// Single-device, scalar TABF endpoint for the resident OpenMM ForceImpl.
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "CudaTabfForce.h"
#include "ColvarsTABFIntegrator.h"
#include "colvarproxy_openmm.h"
#include "colvarbias_abf.h"
#include "colvarbias_meta.h"
#include "colvarbias_restraint.h"
#include "openmm/common/ComputeProgram.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/internal/ContextImpl.h"
#include <algorithm>
#include <cmath>
#include <limits>
using namespace OpenMM;
using namespace ColvarsPlugin;

namespace {
constexpr int capacitySteps = 1024;
// One thread updates the two coupled adaptive degrees of freedom in Colvars
// declaration order. Positions, gradients, auxiliary variables and history
// stay on the OpenMM stream between scheduled outputs or batch boundaries.
const char* endpointSource = R"CUDA(
KERNEL void computeDeviceTabf(GLOBAL const double* p, GLOBAL double* f,
    GLOBAL const int4* mapping, GLOBAL const double4* walls,
    GLOBAL const TabfExtConfig* cfg, GLOBAL TabfExtState* state,
    GLOBAL TabfGaussianState* rng, GLOBAL const TabfBiasSpec* specs, GLOBAL double* bias,
    GLOBAL double* scalar, GLOBAL double* obs, GLOBAL double4* geometry,
    GLOBAL double* potential, GLOBAL int* error,
    long long step, int advance, int first, double dt, double4 box, double energyToKJ,
    int wantEnergy, int replay) {
    if (GLOBAL_ID || *error) return;
    if (!(scalar[0] > 0.0) || !isfinite(scalar[0]) || !isfinite(scalar[1])) { *error = 1; return; }
    for (int j = 0; j < 3*NUM_SELECTED; ++j) f[j] = 0;
    for (int i = 0; i < NUM_CV; ++i) {
        const int4 m = mapping[i];
        double4 g = make_double4(0,0,0,replay ? state[1].actual : scalar[0]);
        if (m.x >= 0) {
            double d[3], boxLen[] = {box.x,box.y,box.z};
            for (int j = 0; j < 3; ++j) {
                d[j] = p[m.y + j*NUM_SELECTED]-p[m.x + j*NUM_SELECTED];
                if (m.z && boxLen[j] > 0.0)
                    d[j] -= floor(d[j]/boxLen[j]+0.5)*boxLen[j];
            }
            double r = sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
            if (!isfinite(r)) { *error = 2; return; }
            const double inv = r > 0.0 ? 1.0/r : 0.0;
            g = make_double4(d[0]*inv,d[1]*inv,d[2]*inv,r);
        }
        geometry[i] = g;
    }

    double total = 0.0;
    double nextAlpha = scalar[0];
    for (int i = 0; i < NUM_CV; ++i) {
        int4 m = mapping[i];
        const double4 g = geometry[i], wall = walls[i];
        const double actual = g.w;
        const double lower = fmin(actual-wall.x,0.0), upper = fmax(actual-wall.y,0.0);
        double force = -wall.z*lower-wall.w*upper;
        total += 0.5*(wall.z*lower*lower + wall.w*upper*upper);
        double reported = actual;
        if (m.w >= 0) {
            const int a = m.w;
            TabfExtView v = tabfExtView(cfg[a], state[a], actual, -scalar[1]/energyToKJ, replay);
            TabfBiasResult b = tabfBias(specs[a],bias,step,advance,first,actual,v.x,v.systemForce,wantEnergy);
            if (b.error) { *error = 10+b.error; return; }
            reported = v.x;
            total += v.springEnergy+b.energy;
            if (advance) {
                double noise = cfg[a].frictionPerFs != 0.0 ? tabfGaussian(*rng) : 0.0;
                int err = tabfExtAdvance(cfg[a],state[a],v,b.force,force,dt,noise,step);
                if (err) { *error = 20+err; return; }
                if (cfg[a].external) nextAlpha = state[a].x;
            }
            force = tabfExtAtomicForce(cfg[a],v,force);
        }
        if (m.x >= 0) {
            double dir[] = {g.x,g.y,g.z};
            for (int j = 0; j < 3; ++j) {
                f[m.x + j*NUM_SELECTED] -= dir[j]*force;
                f[m.y + j*NUM_SELECTED] += dir[j]*force;
            }
        }
        if (advance) {
            obs[i+NUM_CV] = obs[i];
            obs[i] = actual;
        }
        obs[i+2*NUM_CV] = actual;
        obs[i+3*NUM_CV] = reported;
    }
    if (!isfinite(total)) { *error = 3; return; }
    if (advance) scalar[0] = nextAlpha;
    *potential = total*energyToKJ;
}
)CUDA";

template<class T>
void initialize(CudaContext& cuda, CudaArray& array, const std::vector<T>& data, const char* name) {
    array.initialize<T>(cuda, data.size(), name);
    array.upload(data);
}
}

CudaTabfForce::CudaTabfForce(colvarproxy_openmm& proxy, CudaContext& cuda,
                             CudaArray& positions, CudaArray& forces)
    : proxy_(proxy), cuda_(cuda) {
    std::string why;
    periodicBox();
    if (!configureTabfVariables(proxy, variables_, why)) throw OpenMMException(why);
    auto& module = proxy.module();
    const auto& cvs = *module.variables();
    std::vector<mm_int4> mapping;
    std::vector<mm_double4> walls;
    std::vector<TabfExtConfig> configs(2);
    std::vector<TabfExtState> states(2);
    std::vector<double> observations(4*cvs.size());
    for (size_t i=0; i<cvs.size(); ++i) {
        const auto& v = variables_[i];
        mapping.emplace_back(v.atom1,v.atom2,v.minimumImage,v.axis);
        walls.emplace_back(v.lower,v.upper,v.kLower,v.kUpper);
        if (v.axis >= 0) {
            axisVariable_[v.axis] = i;
            configs[v.axis] = CudaTabfDynamicsAdapter::config(*cvs[i]);
            stochastic_ |= configs[v.axis].frictionPerFs != 0.0;
            states[v.axis] = CudaTabfDynamicsAdapter::read(*cvs[i], module.it_restart);
        }
        observations[i] = cvs[i]->actual_value().real_value;
    }
    std::array<colvarbias_abf*,2> abf{};
    std::array<colvarbias_meta*,2> meta{};
    for (auto* b : module.biases) {
        if (dynamic_cast<colvarbias_restraint_harmonic_walls*>(b)) continue;
        if (b->num_variables()!=1) throw OpenMMException("Device TABF requires scalar biases: "+b->name);
        auto cv = std::find(cvs.begin(),cvs.end(),b->variables(0));
        int a = cv != cvs.end() ? variables_[cv-cvs.begin()].axis : -1;
        if (a < 0) throw OpenMMException("Device TABF needs an extended coordinate for bias "+b->name);
        if (auto* v = dynamic_cast<colvarbias_abf*>(b); v && !abf[a]) abf[a]=v;
        else if (auto* v = dynamic_cast<colvarbias_meta*>(b); v && !meta[a]) meta[a]=v;
        else throw OpenMMException("Unsupported or duplicate device TABF bias: "+b->name);
    }
    for (int a=0; a<2; ++a)
        if (!biases_[a].configure(abf[a],meta[a],why)) throw OpenMMException(why);
    TabfGaussianState random{};
    if (stochastic_ && !CudaTabfDynamicsAdapter::readRng(proxy,random))
        throw OpenMMException("Device TABF requires a compatible Colvars RNG checkpoint (Linux libstdc++)");
    initialize(cuda, mapping_, mapping, "tabfVariableAtomIndices");
    initialize(cuda, walls_, walls, "tabfHarmonicWalls");
    initialize(cuda, configs_, configs, "tabfExtendedParameters");
    initialize(cuda, states_, states, "tabfExtendedCoordinates");
    initialize(cuda, rng_, std::vector<TabfGaussianState>{random}, "tabfColvarsRandomState");
    initialize(cuda, diagnostics_, observations, "tabfReportedCoordinates");
    initialize(cuda, error_, std::vector<int>{0}, "tabfEndpointError");
    scalars_.initialize<double>(cuda,2,"tabfAlphaAndDerivative");
    geometry_.initialize<mm_double4>(cuda,cvs.size(),"tabfDistanceGradients");
    potential_.initialize<double>(cuda,1,"tabfBiasPotential");
    specBuffer_.initialize<TabfBiasSpec>(cuda,2,"tabfAdaptiveParameters");
    const std::string source = std::string(tabfDynamicsCudaSource())+cudaTabfBiasSource+endpointSource;
    auto program = cuda.compileProgram(source, {{"NUM_CV",std::to_string(cvs.size())},
        {"NUM_SELECTED",std::to_string(proxy.requestedAtoms().size())}});
    evaluate_ = program->createKernel("computeDeviceTabf");
    packBiases();
    for (auto* array : {&positions,&forces,&mapping_,&walls_,&configs_,&states_,&rng_,&specBuffer_,
                        &biasBuffer_,&scalars_,&diagnostics_,&geometry_,&potential_,&error_})
        evaluate_->addArg(*array);
    for (int i=0; i<8; ++i) evaluate_->addArg();
}

void CudaTabfForce::packBiases() {
    std::vector<double> buffer;
    for (int a=0; a<2; ++a) specs_[a] = biases_[a].pack(buffer,capacitySteps);
    specBuffer_.upload(specs_);
    if (!biasBuffer_.isInitialized()) biasBuffer_.initialize<double>(cuda_,buffer.size(),"tabfAdaptiveGrids");
    else if (biasBuffer_.getSize()!=buffer.size()) biasBuffer_.resize(buffer.size());
    biasBuffer_.upload(buffer);
}

void CudaTabfForce::checkDevice() {
    int error;
    error_.download(&error);
    if (error || poisoned_) {
        poisoned_ = true;
        throw OpenMMException("Invalid CUDA TABF endpoint (error "+std::to_string(error)+
                              "); restore the last paired checkpoint in a new Context");
    }
}

void CudaTabfForce::validateStart(double alpha, long long step) {
    periodicBox();
    if (poisoned_) throw OpenMMException("CUDA TABF Context requires a checkpoint restart");
    if (!pending_) proxy_.validateTabfStart(alpha,step);
    else if (step != lastStep_ || !std::isfinite(alpha) ||
             std::abs(alpha-batchAlpha_) > 1.e-12*std::max(1.0,std::abs(batchAlpha_)))
        throw OpenMMException("OpenMM and device TABF endpoint step or alpha differ");
}

void CudaTabfForce::checkBatchEnd(double alpha) {
    if (pending_) {
        checkDevice();
        batchAlpha_ = alpha;
    }
}

mm_double4 CudaTabfForce::periodicBox() const {
    if (!proxy_.usesPbc_) return mm_double4(0,0,0,0);
    Vec3 a,b,c;
    cuda_.getPeriodicBoxVectors(a,b,c);
    if (b[0] != 0.0 || c[0] != 0.0 || c[1] != 0.0)
        throw OpenMMException("Device TABF requires an orthorhombic periodic box");
    const double scale = proxy_.lengthFromNm();
    const mm_double4 box(a[0]*scale,b[1]*scale,c[2]*scale,0);
    for (double height : {box.x,box.y,box.z})
        if (!(height > 0.0) || !std::isnormal(height*height))
            throw OpenMMException("Invalid device TABF cell");
    return box;
}

double CudaTabfForce::evaluate(ContextImpl& context, bool energy) {
    auto& module = proxy_.module();
    const bool collecting = proxy_.tabf_->isCollectingBias();
    long long step = context.getStepCount();
    if (step < 0 || step >= std::numeric_limits<cvm::step_number>::max())
        throw OpenMMException("Step count exceeds device TABF range");
    if (collecting) ++step;
    else {
        if (proxy_.tabf_->isDeviceBatchActive())
            throw OpenMMException("A physical state update requested a nested Colvars bias during device TABF integration");
        validateStart(proxy_.tabf_->getAlpha(),step);
        finishBatch();
        const double scalars[] = {proxy_.tabf_->getAlpha(),proxy_.tabf_->getAlphaDerivative()};
        scalars_.upload(scalars);
    }
    // OpenMM can invalidate and repeat a force evaluation before incrementing
    // its step. The accepted grids and pre-integration coordinate still apply.
    const bool replay = collecting && step == lastStep_;
    const bool advance = collecting && !replay;
    if (advance && lastStep_ >= 0 && step != lastStep_+1)
        throw OpenMMException("Device TABF requires consecutive accepted endpoints");
    const double dt = 1000.0*context.getIntegrator().getStepSize();
    if (!(dt>0.0) || !std::isfinite(dt)) throw OpenMMException("Invalid device TABF timestep");
    const mm_double4 box = periodicBox();
    if (advance && lastStep_<0 && !proxy_.restartPending_) module.set_initial_step(step);
    module.it = step;
    if (dt != proxy_.lastTimestepFs_) {
        proxy_.set_integration_timestep(dt);
        if (module.update_engine_parameters()) throw OpenMMException("Could not set device TABF timestep");
        proxy_.lastTimestepFs_=dt;
    }
    evaluate_->setArg(8,biasBuffer_);
    evaluate_->setArg(14,step);
    evaluate_->setArg(15,int(advance));
    evaluate_->setArg(16,int(module.step_relative()==0));
    evaluate_->setArg(17,dt);
    evaluate_->setArg(18,box);
    evaluate_->setArg(19,proxy_.energyToKJ());
    const bool output = advance && module.scheduled_output_due();
    evaluate_->setArg(20,int(energy || output));
    evaluate_->setArg(21,int(replay));
    evaluate_->execute(1,1);
    if (advance) {
        lastStep_ = step;
        ++pending_;
        if (output || pending_ == capacitySteps) finishBatch();
        if (output && module.write_scheduled_output()) throw OpenMMException("Colvars scheduled output failed");
    } else if (!collecting) {
        checkDevice();
        std::vector<double> obs;
        diagnostics_.download(obs);
        const auto& cvs = *module.variables();
        for (size_t i=0; i<cvs.size(); ++i)
            publishTabfDiagnostic(*cvs[i],obs[i+2*cvs.size()],obs[i+3*cvs.size()]);
        proxy_.evaluated_ = true;
    }
    if (!energy) return 0;
    double potential;
    potential_.download(&potential);
    return potential;
}

void CudaTabfForce::finishBatch() {
    if (!pending_) return;
    checkDevice();
    std::vector<TabfExtState> states;
    std::vector<double> observations, data;
    TabfGaussianState rng;
    states_.download(states);
    if (stochastic_) rng_.download(&rng);
    diagnostics_.download(observations);
    biasBuffer_.download(data);
    auto& m = proxy_.module();
    const auto& cvs = *m.variables();
    for (int a=0; a<2; ++a) {
        CudaTabfDynamicsAdapter::write(*cvs[axisVariable_[a]],states[a],m.step_relative());
        biases_[a].unpack(specs_[a],data);
    }
    const size_t n = cvs.size();
    publishTabfMonitorsAndWalls(proxy_,variables_,{observations.begin(),observations.begin()+n},
        {observations.begin()+n,observations.begin()+2*n},proxy_.lastTimestepFs_,m.step_relative());
    if (stochastic_) CudaTabfDynamicsAdapter::writeRng(proxy_,rng);
    proxy_.evaluated_ = proxy_.haveStep_ = true;
    proxy_.restartAlphaPending_ = proxy_.restartPending_ = false;
    proxy_.currentStep_ = proxy_.lastUpdatedStep_ = lastStep_;
    proxy_.pendingAlpha_ = states[1].x;
    proxy_.biasUpdateCount_ += pending_;
    pending_ = 0;
    packBiases();
}
