// SPDX-License-Identifier: LGPL-3.0-or-later
#include "CudaTABFStep.h"
#include "CudaColvarsKernel.h"
#include "ColvarsTABFIntegrator.h"
#include "openmm/Kernel.h"
#include "openmm/Platform.h"
#include "openmm/common/ComputeKernel.h"
#include "openmm/common/ComputeProgram.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/cuda/CudaArray.h"
#include "openmm/cuda/CudaBondedUtilities.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaIntegrationUtilities.h"
#include "openmm/cuda/CudaNonbondedUtilities.h"
#include "openmm/cuda/CudaPlatform.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/ForceImpl.h"
#include "openmm/kernels.h"
#include <cmath>
#include <cstdint>

using namespace OpenMM;
using namespace ColvarsPlugin;

namespace {
// posDelta and velocities use OpenMM's mixed type. Always accumulate the
// physical energy and divide by alpha in double precision.
const char* source = R"CUDA(
KERNEL void tabfDrift(GLOBAL mixed4* velm, GLOBAL const mm_long* force,
                      GLOBAL mixed4* posDelta, GLOBAL mixed4* unconstrained,
                      GLOBAL const double* endpoint, GLOBAL const float4* random,
                      GLOBAL int* invalid, int randomOffset, double dt, double decay, double noise) {
    for (int i = GLOBAL_ID; i < NUM_ATOMS; i += GLOBAL_SIZE) {
        mixed4 v = velm[i];
        mixed4 dx = make_mixed4(0, 0, 0, 0);
        if (v.w != 0) {
            const double scale = (0.5*dt*v.w/4294967296.0)/endpoint[0];
            const double vx = v.x + scale*force[i];
            const double vy = v.y + scale*force[i+PADDED];
            const double vz = v.z + scale*force[i+2*PADDED];
            const float4 r = random[randomOffset+i];
            const double sigma = noise*sqrt((double)v.w);
            v.x = decay*vx + sigma*r.x;
            v.y = decay*vy + sigma*r.y;
            v.z = decay*vz + sigma*r.z;
            dx.x = 0.5*dt*(vx+v.x);
            dx.y = 0.5*dt*(vy+v.y);
            dx.z = 0.5*dt*(vz+v.z);
            if (!(isfinite(dx.x) && isfinite(dx.y) && isfinite(dx.z))) atomicExch(invalid, 1);
            velm[i] = v;
        }
        posDelta[i] = unconstrained[i] = dx;
    }
}
KERNEL void tabfConstrainedPositions(GLOBAL real4* posq, GLOBAL real4* correction,
                                     GLOBAL mixed4* velm, GLOBAL const mixed4* constrained,
                                     GLOBAL const mixed4* unconstrained, GLOBAL int* invalid, double invDt) {
    for (int i = GLOBAL_ID; i < NUM_ATOMS; i += GLOBAL_SIZE) {
        mixed4 v = velm[i];
        if (v.w == 0) continue;
        real4 p = posq[i];
        mixed4 d = constrained[i], old = unconstrained[i];
        v.x += (d.x-old.x)*invDt;
        v.y += (d.y-old.y)*invDt;
        v.z += (d.z-old.z)*invDt;
        velm[i] = v;
        mixed x = p.x, y = p.y, z = p.z;
#ifdef USE_MIXED_PRECISION
        real4 c = correction[i];
        x += (mixed)c.x; y += (mixed)c.y; z += (mixed)c.z;
#endif
        x += d.x; y += d.y; z += d.z;
        p.x = x; p.y = y; p.z = z;
        posq[i] = p;
#ifdef USE_MIXED_PRECISION
        correction[i] = make_real4(x-(mixed)p.x, y-(mixed)p.y, z-(mixed)p.z, 0);
#endif
        if (!(isfinite(p.x) && isfinite(p.y) && isfinite(p.z) &&
              isfinite(v.x) && isfinite(v.y) && isfinite(v.z))) atomicExch(invalid, 1);
    }
}
KERNEL void tabfReduceEnergy(GLOBAL const ENERGY_TYPE* energy, GLOBAL double* endpoint,
                             GLOBAL int* invalid, int size, double correction) {
    __shared__ double partial[128];
    double sum = 0;
    for (int i = threadIdx.x; i < size; i += blockDim.x) sum += (double)energy[i];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x/2; stride; stride /= 2) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x+stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const double alpha = endpoint[0];
        const double d = -(correction+partial[0])/(alpha*alpha);
        if (!(isfinite(alpha) && alpha > 0 && isfinite(d))) *invalid = 1;
        endpoint[1] = (*invalid ? __longlong_as_double(0x7ff8000000000000LL) : d);
    }
}
KERNEL void tabfKick(GLOBAL mixed4* velm, GLOBAL const mm_long* force,
                     GLOBAL const double* endpoint, GLOBAL int* invalid, double dt, int physical) {
    for (int i = GLOBAL_ID; i < NUM_ATOMS; i += GLOBAL_SIZE) {
        mixed4 v = velm[i];
        if (v.w == 0) continue;
        double scale = dt*v.w/4294967296.0;
        if (physical) scale *= 0.5/endpoint[0];
        v.x += scale*force[i]; v.y += scale*force[i+PADDED]; v.z += scale*force[i+2*PADDED];
        if (!(isfinite(v.x) && isfinite(v.y) && isfinite(v.z))) atomicExch(invalid, 1);
        velm[i] = v;
    }
}
)CUDA";

class DeviceStep final : public CudaTABFStep {
public:
    DeviceStep(ContextImpl& context, ColvarsTABFIntegrator& integrator, CudaColvarsKernel& bias, CudaContext& cuda)
        : context(context), integrator(integrator), bias(bias), cuda(cuda),
          physicalMask(static_cast<int>(std::uint32_t(1) << integrator.getPhysicalForceGroup())),
          biasMask(static_cast<int>(std::uint32_t(1) << integrator.getColvarsForceGroup())),
          beginKernel(context.getPlatform().createKernel(CalcForcesAndEnergyKernel::Name(), context)) {
        ContextSelector selector(cuda);
        beginKernel.getAs<CalcForcesAndEnergyKernel>().initialize(context.getSystem());
        auto& utils = cuda.getIntegrationUtilities();
        auto& state = bias.deviceTabfState();
        if (state.getElementSize() != sizeof(double) || state.getSize() != 2)
            throw OpenMMException("Invalid CUDA TABF endpoint scalar layout");
        unconstrained.initialize(cuda, cuda.getPaddedNumAtoms(), utils.getPosDelta().getElementSize(), "tabfUnconstrainedDelta");
        auto& forces = cuda.getLongForceBuffer();
        savedPhysical.initialize(cuda, forces.getSize(), forces.getElementSize(), "tabfPhysicalForce");
        invalid.initialize<int>(cuda, 1, "tabfInvalidState");
        const int energyBytes = cuda.getEnergyBuffer().getElementSize();
        if (energyBytes != sizeof(double) && energyBytes != sizeof(float))
            throw OpenMMException("Unsupported OpenMM CUDA energy accumulator precision");
        auto program = cuda.compileProgram(source, {{"NUM_ATOMS", std::to_string(cuda.getNumAtoms())},
            {"PADDED", std::to_string(cuda.getPaddedNumAtoms())},
            {"ENERGY_TYPE", energyBytes == sizeof(double) ? "double" : "float"}});
        drift = program->createKernel("tabfDrift");
        drift->addArg(cuda.getVelm()); drift->addArg(cuda.getLongForceBuffer());
        drift->addArg(utils.getPosDelta()); drift->addArg(unconstrained);
        drift->addArg(state); drift->addArg(utils.getRandom()); drift->addArg(invalid);
        drift->addArg(); drift->addArg(); drift->addArg(); drift->addArg();
        positions = program->createKernel("tabfConstrainedPositions");
        positions->addArg(cuda.getPosq());
        positions->addArg(cuda.getUseMixedPrecision() ? cuda.getPosqCorrection() : cuda.getPosq());
        positions->addArg(cuda.getVelm()); positions->addArg(utils.getPosDelta());
        positions->addArg(unconstrained); positions->addArg(invalid); positions->addArg();
        reduce = program->createKernel("tabfReduceEnergy");
        reduce->addArg(cuda.getEnergyBuffer()); reduce->addArg(state); reduce->addArg(invalid);
        reduce->addArg(static_cast<int>(cuda.getEnergyBuffer().getSize())); reduce->addArg();
        kick = program->createKernel("tabfKick");
        kick->addArg(cuda.getVelm()); kick->addArg(cuda.getLongForceBuffer()); kick->addArg(state);
        kick->addArg(invalid); kick->addArg(); kick->addArg();
    }
    void beginBatch(double alpha, double derivative, double kT, double friction) override {
        ContextSelector selector(cuda);
        if (!std::isfinite(alpha) || alpha <= 0 || !std::isfinite(derivative) ||
            !std::isfinite(kT) || kT < 0 || !std::isfinite(friction) || friction < 0)
            throw OpenMMException("Invalid device TABF integrator globals");
        bias.validateDeviceTabfStart(alpha, context.getStepCount());
        const double state[] = {alpha, derivative};
        bias.deviceTabfState().upload(state);
        const int zero = 0;
        invalid.upload(&zero);
        havePhysical = false; // External position/parameter edits may occur between batches.
        const double dt = integrator.getStepSize();
        const double decay = std::exp(-friction*dt);
        drift->setArg(8, dt); drift->setArg(9, decay); drift->setArg(10, std::sqrt((1-decay)*(1+decay)*kT));
        positions->setArg(6, 1/dt);
        kick->setArg(4, dt);
        cuda.getIntegrationUtilities().setNextStepSize(dt);
    }
    void integratePhysical(bool contextChanged) override {
        ContextSelector selector(cuda);
        auto& utils = cuda.getIntegrationUtilities();
        if (!havePhysical || contextChanged) {
            context.calcForcesAndEnergy(true, false, physicalMask);
            drift->setArg(1, cuda.getLongForceBuffer());
        } else {
            drift->setArg(1, savedPhysical);
        }
        // CustomIntegrator's Gaussian expression consumes one float4 per
        // real particle. Preserve its RNG cursor across backend checkpoints.
        drift->setArg(7, utils.prepareRandomNumbers(cuda.getNumAtoms()));
        drift->execute(cuda.getNumAtoms());
        utils.applyConstraints(integrator.getConstraintTolerance());
        positions->execute(cuda.getNumAtoms());
        utils.computeVirtualSites();
        utils.applyVelocityConstraints(integrator.getConstraintTolerance());
        physicalEndpoint();
        // The next initial half-kick uses this same physical force, with
        // the newly advanced alpha. The Colvars dispatch overwrites it.
        cuda.getLongForceBuffer().copyTo(savedPhysical);
        kick->setArg(5, 1);
        kick->execute(cuda.getNumAtoms());
    }
    void finishBias() override {
        ContextSelector selector(cuda);
        context.calcForcesAndEnergy(true, false, biasMask);
        kick->setArg(5, 0);
        kick->execute(cuda.getNumAtoms());
        cuda.getIntegrationUtilities().applyVelocityConstraints(integrator.getConstraintTolerance());
        cuda.setTime(cuda.getTime()+integrator.getStepSize());
        cuda.setStepCount(cuda.getStepCount()+1);
        cuda.reorderAtoms();
        havePhysical = !cuda.getAtomsWereReordered();
    }
    void endBatch(double& alpha, double& derivative) override {
        ContextSelector selector(cuda);
        int bad;
        invalid.download(&bad);
        if (bad) throw OpenMMException("Nonfinite coordinates, velocities, or physical energy during CUDA TABF integration");
        double state[2];
        bias.deviceTabfState().download(state);
        alpha = state[0];
        derivative = state[1];
        if (!(std::isfinite(alpha) && alpha > 0 && std::isfinite(derivative)))
            throw OpenMMException("Invalid CUDA TABF endpoint feedback");
        bias.finishDeviceTabfBatch(alpha);
    }
private:
    void physicalEndpoint() {
        // Same ordering as OpenMM's single-context CUDA finishComputation.
        // All ForceImpl and post-computation energy contributions (including
        // host-valued constants) are included. Only the final reduction differs.
        context.getLastForceGroups() = physicalMask;
        double contribution;
        do {
            beginKernel.getAs<CalcForcesAndEnergyKernel>().beginComputation(context, true, true, physicalMask);
            contribution = 0;
            for (auto* force : context.getForceImpls())
                contribution += force->calcForcesAndEnergy(context, true, true, physicalMask);
            cuda.getBondedUtilities().computeInteractions(physicalMask);
            cuda.getNonbondedUtilities().computeInteractions(physicalMask, true, true);
            for (auto* post : cuda.getPostComputations())
                contribution += post->computeForceAndEnergy(true, true, physicalMask);
            cuda.getIntegrationUtilities().distributeForcesFromVirtualSites();
        } while (!cuda.getForcesValid());
        reduce->setArg(4, contribution);
        reduce->execute(128, 128);
    }
    ContextImpl& context;
    ColvarsTABFIntegrator& integrator;
    CudaColvarsKernel& bias;
    CudaContext& cuda;
    const int physicalMask, biasMask;
    Kernel beginKernel;
    bool havePhysical = false;
    CudaArray unconstrained, invalid, savedPhysical;
    ComputeKernel drift, positions, reduce, kick;
};
}

std::shared_ptr<CudaTABFStep> ColvarsPlugin::createCudaTABFStep(ContextImpl& context, ColvarsTABFIntegrator& integrator,
                                                                CudaColvarsKernel& bias) {
    auto* data = static_cast<CudaPlatform::PlatformData*>(context.getPlatformData());
    if (context.getPlatform().getName() != "CUDA" || !data || data->contexts.size() != 1)
        throw OpenMMException("Device TABF requires a single OpenMM CUDA device");
    return std::make_shared<DeviceStep>(context, integrator, bias, *data->contexts.front());
}
