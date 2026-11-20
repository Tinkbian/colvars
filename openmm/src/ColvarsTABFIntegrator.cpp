#include "ColvarsTABFIntegrator.h"
#include "ColvarsForce.h"
#include "ColvarsForceImpl.h"
#include "colvarproxy_openmm.h"
#include "openmm/NonbondedForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/System.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/ForceImpl.h"
#include <cmath>
#include <regex>
#include <string>

using namespace OpenMM;
using namespace ColvarsPlugin;

namespace {
void checkAlpha(double alpha) {
    if (!std::isfinite(alpha) || alpha <= 0)
        throw OpenMMException("TABF alpha must be finite and positive");
}
double checkStepSize(double size) {
    if (!std::isfinite(size) || size <= 0)
        throw OpenMMException("TABF step size must be finite and positive");
    return size;
}
void checkVariableName(const std::string& name) {
    static const std::regex identifier("[a-zA-Z_][a-zA-Z_0-9]*");
    static const std::regex reserved("x|v|m|dt|gaussian|uniform|f[0-9]*|energy[0-9]*|colvars_tabf_dE_dalpha|tabf_(kT|friction|decay|noise|unconstrained_x)");
    if (!std::regex_match(name, identifier) || std::regex_match(name, reserved))
        throw OpenMMException("Invalid TABF integrator variable name: "+name);
}
}

ColvarsTABFIntegrator::ColvarsTABFIntegrator(double stepSize, double temperature,
                                             double frictionCoeff, double initialAlpha,
                                             int physicalForceGroup, int colvarsForceGroup,
                                             const std::string& alpha, const std::string& derivative)
    : CustomIntegrator(checkStepSize(stepSize)), physicalGroup_(physicalForceGroup),
      biasGroup_(colvarsForceGroup) {
    checkAlpha(initialAlpha);
    if (!std::isfinite(temperature) || temperature < 0 ||
        !std::isfinite(frictionCoeff) || frictionCoeff < 0)
        throw OpenMMException("TABF temperature and friction must be finite and nonnegative");
    if (physicalGroup_ < 0 || physicalGroup_ > 31 || biasGroup_ < 0 || biasGroup_ > 31 ||
        physicalGroup_ == biasGroup_)
        throw OpenMMException("TABF needs distinct physical and Colvars force groups in [0, 31]");

    checkVariableName(alpha);
    checkVariableName(derivative);
    if (alpha == derivative)
        throw OpenMMException("TABF alpha and derivative variable names must differ");
    alphaIndex_ = addGlobalVariable(alpha, initialAlpha);
    derivativeIndex_ = addGlobalVariable(derivative, 0);
    addGlobalVariable("tabf_kT", 0.00831446261815324*temperature);
    addGlobalVariable("tabf_friction", frictionCoeff);
    addGlobalVariable("tabf_decay", 1.0);
    addGlobalVariable("tabf_noise", 0.0);
    addPerDofVariable("tabf_unconstrained_x", 0.0);
    const std::string physical = std::to_string(physicalGroup_);
    const std::string halfKick = "v+0.5*dt*f"+physical+"/(m*"+alpha+")";

    addComputeGlobal("tabf_decay", "exp(-tabf_friction*dt)");
    addComputeGlobal("tabf_noise", "sqrt((1-tabf_decay)*(1+tabf_decay)*tabf_kT)");
    addComputePerDof("v", halfKick);
    addComputePerDof("x", "x+0.5*dt*v");
    addComputePerDof("v", "tabf_decay*v+tabf_noise*gaussian/sqrt(m)");
    addComputePerDof("x", "x+0.5*dt*v");
    addComputePerDof("tabf_unconstrained_x", "x");
    addConstrainPositions();
    addComputePerDof("v", "v+(x-tabf_unconstrained_x)/dt");
    addConstrainVelocities();
    addComputeGlobal(derivative, "-energy"+physical+"/("+alpha+"*"+alpha+")");
    addComputePerDof("v", halfKick);
    // OpenMM makes changes to Context parameters visible before evaluating
    // forces. Ordinary CustomIntegrator globals can still be stale inside a
    // force callback on Reference. Publish after the physical half kick so
    // the endpoint force and energy can be computed together.
    addComputeGlobal(derivativeParameterName(), derivative);
    addComputePerDof("v", "v+dt*f"+std::to_string(biasGroup_)+"/m");
    addConstrainVelocities();
}

void ColvarsTABFIntegrator::initialize(ContextImpl& c) {
    int colvarsForces = 0;
    const System& system = c.getSystem();
    for (int i = 0; i < system.getNumForces(); ++i) {
        const Force& force = system.getForce(i);
        if (dynamic_cast<const ColvarsForce*>(&force)) {
            ++colvarsForces;
            if (force.getForceGroup() != biasGroup_)
                throw OpenMMException("Place the ColvarsForce in the TABF integrator's Colvars force group");
        } else {
            if (force.getForceGroup() != physicalGroup_)
                throw OpenMMException("Place all physical forces in the TABF integrator's physical force group");
            if (const auto* nonbonded = dynamic_cast<const NonbondedForce*>(&force)) {
                int reciprocal = nonbonded->getReciprocalSpaceForceGroup();
                auto method = nonbonded->getNonbondedMethod();
                bool usesReciprocal = method == NonbondedForce::Ewald || method == NonbondedForce::PME ||
                                      method == NonbondedForce::LJPME;
                if (usesReciprocal && reciprocal != -1 && reciprocal != physicalGroup_)
                    throw OpenMMException("Place the reciprocal-space force in the TABF physical force group");
            }
        }
    }
    if (colvarsForces != 1)
        throw OpenMMException("The TABF integrator requires exactly one ColvarsForce");
    for (ForceImpl* impl : c.getForceImpls()) {
        if (auto* bias = dynamic_cast<ColvarsForceImpl*>(impl)) {
            colvars_ = &bias->proxy();
        } else if (impl->getDefaultParameters().count(derivativeParameterName()))
            throw OpenMMException("Another force uses the reserved TABF derivative Context parameter");
    }
    CustomIntegrator::initialize(c);
}

void ColvarsTABFIntegrator::cleanup() {
    colvars_ = nullptr;
    CustomIntegrator::cleanup();
}

void ColvarsTABFIntegrator::setStepSize(double size) {
    CustomIntegrator::setStepSize(checkStepSize(size));
}

double ColvarsTABFIntegrator::getAlpha() const {
    return collectingBias_ ? stepAlpha_ : getGlobalVariable(alphaIndex_);
}

double ColvarsTABFIntegrator::getAlphaDerivative() const {
    return collectingBias_ ? context->getParameter(derivativeParameterName()) : getGlobalVariable(derivativeIndex_);
}

void ColvarsTABFIntegrator::setNextAlpha(double alpha) {
    if (!collectingBias_)
        throw OpenMMException("TABF alpha feedback requires a scheduled Colvars evaluation");
    checkAlpha(alpha);
    nextAlpha_ = alpha;
}

void ColvarsTABFIntegrator::step(int steps) {
    if (context == nullptr)
        throw OpenMMException("The TABF integrator is not bound to a Context");
    if (steps < 0)
        throw OpenMMException("TABF step count must be nonnegative");
    for (int i = 0; i < steps; ++i) {
        stepAlpha_ = getGlobalVariable(alphaIndex_);
        colvars_->validateTabfStart(stepAlpha_, context->getStepCount());
        // State updates (e.g. center-of-mass removal) may themselves query
        // forces. Only the later endpoint evaluation may advance Colvars.
        if (context->updateContextState())
            stateChanged(State::Positions);
        nextAlpha_ = stepAlpha_;
        collectingBias_ = true;
        try {
            CustomIntegrator::step(1);
        } catch (...) {
            collectingBias_ = false;
            throw;
        }
        collectingBias_ = false;
        // This ordinary OpenMM global participates in checkpoint/saveState.
        // Commit outside the running platform kernel, on every MD step even
        // when Simulation batches multiple steps in a single call.
        if (nextAlpha_ != stepAlpha_)
            setGlobalVariable(alphaIndex_, nextAlpha_);
    }
}
