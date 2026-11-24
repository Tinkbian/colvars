#include "ColvarsTABFIntegrator.h"

#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#ifdef OPENMM_COLVARS_RESIDENT
#include "../resident/src/colvarproxy_openmm.h"
#include "../resident/src/CudaTABFStep.h"
#endif

#include <algorithm>
#include <cmath>
#include <set>

using namespace OpenMM;
using namespace ColvarsPlugin;

namespace {

void validateForceGroup(int group, const std::string& name)
{
    if (group < 0 || group > 31) {
        throw OpenMMException(name + " must be between 0 and 31");
    }
}

void validateName(const std::string& name, const std::string& label)
{
    if (name.empty()) {
        throw OpenMMException(label + " must not be empty");
    }
}

double validateStepSize(double stepSize)
{
    if (!std::isfinite(stepSize) || stepSize < 0.0) {
        throw OpenMMException("stepSize must be finite and non-negative");
    }
    return stepSize;
}

void validateFiniteNonnegative(double value, const std::string& label)
{
    if (!std::isfinite(value) || value < 0.0) {
        throw OpenMMException(label + " must be finite and non-negative");
    }
}

std::string forceVariable(int group)
{
    return "f" + std::to_string(group);
}

std::string energyVariable(int group)
{
    return "energy" + std::to_string(group);
}

} // namespace

ColvarsTABFIntegrator::ColvarsTABFIntegrator(double stepSize,
                                             double temperature,
                                             double frictionCoeff,
                                             double initialAlpha,
                                             int physicalForceGroup,
                                             int colvarsForceGroup,
                                             const std::string& alphaVariableName,
                                             const std::string& alphaDerivativeVariableName)
    : CustomIntegrator(validateStepSize(stepSize)),
      physicalForceGroup_(physicalForceGroup),
      colvarsForceGroup_(colvarsForceGroup),
      alphaVariableName_(alphaVariableName),
      alphaDerivativeVariableName_(alphaDerivativeVariableName)
{
    validateForceGroup(physicalForceGroup_, "physicalForceGroup");
    validateForceGroup(colvarsForceGroup_, "colvarsForceGroup");
    if (physicalForceGroup_ == colvarsForceGroup_) {
        throw OpenMMException("physicalForceGroup and colvarsForceGroup must be different");
    }
    validateName(alphaVariableName_, "alphaVariableName");
    validateName(alphaDerivativeVariableName_, "alphaDerivativeVariableName");
    if (alphaVariableName_ == alphaDerivativeVariableName_) {
        throw OpenMMException("alpha variable and derivative variable names must be different");
    }
    static const std::set<std::string> reservedNames = {"a", "b", "kT", "x1"};
    if (reservedNames.count(alphaVariableName_) != 0 ||
        reservedNames.count(alphaDerivativeVariableName_) != 0) {
        throw OpenMMException("alpha variable names must not conflict with a, b, kT, or x1");
    }
    validateFiniteNonnegative(temperature, "temperature");
    validateFiniteNonnegative(frictionCoeff, "frictionCoeff");
    if (!std::isfinite(initialAlpha) || initialAlpha <= 0.0) {
        throw OpenMMException("initialAlpha must be finite and positive");
    }

#ifdef OPENMM_COLVARS_RESIDENT
    configureEndpoint(temperature, frictionCoeff, initialAlpha);
    const std::string thermostat = "tabf_decay*v+tabf_noise*gaussian/sqrt(m)";
    const std::string unconstrained = "tabf_unconstrained_x";
#else
    const double a = std::exp(-frictionCoeff * stepSize);
    const double b = std::sqrt(std::max(0.0, 1.0 - std::exp(-2.0 * frictionCoeff * stepSize)));

    addGlobalVariable("a", a);
    addGlobalVariable("b", b);
    addGlobalVariable("kT", 0.00831446261815324 * temperature);
    alphaIndex_ = addGlobalVariable(alphaVariableName_, initialAlpha);
    derivativeIndex_ = addGlobalVariable(alphaDerivativeVariableName_, 0.0);
    const std::string thermostat = "a*v + b*sqrt(kT/m)*gaussian";
    const std::string unconstrained = "x1";
#endif
    addPerDofVariable(unconstrained, 0.0);

    const std::string physicalForce = forceVariable(physicalForceGroup_);
    const std::string colvarsForce = forceVariable(colvarsForceGroup_);
    const std::string physicalEnergy = energyVariable(physicalForceGroup_);

    // step() updates Context state before the scheduled Colvars force evaluation.

    addComputePerDof("v", "v+0.5*dt*" + physicalForce + "/(m*" + alphaVariableName_ + ")");
    addComputePerDof("x", "x+0.5*dt*v");
    addComputePerDof("v", thermostat);
    addComputePerDof("x", "x+0.5*dt*v");
    addComputePerDof(unconstrained, "x");
    addConstrainPositions();
    addComputePerDof("v", "v+(x-"+unconstrained+")/dt");
    addConstrainVelocities();

    addComputeGlobal(alphaDerivativeVariableName_,
                     "-" + physicalEnergy + "/(" + alphaVariableName_ + "*" +
                     alphaVariableName_ + ")");
    addComputePerDof("v", "v+0.5*dt*" + physicalForce + "/(m*" + alphaVariableName_ + ")");
    addComputeGlobal(derivativeParameterName(), alphaDerivativeVariableName_);
    // Colvars owns history-dependent state (ABF counts, trajectories, alpha).
    // Evaluate it once at the end of the MD step, then apply the full impulse.
    addComputePerDof("v", "v+dt*" + colvarsForce + "/m");
    addConstrainVelocities();
#ifdef OPENMM_COLVARS_RESIDENT
    endpointComputationCount_ = getNumComputations();
#endif
}

int ColvarsTABFIntegrator::getPhysicalForceGroup() const
{
    return physicalForceGroup_;
}

int ColvarsTABFIntegrator::getColvarsForceGroup() const
{
    return colvarsForceGroup_;
}

const std::string& ColvarsTABFIntegrator::getAlphaVariableName() const
{
    return alphaVariableName_;
}

const std::string& ColvarsTABFIntegrator::getAlphaDerivativeVariableName() const
{
    return alphaDerivativeVariableName_;
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
    if (!std::isfinite(alpha) || alpha <= 0)
        throw OpenMMException("TABF alpha must be finite and positive");
    nextAlpha_ = alpha;
}

void ColvarsTABFIntegrator::step(int steps) {
    if (context == nullptr)
        throw OpenMMException("The TABF integrator is not bound to a Context");
    if (steps < 0)
        throw OpenMMException("TABF step count must be nonnegative");
#ifdef OPENMM_COLVARS_RESIDENT
    if (deviceStep_ && steps > 0) {
        if (devicePoisoned_)
            throw OpenMMException("Restore the last paired TABF checkpoint in a new Context after a failed device step");
        double alpha = getGlobalVariable(alphaIndex_);
        double derivative = getGlobalVariable(derivativeIndex_);
        const double kT = getGlobalVariableByName("tabf_kT");
        const double friction = getGlobalVariableByName("tabf_friction");
        deviceStep_->beginBatch(alpha, derivative, kT, friction);
        deviceBatchActive_ = true;
        try {
            for (int i = 0; i < steps; ++i) {
                const bool changed = context->updateContextState();
                if (changed)
                    stateChanged(State::Positions);
                deviceStep_->integratePhysical(changed);
                collectingBias_ = true;
                deviceStep_->finishBias();
                collectingBias_ = false;
            }
            deviceStep_->endBatch(alpha, derivative);
        } catch (...) {
            collectingBias_ = false;
            deviceBatchActive_ = false;
            devicePoisoned_ = true;
            throw;
        }
        deviceBatchActive_ = false;
        setGlobalVariable(alphaIndex_, alpha);
        setGlobalVariable(derivativeIndex_, derivative);
        const double decay = std::exp(-friction*getStepSize());
        setGlobalVariableByName("tabf_decay", decay);
        setGlobalVariableByName("tabf_noise", std::sqrt((1-decay)*(1+decay)*kT));
        context->setParameter(derivativeParameterName(), derivative);
        stateChanged(State::Positions);
        return;
    }
#endif
    for (int i = 0; i < steps; ++i) {
        stepAlpha_ = getGlobalVariable(alphaIndex_);
#ifdef OPENMM_COLVARS_RESIDENT
        colvars_->validateTabfStart(stepAlpha_, context->getStepCount());
#endif
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
