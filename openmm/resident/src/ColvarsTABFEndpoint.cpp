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

// Preserve the globals and per-DOF layout of existing chinfo checkpoints.
void ColvarsTABFIntegrator::configureEndpoint(double temperature, double frictionCoeff, double initialAlpha) {
    checkStepSize(getStepSize());
    checkVariableName(alphaVariableName_);
    checkVariableName(alphaDerivativeVariableName_);
    alphaIndex_ = addGlobalVariable(alphaVariableName_, initialAlpha);
    derivativeIndex_ = addGlobalVariable(alphaDerivativeVariableName_, 0);
    addGlobalVariable("tabf_kT", 0.00831446261815324*temperature);
    addGlobalVariable("tabf_friction", frictionCoeff);
    addGlobalVariable("tabf_decay", 1.0);
    addGlobalVariable("tabf_noise", 0.0);
    addComputeGlobal("tabf_decay", "exp(-tabf_friction*dt)");
    addComputeGlobal("tabf_noise", "sqrt((1-tabf_decay)*(1+tabf_decay)*tabf_kT)");
}

void ColvarsTABFIntegrator::initialize(ContextImpl& c) {
    int colvarsForces = 0;
    const System& system = c.getSystem();
    for (int i = 0; i < system.getNumForces(); ++i) {
        const Force& force = system.getForce(i);
        if (dynamic_cast<const ColvarsForce*>(&force)) {
            ++colvarsForces;
            if (force.getForceGroup() != colvarsForceGroup_)
                throw OpenMMException("Place the ColvarsForce in the TABF integrator's Colvars force group");
        } else {
            if (force.getForceGroup() != physicalForceGroup_)
                throw OpenMMException("Place all physical forces in the TABF integrator's physical force group");
            if (const auto* nonbonded = dynamic_cast<const NonbondedForce*>(&force)) {
                int reciprocal = nonbonded->getReciprocalSpaceForceGroup();
                auto method = nonbonded->getNonbondedMethod();
                bool usesReciprocal = method == NonbondedForce::Ewald || method == NonbondedForce::PME ||
                                      method == NonbondedForce::LJPME;
                if (usesReciprocal && reciprocal != -1 && reciprocal != physicalForceGroup_)
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
