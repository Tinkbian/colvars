#ifndef OPENMM_COLVARS_TABF_INTEGRATOR_H
#define OPENMM_COLVARS_TABF_INTEGRATOR_H

#include "openmm/CustomIntegrator.h"

class colvarproxy_openmm;

namespace ColvarsPlugin {

/// Langevin integration of U(x)/alpha with unscaled Colvars forces.
/// Colvars samples the constrained endpoint and propagates alpha once per MD
/// step. The derivative is in kJ/mol per unit of dimensionless alpha.
class ColvarsTABFIntegrator : public OpenMM::CustomIntegrator {
public:
    ColvarsTABFIntegrator(double stepSize, double temperature, double frictionCoeff,
                          double initialAlpha = 1.0, int physicalForceGroup = 0,
                          int colvarsForceGroup = 1,
                          const std::string& alphaVariableName = "tabf_alpha",
                          const std::string& alphaDerivativeVariableName = "tabf_dE_dalpha");
    void step(int steps) override;
    void setStepSize(double size) override;
    int getPhysicalForceGroup() const { return physicalGroup_; }
    int getColvarsForceGroup() const { return biasGroup_; }
    const std::string& getAlphaVariableName() const { return getGlobalVariableName(alphaIndex_); }
    const std::string& getAlphaDerivativeVariableName() const { return getGlobalVariableName(derivativeIndex_); }
    double getAlpha() const;
    double getAlphaDerivative() const;

    // Handshake with the Colvars ForceImpl during its scheduled evaluation.
    static constexpr const char* derivativeParameterName() { return "colvars_tabf_dE_dalpha"; }
    bool isCollectingBias() const { return collectingBias_; }
    void setNextAlpha(double alpha);

protected:
    void initialize(OpenMM::ContextImpl& context) override;
    void cleanup() override;

private:
    colvarproxy_openmm* colvars_ = nullptr;
    int physicalGroup_, biasGroup_;
    int alphaIndex_, derivativeIndex_;
    bool collectingBias_ = false;
    double stepAlpha_ = 1.0, nextAlpha_ = 1.0;
};

}
#endif
