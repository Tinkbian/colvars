#ifndef OPENMM_COLVARS_FORCE_H
#define OPENMM_COLVARS_FORCE_H

#include "openmm/Force.h"

#include <string>
#include <vector>

namespace OpenMM { class Context; }
#ifdef OPENMM_COLVARS_RESIDENT
class colvarproxy_openmm;
#endif

namespace ColvarsPlugin {

class ColvarsForceImpl;

/// OpenMM Force wrapper for the Colvars collective variable module.
///
/// The Colvars configuration uses the native OpenMM/GROMACS unit system by
/// default: nanometers for length and kJ/mol for energy. OpenMM positions,
/// energies, and forces are passed through without a custom conversion layer.
class ColvarsForce : public OpenMM::Force {
public:
    enum class ConfigSource {
        File,
        String
    };

    enum class PbcMode {
        Auto,
        Enabled,
        Disabled
    };

    explicit ColvarsForce(const std::string& configFile);

    ColvarsForce(const std::string& configFile,
                 const std::string& outputPrefix,
                 const std::string& lambdaParameterName = "lambda");

    static ColvarsForce fromConfigString(const std::string& configString,
                                         const std::string& outputPrefix = "colvars",
                                         const std::string& lambdaParameterName = "lambda");

    ConfigSource getConfigSource() const;
    const std::string& getConfig() const;

    void setConfigFile(const std::string& configFile);
    void setConfigString(const std::string& configString);

    const std::string& getOutputPrefix() const;
    void setOutputPrefix(const std::string& outputPrefix);

    const std::string& getRestartOutputPrefix() const;
    void setRestartOutputPrefix(const std::string& restartOutputPrefix);

    const std::string& getRestartInputPrefix() const;
    void setRestartInputPrefix(const std::string& restartInputPrefix);

    int getRestartFrequency() const;
    void setRestartFrequency(int restartFrequency);

    const std::string& getLambdaParameterName() const;
    void setLambdaParameterName(const std::string& lambdaParameterName);

    const std::string& getLambdaDerivativeParameterName() const;
    void setLambdaDerivativeParameterName(const std::string& derivativeParameterName);

    const std::string& getTabfAlphaParameterName() const;
    void setTabfAlphaParameterName(const std::string& alphaParameterName);

    const std::string& getTabfAlphaDerivativeParameterName() const;
    void setTabfAlphaDerivativeParameterName(const std::string& derivativeParameterName);

    bool getTabfEnabled() const;
    void setTabfEnabled(bool enabled);

    bool getScaleLambdaDerivativeByTabfAlpha() const;
    void setScaleLambdaDerivativeByTabfAlpha(bool enabled);

    bool getAlchemyEnabled() const;
    void setAlchemyEnabled(bool enabled);

    double getTargetTemperature() const;
    void setTargetTemperature(double temperatureK);

    int getRandomSeed() const;
    void setRandomSeed(int seed);

    const std::string& getUnitSystem() const;

    int getNumParticleCharges() const;
    double getParticleCharge(int particleIndex) const;
    void setParticleCharge(int particleIndex, double chargeElementary);
    void setParticleCharges(const std::vector<double>& chargesElementary);

    bool usesPeriodicBoundaryConditions() const override;
    void setUsesPeriodicBoundaryConditions(bool usesPbc);
    PbcMode getPbcMode() const;
    void setPbcMode(PbcMode mode);

#ifdef OPENMM_COLVARS_RESIDENT
    void setUnitSystem(const std::string& units);
    void setUseGpu(bool enabled) { useGpu_ = enabled; }
    bool getUseGpu() const { return useGpu_; }
    /// Run supported TABF coordinates and biases on a single CUDA device.
    /// Unsupported configurations raise an error when the Context is built.
    void setUseDeviceBias(bool enabled) { useDeviceBias_ = enabled; }
    bool getUseDeviceBias() const { return useDeviceBias_; }
    void setUseGPU(bool enabled) { setUseGpu(enabled); }
    bool getUseGPU() const { return getUseGpu(); }
    bool hasExplicitTabfSetting() const { return tabfConfigured_; }

    /// Diagnostics describe the last evaluation; these do not advance a bias.
    std::string getBackend(OpenMM::Context& context) const;
    double getColvarValue(OpenMM::Context& context, const std::string& name) const;
    long long getBiasUpdateCount(OpenMM::Context& context) const;
    std::string getBiasState(OpenMM::Context& context) const;

    /// Write the current restart state and flush pending bias output files.
    /// Call after the final energy/force evaluation and before deleting Context.
    /// This does not advance adaptive history, and the simulation may continue.
    void flushOutputs(OpenMM::Context& context) const;

#endif

protected:
    OpenMM::ForceImpl* createImpl() const override;

private:
#ifdef OPENMM_COLVARS_RESIDENT
    friend class ::colvarproxy_openmm;
    bool sparseCharges_ = false;
    bool useGpu_ = true;
    bool useDeviceBias_ = false;
    bool tabfConfigured_ = false;
    std::string unitSystem_ = "gromacs";
#endif
    ConfigSource configSource_;
    std::string config_;
    std::string outputPrefix_;
    std::string restartOutputPrefix_;
    std::string restartInputPrefix_;
    std::string lambdaParameterName_;
    std::string lambdaDerivativeParameterName_;
    std::string tabfAlphaParameterName_;
    std::string tabfAlphaDerivativeParameterName_;
    bool alchemyEnabled_;
    bool tabfEnabled_;
    bool scaleLambdaDerivativeByTabfAlpha_;
    PbcMode pbcMode_;
    double targetTemperatureK_;
    int randomSeed_;
    int restartFrequency_;
    std::vector<double> particleCharges_;
};

} // namespace ColvarsPlugin

#endif
