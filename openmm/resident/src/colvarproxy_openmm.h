// Adapted from Colvars PR 979. See ../UPSTREAM.md and ../LICENSE.
#ifndef COLVARPROXY_OPENMM_H
#define COLVARPROXY_OPENMM_H

#include "ColvarsForce.h"
#include "colvarproxy.h"
#include "openmm/Vec3.h"
#include <array>
#include <limits>
#include <random>
#include <mutex>
#include <string>
#include <vector>

namespace OpenMM { class ContextImpl; class System; }
namespace ColvarsPlugin { class ColvarsTABFIntegrator; }

// Colvars currently has shared feature tables and a legacy default proxy.
// Serialize plugin entry points across OpenMM Contexts and select the active
// proxy while holding this recursive lock. Device work itself stays asynchronous.
class OpenMMColvarsLock {
public:
    explicit OpenMMColvarsLock(colvarproxy* active = nullptr);
private:
    std::unique_lock<std::recursive_mutex> lock_;
};

/** Shared host/GPU lifecycle. Construction deliberately defers parsing so a
 * CUDA subclass can establish its stream and memory policy before CV setup. */
class colvarproxy_openmm : public colvarproxy {
public:
    colvarproxy_openmm(const ColvarsPlugin::ColvarsForce&, const OpenMM::System&);
    ~colvarproxy_openmm() override;
    void initializeColvars(OpenMM::ContextImpl& context);
    int setup() override;
    int init_atom(int atomNumber) override;
    int request_atom_charges() override;
    int check_atom_id(int atomNumber) override;
    int set_unit_system(const std::string&, bool checkOnly) override;
    void request_total_force(bool yesno) override;
    bool total_forces_enabled() const override { return total_force_requested; }
    bool total_forces_same_step() const override { return true; }
    bool can_reuse_atom_pairlists() const override { return false; }
    bool integrates_colvars_at_step_end() const override { return tabf_ != nullptr; }
    bool replaying_colvars_endpoint() const override;
    std::string get_integrator_state() const override;
    void set_integrator_state(const std::string&) override;
    int get_tabf_alpha(cvm::real*) override;
    int get_dE_dtabf_alpha(cvm::real*) override;
    int set_tabf_alpha(cvm::real) override;
    int send_tabf_alpha() override;
    double lengthFromNm() const { return 10.0 * angstrom_value_; }
    double energyToKJ() const { return 4.184 / kcal_mol_value_; }
    double forceToOpenMM() const { return lengthFromNm() * energyToKJ(); }
    void add_energy(cvm::real energy) override { biasEnergy_ += energy; }
    void log(const std::string&) override;
    void error(const std::string&) override;
    cvm::real rand_gaussian() override;
    int load_atoms_pdb(const char*, cvm::atom_group&, const std::string&, double) override;
    int load_coords_pdb(const char*, std::vector<cvm::atom_pos>&,
                        const std::vector<int>&, const std::string&, double) override;

    void beginEvaluation(OpenMM::ContextImpl& context);
    void validateTabfStart(double alpha, long long openmmStep);
    double evaluateColvars();
    double calculate(OpenMM::ContextImpl&, const std::vector<OpenMM::Vec3>&);
    void addForcesToOpenMM(std::vector<OpenMM::Vec3>&) const;
    const std::vector<int>& requestedAtoms() const { return atoms_ids; }
    double biasEnergy() const { return biasEnergy_; }
    double getColvarValue(const std::string&) const;
    long long getBiasUpdateCount() const;
    std::string getBiasState();
    void flushOutputs();
    colvarmodule& module() { return *cvmodule; }
protected:
    const ColvarsPlugin::ColvarsForce& owner_;
    const OpenMM::System& system_;
    OpenMM::ContextImpl* currentContext_ = nullptr;
    double biasEnergy_ = 0.0;
    bool initialized_ = false;
    ColvarsPlugin::ColvarsTABFIntegrator* tabf_ = nullptr;
    double pendingAlpha_ = 0.0;
    double alphaLowerBound_ = 0.0, alphaUpperBound_ = std::numeric_limits<double>::infinity();
    bool alphaPending_ = false;
    bool restartAlphaPending_ = false;
    bool evaluated_ = false;
    bool usesPbc_ = false;
    std::array<OpenMM::Vec3, 3> lastBox_{};
    bool haveStep_ = false;
    bool requiresConsecutiveSteps_ = false;
    bool restartPending_ = false;
    long long currentStep_ = 0, lastUpdatedStep_ = 0, initialOpenmmStep_ = 0;
    cvm::step_number initialColvarsStep_ = 0;
    long long biasUpdateCount_ = 0;
    double lastTimestepFs_ = -1.0;
    std::mt19937_64 rng_;
    std::normal_distribution<double> gaussian_{0.0, 1.0};
};
#endif
