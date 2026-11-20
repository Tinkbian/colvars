// Adapted from Colvars PR 979. See ../UPSTREAM.md and ../LICENSE.
#include "colvarproxy_openmm.h"
#include "ColvarsTABFIntegrator.h"
#include "colvarcomp.h"
#include "colvarfile.h"
#include "colvaratoms.h"
#include "colvar.h"
#include "colvarbias.h"
#include "colvarbias_restraint.h"
#include "openmm/Context.h"
#include "openmm/Integrator.h"
#include "openmm/OpenMMException.h"
#include "openmm/System.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/ForceImpl.h"
#include "openmm/kernels.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
using namespace OpenMM;
using namespace ColvarsPlugin;
namespace {
bool hasMonteCarloBarostat(const ContextImpl& context) {
    // All built-in MC barostats, including the RPMD plugin, request this
    // kernel. ForceImpls exist before OpenMM initializes any individual force.
    for (auto* impl : context.getForceImpls()) {
        const auto kernels = impl->getKernelNames();
        if (std::find(kernels.begin(), kernels.end(), ApplyMonteCarloBarostatKernel::Name()) != kernels.end())
            return true;
    }
    return false;
}

std::recursive_mutex& colvarsMutex() {
    // Keep the mutex valid during Python/shared-library teardown.
    static auto* mutex = new std::recursive_mutex;
    return *mutex;
}
bool starts_with_record(const std::string& line)
{
    return line.rfind("ATOM  ", 0) == 0 || line.rfind("HETATM", 0) == 0;
}

double parse_pdb_double(const std::string& line, size_t offset, size_t count)
{
    if (line.size() < offset + count) {
        throw OpenMMException("PDB record is shorter than expected");
    }
    return std::stod(line.substr(offset, count));
}

char normalized_pdb_field(const std::string& field)
{
    if (field.empty()) {
        return '\0';
    }
    return static_cast<char>(std::toupper(static_cast<unsigned char>(field[0])));
}

double pdb_field_value(const std::string& line, char field)
{
    switch (field) {
        case 'O':
            return parse_pdb_double(line, 54, 6);
        case 'B':
            return parse_pdb_double(line, 60, 6);
        case 'X':
            return parse_pdb_double(line, 30, 8);
        case 'Y':
            return parse_pdb_double(line, 38, 8);
        case 'Z':
            return parse_pdb_double(line, 46, 8);
        default:
            throw OpenMMException("Unsupported PDB selector field");
    }
}

bool pdb_field_matches(const std::string& line, char field, double requestedValue)
{
    if (field == '\0') {
        return true;
    }
    const double value = pdb_field_value(line, field);
    if (requestedValue != 0.0) {
        return value == requestedValue;
    }
    return value != 0.0;
}

cvm::atom_pos pdb_position_angstrom(const std::string& line)
{
    return cvm::atom_pos(parse_pdb_double(line, 30, 8),
                         parse_pdb_double(line, 38, 8),
                         parse_pdb_double(line, 46, 8));
}

}

OpenMMColvarsLock::OpenMMColvarsLock(colvarproxy* active) : lock_(colvarsMutex()) {
    if (active) colvarmodule::proxy_static = active;
}

colvarproxy_openmm::colvarproxy_openmm(const ColvarsForce& owner, const System& system)
    : owner_(owner), system_(system),
      rng_(owner.getRandomSeed() == 0 ? std::random_device{}() : owner.getRandomSeed()) {
    OpenMMColvarsLock lock(this);
    version_int = get_version_from_string("2026-11-18");
    engine_name_ = "OpenMM";
    cvmodule = new colvarmodule(this);
    set_output_prefix(owner.getOutputPrefix());
    set_restart_output_prefix(owner.getOutputPrefix());
    if (!owner.getRestartInputPrefix().empty()) set_input_prefix(owner.getRestartInputPrefix());
    set_default_restart_frequency(0);
    add_config(owner.getConfigSource() == ColvarsForce::ConfigSource::File ? "configfile" : "config", owner.getConfig());
    usesPbc_ = owner.getPbcMode() == ColvarsForce::PbcMode::Enabled;
    if (owner.getPbcMode() == ColvarsForce::PbcMode::Auto) {
        for (int i = 0; i < system.getNumForces(); ++i) {
            const Force& f = system.getForce(i);
            if (&f != &owner && f.usesPeriodicBoundaryConditions()) usesPbc_ = true;
        }
    }
}

colvarproxy_openmm::~colvarproxy_openmm() {
    OpenMMColvarsLock lock(this);
    // Delete under the same lock used for evaluation. The base proxy destructor
    // skips this now-null pointer, preventing concurrent feature-table teardown.
    delete cvmodule;
    cvmodule = nullptr;
}

void colvarproxy_openmm::initializeColvars(ContextImpl& context) {
    OpenMMColvarsLock lock(this);
    if (initialized_) return;
    currentContext_ = &context;
    tabf_ = dynamic_cast<ColvarsTABFIntegrator*>(&context.getIntegrator());
    const double timestep = 1000.0 * context.getIntegrator().getStepSize();
    if (!std::isfinite(timestep) || (tabf_ && timestep <= 0))
        throw OpenMMException("Colvars requires a finite timestep and positive dt for TABF");
    set_integration_timestep(timestep);
    if (owner_.getNumParticleCharges() != 0 && owner_.getNumParticleCharges() != system_.getNumParticles())
        throw OpenMMException("Colvars particle charges must contain exactly one entry per System particle");
    int result = set_unit_system(owner_.getUnitSystem(), false);
    result |= set_target_temperature(owner_.getTargetTemperature());
    result |= parse_module_config();
    result |= setup();
    if (result || cvmodule->get_error()) throw OpenMMException("Colvars initialization: " + get_error_msgs());
    // State mutation and fresh force evaluation must be separable for OpenMM.
    int alphaDrivers = 0;
    for (auto* cv : *cvmodule->variables()) {
        if (cv->get_time_step_factor() != 1)
            throw OpenMMException("OpenMM Colvars requires timeStepFactor 1 on every colvar");
        if (cv->is_enabled(colvardeps::f_cv_extended_Lagrangian) && !tabf_)
            throw OpenMMException("OpenMM extendedLagrangian requires ColvarsTABFIntegrator");
        const bool alpha = cv->num_cvcs() == 1 && cv->get_cvcs()[0]->function_type() == "tabfAlpha";
        if (alpha) {
            if (!cv->is_enabled(colvardeps::f_cv_extended_Lagrangian) || ++alphaDrivers > 1)
                throw OpenMMException("TABF requires exactly one extendedLagrangian tabfAlpha driver");
            if (cv->is_enabled(colvardeps::f_cv_lower_boundary)) {
                alphaLowerBound_ = cv->lower_boundary.real_value;
                if (!std::isfinite(alphaLowerBound_) || alphaLowerBound_ <= 0)
                    throw OpenMMException("TABF lowerBoundary must be finite and positive");
            }
            if (cv->is_enabled(colvardeps::f_cv_upper_boundary)) {
                alphaUpperBound_ = cv->upper_boundary.real_value;
                if (!std::isfinite(alphaUpperBound_) || alphaUpperBound_ <= alphaLowerBound_)
                    throw OpenMMException("Invalid TABF upperBoundary");
            }
        }
        if (cv->is_enabled(colvardeps::f_cv_total_force_calc) &&
            !(tabf_ && alpha))
            throw OpenMMException("OpenMM Colvars does not supply total atomic forces for conventional ABF");
    }
    if (tabf_ && alphaDrivers != 1)
        throw OpenMMException("ColvarsTABFIntegrator requires exactly one extendedLagrangian tabfAlpha driver");
    if (cvmodule->scripted_forces())
        throw OpenMMException("OpenMM Colvars does not support scripted forces");
    const bool barostat = hasMonteCarloBarostat(context);
    if (tabf_ && barostat)
        throw OpenMMException("OpenMM TABF does not support Monte Carlo barostats");
    std::vector<colvarbias*> adaptiveBiases;
    for (auto* bias : cvmodule->biases) {
        if (bias->get_time_step_factor() != 1)
            throw OpenMMException("OpenMM Colvars requires timeStepFactor 1 on every bias");
        if (!bias->supports_state_free_evaluation())
            throw OpenMMException("OpenMM Colvars bias does not support repeat-safe evaluation: " + bias->bias_type);
        const auto* restraint = dynamic_cast<const colvarbias_restraint_moving*>(bias);
        if (!restraint || restraint->has_moving_parameters() || bias->is_enabled(colvardeps::f_cvb_calc_ti_samples)) {
            adaptiveBiases.push_back(bias);
            if (barostat)
                throw OpenMMException("OpenMM Colvars supports only static restraints with a Monte Carlo barostat: "
                                      "trial evaluations cannot commit adaptive history for bias " + bias->name);
        }
    }
    requiresConsecutiveSteps_ = tabf_ || !adaptiveBiases.empty();
    cvm::restart_info restart;
    result = cvmodule->setup_input(&restart);
    if (result || cvmodule->get_error()) throw OpenMMException("Colvars initialization: " + get_error_msgs());
    restartPending_ = cvmodule->has_loaded_state();
    if (restartPending_) {
        // The first evaluation will reuse the saved bias. A partially
        // matched file must not silently initialize adaptive histories to zero.
        if (!cvmodule->variables()->empty() && restart.variables.empty())
            throw OpenMMException("Colvars restart contains no matching collective variable");
        for (auto* bias : adaptiveBiases) {
            if (std::find(restart.biases.begin(), restart.biases.end(), bias) == restart.biases.end())
                throw OpenMMException("Colvars restart is missing adaptive bias " + bias->name);
            for (auto* cv : *bias->variables())
                if (std::find(restart.variables.begin(), restart.variables.end(), cv) == restart.variables.end())
                    throw OpenMMException("Colvars restart is missing collective variable " + cv->name +
                                          " for adaptive bias " + bias->name);
        }
        if (restart.has_timestep) set_integration_timestep(restart.timestep);
        if (tabf_ && (!restart.has_timestep || std::abs(restart.timestep-timestep) > 1e-6*timestep ||
                     restart.variables.size() != cvmodule->variables()->size()))
            throw OpenMMException("TABF restart requires matching timestep and all collective variables");
    }
    result = cvmodule->setup_output();
    if (result || cvmodule->get_error()) throw OpenMMException("Colvars initialization: " + get_error_msgs());
    initialized_ = true;
}

int colvarproxy_openmm::setup() {
    OpenMMColvarsLock lock(this);
    int result = colvarproxy::setup();
    return result | cvmodule->update_engine_parameters();
}

int colvarproxy_openmm::init_atom(int atomNumber) {
    const int id = check_atom_id(atomNumber);
    if (id < 0) return -1;
    auto found = std::find(atoms_ids.begin(), atoms_ids.end(), id);
    if (found != atoms_ids.end()) {
        const int index = static_cast<int>(found - atoms_ids.begin());
        ++atoms_refcount[index];
        return index;
    }
    const int index = add_atom_slot(id);
    atoms_masses[index] = system_.getParticleMass(id);
    atoms_charges[index] = owner_.getParticleCharge(id);
    return index;
}

int colvarproxy_openmm::check_atom_id(int atomNumber) {
    if (atomNumber <= 0 || atomNumber > system_.getNumParticles())
        throw OpenMMException("Colvars atomNumbers are one-based: invalid atom " + std::to_string(atomNumber));
    return atomNumber - 1;
}

int colvarproxy_openmm::request_atom_charges() {
    if (owner_.getNumParticleCharges() != system_.getNumParticles())
        throw OpenMMException("Charge-dependent Colvars components require setParticleCharges() before Context creation, with one fixed charge in elementary-charge units per System particle");
    return COLVARS_OK;
}

int colvarproxy_openmm::set_unit_system(const std::string& name, bool checkOnly) {
    if (name != owner_.getUnitSystem()) {
        if (checkOnly) return COLVARS_ERROR;
        throw OpenMMException("Colvars configuration units must match ColvarsForce.setUnitSystem()");
    }
    if (!checkOnly) {
        units = name;
        angstrom_value_ = name == "real" ? 1.0 : 0.1;
        kcal_mol_value_ = name == "real" ? 1.0 : 4.184;
        boltzmann_ = 0.00831446261815324 / energyToKJ();
    }
    return COLVARS_OK;
}

void colvarproxy_openmm::request_total_force(bool yesno) {
    if (yesno && !tabf_)
        throw OpenMMException("OpenMM Colvars cannot supply total atomic forces for ABF");
    // The required projected force is available only for the external alpha.
    // Validate the individual requesting CVs once parsing has finished.
    total_force_requested = yesno;
}

int colvarproxy_openmm::get_tabf_alpha(cvm::real* alpha) {
    if (!tabf_) throw OpenMMException("tabfAlpha requires ColvarsTABFIntegrator");
    *alpha = tabf_->getAlpha();
    return COLVARS_OK;
}
bool colvarproxy_openmm::replaying_colvars_endpoint() const {
    return tabf_ && tabf_->isCollectingBias() && haveStep_ && currentStep_ == lastUpdatedStep_;
}
int colvarproxy_openmm::get_dE_dtabf_alpha(cvm::real* derivative) {
    if (!tabf_) throw OpenMMException("tabfAlpha requires ColvarsTABFIntegrator");
    *derivative = tabf_->getAlphaDerivative() / energyToKJ();
    return COLVARS_OK;
}
int colvarproxy_openmm::set_tabf_alpha(cvm::real alpha) {
    if (!std::isfinite(alpha) || alpha <= 0) throw OpenMMException("Invalid TABF alpha");
    pendingAlpha_ = alpha;
    alphaPending_ = true;
    return COLVARS_OK;
}
int colvarproxy_openmm::send_tabf_alpha() {
    if (alphaPending_) {
        if (!initialized_) restartAlphaPending_ = true;
        else tabf_->setNextAlpha(pendingAlpha_);
        alphaPending_ = false;
    }
    return COLVARS_OK;
}

std::string colvarproxy_openmm::get_integrator_state() const {
    std::ostringstream out;
    out << rng_ << ' ' << gaussian_;
    std::string result = out.str();
    std::replace(result.begin(), result.end(), ' ', ',');
    return result;
}
void colvarproxy_openmm::set_integrator_state(const std::string& state) {
    std::string decoded = state;
    std::replace(decoded.begin(), decoded.end(), ',', ' ');
    std::istringstream in(decoded);
    if (!(in >> rng_ >> gaussian_)) throw OpenMMException("Invalid or missing Colvars endpoint RNG state");
}

void colvarproxy_openmm::log(const std::string& message) {
    std::istringstream lines(message);
    std::string line;
    while (std::getline(lines, line)) std::cout << "colvars: " << line << '\n';
}
void colvarproxy_openmm::error(const std::string& message) {
    add_error_msg(message);
    throw OpenMMException("Colvars-OpenMM: " + message);
}
cvm::real colvarproxy_openmm::rand_gaussian() { return gaussian_(rng_); }

void colvarproxy_openmm::validateTabfStart(double alpha, long long openmmStep) {
    OpenMMColvarsLock lock(this);
    if (restartPending_ && openmmStep != cvmodule->it_restart)
        throw OpenMMException("TABF requires a matching OpenMM checkpoint and Colvars endpoint step");
    if (haveStep_ && openmmStep != lastUpdatedStep_)
        throw OpenMMException("The OpenMM step count and the committed Colvars TABF endpoint differ");
    if (!std::isfinite(alpha) || alpha <= 0)
        throw OpenMMException("TABF alpha must be finite and positive");
    if ((restartAlphaPending_ || haveStep_) && std::abs(alpha - pendingAlpha_) >
            1e-12 * std::max(1.0, std::abs(pendingAlpha_)))
        throw OpenMMException("TABF alpha differs between the Colvars state and OpenMM checkpoint or integrator");
    if (!haveStep_ && !restartPending_ && (alpha < alphaLowerBound_ || alpha > alphaUpperBound_))
        throw OpenMMException("Initial TABF alpha is outside the configured boundaries");
}

void colvarproxy_openmm::beginEvaluation(ContextImpl& context) {
    OpenMMColvarsLock lock(this);
    if (!initialized_) throw OpenMMException("Colvars proxy was not initialized");
    currentContext_ = &context;
    currentStep_ = context.getOwner().getStepCount();
    if (tabf_) {
        // OpenMM increments its step counter after the endpoint force call.
        // Label the accepted state with the step in the resulting checkpoint.
        if (tabf_->isCollectingBias()) {
            if (currentStep_ == std::numeric_limits<long long>::max())
                throw OpenMMException("TABF endpoint step counter overflow");
            ++currentStep_;
        } else {
            validateTabfStart(tabf_->getAlpha(), currentStep_);
        }
        if (total_force_requested) set_total_forces_valid();
    }
    if (haveStep_ && currentStep_ < lastUpdatedStep_)
        throw OpenMMException("OpenMM Colvars cannot move backwards in step count; recreate Context and load a Colvars restart");
    if (usesPbc_) {
        std::array<Vec3, 3> box;
        context.getPeriodicBoxVectors(box[0], box[1], box[2]);
        if (box != lastBox_) {
            const double length = lengthFromNm();
            const cvm::rvector a(length * box[0][0], 0.0, 0.0), b(length * box[1][0], length * box[1][1], 0.0);
            const cvm::rvector c(length * box[2][0], length * box[2][1], length * box[2][2]);
            // The reciprocal-cell calculation uses squared lengths. Reject
            // unrepresentable box heights before changing the saved bias.
            for (double height : {a.x, b.y, c.z}) {
                if (!(height > 0.0) || !std::isnormal(height * height))
                    throw OpenMMException("Cannot represent Colvars cell height at double precision");
            }
            cvm::system_boundary_conditions updated;
            if (b.x == 0.0 && c.x == 0.0 && c.y == 0.0)
                updated.set_boundaries(true, true, true, a, b, c);
            else if (!updated.set_triclinic_minimum_image(a, b, c))
                throw OpenMMException("Cannot reduce triclinic Colvars cell at double precision");
            boundaries_ = updated;
            lastBox_ = box;
        }
    }
    if (!haveStep_) {
        initialOpenmmStep_ = tabf_ ? 0 : currentStep_;
        initialColvarsStep_ = tabf_ ? 0 : (restartPending_ ? cvmodule->it_restart : currentStep_);
    }
    const double timestepFs = 1000.0 * context.getIntegrator().getStepSize();
    if (!std::isfinite(timestepFs))
        throw OpenMMException("OpenMM Colvars requires a finite integration timestep in fs");
    // A checkpoint can map a small OpenMM count onto a much later Colvars
    // step. Both the absolute and the relative step must fit in step_number.
    const auto elapsed = static_cast<std::uint64_t>(currentStep_) - static_cast<std::uint64_t>(initialOpenmmStep_);
    constexpr auto maxStep = std::numeric_limits<cvm::step_number>::max();
    if (elapsed > static_cast<std::uint64_t>(maxStep) ||
        initialColvarsStep_ > maxStep - static_cast<cvm::step_number>(elapsed))
        throw OpenMMException("OpenMM step count exceeds the representable Colvars step range");
    if (requiresConsecutiveSteps_ && haveStep_ && currentStep_ > lastUpdatedStep_ &&
        currentStep_ - 1 != lastUpdatedStep_)
        throw OpenMMException("History-dependent Colvars biases require evaluation on consecutive OpenMM steps; "
                              "include the Colvars force group on each integration step");
    cvmodule->it = initialColvarsStep_ + static_cast<cvm::step_number>(elapsed);
    if (timestepFs != lastTimestepFs_) {
        set_integration_timestep(timestepFs);
        if (cvmodule->update_engine_parameters())
            throw OpenMMException("Colvars could not update the integration timestep");
        lastTimestepFs_ = timestepFs;
    }
    biasEnergy_ = 0.0;
    auto& forces = *modify_atom_applied_forces();
    std::fill(forces.begin(), forces.end(), cvm::rvector(0.0,0.0,0.0));
}

double colvarproxy_openmm::evaluateColvars() {
    OpenMMColvarsLock lock(this);
    // The step saved in a restart has already updated its adaptive history.
    // Its first evaluation restores current forces without depositing again.
    const bool freshStep = !haveStep_ || currentStep_ != lastUpdatedStep_;
    const bool advance = tabf_ ? (tabf_->isCollectingBias() && freshStep) :
                         (freshStep && !restartPending_);
    if (!haveStep_ && advance && currentStep_ != 0 && !restartPending_) {
        // A caller can set the starting OpenMM step after creating the Context.
        cvmodule->set_initial_step(currentStep_);
        for (auto* bias : cvmodule->biases) {
            auto* restraint = dynamic_cast<colvarbias_restraint_moving*>(bias);
            if (restraint && restraint->has_moving_parameters())
                restraint->set_schedule_start_step(cvmodule->it);
        }
    }
    const int result = cvmodule->calc(advance);
    if (result || cvmodule->get_error()) throw OpenMMException("Colvars evaluation: " + get_error_msgs());
    evaluated_ = true;
    if (advance || !tabf_) {
        if (tabf_) send_tabf_alpha();
        haveStep_ = true;
        restartPending_ = false;
        restartAlphaPending_ = false;
        lastUpdatedStep_ = currentStep_;
    }
    if (advance) ++biasUpdateCount_;
    return biasEnergy_ * energyToKJ();
}

double colvarproxy_openmm::calculate(ContextImpl& context, const std::vector<Vec3>& positions) {
    OpenMMColvarsLock lock(this);
    beginEvaluation(context);
    auto& selected = *modify_atom_positions();
    for (size_t i = 0; i < atoms_ids.size(); ++i) {
        const Vec3& p = positions.at(atoms_ids[i]);
        selected[i] = lengthFromNm() * cvm::rvector(p[0],p[1],p[2]);
    }
    return evaluateColvars();
}

void colvarproxy_openmm::addForcesToOpenMM(std::vector<Vec3>& forces) const {
    const auto& applied = *get_atom_applied_forces();
    for (size_t i = 0; i < atoms_ids.size(); ++i)
        forces.at(atoms_ids[i]) += forceToOpenMM() * Vec3(applied[i].x,applied[i].y,applied[i].z);
}

double colvarproxy_openmm::getColvarValue(const std::string& name) const {
    OpenMMColvarsLock lock(const_cast<colvarproxy_openmm*>(this));
    if (!evaluated_) throw OpenMMException("Evaluate energy or forces before reading a Colvars value");
    auto* cv = cvmodule->colvar_by_name(name);
    if (!cv) throw OpenMMException("Unknown Colvars variable: " + name);
    if (cv->value().type() != colvarvalue::type_scalar)
        throw OpenMMException("getColvarValue requires a scalar variable");
    return cv->value().real_value;
}
std::string colvarproxy_openmm::getBiasState() {
    OpenMMColvarsLock lock(this);
    if (!haveStep_ && !restartPending_)
        throw OpenMMException("Evaluate energy or forces before saving Colvars state");
    std::string result;
    if (cvmodule->write_restart_string(result)) throw OpenMMException("Cannot serialize Colvars state");
    return result;
}
void colvarproxy_openmm::flushOutputs() {
    OpenMMColvarsLock lock(this);
    if (!haveStep_) throw OpenMMException("Evaluate energy or forces before flushing Colvars output");
    // Use the core end-of-run output path explicitly, while exceptions can be
    // reported to Python. Destructors must not perform fallible file I/O.
    if (post_run() || cvmodule->get_error())
        throw OpenMMException("Cannot flush Colvars output: " + get_error_msgs());
}
long long colvarproxy_openmm::getBiasUpdateCount() const {
    OpenMMColvarsLock lock(const_cast<colvarproxy_openmm*>(this));
    return biasUpdateCount_;
}

int colvarproxy_openmm::load_atoms_pdb(char const* filename,
                                      cvm::atom_group& atoms,
                                      std::string const& pdbField,
                                      double pdbFieldValue)
{
    const char field = normalized_pdb_field(pdbField);
    if (field == '\0') {
        return cvmodule->error("Error: load_atoms_pdb requires a PDB selector field.\n",
                               COLVARS_INPUT_ERROR);
    }

    const std::string filenameString = filename == nullptr ? std::string() : std::string(filename);
    std::ifstream input(colvars_file_path(filenameString.c_str()));
    if (!input) {
        return cvmodule->error("Error: cannot open PDB file \"" + filenameString + "\".\n",
                               COLVARS_FILE_ERROR);
    }

    auto modifyAtoms = atoms.get_atom_modifier();
    std::string line;
    int pdbAtomIndex = 0;
    while (std::getline(input, line)) {
        if (!starts_with_record(line)) {
            continue;
        }
        if (pdb_field_matches(line, field, pdbFieldValue)) {
            if (pdbAtomIndex >= system_.getNumParticles()) {
                return cvmodule->error(
                    "Error: PDB file \"" + filenameString + "\" selects atom number " +
                        std::to_string(pdbAtomIndex + 1) +
                        ", which is outside the OpenMM System.\n",
                    COLVARS_INPUT_ERROR);
            }
            if (atoms.is_enabled(colvardeps::f_ag_scalable)) {
                modifyAtoms.add_atom_id(pdbAtomIndex);
            } else {
                modifyAtoms.add_atom(cvm::atom_group::init_atom_from_proxy(this, pdbAtomIndex + 1));
            }
        }
        ++pdbAtomIndex;
    }
    return cvmodule->get_error() ? COLVARS_ERROR : COLVARS_OK;
}

int colvarproxy_openmm::load_coords_pdb(char const* filename,
                                       std::vector<cvm::atom_pos>& positions,
                                       std::vector<int> const& sortedIds,
                                       std::string const& pdbField,
                                       double pdbFieldValue)
{
    const char field = normalized_pdb_field(pdbField);
    const std::string filenameString = filename == nullptr ? std::string() : std::string(filename);
    int previousId = -1;
    for (int atomId : sortedIds) {
        if (atomId < 0 || atomId >= system_.getNumParticles()) {
            return cvmodule->error(
                "Error: PDB file \"" + filenameString + "\" requested atom number " +
                    std::to_string(atomId + 1) + ", which is outside the OpenMM System.\n",
                COLVARS_INPUT_ERROR);
        }
        if (atomId <= previousId) {
            return cvmodule->error(
                "Error: requested atom numbers for PDB file \"" + filenameString +
                    "\" must be strictly increasing without duplicates.\n",
                COLVARS_INPUT_ERROR);
        }
        previousId = atomId;
    }

    std::ifstream input(colvars_file_path(filenameString.c_str()));
    if (!input) {
        return cvmodule->error("Error: cannot open PDB file \"" + filenameString + "\".\n",
                               COLVARS_FILE_ERROR);
    }

    std::vector<cvm::atom_pos> selectedPositions;
    std::string line;
    int pdbAtomIndex = 0;
    size_t nextSortedId = 0;
    while (std::getline(input, line)) {
        if (!starts_with_record(line)) {
            continue;
        }

        bool useThisAtom = pdb_field_matches(line, field, pdbFieldValue);
        if (!sortedIds.empty()) {
            useThisAtom = nextSortedId < sortedIds.size() && sortedIds[nextSortedId] == pdbAtomIndex;
            if (useThisAtom) {
                ++nextSortedId;
            }
        }

        if (useThisAtom) {
            selectedPositions.push_back(angstrom_value_ * pdb_position_angstrom(line));
        }
        ++pdbAtomIndex;
    }

    if (!sortedIds.empty() && nextSortedId != sortedIds.size()) {
        return cvmodule->error("Error: PDB file \"" + filenameString +
                                   "\" does not contain requested atom number " +
                                   std::to_string(sortedIds[nextSortedId] + 1) + ".\n",
                               COLVARS_INPUT_ERROR);
    }
    if (!positions.empty() && positions.size() != selectedPositions.size()) {
        return cvmodule->error("Error: PDB coordinate count does not match requested positions.\n",
                               COLVARS_INPUT_ERROR);
    }

    positions = selectedPositions;
    return COLVARS_OK;
}
