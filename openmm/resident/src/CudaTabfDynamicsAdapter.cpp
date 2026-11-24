#include "CudaTabfDynamicsAdapter.h"
#include "colvar.h"
#include "colvarproxy_openmm.h"
#include <algorithm>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>

using ColvarsPlugin::CudaTabfDynamicsAdapter;

TabfExtConfig CudaTabfDynamicsAdapter::config(const colvar& cv) {
    if (cv.value().type() != colvarvalue::type_scalar || !cv.is_enabled(colvardeps::f_cv_extended_Lagrangian) ||
        cv.get_time_step_factor() != 1 || cv.period != 0.0)
        throw std::invalid_argument("Device TABF requires a nonperiodic extended scalar at every MD step");
    return {cv.ext_mass, cv.ext_force_k, cv.ext_temperature*cv.cvmodule->proxy->boltzmann(),
            cv.ext_gamma, cv.lower_boundary.real_value, cv.upper_boundary.real_value,
            cv.is_enabled(colvardeps::f_cv_reflecting_lower_boundary),
            cv.is_enabled(colvardeps::f_cv_reflecting_upper_boundary), cv.is_enabled(colvardeps::f_cv_external)};
}

TabfExtState CudaTabfDynamicsAdapter::read(const colvar& cv, long long lastStep) {
    TabfExtState s{};
    s.noiseDt = cv.ext_noise_dt;
    s.noiseSigma = cv.ext_sigma;
    if (cv.x_ext.type() == colvarvalue::type_notset) return s;
    s.initialized = 1;
    s.lastStep = lastStep;
    s.x = cv.x_ext.real_value;
    s.v = cv.v_ext.real_value;
    s.previousX = cv.prev_x_ext.real_value;
    s.previousV = cv.prev_v_ext.real_value;
    s.actual = cv.endpoint_x.real_value;
    s.actualVelocity = cv.v_fdiff.real_value;
    s.systemForce = cv.ft_reported.real_value;
    s.auxBias = cv.fb.real_value;
    s.actualBias = cv.fb_actual.real_value;
    s.springEnergy = cv.potential_energy;
    s.kineticEnergy = cv.kinetic_energy;
    return s;
}

void CudaTabfDynamicsAdapter::write(colvar& cv, const TabfExtState& s, cvm::step_number relativeStep) {
    if (!s.initialized) return;
    const bool external = cv.is_enabled(colvardeps::f_cv_external);
    cv.x_ext = colvarvalue(s.x);
    cv.v_ext = colvarvalue(s.v);
    cv.prev_x_ext = colvarvalue(s.previousX);
    cv.prev_v_ext = colvarvalue(s.previousV);
    cv.endpoint_x = colvarvalue(s.actual);
    cv.x_reported = cv.prev_x_ext;
    cv.v_reported = cv.prev_v_ext;
    cv.x = colvarvalue(external ? s.x : s.actual);
    cv.x_old = cv.x;
    cv.v_fdiff = colvarvalue(s.actualVelocity);
    cv.fb.real_value = s.auxBias;
    cv.fb_actual.real_value = s.actualBias;
    cv.fr.real_value = external ? s.auxBias + s.actualBias : s.auxBias;
    cv.ft_reported.real_value = s.systemForce;
    cv.ft.real_value = external ? s.systemForce : 0.0;
    cv.f.real_value = external ? cv.fr.real_value : -s.systemForce + s.actualBias;
    if (cv.is_enabled(colvardeps::f_cv_subtract_applied_force)) cv.f_old = cv.f;
    cv.kinetic_energy = s.kineticEnergy;
    cv.potential_energy = s.springEnergy;
    cv.ext_noise_dt = s.noiseDt;
    cv.ext_sigma = s.noiseSigma;
    cv.after_restart = false;
    cv.prev_timestep = relativeStep;
}

bool CudaTabfDynamicsAdapter::readRng(const colvarproxy_openmm& proxy, TabfGaussianState& rng) {
#ifdef __GLIBCXX__
    // libstdc++ writes the 312 engine words and its cursor, followed by the
    // distribution's mean, deviation, spare flag and optional spare deviate.
    std::string encoded = proxy.get_integrator_state();
    std::replace(encoded.begin(), encoded.end(), ',', ' ');
    std::istringstream in(encoded);
    in.imbue(std::locale::classic());
    TabfGaussianState candidate{};
    for (auto& word : candidate.words)
        if (!(in >> word)) return false;
    double mean, deviation;
    if (!(in >> candidate.index >> mean >> deviation >> candidate.hasSpare) ||
        candidate.index < 0 || candidate.index > 312 || mean != 0 || deviation != 1 ||
        (candidate.hasSpare != 0 && candidate.hasSpare != 1)) return false;
    if (candidate.hasSpare && (!(in >> candidate.spare) || !std::isfinite(candidate.spare))) return false;
    in >> std::ws;
    if (!in.eof()) return false;
    rng = candidate;
    return true;
#else
    // The engine sequence is standard, but the normal_distribution algorithm
    // and its state encoding are implementation defined.
    (void)proxy;
    (void)rng;
    return false;
#endif
}

void CudaTabfDynamicsAdapter::writeRng(colvarproxy_openmm& proxy, const TabfGaussianState& rng) {
#ifdef __GLIBCXX__
    std::ostringstream out;
    out.imbue(std::locale::classic());
    for (auto word : rng.words) out << word << ',';
    out << rng.index << ",0,1," << rng.hasSpare;
    if (rng.hasSpare) out << ',' << std::setprecision(std::numeric_limits<double>::max_digits10) << rng.spare;
    proxy.set_integrator_state(out.str());
#else
    (void)proxy;
    (void)rng;
    throw std::logic_error("Device TABF random state requires a supported host distribution");
#endif
}
