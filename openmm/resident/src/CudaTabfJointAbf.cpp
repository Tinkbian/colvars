// Import and export the native Colvars 2D ABF grids at reporting boundaries.
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "CudaTabfJointAbf.h"
#include "colvarbias_abf.h"
#include "openmm/OpenMMException.h"
#include <algorithm>
#include <cmath>
#include <limits>

using namespace ColvarsPlugin;

namespace {
template<class Grid>
bool supportedGrid(const Grid& g, int mult) {
    if (g.nd != 2 || g.mult != mult || g.nx[0] <= 0 || g.nx[1] <= 0)
        return false;
    const size_t count = static_cast<size_t>(g.nx[0])*g.nx[1];
    if (count > static_cast<size_t>(std::numeric_limits<int>::max())/mult ||
        count*mult != g.data.size() || g.nxc[1] != mult || g.nxc[0] != g.nx[1]*mult)
        return false;
    for (int i = 0; i < 2; ++i) {
        if (g.periodic[i] || g.use_actual_value[i] ||
            !std::isfinite(g.lower_boundaries[i].real_value) ||
            !std::isfinite(g.widths[i]) || g.widths[i] <= 0)
            return false;
    }
    return true;
}

template<class Grid>
TabfJointGrid geometry(const Grid& g) {
    return {{g.lower_boundaries[0].real_value, g.lower_boundaries[1].real_value},
            {g.widths[0], g.widths[1]}, {g.nx[0], g.nx[1]}};
}
}

bool CudaTabfJointAbf::configure(colvarbias_abf* abf, const colvar* distance,
                                const colvar* alpha, std::string& why) {
    abf_ = nullptr;
    if (!abf) return true;
    why = "Device joint ABF needs one local, nonperiodic grid for the extended distance and alpha";
    if (abf->num_variables() != 2 || !distance || !alpha || distance == alpha) return false;
    firstAxis_ = abf->variables(0) == distance ? 0 : 1;
    if (abf->variables(0) != (firstAxis_ == 0 ? distance : alpha) ||
        abf->variables(1) != (firstAxis_ == 0 ? alpha : distance))
        return false;
    if (!abf->samples || !abf->gradients || !supportedGrid(*abf->samples, 1) ||
        !supportedGrid(*abf->gradients, 2) || abf->samples->nx != abf->gradients->nx ||
        (abf->pmf && !supportedGrid(*abf->pmf, 1)))
        return false;
    if (abf->shared_on || abf->pabf_freq || abf->b_UI_estimator || abf->b_CZAR_estimator ||
        abf->hide_Jacobian || abf->get_time_step_factor() != 1 ||
        abf->is_enabled(colvardeps::f_cvb_bypass_ext_lagrangian) ||
        abf->is_enabled(colvardeps::f_cvb_scale_biasing_force) ||
        abf->is_enabled(colvardeps::f_cvb_calc_ti_samples) ||
        abf->full_samples > static_cast<size_t>(std::numeric_limits<int>::max())) {
        why = "Device joint ABF supports ordinary current-step ABF with CZAR and UI estimators disabled";
        return false;
    }
    for (int i = 0; i < 2; ++i) {
        if (!abf->variables(i)->is_enabled(colvardeps::f_cv_extended_Lagrangian) ||
            (abf->cap_force && (!std::isfinite(abf->max_force[i]) || abf->max_force[i] < 0.0)))
            return false;
    }
    abf_ = abf;
    why.clear();
    return true;
}

TabfJointSpec CudaTabfJointAbf::pack(std::vector<double>& buffer) const {
    TabfJointSpec s{};
    if (!abf_) return s;
    auto& a = *abf_;
    const size_t required = tjAbfStatusSize + a.samples->data.size() + a.gradients->data.size() +
                            (a.pmf ? a.pmf->data.size() : 0);
    const size_t limit = std::numeric_limits<int>::max();
    if (required > limit || buffer.size() > limit-required)
        throw OpenMM::OpenMMException("Device joint ABF buffer exceeds the CUDA index range");
    s.axis[0] = firstAxis_;
    s.axis[1] = 1-firstAxis_;
    s.updateBias = a.update_bias;
    s.applyBias = a.is_enabled(colvardeps::f_cvb_apply_force);
    s.stepZero = a.is_enabled(colvardeps::f_cvb_step_zero_data);
    s.minSamples = static_cast<int>(a.min_samples);
    s.fullSamples = static_cast<int>(a.full_samples);
    for (int i = 0; i < 2; ++i) s.maxForce[i] = a.cap_force ? a.max_force[i] : -1.0;

    s.status = static_cast<int>(buffer.size());
    buffer.resize(buffer.size()+tjAbfStatusSize);
    s.samples = geometry(*a.samples);
    s.counts = static_cast<int>(buffer.size());
    for (auto n : a.samples->data) {
        // Colvars' gradient grid uses signed int counts to format its output.
        if (n > limit) throw OpenMM::OpenMMException("Joint ABF count exceeds the Colvars output range");
        buffer.push_back(static_cast<double>(n));
    }
    s.gradients = static_cast<int>(buffer.size());
    buffer.insert(buffer.end(), a.gradients->data.begin(), a.gradients->data.end());
    if (a.pmf) {
        s.pmf = geometry(*a.pmf);
        s.potential = static_cast<int>(buffer.size());
        const double minimum = a.pmf->minimum_value();
        for (double energy : a.pmf->data) buffer.push_back(energy-minimum);
    }
    return s;
}

void CudaTabfJointAbf::unpack(const TabfJointSpec& s, const std::vector<double>& buffer) {
    if (!abf_) return;
    auto& a = *abf_;
    const double* counts = buffer.data()+s.counts;
    std::transform(counts, counts+a.samples->data.size(), a.samples->data.begin(),
                   [](double n) { return static_cast<size_t>(n); });
    std::copy_n(buffer.data()+s.gradients, a.gradients->data.size(), a.gradients->data.begin());

    // For ordinary 2D ABF, integrate() recomputes the complete divergence
    // from counts and sums on output. No fields from an intermediate force
    // update enter that solve. The last solved PMF remains live between writes.
    const double* state = buffer.data()+s.status;
    const double extended[] = {state[tjAbfDistance], state[tjAbfAlpha]};
    const double applied[] = {state[tjAbfDistanceForce], state[tjAbfAlphaForce]};
    for (int i = 0; i < 2; ++i) {
        a.colvar_values[i].real_value = extended[s.axis[i]];
        a.previous_colvar_forces[i].real_value = a.colvar_forces[i].real_value = applied[s.axis[i]];
        a.force_bin[i] = a.bin[i] = a.samples->value_to_bin_scalar(a.colvar_values[i], i);
    }
    a.bias_energy = state[tjAbfEnergy];
    a.output_prefix = a.get_cvmodule()->output_prefix();
    if (a.get_cvmodule()->num_biases_feature(colvardeps::f_cvb_calc_pmf) != 1)
        a.output_prefix += "."+a.name;
}
