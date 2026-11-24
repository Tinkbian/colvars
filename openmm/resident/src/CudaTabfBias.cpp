// Bind an already parsed Colvars bias to the scalar CUDA bias representation.
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "CudaTabfBias.h"
#include "colvarbias_abf.h"
#include "colvarbias_meta.h"
#include "openmm/OpenMMException.h"
#include <algorithm>
#include <cmath>
#include <limits>

using namespace ColvarsPlugin;

namespace {
int offset(const std::vector<double>& buffer) {
    if (buffer.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw OpenMM::OpenMMException("CUDA TABF bias buffer exceeds the device index range");
    return static_cast<int>(buffer.size());
}

template<class Grid>
bool supportedGrid(const Grid& grid) {
    return grid.nd == 1 && !grid.periodic[0] && grid.mult == 1 &&
           grid.nx[0] > 0 && grid.nx[0] <= std::numeric_limits<int>::max()/4 &&
           std::isfinite(grid.widths[0]) && grid.widths[0] > 0.0 &&
           std::isfinite(grid.lower_boundaries[0].real_value) &&
           std::isfinite(grid.upper_boundaries[0].real_value);
}

template<class Grid>
TabfBiasGrid gridSpec(const Grid& grid, const std::vector<double>& buffer) {
    return {grid.lower_boundaries[0].real_value, grid.upper_boundaries[0].real_value,
            grid.widths[0], grid.nx[0], offset(buffer)};
}

void appendCounts(std::vector<double>& buffer, const colvar_grid_count& grid) {
    for (auto n : grid.data) {
        if (n > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw OpenMM::OpenMMException("CUDA TABF count exceeds the Colvars ABF output range");
        buffer.push_back(static_cast<double>(n));
    }
}

void restoreCounts(const double* counts, colvar_grid_count& grid) {
    std::transform(counts, counts+grid.data.size(), grid.data.begin(),
                   [](double n) { return static_cast<size_t>(n); });
}

bool scalarBias(colvarbias& bias) {
    return bias.num_variables() == 1 && bias.get_time_step_factor() == 1 &&
           !bias.is_enabled(colvardeps::f_cvb_bypass_ext_lagrangian) &&
           !bias.is_enabled(colvardeps::f_cvb_scale_biasing_force) &&
           !bias.is_enabled(colvardeps::f_cvb_calc_ti_samples);
}
}

bool CudaTabfStateAccess::configure(colvarbias_abf* abf, colvarbias_meta* meta, std::string& why) {
    abf_ = nullptr;
    meta_ = nullptr;
    if (abf && (!scalarBias(*abf) || !supportedGrid(*abf->samples) || abf->shared_on ||
                abf->b_UI_estimator || abf->pabf_freq || abf->hide_Jacobian ||
                abf->samples->use_actual_value[0] ||
                abf->full_samples > static_cast<size_t>(std::numeric_limits<int>::max()) ||
                (abf->cap_force && !std::isfinite(abf->max_force[0])))) {
        why = "CUDA TABF requires scalar nonperiodic ABF with current system forces and local sampling";
        return false;
    }
    if (meta && (!scalarBias(*meta) || !meta->use_grids || meta->expand_grids || meta->keep_hills ||
                 meta->comm != colvarbias_meta::single_replica || meta->ebmeta ||
                 !supportedGrid(*meta->hills_energy) || meta->hills_energy->use_actual_value[0] ||
                 meta->colvar_sigmas.size() != 1 || !(meta->colvar_sigmas[0] > 0.0) ||
                 !std::isfinite(meta->colvar_sigmas[0]) || !std::isfinite(meta->hill_weight) ||
                 !std::isfinite(meta->hill_width) ||
                 (meta->well_tempered && (!(meta->bias_temperature > 0.0) || !std::isfinite(meta->bias_temperature))) ||
                 !meta->new_hill_freq || meta->new_hill_freq > static_cast<size_t>(std::numeric_limits<int>::max()) ||
                 !meta->grids_freq || meta->grids_freq > static_cast<size_t>(std::numeric_limits<int>::max()))) {
        why = "CUDA TABF requires single-replica scalar metadynamics on a fixed nonperiodic grid";
        return false;
    }
    if (abf && meta && abf->variables(0) != meta->variables(0)) {
        why = "CUDA TABF bias pair must act on the same collective variable";
        return false;
    }
    abf_ = abf;
    meta_ = meta;
    return true;
}

TabfBiasSpec CudaTabfStateAccess::pack(std::vector<double>& buffer, int maxNewSteps) const {
    if (maxNewSteps < 0) throw OpenMM::OpenMMException("Negative CUDA TABF batch size");
    TabfBiasSpec s{};
    s.status = offset(buffer);
    buffer.resize(buffer.size()+tbStatusSize);
    if (abf_) {
        auto& a = *abf_;
        s.abf = gridSpec(*a.samples, buffer);
        s.czar = a.z_samples != nullptr;
        s.updateBias = a.update_bias;
        s.applyBias = a.is_enabled(colvardeps::f_cvb_apply_force);
        s.abfStepZero = a.is_enabled(colvardeps::f_cvb_step_zero_data);
        s.minSamples = static_cast<int>(a.min_samples);
        s.fullSamples = static_cast<int>(a.full_samples);
        s.maxForce = a.cap_force ? a.max_force[0] : -1.0;
        appendCounts(buffer, *a.samples);
        buffer.insert(buffer.end(), a.gradients->data.begin(), a.gradients->data.end());
        if (s.czar) {
            appendCounts(buffer, *a.z_samples);
            buffer.insert(buffer.end(), a.z_gradients->data.begin(), a.z_gradients->data.end());
        }
    }
    if (meta_) {
        auto& m = *meta_;
        s.meta = gridSpec(*m.hills_energy, buffer);
        s.sigma = m.colvar_sigmas[0];
        s.hillWeight = m.hill_weight;
        s.wtEnergy = m.well_tempered ? m.bias_temperature*m.get_cvmodule()->proxy->boltzmann() : 0.0;
        s.edgeBins = 3.0*std::floor(m.hill_width)+1.0;
        s.newHillFreq = static_cast<int>(m.new_hill_freq);
        s.gridsFreq = static_cast<int>(m.grids_freq);
        s.metaStepZero = m.is_enabled(colvardeps::f_cvb_step_zero_data);
        s.hardLower = m.hills_energy->hard_lower_boundaries[0];
        s.hardUpper = m.hills_energy->hard_upper_boundaries[0];
        buffer.insert(buffer.end(), m.hills_energy->data.begin(), m.hills_energy->data.end());
        buffer.insert(buffer.end(), m.hills_energy_gradients->data.begin(), m.hills_energy_gradients->data.end());
        s.hills = offset(buffer);
        auto appendHill = [&](colvarbias_meta::hill& hill) {
            if (hill.it > 9007199254740991LL || hill.sigmas.size() != 1 ||
                !(hill.sigmas[0] > 0.0) || !std::isfinite(hill.sigmas[0]) ||
                !std::isfinite(hill.weight()) || !std::isfinite(hill.centers[0].real_value))
                throw OpenMM::OpenMMException("Unrepresentable hill in CUDA TABF restart history");
            buffer.insert(buffer.end(), {static_cast<double>(hill.it), hill.centers[0].real_value,
                                         hill.sigmas[0], hill.weight()});
        };
        for (auto it = m.hills_off_grid.begin(); it != m.new_hills_off_grid_begin; ++it)
            appendHill(*it);
        buffer[s.status+tbProjectedCount] = (offset(buffer)-s.hills)/4;
        for (auto it = m.new_hills_begin; it != m.hills.end(); ++it)
            appendHill(*it);
        s.firstNewHill = (offset(buffer)-s.hills)/4;
        buffer[s.status+tbHillCount] = s.firstNewHill;
        // Enough for any alignment between the absolute step and hill frequency.
        size_t slots = (static_cast<size_t>(maxNewSteps)+s.newHillFreq-1)/s.newHillFreq;
        if (slots > (static_cast<size_t>(std::numeric_limits<int>::max())-buffer.size())/4)
            throw OpenMM::OpenMMException("CUDA TABF hill buffer exceeds the device index range");
        s.hillCapacity = s.firstNewHill+static_cast<int>(slots);
        buffer.resize(buffer.size()+4*slots);
    }
    offset(buffer);
    return s;
}

void CudaTabfStateAccess::unpack(const TabfBiasSpec& s, const std::vector<double>& buffer) {
    const double* status = buffer.data()+s.status;
    const double extended = status[tbExtended];
    if (abf_) {
        auto& a = *abf_;
        const int n = s.abf.bins;
        const double* grids = buffer.data()+s.abf.offset;
        restoreCounts(grids, *a.samples);
        std::copy_n(grids+n, n, a.gradients->data.begin());
        if (s.czar) {
            restoreCounts(grids+2*n, *a.z_samples);
            std::copy_n(grids+3*n, n, a.z_gradients->data.begin());
        }
        if (a.b_integrate) a.pmf->set_div();
        a.colvar_values[0].real_value = extended;
        a.previous_colvar_forces[0].real_value = a.colvar_forces[0].real_value = status[tbAbfForce];
        a.bias_energy = status[tbAbfEnergy];
        a.output_prefix = a.get_cvmodule()->output_prefix();
        if (a.get_cvmodule()->num_biases_feature(colvardeps::f_cvb_calc_pmf) != 1)
            a.output_prefix += "." + a.name;
    }
    if (meta_) {
        auto& m = *meta_;
        const double* grids = buffer.data()+s.meta.offset;
        std::copy_n(grids, s.meta.bins, m.hills_energy->data.begin());
        std::copy_n(grids+s.meta.bins, s.meta.bins, m.hills_energy_gradients->data.begin());
        const int count = static_cast<int>(status[tbHillCount]);
        const int projected = static_cast<int>(status[tbProjectedCount]);
        if (count < s.firstNewHill || count > s.hillCapacity || projected < 0 || projected > count)
            throw OpenMM::OpenMMException("Invalid CUDA TABF metadynamics hill counters");
        m.hills.clear();
        m.hills_off_grid.clear();
        m.new_hills_begin = m.hills.end();
        m.new_hills_off_grid_begin = m.hills_off_grid.end();
        for (int i = 0; i < count; ++i) {
            const double* h = buffer.data()+s.hills+4*i;
            std::vector<colvarvalue> centers{colvarvalue(h[1])};
            const bool edge = m.hills_energy->bin_distance_from_boundaries(centers, true) < s.edgeBins;
            const auto step = static_cast<cvm::step_number>(h[0]);
            const bool newForTrajectory = i >= s.firstNewHill && step > m.last_hill_step;
            colvarbias_meta::hill hill(m.get_cvmodule(), step, h[3], centers, {h[2]});
            if (newForTrajectory) {
                if (m.b_hills_traj) m.hills_traj_os_buf << hill.output_traj();
                m.last_hill_step = step;
            }
            if (i >= projected) {
                m.hills.push_back(hill);
                if (m.new_hills_begin == m.hills.end()) m.new_hills_begin = m.hills.begin();
            }
            if (edge) {
                m.hills_off_grid.push_back(hill);
                if (i >= projected && m.new_hills_off_grid_begin == m.hills_off_grid.end())
                    m.new_hills_off_grid_begin = std::prev(m.hills_off_grid.end());
            }
        }
        m.colvar_values[0].real_value = extended;
        m.previous_colvar_forces[0].real_value = m.colvar_forces[0].real_value = status[tbMetaForce];
        m.bias_energy = status[tbMetaEnergy];
        m.has_data = true;
    }
}
