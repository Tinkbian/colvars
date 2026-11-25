// State transfer for the two-dimensional CUDA TABF metadynamics grid.
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "CudaTabfJointMeta.h"
#include "colvarbias_meta.h"
#include "colvargrid.h"
#include "colvarproxy.h"
#include "openmm/OpenMMException.h"
#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>

using namespace ColvarsPlugin;
using OpenMM::OpenMMException;

namespace {
constexpr int maxGridCells = 8*1024*1024; // three doubles per bin (192 MiB)
constexpr long long maxExactStep = 9007199254740991LL;

int reserveBuffer(std::vector<double>& buf, size_t elements) {
    const size_t bound = std::numeric_limits<int>::max();
    if (buf.size()>bound || elements>bound-buf.size())
        throw OpenMMException("Joint CUDA TABF grid or history exceeds the device index range");
    int offset = static_cast<int>(buf.size());
    buf.resize(buf.size()+elements);
    return offset;
}

bool validSigma(double x) {
    return x>0 && std::isnormal(x*x);
}

int readCount(double x, int maximum) {
    if (!(x>=0 && x<=maximum && std::floor(x)==x))
        throw OpenMMException("Invalid joint CUDA TABF grid or hill counters");
    return static_cast<int>(x);
}
}

bool CudaTabfJointMeta::configure(colvarbias_meta* m, std::string& why) {
    meta_ = nullptr;
    if (!m) return true;
    why = "Joint CUDA TABF requires single-replica 2D grid metadynamics on two nonperiodic extended coordinates, with expansion only on the first axis";
    if (m->num_variables()!=2 || m->get_time_step_factor()!=1 || !m->use_grids ||
        m->comm!=colvarbias_meta::single_replica || m->keep_hills || m->ebmeta ||
        !m->is_enabled(colvardeps::f_cvb_active) || !m->is_enabled(colvardeps::f_cvb_apply_force) ||
        !m->is_enabled(colvardeps::f_cvb_history_dependent) ||
        m->is_enabled(colvardeps::f_cvb_bypass_ext_lagrangian) ||
        m->is_enabled(colvardeps::f_cvb_scale_biasing_force) || m->is_enabled(colvardeps::f_cvb_calc_ti_samples) ||
        m->is_enabled(colvardeps::f_cvb_output_acc_work) || !m->hills_energy || !m->hills_energy_gradients ||
        !m->new_hill_freq || m->new_hill_freq>std::numeric_limits<int>::max() ||
        !m->grids_freq || m->grids_freq>std::numeric_limits<int>::max() ||
        !(m->hill_weight>0) || !std::isfinite(m->hill_weight) ||
        !(m->hill_width>=0 && m->hill_width<(std::numeric_limits<int>::max()-1)/3) ||
        m->colvar_sigmas.size()!=2 || !validSigma(m->colvar_sigmas[0]) || !validSigma(m->colvar_sigmas[1]) ||
        (m->well_tempered && !(m->bias_temperature>0 && std::isfinite(m->bias_temperature))) ||
        m->variables(1)->expand_boundaries) return false;
    const auto& grid = *m->hills_energy;
    const auto& grad = *m->hills_energy_gradients;
    if (grid.nd!=2 || grad.nd!=2 || grid.mult!=1 || grad.mult!=2 || grid.nx!=grad.nx ||
        grid.widths!=grad.widths || grid.nx[0]<=0 || grid.nx[1]<=0 ||
        grid.nx[0]>maxGridCells/grid.nx[1]) return false;
    for (int k=0;k<2;++k) {
        if (grid.periodic[k] || grad.periodic[k] || grid.use_actual_value[k] || grad.use_actual_value[k] ||
            !m->variables(k)->is_enabled(colvardeps::f_cv_extended_Lagrangian) ||
            !std::isfinite(grid.lower_boundaries[k].real_value) || !std::isfinite(grid.upper_boundaries[k].real_value) ||
            !(grid.upper_boundaries[k].real_value>grid.lower_boundaries[k].real_value) ||
            grid.lower_boundaries[k].real_value!=grad.lower_boundaries[k].real_value ||
            grid.upper_boundaries[k].real_value!=grad.upper_boundaries[k].real_value ||
            !(grid.widths[k]>0 && std::isfinite(grid.widths[k])) ||
            (m->variables(k)->expand_boundaries && grid.widths[k]!=m->variables(k)->width)) return false;
    }
    meta_ = m;
    why.clear();
    return true;
}

TabfJointMetaSpec CudaTabfJointMeta::pack(std::vector<double>& buf, int maxNewSteps) const {
    TabfJointMetaSpec s{};
    if (!meta_) return s;
    if (maxNewSteps<0) throw OpenMMException("Negative joint CUDA TABF batch size");
    auto& m = *meta_;
    const auto& grid = *m.hills_energy;
    const auto& grad = *m.hills_energy_gradients;
    s.hillFreq = static_cast<int>(m.new_hill_freq);
    s.gridFreq = static_cast<int>(m.grids_freq);
    s.edgeBins = 3*static_cast<int>(std::floor(m.hill_width))+1;
    s.weight = m.hill_weight;
    s.wtEnergy = m.well_tempered ? m.bias_temperature*m.get_cvmodule()->proxy->boltzmann() : 0.0;
    s.stepZero = m.is_enabled(colvardeps::f_cvb_step_zero_data);
    s.growLower = m.variables(0)->expand_boundaries && !m.variables(0)->is_enabled(colvardeps::f_cv_hard_lower_boundary);
    s.growUpper = m.variables(0)->expand_boundaries && !m.variables(0)->is_enabled(colvardeps::f_cv_hard_upper_boundary);
    for (int k=0;k<2;++k) {
        s.width[k]=grid.widths[k]; s.sigma[k]=m.colvar_sigmas[k];
        s.hardLower[k]=grid.hard_lower_boundaries[k]; s.hardUpper[k]=grid.hard_upper_boundaries[k];
    }
    s.lower1 = grid.lower_boundaries[1].real_value;
    s.upper1 = grid.upper_boundaries[1].real_value;
    s.columns = grid.nx[1];
    const int active = grid.nx[0];
    int maxRows = maxGridCells/s.columns;
    if (active>maxRows) throw OpenMMException("Joint CUDA TABF exceeds the 8 million cell limit");
    // Repack at existing output/batch boundaries. Between boundaries all free
    // rows can be used on either side, even if the entire excursion is one-way.
    int spare = (s.growLower || s.growUpper) ? std::min(maxRows-active,std::max(256,active/2)) : 0;
    s.rows = active+spare;
    s.status = reserveBuffer(buf,tjStatusSize);
    s.grid = reserveBuffer(buf,3ULL*s.rows*s.columns);
    buf[s.status+tjRows]=active;
    buf[s.status+tjLower]=grid.lower_boundaries[0].real_value;
    buf[s.status+tjUpper]=grid.upper_boundaries[0].real_value;
    for (size_t i=0;i<grid.data.size();++i) {
        buf[s.grid+3*i]=grid.data[i];
        buf[s.grid+3*i+1]=grad.data[2*i]; buf[s.grid+3*i+2]=grad.data[2*i+1];
    }
    s.hills = reserveBuffer(buf,0);
    auto append = [&](colvarbias_meta::hill& hill, bool edge) {
        if (hill.it<0 || hill.it>maxExactStep || hill.centers.size()!=2 || hill.sigmas.size()!=2 ||
            !std::isfinite(hill.centers[0].real_value) || !std::isfinite(hill.centers[1].real_value) ||
            !validSigma(hill.sigmas[0]) || !validSigma(hill.sigmas[1]) || !std::isfinite(hill.weight()))
            throw OpenMMException("Unrepresentable hill in joint CUDA metadynamics restart");
        const int o = reserveBuffer(buf,thSize);
        buf[o+thStep]=static_cast<double>(hill.it);
        buf[o+thCenter0]=hill.centers[0].real_value; buf[o+thCenter1]=hill.centers[1].real_value;
        buf[o+thSigma0]=hill.sigmas[0]; buf[o+thSigma1]=hill.sigmas[1];
        buf[o+thWeight]=hill.weight(); buf[o+thEdge]=edge;
    };
    for (auto h=m.hills_off_grid.begin(); h!=m.new_hills_off_grid_begin; ++h) append(*h,true);
    buf[s.status+tjProjected]=(buf.size()-s.hills)/thSize;
    auto edge = m.new_hills_off_grid_begin;
    for (auto h=m.new_hills_begin; h!=m.hills.end(); ++h) {
        bool onEdge = edge!=m.hills_off_grid.end() && edge->it==h->it && edge->replica==h->replica;
        append(*h,onEdge);
        if (onEdge) ++edge;
    }
    if (edge!=m.hills_off_grid.end()) throw OpenMMException("Inconsistent pending joint metadynamics boundary hills");
    s.initialHills = static_cast<int>((buf.size()-s.hills)/thSize);
    buf[s.status+tjCount]=s.initialHills;
    size_t newSlots = (static_cast<size_t>(maxNewSteps)+s.hillFreq-1)/s.hillFreq;
    reserveBuffer(buf,newSlots*thSize);
    s.hillCapacity = s.initialHills+static_cast<int>(newSlots);
    return s;
}

void CudaTabfJointMeta::unpack(const TabfJointMetaSpec& s, const std::vector<double>& buf) {
    if (!meta_) return;
    auto& m = *meta_;
    auto& grid = *m.hills_energy;
    auto& grad = *m.hills_energy_gradients;
    const double* st = buf.data()+s.status;
    int active = readCount(st[tjRows],s.rows), first = readCount(st[tjFirstRow],s.rows-1);
    int count = readCount(st[tjCount],s.hillCapacity), projected = readCount(st[tjProjected],count);
    if (active<grid.nx[0] || count<s.initialHills || !std::isfinite(st[tjLower]) || !std::isfinite(st[tjUpper]))
        throw OpenMMException("Invalid joint CUDA TABF grid or history");
    if (active!=grid.nx[0]) {
        grid.lower_boundaries[0].real_value=grad.lower_boundaries[0].real_value=st[tjLower];
        grid.upper_boundaries[0].real_value=grad.upper_boundaries[0].real_value=st[tjUpper];
        const std::vector<int> shape{active,s.columns};
        if (grid.setup(shape,0.0,1) || grad.setup(shape,0.0,2))
            throw OpenMMException("Could not restore expanded joint CUDA metadynamics grids");
        grid.has_data = grad.has_data = true;
    }
    for (int i=0;i<active;++i) {
        int storage = ((first+i)%s.rows)*s.columns;
        for (int j=0;j<s.columns;++j) {
            int k = i*s.columns+j;
            const double* cell = buf.data()+s.grid+3*(storage+j);
            grid.data[k]=cell[0]; grad.data[2*k]=cell[1]; grad.data[2*k+1]=cell[2];
        }
    }
    m.hills.clear(); m.hills_off_grid.clear();
    m.new_hills_begin=m.hills.end(); m.new_hills_off_grid_begin=m.hills_off_grid.end();
    for (int i=0;i<count;++i) {
        const double* h = buf.data()+s.hills+i*thSize;
        auto step = static_cast<cvm::step_number>(h[thStep]);
        colvarbias_meta::hill hill(m.get_cvmodule(),step,h[thWeight],
                                  {colvarvalue(h[thCenter0]),colvarvalue(h[thCenter1])},
                                  {h[thSigma0],h[thSigma1]});
        if (i>=s.initialHills && step>m.last_hill_step) {
            if (m.b_hills_traj) m.hills_traj_os_buf<<hill.output_traj();
            m.last_hill_step=step;
        }
        if (i>=projected) {
            m.hills.push_back(hill);
            if (m.new_hills_begin==m.hills.end()) m.new_hills_begin=std::prev(m.hills.end());
        }
        if (h[thEdge]) {
            m.hills_off_grid.push_back(hill);
            if (i>=projected && m.new_hills_off_grid_begin==m.hills_off_grid.end())
                m.new_hills_off_grid_begin=std::prev(m.hills_off_grid.end());
        }
    }
    for (int k=0;k<2;++k) {
        m.colvar_values[k].real_value=st[tjValue0+k];
        m.previous_colvar_forces[k].real_value=m.colvar_forces[k].real_value=st[tjForce0+k];
    }
    m.bias_energy=st[tjEnergy];
    m.has_data=true;
}
