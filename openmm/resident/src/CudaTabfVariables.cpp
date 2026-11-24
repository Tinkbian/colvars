#include "CudaTabfVariables.h"
#include "colvarproxy_openmm.h"
#include "colvar.h"
#include "colvaratoms.h"
#include "colvarbias_restraint.h"
#include "colvarcomp.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace ColvarsPlugin {
struct CudaTabfVariableAccess {
    static void diagnostic(colvar& cv, double actual, double reported) {
        if (cv.after_restart) {
            // At the checkpoint itself the core validates the restored geometry.
            // At the next endpoint the atoms have already taken a physical step.
            if (cv.cvmodule->step_relative() == 0 && std::abs(actual-cv.x_restart.real_value) > 0.5*cv.width)
                throw std::runtime_error("The device coordinate of "+cv.name+" differs from its Colvars restart");
            cv.after_restart = false;
        }
        cv.x.real_value = actual;
        cv.x_reported.real_value = reported;
    }

    static void monitor(colvar& cv, double actual, double previous, double force, double dtFs,
                        cvm::step_number relativeStep) {
        diagnostic(cv, actual, actual);
        cv.x_old = cv.endpoint_x = cv.x;
        cv.prev_timestep = relativeStep;
        cv.after_restart = false;
        cv.fb.real_value = 0;
        cv.fb_actual.real_value = cv.f.real_value = force;
        if (cv.is_enabled(colvardeps::f_cv_subtract_applied_force)) cv.f_old = cv.f;
        if (cv.is_enabled(colvardeps::f_cv_fdiff_velocity)) {
            cv.v_reported.real_value = cv.v_fdiff.real_value = relativeStep == 0 ? 0.0 : (actual - previous)/dtFs;
        }
    }

    static bool addSide(double location, double stiffness, double& wall, double& sum) {
        if (!std::isfinite(location) || !(stiffness >= 0.0) || !std::isfinite(stiffness)) return false;
        if (stiffness == 0.0) return true;
        if (sum != 0.0 && wall != location) return false;
        wall = location;
        sum += stiffness;
        return std::isfinite(sum);
    }

    static bool addWall(const colvarbias_restraint_harmonic_walls& wall, size_t i, TabfVariable& v) {
        const double width = wall.variables(i)->width;
        const double scale = wall.force_k/(width*width);
        if (!wall.lower_walls.empty() && !addSide(wall.lower_walls[i].real_value, scale*wall.lower_wall_k,
                                                 v.lower, v.kLower)) return false;
        if (!wall.upper_walls.empty() && !addSide(wall.upper_walls[i].real_value, scale*wall.upper_wall_k,
                                                 v.upper, v.kUpper)) return false;
        return true;
    }

    static void wallOutput(colvarbias_restraint_harmonic_walls& wall, const std::vector<colvar*>& cvs,
                           const std::vector<double>& actual) {
        wall.has_data = true;
        wall.bias_energy = 0.0;
        for (size_t i = 0; i < wall.num_variables(); ++i) {
            const colvar* cv = wall.variables(i);
            const size_t j = std::find(cvs.begin(), cvs.end(), cv) - cvs.begin();
            TabfVariable v;
            addWall(wall, i, v);
            const double lo = std::min(0.0, actual.at(j) - v.lower);
            const double hi = std::max(0.0, actual.at(j) - v.upper);
            wall.bias_energy += 0.5*(v.kLower*lo*lo + v.kUpper*hi*hi);
            wall.colvar_values[i] = cv->value();
            wall.colvar_forces[i].real_value = -(v.kLower*lo + v.kUpper*hi);
            wall.previous_colvar_forces[i] = wall.colvar_forces[i];
        }
    }
};

bool configureTabfVariables(colvarproxy_openmm& proxy, std::vector<TabfVariable>& vars, std::string& why) {
    const auto& cvs = *proxy.module().variables();
    std::vector<TabfVariable> staged(cvs.size());
    int extendedDistances = 0, alphas = 0;
    for (size_t j = 0; j < cvs.size(); ++j) {
        const auto& cv = *cvs[j];
        auto& v = staged[j];
        why = "Unsupported device TABF variable "+cv.name;
        if (!cv.is_enabled() || cv.get_time_step_factor() != 1 || !cv.is_enabled(colvardeps::f_cv_single_cvc) ||
            cv.num_cvcs() != 1 || cv.value().type() != colvarvalue::type_scalar || cv.period != 0.0 ||
            cv.expand_boundaries || !(cv.width > 0.0) || !std::isfinite(cv.width)) return false;
        for (int flag : {colvardeps::f_cv_scripted, colvardeps::f_cv_custom_function, colvardeps::f_cv_runave,
                         colvardeps::f_cv_corrfunc, colvardeps::f_cv_periodic, colvardeps::f_cv_collect_gradient,
                         colvardeps::f_cv_collect_atom_ids, colvardeps::f_cv_hide_Jacobian})
            if (cv.is_enabled(flag)) return false;
        const auto& c = *cv.get_cvcs().front();
        if (!c.is_enabled() || c.sup_coeff != 1.0 || c.sup_np != 1 || c.period != 0.0 ||
            c.is_enabled(colvardeps::f_cvc_debug_gradient)) return false;
        v.extended = cv.is_enabled(colvardeps::f_cv_extended_Lagrangian);
        if (c.function_type() == "tabfAlpha") {
            if (!v.extended || !cv.is_enabled(colvardeps::f_cv_external) || ++alphas != 1) return false;
            v.axis = 1;
            continue;
        }
        if (c.function_type() != "distance" || c.atom_groups.size() != 2 ||
            cv.is_enabled(colvardeps::f_cv_external) || cv.is_enabled(colvardeps::f_cv_total_force_calc)) return false;
        if (v.extended) {
            if (++extendedDistances != 1) return false;
            v.axis = 0;
        }
        for (size_t k = 0; k < 2; ++k) {
            const auto& g = *c.atom_groups[k];
            if (g.b_dummy || g.fitting_group || g.noforce || g.size() != 1 ||
                !(g.total_mass > 0.0) || !std::isfinite(g.total_mass) ||
                g.is_enabled(colvardeps::f_ag_scalable) || g.is_enabled(colvardeps::f_ag_center) ||
                g.is_enabled(colvardeps::f_ag_rotate)) return false;
            const auto& ids = proxy.requestedAtoms();
            auto p = std::find(ids.begin(), ids.end(), g.id(0));
            if (p == ids.end()) return false;
            (k == 0 ? v.atom1 : v.atom2) = static_cast<int>(p - ids.begin());
        }
        v.minimumImage = c.is_enabled(colvardeps::f_cvc_pbc_minimum_image);
    }
    if (extendedDistances != 1 || alphas != 1) {
        why = "Device TABF requires one extended distance and one extended tabfAlpha";
        return false;
    }
    for (auto* bias : proxy.module().biases) {
        auto* wall = dynamic_cast<colvarbias_restraint_harmonic_walls*>(bias);
        if (!wall) continue; // ABF and metadynamics validate their own axes/options.
        why = "Unsupported device TABF wall "+wall->name;
        if (!wall->is_enabled(colvardeps::f_cvb_active) || !wall->is_enabled(colvardeps::f_cvb_apply_force) ||
            wall->get_time_step_factor() != 1 || wall->has_moving_parameters() ||
            !wall->is_enabled(colvardeps::f_cvb_bypass_ext_lagrangian) ||
            wall->is_enabled(colvardeps::f_cvb_calc_ti_samples) ||
            wall->is_enabled(colvardeps::f_cvb_scale_biasing_force) ||
            wall->is_enabled(colvardeps::f_cvb_output_acc_work)) return false;
        for (size_t i = 0; i < wall->num_variables(); ++i) {
            auto pos = std::find(cvs.begin(), cvs.end(), wall->variables(i));
            if (pos == cvs.end() || !CudaTabfVariableAccess::addWall(*wall, i, staged[pos-cvs.begin()]))
                return false;
        }
    }
    vars = std::move(staged);
    why.clear();
    return true;
}

void publishTabfMonitorsAndWalls(colvarproxy_openmm& proxy, const std::vector<TabfVariable>& vars,
                                const std::vector<double>& actual, const std::vector<double>& previousActual,
                                double dtFs, cvm::step_number relativeStep) {
    const auto& cvs = *proxy.module().variables();
    if (vars.size() != cvs.size() || actual.size() != cvs.size() || previousActual.size() != cvs.size())
        throw std::invalid_argument("Incomplete TABF endpoint coordinates for host output");
    for (size_t i = 0; i < cvs.size(); ++i) {
        const auto& v = vars[i];
        if (!v.extended) {
            const double force = -v.kLower*std::min(0.0, actual[i]-v.lower) - v.kUpper*std::max(0.0, actual[i]-v.upper);
            CudaTabfVariableAccess::monitor(*cvs[i], actual[i], previousActual[i], force, dtFs, relativeStep);
        }
    }
    for (auto* bias : proxy.module().biases)
        if (auto* wall = dynamic_cast<colvarbias_restraint_harmonic_walls*>(bias))
            CudaTabfVariableAccess::wallOutput(*wall, cvs, actual);
}

void publishTabfDiagnostic(colvar& cv, double actual, double reported) {
    CudaTabfVariableAccess::diagnostic(cv, actual, reported);
}
}
