// Two-dimensional ABF for the distance and alpha TABF extended coordinates.
// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef OPENMM_COLVARS_CUDA_TABF_JOINT_ABF_H
#define OPENMM_COLVARS_CUDA_TABF_JOINT_ABF_H

#include <string>
#include <vector>

class colvar;
class colvarbias_abf;

namespace ColvarsPlugin {

// Host and NVRTC share these definitions. Grid arrays use the order in the
// ABF configuration. axis[] maps each grid dimension to distance (0) or alpha (1).
#define COLVARS_TABF_JOINT_TYPES \
struct TabfJointGrid { double lower[2], width[2]; int bins[2]; }; \
struct TabfJointSpec { \
    TabfJointGrid samples, pmf; \
    double maxForce[2]; \
    int axis[2], status, counts, gradients, potential; \
    int updateBias, applyBias, stepZero, minSamples, fullSamples; \
}; \
struct TabfJointResult { double force[2], energy; int error; }; \
enum TabfJointStatus { tjAbfDistance, tjAbfAlpha, tjAbfDistanceForce, tjAbfAlphaForce, tjAbfEnergy, tjAbfStatusSize };

COLVARS_TABF_JOINT_TYPES
#define COLVARS_TABF_JOINT_STRING_I(...) #__VA_ARGS__
#define COLVARS_TABF_JOINT_STRING(...) COLVARS_TABF_JOINT_STRING_I(__VA_ARGS__)
inline constexpr char cudaTabfJointAbfSource[] = COLVARS_TABF_JOINT_STRING(COLVARS_TABF_JOINT_TYPES) R"CUDA(

__device__ int tabfJointBin(const TabfJointGrid& grid, int dim, double x) {
    const double scaled = (x-grid.lower[dim])/grid.width[dim];
    if (scaled < 0.0) return -1;
    if (scaled >= grid.bins[dim]) return grid.bins[dim];
    return (int)floor(scaled);
}

__device__ TabfJointResult tabfJointAbf(const TabfJointSpec& s, double* buf,
                                        bool advance, bool first, const double extended[2],
                                        const double systemForce[2]) {
    TabfJointResult result = {{0.0, 0.0}, 0.0, 0};
    if (s.samples.bins[0] == 0) return result;
    int index[2];
    for (int dim = 0; dim < 2; ++dim) {
        const int axis = s.axis[dim];
        if (!isfinite(extended[axis]) || !isfinite(systemForce[axis])) {
            result.error = 1;
            return result;
        }
        index[dim] = tabfJointBin(s.samples, dim, extended[axis]);
    }
    if (index[0] >= 0 && index[0] < s.samples.bins[0] &&
        index[1] >= 0 && index[1] < s.samples.bins[1]) {
        const int bin = index[0]*s.samples.bins[1]+index[1];
        double& n = buf[s.counts+bin];
        double* grad = buf+s.gradients+2*bin;
        if (advance && s.updateBias && (!first || s.stepZero)) {
            if (n >= 2147483647.0) {
                result.error = 2;
                return result;
            }
            n += 1.0;
            for (int dim = 0; dim < 2; ++dim) {
                grad[dim] -= systemForce[s.axis[dim]];
                if (!isfinite(grad[dim])) { result.error = 3; return result; }
            }
        }
        if (s.applyBias && n > s.minSamples) {
            const double scale = n < s.fullSamples ?
                (n-s.minSamples)/(n*(s.fullSamples-s.minSamples)) : 1.0/n;
            for (int dim = 0; dim < 2; ++dim) {
                double f = scale*grad[dim];
                if (s.maxForce[dim] >= 0.0)
                    f = fmax(-s.maxForce[dim], fmin(s.maxForce[dim], f));
                result.force[s.axis[dim]] = f;
            }
        }
    }
    // The nonperiodic PMF grid has a half-bin offset and an additional
    // node at each upper edge. Use its own geometry even outside the ABF grid.
    if (s.applyBias && s.pmf.bins[0] != 0) {
        int node[2];
        for (int dim = 0; dim < 2; ++dim) {
            int i = tabfJointBin(s.pmf, dim, extended[s.axis[dim]]);
            node[dim] = i < 0 ? 0 : (i < s.pmf.bins[dim] ? i : s.pmf.bins[dim]-1);
        }
        result.energy = buf[s.potential+node[0]*s.pmf.bins[1]+node[1]];
    }
    if (advance) {
        double* status = buf+s.status;
        status[tjAbfDistance] = extended[0];
        status[tjAbfAlpha] = extended[1];
        status[tjAbfDistanceForce] = result.force[0];
        status[tjAbfAlphaForce] = result.force[1];
        status[tjAbfEnergy] = result.energy;
    }
    return result;
}
)CUDA";
#undef COLVARS_TABF_JOINT_STRING
#undef COLVARS_TABF_JOINT_STRING_I
#undef COLVARS_TABF_JOINT_TYPES

class CudaTabfJointAbf {
public:
    bool configure(colvarbias_abf* abf, const colvar* distance, const colvar* alpha, std::string& why);
    TabfJointSpec pack(std::vector<double>& buffer) const;
    void unpack(const TabfJointSpec& spec, const std::vector<double>& buffer);

private:
    colvarbias_abf* abf_ = nullptr;
    int firstAxis_ = 0;
};

}
#endif
