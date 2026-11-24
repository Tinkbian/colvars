// Scalar, nonperiodic ABF and grid metadynamics for the CUDA endpoint.
// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef OPENMM_COLVARS_CUDA_TABF_BIAS_H
#define OPENMM_COLVARS_CUDA_TABF_BIAS_H

#include <string>
#include <vector>

class colvarbias_abf;
class colvarbias_meta;

namespace ColvarsPlugin {

// NVRTC and the host compiler must agree on the by-value layout. All buffer
// offsets are absolute indices into the same double array, including counts.
#define COLVARS_TABF_BIAS_TYPES \
struct TabfBiasGrid { double lower, upper, width; int bins, offset; }; \
struct TabfBiasSpec { \
    TabfBiasGrid abf, meta; \
    int czar, updateBias, applyBias, minSamples, fullSamples, abfStepZero; \
    int newHillFreq, gridsFreq, metaStepZero, hardLower, hardUpper; \
    int status, hills, hillCapacity, firstNewHill; \
    double maxForce, sigma, hillWeight, wtEnergy, edgeBins; \
}; \
struct TabfBiasResult { double force, energy; int error; }; \
enum TabfBiasStatus { tbHillCount, tbProjectedCount, tbAbfForce, tbMetaForce, \
    tbAbfEnergy, tbMetaEnergy, tbExtended, tbStatusSize };

COLVARS_TABF_BIAS_TYPES

#define COLVARS_TABF_STRING_I(...) #__VA_ARGS__
#define COLVARS_TABF_STRING(...) COLVARS_TABF_STRING_I(__VA_ARGS__)

inline constexpr char cudaTabfBiasSource[] = COLVARS_TABF_STRING(COLVARS_TABF_BIAS_TYPES) R"CUDA(

// Counts use double because all integers through 2^53 are exact. The Colvars
// scalar grid output uses signed int counts; reject values beyond its range.
__device__ int tabfBiasBin(const TabfBiasGrid& g, double x) {
    double scaled = (x-g.lower)/g.width;
    if (scaled < 0) return -1;
    if (scaled >= g.bins) return g.bins;
    return (int)floor(scaled);
}
__device__ double tabfBiasMean(const TabfBiasSpec& s, double* buf, int i) {
    double n = buf[s.abf.offset+i];
    if (n <= s.minSamples) return 0.0;
    double scale = n < s.fullSamples ? (n-s.minSamples)/(n*(s.fullSamples-s.minSamples)) : 1.0/n;
    return scale*buf[s.abf.offset+s.abf.bins+i];
}
__device__ bool tabfBiasSample(double* counts, int bins, int i, double systemForce) {
    if (i < 0 || i >= bins) return true;
    if (counts[i] >= 2147483647.0) return false;
    counts[i] += 1.0;
    counts[bins+i] -= systemForce;
    return true;
}
__device__ bool tabfBiasEdge(const TabfBiasSpec& s, double x) {
    return (!s.hardLower && (x-s.meta.lower)/s.meta.width < s.edgeBins) ||
           (!s.hardUpper && (s.meta.upper-x)/s.meta.width < s.edgeBins);
}
// A record is {deposition step, center, sigma, weight}. Previously projected
// boundary hills can have a different width from the current configuration.
__device__ void tabfBiasHill(const double* hill, double x, double& energy, double& force) {
    double delta = x-hill[1];
    double sigma2 = hill[2]*hill[2];
    double deviation = delta*delta/sigma2;
    if (deviation <= 23.0) {
        double e = hill[3]*exp(-0.5*deviation);
        energy += e;
        force += (e*(0.5/sigma2))*(2.0*delta);
    }
}
__device__ void tabfBiasMetaValue(const TabfBiasSpec& s, double* buf, double x,
                                  int projected, int count, double& energy, double& force) {
    int bin = tabfBiasBin(s.meta, x);
    bool inside = bin >= 0 && bin < s.meta.bins;
    energy = inside ? buf[s.meta.offset+bin] : 0.0;
    force = inside ? -buf[s.meta.offset+s.meta.bins+bin] : 0.0;
    for (int i = inside ? projected : 0; i < count; ++i) {
        const double* hill = buf+s.hills+4*i;
        if (i >= projected || tabfBiasEdge(s, hill[1]))
            tabfBiasHill(hill, x, energy, force);
    }
}
// Called by a single device thread once per biased coordinate. Diagnostic
// queries set advance=false and never write state. first denotes Colvars'
// relative step zero, not OpenMM step zero. Pass wantEnergy on output steps.
__device__ TabfBiasResult tabfBias(const TabfBiasSpec& s, double* buf, long long step,
                                   bool advance, bool first, double actual, double extended,
                                   double systemForce, bool wantEnergy) {
    TabfBiasResult result = {0.0, 0.0, 0};
    if (!isfinite(actual) || !isfinite(extended) || !isfinite(systemForce)) {
        result.error = 1;
        return result;
    }
    double* status = buf+s.status;
    double abfForce = 0.0, abfEnergy = 0.0, metaForce = 0.0, metaEnergy = 0.0;
    if (s.abf.bins) {
        int n = s.abf.bins, bin = tabfBiasBin(s.abf, extended);
        if (advance && s.updateBias && (!first || s.abfStepZero)) {
            if (!tabfBiasSample(buf+s.abf.offset, n, bin, systemForce) ||
                (s.czar && !tabfBiasSample(buf+s.abf.offset+2*n, n,
                                           tabfBiasBin(s.abf, actual), systemForce))) {
                result.error = 2;
                return result;
            }
        }
        if (s.applyBias) {
            if (bin >= 0 && bin < n) {
                abfForce = tabfBiasMean(s, buf, bin);
                if (s.maxForce >= 0.0)
                    abfForce = fmax(-s.maxForce, fmin(s.maxForce, abfForce));
            }
            if (wantEnergy && bin >= 0) {
                // The 1D core estimates the applied potential using the
                // uncapped bin averages, including the local sampling ramp.
                int home = bin < n ? bin : n-1;
                for (int i = 0; i < home; ++i)
                    abfEnergy -= tabfBiasMean(s, buf, i)*s.abf.width;
                double scaled = (extended-s.abf.lower)/s.abf.width;
                double frac = bin < n ? scaled-floor(scaled) : 1.0;
                abfEnergy -= tabfBiasMean(s, buf, home)*s.abf.width*frac;
            }
        }
    }
    if (s.meta.bins) {
        int count = (int)status[tbHillCount], projected = (int)status[tbProjectedCount];
        if (advance && step%s.newHillFreq == 0 && (!first || s.metaStepZero)) {
            if (count >= s.hillCapacity || step > 9007199254740991LL) {
                result.error = 3;
                return result;
            }
            double weight = s.hillWeight;
            if (s.wtEnergy > 0.0) {
                tabfBiasMetaValue(s, buf, extended, projected, count, metaEnergy, metaForce);
                weight *= exp(-metaEnergy/s.wtEnergy);
            }
            double* hill = buf+s.hills+4*count++;
            hill[0] = (double)step;
            hill[1] = extended;
            hill[2] = s.sigma;
            hill[3] = weight;
            status[tbHillCount] = count;
        }
        if (advance && count != projected && step%s.gridsFreq == 0) {
            for (int bin = 0; bin < s.meta.bins; ++bin) {
                double x = s.meta.lower+s.meta.width*(0.5+bin);
                double e = 0.0, f = 0.0;
                for (int i = projected; i < count; ++i)
                    tabfBiasHill(buf+s.hills+4*i, x, e, f);
                buf[s.meta.offset+bin] += e;
                buf[s.meta.offset+s.meta.bins+bin] -= f;
            }
            projected = count;
            status[tbProjectedCount] = projected;
        }
        tabfBiasMetaValue(s, buf, extended, projected, count, metaEnergy, metaForce);
    }
    if (advance) {
        status[tbAbfForce] = abfForce;
        status[tbAbfEnergy] = abfEnergy;
        status[tbMetaForce] = metaForce;
        status[tbMetaEnergy] = metaEnergy;
        status[tbExtended] = extended;
    }
    result.force = abfForce+metaForce;
    result.energy = abfEnergy+metaEnergy;
    return result;
}
)CUDA";
#undef COLVARS_TABF_STRING
#undef COLVARS_TABF_STRING_I
#undef COLVARS_TABF_BIAS_TYPES

class CudaTabfStateAccess {
public:
    bool configure(colvarbias_abf* abf, colvarbias_meta* meta, std::string& why);
    TabfBiasSpec pack(std::vector<double>& buffer, int maxNewSteps) const;
    void unpack(const TabfBiasSpec& spec, const std::vector<double>& buffer);
private:
    colvarbias_abf* abf_ = nullptr;
    colvarbias_meta* meta_ = nullptr;
};
}
#endif
