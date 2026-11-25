// Joint two-dimensional grid metadynamics on the resident TABF endpoint.
// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef OPENMM_COLVARS_CUDA_TABF_JOINT_META_H
#define OPENMM_COLVARS_CUDA_TABF_JOINT_META_H

#include <string>
#include <vector>

class colvarbias_meta;

namespace ColvarsPlugin {

#define COLVARS_TABF_JOINT_META_TYPES \
struct TabfJointMetaSpec { \
    int rows, columns, grid, hills, hillCapacity, initialHills, status; \
    int growLower, growUpper, hardLower[2], hardUpper[2], edgeBins; \
    int hillFreq, gridFreq, stepZero; \
    double width[2], sigma[2], lower1, upper1, weight, wtEnergy; \
}; \
struct TabfJointMetaResult { double force[2], energy; int error; }; \
enum TabfJointMetaStatus { tjRows, tjFirstRow, tjLower, tjUpper, tjCount, tjProjected, \
                          tjValue0, tjValue1, tjForce0, tjForce1, tjEnergy, tjStatusSize }; \
enum TabfJointHill { thStep, thCenter0, thCenter1, thSigma0, thSigma1, thWeight, thEdge, thSize };

COLVARS_TABF_JOINT_META_TYPES

#define COLVARS_JOINT_META_STR_I(...) #__VA_ARGS__
#define COLVARS_JOINT_META_STR(...) COLVARS_JOINT_META_STR_I(__VA_ARGS__)

inline constexpr char cudaTabfJointMetaSource[] = COLVARS_JOINT_META_STR(COLVARS_TABF_JOINT_META_TYPES) R"CUDA(

// The active rows of the first axis occupy a circular buffer. Slotting new
// rows on either end exactly reproduces the core's zero-filled grid remap.
__device__ int tjCell(const TabfJointMetaSpec& s, const double* st, int i, int j) {
    int row = ((int)st[tjFirstRow]+i)%s.rows;
    return s.grid+3*(row*s.columns+j);
}

__device__ void tjHillValue(const double* h, double x, double y, TabfJointMetaResult& r) {
    double dx = x-h[thCenter0], dy = y-h[thCenter1];
    double sx2 = h[thSigma0]*h[thSigma0], sy2 = h[thSigma1]*h[thSigma1];
    double deviation = dx*dx/sx2+dy*dy/sy2;
    if (deviation > 23.0) return;
    double e = h[thWeight]*exp(-0.5*deviation);
    r.energy += e;
    r.force[0] += (e*(0.5/sx2))*(2.0*dx);
    r.force[1] += (e*(0.5/sy2))*(2.0*dy);
}

__device__ TabfJointMetaResult tjValue(const TabfJointMetaSpec& s, const double* buf,
                                       double x, double y) {
    TabfJointMetaResult r = {};
    const double* st = buf+s.status;
    double i = floor((x-st[tjLower])/s.width[0]), j = floor((y-s.lower1)/s.width[1]);
    bool inside = i>=0 && i<st[tjRows] && j>=0 && j<s.columns;
    int projected = (int)st[tjProjected];
    if (inside) {
        const double* cell = buf+tjCell(s,st,(int)i,(int)j);
        r.energy = cell[0]; r.force[0] = -cell[1]; r.force[1] = -cell[2];
    }
    for (int h = inside ? projected : 0; h<(int)st[tjCount]; ++h) {
        const double* hill = buf+s.hills+thSize*h;
        if (h>=projected || hill[thEdge]) tjHillValue(hill,x,y,r);
    }
    return r;
}

__device__ bool tjExpand(const TabfJointMetaSpec& s, double* st, double x) {
    double bin = floor((x-st[tjLower])/s.width[0]);
    int rows = (int)st[tjRows];
    double lower = s.growLower ? fmax(0.0,s.edgeBins-bin) : 0.0;
    double upper = s.growUpper ? fmax(0.0,bin-(rows-1)+s.edgeBins) : 0.0;
    if (lower+upper > s.rows-rows) return false;
    int left = (int)lower, right = (int)upper;
    st[tjFirstRow] = ((int)st[tjFirstRow]-left+s.rows)%s.rows;
    st[tjRows] = rows+left+right;
    st[tjLower] -= left*s.width[0];
    st[tjUpper] += right*s.width[0];
    return isfinite(st[tjLower]) && isfinite(st[tjUpper]);
}

__device__ bool tjEdge(const TabfJointMetaSpec& s, const double* st, double x, double y) {
    return (!s.hardLower[0] && (x-st[tjLower])/s.width[0]<s.edgeBins) ||
           (!s.hardUpper[0] && (st[tjUpper]-x)/s.width[0]<s.edgeBins) ||
           (!s.hardLower[1] && (y-s.lower1)/s.width[1]<s.edgeBins) ||
           (!s.hardUpper[1] && (s.upper1-y)/s.width[1]<s.edgeBins);
}

__device__ void tjProject(const TabfJointMetaSpec& s, double* buf) {
    double* st = buf+s.status;
    int pending = (int)st[tjProjected], count = (int)st[tjCount];
    if (pending==count) return;
    // A row is skipped if every pending Gaussian is outside its cutoff.
    for (int i=0; i<(int)st[tjRows]; ++i) {
        double x = st[tjLower]+(i+0.5)*s.width[0];
        bool active = false;
        for (int h=pending; h<count; ++h) {
            const double* hill = buf+s.hills+thSize*h;
            double dx = x-hill[thCenter0], sx = hill[thSigma0];
            if (dx*dx/(sx*sx)<=23.0) { active = true; break; }
        }
        if (!active) continue;
        for (int j=0; j<s.columns; ++j) {
            double y = s.lower1+(j+0.5)*s.width[1];
            TabfJointMetaResult sum = {};
            for (int h=pending; h<count; ++h)
                tjHillValue(buf+s.hills+thSize*h,x,y,sum);
            double* cell = buf+tjCell(s,st,i,j);
            cell[0] += sum.energy; cell[1] -= sum.force[0]; cell[2] -= sum.force[1];
        }
    }
    st[tjProjected] = count;
}

__device__ TabfJointMetaResult tabfJointMeta(const TabfJointMetaSpec& s, double* buf,
                                            long long step, bool advance, bool first,
                                            double x, double y) {
    TabfJointMetaResult r = {};
    if (!s.rows) return r;
    if (!isfinite(x) || !isfinite(y)) { r.error = 1; return r; }
    double* st = buf+s.status;
    if (advance) {
        if (!tjExpand(s,st,x)) { r.error = 2; return r; }
        if (step%s.hillFreq==0 && (!first || s.stepZero)) {
            int count = (int)st[tjCount];
            if (count>=s.hillCapacity || step>9007199254740991LL) { r.error = 3; return r; }
            double weight = s.weight;
            if (s.wtEnergy>0.0) weight *= exp(-tjValue(s,buf,x,y).energy/s.wtEnergy);
            if (!isfinite(weight)) { r.error = 4; return r; }
            double* h = buf+s.hills+thSize*count;
            h[thStep] = (double)step; h[thCenter0] = x; h[thCenter1] = y;
            h[thSigma0] = s.sigma[0]; h[thSigma1] = s.sigma[1]; h[thWeight] = weight;
            h[thEdge] = tjEdge(s,st,x,y);
            st[tjCount] = count+1;
        }
        if (step%s.gridFreq==0) tjProject(s,buf);
    }
    r = tjValue(s,buf,x,y);
    if (!isfinite(r.energy) || !isfinite(r.force[0]) || !isfinite(r.force[1])) r.error = 4;
    if (advance && !r.error) {
        st[tjEnergy]=r.energy; st[tjForce0]=r.force[0]; st[tjForce1]=r.force[1];
        st[tjValue0]=x; st[tjValue1]=y;
    }
    return r;
}
)CUDA";

#undef COLVARS_JOINT_META_STR
#undef COLVARS_JOINT_META_STR_I
#undef COLVARS_TABF_JOINT_META_TYPES

class CudaTabfJointMeta {
public:
    bool configure(colvarbias_meta* meta, std::string& why);
    TabfJointMetaSpec pack(std::vector<double>& buffer, int maxNewSteps) const;
    void unpack(const TabfJointMetaSpec& spec, const std::vector<double>& buffer);
private:
    colvarbias_meta* meta_ = nullptr;
};
}
#endif
