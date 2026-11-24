#ifndef OPENMM_RESIDENT_CUDA_TABF_DYNAMICS_ADAPTER_H
#define OPENMM_RESIDENT_CUDA_TABF_DYNAMICS_ADAPTER_H

#include "CudaTabfDynamics.h"
#include "colvarmodule.h"

class colvar;
class colvarproxy_openmm;

namespace ColvarsPlugin {
class CudaTabfDynamicsAdapter {
public:
    static TabfExtConfig config(const colvar& cv);
    static TabfExtState read(const colvar& cv, long long lastStep);
    static void write(colvar& cv, const TabfExtState& state, cvm::step_number relativeStep);
    static bool readRng(const colvarproxy_openmm& proxy, TabfGaussianState& rng);
    static void writeRng(colvarproxy_openmm& proxy, const TabfGaussianState& rng);
};
}
#endif
