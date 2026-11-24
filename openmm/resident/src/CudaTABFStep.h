#ifndef OPENMM_CUDA_TABF_STEP_H
#define OPENMM_CUDA_TABF_STEP_H

#include <memory>

namespace OpenMM { class ContextImpl; }
namespace ColvarsPlugin {
class CudaColvarsKernel;
class ColvarsTABFIntegrator;

// The two scalar transfers occur at public step(n) boundaries so OpenMM's
// ordinary integrator globals still serialize and can be queried by reporters.
class CudaTABFStep {
public:
    virtual ~CudaTABFStep() = default;
    virtual void beginBatch(double alpha, double derivative, double kT, double friction) = 0;
    virtual void integratePhysical(bool contextChanged) = 0;
    virtual void finishBias() = 0;
    virtual void endBatch(double& alpha, double& derivative) = 0;
};
std::shared_ptr<CudaTABFStep> createCudaTABFStep(OpenMM::ContextImpl&, ColvarsTABFIntegrator&, CudaColvarsKernel&);
}
#endif
