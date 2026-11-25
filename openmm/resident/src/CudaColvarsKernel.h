#ifndef OPENMM_CUDA_COLVARS_KERNEL_H
#define OPENMM_CUDA_COLVARS_KERNEL_H
#include <memory>
#include <string>
class colvarproxy_openmm;
namespace OpenMM { class ContextImpl; class CudaArray; }
namespace ColvarsPlugin {
class ColvarsForce;
class CudaColvarsKernel {
public:
    virtual ~CudaColvarsKernel() = default;
    virtual double execute(OpenMM::ContextImpl&, bool includeForces, bool includeEnergy) = 0;
    virtual colvarproxy_openmm& proxy() = 0;
    virtual std::string backend() const = 0;
    virtual OpenMM::CudaArray& deviceTabfState() = 0;
    virtual void validateDeviceTabfStart(double alpha, long long step) = 0;
    virtual void finishDeviceTabfBatch(double alpha) = 0;
    virtual void synchronizeBiasOutput(bool outputMayChange) = 0;
};
std::unique_ptr<CudaColvarsKernel> createCudaColvarsKernel(
    OpenMM::ContextImpl&, const ColvarsForce&);
}
#endif
