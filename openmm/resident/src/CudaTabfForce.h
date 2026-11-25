#ifndef OPENMM_CUDA_TABF_FORCE_H
#define OPENMM_CUDA_TABF_FORCE_H
#include "CudaTabfBias.h"
#include "CudaTabfDynamicsAdapter.h"
#include "CudaTabfJointAbf.h"
#include "CudaTabfJointMeta.h"
#include "CudaTabfVariables.h"
#include "openmm/cuda/CudaArray.h"
#include "openmm/common/ComputeKernel.h"
#include "openmm/common/ComputeVectorTypes.h"
#include <array>

namespace OpenMM { class ContextImpl; }
namespace ColvarsPlugin {
class CudaTabfForce {
public:
    CudaTabfForce(colvarproxy_openmm&, OpenMM::CudaContext&, OpenMM::CudaArray& positions,
                  OpenMM::CudaArray& forces);
    OpenMM::CudaArray& state() { return scalars_; }
    void validateStart(double alpha, long long step);
    double evaluate(OpenMM::ContextImpl& context, bool energy);
    void finishBatch(bool outputMayChange = false);
    void checkBatchEnd(double alpha);
    bool hasJointAbf() const { return useJointAbf_; }
    bool hasJointMeta() const { return useJointMeta_; }
private:
    void packBiases();
    void checkDevice();
    OpenMM::mm_double4 periodicBox() const;
    colvarproxy_openmm& proxy_;
    OpenMM::CudaContext& cuda_;
    std::vector<TabfVariable> variables_;
    std::array<int, 2> axisVariable_{};
    std::array<CudaTabfStateAccess, 2> biases_;
    CudaTabfJointAbf jointAbf_;
    CudaTabfJointMeta jointMeta_;
    TabfJointSpec jointAbfSpec_{};
    TabfJointMetaSpec jointMetaSpec_{};
    OpenMM::CudaArray jointAbfParams_, jointMetaParams_;
    std::vector<TabfBiasSpec> specs_{2};
    OpenMM::CudaArray scalars_, mapping_, walls_, configs_, states_, rng_, specBuffer_,
                      biasBuffer_, diagnostics_, geometry_, potential_, error_;
    OpenMM::ComputeKernel evaluate_;
    long long lastStep_ = -1;
    int pending_ = 0;
    bool poisoned_ = false;
    bool stochastic_ = false;
    bool useJointAbf_ = false, useJointMeta_ = false, needsPack_ = false;
    double batchAlpha_ = 0;
};
}
#endif
