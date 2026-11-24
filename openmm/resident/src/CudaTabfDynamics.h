#ifndef OPENMM_RESIDENT_CUDA_TABF_DYNAMICS_H
#define OPENMM_RESIDENT_CUDA_TABF_DYNAMICS_H

#include <cmath>
#include <type_traits>

// Compile the scalar implementation on the host and supply the same source to
// OpenMM's CUDA compiler. The structs are copied as bytes by the CUDA adapter.
#define TABF_BODY(...) __VA_ARGS__
#define TABF_INLINE inline
#define TABF_MATH(op) std::op
#include "CudaTabfDynamics.inc"
#undef TABF_MATH
#undef TABF_INLINE
#undef TABF_BODY

static_assert(std::is_trivially_copyable<TabfExtState>::value, "CUDA endpoint state must be POD");
static_assert(sizeof(unsigned long long) == 8, "TABF requires 64-bit random words");

namespace ColvarsPlugin {
inline const char* tabfDynamicsCudaSource() {
#define TABF_BODY(...) #__VA_ARGS__ "\n"
    return "#define TABF_INLINE __device__ __forceinline__\n#define TABF_MATH(op) op\n"
#include "CudaTabfDynamics.inc"
           "#undef TABF_INLINE\n#undef TABF_MATH\n";
#undef TABF_BODY
}
}
#endif
