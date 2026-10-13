#include "common.h"

#if defined(FLASH_MLA_IS_BUILD_ON_CUDA) && defined(FLASH_MLA_ENABLE_SM100)
#include "cuda_kernels/sm100/prefill/dense/interface.h"
#endif  // FLASH_MLA_IS_BUILD_ON_CUDA

void register_dense_fwd(pybind11::module_& m) {
#if defined(FLASH_MLA_IS_BUILD_ON_CUDA) && defined(FLASH_MLA_ENABLE_SM100)
    m.def("dense_prefill_fwd",
        &FMHACutlassSM100FwdRun,
        "Run Dense Attention Prefill Forward (cutlass FMHA)");
#else
    (void)m;    // The dense attention kernels are CUDA-only
#endif  // FLASH_MLA_IS_BUILD_ON_CUDA
}
