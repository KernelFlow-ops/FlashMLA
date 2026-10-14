#include "kernel.h"

#include <cstdlib>
#include <cstring>

#include <kerutils/kerutils.cuh>

#include "cuda_kernels/kv_cache_format.h"

namespace sm120::decode::sparse {

enum class Arithmetic {
    BF16,
    FP8X2,
    MXFP8
};

// Read on every call, so that a process can switch between the modes
static Arithmetic get_arithmetic() {
    const char *mode = std::getenv("FLASH_MLA_SM120_DECODE_MODE");
    if (mode == nullptr || mode[0] == '\0' || std::strcmp(mode, "bf16") == 0) {
        return Arithmetic::BF16;
    } else if (std::strcmp(mode, "fp8x2") == 0) {
        return Arithmetic::FP8X2;
    } else if (std::strcmp(mode, "mxfp8") == 0) {
        return Arithmetic::MXFP8;
    }
    KU_ASSERT(false, "FLASH_MLA_SM120_DECODE_MODE must be bf16, fp8x2 or mxfp8, got \"%s\"", mode);
    return Arithmetic::BF16;
}

void run_sparse_decode(const SparseAttnDecodeParams &params) {
    KU_ASSERT(params.h_q == 64 || params.h_q == 128, "h_q (%d) must be 64 or 128", params.h_q);
    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.d_qk == 512);
    KU_ASSERT(params.d_v == 512);
    KU_ASSERT(params.model_type == ModelType::V41 || params.model_type == ModelType::V41_MXFP4, "the main KV cache must be the V4.1 fp8 cache or the MXFP4 cache");
    if (params.model_type == ModelType::V41_MXFP4) {
        KU_ASSERT(params.extra_topk == 0 || params.extra_model_type == ModelType::V41_MXFP4, "an MXFP4 main KV cache takes an MXFP4 extra cache only");
    } else {
        KU_ASSERT(params.extra_model_type == ModelType::V41 || params.extra_model_type == ModelType::V41_FP4);
    }

    // 32-bit loads of Q and 32-bit stores of the output
    KU_ASSERT((int64_t)params.q % 4 == 0 && params.stride_q_b % 2 == 0 && params.stride_q_s_q % 2 == 0 && params.stride_q_h_q % 2 == 0,
        "q must be 4B aligned, with even strides");
    KU_ASSERT((int64_t)params.out % 4 == 0 && params.stride_o_b % 2 == 0 && params.stride_o_s_q % 2 == 0 && params.stride_o_h_q % 2 == 0);
    // cp.async.bulk copies whole rows from 16B aligned addresses
    auto check_cache = [](const char *name, const void *cache, int64_t block_stride, int row_stride, ModelType model_type) {
        const int bytes_per_token = kv_cache_bytes_per_token(model_type);
        KU_ASSERT((int64_t)cache % 16 == 0, "The base address of %s (%p) must be 16B aligned", name, cache);
        KU_ASSERT(row_stride == bytes_per_token, "%s.stride(-2) (%d) must be %d", name, row_stride, bytes_per_token);
        KU_ASSERT(block_stride % 16 == 0, "%s.stride(0) (%ld) must be a multiple of 16", name, block_stride);
    };
    check_cache("k_cache", params.kv, params.stride_kv_block, params.stride_kv_row, params.model_type);
    if (params.extra_topk > 0) {
        check_cache("extra_k_cache", params.extra_kv, params.stride_extra_kv_block, params.stride_extra_kv_row, params.extra_model_type);
    }

    const Arithmetic arithmetic = get_arithmetic();
    if (params.model_type == ModelType::V41_MXFP4) {
        // bf16 (the default) has no exact path for this cache: it runs the 4-term arithmetic that is within the tolerance of the bf16 one
        if (arithmetic == Arithmetic::MXFP8) {
            run_mxfp4(params, 2, 1);
        } else {
            run_mxfp4(params, 4, 2);
        }
        return;
    }
    const bool is_extra_fp4 = params.extra_topk > 0 && params.extra_model_type == ModelType::V41_FP4;
    // fp8x2 has no fp4 extra cache path: one e4m3 term of V on an fp4 tile is outside the tolerance of the upstream tests
    if (arithmetic == Arithmetic::BF16 || (arithmetic == Arithmetic::FP8X2 && is_extra_fp4)) {
        run_bf16(params);
    } else {
        run_mxfp8(params, arithmetic == Arithmetic::FP8X2 ? 2 : 1);
    }
}

}
