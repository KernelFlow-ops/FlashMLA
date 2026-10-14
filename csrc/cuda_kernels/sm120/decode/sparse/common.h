#pragma once

// What the sm120 sparse decoding kernels share: the CTA layout, the walk over the requests of a CTA and
// the end of a request (output scale, LSE, store).

#include "kernel.h"

#include <climits>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <kerutils/kerutils.cuh>

#include "cuda_kernels/kv_cache_format.h"
#include "cuda_kernels/sm120/ptx.h"

namespace sm120::decode::sparse {

using namespace sm120::ptx;

using Fp8Format = KVCacheFormat<ModelType::V41>;
using Fp4Format = KVCacheFormat<ModelType::V41_FP4>;

static constexpr int NUM_HEADS = 64;    // Heads per CTA
static constexpr int D_QK = 512;
static constexpr int D_V = 512;
static constexpr int NUM_CONSUMER_WARPS = 8;
static constexpr int NUM_CONSUMER_THREADS = NUM_CONSUMER_WARPS * 32;
static constexpr int NUM_PRODUCER_WARPS = 4;
static constexpr int NUM_THREADS = NUM_CONSUMER_THREADS + NUM_PRODUCER_WARPS * 32;
static constexpr float MAX_INIT_VAL = -1e30f;       // To avoid (-inf) - (-inf) = NaN
static constexpr float RESCALE_THRESHOLD = 6.0f;    // The running max of a head moves only when the tile max exceeds it by this much (in log2)

static_assert(B_TOPK == 32 && D_QK == D_V);

// Grid: x = block of 64 heads, y = query token, z = request (or partition with split-KV). The CTAs of the head blocks
// and query tokens of a request are launched next to each other, so that the rows they share come from L2.
__device__ __forceinline__ int cta_head_block() { return blockIdx.x; }
__device__ __forceinline__ int cta_s_q_idx() { return blockIdx.y; }
__device__ __forceinline__ int cta_partition_idx() { return blockIdx.z; }

inline dim3 decode_grid(const SparseAttnDecodeParams &params) {
    return dim3(params.h_q / NUM_HEADS, params.s_q, params.enable_split_kv ? params.num_sm_parts : params.b);
}

enum NamedBarriers : uint32_t {
    s_exchanged = 1,
    softmax_done = 2
};

struct Request {
    int start_block_idx, end_block_idx;     // Tiles [start, end): those of `kv` come first, then those of `extra_kv`
    int num_orig_blocks;
    int topk_length, extra_topk_length;
    int n_split_idx;
    bool is_no_split;
};

// The requests [begin, end] of this CTA. Returns false if it has none.
__device__ __forceinline__ bool get_request_range(const SparseAttnDecodeParams &params, int partition_idx, int &begin_req_idx, int &end_req_idx) {
    if (params.enable_split_kv) {
        const DecodingSchedMeta *sched_meta = params.tile_scheduler_metadata_ptr + partition_idx;
        begin_req_idx = sched_meta->begin_req_idx;
        end_req_idx = sched_meta->end_req_idx;
    } else {
        begin_req_idx = end_req_idx = partition_idx;
    }
    return begin_req_idx < params.b;
}

// Reads everything from global memory, so that callers do not have to keep it in registers.
// The block counts must agree with get_decoding_sched_meta.
__device__ __forceinline__ Request get_request(const SparseAttnDecodeParams &params, int partition_idx, int batch_idx) {
    Request req;
    req.topk_length = params.topk_length ? params.topk_length[batch_idx] : params.topk;
    req.extra_topk_length = params.extra_topk == 0 ? 0 : (params.extra_topk_length ? params.extra_topk_length[batch_idx] : params.extra_topk);
    req.num_orig_blocks = max((req.topk_length + B_TOPK - 1) / B_TOPK, 1);
    const int num_blocks = req.num_orig_blocks + (req.extra_topk_length + B_TOPK - 1) / B_TOPK;
    if (params.enable_split_kv) {
        const DecodingSchedMeta *sched_meta = params.tile_scheduler_metadata_ptr + partition_idx;
        const bool is_first_req = batch_idx == sched_meta->begin_req_idx, is_last_req = batch_idx == sched_meta->end_req_idx;
        req.start_block_idx = is_first_req ? sched_meta->begin_block_idx : 0;
        req.end_block_idx = is_last_req ? sched_meta->end_block_idx : num_blocks;
        req.is_no_split = !(is_first_req ? sched_meta->is_first_req_splitted : (is_last_req ? sched_meta->is_last_req_splitted : 0));
        req.n_split_idx = params.num_splits_ptr[batch_idx] + (is_first_req ? sched_meta->begin_split_idx : 0);
    } else {
        req.start_block_idx = 0;
        req.end_block_idx = num_blocks;
        req.is_no_split = true;
        req.n_split_idx = 0;
    }
    return req;
}

// The tensor maps of the caches for the gather4: a cache as a 2-D tensor [rows][row_bytes / 4] of 32-bit elements,
// row = block * rows_per_block + index in the block. rows_per_block = 0: no tensor map (a cache whose rows are copied one
// by one: a block stride that is not a whole number of rows, or a kernel that does not gather this cache with gather4).
struct GatherMaps {
    CUtensorMap kv, extra_kv;
    int kv_rows_per_block, extra_rows_per_block;
};

// The tensor map of a cache of `row_bytes` per token for the gather4, with a box of `box_bytes` (>= row_bytes; a box
// wider than the row reads zeros past its end, or before its start from a negative column); returns rows_per_block,
// 0 without a map. FLASH_MLA_SM120_GATHER4=0 turns the gather4 off (one bulk copy per row), for comparisons.
inline int make_gather_map(CUtensorMap &map, const void *cache, int num_blocks, int page_block_size, int64_t block_stride, int row_bytes, int box_bytes) {
    const char *env = std::getenv("FLASH_MLA_SM120_GATHER4");
    if ((env != nullptr && std::strcmp(env, "0") == 0) || num_blocks <= 0 || block_stride % row_bytes != 0) {
        return 0;
    }
    const int64_t rows_per_block = block_stride / row_bytes;
    const int64_t num_rows = (int64_t)(num_blocks - 1) * rows_per_block + page_block_size;
    if (rows_per_block < page_block_size || num_rows > INT_MAX) {
        return 0;
    }
    static const PFN_cuTensorMapEncodeTiled_v12000 encode = [] {
        void *fn = nullptr;
        cudaDriverEntryPointQueryResult query;
        const cudaError_t err = cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &fn, 12000, cudaEnableDefault, &query);
        return err == cudaSuccess && query == cudaDriverEntryPointSuccess ? reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(fn) : nullptr;
    }();
    if (encode == nullptr) {
        return 0;
    }
    const cuuint64_t dims[2] = {(cuuint64_t)row_bytes / 4, (cuuint64_t)num_rows};
    const cuuint64_t strides[1] = {(cuuint64_t)row_bytes};
    const cuuint32_t box[2] = {(cuuint32_t)box_bytes / 4, 1};
    const cuuint32_t element_strides[2] = {1, 1};
    const CUresult res = encode(&map, CU_TENSOR_MAP_DATA_TYPE_UINT32, 2, const_cast<void*>(cache), dims, strides, box, element_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return res == CUDA_SUCCESS ? (int)rows_per_block : 0;
}

__device__ __forceinline__ float f16_to_f32(uint32_t f16) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(f16)));
}

// The end of a request, for the consumer warps: turns the softmax state into the output scale and the LSE, then
// scales and stores the output. Warp w holds the output columns [64*w, +64) of all 64 heads in
// `o_acc` ([head block][8-column tile][fragment]); `softmax_state` is, for a thread of the softmax warps 0-3,
// {max of head g, max of head g+8, its share of the sum of head g, of head g+8}; `row_scale` has one float
// per head in shared memory.
__device__ __forceinline__ void finish_request(
    const SparseAttnDecodeParams &params, int partition_idx, int batch_idx,
    const float4 *softmax_state, float *row_scale, const float (&o_acc)[4][8][4]
) {
    const int s_q_idx = cta_s_q_idx();
    const int head_base = cta_head_block() * NUM_HEADS;
    const int warp_idx = threadIdx.x / 32;
    const int lane_idx = threadIdx.x % 32;
    const int m_tile = warp_idx % 4;
    const bool is_softmax_warp = warp_idx < 4;
    const int g = lane_idx / 4, t = lane_idx % 4;

    // The first barrier keeps the output scales from overwriting scales of the last tile that are still being read
    const Request req = get_request(params, partition_idx, batch_idx);
    named_barrier_sync(NamedBarriers::s_exchanged, NUM_CONSUMER_THREADS);
    if (is_softmax_warp) {
        const float4 state = *softmax_state;
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            float mi = r == 0 ? state.x : state.y;
            float li = r == 0 ? state.z : state.w;
            li += __shfl_xor_sync(0xffffffffu, li, 1);
            li += __shfl_xor_sync(0xffffffffu, li, 2);
            if (mi == MAX_INIT_VAL) {
                // No valid token. We set li to 0 to fit the definition that li := exp(x[i] - mi)
                li = 0.0f;
                mi = -CUDART_INF_F;
            }
            const int head_idx = head_base + 16 * m_tile + g + 8 * r;
            float o_scale;
            if (req.is_no_split) {
                const float attn_sink = params.attn_sink == nullptr ? -CUDART_INF_F : params.attn_sink[head_idx] * CUDART_L2E_F;
                o_scale = li == 0.0f ? 0.0f : 1.0f / (li + exp2f(attn_sink - mi));
                float cur_lse = mi * CUDART_LN2_F + logf(li);
                cur_lse = cur_lse == -CUDART_INF_F ? +CUDART_INF_F : cur_lse;
                if (t == 0) {
                    params.lse[(int64_t)batch_idx * params.stride_lse_b + (int64_t)s_q_idx * params.stride_lse_s_q + head_idx] = cur_lse;
                }
            } else {
                // Here we leave attn_sink to the combine kernel, otherwise attn_sink will take effect for multiple times
                o_scale = li == 0.0f ? 0.0f : 1.0f / li;
                if (t == 0) {
                    params.lse_accum[(int64_t)req.n_split_idx * params.stride_lse_accum_split + (int64_t)s_q_idx * params.stride_lse_accum_s_q + head_idx] = log2f(li) + mi;
                }
            }
            if (t == 0) {
                row_scale[16 * m_tile + g + 8 * r] = o_scale;
            }
        }
    }
    named_barrier_sync(NamedBarriers::softmax_done, NUM_CONSUMER_THREADS);

    // Scale and store: this thread holds, for every head block, rows g and g+8 and pairs of adjacent columns
    if (req.is_no_split) {
        bf16 *out_row = params.out + (int64_t)batch_idx * params.stride_o_b + (int64_t)s_q_idx * params.stride_o_s_q
            + (int64_t)(head_base + g) * params.stride_o_h_q + warp_idx * 64 + 2 * t;
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            const float scale0 = row_scale[16 * m + g], scale1 = row_scale[16 * m + g + 8];
            bf16 *row0 = out_row + (int64_t)(16 * m) * params.stride_o_h_q;
            bf16 *row1 = row0 + 8 * (int64_t)params.stride_o_h_q;
            #pragma unroll
            for (int n = 0; n < 8; ++n) {
                *reinterpret_cast<__nv_bfloat162*>(row0 + 8 * n) = __float22bfloat162_rn(float2{o_acc[m][n][0] * scale0, o_acc[m][n][1] * scale0});
                *reinterpret_cast<__nv_bfloat162*>(row1 + 8 * n) = __float22bfloat162_rn(float2{o_acc[m][n][2] * scale1, o_acc[m][n][3] * scale1});
            }
        }
    } else {
        float *out_row = params.o_accum + (int64_t)req.n_split_idx * params.stride_o_accum_split + (int64_t)s_q_idx * params.stride_o_accum_s_q
            + (int64_t)(head_base + g) * params.stride_o_accum_h_q + warp_idx * 64 + 2 * t;
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            const float scale0 = row_scale[16 * m + g], scale1 = row_scale[16 * m + g + 8];
            float *row0 = out_row + (int64_t)(16 * m) * params.stride_o_accum_h_q;
            float *row1 = row0 + 8 * (int64_t)params.stride_o_accum_h_q;
            #pragma unroll
            for (int n = 0; n < 8; ++n) {
                *reinterpret_cast<float2*>(row0 + 8 * n) = float2{o_acc[m][n][0] * scale0, o_acc[m][n][1] * scale0};
                *reinterpret_cast<float2*>(row1 + 8 * n) = float2{o_acc[m][n][2] * scale1, o_acc[m][n][3] * scale1};
            }
        }
    }
}

}
