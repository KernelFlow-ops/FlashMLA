/*
Sparse MLA Decoding — SM120 (RTX 50 series / RTX PRO Blackwell)

The exact path: the same arithmetic as the sm100 decoding kernels (bf16 Q, KV dequantized to bf16,
fp32 accumulation, online softmax with lazy rescaling), built on warp-level mma.sync because sm120
has neither tcgen05 / TMEM nor wgmma. `kv` is the V4.1 fp8 cache; `extra_kv` is that or the V4.1 fp4
one. Supports attn_sink, topk_length, the extra cache and split-KV.

One CTA handles one query token and 64 heads (blockIdx.z selects the heads, so h_q = 128 runs two
CTAs per query token), and walks its KV tokens in tiles of B_TOPK.

NUM_THREADS = 384 (3 warpgroups)
- Warps 0-7, the consumers. QK: warp w computes the heads [16*(w%4), +16) over the feature dims
  [256*(w/4), +256), with its part of Q held in registers; the two halves of a head block meet in
  shared memory and warps 0-3 run the softmax. PV: warp w accumulates the output columns [64*w, +64)
  of all 64 heads in registers.
- Warps 8-11, the producers. Warp 8 turns the indices of a tile into addresses and gathers the raw
  rows with one cp.async.bulk per token; all four dequantize the tile to bf16.

I/O: See SparseAttnDecodeParams in csrc/params.h
*/
#include "kernel.h"

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
static constexpr int NUM_THREADS = NUM_CONSUMER_THREADS + 128;
static constexpr int NUM_KV_BUFS = 2;
static constexpr int NUM_MASK_BUFS = 4;
// ldmatrix reads 8 rows at once: a row stride that is an odd multiple of 16 B spreads them over all banks
static constexpr int KV_ROW_BYTES = D_QK * 2 + 16;
static constexpr int P_ROW_BYTES = B_TOPK * 2 + 16;
static constexpr int RAW_ROW_BYTES = Fp8Format::BYTES_PER_TOKEN;    // An fp4 row is shorter and uses the same slot
static constexpr float MAX_INIT_VAL = -1e30f;       // To avoid (-inf) - (-inf) = NaN
static constexpr float RESCALE_THRESHOLD = 6.0f;    // The running max of a head moves only when the tile max exceeds it by this much (in log2)

static_assert(B_TOPK == 32 && D_QK == D_V);
static_assert(Fp4Format::BYTES_PER_TOKEN <= RAW_ROW_BYTES);

enum NamedBarriers : uint32_t {
    s_exchanged = 1,
    softmax_done = 2
};

struct SharedMemoryPlan {
    alignas(128) uint8_t kv[NUM_KV_BUFS][B_TOPK * KV_ROW_BYTES];    // Dequantized tiles: bf16 [token][dim]
    alignas(16) uint8_t raw[B_TOPK * RAW_ROW_BYTES];                // Raw rows of the tile being produced
    alignas(16) uint8_t p[NUM_HEADS * P_ROW_BYTES];                 // bf16 [head][token]
    alignas(16) float s_exchange[NUM_HEADS / 16][32][16];           // QK partial sums of warps 4-7, in fragment layout
    alignas(16) float softmax_state[NUM_HEADS / 16][32][4];         // Per softmax thread: {max of head g, of head g+8, its share of the sum of head g, of head g+8}
    float row_scale[NUM_HEADS];                                     // Per head: the factor for the old output (per tile), then the final output scale
    uint32_t rescale_flags;                                         // One byte per softmax warp: does any of its heads rescale in this tile
    uint32_t valid_mask[NUM_MASK_BUFS];                             // Bit i: token i of the tile is valid
    uint64_t bar_raw;
    uint64_t bar_kv_ready[NUM_KV_BUFS], bar_kv_free[NUM_KV_BUFS];
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

// Two fp16 values times `scale` -> two bf16 values
__device__ __forceinline__ uint32_t scale_f16x2_to_bf16x2(uint32_t f16x2, float scale) {
    const float lo = __half2float(__ushort_as_half(static_cast<unsigned short>(f16x2))) * scale;
    const float hi = __half2float(__ushort_as_half(static_cast<unsigned short>(f16x2 >> 16))) * scale;
    const __nv_bfloat162 result = __float22bfloat162_rn(float2{lo, hi});
    return *reinterpret_cast<const uint32_t*>(&result);
}

__device__ __forceinline__ void store_bf16x8(uint8_t *dst, const uint32_t (&vals)[8], bool is_valid) {
    const uint32_t mask = is_valid ? 0xffffffffu : 0u;
    reinterpret_cast<uint4*>(dst)[0] = uint4{vals[0] & mask, vals[1] & mask, vals[2] & mask, vals[3] & mask};
    reinterpret_cast<uint4*>(dst)[1] = uint4{vals[4] & mask, vals[5] & mask, vals[6] & mask, vals[7] & mask};
}

// Dequantize the dims [128*part, +128) of a V4.1 fp8 row (512 B e4m3 + 16 B ue8m0, one scale per 32 dims) to bf16.
// e4m3 times a power of two is exact in bf16, as in tests/quant.py. An invalid token becomes zeros.
__device__ __forceinline__ void dequant_fp8_row_part(const uint8_t *row, int part, uint8_t *dst, bool is_valid) {
    const uint32_t scales = *reinterpret_cast<const uint32_t*>(row + Fp8Format::QUANT_BYTES + 4 * part);
    #pragma unroll
    for (int tile = 0; tile < 4; ++tile) {
        const uint32_t exponent = (scales >> (8 * tile)) & 0xff;
        const float scale = __uint_as_float(exponent != 0 ? exponent << 23 : 0x00400000u);    // ue8m0: 2^(exponent - 127)
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const uint4 raw = *reinterpret_cast<const uint4*>(row + 128 * part + 32 * tile + 16 * i);
            const uint32_t words[4] = {raw.x, raw.y, raw.z, raw.w};
            uint32_t vals[8];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                vals[2*j] = scale_f16x2_to_bf16x2(e4m3x2_to_f16x2(words[j]), scale);
                vals[2*j+1] = scale_f16x2_to_bf16x2(e4m3x2_to_f16x2(words[j] >> 16), scale);
            }
            store_bf16x8(dst + 2 * (32 * tile + 16 * i), vals, is_valid);
        }
    }
}

// Same for a V4.1 fp4 row (256 B of e2m1 pairs, the even dim in the low nibble, + 32 B e4m3, one scale per 16 dims).
// e2m1 times e4m3 has at most 2 + 4 significant bits, so it is exact in bf16.
__device__ __forceinline__ void dequant_fp4_row_part(const uint8_t *row, int part, uint8_t *dst, bool is_valid) {
    const uint2 scales = *reinterpret_cast<const uint2*>(row + Fp4Format::QUANT_BYTES + 8 * part);
    #pragma unroll
    for (int tile = 0; tile < 8; ++tile) {
        const uint32_t scale_f16x2 = e4m3x2_to_f16x2((tile < 4 ? scales.x : scales.y) >> (16 * ((tile % 4) / 2)));
        const float scale = __half2float(__ushort_as_half(static_cast<unsigned short>(scale_f16x2 >> (16 * (tile % 2)))));
        const uint2 raw = *reinterpret_cast<const uint2*>(row + 64 * part + 8 * tile);
        uint32_t vals[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            vals[j] = scale_f16x2_to_bf16x2(e2m1x2_to_f16x2(((j < 4 ? raw.x : raw.y) >> (8 * (j % 4))) & 0xff), scale);
        }
        store_bf16x8(dst + 2 * 16 * tile, vals, is_valid);
    }
}

__device__ __forceinline__ void producer_main(const SparseAttnDecodeParams &params, SharedMemoryPlan &plan) {
    const int s_q_idx = blockIdx.x;
    const int partition_idx = blockIdx.y;
    const int warp_idx = threadIdx.x / 32 - NUM_CONSUMER_WARPS;
    const int lane_idx = threadIdx.x % 32;

    int begin_req_idx, end_req_idx;
    if (!get_request_range(params, partition_idx, begin_req_idx, end_req_idx)) {
        return;
    }

    uint32_t tile_idx = 0;  // Counts the tiles of this CTA across requests; ring buffers and barrier phases follow it
    #pragma unroll 1
    for (int batch_idx = begin_req_idx; batch_idx <= end_req_idx; ++batch_idx) {
        const Request req = get_request(params, partition_idx, batch_idx);
        const int *indices = params.indices + (int64_t)batch_idx * params.stride_indices_b + (int64_t)s_q_idx * params.stride_indices_s_q;
        const int *extra_indices = params.extra_indices + (int64_t)batch_idx * params.stride_extra_indices_b + (int64_t)s_q_idx * params.stride_extra_indices_s_q;

        #pragma unroll 1
        for (int block_idx = req.start_block_idx; block_idx < req.end_block_idx; ++block_idx, ++tile_idx) {
            const bool is_extra_block = block_idx >= req.num_orig_blocks;
            const bool is_fp4 = is_extra_block && params.extra_model_type == ModelType::V41_FP4;
            const uint32_t row_bytes = is_fp4 ? Fp4Format::BYTES_PER_TOKEN : Fp8Format::BYTES_PER_TOKEN;

            if (warp_idx == 0) {
                // One lane per token. The raw buffer is free once the whole warpgroup has dequantized the previous tile
                if (tile_idx > 0) {
                    mbarrier_wait(&plan.bar_kv_ready[(tile_idx - 1) % NUM_KV_BUFS], ((tile_idx - 1) / NUM_KV_BUFS) & 1);
                }
                const int pos = (is_extra_block ? block_idx - req.num_orig_blocks : block_idx) * B_TOPK + lane_idx;
                const int limit = is_extra_block ? min(req.extra_topk_length, params.extra_topk) : min(req.topk_length, params.topk);
                int index = -1;
                if (pos < limit) {
                    index = (is_extra_block ? extra_indices : indices)[pos];
                }
                const bool is_token_valid = index != -1;
                const uint32_t valid_mask = __ballot_sync(0xffffffffu, is_token_valid);
                if (is_token_valid) {
                    // An invalid token is not copied (its row may hold NaN); the dequantizer writes zeros for it
                    const uint32_t page_block_size = is_extra_block ? params.extra_page_block_size : params.page_block_size;
                    const int64_t block_stride = is_extra_block ? params.stride_extra_kv_block : params.stride_kv_block;
                    const uint8_t *cache = reinterpret_cast<const uint8_t*>(is_extra_block ? params.extra_kv : params.kv);
                    const uint32_t page_idx = (uint32_t)index / page_block_size;
                    const uint32_t idx_in_page = (uint32_t)index - page_idx * page_block_size;
                    bulk_copy_g2s(
                        plan.raw + lane_idx * RAW_ROW_BYTES,
                        cache + page_idx * block_stride + (int64_t)idx_in_page * row_bytes,
                        row_bytes,
                        &plan.bar_raw
                    );
                }
                if (lane_idx == 0) {
                    plan.valid_mask[tile_idx % NUM_MASK_BUFS] = valid_mask;
                    mbarrier_arrive_expect_tx(&plan.bar_raw, __popc(valid_mask) * row_bytes);
                }
            }

            const uint32_t buf_idx = tile_idx % NUM_KV_BUFS;
            mbarrier_wait(&plan.bar_raw, tile_idx & 1);
            if (tile_idx >= NUM_KV_BUFS) {
                mbarrier_wait(&plan.bar_kv_free[buf_idx], (tile_idx / NUM_KV_BUFS - 1) & 1);
            }

            // Lane = token, warp = a quarter of the dims
            const bool is_token_valid = (plan.valid_mask[tile_idx % NUM_MASK_BUFS] >> lane_idx) & 1;
            const uint8_t *row = plan.raw + lane_idx * RAW_ROW_BYTES;
            uint8_t *dst = plan.kv[buf_idx] + lane_idx * KV_ROW_BYTES + warp_idx * (D_QK / 4) * 2;
            if (is_fp4) {
                dequant_fp4_row_part(row, warp_idx, dst, is_token_valid);
            } else {
                dequant_fp8_row_part(row, warp_idx, dst, is_token_valid);
            }
            mbarrier_arrive(&plan.bar_kv_ready[buf_idx]);
        }
    }
}

__device__ __forceinline__ void consumer_main(const SparseAttnDecodeParams &params, SharedMemoryPlan &plan) {
    const int s_q_idx = blockIdx.x;
    const int partition_idx = blockIdx.y;
    const int head_base = blockIdx.z * NUM_HEADS;
    const int warp_idx = threadIdx.x / 32;
    const int lane_idx = threadIdx.x % 32;
    const int m_tile = warp_idx % 4;    // QK: the heads [16*m_tile, +16) ...
    const int q_half = warp_idx / 4;    // ... over the feature dims [256*q_half, +256)
    const bool is_softmax_warp = q_half == 0;
    // Fragment coordinates: this thread holds rows g and g+8 and columns 2t and 2t+1 of every 16x8 output tile
    const int g = lane_idx / 4, t = lane_idx % 4;

    int begin_req_idx, end_req_idx;
    if (!get_request_range(params, partition_idx, begin_req_idx, end_req_idx)) {
        return;
    }

    float4 *s_exchange = reinterpret_cast<float4*>(plan.s_exchange[m_tile][lane_idx]);
    float4 *softmax_state = reinterpret_cast<float4*>(plan.softmax_state[m_tile][lane_idx]);

    uint32_t tile_idx = 0;  // The same count as in producer_main
    #pragma unroll 1
    for (int batch_idx = begin_req_idx; batch_idx <= end_req_idx; ++batch_idx) {
        int block_idx, end_block_idx;
        {
            const Request req = get_request(params, partition_idx, batch_idx);
            block_idx = req.start_block_idx;
            end_block_idx = req.end_block_idx;
        }

        // Q as A fragments of the 16 k-steps of this warp
        uint32_t q_frag[16][4];
        {
            const bf16 *q_row = params.q + (int64_t)batch_idx * params.stride_q_b + (int64_t)s_q_idx * params.stride_q_s_q
                + (int64_t)(head_base + 16 * m_tile + g) * params.stride_q_h_q + (D_QK / 2) * q_half + 2 * t;
            const int64_t row_step = 8 * (int64_t)params.stride_q_h_q;
            #pragma unroll
            for (int k = 0; k < 16; ++k) {
                q_frag[k][0] = *reinterpret_cast<const uint32_t*>(q_row + 16 * k);
                q_frag[k][1] = *reinterpret_cast<const uint32_t*>(q_row + row_step + 16 * k);
                q_frag[k][2] = *reinterpret_cast<const uint32_t*>(q_row + 16 * k + 8);
                q_frag[k][3] = *reinterpret_cast<const uint32_t*>(q_row + row_step + 16 * k + 8);
            }
        }

        float o_acc[4][8][4];   // [head block][output tile within the 64 columns of this warp]
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            #pragma unroll
            for (int n = 0; n < 8; ++n) {
                o_acc[m][n][0] = o_acc[m][n][1] = o_acc[m][n][2] = o_acc[m][n][3] = 0.0f;
            }
        }
        if (is_softmax_warp) {
            *softmax_state = float4{MAX_INIT_VAL, MAX_INIT_VAL, 0.0f, 0.0f};
        }

        #pragma unroll 1
        for (; block_idx < end_block_idx; ++block_idx, ++tile_idx) {
            const uint32_t buf_idx = tile_idx % NUM_KV_BUFS;
            mbarrier_wait(&plan.bar_kv_ready[buf_idx], (tile_idx / NUM_KV_BUFS) & 1);
            const uint32_t kv_addr = smem_addr(plan.kv[buf_idx]);

            // QK: S[head][token] over this warp's half of the dims
            float s_acc[4][4];
            #pragma unroll
            for (int n = 0; n < 4; ++n) {
                s_acc[n][0] = s_acc[n][1] = s_acc[n][2] = s_acc[n][3] = 0.0f;
            }
            {
                // One ldmatrix: 8 tokens x 32 dims, i.e. the B fragments of two k-steps
                const uint32_t base = kv_addr + (lane_idx % 8) * KV_ROW_BYTES + (lane_idx / 8) * 16 + q_half * D_QK;
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    #pragma unroll
                    for (int n = 0; n < 4; ++n) {
                        uint32_t k_frag[4];
                        ldmatrix_x4(k_frag, base + n * (8 * KV_ROW_BYTES) + k * 64);
                        mma_m16n8k16_bf16(s_acc[n], q_frag[2*k], k_frag[0], k_frag[1]);
                        mma_m16n8k16_bf16(s_acc[n], q_frag[2*k+1], k_frag[2], k_frag[3]);
                    }
                }
            }

            // The two halves of a head block meet in shared memory
            if (!is_softmax_warp) {
                #pragma unroll
                for (int n = 0; n < 4; ++n) {
                    s_exchange[n] = float4{s_acc[n][0], s_acc[n][1], s_acc[n][2], s_acc[n][3]};
                }
            }
            named_barrier_sync(NamedBarriers::s_exchanged, NUM_CONSUMER_THREADS);

            if (is_softmax_warp) {
                // Bit 8n + i: the token of column i of output tile n
                const uint32_t valid_mask = plan.valid_mask[tile_idx % NUM_MASK_BUFS] >> (2 * t);
                #pragma unroll
                for (int n = 0; n < 4; ++n) {
                    const float4 other = s_exchange[n];
                    const float other_vals[4] = {other.x, other.y, other.z, other.w};
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float s = (s_acc[n][i] + other_vals[i]) * params.sm_scale_div_log2;
                        s_acc[n][i] = ((valid_mask >> (8 * n + (i & 1))) & 1) ? s : -CUDART_INF_F;
                    }
                }

                const float4 state = *softmax_state;
                float mi[2] = {state.x, state.y}, li[2] = {state.z, state.w};
                float scale_for_old[2];
                bool should_scale_o = false;
                #pragma unroll
                for (int r = 0; r < 2; ++r) {
                    float cur_max = -CUDART_INF_F;
                    #pragma unroll
                    for (int n = 0; n < 4; ++n) {
                        cur_max = max(cur_max, max(s_acc[n][2*r], s_acc[n][2*r+1]));
                    }
                    cur_max = max(cur_max, __shfl_xor_sync(0xffffffffu, cur_max, 1));
                    cur_max = max(cur_max, __shfl_xor_sync(0xffffffffu, cur_max, 2));
                    // By this point cur_max and mi[r] are identical among the 4 threads of the head
                    const bool should_scale = cur_max - mi[r] > RESCALE_THRESHOLD;
                    scale_for_old[r] = should_scale ? exp2f(mi[r] - cur_max) : 1.0f;
                    mi[r] = should_scale ? cur_max : mi[r];
                    should_scale_o |= should_scale;

                    float cur_sum = 0.0f;
                    uint8_t *p_row = plan.p + (16 * m_tile + g + 8 * r) * P_ROW_BYTES + 2 * (2 * t);
                    #pragma unroll
                    for (int n = 0; n < 4; ++n) {
                        const float2 p = float2{exp2f(s_acc[n][2*r] - mi[r]), exp2f(s_acc[n][2*r+1] - mi[r])};
                        cur_sum += p.x + p.y;
                        *reinterpret_cast<__nv_bfloat162*>(p_row + 2 * (8 * n)) = __float22bfloat162_rn(p);
                    }
                    li[r] = li[r] * scale_for_old[r] + cur_sum;
                }
                *softmax_state = float4{mi[0], mi[1], li[0], li[1]};
                if (t == 0) {
                    plan.row_scale[16 * m_tile + g] = scale_for_old[0];
                    plan.row_scale[16 * m_tile + g + 8] = scale_for_old[1];
                }
                const bool any_should_scale_o = __any_sync(0xffffffffu, should_scale_o);
                if (lane_idx == 0) {
                    reinterpret_cast<uint8_t*>(&plan.rescale_flags)[m_tile] = any_should_scale_o;
                }
            }
            named_barrier_sync(NamedBarriers::softmax_done, NUM_CONSUMER_THREADS);

            // Scale O for the heads whose running max moved
            if (plan.rescale_flags != 0) {
                #pragma unroll
                for (int m = 0; m < 4; ++m) {
                    const float scale0 = plan.row_scale[16 * m + g], scale1 = plan.row_scale[16 * m + g + 8];
                    #pragma unroll
                    for (int n = 0; n < 8; ++n) {
                        o_acc[m][n][0] *= scale0;
                        o_acc[m][n][1] *= scale0;
                        o_acc[m][n][2] *= scale1;
                        o_acc[m][n][3] *= scale1;
                    }
                }
            }

            // PV: O[head][dim] over this warp's 64 output columns
            {
                // Lane l addresses row 8*((l/8)&1) + l%8 and columns [8*(l/16), +8) of a 16 x 16 block
                const uint32_t row_in_block = ((lane_idx / 8) & 1) * 8 + lane_idx % 8;
                const uint32_t col_bytes_in_block = (lane_idx / 16) * 16;
                const uint32_t p_base = smem_addr(plan.p) + row_in_block * P_ROW_BYTES + col_bytes_in_block;
                const uint32_t v_base = kv_addr + row_in_block * KV_ROW_BYTES + col_bytes_in_block + warp_idx * 64 * 2;
                #pragma unroll
                for (int k = 0; k < B_TOPK / 16; ++k) {
                    uint32_t p_frag[4][4];
                    #pragma unroll
                    for (int m = 0; m < 4; ++m) {
                        ldmatrix_x4(p_frag[m], p_base + m * (16 * P_ROW_BYTES) + k * 32);
                    }
                    #pragma unroll
                    for (int n = 0; n < 4; ++n) {
                        // 16 tokens x 16 dims, i.e. the B fragments of two output tiles, each used by all four head blocks
                        uint32_t v_frag[4];
                        ldmatrix_x4_trans(v_frag, v_base + k * (16 * KV_ROW_BYTES) + n * 32);
                        #pragma unroll
                        for (int m = 0; m < 4; ++m) {
                            mma_m16n8k16_bf16(o_acc[m][2*n], p_frag[m], v_frag[0], v_frag[1]);
                            mma_m16n8k16_bf16(o_acc[m][2*n+1], p_frag[m], v_frag[2], v_frag[3]);
                        }
                    }
                }
            }
            mbarrier_arrive(&plan.bar_kv_free[buf_idx]);
        }

        // Epilogue. The first barrier keeps the output scales from overwriting scales of the last tile that are still being read
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
                    plan.row_scale[16 * m_tile + g + 8 * r] = o_scale;
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
                const float scale0 = plan.row_scale[16 * m + g], scale1 = plan.row_scale[16 * m + g + 8];
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
                const float scale0 = plan.row_scale[16 * m + g], scale1 = plan.row_scale[16 * m + g + 8];
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

// The name keeps the upstream substring that the benchmark scripts look for
__global__ void __launch_bounds__(NUM_THREADS, 1)
flash_fwd_splitkv_mla_fp8_sparse_kernel(__grid_constant__ const SparseAttnDecodeParams params) {
    extern __shared__ __align__(128) char wksp_buf[];
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);

    if (threadIdx.x == 0) {
        mbarrier_init(&plan.bar_raw, 1);
        #pragma unroll
        for (int i = 0; i < NUM_KV_BUFS; ++i) {
            mbarrier_init(&plan.bar_kv_ready[i], NUM_THREADS - NUM_CONSUMER_THREADS);
            mbarrier_init(&plan.bar_kv_free[i], NUM_CONSUMER_THREADS);
        }
        fence_mbarrier_init();
    }
    __syncthreads();

    if (threadIdx.x < NUM_CONSUMER_THREADS) {
        setmaxnreg_inc<232>();
        consumer_main(params, plan);
    } else {
        setmaxnreg_dec<40>();
        producer_main(params, plan);
    }
}

void run_sparse_decode(const SparseAttnDecodeParams &params) {
    KU_ASSERT(params.h_q == NUM_HEADS || params.h_q == 2 * NUM_HEADS, "h_q (%d) must be 64 or 128", params.h_q);
    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.d_qk == D_QK);
    KU_ASSERT(params.d_v == D_V);
    KU_ASSERT(params.model_type == ModelType::V41, "the main KV cache must be the V4.1 fp8 cache");
    KU_ASSERT(params.extra_model_type == ModelType::V41 || params.extra_model_type == ModelType::V41_FP4);

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

    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    static_assert(smem_size <= 99 * 1024);  // The limit of a block on sm120

    auto kernel = &flash_fwd_splitkv_mla_fp8_sparse_kernel;
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    kernel<<<dim3(params.s_q, params.enable_split_kv ? params.num_sm_parts : params.b, params.h_q / NUM_HEADS), dim3(NUM_THREADS, 1, 1), smem_size, params.stream>>>(params);
    KU_CHECK_KERNEL_LAUNCH();
}

}
