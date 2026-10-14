/*
Sparse MLA Decoding — SM120 (RTX 50 series / RTX PRO Blackwell), the block-scaled fp8 path

Both matrix products run on the block-scaled e4m3 MMA (kind::mxf8f6f4, one power-of-two scale per row
of A and per column of B for every 32 elements of k), which is four times as fast as the bf16 MMA on
this architecture. The main KV cache has to be the V4.1 fp8 cache; the extra cache may be the V4.1 fp4
cache with NUM_TERMS = 1 (EXTRA_FP4, below).

- QK. K is used as stored: the e4m3 bytes of a token are the B operand and its 16 ue8m0 scales (one per
  32 dims) are the scales of its column, so nothing is lost on this side. Q is quantized once per
  request to e4m3 with one power-of-two scale per head and 32 dims.
- PV. The scales of V belong to (token, 32 dims) and a token is an element of k here, so they cannot
  be the scales of a column. The producers rebase every tile: per 32 dims they take the largest
  exponent of the tile as the scale of those columns and divide the bytes of each token by the power
  of two that it is below that. This is exact until a value drops into the subnormal range of e4m3,
  where it is rounded to nearest. P is quantized to e4m3 with a constant scale.
- NUM_TERMS = 2 writes Q and P as the sum of two e4m3 terms (the rounded value and the rounded
  remainder, with a scale 16 times smaller) and runs two MMAs for each. The pair holds 8 significant
  bits, as many as bf16, so the result stays within the tolerance of the bf16 path at half its number
  of MMA cycles. NUM_TERMS = 1 keeps 4 significant bits of Q and P.
- EXTRA_FP4: the tiles of an fp4 extra cache (e2m1 + an e4m3 scale per 16 dims) come after those of
  the main cache in every request. Their QK runs on the fp4 MMA (kind::mxf4nvf4, twice the rate of
  the e4m3 one) with K as stored and Q quantized again, once per request, to FP4_Q_TERMS terms of
  e2m1 with an e4m3 scale per head and 16 dims. For PV the producers round V once to e4m3 on a
  power-of-two scale per tile and 32 dims, which makes it a rebased tile like the others; the scale
  comes from the largest 16-dim scale of the tile (rebase_fp4_tile_part).

The CTA holds one query token and 64 heads, as in the bf16 path (bf16.cu), and walks tiles of B_TOPK
tokens with 8 consumer warps and 4 producer warps.
- The consumer warps work in pairs, one per block of 16 heads: warps 2m and 2m+1 (on different SM
  sub-partitions, so that the pairs sharing one drift apart and overlap their softmax with MMAs) each
  compute S over half of the dims, swap the halves through shared memory, run the same softmax, and keep
  P in registers as the A fragments of PV, each over half of the output columns.
- Producer warp 0 gathers: one cp.async.bulk per row into a ring of NUM_RAW_BUFS tiles, as far ahead
  as the ring is free. The copies are serialized (about 90 clock cycles each, whatever their size),
  so issuing a tile keeps a warp busy, and the row indices are loaded one tile before they are used
  to keep that load off the path.
- The other three producer warps rebase every tile as soon as it has arrived, into one of two V
  buffers, so that the rebase of a tile does not wait for the PV of the tile before it.
- A raw buffer is free again when the consumers have finished its QK and the rebase warps its rebase.

I/O: See SparseAttnDecodeParams in csrc/params.h
*/
#include "common.h"

namespace sm120::decode::sparse {
namespace mxfp8_path {

// Producer warp 0 gathers, the others rebase
static constexpr int NUM_REBASE_WARPS = NUM_PRODUCER_WARPS - 1;
static constexpr int NUM_RAW_BUFS = 3;      // Up to three tiles (50 KB) in flight
static_assert(NUM_RAW_BUFS == 3, "the consumers divide by 3 with __umulhi");
static constexpr int NUM_V_BUFS = 2;        // The tile at PV and the next one, so that a rebase does not wait for the PV before it
static constexpr int NUM_MASK_BUFS = 4;     // A mask lives from the gather of its tile to its softmax: NUM_RAW_BUFS + 1 tiles
static constexpr int NUM_SCALE_GROUPS = D_V / Fp8Format::QUANT_TILE_SIZE;
// ldmatrix reads 8 or 16 rows at once: a row stride that is an odd multiple of 16 B spreads them over all banks
static constexpr int ROW_BYTES = Fp8Format::BYTES_PER_TOKEN;
// P = hi * 2^-2 + lo * 2^-6. Lazy rescaling keeps P <= 2^RESCALE_THRESHOLD, so hi <= 256 fits e4m3 (max 448)
static constexpr float P_GAIN = 4.0f, P_LO_GAIN = 16.0f;
static constexpr uint32_t P_SCALE_HI = 0x7d7d7d7du, P_SCALE_LO = 0x79797979u;
// Q = hi * 2^(s - 127) + lo * 2^(s - 131), with s chosen per head and 32 dims
static constexpr float Q_LO_GAIN = 16.0f;
static constexpr uint32_t Q_LO_SCALE_STEP = 4;

// The rows of an fp4 tile (288 B) go to the same raw buffers at this stride
static constexpr int FP4_ROW_BYTES = Fp4Format::BYTES_PER_TOKEN;
static constexpr int FP4_ROW_STRIDE = FP4_ROW_BYTES + 16;
// Q of an fp4 tile = the sum of this many e2m1 terms
#ifndef MLA_SM120_FP4_Q_TERMS
#define MLA_SM120_FP4_Q_TERMS 2
#endif
static constexpr int FP4_Q_TERMS = MLA_SM120_FP4_Q_TERMS;

// A consumer warp holds 16 heads x 256 columns of the output: 32 tiles of 8 columns
static constexpr int NUM_OUT_TILES = D_V / 2 / 8;
// The named barriers of the consumer pairs: PAIR_BARRIER_BASE + m for the pair of warps 2m and 2m+1
static constexpr uint32_t PAIR_BARRIER_BASE = 3;

static_assert(ROW_BYTES % 32 == 16 && FP4_ROW_STRIDE % 32 == 16 && FP4_ROW_STRIDE <= ROW_BYTES);
static_assert(Fp8Format::QUANT_BYTES == D_QK && Fp8Format::QUANT_TILE_SIZE == 32);
static_assert(Fp4Format::QUANT_BYTES == D_QK / 2 && Fp4Format::QUANT_TILE_SIZE == 16);

struct SharedMemoryPlan {
    alignas(128) uint8_t raw[NUM_RAW_BUFS][B_TOPK * ROW_BYTES];     // Rows as stored: 512 B e4m3 + 16 B ue8m0 (or 256 B e2m1 + 32 B e4m3)
    alignas(16) uint8_t v[NUM_V_BUFS][B_TOPK * ROW_BYTES];          // Rebased tiles: e4m3 [row][dim], token r in row v_row_of_token(r)
    alignas(16) float4 s_exchange[NUM_CONSUMER_WARPS][4][32];       // Per consumer warp: its half of S of a tile, in fragment layout
    uint32_t valid_mask[NUM_MASK_BUFS];                             // Bit i: token i of the tile is valid
    uint8_t is_fp4_tile[NUM_MASK_BUFS];                             // The tile comes from the fp4 extra cache
    alignas(16) uint8_t v_scale[NUM_V_BUFS][NUM_SCALE_GROUPS];      // ue8m0 of a rebased tile: one per 32 dims
    uint64_t bar_raw[NUM_RAW_BUFS];                                 // The rows of a tile have arrived
    uint64_t bar_qk_done[NUM_RAW_BUFS];                             // The consumers have finished QK on a tile
    uint64_t bar_rebased[NUM_RAW_BUFS];                             // The rebase warps have written a tile to its V buffer
    uint64_t bar_v_free[NUM_V_BUFS];                                // The consumers have finished PV on a tile
};

// The tiles of a CTA in order: the blocks [start, end) of each of its requests
struct TileCursor {
    int batch_idx, block_idx, end_block_idx;
    int num_orig_blocks, topk_length, extra_topk_length;
};

__device__ __forceinline__ void enter_request(TileCursor &cursor, const SparseAttnDecodeParams &params, int partition_idx, int batch_idx) {
    const Request req = get_request(params, partition_idx, batch_idx);
    cursor.batch_idx = batch_idx;
    cursor.block_idx = req.start_block_idx;
    cursor.end_block_idx = req.end_block_idx;
    cursor.num_orig_blocks = req.num_orig_blocks;
    cursor.topk_length = req.topk_length;
    cursor.extra_topk_length = req.extra_topk_length;
}

// The cache position of each token of the tile under `cursor`, one lane per token; -1 = not a token.
// A load from global memory that has not been touched before: the gather warp issues it one tile ahead.
__device__ __forceinline__ int load_tile_index(const SparseAttnDecodeParams &params, const TileCursor &cursor) {
    const int s_q_idx = cta_s_q_idx();
    const int lane_idx = threadIdx.x % 32;
    const bool is_extra_block = cursor.block_idx >= cursor.num_orig_blocks;
    const int pos = (is_extra_block ? cursor.block_idx - cursor.num_orig_blocks : cursor.block_idx) * B_TOPK + lane_idx;
    const int limit = is_extra_block ? min(cursor.extra_topk_length, params.extra_topk) : min(cursor.topk_length, params.topk);
    int index = -1;
    if (pos < limit) {
        const int *indices = is_extra_block
            ? params.extra_indices + (int64_t)cursor.batch_idx * params.stride_extra_indices_b + (int64_t)s_q_idx * params.stride_extra_indices_s_q
            : params.indices + (int64_t)cursor.batch_idx * params.stride_indices_b + (int64_t)s_q_idx * params.stride_indices_s_q;
        index = indices[pos];
    }
    return index;
}

// Start the copies of tile `tile_idx` into its raw buffer. One lane per token.
// An invalid token is not copied (its row in the cache may hold NaN); its buffer row keeps stale bytes.
template<bool EXTRA_FP4>
__device__ __forceinline__ void gather_tile(const SparseAttnDecodeParams &params, SharedMemoryPlan &plan, int index, bool is_extra_block, uint32_t raw_idx, uint32_t tile_idx) {
    const int lane_idx = threadIdx.x % 32;
    const bool is_token_valid = index != -1;
    const uint32_t valid_mask = __ballot_sync(0xffffffffu, is_token_valid);
    const bool is_fp4 = EXTRA_FP4 && is_extra_block;
    const uint32_t row_bytes = is_fp4 ? FP4_ROW_BYTES : ROW_BYTES;
    uint64_t *bar = &plan.bar_raw[raw_idx];
    if (is_token_valid) {
        const uint32_t page_block_size = is_extra_block ? params.extra_page_block_size : params.page_block_size;
        const int64_t block_stride = is_extra_block ? params.stride_extra_kv_block : params.stride_kv_block;
        const uint8_t *cache = reinterpret_cast<const uint8_t*>(is_extra_block ? params.extra_kv : params.kv);
        const uint32_t page_idx = (uint32_t)index / page_block_size;
        const uint32_t idx_in_page = (uint32_t)index - page_idx * page_block_size;
        bulk_copy_g2s(
            plan.raw[raw_idx] + lane_idx * (is_fp4 ? FP4_ROW_STRIDE : ROW_BYTES),
            cache + page_idx * block_stride + (int64_t)idx_in_page * row_bytes,
            row_bytes,
            bar
        );
    }
    if (lane_idx == 0) {
        plan.valid_mask[tile_idx % NUM_MASK_BUFS] = valid_mask;
        if (EXTRA_FP4) {
            plan.is_fp4_tile[tile_idx % NUM_MASK_BUFS] = is_fp4;
        }
        mbarrier_arrive_expect_tx(bar, __popc(valid_mask) * row_bytes);
    }
}

// The row of a V buffer that token r of a tile goes to: row 4t + j holds the token of k = 4t + j in the A fragment of
// P (see softmax_and_pv), i.e. tokens 2t, 2t + 1, 8 + 2t, 9 + 2t for j = 0..3, and the same 16 higher
__device__ __forceinline__ int v_row_of_token(int r) {
    return (r & 16) | ((r & 6) << 1) | ((r & 8) >> 2) | (r & 1);
}

// fp16 bits of 2^-shift, shift >= 0 (0 once it is below the smallest subnormal)
__device__ __forceinline__ uint32_t f16_pow2_neg(uint32_t shift) {
    return shift <= 14 ? (15 - shift) << 10 : (shift <= 24 ? 1u << (24 - shift) : 0u);
}

// Rebase the 32-dim groups [group_begin, group_end) of the tile in raw buffer `raw_idx` into V buffer `v_idx`:
// per group the largest exponent among the valid tokens becomes the scale of the tile, and every token is divided
// by the power of two that its own scale is below it. Lane = token; an invalid token becomes zeros.
// The division goes through fp16, where it is exact for shifts up to 15, and back to e4m3 with round to nearest.
__device__ __forceinline__ void rebase_tile_part(SharedMemoryPlan &plan, uint32_t raw_idx, uint32_t v_idx, uint32_t tile_idx, int group_begin, int group_end) {
    const int lane_idx = threadIdx.x % 32;
    const bool is_token_valid = (plan.valid_mask[tile_idx % NUM_MASK_BUFS] >> lane_idx) & 1;
    const uint8_t *row = plan.raw[raw_idx] + lane_idx * ROW_BYTES;
    uint8_t *dst = plan.v[v_idx] + v_row_of_token(lane_idx) * ROW_BYTES;
    const uint32_t out_mask = is_token_valid ? 0xffffffffu : 0u;
    #pragma unroll 1
    for (int group = group_begin; group < group_end; ++group) {
        const uint32_t exponent = is_token_valid ? row[Fp8Format::QUANT_BYTES + group] : 0u;
        const uint32_t max_exponent = __reduce_max_sync(0xffffffffu, exponent);
        const uint32_t factor_bits = f16_pow2_neg(max_exponent - exponent) * 0x00010001u;
        const __half2 factor = *reinterpret_cast<const __half2*>(&factor_bits);
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const uint4 raw = *reinterpret_cast<const uint4*>(row + 32 * group + 16 * i);
            const uint32_t words[4] = {raw.x, raw.y, raw.z, raw.w};
            uint32_t vals[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint32_t halves[2];
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const uint32_t f16x2 = e4m3x2_to_f16x2(words[j] >> (16 * h));
                    const __half2 scaled = __hmul2(*reinterpret_cast<const __half2*>(&f16x2), factor);
                    halves[h] = f16x2_to_e4m3x2(*reinterpret_cast<const uint32_t*>(&scaled));
                }
                vals[j] = (halves[0] | (halves[1] << 16)) & out_mask;
            }
            *reinterpret_cast<uint4*>(dst + 32 * group + 16 * i) = uint4{vals[0], vals[1], vals[2], vals[3]};
        }
        if (lane_idx == 0) {
            plan.v_scale[v_idx][group] = max_exponent;
        }
    }
}

// The same for a tile of the fp4 cache (rows at FP4_ROW_STRIDE), whose values are e2m1 times an e4m3 scale per 16 dims:
// per 32-dim group the scale of the tile is the smallest power of two 2^E with amax / 2^E <= 448, and every value is
// rounded once to e4m3 on it. amax is 6 (the largest e2m1 magnitude) times the largest scale of the valid tokens: a
// bound of their magnitudes that the cache writer reaches unless that scale is clamped or subnormal, and one integer
// max per group instead of a search of the codes of every token. The product e2m1 * (scale / 2^E) is exact in fp16
// down to where e4m3 rounds it to zero anyway.
__device__ __forceinline__ void rebase_fp4_tile_part(SharedMemoryPlan &plan, uint32_t raw_idx, uint32_t v_idx, uint32_t tile_idx, int group_begin, int group_end) {
    const int lane_idx = threadIdx.x % 32;
    const bool is_token_valid = (plan.valid_mask[tile_idx % NUM_MASK_BUFS] >> lane_idx) & 1;
    const uint8_t *row = plan.raw[raw_idx] + lane_idx * FP4_ROW_STRIDE;
    uint8_t *dst = plan.v[v_idx] + v_row_of_token(lane_idx) * ROW_BYTES;
    const uint32_t out_mask = is_token_valid ? 0xffffffffu : 0u;
    #pragma unroll 1
    for (int group = group_begin; group < group_end; ++group) {
        // The e4m3 codes of non-negative values order as the values
        const uint32_t scale_codes = *reinterpret_cast<const uint16_t*>(row + Fp4Format::QUANT_BYTES + 2 * group);
        const uint32_t max_code = __reduce_max_sync(0xffffffffu, is_token_valid ? max(scale_codes & 0xffu, scale_codes >> 8) : 0u);
        const float amax = 6.0f * f16_to_f32(e4m3x2_to_f16x2(max_code));
        // E = ceil(log2(amax / 448)) as ue8m0: with amax = m * 2^e, m in [1, 2), it is e - 8, or e - 7 if m > 1.75; 2^0 without valid tokens
        const uint32_t amax_bits = __float_as_uint(amax);
        const uint32_t max_exponent = amax == 0.0f ? 127u : (amax_bits >> 23) - 8 + ((amax_bits & 0x7fffffu) > 0x600000u ? 1 : 0);
        const uint4 data = *reinterpret_cast<const uint4*>(row + 16 * group);
        const uint32_t scales = e4m3x2_to_f16x2(scale_codes);
        const float scale[2] = {f16_to_f32(scales), f16_to_f32(scales >> 16)};
        // The factors scale / 2^E, at most 448 / 6
        const float inv_pow2 = __uint_as_float((254 - max_exponent) << 23);
        const __half2 factor[2] = {__float2half2_rn(scale[0] * inv_pow2), __float2half2_rn(scale[1] * inv_pow2)};
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const uint32_t words[2] = {h == 0 ? data.x : data.z, h == 0 ? data.y : data.w};
            uint32_t vals[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint32_t halves[2];
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const uint32_t f16x2 = e2m1x2_to_f16x2(words[j / 2] >> (8 * (2 * (j % 2) + i)));
                    const __half2 scaled = __hmul2(*reinterpret_cast<const __half2*>(&f16x2), factor[h]);
                    halves[i] = f16x2_to_e4m3x2(*reinterpret_cast<const uint32_t*>(&scaled));
                }
                vals[j] = (halves[0] | (halves[1] << 16)) & out_mask;
            }
            *reinterpret_cast<uint4*>(dst + 32 * group + 16 * h) = uint4{vals[0], vals[1], vals[2], vals[3]};
        }
        if (lane_idx == 0) {
            plan.v_scale[v_idx][group] = max_exponent;
        }
    }
}

template<bool EXTRA_FP4>
__device__ __forceinline__ void producer_main(const SparseAttnDecodeParams &params, SharedMemoryPlan &plan) {
    const int partition_idx = cta_partition_idx();
    const int warp_idx = threadIdx.x / 32 - NUM_CONSUMER_WARPS;
    const int lane_idx = threadIdx.x % 32;

    int begin_req_idx, end_req_idx;
    if (!get_request_range(params, partition_idx, begin_req_idx, end_req_idx)) {
        return;
    }

    // Tiles are counted across the requests of this CTA; ring buffers and barrier phases follow the count
    uint32_t num_tiles = 0;
    #pragma unroll 1
    for (int batch_idx = begin_req_idx; batch_idx <= end_req_idx; ++batch_idx) {
        const Request req = get_request(params, partition_idx, batch_idx);
        num_tiles += max(req.end_block_idx - req.start_block_idx, 0);
    }
    if (num_tiles == 0) {
        return;
    }

    if (warp_idx == 0) {
        // The gather warp. Issuing the 32 copies of a tile takes it about 3000 cycles (they are serialized), and the
        // indices come from memory nobody has touched, so they are loaded one tile ahead of their use.
        TileCursor cursor;
        enter_request(cursor, params, partition_idx, begin_req_idx);
        #pragma unroll 1
        while (cursor.block_idx >= cursor.end_block_idx) {
            enter_request(cursor, params, partition_idx, cursor.batch_idx + 1);
        }
        int next_index = load_tile_index(params, cursor);
        bool is_next_extra_block = cursor.block_idx >= cursor.num_orig_blocks;
        ++cursor.block_idx;

        uint32_t raw_idx = 0, raw_phase = 0;
        #pragma unroll 1
        for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
            if (tile_idx >= NUM_RAW_BUFS) {
                // The last tile of this buffer: the consumers have to be done with its QK and the rebase warps with its rebase.
                // Its barriers cannot complete again before this tile is gathered, so the phase cannot be missed.
                mbarrier_wait(&plan.bar_qk_done[raw_idx], raw_phase ^ 1);
                mbarrier_wait(&plan.bar_rebased[raw_idx], raw_phase ^ 1);
            }
            const int index = next_index;
            const bool is_extra_block = is_next_extra_block;
            if (tile_idx + 1 < num_tiles) {
                #pragma unroll 1
                while (cursor.block_idx >= cursor.end_block_idx) {
                    enter_request(cursor, params, partition_idx, cursor.batch_idx + 1);
                }
                next_index = load_tile_index(params, cursor);
                is_next_extra_block = cursor.block_idx >= cursor.num_orig_blocks;
                ++cursor.block_idx;
            }
            gather_tile<EXTRA_FP4>(params, plan, index, is_extra_block, raw_idx, tile_idx);
            if (++raw_idx == NUM_RAW_BUFS) {
                raw_idx = 0;
                raw_phase ^= 1;
            }
        }
    } else {
        // A rebase warp: a third of the 32-dim groups of every tile
        const int group_begin = warp_idx == 1 ? 0 : (warp_idx == 2 ? 6 : 11);
        const int group_end = warp_idx == 1 ? 6 : (warp_idx == 2 ? 11 : NUM_SCALE_GROUPS);
        uint32_t raw_idx = 0, raw_phase = 0;
        #pragma unroll 1
        for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
            const uint32_t v_idx = tile_idx % NUM_V_BUFS;
            mbarrier_wait(&plan.bar_raw[raw_idx], raw_phase);
            if (tile_idx >= NUM_V_BUFS) {
                // PV of the tile that was in this V buffer; its barrier completes next for this tile's PV
                mbarrier_wait(&plan.bar_v_free[v_idx], ((tile_idx / NUM_V_BUFS) & 1) ^ 1);
            }
            if (EXTRA_FP4 && plan.is_fp4_tile[tile_idx % NUM_MASK_BUFS]) {
                rebase_fp4_tile_part(plan, raw_idx, v_idx, tile_idx, group_begin, group_end);
            } else {
                rebase_tile_part(plan, raw_idx, v_idx, tile_idx, group_begin, group_end);
            }
            if (lane_idx == 0) {
                mbarrier_arrive(&plan.bar_rebased[raw_idx]);
            }
            if (++raw_idx == NUM_RAW_BUFS) {
                raw_idx = 0;
                raw_phase ^= 1;
            }
        }
    }
}

// Q of the heads g and g+8 of a head block over 256 dims, as the A fragments of 8 MMAs (k = 32 dims each).
// `q_row` points at head g and the first of the 256 dims, `row_step` leads to head g+8.
// Per head and MMA: scale = the smallest power of two with |q| / scale <= 448, hi = e4m3(q / scale) and, with two
// terms, lo = e4m3((q / scale - hi) * 16). hi + lo / 16 is q / scale exactly unless |q| / scale < 2^-6.
// `q_scale[term]` receives this thread's bytes of the scale register: thread t of a quad holds the head
// g + 8*(t&1) and the MMAs [4*(t>>1), +4), i.e. byte-id = k % 4 and thread-id = k / 4 for MMA k.
template<int NUM_TERMS>
__device__ __forceinline__ void quantize_q(const bf16 *q_row, int64_t row_step, uint32_t (&q_frag)[NUM_TERMS][8][4], uint32_t (&q_scale)[NUM_TERMS]) {
    const int t = threadIdx.x % 4;
    // Which of the scales this thread supplies, as masks: comparisons here become predicates that live across the
    // request loop and cost a spilled register
    const uint32_t row_mask[2] = {(uint32_t)(t & 1) - 1u, 0u - (uint32_t)(t & 1)};
    const uint32_t step_mask[2] = {(uint32_t)(t >> 1) - 1u, 0u - (uint32_t)(t >> 1)};
    uint32_t scale_bytes = 0;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        // This thread's elements of MMA k: the dims 4t + {0..3} and 16 + 4t + {0..3} of its 32
        float x[2][8], amax[2];
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            amax[r] = 0.0f;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t two = *reinterpret_cast<const uint32_t*>(q_row + r * row_step + 32 * k + 16 * (i / 2) + 4 * t + 2 * (i % 2));
                x[r][2*i] = __uint_as_float(two << 16);
                x[r][2*i+1] = __uint_as_float(two & 0xffff0000u);
                amax[r] = max(amax[r], max(fabsf(x[r][2*i]), fabsf(x[r][2*i+1])));
            }
            amax[r] = max(amax[r], __shfl_xor_sync(0xffffffffu, amax[r], 1));
            amax[r] = max(amax[r], __shfl_xor_sync(0xffffffffu, amax[r], 2));
        }
        const uint32_t codes = f32x2_to_ue8m0x2_up(amax[0] * (1.0f / 448.0f), amax[1] * (1.0f / 448.0f));
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            // The bounds keep the scale of the second term and the inverse scale in range
            const uint32_t code = min(max((codes >> (8 * r)) & 0xff, 2 * Q_LO_SCALE_STEP), 250u);
            const float inv_scale = __uint_as_float((254 - code) << 23);
            scale_bytes |= (code & row_mask[r] & step_mask[k / 4]) << (8 * (k % 4));
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                uint32_t frag[2] = {0, 0};
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const float y0 = x[r][4*i + 2*j] * inv_scale, y1 = x[r][4*i + 2*j + 1] * inv_scale;
                    const uint32_t hi = f32x2_to_e4m3x2(y0, y1);
                    frag[0] |= hi << (16 * j);
                    if (NUM_TERMS == 2) {
                        const uint32_t hi_f16x2 = e4m3x2_to_f16x2(hi);
                        frag[1] |= f32x2_to_e4m3x2((y0 - f16_to_f32(hi_f16x2)) * Q_LO_GAIN, (y1 - f16_to_f32(hi_f16x2 >> 16)) * Q_LO_GAIN) << (16 * j);
                    }
                }
                #pragma unroll
                for (int term = 0; term < NUM_TERMS; ++term) {
                    q_frag[term][k][r + 2 * i] = frag[term];
                }
            }
        }
    }
    #pragma unroll
    for (int term = 0; term < NUM_TERMS; ++term) {
        q_scale[term] = scale_bytes - term * (Q_LO_SCALE_STEP * 0x01010101u);
    }
}

// S += Q K for the MMAs K_STEP and K_STEP + 1 of a warp, whose K fragments come from one ldmatrix.
// `k_scale`: the threads 0 and 1 of the quad of a token hold its scale bytes of the MMAs [0, 4) and [4, 8).
template<int K_STEP, int NUM_TERMS>
__device__ __forceinline__ void qk_two_steps(float (&s_acc)[4], const uint32_t (&q_frag)[NUM_TERMS][8][4], const uint32_t (&q_scale)[NUM_TERMS], const uint32_t (&k_frag)[4], uint32_t k_scale) {
    #pragma unroll
    for (int term = 0; term < NUM_TERMS; ++term) {
        mma_m16n8k32_mxf8<K_STEP % 4, K_STEP / 4, K_STEP % 4, K_STEP / 4>(s_acc, q_frag[term][K_STEP], k_frag[0], k_frag[1], q_scale[term], k_scale);
    }
    #pragma unroll
    for (int term = 0; term < NUM_TERMS; ++term) {
        mma_m16n8k32_mxf8<(K_STEP + 1) % 4, (K_STEP + 1) / 4, (K_STEP + 1) % 4, (K_STEP + 1) / 4>(s_acc, q_frag[term][K_STEP + 1], k_frag[2], k_frag[3], q_scale[term], k_scale);
    }
}

// The smallest e4m3 value s with 6 s >= amax (amax >= 0, so that no e2m1 code of amax / s saturates), as its code;
// at least 1 (the smallest subnormal, for amax = 0) and at most 0x7e (448). Two at once, `lo` in the low byte.
// e4m3(amax / 6) is the answer or the value below it; 6 s is exact in fp32.
__device__ __forceinline__ uint32_t nvfp4_scale_codes(float amax_lo, float amax_hi) {
    const uint32_t codes = f32x2_to_e4m3x2(amax_lo * (1.0f / 6.0f), amax_hi * (1.0f / 6.0f));
    const uint32_t scales = e4m3x2_to_f16x2(codes);
    uint32_t code_lo = codes & 0xff, code_hi = codes >> 8;
    code_lo += code_lo < 0x7e && 6.0f * f16_to_f32(scales) < amax_lo ? 1 : 0;
    code_hi += code_hi < 0x7e && 6.0f * f16_to_f32(scales >> 16) < amax_hi ? 1 : 0;
    return max(code_lo, 1u) | (max(code_hi, 1u) << 8);
}

// Q of the heads g and g+8 of a head block over 256 dims as NQ terms of e2m1 with an e4m3 scale per head and 16 dims:
// per term the A fragments of 4 MMAs (k = 64 dims each). `q_row` points at head g and the first of the 256 dims,
// `row_step` leads to head g+8. Every term takes per head and 16 dims the scale s of nvfp4_scale_codes and the codes
// e2m1(r * (1 / s)), where r is what the terms before it left of q (q itself for the first).
// `q_scale[term][k / 2]` receives the scale register of MMA k that this thread supplies: thread t of a quad holds the
// 4 scales of the head g + 8*(t&1) for the MMAs k with k % 2 == t / 2, i.e. thread-id = k % 2.
template<int NQ>
__device__ __forceinline__ void quantize_q_nvfp4(const bf16 *q_row, int64_t row_step, uint32_t (&q_frag)[NQ][4][4], uint32_t (&q_scale)[NQ][2]) {
    const int t = threadIdx.x % 4;
    const uint32_t own_select = (t & 1) ? 0x4431u : 0x4420u;    // The scales of this thread's head: registers (t&1) and 2 + (t&1)
    const uint32_t word_select = (t >> 1) ? 0x1504u : 0x5140u;  // With those of thread t^2 into the order of the 16-dim blocks
    const uint32_t step_mask = 0u - (uint32_t)(t >> 1);
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        // x[i]: this thread's elements of register i: head g + 8*(i&1), dims 64k + 32*(i>>1) + 8t + {0..7},
        // which are half of the 16-dim block (t>>1) + 2*(i>>1); thread t^1 holds the other half
        float x[4][8];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t two = *reinterpret_cast<const uint32_t*>(q_row + (i & 1) * row_step + 64 * k + 32 * (i >> 1) + 8 * t + 2 * j);
                x[i][2*j] = __uint_as_float(two << 16);
                x[i][2*j+1] = __uint_as_float(two & 0xffff0000u);
            }
        }
        #pragma unroll
        for (int term = 0; term < NQ; ++term) {
            float amax[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                amax[i] = 0.0f;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    amax[i] = fmaxf(amax[i], fabsf(x[i][j]));
                }
                amax[i] = fmaxf(amax[i], __shfl_xor_sync(0xffffffffu, amax[i], 1));
            }
            const uint32_t codes = nvfp4_scale_codes(amax[0], amax[1]) | (nvfp4_scale_codes(amax[2], amax[3]) << 16);
            const uint32_t scales01 = e4m3x2_to_f16x2(codes), scales23 = e4m3x2_to_f16x2(codes >> 16);
            const float scale[4] = {f16_to_f32(scales01), f16_to_f32(scales01 >> 16), f16_to_f32(scales23), f16_to_f32(scales23 >> 16)};
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float inv_scale = __frcp_rn(scale[i]);
                uint32_t frag = 0;
                #pragma unroll
                for (int j = 0; j < 8; j += 2) {
                    const uint32_t two = f32x2_to_e2m1x2(__fmul_rn(x[i][j], inv_scale), __fmul_rn(x[i][j+1], inv_scale));
                    frag |= two << (4 * j);
                    if (term + 1 < NQ) {
                        const uint32_t vals = e2m1x2_to_f16x2(two);
                        x[i][j] = fmaf(-f16_to_f32(vals), scale[i], x[i][j]);
                        x[i][j+1] = fmaf(-f16_to_f32(vals >> 16), scale[i], x[i][j+1]);
                    }
                }
                q_frag[term][k][i] = frag;
            }
            const uint32_t own = __byte_perm(codes, 0, own_select);
            const uint32_t word = __byte_perm(own, __shfl_xor_sync(0xffffffffu, own, 2), word_select);
            if (k % 2 == 0) {
                q_scale[term][k / 2] = word & ~step_mask;
            } else {
                q_scale[term][k / 2] |= word & step_mask;
            }
        }
    }
}

// S += Q K for the MMAs 2*HALF and 2*HALF + 1 of a warp on a tile of the fp4 cache, whose K fragments come from one
// ldmatrix (8 tokens x 128 dims). The scale register of a token for MMA k is word k of its scales in this half of the
// dims, held by thread k of its quad.
template<int HALF, int NQ>
__device__ __forceinline__ void qk_fp4_two_steps(float (&s_acc)[4], const uint32_t (&q_frag)[NQ][4][4], const uint32_t (&q_scale)[NQ][2], const uint32_t (&k_frag)[4], uint32_t k_scale) {
    #pragma unroll
    for (int term = 0; term < NQ; ++term) {
        mma_m16n8k64_nvf4<0, 2 * HALF>(s_acc, q_frag[term][2 * HALF], k_frag[0], k_frag[1], q_scale[term][HALF], k_scale);
    }
    #pragma unroll
    for (int term = 0; term < NQ; ++term) {
        mma_m16n8k64_nvf4<1, 2 * HALF + 1>(s_acc, q_frag[term][2 * HALF + 1], k_frag[2], k_frag[3], q_scale[term][HALF], k_scale);
    }
}

// O += P V for one 8-column tile and one term of P. Every thread holds the same scales; byte V_SCALE_BYTE of `v_scale`
// belongs to these columns.
template<int V_SCALE_BYTE>
__device__ __forceinline__ void pv_step(float (&o_acc)[4], const uint32_t (&p_frag)[4], uint32_t v0, uint32_t v1, uint32_t p_scale, uint32_t v_scale) {
    mma_m16n8k32_mxf8<0, 0, V_SCALE_BYTE, 0>(o_acc, p_frag, v0, v1, p_scale, v_scale);
}

// QK of a tile of the main cache over this warp's half of the dims: s_acc[n] += S of the 16 heads of its pair and the
// tokens [8n, +8). The row of an invalid token holds stale bytes; its S is discarded in the softmax.
template<int NUM_TERMS>
__device__ __forceinline__ void qk_tile_fp8(const uint8_t *raw, float (&s_acc)[4][4], const uint32_t (&q_frag)[NUM_TERMS][8][4], const uint32_t (&q_scale)[NUM_TERMS]) {
    const int lane_idx = threadIdx.x % 32;
    const int q_half = threadIdx.x / 32 % 2;
    const int g = lane_idx / 4, t = lane_idx % 4;
    // The scales of this thread's token of the first output tile; read again at every use, which is
    // cheaper than four more registers held through QK
    const uint32_t *k_scales = reinterpret_cast<const uint32_t*>(raw + g * ROW_BYTES + Fp8Format::QUANT_BYTES + 8 * q_half + 4 * (t & 1));
    // One ldmatrix: 8 tokens x 64 dims, i.e. the B fragments of two MMAs
    const uint32_t base = smem_addr(raw) + (lane_idx % 8) * ROW_BYTES + (lane_idx / 8) * 16 + q_half * (D_QK / 2);
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        #pragma unroll
        for (int n = 0; n < 4; ++n) {
            uint32_t k_frag[4];
            ldmatrix_x4(k_frag, base + n * (8 * ROW_BYTES) + k * 64);
            const uint32_t k_scale = k_scales[n * (8 * ROW_BYTES / 4)];
            switch (k) {
                case 0: qk_two_steps<0, NUM_TERMS>(s_acc[n], q_frag, q_scale, k_frag, k_scale); break;
                case 1: qk_two_steps<2, NUM_TERMS>(s_acc[n], q_frag, q_scale, k_frag, k_scale); break;
                case 2: qk_two_steps<4, NUM_TERMS>(s_acc[n], q_frag, q_scale, k_frag, k_scale); break;
                default: qk_two_steps<6, NUM_TERMS>(s_acc[n], q_frag, q_scale, k_frag, k_scale); break;
            }
        }
    }
}

// The same for a tile of the fp4 cache: rows at FP4_ROW_STRIDE, 128 B of e2m1 and 16 B of scales per half of the dims
template<int NQ>
__device__ __forceinline__ void qk_tile_fp4(const uint8_t *raw, float (&s_acc)[4][4], const uint32_t (&q_frag)[NQ][4][4], const uint32_t (&q_scale)[NQ][2]) {
    const int lane_idx = threadIdx.x % 32;
    const int q_half = threadIdx.x / 32 % 2;
    const int g = lane_idx / 4, t = lane_idx % 4;
    // Thread t of a quad: the scales of MMA t (dims [64t, +64) of this half) of its token
    const uint32_t *k_scales = reinterpret_cast<const uint32_t*>(raw + g * FP4_ROW_STRIDE + Fp4Format::QUANT_BYTES + 16 * q_half + 4 * t);
    const uint32_t base = smem_addr(raw) + (lane_idx % 8) * FP4_ROW_STRIDE + (lane_idx / 8) * 16 + q_half * (D_QK / 4);
    #pragma unroll
    for (int half = 0; half < 2; ++half) {
        #pragma unroll
        for (int n = 0; n < 4; ++n) {
            uint32_t k_frag[4];
            ldmatrix_x4(k_frag, base + n * (8 * FP4_ROW_STRIDE) + half * 64);
            const uint32_t k_scale = k_scales[n * (8 * FP4_ROW_STRIDE / 4)];
            if (half == 0) {
                qk_fp4_two_steps<0, NQ>(s_acc[n], q_frag, q_scale, k_frag, k_scale);
            } else {
                qk_fp4_two_steps<1, NQ>(s_acc[n], q_frag, q_scale, k_frag, k_scale);
            }
        }
    }
}

// Everything of a tile after QK, for a consumer warp: its half of S and that of its partner meet, the softmax updates
// the running max and sum of the pair's 16 heads (both warps of a pair do the same, with the same result), O is rescaled
// where the max moved, and O += P V over this warp's 256 output columns. P never leaves the registers: the S fragments of
// a thread hold the tokens {8n + 2t, +1} (n = 0..3), which are its A fragment of the PV MMA for the order of k given by
// v_row_of_token, in which the rebase stores the tokens.
template<int NUM_TERMS>
__device__ __forceinline__ void softmax_and_pv(
    const SparseAttnDecodeParams &params, SharedMemoryPlan &plan, uint32_t tile_idx,
    float (&s_acc)[4][4], float (&o_acc)[NUM_OUT_TILES][4], float (&mi)[2], float (&li)[2]
) {
    const int warp_idx = threadIdx.x / 32;
    const int lane_idx = threadIdx.x % 32;
    const int t = lane_idx % 4;
    const uint32_t pair_barrier = PAIR_BARRIER_BASE + warp_idx / 2;

    // The halves of S meet in shared memory. The first barrier: the partner has read what this warp wrote for the tile
    // before
    named_barrier_sync(pair_barrier, 64);
    #pragma unroll
    for (int n = 0; n < 4; ++n) {
        plan.s_exchange[warp_idx][n][lane_idx] = float4{s_acc[n][0], s_acc[n][1], s_acc[n][2], s_acc[n][3]};
    }
    named_barrier_sync(pair_barrier, 64);
    {
        // Bit 8n + i: the token of column i of output tile n
        const uint32_t valid_mask = plan.valid_mask[tile_idx % NUM_MASK_BUFS] >> (2 * t);
        #pragma unroll
        for (int n = 0; n < 4; ++n) {
            const float4 other = plan.s_exchange[warp_idx ^ 1][n][lane_idx];
            const float other_vals[4] = {other.x, other.y, other.z, other.w};
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                // The same sum in both warps, as addition commutes
                const float s = (s_acc[n][i] + other_vals[i]) * params.sm_scale_div_log2;
                s_acc[n][i] = ((valid_mask >> (8 * n + (i & 1))) & 1) ? s : -CUDART_INF_F;
            }
        }
    }

    // P as the A fragments of the PV MMAs: register r holds head g + 8r and k [4t, +4), i.e. its tokens of the output
    // tiles 0 and 1 of S, register 2 + r the same for k [16 + 4t, +4) and the output tiles 2 and 3
    uint32_t p_frag[NUM_TERMS][4];
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

        // The sum uses P before quantization, as the bf16 path does
        float cur_sum = 0.0f;
        uint32_t two[NUM_TERMS][4];
        #pragma unroll
        for (int n = 0; n < 4; ++n) {
            const float p0 = exp2f(s_acc[n][2*r] - mi[r]), p1 = exp2f(s_acc[n][2*r+1] - mi[r]);
            cur_sum += p0 + p1;
            two[0][n] = f32x2_to_e4m3x2(p0 * P_GAIN, p1 * P_GAIN);
            if constexpr (NUM_TERMS == 2) {
                const uint32_t hi_f16x2 = e4m3x2_to_f16x2(two[0][n]);
                two[1][n] = f32x2_to_e4m3x2(
                    (p0 * P_GAIN - f16_to_f32(hi_f16x2)) * P_LO_GAIN,
                    (p1 * P_GAIN - f16_to_f32(hi_f16x2 >> 16)) * P_LO_GAIN
                );
            }
        }
        li[r] = li[r] * scale_for_old[r] + cur_sum;
        #pragma unroll
        for (int term = 0; term < NUM_TERMS; ++term) {
            p_frag[term][r] = two[term][0] | (two[term][1] << 16);
            p_frag[term][2 + r] = two[term][2] | (two[term][3] << 16);
        }
    }

    // Scale O for the heads whose running max moved
    if (__any_sync(0xffffffffu, should_scale_o)) {
        #pragma unroll
        for (int j = 0; j < NUM_OUT_TILES; ++j) {
            o_acc[j][0] *= scale_for_old[0];
            o_acc[j][1] *= scale_for_old[0];
            o_acc[j][2] *= scale_for_old[1];
            o_acc[j][3] *= scale_for_old[1];
        }
    }

    // PV: O[head][dim] over this warp's 256 output columns, one ldmatrix (32 tokens x 16 dims) per two output tiles
    const uint32_t v_idx = tile_idx % NUM_V_BUFS;
    {
        const uint32_t round = __umulhi(tile_idx, 0xaaaaaaabu) >> 1;
        mbarrier_wait(&plan.bar_rebased[tile_idx - NUM_RAW_BUFS * round], round & 1);
    }
    {
        // The scales of this warp's 8 32-dim groups, 4 per register; every thread holds them all
        const uint2 v_scales = *reinterpret_cast<const uint2*>(plan.v_scale[v_idx] + 8 * (warp_idx % 2));
        // V: lane l addresses row l
        const uint32_t v_base = smem_addr(plan.v[v_idx]) + lane_idx * ROW_BYTES + (warp_idx % 2) * (D_V / 2);
        #pragma unroll
        for (int c = 0; c < NUM_OUT_TILES / 2; ++c) {
            uint32_t v_frag[4];
            ldmatrix_x2_trans_b8(v_frag, v_base + 16 * c);
            const uint32_t v_scale = c < NUM_OUT_TILES / 4 ? v_scales.x : v_scales.y;
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                #pragma unroll
                for (int term = 0; term < NUM_TERMS; ++term) {
                    const uint32_t p_scale = term == 0 ? P_SCALE_HI : P_SCALE_LO;
                    // The columns [16c, +16) are in the 32-dim group c / 2 of the warp: byte (c / 2) % 4 of its register
                    switch ((c / 2) % 4) {
                        case 0: pv_step<0>(o_acc[2 * c + h], p_frag[term], v_frag[h], v_frag[2 + h], p_scale, v_scale); break;
                        case 1: pv_step<1>(o_acc[2 * c + h], p_frag[term], v_frag[h], v_frag[2 + h], p_scale, v_scale); break;
                        case 2: pv_step<2>(o_acc[2 * c + h], p_frag[term], v_frag[h], v_frag[2 + h], p_scale, v_scale); break;
                        default: pv_step<3>(o_acc[2 * c + h], p_frag[term], v_frag[h], v_frag[2 + h], p_scale, v_scale); break;
                    }
                }
            }
        }
    }
    if (lane_idx == 0) {
        mbarrier_arrive(&plan.bar_v_free[v_idx]);
    }
}

// The start of a request, for the consumer warps: O = 0, and no max and no sum yet. After Q has been quantized,
// which leaves the quantization more registers.
__device__ __forceinline__ void start_request(float (&o_acc)[NUM_OUT_TILES][4], float (&mi)[2], float (&li)[2]) {
    #pragma unroll
    for (int j = 0; j < NUM_OUT_TILES; ++j) {
        o_acc[j][0] = o_acc[j][1] = o_acc[j][2] = o_acc[j][3] = 0.0f;
    }
    mi[0] = mi[1] = MAX_INIT_VAL;
    li[0] = li[1] = 0.0f;
}

// One tile of the main cache (QK_FP4 = false) or of the fp4 extra cache, for the consumer warps
template<int NUM_TERMS, bool QK_FP4, typename QFrag, typename QScale>
__device__ __forceinline__ void consume_tile(
    const SparseAttnDecodeParams &params, SharedMemoryPlan &plan, uint32_t tile_idx, const QFrag &q_frag, const QScale &q_scale,
    float (&o_acc)[NUM_OUT_TILES][4], float (&mi)[2], float (&li)[2]
) {
    // Raw buffer and barrier phase of the tile; one multiply instead of a division, as registers are short here
    const uint32_t buf_round = __umulhi(tile_idx, 0xaaaaaaabu) >> 1;
    const uint32_t buf_idx = tile_idx - NUM_RAW_BUFS * buf_round;
    mbarrier_wait(&plan.bar_raw[buf_idx], buf_round & 1);

    // QK: S[head][token] over this warp's half of the dims
    float s_acc[4][4];
    #pragma unroll
    for (int n = 0; n < 4; ++n) {
        s_acc[n][0] = s_acc[n][1] = s_acc[n][2] = s_acc[n][3] = 0.0f;
    }
    if constexpr (QK_FP4) {
        qk_tile_fp4<FP4_Q_TERMS>(plan.raw[buf_idx], s_acc, q_frag, q_scale);
    } else {
        qk_tile_fp8<NUM_TERMS>(plan.raw[buf_idx], s_acc, q_frag, q_scale);
    }
    if (threadIdx.x % 32 == 0) {
        mbarrier_arrive(&plan.bar_qk_done[buf_idx]);
    }
    softmax_and_pv<NUM_TERMS>(params, plan, tile_idx, s_acc, o_acc, mi, li);
}

// The end of a request, for a consumer warp: turns the softmax state of its heads into the output scale and the LSE,
// then scales and stores its 16 heads x 256 columns of the output. `mi` / `li`: the max of the heads g and g+8 and this
// thread's share of their sums. The first warp of a pair writes the LSE.
__device__ __forceinline__ void finish_request(
    const SparseAttnDecodeParams &params, int partition_idx, int batch_idx, const float (&mi)[2], const float (&li)[2],
    const float (&o_acc)[NUM_OUT_TILES][4]
) {
    const int s_q_idx = cta_s_q_idx();
    const int warp_idx = threadIdx.x / 32;
    const int lane_idx = threadIdx.x % 32;
    const int g = lane_idx / 4, t = lane_idx % 4;
    const int head = cta_head_block() * NUM_HEADS + 16 * (warp_idx / 2) + g;   // Of row g; row g+8 is head + 8
    const bool writes_lse = warp_idx % 2 == 0 && t == 0;

    const Request req = get_request(params, partition_idx, batch_idx);
    float o_scale[2];
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        float m = mi[r], l = li[r];
        l += __shfl_xor_sync(0xffffffffu, l, 1);
        l += __shfl_xor_sync(0xffffffffu, l, 2);
        if (m == MAX_INIT_VAL) {
            // No valid token. We set l to 0 to fit the definition that l := exp(x[i] - m)
            l = 0.0f;
            m = -CUDART_INF_F;
        }
        const int head_idx = head + 8 * r;
        if (req.is_no_split) {
            const float attn_sink = params.attn_sink == nullptr ? -CUDART_INF_F : params.attn_sink[head_idx] * CUDART_L2E_F;
            o_scale[r] = l == 0.0f ? 0.0f : 1.0f / (l + exp2f(attn_sink - m));
            float cur_lse = m * CUDART_LN2_F + logf(l);
            cur_lse = cur_lse == -CUDART_INF_F ? +CUDART_INF_F : cur_lse;
            if (writes_lse) {
                params.lse[(int64_t)batch_idx * params.stride_lse_b + (int64_t)s_q_idx * params.stride_lse_s_q + head_idx] = cur_lse;
            }
        } else {
            // Here we leave attn_sink to the combine kernel, otherwise attn_sink will take effect for multiple times
            o_scale[r] = l == 0.0f ? 0.0f : 1.0f / l;
            if (writes_lse) {
                params.lse_accum[(int64_t)req.n_split_idx * params.stride_lse_accum_split + (int64_t)s_q_idx * params.stride_lse_accum_s_q + head_idx] = log2f(l) + m;
            }
        }
    }

    // Scale and store: this thread holds rows g and g+8 and pairs of adjacent columns of every output tile
    const int col = (warp_idx % 2) * (D_V / 2) + 2 * t;
    if (req.is_no_split) {
        bf16 *row0 = params.out + (int64_t)batch_idx * params.stride_o_b + (int64_t)s_q_idx * params.stride_o_s_q
            + (int64_t)head * params.stride_o_h_q + col;
        bf16 *row1 = row0 + 8 * (int64_t)params.stride_o_h_q;
        #pragma unroll
        for (int j = 0; j < NUM_OUT_TILES; ++j) {
            *reinterpret_cast<__nv_bfloat162*>(row0 + 8 * j) = __float22bfloat162_rn(float2{o_acc[j][0] * o_scale[0], o_acc[j][1] * o_scale[0]});
            *reinterpret_cast<__nv_bfloat162*>(row1 + 8 * j) = __float22bfloat162_rn(float2{o_acc[j][2] * o_scale[1], o_acc[j][3] * o_scale[1]});
        }
    } else {
        float *row0 = params.o_accum + (int64_t)req.n_split_idx * params.stride_o_accum_split + (int64_t)s_q_idx * params.stride_o_accum_s_q
            + (int64_t)head * params.stride_o_accum_h_q + col;
        float *row1 = row0 + 8 * (int64_t)params.stride_o_accum_h_q;
        #pragma unroll
        for (int j = 0; j < NUM_OUT_TILES; ++j) {
            *reinterpret_cast<float2*>(row0 + 8 * j) = float2{o_acc[j][0] * o_scale[0], o_acc[j][1] * o_scale[0]};
            *reinterpret_cast<float2*>(row1 + 8 * j) = float2{o_acc[j][2] * o_scale[1], o_acc[j][3] * o_scale[1]};
        }
    }
}

// This thread's first row of Q in a request: head g of its warp's head block, the first of its half of the dims.
// Head g+8 is 8 rows further.
__device__ __forceinline__ const bf16 *q_row(const SparseAttnDecodeParams &params, int batch_idx) {
    const int warp_idx = threadIdx.x / 32;
    const int g = threadIdx.x % 32 / 4;
    return params.q + (int64_t)batch_idx * params.stride_q_b + (int64_t)cta_s_q_idx() * params.stride_q_s_q
        + (int64_t)(cta_head_block() * NUM_HEADS + 16 * (warp_idx / 2) + g) * params.stride_q_h_q + (D_QK / 2) * (warp_idx % 2);
}

template<int NUM_TERMS, bool EXTRA_FP4>
__device__ __forceinline__ void consumer_main(const SparseAttnDecodeParams &params, SharedMemoryPlan &plan) {
    const int partition_idx = cta_partition_idx();

    int begin_req_idx, end_req_idx;
    if (!get_request_range(params, partition_idx, begin_req_idx, end_req_idx)) {
        return;
    }

    uint32_t tile_idx = 0;  // The same count as in producer_main
    #pragma unroll 1
    for (int batch_idx = begin_req_idx; batch_idx <= end_req_idx; ++batch_idx) {
        int block_idx, end_block_idx, fp8_end_block_idx;
        {
            const Request req = get_request(params, partition_idx, batch_idx);
            block_idx = req.start_block_idx;
            end_block_idx = req.end_block_idx;
            // The tiles of the main cache come first; with EXTRA_FP4 those of the extra cache are fp4 tiles
            fp8_end_block_idx = EXTRA_FP4 ? min(end_block_idx, req.num_orig_blocks) : end_block_idx;
        }
        float o_acc[NUM_OUT_TILES][4];  // The output tiles of this warp's 256 columns
        float mi[2], li[2];             // The softmax state of the heads g and g+8
        if constexpr (!EXTRA_FP4) {
            // Q as the A fragments of the 8 MMAs (k = 32 dims each) of this warp
            uint32_t q_frag[NUM_TERMS][8][4], q_scale[NUM_TERMS];
            quantize_q<NUM_TERMS>(q_row(params, batch_idx), 8 * (int64_t)params.stride_q_h_q, q_frag, q_scale);
            start_request(o_acc, mi, li);
            #pragma unroll 1
            for (; block_idx < end_block_idx; ++block_idx, ++tile_idx) {
                consume_tile<NUM_TERMS, false>(params, plan, tile_idx, q_frag, q_scale, o_acc, mi, li);
            }
        } else {
            if (block_idx < fp8_end_block_idx) {
                uint32_t q_frag[NUM_TERMS][8][4], q_scale[NUM_TERMS];
                quantize_q<NUM_TERMS>(q_row(params, batch_idx), 8 * (int64_t)params.stride_q_h_q, q_frag, q_scale);
                start_request(o_acc, mi, li);
                #pragma unroll 1
                for (; block_idx < fp8_end_block_idx; ++block_idx, ++tile_idx) {
                    consume_tile<NUM_TERMS, false>(params, plan, tile_idx, q_frag, q_scale, o_acc, mi, li);
                }
            } else {
                start_request(o_acc, mi, li);
            }
            if (block_idx < end_block_idx) {
                // Q again, as the A fragments of the 4 fp4 MMAs (k = 64 dims each) of this warp
                uint32_t q_frag[FP4_Q_TERMS][4][4], q_scale[FP4_Q_TERMS][2];
                quantize_q_nvfp4<FP4_Q_TERMS>(q_row(params, batch_idx), 8 * (int64_t)params.stride_q_h_q, q_frag, q_scale);
                #pragma unroll 1
                for (; block_idx < end_block_idx; ++block_idx, ++tile_idx) {
                    consume_tile<NUM_TERMS, true>(params, plan, tile_idx, q_frag, q_scale, o_acc, mi, li);
                }
            }
        }

        finish_request(params, partition_idx, batch_idx, mi, li, o_acc);
    }
}

// The name keeps the upstream substring that the benchmark scripts look for
template<int NUM_TERMS, bool EXTRA_FP4>
__global__ void __launch_bounds__(NUM_THREADS, 1)
flash_fwd_splitkv_mla_fp8_sparse_kernel(__grid_constant__ const SparseAttnDecodeParams params) {
    extern __shared__ __align__(128) char wksp_buf[];
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int i = 0; i < NUM_RAW_BUFS; ++i) {
            mbarrier_init(&plan.bar_raw[i], 1);
            mbarrier_init(&plan.bar_qk_done[i], NUM_CONSUMER_WARPS);
            mbarrier_init(&plan.bar_rebased[i], NUM_REBASE_WARPS);
        }
        #pragma unroll
        for (int i = 0; i < NUM_V_BUFS; ++i) {
            mbarrier_init(&plan.bar_v_free[i], NUM_CONSUMER_WARPS);
        }
        fence_mbarrier_init();
    }
    __syncthreads();

    // 256 * 232 + 128 * 40 = 256 * 224 + 128 * 56 = 384 * 168 registers: the split has to stay within what the block
    // got at launch, or the last warp waits in setmaxnreg.inc for ever. The rebase runs faster with 56; the consumers
    // with two terms need 232.
    if (threadIdx.x < NUM_CONSUMER_THREADS) {
        setmaxnreg_inc<NUM_TERMS == 1 ? 224 : 232>();
        consumer_main<NUM_TERMS, EXTRA_FP4>(params, plan);
    } else {
        setmaxnreg_dec<NUM_TERMS == 1 ? 56 : 40>();
        producer_main<EXTRA_FP4>(params, plan);
    }
}

template<int NUM_TERMS, bool EXTRA_FP4>
static void launch(const SparseAttnDecodeParams &params) {
    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    static_assert(smem_size <= 99 * 1024);  // The limit of a block on sm120

    auto kernel = &flash_fwd_splitkv_mla_fp8_sparse_kernel<NUM_TERMS, EXTRA_FP4>;
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    kernel<<<decode_grid(params), dim3(NUM_THREADS, 1, 1), smem_size, params.stream>>>(params);
    KU_CHECK_KERNEL_LAUNCH();
}

}   // namespace mxfp8_path

void run_mxfp8(const SparseAttnDecodeParams &params, int num_terms) {
    KU_ASSERT(num_terms == 1 || num_terms == 2);
    const bool is_extra_fp4 = params.extra_topk > 0 && params.extra_model_type == ModelType::V41_FP4;
    KU_ASSERT(params.model_type == ModelType::V41 && (params.extra_topk == 0 || params.extra_model_type == ModelType::V41 || is_extra_fp4),
        "the block-scaled kernel reads a V4.1 fp8 main cache and a V4.1 fp8 or fp4 extra cache");
    KU_ASSERT(!is_extra_fp4 || num_terms == 1, "the block-scaled kernel reads an fp4 extra cache with one term only");
    if (num_terms == 2) {
        mxfp8_path::launch<2, false>(params);
    } else if (is_extra_fp4) {
        mxfp8_path::launch<1, true>(params);
    } else {
        mxfp8_path::launch<1, false>(params);
    }
}

}
