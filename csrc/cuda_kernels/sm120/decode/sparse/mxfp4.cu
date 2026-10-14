/*
Sparse MLA Decoding — SM120 (RTX 50 series / RTX PRO Blackwell), the MXFP4 KV cache path

The KV cache is the MXFP4 cache (ModelType::V41_MXFP4, 272 B per token: 256 B of e2m1 followed by 16 ue8m0 scales, one for
every 32 dims; see KVCacheLayout.V41_MXFP4Sparse in tests/quant.py), for the main cache and, if there is one, the extra cache.
QK runs on the fp4 MMA (kind::mxf4, scale_vec::2X, twice the rate of the block-scaled e4m3 MMA), PV on the block-scaled e4m3 MMA
like the path of mxfp8.cu, whose structure this file follows (see there for the CTA layout, the consumer pairs and the pipeline).

- QK. K is used as stored: the e2m1 bytes of a token are the B operand and its 16 ue8m0 scales are the scales of its column, so
  nothing is lost on this side. Q is quantized once per request to NUM_Q_TERMS terms of e2m1 with one ue8m0 scale per head and
  32 dims: each term is the e2m1 rounding of what the terms before it left of Q (Q itself for the first), on the smallest power
  of two that holds the maximum of the remainder in 6. One QK MMA stream per term.
- PV. The scales of V belong to (token, 32 dims) and a token is an element of k here, so they cannot be the scales of a column.
  The producers rebase every tile: per 32 dims they take the largest scale 2^s_max of the valid tokens, E = s_max - V_HEADROOM as
  the scale of the tile's columns, and give every token the e4m3 value e2m1 * 2^(s - E). V_HEADROOM = 6 is as high as the
  largest e2m1 magnitude allows (6 * 2^6 = 384 <= 448), which leaves 14 binades of scale difference between the tokens of a
  tile before a value of the smaller one falls out of the e4m3 range (the e4m3 of a rebased FP8 tile has 8). The product is
  exact down to there; below it it is rounded to nearest. P is quantized to e4m3 with a constant scale.
- NUM_P_TERMS = 2 writes P as the sum of two e4m3 terms (the rounded value and the rounded remainder, with a scale 16 times
  smaller) and runs two PV MMAs for each; NUM_P_TERMS = 1 keeps 4 significant bits of P. The modes of run.cu: mxfp4x4 is
  NUM_Q_TERMS = 4 with NUM_P_TERMS = 2, mxfp8 is NUM_Q_TERMS = 2 with NUM_P_TERMS = 1. Four terms of Q leave an error of about 1e-4 of
  |Q|, which moves the LSE of a query token that attends a single KV token by up to 3e-6, more than the upstream tolerance (1e-6 absolute
  and 8/65536 relative: it matters where the LSE is below 0.01); the output is well within it. A fifth term (error 1e-5) would pass, but its
  16 registers or 4 KB per warp do not fit next to the 128 registers of O and the 99 KB of shared memory.

I/O: See SparseAttnDecodeParams in csrc/params.h
*/
#include "common.h"

namespace sm120::decode::sparse {
namespace mxfp4_path {

using Mxfp4Format = KVCacheFormat<ModelType::V41_MXFP4>;

// Producer warp 0 gathers, the others rebase
static constexpr int NUM_REBASE_WARPS = NUM_PRODUCER_WARPS - 1;
static constexpr int NUM_RAW_BUFS = 3;      // Up to three tiles (26 KB) in flight
static_assert(NUM_RAW_BUFS == 3, "the consumers divide by 3 with __umulhi");
static constexpr int NUM_V_BUFS = 2;        // The tile at PV and the next one, so that a rebase does not wait for the PV before it
static constexpr int NUM_MASK_BUFS = 4;     // A mask lives from the gather of its tile to its softmax: NUM_RAW_BUFS + 1 tiles
static constexpr int NUM_SCALE_GROUPS = D_V / Mxfp4Format::QUANT_TILE_SIZE;
// ldmatrix reads 8 or 16 rows at once: a row stride that is an odd multiple of 16 B spreads them over all banks
static constexpr int ROW_BYTES = Mxfp4Format::BYTES_PER_TOKEN;      // A raw row: 256 B e2m1 + 16 B ue8m0
static constexpr int SCALE_OFFSET = Mxfp4Format::QUANT_BYTES;
static constexpr int V_ROW_BYTES = D_V + 16;                        // A row of a rebased tile: 512 B e4m3
// P = hi * 2^-2 + lo * 2^-6. Lazy rescaling keeps P <= 2^RESCALE_THRESHOLD, so hi <= 256 fits e4m3 (max 448)
static constexpr float P_GAIN = 4.0f, P_LO_GAIN = 16.0f;
static constexpr uint32_t P_SCALE_HI = 0x7d7d7d7du, P_SCALE_LO = 0x79797979u;
// The tile scale of V is 2^(largest scale - V_HEADROOM): the largest value of the tile, 6 * 2^V_HEADROOM, has to fit e4m3
static constexpr int V_HEADROOM = 6;
static_assert(6 * (1 << V_HEADROOM) <= 448);
static constexpr int MAX_Q_TERMS = 4;
// With four terms the A fragments of Q (64 registers) and the 128 of O leave the consumers too little room: the last slots
// (term, MMA k; order term * 4 + k) of 4 registers each are read back from shared memory when they are used. Fewer terms stay in registers.
static constexpr int MAX_SMEM_Q_SLOTS = 5;
constexpr int num_smem_q_slots(int num_q_terms) { return num_q_terms >= 4 ? MAX_SMEM_Q_SLOTS : 0; }
// The place in shared memory of the slot (term, k) of nq terms, negative if it is in registers
constexpr int smem_q_slot(int nq, int term, int k) { return term * 4 + k - (nq * 4 - num_smem_q_slots(nq)); }
// The ue8m0 code of a Q scale is kept in this range, so that the scale and its inverse are normal floats
static constexpr uint32_t Q_MIN_CODE = 8, Q_MAX_CODE = 250;

// A consumer warp holds 16 heads x 256 columns of the output: 32 tiles of 8 columns
static constexpr int NUM_OUT_TILES = D_V / 2 / 8;
// The named barriers of the consumer pairs: PAIR_BARRIER_BASE + m for the pair of warps 2m and 2m+1
static constexpr uint32_t PAIR_BARRIER_BASE = 3;

static_assert(ROW_BYTES % 32 == 16 && V_ROW_BYTES % 32 == 16);
static_assert(Mxfp4Format::QUANT_BYTES == D_QK / 2 && Mxfp4Format::QUANT_TILE_SIZE == 32 && NUM_SCALE_GROUPS == 16);

struct SharedMemoryPlan {
    alignas(128) uint8_t raw[NUM_RAW_BUFS][B_TOPK * ROW_BYTES];     // Rows as stored
    alignas(16) uint8_t v[NUM_V_BUFS][B_TOPK * V_ROW_BYTES];        // Rebased tiles: e4m3 [row][dim], token r in row v_row_of_token(r)
    alignas(16) float4 s_exchange[NUM_CONSUMER_WARPS][4][32];       // Per consumer warp: its half of S of a tile, in fragment layout
    uint32_t q_scale[NUM_CONSUMER_WARPS][MAX_Q_TERMS][32];          // Per consumer warp and Q term: the scale register of every lane
    alignas(16) uint4 q_smem[NUM_CONSUMER_WARPS][MAX_SMEM_Q_SLOTS][32];  // Per consumer warp and slot of Q: the A fragment of every lane
    uint32_t valid_mask[NUM_MASK_BUFS];                             // Bit i: token i of the tile is valid
    alignas(16) uint8_t v_scale[NUM_V_BUFS][NUM_SCALE_GROUPS];      // ue8m0 of a rebased tile: one per 32 dims
    uint64_t bar_raw[NUM_RAW_BUFS];                                 // The rows of a tile have arrived
    uint64_t bar_qk_done[NUM_RAW_BUFS];                             // The consumers have finished QK on a tile
    uint64_t bar_rebased[NUM_RAW_BUFS];                             // The rebase warps have written a tile to its V buffer
    uint64_t bar_v_free[NUM_V_BUFS];                                // The consumers have finished PV on a tile
};

// d[16x8] += (a[16x64] * sa) * (b[64x8] * sb): e2m1 inputs with a power-of-two scale (ue8m0: the byte x means 2^(x - 127)) for every 32
// elements of k of each row of A and column of B, and an fp32 accumulator; twice the rate of the block-scaled e4m3 MMA. Eight elements
// per register, the first in the low nibble: thread (g, t) holds a[i] = A(g + 8*(i&1), 32*(i>>1) + 8t .. +7),
// b0 / b1 = B(8t .. +7 / 32 + 8t .. +7, g), and d as for bf16. Bytes BYTE and BYTE+1 of a scale register cover k [0, 32) and [32, 64):
// row m takes the `sfa` register of thread 4*(m%8) + 2*THREAD_A + (m >= 8), column n the `sfb` register of thread 4*n + THREAD_B.
// BYTE_A and BYTE_B are 0 or 2.
template<int BYTE_A, int THREAD_A, int BYTE_B, int THREAD_B>
__device__ __forceinline__ void mma_m16n8k64_mxf4(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb) {
    static constexpr uint16_t byte_a = BYTE_A, thread_a = THREAD_A, byte_b = BYTE_B, thread_b = THREAD_B;
    asm volatile("mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
                   "r"(sfa), "h"(byte_a), "h"(thread_a), "r"(sfb), "h"(byte_b), "h"(thread_b));
}

// 16 bytes from shared memory. Volatile, so that the compiler keeps the load next to its use instead of holding the registers
// from the first use of the address to the last
__device__ __forceinline__ uint4 load_shared_v4(const uint4 *ptr) {
    uint4 v;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(smem_addr(ptr)));
    return v;
}

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
__device__ __forceinline__ void gather_tile(const SparseAttnDecodeParams &params, SharedMemoryPlan &plan, int index, bool is_extra_block, uint32_t raw_idx, uint32_t tile_idx) {
    const int lane_idx = threadIdx.x % 32;
    const bool is_token_valid = index != -1;
    const uint32_t valid_mask = __ballot_sync(0xffffffffu, is_token_valid);
    uint64_t *bar = &plan.bar_raw[raw_idx];
    if (is_token_valid) {
        const uint32_t page_block_size = is_extra_block ? params.extra_page_block_size : params.page_block_size;
        const int64_t block_stride = is_extra_block ? params.stride_extra_kv_block : params.stride_kv_block;
        const uint8_t *cache = reinterpret_cast<const uint8_t*>(is_extra_block ? params.extra_kv : params.kv);
        const uint32_t page_idx = (uint32_t)index / page_block_size;
        const uint32_t idx_in_page = (uint32_t)index - page_idx * page_block_size;
        bulk_copy_g2s(
            plan.raw[raw_idx] + lane_idx * ROW_BYTES,
            cache + page_idx * block_stride + (int64_t)idx_in_page * ROW_BYTES,
            ROW_BYTES,
            bar
        );
    }
    if (lane_idx == 0) {
        plan.valid_mask[tile_idx % NUM_MASK_BUFS] = valid_mask;
        mbarrier_arrive_expect_tx(bar, __popc(valid_mask) * ROW_BYTES);
    }
}

// The row of a V buffer that token r of a tile goes to: row 4t + j holds the token of k = 4t + j in the A fragment of
// P (see softmax_and_pv), i.e. tokens 2t, 2t + 1, 8 + 2t, 9 + 2t for j = 0..3, and the same 16 higher
__device__ __forceinline__ int v_row_of_token(int r) {
    return (r & 16) | ((r & 6) << 1) | ((r & 8) >> 2) | (r & 1);
}

// fp16 bits of 2^k for k <= 15 (0 once it is below the smallest subnormal)
__device__ __forceinline__ uint32_t f16_pow2(int k) {
    return k >= -14 ? (uint32_t)(15 + k) << 10 : (k >= -24 ? 1u << (k + 24) : 0u);
}

// Rebase the 32-dim groups [group_begin, group_end) of the tile in raw buffer `raw_idx` into V buffer `v_idx`: per group the largest
// scale among the valid tokens, less V_HEADROOM, becomes the scale of the tile, and the e2m1 values of every token are multiplied
// by the power of two that its own scale is above it. Lane = token; an invalid token becomes zeros.
// The product goes through fp16, where it is exact for scales up to 23 binades below the tile's, and to e4m3 with round to nearest.
__device__ __forceinline__ void rebase_tile_part(SharedMemoryPlan &plan, uint32_t raw_idx, uint32_t v_idx, uint32_t tile_idx, int group_begin, int group_end) {
    const int lane_idx = threadIdx.x % 32;
    const bool is_token_valid = (plan.valid_mask[tile_idx % NUM_MASK_BUFS] >> lane_idx) & 1;
    const uint8_t *row = plan.raw[raw_idx] + lane_idx * ROW_BYTES;
    uint8_t *dst = plan.v[v_idx] + v_row_of_token(lane_idx) * V_ROW_BYTES;
    const uint32_t out_mask = is_token_valid ? 0xffffffffu : 0u;
    #pragma unroll 1
    for (int group = group_begin; group < group_end; ++group) {
        const uint32_t exponent = is_token_valid ? row[SCALE_OFFSET + group] : 0u;
        const uint32_t max_exponent = __reduce_max_sync(0xffffffffu, exponent);
        // No valid token: 2^-127, nothing to scale
        const uint32_t tile_exponent = max_exponent > V_HEADROOM ? max_exponent - V_HEADROOM : 0u;
        const uint32_t factor_bits = f16_pow2((int)exponent - (int)tile_exponent) * 0x00010001u;
        const __half2 factor = *reinterpret_cast<const __half2*>(&factor_bits);
        const uint4 data = *reinterpret_cast<const uint4*>(row + 16 * group);
        // Half h of the group: 16 values (8 bytes, the words 2h and 2h+1) become 16 output bytes. Byte pair j of them
        // (values 4j .. 4j+3) becomes the word j of the output
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
                    const __half2 scaled = __hmul2(*reinterpret_cast<const __half2*>(&f16x2), factor);
                    halves[i] = f16x2_to_e4m3x2(*reinterpret_cast<const uint32_t*>(&scaled));
                }
                vals[j] = (halves[0] | (halves[1] << 16)) & out_mask;
            }
            *reinterpret_cast<uint4*>(dst + 32 * group + 16 * h) = uint4{vals[0], vals[1], vals[2], vals[3]};
        }
        if (lane_idx == 0) {
            plan.v_scale[v_idx][group] = tile_exponent;
        }
    }
}

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
            gather_tile(params, plan, index, is_extra_block, raw_idx, tile_idx);
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
            rebase_tile_part(plan, raw_idx, v_idx, tile_idx, group_begin, group_end);
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

// Q of the heads g and g+8 of a head block over 256 dims, as NQ terms of e2m1 with a ue8m0 scale per head and 32 dims: per term the A
// fragments of 4 MMAs (k = 64 dims each). `q_row` points at head g and the first of the 256 dims, `row_step` leads to head g+8.
// Per term, head and group the scale is the smallest power of two 2^e with amax * fp32(1/6) <= 2^e (amax: of what the terms before it
// left of Q; its code is kept in [Q_MIN_CODE, Q_MAX_CODE]), and the codes are e2m1(r * 2^-e). The remainder after a term is
// r - code * 2^e, in fp32.
// `q_scale[term]` receives this thread's scale register of the term: thread t of a quad holds the 4 scales (the groups
// [4 * (t >> 1), +4) of the 8 of the 256 dims, one byte each, low byte first) of the head g + 8 * (t & 1), which are those of the
// MMAs 2 * (t >> 1) and 2 * (t >> 1) + 1: thread-id = k / 2, byte-id = 2 * (k % 2) for MMA k.
template<int NQ>
__device__ __forceinline__ void quantize_q(const bf16 *q_row, int64_t row_step, uint32_t (&q_frag)[NQ][4][4], uint32_t (&q_scale)[NQ], uint4 (*q_smem)[32]) {
    const int t = threadIdx.x % 4;
    // Which scales this thread supplies, as masks
    const uint32_t row_mask[2] = {(uint32_t)(t & 1) - 1u, 0u - (uint32_t)(t & 1)};
    const uint32_t step_mask[2] = {(uint32_t)(t >> 1) - 1u, 0u - (uint32_t)(t >> 1)};
    #pragma unroll
    for (int term = 0; term < NQ; ++term) {
        q_scale[term] = 0;
    }
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        asm volatile("" ::: "memory");     // The loads of the next MMA stay behind this one, or the compiler holds all four in registers
        // x[i]: this thread's elements of register i: head g + 8*(i&1), dims 64k + 32*(i>>1) + 8t + {0..7}, which are
        // a quarter of the group 2k + (i>>1); the other three quarters are held by the threads t^1, t^2, t^3
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
                amax[i] = fmaxf(amax[i], __shfl_xor_sync(0xffffffffu, amax[i], 2));
            }
            const uint32_t codes01 = f32x2_to_ue8m0x2_up(amax[0] * (1.0f / 6.0f), amax[1] * (1.0f / 6.0f));
            const uint32_t codes23 = f32x2_to_ue8m0x2_up(amax[2] * (1.0f / 6.0f), amax[3] * (1.0f / 6.0f));
            uint32_t frags[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t code = min(max(((i < 2 ? codes01 : codes23) >> (8 * (i & 1))) & 0xff, Q_MIN_CODE), Q_MAX_CODE);
                const float inv_scale = __uint_as_float((254 - code) << 23);
                const float scale = __uint_as_float(code << 23);
                uint32_t frag = 0;
                #pragma unroll
                for (int j = 0; j < 8; j += 2) {
                    const uint32_t two = f32x2_to_e2m1x2(x[i][j] * inv_scale, x[i][j+1] * inv_scale);
                    frag |= two << (4 * j);
                    if (term + 1 < NQ) {
                        const uint32_t vals = e2m1x2_to_f16x2(two);
                        x[i][j] = fmaf(-f16_to_f32(vals), scale, x[i][j]);
                        x[i][j+1] = fmaf(-f16_to_f32(vals >> 16), scale, x[i][j+1]);
                    }
                }
                frags[i] = frag;
                // The scale of the group 2k + (i>>1) of head g + 8*(i&1): byte 2*(k%2) + (i>>1) of the register of thread-id k / 2
                q_scale[term] |= (code & row_mask[i & 1] & step_mask[k / 2]) << (8 * (2 * (k % 2) + (i >> 1)));
            }
            if (smem_q_slot(NQ, term, k) < 0) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    q_frag[term][k][i] = frags[i];
                }
            } else {
                q_smem[smem_q_slot(NQ, term, k)][threadIdx.x % 32] = uint4{frags[0], frags[1], frags[2], frags[3]};     // Read back by this thread only
            }
        }
    }
}

// S += Q K for a tile: this warp's half of the dims, 4 MMAs of 64 dims per 8 tokens and Q term. The K fragments of two MMAs come
// from one ldmatrix (8 tokens x 128 dims). The scale register of a token holds the scales of 4 groups (word t & 1 of the 8 scales of
// this half of the dims): thread-id k / 2 of the MMA k selects it, byte-id 2 * (k % 2) the two groups of the MMA.
// The last slots of Q are not in registers: their A fragments and the scales of all terms wait in shared memory.
template<int NQ>
__device__ __forceinline__ void qk_tile(const SharedMemoryPlan &plan, int buf_idx, float (&s_acc)[4][4], const uint32_t (&q_frag)[NQ][4][4]) {
    const int warp_idx = threadIdx.x / 32;
    const int lane_idx = threadIdx.x % 32;
    const int q_half = warp_idx % 2;
    const int g = lane_idx / 4, t = lane_idx % 4;
    const uint8_t *raw = plan.raw[buf_idx];
    // The scales of this thread's token of the first output tile; read again at every use, which is cheaper than four more registers
    const uint32_t *k_scales = reinterpret_cast<const uint32_t*>(raw + g * ROW_BYTES + SCALE_OFFSET + 8 * q_half + 4 * (t & 1));
    const uint32_t base = smem_addr(raw) + (lane_idx % 8) * ROW_BYTES + (lane_idx / 8) * 16 + q_half * (D_QK / 4);
    // The scales of Q wait in shared memory between the tiles: they are registers the softmax and PV have no use for
    uint32_t q_scale[NQ];
    #pragma unroll
    for (int term = 0; term < NQ; ++term) {
        q_scale[term] = plan.q_scale[warp_idx][term][lane_idx];
    }
    #pragma unroll
    for (int half = 0; half < 2; ++half) {
        #pragma unroll
        for (int n = 0; n < 4; ++n) {
            uint32_t k_frag[4];
            ldmatrix_x4(k_frag, base + n * (8 * ROW_BYTES) + half * 64);
            const uint32_t k_scale = k_scales[n * (8 * ROW_BYTES / 4)];
            #pragma unroll
            for (int term = 0; term < NQ; ++term) {
                uint32_t a[4];
                if (smem_q_slot(NQ, term, 2 * half) >= 0) {
                    const uint4 v = load_shared_v4(&plan.q_smem[warp_idx][smem_q_slot(NQ, term, 2 * half)][lane_idx]);
                    a[0] = v.x, a[1] = v.y, a[2] = v.z, a[3] = v.w;
                } else {
                    a[0] = q_frag[term][2 * half][0], a[1] = q_frag[term][2 * half][1], a[2] = q_frag[term][2 * half][2], a[3] = q_frag[term][2 * half][3];
                }
                if (half == 0) {
                    mma_m16n8k64_mxf4<0, 0, 0, 0>(s_acc[n], a, k_frag[0], k_frag[1], q_scale[term], k_scale);
                } else {
                    mma_m16n8k64_mxf4<0, 1, 0, 1>(s_acc[n], a, k_frag[0], k_frag[1], q_scale[term], k_scale);
                }
            }
            #pragma unroll
            for (int term = 0; term < NQ; ++term) {
                uint32_t a[4];
                if (smem_q_slot(NQ, term, 2 * half + 1) >= 0) {
                    const uint4 v = load_shared_v4(&plan.q_smem[warp_idx][smem_q_slot(NQ, term, 2 * half + 1)][lane_idx]);
                    a[0] = v.x, a[1] = v.y, a[2] = v.z, a[3] = v.w;
                } else {
                    a[0] = q_frag[term][2 * half + 1][0], a[1] = q_frag[term][2 * half + 1][1], a[2] = q_frag[term][2 * half + 1][2], a[3] = q_frag[term][2 * half + 1][3];
                }
                if (half == 0) {
                    mma_m16n8k64_mxf4<2, 0, 2, 0>(s_acc[n], a, k_frag[2], k_frag[3], q_scale[term], k_scale);
                } else {
                    mma_m16n8k64_mxf4<2, 1, 2, 1>(s_acc[n], a, k_frag[2], k_frag[3], q_scale[term], k_scale);
                }
            }
        }
    }
}

// O += P V for one 8-column tile and one term of P. Every thread holds the same scales; byte V_SCALE_BYTE of `v_scale`
// belongs to these columns.
template<int V_SCALE_BYTE>
__device__ __forceinline__ void pv_step(float (&o_acc)[4], const uint32_t (&p_frag)[4], uint32_t v0, uint32_t v1, uint32_t p_scale, uint32_t v_scale) {
    mma_m16n8k32_mxf8<0, 0, V_SCALE_BYTE, 0>(o_acc, p_frag, v0, v1, p_scale, v_scale);
}

// Everything of a tile after QK, for a consumer warp: its half of S and that of its partner meet, the softmax updates
// the running max and sum of the pair's 16 heads (both warps of a pair do the same, with the same result), O is rescaled
// where the max moved, and O += P V over this warp's 256 output columns. P never leaves the registers: the S fragments of
// a thread hold the tokens {8n + 2t, +1} (n = 0..3), which are its A fragment of the PV MMA for the order of k given by
// v_row_of_token, in which the rebase stores the tokens.
template<int NUM_P_TERMS>
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
    uint32_t p_frag[NUM_P_TERMS][4];
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
        uint32_t two[NUM_P_TERMS][4];
        #pragma unroll
        for (int n = 0; n < 4; ++n) {
            const float p0 = exp2f(s_acc[n][2*r] - mi[r]), p1 = exp2f(s_acc[n][2*r+1] - mi[r]);
            cur_sum += p0 + p1;
            two[0][n] = f32x2_to_e4m3x2(p0 * P_GAIN, p1 * P_GAIN);
            if constexpr (NUM_P_TERMS == 2) {
                const uint32_t hi_f16x2 = e4m3x2_to_f16x2(two[0][n]);
                two[1][n] = f32x2_to_e4m3x2(
                    (p0 * P_GAIN - f16_to_f32(hi_f16x2)) * P_LO_GAIN,
                    (p1 * P_GAIN - f16_to_f32(hi_f16x2 >> 16)) * P_LO_GAIN
                );
            }
        }
        li[r] = li[r] * scale_for_old[r] + cur_sum;
        #pragma unroll
        for (int term = 0; term < NUM_P_TERMS; ++term) {
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
        const uint32_t v_base = smem_addr(plan.v[v_idx]) + lane_idx * V_ROW_BYTES + (warp_idx % 2) * (D_V / 2);
        #pragma unroll
        for (int c = 0; c < NUM_OUT_TILES / 2; ++c) {
            uint32_t v_frag[4];
            ldmatrix_x2_trans_b8(v_frag, v_base + 16 * c);
            const uint32_t v_scale = c < NUM_OUT_TILES / 4 ? v_scales.x : v_scales.y;
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                #pragma unroll
                for (int term = 0; term < NUM_P_TERMS; ++term) {
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

// One tile, for the consumer warps
template<int NQ, int NUM_P_TERMS, typename QFrag>
__device__ __forceinline__ void consume_tile(
    const SparseAttnDecodeParams &params, SharedMemoryPlan &plan, uint32_t tile_idx, const QFrag &q_frag,
    float (&o_acc)[NUM_OUT_TILES][4], float (&mi)[2], float (&li)[2]
) {
    // Raw buffer and barrier phase of the tile; one multiply instead of a division, as registers are short here
    const uint32_t buf_round = __umulhi(tile_idx, 0xaaaaaaabu) >> 1;
    const uint32_t buf_idx = tile_idx - NUM_RAW_BUFS * buf_round;
    mbarrier_wait(&plan.bar_raw[buf_idx], buf_round & 1);

    // QK: S[head][token] over this warp's half of the dims. The row of an invalid token holds stale bytes; its S is discarded
    // in the softmax.
    float s_acc[4][4];
    #pragma unroll
    for (int n = 0; n < 4; ++n) {
        s_acc[n][0] = s_acc[n][1] = s_acc[n][2] = s_acc[n][3] = 0.0f;
    }
    qk_tile<NQ>(plan, buf_idx, s_acc, q_frag);
    if (threadIdx.x % 32 == 0) {
        mbarrier_arrive(&plan.bar_qk_done[buf_idx]);
    }
    softmax_and_pv<NUM_P_TERMS>(params, plan, tile_idx, s_acc, o_acc, mi, li);
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

template<int NQ, int NUM_P_TERMS>
__device__ __forceinline__ void consumer_main(const SparseAttnDecodeParams &params, SharedMemoryPlan &plan) {
    const int partition_idx = cta_partition_idx();

    int begin_req_idx, end_req_idx;
    if (!get_request_range(params, partition_idx, begin_req_idx, end_req_idx)) {
        return;
    }

    uint32_t tile_idx = 0;  // The same count as in producer_main
    #pragma unroll 1
    for (int batch_idx = begin_req_idx; batch_idx <= end_req_idx; ++batch_idx) {
        uint32_t tile_end;      // The tiles of this request are [tile_idx, tile_end)
        {
            const Request req = get_request(params, partition_idx, batch_idx);
            tile_end = tile_idx + max(req.end_block_idx - req.start_block_idx, 0);
        }
        float o_acc[NUM_OUT_TILES][4];  // The output tiles of this warp's 256 columns
        float mi[2], li[2];             // The softmax state of the heads g and g+8
        // Q as the A fragments of the 4 MMAs (k = 64 dims each) of this warp
        uint32_t q_frag[NQ][4][4], q_scale[NQ];
        quantize_q<NQ>(q_row(params, batch_idx), 8 * (int64_t)params.stride_q_h_q, q_frag, q_scale, plan.q_smem[threadIdx.x / 32]);
        #pragma unroll
        for (int term = 0; term < NQ; ++term) {
            plan.q_scale[threadIdx.x / 32][term][threadIdx.x % 32] = q_scale[term];     // Read back by this thread only
        }
        start_request(o_acc, mi, li);
        #pragma unroll 1
        for (; tile_idx < tile_end; ++tile_idx) {
            consume_tile<NQ, NUM_P_TERMS>(params, plan, tile_idx, q_frag, o_acc, mi, li);
        }

        finish_request(params, partition_idx, batch_idx, mi, li, o_acc);
    }
}

// The name keeps the upstream substring that the benchmark scripts look for
template<int NQ, int NUM_P_TERMS>
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
    // with two P terms need 232.
    if (threadIdx.x < NUM_CONSUMER_THREADS) {
        setmaxnreg_inc<NUM_P_TERMS == 1 ? 224 : 232>();
        consumer_main<NQ, NUM_P_TERMS>(params, plan);
    } else {
        setmaxnreg_dec<NUM_P_TERMS == 1 ? 56 : 40>();
        producer_main(params, plan);
    }
}

template<int NQ, int NUM_P_TERMS>
static void launch(const SparseAttnDecodeParams &params) {
    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    static_assert(smem_size <= 99 * 1024);  // The limit of a block on sm120

    auto kernel = &flash_fwd_splitkv_mla_fp8_sparse_kernel<NQ, NUM_P_TERMS>;
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    kernel<<<decode_grid(params), dim3(NUM_THREADS, 1, 1), smem_size, params.stream>>>(params);
    KU_CHECK_KERNEL_LAUNCH();
}

}   // namespace mxfp4_path

void run_mxfp4(const SparseAttnDecodeParams &params, int num_q_terms, int num_p_terms) {
    KU_ASSERT(params.model_type == ModelType::V41_MXFP4 && (params.extra_topk == 0 || params.extra_model_type == ModelType::V41_MXFP4),
        "the MXFP4 kernel reads an MXFP4 main cache and an MXFP4 extra cache");
    if (num_q_terms == 4 && num_p_terms == 2) {
        mxfp4_path::launch<4, 2>(params);
    } else if (num_q_terms == 2 && num_p_terms == 1) {
        mxfp4_path::launch<2, 1>(params);
    } else {
        KU_ASSERT(false, "the MXFP4 kernel has no instance with %d terms of Q and %d of P", num_q_terms, num_p_terms);
    }
}

}
