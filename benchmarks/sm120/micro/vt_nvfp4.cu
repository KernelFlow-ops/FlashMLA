// What the producer warps of the sm120 sparse decode kernel would pay per 32-token tile to hand PV a V^T in NVFP4 (T-09, F2).
//
// The producers of mxfp8.cu / mxfp4.cu turn every raw tile into a rebased e4m3 tile (rebase_tile_part, rebase_fp4_tile_part:
// about 2.6-3.1k clk per tile with 3 warps, R-109 / R-116). For PV on the FP4 tensor core they would instead write, per tile,
// V^T as e2m1 with a ue4m3 scale per dim and 16 tokens (layout in fp4_common.h) so that mma.m16n8k64 kind::mxf4nvf4 can read it
// with ldmatrix. This file measures that conversion in a harness shaped like the kernel (one 384-thread block per SM, 170
// blocks, the converter warps free-running over pre-filled raw tiles in shared memory), next to a copy of the current rebase,
// and checks every conversion bit for bit against a host reference.
//
// Methods (METH): REBASE = copy of the rebase of the kernel (e4m3 tile, the baseline); LDM = ldmatrix.m16n16.trans.b8 of
// the raw rows (fp8 source), exact fp32 arithmetic, the 'ceil' scale rule of nvfp4_scale_codes.
//
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -o vt_nvfp4 vt_nvfp4.cu
// Run:   ./vt_nvfp4 test            correctness against the host reference (all methods)
//        ./vt_nvfp4 bench [reps]    clk per tile, isolated and under load

#include "fp4_common.h"

using namespace t09;

enum Meth { M_REBASE = 0, M_LDM = 1, M_COUNT };
inline const char *meth_name(int m) { return m == M_REBASE ? "rebase" : "ldm"; }

// ---------------------------------------------------------------------------------------------
// The baseline: the rebase of the kernels, copied (mxfp8.cu rebase_tile_part / rebase_fp4_tile_part with PAIRED_V = false,
// mxfp4.cu rebase_tile_part). The tile state they read is passed as arguments.
// ---------------------------------------------------------------------------------------------
namespace base {

__device__ __forceinline__ int v_row_of_token(int r) { return (r & 16) | ((r & 6) << 1) | ((r & 8) >> 2) | (r & 1); }
__device__ __forceinline__ uint32_t f16_pow2_neg(uint32_t shift) { return shift <= 14 ? (15 - shift) << 10 : (shift <= 24 ? 1u << (24 - shift) : 0u); }
__device__ __forceinline__ uint32_t f16_pow2(int k) { return k >= -14 ? (uint32_t)(15 + k) << 10 : (k >= -24 ? 1u << (k + 24) : 0u); }

struct VStore {
    uint32_t base, bits;
    __device__ __forceinline__ VStore(const uint8_t *buf, int p, int flip) {
        bits = (flip ^ (p % 8)) << 4;
        base = smem_addr(buf) + p * DIMS;
    }
    __device__ __forceinline__ void store(int group, int ii, const uint32_t (&vals)[4]) const {
        const uint32_t addr = (base + ((32 * group) ^ bits)) ^ (16 * ii);
        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n" ::"r"(addr), "r"(vals[0]), "r"(vals[1]), "r"(vals[2]), "r"(vals[3]));
    }
};

__device__ __forceinline__ void rebase_fp8(const uint8_t *raw, uint8_t *dst, uint8_t *vscale, uint32_t valid, int group_begin, int group_end) {
    const int lane_idx = threadIdx.x % 32;
    const bool is_token_valid = (valid >> lane_idx) & 1;
    const uint8_t *row = raw + fp8_off(lane_idx);
    const int v_row = v_row_of_token(lane_idx);
    const uint32_t out_mask = is_token_valid ? 0xffffffffu : 0u;
    const int half_flip = (lane_idx / 4) % 2;
    const VStore v_store(dst, v_row, half_flip);
    const uint8_t *row_half[2] = {row + 16 * half_flip, row + 16 * (half_flip ^ 1)};
#pragma unroll 1
    for (int group = group_begin; group < group_end; ++group) {
        const uint32_t exponent = is_token_valid ? row[512 + group] : 0u;
        const uint32_t max_exponent = __reduce_max_sync(0xffffffffu, exponent);
        const uint32_t factor_bits = f16_pow2_neg(max_exponent - exponent) * 0x00010001u;
        const __half2 factor = *reinterpret_cast<const __half2 *>(&factor_bits);
#pragma unroll
        for (int ii = 0; ii < 2; ++ii) {
            const uint4 rawv = *reinterpret_cast<const uint4 *>(row_half[ii] + 32 * group);
            const uint32_t words[4] = {rawv.x, rawv.y, rawv.z, rawv.w};
            uint32_t vals[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint32_t halves[2];
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const uint32_t f16x2 = e4m3x2_to_f16x2(words[j] >> (16 * h));
                    const __half2 scaled = __hmul2(*reinterpret_cast<const __half2 *>(&f16x2), factor);
                    halves[h] = f16x2_to_e4m3x2(*reinterpret_cast<const uint32_t *>(&scaled));
                }
                vals[j] = (halves[0] | (halves[1] << 16)) & out_mask;
            }
            v_store.store(group, ii, vals);
        }
        if (lane_idx == 0) vscale[group] = max_exponent;
    }
}

__device__ __forceinline__ void rebase_fp4(const uint8_t *raw, uint8_t *dst, uint8_t *vscale, uint32_t valid, int group_begin, int group_end) {
    const int lane_idx = threadIdx.x % 32;
    const bool is_token_valid = (valid >> lane_idx) & 1;
    const uint8_t *row = raw + fp4_off(lane_idx);
    const int v_row = v_row_of_token(lane_idx);
    const uint32_t out_mask = is_token_valid ? 0xffffffffu : 0u;
    const VStore v_store(dst, v_row, 0);
#pragma unroll 1
    for (int group = group_begin; group < group_end; ++group) {
        const uint32_t scale_codes = *reinterpret_cast<const uint16_t *>(row + 256 + 2 * group);
        const uint32_t max_code = __reduce_max_sync(0xffffffffu, is_token_valid ? max(scale_codes & 0xffu, scale_codes >> 8) : 0u);
        const float amax = 6.0f * f16lo_to_f32(e4m3x2_to_f16x2(max_code));
        const uint32_t amax_bits = __float_as_uint(amax);
        const uint32_t max_exponent = amax == 0.0f ? 127u : (amax_bits >> 23) - 8 + ((amax_bits & 0x7fffffu) > 0x600000u ? 1 : 0);
        const uint4 data = *reinterpret_cast<const uint4 *>(row + 16 * group);
        const uint32_t scales = e4m3x2_to_f16x2(scale_codes);
        const float scale[2] = {f16lo_to_f32(scales), f16hi_to_f32(scales)};
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
                    const __half2 scaled = __hmul2(*reinterpret_cast<const __half2 *>(&f16x2), factor[h]);
                    halves[i] = f16x2_to_e4m3x2(*reinterpret_cast<const uint32_t *>(&scaled));
                }
                vals[j] = (halves[0] | (halves[1] << 16)) & out_mask;
            }
            v_store.store(group, h, vals);
        }
        if (lane_idx == 0) vscale[group] = max_exponent;
    }
}

// mxfp4.cu: raw rows one after another, V buffer rows of 512 + 16 B, V_HEADROOM = 6
constexpr int MX_V_ROW = DIMS + 16;
__device__ __forceinline__ void rebase_mx(const uint8_t *raw, uint8_t *dst_base, uint8_t *vscale, uint32_t valid, int group_begin, int group_end) {
    const int lane_idx = threadIdx.x % 32;
    const bool is_token_valid = (valid >> lane_idx) & 1;
    const uint8_t *row = raw + mx_off(lane_idx);
    uint8_t *dst = dst_base + v_row_of_token(lane_idx) * MX_V_ROW;
    const uint32_t out_mask = is_token_valid ? 0xffffffffu : 0u;
#pragma unroll 1
    for (int group = group_begin; group < group_end; ++group) {
        const uint32_t exponent = is_token_valid ? row[256 + group] : 0u;
        const uint32_t max_exponent = __reduce_max_sync(0xffffffffu, exponent);
        const uint32_t tile_exponent = max_exponent > 6 ? max_exponent - 6 : 0u;
        const uint32_t factor_bits = f16_pow2((int)exponent - (int)tile_exponent) * 0x00010001u;
        const __half2 factor = *reinterpret_cast<const __half2 *>(&factor_bits);
        const uint4 data = *reinterpret_cast<const uint4 *>(row + 16 * group);
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
                    const __half2 scaled = __hmul2(*reinterpret_cast<const __half2 *>(&f16x2), factor);
                    halves[i] = f16x2_to_e4m3x2(*reinterpret_cast<const uint32_t *>(&scaled));
                }
                vals[j] = (halves[0] | (halves[1] << 16)) & out_mask;
            }
            *reinterpret_cast<uint4 *>(dst + 32 * group + 16 * h) = uint4{vals[0], vals[1], vals[2], vals[3]};
        }
        if (lane_idx == 0) vscale[group] = tile_exponent;
    }
}

}  // namespace base

// ---------------------------------------------------------------------------------------------
// V^T NVFP4 from the raw rows with ldmatrix.m16n16.trans.b8 (fp8 source)
//
// One ldmatrix.x2.trans loads 16 dims of 2 x 16 tokens: thread (g, t) gets, for dim 8 h + g of the chunk, the bytes of the tokens
// tok_of(i, t, j) (i = matrix, j = 0..3) in register 2 i + h. These 8 tokens are exactly word t of the V^T chunk of that dim, so
// the thread quantizes them to 8 nibbles and writes the word; the block of 16 tokens that shares a scale is the 8 tokens of
// threads 2q and 2q+1, whose amax meets in one shuffle.
// ---------------------------------------------------------------------------------------------
template <int L>
__device__ __forceinline__ void convert_fp8_ldm(const uint8_t *raw, uint8_t *vt, int hh, uint32_t valid, const int *e1, int gb, int ge) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const uint32_t raw_s = smem_addr(raw), vt_s = smem_addr(vt);
    // ldmatrix row of this lane: row rho of matrix mi is the token tok_of(mi, rho >> 2, rho & 3)
    const uint32_t ldm_addr = raw_s + fp8_off(tok_of(lane >> 4, (lane & 15) >> 2, lane & 3));
    // this thread's tokens (i, j) and their masks
    uint32_t tk_off[2][4], vm[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        vm[i] = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int tok = tok_of(i, t, j);
            tk_off[i][j] = raw_s + fp8_off(tok) + 512;
            vm[i] |= ((valid >> tok) & 1u ? 0xffu : 0u) << (8 * j);
        }
    }
#pragma unroll 1
    for (int G = gb; G < ge; ++G) {
        // value = e4m3 * 2^(e - 127 - E1): the float with the biased exponent e - E1
        float f[2][4];
        const int shift = e1[G];
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t e = lds8(tk_off[i][j] + G);
                f[i][j] = (vm[i] >> (8 * j)) & 1u ? __uint_as_float((e - shift) << 23) : 0.f;
            }
#pragma unroll
        for (int ch = 0; ch < 2; ++ch) {
            uint32_t R[4];
            ldmatrix_x2_trans_b8(R, ldm_addr + 16 * (2 * G + ch));
            float x[2][2][4];  // [h][i][j]
            float amax[2];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                amax[h] = 0.f;
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const uint32_t b = R[2 * i + h] & vm[i];
                    const uint32_t lo = e4m3x2_to_f16x2(b & 0xffffu), hi = e4m3x2_to_f16x2(b >> 16);
                    x[h][i][0] = f16lo_to_f32(lo) * f[i][0];
                    x[h][i][1] = f16hi_to_f32(lo) * f[i][1];
                    x[h][i][2] = f16lo_to_f32(hi) * f[i][2];
                    x[h][i][3] = f16hi_to_f32(hi) * f[i][3];
#pragma unroll
                    for (int j = 0; j < 4; ++j) amax[h] = fmaxf(amax[h], fabsf(x[h][i][j]));
                }
                amax[h] = fmaxf(amax[h], __shfl_xor_sync(0xffffffffu, amax[h], 1));
            }
            const uint32_t codes = nvfp4_scale_codes(amax[0], amax[1]);
            const uint32_t sc = e4m3x2_to_f16x2(codes);
            const float inv[2] = {__frcp_rn(f16lo_to_f32(sc)), __frcp_rn(f16hi_to_f32(sc))};
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                uint32_t word = 0;
#pragma unroll
                for (int i = 0; i < 2; ++i)
#pragma unroll
                    for (int jp = 0; jp < 2; ++jp)
                        word |= f32x2_to_e2m1x2(x[h][i][2 * jp] * inv[h], x[h][i][2 * jp + 1] * inv[h]) << (8 * (2 * i + jp));
                const int d = 16 * (2 * G + ch) + 8 * h + g;
                sts32(vt_s + vt_data_off<L>(d, hh) + 4 * t, word);
                if (!(t & 1)) sts8(vt_s + vt_scale_off<L>(d) + 2 * hh + (t >> 1), (codes >> (8 * h)) & 0xffu);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------------------------
constexpr int VT_L = 0;                                  // the V^T layout of the conversion benchmarks
constexpr int VT_BYTES = vt_buf_bytes<VT_L>();           // 18432 per pair of tiles
constexpr int V_E4M3_BYTES = TILE * DIMS;                // 16384 per rebased tile

// 32-dim groups [gb, ge) of converter warp cw of W
template <int W> __device__ __forceinline__ void group_range(int cw, int &gb, int &ge) {
    if (W == 3) { gb = cw == 0 ? 0 : (cw == 1 ? 6 : 11); ge = cw == 0 ? 6 : (cw == 1 ? 11 : 16); }
    else { gb = cw * (NGROUPS / W); ge = gb + NGROUPS / W; }
}

// Convert tile `raw` (valid mask `valid`) into half `hh` of the pair buffer `vt` (or, for REBASE, into the e4m3 buffer
// vt + slot * (V_E4M3_BYTES + 64) with its 16 group exponents behind it)
template <int SRC, int METH, int W>
__device__ __forceinline__ void convert_tile(const uint8_t *raw, uint8_t *vt, int hh, uint32_t valid, const int *e1, int cw) {
    int gb, ge;
    group_range<W>(cw, gb, ge);
    if constexpr (METH == M_REBASE) {
        uint8_t *dst = vt + hh * (V_E4M3_BYTES + 64);
        uint8_t *vscale = dst + V_E4M3_BYTES;
        if constexpr (SRC == SRC_FP8) base::rebase_fp8(raw, dst, vscale, valid, gb, ge);
        else if constexpr (SRC == SRC_FP4) base::rebase_fp4(raw, dst, vscale, valid, gb, ge);
        else base::rebase_mx(raw, dst, vscale, valid, gb, ge);
    } else if constexpr (METH == M_LDM && SRC == SRC_FP8) {
        convert_fp8_ldm<VT_L>(raw, vt, hh, valid, e1, gb, ge);
    }
}

constexpr bool supported(int src, int meth) {
    return meth == M_REBASE || (meth == M_LDM && src == SRC_FP8);
}

// ---------------------------------------------------------------------------------------------
// Test kernel: one block per pair of tiles, converts with W converter warps (warps 12 - W .. 11), dumps the pair buffer
// ---------------------------------------------------------------------------------------------
constexpr int NTHREADS = 384;
constexpr int OFF_RAW = 0;                               // 2 raw tiles
constexpr int OFF_VT = 2 * RAW_BYTES;                    // 2 pair buffers (REBASE: 2 e4m3 buffers)
constexpr int OFF_GATHER = OFF_VT + 2 * VT_BYTES;        // a scratch tile for the gather warp
constexpr int OFF_MISC = OFF_GATHER + RAW_BYTES;
constexpr int SMEM_BYTES = OFF_MISC + 512;

struct Misc {
    uint32_t valid[2];
    int e1[NGROUPS];
    uint32_t stop;
    uint32_t pad;
    uint64_t gather_bar;
};

template <int SRC, int METH, int W>
__global__ void __launch_bounds__(NTHREADS, 1) vt_test(const uint8_t *images, const uint32_t *valids, const int *e1s, uint8_t *out) {
    extern __shared__ __align__(128) uint8_t smem[];
    Misc &misc = *reinterpret_cast<Misc *>(smem + OFF_MISC);
    const int pair = blockIdx.x;
    for (int i = threadIdx.x; i < 2 * RAW_BYTES / 16; i += NTHREADS)
        reinterpret_cast<uint4 *>(smem + OFF_RAW)[i] = reinterpret_cast<const uint4 *>(images + size_t(pair) * 2 * RAW_BYTES)[i];
    for (int i = threadIdx.x; i < 2 * VT_BYTES / 4; i += NTHREADS) reinterpret_cast<uint32_t *>(smem + OFF_VT)[i] = 0xa5a5a5a5u;
    if (threadIdx.x < 2) misc.valid[threadIdx.x] = valids[pair * 2 + threadIdx.x];
    if (threadIdx.x < NGROUPS) misc.e1[threadIdx.x] = e1s[pair * NGROUPS + threadIdx.x];
    __syncthreads();
    const int warp = threadIdx.x / 32;
    if (warp >= 12 - W) {
#pragma unroll 1
        for (int hh = 0; hh < 2; ++hh)
            convert_tile<SRC, METH, W>(smem + OFF_RAW + hh * RAW_BYTES, smem + OFF_VT, hh, misc.valid[hh], misc.e1, warp - (12 - W));
    }
    __syncthreads();
    for (int i = threadIdx.x; i < VT_BYTES / 16; i += NTHREADS)
        reinterpret_cast<uint4 *>(out + size_t(pair) * VT_BYTES)[i] = reinterpret_cast<uint4 *>(smem + OFF_VT)[i];
}

// ---------------------------------------------------------------------------------------------
// Benchmark kernel: the converter warps free-run over the two raw tiles for `ntiles` tiles and record clk / tile. Under load the
// 8 consumer warps issue the MMAs of the kernel (per pair of tiles and warp: QK 64 mxf8 MMAs on K read from the raw tiles, PV 32
// mxf4nvf4 MMAs on the V^T pair buffer with their ldmatrix) until the converters are done; with the gather, warp 8 (W = 3) copies
// random 528 B rows from global memory with cp.async.bulk into a scratch tile all the time.
// ---------------------------------------------------------------------------------------------
struct BenchParams {
    const uint8_t *images;   // 2 raw tiles
    const uint32_t *valids;  // 2
    const int *e1;           // NGROUPS
    int ntiles;
    uint32_t *dt;            // [gridDim.x][4][ntiles]
    uint32_t afrag[4];
    const uint8_t *gsrc;     // random rows of 528 B
    const int *grows;        // [gridDim.x][4096]
    float *sink;
};

enum Load { LOAD_NONE = 0, LOAD_MMA = 1, LOAD_MMA_GATHER = 2 };

__device__ __forceinline__ uint32_t ld_volatile(const uint32_t *p) {
    uint32_t v;
    asm volatile("ld.volatile.shared.b32 %0, [%1];\n" : "=r"(v) : "r"(smem_addr(p)));
    return v;
}

__device__ __forceinline__ void consumer_load(uint8_t *smem, const BenchParams &p, const uint32_t *stop) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    const uint32_t smem_s = smem_addr(smem);
    AccF32 acc[32];
    AccF32 qk[4];
#pragma unroll
    for (int i = 0; i < 32; ++i) acc_zero(acc[i]);
#pragma unroll
    for (int i = 0; i < 4; ++i) acc_zero(qk[i]);
    const uint32_t a[4] = {p.afrag[0], p.afrag[1], p.afrag[2], p.afrag[3]};
    const uint32_t sfa = 0x38383838u;
    // K: row rho = lane & 7 of matrix mi of the x4: n-tile a0 + (mi >> 1) of the tile, chunk 2 kk + (mi & 1) of the warp's half of the dims
    const uint32_t k_base = smem_s + OFF_RAW + 256 * (warp & 1) + fp8_off(8 * ((lane >> 4) & 1) + (lane & 7)) + 16 * ((lane >> 3) & 1);
    const int vt_dim0 = 256 * (warp & 1);
    uint32_t it = 0;
    while (!ld_volatile(stop)) {
#pragma unroll
        for (int tt = 0; tt < 2; ++tt) {
#pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    uint32_t kb[4];
                    ldmatrix_x4(kb, k_base + tt * RAW_BYTES + 32 * kk + half * 2 * (2 * GROUP_STRIDE));
                    const uint32_t b0[2] = {kb[0], kb[1]}, b1[2] = {kb[2], kb[3]};
                    MXF8_E4M3::run<0, 0, 0, 0>(qk[2 * half], a, b0, 0x7f7f7f7fu, 0x7f7f7f7fu);
                    MXF8_E4M3::run<0, 0, 0, 0>(qk[2 * half + 1], a, b1, 0x7f7f7f7fu, 0x7f7f7f7fu);
                }
            }
        }
        const uint32_t vt_s = smem_s + OFF_VT + (it & 1) * VT_BYTES;
        uint32_t sfb[8];
#pragma unroll
        for (int x = 0; x < 2; ++x) {
            uint32_t f[4];
            ldmatrix_x4(f, vt_s + vt_data_bytes<VT_L>() + (8 * (warp & 1) + 4 * x + (lane >> 3)) * 128 + (lane & 7) * 16);
#pragma unroll
            for (int m = 0; m < 4; ++m) sfb[4 * x + m] = f[m];
        }
#pragma unroll
        for (int np = 0; np < 16; ++np) {
            uint32_t b[4];
            const int d = vt_dim0 + 8 * (2 * np + ((lane >> 3) >> 1)) + (lane & 7);
            ldmatrix_x4(b, vt_s + vt_data_off<VT_L>(d, (lane >> 3) & 1));
            const uint32_t b0[2] = {b[0], b[1]}, b1[2] = {b[2], b[3]};
            // n-tiles 2 np and 2 np + 1: 32-dim group np / 2 of the warp, tid_b = n-tile % 4
            if ((np & 1) == 0) {
                NVF4_4X::run<0, 0, 0, 0>(acc[2 * np], a, b0, sfa, sfb[np / 2]);
                NVF4_4X::run<0, 0, 0, 1>(acc[2 * np + 1], a, b1, sfa, sfb[np / 2]);
            } else {
                NVF4_4X::run<0, 0, 0, 2>(acc[2 * np], a, b0, sfa, sfb[np / 2]);
                NVF4_4X::run<0, 0, 0, 3>(acc[2 * np + 1], a, b1, sfa, sfb[np / 2]);
            }
        }
        ++it;
    }
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 32; ++i) s += acc_sum(acc[i]);
#pragma unroll
    for (int i = 0; i < 4; ++i) s += acc_sum(qk[i]);
    p.sink[blockIdx.x * NTHREADS + threadIdx.x] = s;
}

template <int SRC, int METH, int W, int LOAD>
__global__ void __launch_bounds__(NTHREADS, 1) vt_bench(const __grid_constant__ BenchParams p) {
    extern __shared__ __align__(128) uint8_t smem[];
    Misc &misc = *reinterpret_cast<Misc *>(smem + OFF_MISC);
    for (int i = threadIdx.x; i < 2 * RAW_BYTES / 16; i += NTHREADS)
        reinterpret_cast<uint4 *>(smem + OFF_RAW)[i] = reinterpret_cast<const uint4 *>(p.images)[i];
    for (int i = threadIdx.x; i < 2 * VT_BYTES / 16; i += NTHREADS) reinterpret_cast<uint4 *>(smem + OFF_VT)[i] = uint4{0x38383838u, 0x38383838u, 0x38383838u, 0x38383838u};
    if (threadIdx.x < 2) misc.valid[threadIdx.x] = p.valids[threadIdx.x];
    if (threadIdx.x < NGROUPS) misc.e1[threadIdx.x] = p.e1[threadIdx.x];
    if (threadIdx.x == 0) {
        misc.stop = 0;
        mbarrier_init(&misc.gather_bar, 1);
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();
    const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
    if (warp < 8) {
        setmaxnreg_inc<224>();
        if constexpr (LOAD != LOAD_NONE) consumer_load(smem, p, &misc.stop);
    } else {
        setmaxnreg_dec<56>();
        const int pw = warp - 8;
        if (W == 3 && pw == 0) {
            if constexpr (LOAD == LOAD_MMA_GATHER) {
                // lane l copies row l of the scratch tile from a random source row; 32 rows of 528 B per round
                uint32_t phase = 0, it = 0;
                const int *rows = p.grows + size_t(blockIdx.x) * 4096;
                while (!ld_volatile(&misc.stop)) {
                    if (lane == 0) mbarrier_arrive_expect_tx(&misc.gather_bar, TILE * FP8_ROW);
                    __syncwarp();
                    bulk_copy(smem + OFF_GATHER + lane * FP8_ROW, p.gsrc + size_t(rows[(it * 32 + lane) & 4095]) * FP8_ROW, FP8_ROW, &misc.gather_bar);
                    mbarrier_wait(&misc.gather_bar, phase);
                    phase ^= 1;
                    ++it;
                }
            }
        } else {
            const int cw = W == 3 ? pw - 1 : pw;
            uint32_t *dt = p.dt + (size_t(blockIdx.x) * 4 + cw) * p.ntiles;
            long long prev = clock64_();
#pragma unroll 1
            for (int tile = 0; tile < p.ntiles; ++tile) {
                const int slot = tile & 1;
                convert_tile<SRC, METH, W>(smem + OFF_RAW + slot * RAW_BYTES, smem + OFF_VT, slot, misc.valid[slot], misc.e1, cw);
                const long long now = clock64_();
                if (lane == 0) dt[tile] = uint32_t(now - prev);
                prev = now;
            }
            named_barrier_sync(1, 32 * W);
            if (cw == 0 && lane == 0) asm volatile("st.volatile.shared.b32 [%0], %1;\n" ::"r"(smem_addr(&misc.stop)), "r"(1u) : "memory");
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Host: correctness
// ---------------------------------------------------------------------------------------------

static int g_failures = 0;

template <int SRC, int METH, int W> bool run_test(int pairs_per_scenario) {
    if (!supported(SRC, METH) || METH == M_REBASE) return true;
    const int npairs = pairs_per_scenario * SC_COUNT;
    std::vector<uint8_t> images(size_t(npairs) * 2 * RAW_BYTES);
    std::vector<uint32_t> valids(npairs * 2);
    std::vector<int> e1s(npairs * NGROUPS);
    std::vector<VtRef> refs(npairs * 2);
    for (int p = 0; p < npairs; ++p) {
        const int sc = p % SC_COUNT;
        int e1[NGROUPS];
        make_e1(sc, 7777ull * p + 3, e1);
        for (int hh = 0; hh < 2; ++hh) {
            RawTile tile;
            make_tile(tile, SRC, sc, 1000003ull * p + 17 * hh + 5);
            std::memcpy(&images[(size_t(p) * 2 + hh) * RAW_BYTES], tile.raw.data(), RAW_BYTES);
            valids[p * 2 + hh] = tile.valid;
            std::vector<float> x;
            shifted_values(tile, e1, x);
            vt_reference(x, tile.valid, refs[p * 2 + hh]);
        }
        std::memcpy(&e1s[p * NGROUPS], e1, sizeof(e1));
    }
    uint8_t *d_img, *d_out;
    uint32_t *d_valid;
    int *d_e1;
    CK(cudaMalloc(&d_img, images.size()));
    CK(cudaMalloc(&d_out, size_t(npairs) * VT_BYTES));
    CK(cudaMalloc(&d_valid, valids.size() * 4));
    CK(cudaMalloc(&d_e1, e1s.size() * 4));
    CK(cudaMemcpy(d_img, images.data(), images.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_valid, valids.data(), valids.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_e1, e1s.data(), e1s.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaFuncSetAttribute(vt_test<SRC, METH, W>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    vt_test<SRC, METH, W><<<npairs, NTHREADS, SMEM_BYTES>>>(d_img, d_valid, d_e1, d_out);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<uint8_t> out(size_t(npairs) * VT_BYTES);
    CK(cudaMemcpy(out.data(), d_out, out.size(), cudaMemcpyDeviceToHost));
    long long bad_data = 0, bad_scale = 0, total = 0;
    long long bad_by_sc[SC_COUNT] = {};
    for (int p = 0; p < npairs; ++p) {
        const uint8_t *o = out.data() + size_t(p) * VT_BYTES;
        for (int hh = 0; hh < 2; ++hh) {
            const VtRef &r = refs[p * 2 + hh];
            for (int d = 0; d < DIMS; ++d) {
                for (int t = 0; t < 4; ++t) {
                    uint32_t got;
                    std::memcpy(&got, o + vt_data_off<VT_L>(d, hh) + 4 * t, 4);
                    ++total;
                    if (got != r.data[d * 4 + t]) { ++bad_data; ++bad_by_sc[p % SC_COUNT]; }
                }
                for (int q = 0; q < 2; ++q) {
                    ++total;
                    if (o[vt_scale_off<VT_L>(d) + 2 * hh + q] != r.scale[d * 2 + q]) { ++bad_scale; ++bad_by_sc[p % SC_COUNT]; }
                }
            }
        }
    }
    printf("test %-6s %-7s W=%d: %lld words+scales, data mismatches %lld, scale mismatches %lld %s\n", src_name(SRC), meth_name(METH), W, total,
           bad_data, bad_scale, bad_data + bad_scale == 0 ? "OK" : "FAIL");
    if (bad_data + bad_scale) {
        for (int s = 0; s < SC_COUNT; ++s)
            if (bad_by_sc[s]) printf("    scenario %-14s: %lld mismatches\n", scenario_name(s), bad_by_sc[s]);
        ++g_failures;
    }
    CK(cudaFree(d_img));
    CK(cudaFree(d_out));
    CK(cudaFree(d_valid));
    CK(cudaFree(d_e1));
    return bad_data + bad_scale == 0;
}

// Unit tests of the scalar rules on the device against the host: scale codes, reciprocals, e2m1 rounding
__global__ void rules_kernel(const float *amax, uint32_t *codes, int n, uint32_t *rcp_bits, float *xin, uint32_t *e2m1, int nx) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n / 2; i += gridDim.x * blockDim.x)
        reinterpret_cast<uint16_t *>(codes)[i] = uint16_t(nvfp4_scale_codes(amax[2 * i], amax[2 * i + 1]));
    if (blockIdx.x == 0 && threadIdx.x < 126) rcp_bits[threadIdx.x] = __float_as_uint(__frcp_rn(f16lo_to_f32(e4m3x2_to_f16x2(threadIdx.x + 1))));
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nx / 2; i += gridDim.x * blockDim.x) e2m1[i] = f32x2_to_e2m1x2(xin[2 * i], xin[2 * i + 1]);
}

static void test_rules() {
    Rng r(77);
    std::vector<float> amax;
    // random magnitudes over the whole range, the 6 s grid and its neighbours
    for (int i = 0; i < 400000; ++i) amax.push_back(std::ldexp(1.0f + float(r.u32() & 0xffffff) / 16777216.f, int(r.below(40)) - 25));
    for (uint32_t c = 1; c <= 0x7e; ++c) {
        const float v = 6.0f * float(h_e4m3(c));
        amax.push_back(v);
        amax.push_back(std::nextafter(v, 0.f));
        amax.push_back(std::nextafter(v, 1e30f));
    }
    amax.push_back(0.f);
    amax.push_back(3000.f);
    amax.push_back(1e30f);
    while (amax.size() % 2) amax.push_back(1.f);
    std::vector<float> xin;
    for (int i = 0; i < 400000; ++i) xin.push_back((r.chance(0.5) ? -1.f : 1.f) * std::ldexp(float(r.u32() & 0xffffff) / 16777216.f, int(r.below(6)) - 3));
    for (float m : {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, 0.f, 7.f, 100.f})
        for (float s : {-1.f, 1.f}) {
            xin.push_back(s * m);
            xin.push_back(s * std::nextafter(m, 0.f));
            xin.push_back(s * std::nextafter(m, 1e30f));
        }
    xin.push_back(-0.f);
    while (xin.size() % 2) xin.push_back(1.f);
    float *d_amax, *d_x;
    uint32_t *d_codes, *d_rcp, *d_e2m1;
    CK(cudaMalloc(&d_amax, amax.size() * 4));
    CK(cudaMalloc(&d_x, xin.size() * 4));
    CK(cudaMalloc(&d_codes, amax.size() * 2 + 4));
    CK(cudaMalloc(&d_rcp, 128 * 4));
    CK(cudaMalloc(&d_e2m1, xin.size() * 2 + 4));
    CK(cudaMemcpy(d_amax, amax.data(), amax.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_x, xin.data(), xin.size() * 4, cudaMemcpyHostToDevice));
    rules_kernel<<<64, 256>>>(d_amax, d_codes, int(amax.size()), d_rcp, d_x, d_e2m1, int(xin.size()));
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<uint16_t> codes(amax.size() / 2);
    std::vector<uint32_t> rcp(126);
    std::vector<uint32_t> e2m1(xin.size() / 2);
    CK(cudaMemcpy(codes.data(), d_codes, codes.size() * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(rcp.data(), d_rcp, 126 * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(e2m1.data(), d_e2m1, e2m1.size() * 4, cudaMemcpyDeviceToHost));
    long long bad_code = 0, bad_rcp = 0, bad_e2m1 = 0;
    for (size_t i = 0; i < amax.size(); ++i) {
        const uint32_t got = (codes[i / 2] >> (8 * (i & 1))) & 0xff;
        if (got != h_scale_code(amax[i])) {
            if (bad_code++ < 5) printf("  scale rule: amax %.9g device %02x host %02x\n", amax[i], got, h_scale_code(amax[i]));
        }
    }
    for (uint32_t c = 1; c <= 126; ++c) {
        float want = h_rcp(c);
        uint32_t wb;
        std::memcpy(&wb, &want, 4);
        if (rcp[c - 1] != wb) ++bad_rcp;
    }
    // the kernel stored one byte (zero extended) per pair
    for (size_t i = 0; i < xin.size(); i += 2) {
        const uint32_t byte = e2m1[i / 2] & 0xff;
        for (int k = 0; k < 2; ++k) {
            const uint32_t got = (byte >> (4 * k)) & 15;
            const uint32_t want = h_e2m1_rne(xin[i + k]);
            if (got != want) {
                if (bad_e2m1++ < 5) printf("  e2m1 rule: x %.9g device %x host %x\n", xin[i + k], got, want);
            }
        }
    }
    printf("test rules: scale codes %zu (%lld bad), reciprocals 126 (%lld bad), e2m1 %zu (%lld bad) %s\n", amax.size(), bad_code, bad_rcp, xin.size(), bad_e2m1,
           bad_code + bad_rcp + bad_e2m1 == 0 ? "OK" : "FAIL");
    if (bad_code + bad_rcp + bad_e2m1) ++g_failures;
    CK(cudaFree(d_amax));
    CK(cudaFree(d_x));
    CK(cudaFree(d_codes));
    CK(cudaFree(d_rcp));
    CK(cudaFree(d_e2m1));
}

// ---------------------------------------------------------------------------------------------
// Host: benchmark
// ---------------------------------------------------------------------------------------------
struct BenchCtx {
    uint8_t *images[4];      // per source
    uint32_t *valids[4];
    int *e1;
    uint32_t *dt;
    uint8_t *gsrc;
    int *grows;
    float *sink;
    int ntiles = 72, warmup = 8, reps = 3;
};
constexpr int NBLK = 170;

static BenchCtx g_ctx;

static void bench_setup() {
    BenchCtx &c = g_ctx;
    int e1[NGROUPS];
    make_e1(SC_RANDOM, 4242, e1);
    CK(cudaMalloc(&c.e1, sizeof(e1)));
    CK(cudaMemcpy(c.e1, e1, sizeof(e1), cudaMemcpyHostToDevice));
    for (int src = 0; src < 3; ++src) {
        std::vector<uint8_t> img(2 * RAW_BYTES);
        uint32_t v[2];
        for (int hh = 0; hh < 2; ++hh) {
            RawTile t;
            make_tile(t, src, SC_RANDOM, 99 + 31 * src + hh);
            std::memcpy(&img[hh * RAW_BYTES], t.raw.data(), RAW_BYTES);
            v[hh] = t.valid;
        }
        CK(cudaMalloc(&c.images[src], img.size()));
        CK(cudaMalloc(&c.valids[src], 8));
        CK(cudaMemcpy(c.images[src], img.data(), img.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(c.valids[src], v, 8, cudaMemcpyHostToDevice));
    }
    CK(cudaMalloc(&c.dt, size_t(NBLK) * 4 * c.ntiles * 4));
    const size_t rows = size_t(1) << 21;
    CK(cudaMalloc(&c.gsrc, rows * FP8_ROW));
    CK(cudaMemset(c.gsrc, 0x3c, rows * FP8_ROW));
    std::vector<int> gr(size_t(NBLK) * 4096);
    Rng r(5);
    for (auto &x : gr) x = int(r.below(uint32_t(rows)));
    CK(cudaMalloc(&c.grows, gr.size() * 4));
    CK(cudaMemcpy(c.grows, gr.data(), gr.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&c.sink, size_t(NBLK) * NTHREADS * 4));
}

struct Stat {
    double med = 0, p90 = 0, mean = 0;
};
static Stat stat_of(std::vector<uint32_t> v) {
    Stat s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    s.med = v[v.size() / 2];
    s.p90 = v[size_t(v.size() * 0.9)];
    double sum = 0;
    for (uint32_t x : v) sum += x;
    s.mean = sum / v.size();
    return s;
}

template <int SRC, int METH, int W, int LOAD> static void run_bench() {
    if (!supported(SRC, METH)) return;
    if (LOAD == LOAD_MMA_GATHER && W != 3) return;
    BenchCtx &c = g_ctx;
    BenchParams p{};
    p.images = c.images[SRC];
    p.valids = c.valids[SRC];
    p.e1 = c.e1;
    p.ntiles = c.ntiles;
    p.dt = c.dt;
    for (int i = 0; i < 4; ++i) p.afrag[i] = 0x12345678u * (i + 1);
    p.gsrc = c.gsrc;
    p.grows = c.grows;
    p.sink = c.sink;
    CK(cudaFuncSetAttribute(vt_bench<SRC, METH, W, LOAD>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
    std::vector<uint32_t> samples[4];
    std::vector<double> rep_tile;
    std::vector<uint32_t> host(size_t(NBLK) * 4 * c.ntiles);
    for (int rep = 0; rep < c.reps; ++rep) {
        CK(cudaMemset(c.dt, 0, host.size() * 4));
        vt_bench<SRC, METH, W, LOAD><<<NBLK, NTHREADS, SMEM_BYTES>>>(p);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(host.data(), c.dt, host.size() * 4, cudaMemcpyDeviceToHost));
        double worst = 0;
        for (int cw = 0; cw < W; ++cw) {
            std::vector<uint32_t> v;
            for (int b = 0; b < NBLK; ++b)
                for (int t = c.warmup; t < c.ntiles; ++t) v.push_back(host[(size_t(b) * 4 + cw) * c.ntiles + t]);
            samples[cw].insert(samples[cw].end(), v.begin(), v.end());
            worst = std::max(worst, stat_of(v).med);
        }
        rep_tile.push_back(worst);
    }
    const char *load_name[] = {"isolated", "mma", "mma+gather"};
    printf("%-6s %-7s W=%d %-10s |", src_name(SRC), meth_name(METH), W, load_name[LOAD]);
    double tile = 0;
    for (int cw = 0; cw < W; ++cw) {
        const Stat s = stat_of(samples[cw]);
        printf(" w%d %5.0f/%5.0f", cw, s.med, s.p90);
        tile = std::max(tile, s.med);
    }
    printf(" | tile %5.0f  per-rep:", tile);
    for (double m : rep_tile) printf(" %.0f", m);
    printf("\n");
    fflush(stdout);
}

template <int SRC, int METH> static void run_bench_all() {
    run_bench<SRC, METH, 3, LOAD_NONE>();
    run_bench<SRC, METH, 3, LOAD_MMA>();
    run_bench<SRC, METH, 3, LOAD_MMA_GATHER>();
    run_bench<SRC, METH, 4, LOAD_NONE>();
    run_bench<SRC, METH, 4, LOAD_MMA>();
}

int main(int argc, char **argv) {
    const char *cmd = argc > 1 ? argv[1] : "test";
    if (std::strcmp(cmd, "test") == 0) {
        test_rules();
        run_test<SRC_FP8, M_LDM, 3>(60);
        run_test<SRC_FP8, M_LDM, 4>(30);
        printf(g_failures ? "FAILED\n" : "ALL OK\n");
        return g_failures != 0;
    }
    if (std::strcmp(cmd, "bench") == 0) {
        if (argc > 2) g_ctx.reps = atoi(argv[2]);
        bench_setup();
        printf("clk per 32-token tile and converter warp: median / p90 over %d CTAs x %d tiles x %d reps (170 CTAs, 384 threads)\n", NBLK, g_ctx.ntiles - g_ctx.warmup, g_ctx.reps);
        run_bench_all<SRC_FP8, M_REBASE>();
        run_bench_all<SRC_FP4, M_REBASE>();
        run_bench_all<SRC_MX, M_REBASE>();
        run_bench_all<SRC_FP8, M_LDM>();
        return 0;
    }
    return 0;
}
