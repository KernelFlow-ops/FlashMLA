// Shared pieces of the T-09 micro benchmarks (vt_nvfp4.cu, p_nvfp4.cu, pv_fp4.cu): what PV on the FP4 tensor core needs
// next to the sm120 sparse decode kernel (csrc/cuda_kernels/sm120/decode/sparse/mxfp8.cu).
//
//   * PV runs on mma.m16n8k64 kind::mxf4nvf4 (e2m1 x e2m1, one ue4m3 scale per 16 k of every row of A and every column of B).
//     A = P (16 heads x 64 tokens), B = V^T (64 tokens x 8 dims), k = the tokens of a PAIR of 32-token tiles.
//   * The MMA fixes which k an A / B register holds (thread (g, t) of a quad: A reg r = row g + 8 (r & 1), k = 32 (r >> 1) + 8 t + j,
//     j = 0..7, low nibble first; B reg r = column g, k = 32 r + 8 t + j) and which k block a scale byte covers (byte kb = k
//     [16 kb, +16)); mma_semantics.cu (R-082) measured the scale sources: row m of A takes the register of thread 4 (m % 8) + (m >= 8)
//     (selector 0), column n of B the register of thread 4 n + tid_b.
//   * Which token sits at which k is free as long as P and V^T agree: the S accumulators decide it for P, and the rows the QK
//     ldmatrix is given decide what an S column is. tok_of() below is the order used here (see the V^T layout).
//
// V^T of a pair of tiles (hh = the tile of the pair):
//   data    per dim d, 32 B = 2 x 16 B (hh); in a 16 B chunk word t (t = 0..3) is the B register of thread (g, t): its 8 nibbles
//           m = 4 i + j are the tokens tok_of(i, t, j) of the tile (value of dim d, e2m1, low nibble first)
//   scales  per dim, 4 ue4m3 bytes kb = 2 hh + q, q = 0 / 1 for the two 16-token blocks of a tile: block q = the tokens whose word
//           t has t >> 1 == q, i.e. tokens 8 q .. 8 q + 7 and 16 + 8 q .. 16 + 8 q + 7. Scales of a 32-dim group are one 128 B
//           matrix of 8 rows (dim % 8) x 16 B (dim / 8 % 4, kb), so that one ldmatrix gives a thread the scale register of 4 n-tiles
//           and the MMA picks its dim with tid_b.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "mma_kinds.h"

#define CK(x)                                                                                    \
    do {                                                                                         \
        cudaError_t e_ = (x);                                                                    \
        if (e_ != cudaSuccess) {                                                                 \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);     \
            exit(1);                                                                             \
        }                                                                                        \
    } while (0)

#define HD __host__ __device__

namespace t09 {

// ---------------------------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------------------------
constexpr int TILE = 32;       // tokens per tile
constexpr int DIMS = 512;
constexpr int NGROUPS = 16;    // 32-dim groups

// A raw tile in shared memory as the kernel keeps it (mxfp8.cu): groups of 4 rows at a stride of 2176 B
constexpr int GROUP_STRIDE = 2176;
constexpr int RAW_BYTES = 8 * GROUP_STRIDE;      // 17408
constexpr int FP8_ROW = 528;                     // 512 B e4m3 + 16 B ue8m0 (per 32 dims)
constexpr int FP4_ROW = 288;                     // 256 B e2m1 (even dim in the low nibble) + 32 B e4m3 (per 16 dims)
constexpr int FP4_STRIDE = 304;
constexpr int MX_ROW = 272;                      // 256 B e2m1 + 16 B ue8m0 (per 32 dims); rows one after another (mxfp4.cu)

HD inline int fp8_off(int r) { return (r >> 2) * GROUP_STRIDE + (r & 3) * FP8_ROW; }
HD inline int fp4_off(int r) { return (r >> 2) * GROUP_STRIDE + ((r >> 2) & 1) * 64 + (r & 3) * FP4_STRIDE; }
HD inline int mx_off(int r) { return r * MX_ROW; }

enum Src { SRC_FP8 = 0, SRC_FP4 = 1, SRC_MX = 2, SRC_E4M3 = 3 };
HD inline const char *src_name(int s) { return s == SRC_FP8 ? "fp8" : s == SRC_FP4 ? "fp4x" : s == SRC_MX ? "mxfp4" : "e4m3t"; }

// The token at nibble m = 4 i + j of word t. The order inside a group of 4 is rotated by t so that the 8 rows an S
// column set (see QK) reads fall on 4 different 16 B bank positions twice, as the rows of a tile do now.
HD inline int tok_of(int i, int t, int j) { return 16 * i + 4 * t + ((j + t) & 3); }

// V^T layout. L = 0: 32 B per dim, the two 16 B chunks swapped for dims 4..7 mod 8 (an ldmatrix of 8 dims is conflict free);
// 1: 48 B per dim (padding); 2: 32 B per dim without swizzle; 3: 8-dim matrices of 128 B (chunk-major).
template <int L> HD constexpr int vt_data_bytes() { return L == 1 ? DIMS * 48 : DIMS * 32; }
template <int L> HD inline int vt_data_off(int d, int hh) {
    return L == 0 ? d * 32 + ((hh ^ ((d >> 2) & 1)) << 4)
         : L == 1 ? d * 48 + hh * 16
         : L == 2 ? d * 32 + hh * 16
                  : (((d >> 3) * 2 + hh) << 7) + (d & 7) * 16;
}
constexpr int VT_SCALE_BYTES = DIMS * 4;
template <int L> HD constexpr int vt_buf_bytes() { return vt_data_bytes<L>() + VT_SCALE_BYTES; }
// The scale bytes of dim d (kb = 2 hh + q is added)
template <int L> HD inline int vt_scale_off(int d) { return vt_data_bytes<L>() + (d >> 5) * 128 + (d & 7) * 16 + ((d >> 3) & 3) * 4; }

// ---------------------------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t smem_addr(const void *p) { return uint32_t(__cvta_generic_to_shared(p)); }

__device__ __forceinline__ void mbarrier_init(uint64_t *bar, uint32_t arrivals) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_addr(bar)), "r"(arrivals) : "memory");
}
__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t *bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_addr(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbarrier_wait(uint64_t *bar, uint32_t parity) {
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "WAIT_LOOP:\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
                 "@p bra DONE;\n"
                 "bra WAIT_LOOP;\n"
                 "DONE:\n"
                 "}\n" ::"r"(smem_addr(bar)), "r"(parity) : "memory");
}
__device__ __forceinline__ void bulk_copy(void *dst, const void *src, uint32_t bytes, uint64_t *bar) {
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                 ::"r"(smem_addr(dst)), "l"(src), "r"(bytes), "r"(smem_addr(bar)) : "memory");
}
template <int NUM_REGS> __device__ __forceinline__ void setmaxnreg_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(NUM_REGS)); }
template <int NUM_REGS> __device__ __forceinline__ void setmaxnreg_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(NUM_REGS)); }
__device__ __forceinline__ void named_barrier_sync(uint32_t id, uint32_t n) { asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }

__device__ __forceinline__ uint32_t lds32(uint32_t addr) {
    uint32_t v;
    asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(v) : "r"(addr));
    return v;
}
__device__ __forceinline__ uint32_t lds8(uint32_t addr) {
    uint32_t v;
    asm volatile("ld.shared.u8 %0, [%1];\n" : "=r"(v) : "r"(addr));
    return v;
}
__device__ __forceinline__ void sts32(uint32_t addr, uint32_t v) { asm volatile("st.shared.b32 [%0], %1;\n" ::"r"(addr), "r"(v) : "memory"); }
__device__ __forceinline__ void sts8(uint32_t addr, uint32_t v) { asm volatile("st.shared.b8 [%0], %1;\n" ::"r"(addr), "r"(v) : "memory"); }
__device__ __forceinline__ void sts128(uint32_t addr, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n" ::"r"(addr), "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
}

// 8x8 b16 matrices; lane l gives the (16 B aligned) address of row l % 8 of matrix l / 8; thread (g, t) gets row g, bytes 4t..4t+3
__device__ __forceinline__ void ldmatrix_x1(uint32_t &r0, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n" : "=r"(r0) : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x2(uint32_t (&f)[2], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(f[0]), "=r"(f[1]) : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&f)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(f[0]), "=r"(f[1]), "=r"(f[2]), "=r"(f[3]) : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&f)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(f[0]), "=r"(f[1]), "=r"(f[2]), "=r"(f[3]) : "r"(addr));
}
// Two 16x16 byte matrices, transposed; lane l gives the address of row l % 16 of matrix l / 16 (16 B); thread (g, t) gets in
// register 2 i + h the bytes (rows 4t .. 4t+3, column 8 h + g) of matrix i
__device__ __forceinline__ void ldmatrix_x2_trans_b8(uint32_t (&f)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(f[0]), "=r"(f[1]), "=r"(f[2]), "=r"(f[3]) : "r"(addr));
}

__device__ __forceinline__ uint32_t f32x2_to_e2m1x2(float lo, float hi) {
    uint32_t r;
    asm("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %1, %2; cvt.u32.u8 %0, t; }\n" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}
__device__ __forceinline__ uint32_t f32x2_to_e4m3x2(float lo, float hi) {
    uint16_t r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(r) : "f"(hi), "f"(lo));
    return r;
}
__device__ __forceinline__ uint32_t f16x2_to_e4m3x2(uint32_t v) {
    uint16_t r;
    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;\n" : "=h"(r) : "r"(v));
    return r;
}
__device__ __forceinline__ uint32_t e4m3x2_to_f16x2(uint32_t two_bytes) {
    uint32_t r;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(r) : "h"(static_cast<uint16_t>(two_bytes)));
    return r;
}
__device__ __forceinline__ uint32_t e2m1x2_to_f16x2(uint32_t one_byte) {
    uint32_t r;
    asm("{ .reg .b8 t; cvt.u8.u32 t, %1; cvt.rn.f16x2.e2m1x2 %0, t; }\n" : "=r"(r) : "r"(one_byte));
    return r;
}
__device__ __forceinline__ float f16lo_to_f32(uint32_t v) { return __half2float(__ushort_as_half(static_cast<unsigned short>(v))); }
__device__ __forceinline__ float f16hi_to_f32(uint32_t v) { return __half2float(__ushort_as_half(static_cast<unsigned short>(v >> 16))); }

// The smallest e4m3 value s with 6 s >= amax, as its code (amax >= 0), in [1, 0x7e]; two at once, `lo` in the low byte
// (nvfp4_scale_codes of mxfp8.cu)
__device__ __forceinline__ uint32_t nvfp4_scale_codes(float amax_lo, float amax_hi) {
    const uint32_t codes = f32x2_to_e4m3x2(amax_lo * (1.0f / 6.0f), amax_hi * (1.0f / 6.0f));
    const uint32_t scales = e4m3x2_to_f16x2(codes);
    uint32_t code_lo = codes & 0xff, code_hi = codes >> 8;
    code_lo += code_lo < 0x7e && 6.0f * f16lo_to_f32(scales) < amax_lo ? 1 : 0;
    code_hi += code_hi < 0x7e && 6.0f * f16hi_to_f32(scales) < amax_hi ? 1 : 0;
    return max(code_lo, 1u) | (max(code_hi, 1u) << 8);
}

__device__ __forceinline__ long long clock64_() {
    long long t;
    asm volatile("mov.u64 %0, %%clock64;\n" : "=l"(t));
    return t;
}


// f32 = a.h * b.h (the exact product of two f16 halves, fma.rn.f32.f16 of sm_100+), H = 0 / 1: low / high half. The addend is -0
// so that a -0 product stays -0 (+0 would turn it into +0; the e2m1 code of -0 is 0x8)
template <int H> __device__ __forceinline__ float hmul_f32(uint32_t a2, uint32_t b2) {
    float r;
    if constexpr (H == 0)
        asm("{ .reg .b16 a0, a1, b0, b1; mov.b32 {a0, a1}, %1; mov.b32 {b0, b1}, %2; fma.rn.f32.f16 %0, a0, b0, 0f80000000; }\n" : "=f"(r) : "r"(a2), "r"(b2));
    else
        asm("{ .reg .b16 a0, a1, b0, b1; mov.b32 {a0, a1}, %1; mov.b32 {b0, b1}, %2; fma.rn.f32.f16 %0, a1, b1, 0f80000000; }\n" : "=f"(r) : "r"(a2), "r"(b2));
    return r;
}
// byte M of w (two e2m1, low nibble first) -> two f16
template <int M> __device__ __forceinline__ uint32_t e2m1_byte_f16x2(uint32_t w) {
    uint32_t r;
    if constexpr (M == 0) asm("{ .reg .b8 b0, b1, b2, b3; mov.b32 {b0, b1, b2, b3}, %1; cvt.rn.f16x2.e2m1x2 %0, b0; }\n" : "=r"(r) : "r"(w));
    else if constexpr (M == 1) asm("{ .reg .b8 b0, b1, b2, b3; mov.b32 {b0, b1, b2, b3}, %1; cvt.rn.f16x2.e2m1x2 %0, b1; }\n" : "=r"(r) : "r"(w));
    else if constexpr (M == 2) asm("{ .reg .b8 b0, b1, b2, b3; mov.b32 {b0, b1, b2, b3}, %1; cvt.rn.f16x2.e2m1x2 %0, b2; }\n" : "=r"(r) : "r"(w));
    else asm("{ .reg .b8 b0, b1, b2, b3; mov.b32 {b0, b1, b2, b3}, %1; cvt.rn.f16x2.e2m1x2 %0, b3; }\n" : "=r"(r) : "r"(w));
    return r;
}
__device__ __forceinline__ uint32_t lds_u32(uint32_t addr) { return lds32(addr); }
__device__ __forceinline__ float lds_f32(uint32_t addr) {
    float v;
    asm volatile("ld.shared.f32 %0, [%1];\n" : "=f"(v) : "r"(addr));
    return v;
}

// Two dims at once (8 values each, index = nibble): amax with the partner thread (xor 1), the scale codes, and the e2m1 words.
// `lut_s` is the shared address of the table of __frcp_rn(scale) by code.
__device__ __forceinline__ void quant_pair(const float (&xa)[8], const float (&xb)[8], uint32_t lut_s, uint32_t &wa, uint32_t &wb, uint32_t &codes) {
    float aa = 0.f, ab = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        aa = fmaxf(aa, fabsf(xa[k]));
        ab = fmaxf(ab, fabsf(xb[k]));
    }
    aa = fmaxf(aa, __shfl_xor_sync(0xffffffffu, aa, 1));
    ab = fmaxf(ab, __shfl_xor_sync(0xffffffffu, ab, 1));
    codes = nvfp4_scale_codes(aa, ab);
    const float inva = lds_f32(lut_s + 4 * (codes & 0xffu)), invb = lds_f32(lut_s + 4 * (codes >> 8));
    wa = 0;
    wb = 0;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        wa |= f32x2_to_e2m1x2(xa[2 * p] * inva, xa[2 * p + 1] * inva) << (8 * p);
        wb |= f32x2_to_e2m1x2(xb[2 * p] * invb, xb[2 * p + 1] * invb) << (8 * p);
    }
}

// ---------------------------------------------------------------------------------------------
// Host side: element formats, the quantization rules and a reference of the conversions
// ---------------------------------------------------------------------------------------------
inline double h_e4m3(uint32_t c) {
    const int s = (c >> 7) & 1, e = (c >> 3) & 15, m = c & 7;
    const double v = e == 0 ? std::ldexp(m / 8.0, -6) : std::ldexp(1.0 + m / 8.0, e - 7);
    return s ? -v : v;
}
inline double h_e2m1(uint32_t c) {
    const int s = (c >> 3) & 1, e = (c >> 1) & 3, m = c & 1;
    const double v = e == 0 ? m * 0.5 : std::ldexp(1.0 + m * 0.5, e - 1);
    return s ? -v : v;
}
// cvt.rn.satfinite.e2m1x2.f32 for one value: nearest of {0, .5, 1, 1.5, 2, 3, 4, 6}, ties to the even code, saturating
inline uint32_t h_e2m1_rne(float x) {
    const uint32_t sign = std::signbit(x) ? 8u : 0u;
    const float a = std::fabs(x);
    uint32_t c;
    if (a <= 0.25f) c = 0;
    else if (a < 0.75f) c = 1;
    else if (a <= 1.25f) c = 2;
    else if (a < 1.75f) c = 3;
    else if (a <= 2.5f) c = 4;
    else if (a < 3.5f) c = 5;
    else if (a <= 5.0f) c = 6;
    else c = 7;
    return c | sign;
}
// The smallest e4m3 code in [1, 0x7e] whose value s has 6 s >= amax (0x7e if there is none)
inline uint32_t h_scale_code(float amax) {
    for (uint32_t c = 1; c < 0x7e; ++c)
        if (6.0f * float(h_e4m3(c)) >= amax) return c;
    return 0x7e;
}
inline float h_rcp(uint32_t code) { return float(1.0 / h_e4m3(code)); }  // __frcp_rn of the scale

// Deterministic generator
struct Rng {
    std::mt19937_64 g;
    explicit Rng(uint64_t seed) : g(seed) {}
    uint32_t u32() { return uint32_t(g()); }
    uint32_t below(uint32_t n) { return uint32_t(g() % n); }
    bool chance(double p) { return (g() >> 11) * (1.0 / 9007199254740992.0) < p; }
};

// One raw tile of some source format, as the bytes in the shared memory layout plus the exact value of every element
// (before the first-level shift 2^-E1)
struct RawTile {
    int src = SRC_FP8;
    std::vector<uint8_t> raw;      // RAW_BYTES
    uint32_t valid = 0;            // bit i: token i is valid
    std::vector<double> val;       // [32][512]; invalid tokens: 0
    RawTile() : raw(RAW_BYTES, 0), val(TILE * DIMS, 0.0) {}
};

// Data scenarios (a tile of each is generated from a seed)
enum Scenario { SC_RANDOM = 0, SC_ALL_INVALID, SC_ALL_ZERO, SC_BIG, SC_TINY, SC_EXTREME, SC_OUTLIER, SC_FEW_VALID, SC_COUNT };
inline const char *scenario_name(int s) {
    static const char *n[] = {"random", "all-invalid", "all-zero", "big(s=0x7e)", "tiny(s=1)", "extreme-bytes", "outlier", "few-valid"};
    return n[s];
}

// The 16 (fp8, mx) or 32 (fp4) ue8m0 / e4m3 scale bytes are written at the end of the row; E1[g] is the first-level shift of group g
struct Scenarios {
    // exponent range of the ue8m0 scales (biased) and of E1, per scenario
    static void exp_range(int sc, int &lo, int &hi) {
        switch (sc) {
            case SC_BIG: lo = 130; hi = 142; break;
            case SC_TINY: lo = 60; hi = 80; break;
            case SC_EXTREME: lo = 90; hi = 160; break;
            default: lo = 119; hi = 135; break;
        }
    }
};

inline uint8_t pick_e4m3_byte(Rng &r, int sc, bool allow_nan) {
    static const uint8_t extremes[] = {0x00, 0x80, 0x01, 0x81, 0x07, 0x87, 0x08, 0x38, 0xb8, 0x7e, 0xfe, 0x77, 0x40};
    uint32_t b;
    if (sc == SC_ALL_ZERO) return r.chance(0.5) ? 0x00 : 0x80;
    if (sc == SC_EXTREME && r.chance(0.5)) return extremes[r.below(sizeof(extremes))];
    do b = r.u32() & 0xff; while (!allow_nan && (b & 0x7f) == 0x7f);
    return uint8_t(b);
}
inline uint8_t pick_nib(Rng &r, int sc) {
    if (sc == SC_ALL_ZERO) return r.chance(0.5) ? 0 : 8;
    if (sc == SC_EXTREME && r.chance(0.5)) { static const uint8_t e[] = {0, 8, 1, 9, 7, 15, 2, 6}; return e[r.below(8)]; }
    return uint8_t(r.u32() & 15);
}

// Fill `t` with a tile of source format `src` and scenario `sc`. make_e1 draws the first-level shifts E1 (per 32-dim group;
// the tiles of a CTA share them). Scales are chosen so that every value times 2^-E1 is a normal float or zero.
inline void make_e1(int sc, uint64_t seed, int (&e1)[NGROUPS], bool zero = false) {
    Rng r(seed ^ 0x9e3779b97f4a7c15ull);
    if (zero) { for (int g = 0; g < NGROUPS; ++g) e1[g] = 0; return; }
    for (int g = 0; g < NGROUPS; ++g) e1[g] = (sc == SC_RANDOM || sc == SC_OUTLIER) ? int(r.below(7)) - 3 : (sc == SC_BIG || sc == SC_TINY ? 0 : int(r.below(5)) - 2);
}

inline void make_tile(RawTile &t, int src, int sc, uint64_t seed) {
    Rng r(seed);
    t.src = src;
    std::fill(t.raw.begin(), t.raw.end(), 0);
    std::fill(t.val.begin(), t.val.end(), 0.0);
    int lo, hi;
    Scenarios::exp_range(sc, lo, hi);
    // validity
    uint32_t valid = 0;
    switch (sc) {
        case SC_ALL_INVALID: valid = 0; break;
        case SC_FEW_VALID: for (int k = 0; k < 3; ++k) valid |= 1u << r.below(TILE); break;
        default: for (int i = 0; i < TILE; ++i) if (!r.chance(0.1)) valid |= 1u << i; break;
    }
    t.valid = valid;
    // a token with a much larger exponent than the rest in the outlier scenario
    const int outlier_tok = int(r.below(TILE));
    for (int tok = 0; tok < TILE; ++tok) {
        const bool ok = (valid >> tok) & 1;
        uint8_t *row;
        int off;
        if (src == SRC_FP8) off = fp8_off(tok);
        else if (src == SRC_FP4) off = fp4_off(tok);
        else off = mx_off(tok);
        row = t.raw.data() + off;
        if (src == SRC_FP8) {
            for (int d = 0; d < DIMS; ++d) row[d] = pick_e4m3_byte(r, sc, !ok);
            for (int g = 0; g < NGROUPS; ++g) {
                int e = lo + int(r.below(hi - lo + 1));
                if (sc == SC_OUTLIER) e = tok == outlier_tok ? 140 : 110 + int(r.below(6));
                row[512 + g] = uint8_t(ok ? e : r.u32() & 0xff);
            }
            if (ok)
                for (int d = 0; d < DIMS; ++d) t.val[tok * DIMS + d] = h_e4m3(row[d]) * std::ldexp(1.0, int(row[512 + d / 32]) - 127);
        } else if (src == SRC_FP4) {
            for (int i = 0; i < 256; ++i) row[i] = uint8_t(pick_nib(r, sc) | (pick_nib(r, sc) << 4));
            for (int k = 0; k < 32; ++k) {
                // e4m3 scale per 16 dims: codes 0..0x7e, normal and subnormal; the range follows the scenario
                uint32_t code;
                if (sc == SC_BIG) code = 0x60 + r.below(0x1f);              // up to 0x7e
                else if (sc == SC_TINY) code = r.below(8);                  // zero and subnormals
                else if (sc == SC_EXTREME) { static const uint8_t c[] = {0, 1, 7, 8, 0x38, 0x7e, 0x77}; code = c[r.below(7)]; }
                else code = 0x28 + r.below(0x20);
                if (sc == SC_OUTLIER) code = tok == outlier_tok ? 0x70 : 0x20 + r.below(8);
                row[256 + k] = uint8_t(ok ? code : r.u32() & 0xff);
            }
            if (ok)
                for (int d = 0; d < DIMS; ++d) {
                    const uint32_t nib = (row[d / 2] >> (4 * (d & 1))) & 15;
                    t.val[tok * DIMS + d] = h_e2m1(nib) * h_e4m3(row[256 + d / 16]);
                }
        } else {  // SRC_MX
            for (int i = 0; i < 256; ++i) row[i] = uint8_t(pick_nib(r, sc) | (pick_nib(r, sc) << 4));
            for (int g = 0; g < NGROUPS; ++g) {
                int e = lo + int(r.below(hi - lo + 1));
                if (sc == SC_OUTLIER) e = tok == outlier_tok ? 140 : 110 + int(r.below(6));
                row[256 + g] = uint8_t(ok ? e : r.u32() & 0xff);
            }
            if (ok)
                for (int d = 0; d < DIMS; ++d) {
                    const uint32_t nib = (row[d / 2] >> (4 * (d & 1))) & 15;
                    t.val[tok * DIMS + d] = h_e2m1(nib) * std::ldexp(1.0, int(row[256 + d / 32]) - 127);
                }
        }
    }
}

// x[tok][dim] as the converters see it: float(value * 2^-E1) (checked to be exact)
inline void shifted_values(const RawTile &t, const int (&e1)[NGROUPS], std::vector<float> &x) {
    x.assign(TILE * DIMS, 0.f);
    for (int tok = 0; tok < TILE; ++tok)
        for (int d = 0; d < DIMS; ++d) {
            const double v = std::ldexp(t.val[tok * DIMS + d], -e1[d / 32]);
            const float f = float(v);
            if (double(f) != v) {
                printf("make_tile: value %g * 2^-%d is not an exact float (tok %d dim %d)\n", t.val[tok * DIMS + d], e1[d / 32], tok, d);
                exit(1);
            }
            x[tok * DIMS + d] = f;
        }
}

// Token orders of the nibbles of a V^T word: ORDER_LDM (nibble 4 i + j of word t = tok_of(i, t, j), what the ldmatrix.trans.b8
// route of the fp8 rows produces) and ORDER_NAT (nibble 2 m + c of word t = token 8 m + 2 t + c, the order of the S
// accumulators of an unpermuted QK, what the ldmatrix.trans.b16 route of the nibble rows produces). Block q = word 2q, 2q+1.
enum Order { ORDER_LDM = 0, ORDER_NAT = 1 };
HD inline int tok_of_order(int order, int t, int nib) { return order == ORDER_LDM ? tok_of(nib >> 2, t, nib & 3) : 8 * (nib >> 1) + 2 * t + (nib & 1); }

// Expected V^T of one tile: data[d][4 words] and scale codes sc[d][2]
struct VtRef {
    std::vector<uint32_t> data;   // [512][4]
    std::vector<uint8_t> scale;   // [512][2]
    VtRef() : data(DIMS * 4, 0), scale(DIMS * 2, 0) {}
};
inline void vt_reference(const std::vector<float> &x, uint32_t valid, VtRef &ref, int order) {
    for (int d = 0; d < DIMS; ++d)
        for (int q = 0; q < 2; ++q) {
            float amax = 0.f;
            for (int t = 2 * q; t < 2 * q + 2; ++t)
                for (int m = 0; m < 8; ++m) {
                    const int tok = tok_of_order(order, t, m);
                    if ((valid >> tok) & 1) amax = std::max(amax, std::fabs(x[tok * DIMS + d]));
                }
            const uint32_t code = h_scale_code(amax);
            const float inv = h_rcp(code);
            ref.scale[d * 2 + q] = uint8_t(code);
            for (int t = 2 * q; t < 2 * q + 2; ++t) {
                uint32_t word = 0;
                for (int m = 0; m < 8; ++m) {
                    const int tok = tok_of_order(order, t, m);
                    const uint32_t nib = ((valid >> tok) & 1) ? h_e2m1_rne(x[tok * DIMS + d] * inv) : 0u;
                    word |= nib << (4 * m);
                }
                ref.data[d * 4 + t] = word;
            }
        }
}

}  // namespace t09
