// What the sm120 `mma.sync` kinds actually compute, checked on the device against a host reference:
//   1. which (thread, byte) of the scale registers feeds each row of A / column of B, for every
//      byte-id / thread-id selector that assembles;
//   2. full random 16 x 8 x K products (fragment layout, e2m1 container position, scale decoding);
//   3. how many bits the f32 accumulator keeps, across MMAs and inside one MMA.
//
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -o mma_semantics mma_semantics.cu
// Run:   ./mma_semantics

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <cuda_runtime.h>

#include "mma_kinds.h"

#define CK(x)                                                                                    \
    do {                                                                                         \
        cudaError_t e_ = (x);                                                                    \
        if (e_ != cudaSuccess) {                                                                 \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);     \
            exit(1);                                                                             \
        }                                                                                        \
    } while (0)

// ---------------------------------------------------------------------------------------------
// Element formats
// ---------------------------------------------------------------------------------------------

enum Elem { EL_BF16, EL_E4M3, EL_E2M1_IN_8, EL_E2M1 };
enum ScaleType { SF_NONE, SF_UE8M0, SF_UE4M3 };

static double dec_e4m3(uint32_t x) {
    const int s = (x >> 7) & 1, e = (x >> 3) & 15, m = x & 7;
    const double v = e == 0 ? std::ldexp(m / 8.0, -6) : std::ldexp(1.0 + m / 8.0, e - 7);
    return s ? -v : v;
}
static double dec_e2m1(uint32_t x) {
    const int s = (x >> 3) & 1, e = (x >> 1) & 3, m = x & 1;
    const double v = e == 0 ? m * 0.5 : std::ldexp(1.0 + m * 0.5, e - 1);
    return s ? -v : v;
}
static double dec_bf16(uint32_t x) {
    uint32_t bits = x << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}
static double dec_scale(ScaleType t, uint32_t x) {
    if (t == SF_UE8M0) return std::ldexp(1.0, int(x) - 127);
    if (t == SF_UE4M3) return dec_e4m3(x & 0x7f);
    return 1.0;
}
// e2m1 inside an 8-bit container of the f8f6f4 kinds: the 4 bits start at `g_e2m1_shift`
static int g_e2m1_shift = 2;
static double dec_elem(Elem t, uint32_t code) {
    switch (t) {
        case EL_BF16: return dec_bf16(code);
        case EL_E4M3: return dec_e4m3(code);
        case EL_E2M1_IN_8: return dec_e2m1((code >> g_e2m1_shift) & 15);
        default: return dec_e2m1(code);
    }
}
// The code of +2^x
static uint32_t pow2_code(Elem t, int x) {
    switch (t) {
        case EL_BF16: return uint32_t(x + 127) << 7;
        case EL_E4M3: return x >= -6 ? uint32_t(x + 7) << 3 : 1u << (x + 9);
        case EL_E2M1_IN_8: return (x == -1 ? 1u : uint32_t(x + 1) << 1) << g_e2m1_shift;
        default: return x == -1 ? 1u : uint32_t(x + 1) << 1;
    }
}
static uint32_t random_code(Elem t, std::mt19937 &rng) {
    switch (t) {
        case EL_BF16: return ((rng() & 1) << 15) | ((127 + int(rng() % 9) - 4) << 7) | (rng() & 127);
        case EL_E4M3: {
            uint32_t c;
            do c = rng() & 255; while ((c & 0x7f) == 0x7f);
            return c;
        }
        case EL_E2M1_IN_8: return (rng() & 15) << g_e2m1_shift;
        default: return rng() & 15;
    }
}

template <class Op> struct Info;
#define INFO(OP, A_, B_, SF_, SFK_)                                                              \
    template <> struct Info<OP> {                                                                \
        static constexpr Elem kA = A_, kB = B_;                                                  \
        static constexpr ScaleType kSf = SF_;                                                    \
        static constexpr int kSfK = SFK_; /* k elements covered by one scale */                  \
    };
INFO(BF16_F32, EL_BF16, EL_BF16, SF_NONE, 16)
INFO(E4M3_F32, EL_E4M3, EL_E4M3, SF_NONE, 32)
INFO(KF8_E4M3_F32, EL_E4M3, EL_E4M3, SF_NONE, 32)
INFO(KF8_E4M3_E2M1_F32, EL_E4M3, EL_E2M1_IN_8, SF_NONE, 32)
INFO(KF8_E2M1_F32, EL_E2M1_IN_8, EL_E2M1_IN_8, SF_NONE, 32)
INFO(MXF8_E4M3, EL_E4M3, EL_E4M3, SF_UE8M0, 32)
INFO(MXF8_E4M3_E2M1, EL_E4M3, EL_E2M1_IN_8, SF_UE8M0, 32)
INFO(MXF8_E2M1, EL_E2M1_IN_8, EL_E2M1_IN_8, SF_UE8M0, 32)
INFO(MXF4_2X, EL_E2M1, EL_E2M1, SF_UE8M0, 32)
INFO(NVF4_4X, EL_E2M1, EL_E2M1, SF_UE4M3, 16)

// ---------------------------------------------------------------------------------------------
// One MMA on fragments prepared by the host
// ---------------------------------------------------------------------------------------------

struct Frag {
    uint32_t a[4];
    uint32_t b[2];
    float c[4];
    uint32_t sfa, sfb;
};

template <class Op, int BA, int TA, int BB, int TB>
__global__ void one_mma_kernel(const Frag *in, float *out) {
    const Frag f = in[threadIdx.x];
    AccF32 acc;
    for (int i = 0; i < 4; ++i) acc.v[i] = f.c[i];
    Op::template run<BA, TA, BB, TB>(acc, f.a, f.b, f.sfa, f.sfb);
    for (int i = 0; i < 4; ++i) out[threadIdx.x * 4 + i] = acc.v[i];
}

static Frag *g_frag_d;
static float *g_out_d;

using Launcher = void (*)();
template <class Op, int BA, int TA, int BB, int TB> void launch() {
    one_mma_kernel<Op, BA, TA, BB, TB><<<1, 32>>>(g_frag_d, g_out_d);
    CK(cudaDeviceSynchronize());
}

// The operands of one MMA in matrix form. A is [16][K], B is [8][K] (codes), C / D are [16][8].
struct Problem {
    int K, elem_bits;
    uint32_t a[16][64], b[8][64];
    float c[16][8];
    uint32_t sfa[32], sfb[32];  // scale register of every thread
    double d[16][8];            // result
    void clear() { memset(this, 0, offsetof(Problem, d)); }
};

static void run_problem(Problem &p, Launcher fn) {
    const int per_reg = 32 / p.elem_bits;
    Frag frag[32];
    for (int T = 0; T < 32; ++T) {
        const int g = T / 4, t = T % 4;
        for (int r = 0; r < 4; ++r) {
            uint32_t v = 0;
            for (int j = 0; j < per_reg; ++j)
                v |= p.a[g + 8 * (r & 1)][(p.K / 2) * (r >> 1) + per_reg * t + j] << (p.elem_bits * j);
            frag[T].a[r] = v;
        }
        for (int r = 0; r < 2; ++r) {
            uint32_t v = 0;
            for (int j = 0; j < per_reg; ++j) v |= p.b[g][(p.K / 2) * r + per_reg * t + j] << (p.elem_bits * j);
            frag[T].b[r] = v;
        }
        for (int i = 0; i < 4; ++i) frag[T].c[i] = p.c[g + 8 * (i >> 1)][2 * t + (i & 1)];
        frag[T].sfa = p.sfa[T];
        frag[T].sfb = p.sfb[T];
    }
    float out[32 * 4];
    CK(cudaMemcpy(g_frag_d, frag, sizeof(frag), cudaMemcpyHostToDevice));
    fn();
    CK(cudaMemcpy(out, g_out_d, sizeof(out), cudaMemcpyDeviceToHost));
    for (int T = 0; T < 32; ++T)
        for (int i = 0; i < 4; ++i) p.d[T / 4 + 8 * (i >> 1)][2 * (T % 4) + (i & 1)] = out[T * 4 + i];
}

// ---------------------------------------------------------------------------------------------
// 1. Which (thread, byte) supplies each scale
// ---------------------------------------------------------------------------------------------

struct Selector {
    int ba, ta, bb, tb;
    Launcher fn;
};

// Returns the mapping as source[m or n][k block] = thread * 4 + byte, or -1 if it is inconsistent
template <class Op>
static void identify(const Selector &sel, bool side_a, int source[16][4]) {
    const ScaleType st = Info<Op>::kSf;
    const int blocks = Op::K / Info<Op>::kSfK;
    const int unit = st == SF_UE8M0 ? 127 : 0x38;
    const int rows = side_a ? 16 : 8;
    for (int kb = 0; kb < blocks; ++kb) {
        int thread_of[16], byte_of[16];
        for (int pass = 0; pass < 2; ++pass) {
            Problem p;
            p.clear();
            p.K = Op::K;
            p.elem_bits = Op::kElemBits;
            // Ones inside k block `kb` only, so the result isolates that block's scales
            for (int k = kb * Info<Op>::kSfK; k < (kb + 1) * Info<Op>::kSfK; ++k) {
                for (int m = 0; m < 16; ++m) p.a[m][k] = pow2_code(Info<Op>::kA, 0);
                for (int n = 0; n < 8; ++n) p.b[n][k] = pow2_code(Info<Op>::kB, 0);
            }
            // pass 0: every thread has its own scale value; pass 1: every byte position has
            uint32_t candidates[32];
            const int num_candidates = pass == 0 ? 32 : 4;
            for (int T = 0; T < 32; ++T) {
                uint32_t reg = 0;
                for (int j = 0; j < 4; ++j) {
                    const uint32_t code = pass == 0 ? (st == SF_UE8M0 ? 111u + T : 0x20u + T)
                                                    : (st == SF_UE8M0 ? 127u + j : 0x38u + 8u * j);
                    candidates[pass == 0 ? T : j] = code;
                    reg |= code << (8 * j);
                }
                const uint32_t unit_reg = unit * 0x01010101u;
                p.sfa[T] = side_a ? reg : unit_reg;
                p.sfb[T] = side_a ? unit_reg : reg;
            }
            run_problem(p, sel.fn);
            for (int i = 0; i < rows; ++i) {
                int found = -1;
                for (int j = 0; j < (side_a ? 8 : 16); ++j) {
                    const double v = (side_a ? p.d[i][j] : p.d[j][i]) / Info<Op>::kSfK;
                    int match = -1;
                    for (int c = 0; c < num_candidates; ++c)
                        if (v == dec_scale(st, candidates[c])) match = c;
                    if (j == 0) found = match;
                    else if (match != found) found = -1;
                }
                (pass == 0 ? thread_of : byte_of)[i] = found;
            }
        }
        for (int i = 0; i < rows; ++i)
            source[i][kb] = thread_of[i] < 0 || byte_of[i] < 0 ? -1 : thread_of[i] * 4 + byte_of[i];
    }
}

template <class Op>
static void report_selectors(const std::vector<Selector> &sels_a, const std::vector<Selector> &sels_b) {
    const int blocks = Op::K / Info<Op>::kSfK;
    printf("%s: %d scale(s) per row / column, one per %d k\n", Op::kName, blocks, Info<Op>::kSfK);
    for (int side = 0; side < 2; ++side) {
        const bool side_a = side == 0;
        for (const Selector &sel : side_a ? sels_a : sels_b) {
            int source[16][4];
            identify<Op>(sel, side_a, source);
            const int rows = side_a ? 16 : 8;
            // Fit: thread = 4 * (row % 8) + thread_base + (row >= 8 ? thread_hi : 0), byte = byte_base + k block
            const int thread_base = source[0][0] < 0 ? -1 : (source[0][0] / 4) % 4;
            const int thread_hi = !side_a || source[8][0] < 0 ? 0 : (source[8][0] / 4) % 4 - thread_base;
            const int byte_base = source[0][0] < 0 ? -1 : source[0][0] % 4;
            bool regular = thread_base >= 0;
            for (int i = 0; i < rows && regular; ++i)
                for (int kb = 0; kb < blocks; ++kb)
                    regular = regular && source[i][kb] == (4 * (i % 8) + thread_base + (i >= 8 ? thread_hi : 0)) * 4 + byte_base + kb;
            printf("  SF_%c byte-id %d thread-id %d: ", side_a ? 'A' : 'B', side_a ? sel.ba : sel.bb, side_a ? sel.ta : sel.tb);
            if (regular) {
                if (side_a)
                    printf("row m <- thread 4*(m%%8)+%d (m<8) / +%d (m>=8), byte %d + kblock\n", thread_base, thread_base + thread_hi, byte_base);
                else
                    printf("col n <- thread 4*n+%d, byte %d + kblock\n", thread_base, byte_base);
            } else {
                printf("irregular:");
                for (int i = 0; i < rows; ++i)
                    for (int kb = 0; kb < blocks; ++kb) printf(" %d.%d", source[i][kb] < 0 ? -1 : source[i][kb] / 4, source[i][kb] < 0 ? 0 : source[i][kb] % 4);
                printf("\n");
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// 2. Random products against a double precision reference
// ---------------------------------------------------------------------------------------------

template <class Op> static double verify_random(Launcher fn, int trials, uint32_t seed) {
    std::mt19937 rng(seed);
    const ScaleType st = Info<Op>::kSf;
    const int sfk = Info<Op>::kSfK, blocks = Op::K / sfk;
    double worst = 0.0;
    for (int trial = 0; trial < trials; ++trial) {
        Problem p;
        p.clear();
        p.K = Op::K;
        p.elem_bits = Op::kElemBits;
        for (int k = 0; k < Op::K; ++k) {
            for (int m = 0; m < 16; ++m) p.a[m][k] = random_code(Info<Op>::kA, rng);
            for (int n = 0; n < 8; ++n) p.b[n][k] = random_code(Info<Op>::kB, rng);
        }
        for (int m = 0; m < 16; ++m)
            for (int n = 0; n < 8; ++n) p.c[m][n] = float(int(rng() % 2001) - 1000) / 8.f;
        uint32_t sa[16][4] = {}, sb[8][4] = {};
        auto random_scale = [&]() -> uint32_t {
            if (st == SF_UE8M0) return 127 + int(rng() % 13) - 6;
            if (st == SF_UE4M3) return 0x20 + rng() % 0x30;  // 0.125 ... 15
            return 0;
        };
        for (int kb = 0; kb < blocks; ++kb) {
            for (int m = 0; m < 16; ++m) sa[m][kb] = random_scale();
            for (int n = 0; n < 8; ++n) sb[n][kb] = random_scale();
        }
        // Scale placement for selectors (0, 0): row m in thread 4*(m%8) + (m>=8), column n in thread 4*n
        for (int kb = 0; kb < blocks; ++kb) {
            for (int m = 0; m < 16; ++m) p.sfa[4 * (m % 8) + (m >= 8)] |= sa[m][kb] << (8 * kb);
            for (int n = 0; n < 8; ++n) p.sfb[4 * n] |= sb[n][kb] << (8 * kb);
        }
        run_problem(p, fn);
        for (int m = 0; m < 16; ++m)
            for (int n = 0; n < 8; ++n) {
                double ref = p.c[m][n], mag = std::fabs(ref);
                for (int k = 0; k < Op::K; ++k) {
                    const double term = dec_elem(Info<Op>::kA, p.a[m][k]) * dec_elem(Info<Op>::kB, p.b[n][k]) *
                                        dec_scale(st, sa[m][k / sfk]) * dec_scale(st, sb[n][k / sfk]);
                    ref += term;
                    mag += std::fabs(term);
                }
                const double err = std::fabs(p.d[m][n] - ref) / (mag > 0 ? mag : 1.0);
                if (!(err <= worst)) worst = err;  // also catches NaN
            }
    }
    return worst;
}

// ---------------------------------------------------------------------------------------------
// 3. Accumulator precision
// ---------------------------------------------------------------------------------------------

// C = 2^k plus one product of 1: the largest k for which D - C is still exactly 1
template <class Op> static int accumulate_bits(Launcher fn) {
    int last_exact = -1;
    for (int k = 0; k <= 26; ++k) {
        Problem p;
        p.clear();
        p.K = Op::K;
        p.elem_bits = Op::kElemBits;
        for (int m = 0; m < 16; ++m) p.a[m][0] = pow2_code(Info<Op>::kA, 0);
        for (int n = 0; n < 8; ++n) p.b[n][0] = pow2_code(Info<Op>::kB, 0);
        for (int m = 0; m < 16; ++m)
            for (int n = 0; n < 8; ++n) p.c[m][n] = std::ldexp(1.f, k);
        for (int T = 0; T < 32; ++T) p.sfa[T] = p.sfb[T] = Op::kScale;
        run_problem(p, fn);
        bool exact = true;
        for (int m = 0; m < 16; ++m)
            for (int n = 0; n < 8; ++n) exact = exact && p.d[m][n] - std::ldexp(1.0, k) == 1.0;
        if (exact && last_exact == k - 1) last_exact = k;
    }
    return last_exact;
}

// Inside one MMA: one product of 2^hi plus `count` products of 2^(hi - r) each; the largest r for
// which the small terms survive exactly. Exponents are split over A and B within their ranges.
template <class Op> static int product_sum_bits(Launcher fn, int count) {
    const Elem ea = Info<Op>::kA, eb = Info<Op>::kB;
    auto range = [](Elem e, int &lo, int &hi) {
        if (e == EL_BF16) { lo = -40; hi = 40; }
        else if (e == EL_E4M3) { lo = -9; hi = 8; }
        else { lo = -1; hi = 2; }
    };
    int alo, ahi, blo, bhi;
    range(ea, alo, ahi);
    range(eb, blo, bhi);
    const int hi = ea == EL_BF16 ? 0 : ahi + bhi;
    int last_exact = 0;
    for (int r = 1; r <= 30; ++r) {
        const int target = hi - r;  // exponent of each small product
        if (target < alo + blo) break;
        int xa = target / 2;
        xa = xa < alo ? alo : (xa > ahi ? ahi : xa);
        int xb = target - xa;
        if (xb < blo) { xb = blo; xa = target - xb; }
        if (xb > bhi) { xb = bhi; xa = target - xb; }
        Problem p;
        p.clear();
        p.K = Op::K;
        p.elem_bits = Op::kElemBits;
        for (int m = 0; m < 16; ++m) {
            p.a[m][0] = pow2_code(ea, ea == EL_BF16 ? 0 : ahi);
            for (int k = 1; k <= count; ++k) p.a[m][k] = pow2_code(ea, xa);
        }
        for (int n = 0; n < 8; ++n) {
            p.b[n][0] = pow2_code(eb, eb == EL_BF16 ? 0 : bhi);
            for (int k = 1; k <= count; ++k) p.b[n][k] = pow2_code(eb, xb);
        }
        for (int T = 0; T < 32; ++T) p.sfa[T] = p.sfb[T] = Op::kScale;
        run_problem(p, fn);
        const double expect = std::ldexp(1.0, hi) + count * std::ldexp(1.0, target);
        if (float(expect) != expect) break;  // not representable in f32 any more
        bool exact = true;
        for (int m = 0; m < 16; ++m)
            for (int n = 0; n < 8; ++n) exact = exact && p.d[m][n] == expect;
        if (exact && last_exact == r - 1) last_exact = r;
    }
    return last_exact;
}

template <class Op> static void report_numerics(Launcher fn) {
    const double err = verify_random<Op>(fn, 200, 1234);
    printf("%-20s random products: worst |d - ref| / sum|terms| = %.2e %s   accumulate: exact up to C = 2^%d"
           "   in-MMA sum: 1 small term exact down to 2^-%d, 15 small terms down to 2^-%d\n",
           Op::kName, err, err < 1e-6 ? "(ok)" : "(MISMATCH)", accumulate_bits<Op>(fn), product_sum_bits<Op>(fn, 1),
           product_sum_bits<Op>(fn, 15));
}

#define SEL(OP, BA, TA, BB, TB) Selector{BA, TA, BB, TB, &launch<OP, BA, TA, BB, TB>}

int main() {
    CK(cudaMalloc(&g_frag_d, sizeof(Frag) * 32));
    CK(cudaMalloc(&g_out_d, sizeof(float) * 32 * 4));

    printf("== 1. scale factor sources ==\n");
#define SELECTORS_1X(OP)                                                                         \
    report_selectors<OP>(                                                                        \
        {SEL(OP, 0, 0, 0, 0), SEL(OP, 1, 0, 0, 0), SEL(OP, 2, 0, 0, 0), SEL(OP, 3, 0, 0, 0),    \
         SEL(OP, 0, 1, 0, 0), SEL(OP, 3, 1, 0, 0)},                                              \
        {SEL(OP, 0, 0, 0, 0), SEL(OP, 0, 0, 1, 0), SEL(OP, 0, 0, 2, 0), SEL(OP, 0, 0, 3, 0),    \
         SEL(OP, 0, 0, 0, 1), SEL(OP, 0, 0, 0, 2), SEL(OP, 0, 0, 0, 3), SEL(OP, 0, 0, 3, 3)});
    SELECTORS_1X(MXF8_E4M3)
    report_selectors<MXF4_2X>(
        {SEL(MXF4_2X, 0, 0, 0, 0), SEL(MXF4_2X, 2, 0, 0, 0), SEL(MXF4_2X, 0, 1, 0, 0), SEL(MXF4_2X, 2, 1, 0, 0)},
        {SEL(MXF4_2X, 0, 0, 0, 0), SEL(MXF4_2X, 0, 0, 2, 0), SEL(MXF4_2X, 0, 0, 0, 1), SEL(MXF4_2X, 0, 0, 0, 2),
         SEL(MXF4_2X, 0, 0, 2, 3)});
    report_selectors<NVF4_4X>({SEL(NVF4_4X, 0, 0, 0, 0), SEL(NVF4_4X, 0, 1, 0, 0)},
                              {SEL(NVF4_4X, 0, 0, 0, 0), SEL(NVF4_4X, 0, 0, 0, 1), SEL(NVF4_4X, 0, 0, 0, 2),
                               SEL(NVF4_4X, 0, 0, 0, 3)});

    printf("== 2 / 3. numerics (selectors 0, e2m1 at bit %d of its 8-bit container) ==\n", g_e2m1_shift);
#define NUMERICS(OP) report_numerics<OP>(&launch<OP, 0, 0, 0, 0>);
    NUMERICS(BF16_F32)
    NUMERICS(E4M3_F32)
    NUMERICS(KF8_E4M3_F32)
    NUMERICS(KF8_E4M3_E2M1_F32)
    NUMERICS(KF8_E2M1_F32)
    NUMERICS(MXF8_E4M3)
    NUMERICS(MXF8_E4M3_E2M1)
    NUMERICS(MXF8_E2M1)
    NUMERICS(MXF4_2X)
    NUMERICS(NVF4_4X)

    // The container position of e2m1 in the 8-bit kinds: only one shift can match the reference
    printf("== e2m1 container position (KF8_E4M3_E2M1_F32) ==\n");
    for (int shift : {0, 1, 2, 3, 4}) {
        g_e2m1_shift = shift;
        const double err = verify_random<KF8_E4M3_E2M1_F32>(&launch<KF8_E4M3_E2M1_F32, 0, 0, 0, 0>, 20, 99);
        printf("  bits [%d:%d]: worst error %.2e %s\n", shift + 3, shift, err, err < 1e-6 ? "(match)" : "");
    }
    g_e2m1_shift = 2;

    CK(cudaFree(g_frag_d));
    CK(cudaFree(g_out_d));
    return 0;
}
