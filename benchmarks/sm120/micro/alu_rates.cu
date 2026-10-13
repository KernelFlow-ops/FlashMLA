// Throughput of the non-tensor instructions the sm120 decode kernel leans on: integer / bit ops
// (exponent rebase, fp4 unpacking), cvt between the quantized formats and f16 / f32 (dequantize,
// quantize Q and P), and the MUFU functions of the softmax. Eight independent chains per thread
// with every warp of the SM busy, so the numbers are issue rates, not latencies.
//
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -o alu_rates alu_rates.cu
// Run:   ./alu_rates [warps_per_sm=16]

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define CK(x)                                                                                    \
    do {                                                                                         \
        cudaError_t e_ = (x);                                                                    \
        if (e_ != cudaSuccess) {                                                                 \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);     \
            exit(1);                                                                             \
        }                                                                                        \
    } while (0)

__device__ __forceinline__ uint64_t globaltimer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ uint32_t bits_of(uint32_t v) { return v; }
__device__ __forceinline__ uint32_t bits_of(uint16_t v) { return v; }
__device__ __forceinline__ uint32_t bits_of(float v) { return __float_as_uint(v); }

// Float / cvt ops need valid operands, so they come with glue. One chain step is:
// `in` = a valid operand made from the chain value x, r = op(in), x = r ^ k,
// where k is a per-thread value the compiler cannot see through. With WITH_OP = false the op is
// left out (r = the bits of `in`), which gives the cost of the glue alone; the difference is the
// cost of the measured instruction.
#define DEFINE_OP(NAME, IN_T, PREP, ...)                                                         \
    struct NAME {                                                                                \
        static constexpr const char *kName = #NAME;                                              \
        static constexpr bool kPure = false;                                                     \
        template <bool WITH_OP>                                                                  \
        __device__ __forceinline__ static uint32_t step(uint32_t x, uint32_t k) {                \
            const IN_T in = PREP;                                                                \
            uint32_t r;                                                                          \
            if (WITH_OP) { __VA_ARGS__ } else { r = bits_of(in); }                               \
            return r ^ k;                                                                        \
        }                                                                                        \
    };

#define UNIT_F32(x) __uint_as_float(((x) & 0x007fffffu) | 0x3f000000u)  // in [0.5, 1)

// Integer / bit ops as chains of the bare instruction, x = op(x, k, k2). Inline asm, so that the
// compiler neither folds the chain nor adds glue; there is no glue-only twin for them.
#define DEFINE_PURE(NAME, PTX)                                                                   \
    struct NAME {                                                                                \
        static constexpr const char *kName = #NAME;                                              \
        static constexpr bool kPure = true;                                                      \
        template <bool WITH_OP>                                                                  \
        __device__ __forceinline__ static uint32_t step(uint32_t x, uint32_t k) {                \
            uint32_t r;                                                                          \
            const uint32_t k2 = k ^ 0x4520u;                                                     \
            asm volatile(PTX : "=r"(r) : "r"(x), "r"(k), "r"(k2));                               \
            return r;                                                                            \
        }                                                                                        \
    };

DEFINE_PURE(IADD, "add.u32 %0, %1, %2;")
DEFINE_PURE(ISUB, "sub.u32 %0, %1, %2;")
DEFINE_PURE(AND, "and.b32 %0, %1, %2;")
DEFINE_PURE(XOR, "xor.b32 %0, %1, %2;")
DEFINE_PURE(LOP3, "lop3.b32 %0, %1, %2, %3, 0x96;")
DEFINE_PURE(SHR_IMM, "shr.u32 %0, %1, 3;")
DEFINE_PURE(SHL_REG, "shl.b32 %0, %1, %2;")
DEFINE_PURE(PRMT, "prmt.b32 %0, %1, %2, %3;")
DEFINE_PURE(IMUL, "mul.lo.u32 %0, %1, %2;")
DEFINE_PURE(UMAX, "max.u32 %0, %1, %2;")
DEFINE_PURE(POPC, "popc.b32 %0, %1;")
DEFINE_PURE(CLZ, "clz.b32 %0, %1;")
// Bare float chains (the values saturate or go NaN after a while, which does not change the
// timing) and a cvt round trip of two instructions, to cross-check the glue subtraction below
DEFINE_PURE(PURE_FMUL, "mul.f32 %0, %1, %2;")
DEFINE_PURE(PURE_FFMA, "fma.rn.f32 %0, %1, %2, %3;")
DEFINE_PURE(PURE_EX2, "ex2.approx.ftz.f32 %0, %1;")
DEFINE_PURE(PURE_CVT_E4M3_F16_PAIR, "{ .reg .b16 t; cvt.rn.satfinite.e4m3x2.f16x2 t, %1; cvt.rn.f16x2.e4m3x2 %0, t; }")

DEFINE_OP(FMUL, float, UNIT_F32(x), r = __float_as_uint(in * 1.37f);)
DEFINE_OP(FFMA, float, UNIT_F32(x), r = __float_as_uint(fmaf(in, 1.37f, 0.25f));)
DEFINE_OP(EX2, float, UNIT_F32(x), float f; asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(f) : "f"(in)); r = __float_as_uint(f);)
DEFINE_OP(LG2, float, UNIT_F32(x), float f; asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(f) : "f"(in)); r = __float_as_uint(f);)
DEFINE_OP(RCP, float, UNIT_F32(x), float f; asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(f) : "f"(in)); r = __float_as_uint(f);)
// e4m3x2 -> f16x2 (NaN codes masked out)
DEFINE_OP(CVT_E4M3X2_TO_F16X2, uint16_t, uint16_t(x & 0x3f3fu), asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(in));)
// f16x2 -> e4m3x2
DEFINE_OP(CVT_F16X2_TO_E4M3X2, uint32_t, (x & 0x03ff03ffu) | 0x3c003c00u,
          uint16_t h; asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(h) : "r"(in)); r = h;)
// (f32, f32) -> e4m3x2
DEFINE_OP(CVT_F32_F32_TO_E4M3X2, float, UNIT_F32(x),
          uint16_t h; asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(h) : "f"(in), "f"(in)); r = h;)
// (f32, f32) -> e2m1x2
DEFINE_OP(CVT_F32_F32_TO_E2M1X2, float, UNIT_F32(x),
          asm volatile("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %1, %2; cvt.u32.u8 %0, t; }" : "=r"(r) : "f"(in), "f"(in));)
// e2m1x2 -> f16x2
DEFINE_OP(CVT_E2M1X2_TO_F16X2, uint32_t, x & 255u,
          asm volatile("{ .reg .b8 t; cvt.u8.u32 t, %1; cvt.rn.f16x2.e2m1x2 %0, t; }" : "=r"(r) : "r"(in));)
// f16 -> f32
DEFINE_OP(CVT_F16_TO_F32, uint16_t, uint16_t((x & 0x03ffu) | 0x3c00u),
          float f; asm volatile("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(in)); r = __float_as_uint(f);)
// (f32, f32) -> bf16x2
DEFINE_OP(CVT_F32_F32_TO_BF16X2, float, UNIT_F32(x), asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(in), "f"(in));)
// (f32, f32) -> ue8m0x2, exponents rounded up
DEFINE_OP(CVT_F32_F32_TO_UE8M0X2, float, UNIT_F32(x),
          uint16_t h; asm volatile("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(h) : "f"(in), "f"(in)); r = h;)

static constexpr int kChains = 8;

template <class Op, bool WITH_OP>
__global__ void __launch_bounds__(512) rate_kernel(int iters, const uint32_t *seed, uint32_t *sink, unsigned long long *timing) {
    uint32_t x[kChains];
    const uint32_t k = seed[threadIdx.x & 255] | 1u;
    for (int j = 0; j < kChains; ++j) x[j] = seed[(threadIdx.x + 17 * j) & 255];
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        timing[0] = clock64();
        timing[1] = globaltimer();
    }
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int j = 0; j < kChains; ++j) x[j] = Op::template step<WITH_OP>(x[j], k);
    }
    uint32_t s = 0;
    for (int j = 0; j < kChains; ++j) s ^= x[j];
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        timing[2] = clock64();
        timing[3] = globaltimer();
    }
    if (s == seed[0]) sink[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

// Clocks per chain step of one warp while every warp of the SM runs the same loop
template <class Op, bool WITH_OP>
static double clk_per_step(int warps_per_sm, int num_sms, const uint32_t *seed_d, uint32_t *sink, unsigned long long *timing_d) {
    const int iters = 1 << 16;
    unsigned long long timing[4];
    double best = 1e30;
    for (int rep = 0; rep < 4; ++rep) {
        rate_kernel<Op, WITH_OP><<<num_sms, warps_per_sm * 32>>>(iters, seed_d, sink, timing_d);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(timing, timing_d, sizeof(timing), cudaMemcpyDeviceToHost));
        const double clk = double(timing[2] - timing[0]);
        if (rep > 0 && clk < best) best = clk;
    }
    return best / (double(iters) * kChains);
}

int main(int argc, char **argv) {
    const int warps_per_sm = argc > 1 ? atoi(argv[1]) : 16;
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    uint32_t seed[256], *seed_d, *sink;
    unsigned long long *timing_d;
    for (int i = 0; i < 256; ++i) seed[i] = uint32_t(i) * 2654435761u + 0x9e3779b9u;
    CK(cudaMalloc(&seed_d, sizeof(seed)));
    CK(cudaMalloc(&sink, sizeof(uint32_t) * 170 * 1024));
    CK(cudaMalloc(&timing_d, sizeof(unsigned long long) * 4));
    CK(cudaMemcpy(seed_d, seed, sizeof(seed), cudaMemcpyHostToDevice));

    printf("%s, %d warps per SM (one block per SM), %d chains per thread\n", prop.name, warps_per_sm, kChains);
    printf("%-24s %14s %14s %12s %24s\n", "op", "clk/step", "glue only", "op alone", "warp instr / clk / SM");
#define RUN(OP)                                                                                  \
    {                                                                                            \
        const double with = clk_per_step<OP, true>(warps_per_sm, prop.multiProcessorCount, seed_d, sink, timing_d);  \
        const double glue = OP::kPure ? 0.0 : clk_per_step<OP, false>(warps_per_sm, prop.multiProcessorCount, seed_d, sink, timing_d); \
        printf("%-24s %14.2f %14.2f %12.2f %24.2f\n", OP::kName, with, glue, with - glue, warps_per_sm / (with - glue)); \
    }
    RUN(IADD) RUN(ISUB) RUN(AND) RUN(XOR) RUN(LOP3) RUN(SHR_IMM) RUN(SHL_REG) RUN(PRMT) RUN(IMUL) RUN(UMAX) RUN(POPC) RUN(CLZ)
    RUN(PURE_FMUL) RUN(PURE_FFMA) RUN(PURE_EX2) RUN(PURE_CVT_E4M3_F16_PAIR)
    RUN(FMUL) RUN(FFMA) RUN(EX2) RUN(LG2) RUN(RCP)
    RUN(CVT_E4M3X2_TO_F16X2) RUN(CVT_F16X2_TO_E4M3X2) RUN(CVT_F32_F32_TO_E4M3X2) RUN(CVT_F32_F32_TO_E2M1X2)
    RUN(CVT_E2M1X2_TO_F16X2) RUN(CVT_F16_TO_F32) RUN(CVT_F32_F32_TO_BF16X2) RUN(CVT_F32_F32_TO_UE8M0X2)
#undef RUN
    CK(cudaFree(seed_d));
    CK(cudaFree(sink));
    CK(cudaFree(timing_d));
    return 0;
}
