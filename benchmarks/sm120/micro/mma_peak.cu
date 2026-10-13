// Peak throughput and latency of the sm120 `mma.sync` kinds that matter for MLA decoding:
//   m16n8k16 bf16 / f16, m16n8k32 e4m3 (legacy, kind::f8f6f4, kind::mxf8f6f4 block-scaled),
//   m16n8k64 e2m1 (kind::mxf4, kind::mxf4nvf4 block-scaled).
// Operands live in registers, so this measures the tensor pipe alone; see `mma_ldmatrix.cu` for the
// ldmatrix -> mma chain. The SM clock is derived from clock64 against globaltimer in the same run.
//
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -o mma_peak mma_peak.cu
//        (`-arch=sm_120a` emits `.target sm_120`, which rejects the f8f6f4 / block-scaled kinds)
// Run:   ./mma_peak [warps_per_block=8] [blocks_per_sm=2] [ilp=4] [target_ms=40] [lat]

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

__device__ __forceinline__ uint64_t globaltimer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}

// Random operand bits that are finite in every format: bf16/f16 with |x| < 2, e4m3 with bit 6
// cleared (|x| <= 1.875, never NaN), e2m1 nibbles are always finite.
template <class Op>
__device__ __forceinline__ uint32_t operand_bits(uint32_t r) {
    if constexpr (Op::K == 16) return (r & 0x803f803fu) | 0x3f803f80u;
    else return r & 0xbfbfbfbfu;
}

// timing[0..1] = clock64 / globaltimer at the start, timing[2..3] at the end (block 0, thread 0)
template <class Op, int ILP>
__global__ void __launch_bounds__(1024) throughput_kernel(int iters, uint32_t seed, float *sink,
                                                          unsigned long long *timing) {
    uint32_t a[4], b[2];
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < 4; ++i) a[i] = operand_bits<Op>(hash32(seed + tid * 8 + i));
    for (int i = 0; i < 2; ++i) b[i] = operand_bits<Op>(hash32(seed + tid * 8 + 4 + i));
    const uint32_t sfa = Op::kScale, sfb = Op::kScale;
    typename Op::Acc acc[ILP];
    for (int j = 0; j < ILP; ++j) acc_zero(acc[j]);

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        timing[0] = clock64();
        timing[1] = globaltimer();
    }
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int j = 0; j < ILP; ++j) Op::run(acc[j], a, b, sfa, sfb);
    }
    float s = 0.f;
    for (int j = 0; j < ILP; ++j) s += acc_sum(acc[j]);
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        timing[2] = clock64();
        timing[3] = globaltimer();
    }
    if (s == 1234.5f) sink[tid] = s;  // never true in practice; keeps the chain alive
}

struct Config {
    int warps_per_block, blocks_per_sm, ilp;
    float target_ms;
    int num_sms;
};

template <class Op, int ILP>
void measure(const char *name, const Config &cfg, float *sink, unsigned long long *timing_d) {
    const int threads = cfg.warps_per_block * 32;
    const int blocks = cfg.num_sms * cfg.blocks_per_sm;
    auto kernel = throughput_kernel<Op, ILP>;
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));

    // Calibrate the iteration count to the target duration
    int iters = 256;
    float ms = 0.f;
    for (int attempt = 0; attempt < 6; ++attempt) {
        CK(cudaEventRecord(e0));
        kernel<<<blocks, threads>>>(iters, 1u, sink, timing_d);
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        CK(cudaEventElapsedTime(&ms, e0, e1));
        if (ms > cfg.target_ms * 0.5f) break;
        iters = (int)(iters * (cfg.target_ms / (ms > 0.01f ? ms : 0.01f)));
        if (iters > (1 << 24)) iters = 1 << 24;
    }
    float best = 1e30f;
    unsigned long long timing[4];
    double best_clock_ghz = 0.0;
    for (int rep = 0; rep < 5; ++rep) {
        CK(cudaEventRecord(e0));
        kernel<<<blocks, threads>>>(iters, 2u + rep, sink, timing_d);
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        CK(cudaEventElapsedTime(&ms, e0, e1));
        CK(cudaMemcpy(timing, timing_d, sizeof(timing), cudaMemcpyDeviceToHost));
        if (ms < best) {
            best = ms;
            best_clock_ghz = double(timing[2] - timing[0]) / double(timing[3] - timing[1]);
        }
    }
    const double flops = 2.0 * Op::M * Op::N * Op::K * double(iters) * ILP * (double(blocks) * cfg.warps_per_block);
    const double tflops = flops / (best * 1e-3) / 1e12;
    const double flop_per_clk_sm = flops / (best * 1e-3) / (best_clock_ghz * 1e9) / cfg.num_sms;
    printf("%-22s ilp=%d warps/SM=%2d  %8.3f ms  %8.1f TFLOPS  clk %.3f GHz  %7.0f FLOP/clk/SM  %6.1f clk/mma/SMSP\n",
           name, ILP, cfg.warps_per_block * cfg.blocks_per_sm, best, tflops, best_clock_ghz, flop_per_clk_sm,
           2.0 * Op::M * Op::N * Op::K / (flop_per_clk_sm / 4.0));
    CK(cudaEventDestroy(e0));
    CK(cudaEventDestroy(e1));
}

// Dependent-chain latency: a single warp, one accumulator
template <class Op>
void measure_latency(const char *name, float *sink, unsigned long long *timing_d) {
    const int iters = 1 << 16;
    throughput_kernel<Op, 1><<<1, 32>>>(iters, 7u, sink, timing_d);
    CK(cudaDeviceSynchronize());
    unsigned long long timing[4];
    CK(cudaMemcpy(timing, timing_d, sizeof(timing), cudaMemcpyDeviceToHost));
    printf("%-22s latency %6.1f clk\n", name, double(timing[2] - timing[0]) / iters);
}

template <class Op>
void run_one(const char *name, const Config &cfg, float *sink, unsigned long long *timing_d) {
    switch (cfg.ilp) {
        case 1: measure<Op, 1>(name, cfg, sink, timing_d); break;
        case 2: measure<Op, 2>(name, cfg, sink, timing_d); break;
        case 4: measure<Op, 4>(name, cfg, sink, timing_d); break;
        case 8: measure<Op, 8>(name, cfg, sink, timing_d); break;
        default: printf("unsupported ilp %d\n", cfg.ilp); exit(1);
    }
}

int main(int argc, char **argv) {
    Config cfg;
    cfg.warps_per_block = argc > 1 ? atoi(argv[1]) : 8;
    cfg.blocks_per_sm = argc > 2 ? atoi(argv[2]) : 2;
    cfg.ilp = argc > 3 ? atoi(argv[3]) : 4;
    cfg.target_ms = argc > 4 ? atof(argv[4]) : 40.f;
    const bool latency = argc > 5 && strcmp(argv[5], "lat") == 0;
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    cfg.num_sms = prop.multiProcessorCount;
    printf("%s, %d SMs, cc %d.%d\n", prop.name, prop.multiProcessorCount, prop.major, prop.minor);

    float *sink;
    unsigned long long *timing_d;
    CK(cudaMalloc(&sink, sizeof(float) * 1024 * 1024 * 4));
    CK(cudaMalloc(&timing_d, sizeof(unsigned long long) * 4));

#define RUN(OP) \
    if (latency) measure_latency<OP>(#OP, sink, timing_d); else run_one<OP>(#OP, cfg, sink, timing_d);
    RUN(BF16_F32)
    RUN(F16_F32)
    RUN(F16_F16)
    RUN(E4M3_F32)
    RUN(E4M3_F16)
    RUN(KF8_E4M3_F32)
    RUN(KF8_E4M3_F16)
    RUN(KF8_E4M3_E2M1_F32)
    RUN(KF8_E2M1_F32)
    RUN(MXF8_E4M3)
    RUN(MXF8_E4M3_E2M1)
    RUN(MXF8_E2M1)
    RUN(MXF4_2X)
    RUN(NVF4_4X)
#undef RUN
    CK(cudaFree(sink));
    CK(cudaFree(timing_d));
    return 0;
}
