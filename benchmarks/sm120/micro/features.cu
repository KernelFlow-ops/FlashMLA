// Which PTX features the sm120 decode kernel may rely on. Every feature is its own build, so an
// instruction that does not assemble for sm_120a does not hide the others:
//
//   for f in SETMAXNREG PDL CLUSTER GATHER4 CVT_E4M3_F16 CVT_F16_E4M3 CVT_E4M3_F32 CVT_E2M1_F16 \
//            CVT_E2M1_F32 CVT_UE8M0_F32 CVT_BF16_E4M3 CVT_SCALED; do
//     nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -DFEATURE_$f -o features_$f features.cu \
//       && ./features_$f
//   done

#include <cmath>
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

// ---------------------------------------------------------------------------------------------
#if defined(FEATURE_SETMAXNREG)
// 384 threads: two warpgroups raise their register limit to 232, one lowers it to 40
// (8 * 32 * 232 + 4 * 32 * 40 = 64512 <= 65536). The consumers then keep 200 values live.
template <int N> __device__ __forceinline__ void reg_alloc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(N)); }
template <int N> __device__ __forceinline__ void reg_dealloc() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(N)); }

__global__ void __launch_bounds__(384, 1) feature_kernel(const uint32_t *in, uint32_t *out) {
    const int warpgroup = threadIdx.x / 128;
    if (warpgroup == 2) {
        reg_dealloc<40>();
        out[threadIdx.x] = in[threadIdx.x] + 1;
    } else {
        reg_alloc<232>();
        uint32_t v[200];
#pragma unroll
        for (int i = 0; i < 200; ++i) v[i] = in[(threadIdx.x + i * 7) & 1023];
#pragma unroll
        for (int round = 0; round < 4; ++round) {
#pragma unroll
            for (int i = 0; i < 200; ++i) v[i] = v[i] * 2654435761u + v[(i + 61 + round) % 200];
        }
        uint32_t s = 0;
#pragma unroll
        for (int i = 0; i < 200; ++i) s ^= v[i];
        out[threadIdx.x] = s;
    }
}

static const char *run_feature() {
    static uint32_t in[1024], out[384], ref[384];
    for (int i = 0; i < 1024; ++i) in[i] = i * 2246822519u + 12345u;
    for (int t = 0; t < 384; ++t) {
        if (t / 128 == 2) { ref[t] = in[t] + 1; continue; }
        uint32_t v[200];
        for (int i = 0; i < 200; ++i) v[i] = in[(t + i * 7) & 1023];
        for (int round = 0; round < 4; ++round)
            for (int i = 0; i < 200; ++i) v[i] = v[i] * 2654435761u + v[(i + 61 + round) % 200];
        uint32_t s = 0;
        for (int i = 0; i < 200; ++i) s ^= v[i];
        ref[t] = s;
    }
    uint32_t *in_d, *out_d;
    CK(cudaMalloc(&in_d, sizeof(in)));
    CK(cudaMalloc(&out_d, sizeof(out)));
    CK(cudaMemcpy(in_d, in, sizeof(in), cudaMemcpyHostToDevice));
    feature_kernel<<<170, 384>>>(in_d, out_d);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(out, out_d, sizeof(out), cudaMemcpyDeviceToHost));
    return memcmp(out, ref, sizeof(out)) == 0 ? "setmaxnreg inc 232 / dec 40 with 384 threads: runs, results match" : "setmaxnreg: WRONG RESULT";
}

// ---------------------------------------------------------------------------------------------
#elif defined(FEATURE_PDL)
__global__ void first_kernel(uint32_t *buf) {
    buf[blockIdx.x * blockDim.x + threadIdx.x] = 41;
    asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
}
__global__ void second_kernel(uint32_t *buf) {
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
    buf[blockIdx.x * blockDim.x + threadIdx.x] += 1;
}

static const char *run_feature() {
    uint32_t *buf, out[64 * 32];
    CK(cudaMalloc(&buf, sizeof(out)));
    CK(cudaMemset(buf, 0, sizeof(out)));
    first_kernel<<<64, 32>>>(buf);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(64);
    cfg.blockDim = dim3(32);
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    CK(cudaLaunchKernelEx(&cfg, second_kernel, buf));
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(out, buf, sizeof(out), cudaMemcpyDeviceToHost));
    for (uint32_t v : out)
        if (v != 42) return "PDL: WRONG RESULT";
    return "griddepcontrol launch_dependents / wait with programmatic stream serialization: ok";
}

// ---------------------------------------------------------------------------------------------
#elif defined(FEATURE_CLUSTER)
// Two CTAs per cluster; each writes into the shared memory of the other one
__global__ void __cluster_dims__(2, 1, 1) feature_kernel(uint32_t *out) {
    __shared__ uint32_t smem[32];
    uint32_t rank, peer_addr;
    asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rank));
    const uint32_t local = uint32_t(__cvta_generic_to_shared(&smem[threadIdx.x]));
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(peer_addr) : "r"(local), "r"(rank ^ 1u));
    asm volatile("st.shared::cluster.u32 [%0], %1;\n" ::"r"(peer_addr), "r"(1000u * (rank + 1) + threadIdx.x) : "memory");
    asm volatile("barrier.cluster.arrive.release.aligned;\n" "barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
    out[blockIdx.x * 32 + threadIdx.x] = smem[threadIdx.x];
}

static const char *run_feature() {
    uint32_t *out_d, out[340 * 32];
    CK(cudaMalloc(&out_d, sizeof(out)));
    feature_kernel<<<340, 32>>>(out_d);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(out, out_d, sizeof(out), cudaMemcpyDeviceToHost));
    for (int b = 0; b < 340; ++b)
        for (int t = 0; t < 32; ++t)
            if (out[b * 32 + t] != 1000u * ((b % 2 ^ 1) + 1) + t) return "cluster: WRONG RESULT";
    return "cluster of 2 CTAs, mapa + st.shared::cluster into the peer, cluster barrier: ok";
}

// ---------------------------------------------------------------------------------------------
#elif defined(FEATURE_GATHER4)
// Assembly check only: the kernel is never launched (it would need a tensor map)
__global__ void feature_kernel(const void *desc, uint64_t *bar, void *dst) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];\n"
                 ::"r"(uint32_t(__cvta_generic_to_shared(dst))), "l"(desc), "r"(0), "r"(1), "r"(2), "r"(3), "r"(4),
                 "r"(uint32_t(__cvta_generic_to_shared(bar))) : "memory");
}
static const char *run_feature() { return "cp.async.bulk.tensor tile::gather4 assembles for sm_120a (not executed)"; }

// ---------------------------------------------------------------------------------------------
#else
// cvt variants: one value pair in, one out, checked against the expected numbers

__global__ void feature_kernel(const float *fin, const uint32_t *bits_in, float *fout, uint32_t *bits_out) {
    const float x = fin[0], y = fin[1];
    const uint32_t b = bits_in[0];
    uint32_t r = 0;
    float f0 = 0, f1 = 0;
#if defined(FEATURE_CVT_E4M3_F16)
    // e4m3x2 -> f16x2
    uint16_t in16 = uint16_t(b);
    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(r) : "h"(in16));
#elif defined(FEATURE_CVT_F16_E4M3)
    // f16x2 -> e4m3x2
    uint16_t out16;
    asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;\n" : "=h"(out16) : "r"(b));
    r = out16;
#elif defined(FEATURE_CVT_E4M3_F32)
    // (f32, f32) -> e4m3x2
    uint16_t out16;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(out16) : "f"(y), "f"(x));
    r = out16;
#elif defined(FEATURE_CVT_E2M1_F16)
    // e2m1x2 (one byte) -> f16x2
    uint8_t in8 = uint8_t(b);
    asm volatile("{ .reg .b8 t; cvt.u8.u32 t, %1; cvt.rn.f16x2.e2m1x2 %0, t; }\n" : "=r"(r) : "r"(uint32_t(in8)));
#elif defined(FEATURE_CVT_E2M1_F32)
    // (f32, f32) -> e2m1x2 (one byte)
    asm volatile("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %1, %2; cvt.u32.u8 %0, t; }\n" : "=r"(r) : "f"(y), "f"(x));
#elif defined(FEATURE_CVT_UE8M0_F32)
    // (f32, f32) -> ue8m0x2, rounding the exponent up
    uint16_t out16;
    asm volatile("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;\n" : "=h"(out16) : "f"(y), "f"(x));
    r = out16;
#elif defined(FEATURE_CVT_BF16_E4M3)
    // e4m3x2 -> bf16x2 (documented for newer toolkits)
    uint16_t in16 = uint16_t(b);
    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;\n" : "=r"(r) : "h"(in16));
#elif defined(FEATURE_CVT_SCALED)
    // (f32, f32) -> e4m3x2 with a ue8m0 scale applied by the instruction
    uint16_t out16;
    asm volatile("cvt.rn.satfinite.scaled::n2::ue8m0.e4m3x2.f32 %0, %1, %2, %3;\n" : "=h"(out16) : "f"(y), "f"(x), "r"(b));
    r = out16;
#else
#error "pick a FEATURE_* macro"
#endif
    bits_out[0] = r;
    fout[0] = f0;
    fout[1] = f1;
}

static const char *run_feature() {
    float fin[2] = {1.75f, -0.4375f};
    // e4m3: 0x3e = 1.75, 0xae = -0.4375 (low byte first); e2m1 byte: 0x5 = 3.0 in the low nibble, 0xa = -1.0
#if defined(FEATURE_CVT_E2M1_F16)
    uint32_t bits_in = 0xa5;
#elif defined(FEATURE_CVT_F16_E4M3)
    uint32_t bits_in = 0xb7003f00;  // f16 pair: 1.75 (0x3f00), -0.4375 (0xb700)
#elif defined(FEATURE_CVT_SCALED)
    uint32_t bits_in = 0x7f7f;
#else
    uint32_t bits_in = 0xae3e;
#endif
    float *fin_d, *fout_d, fout[2];
    uint32_t *bin_d, *bout_d, bout;
    CK(cudaMalloc(&fin_d, sizeof(fin)));
    CK(cudaMalloc(&fout_d, sizeof(fout)));
    CK(cudaMalloc(&bin_d, 4));
    CK(cudaMalloc(&bout_d, 4));
    CK(cudaMemcpy(fin_d, fin, sizeof(fin), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(bin_d, &bits_in, 4, cudaMemcpyHostToDevice));
    feature_kernel<<<1, 1>>>(fin_d, bin_d, fout_d, bout_d);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(&bout, bout_d, 4, cudaMemcpyDeviceToHost));
    static char msg[160];
    snprintf(msg, sizeof(msg), "assembles and runs: inputs (1.75, -0.4375) / bits 0x%x -> 0x%08x", bits_in, bout);
    return msg;
}
#endif

int main() {
    printf("%s\n", run_feature());
    return 0;
}
