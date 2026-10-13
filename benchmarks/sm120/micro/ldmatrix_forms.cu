// The `ldmatrix` forms that feed 8-bit / 4-bit MMA operands on sm120:
//   m8n8.x4.b16                      rows of 16 bytes, as-is            (K as the B operand of QK)
//   m16n16.x1/x2.trans.b8            16 x 16 bytes, transposed          (V as the B operand of PV)
//   m8n16.x4.b8x16.b4x16_p64         16 packed 4-bit -> 16 bytes        (packed fp4 rows)
//   m16n16.x2.trans.b8x16.b4x16_p64  the same, transposed
// Part 1 prints where every output byte comes from. Part 2 measures the throughput of each form
// alone and chained into two block-scaled MMAs, for the row strides of the KV cache formats.
//
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -o ldmatrix_forms ldmatrix_forms.cu
// Run:   ./ldmatrix_forms [warps_per_block=8] [blocks_per_sm=1] [all]     (all = every row stride)
//        ./ldmatrix_forms unaligned <form 0..3> <byte offset>

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

// kRowsPerMatrix lanes give the row addresses of one matrix; kMatrices matrices per instruction.
// kPacked: the source row is 16 packed 4-bit elements (8 bytes) instead of 16 bytes.
#define DEFINE_LDMATRIX(NAME, REGS, ROWS, MATRICES, TRANS, PACKED, PTX, OUTS)                    \
    struct NAME {                                                                                \
        static constexpr int kRegs = REGS, kRowsPerMatrix = ROWS, kMatrices = MATRICES;          \
        static constexpr bool kTrans = TRANS, kPacked = PACKED;                                  \
        static constexpr const char *kName = #NAME;                                              \
        __device__ __forceinline__ static void load(uint32_t (&r)[4], uint32_t addr) {           \
            asm volatile(PTX OUTS : "r"(addr));                                                  \
        }                                                                                        \
    };
#define LD_OUT1 : "=r"(r[0])
#define LD_OUT2 : "=r"(r[0]), "=r"(r[1])
#define LD_OUT4 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])

DEFINE_LDMATRIX(M8N8_X4_B16, 4, 8, 4, false, false,
                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n", LD_OUT4)
DEFINE_LDMATRIX(M16N16_X1_TRANS_B8, 2, 16, 1, true, false,
                "ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {%0,%1}, [%2];\n", LD_OUT2)
DEFINE_LDMATRIX(M16N16_X2_TRANS_B8, 4, 16, 2, true, false,
                "ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3}, [%4];\n", LD_OUT4)
DEFINE_LDMATRIX(M8N16_X4_P64, 4, 8, 4, false, true,
                "ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64 {%0,%1,%2,%3}, [%4];\n", LD_OUT4)
DEFINE_LDMATRIX(M16N16_X2_TRANS_P64, 4, 16, 2, true, true,
                "ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8x16.b4x16_p64 {%0,%1,%2,%3}, [%4];\n", LD_OUT4)

__device__ __forceinline__ uint64_t globaltimer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// ---------------------------------------------------------------------------------------------
// 1. Semantics
// ---------------------------------------------------------------------------------------------

static constexpr int kProbeBytes = 4096;
static constexpr int kProbeStride = 64;  // lane l reads the row at l * kProbeStride + offset
static int g_probe_offset = 0;            // byte offset of every row address (alignment check)

template <class Form>
__global__ void probe_kernel(const uint8_t *src, uint32_t *out, int offset) {
    __shared__ __align__(16) uint8_t smem[kProbeBytes];
    for (int i = threadIdx.x; i < kProbeBytes; i += blockDim.x) smem[i] = src[i];
    __syncthreads();
    uint32_t r[4] = {0, 0, 0, 0};
    Form::load(r, uint32_t(__cvta_generic_to_shared(smem + threadIdx.x * kProbeStride + offset)));
    for (int i = 0; i < 4; ++i) out[threadIdx.x * 4 + i] = r[i];
}

// Loads `src` into shared memory, runs the form once and returns the 4 registers of every thread
template <class Form> static void run_probe(uint8_t *src_d, uint32_t *out_d, const uint8_t *src, uint32_t *out) {
    CK(cudaMemcpy(src_d, src, kProbeBytes, cudaMemcpyHostToDevice));
    probe_kernel<Form><<<1, 32>>>(src_d, out_d, g_probe_offset);
    CK(cudaMemcpy(out, out_d, sizeof(uint32_t) * 32 * 4, cudaMemcpyDeviceToHost));
}

// The packed forms expand every 4-bit element to one byte: the bit position of the element in it
template <class Form> static int packed_shift(uint8_t *src_d, uint32_t *out_d) {
    static uint8_t src[kProbeBytes];
    uint32_t out[32 * 4];
    memset(src, 0xff, sizeof(src));
    run_probe<Form>(src_d, out_d, src, out);
    for (int s = 0; s <= 4; ++s)
        if ((out[0] & 255u) == (15u << s)) return s;
    return -1;
}

// Source position (in elements: bytes, or 4-bit elements for the packed forms) of every output
// byte, recovered by loading patterns that spell the position digit by digit.
template <class Form> static void probe(uint8_t *src_d, uint32_t *out_d, int shift, int pos[32][4][4]) {
    const int digit_bits = Form::kPacked ? 4 : 8;
    const int elems = Form::kPacked ? kProbeBytes * 2 : kProbeBytes;
    const int passes = Form::kPacked ? 4 : 2;
    static uint8_t src[kProbeBytes];
    uint32_t out[32 * 4];
    memset(pos, 0, sizeof(int) * 32 * 4 * 4);
    for (int pass = 0; pass < passes; ++pass) {
        memset(src, 0, sizeof(src));
        for (int e = 0; e < elems; ++e) {
            const int digit = (e >> (digit_bits * pass)) & ((1 << digit_bits) - 1);
            if (Form::kPacked) src[e / 2] |= digit << (4 * (e & 1));
            else src[e] = uint8_t(digit);
        }
        run_probe<Form>(src_d, out_d, src, out);
        for (int T = 0; T < 32; ++T)
            for (int r = 0; r < Form::kRegs; ++r)
                for (int j = 0; j < 4; ++j) {
                    const int byte = (out[T * 4 + r] >> (8 * j)) & 255;
                    const int digit = Form::kPacked ? (byte >> shift) & 15 : byte;
                    pos[T][r][j] |= digit << (digit_bits * pass);
                }
    }
}

template <class Form> static void report_semantics(uint8_t *src_d, uint32_t *out_d, int offset = 0) {
    int pos[32][4][4];
    g_probe_offset = offset;
    const int shift = Form::kPacked ? packed_shift<Form>(src_d, out_d) : 0;
    probe<Form>(src_d, out_d, shift < 0 ? 0 : shift, pos);
    const int elems_per_row_step = Form::kPacked ? kProbeStride * 2 : kProbeStride;
    const int offset_elems = Form::kPacked ? offset * 2 : offset;
    for (int T = 0; T < 32; ++T)
        for (int r = 0; r < Form::kRegs; ++r)
            for (int j = 0; j < 4; ++j) pos[T][r][j] -= offset_elems;
    printf("%s (%d regs, %d row addresses from lanes 0..%d), row addresses = 16n + %d", Form::kName, Form::kRegs,
           Form::kRowsPerMatrix * Form::kMatrices, Form::kRowsPerMatrix * Form::kMatrices - 1, offset);
    if (Form::kPacked) printf(" [columns count 4-bit elements; each lands at bits [%d:%d] of its byte]", shift + 3, shift);
    printf("\n");
    // The source (row lane, column) of every byte for a few threads
    for (int T : {0, 1, 5}) {
        printf("  thread %2d:", T);
        for (int r = 0; r < Form::kRegs; ++r) {
            printf("  r%d =", r);
            for (int j = 0; j < 4; ++j) printf(" (%d,%d)", pos[T][r][j] / elems_per_row_step, pos[T][r][j] % elems_per_row_step);
        }
        printf("\n");
    }
    // Fit against the two expected layouts, g = T / 4, t = T % 4:
    //   as-is:      reg i, byte j      <- row lane 8i + g, column 4t + j
    //   transposed: reg 2i + h, byte j <- row lane 16i + 4t + j, column 8h + g
    bool fit = true;
    for (int T = 0; T < 32; ++T)
        for (int r = 0; r < Form::kRegs; ++r)
            for (int j = 0; j < 4; ++j) {
                const int g = T / 4, t = T % 4;
                const int lane = Form::kTrans ? 16 * (r / 2) + 4 * t + j : 8 * r + g;
                const int col = Form::kTrans ? 8 * (r % 2) + g : 4 * t + j;
                fit = fit && pos[T][r][j] == lane * elems_per_row_step + col;
            }
    printf("  %s\n", !fit ? "does NOT follow the expected layout (see the samples above)"
                          : Form::kTrans ? "fit: reg 2i+h byte j <- row lane 16i + 4t + j, column 8h + g"
                                         : "fit: reg i byte j <- row lane 8i + g, column 4t + j");
}

// ---------------------------------------------------------------------------------------------
// 2. Throughput
// ---------------------------------------------------------------------------------------------

// Every step loads one instruction worth of fragments from a 32-row tile whose rows are `stride`
// bytes apart, walking along the row.
//   ALONE:     only the loads
//   CHAIN:     load, then two block-scaled MMAs on what it loaded
//   LOOKAHEAD: the same work, but the load of step i + 1 is issued before the MMAs of step i
//   LATENCY:   only the loads, each address depending on the previous result
//   CHAIN8 / LOOKAHEAD8: as CHAIN / LOOKAHEAD with eight MMAs per load: the two B fragments of
//              the load against four different A fragments, which is PV with one warp owning a
//              range of V columns for four m tiles
enum Mode { ALONE, CHAIN, LOOKAHEAD, LATENCY, CHAIN8, LOOKAHEAD8 };
static const char *const kModeNames[] = {"alone", "load, 2 mma", "lookahead 2", "latency", "load, 8 mma", "lookahead 8"};

template <class Form, int MODE>
// At most 512 threads per block, so that a thread may use 128 registers (16 accumulators need 64)
__global__ void __launch_bounds__(512) chain_kernel(int iters, int stride, float *sink, unsigned long long *timing) {
    extern __shared__ __align__(16) uint8_t smem[];
    const int tile_bytes = 32 * stride;
    for (int i = threadIdx.x; i < tile_bytes; i += blockDim.x) smem[i] = uint8_t((i * 37 + 11) & 0x3f);
    __syncthreads();
    const int lane = threadIdx.x % 32;
    // Row (token) and column offset supplied by this lane, see the layouts in part 1
    const int row = Form::kTrans ? lane : lane % 8;
    // Every row address must be a multiple of 16 (anything else traps), also for the packed forms,
    // which then use the first 8 bytes of each 16
    const int col = Form::kTrans ? 0 : (lane / 8) * 16;
    const uint32_t base = uint32_t(__cvta_generic_to_shared(smem)) + row * stride + col;
    const int col_step = Form::kTrans ? 16 : 64;
    const int col_mask = 255;

    constexpr bool kLookahead = MODE == LOOKAHEAD || MODE == LOOKAHEAD8;
    constexpr int kTilesA = (MODE == CHAIN8 || MODE == LOOKAHEAD8) ? 4 : (MODE == CHAIN || MODE == LOOKAHEAD) ? 1 : 0;
    uint32_t a[4][4];
    for (int m = 0; m < 4; ++m)
        for (int i = 0; i < 4; ++i) a[m][i] = 0x38383838u + 0x01010101u * ((threadIdx.x + i + 3 * m) & 7);
    AccF32 acc[16];
    for (int j = 0; j < 16; ++j) acc_zero(acc[j]);
    uint32_t x = 0;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        timing[0] = clock64();
        timing[1] = globaltimer();
    }
    uint32_t next[4] = {0, 0, 0, 0};
    if (kLookahead) Form::load(next, base);
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            uint32_t r[4] = {0, 0, 0, 0};
            const uint32_t addr = base + (((it * 2 + u + kLookahead) * col_step) & col_mask);
            if (kLookahead) {
                for (int i = 0; i < 4; ++i) r[i] = next[i];
                Form::load(next, addr);
            } else if (MODE == LATENCY) {
                Form::load(r, addr + (x & 16));  // the smem bytes are < 0x40, so this is 0 or 16
            } else {
                Form::load(r, addr);
            }
            if (kTilesA > 0) {
                const uint32_t b0[2] = {r[0], r[Form::kTrans ? 2 : 1]};
                const uint32_t b1[2] = {r[Form::kTrans ? 1 : 2], r[3]};
#pragma unroll
                for (int m = 0; m < kTilesA; ++m) {
                    MXF8_E4M3::run(acc[8 * u + 2 * m], a[m], b0, 0x7f7f7f7fu, 0x7f7f7f7fu);
                    MXF8_E4M3::run(acc[8 * u + 2 * m + 1], a[m], b1, 0x7f7f7f7fu, 0x7f7f7f7fu);
                }
            } else if (MODE == LATENCY) {
                x = r[0];
            } else {
                x ^= r[0] ^ r[1] ^ r[2] ^ r[3];
            }
        }
    }
    x ^= next[0];
    float s = float(x & 1);
    for (int j = 0; j < 16; ++j) s += acc_sum(acc[j]);
    __syncthreads();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        timing[2] = clock64();
        timing[3] = globaltimer();
    }
    // `x` depends on the loaded data, so the loads stay alive without the MMAs too
    if (s == 1234.5f || x == 0xdeadbeefu) sink[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

template <class Form, int MODE>
static void measure(int warps_per_block, int blocks_per_sm, int stride, int num_sms, float *sink, unsigned long long *timing_d) {
    // The latency chain runs as a single warp on the whole device
    const int blocks = MODE == LATENCY ? 1 : num_sms * blocks_per_sm;
    const int threads = MODE == LATENCY ? 32 : warps_per_block * 32;
    const size_t smem_bytes = 32 * stride;
    auto kernel = chain_kernel<Form, MODE>;
    CK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem_bytes)));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    const int iters = 1 << 18;
    float best = 1e30f, ms;
    unsigned long long timing[4];
    double clock_ghz = 0;
    for (int rep = 0; rep < 4; ++rep) {
        CK(cudaEventRecord(e0));
        kernel<<<blocks, threads, smem_bytes>>>(iters, stride, sink, timing_d);
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        CK(cudaEventElapsedTime(&ms, e0, e1));
        CK(cudaMemcpy(timing, timing_d, sizeof(timing), cudaMemcpyDeviceToHost));
        if (rep > 0 && ms < best) {
            best = ms;
            clock_ghz = double(timing[2] - timing[0]) / double(timing[3] - timing[1]);
        }
    }
    const double clk = best * 1e-3 * clock_ghz * 1e9;
    if (MODE == LATENCY) {
        printf("  %-22s stride %3d  %-11s  %5.1f clk\n", Form::kName, stride, kModeNames[MODE], clk / (2.0 * iters));
    } else {
        // Loads per SMSP: 2 per iteration and warp, a quarter of the SM's warps per SMSP
        const double loads_per_smsp = 2.0 * iters * warps_per_block * blocks_per_sm / 4.0;
        printf("  %-22s stride %3d  %-11s  %7.2f ms  %5.1f clk per load per SMSP", Form::kName, stride,
               kModeNames[MODE], best, clk / loads_per_smsp);
        const int mmas_per_load = (MODE == CHAIN8 || MODE == LOOKAHEAD8) ? 8 : 2;
        if (MODE != ALONE) printf("  = %5.1f clk per mma (16.0 = tensor peak)", clk / loads_per_smsp / mmas_per_load);
        printf("\n");
    }
    CK(cudaEventDestroy(e0));
    CK(cudaEventDestroy(e1));
}

int main(int argc, char **argv) {
    const int warps_per_block = argc > 1 ? atoi(argv[1]) : 8;
    const int blocks_per_sm = argc > 2 ? atoi(argv[2]) : 1;
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    uint8_t *src_d;
    uint32_t *out_d;
    float *sink;
    unsigned long long *timing_d;
    CK(cudaMalloc(&src_d, kProbeBytes));
    CK(cudaMalloc(&out_d, sizeof(uint32_t) * 32 * 4));
    CK(cudaMalloc(&sink, sizeof(float) * 1024 * 1024));
    CK(cudaMalloc(&timing_d, sizeof(unsigned long long) * 4));

    // `./ldmatrix_forms unaligned <form> <offset>`: one probe with row addresses 16n + offset, in
    // its own process because a misaligned address traps and takes the context down
    if (argc > 3 && strcmp(argv[1], "unaligned") == 0) {
        const int form = atoi(argv[2]), offset = atoi(argv[3]);
        if (form == 0) report_semantics<M8N8_X4_B16>(src_d, out_d, offset);
        if (form == 1) report_semantics<M16N16_X2_TRANS_B8>(src_d, out_d, offset);
        if (form == 2) report_semantics<M8N16_X4_P64>(src_d, out_d, offset);
        if (form == 3) report_semantics<M16N16_X2_TRANS_P64>(src_d, out_d, offset);
        return 0;
    }

    printf("== 1. where every output byte comes from: (row lane, column) ==\n");
    report_semantics<M8N8_X4_B16>(src_d, out_d);
    report_semantics<M16N16_X1_TRANS_B8>(src_d, out_d);
    report_semantics<M16N16_X2_TRANS_B8>(src_d, out_d);
    report_semantics<M8N16_X4_P64>(src_d, out_d);
    report_semantics<M16N16_X2_TRANS_P64>(src_d, out_d);

    printf("== 2. throughput, %d warps per SM (%d x %d), SM clock from clock64 / globaltimer ==\n",
           warps_per_block * blocks_per_sm, warps_per_block, blocks_per_sm);
    // Stride 512 puts every row on the same banks; 528 / 272 / 304 are the KV formats (304 = 288 padded)
    const bool all_strides = argc > 3 && strcmp(argv[3], "all") == 0;
    for (int stride : {512, 528, 544, 272, 288, 304}) {
        if (!all_strides && stride != 528 && stride != 544) continue;
#define RUN(FORM, MODE) \
    measure<FORM, MODE>(warps_per_block, blocks_per_sm, stride, prop.multiProcessorCount, sink, timing_d);
#define ALL_MODES(FORM) RUN(FORM, ALONE) RUN(FORM, LATENCY) RUN(FORM, CHAIN) RUN(FORM, LOOKAHEAD) RUN(FORM, CHAIN8) RUN(FORM, LOOKAHEAD8)
        ALL_MODES(M8N8_X4_B16)
        RUN(M16N16_X1_TRANS_B8, ALONE)
        RUN(M16N16_X1_TRANS_B8, LATENCY)
        ALL_MODES(M16N16_X2_TRANS_B8)
        ALL_MODES(M8N16_X4_P64)
        ALL_MODES(M16N16_X2_TRANS_P64)
#undef ALL_MODES
#undef RUN
    }
    CK(cudaFree(src_d));
    CK(cudaFree(out_d));
    CK(cudaFree(sink));
    CK(cudaFree(timing_d));
    return 0;
}
