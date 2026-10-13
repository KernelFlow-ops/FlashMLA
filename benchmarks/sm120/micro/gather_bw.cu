// Bandwidth of gathering quantized KV rows from global memory into shared memory, which is what
// bounds sparse MLA decoding with 64 heads. Each CTA gathers `tokens` rows of `row_bytes` (528 for
// the fp8 cache, 288 / 272 for the fp4 ones) selected by an index list, tile by tile, keeping
// `stages` tiles in flight, and does nothing else with them.
//   BULK:  one `cp.async.bulk` per row, issued by a single warp, completion through an mbarrier
//   CP16:  `cp.async` of 16 bytes per thread, completion through commit / wait groups
// Index lists are random rows of a buffer far larger than L2, or sequential rows (the streaming
// ceiling). Every repetition shifts the rows so that nothing is reused from L2.
//
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -o gather_bw gather_bw.cu
// Run:   ./gather_bw [ctas=256] [tokens=640]

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <cuda_runtime.h>

#define CK(x)                                                                                    \
    do {                                                                                         \
        cudaError_t e_ = (x);                                                                    \
        if (e_ != cudaSuccess) {                                                                 \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);     \
            exit(1);                                                                             \
        }                                                                                        \
    } while (0)

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
__device__ __forceinline__ void cp_async_16(void *dst, const void *src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr(dst)), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

static constexpr int kMaxStages = 4;

struct Params {
    const uint8_t *kv;    // [num_rows][row_bytes]
    const int *indices;   // [ctas][tokens]
    uint32_t *sink;       // [ctas]
    int num_rows, row_bytes, tokens, tile, row_shift;
};

// Shared memory: stages * tile * row_bytes of rows, then the barriers. The index list is read
// from global memory (it is small, sequential, and in the real kernel it is staged the same way).
template <int STAGES>
__global__ void __launch_bounds__(512) bulk_kernel(Params p) {
    extern __shared__ __align__(16) uint8_t smem[];
    const int tile_bytes = p.tile * p.row_bytes;
    uint64_t *bars = reinterpret_cast<uint64_t *>(smem + STAGES * tile_bytes);
    if (threadIdx.x == 0)
        for (int s = 0; s < STAGES; ++s) mbarrier_init(&bars[s], 1);
    __syncthreads();
    if (threadIdx.x >= 32) return;  // a single producer warp; the others stand for the consumers
    const int *indices = p.indices + size_t(blockIdx.x) * p.tokens;
    const int num_tiles = p.tokens / p.tile;
    uint32_t x = 0;
    for (int t = 0; t < num_tiles + STAGES; ++t) {
        const int s = t % STAGES;
        if (t >= STAGES) {
            // The tile that used this stage `STAGES` tiles ago has to land before it is overwritten
            mbarrier_wait(&bars[s], ((t - STAGES) / STAGES) & 1);
            x ^= *reinterpret_cast<const uint32_t *>(smem + s * tile_bytes + threadIdx.x * 16);
        }
        if (t < num_tiles) {
            if (threadIdx.x == 0) mbarrier_arrive_expect_tx(&bars[s], tile_bytes);
            for (int r = threadIdx.x; r < p.tile; r += 32) {
                int row = indices[t * p.tile + r] + p.row_shift;
                row = row >= p.num_rows ? row - p.num_rows : row;
                bulk_copy(smem + s * tile_bytes + r * p.row_bytes, p.kv + size_t(row) * p.row_bytes, p.row_bytes, &bars[s]);
            }
        }
    }
    if (threadIdx.x == 0) p.sink[blockIdx.x] = x;
}

template <int STAGES>
__global__ void __launch_bounds__(512) cp16_kernel(Params p) {
    extern __shared__ __align__(16) uint8_t smem[];
    const int tile_bytes = p.tile * p.row_bytes;
    const int chunks_per_row = p.row_bytes / 16, chunks_per_tile = p.tile * chunks_per_row;
    const int *indices = p.indices + size_t(blockIdx.x) * p.tokens;
    const int num_tiles = p.tokens / p.tile;
    // c / chunks_per_row without a division: exact for c < 2^20 / chunks_per_row
    const uint32_t inv_chunks_per_row = (1u << 20) / chunks_per_row + 1;
    uint32_t x = 0;
    for (int t = 0; t < num_tiles; ++t) {
        const int s = t % STAGES;
        for (int c = threadIdx.x; c < chunks_per_tile; c += blockDim.x) {
            const int r = int((uint32_t(c) * inv_chunks_per_row) >> 20), off = (c - r * chunks_per_row) * 16;
            int row = indices[t * p.tile + r] + p.row_shift;
            row = row >= p.num_rows ? row - p.num_rows : row;
            cp_async_16(smem + s * tile_bytes + c * 16, p.kv + size_t(row) * p.row_bytes + off);
        }
        cp_async_commit();
        // At most STAGES - 1 tiles stay in flight, so the stage of the next tile is free
        cp_async_wait<STAGES - 1>();
        __syncthreads();
        if (t >= STAGES - 1) x ^= *reinterpret_cast<const uint32_t *>(smem + ((t + 1) % STAGES) * tile_bytes + (threadIdx.x % 32) * 16);
    }
    cp_async_wait<0>();
    __syncthreads();
    if (threadIdx.x == 0) p.sink[blockIdx.x] = x;
}

enum Method { BULK, CP16 };

struct Result {
    double us, gbps;
};

static Result run(Method method, int stages, int threads, Params p, int ctas, int reps) {
    const size_t smem_bytes = size_t(stages) * p.tile * p.row_bytes + kMaxStages * sizeof(uint64_t);
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    std::vector<float> times;
    for (int rep = 0; rep < reps + 2; ++rep) {
        p.row_shift = int((size_t(rep) * 1000003u * 97u) % size_t(p.num_rows));
        CK(cudaEventRecord(e0));
#define LAUNCH(KERNEL, S)                                                                        \
    case S:                                                                                      \
        CK(cudaFuncSetAttribute(KERNEL<S>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem_bytes))); \
        KERNEL<S><<<ctas, threads, smem_bytes>>>(p);                                             \
        break;
        if (method == BULK) {
            switch (stages) { LAUNCH(bulk_kernel, 1) LAUNCH(bulk_kernel, 2) LAUNCH(bulk_kernel, 3) LAUNCH(bulk_kernel, 4) }
        } else {
            switch (stages) { LAUNCH(cp16_kernel, 1) LAUNCH(cp16_kernel, 2) LAUNCH(cp16_kernel, 3) LAUNCH(cp16_kernel, 4) }
        }
#undef LAUNCH
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        float ms;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        if (rep >= 2) times.push_back(ms);
    }
    CK(cudaEventDestroy(e0));
    CK(cudaEventDestroy(e1));
    std::sort(times.begin(), times.end());
    const double ms = times[times.size() / 2];
    const double bytes = double(ctas) * (p.tokens / p.tile * p.tile) * p.row_bytes;
    return {ms * 1e3, bytes / (ms * 1e-3) / 1e9};
}

int main(int argc, char **argv) {
    const int ctas = argc > 1 ? atoi(argv[1]) : 256;
    const int tokens = argc > 2 ? atoi(argv[2]) : 640;
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("%s, %d SMs; %d CTAs x %d rows per CTA; median of 15 launches after 2 warm-ups\n", prop.name,
           prop.multiProcessorCount, ctas, tokens);

    // 3 GiB of rows: 31x the 96 MB L2
    const size_t buffer_bytes = size_t(3) << 30;
    uint8_t *kv;
    CK(cudaMalloc(&kv, buffer_bytes));
    CK(cudaMemset(kv, 0x5a, buffer_bytes));
    int *indices_d;
    uint32_t *sink;
    CK(cudaMalloc(&indices_d, sizeof(int) * size_t(ctas) * tokens));
    CK(cudaMalloc(&sink, sizeof(uint32_t) * ctas));
    std::mt19937_64 rng(2026);

    for (int row_bytes : {528, 288, 272}) {
        const int num_rows = int(buffer_bytes / row_bytes);
        for (int random_rows = 1; random_rows >= 0; --random_rows) {
            std::vector<int> indices(size_t(ctas) * tokens);
            for (size_t i = 0; i < indices.size(); ++i) indices[i] = random_rows ? int(rng() % num_rows) : int(i);
            CK(cudaMemcpy(indices_d, indices.data(), sizeof(int) * indices.size(), cudaMemcpyHostToDevice));
            printf("== %d B rows, %s; %.1f MB per launch ==\n", row_bytes, random_rows ? "random rows" : "sequential rows",
                   double(ctas) * tokens * row_bytes / 1e6);
            for (int tile : {32, 64}) {
                for (int stages = 1; stages <= kMaxStages; ++stages) {
                    const size_t in_flight = size_t(stages) * tile * row_bytes;
                    if (in_flight + 64 > 101376) continue;  // the per-block shared memory limit
                    Params p{kv, indices_d, sink, num_rows, row_bytes, tokens, tile, 0};
                    const Result bulk = run(BULK, stages, 64, p, ctas, 15);
                    const Result cp256 = run(CP16, stages, 256, p, ctas, 15);
                    const Result cp384 = run(CP16, stages, 384, p, ctas, 15);
                    printf("  tile %2d x %d stages (%5.1f KB in flight per CTA):  bulk %7.1f us %6.0f GB/s   cp.async16 x256 thr %7.1f us %6.0f GB/s"
                           "   x384 thr %7.1f us %6.0f GB/s\n",
                           tile, stages, in_flight / 1024.0, bulk.us, bulk.gbps, cp256.us, cp256.gbps, cp384.us, cp384.gbps);
                }
            }
        }
    }
    CK(cudaFree(kv));
    CK(cudaFree(indices_d));
    CK(cudaFree(sink));
    return 0;
}
