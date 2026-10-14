// Steady-state cost of gathering one tile of 32 random KV rows into shared memory while every SM of the
// GPU does the same, which is the situation of the sm120 sparse decode kernel (one 384-thread block per
// SM: 8 consumer warps and 4 producer warps, a ring of 3 tiles in shared memory).
//
// The producers fill the ring from a source buffer far larger than L2 (3 GiB, 32 random row numbers per
// tile, read one tile ahead). The consumer warps only wait until a tile has landed, optionally spin for
// `busy` clocks, and release the stage. Methods (who issues, with what):
//   A  one warp, one cp.async.bulk per row, mbarrier complete_tx            (what the kernel does now)
//   B  four warps, 8 bulk copies each
//   C  four warps (128 threads), cp.async 16 B per thread, cp.async.mbarrier.arrive.noinc
//   D  all 12 warps (384 threads) issue cp.async 16 B and consume (reference only: no producer split)
//   E  128 producer threads: ld.global.v4 into registers, st.shared, mbarrier.arrive; one tile of loads
//      in flight per thread
//   F  as E, but the loads of the next tile are issued before the stores of the current one (two tiles
//      of loads in flight per thread, twice the registers)
//   G  as C, but completion is tracked with cp.async.commit_group / wait_group and the producer arrives
//      on the barrier itself (the producer is blocked until the tile has landed)
//   H  one lane of one warp, 8 TMA cp.async.bulk.tensor.2d tile::gather4 (4 rows each) per tile; the
//      source is a 2-D tensor map [rows][R / 4] of 32-bit elements built on the host
//   2  (H2) as H, but 8 lanes issue one gather4 each
//   I  as H, but every group of 4 rows starts at a 128-byte aligned address (the groups are padded)
// Row lengths: 528 (fp8 cache), 288 (V4.1 fp4 extra cache), 272 (mxfp4 cache).
//
// Every launch is checked first: with VERIFY on, the consumers compare each byte of every tile with the
// source row. The timed launches then run without the check. The reported time per tile is the interval
// between the ready times of tile `t_begin` and tile `t_end` of the consumer warp 0 (clock64), averaged
// over the CTAs; GB/s is the sum over CTAs of 32 rows per measured tile interval (globaltimer).
//
// Build: nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a -o gather_tile gather_tile.cu
// Run:   ./gather_tile <A..I, 2 = H2> <row_bytes: 528|288|272> <busy_clk> [reps=7] [tiles_per_cta=256] [profile=0] [oob_every=0]
//        (one configuration per process, so that it can be run under the GPU lock one at a time)
//        oob_every = n > 0 (H, H2, I only): entry e of the index list with e % n == 0 gets the row coordinate -1,
//        out of the tensor; the check then expects zeros in that row

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cudaTypedefs.h>

#define CK(x)                                                                                    \
    do {                                                                                         \
        cudaError_t e_ = (x);                                                                    \
        if (e_ != cudaSuccess) {                                                                 \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);     \
            exit(1);                                                                             \
        }                                                                                        \
    } while (0)

static constexpr int TILE = 32;          // rows per tile
static constexpr int STAGES = 3;         // tiles in the ring
static constexpr int NTHREADS = 384;     // 8 consumer warps + 4 producer warps
static constexpr int SMEM_LIMIT = 101376;  // the per-block limit, which is what the decode kernel asks for
static constexpr int INDEX_PAD = 8 * TILE;  // the index prefetch may run a few tiles past the end

__device__ __forceinline__ uint32_t smem_addr(const void *p) { return uint32_t(__cvta_generic_to_shared(p)); }

__device__ __forceinline__ void mbarrier_init(uint64_t *bar, uint32_t arrivals) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_addr(bar)), "r"(arrivals) : "memory");
}
__device__ __forceinline__ void mbarrier_arrive(uint64_t *bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_addr(bar)) : "memory");
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
// Arrives on the barrier once all earlier cp.async of this thread have landed; the arrival is not added to
// the pending count by the instruction (.noinc), so the barrier is initialised with one arrival per thread.
__device__ __forceinline__ void cp_async_mbarrier_arrive_noinc(uint64_t *bar) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(smem_addr(bar)) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }
__device__ __forceinline__ uint4 ldg_v4(const void *p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}
// 4 rows of a 2-D tensor (coordinates: column `col`, rows r0..r3) into shared memory, one row after another
__device__ __forceinline__ void tma_gather4(void *dst, const CUtensorMap *map, int col, int r0, int r1, int r2, int r3, uint64_t *bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];\n"
                 ::"r"(smem_addr(dst)), "l"(map), "r"(col), "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(smem_addr(bar)) : "memory");
}
__device__ __forceinline__ unsigned long long globaltimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(t));
    return t;
}

struct Params {
    CUtensorMap map;     // H, H2, I: kv as [num_rows][R / 4] of 32-bit elements, box R / 4 x 1
    int oob_every;
    const uint8_t *kv;   // [num_rows][R]
    const int *indices;  // [ctas][tiles * TILE]
    int tiles, busy_clk, t_begin, t_end;
    int profile;  // 1: producer thread 256 also records how long it waits for free stages
    unsigned long long *stats;  // [ctas][8]: clock64 begin / end, globaltimer begin / end, smid
    unsigned int *errors;
};

// How the 16-byte chunks of a tile (TILE * R / 16 of them, chunk c = byte 16 c of the tile in shared memory
// = byte 16 (c % CPR) of row c / CPR) are dealt to NP threads: thread ptid takes chunks ptid, ptid + NP, ...
template <int R, int NP>
struct Gather {
    static constexpr int CPR = R / 16, CPT = TILE * CPR, K = (CPT + NP - 1) / NP;
    // The row numbers of the chunks of this thread, from the index list of one tile
    __device__ __forceinline__ static void load_rows(int (&rows)[K], const int *tile_idx, int ptid) {
#pragma unroll
        for (int k = 0; k < K; ++k) {
            const int c = ptid + k * NP;
            rows[k] = c < CPT ? tile_idx[c / CPR] : 0;
        }
    }
    __device__ __forceinline__ static void issue_cp(const int (&rows)[K], const uint8_t *kv, uint8_t *dst, int ptid) {
#pragma unroll
        for (int k = 0; k < K; ++k) {
            const int c = ptid + k * NP;
            if (c < CPT) cp_async_16(dst + c * 16, kv + size_t(rows[k]) * R + (c % CPR) * 16);
        }
    }
    __device__ __forceinline__ static void load_data(const int (&rows)[K], uint4 (&v)[K], const uint8_t *kv, int ptid) {
#pragma unroll
        for (int k = 0; k < K; ++k) {
            const int c = ptid + k * NP;
            if (c < CPT) v[k] = ldg_v4(kv + size_t(rows[k]) * R + (c % CPR) * 16);
        }
    }
    __device__ __forceinline__ static void store_data(const uint4 (&v)[K], uint8_t *dst, int ptid) {
#pragma unroll
        for (int k = 0; k < K; ++k) {
            const int c = ptid + k * NP;
            if (c < CPT) *reinterpret_cast<uint4 *>(dst + c * 16) = v[k];
        }
    }
};

template <int M>
__host__ __device__ constexpr uint32_t full_arrivals() {
    return M == 'A' || M == 'H' || M == '2' || M == 'I' ? 1 : M == 'B' ? 4 : M == 'D' ? NTHREADS : 128;
}
template <int M>
__host__ __device__ constexpr uint32_t empty_arrivals() {
    return M == 'D' ? NTHREADS / 32 : 8;  // one per consuming warp
}

// The byte offset of row r in a tile: rows one after another, except for I, where every group of 4 rows starts
// at a multiple of 128 bytes
template <int M, int R>
__host__ __device__ constexpr int group_stride() { return M == 'I' ? (4 * R + 127) / 128 * 128 : 4 * R; }
template <int M, int R>
__host__ __device__ constexpr int row_offset(int r) { return (r / 4) * group_stride<M, R>() + (r % 4) * R; }
template <int M, int R>
__host__ __device__ constexpr int tile_bytes() { return TILE / 4 * group_stride<M, R>(); }
// The row coordinate of entry e of the index list for the TMA methods: -1 (out of the tensor) for the rows tested
// as invalid
__device__ __forceinline__ int tma_row(const Params &p, int e, int row) { return p.oob_every > 0 && e % p.oob_every == 0 ? -1 : row; }

template <int M, int R, bool VERIFY>
__global__ void __launch_bounds__(NTHREADS, 1) tile_kernel(const __grid_constant__ Params p) {
    constexpr int TB = tile_bytes<M, R>();
    constexpr bool TMA = M == 'H' || M == '2' || M == 'I';
    extern __shared__ __align__(128) uint8_t smem[];
    uint64_t *full = reinterpret_cast<uint64_t *>(smem + STAGES * TB);
    uint64_t *empty = full + STAGES;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int *idx = p.indices + size_t(blockIdx.x) * p.tiles * TILE;

    for (int i = tid; i < STAGES * TB / 4; i += NTHREADS) reinterpret_cast<uint32_t *>(smem)[i] = 0xcdcdcdcdu;
    if (tid == 0) {
        for (int s = 0; s < STAGES; ++s) {
            mbarrier_init(&full[s], full_arrivals<M>());
            mbarrier_init(&empty[s], empty_arrivals<M>());
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();

    long long clk_begin = 0, clk_end = 0;
    unsigned long long ns_begin = 0, ns_end = 0;

    // Wait for tile t, take the timestamps (warp 0 lane 0), check the bytes, spin, release the stage
    auto consume = [&](int t, int ctid, int nc) {
        const int s = t % STAGES;
        mbarrier_wait(&full[s], (t / STAGES) & 1);
        if (!VERIFY && tid == 0) {
            if (t == p.t_begin) { clk_begin = clock64(); ns_begin = globaltimer(); }
            if (t == p.t_end) { clk_end = clock64(); ns_end = globaltimer(); }
        }
        if (VERIFY) {
            constexpr int CPR = R / 16, CPT = TILE * CPR;
            for (int c = ctid; c < CPT; c += nc) {
                const int e = t * TILE + c / CPR;
                const bool oob = TMA && p.oob_every > 0 && e % p.oob_every == 0;
                const uint4 want = oob ? make_uint4(0, 0, 0, 0) : *reinterpret_cast<const uint4 *>(p.kv + size_t(idx[e]) * R + (c % CPR) * 16);
                const uint4 got = *reinterpret_cast<const uint4 *>(smem + s * TB + row_offset<M, R>(c / CPR) + (c % CPR) * 16);
                if (want.x != got.x || want.y != got.y || want.z != got.z || want.w != got.w) atomicAdd(p.errors, 1u);
            }
        }
        if (p.busy_clk > 0) {
            const long long t0 = clock64();
            while (clock64() - t0 < p.busy_clk) {}
        }
        __syncwarp();
        if (lane == 0) mbarrier_arrive(&empty[s]);
    };
    // Producers wait for the consumers to release the stage that tile t reuses
    // (with p.profile, thread 256 records its time in this wait between tiles t_begin and t_end: the rest of
    // the producer's time is what issuing the copies costs it)
    long long prod_c0 = 0, prod_c1 = 0, prod_w0 = 0, prod_w1 = 0, prod_wait = 0;
    auto wait_stage_free = [&](int t) {
        const bool prof = p.profile && tid == 256;
        long long now = 0;
        if (prof) {
            now = clock64();
            if (t == p.t_begin) { prod_c0 = now; prod_w0 = prod_wait; }
            if (t == p.t_end) { prod_c1 = now; prod_w1 = prod_wait; }
        }
        if (t >= STAGES) mbarrier_wait(&empty[t % STAGES], (t / STAGES - 1) & 1);
        if (prof) prod_wait += clock64() - now;
    };

    if constexpr (M == 'D') {
        // Every warp issues a share of each tile and consumes; tiles t + 1 and t + 2 are in flight while t is used
        using G = Gather<R, NTHREADS>;
        int rows[G::K];
        G::load_rows(rows, idx, tid);
        G::issue_cp(rows, p.kv, smem, tid);
        cp_async_mbarrier_arrive_noinc(&full[0]);
        G::load_rows(rows, idx + TILE, tid);
        G::issue_cp(rows, p.kv, smem + TB, tid);
        cp_async_mbarrier_arrive_noinc(&full[1]);
        G::load_rows(rows, idx + 2 * TILE, tid);
        for (int t = 0; t < p.tiles; ++t) {
            if (t + 2 < p.tiles) {
                int cur[G::K];
#pragma unroll
                for (int k = 0; k < G::K; ++k) cur[k] = rows[k];
                G::load_rows(rows, idx + (t + 3) * TILE, tid);
                wait_stage_free(t + 2);
                G::issue_cp(cur, p.kv, smem + ((t + 2) % STAGES) * TB, tid);
                cp_async_mbarrier_arrive_noinc(&full[(t + 2) % STAGES]);
            }
            consume(t, tid, NTHREADS);
        }
    } else if (warp < 8) {
        for (int t = 0; t < p.tiles; ++t) consume(t, tid, 256);
    } else if constexpr (M == 'A') {
        // One warp (warp 8), lane l = row l of the tile; the other producer warps have nothing to do
        if (warp != 8) return;
        int next = idx[lane];
        for (int t = 0; t < p.tiles; ++t) {
            const int cur = next;
            next = idx[(t + 1) * TILE + lane];
            const int s = t % STAGES;
            wait_stage_free(t);
            bulk_copy(smem + s * TB + lane * R, p.kv + size_t(cur) * R, R, &full[s]);
            if (lane == 0) mbarrier_arrive_expect_tx(&full[s], TILE * R);
        }
    } else if constexpr (TMA) {
        // One warp (warp 8), lane l holds the row of entry l of the tile. H and I: lane 0 issues the 8 gather4;
        // H2: lane g issues group g
        if (warp != 8) return;
        int next = idx[lane];
        for (int t = 0; t < p.tiles; ++t) {
            const int cur = tma_row(p, t * TILE + lane, next);
            next = idx[(t + 1) * TILE + lane];
            const int s = t % STAGES;
            wait_stage_free(t);
            if (lane == 0) mbarrier_arrive_expect_tx(&full[s], TILE * R);
            if constexpr (M == '2') {
                const int g = lane % 8;
                const int r0 = __shfl_sync(0xffffffffu, cur, 4 * g), r1 = __shfl_sync(0xffffffffu, cur, 4 * g + 1);
                const int r2 = __shfl_sync(0xffffffffu, cur, 4 * g + 2), r3 = __shfl_sync(0xffffffffu, cur, 4 * g + 3);
                if (lane < 8) tma_gather4(smem + s * TB + g * group_stride<M, R>(), &p.map, 0, r0, r1, r2, r3, &full[s]);
            } else {
                int r[TILE];
#pragma unroll
                for (int i = 0; i < TILE; ++i) r[i] = __shfl_sync(0xffffffffu, cur, i);
                if (lane == 0) {
#pragma unroll
                    for (int g = 0; g < 8; ++g)
                        tma_gather4(smem + s * TB + g * group_stride<M, R>(), &p.map, 0, r[4 * g], r[4 * g + 1], r[4 * g + 2], r[4 * g + 3], &full[s]);
                }
            }
        }
    } else if constexpr (M == 'B') {
        // Four warps, 8 rows each (lanes 0-7)
        const int w = warp - 8;
        int next = lane < 8 ? idx[w * 8 + lane] : 0;
        for (int t = 0; t < p.tiles; ++t) {
            const int cur = next;
            next = lane < 8 ? idx[(t + 1) * TILE + w * 8 + lane] : 0;
            const int s = t % STAGES;
            wait_stage_free(t);
            if (lane < 8) bulk_copy(smem + s * TB + (w * 8 + lane) * R, p.kv + size_t(cur) * R, R, &full[s]);
            if (lane == 0) mbarrier_arrive_expect_tx(&full[s], 8 * R);
        }
    } else if constexpr (M == 'C') {
        using G = Gather<R, 128>;
        const int ptid = tid - 256;
        int rows[G::K];
        G::load_rows(rows, idx, ptid);
        for (int t = 0; t < p.tiles; ++t) {
            int cur[G::K];
#pragma unroll
            for (int k = 0; k < G::K; ++k) cur[k] = rows[k];
            G::load_rows(rows, idx + (t + 1) * TILE, ptid);
            const int s = t % STAGES;
            wait_stage_free(t);
            G::issue_cp(cur, p.kv, smem + s * TB, ptid);
            cp_async_mbarrier_arrive_noinc(&full[s]);
        }
    } else if constexpr (M == 'G') {
        // commit / wait_group: the producer itself waits for tile t - 1 to land before it announces it, so up
        // to two tiles are in flight per producer thread
        using G = Gather<R, 128>;
        const int ptid = tid - 256;
        int rows[G::K];
        G::load_rows(rows, idx, ptid);
        for (int t = 0; t < p.tiles; ++t) {
            int cur[G::K];
#pragma unroll
            for (int k = 0; k < G::K; ++k) cur[k] = rows[k];
            G::load_rows(rows, idx + (t + 1) * TILE, ptid);
            const int s = t % STAGES;
            wait_stage_free(t);
            G::issue_cp(cur, p.kv, smem + s * TB, ptid);
            cp_async_commit();
            if (t >= 1) {
                asm volatile("cp.async.wait_group 1;\n" ::: "memory");
                mbarrier_arrive(&full[(t - 1) % STAGES]);
            }
        }
        cp_async_wait_all();
        mbarrier_arrive(&full[(p.tiles - 1) % STAGES]);
    } else if constexpr (M == 'E') {
        using G = Gather<R, 128>;
        const int ptid = tid - 256;
        int rows[G::K];
        uint4 v[G::K];
        G::load_rows(rows, idx, ptid);
        for (int t = 0; t < p.tiles; ++t) {
            int cur[G::K];
#pragma unroll
            for (int k = 0; k < G::K; ++k) cur[k] = rows[k];
            G::load_rows(rows, idx + (t + 1) * TILE, ptid);
            const int s = t % STAGES;
            G::load_data(cur, v, p.kv, ptid);
            wait_stage_free(t);
            G::store_data(v, smem + s * TB, ptid);
            mbarrier_arrive(&full[s]);
        }
    } else if constexpr (M == 'F') {
        // The loads of tile t + 1 are issued before the stores of tile t
        using G = Gather<R, 128>;
        const int ptid = tid - 256;
        int rows_now[G::K], rows_next[G::K];
        uint4 v[G::K], v_next[G::K];
        G::load_rows(rows_now, idx, ptid);
        G::load_data(rows_now, v, p.kv, ptid);
        G::load_rows(rows_next, idx + TILE, ptid);
        for (int t = 0; t < p.tiles; ++t) {
            G::load_data(rows_next, v_next, p.kv, ptid);
            G::load_rows(rows_next, idx + (t + 2) * TILE, ptid);
            const int s = t % STAGES;
            wait_stage_free(t);
            G::store_data(v, smem + s * TB, ptid);
            mbarrier_arrive(&full[s]);
#pragma unroll
            for (int k = 0; k < G::K; ++k) v[k] = v_next[k];
        }
    }

    if (tid == 0) {
        unsigned int smid;
        asm volatile("mov.u32 %0, %%smid;\n" : "=r"(smid));
        unsigned long long *st = p.stats + size_t(blockIdx.x) * 8;
        st[0] = clk_begin; st[1] = clk_end; st[2] = ns_begin; st[3] = ns_end; st[4] = smid;
    }
    if (tid == 256 && p.profile && !VERIFY) {
        unsigned long long *st = p.stats + size_t(blockIdx.x) * 8;
        st[5] = prod_c1 - prod_c0; st[6] = prod_w1 - prod_w0;
    }
}

__device__ __forceinline__ uint64_t splitmix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}
// Every 16-byte chunk of the source gets its own pseudo-random content, so a misplaced or stale chunk shows
__global__ void fill_kernel(uint4 *buf, size_t n16) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n16; i += size_t(gridDim.x) * blockDim.x) {
        const uint64_t a = splitmix(i), b = splitmix(a ^ 0x1234567ull);
        buf[i] = make_uint4(uint32_t(a), uint32_t(a >> 32), uint32_t(b), uint32_t(b >> 32));
    }
}

template <int M, int R>
static void launch_one(bool verify, Params p, int ctas) {
    if (verify) {
        auto k = tile_kernel<M, R, true>;
        CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_LIMIT));
        CK(cudaGetLastError());  // a size above the limit fails silently here
        k<<<ctas, NTHREADS, SMEM_LIMIT>>>(p);
    } else {
        auto k = tile_kernel<M, R, false>;
        CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_LIMIT));
        CK(cudaGetLastError());
        k<<<ctas, NTHREADS, SMEM_LIMIT>>>(p);
    }
    CK(cudaGetLastError());
}

template <int M>
static void launch_rows(int row_bytes, bool verify, Params p, int ctas) {
    switch (row_bytes) {
        case 528: launch_one<M, 528>(verify, p, ctas); break;
        case 288: launch_one<M, 288>(verify, p, ctas); break;
        case 272: launch_one<M, 272>(verify, p, ctas); break;
        default: printf("unsupported row length %d\n", row_bytes); exit(1);
    }
}

static void launch(char method, int row_bytes, bool verify, Params p, int ctas) {
    switch (method) {
        case 'A': launch_rows<'A'>(row_bytes, verify, p, ctas); break;
        case 'B': launch_rows<'B'>(row_bytes, verify, p, ctas); break;
        case 'C': launch_rows<'C'>(row_bytes, verify, p, ctas); break;
        case 'D': launch_rows<'D'>(row_bytes, verify, p, ctas); break;
        case 'E': launch_rows<'E'>(row_bytes, verify, p, ctas); break;
        case 'F': launch_rows<'F'>(row_bytes, verify, p, ctas); break;
        case 'G': launch_rows<'G'>(row_bytes, verify, p, ctas); break;
        case 'H': launch_rows<'H'>(row_bytes, verify, p, ctas); break;
        case '2': launch_rows<'2'>(row_bytes, verify, p, ctas); break;
        case 'I': launch_rows<'I'>(row_bytes, verify, p, ctas); break;
        default: printf("unknown method %c\n", method); exit(1);
    }
}

static std::string smi(const char *query) {
    std::string cmd = std::string("nvidia-smi --query-gpu=") + query + " --format=csv,noheader,nounits 2>/dev/null";
    FILE *f = popen(cmd.c_str(), "r");
    char buf[128] = {0};
    if (f) { if (!fgets(buf, sizeof(buf), f)) buf[0] = 0; pclose(f); }
    std::string s(buf);
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
    return s;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        printf("usage: %s <A..I, 2 = H2> <row_bytes 528|288|272> <busy_clk> [reps=7] [tiles_per_cta=256] [profile=0] [oob_every=0]\n", argv[0]);
        return 1;
    }
    const char method = argv[1][0];
    const int row_bytes = atoi(argv[2]), busy = atoi(argv[3]);
    const int reps = argc > 4 ? atoi(argv[4]) : 7;
    const int tiles = argc > 5 ? atoi(argv[5]) : 256;
    const int profile = argc > 6 ? atoi(argv[6]) : 0;
    const int oob_every = argc > 7 ? atoi(argv[7]) : 0;
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    const int ctas = prop.multiProcessorCount;
    const int t_begin = 8, t_end = tiles - 8;
    const int warmups = 2;

    const size_t buffer_bytes = size_t(3) << 30;  // 31x the L2
    const int num_rows = int(buffer_bytes / row_bytes);
    uint8_t *kv;
    CK(cudaMalloc(&kv, buffer_bytes));
    fill_kernel<<<1024, 256>>>(reinterpret_cast<uint4 *>(kv), buffer_bytes / 16);
    CK(cudaGetLastError());

    // One index set for the check and one per launch, all random rows of the whole buffer
    const int sets = 1 + warmups + reps;
    const size_t per_set = size_t(ctas) * tiles * TILE;
    std::vector<int> host(per_set * sets + INDEX_PAD);
    std::mt19937_64 rng(2026);
    for (size_t i = 0; i < per_set * sets; ++i) host[i] = int(rng() % num_rows);
    for (int i = 0; i < INDEX_PAD; ++i) host[per_set * sets + i] = 0;
    int *indices_d;
    CK(cudaMalloc(&indices_d, host.size() * sizeof(int)));
    CK(cudaMemcpy(indices_d, host.data(), host.size() * sizeof(int), cudaMemcpyHostToDevice));
    unsigned long long *stats;
    unsigned int *errors;
    CK(cudaMalloc(&stats, size_t(ctas) * 8 * sizeof(unsigned long long)));
    CK(cudaMalloc(&errors, sizeof(unsigned int)));
    CK(cudaMemset(errors, 0, sizeof(unsigned int)));

    Params p{};
    p.oob_every = oob_every;
    p.kv = kv; p.indices = indices_d; p.tiles = tiles; p.busy_clk = busy; p.t_begin = t_begin; p.t_end = t_end;
    p.profile = profile; p.stats = stats; p.errors = errors;
    if (method == 'H' || method == '2' || method == 'I') {
        PFN_cuTensorMapEncodeTiled_v12000 encode = nullptr;
        cudaDriverEntryPointQueryResult q;
        CK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", reinterpret_cast<void **>(&encode), 12000, cudaEnableDefault, &q));
        if (q != cudaDriverEntryPointSuccess || encode == nullptr) { printf("no cuTensorMapEncodeTiled\n"); return 1; }
        const cuuint64_t dims[2] = {cuuint64_t(row_bytes / 4), cuuint64_t(num_rows)};
        const cuuint64_t strides[1] = {cuuint64_t(row_bytes)};
        const cuuint32_t box[2] = {cuuint32_t(row_bytes / 4), 1};
        const cuuint32_t estr[2] = {1, 1};
        const CUresult r = encode(&p.map, CU_TENSOR_MAP_DATA_TYPE_UINT32, 2, kv, dims, strides, box, estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                  CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if (r != CUDA_SUCCESS) { printf("cuTensorMapEncodeTiled failed: %d\n", int(r)); return 1; }
    }

    // Correctness: every tile of every CTA, byte by byte
    p.indices = indices_d;
    launch(method, row_bytes, true, p, ctas);
    CK(cudaDeviceSynchronize());
    unsigned int bad = 0;
    CK(cudaMemcpy(&bad, errors, sizeof(bad), cudaMemcpyDeviceToHost));

    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    std::vector<double> clk_per_tile, gbps, ghz, evt_us, spread_lo, spread_hi, prod_busy;
    int distinct_sms = 0;
    std::vector<unsigned long long> st(size_t(ctas) * 8);
    for (int rep = 0; rep < warmups + reps; ++rep) {
        p.indices = indices_d + per_set * (1 + rep);
        CK(cudaEventRecord(e0));
        launch(method, row_bytes, false, p, ctas);
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        if (rep < warmups) continue;
        float ms;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        CK(cudaMemcpy(st.data(), stats, st.size() * sizeof(st[0]), cudaMemcpyDeviceToHost));
        double sum_clk = 0, sum_rate = 0, sum_ns = 0, sum_cyc = 0, lo = 1e18, hi = 0, sum_prod = 0;
        std::vector<char> seen(1024, 0);
        const int n = t_end - t_begin;
        for (int b = 0; b < ctas; ++b) {
            const double clk = double(st[b * 8 + 1] - st[b * 8 + 0]), ns = double(st[b * 8 + 3] - st[b * 8 + 2]);
            sum_clk += clk / n; sum_cyc += clk; sum_ns += ns;
            sum_rate += double(TILE) * row_bytes / (ns / n);  // bytes per ns = GB/s
            lo = std::min(lo, clk / n); hi = std::max(hi, clk / n);
            sum_prod += double(st[b * 8 + 5] - st[b * 8 + 6]) / n;  // producer clocks per tile not spent waiting for a stage
            if (st[b * 8 + 4] < 1024) seen[st[b * 8 + 4]] = 1;
        }
        distinct_sms = 0;
        for (char c : seen) distinct_sms += c;
        clk_per_tile.push_back(sum_clk / ctas);
        gbps.push_back(sum_rate);
        ghz.push_back(sum_cyc / sum_ns);
        evt_us.push_back(ms * 1e3);
        spread_lo.push_back(lo);
        spread_hi.push_back(hi);
        prod_busy.push_back(sum_prod / ctas);
    }
    const std::string temp = smi("temperature.gpu"), sm_mhz = smi("clocks.sm");
    auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
    auto mn = [](const std::vector<double> &v) { return *std::min_element(v.begin(), v.end()); };
    auto mx = [](const std::vector<double> &v) { return *std::max_element(v.begin(), v.end()); };
    const double evt_gbps = double(ctas) * tiles * TILE * row_bytes / (med(evt_us) * 1e3);
    printf("RESULT method=%c oob_every=%d row=%d busy=%d ctas=%d tiles=%d reps=%d verify=%s errors=%u distinct_sms=%d | "
           "clk_per_tile=%.0f (rep min %.0f max %.0f; CTA min %.0f max %.0f) GBps=%.0f (rep min %.0f max %.0f) "
           "ratio_vs_1530=%.3f sm_ghz=%.3f event_us=%.1f event_GBps=%.0f temp_C=%s smi_sm_mhz=%s profile=%d prod_busy_clk_per_tile=%.0f\n",
           method, oob_every, row_bytes, busy, ctas, tiles, reps, bad == 0 ? "OK" : "FAIL", bad, distinct_sms,
           med(clk_per_tile), mn(clk_per_tile), mx(clk_per_tile), med(spread_lo), med(spread_hi), med(gbps),
           mn(gbps), mx(gbps), med(gbps) / 1530.0, med(ghz), med(evt_us), evt_gbps, temp.c_str(), sm_mhz.c_str(), profile, med(prod_busy));
    CK(cudaFree(kv));
    CK(cudaFree(indices_d));
    CK(cudaFree(stats));
    CK(cudaFree(errors));
    return bad == 0 ? 0 : 2;
}
