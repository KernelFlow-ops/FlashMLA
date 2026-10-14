#pragma once

#include <cstdint>

// Thin wrappers over the PTX instructions the sm120 kernels are built on. sm120 has neither wgmma nor
// tcgen05, so the kernels use warp-level mma.sync, ldmatrix and cp.async.bulk directly.
namespace sm120::ptx {

__device__ __forceinline__ uint32_t smem_addr(const void *ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// Register budget of the calling warp; every warp of a warpgroup has to call it (as cutlass::arch::warpgroup_reg_alloc)
template<int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" :: "n"(NUM_REGS));
}

template<int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_dec() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" :: "n"(NUM_REGS));
}

// `num_threads` threads (whole warps) meet at barrier `id`; id 0 belongs to __syncthreads
__device__ __forceinline__ void named_barrier_sync(uint32_t id, uint32_t num_threads) {
    asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(num_threads) : "memory");
}

__device__ __forceinline__ void mbarrier_init(uint64_t *bar, uint32_t num_arrivals) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(smem_addr(bar)), "r"(num_arrivals) : "memory");
}

__device__ __forceinline__ void fence_mbarrier_init() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive(uint64_t *bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(smem_addr(bar)) : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t *bar, uint32_t num_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(smem_addr(bar)), "r"(num_bytes) : "memory");
}

// Wait for the phase of parity `phase`, i.e. for the k-th completion of the barrier with phase = k & 1
__device__ __forceinline__ void mbarrier_wait(uint64_t *bar, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "@p bra DONE;\n"
        "bra LAB_WAIT;\n"
        "DONE:\n"
        "}\n" :: "r"(smem_addr(bar)), "r"(phase) : "memory");
}

// Global -> shared copy of `num_bytes` bytes, completing `num_bytes` of the transaction count of `bar`.
// Both addresses and the size have to be multiples of 16.
__device__ __forceinline__ void bulk_copy_g2s(void *dst, const void *src, uint32_t num_bytes, uint64_t *bar) {
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
                 :: "r"(smem_addr(dst)), "l"(src), "r"(num_bytes), "r"(smem_addr(bar)) : "memory");
}

// Four 8x8 matrices of 16-bit elements. Lane l supplies the address of row l % 8 of matrix l / 8 (16 bytes,
// 16-byte aligned). Thread (g = lane / 4, t = lane % 4) gets, in register i, the elements (g, 2t) and (g, 2t+1)
// of matrix i: the A fragment of m16n8k16 for matrices {rows 0-7, rows 8-15} x {k 0-7, k 8-15}, or two B
// fragments for matrices holding n as rows.
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&frag)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(frag[0]), "=r"(frag[1]), "=r"(frag[2]), "=r"(frag[3]) : "r"(addr));
}

// Same, transposed: thread (g, t) gets the elements (2t, g) and (2t+1, g) of matrix i, i.e. B fragments
// of m16n8k16 for matrices holding k as rows.
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&frag)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(frag[0]), "=r"(frag[1]), "=r"(frag[2]), "=r"(frag[3]) : "r"(addr));
}

// Two 16x16 matrices of bytes, transposed. Lane l supplies the address of row l % 16 of matrix l / 16 (16 bytes,
// 16-byte aligned). Thread (g, t) gets, in register 2*i + h, the bytes (4t .. 4t+3, 8*h + g) of matrix i: for
// matrices holding k as rows, the registers {h, 2 + h} are the B fragment of m16n8k32 for the columns [8*h, +8).
__device__ __forceinline__ void ldmatrix_x2_trans_b8(uint32_t (&frag)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(frag[0]), "=r"(frag[1]), "=r"(frag[2]), "=r"(frag[3]) : "r"(addr));
}

// d[16x8] += a[16x16] * b[16x8], bf16 inputs and an fp32 accumulator. Thread (g, t) holds
// a[i] = A(g + 8*(i&1), 8*(i>>1) + 2t .. +1), b0 / b1 = B(2t .. +1 / 8 + 2t .. +1, g), d[i] = D(g + 8*(i>>1), 2t + (i&1))
__device__ __forceinline__ void mma_m16n8k16_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// d[16x8] += (a[16x32] * sa) * (b[32x8] * sb): e4m3 inputs, one power-of-two scale per row of A and per column
// of B (ue8m0: the byte x means 2^(x - 127)), and an fp32 accumulator. It runs at four times the rate of the
// bf16 MMA. One byte per element, low byte first: thread (g, t) holds
// a[i] = A(g + 8*(i&1), 16*(i>>1) + 4t .. +3), b0 / b1 = B(4t .. +3 / 16 + 4t .. +3, g), and d as for bf16.
// The scales are bytes of the `sfa` / `sfb` registers of the warp: row m takes byte BYTE_A of thread
// 4*(m%8) + 2*THREAD_A + (m >= 8), column n takes byte BYTE_B of thread 4*n + THREAD_B. THREAD_A is 0 or 1.
template<int BYTE_A, int THREAD_A, int BYTE_B, int THREAD_B>
__device__ __forceinline__ void mma_m16n8k32_mxf8(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb) {
    static constexpr uint16_t byte_a = BYTE_A, thread_a = THREAD_A, byte_b = BYTE_B, thread_b = THREAD_B;
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
                   "r"(sfa), "h"(byte_a), "h"(thread_a), "r"(sfb), "h"(byte_b), "h"(thread_b));
}

// d[16x8] += (a[16x64] * sa) * (b[64x8] * sb): e2m1 inputs with an e4m3 scale (read as unsigned) for every 16 elements of
// k of each row of A and column of B, and an fp32 accumulator; twice the rate of the block-scaled e4m3 MMA. Eight elements
// per register, the first in the low nibble: thread (g, t) holds a[i] = A(g + 8*(i&1), 32*(i>>1) + 8t .. +7),
// b0 / b1 = B(8t .. +7 / 32 + 8t .. +7, g), and d as for bf16. Byte i of a scale register covers k [16i, +16): row m takes
// the `sfa` register of thread 4*(m%8) + 2*THREAD_A + (m >= 8), column n the `sfb` register of thread 4*n + THREAD_B.
template<int THREAD_A, int THREAD_B>
__device__ __forceinline__ void mma_m16n8k64_nvf4(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb) {
    static constexpr uint16_t byte_id = 0, thread_a = THREAD_A, thread_b = THREAD_B;
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
                   "r"(sfa), "h"(byte_id), "h"(thread_a), "r"(sfb), "h"(byte_id), "h"(thread_b));
}

// Two fp32 values -> two e2m1 values in one byte (`lo` in the low nibble), round to nearest, saturating
__device__ __forceinline__ uint32_t f32x2_to_e2m1x2(float lo, float hi) {
    uint32_t result;
    asm("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %1, %2; cvt.u32.u8 %0, t; }\n" : "=r"(result) : "f"(hi), "f"(lo));
    return result;
}

// Two fp32 values -> two e4m3 values (`lo` in the low byte), round to nearest, saturating
__device__ __forceinline__ uint32_t f32x2_to_e4m3x2(float lo, float hi) {
    uint16_t result;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(result) : "f"(hi), "f"(lo));
    return result;
}

// Two fp16 values (low half first) -> two e4m3 values (low byte first), round to nearest, saturating
__device__ __forceinline__ uint32_t f16x2_to_e4m3x2(uint32_t f16x2) {
    uint16_t result;
    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;\n" : "=h"(result) : "r"(f16x2));
    return result;
}

// Two non-negative fp32 values -> the smallest powers of two not below them, as ue8m0 (`lo` in the low byte)
__device__ __forceinline__ uint32_t f32x2_to_ue8m0x2_up(float lo, float hi) {
    uint16_t result;
    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;\n" : "=h"(result) : "f"(hi), "f"(lo));
    return result;
}

// Two e4m3 values (low byte first) -> two fp16 values (low half first), exact
__device__ __forceinline__ uint32_t e4m3x2_to_f16x2(uint32_t two_bytes) {
    uint32_t result;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(result) : "h"(static_cast<uint16_t>(two_bytes)));
    return result;
}

// Two e2m1 values in one byte (low nibble first) -> two fp16 values (low half first), exact
__device__ __forceinline__ uint32_t e2m1x2_to_f16x2(uint32_t one_byte) {
    uint32_t result;
    asm("{ .reg .b8 t; cvt.u8.u32 t, %1; cvt.rn.f16x2.e2m1x2 %0, t; }\n" : "=r"(result) : "r"(one_byte));
    return result;
}

}
