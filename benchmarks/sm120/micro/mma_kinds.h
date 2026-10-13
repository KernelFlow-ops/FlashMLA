// Thin wrappers over the sm120 `mma.sync` kinds used by the micro benchmarks.
// Every kind takes the same register shapes: A = 4 x b32, B = 2 x b32, and an f32 (4 regs) or
// f16x2 (2 regs) accumulator. Block-scaled kinds additionally take one b32 of scale data per
// operand plus the byte-id / thread-id selectors (template arguments, they must be constants).
#pragma once

#include <cstdint>
#include <cuda_fp16.h>

struct AccF32 { float v[4]; };
struct AccF16 { uint32_t v[2]; };

__device__ __forceinline__ void acc_zero(AccF32 &d) { d.v[0] = d.v[1] = d.v[2] = d.v[3] = 0.f; }
__device__ __forceinline__ void acc_zero(AccF16 &d) { d.v[0] = d.v[1] = 0u; }
__device__ __forceinline__ float acc_sum(const AccF32 &d) { return d.v[0] + d.v[1] + d.v[2] + d.v[3]; }
__device__ __forceinline__ float acc_sum(const AccF16 &d) {
    float s = 0.f;
    for (int i = 0; i < 2; ++i) {
        __half2 h = *reinterpret_cast<const __half2 *>(&d.v[i]);
        s += __half2float(h.x) + __half2float(h.y);
    }
    return s;
}

#define MMA_F32_OPERANDS                                                                         \
    : "+f"(d.v[0]), "+f"(d.v[1]), "+f"(d.v[2]), "+f"(d.v[3])                                     \
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1])
#define MMA_F16_OPERANDS                                                                         \
    : "+r"(d.v[0]), "+r"(d.v[1])                                                                 \
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1])
#define MMA_SF_OPERANDS                                                                          \
    : "+f"(d.v[0]), "+f"(d.v[1]), "+f"(d.v[2]), "+f"(d.v[3])                                     \
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sfa), "h"(bid_a),    \
      "h"(tid_a), "r"(sfb), "h"(bid_b), "h"(tid_b)

#define MMA_F32_FRAG "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
#define MMA_F16_FRAG "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
#define MMA_SF_FRAG "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%14,%15};\n"

// ELEM_BITS is the width of one A / B element in its register (8, or 4 for the k64 kinds);
// kScale is a scale register that means 1.0 in every byte.
#define DEFINE_PLAIN_MMA(NAME, ACC, K_, ELEM_BITS, OPERANDS, PTX)                                \
    struct NAME {                                                                                \
        using Acc = ACC;                                                                         \
        static constexpr int M = 16, N = 8, K = K_, kElemBits = ELEM_BITS;                       \
        static constexpr bool kBlockScaled = false;                                              \
        static constexpr uint32_t kScale = 0;                                                    \
        static constexpr const char *kName = #NAME;                                              \
        template <int BA = 0, int TA = 0, int BB = 0, int TB = 0>                                \
        __device__ __forceinline__ static void run(Acc &d, const uint32_t (&a)[4],               \
                                                   const uint32_t (&b)[2], uint32_t, uint32_t) { \
            asm volatile(PTX OPERANDS);                                                          \
        }                                                                                        \
    };

#define DEFINE_SF_MMA(NAME, K_, ELEM_BITS, SCALE, PTX)                                           \
    struct NAME {                                                                                \
        using Acc = AccF32;                                                                      \
        static constexpr int M = 16, N = 8, K = K_, kElemBits = ELEM_BITS;                       \
        static constexpr bool kBlockScaled = true;                                               \
        static constexpr uint32_t kScale = SCALE;                                                \
        static constexpr const char *kName = #NAME;                                              \
        template <int BA = 0, int TA = 0, int BB = 0, int TB = 0>                                \
        __device__ __forceinline__ static void run(Acc &d, const uint32_t (&a)[4],               \
                                                   const uint32_t (&b)[2], uint32_t sfa,         \
                                                   uint32_t sfb) {                               \
            static constexpr uint16_t bid_a = BA, tid_a = TA, bid_b = BB, tid_b = TB;            \
            asm volatile(PTX MMA_SF_OPERANDS);                                                   \
        }                                                                                        \
    };

DEFINE_PLAIN_MMA(BF16_F32, AccF32, 16, 16, MMA_F32_OPERANDS,
                 "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " MMA_F32_FRAG)
DEFINE_PLAIN_MMA(F16_F32, AccF32, 16, 16, MMA_F32_OPERANDS,
                 "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 " MMA_F32_FRAG)
DEFINE_PLAIN_MMA(F16_F16, AccF16, 16, 16, MMA_F16_OPERANDS,
                 "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 " MMA_F16_FRAG)
DEFINE_PLAIN_MMA(E4M3_F32, AccF32, 32, 8, MMA_F32_OPERANDS,
                 "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " MMA_F32_FRAG)
DEFINE_PLAIN_MMA(E4M3_F16, AccF16, 32, 8, MMA_F16_OPERANDS,
                 "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 " MMA_F16_FRAG)
DEFINE_PLAIN_MMA(KF8_E4M3_F32, AccF32, 32, 8, MMA_F32_OPERANDS,
                 "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 " MMA_F32_FRAG)
DEFINE_PLAIN_MMA(KF8_E4M3_F16, AccF16, 32, 8, MMA_F16_OPERANDS,
                 "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f16.e4m3.e4m3.f16 " MMA_F16_FRAG)
DEFINE_PLAIN_MMA(KF8_E4M3_E2M1_F32, AccF32, 32, 8, MMA_F32_OPERANDS,
                 "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e2m1.f32 " MMA_F32_FRAG)
DEFINE_PLAIN_MMA(KF8_E2M1_F32, AccF32, 32, 8, MMA_F32_OPERANDS,
                 "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32 " MMA_F32_FRAG)
DEFINE_SF_MMA(MXF8_E4M3, 32, 8, 0x7f7f7f7fu,
              "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 " MMA_SF_FRAG)
DEFINE_SF_MMA(MXF8_E4M3_E2M1, 32, 8, 0x7f7f7f7fu,
              "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0 " MMA_SF_FRAG)
DEFINE_SF_MMA(MXF8_E2M1, 32, 8, 0x7f7f7f7fu,
              "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e2m1.f32.ue8m0 " MMA_SF_FRAG)
DEFINE_SF_MMA(MXF4_2X, 64, 4, 0x7f7f7f7fu,
              "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 " MMA_SF_FRAG)
DEFINE_SF_MMA(NVF4_4X, 64, 4, 0x38383838u,
              "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 " MMA_SF_FRAG)
