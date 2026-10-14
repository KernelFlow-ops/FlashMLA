#pragma once

#include "params.h"

// Sparse (DSA) decoding kernels for sm120 (RTX 50 series / RTX PRO Blackwell), built on warp-level mma.sync.
// One CTA handles one query token and 64 heads; h_q = 128 runs two CTAs per query token.
namespace sm120::decode::sparse {

// KV tokens per tile, also the block size of the split-KV scheduler (get_decoding_sched_meta)
static constexpr int B_TOPK = 32;

// Checks the inputs and runs the kernel that the environment variable FLASH_MLA_SM120_DECODE_MODE selects:
//   bf16 (default)  bf16.cu: bf16 Q, KV dequantized to bf16, the arithmetic of the sm100 kernels
//   fp8x2           mxfp8.cu with two e4m3 terms for Q and P: within the tolerance of bf16, half the MMA cycles
//   mxfp8           mxfp8.cu with one e4m3 term: a quarter of the MMA cycles, 4 significant bits of Q and P
// The fp8 kernels read V4.1 fp8 caches only; with an fp4 extra cache the bf16 kernel runs whatever the mode.
void run_sparse_decode(const SparseAttnDecodeParams &params);

// The kernels behind it
void run_bf16(const SparseAttnDecodeParams &params);
void run_mxfp8(const SparseAttnDecodeParams &params, int num_terms);

}
