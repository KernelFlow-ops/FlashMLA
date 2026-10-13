#pragma once

#include "params.h"

// Sparse (DSA) decoding kernels for sm120 (RTX 50 series / RTX PRO Blackwell), built on warp-level mma.sync.
// One CTA handles one query token and 64 heads; h_q = 128 runs two CTAs per query token.
namespace sm120::decode::sparse {

// KV tokens per tile, also the block size of the split-KV scheduler (get_decoding_sched_meta)
static constexpr int B_TOPK = 32;

void run_sparse_decode(const SparseAttnDecodeParams &params);

}
