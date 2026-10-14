# FlashMLA on RTX 50 series / sm120: DeepSeek-V4.1 sparse decoding

This branch adds an sm120 (RTX 5090 and the other RTX 50 / RTX PRO Blackwell GPUs) implementation of the
DeepSeek-V4.1 sparse attention (DSA) decoding kernel, `flash_mla_with_kvcache`. Upstream has no kernel that runs on this architecture:
its sparse decoding kernels are written for sm100 (tcgen05 / TMEM, which sm120 does not have, and more shared memory than the
101,376 B an sm120 block gets).

The kernel runs QK and PV on the block-scaled MMAs of sm120 (`mma.sync` `kind::mxf8f6f4`, `kind::mxf4nvf4`,
`kind::mxf4`), which do 4x (FP8) and 8x (FP4) the work per clock of the BF16 MMA on this GPU. It offers precision modes
from exact (the default; it passes the upstream test with the upstream tolerances) to fast approximate ones.

Scope:

* Supported: sparse decoding with the V4.1 caches of upstream (FP8 main cache, FP8 or NVFP4-style FP4 extra cache) plus a new
  **MXFP4** cache (main and extra), h_q = 64 / 128, `attn_sink`, `topk_length` / `extra_topk_length`, split-KV, batch invariance, CUDA Graph capture.
* Not included: sparse prefill, dense MHA and the fused norm-RoPE-attn-RoPE-cast kernel on sm120.

## Build and run

```bash
# sm_120a only (the sm100 and sm120 targets cannot be mixed in one build); NVCC >= 12.8, tested with CUDA 13.0 + torch 2.12 (cu130)
FLASH_MLA_CUDA_ARCHS=sm_120a python setup.py build_ext --inplace

# the upstream test (exact mode); tests/lib.py only learned to accept compute capability 12
cd tests && python test-sparse-decode.py
# the same in a fast mode
FLASH_MLA_SM120_DECODE_MODE=fp8x2 python test-sparse-decode.py
# MXFP4 cache: correctness and corner cases, both of its modes (the script sets the mode itself)
python test-sparse-decode-sm120-mxfp4.py
```

The mode is read from `FLASH_MLA_SM120_DECODE_MODE` at every call, so one process can switch between modes. The API itself does not change.

| mode | arithmetic | accuracy (vs `tests/ref.py`) | caches |
|---|---|---|---|
| `bf16` (default) | KV dequantized to bf16 in registers, BF16 MMA | exact: the upstream test passes 1534 / 1534 | FP8 main + FP8 / FP4 extra |
| `fp8x2` | Q and P each as the sum of two e4m3 terms (8 significant bits), block-scaled e4m3 MMA | passes the upstream test 1534 / 1534 (out cos_diff about 9e-7) | FP8 main + FP8 extra; with an FP4 extra cache it runs the `bf16` kernel |
| `mxfp8` | Q and P as one e4m3 term; FP4 extra tiles: QK on the NVFP4 MMA with Q as two e2m1 terms | out cos_diff 2.0e-4 to 2.2e-4 (FP8 extra), 3.6e-4 to 3.7e-4 (FP4 extra); same class as FlashInfer's default FP8 mode (2.2e-4 to 2.6e-4 / 4.0e-4 to 4.2e-4). Fails the upstream out tolerance (5e-6) by design | all, including MXFP4 (Q as two MXFP4 terms) |
| `mxfp4x4` | MXFP4 cache only: Q as four MXFP4 terms on the FP4 MMA, P as two e4m3 terms | out within the upstream tolerance (cos_diff <= 1.2e-6); the lse of a query token with exactly one valid KV token is off by up to 3.4e-6 (upstream limit 1e-6) | MXFP4 main (+ MXFP4 extra) |

With an MXFP4 cache, `bf16` and `fp8x2` raise an error. No arithmetic that fits the kernel's registers and shared memory meets
the upstream lse tolerance on single-token rows, so the caller has to pick `mxfp4x4` or `mxfp8` explicitly.

KV cache formats (bytes per token, `tests/quant.py`):

| format | layout | bytes / token | where |
|---|---|---|---|
| V4.1 FP8 (upstream) | 512 B e4m3 + 16 B ue8m0 (one scale per 32 dims) | 528 | main or extra |
| V4.1 FP4 (upstream) | 256 B e2m1 + 32 B e4m3 (one scale per 16 dims) | 288 | extra only (as upstream) |
| MXFP4 (new, `KVCacheLayout.V41_MXFP4Sparse`) | 256 B e2m1 + 16 B ue8m0 (one scale per 32 dims) | 272 | main, and extra next to an MXFP4 main cache |

The format is recognized by the last dimension of the cache tensor, as upstream does.

## Results (RTX 5090)

**How to read the tables.**
* Times are microseconds per decoding call, as `tests/test-sparse-decode.py` measures them: kineto, first kernel start to last kernel end,
  after an 8 GB L2 flush, median of runs.
* Shapes are the production shapes of the upstream test: s_q = 4, topk 128 (main cache) + 512 (extra cache), d_qk = d_v = 512, with
  the KV lengths of the test. The last two lines of each block are b = 1, s_q = 4096.
* "P1 baseline" is the first working sm120 kernel of this branch: the upstream algorithm moved to `mma.sync`, with KV dequantized to
  bf16. Today's `bf16` mode is that algorithm. The factor in brackets is the speedup over the baseline.
* Conditions: one RTX 5090 (170 SMs), driver clocks not locked (2.4 to 2.9 GHz, recorded per run), nothing else on the GPU.
* Peaks measured on this card with `benchmarks/sm120/micro`: BF16 MMA 252 TFLOPS, block-scaled FP8 MMA 4x that, FP4 8x, random row gather into shared memory 1.53 TB/s.

### Production shapes

| h | b | P1 baseline (exact) | bf16 (exact) | fp8x2 | mxfp8 |
|---|---|---|---|---|---|
| **extra FP8** | | | | | |
| 64 | 64 | 162.1 | 163.6 (0.99x) | 114.6 (1.41x) | 85.4 (1.90x) |
| 64 | 128 | 303.6 | 306.7 (0.99x) | 210.8 (1.44x) | 151.7 (2.00x) |
| 64 | 256 | 517.8 | 520.7 (0.99x) | 365.3 (1.42x) | 260.7 (1.99x) |
| 64 | 512 | 946.3 | 953.2 (0.99x) | 675.5 (1.40x) | 478.7 (1.98x) |
| 128 | 64 | 302.2 | 301.7 (1.00x) | 207.0 (1.46x) | 144.1 (2.10x) |
| 128 | 128 | 519.2 | 522.1 (0.99x) | 360.3 (1.44x) | 249.4 (2.08x) |
| 128 | 256 | 947.9 | 949.5 (1.00x) | 661.7 (1.43x) | 459.5 (2.06x) |
| 128 | 512 | 1813.1 | 1810.8 (1.00x) | 1280.9 (1.42x) | 886.5 (2.05x) |
| 64 | 1 (s_q 4096) | 1787.5 | 1798.9 (0.99x) | 1245.8 (1.43x) | 863.2 (2.07x) |
| 128 | 1 (s_q 4096) | 3501.9 | 3507.8 (1.00x) | 2465.2 (1.42x) | 1684.5 (2.08x) |
| **extra FP4** | | | | | |
| 64 | 64 | 161.6 | 163.0 (0.99x) | bf16 path | 91.5 (1.77x) |
| 64 | 128 | 305.4 | 307.1 (0.99x) | bf16 path | 162.8 (1.88x) |
| 64 | 256 | 519.7 | 523.1 (0.99x) | bf16 path | 280.3 (1.85x) |
| 64 | 512 | 945.7 | 950.3 (1.00x) | bf16 path | 512.5 (1.85x) |
| 128 | 64 | 302.2 | 304.0 (0.99x) | bf16 path | 157.3 (1.92x) |
| 128 | 128 | 518.5 | 520.6 (1.00x) | bf16 path | 274.9 (1.89x) |
| 128 | 256 | 945.4 | 950.1 (1.00x) | bf16 path | 500.8 (1.89x) |
| 128 | 512 | 1803.5 | 1809.8 (1.00x) | bf16 path | 946.3 (1.91x) |
| 64 | 1 (s_q 4096) | 1795.8 | 1800.2 (1.00x) | bf16 path | 966.9 (1.86x) |
| 128 | 1 (s_q 4096) | 3502.5 | 3510.0 (1.00x) | bf16 path | 1878.6 (1.86x) |

Where the kernels stand against the hardware (b = 512, upstream test counts of FLOPs and bytes):
* `bf16`: 180 to 190 TFLOPS, 71 to 75% of the BF16 MMA peak. It is bound by the BF16 MMA, which is half rate on GeForce parts.
* `mxfp8`, h = 64 with an FP8 extra cache: 1397 GB/s, 91% of the measured gather limit. It is memory bound; removing all of its
  softmax and rebase work saves only 2.4%.
* `mxfp8`, h = 128: 388 TFLOPS. The rebase of V and the softmax still cost 8 to 15% each, see "Open items".
* `fp8x2`: 254 to 268 TFLOPS, i.e. 510 to 540 TFLOPS of MMA work since every product runs twice, about half the block-scaled FP8 peak.

### MXFP4 cache (main and extra cache MXFP4, same shapes)

Measured with the same method (kineto + 8 GB L2 flush) in one session against the FP8 cache, one row per mode.

| mode | h | b = 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|
| `mxfp8`, MXFP4 cache | 64 | 78.2 | 135.9 | 249.1 | 461.5 |
| `mxfp8`, FP8 cache | 64 | 85.0 | 148.1 | 258.9 | 477.3 |
| `mxfp8`, MXFP4 cache | 128 | 139.7 | 239.2 | 451.0 | 861.0 |
| `mxfp8`, FP8 cache | 128 | 143.9 | 246.8 | 458.0 | 881.6 |
| `mxfp4x4`, MXFP4 cache | 64 | 110.3 | 208.3 | 357.1 | 655.6 |
| `fp8x2`, FP8 cache | 64 | 113.9 | 209.5 | 363.0 | 672.4 |
| `mxfp4x4`, MXFP4 cache | 128 | 205.5 | 354.7 | 649.0 | 1241.6 |
| `fp8x2`, FP8 cache | 128 | 206.1 | 355.7 | 659.1 | 1268.7 |

The MXFP4 cache halves the bytes read per token (272 vs 528), but these kernels are not bound by memory reads. They are 0.3 to 8% faster
than on the FP8 cache (most at small b), and the cache takes half the memory.

### Against FlashInfer (0.7.1, `sparse_mla_sm120`, same data, same timing method)

At the production shapes (T = b * s_q = 256 to 2048 query tokens), FlashInfer's wrapper uses its prefill kernel. Its decode kernels take at most 64 tokens per call.
* FlashInfer's exact mode (`bf16`) passes the upstream tolerance like ours.
* FlashInfer's default mode computes with FP8 operands; its errors are given in the mode table above. It has no FP8 kernel for h = 128 at these shapes; the call raises.

| extra | h | b | FlashInfer bf16 | ours bf16 | FlashInfer default | ours mxfp8 |
|---|---|---|---|---|---|---|
| FP8 | 64 | 64 | 375 | 163.6 (2.3x) | 326 | 85.4 (3.8x) |
| FP8 | 64 | 512 | 2792 | 953.2 (2.9x) | 2276 | 478.7 (4.8x) |
| FP8 | 128 | 64 | 705 | 301.7 (2.3x) | raises | 144.1 |
| FP8 | 128 | 512 | 5481 | 1810.8 (3.0x) | raises | 886.5 |
| FP4 | 64 | 64 | 397 | 163.0 (2.4x) | 333 | 91.5 (3.6x) |
| FP4 | 64 | 512 | 2991 | 950.3 (3.1x) | 2356 | 512.5 (4.6x) |
| FP4 | 128 | 64 | 766 | 304.0 (2.5x) | raises | 157.3 |
| FP4 | 128 | 512 | 5881 | 1809.8 (3.2x) | raises | 946.3 |

### Small batches (T = b * s_q <= 64)

A CTA serves one query token and 64 heads, so a small batch leaves most SMs idle. Upstream's rule (split only when topk + extra_topk > 640) kept
these shapes on one CTA per token. The baseline took 72 to 83 us whatever the size: 20 tiles in sequence. On sm120 the requests are
now split over the SMs whenever b * s_q * (h_q / 64) <= num_SMs / 2 and batch invariance is not requested. At 128 CTAs, splitting
was 3 to 5% slower and stays off.

Kineto times in us (decode + combine kernel), 128 + 512 tokens, extra cache FP8. FlashInfer runs its decode kernel at these sizes.

| shape | T | baseline | bf16 | fp8x2 | mxfp8 | FlashInfer bf16 | FlashInfer default |
|---|---|---|---|---|---|---|---|
| h64, b1, s_q 1 | 1 | 72.7 | 19.1 | 16.7 | 16.0 | 16.9 | 14.8 |
| h64, b4, s_q 1 | 4 | 73.3 | 23.1 | 20.7 | 20.2 | 19.1 | 17.2 |
| h64, b1, s_q 4 | 4 | 73.2 | 24.2 | 22.4 | 21.0 | 17.1 | 15.4 |
| h64, b4, s_q 4 | 16 | 74.1 | 28.4 | 23.1 | 22.1 | 48.9 | 32.1 |
| h64, b16, s_q 4 | 64 | 76.6 | 55.9 | 39.3 | 33.0 | 137.3 | 87.9 |
| h128, b1, s_q 1 | 1 | 73.3 | 20.8 | 19.3 | 18.1 | 16.9 | 14.9 |
| h128, b1, s_q 4 | 4 | 73.5 | 24.7 | 21.3 | 19.7 | 27.4 | 19.3 |
| h128, b4, s_q 4 | 16 | 75.0 | 36.7 | 28.9 | 25.8 | 73.3 | 46.6 |
| h128, b16, s_q 4 | 64 | 82.6 | 83.7 | 54.7 | 39.9 | 263.1 | 245.1 |

From T = 16 on, our exact mode is 2 to 3x faster than FlashInfer's. At T = 1, and at T = 4 with h = 64, FlashInfer's exact mode is still
1.1 to 1.4x faster than ours (its default mode is 1.1 to 1.4x faster than our `mxfp8`). About 6 us of our time there is the combine kernel. Under a CUDA Graph (`benchmarks/sm120/bench_decode_graph.py`), the bf16 time per layer of
h64 b1 s_q 1 goes from 71.8 to 17.8 us, and graph replay equals eager bit for bit for every shape and mode.

## Correctness checks

Every change was checked with the following:
* the upstream `tests/test-sparse-decode.py`: all 1534 cases. `bf16` and `fp8x2` pass all of them; `mxfp8` passes 253, the cases
  whose tolerance an approximate mode can meet.
* `tests/test-sparse-decode-sm120-mxfp4.py`: 856 cases of the MXFP4 cache, written after the upstream test, with its corner cases.
  These include page sizes 1 to 576, rows with no valid token, `topk_length`, sinks, and split and no split.
* `tests/eval-sm120-modes.py`: a float64 emulator of every mode. The kernels' bf16 outputs equal the emulator's in 99.9 to 100% of the
  elements, on upstream data and on high-variance logits, which upstream data does not exercise (its attention is close to uniform).
* `benchmarks/sm120/bench_decode_graph.py`: CUDA Graph replay equals eager bit for bit for all production and small shapes.
* The builds have no register spills.

## Kernel design

`csrc/cuda_kernels/sm120/decode/sparse/`: `bf16.cu` (exact path), `mxfp8.cu` (FP8 main cache, `fp8x2` / `mxfp8`, FP4 extra tiles), `mxfp4.cu`
(MXFP4 cache), `run.cu` (mode dispatch). The scheduler metadata and the combine kernel are upstream's (`smxx`).

The block-scaled kernels (`mxfp8.cu`, `mxfp4.cu`):

* **CTA = one query token x 64 heads, 384 threads.** 8 consumer warps and 4 producer warps, with registers moved to the consumers by
  `setmaxnreg` (232 / 40, or 224 / 56 where the producers convert FP4 tiles).
  h = 128 runs two CTAs, launched next to each other so that the second read of a KV tile hits L2.
* **QK on the block-scaled FP8 MMA with K as stored.** The e4m3 bytes of a cached token are the B operand and its ue8m0 scales are
  the MMA's column scales, so nothing is dequantized. Q is quantized once per request. FP4 extra tiles run their QK on the
  NVFP4 MMA (`kind::mxf4nvf4`, 2x the FP8 rate) with Q quantized to two e2m1 terms. MXFP4 tiles use `kind::mxf4`.
* **PV needs V's scales on the k dimension, which the MMA cannot take.** The producer warps *rebase* each 32-token tile once. Per 32
  dims they take the largest exponent in the tile as the column scale and shift each token's bytes down by its exponent difference.
  This is exact unless a value becomes subnormal in e4m3, where it is rounded to nearest. The work is done once per tile, whatever h is. It is integer
  only, and it keeps PV on the full-rate block-scaled MMA. (FlashInfer folds the scales into P and runs PV on the half-rate
  unscaled e4m3 MMA.) FP4 and MXFP4 tiles are converted once to rebased e4m3.
* **fp8x2.** Q and P are each written as hi + lo e4m3 terms, with the lo scale 16x smaller: 8 significant bits, as many as bf16.
  The two MMAs per product still cost half the cycles of one BF16 MMA. This is why it passes the upstream tolerance at 1.4x the speed of bf16.
* **Producers.** Warp 0 gathers tiles into a ring of 3 raw buffers. Rows of FP8 tiles use TMA `tile::gather4` (4 rows per instruction,
  about a third of the issue cost of one bulk copy per row). FP4 rows use one `cp.async.bulk` each. Warps 1 to 3 rebase into 2 V
  buffers, so that a rebase never waits for the previous PV. Row indices are loaded one tile ahead.
* **Consumers in pairs.** Warps 2m and 2m+1 sit on different SM sub-partitions and share 16 heads. Each computes S over half of the dims,
  they swap halves through shared memory, and each runs PV over half of the output columns. The two pairs on one sub-partition
  drift apart, so one pair's softmax overlaps the other's MMAs (`fp8x2` -7 to -9%).
* **Split for small batches**, see above.

## What did not help (measured, not kept)

* Forcing split-KV at the production shapes: the combine kernel eats the saved tail wave, 3 to 7% slower at b >= 256.
* Gathering FP4 rows with `cp.async` from 128 producer threads: 4 to 6% slower. The extra issue slots compete with the consumers.
* Lookup-table or two-pass rebase of FP4 tiles: 1 to 4% slower. The rebase warp is bound by its latency chain, not by instruction count.
* An integer fast path of the FP8 rebase (97.5% hit rate): up to 5% slower in `mxfp8`, no gain in `fp8x2`. The per-half vote serializes the two halves.
* The gather warp taking over part of the rebase after `gather4` freed it: 3.5 to 5% slower. It waits on the previous tile, so prefetch gets shallower.
* Consumer ping-pong (QK of tile t next to PV of tile t-1): 3 to 5% slower. It holds the two V buffers longer than the rebase period allows.
* `mxfp8` with Q as one term and P as two (or the reverse): no such mode is both cheaper than `fp8x2` and within the upstream tolerance.

## Open items

* FP8 main cache with an MXFP4 extra cache: rejected with an error.
* `fp8x2` with an FP4 extra cache runs the bf16 kernel. Staying within tolerance would need four Q terms, a pre-scale of Q and two V terms.
* At h = 128 and with FP4 extra caches, the remaining time is the V rebase, the softmax and, for FP4, the per-row copies.
  Knocking them out together saves 19 to 25%, any one alone much less. Candidates: in-place rebase, a third V buffer, or `gather4` for FP4 rows.
* The re-quantization of Q at the switch from main to FP4 extra tiles costs 2 to 3% at b = 512 and 5 to 7% at b = 64 to 128.
* Smallest batches (T <= 4, h = 64): the combine kernel and the per-CTA setup leave us 1.1 to 1.4x behind FlashInfer.
* The defaults are unchanged: `bf16`; `bf16` / `fp8x2` raise on an MXFP4 cache.
