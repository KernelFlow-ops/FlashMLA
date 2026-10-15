# FlashMLA on RTX 50 series / sm120: DeepSeek-V4.1 sparse decoding

> An illustrated overview of this work (background, results, design, figures) is the [top-level README](../../README.md); this page is the technical reference.

This branch adds an sm120 (RTX 5090 and the other RTX 50 / RTX PRO Blackwell GPUs) implementation of the
DeepSeek-V4.1 sparse attention (DSA) decoding kernel, `flash_mla_with_kvcache`. Upstream has no kernel that runs on this architecture:
its sparse decoding kernels are written for sm100 (tcgen05 / TMEM, which sm120 does not have, and more shared memory than the
101,376 B an sm120 block gets).

The kernel runs QK and PV on the block-scaled MMAs of sm120 (`mma.sync` `kind::mxf8f6f4`, `kind::mxf4nvf4`,
`kind::mxf4`), which do 4x (FP8) and 8x (FP4) the work per clock of the BF16 MMA on this GPU. It offers precision modes
from exact (the default; it passes the upstream test with the upstream tolerances) to fast approximate ones.

Scope:

* Supported: sparse decoding with the V4.1 caches of upstream (FP8 main cache, FP8 or NVFP4-style FP4 extra cache), the V4.1 FP4
  cache **as the main cache** too (with a V4.1 FP4 extra cache or none), and a new **MXFP4** cache (main and extra), h_q = 64 / 128, `attn_sink`, `topk_length` / `extra_topk_length`, split-KV, batch invariance, CUDA Graph capture.
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
# V4.1 FP4 (NVFP4) cache as the main cache: correctness and corner cases, all modes
python test-sparse-decode-sm120-nvfp4.py
# kernel time of every cache format and mode (the tables below; exclusive GPU)
cd .. && PYTHONPATH=. python benchmarks/sm120/bench_formats.py
```

The mode is read from `FLASH_MLA_SM120_DECODE_MODE` at every call, so one process can switch between modes. The API itself does not change.

| mode | arithmetic | accuracy (vs `tests/ref.py`) | caches |
|---|---|---|---|
| `bf16` (default) | KV dequantized to bf16 in registers, BF16 MMA | exact: the upstream test passes 1534 / 1534; on the V4.1 FP4 main cache all 860 cases of `test-sparse-decode-sm120-nvfp4.py` pass the upstream tolerances | FP8 main + FP8 / FP4 extra; V4.1 FP4 main (+ V4.1 FP4 extra) |
| `fp8x2` | Q and P each as the sum of two e4m3 terms (8 significant bits), block-scaled e4m3 MMA | passes the upstream test 1534 / 1534 (out cos_diff about 9e-7) | FP8 main + FP8 extra; with V4.1 FP4 tiles (an FP4 extra or main cache) it runs the `bf16` kernel |
| `mxfp8` | Q and P as one e4m3 term; V4.1 FP4 tiles: QK on the NVFP4 MMA with Q as two e2m1 terms | out cos_diff 2.0e-4 to 2.2e-4 (FP8 extra), 3.6e-4 to 3.7e-4 (FP4 extra), median 4.2e-4 (V4.1 FP4 main cache); same class as FlashInfer's default FP8 mode (2.2e-4 to 2.6e-4 / 4.0e-4 to 4.2e-4). Fails the upstream out tolerance (5e-6) by design | all, including MXFP4 (Q as two MXFP4 terms) and the V4.1 FP4 main cache |
| `mxfp4x4` | MXFP4 cache only: Q as four MXFP4 terms on the FP4 MMA, P as two e4m3 terms | out within the upstream tolerance (cos_diff <= 1.2e-6); the lse of a query token with exactly one valid KV token is off by up to 3.4e-6 (upstream limit 1e-6) | MXFP4 main (+ MXFP4 extra) |

With an MXFP4 cache, `bf16` and `fp8x2` raise an error. No arithmetic that fits the kernel's registers and shared memory meets
the upstream lse tolerance on single-token rows, so the caller has to pick `mxfp4x4` or `mxfp8` explicitly.

KV cache formats (bytes per token, `tests/quant.py`):

| format | layout | bytes / token | where |
|---|---|---|---|
| V4.1 FP8 (upstream) | 512 B e4m3 + 16 B ue8m0 (one scale per 32 dims) | 528 | main or extra |
| V4.1 FP4 (upstream, NVFP4-style) | 256 B e2m1 + 32 B e4m3 (one scale per 16 dims) | 288 | extra next to an FP8 main cache (as upstream); on sm120 also main, with a V4.1 FP4 extra cache or none |
| MXFP4 (new, `KVCacheLayout.V41_MXFP4Sparse`) | 256 B e2m1 + 16 B ue8m0 (one scale per 32 dims) | 272 | main, and extra next to an MXFP4 main cache |

The format is recognized by the last dimension of the cache tensor, as upstream does.

How much the 4-bit formats lose (`tests/eval-sm120-storage.py`: fp64 attention on the dequantized cache against fp64 attention on the
unquantized bf16 KV, upstream-style data): the output's relative RMS error is 2.6e-2 with the FP8 cache, 9.5e-2 with the V4.1 FP4 cache
and 1.15e-1 with MXFP4. The V4.1 FP4 (NVFP4-style) cache is 17 to 18% more accurate than MXFP4 on such data, 12% with a spread of token
amplitudes and 56% with outlier channels, for 5.9% more bytes. The kernel's modes add little to it: `mxfp8` adds 0.1 to 4.5% on the
V4.1 FP4 cache, the exact modes nothing measurable. The cache writer matters as much: the upstream writer rounds amax / 6 to the
nearest e4m3 scale, which is 5 to 11% more accurate than rounding it up, and choosing per block between 6 and 4 as the largest
magnitude ("Four Over Six") gains another 8 to 15% with the same bytes. A second, per-tensor power-of-two scale (as in two-level NVFP4)
changes nothing on such data; it matters only for values far outside the e4m3 scale range (x 2^-12, x 2^14), where the one-level format
breaks down.

## Results (RTX 5090)

**How to read the tables.**
* Times are microseconds per decoding call: the time of the decode kernel as `tests/test-sparse-decode.py` measures it (kineto, 8 GB
  L2 flush before every call, 10 calls), taken with `benchmarks/sm120/bench_formats.py`.
* Shapes are the production shapes of the upstream test: s_q = 4, topk 128 (main cache, page 256) + 512 (extra cache, page 64),
  d_qk = d_v = 512, with the KV lengths of the test. The last two lines of each table are b = 1, s_q = 4096 ("prefill as decode").
* "P1 baseline" is the first working sm120 kernel of this branch (24ccda5): the upstream algorithm moved to `mma.sync`, with KV
  dequantized to bf16. Today's `bf16` mode is that algorithm. The factor in brackets is the P1 baseline's time divided by the cell's,
  i.e. the speedup. A 4-bit main cache is compared with the P1 baseline on the upstream caches it replaces: MXFP4 (main and extra)
  with FP8 main + FP8 extra, the V4.1 FP4 main cache (with a V4.1 FP4 extra cache) with FP8 main + V4.1 FP4 extra.
* All columns come from one session: build bd6f23e and the P1 baseline, each in its own process, alternating (2 passes in each of 2
  rounds). A cell is the median of its 4 samples, which were never more than 0.8% apart.
* Conditions: one RTX 5090 (170 SMs), driver clocks not locked (2.8 to 2.9 GHz under load), nothing else on the GPU.
* Peaks measured on this card with `benchmarks/sm120/micro`: BF16 MMA 252 TFLOPS, block-scaled FP8 MMA 4x that, FP4 8x, random row gather into shared memory 1.53 TB/s.

### FP8 main cache (the upstream formats): FP8 or V4.1 FP4 extra cache

`fp8x2` with a V4.1 FP4 extra cache runs the `bf16` kernel and is left out.

| h | b | FP8 extra: P1 baseline | bf16 | fp8x2 | mxfp8 | FP4 extra: P1 baseline | bf16 | mxfp8 |
|---|---|---|---|---|---|---|---|---|
| 64 | 64 | 162.8 | 163.2 (1.00x) | 113.7 (1.43x) | 84.6 (1.92x) | 162.5 | 162.6 (1.00x) | 90.7 (1.79x) |
| 64 | 128 | 302.7 | 302.7 (1.00x) | 209.0 (1.45x) | 148.4 (2.04x) | 302.0 | 302.3 (1.00x) | 160.2 (1.89x) |
| 64 | 256 | 517.6 | 520.0 (1.00x) | 361.2 (1.43x) | 256.9 (2.01x) | 518.1 | 519.6 (1.00x) | 281.0 (1.84x) |
| 64 | 512 | 950.8 | 953.1 (1.00x) | 667.3 (1.42x) | 474.4 (2.00x) | 950.6 | 950.7 (1.00x) | 509.0 (1.87x) |
| 128 | 64 | 299.9 | 299.7 (1.00x) | 206.1 (1.46x) | 143.2 (2.09x) | 298.4 | 300.1 (0.99x) | 161.4 (1.85x) |
| 128 | 128 | 520.4 | 516.9 (1.01x) | 355.2 (1.46x) | 245.3 (2.12x) | 520.4 | 517.8 (1.01x) | 271.5 (1.92x) |
| 128 | 256 | 952.2 | 948.4 (1.00x) | 654.7 (1.45x) | 453.5 (2.10x) | 951.2 | 948.0 (1.00x) | 499.2 (1.91x) |
| 128 | 512 | 1814.6 | 1811.6 (1.00x) | 1260.2 (1.44x) | 872.2 (2.08x) | 1813.6 | 1810.4 (1.00x) | 937.0 (1.94x) |
| 64 | 1 (s_q 4096) | 1793.0 | 1797.5 (1.00x) | 1224.9 (1.46x) | 848.1 (2.11x) | 1798.1 | 1796.5 (1.00x) | 960.0 (1.87x) |
| 128 | 1 (s_q 4096) | 3515.1 | 3510.8 (1.00x) | 2423.5 (1.45x) | 1658.3 (2.12x) | 3512.9 | 3509.8 (1.00x) | 1866.4 (1.88x) |
| geomean | |  | (1.00x) | (1.45x) | (2.06x) |  | (1.00x) | (1.87x) |

Where the kernels stand against the hardware (b = 512, upstream test counts of FLOPs and bytes, build a006306, whose times are within
2% of the table):
* `bf16`: 180 to 190 TFLOPS, 71 to 75% of the BF16 MMA peak. It is bound by the BF16 MMA, which is half rate on GeForce parts.
* `mxfp8`, h = 64 with an FP8 extra cache: 1397 GB/s, 91% of the measured gather limit. It is memory bound; removing all of its
  softmax and rebase work saves only 2.4%.
* `mxfp8`, h = 128: 388 TFLOPS. The rebase of V and the softmax still cost 8 to 15% each, see "Open items".
* `fp8x2`: 254 to 268 TFLOPS, i.e. 510 to 540 TFLOPS of MMA work since every product runs twice, about half the block-scaled FP8 peak.

### 4-bit main caches (sm120 only): MXFP4 main + extra, V4.1 FP4 main + extra

`bf16` and `fp8x2` raise an error on an MXFP4 cache; `fp8x2` on the V4.1 FP4 main cache runs the `bf16` kernel and is left out.

| h | b | MXFP4: mxfp8 | mxfp4x4 | V4.1 FP4 (NVFP4): bf16 | mxfp8 |
|---|---|---|---|---|---|
| 64 | 64 | 71.8 (2.27x) | 109.9 (1.48x) | 163.3 (0.99x) | 74.9 (2.17x) |
| 64 | 128 | 130.3 (2.32x) | 206.2 (1.47x) | 301.4 (1.00x) | 138.8 (2.18x) |
| 64 | 256 | 224.5 (2.31x) | 355.3 (1.46x) | 519.4 (1.00x) | 240.1 (2.16x) |
| 64 | 512 | 416.6 (2.28x) | 651.8 (1.46x) | 953.6 (1.00x) | 443.5 (2.14x) |
| 128 | 64 | 126.8 (2.37x) | 204.1 (1.47x) | 299.0 (1.00x) | 135.8 (2.20x) |
| 128 | 128 | 220.9 (2.36x) | 351.9 (1.48x) | 519.0 (1.00x) | 237.2 (2.19x) |
| 128 | 256 | 407.0 (2.34x) | 644.0 (1.48x) | 950.5 (1.00x) | 435.2 (2.19x) |
| 128 | 512 | 784.1 (2.31x) | 1231.4 (1.47x) | 1815.5 (1.00x) | 838.0 (2.16x) |
| 64 | 1 (s_q 4096) | 778.3 (2.30x) | 1222.2 (1.47x) | 1806.5 (1.00x) | 822.9 (2.19x) |
| 128 | 1 (s_q 4096) | 1524.6 (2.31x) | 2421.1 (1.45x) | 3527.9 (1.00x) | 1618.1 (2.17x) |
| geomean | | (2.32x) | (1.47x) | (1.00x) | (2.17x) |

* MXFP4 (272 B / token) in `mxfp8` mode is 8 to 15% faster than `mxfp8` on the FP8 cache (528 B / token), `mxfp4x4` 0 to 3% faster
  than `fp8x2`. Half the bytes do not make it twice as fast: these kernels are limited by the rebase, the softmax and the issue of the
  copies, not by DRAM. The cache takes half the memory.
* The V4.1 FP4 main cache (288 B / token) in `mxfp8` mode is 4 to 7% slower than MXFP4 (below): it reads 5.9% more bytes, and its
  rebase multiplies e2m1 values by e4m3 scales where MXFP4 only shifts exponents. It is 11 to 18% faster than `mxfp8` with an FP8 main
  cache and a V4.1 FP4 extra cache: every tile takes the FP4 path, Q is quantized once, and all rows come with gather4. Its `bf16`
  mode is exact and as fast as the P1 baseline.

| h | b | V4.1 FP4 (us) | MXFP4 (us) | ratio |
|---|---|---|---|---|
| 64 | 64 | 74.9 | 71.8 | 1.043 |
| 64 | 128 | 138.8 | 130.3 | 1.065 |
| 64 | 256 | 240.1 | 224.5 | 1.070 |
| 64 | 512 | 443.5 | 416.6 | 1.065 |
| 128 | 64 | 135.8 | 126.8 | 1.071 |
| 128 | 128 | 237.2 | 220.9 | 1.074 |
| 128 | 256 | 435.2 | 407.0 | 1.069 |
| 128 | 512 | 838.0 | 784.1 | 1.069 |
| 64 | 1 (s_q 4096) | 822.9 | 778.3 | 1.057 |
| 128 | 1 (s_q 4096) | 1618.1 | 1524.6 | 1.061 |

### Against FlashInfer (0.7.1, `sparse_mla_sm120`, same data, same timing method)

Our columns here are from build a006306, measured in the same session as FlashInfer.

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

From T = 16 on, our exact mode is 1.7 to 3.1x faster than FlashInfer's. At T = 1, and at T = 4 with h = 64, FlashInfer's exact mode is still
1.1 to 1.4x faster than ours (its default mode is 1.1 to 1.4x faster than our `mxfp8`). About 6 us of our time there is the combine kernel. Under a CUDA Graph (`benchmarks/sm120/bench_decode_graph.py`), the bf16 time per layer of
h64 b1 s_q 1 goes from 71.8 to 17.8 us, and graph replay equals eager bit for bit for every shape and mode.

## Correctness checks

Every change was checked with the following:
* the upstream `tests/test-sparse-decode.py`: all 1534 cases. `bf16` and `fp8x2` pass all of them; `mxfp8` passes 253, the cases
  whose tolerance an approximate mode can meet.
* `tests/test-sparse-decode-sm120-mxfp4.py`: 856 cases of the MXFP4 cache, written after the upstream test, with its corner cases.
  These include page sizes 1 to 576, rows with no valid token, `topk_length`, sinks, and split and no split.
* `tests/test-sparse-decode-sm120-nvfp4.py`: 860 cases of the V4.1 FP4 main cache, with and without a V4.1 FP4 extra cache, with the
  corner cases of the MXFP4 test. `bf16` passes the upstream tolerances, `fp8x2` equals `bf16` bit for bit, `mxfp8` is checked against
  its emulator row.
* `tests/eval-sm120-modes.py`: a float64 emulator of every mode. The kernels' bf16 outputs equal the emulator's in 99.9 to 100% of the
  elements, on upstream data and on high-variance logits, which upstream data does not exercise (its attention is close to uniform).
* `benchmarks/sm120/bench_decode_graph.py`: CUDA Graph replay equals eager bit for bit for all production and small shapes.
* The builds have no register spills.

## Kernel design

`csrc/cuda_kernels/sm120/decode/sparse/`: `bf16.cu` (exact path, all formats but MXFP4), `mxfp8.cu` (`fp8x2` / `mxfp8` on the FP8 main
cache with its FP4 extra tiles, and `mxfp8` on the V4.1 FP4 main cache), `mxfp4.cu` (MXFP4 cache), `run.cu` (mode dispatch). The scheduler metadata and the combine kernel are upstream's (`smxx`).

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
  about a third of the issue cost of one bulk copy per row). So do the rows of the V4.1 FP4 main cache and those of the MXFP4 cache in `mxfp8` mode (`mxfp4x4` has no shared memory left for this layout), each row
  in a box a little wider than the row, with the odd groups of 4 rows shifted by 16 B so that the 8 rows an `ldmatrix` reads fall on 8
  different bank positions. The rows of a V4.1 FP4 extra cache next to an FP8 main cache use one `cp.async.bulk` each. Warps 1 to 3 rebase into 2 V
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
* `fp8x2` natively on V4.1 FP4 tiles (Q as four NVFP4 terms after a per-head power-of-two prescale, P as two terms, V as hi + lo e4m3
  terms): the third PV MMA and the room for the lo terms made it 5 to 11% slower than the `bf16` kernel it would replace, and rows with a
  single valid token still missed the upstream lse tolerance.
* PV on the FP4 MMA as well (SageAttention3-style: P as two-level NVFP4, V transposed and re-quantized to NVFP4 by the producers): the
  conversion costs the producers 2 to 3.5x the rebase, more than the faster MMA can save.
* `gather4` for the rows of a V4.1 FP4 extra cache next to an FP8 main cache: at most 1 to 3%, slower at h = 128, b = 512.

## Open items

* FP8 main cache with an MXFP4 extra cache, and the V4.1 FP4 main cache with an FP8 or MXFP4 extra cache: rejected with an error.
* `fp8x2` runs the bf16 kernel on V4.1 FP4 tiles (see above for the native attempt).
* At h = 128 and with FP4 tiles, the remaining time is the V rebase and the softmax. On the V4.1 FP4 main cache the rebase still costs
  3 to 8%; a lookup table does not make it cheaper. The two CTAs of h = 128 convert the same tiles, and sharing that work is untried.
* A second-level scale for the V4.1 FP4 cache (one power of two per cache, which would fold exactly into the logit scale and the rebase
  exponent) is not exposed: it needs an API argument, and it only matters for caches whose values are far outside the e4m3 scale range.
* The re-quantization of Q at the switch from main to FP4 extra tiles costs 2 to 3% at b = 512 and 5 to 7% at b = 64 to 128.
* Smallest batches (T <= 4, h = 64): the combine kernel and the per-CTA setup leave us 1.1 to 1.4x behind FlashInfer.
* The defaults are unchanged: `bf16`; `bf16` / `fp8x2` raise on an MXFP4 cache.
