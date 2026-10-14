# sm120 (RTX 5090) benchmarks

## `bench_flashinfer.py` - FlashInfer as the external baseline

Runs FlashInfer 0.7.1's `sparse_mla_sm120` (the DeepSeek-V4.1 sparse MLA decode of its `SparseMLASm120Wrapper`)
on the data of `tests/test-sparse-decode.py` and reports, per shape, its CUDA kernels with their times and its
error against `tests/ref.py`. The times are the speed to beat; the errors are the bar for the quantized fast modes
("not worse than FlashInfer on the same input").

It needs a built `flash_mla` (`tests/lib.py` imports it) and FlashInfer 0.7.1 (`flashinfer-python`). FlashInfer
compiles its kernels on first use, so nvcc has to be on the path. If FlashInfer should stay out of the Python
environment, install it into a directory of its own (`pip install --target <dir> --no-deps flashinfer-python` plus
its dependencies) and put that directory on `PYTHONPATH`:

```bash
export PYTHONPATH=<dir with flashinfer> FLASHINFER_WORKSPACE_BASE=<dir for its JIT cache> \
       CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
# Nothing else may use the GPU while this times; error-only runs (--no-time) are not sensitive to that
python benchmarks/sm120/bench_flashinfer.py --production --features --csv out.csv --md out.md
```

Options: `--production` (the 16 production shapes: h in {64, 128}, b in {64, 128, 256, 512}, s_q 4, topk 128 +
extra 512, FP8 or FP4 extra cache), `--features` (4 small shapes with page sizes 61 / 123, `topk_length`,
`extra_topk_length`, `attn_sink`; errors only), `--empty-rows` (2 shapes with rows that have no valid KV),
`--only <substr,...>`, `--no-time`, `--precision {default,fp8,bf16}` (the wrapper's `compute_precision`),
`--num-runs` (kk.bench runs per measurement, 10 as upstream), `--bench-repeats` (separate measurements, median
reported), `--no-chunk64`.

Things to know before comparing numbers:

* **Cache layout.** FlashMLA stores a token as `[data, scales]`; FlashInfer's DSV4_1 expects the scales as a footer
  of the page (`[page x data | page x scales]`). The script permutes the bytes once, outside the timed region. Passing
  the FlashMLA cache as is returns an all-zero output and lse = -inf, without an error.
* **Paths.** FlashInfer's decode kernels take at most 64 tokens per call, so the production shapes (T = b * s_q =
  256..2048) reach its prefill kernel through the wrapper (`default`). That kernel has no h = 128 FP8 instantiation
  and the call raises; the table then has no `default` row for those shapes. `chunk64` calls the wrapper on 64-token
  slices (decode kernels, any h); its `sum` is the GPU time and its `span` also contains the launch gaps.
* **Timing columns.** `span` = first kernel start to last kernel end of one call, what the upstream test calls e2e;
  `sum` = sum of the kernel durations. The kernels of `kk.bench`'s own L2 flush are excluded.
* **LSE.** FlashInfer's LSE is base 2 and includes `attn_sink`; the script asks for the natural log and converts the
  reference (`logaddexp(lse_ref, sink)`, the +inf sentinel of empty rows becomes -inf first).
* **attn_sink = +inf** (present in the test data): FlashInfer's prefill kernel returns NaN for those heads and its
  decode kernel returns lse = -inf. The headline errors exclude such heads; the anomalous elements over all heads are
  reported in separate columns.
* **Accuracy classes.** The default mode computes with FP8 operands and misses the upstream out tolerance by 50x
  (cos_diff about 2.5e-4 with an FP8 extra cache, 4e-4 with FP4). `--precision bf16` passes it on every shape, which
  is also the check that the conventions above are right.
