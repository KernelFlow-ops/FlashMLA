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
`--small` (the 24 small-batch shapes b in {1, 4, 16} x s_q in {1, 4} x h x FP8/FP4 extra, T <= 64, which FlashInfer's decode
kernel takes directly; run it once with the default precision and once with `--precision bf16`),
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

## `bench_decode_graph.py` - the decode kernel under a CUDA Graph, and the graph-vs-eager bit check

A serving loop calls the decode kernel once per layer, with the layer's own q / KV cache / indices, usually inside a
CUDA Graph. `tests/test-sparse-decode.py` times one call after an 8 GB L2 flush; this script measures the multi-layer loop and
checks that the graph gives the same bits as eager.

```bash
export PYTHONPATH=<worktree>:<worktree>/tests          # the flash_mla of THIS tree must be the one that is imported
# timing: exclusive GPU. correctness only (capture + replay + bit comparison, no timing): add --no-time
python benchmarks/sm120/bench_decode_graph.py --production --secondary --csv out.csv --md out.md
```

Per shape it reports, per layer and in microseconds: the kineto time of one call (splitkv kernel, combine kernel if the shape
splits, e2e; the upstream test's method), the wall clock of an eager Python loop over the L layers, and the wall clock of
the L layers captured in one `torch.cuda.CUDAGraph` and replayed back to back. TFLOPS and GB/s use the counts of the upstream
test (`lib.count_flop_and_mem_vol_for_decode`) and are given as a percentage of the BF16 mma peak (252 TFLOPS, measured with `micro/mma_peak.cu`) and of
the random-gather limit (1.53 TB/s, measured with `micro/gather_bw.cu`). `bit_identical` compares out (`view(int16)`) and lse (`view(int32)`) of every layer
between graph replay and eager, after a replay into NaN-poisoned buffers and again after the timed replays; `eager_repeat`
does the same for two eager runs.

* L layers: every layer has its own data; L is the smallest number whose summed footprint (unique KV bytes + Q + O) exceeds the
  96 MB L2 (at least `--min-layers` 4, `--layers N` overrides). The tiniest shapes therefore use ~200 layers. If the data does
  not fit in GPU memory L is reduced and the row says so.
* Shapes: `--production` = the 16 upstream shapes (varlen KV lengths, as upstream). `--secondary` = `small` (b in {1, 4, 16} x
  s_q in {1, 4} x h x FP8/FP4 extra, all KV lengths full, so all 640 indices are valid), `topk2048` (topk 2048, no extra
  cache, b in {1, 4, 16, 64}: the split-KV kernel plus the combine kernel) and `longq` (b 1, s_q 4096).
* `eager` calls `lib.run_flash_mla_decode`, i.e. it includes the test wrapper's Python (slicing, argument building) and the
  `torch.empty` of the outputs. One eager call costs about 37 us of CPU time on the RTX 5090 test machine (the topk 2048 shapes at b 1 / 4,
  whose GPU time is shorter, are CPU-bound at 37.5 us per layer; it varies with the load of the host). When the GPU time of a
  call is longer than that (all production shapes) the launches are hidden and eager equals graph within 1-2%. Shorter calls are
  CPU-bound in eager: the small-batch shapes, which are split over the SMs on sm120 (15 to 56 us of GPU time, 43 to 46 us per eager
  call with the scheduler and combine launches), run 1.3 to 3x faster in the graph, and the topk 2048 shapes at b 1 / 4 18-27% faster.
* `--nvtx` wraps every layer (and every replay) in an nvtx range for nsys. Under a profiler the kineto step is skipped
  (CUPTI is taken); `nsys profile -t cuda,nvtx --cuda-graph-trace=node python ... --only h64_b64_fp8 --nvtx --no-kineto`
  lists the kernels of the graph nodes.
* `--only <substr,...>`, `--list`, `--repeats`, `--block-ms`, `--bf16-tflops`, `--gather-gbps`, `--no-graph`, `--no-kineto`.
* The exit code is 2 if any shape is not bit-identical (graph vs eager, or eager vs eager).
