"""
CUDA Graph benchmark of the sparse decode kernel (DeepSeek-V4.1, `flash_mla.flash_mla_with_kvcache`), with the
graph-vs-eager bit-for-bit check.

A real model calls the decode kernel once per layer, with the layer's own q / KV cache / indices, and runs the whole
forward pass under a CUDA Graph. The per-call kineto number of `tests/test-sparse-decode.py` (one call, 8 GB L2 flush
before it) leaves out the launch overhead and the state of the L2. This script measures what a serving loop sees.

Per shape (a row of the output)
  1. Generate L independent layers with `lib.generate_testcase_for_decode` (own q, KV caches, indices; seed =
     seed + 100000 * case index + layer). The reference data (de-quantized KV copies) is freed after generation.
     L = the smallest number of layers whose summed footprint (unique KV bytes read + Q + O, `lib.count_flop_and_mem_vol_for_decode`)
     exceeds the L2 cache (96 MB on the RTX 5090), but at least `--min-layers` (4); `--layers N` overrides it. If the
     data does not fit in GPU memory L is reduced and `layers_note` says so.
  2. `eager kineto`: layer 0, `kk.bench` (kineto; 8 GB L2 flush before every call), exactly the upstream test's
     method. splitkv / combine kernel time and e2e (first kernel start to last kernel end of one call).
  3. `eager wall`: the same L calls in a Python loop, repeated; wall clock / (passes * L), synchronized before and after.
     Contains the launch overhead (Python wrapper, torch.empty of the outputs, the kernel launches).
  4. `graph`: the L calls captured into ONE `torch.cuda.CUDAGraph`, replayed repeatedly; wall clock / (replays * L).
  5. Bit-for-bit check. Eager outputs are produced twice (`eager_repeat`); the graph's output buffers are poisoned with NaN
     and replayed once, and compared with the eager outputs after that replay and again after all timed replays:
     `out.view(int16)` and `lse.view(int32)` with `torch.equal`, for each layer.
  6. TFLOPS and GB/s with the counts of the upstream test (`lib.count_flop_and_mem_vol_for_decode`: 2 * h * attended tokens
     * (d_qk + d_v) FLOP; unique KV bytes + Q + O), as % of the BF16 mma peak (252 TFLOPS, micro/mma_peak.cu) and of the random-gather
     bandwidth limit (1.53 TB/s, micro/gather_bw.cu). The MXFP8 block-scaled peak is 4x the BF16 one.

Shapes
  --production  the 16 shapes of tests/test-sparse-decode.py: h in {64, 128} x b in {64, 128, 256, 512} x extra cache FP8/FP4,
                s_q 4, topk 128 + extra 512, page 256 / 64, varlen KV lengths (the upstream data)
  --secondary   small batch: b in {1, 4, 16} x s_q in {1, 4} x h in {64, 128}, 128 + 512, extra FP8/FP4 (`small`; all KV lengths full);
                topk 2048 without extra cache, b in {1, 4, 16, 64}, s_q 1 (`topk2048`; split-KV + combine kernel);
                b 1, s_q 4096, 128 + 512 (`longq`; the upstream "prefill as decode" shapes)

The sched-meta object of every layer is created by the first (eager) call and reused (split-KV shapes only; the
production shapes do not split: topk + extra_topk <= 640). The data does not have to be re-created for the timing.

Run (the GPU has to be free of other work: take an exclusive lock around timing runs; `--no-time` is the
correctness-only mode and can run under a shared lock):
  python benchmarks/sm120/bench_decode_graph.py --production --secondary --csv out.csv --md out.md
nsys: add `--only h64_b64_fp8 --nvtx`; under nsys the kineto step is skipped (kineto and nsys cannot share CUPTI).
"""

import argparse
import csv
import dataclasses
import gc
import math
import os
import subprocess
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

_TESTS_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests"))
sys.path.insert(0, _TESTS_DIR)

import torch
import kernelkit as kk

import flash_mla
import lib
import quant
from lib import RawTestParamForDecode as RawTestParam

FP8, FP4 = quant.KVCacheLayout.V41_FP8Sparse, quant.KVCacheLayout.V41_FP4Sparse
SPLITKV_KERNEL = "flash_fwd_splitkv_mla_fp8_sparse_kernel"
COMBINE_KERNEL = "flash_fwd_mla_combine_kernel"
FLUSH_BYTES = 8_000_000_000       # kk.bench allocates this much for the L2 flush before every call


# ---------------------------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------------------------

@dataclasses.dataclass
class Case:
    name: str
    group: str      # "production" | "secondary"
    kind: str       # "production" | "small" | "topk2048" | "longq"
    raw: RawTestParam
    index: int = 0  # position in the full list (before --only); fixes the seeds


def production_cases() -> List[Case]:
    """The 16 production shapes of tests/test-sparse-decode.py (`base_and_bszs`), same as bench_flashinfer.py."""
    cases = []
    for extra_layout in (None, FP4):
        for h in (64, 128):
            for b in (64, 128, 256, 512):
                raw = RawTestParam(b, h, 4, 1, 256, True, topk=128, d_qk=512, extra_s_k=2048, extra_topk=512,
                                   block_size=256, extra_block_size=64, kvcache_layout=FP8, extra_kvcache_layout=extra_layout)
                cases.append(Case(f"h{h}_b{b}_{'fp4' if extra_layout else 'fp8'}", "production", "production", raw))
    return cases


def small_cases() -> List[Case]:
    """b in {1, 4, 16} x s_q in {1, 4} x h in {64, 128}, 128 + 512, extra FP8 / FP4 (production page sizes; every request has
    its full 256 / 2048 tokens, so all 640 indices are valid). Same list as bench_flashinfer.py --small."""
    cases = []
    for extra_layout in (None, FP4):
        for h in (64, 128):
            for s_q in (1, 4):
                for b in (1, 4, 16):
                    raw = RawTestParam(b, h, s_q, 1, 256, False, topk=128, d_qk=512, extra_s_k=2048, extra_topk=512,
                                       block_size=256, extra_block_size=64, kvcache_layout=FP8, extra_kvcache_layout=extra_layout)
                    cases.append(Case(f"h{h}_b{b}_sq{s_q}_{'fp4' if extra_layout else 'fp8'}", "secondary", "small", raw))
    return cases


def topk2048_cases() -> List[Case]:
    """topk 2048, no extra cache: 2048 > 640 so the call is the split-KV kernel + the combine kernel."""
    cases = []
    for h in (64, 128):
        for b in (1, 4, 16, 64):
            raw = RawTestParam(b, h, 1, 1, 4096, False, topk=2048, d_qk=512, block_size=64, kvcache_layout=FP8)
            cases.append(Case(f"h{h}_b{b}_topk2048", "secondary", "topk2048", raw))
    return cases


def longq_cases() -> List[Case]:
    """b 1, s_q 4096 (the upstream 'prefill as decode' shapes), 128 + 512."""
    cases = []
    for extra_layout in (FP8, FP4):
        for h in (64, 128):
            raw = RawTestParam(1, h, 4096, 1, 4096, False, 128, extra_s_k=4096, extra_topk=512, extra_block_size=64, d_qk=512,
                               kvcache_layout=FP8, extra_kvcache_layout=extra_layout)
            cases.append(Case(f"h{h}_b1_sq4096_{'fp4' if extra_layout is FP4 else 'fp8'}", "secondary", "longq", raw))
    return cases


def all_cases() -> List[Case]:
    cases = production_cases() + small_cases() + topk2048_cases() + longq_cases()
    for i, c in enumerate(cases):
        c.index = i
    return cases


# ---------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------

def tensor_bytes(x: Optional[torch.Tensor]) -> int:
    return 0 if x is None else x.numel() * x.element_size()


def strip_reference_data(t: lib.TestcaseForDecode):
    """Free what only the reference needs (the de-quantized bf16 copy of the caches, the NaN-masked originals)."""
    for scope in (t.kv_scope, t.extra_kv_scope):
        if scope is not None:
            scope.blocked_k = None
            scope.abs_indices = None
            scope.block_table = None


def alloc_bytes(t: lib.TestcaseForDecode) -> int:
    n = tensor_bytes(t.q) + tensor_bytes(t.attn_sink)
    for scope in (t.kv_scope, t.extra_kv_scope):
        if scope is not None:
            n += tensor_bytes(scope.blocked_k_quantized) + tensor_bytes(scope.indices_in_kvcache) + tensor_bytes(scope.topk_length)
    return n


class Layers:
    """The L layers of one case: params, testcases, sched metas and the flop / byte counts."""

    def __init__(self):
        self.p: List[lib.TestParam] = []
        self.t: List[lib.TestcaseForDecode] = []
        self.meta = []
        self.flop: List[float] = []
        self.mem_vol: List[float] = []
        self.nvtx = False

    def add(self, case: Case, seed: int):
        raw = dataclasses.replace(case.raw, seed=seed)
        p = raw.to_test_param()
        assert p.can_run_on_and_clamp(lib.TestTarget.DECODE)
        t = lib.generate_testcase_for_decode(p)
        st = lib.count_flop_and_mem_vol_for_decode(p, t)
        strip_reference_data(t)
        self.p.append(p)
        self.t.append(t)
        self.meta.append(flash_mla.get_mla_metadata()[0])
        self.flop.append(st.flop)
        self.mem_vol.append(st.mem_vol)
        torch.cuda.empty_cache()

    def __len__(self):
        return len(self.t)

    def run(self, l: int):
        if self.nvtx:
            torch.cuda.nvtx.range_push(f"layer{l}")
        r = lib.run_flash_mla_decode(self.p[l], self.t[l], self.meta[l], None)
        if self.nvtx:
            torch.cuda.nvtx.range_pop()
        return r

    def run_all(self):
        for l in range(len(self.t)):
            self.run(l)


def choose_layers(layers: Layers, args, l2_bytes: int) -> Tuple[int, str]:
    """Decide L from layer 0 (already generated). Returns (L, note)."""
    p, t = layers.p[0], layers.t[0]
    mem_vol = layers.mem_vol[0]
    per_layer_alloc = alloc_bytes(t)
    out_bytes = p.decode.b * p.s_q * p.h_q * (p.d_v * 2 + 4)

    if args.layers:
        want, note = args.layers, "fixed by --layers"
    else:
        want = max(args.min_layers, int(l2_bytes // mem_vol) + 1)
        note = ""
    # memory: data + the eager outputs kept for the comparison + the graph's outputs; kk.bench needs its flush buffer
    total = torch.cuda.get_device_properties(0).total_memory
    budget = total - (FLUSH_BYTES if args.time and args.kineto else 0) - args.mem_reserve_gb * 2**30
    per_layer = per_layer_alloc + 2 * out_bytes
    fit = int((budget - 2 * out_bytes) // per_layer)
    if fit < want:
        if fit < 1:
            raise MemoryError(f"even one layer does not fit ({per_layer / 2**30:.2f} GiB per layer)")
        note = f"reduced from {want} to {fit} (GPU memory)"
        want = fit
    elif not note and want > args.min_layers:
        note = f"footprint {mem_vol / 2**20:.2f} MiB/layer"
    return want, note


# ---------------------------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------------------------

_flush_names: Optional[set] = None


def kineto_measure(fn: Callable[[], object], num_runs: int, repeats: int) -> Dict:
    """kk.bench of `fn` (one decode call), as tests/test-sparse-decode.py does; median of `repeats` measurements."""
    global _flush_names
    if _flush_names is None:
        _flush_names = set(kk.bench(lambda: None, num_runs).get_kernel_names())
    splitkv, combine, e2e, other = [], [], [], set()
    for _ in range(repeats):
        res = kk.bench(fn, num_runs)
        names = res.get_kernel_names()
        sk = [n for n in names if SPLITKV_KERNEL in n]
        ck = [n for n in names if COMBINE_KERNEL in n]
        if len(sk) != 1 or len(ck) > 1:
            raise RuntimeError(f"unexpected kernels in the timed call: {names}")
        other |= {n for n in names if n not in _flush_names and n not in sk and n not in ck and not n.startswith("cuda")}
        splitkv.append(res.get_kernel_time(SPLITKV_KERNEL) * 1e6)
        if ck:
            combine.append(res.get_kernel_time(COMBINE_KERNEL) * 1e6)
            e2e.append(res.get_e2e_time([SPLITKV_KERNEL, COMBINE_KERNEL]) * 1e6)
        else:
            e2e.append(splitkv[-1])
    return dict(splitkv_us=median(splitkv), combine_us=median(combine) if combine else None, e2e_us=median(e2e),
                e2e_spread_pct=spread_pct(e2e), other_kernels=sorted(_strip(n) for n in other))


def _strip(name: str) -> str:
    depth = 0
    for i, ch in enumerate(name):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "(" and depth == 0:
            return name[:i].removeprefix("void ").strip()
    return name


def median(xs: List[float]) -> float:
    s = sorted(xs)
    return s[len(s) // 2] if len(s) % 2 else 0.5 * (s[len(s) // 2 - 1] + s[len(s) // 2])


def spread_pct(xs: List[float]) -> float:
    return (max(xs) - min(xs)) / median(xs) * 100.0 if len(xs) > 1 else 0.0


def time_wall(call: Callable[[], None], units_per_call: int, args) -> Dict:
    """Wall clock per unit (= per layer) of `call`: warm up for `warmup_s` seconds, pick the number of calls so that one
    block lasts about `block_ms`, then time `repeats` blocks (synchronized before and after each)."""
    torch.cuda.synchronize()
    t_end = time.perf_counter() + args.warmup_s
    while time.perf_counter() < t_end:
        for _ in range(4):
            call()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    est = (time.perf_counter() - t0) / 3
    n = int(min(max(math.ceil(args.block_ms * 1e-3 / est), 3), 100000))
    blocks = []
    for _ in range(args.repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            call()
        torch.cuda.synchronize()
        blocks.append((time.perf_counter() - t0) / (n * units_per_call) * 1e6)
    return dict(us=median(blocks), spread_pct=spread_pct(blocks), calls_per_block=n, blocks=blocks)


def sample_gpu_state(busy: Callable[[], None]) -> Dict:
    """SM clock / temperature / power while the GPU is busy: `busy` enqueues about half a second of work, then nvidia-smi."""
    try:
        busy()
        r = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        torch.cuda.synchronize()
        clk, temp, pw = [float(x) for x in r.stdout.strip().split(",")]
        return dict(sm_mhz=clk, temp_c=temp, power_w=pw)
    except Exception as e:      # pragma: no cover
        torch.cuda.synchronize()
        print(f"  warning: could not sample the GPU state: {e}", flush=True)
        return dict(sm_mhz=None, temp_c=None, power_w=None)


# ---------------------------------------------------------------------------------------------
# Bit-for-bit comparison
# ---------------------------------------------------------------------------------------------

def bit_view(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.bfloat16:
        return x.view(torch.int16)
    if x.dtype == torch.float32:
        return x.view(torch.int32)
    raise TypeError(x.dtype)


def bit_mismatches(a: torch.Tensor, b: torch.Tensor) -> int:
    """Number of elements whose bit patterns differ (-1 if the shapes differ)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return -1
    va, vb = bit_view(a), bit_view(b)
    if torch.equal(va, vb):
        return 0
    return int((va != vb).sum().item())


def compare_layers(ref_outs, ref_lses, outs, lses) -> Tuple[bool, List[str]]:
    """Compare per layer; returns (all identical, ['layer 3 out 1234 elements', ...])."""
    bad = []
    for l, (ro, rl, o, ls) in enumerate(zip(ref_outs, ref_lses, outs, lses)):
        n = bit_mismatches(ro, o)
        if n:
            bad.append(f"layer {l} out {n} elements")
        n = bit_mismatches(rl, ls)
        if n:
            bad.append(f"layer {l} lse {n} elements")
    return not bad, bad


# ---------------------------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------------------------

def run_case(case: Case, args, l2_bytes: int) -> Dict:
    raw = case.raw
    row: Dict = dict(
        case=case.name, group=case.group, kind=case.kind, h=raw.h_q, b=raw.b, s_q=raw.s_q, topk=raw.topk,
        extra_topk=raw.extra_topk or 0, extra_fmt=("fp4" if raw.extra_kvcache_layout is FP4 else "fp8") if raw.extra_topk else "-",
        page=raw.block_size, extra_page=raw.extra_block_size or 0, varlen=int(raw.is_varlen), seed=args.seed + 100000 * case.index,
    )
    # ---- data: layer 0 first (its footprint decides L), then the others
    layers = Layers()
    layers.nvtx = args.nvtx
    t0 = time.perf_counter()
    layers.add(case, args.seed + 100000 * case.index)
    L, note = choose_layers(layers, args, l2_bytes)
    row["layers"], row["layers_note"] = L, note
    for l in range(1, L):
        layers.add(case, args.seed + 100000 * case.index + l)
    row["gen_s"] = time.perf_counter() - t0
    row["flop_per_layer"] = sum(layers.flop) / L
    row["mem_vol_per_layer"] = sum(layers.mem_vol) / L
    row["footprint_total_mib"] = sum(layers.mem_vol) / 2**20
    row["split_kv"] = int(raw.topk + (raw.extra_topk or 0) > 640)

    # ---- eager: first call of every layer (creates the sched metas), outputs kept; second call compared bit-for-bit
    outs, lses = [], []
    eager_repeat_bad: List[str] = []
    for l in range(L):
        o, ls = layers.run(l)
        outs.append(o)
        lses.append(ls)
    torch.cuda.synchronize()
    for l in range(L):
        o2, l2 = layers.run(l)
        n = bit_mismatches(outs[l], o2)
        if n:
            eager_repeat_bad.append(f"layer {l} out {n} elements")
        n = bit_mismatches(lses[l], l2)
        if n:
            eager_repeat_bad.append(f"layer {l} lse {n} elements")
        del o2, l2
    torch.cuda.synchronize()
    row["eager_repeat_identical"] = int(not eager_repeat_bad)
    out_finite = all(bool(torch.isfinite(o.float()).all().item()) for o in outs) if args.finite_check else None
    row["out_finite"] = None if out_finite is None else int(out_finite)

    # ---- eager, kineto (layer 0 only; what tests/test-sparse-decode.py measures)
    use_kineto = args.time and args.kineto and not kk.is_using_profiling_tools()
    if use_kineto:
        km = kineto_measure(lambda: layers.run(0), args.num_runs, args.bench_repeats)
        row.update(kineto_splitkv_us=km["splitkv_us"], kineto_combine_us=km["combine_us"], kineto_e2e_us=km["e2e_us"],
                   kineto_spread_pct=km["e2e_spread_pct"], other_kernels="; ".join(km["other_kernels"]))

    # ---- eager, wall clock of the L-layer loop
    if args.time:
        ew = time_wall(layers.run_all, L, args)
        row.update(eager_wall_us=ew["us"], eager_wall_spread_pct=ew["spread_pct"], eager_passes=ew["calls_per_block"])

    # ---- CUDA graph
    g = None
    graph_outs: List[torch.Tensor] = []
    graph_lses: List[torch.Tensor] = []
    if args.graph:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            layers.run_all()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for l in range(L):
                o, ls = layers.run(l)
                graph_outs.append(o)
                graph_lses.append(ls)
        for o, ls in zip(graph_outs, graph_lses):       # poison: the replay has to write every element
            o.fill_(float("nan"))
            ls.fill_(float("nan"))
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        ok1, bad1 = compare_layers(outs, lses, graph_outs, graph_lses)

        def replay():
            if args.nvtx:
                torch.cuda.nvtx.range_push("graph_replay")
            g.replay()
            if args.nvtx:
                torch.cuda.nvtx.range_pop()

        if args.time:
            gw = time_wall(replay, L, args)
            row.update(graph_us=gw["us"], graph_spread_pct=gw["spread_pct"], graph_replays=gw["calls_per_block"])
            n_busy = max(1, int(0.5e6 / (gw["us"] * L)))

            def busy():
                for _ in range(n_busy):
                    g.replay()
            row.update(sample_gpu_state(busy))
        # replay after the timed replays: still the same bits
        torch.cuda.synchronize()
        ok2, bad2 = compare_layers(outs, lses, graph_outs, graph_lses)
        row["bit_identical"] = int(ok1 and ok2)
        row["bit_identical_first_replay"] = int(ok1)
        row["bit_identical_after_timing"] = int(ok2)
        row["mismatches"] = "; ".join(bad1 or bad2)
    row["eager_repeat_mismatches"] = "; ".join(eager_repeat_bad)

    # ---- derived numbers
    flop, mem = row["flop_per_layer"], row["mem_vol_per_layer"]

    def rates(prefix: str, us: Optional[float]):
        if not us:
            return
        row[f"{prefix}_tflops"] = flop / (us * 1e-6) / 1e12
        row[f"{prefix}_gbps"] = mem / (us * 1e-6) / 1e9
        row[f"{prefix}_pct_bf16"] = row[f"{prefix}_tflops"] / args.bf16_tflops * 100.0
        row[f"{prefix}_pct_gather"] = row[f"{prefix}_gbps"] / args.gather_gbps * 100.0

    rates("kineto", row.get("kineto_e2e_us"))
    rates("eager", row.get("eager_wall_us"))
    rates("graph", row.get("graph_us"))
    if row.get("graph_us") and row.get("eager_wall_us"):
        row["graph_over_eager"] = row["graph_us"] / row["eager_wall_us"]
    if row.get("graph_us") and row.get("kineto_e2e_us"):
        row["graph_over_kineto"] = row["graph_us"] / row["kineto_e2e_us"]

    del g, graph_outs, graph_lses, outs, lses, layers
    gc.collect()
    torch.cuda.empty_cache()
    return row


def print_row(row: Dict):
    def f(key, spec=".1f", unit=""):
        v = row.get(key)
        return "-" if v is None else format(v, spec) + unit

    parts = [f"  L={row['layers']:<3d}"]
    if "kineto_e2e_us" in row:
        comb = f" (splitkv {f('kineto_splitkv_us')} + combine {f('kineto_combine_us')})" if row.get("kineto_combine_us") else ""
        parts.append(f"kineto {f('kineto_e2e_us')} us{comb} {f('kineto_tflops', '.0f')} TFLOPS {f('kineto_pct_bf16', '.0f')}% bf16")
    if "eager_wall_us" in row:
        parts.append(f"eager {f('eager_wall_us')} us")
    if "graph_us" in row:
        parts.append(f"graph {f('graph_us')} us ({f('graph_over_eager', '.3f')}x eager) {f('graph_tflops', '.0f')} TFLOPS "
                     f"{f('graph_pct_bf16', '.0f')}% bf16 {f('graph_gbps', '.0f')} GB/s {f('graph_pct_gather', '.0f')}% gather")
    if "bit_identical" in row:
        parts.append("bit-identical " + ("YES" if row["bit_identical"] else "NO: " + row.get("mismatches", "")))
    parts.append("eager x2 " + ("same" if row["eager_repeat_identical"] else "DIFFERENT: " + row["eager_repeat_mismatches"]))
    if row.get("layers_note"):
        parts.append(f"[{row['layers_note']}]")
    print(" | ".join(parts), flush=True)


# ---------------------------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------------------------

CSV_COLUMNS = [
    "case", "group", "kind", "h", "b", "s_q", "topk", "extra_topk", "extra_fmt", "page", "extra_page", "varlen", "split_kv", "seed",
    "layers", "layers_note", "footprint_total_mib", "flop_per_layer", "mem_vol_per_layer",
    "kineto_splitkv_us", "kineto_combine_us", "kineto_e2e_us", "kineto_spread_pct", "other_kernels",
    "eager_wall_us", "eager_wall_spread_pct", "eager_passes", "graph_us", "graph_spread_pct", "graph_replays",
    "graph_over_eager", "graph_over_kineto",
    "kineto_tflops", "kineto_gbps", "kineto_pct_bf16", "kineto_pct_gather",
    "eager_tflops", "eager_gbps", "eager_pct_bf16", "eager_pct_gather",
    "graph_tflops", "graph_gbps", "graph_pct_bf16", "graph_pct_gather",
    "bit_identical", "bit_identical_first_replay", "bit_identical_after_timing", "mismatches",
    "eager_repeat_identical", "eager_repeat_mismatches", "out_finite", "sm_mhz", "temp_c", "power_w", "gen_s",
]


def write_csv(path: str, rows: List[Dict]):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})


def write_markdown(path: str, rows: List[Dict], args, cmd: str):
    def f(row, key, spec=".1f"):
        v = row.get(key)
        return "-" if v is None else format(v, spec)

    def yn(row, key):
        v = row.get(key)
        return "-" if v is None else ("yes" if v else "NO")

    gpu = rows[0] if rows else {}
    lines = [
        "# Sparse decode under a CUDA Graph (sm120)",
        "",
        f"- GPU {torch.cuda.get_device_name(0)}, torch {torch.__version__}, date {time.strftime('%Y-%m-%d %H:%M')}; command `{cmd}`",
        f"- `kineto` = layer 0 alone, `kk.bench` ({args.num_runs} runs, 8 GB L2 flush before each, median of {args.bench_repeats} measurements): "
        "e2e = first kernel start to last kernel end of one call (the upstream test's time). `eager` = the same L layers in a Python loop "
        f"(`lib.run_flash_mla_decode`), wall clock per layer, median of {args.repeats} blocks of about {args.block_ms:.0f} ms. "
        "`graph` = the L layers captured in one `torch.cuda.CUDAGraph`, wall clock per layer of back-to-back replays (same blocks).",
        f"- L layers = smallest number whose summed footprint (unique KV bytes + Q + O) exceeds the L2 ({args.l2_mib:.0f} MiB), at least {args.min_layers}; "
        "every layer has its own q / KV / indices. TFLOPS and GB/s with the counts of tests/test-sparse-decode.py; "
        f"%bf16 relative to {args.bf16_tflops:g} TFLOPS (micro/mma_peak.cu), %gather relative to {args.gather_gbps:g} GB/s (micro/gather_bw.cu).",
        "- `bit-identical` = graph replay equals eager bit for bit (`out` as int16, `lse` as int32, every layer; after the first replay "
        "into NaN-poisoned buffers and again after the timed replays). `eager x2` = two eager runs equal bit for bit.",
        "",
    ]
    for title, groups in (("Production shapes", ("production",)), ("Secondary shapes", ("secondary",))):
        g = [r for r in rows if r["group"] in groups]
        if not g:
            continue
        lines += [f"## {title}", "",
                  "| case | kind | L | kineto splitkv us | kineto combine us | kineto e2e us | eager us/layer | graph us/layer | graph / eager | graph / kineto | "
                  "TFLOPS kineto | TFLOPS graph | %bf16 graph | GB/s graph | %gather graph | bit-identical | eager x2 | SM MHz | temp C | W |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in g:
            lines.append(
                f"| {r['case']} | {r['kind']} | {r['layers']} | {f(r, 'kineto_splitkv_us')} | {f(r, 'kineto_combine_us')} | {f(r, 'kineto_e2e_us')} | "
                f"{f(r, 'eager_wall_us')} | {f(r, 'graph_us')} | {f(r, 'graph_over_eager', '.3f')} | {f(r, 'graph_over_kineto', '.3f')} | "
                f"{f(r, 'kineto_tflops', '.0f')} | {f(r, 'graph_tflops', '.0f')} | {f(r, 'graph_pct_bf16', '.0f')} | {f(r, 'graph_gbps', '.0f')} | "
                f"{f(r, 'graph_pct_gather', '.0f')} | {yn(r, 'bit_identical')} | {yn(r, 'eager_repeat_identical')} | "
                f"{f(r, 'sm_mhz', '.0f')} | {f(r, 'temp_c', '.0f')} | {f(r, 'power_w', '.0f')} |")
        lines.append("")
        notes = [r for r in g if r.get("layers_note") or r.get("mismatches") or r.get("eager_repeat_mismatches") or r.get("other_kernels")]
        if notes:
            lines += ["Notes:", ""]
            for r in notes:
                bits = []
                if r.get("layers_note"):
                    bits.append(f"layers: {r['layers_note']}")
                if r.get("mismatches"):
                    bits.append(f"graph vs eager mismatches: {r['mismatches']}")
                if r.get("eager_repeat_mismatches"):
                    bits.append(f"eager vs eager mismatches: {r['eager_repeat_mismatches']}")
                if r.get("other_kernels"):
                    bits.append(f"other kernels in the call: {r['other_kernels']}")
                lines.append(f"- {r['case']}: " + "; ".join(bits))
            lines.append("")
    with open(path, "w", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--production", action="store_true", help="the 16 production shapes")
    parser.add_argument("--secondary", action="store_true", help="small batch / topk 2048 / s_q 4096 shapes (see the module doc)")
    parser.add_argument("--only", type=str, default=None, help="comma separated substrings; keep the cases whose name contains one of them")
    parser.add_argument("--list", action="store_true", help="print the selected cases and exit")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--md", type=str, default=None)
    parser.add_argument("--layers", type=int, default=0, help="number of layers per case (default: enough to exceed the L2, at least --min-layers)")
    parser.add_argument("--min-layers", type=int, default=4)
    parser.add_argument("--mem-reserve-gb", type=float, default=1.5, help="GPU memory left free besides the 8 GB flush buffer of kk.bench")
    parser.add_argument("--no-time", dest="time", action="store_false", help="correctness only (capture, replay, bit comparison); no timing, can run under a shared lock")
    parser.add_argument("--no-kineto", dest="kineto", action="store_false", help="skip the kineto measurement (automatic under nsys / ncu)")
    parser.add_argument("--no-graph", dest="graph", action="store_false", help="skip the CUDA graph (eager only)")
    parser.add_argument("--nvtx", action="store_true", help="wrap every layer in an nvtx range (and every replay in 'graph_replay'); for nsys")
    parser.add_argument("--num-runs", type=int, default=10, help="runs per kk.bench call (the upstream test uses 10)")
    parser.add_argument("--bench-repeats", type=int, default=3, help="separate kk.bench calls; the median is reported")
    parser.add_argument("--repeats", type=int, default=7, help="timed blocks for the eager loop and for the graph; the median is reported")
    parser.add_argument("--block-ms", type=float, default=150.0, help="target duration of one timed block")
    parser.add_argument("--warmup-s", type=float, default=0.4, help="seconds of warm-up before each wall-clock measurement (clock ramp)")
    parser.add_argument("--bf16-tflops", type=float, default=252.0, help="BF16 mma peak, from micro/mma_peak.cu")
    parser.add_argument("--gather-gbps", type=float, default=1530.0, help="random gather bandwidth limit, from micro/gather_bw.cu")
    parser.add_argument("--no-finite-check", dest="finite_check", action="store_false", help="skip the check that the eager outputs are finite")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cases: List[Case] = []
    full = all_cases()
    if args.production:
        cases += [c for c in full if c.group == "production"]
    if args.secondary:
        cases += [c for c in full if c.group == "secondary"]
    if args.only:
        keys = args.only.split(",")
        cases = [c for c in cases if any(k in c.name for k in keys)]
    if not cases:
        parser.error("no cases: pass --production and/or --secondary (and check --only)")
    if args.list:
        for c in cases:
            print(c.name, c.kind)
        return

    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    props = torch.cuda.get_device_properties(0)
    l2_bytes = getattr(props, "L2_cache_size", 0) or 96 * 2**20
    args.l2_mib = l2_bytes / 2**20

    print(f"{props.name}, sm_{props.major}{props.minor}, L2 {args.l2_mib:.0f} MiB, torch {torch.__version__}, {len(cases)} cases, "
          f"timing {'on' if args.time else 'off'}", flush=True)
    if args.time:
        print("note: the times are only valid if nothing else uses the GPU during the run (hold the exclusive lock)", flush=True)
    if args.time and args.kineto and kk.is_using_profiling_tools():
        print("note: running under a profiler; the kineto step is skipped", flush=True)

    rows: List[Dict] = []
    with torch.inference_mode():
        for i, case in enumerate(cases):
            print(f"[{i + 1}/{len(cases)}] {case.name}  ({case.kind}) b={case.raw.b} h={case.raw.h_q} s_q={case.raw.s_q}", flush=True)
            try:
                row = run_case(case, args, l2_bytes)
            except MemoryError as e:
                print(f"  skipped: {e}", flush=True)
                continue
            rows.append(row)
            print_row(row)

    cmd = " ".join(["python", "benchmarks/sm120/bench_decode_graph.py"] + sys.argv[1:])
    if args.csv:
        write_csv(args.csv, rows)
        print(f"wrote {args.csv}")
    if args.md:
        write_markdown(args.md, rows, args, cmd)
        print(f"wrote {args.md}")
    bad = [r["case"] for r in rows if "bit_identical" in r and not r["bit_identical"]]
    bad_eager = [r["case"] for r in rows if not r["eager_repeat_identical"]]
    if bad or bad_eager:
        print(f"NOT bit-identical (graph vs eager): {bad}; eager vs eager differs: {bad_eager}", flush=True)
        sys.exit(2)
    print("all shapes: graph replay == eager bit for bit" if args.graph else "done")


if __name__ == "__main__":
    main()
