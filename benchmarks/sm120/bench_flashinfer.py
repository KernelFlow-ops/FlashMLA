"""
External baseline for the sm120 sparse decode kernel: FlashInfer's `sparse_mla_sm120` run on the
FlashMLA decode test data, with its time and its error against `tests/ref.py`.

What it does, per shape
  1. Generates the testcase with `lib.generate_testcase_for_decode` (the same data as
     `tests/test-sparse-decode.py`: KV quantized then de-quantized for the reference, unused KV rows
     NaN, invalid indices -1, attn_sink with +-inf entries).
  2. Re-lays the quantized caches out for FlashInfer and calls it (see "Conventions").
  3. Compares its out / lse with `ref.ref_sparse_attn_decode`.
  4. Times the call with `kk.bench` (kineto, 8 GB L2 flush before every call) and lists the CUDA
     kernels of one call with their times.

Conventions (each one of these makes the numbers meaningless if wrong)
  * KV layout. FlashMLA (main, 2026-09-30) stores each token as [data, scales]: 512 B e4m3 + 16 B
    ue8m0 (FP8), 256 B e2m1 + 32 B e4m3 (FP4). FlashInfer 0.7.1 (DSV4_1, `kv_scale_format=
    "ue8m0_g32"`) expects the scales as a footer of the PAGE: [page_size x data | page_size x
    scales]. The two are a byte permutation of each other, so the script converts the cache once,
    outside of the timed region; no value changes.
  * Tokens. FlashMLA's [b, s_q] is flattened into T = b * s_q tokens; `topk_length` and
    `extra_topk_length` ([b]) are repeated s_q times.
  * LSE. FlashInfer's LSE is base 2 and (with attn_sink) includes the sink:
    lse' = log(exp(lse) + exp(sink)). The script asks for the natural log (`lse_scale = ln 2`) and
    converts the REFERENCE to FlashInfer's convention: lse_ref' = logaddexp(lse_ref, sink), where
    FlashMLA's "no valid KV" sentinel (+inf) is first turned back into -inf. Output is unaffected.
  * attn_sink = +inf. The test data has heads with sink = +inf (output must be 0). FlashInfer's
    prefill kernel returns NaN for those heads and its decode kernel returns lse = -inf instead of
    +inf. The headline errors therefore exclude heads whose sink is +inf ("clean heads"); the number
    of anomalous (NaN / inf mismatching) elements over ALL heads is reported separately.
  * Paths. `default` = one `SparseMLASm120Wrapper.run` call on all T tokens. FlashInfer's decode
    kernels only take T <= 64 tokens; above that the wrapper uses its prefill kernel, which has no
    h = 128 FP8 instantiation (the call then raises). `chunk64` = the same wrapper called on
    consecutive slices of 64 tokens, i.e. the decode kernels for any T and h; its kernel time is the
    sum over the slices and its wall span also contains the CPU gaps between the launches.

`--small` adds 24 small-batch shapes (b in {1, 4, 16} x s_q in {1, 4} x h in {64, 128} x extra FP8 / FP4, T <= 64), where the
wrapper calls FlashInfer's decode kernel directly; run it once with the default precision and once with `--precision bf16`.

Error definitions are those of `kk.check_is_allclose`: abs = |ans - ref|, rel = abs / (|ref| + 1e-6),
cos_diff = 1 - 2 <ans, ref> / (|ans|^2 + |ref|^2); an element passes if abs < abs_tol or rel < rel_tol.
Tolerances are those of the upstream test: out abs 1e-3 / rel 2.01/128 / cos 5e-6; lse abs 1e-6 /
rel 8.01/65536.

Run (nothing else may use the GPU while it times; flash_mla has to be built and FlashInfer importable,
see benchmarks/sm120/README.md):
  python benchmarks/sm120/bench_flashinfer.py --production --features --csv out.csv --md out.md
"""

import argparse
import csv
import dataclasses
import math
import os
import re
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

_TESTS_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests"))
sys.path.insert(0, _TESTS_DIR)

import torch
import kernelkit as kk

import lib
import quant
import ref
from lib import RawTestParamForDecode as RawTestParam

try:
    import flashinfer
    from flashinfer.mla import SparseMLASm120Wrapper
except ImportError as e:    # pragma: no cover
    raise SystemExit(
        f"Cannot import FlashInfer ({e}). Install flashinfer-python, or put the directory it was installed to on PYTHONPATH "
        "(see benchmarks/sm120/README.md)")

# The upstream test tolerances (tests/test-sparse-decode.py)
OUT_TOL = dict(abs_tol=1e-3, rel_tol=2.01 / 128, cos_tol=5e-6)
LSE_TOL = dict(abs_tol=1e-6, rel_tol=8.01 / 65536, cos_tol=None)
DECODE_MAX_TOKENS = 64      # FlashInfer's decode kernels take at most this many tokens per call
FP8, FP4 = quant.KVCacheLayout.V41_FP8Sparse, quant.KVCacheLayout.V41_FP4Sparse


# ---------------------------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------------------------

@dataclasses.dataclass
class Case:
    name: str
    group: str      # "production" | "small" | "feature"
    raw: RawTestParam


def production_cases() -> List[Case]:
    """The 16 production shapes of tests/test-sparse-decode.py (`base_and_bszs`)."""
    cases = []
    for extra_layout in (None, FP4):
        for h in (64, 128):
            for b in (64, 128, 256, 512):
                raw = RawTestParam(b, h, 4, 1, 256, True, topk=128, d_qk=512, extra_s_k=2048, extra_topk=512,
                                   block_size=256, extra_block_size=64, kvcache_layout=FP8, extra_kvcache_layout=extra_layout)
                cases.append(Case(f"h{h}_b{b}_{'fp4' if extra_layout else 'fp8'}", "production", raw))
    return cases


def small_cases() -> List[Case]:
    """Small batches, T = b * s_q <= 64 so the wrapper takes FlashInfer's decode kernel (its design range): b in {1, 4, 16} x
    s_q in {1, 4} x h in {64, 128}, topk 128 + extra 512, FP8 / FP4 extra cache, production page sizes, every request with
    its full 256 / 2048 tokens. The same list (names and data parameters) as `small_cases` of bench_decode_graph.py, so the
    two scripts time the same shapes."""
    cases = []
    for extra_layout in (None, FP4):
        for h in (64, 128):
            for s_q in (1, 4):
                for b in (1, 4, 16):
                    raw = RawTestParam(b, h, s_q, 1, 256, False, topk=128, d_qk=512, extra_s_k=2048, extra_topk=512,
                                       block_size=256, extra_block_size=64, kvcache_layout=FP8, extra_kvcache_layout=extra_layout)
                    cases.append(Case(f"h{h}_b{b}_sq{s_q}_{'fp4' if extra_layout else 'fp8'}", "small", raw))
    return cases


def feature_cases() -> List[Case]:
    """Small shapes with the features of the upstream corner cases (odd page sizes, lengths, sink)."""
    def mk(name: str, h: int, extra_layout, **feats) -> Case:
        raw = RawTestParam(4, h, 3, 1, 1024, True, 576, extra_s_k=1024, extra_topk=576, block_size=61, extra_block_size=123,
                           d_qk=512, kvcache_layout=FP8, extra_kvcache_layout=extra_layout, **feats)
        return Case(name, "feature", raw)
    return [
        mk("feat_h64_fp8_L_S", 64, None, have_topk_length=True, enable_attn_sink=True),
        mk("feat_h64_fp4_E_S", 64, FP4, have_extra_topk_length=True, enable_attn_sink=True),
        mk("feat_h128_fp8_L_E", 128, None, have_topk_length=True, have_extra_topk_length=True, enable_attn_sink=False),
        mk("feat_h128_fp4_L_E_S", 128, FP4, have_topk_length=True, have_extra_topk_length=True, enable_attn_sink=True),
    ]


def empty_row_cases() -> List[Case]:
    """Rows without any valid KV: FlashMLA returns out = 0, lse = +inf; FlashInfer returns out = 0 and lse = -inf
    (no sink) or the sink (finite sink). Checks the sentinel conversion. `zero_len`: some sequences have length 0
    (mixed with normal rows); `all_invalid`: every index is -1."""
    def mk(name: str, extra_layout, **feats) -> Case:
        raw = RawTestParam(8, 64, 3, 1, 1024, True, 576, enable_attn_sink=True, extra_s_k=1024, extra_topk=576, block_size=61,
                           extra_block_size=123, d_qk=512, kvcache_layout=FP8, extra_kvcache_layout=extra_layout, **feats)
        return Case(name, "feature", raw)
    return [mk("feat_h64_fp8_zero_len", None, have_zero_seqlen_k=True),
            mk("feat_h64_fp4_all_invalid", FP4, is_all_indices_invalid=True)]


# ---------------------------------------------------------------------------------------------
# FlashMLA test data -> FlashInfer inputs
# ---------------------------------------------------------------------------------------------

def to_page_footer_layout(cache: torch.Tensor, data_bytes: int, scale_bytes: int) -> torch.Tensor:
    """[nb, page, 1, data+scale] per-token [data, scales]  ->  [nb, page * (data+scale)] with
    [page x data | page x scales] per page (FlashInfer's DSV4_1 footer layout). A byte permutation."""
    nb, page, _, bytes_per_token = cache.shape
    assert bytes_per_token == data_bytes + scale_bytes, (cache.shape, data_bytes, scale_bytes)
    c = cache.view(torch.uint8).view(nb, page, bytes_per_token)
    return torch.cat([c[..., :data_bytes].reshape(nb, page * data_bytes),
                      c[..., data_bytes:].reshape(nb, page * scale_bytes)], dim=1).contiguous()


@dataclasses.dataclass
class FlashInferInputs:
    q: torch.Tensor                 # [T, h, 512] bf16
    kv: torch.Tensor                # [nb, page * 528] uint8
    indices: torch.Tensor           # [T, topk] int32
    extra_kv: Optional[torch.Tensor]
    extra_indices: Optional[torch.Tensor]
    topk_length: Optional[torch.Tensor]         # [T]
    extra_topk_length: Optional[torch.Tensor]   # [T]
    attn_sink: Optional[torch.Tensor]           # [h] float32
    sm_scale: float
    extra_fp4: bool
    out: torch.Tensor               # [T, h, 512] bf16
    lse: torch.Tensor               # [T, h] float32, natural log


def make_inputs(p: lib.TestParam, t: lib.TestcaseForDecode) -> FlashInferInputs:
    b, s_q, h = p.decode.b, p.s_q, p.h_q
    T = b * s_q
    extra_fp4 = t.extra_kv_scope is not None and t.extra_kv_scope.kvcache_layout == FP4

    def tok_len(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return None if x is None else x.repeat_interleave(s_q).to(torch.int32).contiguous()

    extra_kv = extra_indices = None
    if t.extra_kv_scope is not None:
        data_bytes, scale_bytes = (256, 32) if extra_fp4 else (512, 16)
        extra_kv = to_page_footer_layout(t.extra_kv_scope.get_kvcache_for_flash_mla(), data_bytes, scale_bytes)
        extra_indices = t.extra_kv_scope.indices_in_kvcache.reshape(T, -1).contiguous()
    return FlashInferInputs(
        q=t.q.reshape(T, h, p.d_qk).contiguous(),
        kv=to_page_footer_layout(t.kv_scope.get_kvcache_for_flash_mla(), 512, 16),
        indices=t.kv_scope.indices_in_kvcache.reshape(T, -1).contiguous(),
        extra_kv=extra_kv,
        extra_indices=extra_indices,
        topk_length=tok_len(t.kv_scope.topk_length),
        extra_topk_length=tok_len(t.extra_kv_scope.topk_length) if t.extra_kv_scope is not None else None,
        attn_sink=t.attn_sink,
        sm_scale=t.sm_scale,
        extra_fp4=extra_fp4,
        out=torch.empty(T, h, p.d_v, dtype=torch.bfloat16),
        lse=torch.empty(T, h, dtype=torch.float32),
    )


def make_call(x: FlashInferInputs, path: str, precision: str) -> Tuple[Callable[[], None], int]:
    """Return (fn, number of wrapper calls per fn()) for a path: "default" or "chunk64"."""
    wrapper = SparseMLASm120Wrapper(kv_scale_format="ue8m0_g32", extra_kv_fp4=x.extra_fp4, compute_precision=precision)
    T = x.q.shape[0]

    def sl(tensor: Optional[torch.Tensor], s: int, e: int) -> Optional[torch.Tensor]:
        return None if tensor is None else tensor[s:e]

    if path == "default":
        bounds = [(0, T)]
    elif path == "chunk64":
        bounds = [(s, min(s + DECODE_MAX_TOKENS, T)) for s in range(0, T, DECODE_MAX_TOKENS)]
    else:
        raise ValueError(path)
    # Slices are created once, outside the timed function
    calls = [((x.q[s:e], x.kv, x.indices[s:e], x.out[s:e], x.sm_scale),
              dict(topk_length=sl(x.topk_length, s, e), attn_sink=x.attn_sink, extra_kv_cache=x.extra_kv,
                   extra_indices=sl(x.extra_indices, s, e), extra_topk_length=sl(x.extra_topk_length, s, e),
                   out_lse=x.lse[s:e], lse_scale=math.log(2)))
             for s, e in bounds]

    def fn():
        for call_args, call_kwargs in calls:
            wrapper.run(*call_args, **call_kwargs)
    return fn, len(calls)


# ---------------------------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------------------------

def error_metrics(ans: torch.Tensor, ref_: torch.Tensor, abs_tol: float, rel_tol: float, cos_tol: Optional[float]) -> Dict[str, float]:
    """Max abs / rel error, cos_diff and the number of tolerance violations, with the definitions of
    `kk.check_is_allclose`. Positions where one side is NaN / inf and the other is not (or differs) are
    counted in `anomalies` and left out of the other numbers."""
    ans = ans.detach().float().reshape(-1)
    ref_ = ref_.detach().float().reshape(-1)
    nonfinite = ~torch.isfinite(ans) | ~torch.isfinite(ref_)
    same = nonfinite & ((ans == ref_) | (torch.isnan(ans) & torch.isnan(ref_)))
    anomalies = int((nonfinite & ~same).sum().item())
    zero = torch.zeros((), device=ans.device)
    a = torch.where(nonfinite, zero, ans)
    r = torch.where(nonfinite, zero, ref_)
    abs_err = (a - r).abs()
    rel_err = abs_err / (r.abs() + 1e-6)
    viol = int((~((abs_err < abs_tol) | (rel_err < rel_tol))).sum().item())
    return dict(
        cos_diff=float(kk.get_cos_diff(a, r)) if cos_tol is not None else float("nan"),
        max_abs=float(abs_err.max().item()),
        max_rel=float(rel_err.max().item()),
        viol=viol,
        anomalies=anomalies,
        passed=bool(viol == 0 and anomalies == 0 and (cos_tol is None or abs(kk.get_cos_diff(a, r)) <= cos_tol)),
    )


def compare_to_reference(p: lib.TestParam, t: lib.TestcaseForDecode, x: FlashInferInputs) -> Dict[str, float]:
    b, s_q, h, d_v = p.decode.b, p.s_q, p.h_q, p.d_v
    out_ref, lse_ref = ref.ref_sparse_attn_decode(p, t)         # [b, s_q, h, dv] bf16, [b, h, s_q] fp32
    out = x.out.view(b, s_q, h, d_v)
    lse = x.lse.view(b, s_q, h)

    # FlashMLA lse (+inf for rows without valid KV, no sink) -> FlashInfer lse (includes the sink)
    lse_ref = lse_ref.transpose(1, 2).float()
    lse_ref = torch.where(torch.isposinf(lse_ref), torch.full_like(lse_ref, float("-inf")), lse_ref)
    if x.attn_sink is not None:
        lse_ref = torch.logaddexp(lse_ref, x.attn_sink.view(1, 1, h).float())

    clean = torch.ones(h, dtype=torch.bool, device=out.device)
    if x.attn_sink is not None:
        clean = ~torch.isposinf(x.attn_sink)
    res: Dict[str, float] = dict(inf_sink_heads=int((~clean).sum().item()))
    for key, m in error_metrics(out[:, :, clean], out_ref[:, :, clean], **OUT_TOL).items():
        res[f"out_{key}"] = m
    for key, m in error_metrics(lse[:, :, clean], lse_ref[:, :, clean], **LSE_TOL).items():
        res[f"lse_{key}"] = m
    # Anomalies over all heads (NaN in out, wrong inf in lse), reported on their own
    res["out_anomalies_all"] = error_metrics(out, out_ref, **OUT_TOL)["anomalies"]
    res["lse_anomalies_all"] = error_metrics(lse, lse_ref, **LSE_TOL)["anomalies"]
    return res


# ---------------------------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------------------------

_flush_kernel_names: Optional[Dict[str, int]] = None   # name -> launches in a kk.bench of an empty function


def _strip_arguments(name: str) -> str:
    """'void ns::kernel<(Enum)5, 64>(float*, int)' -> 'ns::kernel<(Enum)5, 64>'"""
    depth = 0
    for i, ch in enumerate(name):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "(" and depth == 0:
            return name[:i].removeprefix("void ").strip()
    return name


def _short_kernel_name(name: str) -> str:
    """'void ns::kernel<(Enum)5, 64>(float*, int)' -> 'kernel'"""
    m = re.match(r"(?:[\w]+::)*(\w+)\s*(?:<|$)", _strip_arguments(name))
    return m.group(1) if m else name[:60]


def time_call(fn: Callable[[], None], num_runs: int, repeats: int) -> Dict:
    """Time `fn` with kk.bench (as tests/test-sparse-decode.py does) and list the CUDA kernels it launches.

    Returns kernels: [(full name, short name, launches per call, total time per call in us)], and for
    each of `repeats` separate kk.bench calls the sum of the kernel times and the wall span from the
    first kernel start to the last kernel end of one call (what the upstream test calls e2e)."""
    global _flush_kernel_names
    if _flush_kernel_names is None:
        # Kernels that kk.bench itself launches (the L2 flush), measured with an empty function
        flush = kk.bench(lambda: None, num_runs)
        _flush_kernel_names = {n: len(r) for n, r in flush.time_ranges.items()}
    sums, spans, kernels = [], [], None
    for _ in range(repeats):
        res = kk.bench(fn, num_runs)
        names = [n for n in res.get_kernel_names() if not n.startswith("cuda") and n not in _flush_kernel_names]
        assert names, f"no CUDA kernel found, saw {res.get_kernel_names()}"
        for n, cnt in _flush_kernel_names.items():
            if not n.startswith("cuda") and n in res.time_ranges and len(res.time_ranges[n]) != cnt:
                print(f"  warning: kernel {n[:60]} is launched by the timed function too; it is not counted", flush=True)
        per_kernel = []
        for n in names:
            ranges = res.time_ranges[n]
            assert len(ranges) % num_runs == 0, (n, len(ranges), num_runs)
            per_kernel.append((n, _short_kernel_name(n), len(ranges) // num_runs, sum(e - s for s, e in ranges) / num_runs * 1e6))
        sums.append(sum(k[3] for k in per_kernel))
        spans.append(res.get_e2e_time(names) * 1e6)
        kernels = per_kernel
    return dict(kernels=kernels, sum_us=sums, span_us=spans)


def median(xs: List[float]) -> float:
    s = sorted(xs)
    return s[len(s) // 2] if len(s) % 2 else 0.5 * (s[len(s) // 2 - 1] + s[len(s) // 2])


# ---------------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------------

def feature_string(p: lib.TestParam, x: FlashInferInputs) -> str:
    return "".join([
        "V" if p.decode.is_varlen else "-",
        "L" if x.topk_length is not None else "-",
        "E" if x.extra_topk_length is not None else "-",
        "S" if x.attn_sink is not None else "-",
    ])


def run_case(case: Case, seed: int, args) -> List[Dict]:
    raw = dataclasses.replace(case.raw, seed=seed)
    p = raw.to_test_param()
    assert p.can_run_on_and_clamp(lib.TestTarget.DECODE)
    t = lib.generate_testcase_for_decode(p)
    x = make_inputs(p, t)
    T = p.decode.b * p.s_q
    do_time = args.time and (case.group in ("production", "small") or args.time_features)

    paths = ["default"]
    if args.chunk64 and T > DECODE_MAX_TOKENS:
        paths.append("chunk64")

    base = dict(
        case=case.name, group=case.group, h=p.h_q, b=p.decode.b, s_q=p.s_q, T=T, topk=p.topk, extra_topk=p.decode.extra_topk,
        extra_fmt="fp4" if x.extra_fp4 else "fp8", page=p.decode.block_size, extra_page=p.decode.extra_block_size,
        feats=feature_string(p, x), precision=args.precision, seed=seed,
    )
    rows = []
    for path in paths:
        row = dict(base, path=path)
        try:
            fn, num_calls = make_call(x, path, args.precision)
            row["calls"] = num_calls
            x.out.fill_(float("nan"))
            x.lse.fill_(float("nan"))
            fn()
            torch.cuda.synchronize()
        except (ValueError, RuntimeError) as e:
            row["status"] = "unsupported: " + str(e).strip().splitlines()[0][:160]
            rows.append(row)
            print(f"  {case.name:24s} {path:8s} {row['status']}", flush=True)
            continue
        row["status"] = "ok"
        row.update(compare_to_reference(p, t, x))
        rows.append(row)
        # `x` holds the FlashInfer result of this path until the next path overwrites it
        if do_time:
            row["_timing_fn"] = fn
    # Free the reference data before timing (kk.bench allocates an 8 GB flush buffer per call)
    del t
    torch.cuda.empty_cache()
    for row in rows:
        fn = row.pop("_timing_fn", None)
        if fn is not None:
            tm = time_call(fn, args.num_runs, args.bench_repeats)
            row["kernels"] = tm["kernels"]
            row["sum_us"] = median(tm["sum_us"])
            row["span_us"] = median(tm["span_us"])
            row["span_spread_pct"] = (max(tm["span_us"]) - min(tm["span_us"])) / row["span_us"] * 100.0
        print_row(row)
    return rows


def print_row(row: Dict):
    if row["status"] != "ok":
        return
    timing = f"span {row['span_us']:8.1f} us (sum {row['sum_us']:8.1f}, spread {row['span_spread_pct']:.1f}%)" if "span_us" in row else "no timing"
    print(f"  {row['case']:24s} {row['path']:8s} {timing} | out cos {row['out_cos_diff']:.2e} abs {row['out_max_abs']:.2e} "
          f"rel {row['out_max_rel']:.2e} {'PASS' if row['out_passed'] else 'fail'} | lse abs {row['lse_max_abs']:.2e} "
          f"rel {row['lse_max_rel']:.2e} {'PASS' if row['lse_passed'] else 'fail'} | anomalies(all heads) out {row['out_anomalies_all']} "
          f"lse {row['lse_anomalies_all']}", flush=True)


CSV_COLUMNS = [
    "case", "group", "path", "status", "precision", "h", "b", "s_q", "T", "topk", "extra_topk", "extra_fmt", "page", "extra_page", "feats",
    "seed", "calls", "span_us", "sum_us", "span_spread_pct", "kernels",
    "out_cos_diff", "out_max_abs", "out_max_rel", "out_viol", "out_passed", "lse_max_abs", "lse_max_rel", "lse_viol", "lse_passed",
    "inf_sink_heads", "out_anomalies", "lse_anomalies", "out_anomalies_all", "lse_anomalies_all",
]


def kernels_cell(row: Dict, short: bool) -> str:
    if "kernels" not in row:
        return ""
    return "; ".join(
        (f"{k[1] if short else _strip_arguments(k[0])}" + (f" x{k[2]}" if k[2] > 1 else "") + f": {k[3]:.1f}") for k in row["kernels"])


def write_csv(path: str, rows: List[Dict]):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            r = dict(row)
            r["kernels"] = kernels_cell(row, short=False)
            for k, v in list(r.items()):
                if isinstance(v, float):
                    r[k] = f"{v:.6g}"
            w.writerow(r)


def write_markdown(path: str, rows: List[Dict], args, cmd: str):
    def fmt(v, spec):
        return "-" if v is None else format(v, spec)

    lines = [
        "# FlashInfer sm120 sparse MLA decode on the FlashMLA decode test data",
        "",
        f"- GPU {torch.cuda.get_device_name(0)}, torch {torch.__version__}, FlashInfer {flashinfer.__version__}, compute precision `{args.precision}`",
        f"- Date {time.strftime('%Y-%m-%d %H:%M')}; command `{cmd}`",
        f"- Timing: `kk.bench`, {args.num_runs} runs per measurement with an 8 GB L2 flush before each, median of {args.bench_repeats} measurements; "
        "`span` = first kernel start to last kernel end of one call (what tests/test-sparse-decode.py calls e2e), `sum` = sum of the kernel durations.",
        "- Errors vs `ref.ref_sparse_attn_decode` on heads whose attn_sink is not +inf (FlashInfer returns NaN / a wrong lse there; counted in `anom`). "
        "`viol` = elements failing both abs and rel tolerance; PASS = no violation, no anomaly, and (out only) cos_diff <= 5e-6. "
        "out tolerances abs 1e-3 / rel 2.01/128 / cos 5e-6, lse abs 1e-6 / rel 8.01/65536. LSE compared in FlashInfer's convention (natural log, includes sink).",
        "",
    ]
    for group in ("production", "small", "feature"):
        g = [r for r in rows if r["group"] == group]
        if not g:
            continue
        lines += [f"## {group} shapes", "",
                  "| case | path | h | b | s_q | T | topk+extra | extra | feats | kernels (us per call) | span us | sum us | out cos_diff | out max abs | out max rel | out | lse max abs | lse max rel | lse | anom out/lse (clean heads) | anom out/lse (all heads) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in g:
            if r["status"] != "ok":
                lines.append(f"| {r['case']} | {r['path']} | {r['h']} | {r['b']} | {r['s_q']} | {r['T']} | {r['topk']}+{r['extra_topk']} | {r['extra_fmt']} | {r['feats']} | {r['status']} |" + " |" * 12)
                continue
            lines.append(
                f"| {r['case']} | {r['path']} | {r['h']} | {r['b']} | {r['s_q']} | {r['T']} | {r['topk']}+{r['extra_topk']} | {r['extra_fmt']} | {r['feats']} | "
                f"{kernels_cell(r, short=True) or '-'} | {fmt(r.get('span_us'), '.1f')} | {fmt(r.get('sum_us'), '.1f')} | "
                f"{r['out_cos_diff']:.2e} | {r['out_max_abs']:.2e} | {r['out_max_rel']:.2e} | {'PASS' if r['out_passed'] else 'fail'} | "
                f"{r['lse_max_abs']:.2e} | {r['lse_max_rel']:.2e} | {'PASS' if r['lse_passed'] else 'fail'} | "
                f"{r['out_anomalies']}/{r['lse_anomalies']} | {r['out_anomalies_all']}/{r['lse_anomalies_all']} |")
        lines.append("")
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--production", action="store_true", help="the 16 production shapes (timing + errors)")
    parser.add_argument("--small", action="store_true", help="the 24 small-batch shapes (T <= 64: FlashInfer's decode kernel; timing + errors)")
    parser.add_argument("--features", action="store_true", help="the 4 small feature shapes (errors only unless --time-features)")
    parser.add_argument("--empty-rows", action="store_true", help="also run 2 shapes whose rows have no valid KV (sentinel check)")
    parser.add_argument("--only", type=str, default=None, help="comma separated substrings; keep the cases whose name contains one of them")
    parser.add_argument("--csv", type=str, default=None, help="write one CSV row per (case, path)")
    parser.add_argument("--md", type=str, default=None, help="write a Markdown table")
    parser.add_argument("--no-time", dest="time", action="store_false", help="errors only")
    parser.add_argument("--time-features", action="store_true", help="also time the feature shapes")
    parser.add_argument("--num-runs", type=int, default=10, help="runs per kk.bench call (the upstream test uses 10)")
    parser.add_argument("--bench-repeats", type=int, default=3, help="separate kk.bench calls per measurement; the median is reported")
    parser.add_argument("--no-chunk64", dest="chunk64", action="store_false", help="skip the 64-token-slice decode path")
    parser.add_argument("--precision", choices=["default", "fp8", "bf16"], default="default", help="SparseMLASm120Wrapper compute_precision")
    parser.add_argument("--seed", type=int, default=0, help="seed of the first case; case i uses seed + i")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")

    cases: List[Case] = []
    if args.production:
        cases += production_cases()
    if args.small:
        cases += small_cases()
    if args.features:
        cases += feature_cases()
    if args.empty_rows:
        cases += empty_row_cases()
    if args.only:
        keys = args.only.split(",")
        cases = [c for c in cases if any(k in c.name for k in keys)]
    if not cases:
        parser.error("no cases: pass --production, --small, --features and/or --empty-rows (and check --only)")

    print(f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, FlashInfer {flashinfer.__version__}, "
          f"{len(cases)} cases, precision {args.precision}, timing {'on' if args.time else 'off'}", flush=True)
    if args.time:
        print("note: the times are only valid if nothing else uses the GPU during the run", flush=True)

    rows: List[Dict] = []
    with torch.inference_mode():
        for i, case in enumerate(cases):
            print(f"[{i + 1}/{len(cases)}] {case.name}  {case.raw.b=} {case.raw.h_q=} {case.raw.s_q=}", flush=True)
            rows += run_case(case, args.seed + i, args)
            torch.cuda.empty_cache()

    cmd = " ".join(["python", "benchmarks/sm120/bench_flashinfer.py"] + sys.argv[1:])
    if args.csv:
        write_csv(args.csv, rows)
        print(f"wrote {args.csv}")
    if args.md:
        write_markdown(args.md, rows, args, cmd)
        print(f"wrote {args.md}")


if __name__ == "__main__":
    main()
