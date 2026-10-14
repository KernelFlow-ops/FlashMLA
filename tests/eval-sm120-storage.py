"""
Storage and composite error tables of the 4-bit KV cache formats (T-10, sm120 phase 6 G1). Run from tests/:

    python eval-sm120-storage.py --csv storage.csv --md storage.md
    python eval-sm120-storage.py --selftest

Data. lib.generate_testcase_for_decode quantizes the KV cache (KVScope.quant_and_dequant_) and replaces the bf16 KV by its dequantization, so the unquantized KV is gone afterwards.
Here the generator is run with quant_and_dequant_ replaced by a no-op: the random streams, the shapes and the indices are exactly those of the upstream generator, and
`scope.blocked_k` stays the unquantized bf16 KV (randn * 0.1 clamped to +-1, NaN in the rows no index points to). The data classes of eval-sm120-modes.py (upstream, hv4, hv16,
kampw4: the logit spread is set by scaling sm_scale, the K gains by the same generator as apply_k_gain) are generated this way, plus three more:
    chan    K *= a per-dimension amplitude exp(0.75 * N(0, 1)) (one draw for the run, shared by both caches) and 8 dimensions (random) times 20
    tokamp  K *= a per-token amplitude exp(1.5 * N(0, 1))
    small   K *= 2^-12
    big     K *= 2^14 (not asked for in the task card: the other end of the range, where the largest block scale of a one-level cache, amax / 6, exceeds the e4m3 limit 448)
All four, like hv4 and kampw4, get sm_scale such that the logits have a standard deviation of 4 nat (measured on the unquantized KV, once, shared by every format).
The data has a main cache (192 tokens per query token) and an extra cache (576 tokens), always in the same format.

Tables.
  storage    attention in float64 on the dequantized cache against float64 attention on the unquantized KV (same Q, indices, sm_scale, attention sink): only the format error.
             Metrics: relative RMS of the KV elements (over the finite elements of the unquantized KV), out cos_diff (1 - 2 <a, b> / (|a|^2 + |b|^2)), out relative RMS (|a - b| / |b|),
             lse max abs error (nat)
  composite  the same cache run through the arithmetic of a tier of the kernel (sm120_emu.simulate_decode), against the same float64 reference on the unquantized KV: what the model sees.
             The bf16 output rounding of the emulator is part of the numbers; the row `floor` is the float64 reference rounded to bf16
Formats. fp8 = V41_FP8Sparse (quant.py), mxfp4 = V41_MXFP4Sparse (quant.py), `-nf` = the same without the 2^-13 floor of the ue8m0 scale (eval-sm120-modes.quantize_without_scale_floor);
  nv1-rn = V41_FP4Sparse (quant.py), nv1-ceil = the same bytes layout with the e4m3 scale rounded up; nv2t-* = two-level, one 2^g per cache tensor, nv2k-* = one 2^g per token (storage only:
  the kernel folds a per-tensor g into the logit scale and the rebase exponent), `-46` = Four Over Six. See quant.quantize_k_cache_fp4_2l.
"""
import argparse
import contextlib
import csv
import dataclasses
import importlib.util
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch

import lib
import quant
import ref
import sm120_emu as emu

_spec = importlib.util.spec_from_file_location("ev", os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval-sm120-modes.py"))
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)
L = quant.KVCacheLayout

# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------
CLASS_CFG: Dict[str, dict] = {k: ev.CLASSES[k] for k in ("upstream", "hv4", "hv16", "kampw4")}
CLASS_CFG.update({
    "chan": dict(sigma=4.0, kamp=False, topk_length=False, special="chan"),
    "tokamp": dict(sigma=4.0, kamp=False, topk_length=False, special="tokamp"),
    "small": dict(sigma=4.0, kamp=False, topk_length=False, special="small"),
    "big": dict(sigma=4.0, kamp=False, topk_length=False, special="big"),
})
EXTREME_CLASSES = ("small", "big")          # the classes that break a one-level cache: kept out of the geometric means and shown on their own


@contextlib.contextmanager
def keep_unquantized():
    """lib's quantize-and-dequantize step becomes a no-op: scope.blocked_k stays the unquantized bf16 KV"""
    orig = lib.KVScope.quant_and_dequant_
    lib.KVScope.quant_and_dequant_ = lambda self: None
    try:
        yield
    finally:
        lib.KVScope.quant_and_dequant_ = orig


def scopes_of(t) -> list:
    return [t.kv_scope] + ([t.extra_kv_scope] if t.extra_kv_scope is not None else [])


def apply_gain_unquantized(t, seed: int, spread: float):
    """ev.apply_k_gain without the re-quantization: every (token, 32-dim group) of K times 2^U(-2 spread, 0), the same generator and draws"""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    for scope in scopes_of(t):
        k = scope.blocked_k
        u = torch.rand(*k.shape[:-1], 16, generator=gen, dtype=torch.float32) * (2 * spread) - 2 * spread
        scope.blocked_k = (k.float().view(*k.shape[:-1], 16, 32) * torch.exp2(u).unsqueeze(-1)).view(k.shape).to(torch.bfloat16)


def apply_special(t, kind: str, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed + 7)
    if kind == "chan":
        amp = torch.exp(0.75 * torch.randn(512, generator=gen, dtype=torch.float32, device="cuda"))
        amp[torch.randperm(512, generator=gen, device="cuda")[:8]] *= 20.0
    for scope in scopes_of(t):
        k = scope.blocked_k
        if kind == "chan":
            f = amp
        elif kind == "tokamp":
            f = torch.exp(1.5 * torch.randn(*k.shape[:-1], 1, generator=gen, dtype=torch.float32, device="cuda"))
        elif kind == "small":
            f = 2.0 ** -12
        else:
            assert kind == "big"
            f = 2.0 ** 14
        scope.blocked_k = (k.float() * f).to(torch.bfloat16)


def logits_f64(t, kvs, sm_scale: float):
    """(w [b*s_q, h, N] float64 logits * sm_scale with -inf for invalid tokens, invalid [b*s_q, 1, N], gathered KV [b*s_q, N, d] float64 with NaN set to 0)"""
    b, s_q, h, d = t.q.shape
    gs, invs = [], []
    for sc, kv in zip(scopes_of(t), kvs):
        topk = sc.indices_in_kvcache.size(-1)
        idx = torch.clamp_min(sc.indices_in_kvcache, 0)
        gs.append(kv.reshape(-1, d).index_select(0, idx.reshape(-1).long()).view(b, s_q, topk, d))
        inv = sc.indices_in_kvcache == -1
        if sc.topk_length is not None:
            inv = inv | (torch.arange(topk, device=inv.device).view(1, 1, topk) >= sc.topk_length.view(b, 1, 1))
        invs.append(inv)
    g = torch.cat(gs, 2).reshape(b * s_q, -1, d).double()
    g = torch.where(g != g, torch.zeros_like(g), g)
    invalid = torch.cat(invs, 2).reshape(b * s_q, 1, -1)
    w = torch.matmul(t.q.double().reshape(b * s_q, h, d), g.transpose(-1, -2)) * sm_scale
    return w.masked_fill(invalid, float("-inf")), invalid, g


def attn_f64(t, kvs, sm_scale: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """ref.ref_sparse_attn_decode in float64 for the given KV tensors (one per cache, [num_blocks, block_size, 1, 512]): (out [b, s_q, h, 512] float64, lse [b, h, s_q] float64)"""
    b, s_q, h, d = t.q.shape
    w, invalid, g = logits_f64(t, kvs, sm_scale)
    lse = torch.logsumexp(w, -1)
    out = torch.matmul(torch.exp(w - lse.unsqueeze(-1)), g).view(b, s_q, h, d)
    lse = lse.view(b, s_q, h)
    if t.attn_sink is not None:
        out = out * (1.0 / (1.0 + torch.exp(t.attn_sink.double().view(1, 1, h) - lse))).unsqueeze(-1)
    lonely = lse == float("-inf")
    out = out.masked_fill(lonely.unsqueeze(-1), 0.0)
    lse = lse.masked_fill(lonely, float("inf"))
    return out, lse.transpose(1, 2)


def logit_std_f64(t, kvs) -> float:
    """ev.logit_std on the unquantized KV in float64: mean over the (query token, head) rows with >= 8 valid tokens of the std over the valid tokens of logit * sm_scale"""
    w, invalid, _ = logits_f64(t, kvs, t.sm_scale)
    v = (~invalid).double()
    x = torch.where(invalid, torch.zeros_like(w), w)
    n = v.sum(-1)
    mean = (x * v).sum(-1) / n.clamp_min(1)
    var = (((x - mean.unsqueeze(-1)) * v) ** 2).sum(-1) / n.clamp_min(1)
    rows = (n >= 8).expand_as(var)
    return float(var.sqrt()[rows].mean()) if rows.any() else float("nan")


def make_unquantized_case(cls: str, h: int, seed: int, b: int, s_q: int):
    cfg = CLASS_CFG[cls]
    raw = lib.RawTestParamForDecode(
        b=b, h_q=h, s_q=s_q, h_kv=1, s_kv=1024, is_varlen=True, topk=192, have_topk_length=cfg["topk_length"], enable_attn_sink=True,
        extra_s_k=2048, extra_topk=576, block_size=64, extra_block_size=64, have_extra_topk_length=cfg["topk_length"],
        d_qk=512, kvcache_layout=None, extra_kvcache_layout=None, check_correctness=True, num_runs=0, seed=seed)
    p = raw.to_test_param()
    with keep_unquantized():
        t = lib.generate_testcase_for_decode(p)
    if cfg["kamp"]:
        apply_gain_unquantized(t, seed, cfg.get("kamp_spread", 6.0))
    if cfg.get("special"):
        apply_special(t, cfg["special"], seed)
    if cfg["sigma"] is not None:
        t.sm_scale *= cfg["sigma"] / logit_std_f64(t, [sc.blocked_k for sc in scopes_of(t)])
    return p, t


# ----------------------------------------------------------------------------------------------
# Formats: kv [num_blocks, block_size, 1, 512] bf16 -> (layout, bytes, dequantized bf16, g)
# ----------------------------------------------------------------------------------------------
def _plain(layout, writer=None):
    def f(k):
        q = (writer or quant.quantize_k_cache)(k, layout)
        return layout, q, quant.dequantize_k_cache(q, layout), 0
    return f


def _two_level(rule: str, four_six: bool, granularity: str, g0: bool = False):
    def f(k):
        q, g = quant.quantize_k_cache_fp4_2l(k, rule, four_six, granularity, g=0 if g0 else None)
        return L.V41_FP4Sparse, q, quant.dequantize_k_cache_fp4_2l(q, g), g
    return f


FORMATS = {
    "fp8": _plain(L.V41_FP8Sparse),
    "fp8-nf": _plain(L.V41_FP8Sparse, ev.quantize_without_scale_floor),
    "mxfp4": _plain(L.V41_MXFP4Sparse),
    "mxfp4-nf": _plain(L.V41_MXFP4Sparse, ev.quantize_without_scale_floor),
    "nv1-rn": _plain(L.V41_FP4Sparse),
    "nv1-ceil": _two_level("ceil", False, "tensor", g0=True),
    "nv2t-rn": _two_level("rn", False, "tensor"),
    "nv2t-ceil": _two_level("ceil", False, "tensor"),
    "nv2k-rn": _two_level("rn", False, "token"),
    "nv1-rn-46": _two_level("rn", True, "tensor", g0=True),
    "nv2t-rn-46": _two_level("rn", True, "tensor"),
}
FORMAT_ORDER = list(FORMATS)


def tiers_of(fmt: str) -> List[str]:
    """The kernel tiers (rows of eval-sm120-modes.py) a format is read by, plus `exact` (the upstream bf16 arithmetic on the dequantized cache)"""
    fam = "fp8" if fmt.startswith("fp8") else "mxfp4" if fmt.startswith("mxfp4") else "nvfp4"
    return {"fp8": ["kernel-mxfp8", "kernel-fp8x2"], "mxfp4": ["kernel-mxfp4-mxfp8", "kernel-mxfp4-fp8x2"], "nvfp4": ["kernel-nvfp4-mxfp8", "kernel-nvfp4-fp8x2"]}[fam] + ["exact"]


def tier_group(tier: str) -> str:
    return "mxfp8" if tier.endswith("mxfp8") else "fp8x2" if tier.endswith("fp8x2") else "exact"


# ----------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------
def kv_rms(kvs_pre, kvs_deq) -> float:
    num = den = 0.0
    for a, d in zip(kvs_pre, kvs_deq):
        m = torch.isfinite(a.float())
        a64 = a.double()[m]
        num += float((d.double()[m] - a64).pow(2).sum())
        den += float(a64.pow(2).sum())
    return math.sqrt(num / max(den, 1e-300))


def err_metrics(out, lse, out_ref, lse_ref) -> dict:
    o, r = out.double(), out_ref.double()
    cos = 1.0 - 2.0 * float((o * r).sum()) / max(float((o * o).sum() + (r * r).sum()), 1e-300)
    rel = math.sqrt(float((o - r).pow(2).sum())) / max(math.sqrt(float(r.pow(2).sum())), 1e-300)
    lse, lse_ref = lse.double(), lse_ref.double()
    fin = torch.isfinite(lse) & torch.isfinite(lse_ref)
    return dict(cos_diff=cos, rel_rms=rel, lse_max_abs=float((lse - lse_ref).abs()[fin].max()) if fin.any() else 0.0,
                lse_inf_mismatch=int((torch.isinf(lse) != torch.isinf(lse_ref)).sum()))


def run_case(cls: str, h: int, seed: int, b: int, s_q: int, combos: dict, formats: List[str], log) -> List[dict]:
    rows = []
    with ev.upstream_defaults():
        p, t = make_unquantized_case(cls, h, seed, b, s_q)
    with torch.inference_mode():
        scopes = scopes_of(t)
        kvs_pre = [sc.blocked_k for sc in scopes]
        out_ref, lse_ref = attn_f64(t, kvs_pre, t.sm_scale)
        base = dict(cls=cls, h=h, seed=seed, sigma=logit_std_f64(t, kvs_pre))
        rows.append(dict(base, kind="composite", format="bf16", tier="floor", kv_rms=0.0,
                         **err_metrics(out_ref.to(torch.bfloat16), lse_ref.float(), out_ref, lse_ref)))
        for fmt in formats:
            infos = [FORMATS[fmt](sc.blocked_k) for sc in scopes]
            deq = [i[2] for i in infos]
            krms = kv_rms(kvs_pre, deq)
            out_q, lse_q = attn_f64(t, deq, t.sm_scale)
            rows.append(dict(base, kind="storage", format=fmt, tier="fp64", kv_rms=krms, **err_metrics(out_q, lse_q, out_ref, lse_ref)))
            t_fmt = dataclasses.replace(t)
            new_scopes = []
            for sc, (layout, qb, dq, g) in zip(scopes, infos):
                s2 = dataclasses.replace(sc, kvcache_layout=layout, blocked_k_quantized=qb, blocked_k=dq)
                s2.fp4_g = g
                new_scopes.append(s2)
            t_fmt.kv_scope, t_fmt.extra_kv_scope = new_scopes[0], (new_scopes[1] if len(new_scopes) > 1 else None)
            for tier in tiers_of(fmt):
                out_e, lse_e = emu.simulate_decode(p, t_fmt, combos[tier])
                rows.append(dict(base, kind="composite", format=fmt, tier=tier, kv_rms=krms, **err_metrics(out_e, lse_e, out_ref, lse_ref)))
        del out_ref, lse_ref, kvs_pre
    torch.cuda.empty_cache()
    return rows


# ----------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------
def fnum(x: float) -> str:
    return "nan" if x != x else f"{x:.2e}"


def agg(rows: List[dict], key: str, how: str) -> Optional[float]:
    v = [r[key] for r in rows]
    if not v:
        return None
    if any(x != x for x in v):
        return float("nan")
    return max(v) if how == "max" else sum(v) / len(v)


def pct(a: Optional[float], b: Optional[float]) -> str:
    """Reduction of B relative to A in percent (positive: B is better)"""
    if a is None or b is None or a != a or b != b or a <= 0:
        return "n/a"
    return f"{(1.0 - b / a) * 100:+.0f}%"


def geo_pct(pairs: List[Tuple[float, float]]) -> str:
    ps = [(a, b) for a, b in pairs if a is not None and b is not None and a == a and b == b and a > 0 and b > 0]
    if not ps:
        return "n/a"
    r = math.exp(sum(math.log(b / a) for a, b in ps) / len(ps))
    return f"{(1.0 - r) * 100:+.0f}%"


METRICS = (("kv_rms", "KV element relative RMS", "mean"), ("cos_diff", "out cos_diff", "mean"), ("rel_rms", "out relative RMS", "mean"), ("lse_max_abs", "lse max abs error (nat)", "max"))


def build_md(rows: List[dict], classes: List[str], formats: List[str], args) -> str:
    cls = [c for c in classes if any(r["cls"] == c for r in rows)]
    n_cases = {c: len({(r["h"], r["seed"]) for r in rows if r["cls"] == c}) for c in cls}
    out = ["# Storage and composite error of the 4-bit KV cache formats (T-10, sm120 phase 6 G1)\n",
           f"Cases per data class: {', '.join(f'{c} {n_cases[c]}' for c in cls)} (heads {args.heads}, b {args.b}, s_q {args.s_q}, main cache 192 + extra cache 576 tokens, both in the format). "
           "Cell = mean over the cases (lse: worst case). Method: see the head of eval-sm120-storage.py.\n"]
    sel = lambda kind, fmt, tier, c: [r for r in rows if r["kind"] == kind and r["format"] == fmt and r["tier"] == tier and r["cls"] == c]

    out.append("\n## 1. Storage error (float64 attention on the dequantized cache against float64 attention on the unquantized KV)\n")
    for key, title, how in METRICS:
        out.append(f"\n### {title}\n")
        out.append("| format | " + " | ".join(cls) + " |")
        out.append("|---|" + "---|" * len(cls))
        for fmt in formats:
            out.append(f"| {fmt} | " + " | ".join(fnum(agg(sel("storage", fmt, "fp64", c), key, how)) if sel("storage", fmt, "fp64", c) else "" for c in cls) + " |")
    out.append("\n### Improvement of B over A (reduction of the error in percent, positive = B better); the last column is the geometric mean over the classes without small and big\n")
    pairs = [("mxfp4", "nv1-rn", "NVFP4 one level (rn) over MXFP4 (current writer)"), ("mxfp4-nf", "nv1-rn", "NVFP4 one level (rn) over MXFP4 without the scale floor"),
             ("mxfp4", "nv2t-rn", "NVFP4 two level per tensor over MXFP4 (current writer)"), ("mxfp4-nf", "nv2t-rn", "NVFP4 two level per tensor over MXFP4 without the scale floor"),
             ("nv1-rn", "nv2t-rn", "two level per tensor over one level (rn)"), ("nv2t-rn", "nv2k-rn", "per-token g over per-tensor g (rn)"),
             ("nv1-rn", "nv1-ceil", "scale rule ceil over rn (one level)"), ("nv2t-rn", "nv2t-ceil", "scale rule ceil over rn (two level)"),
             ("nv1-rn", "nv1-rn-46", "Four Over Six over plain (one level, rn)"), ("nv2t-rn", "nv2t-rn-46", "Four Over Six over plain (two level, rn)"),
             ("mxfp4", "nv2t-rn-46", "NVFP4 two level + Four Over Six over MXFP4 (current writer)")]
    for key, title, how in METRICS[:3]:
        out.append(f"\n#### {title}\n")
        out.append("| B over A | " + " | ".join(cls) + " | geo mean (no small / big) |")
        out.append("|---|" + "---|" * (len(cls) + 1))
        for a, bb, desc in pairs:
            if a not in formats or bb not in formats:
                continue
            cells = [(agg(sel("storage", a, "fp64", c), key, "mean"), agg(sel("storage", bb, "fp64", c), key, "mean")) for c in cls]
            out.append(f"| {desc} (`{bb}` over `{a}`) | " + " | ".join(pct(x, y) for x, y in cells) + f" | {geo_pct([cc for cc, c in zip(cells, cls) if c not in EXTREME_CLASSES])} |")

    out.append("\n## 2. Composite error (the cache read by a kernel tier, against float64 attention on the unquantized KV: what the model sees)\n")
    out.append("`floor` = the float64 reference rounded to bf16 (the output rounding every row contains). Tiers: `kernel-*-mxfp8` = the mxfp8 tier, `kernel-*-fp8x2` = the precise tier of the format's kernel, `exact` = bf16 arithmetic of the upstream reference on the dequantized cache.\n")
    comp_rows = [(f, t) for f in ["bf16"] + formats for t in (["floor"] if f == "bf16" else tiers_of(f))]
    for key, title, how in METRICS[:4]:
        out.append(f"\n### {title}\n")
        out.append("| format | tier | " + " | ".join(cls) + " |")
        out.append("|---|---|" + "---|" * len(cls))
        for fmt, tier in comp_rows:
            out.append(f"| {fmt} | {tier} | " + " | ".join(fnum(agg(sel("composite", fmt, tier, c), key, how)) if sel("composite", fmt, tier, c) else "" for c in cls) + " |")
    out.append("\n### NVFP4 against MXFP4, same tier group (reduction of the composite error in percent, positive = NVFP4 better); the last column is the geometric mean over the classes without small and big\n")
    cpairs = []
    for grp, label in (("mxfp8", "mxfp8 tier"), ("fp8x2", "precise tier"), ("exact", "exact arithmetic")):
        for a in ("mxfp4", "mxfp4-nf"):
            for bb in ("nv1-rn", "nv2t-rn", "nv2t-rn-46"):
                if a in formats and bb in formats:
                    cpairs.append((grp, a, bb, label))
    tier_for = lambda fmt, grp: next(t for t in tiers_of(fmt) if tier_group(t) == grp)
    for key, title, how in METRICS[1:3]:
        out.append(f"\n#### {title}\n")
        out.append("| comparison | " + " | ".join(cls) + " | geo mean (no small / big) |")
        out.append("|---|" + "---|" * (len(cls) + 1))
        for grp, a, bb, label in cpairs:
            cells = [(agg(sel("composite", a, tier_for(a, grp), c), key, "mean"), agg(sel("composite", bb, tier_for(bb, grp), c), key, "mean")) for c in cls]
            out.append(f"| {label}: `{bb}` ({tier_for(bb, grp)}) over `{a}` ({tier_for(a, grp)}) | " + " | ".join(pct(x, y) for x, y in cells) + f" | {geo_pct([cc for cc, c in zip(cells, cls) if c not in EXTREME_CLASSES])} |")
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------------------------
# Self test
# ----------------------------------------------------------------------------------------------
def selftest() -> bool:
    ok = True

    def report(name, passed, detail=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}", flush=True)

    for cls in ("upstream", "hv4", "kampw4", "chan", "tokamp", "small", "big"):
        with ev.upstream_defaults():
            p, t = make_unquantized_case(cls, 64, 5, 4, 2)
        with torch.inference_mode():
            kvs = [sc.blocked_k for sc in scopes_of(t)]
            o64, l64 = attn_f64(t, kvs, t.sm_scale)
            with ev.upstream_defaults():
                o32, l32 = ref.ref_sparse_attn_decode(p, t)         # float32 reference on the same (unquantized) KV: blocked_k is what it gathers from
            m = err_metrics(o32, l32, o64, l64)
        report(f"float64 attention == ref.ref_sparse_attn_decode on the unquantized KV [{cls}]", m["cos_diff"] < 2e-5 and m["rel_rms"] < 1e-2 and m["lse_max_abs"] < 1e-3 and m["lse_inf_mismatch"] == 0,
               f"(cos_diff {m['cos_diff']:.1e}, rel_rms {m['rel_rms']:.1e}, lse {m['lse_max_abs']:.1e}; measured std of logit * sm_scale {logit_std_f64(t, kvs):.3f} nat)")
        has_nan = any(bool(torch.isnan(k.float()).any()) for k in kvs)
        report(f"unquantized KV kept (bf16, NaN in the rows no index points to) [{cls}]", all(k.dtype == torch.bfloat16 for k in kvs) and has_nan)
    # the cases are the upstream generator's: the indices and Q equal those of a run with the quantization step in place
    with ev.upstream_defaults():
        p, t = make_unquantized_case("upstream", 64, 9, 2, 2)
        t2 = lib.generate_testcase_for_decode(p)
    report("same Q, indices, attention sink and cache lengths as the unmodified upstream generator", torch.equal(t.q, t2.q) and torch.equal(t.kv_scope.indices_in_kvcache, t2.kv_scope.indices_in_kvcache)
           and torch.equal(t.kv_scope.cache_seqlens, t2.kv_scope.cache_seqlens) and torch.equal(t.attn_sink, t2.attn_sink) and torch.equal(t.extra_kv_scope.indices_in_kvcache, t2.extra_kv_scope.indices_in_kvcache))
    # the quantization of a format applied to the kept KV is the dequantized cache the upstream generator keeps (V41_FP8Sparse default layout)
    with torch.inference_mode():
        lay, qb, dq, g = FORMATS["fp8"](t.kv_scope.blocked_k)
        report("format fp8 applied to the kept KV == the cache lib's quantize_and_dequantize keeps", torch.equal(torch.nan_to_num(dq.float(), nan=7e7), torch.nan_to_num(t2.kv_scope.blocked_k.float(), nan=7e7)))
    # the composite path: an fp8 cache through kernel-fp8x2, error well below the storage error of a 4-bit format
    combos = {n: m for n, _, m, _ in ev.build_combos()}
    rows = run_case("upstream", 64, 3, 2, 2, combos, ["fp8", "mxfp4", "nv2t-rn"], log=lambda s: None)
    get = lambda kind, fmt, tier: next(r for r in rows if r["kind"] == kind and r["format"] == fmt and r["tier"] == tier)
    floor = get("composite", "bf16", "floor")
    report("floor row: the bf16 rounding of the float64 reference (rel_rms about 2^-9 / sqrt(3) ... 2^-8)", 1e-3 < floor["rel_rms"] < 4e-3, f"({floor['rel_rms']:.2e})")
    s_fp8, s_mx, s_nv = get("storage", "fp8", "fp64"), get("storage", "mxfp4", "fp64"), get("storage", "nv2t-rn", "fp64")
    report("storage error ordering fp8 < nvfp4 < mxfp4 on upstream data", s_fp8["rel_rms"] < s_nv["rel_rms"] < s_mx["rel_rms"] and s_fp8["kv_rms"] < s_nv["kv_rms"] < s_mx["kv_rms"],
           f"(KV rms {s_fp8['kv_rms']:.3f} / {s_nv['kv_rms']:.3f} / {s_mx['kv_rms']:.3f})")
    c_fp8 = get("composite", "fp8", "kernel-fp8x2")
    report("composite fp8 cache through kernel-fp8x2 is near the floor + fp8 storage error", c_fp8["rel_rms"] < 3 * (floor["rel_rms"] + s_fp8["rel_rms"]), f"({c_fp8['rel_rms']:.2e})")
    c_nv = get("composite", "nv2t-rn", "kernel-nvfp4-fp8x2")
    report("composite NVFP4 precise tier ~ the storage error (the arithmetic adds little)", c_nv["rel_rms"] < 1.5 * s_nv["rel_rms"] + 3 * floor["rel_rms"], f"({c_nv['rel_rms']:.2e} against storage {s_nv['rel_rms']:.2e})")
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classes", default="upstream,hv4,hv16,kampw4,chan,tokamp,small,big")
    ap.add_argument("--formats", default=",".join(FORMAT_ORDER))
    ap.add_argument("--heads", default="64,128")
    ap.add_argument("--repeat", type=int, default=3, help="Number of seeds per (data class, heads)")
    ap.add_argument("-b", type=int, default=16)
    ap.add_argument("--s-q", type=int, default=4)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--md", default=None)
    ap.add_argument("--from-csv", default=None, help="Do not run: read the rows of an earlier --csv and write the markdown (with --md)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    classes = [c for c in args.classes.split(",") if c]
    formats = [f for f in args.formats.split(",") if f]
    if args.from_csv:
        def conv(v):
            for t_ in (int, float):
                try:
                    return t_(v)
                except ValueError:
                    pass
            return v
        with open(args.from_csv, newline="") as f:
            rows = [{k: conv(v) for k, v in r.items()} for r in csv.DictReader(f)]
    else:
        combos = {n: m for n, _, m, _ in ev.build_combos()}
        rows = []
        t0 = time.time()
        seed = 500
        for cls in classes:
            for h in [int(x) for x in args.heads.split(",")]:
                for rep in range(args.repeat):
                    tc = time.time()
                    new = run_case(cls, h, seed, args.b, args.s_q, combos, formats, log=print)
                    rows += new
                    s = {r["format"]: r for r in new if r["kind"] == "storage"}
                    print(f"{cls:8s} h={h:3d} seed={seed}: sigma {new[-1]['sigma']:.2f} nat; storage out rel_rms " + ", ".join(f"{f} {s[f]['rel_rms']:.1e}" for f in ("mxfp4", "nv1-rn", "nv2t-rn") if f in s)
                          + f" ({time.time() - tc:.1f} s)", flush=True)
                    seed += 1
        print(f"{len(rows)} rows in {time.time() - t0:.0f} s; peak GPU memory allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    if args.csv and not args.from_csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    if args.md:
        with open(args.md, "w") as f:
            f.write(build_md(rows, classes, formats, args))


if __name__ == "__main__":
    main()
