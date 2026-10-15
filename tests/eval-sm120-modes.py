"""
Error table of the sm120 precision modes against the upstream reference (tests/ref.py::ref_sparse_attn_decode).

Every row of the table is a combination of choices (how Q, K, P, V enter the block-scaled MMAs, how the softmax max is tracked, ...) emulated bit for bit by
sm120_emu.py on the data of the upstream test generator (`upstream`) or on data with a much larger spread of the attention logits (`hv*`, `kamp*`), where the
quantization of P and Q shows up. Run from tests/:

    python eval-sm120-modes.py --check-exact                  # acceptance: the exact mode passes the upstream tolerances on every dataset
    python eval-sm120-modes.py --csv out.csv --md out.md      # the whole table
    python eval-sm120-modes.py --list
"""
import argparse
import contextlib
import csv
import dataclasses
import fnmatch
import math
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import kernelkit as kk

import lib
import quant
import ref
import sm120_emu as emu

L = quant.KVCacheLayout
UPSTREAM_TOL = dict(out=dict(abs_tol=1e-3, rel_tol=2.01 / 128, cos_diff_tol=5e-6), lse=dict(abs_tol=1e-6, rel_tol=8.01 / 65536))


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------
@contextlib.contextmanager
def upstream_defaults():
    """The state tests/test-sparse-decode.py generates and checks data in (bf16 default dtype, cuda default device), but with exact fp32 matmuls"""
    prev_dtype, prev_prec = torch.get_default_dtype(), torch.get_float32_matmul_precision()
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.set_default_dtype(torch.bfloat16)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.device("cuda"), torch.inference_mode():
            yield
    finally:
        torch.set_default_dtype(prev_dtype)
        torch.set_float32_matmul_precision(prev_prec)
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32


# cache layout pairs (main, extra); the extra cache may be absent
FAMILIES: Dict[str, Tuple[quant.KVCacheLayout, Optional[quant.KVCacheLayout]]] = {
    "fp8": (L.V41_FP8Sparse, None),
    "fp8+fp8": (L.V41_FP8Sparse, L.V41_FP8Sparse),
    "fp8+fp4": (L.V41_FP8Sparse, L.V41_FP4Sparse),
    "fp8+mxfp4": (L.V41_FP8Sparse, L.V41_MXFP4Sparse),
    "mxfp4": (L.V41_MXFP4Sparse, L.V41_MXFP4Sparse),
    "mxfp4-main": (L.V41_MXFP4Sparse, None),
    "nvfp4-main": (L.V41_FP4Sparse, None),                   # T-10 (phase 6 G1): V41_FP4Sparse as the main cache. The upstream kernels accept it only as an extra cache
    "nvfp4": (L.V41_FP4Sparse, L.V41_FP4Sparse),
}

# data classes: sigma = target standard deviation (over valid tokens, nat) of logit * sm_scale; kamp = every token and 32-dim group of K gets a gain 2^U(-6, 6)
CLASSES: Dict[str, dict] = {
    "upstream": dict(sigma=None, kamp=False, topk_length=True),
    "hv1": dict(sigma=1.0, kamp=False, topk_length=False),
    "hv4": dict(sigma=4.0, kamp=False, topk_length=False),
    "hv16": dict(sigma=16.0, kamp=False, topk_length=False),
    "kamp1": dict(sigma=1.0, kamp=True, topk_length=False),
    "kamp4": dict(sigma=4.0, kamp=True, topk_length=False),
    "kampw4": dict(sigma=4.0, kamp=True, kamp_spread=12.0, topk_length=False),       # gains 2^U(-12, 12): beyond what e4m3 can absorb in the V rebase
    "kampw16": dict(sigma=16.0, kamp=True, kamp_spread=12.0, topk_length=False),
}


@dataclasses.dataclass
class Case:
    cls: str
    family: str
    h: int
    p: lib.TestParam
    t: lib.TestcaseForDecode
    sigma_measured: float
    seed: int = 0
    gap_mean: float = 0.0
    gap_max: float = 0.0


def quantize_without_scale_floor(k: torch.Tensor, layout: quant.KVCacheLayout, shift: int = 40) -> torch.Tensor:
    """
    quant.quantize_k_cache with the ue8m0 scale floor of 2^-13 removed (V41_FP8Sparse / V41_MXFP4Sparse): quantize 2^shift * k and subtract `shift` from the stored
    scale exponent, so that groups with a tiny amax keep full e4m3 / e2m1 precision and the exponents of the data are not squeezed into [-13, ...]. The ue4m3 scale of
    V41_FP4Sparse has its own floor (2^-9) as part of the format and is left as the writer makes it
    """
    if layout == L.V41_FP4Sparse:
        return quant.quantize_k_cache(k, layout)
    q = quant.quantize_k_cache((k.float() * 2.0 ** shift).to(torch.bfloat16), layout)
    off = 512 if layout == L.V41_FP8Sparse else 256
    sc = q.view(q.shape[0], q.shape[1], -1)[..., off:off + 16].view(torch.uint8)
    sc.copy_(torch.where(sc != 0xFF, sc - shift, sc))        # 0xFF = NaN scale of the unused rows
    return q


def apply_k_gain(t: lib.TestcaseForDecode, seed: int, spread: float = 6.0):
    """
    Multiply every (token, 32-dim group) of K by the gain 2^U(-2 * spread, 0) and re-quantize the caches without the scale floor, so the group exponents of nearby tokens
    differ by up to 2 * spread. The largest gain is 1, so the absolute scale of V (and with it of the output error) never exceeds that of the upstream data and the
    upstream tolerances stay meaningful
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)
    for scope in [t.kv_scope] + ([t.extra_kv_scope] if t.extra_kv_scope is not None else []):
        k = scope.blocked_k                                      # [num_blocks, block_size, 1, 512] bf16, NaN for unused rows
        u = torch.rand(*k.shape[:-1], 16, generator=gen, dtype=torch.float32) * (2 * spread) - 2 * spread
        k_new = (k.float().view(*k.shape[:-1], 16, 32) * torch.exp2(u).unsqueeze(-1)).view(k.shape).to(torch.bfloat16)
        layout = scope.kvcache_layout or L.V41_FP8Sparse
        if layout == L.V41_FP4Sparse and _FP4_2L_CFG is not None:
            scope.blocked_k_quantized, scope.fp4_g = quant.quantize_k_cache_fp4_2l(k_new, **_FP4_2L_CFG)
            scope.blocked_k = quant.dequantize_k_cache_fp4_2l(scope.blocked_k_quantized, scope.fp4_g)
            continue
        scope.blocked_k_quantized = quantize_without_scale_floor(k_new, layout)
        scope.blocked_k = quant.dequantize_k_cache(scope.blocked_k_quantized, layout)


def exp_gap_stats(t: lib.TestcaseForDecode) -> Tuple[float, float]:
    """
    Spread of the 32-dim group exponents among the valid tokens of one block of 64 (what the V rebase has to absorb): (mean, max) of max e - min e over (query token, block, group),
    groups whose data is all zero are ignored (their exponent is irrelevant)
    """
    b, s_q, h, d = t.q.shape
    gaps = []
    for scope in [t.kv_scope] + ([t.extra_kv_scope] if t.extra_kv_scope is not None else []):
        sd = emu.gather_scope(scope, b, s_q)
        codes, e = emu.eight_bit_terms(sd, "x1")[0]
        bs, T, _ = e.shape
        nz = codes.view(bs, T, 16, 32).abs().amax(-1) > 0
        e = e.view(bs, T // 64, 64, 16)
        v = (sd.valid.view(bs, T, 1) & nz).view(bs, T // 64, 64, 16)
        big = 1 << 20
        mx = torch.where(v, e, torch.full_like(e, -big)).amax(2)
        mn = torch.where(v, e, torch.full_like(e, big)).amin(2)
        cnt = v.sum(2)
        gaps.append((mx - mn)[cnt >= 2].float())
    g = torch.cat(gaps)
    return (float(g.mean()), float(g.max())) if g.numel() else (0.0, 0.0)


def logit_std(t: lib.TestcaseForDecode) -> float:
    """Mean over (query token, head) rows of the std over valid tokens of logit * sm_scale (exact bf16 path)"""
    b, s_q, h, d = t.q.shape
    q = t.q.reshape(b * s_q, h, d).float()
    qt = emu.quantize_q(q, "bf16")
    ls, vs = [], []
    for scope in [t.kv_scope] + ([t.extra_kv_scope] if t.extra_kv_scope is not None else []):
        sd = emu.gather_scope(scope, b, s_q)
        ls.append(emu.qk_logits(qt, emu.k_terms(sd, emu.EXACT_OPS, emu.Mode()), "f64")), vs.append(sd.valid)
    x = torch.cat(ls, dim=2).double() * t.sm_scale
    v = torch.cat(vs, dim=1).unsqueeze(1).double()
    n = v.sum(-1)
    mean = (x * v).sum(-1) / n.clamp_min(1)
    var = (((x - mean.unsqueeze(-1)) * v) ** 2).sum(-1) / n.clamp_min(1)
    rows = (n >= 8).expand_as(var)
    return float(var.sqrt()[rows].mean()) if rows.any() else float("nan")


_FP4_2L_CFG: Optional[dict] = None      # set by fp4_two_level_writer: the arguments of quant.quantize_k_cache_fp4_2l for the V41_FP4Sparse caches of the data being generated


@contextlib.contextmanager
def fp4_two_level_writer(cfg: Optional[dict]):
    """While active, the quant_and_dequant_ of lib.KVScope (and the re-quantization of apply_k_gain) write V41_FP4Sparse caches with the two-level writer and keep the integer(s) g in scope.fp4_g"""
    global _FP4_2L_CFG
    if cfg is None:
        yield
        return
    orig = lib.KVScope.quant_and_dequant_

    def patched(self):
        if (self.kvcache_layout or L.V41_FP8Sparse) != L.V41_FP4Sparse:
            return orig(self)
        self.blocked_k_quantized, self.fp4_g = quant.quantize_k_cache_fp4_2l(self.blocked_k, **cfg)
        self.blocked_k = quant.dequantize_k_cache_fp4_2l(self.blocked_k_quantized, self.fp4_g)

    lib.KVScope.quant_and_dequant_ = patched
    _FP4_2L_CFG = cfg
    try:
        yield
    finally:
        lib.KVScope.quant_and_dequant_ = orig
        _FP4_2L_CFG = None


def make_case(cls: str, family: str, h: int, seed: int, b: int, s_q: int, topk: int = 192, extra_topk: int = 576, fp4_2l: Optional[dict] = None) -> Case:
    with fp4_two_level_writer(fp4_2l):
        return _make_case(cls, family, h, seed, b, s_q, topk, extra_topk)


def _make_case(cls: str, family: str, h: int, seed: int, b: int, s_q: int, topk: int, extra_topk: int) -> Case:
    cfg = CLASSES[cls]
    main, extra = FAMILIES[family]
    has_extra = extra is not None
    raw = lib.RawTestParamForDecode(
        b=b, h_q=h, s_q=s_q, h_kv=1, s_kv=1024, is_varlen=True, topk=topk,
        have_topk_length=cfg["topk_length"], enable_attn_sink=True,
        extra_s_k=2048 if has_extra else None, extra_topk=extra_topk if has_extra else None,
        block_size=64, extra_block_size=64 if has_extra else None,
        have_extra_topk_length=cfg["topk_length"] and has_extra,
        d_qk=512, kvcache_layout=main, extra_kvcache_layout=extra, check_correctness=True, num_runs=0, seed=seed)
    p = raw.to_test_param()
    t = lib.generate_testcase_for_decode(p)
    if cfg["kamp"]:
        apply_k_gain(t, seed, cfg.get("kamp_spread", 6.0))
    if cfg["sigma"] is not None:
        t.sm_scale *= cfg["sigma"] / logit_std(t)
    return Case(cls, family, h, p, t, logit_std(t))


CORNER_EXTRA = {"nvfp4": dict(have_zero_seqlen_k=True, have_topk_length=True, have_extra_topk_length=True), "nvfp4-main": dict(have_zero_seqlen_k=True, have_topk_length=True)}      # T-10: appended after the four fixed cases


def make_corner_cases(seed0: int, b: int, s_q: int, extra_families: Tuple[str, ...] = ()) -> List[Case]:
    """All-invalid indices and zero-length sequences with attn_sink / topk_length: lonely query tokens (out = 0, lse = +inf) and empty blocks"""
    cases = []
    for i, (kw, family) in enumerate([(dict(is_all_indices_invalid=True), "fp8+fp4"), (dict(have_zero_seqlen_k=True, have_topk_length=True, have_extra_topk_length=True), "fp8+fp8"),
                                      (dict(have_zero_seqlen_k=True, have_topk_length=True), "fp8"), (dict(have_zero_seqlen_k=True, have_topk_length=True, have_extra_topk_length=True), "mxfp4")]
                                     + [(CORNER_EXTRA[f], f) for f in extra_families]):
        main, extra = FAMILIES[family]
        has_extra = extra is not None
        base = dict(b=b, h_q=64, s_q=s_q, h_kv=1, s_kv=512, is_varlen=True, topk=128, enable_attn_sink=True, extra_s_k=512 if has_extra else None, extra_topk=128 if has_extra else None,
                    block_size=64, extra_block_size=64 if has_extra else None, d_qk=512, kvcache_layout=main, extra_kvcache_layout=extra, check_correctness=True, num_runs=0, seed=seed0 + i)
        if not has_extra:
            kw = {k: v for k, v in kw.items() if k != "have_extra_topk_length"}
        base.update(kw)
        p = lib.RawTestParamForDecode(**base).to_test_param()
        t = lib.generate_testcase_for_decode(p)
        cases.append(Case("corner", family, 64, p, t, float("nan")))
    return cases


# ----------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------
def metrics(out: torch.Tensor, lse: torch.Tensor, out_ref: torch.Tensor, lse_ref: torch.Tensor) -> dict:
    o, r = out.float(), out_ref.float()
    d = (o - r).abs()
    max_abs = float(d.max())
    ref_max = float(r.abs().max())
    rel_el = (d / (r.abs() + 1e-6)).masked_fill(d < UPSTREAM_TOL["out"]["abs_tol"], 0)      # relative error counted only where the absolute error matters (the upstream rule)
    out_ok = kk.check_is_allclose("out", out, out_ref, quiet=True, **UPSTREAM_TOL["out"])
    tol = UPSTREAM_TOL["out"]
    raw_rel = d / (r.abs() + 1e-6)          # the elementwise rule of kk.check_is_allclose, kept apart from the cos_diff rule to see which one fails
    elem_ok = bool(((d.masked_fill(raw_rel < tol["rel_tol"], 0) < tol["abs_tol"]) | (raw_rel.masked_fill(d < tol["abs_tol"], 0) < tol["rel_tol"])).all())
    lse_ok = kk.check_is_allclose("lse", lse, lse_ref, quiet=True, **UPSTREAM_TOL["lse"])
    fin = torch.isfinite(lse) & torch.isfinite(lse_ref)
    inf_mismatch = int((torch.isinf(lse) != torch.isinf(lse_ref)).sum() + (torch.isinf(lse) & torch.isinf(lse_ref) & (lse != lse_ref)).sum())
    dl = (lse - lse_ref).abs()
    rn = (d.double().pow(2).sum(-1).sqrt() / r.double().pow(2).sum(-1).sqrt().clamp_min(1e-30)).flatten()      # relative L2 error of every (query token, head) row
    rw = r.double().pow(2).sum(-1).sqrt().flatten()
    rows_ok = rw > 1e-2 * rw[rw > 0].median() if (rw > 0).any() else rw > 0                                       # ignore all-zero rows and rows far below the typical output size
    nz = rw > 0
    return dict(
        row_rel_p99=float(torch.quantile(rn[rows_ok].float(), 0.99)) if rows_ok.any() else 0.0, row_rel_max=float(rn[rows_ok].max()) if rows_ok.any() else 0.0,
        rowall_rel_p99=float(torch.quantile(rn[nz].float(), 0.99)) if nz.any() else 0.0, rowall_rel_max=float(rn[nz].max()) if nz.any() else 0.0,
        cos_diff=float(kk.get_cos_diff(out, out_ref)),
        rel_rms=float((d.double().pow(2).sum() / r.double().pow(2).sum().clamp_min(1e-30)).sqrt()),
        max_abs=max_abs, max_rel_norm=max_abs / max(ref_max, 1e-30), max_rel_el=float(rel_el.max()),
        out_ok=bool(out_ok), out_elem_ok=elem_ok, out_cos_ok=bool(float(kk.get_cos_diff(out, out_ref)) <= tol["cos_diff_tol"]),
        lse_max_abs=float(dl[fin].max()) if fin.any() else 0.0,
        lse_max_rel=float((dl / (lse_ref.abs() + 1e-6))[fin].max()) if fin.any() else 0.0,
        lse_ok=bool(lse_ok) and inf_mismatch == 0, lse_inf_mismatch=inf_mismatch,
    )


# ----------------------------------------------------------------------------------------------
# Mode combinations
# ----------------------------------------------------------------------------------------------
OM = emu.OperandMode
ALL = ("fp8", "fp8+fp8", "fp8+fp4", "fp8+mxfp4", "mxfp4", "mxfp4-main", "nvfp4-main", "nvfp4")
FP8_LIKE = ("fp8", "fp8+fp8", "fp8+fp4", "fp8+mxfp4", "mxfp4", "mxfp4-main", "nvfp4-main", "nvfp4")      # every family has MXFP8-compatible operands (fp4 caches via the k8 conversion)
NVFP4_KERNEL_FAMILIES = ("fp8+fp4", "nvfp4", "nvfp4-main")           # the caches with V41_FP4Sparse tiles (T-10: the rows kernel-nvfp4-*)
KERNEL_FAMILIES = ("fp8", "fp8+fp8")                                  # the cache types the P2 kernel supports
MXFP4_KERNEL_FAMILIES = ("mxfp4", "mxfp4-main")                       # the cache types mxfp4.cu supports
HAS_FP4 = ("fp8+fp4",)
HAS_MXFP4 = ("fp8+mxfp4", "mxfp4", "mxfp4-main")


def ops8(q: str = "mxfp8x1", v: str = "rebase", k4: str = "k8x1", v8: str = "x1") -> dict:
    """Operands of an MXFP8 pipeline for the three cache layouts (fp4 caches are converted to e4m3 + ue8m0 first)"""
    return dict(fp8=OM(q, "native", v), mxfp4=OM(q, "native", v), fp4=OM(q, k4, v, v8))


def build_combos() -> List[Tuple[str, str, emu.Mode, Tuple[str, ...]]]:
    """(name, description, mode, families it applies to)"""
    C = []

    def add(name, desc, families=ALL, **kw):
        C.append((name, desc, emu.Mode(**kw), families))

    # ---- exact: bf16 Q, bf16 K / V from the dequantized cache, P rounded to bf16 (the upstream arithmetic)
    add("exact", "q-bf16 k-exact p-bf16 v-exact (upstream arithmetic), m-lazy, sum-exact")
    add("exact-mfinal", "exact with m-final", m="final")
    add("exact-f32seg", "exact with fp32 accumulation per 32-k MMA step", accum="f32seg")
    add("exact-sumq", "exact with the denominator from bf16-rounded P", sum="quant")

    # ---- MXFP8 pipeline: Q e4m3 (+ ue8m0 per 32), K native, P e4m3, V rebased
    mx = ops8()
    add("mxfp8", "q-mxfp8x1 p-e4m3x1 v-rebase (const SF_A, m-lazy, B=64, sum-exact)", FP8_LIKE, **mx, p="e4m3x1", p_sf="const")
    add("mxfp8-Q", "only Q -> mxfp8x1", FP8_LIKE, **ops8(v="exact"))
    add("mxfp8-P", "only P -> e4m3x1", p="e4m3x1")
    add("mxfp8-V", "only V -> rebase (B=64)", FP8_LIKE, **{k: dataclasses.replace(OM(), v="rebase", v8="x1") for k in ("fp8", "fp4", "mxfp4")})
    add("mxfp8-V-rz", "only V -> rebase with truncation (round toward zero) when the shift lands in subnormals", FP8_LIKE, rebase_round="rz",
        **{k: dataclasses.replace(OM(), v="rebase", v8="x1") for k in ("fp8", "fp4", "mxfp4")})
    add("mxfp8-P-sfrow", "only P -> e4m3x1 with SF_A per row and 32 tokens", p="e4m3x1", p_sf="row")
    rb = {k: dataclasses.replace(OM(), v="rebase", v8="x1") for k in ("fp8", "fp4", "mxfp4")}
    add("exact-f32p", "bf16 QK and PV with P left in float32: the reference arithmetic up to the output rounding (baseline of the f32p rows)", p="f32")
    add("vrebase-f32p", "V rebase (B=64) with P in float32: the error of the V rebase alone, no P rounding noise", FP8_LIKE, p="f32", **rb)
    add("vrebase-f32p-rz", "same with truncation when the shift lands in subnormals", FP8_LIKE, p="f32", rebase_round="rz", **rb)
    add("vrebase-f32p-B32", "same with blocks of 32 tokens", FP8_LIKE, p="f32", B=32, **rb)
    add("fp4-v8x2-f32p", "V of the fp4 extra as hi + lo (v8 x2) rebased, P in float32", HAS_FP4, p="f32", fp4=OM("bf16", "exact", "rebase", "x2"))
    add("mxfp8-sfrow", "mxfp8 with SF_A per row and 32 tokens (from the block max P)", FP8_LIKE, **mx, p="e4m3x1", p_sf="row")
    add("mxfp8-mfinal", "mxfp8 with m-final (P <= 1, const SF_A = 2^-8)", FP8_LIKE, **mx, p="e4m3x1", p_sf="const", m="final")
    add("mxfp8-mfinal-sfrow", "mxfp8 with m-final and SF_A per row / 32 tokens", FP8_LIKE, **mx, p="e4m3x1", p_sf="row", m="final")
    add("mxfp8-sumq", "mxfp8 with the denominator from the quantized P", FP8_LIKE, **mx, p="e4m3x1", sum="quant")
    add("mxfp8-B32", "mxfp8 with B=32 (lazy-max blocks and rebase blocks of 32 tokens)", FP8_LIKE, **mx, p="e4m3x1", B=32)
    add("mxfp8-f32seg", "mxfp8 with fp32 accumulation per 32-k MMA step", FP8_LIKE, **mx, p="e4m3x1", accum="f32seg")

    # ---- fp8x2: Q and P as hi + lo e4m3 terms (lo scale 2^-4 of hi), V rebased
    x2 = ops8(q="mxfp8x2")
    add("fp8x2", "q-mxfp8x2 p-e4m3x2 v-rebase (const SF_A, m-lazy, B=64, sum-exact)", FP8_LIKE, **x2, p="e4m3x2", p_sf="const")
    add("q1p2", "q-mxfp8x1 p-e4m3x2 v-rebase (one QK MMA stream, two PV streams)", FP8_LIKE, **mx, p="e4m3x2")
    add("q2p1", "q-mxfp8x2 p-e4m3x1 v-rebase (two QK streams, one PV stream)", FP8_LIKE, **x2, p="e4m3x1")
    add("fp8x2-Q", "only Q -> mxfp8x2 (lo scale 2^-4)", FP8_LIKE, **ops8(q="mxfp8x2", v="exact"))
    add("fp8x2-Q-s0", "only Q -> mxfp8x2 with the lo term on the same scale as hi (s=0)", FP8_LIKE, q_lo_shift=0, **ops8(q="mxfp8x2", v="exact"))
    add("fp8x2-P", "only P -> e4m3x2 (lo scale 2^-4)", p="e4m3x2")
    add("fp8x2-P-s0", "only P -> e4m3x2 with s=0", p="e4m3x2", p_lo_shift=0)
    add("fp8x2-P-sfrow", "only P -> e4m3x2 with SF_A per row and 32 tokens", p="e4m3x2", p_sf="row")
    add("fp8x2-V", "only V -> rebase (same as mxfp8-V: V has no x2 form for fp8 caches)", FP8_LIKE, **{k: dataclasses.replace(OM(), v="rebase", v8="x1") for k in ("fp8", "fp4", "mxfp4")})
    add("fp8x2-sfrow", "fp8x2 with SF_A per row and 32 tokens", FP8_LIKE, **x2, p="e4m3x2", p_sf="row")
    add("fp8x2-mfinal", "fp8x2 with m-final", FP8_LIKE, **x2, p="e4m3x2", p_sf="const", m="final")
    add("fp8x2-mfinal-sfrow", "fp8x2 with m-final and SF_A per row / 32 tokens", FP8_LIKE, **x2, p="e4m3x2", p_sf="row", m="final")
    add("fp8x2-sumq", "fp8x2 with the denominator from the quantized P", FP8_LIKE, **x2, p="e4m3x2", sum="quant")
    add("fp8x2-B32", "fp8x2 with B=32", FP8_LIKE, **x2, p="e4m3x2", B=32)
    add("fp8x2-f32seg", "fp8x2 with fp32 accumulation per 32-k MMA step", FP8_LIKE, **x2, p="e4m3x2", accum="f32seg")

    # ---- the P2 kernel (csrc/cuda_kernels/sm120/decode/sparse/mxfp8.cu), parameter for parameter: Q scale = ceil-to-power-of-two(amax * fp32(1/448)) (code clamped to [8, 250]), lo scale 2^-4;
    #      P = hi * 2^-2 + lo * 2^-6 (constant SF), lazy max (threshold 6), rowsum from the unquantized P; V rebased per 32 tokens with a fp16 multiply and RN back to e4m3 (= single RN);
    #      accumulation in fp32 per 32-k MMA step. NUM_TERMS selects both the Q terms and the P terms; the mixed variants are the candidates for a cheaper middle tier
    kern = dict(B=32, accum="f32seg", q_kernel_rule=True, m="lazy", p_sf="const", sum="exact", q_lo_shift=4, p_lo_shift=4)
    add("kernel-fp8x2", "the P2 kernel with NUM_TERMS=2: Q x2 + P x2 + V rebase, B=32, fp32 per-step accumulation", KERNEL_FAMILIES, **ops8(q="mxfp8x2"), p="e4m3x2", **kern)
    add("kernel-mxfp8", "the P2 kernel with NUM_TERMS=1: Q x1 + P x1 + V rebase", KERNEL_FAMILIES, **ops8(q="mxfp8x1"), p="e4m3x1", **kern)
    add("kernel-q2p1", "middle tier: Q x2 + P x1 (one MMA less than fp8x2 in PV, a quarter of all MMAs saved) with the kernel parameters", KERNEL_FAMILIES, **ops8(q="mxfp8x2"), p="e4m3x1", **kern)
    add("kernel-q1p2", "other middle tier: Q x1 + P x2 with the kernel parameters", KERNEL_FAMILIES, **ops8(q="mxfp8x1"), p="e4m3x2", **kern)

    # ---- V41_FP4 extra cache (e2m1 x e4m3 per 16): K / V conversions and the native NVFP4 Q
    add("fp4-kexact", "main: mxfp8 pipeline; fp4 extra: bf16 QK and bf16 PV (k-exact)", HAS_FP4, fp8=mx["fp8"], fp4=OM(), p="e4m3x1")
    add("fp4-k8x1", "mxfp8 pipeline, fp4 extra converted per token to e4m3 + ue8m0 (k8x1, v8 x1)", HAS_FP4, **mx, p="e4m3x1")
    add("fp4-k8x2", "mxfp8x1 Q, fp4 extra as hi + lo e4m3 terms for K and V (k8x2, v8 x2)", HAS_FP4, **ops8(k4="k8x2", v8="x2"), p="e4m3x1")
    add("fp4-fp8x2-k8x1", "fp8x2 pipeline, fp4 extra via k8x1", HAS_FP4, **x2, p="e4m3x2")
    add("fp4-fp8x2-k8x2", "fp8x2 pipeline, fp4 extra via k8x2 / v8 x2", HAS_FP4, **ops8(q="mxfp8x2", k4="k8x2", v8="x2"), p="e4m3x2")
    add("fp4-k8x1-qk", "QK only: q-mxfp8x1, fp4 extra converted to e4m3 + ue8m0 (k8x1); everything else exact", HAS_FP4, fp4=OM("mxfp8x1", "k8x1", "exact"))
    add("fp4-k8x2-qk", "QK only: q-mxfp8x2, fp4 extra as hi + lo (k8x2); everything else exact", HAS_FP4, fp4=OM("mxfp8x2", "k8x2", "exact"))
    add("fp4-k8x2-qk-lolo", "QK only: q-mxfp8x2 x k8x2 with all four MMA pairs (lo x lo issued)", HAS_FP4, fp4=OM("mxfp8x2", "k8x2", "exact"), lolo=True)
    add("fp4-fp8x2-k8x2-lolo", "fp4-fp8x2-k8x2 with the lo x lo MMAs issued in QK and PV", HAS_FP4, **ops8(q="mxfp8x2", k4="k8x2", v8="x2"), p="e4m3x2", lolo=True)
    add("fp4-v8x1-pv", "PV only: V of the fp4 extra converted (v8 x1) and rebased; Q, K, P exact", HAS_FP4, fp4=OM("bf16", "exact", "rebase", "x1"))
    add("fp4-v8x2-pv", "PV only: V of the fp4 extra as hi + lo (v8 x2) and rebased; Q, K, P exact", HAS_FP4, fp4=OM("bf16", "exact", "rebase", "x2"))
    for n in (1, 2, 3, 4, 6):
        add(f"nvfp4x{n}-qk", f"q-nvfp4x{n} + k-fp4-native on the fp4 extra (QK only; everything else exact)", HAS_FP4, fp4=OM(f"nvfp4x{n}", "native", "exact"))
    for n in (4, 5, 6, 8):
        add(f"nvfp4x{n}pa-qk", f"q-nvfp4x{n} with a per-row power-of-two prescale (moves the ue4m3 scale floor 2^-9 away from the residual terms) + k-fp4-native on the fp4 extra (QK only)",
            HAS_FP4, fp4=OM(f"nvfp4x{n}pa", "native", "exact"))
    add("nvfp4x6pa-full", "main fp8x2 pipeline; fp4 extra: q-nvfp4x6 with prescale, native QK, v rebase with v8 x2; p-e4m3x2", HAS_FP4, fp8=x2["fp8"], fp4=OM("nvfp4x6pa", "native", "rebase", "x2"), p="e4m3x2")
    add("nvfp4x1-full", "main mxfp8 pipeline; fp4 extra: q-nvfp4x1 native QK, v rebase (v8 x1); p-e4m3x1", HAS_FP4, fp8=mx["fp8"], fp4=OM("nvfp4x1", "native", "rebase", "x1"), p="e4m3x1")
    add("nvfp4x2-full", "main fp8x2 pipeline; fp4 extra: q-nvfp4x2 native QK, v rebase (v8 x1); p-e4m3x2", HAS_FP4, fp8=x2["fp8"], fp4=OM("nvfp4x2", "native", "rebase", "x1"), p="e4m3x2")

    for n in (3, 4):
        add(f"nvfp4x{n}-full", f"main fp8x2 pipeline; fp4 extra: q-nvfp4x{n} native QK, v rebase with v8 x2; p-e4m3x2", HAS_FP4, fp8=x2["fp8"], fp4=OM(f"nvfp4x{n}", "native", "rebase", "x2"), p="e4m3x2")

    # ---- P3 first step: mxfp8 mode with a V41_FP4 extra cache (the kernel rows). The main cache tile = kernel-mxfp8 exactly. The fp4 tile: QK on mxf4nvf4 4X (m16n8k64) with K as stored
    #      (e2m1 + a ue4m3 scale per 16 dims) and Q re-quantized from bf16 into N NVFP4 terms (scale = the smallest e4m3 >= amax / 6 per head and 16 dims, residual chain, one MMA stream per
    #      term, fp32 accumulation per k64 instruction); an optional per-head power-of-two prescale of Q (folded into the logit scale). PV: P as e4m3 (like the main tile), V converted by the
    #      producer straight from e2m1 x s16 to e4m3 on a scale 2^E per (tile of 32 tokens, 32 dims), E = ceil(log2(amax / 448)) over the valid tokens (d1: one term, d2: hi + lo)
    nv1 = lambda n, pre="": OM(f"nvfp4x{n}{pre}", "native", "rebase", "d1")
    for n in (1, 2, 3, 4):
        add(f"p3-mxfp8-nvq{n}", f"P3 step 1, Q of the fp4 tile as {n} NVFP4 term(s), no prescale; main tile = kernel-mxfp8; fp4 V one e4m3 term (d1); P e4m3 x1; B=32", HAS_FP4, fp8=mx["fp8"], fp4=nv1(n), p="e4m3x1", **kern)
    for n in (2, 3, 4):
        add(f"p3-mxfp8-nvq{n}pa", f"P3 step 1, {n} NVFP4 Q terms with the per-head prescale 2^floor(log2(2688 / amax_head)) (largest group scale <= 448)", HAS_FP4, fp8=mx["fp8"], fp4=nv1(n, "pa"), p="e4m3x1", **kern)
    for n in (2, 3, 4):
        add(f"p3-mxfp8-nvq{n}pe", f"P3 step 1, {n} NVFP4 Q terms with the exponent-only prescale 2^(10 - floor(log2(amax_head))) (head amax in [1024, 2048))", HAS_FP4, fp8=mx["fp8"], fp4=nv1(n, "pe"), p="e4m3x1", **kern)
    # floor of the mode: the main tile as above, an ideal fp4 tile (bf16 QK and bf16 PV on the dequantized cache); Q of the fp4 tile requantized from the e4m3 Q of the main tile (q8) instead of bf16 Q
    add("p3-mxfp8-fp4exact", "P3 step 1 floor: main tile = kernel-mxfp8, fp4 tile exact (bf16 QK and PV), P e4m3x1, B=32: the error the mode has even with a perfect fp4 tile", HAS_FP4, fp8=mx["fp8"], fp4=OM(), p="e4m3x1", **kern)
    q8 = lambda n, pre="": dataclasses.replace(nv1(n, pre), q_src="mxfp8x1")
    for n in (1, 2, 3, 4):
        add(f"p3-mxfp8-nvq{n}-q8", f"as p3-mxfp8-nvq{n} but the NVFP4 Q terms are quantized from the e4m3 Q of the main tile (mxfp8x1, kernel rule) instead of bf16 Q", HAS_FP4, fp8=mx["fp8"], fp4=q8(n), p="e4m3x1", **kern)
    for n in (2, 3):
        add(f"p3-mxfp8-nvq{n}pe-q8", f"as p3-mxfp8-nvq{n}-q8 with the exponent prescale 'pe'", HAS_FP4, fp8=mx["fp8"], fp4=q8(n, "pe"), p="e4m3x1", **kern)
    for n in (2, 3):
        add(f"p3-mxfp8-nvq{n}-v2", f"as p3-mxfp8-nvq{n} but the V of the fp4 tile as hi + lo e4m3 (d2): the PV of the fp4 tile issues P V_hi and P V_lo (two MMAs instead of one)", HAS_FP4,
            fp8=mx["fp8"], fp4=OM(f"nvfp4x{n}", "native", "rebase", "d2"), p="e4m3x1", **kern)
    for n in (2, 3):
        add(f"p3-mxfp8-nvq{n}s", f"as p3-mxfp8-nvq{n} with the V scale from 6 x the largest 16-dim scale of the valid tokens (d1s) instead of their amax", HAS_FP4,
            fp8=mx["fp8"], fp4=OM(f"nvfp4x{n}", "native", "rebase", "d1s"), p="e4m3x1", **kern)
    add("p3-fp8x2-nvq4pa-v1", "control: main tile = kernel-fp8x2 (Q x2, P x2); fp4 tile: 4 prescaled NVFP4 Q terms, P x2, V one e4m3 term (d1)", HAS_FP4, fp8=x2["fp8"], fp4=nv1(4, "pa"), p="e4m3x2", **kern)
    add("p3-fp8x2-nvq4pa-v2", "control: as p3-fp8x2-nvq4pa-v1 with V as hi + lo (d2); PV issues P_hi V_hi, P_hi V_lo, P_lo V_hi (lo x lo not issued)", HAS_FP4, fp8=x2["fp8"],
        fp4=OM("nvfp4x4pa", "native", "rebase", "d2"), p="e4m3x2", **kern)
    # attribution rows (one source of error each, the rest exact; the kernel parameters otherwise)
    for n in (1, 2, 3, 4):
        add(f"p3-qk-nvq{n}", f"QK only of the fp4 tile: {n} NVFP4 Q term(s) (kernel scale rule), no prescale; everything else exact", HAS_FP4, fp4=OM(f"nvfp4x{n}", "native", "exact"), **kern)
        add(f"p3-qk-nvq{n}pa", f"QK only of the fp4 tile: {n} NVFP4 Q term(s) with the prescale 'pa'; everything else exact", HAS_FP4, fp4=OM(f"nvfp4x{n}pa", "native", "exact"), **kern)
    add("p3-pv-d1", "PV only: V of the fp4 tile as one e4m3 term on the tile scale (d1); P bf16", HAS_FP4, fp4=OM("bf16", "exact", "rebase", "d1"), **kern)
    add("p3-pv-d2", "PV only: V of the fp4 tile as hi + lo e4m3 on the tile scale (d2); P bf16", HAS_FP4, fp4=OM("bf16", "exact", "rebase", "d2"), **kern)
    add("p3-p1-only", "P as one e4m3 term only (B=32, constant SF_A); everything else exact: the error floor of the one-term P in the mxfp8 mode", HAS_FP4, p="e4m3x1", **kern)
    add("p3-p2-only", "P as two e4m3 terms only; everything else exact", HAS_FP4, p="e4m3x2", **kern)

    # ---- T-10 (phase 6 G1): the kernel rows for caches with V41_FP4Sparse tiles (fp8+fp4: the extra cache; nvfp4-main: the main cache; nvfp4: both). The FP4 tiles are the same wherever they sit.
    #      kernel-nvfp4-mxfp8 = the mxfp8.cu instance <1, true> bit for bit (= p3-mxfp8-nvq2s: Q two NVFP4 terms with the kernel scale rule, K as stored, V one e4m3 term on the scale
    #      2^ceil(log2(6 * largest 16-dim scale of the tile / 448)), P one e4m3 term, B = 32, fp32 per k64 instruction). The 2^g of a two-level cache is part of the stored scales here
    #      (gather_scope / decode_native), which is what the kernel does by folding it into the logit scale and the rebase tile exponent
    #      kernel-nvfp4-fp8x2 = the precise tier (G2a): main FP8 tile = kernel-fp8x2; FP4 tile: Q four NVFP4 terms with a per-head power-of-two second level (pa), P two e4m3 terms,
    #      V hi + lo (d2, lo scale 2^-4), lo x lo not issued (= p3-fp8x2-nvq4pa-v2). Variants: Q x3 / x5, V scale from 6 x the largest 16-dim scale (d2s), Q without the second level
    fp4_tile = lambda q, v8="d2": OM(q, "native", "rebase", v8)
    add("kernel-nvfp4-mxfp8", "the mxfp8 tier on V41_FP4Sparse tiles (= mxfp8.cu <1, true>): Q x2 NVFP4, V one e4m3 term (d1s), P x1, B=32, fp32 per k64 instruction; main FP8 tile = kernel-mxfp8; "
        "an fp4 extra cache next to an fp8 main cache comes first", NVFP4_KERNEL_FAMILIES, fp8=mx["fp8"], fp4=OM("nvfp4x2", "native", "rebase", "d1s"), p="e4m3x1", extra_first=True, **kern)
    add("kernel-nvfp4-fp8x2", "the precise tier on V41_FP4Sparse tiles: Q x4 NVFP4 with a per-head 2^k second level (pa), V hi + lo (d2), P x2, B=32; main FP8 tile = kernel-fp8x2",
        NVFP4_KERNEL_FAMILIES, fp8=x2["fp8"], fp4=fp4_tile("nvfp4x4pa"), p="e4m3x2", **kern)
    add("kernel-nvfp4-fp8x2-q3", "kernel-nvfp4-fp8x2 with Q x3 (second level)", NVFP4_KERNEL_FAMILIES, fp8=x2["fp8"], fp4=fp4_tile("nvfp4x3pa"), p="e4m3x2", **kern)
    add("kernel-nvfp4-fp8x2-q5", "kernel-nvfp4-fp8x2 with Q x5 (second level)", NVFP4_KERNEL_FAMILIES, fp8=x2["fp8"], fp4=fp4_tile("nvfp4x5pa"), p="e4m3x2", **kern)
    add("kernel-nvfp4-fp8x2-q4one", "kernel-nvfp4-fp8x2 with Q x4 on a single level (no second level): the control that does not pass", NVFP4_KERNEL_FAMILIES, fp8=x2["fp8"], fp4=fp4_tile("nvfp4x4"), p="e4m3x2", **kern)
    add("kernel-nvfp4-fp8x2-v2s", "kernel-nvfp4-fp8x2 with the V scale from 6 x the largest 16-dim scale (d2s)", NVFP4_KERNEL_FAMILIES, fp8=x2["fp8"], fp4=fp4_tile("nvfp4x4pa", "d2s"), p="e4m3x2", **kern)
    add("kernel-nvfp4-fp8x2-q3v2s", "kernel-nvfp4-fp8x2 with Q x3 (second level) and d2s", NVFP4_KERNEL_FAMILIES, fp8=x2["fp8"], fp4=fp4_tile("nvfp4x3pa", "d2s"), p="e4m3x2", **kern)
    add("kernel-nvfp4-fp8x2-q5v2s", "kernel-nvfp4-fp8x2 with Q x5 (second level) and d2s", NVFP4_KERNEL_FAMILIES, fp8=x2["fp8"], fp4=fp4_tile("nvfp4x5pa", "d2s"), p="e4m3x2", **kern)
    add("kernel-nvfp4-q4p1", "precise-tier variant: Q x4 (second level), V hi + lo, P x1", NVFP4_KERNEL_FAMILIES, fp8=mx["fp8"], fp4=fp4_tile("nvfp4x4pa"), p="e4m3x1", **kern)

    # ---- MXFP4 (e2m1 + ue8m0 per 32): Q as e4m3 (mxf8f6f4) or e2m1 (mxf4)
    for q in ("mxfp8x1", "mxfp8x2", "mxfp4x1", "mxfp4x2", "mxfp4x3", "mxfp4x4", "mxfp4x6"):
        add(f"mxfp4-{q}-qk", f"q-{q} on the MXFP4 cache (QK only; everything else exact)", HAS_MXFP4, mxfp4=OM(q, "native", "exact"))
    add("mxfp4-mxfp8x1-full", "= mxfp8 pipeline on the MXFP4 cache (e2m1 is an exact e4m3 value)", HAS_MXFP4, **mx, p="e4m3x1")
    add("mxfp4-mxfp4x2-full", "main fp8x2 pipeline; mxfp4 cache: q-mxfp4x2 QK, v rebase; p-e4m3x2", HAS_MXFP4, fp8=x2["fp8"], mxfp4=OM("mxfp4x2", "native", "rebase"), p="e4m3x2")
    for n in (3, 4):
        add(f"mxfp4-mxfp4x{n}-full", f"main fp8x2 pipeline; mxfp4 cache: q-mxfp4x{n} QK, v rebase; p-e4m3x2", HAS_MXFP4, fp8=x2["fp8"], mxfp4=OM(f"mxfp4x{n}", "native", "rebase"), p="e4m3x2")

    # ---- the MXFP4 kernel (csrc/cuda_kernels/sm120/decode/sparse/mxfp4.cu), parameter for parameter: QK on mxf4 2X (m16n8k64, fp32 per instruction), Q as N e2m1 terms with a
    #      ue8m0 scale per head and 32 dims (ceil-to-power-of-two(amax * fp32(1/6)), code clamped to [8, 250], residual chain), P = hi * 2^-2 + lo * 2^-6 as for mxfp8.cu,
    #      V rebased per 32 tokens with the block exponent = (largest exponent of the valid tokens) - 6. mxfp8 tier: 2 Q terms, 1 P term; mxfp4x4 tier: 4 and 2
    mxk = lambda n: dict(mxfp4=OM(f"mxfp4x{n}", "native", "rebase", v_off=-6))
    kern4 = dict(kern, accum="f32pair")       # QK of the two warps of a pair: partial sums over the two halves of the dims
    add("kernel-mxfp4-mxfp8", "the MXFP4 kernel, mxfp8 tier: Q x2 (mxfp4), P x1, V rebase with 6 exponent steps of headroom, B=32, fp32 per-instruction accumulation (partial sums per warp of a pair)", MXFP4_KERNEL_FAMILIES,
        **mxk(2), p="e4m3x1", **kern4)
    add("kernel-mxfp4-fp8x2", "the MXFP4 kernel, mxfp4x4 tier: Q x4 (mxfp4), P x2, V rebase with headroom", MXFP4_KERNEL_FAMILIES, **mxk(4), p="e4m3x2", **kern4)
    add("kernel-mxfp4-q3p2", "MXFP4 kernel variant: Q x3, P x2", MXFP4_KERNEL_FAMILIES, **mxk(3), p="e4m3x2", **kern4)
    add("kernel-mxfp4-q4p1", "MXFP4 kernel variant: Q x4, P x1", MXFP4_KERNEL_FAMILIES, **mxk(4), p="e4m3x1", **kern4)
    add("kernel-mxfp4-q2p2", "MXFP4 kernel variant: Q x2, P x2", MXFP4_KERNEL_FAMILIES, **mxk(2), p="e4m3x2", **kern4)
    add("kernel-mxfp4-nohead", "kernel-mxfp4-fp8x2 without the V headroom (rebase to the largest exponent, like mxfp8.cu)", MXFP4_KERNEL_FAMILIES,
        mxfp4=OM("mxfp4x4", "native", "rebase"), p="e4m3x2", **kern4)

    # ---- FP4 PV ("nvfp4" mode, T-08): QK and PV on kind::mxf4nvf4 / mxf4. Every row uses the 64-token pair (B = 64), fp32 accumulation per instruction, P kperm groups unless named otherwise.
    #      The reference row reproduces the algorithm of FlashInfer's DSv4 NVFP4 kernel (sparse_mla_sm120/kernels/dsv4_nvfp4/grouped_attention.cuh) on the V4.1 cache formats: Q one NVFP4 term (rule rn),
    #      K native NVFP4 (FP8 / MXFP4 caches re-quantized per token and 16 dims, nvf4r, rule rn), online max updated for every 64 candidates (eager: P <= 1), P rounded to bf16 and quantized per
    #      (head, 16 consecutive candidates) with the rn rule and no first level, V^T quantized per (dimension, 16 consecutive tokens) with the rn rule and no first level, fp32 accumulation per
    #      64 candidates, row sum from the unquantized P. (Differences to FlashInfer: all 512 dims are NVFP4 here, FlashInfer keeps 64 rope dims in bf16.)
    def nv_modes(qn, prule, vrule, p_l1="sa3", v_l1="grp", p46=False, v46=False, m="eager", qk_rule=None, p_group="kperm", p_pre="f32", **extra):
        qk_rule = qk_rule or prule
        v = dict(v_l1=v_l1, v_rule=vrule, v46=v46, v_smooth=extra.pop("v_smooth", 0))
        ks = extra.pop("k_smooth", 0)
        other = OM(f"nvfp4x{qn}", "nvf4r", "nvf4t", k_rule=qk_rule, k_smooth=ks, **v)
        native = OM(f"nvfp4x{qn}", "nvf4r" if ks else "native", "nvf4t", k_rule=qk_rule, k_smooth=ks, **v)
        return dict(fp8=other, mxfp4=other, fp4=native, p="nvfp4", p_l1=p_l1, p_rule=prule, p46=p46, p_group=p_group, p_pre=p_pre, m=m, B=64, accum="f32seg", q_rule=qk_rule, **extra)

    add("fi-dsv4-nvfp4", "FlashInfer DSv4 NVFP4 algorithm on the V4.1 caches: Q nvfp4x1 (rn), K native / nvf4r (rn), eager max, P bf16 -> NVFP4 per 16 contiguous candidates (rn, no first level), "
        "V^T NVFP4 per 16 tokens (rn, no first level), fp32 per 64 candidates", ALL,
        **nv_modes(1, "rn", "rn", p_l1="none", v_l1="none", p_group="contig", p_pre="bf16"))
    cname = lambda r: {"ceil": "c", "rn": "r"}[r]
    for qn in (1, 2):
        for pl1 in ("sa3", "none"):
            for pr in ("ceil", "rn"):
                for vl1 in ("grp", "none"):
                    for vr in ("ceil", "rn"):
                        add(f"nvfp4-q{qn}-p{'s' if pl1 == 'sa3' else 'n'}{cname(pr)}-v{'g' if vl1 == 'grp' else 'n'}{cname(vr)}",
                            f"Q nvfp4x{qn}; P two-level NVFP4 (first level {pl1}, {pr} scale rule); V^T NVFP4 (first level {vl1}, {vr}); Q / K scale rule = the P rule; eager max, kperm groups",
                            ALL, **nv_modes(qn, pr, vr, p_l1=pl1, v_l1=vl1))
    for qn in (1, 2):
        for r in ("ceil", "rn"):
            for tag, kw in (("p46", dict(p46=True)), ("v46", dict(v46=True)), ("pv46", dict(p46=True, v46=True))):
                add(f"nvfp4-q{qn}-ps{cname(r)}-vg{cname(r)}-{tag}", f"as nvfp4-q{qn}-ps{cname(r)}-vg{cname(r)} with Four Over Six on {'P' if tag == 'p46' else 'V^T' if tag == 'v46' else 'P and V^T'} (the first level targets 1536 instead of 2688)",
                    ALL, **nv_modes(qn, r, r, **kw))
            add(f"nvfp4-q{qn}-ps{cname(r)}-vg{cname(r)}-lazy", f"as nvfp4-q{qn}-ps{cname(r)}-vg{cname(r)} with the lazy max (threshold 6; Pmax = 64, G = 42)", ALL, **nv_modes(qn, r, r, m="lazy"))
    # attribution: one operand in FP4, the rest exact
    for r in ("ceil", "rn"):
        for l1 in ("sa3", "none"):
            add(f"nvfp4-attr-P{'' if l1 == 'sa3' else '0'}-{cname(r)}", f"only P -> two-level NVFP4 (first level {l1}, {r}), eager max; QK and V exact", ALL, p="nvfp4", p_l1=l1, p_rule=r, m="eager", B=64)
        for l1 in ("grp", "none"):
            vo = OM("bf16", "exact", "nvf4t", v_l1=l1, v_rule=r)
            add(f"nvfp4-attr-V{'' if l1 == 'grp' else '0'}-{cname(r)}", f"only V^T -> NVFP4 (first level {l1}, {r}); P in float32 (no quantization), eager max, QK exact", ALL, fp8=vo, fp4=vo, mxfp4=vo, p="nvfp4", p_rule="exact", m="eager", B=64)
        for qn in (1, 2):
            qo = dict(fp8=OM(f"nvfp4x{qn}", "nvf4r", "exact", k_rule=r), mxfp4=OM(f"nvfp4x{qn}", "nvf4r", "exact", k_rule=r), fp4=OM(f"nvfp4x{qn}", "native", "exact"))
            add(f"nvfp4-attr-QK{qn}-{cname(r)}", f"only QK -> Q nvfp4x{qn} ({r}) x K native NVFP4 (FP8 / MXFP4 caches: nvf4r, {r}); P bf16, V exact", ALL, accum="f32seg", q_rule=r, **qo)
    # FP8 tile (and MXFP4 cache) QK left on its native block-scaled kind while PV is FP4: does the K of the FP8 tile need to go to FP4?
    for qn in (1, 2):
        for r in ("ceil", "rn"):
            kw = nv_modes(2, r, r)
            kw["fp8"] = OM(f"mxfp8x{qn}", "native", "nvf4t", v_l1="grp", v_rule=r)
            add(f"nvfp4-f8m{qn}-{cname(r)}", f"as nvfp4-q2-ps{cname(r)}-vg{cname(r)} but the FP8 tiles keep their native QK: Q mxfp8x{qn} x K as stored (mxf8f6f4); PV FP4", ("fp8+fp4", "fp8+fp8"), **kw)
    for qn in (2, 4):
        for r in ("ceil", "rn"):
            kw = nv_modes(2, r, r)
            kw["mxfp4"] = OM(f"mxfp4x{qn}", "native", "nvf4t", v_l1="grp", v_rule=r)
            add(f"nvfp4-m4q{qn}-{cname(r)}", f"as nvfp4-q2-ps{cname(r)}-vg{cname(r)} but the MXFP4 cache keeps its native QK: Q mxfp4x{qn} (kernel rule) x K as stored (mxf4); PV NVFP4", HAS_MXFP4, q_kernel_rule=True, **kw)
    for qn in (2, 4):
        add(f"nvfp4-mx4-q{qn}", f"MXFP4 cache all on kind::mxf4: Q mxfp4x{qn} native QK, P e2m1 + ue8m0 per 32 tokens, V^T mxf4r (ue8m0 per 32 tokens, e2m1 re-rounded); eager max",
            MXFP4_KERNEL_FAMILIES, mxfp4=OM(f"mxfp4x{qn}", "native", "mxf4r"), p="mxfp4", m="eager", B=64, accum="f32seg", q_kernel_rule=True)
    # smoothing ablations on top of nvfp4-q2-ps?-vg? (tile 32 tokens for K and V, 64 heads for Q); k0 = K re-quantized from the FP4 cache too, without smoothing
    for r in ("ceil", "rn"):
        for tag, kw in (("q", dict(q_smooth=True)), ("k", dict(k_smooth=32)), ("k0", dict(k_smooth=0)), ("v", dict(v_smooth=32)), ("qkv", dict(q_smooth=True, k_smooth=32, v_smooth=32))):
            if tag == "k0":
                mm = nv_modes(2, r, r)
                mm["fp4"] = dataclasses.replace(mm["fp4"], k="nvf4r")
                add(f"nvfp4-sm-k0-{cname(r)}", f"nvfp4-q2-ps{cname(r)}-vg{cname(r)} with the FP4 cache K also re-quantized (nvf4r), no smoothing: the baseline of the K smoothing row", ALL, **mm)
                continue
            add(f"nvfp4-sm-{tag}-{cname(r)}", f"nvfp4-q2-ps{cname(r)}-vg{cname(r)} with smoothing: {tag} (Q minus the mean over 64 heads / K and V minus the mean of the 32-token tile; the means go back exactly)", ALL, **nv_modes(2, r, r, **kw))
    return C


# ----------------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------------
def fmt(x: float) -> str:
    return f"{x:.1e}"


def summarize(rows: List[dict], combos, classes: List[str]) -> str:
    """Markdown: per cache family a table combination x data class (worst case over heads / seeds), then the detail table, then a pass summary"""
    fams = [f for f in FAMILIES if any(r["family"] == f for r in rows)]
    order = [c[0] for c in combos]
    desc = {c[0]: c[1] for c in combos}
    by = {}
    for r in rows:
        by.setdefault((r["family"], r["combo"], r["cls"]), []).append(r)
    out = []
    out.append("# sm120 precision modes: error against the upstream reference\n")
    out.append("Cell = worst case over the cases of that data class (heads 64 / 128, repeated seeds). `OK` = every case passes the upstream tolerances "
               "(out: cos_diff <= 5e-6 and abs 1e-3 / rel 2.01/128 elementwise; lse: abs 1e-6 / rel 8.01/65536). `X` = at least one case fails; the letters say which rule fails: "
               "C = out cos_diff, E = out elementwise, L = lse. The number is the worst out cos_diff.\n")
    for f in fams:
        main, extra = FAMILIES[f]
        out.append(f"\n## cache family `{f}` (main {main.name}, extra {extra.name if extra else 'none'})\n")
        cls_present = [c for c in classes if any(k[0] == f and k[2] == c for k in by)]
        out.append("| combination | " + " | ".join(cls_present) + " |")
        out.append("|---|" + "---|" * len(cls_present))
        names = [n for n in order if any(k[0] == f and k[1] == n for k in by)]
        for name in names:
            cells = []
            for c in cls_present:
                rs = by.get((f, name, c))
                if not rs:
                    cells.append("")
                    continue
                flags = ("C" if any(not r["out_cos_ok"] for r in rs) else "") + ("E" if any(not r["out_elem_ok"] for r in rs) else "") + ("L" if any(not r["lse_ok"] for r in rs) else "")
                cells.append(("X " if flags else "OK ") + fmt(max(r["cos_diff"] for r in rs)) + (f" {flags}" if flags else ""))
            out.append(f"| {name} | " + " | ".join(cells) + " |")
        out.append(f"\nDetail (`{f}`), per cell: worst out relative RMS error / worst 99th percentile over (query token, head) rows of the row-wise relative L2 error / "
                   "worst lse absolute error / worst lse relative error.\n")
        out.append("| combination | " + " | ".join(cls_present) + " |")
        out.append("|---|" + "---|" * len(cls_present))
        for name in names:
            cells = []
            for c in cls_present:
                rs = by.get((f, name, c))
                if not rs:
                    cells.append("")
                    continue
                cells.append(" / ".join([fmt(max(r["rel_rms"] for r in rs)), fmt(max(r["rowall_rel_p99"] for r in rs)), fmt(max(r["lse_max_abs"] for r in rs)), fmt(max(r["lse_max_rel"] for r in rs))]))
            out.append(f"| {name} | " + " | ".join(cells) + " |")
    # pass summary
    out.append("\n## Pass summary\n")
    out.append("`U` = passes the upstream tolerances on every `upstream` and `corner` case, `H` = passes on every high-variance case (`hv*`, `kamp*`), `-` = fails somewhere.\n")
    out.append("| combination | " + " | ".join(fams) + " |")
    out.append("|---|" + "---|" * len(fams))
    up_cls = [c for c in classes if c in ("upstream", "corner")]
    hv_cls = [c for c in classes if c not in ("upstream", "corner")]
    for name in order:
        cells = []
        any_row = False
        for f in fams:
            def ok(cl):
                rs = [r for c in cl for r in by.get((f, name, c), [])]
                return None if not rs else all(r["out_ok"] and r["lse_ok"] for r in rs)
            u, hh = ok(up_cls), ok(hv_cls)
            if u is None and hh is None:
                cells.append("")
                continue
            any_row = True
            cells.append(("U" if u else "-") + " " + ("H" if hh else "-"))
        if any_row:
            out.append(f"| {name} | " + " | ".join(cells) + " |")
    out.append("\n## Combination definitions\n")
    for name in order:
        out.append(f"- `{name}`: {desc[name]}")
    return "\n".join(out) + "\n"


NUM_CORNER_CASES = 4        # the fixed corner cases; the families of CORNER_EXTRA that are asked for add one case each (spec[2])


def case_specs(classes: List[str], families: List[str], heads: List[int], repeat: int = 1) -> List[tuple]:
    """The data to generate, in order, with their seeds: ('corner', seed) stands for NUM_CORNER_CASES cases, (cls, family, h, seed) for one"""
    specs, seed = [], 100
    for cls in classes:
        for rep in range(repeat):
            if cls == "corner":
                specs.append(("corner", seed) + ((tuple(f for f in families if f in CORNER_EXTRA),) if any(f in CORNER_EXTRA for f in families) else ()))
                seed += 10
                continue
            for family in families:
                for h in heads:
                    specs.append((cls, family, h, seed))
                    seed += 1
    return specs


def iter_cases(specs: List[tuple], b: int, s_q: int, quiet: bool = False, fp4_2l: Optional[dict] = None):
    """Generate the cases one at a time (a case holds the caches of its batch: keeping all of them alive costs more than 20 GB of GPU memory)"""
    for spec in specs:
        with upstream_defaults():
            if spec[0] == "corner":
                cs = make_corner_cases(spec[1], b, s_q, spec[2] if len(spec) > 2 else ())
            else:
                cls, family, h, seed = spec
                t0 = time.time()
                c = make_case(cls, family, h, seed, b, s_q, fp4_2l=fp4_2l)
                c.seed = seed
                c.gap_mean, c.gap_max = exp_gap_stats(c.t)
                if not quiet:
                    print(f"  data {cls:8s} {family:10s} h={h:3d} seed={c.seed}: measured logit*sm_scale std {c.sigma_measured:.3f} nat, block exponent gap mean {c.gap_mean:.1f} max {c.gap_max:.0f} ({time.time() - t0:.1f} s)", flush=True)
                cs = [c]
        for c in cs:
            yield c


def build_cases(classes: List[str], families: List[str], heads: List[int], b: int, s_q: int, repeat: int = 1, quiet: bool = False, fp4_2l: Optional[dict] = None) -> List[Case]:
    return list(iter_cases(case_specs(classes, families, heads, repeat), b, s_q, quiet, fp4_2l))


def run(cases, combos, log=print, total: Optional[int] = None) -> List[dict]:
    rows = []
    total = len(cases) if total is None else total
    for ci, c in enumerate(cases):
        with upstream_defaults():
            out_ref, lse_ref = ref.ref_sparse_attn_decode(c.p, c.t)
        todo = [x for x in combos if c.family in x[3]]
        for name, desc, mode, fams in todo:
            t0 = time.time()
            with torch.inference_mode():
                out, lse = emu.simulate_decode(c.p, c.t, mode)
                m = metrics(out, lse, out_ref, lse_ref)
            rows.append(dict(combo=name, cls=c.cls, family=c.family, h=c.h, seed=c.seed, sigma=c.sigma_measured, gap_mean=c.gap_mean, gap_max=c.gap_max, **m))
            log(f"[{ci + 1}/{total}] {c.cls:8s} {c.family:10s} h={c.h:3d} {name:20s} cos={m['cos_diff']:.2e} rel_rms={m['rel_rms']:.2e} out_ok={m['out_ok']} lse_ok={m['lse_ok']} ({time.time() - t0:.1f} s)")
        del out_ref, lse_ref, c
        torch.cuda.empty_cache()
    return rows


# ----------------------------------------------------------------------------------------------
# Self test of the emulator: an independent, slow, per-token / per-head / per-element implementation of the same arithmetic
# ----------------------------------------------------------------------------------------------
_E2M1_TAB = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _ceil_log2_py(x: float) -> int:
    m, e = math.frexp(x)
    return e - 1 if m == 0.5 else e


def _slow_fp4_dequant(row: torch.Tensor) -> torch.Tensor:
    """One V41_FP4Sparse cache row (uint8 [bytes], cpu) -> the 512 values e2m1 * ue4m3 scale per 16 dims (float32, exact)"""
    nib = []
    for i in range(256):
        byte = int(row[i])
        for code in (byte & 0xF, byte >> 4):
            nib.append((-1.0 if code & 8 else 1.0) * _E2M1_TAB[code & 7])
    return torch.tensor(nib, dtype=torch.float32) * row[256:288].view(torch.float8_e4m3fn).float().repeat_interleave(16)


def _slow_token_terms8(layout, row: torch.Tensor, form: str, lo_shift: int):
    """One cache row (uint8 [bytes], cpu) -> [(codes float32 [512], exps list[16])]: the MXFP8 form (data * 2^exp per 32 dims)"""
    import numpy as np
    if layout == L.V41_FP8Sparse:
        return [(row[:512].view(torch.float8_e4m3fn).float(), [int(x) - 127 for x in row[512:528]])]
    nib = []
    for i in range(256):
        byte = int(row[i])
        for code in (byte & 0xF, byte >> 4):
            nib.append((-1.0 if code & 8 else 1.0) * _E2M1_TAB[code & 7])
    vals = torch.tensor(nib, dtype=torch.float32)
    if layout == L.V41_MXFP4Sparse:
        return [(vals, [int(x) - 127 for x in row[256:272]])]
    sf16 = row[256:288].view(torch.float8_e4m3fn).float()
    v = vals * sf16.repeat_interleave(16)
    codes = [torch.zeros(512), torch.zeros(512)]
    exps = [[], []]
    for g in range(16):
        x = v[32 * g:32 * g + 32]
        amax = float(x.abs().max())
        e = _ceil_log2_py(float(np.float32(amax) / np.float32(448.0))) if amax > 0 else -127
        r = x
        for i in range(1 if form == "x1" else 2):
            ee = max(e - i * lo_shift, -127)
            d = emu.e4m3_rn(r / (2.0 ** ee))
            codes[i][32 * g:32 * g + 32] = d
            exps[i].append(ee)
            r = r - d * (2.0 ** ee)
    n = 1 if form == "x1" else 2
    return [(codes[i], exps[i]) for i in range(n)]


def slow_decode_token(t, mode: emu.Mode, bi: int, si: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """The arithmetic of simulate_decode for one query token written with explicit loops over heads, blocks, tokens (cpu, no batching). Returns (out [h, 512] bf16, lse [h])"""
    import numpy as np
    assert mode.accum == "f64"
    q = t.q[bi, si].float().cpu()
    h = q.shape[0]
    scopes = [t.kv_scope] + ([t.extra_kv_scope] if t.extra_kv_scope is not None else [])
    toks = []        # [(layout, valid, row uint8 or None, kdeq float32 or None)]
    for sc in scopes:
        layout = sc.kvcache_layout or L.V41_FP8Sparse
        rows = sc.blocked_k_quantized.view(torch.uint8).reshape(-1, layout.get_bytes_per_token())
        kd = sc.blocked_k.reshape(-1, 512)
        idx = sc.indices_in_kvcache[bi, si].tolist()
        tl = int(sc.topk_length[bi]) if sc.topk_length is not None else len(idx)
        sub = []
        for n, ix in enumerate(idx):
            if ix == -1 or n >= tl:
                sub.append((layout, False, None, None))
            else:
                sub.append((layout, True, rows[ix].cpu(), kd[ix].float().cpu()))
        while len(sub) % 64:
            sub.append((layout, False, None, None))
        toks += sub
    N = len(toks)
    B = mode.B

    # per token: dense K terms for QK, V terms for PV
    kterm = []       # per token list of float64 dense vectors (+ the Q mode they pair with)
    vterm = []       # per token list of (data float32 [512] or None, exps list or None) / exact vector
    for layout, valid, row, kdeq in toks:
        om = mode.ops(layout)
        if not valid:
            kterm.append(None), vterm.append(None)
            continue
        kind, _ = emu.parse_q_mode(om.q)
        if kind == "bf16":
            kt = [kdeq.double()]
        elif kind == "mxfp8":
            kt = [c.double() * torch.tensor([2.0 ** e for e in ex], dtype=torch.float64).repeat_interleave(32)
                  for c, ex in _slow_token_terms8(layout, row, "x1" if om.k == "native" else om.k[2:], mode.k_lo_shift)]
        else:
            vals = []
            for i in range(256):
                byte = int(row[i])
                for code in (byte & 0xF, byte >> 4):
                    vals.append((-1.0 if code & 8 else 1.0) * _E2M1_TAB[code & 7])
            vals = torch.tensor(vals, dtype=torch.float64)
            if layout == L.V41_FP4Sparse:
                kt = [vals * row[256:288].view(torch.float8_e4m3fn).double().repeat_interleave(16)]
            else:
                kt = [vals * torch.tensor([2.0 ** (int(x) - 127) for x in row[256:272]], dtype=torch.float64).repeat_interleave(32)]
        kterm.append(kt)
        if om.v == "exact":
            vterm.append([kdeq.float()])
        elif om.v8 in ("d1", "d2", "d1s", "d2s"):
            vterm.append([_slow_fp4_dequant(row)])           # the direct forms work on the block, see the PV loop
        else:
            vterm.append(_slow_token_terms8(layout, row, om.v8, mode.k_lo_shift))

    scale = np.float32(float(torch.tensor(t.sm_scale * emu.LOG2E, dtype=torch.float32)))
    outs, lses = [], []
    for hd in range(h):
        # ---- logits of the head: sum of the MMA term pairs (lo x lo pairs are not issued), exact in float64, rounded to float32
        qcache = {}
        logit = np.zeros(N, dtype=np.float32)
        for n, (layout, valid, row, kdeq) in enumerate(toks):
            if not valid:
                continue
            om = mode.ops(layout)
            if (om.q, om.q_src) not in qcache:
                qt = emu.quantize_q(emu.source_q(q[hd:hd + 1], om, mode), om.q, mode.q_lo_shift, mode.q_kernel_rule)
                qcache[(om.q, om.q_src)] = [(v[0].double() * s[0].double().repeat_interleave(qt.g)) for v, s in zip(qt.vals, qt.sf)]
            acc = 0.0
            for i, qd in enumerate(qcache[(om.q, om.q_src)]):
                for j, kd_ in enumerate(kterm[n]):
                    if i > 0 and j > 0 and not mode.lolo:
                        continue
                    acc += float((qd * kd_).sum())
            logit[n] = np.float32(acc)
        valid_arr = np.array([tk[1] for tk in toks])
        l2 = np.where(valid_arr, logit * scale, np.float32(-np.inf)).astype(np.float32)
        # ---- block maxima and the max in effect
        nblk = N // B
        bmax = l2.reshape(nblk, B).max(1)
        if mode.m == "final":
            rm = bmax.max()
            ms = [np.float32(0.0 if np.isinf(rm) else rm)] * nblk
            ss = [np.float32(1.0)] * nblk
        else:
            m = np.float32(emu.MAX_INIT_VAL)
            ms, ss = [], []
            for j in range(nblk):
                if bmax[j] - m > emu.LAZY_THRESHOLD:
                    s = np.float32(np.exp2(np.float32(m) - np.float32(bmax[j])))
                    m = np.float32(bmax[j])
                else:
                    s = np.float32(1.0)
                ms.append(m), ss.append(s)
        O = np.zeros(512, dtype=np.float32)
        l = np.float32(0.0)
        for j in range(nblk):
            n0 = j * B
            m = ms[j]
            P = np.zeros(B, dtype=np.float32)
            for k in range(B):
                if valid_arr[n0 + k]:
                    P[k] = np.exp2(np.float32(np.float64(logit[n0 + k]) * np.float64(scale) - np.float64(m)))
            # --- P terms per 32-token k block
            pterm = [[], []]        # [term][kb] = (values float64 [32], sf float)
            for kb in range(B // 32):
                x = P[kb * 32:(kb + 1) * 32]
                if mode.p == "bf16":
                    pterm[0].append((torch.from_numpy(x).to(torch.bfloat16).float().double().numpy(), 1.0))
                    continue
                if mode.p_sf == "const":
                    e = math.ceil(math.log2((1.0 if mode.m == "final" else 2.0 ** emu.LAZY_THRESHOLD) / 448.0))
                else:
                    amax = float(x.max())
                    e = _ceil_log2_py(float(np.float32(amax) / np.float32(448.0))) if amax > 0 else 0
                r = torch.from_numpy(x.copy())
                for i in range(1 if mode.p == "e4m3x1" else 2):
                    ee = e - i * mode.p_lo_shift
                    d = emu.e4m3_rn(r / (2.0 ** ee))
                    pterm[i].append((d.double().numpy(), 2.0 ** ee))
                    r = r - d * (2.0 ** ee)
            # --- denominator
            if mode.sum == "exact":
                ls = np.float32(P.astype(np.float64).sum())
            else:
                tot = 0.0
                for kb in range(B // 32):
                    for i in range(2):
                        if pterm[i] and len(pterm[i]) > kb:
                            tot += float((pterm[i][kb][0] * pterm[i][kb][1]).sum())
                ls = np.float32(tot)
            # --- V terms of the block (rebase per 32 dims over the valid tokens of the block)
            om_blk = mode.ops(toks[n0][0])
            direct = om_blk.v == "rebase" and om_blk.v8 in ("d1", "d2", "d1s", "d2s")
            nv = max(len(vterm[n0 + k]) for k in range(B) if valid_arr[n0 + k]) if valid_arr[n0:n0 + B].any() else 1
            nv = (2 if om_blk.v8 in ("d2", "d2s") else 1) if direct else nv
            vdat = [np.zeros((B, 512)) for _ in range(nv)]       # float64 data
            vsf = [np.ones(512) for _ in range(nv)]              # SF_B per column
            if om_blk.v == "exact":
                for k in range(B):
                    if valid_arr[n0 + k]:
                        vdat[0][k] = vterm[n0 + k][0].double().numpy()
            elif direct:
                # e2m1 * s16 straight to e4m3 on the scale 2^E of (this block, 32 dims), E = ceil(log2(amax / 448)) over the valid tokens; d2: the lo term on 2^(E - 4)
                vk = [k for k in range(B) if valid_arr[n0 + k]]
                for g in range(16):
                    if om_blk.v8.endswith("s"):
                        # 6 x the largest 16-dim scale of the valid tokens
                        amax = max((6.0 * float(toks[n0 + k][2][256 + 2 * g:256 + 2 * g + 2].view(torch.float8_e4m3fn).float().max()) for k in vk), default=0.0)
                    else:
                        amax = max((float(vterm[n0 + k][0][32 * g:32 * g + 32].abs().max()) for k in vk), default=0.0)
                    E = _ceil_log2_py(float(np.float32(amax) / np.float32(448.0))) if amax > 0 else 0
                    for jt in range(nv):
                        vsf[jt][32 * g:32 * g + 32] = 2.0 ** (E - 4 * jt)
                    for k in vk:
                        x = vterm[n0 + k][0][32 * g:32 * g + 32]
                        hi = emu.e4m3_rn(x / (2.0 ** E))
                        vdat[0][k, 32 * g:32 * g + 32] = hi.double().numpy()
                        if nv == 2:
                            lo = emu.e4m3_rn((x - hi * (2.0 ** E)) / (2.0 ** (E - 4)))
                            vdat[1][k, 32 * g:32 * g + 32] = lo.double().numpy()
            else:
                for jt in range(nv):
                    for g in range(16):
                        es = [vterm[n0 + k][jt][1][g] for k in range(B) if valid_arr[n0 + k]]
                        emax = max(max(es) + om_blk.v_off, emu.UE8M0_MIN_EXP) if es else 0
                        vsf[jt][32 * g:32 * g + 32] = 2.0 ** emax
                        for k in range(B):
                            if valid_arr[n0 + k]:
                                c = vterm[n0 + k][jt][0][32 * g:32 * g + 32].double() * (2.0 ** (vterm[n0 + k][jt][1][g] - emax))
                                c = emu.e4m3_rz(c.float()) if mode.rebase_round == "rz" else emu.e4m3_rn(c.float())
                                vdat[jt][k, 32 * g:32 * g + 32] = c.double().numpy()
            # --- PV
            contrib = np.zeros(512)
            for kb in range(B // 32):
                for i in range(len(pterm)):
                    if not pterm[i]:
                        continue
                    pv_, psf_ = pterm[i][kb]
                    for jt in range(nv):
                        if i > 0 and jt > 0 and not mode.lolo:
                            continue
                        contrib += (pv_ @ vdat[jt][kb * 32:(kb + 1) * 32]) * psf_ * vsf[jt]
            O = (O * ss[j]).astype(np.float32)
            l = np.float32(l * ss[j] + ls)
            O = (O + contrib.astype(np.float32)).astype(np.float32)
        sink = float(t.attn_sink[hd]) * emu.LOG2E if t.attn_sink is not None else -math.inf
        mlast = ms[-1]
        if l == 0:
            outs.append(torch.zeros(512, dtype=torch.bfloat16)), lses.append(torch.tensor(math.inf))
            continue
        den = np.float32(l + np.float32(np.exp2(np.float32(sink - float(mlast))) if np.isfinite(sink) else (0.0 if sink < 0 else np.inf)))
        outs.append(torch.from_numpy((O * np.float32(1.0 / den)).astype(np.float32)).to(torch.bfloat16))
        lses.append(torch.tensor(float(np.float32(mlast) * np.float32(emu.LN2) + np.log(l)), dtype=torch.float32))
    return torch.stack(outs), torch.stack(lses)


# ----------------------------------------------------------------------------------------------
# FP4 PV (T-08): independent scalar implementations (python loops over tokens / dimensions / groups, numpy float32, the small float tables) and their comparison with the vectorized code
# ----------------------------------------------------------------------------------------------
_T_E4M3 = emu._small_float_table(3, 7, 0x7E)
_T_E2M1 = emu._small_float_table(1, 1, 7)


def _exp2_f32(x):
    """exp2 of a float32 numpy array, evaluated by the device (the same implementation as the vectorized code; exp2 itself is not under test)"""
    return torch.exp2(torch.tensor(np.asarray(x, dtype=np.float32), device="cuda")).cpu().numpy()


def slow_nv_group(x, rule: str, four_six: bool):
    """One NVFP4 group (numpy float32 [16]) quantized with python scalars: (codes [16 python floats], scale python float). See emu.nv_quant for the rules"""
    def one(qmax: float):
        amax = float(np.abs(x).max())
        if rule == "ceil":
            s = next((v for v in _T_E4M3[1:] if qmax * v >= amax), _T_E4M3[-1])          # the smallest e4m3 value with qmax * s >= amax, at least 2^-9
            inv = np.float32(1.0) / np.float32(s)
        else:
            s = emu._table_round(float(np.float32(amax) / np.float32(qmax)), _T_E4M3)    # e4m3_rn(amax / qmax), IEEE float32 division, saturating, subnormals kept
            inv = np.float32(0.0) if s == 0.0 else np.float32(1.0) / np.float32(s)
        return [emu._table_round(float(np.float32(xi) * inv), _T_E2M1) for xi in x], s
    c6, s6 = one(6.0)
    if not four_six:
        return c6, s6
    c4, s4 = one(4.0)
    e6 = sum((float(xi) - ci * s6) ** 2 for xi, ci in zip(x, c6))
    e4 = sum((float(xi) - ci * s4) ** 2 for xi, ci in zip(x, c4))
    return (c4, s4) if e4 < e6 else (c6, s6)


def slow_kperm_blocks():
    """The four 16-token scale blocks of a pair of tiles, derived from the fragment layouts instead of the closed form: S accumulator (n, t, j) -> token 8 n + 2 t + j; A fragment: k = 8 t + 2 (n % 4) + j (+ 32 for n >= 4)"""
    blocks = {}
    for n in range(8):
        for t in range(4):
            for j in range(2):
                token = 8 * n + 2 * t + j
                k = 8 * t + 2 * (n % 4) + j + (32 if n >= 4 else 0)
                blocks.setdefault(k // 16, []).append(token)
    return [sorted(blocks[b]) for b in range(4)]


def slow_p_block(P64, mode: emu.Mode):
    """
    One block of 64 tokens of P (numpy float32 [64], the unquantized numerators of one row) -> (dequantized values in the G-scaled domain, numpy float64 [64], G). nvfp4: bf16 rounding (p_pre),
    G = p_gain, scale groups of 16 tokens (kperm: slow_kperm_blocks(), contig: consecutive), ue4m3 scale rule; mxfp4: the 32-token tiles, ue8m0 scale 2^ceil(log2(fp32(amax) * fp32(1 / 6)))
    """
    if mode.p == "nvfp4" and mode.p_rule == "exact":
        return P64.astype(np.float64), 1.0
    x = P64.astype(np.float32)
    if mode.p_pre == "bf16":
        x = torch.from_numpy(x).to(torch.bfloat16).float().numpy()
    deq = np.zeros(64)
    if mode.p == "mxfp4":
        for tile in range(2):
            xt = x[32 * tile:32 * tile + 32]
            y = np.float32(float(np.abs(xt).max())) * np.float32(emu.Q_RECIP_6)
            e = 0 if y == 0 else _ceil_log2_py(float(y))
            for k in range(32):
                deq[32 * tile + k] = emu._table_round(float(xt[k] / np.float32(2.0 ** e)), _T_E2M1) * 2.0 ** e
        return deq, 1.0
    G = emu.p_gain(mode)
    if G != 1.0:
        x = (x * np.float32(G)).astype(np.float32)
    blocks = slow_kperm_blocks() if mode.p_group == "kperm" else [list(range(16 * i, 16 * i + 16)) for i in range(4)]
    for tokens in blocks:
        codes, s = slow_nv_group(x[tokens], mode.p_rule, mode.p46)
        for tk, c in zip(tokens, codes):
            deq[tk] = c * s
    return deq, G


def slow_decode_token_fp4(t, mode: emu.Mode, bi: int, si: int, parts: Optional[dict] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Scalar implementation of simulate_decode for the FP4 PV modes (mode.p = nvfp4 / mxfp4) for one query token: explicit loops over tokens, heads, dimensions and scale groups; the cache bytes are
    decoded here (not by sm120_emu). Returns (out [h, 512] bf16, lse [h]); if `parts` is a dict, the dequantized V' (before the output column multiplier, natural token order) and the column multiplier are stored in it
    """
    assert mode.p in emu.FP4_P and mode.B == 64 and mode.sum == "exact"
    q = t.q[bi, si].float().cpu()
    h = q.shape[0]
    scopes = [t.kv_scope] + ([t.extra_kv_scope] if t.extra_kv_scope is not None else [])
    toks = []
    for sc in scopes:
        layout = sc.kvcache_layout or L.V41_FP8Sparse
        rows = sc.blocked_k_quantized.view(torch.uint8).reshape(-1, layout.get_bytes_per_token())
        kd = sc.blocked_k.reshape(-1, 512)
        idx = sc.indices_in_kvcache[bi, si].tolist()
        tl = int(sc.topk_length[bi]) if sc.topk_length is not None else len(idx)
        sub = []
        for n, ix in enumerate(idx):
            if ix == -1 or n >= tl:
                sub.append((layout, False, None, None))
            else:
                sub.append((layout, True, rows[ix].cpu(), kd[ix].float().cpu().numpy()))
        while len(sub) % 64:
            sub.append((layout, False, None, None))
        toks += sub
    N = len(toks)
    nblk = N // 64
    valid_arr = np.array([tk[1] for tk in toks])
    qrule = None if mode.q_rule == "legacy" else mode.q_rule

    # ---- tile means (smoothing)
    def tile_means(lo: int, hi: int, tile: int):
        """means of the exact K over the valid tokens of the tiles of positions lo .. hi: {tile index (relative to lo): float32 [512]}"""
        out = {}
        for k0 in range(lo, hi, tile):
            vs = [toks[n][3].astype(np.float64) for n in range(k0, k0 + tile) if toks[n][1]]
            out[(k0 - lo) // tile] = (np.sum(vs, axis=0) / len(vs)).astype(np.float32) if vs else np.zeros(512, np.float32)
        return out

    scope_ranges, pos = [], 0
    for sc in scopes:
        topk = sc.indices_in_kvcache.shape[-1]
        n_sc = topk + (-topk) % 64
        scope_ranges.append((pos, pos + n_sc))
        pos += n_sc

    # ---- Q (the NVFP4 / MXFP4 / MXFP8 Q quantizers are checked apart against scalar implementations)
    qbar_h = None
    qmain = q
    if mode.q_smooth:
        hg = 64 if h % 64 == 0 else h
        qbar_g = q.view(h // hg, hg, 512).double().mean(1).float()
        qbar_h = qbar_g.repeat_interleave(hg, dim=0)                                             # [h, 512]
        qmain = (q.view(h // hg, hg, 512) - qbar_g.unsqueeze(1)).view(h, 512)

    # ---- K operands: dense float64 vectors of the terms; per token compensation of the K smoothing
    kterm, kcomp_mean = [None] * N, [None] * N
    qk_kinds = {}
    for si_, (lo, hi) in enumerate(scope_ranges):
        layout = toks[lo][0]
        om = mode.ops(layout)
        qk_kinds[si_] = om
        kbar = tile_means(lo, hi, om.k_smooth) if om.k_smooth else None
        for n in range(lo, hi):
            layout, valid, row, kdeq = toks[n]
            if not valid:
                continue
            kind, _ = emu.parse_q_mode(om.q)
            if kind == "bf16":
                kterm[n] = [torch.from_numpy(kdeq.astype(np.float64))]
                continue
            if kind == "mxfp8":
                kterm[n] = [c.double() * torch.tensor([2.0 ** e for e in ex], dtype=torch.float64).repeat_interleave(32)
                            for c, ex in _slow_token_terms8(layout, row, "x1" if om.k == "native" else om.k[2:], mode.k_lo_shift)]
                continue
            if om.k == "nvf4r":
                xk = kdeq
                if om.k_smooth:
                    kbar_t = kbar[(n - lo) // om.k_smooth]
                    xk = (kdeq - kbar_t).astype(np.float32)
                    kcomp_mean[n] = kbar_t
                dense = np.zeros(512)
                for g in range(32):
                    codes, s = slow_nv_group(xk[16 * g:16 * g + 16], om.k_rule, False)
                    dense[16 * g:16 * g + 16] = np.array(codes) * s
                kterm[n] = [torch.from_numpy(dense)]
                continue
            vals = []
            for i in range(256):
                byte = int(row[i])
                for code in (byte & 0xF, byte >> 4):
                    vals.append((-1.0 if code & 8 else 1.0) * _E2M1_TAB[code & 7])
            vals = torch.tensor(vals, dtype=torch.float64)
            if layout == L.V41_FP4Sparse:
                kterm[n] = [vals * row[256:288].view(torch.float8_e4m3fn).double().repeat_interleave(16)]
            else:
                kterm[n] = [vals * torch.tensor([2.0 ** (int(x) - 127) for x in row[256:272]], dtype=torch.float64).repeat_interleave(32)]

    # ---- V operand: dequantized V' per token (natural order, float64) and the column multiplier
    vdeq = np.zeros((N, 512))
    col = np.ones(512, dtype=np.float32)
    vbar_tiles = None
    oms = [mode.ops(toks[lo][0]) for lo, hi in scope_ranges]
    blocks16 = slow_kperm_blocks() if (mode.p == "nvfp4" and mode.p_group == "kperm") else [list(range(16 * i, 16 * i + 16)) for i in range(4)]
    if mode.p == "mxfp4":
        for si_, (lo, hi) in enumerate(scope_ranges):
            assert oms[si_].v == "mxf4r"
            tok_vals, tok_exps = {}, {}
            for n in range(lo, hi):
                if toks[n][1]:
                    row = toks[n][2]
                    vals = []
                    for i in range(256):
                        byte = int(row[i])
                        for code in (byte & 0xF, byte >> 4):
                            vals.append((-1.0 if code & 8 else 1.0) * _E2M1_TAB[code & 7])
                    tok_vals[n], tok_exps[n] = np.array(vals), [int(x) - 127 for x in row[256:272]]
            for k0 in range(lo, hi, 32):
                vs = [n for n in range(k0, k0 + 32) if toks[n][1]]
                for g in range(16):
                    if not vs:
                        continue
                    emax = max(tok_exps[n][g] for n in vs)
                    for n in vs:
                        for d in range(32 * g, 32 * g + 32):
                            vdeq[n, d] = emu._table_round(tok_vals[n][d] * 2.0 ** (tok_exps[n][g] - emax), _T_E2M1) * 2.0 ** emax
    else:
        om0 = oms[0]
        assert all(o.v == oms[0].v for o in oms) and om0.v in ("nvf4t", "exact")
        top = emu.V1_TOP_46 if om0.v46 else emu.V1_TOP
        xv = np.zeros((N, 512), dtype=np.float32)
        if om0.v == "exact":
            for n in range(N):
                if toks[n][1]:
                    vdeq[n] = toks[n][3]
        for si_, (lo, hi) in enumerate(scope_ranges if om0.v == "nvf4t" else []):
            kbar = tile_means(lo, hi, om0.v_smooth) if om0.v_smooth else None
            for n in range(lo, hi):
                if toks[n][1]:
                    xv[n] = toks[n][3] - kbar[(n - lo) // om0.v_smooth] if om0.v_smooth else toks[n][3]
            if kbar is not None:
                vbar_tiles = (vbar_tiles or []) + [kbar[i] for i in range(len(kbar))]
        E1 = [0] * 16
        if om0.v == "nvf4t" and om0.v_l1 == "grp":
            for g in range(16):
                amax = float(np.abs(xv[:, 32 * g:32 * g + 32]).max())
                if amax > 0:
                    e = math.ceil(math.log2(amax / top))
                    while amax > top * 2.0 ** e:
                        e += 1
                    while amax <= top * 2.0 ** (e - 1):
                        e -= 1
                    E1[g] = e
                col[32 * g:32 * g + 32] = np.float32(2.0 ** E1[g])
        xs = np.zeros_like(xv)
        for g in range(16):
            xs[:, 32 * g:32 * g + 32] = xv[:, 32 * g:32 * g + 32] * np.float32(2.0 ** -E1[g])
        for j in range(nblk if om0.v == "nvf4t" else 0):
            for tokens in blocks16:
                idx = [64 * j + tk for tk in tokens]
                for d in range(512):
                    codes, s = slow_nv_group(xs[idx, d].astype(np.float32), om0.v_rule, om0.v46)
                    for n, c in zip(idx, codes):
                        vdeq[n, d] = c * s
    if parts is not None:
        parts["vdeq"], parts["col"] = vdeq, col

    # ---- per head
    scale = np.float32(float(torch.tensor(t.sm_scale * emu.LOG2E, dtype=torch.float32)))
    G = emu.p_gain(mode)
    outs, lses = [], []
    for hd in range(h):
        qcache = {}
        logit = np.zeros(N, dtype=np.float32)
        for n, (layout, valid, row, kdeq) in enumerate(toks):
            if not valid:
                continue
            si_ = next(i for i, (lo, hi) in enumerate(scope_ranges) if lo <= n < hi)
            om = oms[si_]
            if (om.q, om.q_src) not in qcache:
                qt = emu.quantize_q(emu.source_q(qmain[hd:hd + 1], om, mode), om.q, mode.q_lo_shift, mode.q_kernel_rule, qrule)
                qcache[(om.q, om.q_src)] = ([(v[0].double() * s_[0].double().repeat_interleave(qt.g)).numpy() for v, s_ in zip(qt.vals, qt.sf)], qt.k_step)
            qd, ks = qcache[(om.q, om.q_src)]
            pairs = [(i, j) for i in range(len(qd)) for j in range(len(kterm[n])) if mode.lolo or not (i > 0 and j > 0)]
            kd_ = [k_.numpy() for k_ in kterm[n]]
            if mode.accum == "f64":
                logit[n] = np.float32(sum(float((qd[i] * kd_[j]).sum()) for i, j in pairs))
            else:
                acc = np.float32(0.0)
                for st in range(512 // ks):
                    for i, j in pairs:
                        acc = np.float32(acc + np.float32(float((qd[i][ks * st:ks * st + ks] * kd_[j][ks * st:ks * st + ks]).sum())))
                logit[n] = acc
            if qbar_h is not None:
                logit[n] = np.float32(logit[n] + np.float32(float((qbar_h[hd].double().numpy() * kdeq.astype(np.float64)).sum())))
            if kcomp_mean[n] is not None:
                logit[n] = np.float32(logit[n] + np.float32(float((q[hd].double().numpy() * kcomp_mean[n].astype(np.float64)).sum())))
        l2 = np.where(valid_arr, logit * scale, np.float32(-np.inf)).astype(np.float32)
        bmax = l2.reshape(nblk, 64).max(1)
        if mode.m == "final":
            rm = bmax.max()
            ms, ss = [np.float32(0.0 if np.isinf(rm) else rm)] * nblk, [np.float32(1.0)] * nblk
        else:
            thr = np.float32(0.0 if mode.m == "eager" else emu.LAZY_THRESHOLD)
            m = np.float32(emu.MAX_INIT_VAL)
            ms, ss = [], []
            for j in range(nblk):
                if np.float32(bmax[j] - m) > thr:
                    s_ = _exp2_f32(np.float32(np.float32(m) - np.float32(bmax[j])))
                    m = np.float32(bmax[j])
                else:
                    s_ = np.float32(1.0)
                ms.append(m), ss.append(np.float32(s_))
        # numerators of every block (device exp2 on the float32 argument)
        arg = np.zeros(N, dtype=np.float32)
        for j in range(nblk):
            for k in range(64):
                n = 64 * j + k
                if valid_arr[n]:
                    arg[n] = np.float32(np.float64(logit[n]) * np.float64(scale) - np.float64(ms[j]))
        Pall = np.where(valid_arr, _exp2_f32(arg), np.float32(0.0)).astype(np.float32)
        O = np.zeros(512, dtype=np.float32)
        l = np.float32(0.0)
        for j in range(nblk):
            P64 = Pall[64 * j:64 * j + 64]
            pdeq, _ = slow_p_block(P64, mode)
            ls = np.float32(P64.astype(np.float64).sum())
            contrib = pdeq @ vdeq[64 * j:64 * j + 64]                                              # exact in float64: the 64 products of one instruction
            if vbar_tiles is not None:
                tile = oms[0].v_smooth
                for tt in range(64 // tile):
                    ti = (64 * j) // tile + tt
                    contrib = contrib + (G * float(P64[tt * tile:(tt + 1) * tile].astype(np.float64).sum())) * (vbar_tiles[ti].astype(np.float64) / col.astype(np.float64))
            O = (O * ss[j]).astype(np.float32)
            l = np.float32(l * ss[j] + ls)
            O = (O + contrib.astype(np.float32)).astype(np.float32)
        if G != 1.0:
            O = (O / np.float32(G)).astype(np.float32)
        O = (O * col).astype(np.float32)
        sink = float(t.attn_sink[hd]) * emu.LOG2E if t.attn_sink is not None else -math.inf
        mlast = ms[-1]
        if l == 0:
            outs.append(torch.zeros(512, dtype=torch.bfloat16)), lses.append(torch.tensor(math.inf))
            continue
        den = np.float32(l + np.float32(_exp2_f32(np.float32(sink - float(mlast))) if np.isfinite(sink) else (0.0 if sink < 0 else np.inf)))
        outs.append(torch.from_numpy((O * np.float32(1.0 / den)).astype(np.float32)).to(torch.bfloat16))
        lses.append(torch.tensor(float(np.float32(mlast) * np.float32(emu.LN2) + np.log(l)), dtype=torch.float32))
    return torch.stack(outs), torch.stack(lses)


def selftest_fp4() -> bool:
    """T-08 primitives against independent scalar implementations"""
    ok = True

    def report(name, passed, detail=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}", flush=True)

    g = torch.Generator(device="cuda").manual_seed(5)
    # (1) nv_quant: rules x Four Over Six on groups of every magnitude, zero groups, scale boundaries, saturation
    ng = 1500
    gain = torch.exp2(torch.rand(ng, 1, generator=g, device="cuda") * 30 - 18)
    xg = torch.randn(ng, 16, generator=g, device="cuda") * gain
    xg[:100] = 0.0                                                                    # zero groups
    xg[100:200] = xg[100:200] * (torch.rand(100, 16, generator=g, device="cuda") < 0.15)         # sparse groups
    tab = torch.tensor(_T_E4M3[1:], dtype=torch.float64, device="cuda")
    bnd = (tab[torch.randint(0, tab.numel(), (150,), generator=g, device="cuda")] * 6.0)         # amax exactly 6 * (an e4m3 value) and its float32 neighbours
    bnd = bnd.float() * torch.tensor([1.0, 1.0 + 2.0 ** -23, 1.0 - 2.0 ** -23], device="cuda")[torch.randint(0, 3, (150,), generator=g, device="cuda")]
    xg[200:350] = xg[200:350] / xg[200:350].abs().amax(-1, keepdim=True) * bnd.unsqueeze(-1)
    xg[350:380] = xg[350:380] * 1e6                                                   # beyond 6 * 448: saturation
    xg[380:410] = xg[380:410] * 1e-9                                                  # below the scale floor
    xg = xg.to(torch.bfloat16).float()
    xg[410:450] = xg[410:450] * (1.0 + 2.0 ** -20)                                    # float32 values that are not bf16
    for rule in ("ceil", "rn"):
        for fs in (False, True):
            c, s = emu.nv_quant(xg, rule, fs)
            xs = xg.cpu().numpy()
            bad = 0
            cc, ss_ = c.cpu().numpy(), s.cpu().numpy()
            for i in range(ng):
                codes, sc = slow_nv_group(xs[i], rule, fs)
                bad += not (np.array_equal(cc[i], np.array(codes, dtype=np.float32)) and float(ss_[i]) == sc)
            report(f"nv_quant [{rule}{' 4/6' if fs else ''}] == scalar implementation", bad == 0, f"({bad} of {ng} groups differ)")
    # fdiv is the IEEE division, and differs from torch's multiplication by the reciprocal for some inputs
    a = torch.exp2(torch.rand(1 << 20, generator=g, device="cuda") * 20 - 10)
    ref = (a.cpu().numpy() / np.float32(6.0)).astype(np.float32)
    report("fdiv(a, 6) == numpy float32 division (IEEE); torch's a / 6.0 is a * fp32(1/6)", torch.equal(emu.fdiv(a, 6.0).cpu(), torch.from_numpy(ref)), f"(a / 6.0 differs from IEEE for {int((a / 6.0 != emu.fdiv(a, 6.0)).sum())} of {a.numel()})")
    # the 'rn' Q rule against a scalar implementation (zero-scale groups included)
    qn = (torch.randn(2, 3, 512, generator=g, device="cuda") * torch.exp2(torch.rand(2, 3, 512, generator=g, device="cuda") * 8 - 4)).to(torch.bfloat16).float()
    qn[0, 0, :16] *= 1e-6
    qn[0, 1, :] = 0.0
    qn[1, 2, 16:32] *= 3e3
    for n in (1, 2):
        rec = emu.quantize_q(qn, f"nvfp4x{n}", 4, False, "rn")
        bad = 0
        for b_ in range(2):
            for r_ in range(3):
                dense = np.zeros(512)
                r = qn[b_, r_].cpu().numpy().astype(np.float32)
                for _ in range(n):
                    nr = r.copy()
                    for g_ in range(32):
                        codes, sc = slow_nv_group(r[16 * g_:16 * g_ + 16], "rn", False)
                        for k_ in range(16):
                            dense[16 * g_ + k_] += codes[k_] * sc
                            nr[16 * g_ + k_] = np.float32(float(r[16 * g_ + k_]) - codes[k_] * sc)
                    r = nr
                bad += not np.array_equal(rec.dense()[b_, r_].cpu().numpy(), dense)
        report(f"quantize_q nvfp4x{n} rule rn (FlashInfer) == scalar implementation", bad == 0, f"({bad} of 6 rows differ)")

    # (2) P quantization: kperm from the fragment layout, G, bf16 pre-rounding, rules, 4/6, mxfp4
    kp = slow_kperm_blocks()
    closed = [[tk for tk in range(64) if (tk // 32) * 2 + (tk % 8) // 4 == b] for b in range(4)]
    perm0 = emu.kperm_index(64, "cuda").cpu().tolist()
    report("kperm_index blocks == the blocks derived from the S accumulator / A fragment layouts", kp == closed and [sorted(perm0[16 * b:16 * b + 16]) for b in range(4)] == kp and sorted(perm0) == list(range(64)))
    bs_, h_, N_ = 2, 3, 128
    lg = torch.randn(bs_, h_, N_, generator=g, device="cuda") * 3
    Pm = torch.exp2((lg - lg.amax(-1, keepdim=True)).clamp_min(-30) + torch.rand(bs_, h_, N_, generator=g, device="cuda") * 6)       # numerators in (0, 64]
    Pm[0, 0, 10] = 0.0
    Pm[1, 1, 64:96] = 0.0
    cfgs = []
    for m in ("lazy", "eager"):
        for l1 in ("none", "sa3"):
            for rule in ("ceil", "rn"):
                for fs in (False, True):
                    for grp in ("kperm", "contig"):
                        cfgs.append(emu.Mode(p="nvfp4", p_l1=l1, p_rule=rule, p46=fs, p_group=grp, p_pre="bf16" if (l1 == "sa3") == fs else "f32", m=m, B=64))
    cfgs.append(emu.Mode(p="mxfp4", B=64))
    cfgs.append(emu.Mode(p="mxfp4", p_pre="bf16", B=64))
    bad_cfg = []
    for mode in cfgs:
        pc, psf, Gp, perm = emu.quantize_p_fp4(Pm, mode)
        gg = 16 if mode.p == "nvfp4" else 32
        deq = pc.double() * psf.double().repeat_interleave(gg, dim=-1)
        if perm is not None:
            nat = torch.zeros_like(deq)
            nat[..., perm] = deq
            deq = nat
        got = deq.cpu().numpy()
        bad = 0
        for b_ in range(bs_):
            for r_ in range(h_):
                for j in range(N_ // 64):
                    want, Gs = slow_p_block(Pm[b_, r_, 64 * j:64 * j + 64].cpu().numpy(), mode)
                    bad += not (np.array_equal(got[b_, r_, 64 * j:64 * j + 64], want) and Gs == Gp)
        if bad:
            bad_cfg.append((mode.m, mode.p, mode.p_l1, mode.p_rule, mode.p46, mode.p_group, mode.p_pre, bad))
    report("quantize_p_fp4 == scalar implementation (nvfp4: G, bf16 pre-rounding, ceil / rn, 4/6, kperm / contig; mxfp4)", not bad_cfg, f"({len(cfgs)} configs x 6 rows x 2 blocks; failing: {bad_cfg[:3]})")

    # (3) one fp32 rounding per 64-token instruction: crafted codes where the exact sum (rounded once) differs from a sequential fp32 sum
    nb = 3
    pc = torch.ones(1, 2, 64 * nb, device="cuda")                                   # P code 1 everywhere
    psf = torch.ones(1, 2, 4 * nb, device="cuda")
    vc = torch.zeros(1, 64 * nb, 512, device="cuda")
    vsf = torch.ones(1, 4 * nb, 512, device="cuda")
    for j in range(nb):
        vc[0, 64 * j, :] = 4.0                                                      # one product 4 * 2^(18 + j) = 2^(20 + j) (alone in its group of 16 tokens)
        vc[0, 64 * j + 16:64 * j + 64, :] = 1.0                                     # 48 products 2^-9: their exact sum 0.094 is above half an ulp (0.0625) of 2^20, each single one is lost in a sequential sum
        vsf[0, 4 * j, :] = 2.0 ** (18 + j)
        vsf[0, 4 * j + 1:4 * j + 4, :] = 2.0 ** -9
    psf[0, 0, :] = torch.exp2(torch.randint(-1, 2, (4 * nb,), generator=g, device="cuda").float())
    ls = torch.zeros(1, 2, nb, device="cuda")
    s_blk = torch.ones(1, 2, nb, device="cuda")
    s_blk[0, 1, 1] = 0.5
    O, _ = emu.pv_fp4_core(pc, psf, vc, vsf, 16, ls, s_blk, None)
    from fractions import Fraction
    want = np.zeros(2, dtype=np.float32)
    seq0 = np.float32(0.0)
    for hh in range(2):
        acc = np.float32(0.0)
        for j in range(nb):
            tot = Fraction(0)
            for k in range(64):
                gi = (64 * j + k) // 16
                tot += Fraction(float(pc[0, hh, 64 * j + k])) * Fraction(float(psf[0, hh, gi])) * Fraction(float(vc[0, 64 * j + k, 0])) * Fraction(float(vsf[0, gi, 0]))
                if hh == 1 and j == 0:
                    seq0 = np.float32(seq0 + np.float32(float(pc[0, hh, k]) * float(psf[0, hh, gi]) * float(vc[0, k, 0]) * float(vsf[0, gi, 0])))
            acc = np.float32(acc * np.float32(s_blk[0, hh, j].item()))
            acc = np.float32(acc + np.float32(float(tot)))                          # float(Fraction): the correctly rounded double, then float32 (the sums here are exact in double)
        want[hh] = acc
    got = O[0].cpu().numpy()
    report("pv_fp4_core: one fp32 rounding of the exact sum per 64-token instruction == exact Fraction reference", np.array_equal(got, np.broadcast_to(want[:, None], (2, 512))), f"(got {got[:, 0].tolist()}, want {want.tolist()})")
    exact0 = np.float32(2.0 ** 20 + 48 * 2.0 ** -9)
    report("  the crafted block tells the exact sum from a sequential fp32 sum (so the test can fail)", float(seq0) == 2.0 ** 20 and float(exact0) != float(seq0), f"(sequential {float(seq0)}, exact-then-round {float(exact0)})")

    # (4) the full pipeline against the scalar implementation; also the dequantized V' and the first-level multiplier
    OMf = emu.OperandMode
    ok_modes = []

    def nv_om(qn, k="nvf4r", **kw):
        return OMf(qn, k, "nvf4t", **kw)

    fi_k = nv_om("nvfp4x1", k_rule="rn", v_rule="rn")
    fi = emu.Mode(fp8=fi_k, mxfp4=fi_k, fp4=nv_om("nvfp4x1", "native", v_rule="rn"), p="nvfp4", p_rule="rn", p_group="contig", p_pre="bf16", m="eager", B=64, accum="f32seg", q_rule="rn")
    mine = emu.Mode(fp8=nv_om("nvfp4x2", v_l1="grp"), mxfp4=nv_om("nvfp4x2", v_l1="grp"), fp4=nv_om("nvfp4x2", "native", v_l1="grp"), p="nvfp4", p_l1="sa3", p_rule="ceil", m="lazy", B=64, accum="f32seg",
                    q_rule="ceil")
    mine46 = emu.Mode(fp8=nv_om("nvfp4x1", k_rule="rn", v_l1="grp", v_rule="rn", v46=True), mxfp4=nv_om("nvfp4x1", k_rule="rn", v_l1="grp", v_rule="rn", v46=True),
                      fp4=nv_om("nvfp4x1", "native", v_l1="grp", v_rule="rn", v46=True), p="nvfp4", p_l1="sa3", p_rule="rn", p46=True, m="eager", B=64, accum="f64", q_rule="rn")
    f8m = emu.Mode(fp8=OMf("mxfp8x2", "native", "nvf4t", v_l1="grp"), mxfp4=OMf("mxfp8x1", "native", "nvf4t", v_l1="grp"), fp4=nv_om("nvfp4x2", "native", v_l1="grp"), p="nvfp4", p_l1="sa3", p_rule="ceil",
                   m="final", B=64, accum="f32seg", q_rule="ceil")
    mx4 = emu.Mode(mxfp4=OMf("mxfp4x4", "native", "mxf4r"), fp8=OMf("mxfp8x1", "native", "nvf4t"), p="mxfp4", m="eager", B=64, accum="f64", q_kernel_rule=True)
    smooth = emu.Mode(fp8=nv_om("nvfp4x2", k_rule="rn", v_rule="rn", v_smooth=32, k_smooth=32), fp4=nv_om("nvfp4x2", "native", v_rule="rn", v_smooth=32), mxfp4=nv_om("nvfp4x2", k_rule="rn", v_rule="rn", v_smooth=32, k_smooth=32),
                      p="nvfp4", p_l1="sa3", p_rule="rn", m="eager", B=64, accum="f64", q_rule="rn", q_smooth=True)
    cases = [("fp8+fp4", "fi-dsv4-nvfp4 (Q rn x1, K nvf4r rn, P rn contig bf16 eager, V^T rn)", fi),
             ("fp8+fp4", "sa3 ceil kperm lazy, Q x2, V grp ceil, f32seg", mine),
             ("fp8+fp4", "sa3 rn 4/6 on P and V, eager, Q x1 rn, V grp", mine46),
             ("fp8+fp4", "FP8 tile Q mxfp8x2 native QK (f8m), P ceil sa3 final", f8m),
             ("mxfp4", "MXFP4 cache mxfp4x4 Q native + P mxfp4 + V mxf4r, eager", mx4),
             ("fp8+fp8", "fi-dsv4-nvfp4 on two FP8 caches", fi),
             ("mxfp4", "fi-dsv4-nvfp4 on the MXFP4 cache (K requantized)", fi),
             ("fp8+fp4", "smoothing of Q, K, V (tile 32)", smooth),
             ("fp8+fp4", "smoothing of V (tile 32) with the first level grp, K nvf4r smoothing tile 32, sa3 ceil", emu.Mode(fp8=nv_om("nvfp4x2", k_smooth=32, v_l1="grp", v_smooth=32), mxfp4=nv_om("nvfp4x2", k_smooth=32, v_l1="grp", v_smooth=32),
                                                                                                         fp4=nv_om("nvfp4x2", "nvf4r", k_smooth=32, v_l1="grp", v_smooth=32), p="nvfp4", p_l1="sa3", p_rule="ceil", m="lazy", B=64, accum="f64", q_rule="ceil")),
             ("fp8+fp4", "attribution: only P in FP4 (sa3 ceil kperm, eager), QK and V exact", emu.Mode(p="nvfp4", p_l1="sa3", m="eager", B=64)),
             ("fp8+fp4", "attribution: only V^T in FP4 (grp ceil), P float32, QK exact", emu.Mode(fp8=OMf("bf16", "exact", "nvf4t", v_l1="grp"), fp4=OMf("bf16", "exact", "nvf4t", v_l1="grp"), p="nvfp4", p_rule="exact", m="eager", B=64))]
    for cls in ("hv4", "kampw4"):
        for family, label, mode in cases:
            if cls == "kampw4" and family != "fp8+fp4":
                continue
            with upstream_defaults():
                main, extra = FAMILIES[family]
                cfg = CLASSES[cls]
                raw = lib.RawTestParamForDecode(b=2, h_q=4, s_q=1, h_kv=1, s_kv=512, is_varlen=True, topk=96, have_topk_length=True, enable_attn_sink=True,
                                                extra_s_k=512 if extra else None, extra_topk=80 if extra else None, block_size=64, extra_block_size=64 if extra else None,
                                                have_extra_topk_length=bool(extra), d_qk=512, kvcache_layout=main, extra_kvcache_layout=extra, num_runs=0, seed=77)
                p = raw.to_test_param()
                tt = lib.generate_testcase_for_decode(p)
                if cfg["kamp"]:
                    apply_k_gain(tt, 5, cfg.get("kamp_spread", 6.0))
                tt.sm_scale *= cfg["sigma"] / logit_std(tt)
                if tt.attn_sink is not None:
                    tt.attn_sink[0], tt.attn_sink[1], tt.attn_sink[2], tt.attn_sink[3] = 0.3, float("inf"), float("-inf"), -0.5
            with torch.inference_mode():
                out, lse = emu.simulate_decode(p, tt, mode)
            worst_ulp, worst_lse, vbad = 0.0, 0.0, 0
            for bi in range(2):
                parts = {}
                so, sl = slow_decode_token_fp4(tt, mode, bi, 0, parts)
                o = out[bi, 0].float().cpu()
                so = so.float()
                nz = (so != 0) | (o != 0)
                rel = ((o - so).abs() / (so.abs().clamp_min(1e-30)))[nz]
                worst_ulp = max(worst_ulp, float(rel.max()) / 2.0 ** -7 if rel.numel() else 0.0)
                lf = lse[bi, :, 0].cpu()
                fin = torch.isfinite(sl) & torch.isfinite(lf)
                worst_lse = max(worst_lse, float((lf[fin] - sl[fin]).abs().max()) if fin.any() else 0.0)
                assert torch.equal(torch.isinf(lf), torch.isinf(sl))
                if bi == 0:
                    # dequantized V' of the vectorized code (natural token order) against the scalar one
                    sds = [emu.gather_scope(s_, 2, 1) for s_ in ([tt.kv_scope] + ([tt.extra_kv_scope] if tt.extra_kv_scope is not None else []))]
                    codes_, sfv_, col_, vbar_ = emu.v_fp4_terms(sds, [mode.ops(sd.layout) for sd in sds], mode)
                    gg = 16 if mode.p == "nvfp4" else 32
                    deq = (codes_.double() * sfv_.double().repeat_interleave(gg, dim=1))
                    if mode.p == "nvfp4" and mode.p_group == "kperm":
                        perm = emu.kperm_index(deq.shape[1], deq.device)
                        nat = torch.zeros_like(deq)
                        nat[:, perm] = deq
                        deq = nat
                    vbad = int((deq[0].cpu().numpy() != parts["vdeq"]).sum()) + int((col_[0].cpu().numpy() != parts["col"]).sum())
            report(f"vectorized == scalar implementation [{cls} {family} {label}]", worst_ulp <= 1.01 and worst_lse < 5e-5 and vbad == 0,
                   f"(max out diff {worst_ulp:.2f} bf16 ulp, max lse diff {worst_lse:.1e}, dequantized V' differs in {vbad} elements)")
    return ok


# ----------------------------------------------------------------------------------------------
# Two-level NVFP4 cache (T-10): the writer against an independent scalar implementation, g = 0 against the upstream V41_FP4Sparse writer, the emulator's second-level path
# ----------------------------------------------------------------------------------------------
def slow_fp4_2l(x, rule: str, four_six: bool, granularity: str, g_fixed: Optional[int] = None):
    """
    The two-level NVFP4 writer with python scalars and numpy float32 (x: float32 [n, 512], rows are tokens). Returns (dequantized values float64 [n, 512], block scales float64 [n, 32], g list[int]); NaN scale = poisoned block.
    Torch on CUDA divides a float32 tensor by a python scalar as a multiplication with fp32(1 / scalar) (the upstream writer's `amax / 6.0`), so the scale estimate is a * fp32(1 / qmax)
    """
    n = x.shape[0]
    top = quant.FP4_2L_TOP_46 if four_six else quant.FP4_2L_TOP
    finite = np.where(np.isfinite(x), np.abs(x), 0.0)

    def choose_g(amax: float) -> int:
        if not amax > 0:
            return 0
        m, e = math.frexp(float(np.float64(amax) / top))
        return e - 1 if m == 0.5 else e

    gs = [g_fixed if g_fixed is not None else choose_g(float(finite.max()))] * n if granularity == "tensor" else [choose_g(float(finite[i].max())) for i in range(n)]
    vals = np.zeros((n, 512))
    scl = np.zeros((n, 32))
    for i in range(n):
        G = np.float32(2.0 ** gs[i])
        for t in range(32):
            xt = x[i, 16 * t:16 * t + 16]
            if not np.isfinite(xt).all():
                scl[i, t] = np.nan
                vals[i, 16 * t:16 * t + 16] = np.nan
                continue
            a = np.float32(np.abs(xt).max()) / G

            def one(qmax: float):
                if rule == "ceil":
                    s = next((v for v in _T_E4M3[1:] if qmax * v >= float(a)), _T_E4M3[-1])
                else:
                    est = np.float32(a * np.float32(np.float32(1.0) / np.float32(qmax)))
                    s = emu._table_round(float(min(max(est, np.float32(2.0 ** -9)), np.float32(448.0))), _T_E4M3)
                codes = [emu._table_round(float(np.float32(v) / np.float32(np.float32(s) * G)), _T_E2M1) for v in xt]
                return codes, s

            c6, s6 = one(6.0)
            best = (c6, s6)
            if four_six:
                c4, s4 = one(4.0)
                e6 = sum((float(v) - c * s6 * float(G)) ** 2 for v, c in zip(xt, c6))
                e4 = sum((float(v) - c * s4 * float(G)) ** 2 for v, c in zip(xt, c4))
                if e4 < e6:
                    best = (c4, s4)
            vals[i, 16 * t:16 * t + 16] = [c * best[1] * float(G) for c in best[0]]
            scl[i, t] = best[1]
    return vals, scl, gs


def selftest_two_level(verbose: bool = True) -> bool:
    ok = True

    def report(name, passed, detail=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}", flush=True)

    layout = L.V41_FP4Sparse
    dev = "cuda"
    bytes_eq = lambda a, b: torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
    with torch.inference_mode():
        # (1) g = 0 (rule rn, no Four Over Six) is the upstream V41_FP4Sparse writer bit for bit; so is the dequantization
        g = torch.Generator(device=dev).manual_seed(4321)
        x1 = emu._layout_test_input().to(dev)                                                    # large / tiny / spiky tiles, a zero row, NaN rows, a NaN inside one tile
        spread = torch.exp2(torch.randint(-14, 10, (6, 64, 1, 32), generator=g, device=dev).float()).repeat_interleave(16, dim=-1)
        x2 = (torch.randn(6, 64, 1, 512, generator=g, device=dev) * spread).to(torch.bfloat16)  # blocks whose amplitude differs by 2^24
        x2[2, 7, 0, 100] = float("inf")                                                           # an infinity poisons its block
        for nm, x in (("layout test input", x1), ("block amplitudes 2^U(-14, 10)", x2)):
            ref = quant.quantize_k_cache(x, layout)
            mine, g0 = quant.quantize_k_cache_fp4_2l(x, "rn", False, "tensor", g=0)
            report(f"two-level writer with g = 0 == quant.quantize_k_cache(V41_FP4Sparse) bytes [{nm}]", bytes_eq(ref, mine) and g0 == 0)
            d_ref = quant.dequantize_k_cache(ref, layout)
            d_mine = quant.dequantize_k_cache_fp4_2l(mine, 0)
            report(f"two-level dequantization with g = 0 == quant.dequantize_k_cache [{nm}]", torch.equal(torch.nan_to_num(d_ref.float(), nan=7e7), torch.nan_to_num(d_mine.float(), nan=7e7)))
        # (2) the writer against the independent scalar implementation: tensor / token g, rule rn / ceil, Four Over Six off / on
        gen = torch.Generator(device=dev).manual_seed(99)
        xs = (torch.randn(2, 8, 1, 512, generator=gen, device=dev) * 0.1) * torch.exp2(torch.randint(-9, 8, (2, 8, 1, 32), generator=gen, device=dev).float()).repeat_interleave(16, dim=-1)
        xs[0, 0] = 0.0                                                                            # a zero token
        xs[0, 1, 0, :16] = 6.0 * 448.0 * 2.0 ** torch.randint(-3, 3, (16,), generator=gen, device=dev).float() / 6.0      # block with a spiky amplitude pattern
        xs[0, 2, 0, 16:32] = torch.tensor([1.0, 1.5, 0.75, 0.25, 6.0, 5.0, 4.0, 3.5, 2.5, 1.25, 0.5, 0.125, 7.0, 0.0, -2.0, -6.0], device=dev) * 0.01      # e2m1 grid / tie points
        xs[1, 3, 0, 40] = float("nan")                                                            # NaN in one block
        xs = xs.to(torch.bfloat16)
        xnp = xs.reshape(16, 512).float().cpu().numpy()
        for gran in ("tensor", "token"):
            for rule in ("rn", "ceil"):
                for fs in (False, True):
                    q, gg = quant.quantize_k_cache_fp4_2l(xs, rule, fs, gran)
                    dq = quant.dequantize_k_cache_fp4_2l(q, gg).reshape(16, 512).double().cpu().numpy()
                    sc = q.view(16, -1)[:, 256:].view(torch.float8_e4m3fn).double().cpu().numpy()
                    sv, ss, sg = slow_fp4_2l(xnp, rule, fs, gran)
                    gv = [int(gg)] * 16 if gran == "tensor" else [int(v) for v in gg.reshape(-1).cpu()]
                    nan_blocks = np.isnan(ss)
                    same_s = np.array_equal(np.isnan(sc), nan_blocks) and np.array_equal(sc[~nan_blocks], ss[~nan_blocks])
                    nanv = np.repeat(nan_blocks, 16, axis=1)
                    same_v = np.array_equal(np.isnan(dq), nanv) and np.array_equal(dq[~nanv], sv[~nanv])
                    report(f"two-level writer [{gran}, {rule}{', 4/6' if fs else ''}] == independent scalar implementation", same_s and same_v and gv == sg,
                           f"(g {sorted(set(gv))[:4]}; {int((~nan_blocks).sum())} blocks, scales equal={same_s}, values equal={same_v})")
        # (3) scale invariance: a power of two on the data moves g and nothing else (the property that keeps small-magnitude caches alive); 4/6 too
        xc = (torch.randn(3, 64, 1, 512, generator=gen, device=dev) * 0.1).to(torch.bfloat16)
        for rule, fs in (("rn", False), ("ceil", False), ("rn", True)):
            qa, ga = quant.quantize_k_cache_fp4_2l(xc, rule, fs, "tensor")
            qb, gb = quant.quantize_k_cache_fp4_2l((xc.float() * 2.0 ** -12).to(torch.bfloat16), rule, fs, "tensor")
            report(f"x 2^-12 on the data only moves g by -12, same bytes [{rule}{', 4/6' if fs else ''}]", bytes_eq(qa, qb) and gb == ga - 12, f"(g {ga} -> {gb})")
        q1, _ = quant.quantize_k_cache_fp4_2l((xc.float() * 2.0 ** -12).to(torch.bfloat16), "rn", False, "tensor", g=0)
        d1 = quant.dequantize_k_cache(q1, layout).float()
        report("without the second level (g = 0) the same data x 2^-12 is flushed by the scale floor 2^-9 (the failure the second level removes)",
               float((d1 - xc.float() * 2.0 ** -12).norm() / (xc.float() * 2.0 ** -12).norm()) > 0.9, f"(relative error {float((d1 - xc.float() * 2.0 ** -12).norm() / (xc.float() * 2.0 ** -12).norm()):.2f})")
        # (4) properties: rule ceil never saturates a code (6 s G >= amax of the block, s < 448); Four Over Six is blockwise never worse than the same g without it
        qc, gc = quant.quantize_k_cache_fp4_2l(xc, "ceil", False, "tensor")
        sc_ = qc.view(3, 64, -1)[..., 256:].view(torch.float8_e4m3fn).float()
        am = xc.float().view(3, 64, 32, 16).abs().amax(-1)
        report("rule ceil: 6 * s * G >= block amax wherever s < 448", bool(((6.0 * sc_ * 2.0 ** gc >= am) | (sc_ >= 448.0)).all()))
        for rule in ("rn", "ceil"):
            g46 = int(quant.fp4_2l_choose_g(xc.float().abs().amax(), True))
            qa, _ = quant.quantize_k_cache_fp4_2l(xc, rule, False, "tensor", g=g46)
            qb, _ = quant.quantize_k_cache_fp4_2l(xc, rule, True, "tensor", g=g46)
            ea = (quant.dequantize_k_cache_fp4_2l(qa, g46).float() - xc.float()).view(3, 64, 32, 16).pow(2).sum(-1)
            eb = (quant.dequantize_k_cache_fp4_2l(qb, g46).float() - xc.float()).view(3, 64, 32, 16).pow(2).sum(-1)
            report(f"Four Over Six [{rule}]: blockwise squared error never above the 6-only quantization at the same g", bool((eb <= ea * (1 + 1e-6)).all()), f"(total error {float(eb.sum() / ea.sum()):.3f} of the 6-only error)")
        # (5) the emulator's second level: a cache with g != 0 and shifted scale bytes is the same cache as the plain one -> every row gives the same output bit for bit
        names = ["exact", "kernel-nvfp4-mxfp8", "kernel-nvfp4-fp8x2", "kernel-nvfp4-fp8x2-v2s", "p3-mxfp8-nvq2", "fp4-k8x2", "fp4-v8x2-pv"]
        modes = {nm: mo for nm, _, mo, _ in build_combos() if nm in names}
        fam_of = {nm: fams for nm, _, _, fams in build_combos()}
        for family in ("fp8+fp4", "nvfp4", "nvfp4-main"):
            for variant in ("tensor g = -3", "token g in -1 .. -5"):
                with upstream_defaults():
                    c = make_case("hv4", family, 64, 11, 2, 2)
                    t = c.t
                    t2 = dataclasses.replace(t)
                    good = True
                    for fld in ("kv_scope", "extra_kv_scope"):
                        sc0 = getattr(t, fld)
                        if sc0 is None or (sc0.kvcache_layout or L.V41_FP8Sparse) != layout:
                            continue
                        nb, bs = sc0.blocked_k_quantized.shape[:2]
                        raw = sc0.blocked_k_quantized.view(nb, bs, 288).clone()
                        s = raw[..., 256:].view(torch.float8_e4m3fn).float()
                        if variant.startswith("tensor"):
                            gq = -3
                            mul = torch.full((nb, bs, 1), 8.0, device=s.device)
                        else:
                            gq = torch.randint(-5, 0, (nb, bs), generator=gen, device=dev, dtype=torch.int32)
                            mul = torch.exp2(-gq.float()).unsqueeze(-1)
                        ns = s * mul
                        raw[..., 256:] = ns.to(torch.float8_e4m3fn).view(torch.uint8).view(torch.float8_e4m3fn)
                        back = raw[..., 256:].view(torch.float8_e4m3fn).float()
                        good &= bool(torch.equal(torch.nan_to_num(back, nan=-1.0), torch.nan_to_num(ns, nan=-1.0)))        # the shift is exact (no scale left the e4m3 grid)
                        sc2 = dataclasses.replace(sc0)
                        sc2.blocked_k_quantized = raw.view(nb, bs, 1, 288)
                        sc2.fp4_g = gq
                        dq2 = quant.dequantize_k_cache_fp4_2l(sc2.blocked_k_quantized, gq)
                        good &= bool(torch.equal(torch.nan_to_num(dq2.float(), nan=7e7), torch.nan_to_num(sc0.blocked_k.float(), nan=7e7)))
                        setattr(t2, fld, sc2)
                    report(f"[{family}, {variant}] shifted scales + 2^g dequantize to the plain cache", good)
                    allsame = True
                    run_names = [nm for nm in modes if family in fam_of[nm]]
                    for nm in run_names:
                        mo = modes[nm]
                        o1, l1 = emu.simulate_decode(c.p, t, mo)
                        o2, l2 = emu.simulate_decode(c.p, t2, mo)
                        same = torch.equal(o1, o2) and torch.equal(l1, l2)
                        allsame &= same
                        if not same:
                            print(f"    differs: {nm}")
                    report(f"[{family}, {variant}] emulator output bit-identical to the plain cache ({', '.join(run_names)})", allsame)
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return ok


def selftest_framework(verbose: bool = True) -> bool:
    """The vectorized emulator == the slow independent implementation, on small cases that exercise lazy max updates, empty blocks, the fp4 conversions, rebase"""
    ok = True

    def report(name, passed, detail=""):
        nonlocal ok
        ok &= bool(passed)
        print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}", flush=True)

    # (1) block-scaled QK == dense dot product of the dequantized operands, for every Q representation
    with torch.inference_mode():
        g = torch.Generator(device="cuda").manual_seed(0)
        q = (torch.randn(4, 8, 512, generator=g, device="cuda") * torch.exp2(torch.rand(4, 8, 512, generator=g, device="cuda") * 6 - 3)).to(torch.bfloat16).float()
        for name in ("bf16", "mxfp8x1", "mxfp8x2", "nvfp4x1", "nvfp4x2", "nvfp4x4", "mxfp4x1", "mxfp4x2", "mxfp4x4", "nvfp4x4p8", "nvfp4x6pa"):
            qt = emu.quantize_q(q, name)
            kv = (torch.randn(4, 16, 512, generator=g, device="cuda") * 0.3)
            kvt = emu.quantize_q(kv, "mxfp8x1" if qt.g == 32 else "nvfp4x2")      # any Terms with the right group size
            got = emu.qk_logits(qt, kvt, "f64")
            exp_ = sum((qd.double() @ kd.double().transpose(-1, -2)) for qd, kd in [(qt.dense(), kvt.dense())])
            # the lo x lo pair is not issued: subtract it (only Q x K with both multi-term)
            if len(qt.vals) > 1 and len(kvt.vals) > 1:
                lo_q = sum(v.double() * s.double().repeat_interleave(qt.g, dim=-1) for v, s in zip(qt.vals[1:], qt.sf[1:]))
                lo_k = sum(v.double() * s.double().repeat_interleave(kvt.g, dim=-1) for v, s in zip(kvt.vals[1:], kvt.sf[1:]))
                exp_ = exp_ - lo_q @ lo_k.transpose(-1, -2)
            d = (got.double() - exp_).abs().max() / exp_.abs().max()
            err_q = ((qt.dense() - q.double()).norm() / q.double().norm()).item()
            report(f"qk_logits(Q {name}) == dense dot of the dequantized operands", d < 1e-6, f"(rel diff {float(d):.1e}; Q representation error {err_q:.2e})")
        # the kernel's Q scale rule against an independent scalar implementation (numpy float32, struct for the codes), including amax = 448 * 2^k and zero / tiny groups
        import numpy as np
        qk = q.clone()
        qk[0, 0, :32] = 448.0                                  # amax exactly 448: scale must be 1 (or 2 if the fp32 reciprocal rounds up)
        qk[0, 1, :32] = 0.0                                    # all-zero group
        qk[0, 2, :32] = qk[0, 2, :32] * 1e-12                  # tiny group
        qk[0, 3, :32] = 224.0 * torch.sign(qk[0, 3, :32])      # amax exactly 224
        qt = emu.quantize_q(qk, "mxfp8x2", 4, True)
        bad = 0
        rec = qt.dense()
        rf = np.float32(1.0 / 448.0)
        for b_ in range(qk.shape[0]):
            for r_ in range(qk.shape[1]):
                for g_ in range(16):
                    x = qk[b_, r_, 32 * g_:32 * g_ + 32].double().cpu().numpy()
                    x32 = np.float32(np.abs(x).max()) * rf
                    code = 0 if x32 == 0 else int(np.ceil(np.log2(np.float64(x32)) - 1e-12 * 0)) + 127
                    if x32 > 0 and 2.0 ** (code - 127 - 1) >= np.float64(x32):    # ceil(log2) guard against float log2 rounding
                        code -= 1
                    if x32 > 0 and 2.0 ** (code - 127) < np.float64(x32):
                        code += 1
                    code = min(max(code, 8), 250)
                    sc = 2.0 ** (code - 127)
                    y = np.clip(x / sc, -448, 448)
                    hi = torch.tensor(y).to(torch.float32).clamp(-448, 448).to(torch.float8_e4m3fn).to(torch.float64).numpy()
                    lo = torch.tensor((y - hi) * 16.0).to(torch.float32).clamp(-448, 448).to(torch.float8_e4m3fn).to(torch.float64).numpy()
                    mine = rec[b_, r_, 32 * g_:32 * g_ + 32].cpu().numpy()
                    want = hi * sc + lo * sc / 16.0
                    if not np.array_equal(mine, want):
                        bad += 1
        report("kernel Q rule (amax * fp32(1/448), code clamp [8, 250], hi/lo e4m3) == independent scalar implementation", bad == 0, f"({bad} of {qk.shape[0] * qk.shape[1] * 16} groups differ)")
        # the P3 kernel's NVFP4 Q rule (scale = smallest e4m3 >= amax / 6 per 16 dims, residual chain, prescale 'pe' / 'pa' per row) against an independent scalar implementation built from
        # the small float tables (first principles: table search for the scale, table rounding of the e2m1 codes, numpy float32 reciprocal and multiply, fp32 residuals)
        t_e4m3 = emu._small_float_table(3, 7, 0x7E)
        t_e2m1 = emu._small_float_table(1, 1, 7)

        def slow_nv(rowf: "np.ndarray", n: int, pre: str) -> "np.ndarray":
            amax_row = float(np.abs(rowf).max())
            if amax_row == 0.0:
                return np.zeros(512)
            if pre == "pe":
                k = min(max(10 - (math.frexp(amax_row)[1] - 1), -30), 30)
            elif pre == "pa":
                k = min(max(math.floor(math.log2(448.0 * 6.0 / amax_row)), 0), 30)
            else:
                k = 0
            r = (rowf.astype(np.float64) * 2.0 ** k).astype(np.float32)
            dense = np.zeros(512)
            for _ in range(n):
                nr = r.copy()
                for g_ in range(32):
                    x = r[16 * g_:16 * g_ + 16]
                    amax = float(np.abs(x).max())
                    sc = next((v for v in t_e4m3[1:] if 6.0 * v >= amax), t_e4m3[-1])
                    inv = np.float32(1.0) / np.float32(sc)
                    for j_ in range(16):
                        d = emu._table_round(float(x[j_] * inv), t_e2m1)           # y = fp32 (x * fp32 (1 / s))
                        dense[16 * g_ + j_] += d * sc
                        nr[16 * g_ + j_] = np.float32(float(x[j_]) - d * sc)        # fp32 residual (the product is exact: one rounding, like an fma)
                r = nr
            return dense / 2.0 ** k

        qn = q.clone()
        qn[0, 0, :16] = qn[0, 0, :16] * 1e-12                  # tiny group: the scale floor 2^-9
        qn[0, 1, :] = 0.0                                      # all-zero row
        qn[1, 0, :] = qn[1, 0, :] * 300.0                      # large row: negative exponent prescale / scale clamp
        qn[1, 1, 5] = 1344.0                                   # amax = 2688 / 2: exact boundary of the 'pa' rule
        qn[2, 0, 7] = 2688.0
        qn[2, 1, 3] = 3000.0                                   # beyond the range of the ue4m3 scale without prescale
        qn = qn.to(torch.bfloat16).float()
        rows = [(b_, r_) for b_ in range(3) for r_ in range(qn.shape[1])]
        for name, n, pre in (("nvfp4x1", 1, ""), ("nvfp4x2pe", 2, "pe"), ("nvfp4x3pa", 3, "pa"), ("nvfp4x4pe", 4, "pe"), ("nvfp4x4pa", 4, "pa"), ("nvfp4x4", 4, "")):
            rec = emu.quantize_q(qn, name, 4, True).dense().cpu().numpy()
            bad = sum(not np.array_equal(rec[b_, r_], slow_nv(qn[b_, r_].cpu().numpy(), n, pre)) for b_, r_ in rows)
            report(f"kernel NVFP4 Q rule [{name}] == independent scalar implementation", bad == 0, f"({bad} of {len(rows)} rows differ)")
        # the rounding-up scale rule: the first term never saturates (group amax <= 6 * scale) wherever the scale is below its upper limit 448; the plain rule (nearest e4m3) does not guarantee it
        amax_g = qn.double().view(*qn.shape[:-1], 32, 16).abs().amax(-1)
        sc0 = emu.quantize_q(qn, "nvfp4x4", 4, True).sf[0].double()
        sc0_plain = emu.quantize_q(qn, "nvfp4x4", 4, False).sf[0].double()
        report("kernel NVFP4 Q rule: group amax <= 6 * scale wherever scale < 448", bool((amax_g[sc0 < 448.0] <= 6.0 * sc0[sc0 < 448.0]).all()),
               f"(the nearest-e4m3 rule violates it for {int((amax_g > 6.0 * sc0_plain)[sc0_plain < 448.0].sum())} of {amax_g.numel()} groups)")
        # the MXFP4 kernel's Q rule (mxfp4.cu): per term and group of 32 dims the ue8m0 code = ceil-to-power-of-two(fp32(amax) * fp32(1/6)) of the residual, clamped to [8, 250];
        # codes e2m1_rn(x * 2^-e) from the table of e2m1 values; the residual x - code * 2^e in fp32. Independent scalar implementation (numpy float32, math.frexp, table rounding)
        t_e2m1m = emu._small_float_table(1, 1, 7)
        qm = q.clone()
        qm[0, 0, :32] = 6.0 * torch.sign(qm[0, 0, :32])        # amax exactly 6: x = fp32(6 * fp32(1/6)) is 1 or one ulp away from it
        qm[0, 1, :32] = 0.0                                    # all-zero group
        qm[0, 2, :32] = qm[0, 2, :32] * 1e-37                  # below the smallest code: the clamp at 8
        qm[0, 3, :32] = 3.0 * torch.sign(qm[0, 3, :32])        # amax 3
        qm[1, 0, :32] = 1e38 * torch.sign(qm[1, 0, :32])       # above the largest code: the clamp at 250, e2m1 saturation
        qm[1, 1, :32] = 384.0 * torch.sign(qm[1, 1, :32])      # 6 * 2^6
        qm[1, 2, :31] = 0.0                                    # a lone value in the group
        qm = qm.to(torch.bfloat16).float()
        rf6 = np.float32(1.0 / 6.0)

        def slow_mx4(grp: "np.ndarray", n: int) -> "np.ndarray":
            r = grp.astype(np.float32)
            dense = np.zeros(32)
            for _ in range(n):
                x32 = np.float32(np.abs(r).max()) * rf6
                if x32 == 0:
                    code = 0
                else:
                    mm, ee = math.frexp(float(x32))
                    code = (ee - 1 if mm == 0.5 else ee) + 127
                code = min(max(code, 8), 250)
                sc = 2.0 ** (code - 127)
                inv = np.float32(2.0 ** -(code - 127))
                nr = r.copy()
                for j_ in range(32):
                    d = emu._table_round(float(r[j_] * inv), t_e2m1m)
                    dense[j_] += d * sc
                    nr[j_] = np.float32(float(r[j_]) - d * sc)
                r = nr
            return dense

        for n in (1, 2, 3, 4):
            rec = emu.quantize_q(qm, f"mxfp4x{n}", 4, True).dense().cpu().numpy()
            bad = 0
            for b_ in range(qm.shape[0]):
                for r_ in range(qm.shape[1]):
                    for g_ in range(16):
                        bad += not np.array_equal(rec[b_, r_, 32 * g_:32 * g_ + 32], slow_mx4(qm[b_, r_, 32 * g_:32 * g_ + 32].cpu().numpy(), n))
            report(f"kernel MXFP4 Q rule [mxfp4x{n}] == independent scalar implementation", bad == 0, f"({bad} of {qm.shape[0] * qm.shape[1] * 16} groups differ)")
        # each term of the rule is the plain rule's grid with a different (never smaller) scale: the first term never saturates unless the code is clamped
        am4 = qm.double().view(*qm.shape[:-1], 16, 32).abs().amax(-1)
        sc4 = emu.quantize_q(qm, "mxfp4x1", 4, True).sf[0].double()
        sat_ok = (am4 <= 6.0 * sc4 * (1 + 2.0 ** -22)) | (sc4 >= 2.0 ** (250 - 127)) | (am4 == 0)
        report("kernel MXFP4 Q rule: group amax <= 6 * scale unless the code is clamped at 250 (up to the fp32 reciprocal rounding)", bool(sat_ok.all()), f"({int((~sat_ok).sum())} groups violate it)")
        qd8 = emu.quantize_q(torch.cat([q, qk], 0), "mxfp8x1", 4, True).dense()
        report("mxfp8x1 Q (kernel rule) dequantized to float32 is exact (the source of the -q8 rows)", torch.equal(qd8.float().double(), qd8))
        # the scale itself, exhaustively over every finite non-negative bf16 and a log-uniform float32 sample: the smallest e4m3 value v with 6 * v >= amax (searchsorted on the table), clamped to [2^-9, 448]
        tab = torch.tensor(t_e4m3[1:], dtype=torch.float64, device="cuda")                       # 2^-9 .. 448
        am = torch.cat([torch.arange(0, 0x7F80, dtype=torch.int32, device="cuda").to(torch.int16).view(torch.bfloat16).float(),
                        torch.exp2(torch.rand(1 << 22, generator=g, device="cuda") * 30 - 20), tab.float() * 6.0, (tab.float() * 6.0) * (1 + 2.0 ** -23), (tab.float() * 6.0) * (1 - 2.0 ** -23)])
        want = tab[torch.searchsorted(tab * 6.0, am.double(), right=False).clamp(max=tab.numel() - 1)]
        report("ue4m3_scale_ceil == smallest e4m3 >= amax / 6 (searchsorted on the table), all bf16 + 4M float32 + the tie neighbourhoods", torch.equal(emu.ue4m3_scale_ceil(am).double(), want), f"({am.numel()} values)")
        # fp32 accumulation per MMA instruction: one rounding per k64 instruction for the fp4 kinds (qk_logits(accum="f32seg") vs an independent loop)
        qa = emu.quantize_q(qn[:2, :2], "nvfp4x3pe", 4, True)
        kc = emu.e2m1_rn(torch.randn(2, 6, 512, generator=g, device="cuda") * 2)
        ksf = emu.e4m3_rn(torch.exp2(torch.rand(2, 6, 32, generator=g, device="cuda") * 6 - 5))
        got = emu.qk_logits(qa, emu.Terms([kc], [ksf], 16, 64), "f32seg").cpu().numpy()
        qd = [(v.double() * sf_.double().repeat_interleave(16, dim=-1)).cpu().numpy() for v, sf_ in zip(qa.vals, qa.sf)]
        kd = (kc.double() * ksf.double().repeat_interleave(16, dim=-1)).cpu().numpy()
        want = np.zeros((2, 2, 6), dtype=np.float32)
        for b_ in range(2):
            for h_ in range(2):
                for t_ in range(6):
                    acc = np.float32(0.0)
                    for st in range(8):
                        for i_ in range(len(qd)):
                            acc = np.float32(acc + np.float32(float((qd[i_][b_, h_, 64 * st:64 * st + 64] * kd[b_, t_, 64 * st:64 * st + 64]).sum())))
                    want[b_, h_, t_] = acc
        report("qk_logits(f32seg) of the fp4 kinds: one fp32 rounding per k64 instruction == independent loop", np.array_equal(got, want), f"({int((got != want).sum())} of {want.size} differ)")
        # the same with the warp-pair split (f32pair): two partial sums over steps 0..3 and 4..7, added at the end
        qm3 = emu.quantize_q(qm[:2, :2], "mxfp4x3", 4, True)
        kc = emu.e2m1_rn(torch.randn(2, 6, 512, generator=g, device="cuda") * 2)
        ksf = torch.exp2(torch.randint(-6, 3, (2, 6, 16), generator=g, device="cuda").float())
        got = emu.qk_logits(qm3, emu.Terms([kc], [ksf], 32, 64), "f32pair").cpu().numpy()
        qd = [(v.double() * sf_.double().repeat_interleave(32, dim=-1)).cpu().numpy() for v, sf_ in zip(qm3.vals, qm3.sf)]
        kd = (kc.double() * ksf.double().repeat_interleave(32, dim=-1)).cpu().numpy()
        want = np.zeros((2, 2, 6), dtype=np.float32)
        for b_ in range(2):
            for h_ in range(2):
                for t_ in range(6):
                    halves = []
                    for steps in (range(0, 4), range(4, 8)):
                        acc = np.float32(0.0)
                        for st in steps:
                            for i_ in range(len(qd)):
                                acc = np.float32(acc + np.float32(float((qd[i_][b_, h_, 64 * st:64 * st + 64] * kd[b_, t_, 64 * st:64 * st + 64]).sum())))
                        halves.append(acc)
                    want[b_, h_, t_] = np.float32(halves[0] + halves[1])
        report("qk_logits(f32pair) of the mxf4 kind: two partial sums over the halves of the dims == independent loop", np.array_equal(got, want), f"({int((got != want).sum())} of {want.size} differ)")
        # the prescale only moves the ue4m3 scale floor: x6 with it must be much more accurate than x6 without it, and 'pa' must never clip (error bounded by the last term)
        e_plain = ((emu.quantize_q(q, "nvfp4x6").dense() - q.double()).norm() / q.double().norm()).item()
        e_pa = ((emu.quantize_q(q, "nvfp4x6pa").dense() - q.double()).norm() / q.double().norm()).item()
        report("nvfp4 prescale lowers the Q representation error floor", e_pa < 0.05 * e_plain and e_plain < 3e-4, f"(x6 plain {e_plain:.2e}, x6 prescaled {e_pa:.2e})")
    # (2) full pipeline vs the slow implementation
    mx = ops8()
    x2 = ops8(q="mxfp8x2")
    cases = [
        ("fp8+fp4", "mxfp8 lazy", emu.Mode(**mx, p="e4m3x1")),
        ("fp8+fp4", "fp8x2 sfrow", emu.Mode(**x2, p="e4m3x2", p_sf="row")),
        ("fp8+fp4", "fp4 k8x2/v8x2, B32, mfinal, sumq", emu.Mode(**ops8(q="mxfp8x2", k4="k8x2", v8="x2"), p="e4m3x2", B=32, m="final", sum="quant")),
        ("fp8+fp4", "nvfp4x2 native QK, exact PV", emu.Mode(fp4=OM("nvfp4x2", "native", "exact"))),
        ("fp8+fp4", "nvfp4x1 full + rz", emu.Mode(fp8=mx["fp8"], fp4=OM("nvfp4x1", "native", "rebase", "x1"), p="e4m3x1", rebase_round="rz")),
        ("mxfp4", "mxfp4x2 QK, rebase", emu.Mode(mxfp4=OM("mxfp4x2", "native", "rebase"), fp8=mx["fp8"], p="e4m3x2")),
        ("fp8+mxfp4", "mxfp8x2 + p bf16 + rebase", emu.Mode(**x2)),
        ("mxfp4", "MXFP4 kernel, mxfp4x4 tier (Q x4 mxfp4 with the kernel rule, P x2, V headroom -6, B32)",
         emu.Mode(mxfp4=OM("mxfp4x4", "native", "rebase", v_off=-6), p="e4m3x2", B=32, q_kernel_rule=True)),
        ("mxfp4-main", "MXFP4 kernel, mxfp8 tier (Q x2 mxfp4, P x1, V headroom -6, B32)",
         emu.Mode(mxfp4=OM("mxfp4x2", "native", "rebase", v_off=-6), p="e4m3x1", B=32, q_kernel_rule=True)),
        ("mxfp4", "MXFP4 rebase with a large headroom (-20) and rz, Q x3, P bf16",
         emu.Mode(mxfp4=OM("mxfp4x3", "native", "rebase", v_off=-20), rebase_round="rz", B=32, q_kernel_rule=True)),
        ("fp8+fp4", "P3 mxfp8 step 1 (3 NVFP4 Q terms with the exponent prescale, V direct d1, B32, kernel Q rules)",
         emu.Mode(fp8=mx["fp8"], fp4=OM("nvfp4x3pe", "native", "rebase", "d1"), p="e4m3x1", B=32, q_kernel_rule=True)),
        ("fp8+fp4", "P3 mxfp8 kernel (2 NVFP4 Q terms, V direct d1s: the scale from 6 x the largest 16-dim scale, B32, kernel Q rules)",
         emu.Mode(fp8=mx["fp8"], fp4=OM("nvfp4x2", "native", "rebase", "d1s"), p="e4m3x1", B=32, q_kernel_rule=True)),
        ("fp8+fp4", "P3 fp8x2 control (4 NVFP4 Q terms with the prescale 'pa', V direct hi + lo d2, P x2, B32)",
         emu.Mode(fp8=x2["fp8"], fp4=OM("nvfp4x4pa", "native", "rebase", "d2"), p="e4m3x2", B=32, q_kernel_rule=True)),
        ("fp8+fp4", "P3 q8 source (2 NVFP4 Q terms from the mxfp8x1 Q, prescale 'pe', V d1, B32)",
         emu.Mode(fp8=mx["fp8"], fp4=dataclasses.replace(OM("nvfp4x2pe", "native", "rebase", "d1"), q_src="mxfp8x1"), p="e4m3x1", B=32, q_kernel_rule=True)),
        ("nvfp4", "T-10 mxfp8 tier on FP4 main + extra (Q x2, V d1s, P x1, B32, kernel Q rules; f64 accumulation)",
         emu.Mode(fp4=OM("nvfp4x2", "native", "rebase", "d1s"), p="e4m3x1", B=32, q_kernel_rule=True)),
        ("nvfp4", "T-10 precise tier on FP4 main + extra (Q x4 second level pa, V d2, P x2, B32)",
         emu.Mode(fp4=OM("nvfp4x4pa", "native", "rebase", "d2"), p="e4m3x2", B=32, q_kernel_rule=True)),
        ("nvfp4-main", "T-10 FP4 main cache only (Q x3 pa, V d2s, P x2, B32)",
         emu.Mode(fp4=OM("nvfp4x3pa", "native", "rebase", "d2s"), p="e4m3x2", B=32, q_kernel_rule=True)),
        ("fp8+fp8", "exact", emu.Mode()),
        ("fp8+fp4", "k8x2 / v8x2 with lolo", emu.Mode(**ops8(q="mxfp8x2", k4="k8x2", v8="x2"), p="e4m3x2", lolo=True)),
        ("fp8+fp8", "P2 kernel parameters (Q x2, P x2, B32, kernel Q rule; f64 accumulation, the slow reference has no f32seg)", emu.Mode(**ops8(q="mxfp8x2"), p="e4m3x2", B=32, q_kernel_rule=True)),
        ("fp8", "P2 kernel middle tier (Q x2, P x1)", emu.Mode(**ops8(q="mxfp8x2"), p="e4m3x1", B=32, q_kernel_rule=True)),
    ]
    for cls in ("hv4", "kamp4", "kampw4"):
        for family, label, mode in cases:
            with upstream_defaults():
                main, extra = FAMILIES[family]
                cfg = CLASSES[cls]
                raw = lib.RawTestParamForDecode(b=2, h_q=4, s_q=1, h_kv=1, s_kv=512, is_varlen=True, topk=96, have_topk_length=True, enable_attn_sink=True,
                                                extra_s_k=512 if extra else None, extra_topk=80 if extra else None, block_size=64, extra_block_size=64 if extra else None,
                                                have_extra_topk_length=bool(extra), d_qk=512, kvcache_layout=main, extra_kvcache_layout=extra, num_runs=0, seed=77)
                p = raw.to_test_param()
                t = lib.generate_testcase_for_decode(p)
                if cfg["kamp"]:
                    apply_k_gain(t, 5, cfg.get("kamp_spread", 6.0))
                t.sm_scale *= cfg["sigma"] / logit_std(t)
                if t.attn_sink is not None:
                    t.attn_sink[0], t.attn_sink[1], t.attn_sink[2], t.attn_sink[3] = 0.3, float("inf"), float("-inf"), -0.5
            with torch.inference_mode():
                out, lse = emu.simulate_decode(p, t, mode)
            worst_ulp, worst_lse = 0, 0.0
            for bi in range(2):
                so, sl = slow_decode_token(t, mode, bi, 0)
                o = out[bi, 0].float().cpu()
                so = so.float()
                nz = (so != 0) | (o != 0)
                rel = ((o - so).abs() / (so.abs().clamp_min(1e-30)))[nz]
                worst_ulp = max(worst_ulp, float(rel.max()) / 2.0 ** -7 if rel.numel() else 0.0)     # 1 bf16 ulp is at most 2^-7 of the value
                lf = lse[bi, :, 0].cpu()
                fin = torch.isfinite(sl) & torch.isfinite(lf)
                worst_lse = max(worst_lse, float((lf[fin] - sl[fin]).abs().max()) if fin.any() else 0.0)
                assert torch.equal(torch.isinf(lf), torch.isinf(sl))
            report(f"vectorized == slow independent implementation [{cls} {family} {label}]", worst_ulp <= 1.01 and worst_lse < 5e-5, f"(max out diff {worst_ulp:.2f} bf16 ulp, max lse diff {worst_lse:.1e})")
    ok &= selftest_fp4()
    ok &= selftest_two_level()
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return ok


# ----------------------------------------------------------------------------------------------
# Gate against a reference row (T-08): a combination passes when, in every cell (data class, heads) of the family, its out cos_diff and its lse error are not worse than the reference's
# ----------------------------------------------------------------------------------------------
GATE_RTOL = 0.01                                     # 'not worse' allows 1 % (the logits of rows with the same QK are identical, the rest differs by float32 rounding noise)
GATE_ATOL = dict(cos_diff=1e-9, lse_max_abs=1e-6)    # and an absolute floor (1e-6 is the float32 resolution of an lse of ~10)
TILE_WEIGHTS = (192, 576)                            # tokens of the main and the extra cache in the generated data (topk 192, extra 576)


def mode_cost(mode: emu.Mode, family: str) -> Optional[Tuple[float, float]]:
    """
    A rough cost of a full FP4 pipeline row per family, token-weighted over the tiles: (MMA instructions per 8 tokens and 16 heads, extra producer / consumer work in arbitrary units).
    MMA: QK = Q terms x 8 (k64 instructions of the FP4 kinds over the 512 dims) or x 16 (k32 mxf8f6f4), PV = 8 (one k64 instruction per 8 output columns and 64 tokens, 512 / 8 / 8).
    Work: producer = V^T conversion (nvf4t 3, +1 first level grp, +2 Four Over Six, +0.5 ceil rule, +1 smoothing; mxf4r 1) + K re-quantization (nvf4r 2, +0.5 ceil, +2 smoothing);
    consumer = P quantization (nvfp4 2, +0.5 ceil, +2 Four Over Six, +0.5 bf16 pre-rounding; mxfp4 1.5) + 1 for the eager max (rescale of the accumulators every tile) + 1 for the Q smoothing.
    None for rows that are not complete FP4 pipelines (attribution rows, the other modes)
    """
    if mode.p not in emu.FP4_P or mode.p_rule == "exact":
        return None
    main, extra = FAMILIES[family]
    tiles = [(main, TILE_WEIGHTS[0])] + ([(extra, TILE_WEIGHTS[1])] if extra else [])
    tot = float(sum(w for _, w in tiles))
    mma = work = 0.0
    for layout, w in tiles:
        om = mode.ops(layout)
        kind, n = emu.parse_q_mode(om.q)
        if kind == "bf16" or om.v not in ("nvf4t", "mxf4r"):
            return None
        prod = 0.0
        if om.k == "nvf4r":
            prod += 2 + (0.5 if om.k_rule == "ceil" else 0) + (2 if om.k_smooth else 0)
        if om.v == "nvf4t":
            prod += 3 + (1 if om.v_l1 == "grp" else 0) + (2 if om.v46 else 0) + (0.5 if om.v_rule == "ceil" else 0) + (1 if om.v_smooth else 0)
        else:
            prod += 1
        mma += w * (n * (16 if kind == "mxfp8" else 8) + 8)
        work += w * prod
    cons = (2 + (0.5 if mode.p_rule == "ceil" else 0) + (2 if mode.p46 else 0) + (0.5 if mode.p_pre == "bf16" else 0)) if mode.p == "nvfp4" else 1.5
    cons += (1 if mode.m == "eager" else 0) + (1 if mode.q_smooth else 0)
    return mma / tot, work / tot + cons


def gate_tables(rows: List[dict], combos, ref: str) -> str:
    fams = [f for f in FAMILIES if any(r["family"] == f for r in rows)]
    by = {}
    for r in rows:
        by.setdefault((r["family"], r["combo"], r["cls"], r["h"]), []).append(r)
    cell = lambda f, n, c, h, key: max(r[key] for r in by[(f, n, c, h)])
    mode_of = {c[0]: c[2] for c in combos}
    desc = {c[0]: c[1] for c in combos}
    keys = (("cos_diff", "out cos_diff"), ("lse_max_abs", "lse max abs error"))
    out = [f"\n## Gate against `{ref}`\n",
           f"A combination passes when in every cell (data class x heads; worst case over the repeated seeds) of the family both its out cos_diff and its lse max abs error are not worse than those of `{ref}` "
           f"(tolerance: {GATE_RTOL:.0%} relative + {GATE_ATOL['cos_diff']:g} / {GATE_ATOL['lse_max_abs']:g} absolute; `strict` counts the cells without the tolerance). "
           "`margin` = reference error / candidate error in the worst cell (>= 1: not worse; the cell is named). `cost` = (MMA instructions per 8 tokens and 16 heads, extra producer + consumer work), see `mode_cost`; "
           "rows that are not complete FP4 pipelines have none.\n"]
    summary = []
    for f in fams:
        cells = sorted({(r["cls"], r["h"]) for r in rows if r["family"] == f and r["combo"] == ref}, key=lambda x: (list(CLASSES).index(x[0]) if x[0] in CLASSES else -1, x[1]))
        if not cells:
            continue
        main, extra = FAMILIES[f]
        out.append(f"\n### family `{f}` (main {main.name}, extra {extra.name if extra else 'none'}), {len(cells)} cells\n")
        out.append(f"Reference `{ref}`, worst case per cell: out cos_diff / lse max abs error\n")
        out.append("| cell | " + " | ".join(f"{c}/{h}" for c, h in cells) + " |")
        out.append("|---|" + "---|" * len(cells))
        out.append("| cos_diff | " + " | ".join(fmt(cell(f, ref, c, h, "cos_diff")) for c, h in cells) + " |")
        out.append("| lse abs | " + " | ".join(fmt(cell(f, ref, c, h, "lse_max_abs")) for c, h in cells) + " |")
        out.append("\n| combination | cells passed (strict / with tolerance) | cos margin (worst cell) | lse margin (worst cell) | cost (MMA, work) | verdict |")
        out.append("|---|---|---|---|---|---|")
        passing = []
        names = [c[0] for c in combos if c[0] != ref and all((f, c[0], cl, h) in by for cl, h in cells)]
        for name in names:
            res = {}
            for key, _ in keys:
                strict = tol = 0
                worst, wcell = float("inf"), None
                for cl, h in cells:
                    a, b = cell(f, ref, cl, h, key), cell(f, name, cl, h, key)
                    strict += b <= a + GATE_ATOL[key]
                    tol += b <= a * (1 + GATE_RTOL) + GATE_ATOL[key]
                    if not (a == 0 and b == 0):
                        mg = a / b if b > 0 else float("inf")
                        if mg < worst:
                            worst, wcell = mg, (cl, h)
                res[key] = (strict, tol, worst, wcell)
            cs, ls_ = res["cos_diff"], res["lse_max_abs"]
            both_strict = sum(1 for cl, h in cells if cell(f, name, cl, h, "cos_diff") <= cell(f, ref, cl, h, "cos_diff") + GATE_ATOL["cos_diff"]
                              and cell(f, name, cl, h, "lse_max_abs") <= cell(f, ref, cl, h, "lse_max_abs") + GATE_ATOL["lse_max_abs"])
            both_tol = sum(1 for cl, h in cells if cell(f, name, cl, h, "cos_diff") <= cell(f, ref, cl, h, "cos_diff") * (1 + GATE_RTOL) + GATE_ATOL["cos_diff"]
                           and cell(f, name, cl, h, "lse_max_abs") <= cell(f, ref, cl, h, "lse_max_abs") * (1 + GATE_RTOL) + GATE_ATOL["lse_max_abs"])
            ok = both_tol == len(cells)
            cost = mode_cost(mode_of[name], f)
            mg = lambda t: ("inf" if t[2] == float("inf") else f"{t[2]:.2f}") + (f" ({t[3][0]}/{t[3][1]})" if t[3] else "")
            out.append(f"| {name} | {both_strict} / {both_tol} of {len(cells)} | {mg(cs)} | {mg(ls_)} | " + (f"{cost[0]:.0f}, {cost[1]:.1f}" if cost else "-") + f" | {'PASS' if ok else 'fail'} |")
            if ok and cost and name.startswith("nvfp4-"):
                passing.append((cost[0], cost[1], name, min(cs[2], ls_[2]), both_strict == len(cells)))
        passing.sort()
        summary.append((f, passing))
    out.append("\n## Cheapest combinations that pass the gate\n")
    out.append("Ordered by (MMA instructions, extra work); `margin` = the smaller of the two worst-cell margins; `strict` = also passes without the tolerance.\n")
    for f, passing in summary:
        out.append(f"\n- family `{f}`: " + ("none of the `nvfp4-*` rows passes" if not passing else ""))
        for mma, work, name, mg, strict in passing[:8]:
            out.append(f"  - `{name}`: MMA {mma:.0f}, work {work:.1f}, worst margin {mg:.2f}{'' if strict else ' (only with the tolerance)'}")
    out.append("\n## Row definitions of the FP4 PV rows\n")
    for c in combos:
        if c[0].startswith("nvfp4-") or c[0] == ref:
            out.append(f"- `{c[0]}`: {c[1]}")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classes", default="upstream,corner,hv1,hv4,hv16,kamp1,kamp4,kampw4,kampw16", help="Comma separated data classes (corner = all-invalid / empty rows)")
    ap.add_argument("--families", default=",".join(FAMILIES), help="Comma separated cache layout families")
    ap.add_argument("--heads", default="64,128")
    ap.add_argument("--combos", default="*", help="Comma separated fnmatch patterns of combination names")
    ap.add_argument("-b", type=int, default=16)
    ap.add_argument("--s-q", type=int, default=4)
    ap.add_argument("--repeat", type=int, default=1, help="Number of seeds per (data class, family, heads)")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--md", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--gate-ref", default=None, help="Append the gate section against this combination (e.g. fi-dsv4-nvfp4) to the markdown")
    ap.add_argument("--from-csv", default=None, help="Do not run: read the rows of an earlier --csv and write the markdown (with --md / --gate-ref)")
    ap.add_argument("--selftest", action="store_true", help="Check the emulator against an independent slow implementation")
    ap.add_argument("--selftest-fp4", action="store_true", help="Only the FP4 PV part of --selftest (nvfp4 / mxfp4 P, V^T, K requantization, FlashInfer reference row)")
    ap.add_argument("--check-exact", action="store_true", help="Run only the exact combination; exit 1 if it fails the upstream tolerances anywhere")
    ap.add_argument("--fp4-2l", default="off", choices=["off", "tensor", "token"], help="Write the V41_FP4Sparse caches of the generated data with the two-level NVFP4 writer (quant.quantize_k_cache_fp4_2l): one 2^g per cache tensor / per token. "
                    "off = the upstream writer (g = 0)")
    ap.add_argument("--fp4-rule", default="rn", choices=["rn", "ceil"], help="With --fp4-2l: the e4m3 rule of the block scale")
    ap.add_argument("--fp4-46", action="store_true", help="With --fp4-2l: Four Over Six")
    args = ap.parse_args()

    combos = build_combos()
    if args.selftest_fp4:
        ok = selftest_fp4()
        print("ALL PASS" if ok else "SOME CHECKS FAILED")
        sys.exit(0 if ok else 1)
    if args.selftest:
        sys.exit(0 if selftest_framework() else 1)
    if args.list:
        for name, desc, mode, fams in combos:
            print(f"{name:22s} {','.join(fams):40s} {desc}")
        return
    if args.from_csv:
        def conv(v: str):
            if v in ("True", "False"):
                return v == "True"
            for t_ in (int, float):
                try:
                    return t_(v)
                except ValueError:
                    pass
            return v
        with open(args.from_csv, newline="") as f:
            rows = [{k: conv(v) for k, v in r.items()} for r in csv.DictReader(f)]
        names = {r["combo"] for r in rows}
        combos = [x for x in combos if x[0] in names]
        classes = [c for c in args.classes.split(",") if c and any(r["cls"] == c for r in rows)]
        md = summarize(rows, combos, classes)
        if args.gate_ref:
            md += gate_tables(rows, combos, args.gate_ref)
        with open(args.md, "w") as f:
            f.write(md)
        return
    pats = args.combos.split(",")
    if args.check_exact:
        pats = ["exact"]
    combos = [x for x in combos if any(fnmatch.fnmatch(x[0], p) for p in pats)]
    classes = [x for x in args.classes.split(",") if x]
    specs = case_specs(classes, args.families.split(","), [int(x) for x in args.heads.split(",")], args.repeat)
    total = sum((NUM_CORNER_CASES + (len(sp[2]) if len(sp) > 2 else 0)) if sp[0] == "corner" else 1 for sp in specs)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    fp4_2l = None if args.fp4_2l == "off" else dict(granularity=args.fp4_2l, rule=args.fp4_rule, four_six=args.fp4_46)
    rows = run(iter_cases(specs, args.b, args.s_q, fp4_2l=fp4_2l), combos, total=total)
    print(f"{len(rows)} runs in {time.time() - t0:.0f} s; peak GPU memory allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    if args.md:
        with open(args.md, "w") as f:
            f.write(summarize(rows, combos, classes))
            if args.gate_ref:
                f.write(gate_tables(rows, combos, args.gate_ref))
    if args.check_exact:
        bad = [r for r in rows if not (r["out_ok"] and r["lse_ok"])]
        print(f"exact mode: {len(rows) - len(bad)}/{len(rows)} cases pass the upstream tolerances")
        for r in bad:
            print("  FAIL", {k: r[k] for k in ("cls", "family", "h", "cos_diff", "max_abs", "lse_max_abs", "lse_max_rel", "out_ok", "lse_ok")})
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
