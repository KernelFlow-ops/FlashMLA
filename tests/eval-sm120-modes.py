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


def make_case(cls: str, family: str, h: int, seed: int, b: int, s_q: int, topk: int = 192, extra_topk: int = 576) -> Case:
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


def make_corner_cases(seed0: int, b: int, s_q: int) -> List[Case]:
    """All-invalid indices and zero-length sequences with attn_sink / topk_length: lonely query tokens (out = 0, lse = +inf) and empty blocks"""
    cases = []
    for i, (kw, family) in enumerate([(dict(is_all_indices_invalid=True), "fp8+fp4"), (dict(have_zero_seqlen_k=True, have_topk_length=True, have_extra_topk_length=True), "fp8+fp8"),
                                      (dict(have_zero_seqlen_k=True, have_topk_length=True), "fp8"), (dict(have_zero_seqlen_k=True, have_topk_length=True, have_extra_topk_length=True), "mxfp4")]):
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
ALL = ("fp8", "fp8+fp8", "fp8+fp4", "fp8+mxfp4", "mxfp4", "mxfp4-main")
FP8_LIKE = ("fp8", "fp8+fp8", "fp8+fp4", "fp8+mxfp4", "mxfp4", "mxfp4-main")      # every family has MXFP8-compatible operands (fp4 caches via the k8 conversion)
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

    # ---- MXFP4 (e2m1 + ue8m0 per 32): Q as e4m3 (mxf8f6f4) or e2m1 (mxf4)
    for q in ("mxfp8x1", "mxfp8x2", "mxfp4x1", "mxfp4x2", "mxfp4x3", "mxfp4x4", "mxfp4x6"):
        add(f"mxfp4-{q}-qk", f"q-{q} on the MXFP4 cache (QK only; everything else exact)", HAS_MXFP4, mxfp4=OM(q, "native", "exact"))
    add("mxfp4-mxfp8x1-full", "= mxfp8 pipeline on the MXFP4 cache (e2m1 is an exact e4m3 value)", HAS_MXFP4, **mx, p="e4m3x1")
    add("mxfp4-mxfp4x2-full", "main fp8x2 pipeline; mxfp4 cache: q-mxfp4x2 QK, v rebase; p-e4m3x2", HAS_MXFP4, fp8=x2["fp8"], mxfp4=OM("mxfp4x2", "native", "rebase"), p="e4m3x2")
    for n in (3, 4):
        add(f"mxfp4-mxfp4x{n}-full", f"main fp8x2 pipeline; mxfp4 cache: q-mxfp4x{n} QK, v rebase; p-e4m3x2", HAS_MXFP4, fp8=x2["fp8"], mxfp4=OM(f"mxfp4x{n}", "native", "rebase"), p="e4m3x2")

    # ---- the MXFP4 kernel (csrc/cuda_kernels/sm120/decode/sparse/mxfp4.cu), parameter for parameter: QK on mxf4 2X (m16n8k64, fp32 per instruction), Q as N e2m1 terms with a
    #      ue8m0 scale per head and 32 dims (ceil-to-power-of-two(amax * fp32(1/6)), code clamped to [8, 250], residual chain), P = hi * 2^-2 + lo * 2^-6 as for mxfp8.cu,
    #      V rebased per 32 tokens with the block exponent = (largest exponent of the valid tokens) - 6. mxfp8 tier: 2 Q terms, 1 P term; fp8x2 / bf16 tier: 4 and 2
    mxk = lambda n: dict(mxfp4=OM(f"mxfp4x{n}", "native", "rebase", v_off=-6))
    add("kernel-mxfp4-mxfp8", "the MXFP4 kernel, mxfp8 tier: Q x2 (mxfp4), P x1, V rebase with 6 exponent steps of headroom, B=32, fp32 per-instruction accumulation", MXFP4_KERNEL_FAMILIES,
        **mxk(2), p="e4m3x1", **kern)
    add("kernel-mxfp4-fp8x2", "the MXFP4 kernel, fp8x2 / bf16 tier: Q x4 (mxfp4), P x2, V rebase with headroom", MXFP4_KERNEL_FAMILIES, **mxk(4), p="e4m3x2", **kern)
    add("kernel-mxfp4-q3p2", "MXFP4 kernel variant: Q x3, P x2", MXFP4_KERNEL_FAMILIES, **mxk(3), p="e4m3x2", **kern)
    add("kernel-mxfp4-q4p1", "MXFP4 kernel variant: Q x4, P x1", MXFP4_KERNEL_FAMILIES, **mxk(4), p="e4m3x1", **kern)
    add("kernel-mxfp4-q2p2", "MXFP4 kernel variant: Q x2, P x2", MXFP4_KERNEL_FAMILIES, **mxk(2), p="e4m3x2", **kern)
    add("kernel-mxfp4-nohead", "kernel-mxfp4-fp8x2 without the V headroom (rebase to the largest exponent, like mxfp8.cu)", MXFP4_KERNEL_FAMILIES,
        mxfp4=OM("mxfp4x4", "native", "rebase"), p="e4m3x2", **kern)
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


NUM_CORNER_CASES = 4


def case_specs(classes: List[str], families: List[str], heads: List[int], repeat: int = 1) -> List[tuple]:
    """The data to generate, in order, with their seeds: ('corner', seed) stands for NUM_CORNER_CASES cases, (cls, family, h, seed) for one"""
    specs, seed = [], 100
    for cls in classes:
        for rep in range(repeat):
            if cls == "corner":
                specs.append(("corner", seed))
                seed += 10
                continue
            for family in families:
                for h in heads:
                    specs.append((cls, family, h, seed))
                    seed += 1
    return specs


def iter_cases(specs: List[tuple], b: int, s_q: int, quiet: bool = False):
    """Generate the cases one at a time (a case holds the caches of its batch: keeping all of them alive costs more than 20 GB of GPU memory)"""
    for spec in specs:
        with upstream_defaults():
            if spec[0] == "corner":
                cs = make_corner_cases(spec[1], b, s_q)
            else:
                cls, family, h, seed = spec
                t0 = time.time()
                c = make_case(cls, family, h, seed, b, s_q)
                c.seed = seed
                c.gap_mean, c.gap_max = exp_gap_stats(c.t)
                if not quiet:
                    print(f"  data {cls:8s} {family:10s} h={h:3d} seed={c.seed}: measured logit*sm_scale std {c.sigma_measured:.3f} nat, block exponent gap mean {c.gap_mean:.1f} max {c.gap_max:.0f} ({time.time() - t0:.1f} s)", flush=True)
                cs = [c]
        for c in cs:
            yield c


def build_cases(classes: List[str], families: List[str], heads: List[int], b: int, s_q: int, repeat: int = 1, quiet: bool = False) -> List[Case]:
    return list(iter_cases(case_specs(classes, families, heads, repeat), b, s_q, quiet))


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
        ("mxfp4", "MXFP4 kernel, fp8x2 tier (Q x4 mxfp4 with the kernel rule, P x2, V headroom -6, B32)",
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
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return ok


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
    ap.add_argument("--selftest", action="store_true", help="Check the emulator against an independent slow implementation")
    ap.add_argument("--check-exact", action="store_true", help="Run only the exact combination; exit 1 if it fails the upstream tolerances anywhere")
    args = ap.parse_args()

    combos = build_combos()
    if args.selftest:
        sys.exit(0 if selftest_framework() else 1)
    if args.list:
        for name, desc, mode, fams in combos:
            print(f"{name:22s} {','.join(fams):40s} {desc}")
        return
    pats = args.combos.split(",")
    if args.check_exact:
        pats = ["exact"]
    combos = [x for x in combos if any(fnmatch.fnmatch(x[0], p) for p in pats)]
    classes = [x for x in args.classes.split(",") if x]
    specs = case_specs(classes, args.families.split(","), [int(x) for x in args.heads.split(",")], args.repeat)
    total = sum(NUM_CORNER_CASES if sp[0] == "corner" else 1 for sp in specs)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    rows = run(iter_cases(specs, args.b, args.s_q), combos, total=total)
    print(f"{len(rows)} runs in {time.time() - t0:.0f} s; peak GPU memory allocated {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, reserved {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    if args.md:
        with open(args.md, "w") as f:
            f.write(summarize(rows, combos, classes))
    if args.check_exact:
        bad = [r for r in rows if not (r["out_ok"] and r["lse_ok"])]
        print(f"exact mode: {len(rows) - len(bad)}/{len(rows)} cases pass the upstream tolerances")
        for r in bad:
            print("  FAIL", {k: r[k] for k in ("cls", "family", "h", "cos_diff", "max_abs", "lse_max_abs", "lse_max_rel", "out_ok", "lse_ok")})
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
