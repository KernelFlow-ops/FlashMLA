"""
Bit-exact torch emulation of the arithmetic of the sm120 (RTX 5090) precision modes.

The decode kernel for sm120 can run QK and PV on the block-scaled tensor-core instructions
(mxf8f6f4 1X: e4m3 / e2m1 data + ue8m0 scale per 32 k, mxf4nvf4 4X: e2m1 data + ue4m3 scale per 16 k,
mxf4 2X: e2m1 + ue8m0 per 32 k). That requires quantizing Q and P, and re-expressing V on a common
per-column scale ("rebase"). This module reproduces that arithmetic so that the error of every
combination of choices can be measured against the upstream reference before any kernel is written.

Part 1 (this part): quantization primitives with the semantics of the PTX conversions
  - e4m3  : cvt.rn.satfinite.e4m3x2.f32   (round to nearest even, subnormals kept, saturate to +-448)
  - e2m1  : cvt.rn.satfinite.e2m1x2.f32   (round to nearest even, saturate to +-6)
  - ue8m0 : power of two scale, exponent = ceil(log2(x)) (cvt.rp.satfinite.ue8m0x2.f32)
  - ue4m3 : unsigned e4m3 scale (same magnitudes as e4m3)
"""
import dataclasses
import math
import re
from typing import List, Optional, Tuple

import torch

import quant

E4M3_MAX = 448.0
E2M1_MAX = 6.0
E4M3_MIN_EXP = -6      # exponent of the smallest normal e4m3 (2^-6); subnormal step is 2^-9
E2M1_MIN_EXP = 0       # exponent of the smallest normal e2m1 (1.0); subnormal step is 0.5
UE8M0_MIN_EXP = -127
UE8M0_MAX_EXP = 127


# ----------------------------------------------------------------------------------------------
# Power of two, exponent extraction
# ----------------------------------------------------------------------------------------------
def pow2f(e: torch.Tensor) -> torch.Tensor:
    """2^e as float32 for integer e, built from the bit pattern (exact, including the float32 subnormals 2^-149 .. 2^-127)."""
    e = e.to(torch.int32).clamp(-149, 127)
    normal = ((e + 127) << 23).view(torch.float32)
    sub = (torch.ones_like(e) << (e + 149).clamp(0, 31)).view(torch.float32)    # subnormal: bit (e + 149) of the mantissa
    return torch.where(e >= -126, normal, sub)


def floor_log2(a: torch.Tensor) -> torch.Tensor:
    """floor(log2(a)) as int32 for a > 0 (exact, from the exponent field). For a == 0 returns -1 (callers clamp)."""
    _, e = torch.frexp(a)
    return e.to(torch.int32) - 1


def ceil_log2(a: torch.Tensor) -> torch.Tensor:
    """ceil(log2(a)) as int32 for a > 0, exact: a = m * 2^e with m in [0.5, 1); ceil(log2(a)) = e - 1 if m == 0.5 else e."""
    m, e = torch.frexp(a)
    return e.to(torch.int32) - (m == 0.5).to(torch.int32)


# ----------------------------------------------------------------------------------------------
# Generic small float rounding
# ----------------------------------------------------------------------------------------------
def _round_to_fp(x: torch.Tensor, mbits: int, min_exp: int, max_val: float, rz: bool = False) -> torch.Tensor:
    """
    Round x to the nearest value of a sign / exponent / `mbits`-bit-mantissa float format whose smallest normal exponent is `min_exp`
    (subnormals have the grid step of the smallest binade), saturating at +-max_val. Round to nearest, ties to even (rz=True: toward zero).
    NaN propagates. Works for float32 and float64 inputs; the computation is exact (a power-of-two scaling, an integer rounding, a scaling back)
    """
    a = x.abs()
    e = torch.clamp(floor_log2(a), min=min_exp)
    step = pow2f(e - mbits).to(a.dtype)
    t = a / step                      # exact: step is a power of two
    t = torch.floor(t) if rz else torch.round(t)    # torch.round: half to even
    q = torch.clamp(t * step, max=max_val)
    return torch.copysign(q, x)


def e4m3_rn(x: torch.Tensor) -> torch.Tensor:
    """Round to the nearest e4m3 value (RN-even, satfinite). Returns the value in the dtype of x (float32 or float64)"""
    return _round_to_fp(x, 3, E4M3_MIN_EXP, E4M3_MAX)


def e4m3_rz(x: torch.Tensor) -> torch.Tensor:
    """Round toward zero to e4m3 (saturating). Only used for the 'what if the rebase truncates' ablation"""
    return _round_to_fp(x, 3, E4M3_MIN_EXP, E4M3_MAX, rz=True)


def e2m1_rn(x: torch.Tensor) -> torch.Tensor:
    """Round to the nearest e2m1 value (RN-even, satfinite to +-6)"""
    return _round_to_fp(x, 1, E2M1_MIN_EXP, E2M1_MAX)


def bf16_rn(x: torch.Tensor) -> torch.Tensor:
    """Round to bf16 (RN-even) and return as float32"""
    return x.to(torch.bfloat16).to(torch.float32)


def ue4m3_rn(x: torch.Tensor) -> torch.Tensor:
    """Round a non-negative scale to the nearest ue4m3 value (the magnitudes of e4m3), clamped to [2^-9, 448] like quant.py does for the FP4 cache scale"""
    return e4m3_rn(torch.clamp(x, 2.0 ** -9, E4M3_MAX))


# ----------------------------------------------------------------------------------------------
# ue8m0 scales
# ----------------------------------------------------------------------------------------------
def ue8m0_exp(scale_inv: torch.Tensor, floor_val: Optional[float] = 1e-4) -> torch.Tensor:
    """
    Exponent (int32) of the ue8m0 scale 2^ceil(log2(scale_inv)); the scale is the smallest power of two >= scale_inv.
    floor_val is the clamp of the cache writer (quant._cast_scale_inv_to_ue8m0 clamps scale_inv to >= 1e-4, i.e. a minimum scale of 2^-13); None: no clamp
    (used for the P scale, where tiny values are legitimate; amax == 0 then gives the exponent 0)
    """
    if floor_val is not None:
        e = ceil_log2(torch.clamp_min(scale_inv, floor_val))
    else:
        pos = scale_inv > 0
        e = ceil_log2(torch.where(pos, scale_inv, torch.ones_like(scale_inv)))
        e = torch.where(pos, e, torch.zeros_like(e))
    return torch.clamp(e, UE8M0_MIN_EXP, UE8M0_MAX_EXP)


def ue8m0_byte_to_exp(b: torch.Tensor) -> torch.Tensor:
    """The exponent of a stored ue8m0 byte (bias 127). 0xFF is NaN in e8m0; it maps to exponent 128 here (callers never use NaN-scale rows)"""
    return b.to(torch.int32) - 127

# ----------------------------------------------------------------------------------------------
# Block-scaled operands:  value = sum_i vals_i * sf_i
# ----------------------------------------------------------------------------------------------
@dataclasses.dataclass
class Terms:
    """
    A tensor expressed as the sum of MMA operands. Every term is data of the MMA input type (e4m3 / e2m1 values, or bf16 values for the exact
    path) times a block scale per `g` consecutive elements of the last dimension (g = 32: ue8m0, g = 16: ue4m3). A term is one MMA instruction stream
    """
    vals: List[torch.Tensor]    # each [..., 512] float32
    sf: List[torch.Tensor]      # each [..., 512 // g] float32 (a power of two for ue8m0, an e4m3 value for ue4m3)
    g: int

    def dense(self) -> torch.Tensor:
        """The represented tensor in float64"""
        out = 0
        for v, s in zip(self.vals, self.sf):
            out = out + v.double() * s.double().repeat_interleave(self.g, dim=-1)
        return out


_QMODE_RE = re.compile(r"(mxfp8|nvfp4|mxfp4)x([1-8])(?:p([0-9]+|a))?")


def parse_q_mode(name: str) -> Tuple[str, int]:
    """
    'bf16' -> ('bf16', 1); 'mxfp8x2' -> ('mxfp8', 2); also nvfp4x{1..8}, mxfp4x{1..8}. A suffix 'p<k>' (e.g. 'nvfp4x6p8') means Q is multiplied by 2^k before it is quantized
    and the scales are divided by 2^k afterwards (a power-of-two prescale, folded into the logit scale on the device); 'pa' picks k per row as the largest value for which
    amax * 2^k / 6 still fits the ue4m3 range (<= 448); it is ignored here, see quantize_q
    """
    if name == "bf16":
        return "bf16", 1
    m = _QMODE_RE.fullmatch(name)
    assert m is not None, f"unknown Q mode {name}"
    return m.group(1), int(m.group(2))


Q_RECIP_448 = float(torch.tensor(1.0 / 448.0, dtype=torch.float32))       # the fp32 constant the P2 kernel multiplies amax with


def quantize_q(q: torch.Tensor, name: str, lo_shift: int = 4, kernel_rule: bool = False) -> Terms:
    """
    Q [..., 512] (bf16 values, given as float32) -> operand terms of the A matrix.
      bf16   : Q as is
      mxfp8  : per (row, 32 dims) ue8m0 scale 2^ceil(log2(amax / 448)) (the cache rule), e4m3 data. Term i > 0: the residual divided by 2^(e - i * lo_shift),
               i.e. the scale of the lo term is 2^-lo_shift times the scale of the previous term
      nvfp4  : per (row, 16 dims) ue4m3 scale round_e4m3(amax / 6), e2m1 data; every further term quantizes the residual with its own scale
      mxfp4  : per (row, 32 dims) ue8m0 scale 2^ceil(log2(amax / 6)), e2m1 data; further terms as for nvfp4
    kernel_rule (mxfp8 only): the scale rule of the P2 kernel (mxfp8.cu quantize_q): ue8m0 code = ceil-to-power-of-two(amax * fp32(1/448)) clamped to [2 * lo_shift, 250], i.e. the
    exponent to [2 * lo_shift - 127, 123] and no 1e-4 floor of the cache writer; the multiplication by the fp32 reciprocal differs from the division by 448 only when amax is within
    one fp32 ulp of 448 * 2^k
    """
    kind, n = parse_q_mode(name)
    lead = q.shape[:-1]
    if kind == "bf16":
        return Terms([q], [torch.ones(*lead, 16, dtype=torch.float32, device=q.device)], 32)
    m = _QMODE_RE.fullmatch(name)
    if m.group(3) is not None:      # prescale: quantize 2^k * Q and give the scales back divided by 2^k (exact; only moves the ue4m3 scale floor 2^-9 relative to Q)
        if m.group(3) == "a":
            amax = q.abs().amax(-1, keepdim=True)
            k = torch.floor(torch.log2(448.0 * E2M1_MAX / amax.clamp_min(1e-30))).clamp(0, 30)
        else:
            k = torch.full(lead + (1,), float(int(m.group(3))), dtype=torch.float32, device=q.device)
        t = quantize_q(q * torch.exp2(k), name[:name.rindex("p")], lo_shift)
        return Terms(t.vals, [sf * torch.exp2(-k) for sf in t.sf], t.g)
    vals, sfs = [], []
    if kind == "mxfp8":
        r = q.reshape(*lead, 16, 32)
        if kernel_rule:
            x = r.abs().amax(-1) * Q_RECIP_448
            e0 = torch.clamp(torch.where(x > 0, ue8m0_exp(x, floor_val=None), torch.full_like(ue8m0_exp(x, None), UE8M0_MIN_EXP)), 2 * lo_shift - 127, 250 - 127)
        else:
            e0 = ue8m0_exp(r.abs().amax(-1) / E4M3_MAX)
        for i in range(n):
            sf = pow2f(e0 - i * lo_shift)
            d = e4m3_rn(r / sf.unsqueeze(-1))
            vals.append(d.reshape(*lead, 512)), sfs.append(sf)
            r = r - d * sf.unsqueeze(-1)
        return Terms(vals, sfs, 32)
    if kind == "nvfp4":
        r = q.reshape(*lead, 32, 16)
        for i in range(n):
            sf = ue4m3_rn(r.abs().amax(-1) / E2M1_MAX)
            d = e2m1_rn(r / sf.unsqueeze(-1))
            vals.append(d.reshape(*lead, 512)), sfs.append(sf)
            r = r - d * sf.unsqueeze(-1)
        return Terms(vals, sfs, 16)
    r = q.reshape(*lead, 16, 32)       # mxfp4
    for i in range(n):
        sf = pow2f(ue8m0_exp(r.abs().amax(-1) / E2M1_MAX, floor_val=1e-4 if i == 0 else None))
        d = e2m1_rn(r / sf.unsqueeze(-1))
        vals.append(d.reshape(*lead, 512)), sfs.append(sf)
        r = r - d * sf.unsqueeze(-1)
    return Terms(vals, sfs, 32)


# ----------------------------------------------------------------------------------------------
# KV cache rows -> operands
# ----------------------------------------------------------------------------------------------
@dataclasses.dataclass
class ScopeData:
    """The tokens one query token attends to in one cache (main or extra), gathered; T is padded with invalid tokens to a multiple of 64"""
    layout: "quant.KVCacheLayout"
    valid: torch.Tensor     # [bs, T] bool: index != -1 and below topk_length. Invalid tokens are excluded from everything (logit = -inf, V = 0, no emax)
    raw: torch.Tensor       # [bs, T, bytes_per_token] uint8 cache bytes, zero for invalid tokens
    kdeq: torch.Tensor      # [bs, T, 512] bf16: the exact dequantization the reference uses, zero for invalid tokens
    _cache: dict = dataclasses.field(default_factory=dict)


def gather_scope(scope, b: int, s_q: int) -> ScopeData:
    idx = scope.indices_in_kvcache                      # [b, s_q, topk]
    topk = idx.size(-1)
    invalid = idx == -1
    if scope.topk_length is not None:
        invalid = invalid | (torch.arange(topk, device=idx.device).view(1, 1, topk) >= scope.topk_length.view(b, 1, 1))
    flat = torch.clamp_min(idx, 0).reshape(-1).long()
    layout = scope.kvcache_layout or quant.KVCacheLayout.V41_FP8Sparse
    nbytes = layout.get_bytes_per_token()
    raw = scope.blocked_k_quantized.view(torch.uint8).reshape(-1, nbytes).index_select(0, flat).view(b * s_q, topk, nbytes)
    kdeq = scope.blocked_k.reshape(-1, 512).index_select(0, flat).view(b * s_q, topk, 512)
    valid = ~invalid.reshape(b * s_q, topk)
    pad = (-topk) % 64
    valid = torch.nn.functional.pad(valid, (0, pad), value=False)
    raw = torch.nn.functional.pad(raw, (0, 0, 0, pad), value=0)
    kdeq = torch.nn.functional.pad(kdeq, (0, 0, 0, pad), value=0)
    raw = torch.where(valid.unsqueeze(-1), raw, torch.zeros_like(raw))             # NaN rows of the upstream tests / padding never reach any computation
    kdeq = torch.where(valid.unsqueeze(-1), kdeq, torch.zeros_like(kdeq))
    return ScopeData(layout, valid, raw, kdeq)


def _e2m1_from_packed(data: torch.Tensor) -> torch.Tensor:
    """[..., 256] uint8 (even element in the low nibble) -> [..., 512] float32 e2m1 values"""
    codes = torch.stack([data & 0xF, data >> 4], dim=-1).reshape(*data.shape[:-1], 512)
    return quant._dequantize_e2m1(codes)


def decode_native(sd: ScopeData):
    """Data values and scales as stored: (vals [bs, T, 512] f32, ue8m0 exponents [bs, T, 16] i32 or None, ue4m3 scales [bs, T, 32] f32 or None)"""
    if "native" in sd._cache:
        return sd._cache["native"]
    raw = sd.raw
    if sd.layout == quant.KVCacheLayout.V41_FP8Sparse:
        r = (raw[..., :512].contiguous().view(torch.float8_e4m3fn).float(), ue8m0_byte_to_exp(raw[..., 512:528]), None)
    elif sd.layout == quant.KVCacheLayout.V41_MXFP4Sparse:
        r = (_e2m1_from_packed(raw[..., :256]), ue8m0_byte_to_exp(raw[..., 256:272]), None)
    elif sd.layout == quant.KVCacheLayout.V41_FP4Sparse:
        r = (_e2m1_from_packed(raw[..., :256]), None, raw[..., 256:288].contiguous().view(torch.float8_e4m3fn).float())
    else:
        raise NotImplementedError(sd.layout)
    sd._cache["native"] = r
    return r


def eight_bit_terms(sd: ScopeData, form: str, lo_shift: int = 4) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    The tokens as MXFP8 B-operand terms [(data [bs, T, 512] f32 e4m3 values, ue8m0 exponents [bs, T, 16] i32)], value = sum data * 2^exp.
    V41_FP8Sparse and V41_MXFP4Sparse are already in this form (an e2m1 value is exactly an e4m3 value). V41_FP4Sparse (e2m1 x e4m3 scale per 16) is
    converted per token and 32 dims to e4m3 x 2^e with e = ceil(log2(amax / 448)): 'x1' = one term (second rounding), 'x2' = hi + lo with the
    lo scale 2^-lo_shift lower
    """
    key = ("8bit", form, lo_shift)
    if key in sd._cache:
        return sd._cache[key]
    vals, exps, sfv = decode_native(sd)
    if exps is not None:
        out = [(vals, exps)]
    else:
        assert form in ("x1", "x2"), form
        bs, T, _ = vals.shape
        v = (vals * sfv.repeat_interleave(16, dim=-1)).view(bs, T, 16, 32)       # exact in float32 (<= 6 significant bits)
        amax = v.abs().amax(-1)
        e = torch.where(amax > 0, ue8m0_exp(amax / E4M3_MAX, floor_val=None), torch.full_like(ue8m0_exp(amax, None), UE8M0_MIN_EXP))     # an all-zero group gets the smallest exponent (it must not raise a block max)
        out, r = [], v
        for i in range(1 if form == "x1" else 2):
            e_i = torch.clamp(e - i * lo_shift, min=UE8M0_MIN_EXP)
            sf = pow2f(e_i)
            d = e4m3_rn(r / sf.unsqueeze(-1))
            out.append((d.reshape(bs, T, 512), e_i))
            r = r - d * sf.unsqueeze(-1)
    sd._cache[key] = out
    return out


def rebase_v(codes: torch.Tensor, exps: torch.Tensor, valid: torch.Tensor, B: int, rz: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Re-express V = codes * 2^exps (per token and 32 dims) on one scale per (block of B tokens, 32 dims): emax = max exponent over the VALID tokens of the
    block; codes' = round_e4m3(codes * 2^(exp - emax)) (RN-even, or toward zero for rz); the MMA scale is 2^emax.
    Returns (codes' [bs, T, 512], SF_B [bs, T // 32, 512]) where SF_B is per 32-token k-block and per column (a B = 64 block uses the same scale for both k-blocks)
    """
    bs, T, _ = codes.shape
    nb = T // B
    NEG = -(1 << 20)
    e = exps.view(bs, nb, B, 16)
    vmask = valid.view(bs, nb, B, 1)
    emax = torch.where(vmask, e, torch.full_like(e, NEG)).amax(2)               # [bs, nb, 16]
    emax = torch.where(emax == NEG, torch.zeros_like(emax), emax)               # block without a valid token: irrelevant, V = 0
    shift = torch.where(vmask, e - emax.unsqueeze(2), torch.zeros_like(e))      # <= 0
    scaled = codes.view(bs, nb, B, 16, 32) * pow2f(shift).unsqueeze(-1)         # exact: a power of two
    cp = e4m3_rz(scaled) if rz else e4m3_rn(scaled)
    cp = torch.where(vmask.unsqueeze(-1), cp, torch.zeros_like(cp)).view(bs, T, 512)
    sfb = pow2f(emax).repeat_interleave(B // 32, dim=1).repeat_interleave(32, dim=-1)       # [bs, T // 32, 512]
    return cp, sfb


# ----------------------------------------------------------------------------------------------
# Modes
# ----------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class OperandMode:
    """How the tokens of one cache layout enter the MMAs"""
    q: str = "bf16"         # Q: bf16 | mxfp8x{1,2} | nvfp4x{1..4} | mxfp4x{1..4}
    k: str = "exact"        # K (QK B operand): exact (bf16, only with q = bf16) | native | k8x1 | k8x2 (fp4 cache converted to e4m3 + ue8m0)
    v: str = "exact"        # V (PV B operand): exact (bf16) | rebase (e4m3 on a common scale per block and 32 dims)
    v8: str = "x1"          # the 8-bit form V is built from when the cache is not MXFP8 already (V41_FP4Sparse): x1 | x2


EXACT_OPS = OperandMode()


@dataclasses.dataclass(frozen=True)
class Mode:
    fp8: OperandMode = EXACT_OPS        # tokens of V41_FP8Sparse caches
    fp4: OperandMode = EXACT_OPS        # tokens of V41_FP4Sparse caches
    mxfp4: OperandMode = EXACT_OPS      # tokens of V41_MXFP4Sparse caches
    q_lo_shift: int = 4                 # the scale of the second (third, ...) Q term is 2^-q_lo_shift of the previous one (mxfp8 only; 0 or 4)
    k_lo_shift: int = 4                 # same for the lo term of the k8x2 / v8 x2 conversion
    p: str = "bf16"                     # P operand: bf16 | e4m3x1 | e4m3x2 | f32 (unrounded, only to isolate other errors)
    p_sf: str = "const"                 # SF_A of P: const (global constant) | row (per row and 32 tokens from that block's max)
    p_lo_shift: int = 4                 # lo term scale of p-e4m3x2
    m: str = "lazy"                     # row max used for P: lazy (upstream: update only when a block exceeds it by > 6) | final (row max over all tokens)
    B: int = 64                         # tokens per block: lazy-max update granularity and V rebase granularity (32 or 64)
    sum: str = "exact"                  # softmax denominator: exact (sum of fp32 P) | quant (sum of the quantized P that the MMA uses)
    accum: str = "f64"                  # f64: exact sum rounded to fp32 once per logit / per block; f32seg: every 32-k MMA step rounded to fp32 and accumulated in fp32
    rebase_round: str = "rn"            # rn | rz: rounding when the rebase shifts a code into the subnormal range
    q_kernel_rule: bool = False         # Q scale rule of the P2 kernel (fp32 reciprocal of 448, exponent clamped to [2 * q_lo_shift - 127, 123], no 1e-4 floor); see quantize_q
    lolo: bool = False                  # also issue the lo x lo MMA when both operands have a lo term (default: skipped, 3 instead of 4 MMAs for x2 x x2)

    def ops(self, layout) -> OperandMode:
        return {quant.KVCacheLayout.V41_FP8Sparse: self.fp8, quant.KVCacheLayout.V41_FP4Sparse: self.fp4, quant.KVCacheLayout.V41_MXFP4Sparse: self.mxfp4}[layout]


def check_operand_mode(layout, om: OperandMode):
    L = quant.KVCacheLayout
    kind, _ = parse_q_mode(om.q)
    if kind == "bf16":
        assert om.k == "exact", f"{om}: q-bf16 needs k-exact"
    elif kind == "mxfp8":
        assert (om.k == "native" and layout in (L.V41_FP8Sparse, L.V41_MXFP4Sparse)) or (om.k in ("k8x1", "k8x2") and layout == L.V41_FP4Sparse), f"{layout}: {om}"
    elif kind == "nvfp4":
        assert layout == L.V41_FP4Sparse and om.k == "native", f"{layout}: {om}"
    else:
        assert layout == L.V41_MXFP4Sparse and om.k == "native", f"{layout}: {om}"
    assert om.v in ("exact", "rebase")
    assert om.v8 in ("x1", "x2")


def k_terms(sd: ScopeData, om: OperandMode, mode: Mode) -> Terms:
    """The B operand of QK for the tokens of one cache"""
    check_operand_mode(sd.layout, om)
    kind, _ = parse_q_mode(om.q)
    bs, T, _ = sd.kdeq.shape
    if kind == "bf16":
        return Terms([sd.kdeq.float()], [torch.ones(bs, T, 16, dtype=torch.float32, device=sd.kdeq.device)], 32)
    if kind == "mxfp8":
        form = "x1" if om.k == "native" else om.k[2:]
        terms = eight_bit_terms(sd, form, mode.k_lo_shift)
        return Terms([c for c, _ in terms], [pow2f(e) for _, e in terms], 32)
    vals, exps, sfv = decode_native(sd)
    if kind == "nvfp4":
        return Terms([vals], [sfv], 16)
    return Terms([vals], [pow2f(exps)], 32)


def v_terms(sd: ScopeData, om: OperandMode, mode: Mode) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """The B operand of PV: ([data [bs, T, 512]], [SF_B [bs, T // 32, 512]]) terms"""
    bs, T, _ = sd.kdeq.shape
    if om.v == "exact":
        return [sd.kdeq.float()], [torch.ones(bs, T // 32, 512, dtype=torch.float32, device=sd.kdeq.device)]
    terms = eight_bit_terms(sd, om.v8, mode.k_lo_shift)
    vals, sfbs = [], []
    for codes, exps in terms:
        c, s = rebase_v(codes, exps, sd.valid, mode.B, rz=(mode.rebase_round == "rz"))
        vals.append(c), sfbs.append(s)
    return vals, sfbs


# ----------------------------------------------------------------------------------------------
# QK
# ----------------------------------------------------------------------------------------------
def _pair_partials(qv, qsf, kv, ksf, g: int) -> torch.Tensor:
    """sum over each scale group of qv * kv, times the two block scales: [c, G, h, T] float64 (exact products and sums)"""
    c, h, _ = qv.shape
    T = kv.size(1)
    G = 512 // g
    a = qv.view(c, h, G, g).permute(0, 2, 1, 3).double()
    b = kv.view(c, T, G, g).permute(0, 2, 3, 1).double()
    part = torch.matmul(a, b)
    return part * qsf.permute(0, 2, 1).unsqueeze(-1).double() * ksf.permute(0, 2, 1).unsqueeze(2).double()


def qk_logits(qt: Terms, kt: Terms, accum: str, budget: int = 1 << 30, lolo: bool = False) -> torch.Tensor:
    """Raw logits q.k [bs, h, T] float32 of all MMA term pairs (lo x lo pairs are not issued unless lolo)"""
    assert qt.g == kt.g
    g = qt.g
    G = 512 // g
    bs, h, _ = qt.vals[0].shape
    T = kt.vals[0].shape[1]
    pairs = [(i, j) for i in range(len(qt.vals)) for j in range(len(kt.vals)) if lolo or not (i > 0 and j > 0)]
    chunk = max(1, int(budget // (G * h * T * 8 * (len(pairs) + 1))))
    out = torch.empty(bs, h, T, dtype=torch.float32, device=qt.vals[0].device)
    for c0 in range(0, bs, chunk):
        sl = slice(c0, min(bs, c0 + chunk))
        parts = [_pair_partials(qt.vals[i][sl], qt.sf[i][sl], kt.vals[j][sl], kt.sf[j][sl], g) for i, j in pairs]       # each [c, G, h, T]
        if accum == "f64":
            out[sl] = sum(p.sum(1) for p in parts).float()
        else:
            acc = torch.zeros(parts[0].shape[0], h, T, dtype=torch.float32, device=out.device)
            gps = 32 // g       # groups per 32-element MMA k step
            for s in range(16):
                for p in parts:
                    acc = acc + p[:, s * gps:(s + 1) * gps].sum(1).float()
            out[sl] = acc
    return out


# ----------------------------------------------------------------------------------------------
# Softmax, P quantization, PV
# ----------------------------------------------------------------------------------------------
LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
LAZY_THRESHOLD = 6.0
MAX_INIT_VAL = -1e30


def quantize_p(P: torch.Tensor, mode: Mode) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    P [bs, h, N] fp32 (>= 0) -> A-operand terms of PV: ([data [bs, h, N]], [SF_A [bs, h, N // 32]]) with the value sum data * SF_A (per row and 32 tokens)
    """
    bs, h, N = P.shape
    nkb = N // 32
    if mode.p == "bf16":
        return [bf16_rn(P)], [torch.ones(bs, h, nkb, dtype=torch.float32, device=P.device)]
    if mode.p == "f32":         # P left in float32 (not an MMA input type; used to isolate the error of V)
        return [P], [torch.ones(bs, h, nkb, dtype=torch.float32, device=P.device)]
    n = {"e4m3x1": 1, "e4m3x2": 2}[mode.p]
    if mode.p_sf == "const":
        pmax = 1.0 if mode.m == "final" else 2.0 ** LAZY_THRESHOLD       # largest P the softmax can produce
        e = torch.full((bs, h, nkb), math.ceil(math.log2(pmax / E4M3_MAX)), dtype=torch.int32, device=P.device)
    else:
        e = ue8m0_exp(P.view(bs, h, nkb, 32).amax(-1) / E4M3_MAX, floor_val=None)
    vals, sfs = [], []
    r = P.view(bs, h, nkb, 32)
    for i in range(n):
        sf = pow2f(e - i * mode.p_lo_shift)
        d = e4m3_rn(r / sf.unsqueeze(-1))
        vals.append(d.view(bs, h, N)), sfs.append(sf)
        r = r - d * sf.unsqueeze(-1)
    return vals, sfs


def softmax_pv(logits: torch.Tensor, valid: torch.Tensor, vv: List[torch.Tensor], vsf: List[torch.Tensor], sm_scale: float,
               attn_sink: Optional[torch.Tensor], mode: Mode, budget: int = 1 << 29) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    logits [bs, h, N] raw q.k float32 (N a multiple of 64; blocks of mode.B tokens never straddle two caches), valid [bs, N],
    V terms vv [bs, N, 512] / vsf [bs, N // 32, 512]. Returns (out [bs, h, 512] bf16, lse [bs, h] float32, natural log)
    """
    bs, h, N = logits.shape
    B = mode.B
    nblk, nkb = N // B, N // 32
    dev = logits.device
    scale = float(torch.tensor(sm_scale * LOG2E, dtype=torch.float32))                        # sm_scale_div_log2 as a float32
    validh = valid.unsqueeze(1)
    l2 = torch.where(validh, logits * scale, torch.full_like(logits, -math.inf))              # log2-domain logits
    bmax = l2.view(bs, h, nblk, B).amax(-1)                                                    # [bs, h, nblk]

    # row max in effect for every block, and the factor applied to the accumulators before the block is added
    m_blk = torch.empty(bs, h, nblk, dtype=torch.float32, device=dev)
    s_blk = torch.ones(bs, h, nblk, dtype=torch.float32, device=dev)
    if mode.m == "final":
        rmax = bmax.amax(-1)
        m_blk[:] = torch.where(torch.isinf(rmax), torch.zeros_like(rmax), rmax).unsqueeze(-1)
    else:
        m = torch.full((bs, h), MAX_INIT_VAL, dtype=torch.float32, device=dev)
        for j in range(nblk):
            upd = (bmax[..., j] - m) > LAZY_THRESHOLD
            m_new = torch.where(upd, bmax[..., j], m)
            s_blk[..., j] = torch.where(upd, torch.exp2(m - m_new), torch.ones_like(m))
            m = m_new
            m_blk[..., j] = m
    m_tok = m_blk.repeat_interleave(B, dim=-1)                                                  # [bs, h, N]
    # P = 2^(logit * scale - m), one fma (single rounding), then exp2
    d = (logits.double() * scale - m_tok.double()).float()
    P = torch.where(validh, torch.exp2(d), torch.zeros_like(d))

    pv, psf = quantize_p(P, mode)
    if mode.sum == "exact":
        Peff = P
    else:
        Peff = sum(v.double() * s.double().repeat_interleave(32, dim=-1) for v, s in zip(pv, psf)).float()
    ls = Peff.double().view(bs, h, nblk, B).sum(-1).float()                                     # [bs, h, nblk] per-block row sums

    # PV: for every k block of 32 tokens, A (P terms, SF_A per row) x B (V terms, SF_B per column); lo x lo pairs are skipped unless mode.lolo
    pairs = [(i, j) for i in range(len(pv)) for j in range(len(vv)) if mode.lolo or not (i > 0 and j > 0)]
    O = torch.zeros(bs, h, 512, dtype=torch.float32, device=dev)
    l = torch.zeros(bs, h, dtype=torch.float32, device=dev)
    chunk = max(1, int(budget // (nkb * h * 512 * 8 * (len(pairs) + 1))))
    kb_per_blk = B // 32
    for c0 in range(0, bs, chunk):
        sl = slice(c0, min(bs, c0 + chunk))
        c = sl.stop - sl.start
        parts = []
        for i, j in pairs:
            a = pv[i][sl].view(c, h, nkb, 32).permute(0, 2, 1, 3).double()                      # [c, nkb, h, 32]
            b = vv[j][sl].view(c, nkb, 32, 512).double()
            inner = torch.matmul(a, b)                                                          # [c, nkb, h, 512] exact
            parts.append(inner * psf[i][sl].permute(0, 2, 1).unsqueeze(-1).double() * vsf[j][sl].unsqueeze(2).double())
        Oc = torch.zeros(c, h, 512, dtype=torch.float32, device=dev)
        lc = torch.zeros(c, h, dtype=torch.float32, device=dev)
        for jb in range(nblk):
            sj = s_blk[sl, :, jb]
            Oc = Oc * sj.unsqueeze(-1)
            lc = lc * sj + ls[sl, :, jb]
            kbs = slice(jb * kb_per_blk, (jb + 1) * kb_per_blk)
            if mode.accum == "f64":
                Oc = Oc + sum(p[:, kbs].sum(1) for p in parts).float()
            else:
                for kb in range(kbs.start, kbs.stop):
                    for p in parts:
                        Oc = Oc + p[:, kb].float()
        O[sl], l[sl] = Oc, lc

    m_last = m_blk[..., -1]
    sink = (attn_sink.float() * LOG2E) if attn_sink is not None else torch.full((h,), -math.inf, dtype=torch.float32, device=dev)
    denom = l + torch.exp2(sink.view(1, h) - m_last)
    o_scale = torch.where(l == 0, torch.zeros_like(l), 1.0 / denom)
    out = (O * o_scale.unsqueeze(-1)).to(torch.bfloat16)
    lse = torch.where(l == 0, torch.full_like(l, math.inf), m_last * LN2 + torch.log(l))
    return out, lse


def simulate_decode(p, t, mode: Mode) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Emulate the decode attention of a testcase (lib.TestcaseForDecode) in the given precision mode.
    Returns (out [b, s_q, h, 512] bf16, lse [b, h, s_q] float32), the same layout as ref.ref_sparse_attn_decode
    """
    b, s_q, h, d = t.q.shape
    bs = b * s_q
    q = t.q.reshape(bs, h, d).float()
    scopes = [t.kv_scope] + ([t.extra_kv_scope] if t.extra_kv_scope is not None else [])
    sds = [gather_scope(s, b, s_q) for s in scopes]
    q_cache = {}
    logits, valids, v_lists = [], [], []
    for sd in sds:
        om = mode.ops(sd.layout)
        if om.q not in q_cache:
            q_cache[om.q] = quantize_q(q, om.q, mode.q_lo_shift, mode.q_kernel_rule)
        logits.append(qk_logits(q_cache[om.q], k_terms(sd, om, mode), mode.accum, lolo=mode.lolo))
        valids.append(sd.valid)
        v_lists.append(v_terms(sd, om, mode))
    nv = max(len(v[0]) for v in v_lists)
    vv, vsf = [], []
    for j in range(nv):
        vv.append(torch.cat([(v[0][j] if j < len(v[0]) else torch.zeros_like(v[0][0])) for v in v_lists], dim=1))
        vsf.append(torch.cat([(v[1][j] if j < len(v[1]) else torch.ones_like(v[1][0])) for v in v_lists], dim=1))
    out, lse = softmax_pv(torch.cat(logits, dim=2), torch.cat(valids, dim=1), vv, vsf, t.sm_scale, t.attn_sink, mode)
    return out.view(b, s_q, h, 512), lse.view(b, s_q, h).transpose(1, 2).contiguous()


# ----------------------------------------------------------------------------------------------
# Self test of the primitives (acceptance 1): bit-exact against torch / quant.py, and against an independent table based reference
# ----------------------------------------------------------------------------------------------
def _all_f32_bits(lo: int, hi: int, device) -> torch.Tensor:
    """float32 tensor whose bit patterns are lo .. hi-1 (as unsigned 32-bit numbers)"""
    v = torch.arange(lo, hi, dtype=torch.int64, device=device)
    v = torch.where(v >= (1 << 31), v - (1 << 32), v).to(torch.int32)
    return v.view(torch.float32)


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Elementwise: bit-identical (both NaN counts as identical); distinguishes +0 and -0"""
    return (a.view(torch.int32) == b.view(torch.int32)) | (torch.isnan(a) & torch.isnan(b))


def _small_float_table(mbits: int, bias: int, max_code: int) -> List[float]:
    """Non-negative values of the codes 0 .. max_code of a sign-less small float (mantissa `mbits` bits, exponent bias `bias`), computed from the definition"""
    vals = []
    for c in range(max_code + 1):
        e, m = c >> mbits, c & ((1 << mbits) - 1)
        vals.append(m * 2.0 ** (1 - bias - mbits) if e == 0 else (1 + m / (1 << mbits)) * 2.0 ** (e - bias))
    return vals


def _table_round(x: float, table: List[float]) -> float:
    """Independent reference: nearest table value of |x| (float64 arithmetic), ties to the even code, saturating; sign restored"""
    a = abs(x)
    if a >= table[-1]:
        return math.copysign(table[-1], x)
    best = min(range(len(table)), key=lambda i: (abs(table[i] - a), i & 1))
    return math.copysign(table[best], x)


def selftest_primitives(exhaustive: bool = True, device: str = "cuda") -> bool:
    import numpy as np
    import quant

    ok = True

    def report(name: str, passed: bool, detail: str = ""):
        nonlocal ok
        ok &= bool(passed)
        print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}")

    # ---- tables from the definitions (independent of torch conversions)
    t_e4m3 = _small_float_table(3, 7, 0x7E)      # 0x7F is NaN
    t_e2m1 = _small_float_table(1, 1, 7)
    assert t_e4m3[-1] == 448.0 and t_e2m1 == [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    assert torch.equal(torch.tensor(t_e4m3).to(torch.float8_e4m3fn).float(), torch.tensor(t_e4m3).float()), "table is not e4m3"

    # ---- midpoints (exact ties), their float32 neighbours: mine == torch == independent reference
    def tie_probe(table: List[float]) -> torch.Tensor:
        xs = []
        for lo, hi in zip(table[:-1], table[1:]):
            mid = np.float32((lo + hi) / 2)
            assert float(mid) == (lo + hi) / 2, "midpoint is not exact in float32"
            xs += [mid, np.nextafter(mid, np.float32(np.inf)), np.nextafter(mid, np.float32(-np.inf))]
        top = np.float32(table[-1])
        xs += [top, np.nextafter(top, np.float32(np.inf)), top * np.float32(1.0625), np.float32(1e30), np.float32(np.inf)]
        xs = np.array(xs, dtype=np.float32)
        return torch.from_numpy(np.concatenate([xs, -xs, np.zeros(1, np.float32), -np.zeros(1, np.float32)])).to(device)

    probe = tie_probe(t_e4m3)
    expect = torch.tensor([_table_round(float(v), t_e4m3) for v in probe.cpu()], dtype=torch.float32, device=device)
    torch_ref = probe.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float()
    report("e4m3 ties / neighbours: mine == independent table", _same_bits(e4m3_rn(probe), expect).all(), f"({probe.numel()} probes)")
    report("e4m3 ties / neighbours: torch conversion == independent table", _same_bits(torch_ref, expect).all())

    probe = tie_probe(t_e2m1)
    expect = torch.tensor([_table_round(float(v), t_e2m1) for v in probe.cpu()], dtype=torch.float32, device=device)
    ref_q = quant._dequantize_e2m1(quant._quantize_to_e2m1(probe))
    report("e2m1 ties / neighbours: mine == independent table", _same_bits(e2m1_rn(probe), expect).all(), f"({probe.numel()} probes)")
    report("e2m1 ties / neighbours: quant._quantize_to_e2m1 == independent table", _same_bits(ref_q, expect).all())

    # ---- every float32 bit pattern (or a big random sample when exhaustive=False)
    chunk = 1 << 23
    n_chunks = (1 << 32) // chunk if exhaustive else 64
    rng = torch.Generator(device="cpu").manual_seed(0)
    bad4 = bad2 = n_cmp = 0
    for ci in range(n_chunks):
        if exhaustive:
            x = _all_f32_bits(ci * chunk, (ci + 1) * chunk, device)
        else:
            x = torch.randint(-(1 << 31), 1 << 31, (chunk,), generator=rng, dtype=torch.int32).to(device).view(torch.float32)
        ref4 = x.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float()   # clamp first: torch turns overflow into NaN, the PTX conversion saturates
        bad4 += int((~_same_bits(e4m3_rn(x), ref4)).sum())
        ref2 = quant._dequantize_e2m1(quant._quantize_to_e2m1(x))
        bad2 += int((~_same_bits(e2m1_rn(x), ref2) & ~torch.isnan(x)).sum())     # quant maps NaN to code 0 (the NaN lives in the scale), skip NaN
        n_cmp += x.numel()
        del x, ref4, ref2
    scope = "all 2^32 float32 bit patterns" if exhaustive else f"{n_cmp} random float32 bit patterns"
    report(f"e4m3_rn == torch float8_e4m3fn conversion (clamped to +-448 first), {scope}", bad4 == 0, f"mismatches={bad4}")
    report(f"e2m1_rn == quant._dequantize_e2m1(quant._quantize_to_e2m1), {scope}", bad2 == 0, f"mismatches={bad2}")

    # float64 input (exactly representable in float32) uses the same grid
    x64 = (torch.randn(1 << 20, device=device) * 10).double()     # exactly representable in float32
    report("e4m3_rn / e2m1_rn: float64 input == float32 input", torch.equal(e4m3_rn(x64), e4m3_rn(x64.float()).double()) and torch.equal(e2m1_rn(x64), e2m1_rn(x64.float()).double()))

    # ---- ue8m0
    # (1) every scale that can come from bf16 data (amax / 448 and amax / 6, amax any finite non-negative bf16): identical to quant._cast_scale_inv_to_ue8m0
    allbf = torch.arange(0, 0x7F80, dtype=torch.int32, device=device).to(torch.int16).view(torch.bfloat16).float()    # all finite non-negative bf16
    for name, den in (("amax/448", 448.0), ("amax/6", 6.0)):
        s = allbf / den
        report(f"ue8m0_exp == quant._cast_scale_inv_to_ue8m0 for every finite non-negative bf16 {name}",
               torch.equal(pow2f(ue8m0_exp(s)), quant._cast_scale_inv_to_ue8m0(s)), f"({s.numel()} values)")
    # (2) arbitrary float32: they can differ only where torch's float32 log2 rounds to an integer (value just above a power of two)
    g = torch.Generator(device="cpu").manual_seed(1)
    x = torch.exp2(torch.rand(1 << 24, generator=g) * 40 - 25).to(device)       # log-uniform over 2^-25 .. 2^15
    pw = torch.exp2(torch.arange(-24, 15, device=device).float()).repeat_interleave(2048)    # +-1024 ulp around each power of two
    off = torch.arange(-1024, 1024, device=device, dtype=torch.int32).repeat(39)
    x = torch.cat([x, (pw.view(torch.int32) + off).view(torch.float32)])
    mine, ref = pow2f(ue8m0_exp(x)), quant._cast_scale_inv_to_ue8m0(x)
    diff = mine != ref
    rel = x[diff] / ref[diff] - 1       # position relative to torch's power of two
    explained = bool(((mine[diff] == ref[diff] * 2) & (rel > 0) & (rel < 2e-6)).all())
    report("ue8m0_exp vs torch on arbitrary float32: only differences are torch's log2 rounding to the integer just above a power of two", explained,
           f"(differences {int(diff.sum())} of {x.numel()}, max relative distance above the power of two {float(rel.max()) if diff.any() else 0.0:.2e}; mine is the exact ceil, see next line)")
    # (3) mine is the exact ceil: 2^(e-1) < x <= 2^e (above the clamp)
    xx = x[x > 1e-4]
    e = ue8m0_exp(xx)
    report("ue8m0_exp exactness: 2^(e-1) < x <= 2^e", ((pow2f(e) >= xx) & (pow2f(e - 1) < xx)).all())
    # (4) round trip through the e8m0 storage type
    es = torch.arange(-127, 128, device=device, dtype=torch.int32)
    stored = pow2f(es).to(torch.float8_e8m0fnu)
    report("pow2f / ue8m0 byte round trip (exponent -127 .. 127)", torch.equal(stored.view(torch.uint8).to(torch.int32) - 127, es) and torch.equal(stored.float(), pow2f(es)))
    report("pow2f(e) == 2.0 ** e (float64 reference) for e in -149 .. 127",
           torch.equal(pow2f(torch.arange(-149, 128, device=device)).double(), torch.tensor([2.0 ** k for k in range(-149, 128)], dtype=torch.float64, device=device)))
    # (5) amax == 0
    report("ue8m0_exp(0) == -13 (cache clamp 1e-4), ue8m0_exp(0, None) == 0",
           int(ue8m0_exp(torch.zeros(1, device=device))) == -13 and int(ue8m0_exp(torch.zeros(1, device=device), None)) == 0)

    # ---- ue4m3 scale (NVFP4): the same as quant.py's FP4 cache writer: clamp(amax / 6, 2^-9, 448).to(e4m3)
    amax = torch.cat([torch.exp2(torch.rand(1 << 22, generator=g) * 30 - 18).to(device), allbf, torch.zeros(1, device=device)])
    ref = torch.clamp(amax / 6.0, 2.0 ** -9, 448.0).to(torch.float8_e4m3fn).float()
    report("ue4m3_rn(amax / 6) == quant.py FP4 cache scale", torch.equal(ue4m3_rn(amax / 6.0), ref), f"({amax.numel()} values)")

    # ---- truncating variant: toward zero, never farther from zero than RN, equal to RN when RN rounds down
    x = torch.randn(1 << 20, device=device) * 20
    rz, rn = e4m3_rz(x), e4m3_rn(x)
    report("e4m3_rz is toward zero", ((rz.abs() <= x.abs()) | (x.abs() > E4M3_MAX)).all() and ((rz == rn) | (rz.abs() < rn.abs())).all())

    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return ok

# sha256 of the quantized / dequantized bytes of the two pre-existing layouts on `_layout_test_input()`, recorded from quant.py at blackwell-5090@5c463c9
# (before V41_MXFP4Sparse was added). The new layout must not change them.
_OLD_LAYOUT_HASHES = {
    "V41_FP8Sparse": ("5dabbc028e5589481e60f91bde343fd11edd78e229d2132892c7cc98034f5672", "75e34d7e1e003e5499432913ac0e8fd5e2b6e189948d3971918e136e26874cc3"),
    "V41_FP4Sparse": ("1c70ee7ac7a53cd77dcdc959a0892b51b58f92f14c51a2628d20aa009501265f", "f56c494bf3381366e3d0a4bcf2db8d81f82ecd24010dea6c904b075dd9bf4ee7"),
}


def _layout_test_input() -> torch.Tensor:
    torch.manual_seed(1234)
    x = torch.randn(6, 64, 1, 512) * 0.1
    x[1] *= 30.0                    # large values
    x[2] *= 0.002                   # tiny values (scale clamps)
    x[3, :, :, ::7] *= 12.0         # spiky groups
    x[4, 5] = 0.0                   # an all-zero row
    x[0, 3] = float("nan")          # NaN row (unused-KV rows of the upstream tests)
    x[5, 9, 0, 40] = float("nan")   # NaN inside one tile
    return x.clamp(-1e3, 1e3).to(torch.bfloat16)


def selftest_layouts() -> bool:
    """Acceptance 3: old layouts unchanged byte for byte, V41_MXFP4Sparse is self consistent and works through the upstream testcase generator"""
    import hashlib
    import quant

    ok = True

    def report(name: str, passed: bool, detail: str = ""):
        nonlocal ok
        ok &= bool(passed)
        print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}")

    def sha(t: torch.Tensor) -> str:
        return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()

    x = _layout_test_input()
    for name, (hq, hd) in _OLD_LAYOUT_HASHES.items():
        layout = quant.KVCacheLayout[name]
        q = quant.quantize_k_cache(x, layout)
        d = quant.dequantize_k_cache(q, layout)
        report(f"{name}: quantized and dequantized bytes unchanged", sha(q) == hq and sha(d) == hd)

    L = quant.KVCacheLayout.V41_MXFP4Sparse
    report("V41_MXFP4Sparse: 272 B/token, meta (512, 448, 64, 32, 16)", L.get_bytes_per_token() == 272 and L.get_meta() == (512, 448, 64, 32, 16))
    q = quant.quantize_k_cache(x, L)
    d = quant.dequantize_k_cache(q, L)
    report("V41_MXFP4Sparse: shapes / dtypes", q.shape == (6, 64, 1, 272) and q.dtype == torch.float8_e4m3fn and d.shape == (6, 64, 1, 512) and d.dtype == torch.bfloat16)

    # independent recomputation of the format from the definition, row by row group
    xf = x.float().view(6, 64, 16, 32)
    amax = xf.abs().amax(dim=-1)                              # NaN rows give NaN here; handled below
    exp = ue8m0_exp(torch.nan_to_num(amax, nan=1.0) / 6.0)    # the cache clamps the scale at 2^-13
    scale = pow2f(exp)
    stored = q.view(6, 64, 272)[..., 256:].view(torch.float8_e8m0fnu)
    clean = ~torch.isnan(xf).any(dim=-1)                      # tiles without NaN
    report("V41_MXFP4Sparse: stored scale byte == ue8m0 of ceil(amax / 6)", torch.equal(stored.float()[clean], scale[clean]))
    expect = (e2m1_rn(xf / scale.unsqueeze(-1)) * scale.unsqueeze(-1)).to(torch.bfloat16).view(6, 64, 1, 512)
    cleanrow = clean.all(dim=-1)
    report("V41_MXFP4Sparse: dequantize(quantize(x)) == e2m1_rn(x / s) * s computed independently", torch.equal(d.view(6, 64, 512)[cleanrow].view(torch.int16), expect.view(6, 64, 512)[cleanrow].view(torch.int16)))
    err = (d.float().view(6, 64, 16, 32) - xf).abs()
    report("V41_MXFP4Sparse: |x - dequant(x)| <= scale in every clean tile (e2m1 half step at the top is 1)", bool((err[clean] <= scale.unsqueeze(-1).expand_as(err)[clean]).all()))
    # nibble order: even element in the low nibble
    data = q.view(6, 64, 272)[..., :256].view(torch.uint8)
    codes = torch.empty(6, 64, 512, dtype=torch.uint8)
    codes[..., 0::2] = data & 0xF
    codes[..., 1::2] = data >> 4
    exp_codes = quant._quantize_to_e2m1(xf / scale.unsqueeze(-1)).view(6, 64, 512)
    report("V41_MXFP4Sparse: nibble order (even element in the low nibble)", torch.equal(codes[cleanrow], exp_codes[cleanrow]))
    report("V41_MXFP4Sparse: NaN tile -> NaN scale -> NaN dequantized values; zero row -> zeros",
           bool(torch.isnan(d[0, 3]).all()) and bool(torch.isnan(d[5, 9, 0, 32:64]).all()) and bool((d[4, 5] == 0).all()) and not bool(torch.isnan(d[5, 9, 0, :32]).any()))

    # through the upstream testcase generator (main and extra caches in the new layout)
    import lib
    prev_dtype, prev_dev = torch.get_default_dtype(), torch.get_default_device()
    try:
        torch.set_default_dtype(torch.bfloat16)
        torch.set_default_device("cuda")
        p = lib.RawTestParamForDecode(b=2, h_q=64, s_q=2, h_kv=1, s_kv=256, is_varlen=True, topk=64, have_topk_length=True, extra_s_k=256, extra_topk=64,
                                      block_size=64, extra_block_size=64, kvcache_layout=L, extra_kvcache_layout=L, seed=3).to_test_param()
        t = lib.generate_testcase_for_decode(p)
        got = t.kv_scope.blocked_k_quantized
        report("generate_testcase_for_decode works with V41_MXFP4Sparse as main and extra layout", got.shape[-1] == 272 and t.extra_kv_scope.blocked_k_quantized.shape[-1] == 272)
        report("the dequantized KV of the testcase is the dequantization of its cache bytes",
               torch.equal(torch.nan_to_num(t.kv_scope.blocked_k.float()), torch.nan_to_num(quant.dequantize_k_cache(got, L).float())))
    finally:
        torch.set_default_dtype(prev_dtype)
        torch.set_default_device(prev_dev)

    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return ok


if __name__ == "__main__":
    import argparse
    import sys
    ap = argparse.ArgumentParser(description="sm120 precision mode emulator")
    ap.add_argument("--selftest", action="store_true", help="Bit-exact check of the quantization primitives, then the KV cache layouts")
    ap.add_argument("--quick", action="store_true", help="With --selftest: a random sample of float32 bit patterns instead of all 2^32")
    args = ap.parse_args()
    if args.selftest:
        r1 = selftest_primitives(exhaustive=not args.quick)
        r2 = selftest_layouts()
        sys.exit(0 if (r1 and r2) else 1)
    ap.print_help()
