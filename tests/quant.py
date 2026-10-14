import enum
from typing import Tuple

import torch

import kernelkit as kk

class KVCacheLayout(enum.Enum):
    V41_FP8Sparse = 0
    V41_FP4Sparse = 1
    V41_MXFP4Sparse = 2     # OCP MX-style fp4: e2m1 data + one ue8m0 (power of two) scale per 32 elements; 272 B/token. Read by the sm120 decode kernel only (csrc/cuda_kernels/sm120/decode/sparse/mxfp4.cu)

    def get_meta(self) -> Tuple[int, int, int, int, int]:
        # Return: (d, d_nope, d_rope, tile_size, num_tiles)
        return {
            KVCacheLayout.V41_FP8Sparse: (512, 448, 64, 32, 16),  # 14 NoPE + 2 RoPE tiles
            KVCacheLayout.V41_FP4Sparse: (512, 448, 64, 16, 32),  # 28 NoPE + 4 RoPE tiles, all fp4
            KVCacheLayout.V41_MXFP4Sparse: (512, 448, 64, 32, 16),  # 14 NoPE + 2 RoPE tiles, all fp4
        }[self]

    def get_bytes_per_token(self) -> int:
        d, d_nope, d_rope, tile_size, num_tiles = self.get_meta()
        return {
            KVCacheLayout.V41_FP8Sparse: d_nope + d_rope + num_tiles,
            KVCacheLayout.V41_FP4Sparse: d // 2 + num_tiles,
            KVCacheLayout.V41_MXFP4Sparse: d // 2 + num_tiles,
        }[self]

def _cast_scale_inv_to_ue8m0(scales_inv: torch.Tensor, out_dtype = torch.float32) -> torch.Tensor:
    return torch.pow(2, torch.clamp_min(scales_inv, 1e-4).log2().ceil()).to(out_dtype)    # This 1e-4 aligns with Tile Kernel's FP8_AMAX_MARGIN

_E2M1_MAGNITUDES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)  # Indexed by the 3 low bits of the code, bit 3 is the sign

def _quantize_to_e2m1(x: torch.Tensor) -> torch.Tensor:
    """
    Round to the nearest fp4_e2m1 value with the semantics of PTX `cvt.rn.satfinite.e2m1x2.f32` (ties to even, saturating to +-6)
    and return the 4-bit codes as uint8. NaN is mapped to code 0 (fp4 has no NaN; the caller keeps the NaN in the scale instead)
    """
    x = x.float()
    if x.numel() > (1 << 26):
        # The tie detection below broadcasts an elements x 7 intermediate; quantize big caches in chunks
        out = torch.empty(x.shape, dtype=torch.uint8, device=x.device)
        for i in range(0, x.numel(), 1 << 26):
            out.reshape(-1)[i:i + (1 << 26)] = _quantize_to_e2m1(x.reshape(-1)[i:i + (1 << 26)])
        return out
    mags = _E2M1_MAGNITUDES.to(x.device)
    sign = (torch.signbit(x)).to(torch.uint8) << 3
    a = torch.nan_to_num(x.abs(), nan=0.0, posinf=6.0).clamp_max(6.0)
    mids = (mags[:-1] + mags[1:]) / 2
    code = torch.bucketize(a, mids, right=True)
    on_tie = (a.unsqueeze(-1) == mids).any(dim=-1)
    tie_code = torch.bucketize(a, mids, right=False)    # On a tie: the lower of the two candidate codes
    code = torch.where(on_tie, tie_code + (tie_code & 1), code)
    return (sign | code.to(torch.uint8)).to(torch.uint8)

def _dequantize_e2m1(codes: torch.Tensor) -> torch.Tensor:
    mags = _E2M1_MAGNITUDES.to(codes.device)
    val = mags[(codes & 7).long()]
    return torch.where((codes & 8) != 0, -val, val)

def quantize_k_cache(
    input_k_cache: torch.Tensor,    # (num_blocks, block_size, h_k, d)
    kvcache_layout: KVCacheLayout,
) -> torch.Tensor:
    """
    Quantize the k-cache
    Each token stores its quantized raw data followed immediately by its scales.
    """
    d, d_nope, d_rope, tile_size, num_tiles = kvcache_layout.get_meta()
    assert input_k_cache.shape[-1] == d
    num_blocks, block_size, h_k, _ = input_k_cache.shape
    assert h_k == 1
    input_k_cache = input_k_cache.squeeze(2)    # [num_blocks, block_size, d]

    if kvcache_layout == KVCacheLayout.V41_FP8Sparse:
        # Token layout: [512 B fp8 data, 16 B ue8m0 scales].
        bytes_per_token = kvcache_layout.get_bytes_per_token()
        result = torch.empty((num_blocks, block_size, bytes_per_token), dtype=torch.float8_e4m3fn, device=input_k_cache.device)
        result_k_nope_rope_part = result[..., :d_nope+d_rope]
        result_k_scale_factor_view_as_dtype = torch.int8 if kk.is_on_ascend_platform() else torch.float8_e8m0fnu    # torch_npu has bug when copying from/to `torch.float8_e8m0fnu`
        result_k_scale_factor = result[..., d_nope+d_rope:].view(result_k_scale_factor_view_as_dtype)  # [num_blocks, block_size, 16]

        is_ascend = kk.is_on_ascend_platform()
        for tile_idx in range(0, 16):
            cur_scale_factors_inv = torch.abs(input_k_cache[..., tile_idx*tile_size:(tile_idx+1)*tile_size]).max(dim=-1).values.float() / 448.0
            cur_scale_factors_inv = _cast_scale_inv_to_ue8m0(cur_scale_factors_inv)
            if is_ascend:
                # torch_npu can not copy from float32 to float8_e8m0fnu while preserving the bit pattern,
                # so compute the e8m0 byte pattern (= bf16 bits >> 7) on CPU and write it as int8 bits.
                scale_bits = cur_scale_factors_inv.view(torch.int32) // 2**23
                result_k_scale_factor[:, :, tile_idx] = scale_bits.to(torch.int8)
            else:
                result_k_scale_factor[:, :, tile_idx] = cur_scale_factors_inv.to(torch.float8_e8m0fnu)
            cur_scale_factors_inv = cur_scale_factors_inv.view(num_blocks, block_size, 1)
            cur_quantized_nope = (input_k_cache[..., tile_idx*tile_size:(tile_idx+1)*tile_size].float() / cur_scale_factors_inv.float()).to(torch.float8_e4m3fn)
            result_k_nope_rope_part[:, :, tile_idx*tile_size:(tile_idx+1)*tile_size] = cur_quantized_nope

        result = result.view(num_blocks, block_size, 1, -1)
        return result

    elif kvcache_layout == KVCacheLayout.V41_FP4Sparse:
        # Token layout: [256 B fp4 data, 32 B e4m3 scales]. Element i of a row is in byte i//2, even
        # elements in the low nibble. The scale is amax / 6 (6 = max magnitude of e2m1) rounded to e4m3, without a per-tensor scale
        bytes_per_token = kvcache_layout.get_bytes_per_token()
        result = torch.empty((num_blocks, block_size, bytes_per_token), dtype=torch.float8_e4m3fn, device=input_k_cache.device)
        result_k_data = result[..., :d//2].view(torch.uint8)
        result_k_scale = result[..., d//2:]

        # Chunk along blocks: the fp32 intermediates below are 4x the size of the (bf16) cache, and a
        # whole-cache version would need several of them alive at once (e.g. 3 x 30.7GB for a 15GB cache)
        blocks_per_chunk = max(1, (1 << 26) // (block_size * d))
        for b0 in range(0, num_blocks, blocks_per_chunk):
            b1 = min(b0 + blocks_per_chunk, num_blocks)
            n = b1 - b0
            x = input_k_cache[b0: b1].float()
            amax = torch.nan_to_num(x.abs(), nan=float("inf")).view(n, block_size, num_tiles, tile_size).amax(dim=-1)   # A NaN element poisons the whole tile
            scale = torch.clamp(amax / 6.0, 2.0**-9, 448.0).to(torch.float8_e4m3fn)     # Clamp to the e4m3 range first: torch maps overflow to NaN
            if kk.is_on_ascend_platform():
                # torch_npu cannot fill FP8 tensors; write the E4M3 NaN encoding through its byte view.
                scale.view(torch.int8).masked_fill_(torch.isinf(amax), 0x7f)
            else:
                scale = torch.where(torch.isinf(amax), torch.full_like(scale, float("nan")), scale)
            normalized = x.view(n, block_size, num_tiles, tile_size) / scale.float().unsqueeze(-1)
            if input_k_cache.device.type == "npu":
                import torch_npu
                # Accept the native cast's FP32 rounding near E2M1 midpoints.
                result_k_data[b0: b1] = torch_npu.npu_dtype_cast(
                    normalized, torch_npu.float4_e2m1fn_x2
                ).view(n, block_size, d // 2)
            else:
                codes = _quantize_to_e2m1(normalized).view(n, block_size, d)
                result_k_data[b0: b1] = codes[..., 0::2] | (codes[..., 1::2] << 4)
            result_k_scale[b0: b1] = scale

        result = result.view(num_blocks, block_size, 1, -1)
        return result

    elif kvcache_layout == KVCacheLayout.V41_MXFP4Sparse:
        # Token layout: [256 B fp4 data, 16 B ue8m0 scales]. The data is packed like V41_FP4Sparse (element i of a row is in byte i//2, even
        # elements in the low nibble). One scale per 32 elements, a power of two: the smallest 2^e >= amax / 6 (6 = max magnitude of e2m1), i.e.
        # the same rule as V41_FP8Sparse with 6 instead of 448 (including its lower clamp of the scale)
        assert not kk.is_on_ascend_platform(), "V41_MXFP4Sparse is not supported on Ascend"
        bytes_per_token = kvcache_layout.get_bytes_per_token()
        result = torch.empty((num_blocks, block_size, bytes_per_token), dtype=torch.float8_e4m3fn, device=input_k_cache.device)
        result_k_data = result[..., :d//2].view(torch.uint8)
        result_k_scale = result[..., d//2:].view(torch.float8_e8m0fnu)

        blocks_per_chunk = max(1, (1 << 26) // (block_size * d))
        for b0 in range(0, num_blocks, blocks_per_chunk):
            b1 = min(b0 + blocks_per_chunk, num_blocks)
            n = b1 - b0
            x = input_k_cache[b0: b1].float()
            amax = torch.nan_to_num(x.abs(), nan=float("inf")).view(n, block_size, num_tiles, tile_size).amax(dim=-1)   # A NaN element poisons the whole tile
            scale = _cast_scale_inv_to_ue8m0(amax / 6.0)
            scale = torch.where(torch.isinf(amax), torch.full_like(scale, float("nan")), scale)
            normalized = x.view(n, block_size, num_tiles, tile_size) / scale.unsqueeze(-1)
            codes = _quantize_to_e2m1(normalized).view(n, block_size, d)
            result_k_data[b0: b1] = codes[..., 0::2] | (codes[..., 1::2] << 4)
            result_k_scale[b0: b1] = scale.to(torch.float8_e8m0fnu)

        result = result.view(num_blocks, block_size, 1, -1)
        return result

    else:
        raise NotImplementedError(f"Unsupported kvcache_layout: {kvcache_layout}")
    

def dequantize_k_cache(
    quant_k_cache: torch.Tensor,    # (num_blocks, block_size, 1, bytes_per_token)
    kvcache_layout: KVCacheLayout,
) -> torch.Tensor:
    """
    De-quantize the k-cache
    """
    d, d_nope, d_rope, tile_size, num_tiles = kvcache_layout.get_meta()
    num_blocks, block_size, h_k, _ = quant_k_cache.shape
    assert h_k == 1
    result = torch.empty((num_blocks, block_size, d), dtype=torch.bfloat16, device=quant_k_cache.device)

    if kvcache_layout == KVCacheLayout.V41_FP8Sparse:
        quant_k_cache = quant_k_cache.view(num_blocks, block_size, -1)
        input_nope_rope = quant_k_cache[..., :d_nope+d_rope]
        input_scale = quant_k_cache[..., d_nope+d_rope:].view(torch.float8_e8m0fnu)  # [num_blocks, block_size, 16]

        # Dequant NoPE (tiles 0-13, each tile size 32)
        is_ascend = kk.is_on_ascend_platform()
        for tile_idx in range(0, 16):
            cur_nope_rope = input_nope_rope[..., tile_idx*tile_size:(tile_idx+1)*tile_size].to(torch.bfloat16)
            if is_ascend:
                cur_scales = (input_scale[:, :, tile_idx].view(torch.int8).to(torch.int16) * 2**7).view(torch.bfloat16) # torch_npu can not cast float8_e8m0 to bf16 on NPU
            else:
                cur_scales = input_scale[:, :, tile_idx].to(torch.bfloat16)
            cur_scales = cur_scales.unsqueeze(-1)
            result[..., tile_idx*tile_size:(tile_idx+1)*tile_size] = cur_nope_rope * cur_scales
            
    elif kvcache_layout == KVCacheLayout.V41_FP4Sparse:
        quant_k_cache = quant_k_cache.view(num_blocks, block_size, -1)
        input_data = quant_k_cache[..., :d//2].view(torch.uint8)
        input_scale = quant_k_cache[..., d//2:].view(torch.float8_e4m3fn)

        # Chunk along blocks to bound intermediate memory for large caches.
        blocks_per_chunk = max(1, (1 << 26) // (block_size * d))
        for b0 in range(0, num_blocks, blocks_per_chunk):
            b1 = min(b0 + blocks_per_chunk, num_blocks)
            if quant_k_cache.device.type == "npu":
                import torch_npu
                # Packed FP4 cast needs contiguous input; raw page rows skip the scale region.
                values = torch_npu.npu_dtype_cast(
                    input_data[b0:b1].contiguous(), torch.bfloat16, input_dtype=torch_npu.float4_e2m1fn_x2
                ).view(b1 - b0, block_size, num_tiles, tile_size)
            else:
                codes = torch.empty((b1 - b0, block_size, d), dtype=torch.uint8, device=quant_k_cache.device)
                codes[..., 0::2] = input_data[b0:b1] & 0xF
                codes[..., 1::2] = input_data[b0:b1] >> 4
                values = _dequantize_e2m1(codes).view(b1 - b0, block_size, num_tiles, tile_size)
            # e2m1 x e4m3 has at most 2 + 4 significant bits, so the product is exact in bf16, as in the kernel
            result[b0:b1] = (values * input_scale[b0:b1].to(values.dtype).unsqueeze(-1)).view(b1 - b0, block_size, d).to(torch.bfloat16)

    elif kvcache_layout == KVCacheLayout.V41_MXFP4Sparse:
        assert not kk.is_on_ascend_platform(), "V41_MXFP4Sparse is not supported on Ascend"
        quant_k_cache = quant_k_cache.view(num_blocks, block_size, -1)
        input_data = quant_k_cache[..., :d//2].view(torch.uint8)
        input_scale = quant_k_cache[..., d//2:].view(torch.float8_e8m0fnu)  # [num_blocks, block_size, 16]

        blocks_per_chunk = max(1, (1 << 26) // (block_size * d))
        for b0 in range(0, num_blocks, blocks_per_chunk):
            b1 = min(b0 + blocks_per_chunk, num_blocks)
            codes = torch.empty((b1 - b0, block_size, d), dtype=torch.uint8, device=quant_k_cache.device)
            codes[..., 0::2] = input_data[b0:b1] & 0xF
            codes[..., 1::2] = input_data[b0:b1] >> 4
            values = _dequantize_e2m1(codes).view(b1 - b0, block_size, num_tiles, tile_size)
            # e2m1 x a power of two is exact in bf16
            result[b0:b1] = (values * input_scale[b0:b1].to(values.dtype).unsqueeze(-1)).view(b1 - b0, block_size, d).to(torch.bfloat16)

    else:
        raise NotImplementedError(f"Unsupported kvcache_layout: {kvcache_layout}")
    
    result = result.view(num_blocks, block_size, 1, d)
    return result


def abs_indices2indices_in_kvcache(
    abs_indices: torch.Tensor,  # [b, s_q, topk]
    block_table: torch.Tensor,  # [b, /]
    block_size: int,
) -> torch.Tensor:
    """
    Convert abs_indices (logical index, ranging from 0 to s_k-1) to index expected by the sparse attn kernel
    Equivalent to:
    
    b, s_q, topk = abs_indices.shape
    indices_in_kvcache = torch.empty_like(abs_indices)
    for i in range(b):
        cur_abs_indices = abs_indices[i, :, :].clone()  # [s_q, topk]
        invalid_mask = cur_abs_indices == -1
        cur_abs_indices[invalid_mask] = 0
        cur_indices_in_kvcache = block_table[i].index_select(0, cur_abs_indices.flatten()//block_size).view(s_q, topk)*block_size + cur_abs_indices%block_size
        cur_indices_in_kvcache[invalid_mask] = -1
        indices_in_kvcache[i] = cur_indices_in_kvcache
    return indices_in_kvcache

    """
    b, s_q, topk = abs_indices.shape
    _, max_blocks_per_seq = block_table.shape

    abs_indices = abs_indices.clone()
    invalid_mask = abs_indices == -1
    abs_indices[invalid_mask] = 0

    real_block_idxs = block_table.view(-1).index_select(0, (abs_indices//block_size + torch.arange(0, b).view(b, 1, 1)*max_blocks_per_seq).view(-1))
    indices_in_kvcache = real_block_idxs.view(b, s_q, topk)*block_size + abs_indices%block_size
    indices_in_kvcache[invalid_mask] = -1

    return indices_in_kvcache


# ----------------------------------------------------------------------------------------------
# Two-level NVFP4 cache (sm120 phase 6, G1). The bytes are exactly those of V41_FP4Sparse (256 B of e2m1 + 32 B of e4m3 scales, one per 16 dims, 288 B / token); the second level
# is a power of two G = 2^g kept outside the bytes (one integer per cache tensor, or per token for the storage comparison): value = e2m1 * s * G.
#   g = ceil(log2(amax / top)) with top = 2688 = 6 * 448 (1536 = 4 * 384 with Four Over Six: the scale amax / 4 may be rounded up and must stay <= 448), so that the largest
#   block scale s = amax_block / 6 / G fits e4m3. s = e4m3_rule(amax_block / G / 6) clamped to [2^-9, 448]: rule 'rn' = round to nearest (the V41_FP4Sparse writer: with g = 0 the
#   result is bit for bit that of quantize_k_cache), rule 'ceil' = the smallest e4m3 value with 6 * s >= amax_block / G (no code saturates).
#   Four Over Six (arXiv 2512.02010): every block is quantized with the scale mapped to 6 and mapped to 4 (amax / 4, codes 0 .. 4 only), the one with the smaller squared error is kept (ties: 6)
# ----------------------------------------------------------------------------------------------
FP4_2L_TOP = 2688.0
FP4_2L_TOP_46 = 1536.0


def fp4_2l_choose_g(amax: torch.Tensor, four_six: bool = False) -> torch.Tensor:
    """g = ceil(log2(amax / top)) as int32 (exact: from the float64 frexp of the quotient); 0 where amax is 0 (or not positive)"""
    a = amax.double() / (FP4_2L_TOP_46 if four_six else FP4_2L_TOP)
    pos = a > 0
    m, e = torch.frexp(torch.where(pos, a, torch.ones_like(a)))
    g = e.to(torch.int32) - (m == 0.5).to(torch.int32)
    return torch.where(pos, g, torch.zeros_like(g))


def _fp4_2l_pow2(g) -> torch.Tensor:
    """2^g as float32 (exact) for an int, or an int tensor"""
    if isinstance(g, torch.Tensor):
        return torch.ldexp(torch.ones(g.shape, dtype=torch.float32, device=g.device), g.to(torch.int32))
    return torch.tensor(2.0 ** int(g), dtype=torch.float32)


def _fp4_2l_try(x: torch.Tensor, amax: torch.Tensor, G: torch.Tensor, rule: str, qmax: float):
    """One quantization of the blocks x [..., ntile, 16] (float32) with the scale mapped to qmax (6 or 4): (codes uint8 [..., ntile, 16], scale float8_e4m3fn [..., ntile])"""
    a = amax / G                                                    # exact (G is a power of two)
    scale = torch.clamp(a / qmax, 2.0 ** -9, 448.0).to(torch.float8_e4m3fn)
    if rule == "ceil":
        s = scale.float()
        need = (s * qmax < a) & (s < 448.0)                         # the next e4m3 value above (codes of positive e4m3 values are ordered)
        scale = torch.where(need, (scale.view(torch.uint8) + 1).view(torch.float8_e4m3fn), scale)
    else:
        assert rule == "rn", rule
    scale = torch.where(torch.isinf(amax), torch.full_like(scale, float("nan")), scale)       # a NaN element poisons the whole block, like the V41_FP4Sparse writer
    normalized = x / (scale.float() * G).unsqueeze(-1)
    return _quantize_to_e2m1(normalized), scale


def quantize_fp4_2l_blocks(x: torch.Tensor, G: torch.Tensor, rule: str = "rn", four_six: bool = False):
    """
    x [..., ntile, 16] float32, G float32 broadcastable to [..., ntile] (a power of two) -> (codes uint8 [..., ntile, 16], scale float8_e4m3fn [..., ntile]); value = e2m1(code) * scale * G
    """
    amax = torch.nan_to_num(x.abs(), nan=float("inf")).amax(dim=-1)                # A NaN element poisons the whole block
    c6, s6 = _fp4_2l_try(x, amax, G, rule, 6.0)
    if not four_six:
        return c6, s6
    c4, s4 = _fp4_2l_try(x, amax, G, rule, 4.0)
    xd = x.double()
    e6 = (xd - _dequantize_e2m1(c6).double() * (s6.double() * G.double()).unsqueeze(-1)).pow(2).sum(-1)
    e4 = (xd - _dequantize_e2m1(c4).double() * (s4.double() * G.double()).unsqueeze(-1)).pow(2).sum(-1)
    pick4 = e4 < e6
    return torch.where(pick4.unsqueeze(-1), c4, c6), torch.where(pick4, s4, s6)


def quantize_k_cache_fp4_2l(
    input_k_cache: torch.Tensor,    # (num_blocks, block_size, 1, d)
    rule: str = "rn",
    four_six: bool = False,
    granularity: str = "tensor",    # 'tensor': one g for the whole cache (from the amax over all non-NaN elements); 'token': one g per token
    g=None,                         # tensor granularity only: use this integer instead of choosing it (g = 0, rule 'rn', no Four Over Six: the bytes of quantize_k_cache(.., V41_FP4Sparse))
):
    """
    Two-level NVFP4 quantization of a k-cache. Returns (bytes (num_blocks, block_size, 1, 288) float8_e4m3fn, laid out like quantize_k_cache(.., V41_FP4Sparse), g),
    g = a python int (tensor granularity) or an int32 tensor (num_blocks, block_size) (token granularity)
    """
    assert granularity in ("tensor", "token")
    layout = KVCacheLayout.V41_FP4Sparse
    d, d_nope, d_rope, tile_size, num_tiles = layout.get_meta()
    assert input_k_cache.shape[-1] == d
    num_blocks, block_size, h_k, _ = input_k_cache.shape
    assert h_k == 1
    x_all = input_k_cache.squeeze(2)
    blocks_per_chunk = max(1, (1 << 26) // (block_size * d))
    dev = input_k_cache.device
    if granularity == "tensor":
        if g is None:
            amax_t = 0.0
            for b0 in range(0, num_blocks, blocks_per_chunk):
                amax_t = max(amax_t, float(torch.nan_to_num(x_all[b0: b0 + blocks_per_chunk].float().abs(), nan=0.0, posinf=0.0).amax()))
            g = int(fp4_2l_choose_g(torch.tensor(amax_t, dtype=torch.float32), four_six))
        g = int(g)
        G_t = _fp4_2l_pow2(g).to(dev)
        g_out = g
    else:
        assert g is None
        g_out = torch.empty((num_blocks, block_size), dtype=torch.int32, device=dev)
    result = torch.empty((num_blocks, block_size, layout.get_bytes_per_token()), dtype=torch.float8_e4m3fn, device=dev)
    result_k_data = result[..., :d // 2].view(torch.uint8)
    result_k_scale = result[..., d // 2:]
    for b0 in range(0, num_blocks, blocks_per_chunk):
        b1 = min(b0 + blocks_per_chunk, num_blocks)
        n = b1 - b0
        x = x_all[b0: b1].float()
        if granularity == "token":
            amax_tok = torch.nan_to_num(x.abs(), nan=0.0, posinf=0.0).amax(dim=-1)                  # [n, block_size]
            g_out[b0: b1] = fp4_2l_choose_g(amax_tok, four_six)
            G_t = _fp4_2l_pow2(g_out[b0: b1]).unsqueeze(-1)                                         # [n, block_size, 1]
        codes, scale = quantize_fp4_2l_blocks(x.view(n, block_size, num_tiles, tile_size), G_t, rule, four_six)
        codes = codes.view(n, block_size, d)
        result_k_data[b0: b1] = codes[..., 0::2] | (codes[..., 1::2] << 4)
        result_k_scale[b0: b1] = scale
    return result.view(num_blocks, block_size, 1, -1), g_out


def dequantize_k_cache_fp4_2l(quant_k_cache: torch.Tensor, g) -> torch.Tensor:
    """(num_blocks, block_size, 1, 288) bytes + g (int or int32 tensor (num_blocks, block_size)) -> bf16 (num_blocks, block_size, 1, 512); e2m1 * e4m3 * 2^g is exact in bf16"""
    d, d_nope, d_rope, tile_size, num_tiles = KVCacheLayout.V41_FP4Sparse.get_meta()
    num_blocks, block_size, h_k, _ = quant_k_cache.shape
    q = quant_k_cache.view(num_blocks, block_size, -1)
    data = q[..., :d // 2].view(torch.uint8)
    scale = q[..., d // 2:].view(torch.float8_e4m3fn)
    result = torch.empty((num_blocks, block_size, d), dtype=torch.bfloat16, device=quant_k_cache.device)
    blocks_per_chunk = max(1, (1 << 26) // (block_size * d))
    for b0 in range(0, num_blocks, blocks_per_chunk):
        b1 = min(b0 + blocks_per_chunk, num_blocks)
        codes = torch.empty((b1 - b0, block_size, d), dtype=torch.uint8, device=quant_k_cache.device)
        codes[..., 0::2] = data[b0:b1] & 0xF
        codes[..., 1::2] = data[b0:b1] >> 4
        values = _dequantize_e2m1(codes).view(b1 - b0, block_size, num_tiles, tile_size)
        if isinstance(g, torch.Tensor):
            G = _fp4_2l_pow2(g[b0:b1]).to(values.device).view(b1 - b0, block_size, 1, 1)
        else:
            G = _fp4_2l_pow2(g).to(values.device)
        result[b0:b1] = ((values * scale[b0:b1].to(values.dtype).unsqueeze(-1)) * G).view(b1 - b0, block_size, d).to(torch.bfloat16)
    return result.view(num_blocks, block_size, 1, d)
