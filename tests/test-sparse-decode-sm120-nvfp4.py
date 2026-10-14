"""
Correctness test of the sm120 (RTX 5090 / RTX PRO Blackwell) sparse decode kernels on a V4.1 fp4 KV cache as the MAIN cache (V41_FP4Sparse: 256 B e2m1
+ 32 B e4m3, one scale per 16 dims = 288 B per token), with or without a V41_FP4Sparse extra cache. The cases follow tests/test-sparse-decode.py (block
sizes 1 ... 576, b = 1 ... 321, s_q = 1 / 3, all-invalid indices, zero-length sequences, topk_length / extra_topk_length, attn_sink, requests that are and
are not split over CTAs) and tests/test-sparse-decode-sm120-mxfp4.py.

Every case is run in the three modes of FLASH_MLA_SM120_DECODE_MODE that this cache has:
  bf16    (also the default) the exact tier: e2m1 x e4m3 is exact in bf16, so the output and the LSE must pass the upstream tolerances against
          tests/ref.py on the dequantized cache.
  fp8x2   has no fp4 tile path (one e4m3 term of V is outside the tolerance of the upstream tests): it runs the bf16 kernel, so the result must be
          bit-identical to the bf16 mode.
  mxfp8   all tiles on the NVFP4 MMA for QK (Q as 2 NVFP4 terms) and the e4m3 MMA for PV (V rebased once per tile, P as one e4m3 term): stays below a
          generous bound against the reference, and is compared with its emulator row (p3-mxfp8-nvq2s, tests/sm120_emu.py): about 99.9% of the bf16
          outputs are bit-identical when no request is split over CTAs (a split changes the running max the P terms are scaled by; then only the
          errors against the reference are compared). The fraction is checked for h_q = 128 and recorded for h_q = 64.
No mode may produce NaN. mxfp4x4 (for MXFP4 caches) and the pairs of caches that have no kernel must raise an error.

  cd tests && python test-sparse-decode-sm120-nvfp4.py [--quick] [--emu-max-bq N] [--run-to-finish]
"""
import argparse
import copy
import dataclasses
import importlib.util
import os
import random
import statistics
import sys
import time
from typing import List, Tuple

import torch
import kernelkit as kk

import flash_mla

import lib
from lib import TestParam
from lib import RawTestParamForDecode as RawTestParam
import quant
import ref

L = quant.KVCacheLayout
MODES = ("bf16", "fp8x2", "mxfp8")
EMU_ROW = "p3-mxfp8-nvq2s"      # the emulator row of the fp4 tile of mxfp8.cu: Q as 2 NVFP4 terms, V one e4m3 term on a scale from the largest 16-dim scale (d1s), P x1

# The upstream tolerances (tests/test-sparse-decode.py)
OUT_TOL = dict(abs_tol=1e-3, rel_tol=2.01 / 128, cos_diff_tol=5e-6)
LSE_TOL = dict(abs_tol=1e-6, rel_tol=8.01 / 65536)
# The mxfp8 tier: against the emulator row (the kernel differs from it only by the order of the fp32 additions inside the MMAs), and against the reference
# kernel vs emulator: the only difference is the order of the fp32 additions inside the MMAs, which rarely moves a P value across a rounding boundary of
# e4m3; one such flip in a request with few valid tokens changes its 512 outputs by a few percent (cos_diff up to 7e-7 measured over 860 cases, the median
# is 2e-10). The mode's own error against the reference is 4e-4, so this bound still separates a real difference by two orders of magnitude.
EMU_COS_TOL = 2e-6
EMU_SAME_BF16 = 0.999           # fraction of bit-identical bf16 outputs, no split request, h_q = 128
MXFP8_REF_COS_TOL = 2e-3        # kernel vs reference (the upstream data with an fp4 extra cache: about 3.4e-4)
MXFP8_REF_LSE_TOL = 1e-2
SPLIT_ERR_FACTOR = 2.0          # with split requests: cos_diff(kernel, ref) <= this * cos_diff(emulator, ref) + SPLIT_ERR_SLACK
SPLIT_ERR_SLACK = 2e-6


def gen_testcase(quick: bool) -> List[RawTestParam]:
    kk.set_random_seed(0)

    correctness_cases = []
    corner_cases = []
    for have_extra_k in [False, True]:
        kvcache_layout, extra_kvcache_layout = L.V41_FP4Sparse, (L.V41_FP4Sparse if have_extra_k else None)
        for h_q in [64, 128]:
            have_topk_len = random.choice([False, True])
            have_extra_topk_len = random.choice([False, True, True]) if have_extra_k else False
            cur_correctness_cases = [
                RawTestParam(b, h_q, s_q, 1, s_k, is_varlen, topk,
                             have_topk_length=have_topk_len,
                             enable_attn_sink=random.randint(0, 1) == 1,
                             extra_s_k=extra_s_k,
                             extra_topk=extra_topk,
                             block_size=block_size,
                             extra_block_size=extra_block_size,
                             have_extra_topk_length=have_extra_topk_len,
                             d_qk=512,
                             kvcache_layout=kvcache_layout,
                             extra_kvcache_layout=extra_kvcache_layout,
                             check_correctness=True,
                             num_runs=0)
                for (s_k, topk, block_size) in [
                    (512, 64, 5),
                    (512, 64, 69),
                    (1024, 576, 2),
                    (1024, 576, 61),
                    (2046, 2048, 1),
                    (2046, 2048, 64),
                    (2046, 2048, 576)
                ]
                for (extra_s_k, extra_topk, extra_block_size) in ([
                    (512, 64, 5),
                    (512, 64, 69),
                    (1024, 576, 2),
                    (1024, 576, 123),
                    (2046, 2048, 1),
                    (2046, 2048, 576)
                ] if have_extra_k else [(None, None, None)])
                for b in [4, 74, 321]
                for s_q in [1, 3]
                for is_varlen in ([True, False] if (b == 74 and not have_topk_len and not have_extra_topk_len) else [True])    # With b=74 and no varlen/topk_len/extra_topk_len, no request should be split-kv
            ]
            correctness_cases.extend(cur_correctness_cases)

            cur_corner_cases = [
                RawTestParam(b, h_q, s_q, 1, s_k, is_varlen, topk,
                             is_all_indices_invalid=is_all_indices_invalid,
                             have_zero_seqlen_k=have_zero_seqlen_k,
                             have_topk_length=have_topk_len,
                             enable_attn_sink=enable_attn_sink,
                             extra_s_k=extra_s_k,
                             extra_topk=extra_topk,
                             block_size=block_size,
                             extra_block_size=extra_block_size,
                             have_extra_topk_length=have_extra_topk_len,
                             d_qk=512,
                             kvcache_layout=kvcache_layout,
                             extra_kvcache_layout=extra_kvcache_layout,
                             check_correctness=True,
                             num_runs=0)
                for (s_k, topk, block_size) in [
                    (512, 64, 61),
                    (650, 576, 53),
                ]
                for (extra_s_k, extra_topk, extra_block_size) in ([
                    (512, 64, 61),
                    (650, 576, 53),
                ] if have_extra_k else [(None, None, None)])
                for b in [1, 74, 321]
                for s_q in [3]
                for is_varlen in ([True, False] if (b == 74 and not have_topk_len and not have_extra_topk_len) else [True])
                for is_all_indices_invalid in [True, False]
                for have_zero_seqlen_k in [True, False]
                for enable_attn_sink in [True, False]
                if (is_all_indices_invalid or have_zero_seqlen_k or enable_attn_sink)
            ]
            corner_cases.extend(cur_corner_cases)

    # head128 with extra_topk not a multiple of the tile (32 tokens): the partial last tile goes through the mask
    partial_extra_block_cases = [
        RawTestParam(4, h_q, 3, 1, 512, True, 128,
                     extra_s_k=512, extra_topk=extra_topk, block_size=64, extra_block_size=64,
                     d_qk=512,
                     kvcache_layout=L.V41_FP4Sparse,
                     extra_kvcache_layout=L.V41_FP4Sparse,
                     check_correctness=True, num_runs=0)
        for h_q in [64, 128]
        for extra_topk in [100, 33, 31, 1]
    ]
    # Production shapes (DeepSeek-V4.1 decode: s_q 4, topk 128 + extra 512, block 256 / 64); the timing of them is benchmarks / nbm.py
    production_cases = [
        RawTestParam(b, h_q, 4, 1, 256, True, topk=128, d_qk=512, extra_s_k=2048, extra_topk=512, block_size=256, extra_block_size=64,
                     kvcache_layout=L.V41_FP4Sparse, extra_kvcache_layout=L.V41_FP4Sparse, check_correctness=True, num_runs=0)
        for h_q in [64, 128]
        for b in [64, 128, 256, 512]
    ] + [
        # no extra cache
        RawTestParam(b, h_q, 4, 1, 256, True, topk=128, d_qk=512, block_size=256,
                     kvcache_layout=L.V41_FP4Sparse, check_correctness=True, num_runs=0)
        for h_q in [64, 128]
        for b in [64, 512]
    ]
    cases = correctness_cases + corner_cases + partial_extra_block_cases + production_cases
    if quick:
        rng = random.Random(1)
        keep = partial_extra_block_cases + production_cases
        rest = correctness_cases + corner_cases
        cases = rng.sample(rest, k=len(rest) // 8) + keep
    return cases


@dataclasses.dataclass
class CaseResult:
    ok: bool
    detail: str


_counter = kk.Counter()
_stats = {"bf16_cos": [], "mxfp8_ref_cos": [], "mxfp8_ref_dlse": [], "emu_cos": [], "same": {64: [], 128: []}}


def cos_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(kk.get_cos_diff(a, b))


def run_mode(p: TestParam, t, sched_meta, mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
    os.environ["FLASH_MLA_SM120_DECODE_MODE"] = mode
    out, lse = lib.run_flash_mla_decode(p, t, sched_meta, None)
    torch.cuda.synchronize()
    return out, lse


@torch.inference_mode()
def test_case(p: TestParam, ev, emu_max_bq: int) -> CaseResult:
    if p.seed == -1:
        global _counter
        p.seed = _counter.next()
    assert p.decode
    print("================")
    print(f"Running on {p}")
    torch.cuda.empty_cache()

    t = lib.generate_testcase_for_decode(p)
    sched_meta, _ = flash_mla.get_mla_metadata()
    out_ref, lse_ref = ref.ref_sparse_attn_decode(p, t)
    torch.cuda.synchronize()

    ok = True
    detail = []
    results = {}
    any_split = False
    for mode in MODES:
        out, lse = run_mode(p, t, sched_meta, mode)
        results[mode] = (out, lse)
        if mode == MODES[0] and sched_meta.num_splits is not None:
            ns = sched_meta.num_splits
            any_split = bool(((ns[1:] - ns[:-1]) > 1).any())
        if bool(torch.isnan(out.float()).any() or torch.isnan(lse).any()):
            print(f"[{mode}] NaN in the output")
            ok = False
            detail.append(f"{mode}:NaN")
            continue
        c = cos_diff(out, out_ref)
        inf_match = bool(torch.equal(torch.isinf(lse), torch.isinf(lse_ref)))
        fin = torch.isfinite(lse) & torch.isfinite(lse_ref)
        d_lse = float((lse - lse_ref)[fin].abs().max()) if bool(fin.any()) else 0.0
        if mode == "bf16":
            out_ok = kk.check_is_allclose(f"out[{mode}]", out, out_ref, **OUT_TOL)
            lse_ok = kk.check_is_allclose(f"lse[{mode}]", lse, lse_ref, **LSE_TOL)
            print(f"[{mode}] vs ref: cos_diff {c:.2e}, max |d lse| {d_lse:.2e}")
            _stats["bf16_cos"].append(c)
            if not (out_ok and lse_ok and inf_match):
                ok = False
                detail.append(f"{mode}:ref")
        elif mode == "fp8x2":
            bf16_out, bf16_lse = results["bf16"]
            same = torch.equal(out, bf16_out) and torch.equal(lse, bf16_lse)
            print(f"[{mode}] vs bf16 mode: {'bit-identical' if same else 'DIFFERENT'}")
            if not same:
                ok = False
                detail.append(f"{mode}:not-bf16")
        else:
            print(f"[{mode}] vs ref: cos_diff {c:.2e}, max |d lse| {d_lse:.2e}, inf pattern {'same' if inf_match else 'DIFFERENT'}")
            _stats["mxfp8_ref_cos"].append(c)
            _stats["mxfp8_ref_dlse"].append(d_lse)
            if not (c <= MXFP8_REF_COS_TOL and inf_match and d_lse <= MXFP8_REF_LSE_TOL):
                ok = False
                detail.append(f"{mode}:ref")
            bq = p.decode.b * p.s_q
            if bq <= emu_max_bq:
                with ev.upstream_defaults():
                    out_e, lse_e = ev.emu.simulate_decode(p, t, ev.COMBOS[EMU_ROW])
                out_e = out_e.to(out.dtype).view_as(out)
                lse_e = lse_e.view_as(lse)
                out_e_ref_cos = cos_diff(out_e, out_ref)
                ce = cos_diff(out, out_e)
                same = float((out == out_e).double().mean())
                inf_e = bool(torch.equal(torch.isinf(lse), torch.isinf(lse_e)))
                fin_e = torch.isfinite(lse) & torch.isfinite(lse_e)
                d_lse_e = float((lse - lse_e)[fin_e].abs().max()) if bool(fin_e.any()) else 0.0
                print(f"[{mode}] vs emulator: cos_diff {ce:.2e}, bit-identical bf16 {100 * same:.3f}%, max |d lse| {d_lse_e:.2e}, inf pattern {'same' if inf_e else 'DIFFERENT'}"
                      f"{', split requests: emulator error vs ref ' + format(out_e_ref_cos, '.2e') if any_split else ''}")
                if any_split:
                    good = inf_e and c <= SPLIT_ERR_FACTOR * out_e_ref_cos + SPLIT_ERR_SLACK
                else:
                    _stats["emu_cos"].append(ce)
                    _stats["same"][p.h_q].append(same)
                    good = ce <= EMU_COS_TOL and inf_e and d_lse_e <= 1e-4 and (p.h_q != 128 or same >= EMU_SAME_BF16)
                if not good:
                    ok = False
                    detail.append(f"{mode}:emu")
                del out_e, lse_e
    # the kernels are deterministic: a second run is bit-identical
    for mode in MODES:
        out2, lse2 = run_mode(p, t, sched_meta, mode)
        if not (torch.equal(out2, results[mode][0]) and torch.equal(lse2, results[mode][1])):
            print(f"[{mode}] a second run differs")
            ok = False
            detail.append(f"{mode}:nondeterministic")
    del t, out_ref, lse_ref, results
    return CaseResult(ok, ",".join(detail))


def expect_raise(what: str, p: TestParam, mode: str) -> bool:
    t = lib.generate_testcase_for_decode(p)
    sched_meta, _ = flash_mla.get_mla_metadata()
    os.environ["FLASH_MLA_SM120_DECODE_MODE"] = mode
    try:
        lib.run_flash_mla_decode(p, t, sched_meta, None)
        torch.cuda.synchronize()
    except Exception as e:     # noqa: BLE001
        print(f"{what} (mode '{mode}') raises as expected: {type(e).__name__}: {str(e).splitlines()[0][:200]}")
        return True
    print(f"{what} (mode '{mode}') did not raise")
    return False


def test_rejected() -> bool:
    """mxfp4x4 on an fp4 main cache (with and without an fp4 extra cache), and the pairs of caches that have no kernel, must raise"""
    ok = True
    kw = dict(block_size=64, check_correctness=False, num_runs=0)
    def param(main, extra, seed):
        extra_kw = dict(extra_s_k=256, extra_topk=64, extra_block_size=64) if extra is not None else {}
        return RawTestParam(2, 64, 1, 1, 256, True, 64, kvcache_layout=main, extra_kvcache_layout=extra, seed=seed, **kw, **extra_kw).to_test_param()

    ok &= expect_raise("FP4 main cache", param(L.V41_FP4Sparse, None, 11), "mxfp4x4")
    ok &= expect_raise("FP4 main + FP4 extra cache", param(L.V41_FP4Sparse, L.V41_FP4Sparse, 12), "mxfp4x4")
    ok &= expect_raise("FP8 main + FP4 extra cache", param(L.V41_FP8Sparse, L.V41_FP4Sparse, 13), "mxfp4x4")
    for mode in ("bf16", "mxfp8"):
        ok &= expect_raise("FP4 main + FP8 extra cache", param(L.V41_FP4Sparse, L.V41_FP8Sparse, 14), mode)
        ok &= expect_raise("FP4 main + MXFP4 extra cache", param(L.V41_FP4Sparse, L.V41_MXFP4Sparse, 15), mode)
        ok &= expect_raise("MXFP4 main + FP4 extra cache", param(L.V41_MXFP4Sparse, L.V41_FP4Sparse, 16), mode)
    return bool(ok)


def summary() -> None:
    def stat(name, xs, fmt):
        if xs:
            print(f"  {name}: n {len(xs)}, min {fmt.format(min(xs))}, median {fmt.format(statistics.median(xs))}, max {fmt.format(max(xs))}")
    print("statistics over the cases run:")
    stat("bf16 out cos_diff vs ref", _stats["bf16_cos"], "{:.2e}")
    stat("mxfp8 out cos_diff vs ref", _stats["mxfp8_ref_cos"], "{:.2e}")
    stat("mxfp8 max |d lse| vs ref", _stats["mxfp8_ref_dlse"], "{:.2e}")
    stat("mxfp8 vs emulator cos_diff (no split)", _stats["emu_cos"], "{:.2e}")
    for h in (64, 128):
        stat(f"mxfp8 vs emulator bit-identical fraction, h_q {h} (no split)", _stats["same"][h], "{:.5f}")


def main():
    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(32)

    parser = argparse.ArgumentParser()
    parser.add_argument("-rf", "--run-to-finish", action="store_true", help="Don't exit when a testcase is failed")
    parser.add_argument("--quick", action="store_true", help="An eighth of the correctness / corner cases, plus the partial-tile and production cases")
    parser.add_argument("--emu-max-bq", type=int, default=330, help="Compare the mxfp8 mode with the emulator for cases with b * s_q <= this (the emulator runs in float64)")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=None)
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("eval_modes", os.path.join(os.path.dirname(os.path.abspath(lib.__file__)), "eval-sm120-modes.py"))
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    ev.COMBOS = {name: mode for name, _, mode, _ in ev.build_combos()}

    raw_testcases = gen_testcase(args.quick)
    testcases = []
    for t in (r.to_test_param() for r in raw_testcases):
        t = copy.deepcopy(t)
        if t.can_run_on_and_clamp(lib.TestTarget.DECODE):
            testcases.append(t)
    testcases = testcases[args.start:args.stop]
    print(f"{kk.colors['CYAN_BG']}{len(testcases)} testcases to run{kk.colors['CLEAR']}")

    failed = []
    t0 = time.time()
    for i, tc in enumerate(testcases):
        print(f"[{i + 1}/{len(testcases)}, {i / len(testcases) * 100:3.0f}%]  ", end="")
        r = test_case(tc, ev, args.emu_max_bq)
        if not r.ok:
            failed.append((tc, r.detail))
            print(f"{kk.colors['RED_BG']}FAILED ({r.detail}){kk.colors['CLEAR']}")
            if not args.run_to_finish:
                sys.exit(1)
    rejected_ok = test_rejected()
    summary()
    n_ok = len(testcases) - len(failed)
    if n_ok == len(testcases) and rejected_ok:
        print(f"{kk.colors['GREEN_BG']}{n_ok}/{len(testcases)} correctness cases passed in {time.time() - t0:.0f} s, rejected combinations raise{kk.colors['CLEAR']}")
    else:
        print(f"{kk.colors['RED_BG']}{n_ok}/{len(testcases)} correctness cases passed, rejected combinations raise: {rejected_ok}{kk.colors['CLEAR']}")
        for tc, d in failed:
            print(f"\t{d}: {tc},")
        sys.exit(1)


if __name__ == "__main__":
    main()
