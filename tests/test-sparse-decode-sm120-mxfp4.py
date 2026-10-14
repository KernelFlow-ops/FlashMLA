"""
Correctness test of the sm120 (RTX 5090 / RTX PRO Blackwell) sparse decode kernel on an MXFP4 KV cache (V41_MXFP4Sparse: 256 B e2m1 + 16 B ue8m0
per token), main cache with or without an MXFP4 extra cache. The cases follow tests/test-sparse-decode.py (block sizes 1 ... 576, b = 1 ... 321,
s_q = 1 / 3, all-invalid indices, zero-length sequences, topk_length / extra_topk_length, attn_sink, requests that are and are not split over CTAs).

Every case is run in the two arithmetic modes of FLASH_MLA_SM120_DECODE_MODE that an MXFP4 cache has:
  mxfp4x4   Q as 4 MXFP4 terms, P as 2 e4m3 terms. The output must pass the upstream tolerances against tests/ref.py on the dequantized cache. The LSE
            too, except for the query tokens whose reference LSE is below 0.05 (a single valid KV token): four terms of Q move it by up to 3e-6,
            more than the upstream tolerance of 1e-6 absolute, and the test allows 1e-5 there and counts how many rows are beyond the upstream limit.
  mxfp8     Q as 2 MXFP4 terms, P as 1 e4m3 term: stays below a generous bound against the reference (its error is the 4 bit Q / P)
Both are compared with their emulator rows (kernel-mxfp4-fp8x2 / kernel-mxfp4-mxfp8 in tests/sm120_emu.py): about 99.9% of the bf16 outputs are
bit-identical when no request is split over CTAs (a split changes the running max the P terms are scaled by; then only the errors against the
reference are compared). No mode may produce NaN. The bf16 (default) and fp8x2 modes, and an FP8 main cache with an MXFP4 extra cache, must raise
an error: no arithmetic within the upstream tolerance exists for the first two, and the third is not implemented.

  cd tests && python test-sparse-decode-sm120-mxfp4.py [--quick] [--emu-max-bq N] [--run-to-finish]
"""
import argparse
import copy
import dataclasses
import importlib.util
import os
import random
import sys
import time
from typing import List, Optional, Tuple

import torch
import kernelkit as kk

import flash_mla

import lib
from lib import TestParam
from lib import RawTestParamForDecode as RawTestParam
import quant
import ref

L = quant.KVCacheLayout
MODES = ("mxfp4x4", "mxfp8")
EMU_ROWS = {"mxfp4x4": "kernel-mxfp4-fp8x2", "mxfp8": "kernel-mxfp4-mxfp8"}
TINY_LSE = 0.05                 # reference LSE below this: a query token with a single valid KV token (or a few with logits near 0)
TINY_LSE_ABS_TOL = 1e-5         # the allowed LSE error there in the mxfp4x4 mode (upstream: 1e-6; measured <= 3e-6)

# The upstream tolerances (tests/test-sparse-decode.py)
OUT_TOL = dict(abs_tol=1e-3, rel_tol=2.01 / 128, cos_diff_tol=5e-6)
LSE_TOL = dict(abs_tol=1e-6, rel_tol=8.01 / 65536)
# The mxfp8 tier: against the emulator row (the kernel differs from it only by the order of the fp32 additions inside the MMAs), and against the reference
EMU_COS_TOL = 1e-7              # kernel vs emulator: measured <= 1e-8
EMU_SAME_BF16 = 0.998           # fraction of bit-identical bf16 outputs: measured >= 99.9%
MXFP8_REF_COS_TOL = 1e-3        # kernel vs reference: measured 1.3e-4 ... 2.7e-4 on this data
SPLIT_ERR_FACTOR = 2.0          # with split requests: cos_diff(kernel, ref) <= this * cos_diff(emulator, ref) + SPLIT_ERR_SLACK
SPLIT_ERR_SLACK = 2e-6


def gen_testcase(quick: bool) -> List[RawTestParam]:
    kk.set_random_seed(0)

    correctness_cases = []
    corner_cases = []
    for have_extra_k in [False, True]:
        kvcache_layout, extra_kvcache_layout = L.V41_MXFP4Sparse, (L.V41_MXFP4Sparse if have_extra_k else None)
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
                     kvcache_layout=L.V41_MXFP4Sparse,
                     extra_kvcache_layout=L.V41_MXFP4Sparse,
                     check_correctness=True, num_runs=0)
        for h_q in [64, 128]
        for extra_topk in [100, 33, 31, 1]
    ]
    # Production shapes (DeepSeek-V4.1 decode: s_q 4, topk 128 + extra 512, block 256 / 64); the timing of them is benchmarks / quick_bench
    production_cases = [
        RawTestParam(b, h_q, 4, 1, 256, True, topk=128, d_qk=512, extra_s_k=2048, extra_topk=512, block_size=256, extra_block_size=64,
                     kvcache_layout=L.V41_MXFP4Sparse, extra_kvcache_layout=L.V41_MXFP4Sparse, check_correctness=True, num_runs=0)
        for h_q in [64, 128]
        for b in [64, 128, 256, 512]
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


def cos_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(kk.get_cos_diff(a, b))


def run_mode(p: TestParam, t, sched_meta, mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
    os.environ["FLASH_MLA_SM120_DECODE_MODE"] = mode
    out, lse = lib.run_flash_mla_decode(p, t, sched_meta, None)
    torch.cuda.synchronize()
    return out, lse


def check_lse(name: str, lse: torch.Tensor, lse_ref: torch.Tensor) -> Tuple[bool, int]:
    """The upstream tolerance, with TINY_LSE_ABS_TOL absolute on the rows whose reference LSE is below TINY_LSE. Returns (ok, rows beyond the upstream limit)"""
    d = (lse - lse_ref).abs()
    fin = torch.isfinite(lse) & torch.isfinite(lse_ref)
    beyond = fin & (d > LSE_TOL["abs_tol"]) & (d / (lse_ref.abs() + 1e-6) > LSE_TOL["rel_tol"])
    tiny = fin & (lse_ref.abs() < TINY_LSE) & (d <= TINY_LSE_ABS_TOL)
    adj = torch.where(tiny, lse_ref, lse)
    return kk.check_is_allclose(name, adj, lse_ref, **LSE_TOL), int(beyond.sum())


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
        if mode == "mxfp4x4":
            out_ok = kk.check_is_allclose(f"out[{mode}]", out, out_ref, **OUT_TOL)
            lse_ok, n_beyond = check_lse(f"lse[{mode}]", lse, lse_ref)
            print(f"[{mode}] vs ref: cos_diff {c:.2e}, max |d lse| {d_lse:.2e}, rows beyond the upstream LSE limit {n_beyond} of {int(fin.sum())}")
            if not (out_ok and lse_ok and inf_match):
                ok = False
                detail.append(f"{mode}:ref")
        else:
            print(f"[{mode}] vs ref: cos_diff {c:.2e}, max |d lse| {d_lse:.2e}, inf pattern {'same' if inf_match else 'DIFFERENT'}")
            if not (c <= MXFP8_REF_COS_TOL and inf_match and d_lse <= 1e-2):
                ok = False
                detail.append(f"{mode}:ref")
        bq = p.decode.b * p.s_q
        if bq <= emu_max_bq:
            with ev.upstream_defaults():
                out_e, lse_e = ev.emu.simulate_decode(p, t, ev.COMBOS[EMU_ROWS[mode]])
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
                good = ce <= EMU_COS_TOL and same >= EMU_SAME_BF16 and inf_e and d_lse_e <= 1e-4
            if not good:
                ok = False
                detail.append(f"{mode}:emu")
            del out_e, lse_e
    # the kernel is deterministic: a second run is bit-identical
    for mode in MODES:
        out2, lse2 = run_mode(p, t, sched_meta, mode)
        if not (torch.equal(out2, results[mode][0]) and torch.equal(lse2, results[mode][1])):
            print(f"[{mode}] a second run differs")
            ok = False
            detail.append(f"{mode}:nondeterministic")
    del t, out_ref, lse_ref, results
    return CaseResult(ok, ",".join(detail))


def test_rejected_modes() -> bool:
    """bf16 (also when FLASH_MLA_SM120_DECODE_MODE is unset) and fp8x2 on an MXFP4 cache, and mxfp4x4 on an FP8 cache, must raise"""
    ok = True
    p = RawTestParam(2, 64, 1, 1, 256, True, 64, block_size=64, kvcache_layout=L.V41_MXFP4Sparse, check_correctness=False, num_runs=0, seed=3).to_test_param()
    t = lib.generate_testcase_for_decode(p)
    sched_meta, _ = flash_mla.get_mla_metadata()
    for mode in ("", "bf16", "fp8x2"):
        os.environ["FLASH_MLA_SM120_DECODE_MODE"] = mode
        try:
            lib.run_flash_mla_decode(p, t, sched_meta, None)
            torch.cuda.synchronize()
            print(f"MXFP4 cache with mode '{mode}' did not raise")
            ok = False
        except Exception as e:     # noqa: BLE001
            print(f"MXFP4 cache with mode '{mode}' raises as expected: {type(e).__name__}: {str(e).splitlines()[0][:200]}")
    os.environ["FLASH_MLA_SM120_DECODE_MODE"] = "mxfp4x4"
    p = RawTestParam(2, 64, 1, 1, 256, True, 64, block_size=64, kvcache_layout=L.V41_FP8Sparse, check_correctness=False, num_runs=0, seed=3).to_test_param()
    t = lib.generate_testcase_for_decode(p)
    sched_meta, _ = flash_mla.get_mla_metadata()
    try:
        lib.run_flash_mla_decode(p, t, sched_meta, None)
        torch.cuda.synchronize()
        print("mode mxfp4x4 on an FP8 cache did not raise")
        ok = False
    except Exception as e:     # noqa: BLE001
        print(f"mode mxfp4x4 on an FP8 cache raises as expected: {type(e).__name__}: {str(e).splitlines()[0][:200]}")
    return ok


def test_unsupported_mix() -> bool:
    """FP8 main cache + MXFP4 extra cache is not implemented: it must raise, not compute on a misread cache"""
    p = RawTestParam(2, 64, 1, 1, 256, True, 64, extra_s_k=256, extra_topk=64, block_size=64, extra_block_size=64,
                     kvcache_layout=L.V41_FP8Sparse, extra_kvcache_layout=L.V41_MXFP4Sparse, check_correctness=False, num_runs=0, seed=5).to_test_param()
    t = lib.generate_testcase_for_decode(p)
    sched_meta, _ = flash_mla.get_mla_metadata()
    os.environ["FLASH_MLA_SM120_DECODE_MODE"] = "bf16"
    try:
        lib.run_flash_mla_decode(p, t, sched_meta, None)
        torch.cuda.synchronize()
    except Exception as e:     # noqa: BLE001
        print(f"FP8 main + MXFP4 extra raises as expected: {type(e).__name__}: {str(e).splitlines()[0][:160]}")
        return True
    print("FP8 main + MXFP4 extra did not raise")
    return False


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
    mix_ok = test_unsupported_mix() and test_rejected_modes()
    n_ok = len(testcases) - len(failed)
    if n_ok == len(testcases) and mix_ok:
        print(f"{kk.colors['GREEN_BG']}{n_ok}/{len(testcases)} correctness cases passed in {time.time() - t0:.0f} s, rejected modes raise{kk.colors['CLEAR']}")
    else:
        print(f"{kk.colors['RED_BG']}{n_ok}/{len(testcases)} correctness cases passed, rejected modes raise: {mix_ok}{kk.colors['CLEAR']}")
        for tc, d in failed:
            print(f"\t{d}: {tc},")
        sys.exit(1)


if __name__ == "__main__":
    main()
