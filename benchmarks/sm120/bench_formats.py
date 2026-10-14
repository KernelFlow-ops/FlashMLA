"""
Kernel time of the sm120 sparse decode kernel for every KV cache format and precision mode, on the production shapes of
tests/test-sparse-decode.py, with all of them interleaved in one process.

Per shape, the data of every format is generated first (`lib.generate_testcase_for_decode`, seed 0, the same generator for
all formats), then `--passes` rounds go over the configs "format:mode". Every config of a round sees the same clocks, power and
L2 state, so the ratios between them are steadier than those of separate runs. Timing is the upstream test's: `kk.bench`
(kineto, 8 GB L2 flush before every call, `--runs` calls), the time of the decode kernel alone.

Formats
  fp8       V4.1 fp8 main cache + V4.1 fp8 extra cache (upstream)
  fp8+fp4   V4.1 fp8 main cache + V4.1 fp4 extra cache (upstream)
  mxfp4     MXFP4 main + MXFP4 extra cache (KVCacheLayout.V41_MXFP4Sparse, sm120 only)
  nvfp4     V4.1 fp4 main + V4.1 fp4 extra cache (sm120 only)
Modes: the values of FLASH_MLA_SM120_DECODE_MODE (bf16, fp8x2, mxfp8, mxfp4x4); see docs/sm120/README.md for which mode runs
which kernel on which format.

Shapes
  h{64,128}_b{64,128,256,512}   s_q 4, topk 128 (page 256) + extra topk 512 (page 64), the KV lengths of the upstream test
  h{64,128}_sq4096              b 1, s_q 4096, topk 128 + 512, the upstream "prefill as decode" shapes

Output: one line per shape and round, a CSV (`--csv`: shape, config, pass, us) and a markdown table of the medians (`--md`).

Usage (exclusive GPU; `lib` and `quant` come from this tree's tests/, `flash_mla` from PYTHONPATH):
  PYTHONPATH=<tree> python benchmarks/sm120/bench_formats.py --csv new.csv --md new.md
An older build is measured with this tree's data generator by putting it on PYTHONPATH and asking only for the formats it
supports, e.g. the first sm120 kernel (bf16 only, FP8 main cache):
  PYTHONPATH=<old tree> python benchmarks/sm120/bench_formats.py --configs fp8:bf16,fp8+fp4:bf16 --csv old.csv
Alternate the processes of the builds to compare (old, new, old, new) and take the medians over all of them.
"""
import argparse
import csv
import os
import statistics
import subprocess
import sys

_TESTS_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests"))
sys.path.insert(0, _TESTS_DIR)

import torch
import kernelkit as kk

import flash_mla
import lib
import quant
from lib import RawTestParamForDecode as P

KERNEL = "flash_fwd_splitkv_mla_fp8_sparse_kernel"
L = quant.KVCacheLayout
FORMATS = {
    "fp8": (L.V41_FP8Sparse, L.V41_FP8Sparse),
    "fp8+fp4": (L.V41_FP8Sparse, L.V41_FP4Sparse),
    "mxfp4": (L.V41_MXFP4Sparse, L.V41_MXFP4Sparse),
    "nvfp4": (L.V41_FP4Sparse, L.V41_FP4Sparse),
}
DEFAULT_SHAPES = ",".join([f"h{h}_b{b}" for h in (64, 128) for b in (64, 128, 256, 512)] + ["h64_sq4096", "h128_sq4096"])
DEFAULT_CONFIGS = "fp8:bf16,fp8:fp8x2,fp8:mxfp8,fp8+fp4:bf16,fp8+fp4:mxfp8,mxfp4:mxfp8,mxfp4:mxfp4x4,nvfp4:bf16,nvfp4:mxfp8"


def make_param(fmt: str, name: str) -> P:
    main_layout, extra_layout = FORMATS[fmt]
    h = int(name.split("_")[0][1:])
    if name.endswith("_sq4096"):
        return P(1, h, 4096, 1, 4096, False, 128, extra_s_k=4096, extra_topk=512, extra_block_size=64,
                 kvcache_layout=main_layout, extra_kvcache_layout=extra_layout)
    b = int(name.split("_")[1][1:])
    return P(b, h, 4, 1, 256, True, topk=128, extra_s_k=2048, extra_topk=512, block_size=256, extra_block_size=64,
             kvcache_layout=main_layout, extra_kvcache_layout=extra_layout)


def gpu_state() -> str:
    return subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--configs", default=DEFAULT_CONFIGS, help="format:mode,...")
    parser.add_argument("--shapes", default=DEFAULT_SHAPES)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--md", default=None)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda:0")
    configs = [tuple(c.split(":")) for c in args.configs.split(",")]
    formats = list(dict.fromkeys(f for f, _ in configs))
    print(f"build: {os.path.dirname(flash_mla.__file__)}  tag: {args.tag}  configs: {args.configs}  passes: {args.passes}", flush=True)
    print(f"gpu before: {gpu_state()}", flush=True)
    rows, table = [], {}
    for name in args.shapes.split(","):
        data = {}
        for fmt in formats:
            p = make_param(fmt, name).to_test_param()
            p.seed = 0
            data[fmt] = (p, lib.generate_testcase_for_decode(p), flash_mla.get_mla_metadata()[0])
        table[name] = {f"{f}:{m}": [] for f, m in configs}
        for ps in range(args.passes):
            line = f"{name:12s} pass {ps + 1}"
            for fmt, mode in configs:
                p, t, sched_meta = data[fmt]
                os.environ["FLASH_MLA_SM120_DECODE_MODE"] = mode
                lib.run_flash_mla_decode(p, t, sched_meta, None)
                result = kk.bench(lambda: lib.run_flash_mla_decode(p, t, sched_meta, None), args.runs)
                us = result.get_kernel_time(KERNEL) * 1e6
                table[name][f"{fmt}:{mode}"].append(us)
                rows.append((name, f"{fmt}:{mode}", ps + 1, f"{us:.2f}"))
                line += f"  {fmt}:{mode} {us:8.1f}"
            print(line, flush=True)
        del data
        torch.cuda.empty_cache()
    print(f"gpu after: {gpu_state()}", flush=True)

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["shape", "config", "pass", "us"])
            w.writerows(rows)
    keys = [f"{f}:{m}" for f, m in configs]
    lines = ["| shape | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for name, row in table.items():
        lines.append(f"| {name} | " + " | ".join(f"{statistics.median(row[k]):.1f}" for k in keys) + " |")
    out = "\n".join(lines)
    print(out)
    if args.md:
        with open(args.md, "w", newline="\n") as fh:
            fh.write(out + "\n")


if __name__ == "__main__":
    main()
