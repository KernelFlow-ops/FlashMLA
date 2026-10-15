"""Fig. 1 - the operator, why sm120 needs its own kernel, the mode x cache-format map, headline speedups."""
from figcommon import (BLUE, GRAY, GREEN, LIGHT, ORANGE, PINK, PURPLE, RED, TEAL, FILL, INK, STROKE, Sketch, footer, header,
                       text_width)

W, H = 2000, 1346
s = Sketch(W, H, seed=11)
header(s, 1, "DeepSeek-V4.1 sparse MLA decoding on the RTX 5090 (sm_120): operator, constraints, kernel family",
       "This branch adds the first sm120 implementation of flash_mla_with_kvcache. The default stays exact; faster arithmetic is opt-in "
       "through FLASH_MLA_SM120_DECODE_MODE.")

# ============================================================================================ A: the operator
AX, AY, AW, AH = 24, 112, 1010, 566
s.group(AX, AY, AW, AH, "A  The operator: one decoding step of one layer", stroke=BLUE, size=17)
s.box(50, 152, 260, 92, "q  (bf16)", ["[b, s_q, h_q, 512]", "h_q = 64 or 128 heads"], fill=FILL["blue"], stroke=BLUE)
s.box(50, 262, 260, 112, "indices", ["main:  [b, s_q, 128]", "extra: [b, s_q, 512]", "-1 = no token"], fill=FILL["gray"], stroke=GRAY)
s.box(50, 392, 260, 72, "attn_sink, topk_length", ["per head / per query (optional)"], fill=FILL["gray"], stroke=GRAY, tsize=15)
s.box(350, 152, 280, 132, "main KV cache", ["paged, 256 tokens per page", "one token = 512 dims, quantized:", "528 B (FP8), 288 B (V4.1 FP4)", "or 272 B (MXFP4)"],
      fill=FILL["yellow"], stroke=STROKE["yellow"], size=13.5)
s.box(350, 302, 280, 112, "extra KV cache", ["paged, 64 tokens per page", "FP8 or V4.1 FP4 next to an FP8", "main cache; else as the main one"],
      fill=FILL["yellow"], stroke=STROKE["yellow"], size=13.5)
s.arrow([(310, 318), (350, 318)], GRAY, 1.8)
s.arrow([(310, 290), (330, 290), (330, 220), (350, 220)], GRAY, 1.8)
s.text(355, 445, "gather 128 + 512 = 640 rows", 14, ORANGE, bold=True)
s.text(355, 465, "per query token (sparse top-k)", 14, ORANGE, bold=True)
# compute column
cx = 680
steps = [("S = q K^T", "contract 512 dims, h_q x 640 scores", BLUE),
         ("softmax (+ sink)", "online, log2 domain; lse per head", PURPLE),
         ("O = P V", "V = K: the same 512-dim token", BLUE)]
for i, (t, sub, col) in enumerate(steps):
    y = 152 + i * 96
    s.box(cx, y, 320, 74, t, [sub], fill=FILL["blue" if col == BLUE else "purple"], stroke=col, tsize=17, size=13.5)
    if i:
        s.arrow([(cx + 160, y - 22), (cx + 160, y)], INK, 1.8)
s.arrow([(630, 230), (680, 190)], ORANGE, 2.0)
s.arrow([(630, 360), (680, 375)], ORANGE, 2.0)
s.box(cx, 452, 320, 64, "out [b, s_q, h_q, 512] bf16", ["+ lse [b, s_q, h_q] fp32"], fill=FILL["green"], stroke=GREEN, tsize=15.5, size=13.5)
s.arrow([(cx + 160, 418), (cx + 160, 452)], INK, 1.8)
# launch chain
ly = 590
chain = [("get_mla_metadata", "upstream (smxx)", GRAY, FILL["gray"]), ("sparse decode kernel", "this work: sm120", RED, FILL["red"]),
         ("combine", "upstream; only when split", GRAY, FILL["gray"])]
for i, (t, sub, col, fill) in enumerate(chain):
    x = 50 + i * 330
    s.box(x, ly - 30, 280, 62, t, [sub], fill=fill, stroke=col, tsize=15.5, size=13)
    if i:
        s.arrow([(x - 50, ly), (x, ly)], INK, 1.8)
s.text(50, ly - 44, "launch chain of one call (all three can be captured in one CUDA graph):", 13.5, GRAY)
s.text(50, AY + AH - 14, "Work per gathered token: 2 x 512 x 2 x h_q = 2048 h_q FLOP (QK and PV).  h_q = 128 runs as two 64-head CTAs.", 13.5, INK)

# ============================================================================================ B: constraints
BX, BW = 1050, 926
s.group(BX, AY, BW, AH, "B  Why upstream cannot run here, and what sm120 offers (measured)", stroke=RED, size=17)
s.box(BX + 24, 152, 360, 196, "upstream kernel (sm100)", ["tcgen05 MMA + tensor memory", "(TMEM): not on sm120", "needs more shared memory than", "the 101,376 B of an sm120 block",
                                                     ("-> nothing in upstream runs on", RED, True), ("    an RTX 50 GPU", RED, True)],
      fill=FILL["red"], stroke=RED, size=14, align="left")
s.box(BX + 404, 152, 498, 196, "sm120 (RTX 5090, 170 SMs, 32 GB)", ["warp-level mma.sync only (no wgmma / tcgen05)",
                                                                  "block-scaled MMA: MXFP8 and FP4 kinds",
                                                                  "101,376 B shared memory per block",
                                                                  "TMA incl. tile::gather4, mbarrier, setmaxnreg",
                                                                  "random-row gather: 1.53 TB/s (85% of 1.79)",
                                                                  "BF16 MMA at half rate (GeForce)"],
      fill=FILL["green"], stroke=GREEN, size=14, align="left")
# MMA rate bars
by = 392
s.text(BX + 24, by, "MMA work per clock and SM, measured (FP32 accumulation):", 15, INK, bold=True)
rates = [("BF16  (HMMA)", 511, "1x", GRAY, "252 TFLOPS"), ("e4m3, unscaled  (QMMA)", 1023, "2x", GRAY, "495 TFLOPS"),
         ("MXFP8 block-scaled  (QMMA.SF)", 2040, "4x", TEAL, "954-972 TFLOPS"), ("FP4 block-scaled  (OMMA.SF)", 4090, "8x", GREEN, "1949-1969 TFLOPS")]
bx0, bscale = BX + 290, 300 / 4096
for i, (lab, v, mult, col, tf) in enumerate(rates):
    y = by + 22 + i * 36
    s.text(bx0 - 12, y + 18, lab, 13.5, INK, "end")
    key = "teal" if col == TEAL else "green" if col == GREEN else "gray"
    s.rect(bx0, y + 4, max(v * bscale, 8), 22, col, FILL[key], 1.5, r=2, rough=0.5, hatch=LIGHT[key], hatch_gap=7)
    s.text(bx0 + v * bscale + 10, y + 20, f"{v} FLOP  {mult}  = {tf}", 13.5, col, bold=True)
s.lines(BX + 24, by + 186, ["e2m1 placed in the 8-bit kinds runs at 1023: only the FP4 kinds are real FP4 (both operands e2m1).",
                            ("The exact kernel is bound by the half-rate BF16 MMA; the block-scaled kernels are bound by data", INK),
                            ("movement, rescaling and softmax, not by the tensor cores (Fig. 2, 3).", INK)], 13.5, GRAY, lh=1.45, maxw=BW - 40)

# ============================================================================================ C: modes x formats
CX, CY, CW, CH = 24, 702, 1310, 600
s.group(CX, CY, CW, CH, "C  One API, four arithmetic modes, four cache combinations: which kernel runs, how accurate it is, how fast", stroke=PURPLE, size=17)
modes = [("bf16 (default)", "gray"), ("fp8x2", "purple"), ("mxfp8", "teal"), ("mxfp4x4", "pink")]
rows = [("FP8 main + FP8 extra", "the upstream formats", "blue"),
        ("FP8 main + V4.1 FP4 extra", "the upstream formats", "blue"),
        ("V4.1 FP4 main", "+ V4.1 FP4 extra; sm120 only", "green"),
        ("MXFP4 main", "+ MXFP4 extra; new format", "orange")]
E = ("error", ["not offered for this cache"], "red")
cells = [
    [("bf16.cu", ["exact (reference arithmetic)", "1.00x"], "gray"), ("mxfp8.cu <2 terms>", ["upstream tolerance, cos 9e-7", "1.45x"], "purple"),
     ("mxfp8.cu <1 term>", ["cos_diff 2.0-2.2e-4", "2.06x"], "teal"), ("error", ["MXFP4 caches only"], "red")],
    [("bf16.cu", ["exact", "1.00x"], "gray"), ("runs bf16.cu", ["exact", "1.00x"], "gray"),
     ("mxfp8.cu <1, FP4 extra>", ["cos_diff 3.6-3.7e-4", "1.87x"], "teal"), ("error", ["MXFP4 caches only"], "red")],
    [("bf16.cu <MAIN_FP4>", ["exact", "1.00x"], "gray"), ("runs bf16.cu", ["exact", "1.00x"], "gray"),
     ("mxfp8.cu <MAIN_FP4>", ["cos_diff 4.2e-4 (median)", "2.17x"], "teal"), ("error", ["MXFP4 caches only"], "red")],
    [("error", ["no path meets the upstream", "lse tolerance"], "red"), ("error", ["same"], "red"),
     ("mxfp4.cu <Q x2, P x1>", ["cos_diff about 2e-4", "2.32x"], "teal"), ("mxfp4.cu <Q x4, P x2>", ["out within tolerance; lse", "of 1-token rows 3.4e-6", "1.47x"], "pink")]]
lx, cw, rh, ty = CX + 22, 254, 112, CY + 46
s.text(lx, ty + 22, "cache  \\  mode", 15, GRAY, bold=True)
for j, (m, col) in enumerate(modes):
    x = lx + 270 + j * cw
    s.badge(x + cw / 2 - 6, ty + 16, m, FILL[col], STROKE[col], 15)
for i, (r1, r2, fcol) in enumerate(rows):
    y = ty + 44 + i * rh
    s.rect(lx, y, 256, rh - 12, STROKE[fcol], FILL[fcol], 1.6, r=8, rough=0.6)
    s.text(lx + 12, y + 34, r1, 15.5, STROKE[fcol], bold=True, maxw=236)
    s.text(lx + 12, y + 58, r2, 13.5, INK, maxw=236)
    for j in range(4):
        t, ls, col = cells[i][j]
        x = lx + 270 + j * cw
        hatch = LIGHT["red"] if col == "red" else None
        fill = "#fff5f5" if col == "red" else FILL[col]
        s.rect(x, y, cw - 14, rh - 12, STROKE[col], fill, 1.5, r=8, rough=0.6, hatch=hatch, hatch_gap=12)
        speed = ls[-1] if ls[-1].endswith("x") and col != "red" else None
        body = ls[:-1] if speed else ls
        s.text(x + 10, y + 26, t, 14.5, STROKE[col], bold=True, maxw=cw - 30)
        if speed:
            s.text(x + cw - 26, y + rh - 22, speed, 18, STROKE[col], "end", bold=True)
        for k, l in enumerate(body):
            s.text(x + 10, y + 52 + k * 21, l, 13.5, INK, maxw=cw - 30)
s.text(lx, CY + CH - 16, "Speed: geometric mean over the 10 production shapes against the first sm120 kernel on the upstream cache it replaces (Fig. 7).",
       13.5, GRAY)

# ============================================================================================ D: headline speedups
DX = CX + CW + 16
DW = W - 24 - DX
s.group(DX, CY, DW, CH, "D  Speedup over the first sm120 kernel", stroke=GREEN, size=17)
bars = [("bf16 (exact), every format", 1.00, "gray", None), ("fp8x2, FP8 cache", 1.45, "purple", None), ("mxfp4x4, MXFP4 cache", 1.47, "pink", None),
        ("mxfp8, FP8 + V4.1 FP4 extra", 1.87, "teal", LIGHT["blue"]), ("mxfp8, FP8 cache", 2.06, "teal", None),
        ("mxfp8, V4.1 FP4 main", 2.17, "teal", LIGHT["green"]), ("mxfp8, MXFP4 cache", 2.32, "teal", LIGHT["orange"])]
x0 = DX + 236
sc = (DW - 236 - 70) / 2.4
for i, (lab, v, col, hatch) in enumerate(bars):
    y = CY + 52 + i * 52
    s.text(x0 - 10, y + 20, lab, 13.5, INK, "end")
    s.rect(x0, y + 4, v * sc, 26, STROKE[col], FILL[col], 1.6, r=2, rough=0.5, hatch=hatch, hatch_gap=7)
    s.text(x0 + v * sc + 8, y + 23, f"{v:.2f}x", 15, STROKE[col], bold=True)
s.line(x0 + sc, CY + 44, x0 + sc, CY + 52 + 7 * 52, RED, 1.5, rough=0.3, passes=1, dash="5 5")
s.lines(DX + 22, CY + 440, ["geometric mean, 10 production shapes; 1.00x = the first",
                            "working sm120 kernel (exact, KV dequantized to bf16)",
                            ("FlashInfer 0.7.1 on the same data:", INK, True),
                            "its exact mode is 2.3-3.2x slower than ours, its FP8 mode",
                            "3.6-4.8x slower than our mxfp8 (h_q = 64); at h_q = 128",
                            "it has no FP8 kernel for these shapes"], 13.5, lh=1.38, maxw=DW - 40)

footer(s, "RTX 5090, CUDA 13.0, torch 2.12 (cu130). Kernel times from the upstream test's method (kineto, 8 GB L2 flush before every call); "
          "errors against tests/ref.py, which dequantizes the same cache.")
s.save("../fig1_overview.svg")
print("fig1 ok")
