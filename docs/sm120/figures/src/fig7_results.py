"""Fig. 7 - results: speedup of every fast configuration at the upstream test's shapes, how we got there, FlashInfer."""
from figcommon import (BLUE, GRAY, GREEN, LIGHT, ORANGE, PINK, PURPLE, RED, TEAL, YELLOW, FILL, INK, STROKE, Axes, Sketch, dot, footer,
                       header, legend, text_width)

W, H = 2000, 1330
s = Sketch(W, H, seed=71)
header(s, 7, "Results on one RTX 5090: 1.45-2.32x over the first sm120 kernel, 2.3-4.8x over FlashInfer at the test's shapes",
       "Decode kernel time per call (kineto, 8 GB L2 flush before every call), the shapes of tests/test-sparse-decode.py: s_q = 4, "
       "topk 128 (main) + 512 (extra).")

# ============================================================================================ A: grouped speedups
AX, AY, AW, AH = 24, 112, W - 48, 560
s.group(AX, AY, AW, AH, "A  Speedup over the P1 baseline (its time / ours); all columns from one session", stroke=TEAL, size=17)
shapes = ["h64 b64", "h64 b128", "h64 b256", "h64 b512", "h128 b64", "h128 b128", "h128 b256", "h128 b512", "h64 prefill", "h128 prefill"]
cfgs = [("fp8x2, FP8 main + FP8 extra", "purple", [1.43, 1.45, 1.43, 1.42, 1.46, 1.46, 1.45, 1.44, 1.46, 1.45], 1.45),
        ("mxfp4x4, MXFP4", "pink", [1.48, 1.47, 1.46, 1.46, 1.47, 1.48, 1.48, 1.47, 1.47, 1.45], 1.47),
        ("mxfp8, FP8 main + V4.1 FP4 extra", "teal", [1.79, 1.89, 1.84, 1.87, 1.85, 1.92, 1.91, 1.94, 1.87, 1.88], 1.87),
        ("mxfp8, FP8 main + FP8 extra", "blue", [1.92, 2.04, 2.01, 2.00, 2.09, 2.12, 2.10, 2.08, 2.11, 2.12], 2.06),
        ("mxfp8, V4.1 FP4 main + extra", "green", [2.17, 2.18, 2.16, 2.14, 2.20, 2.19, 2.19, 2.16, 2.19, 2.17], 2.17),
        ("mxfp8, MXFP4 main + extra", "orange", [2.27, 2.32, 2.31, 2.28, 2.37, 2.36, 2.34, 2.31, 2.30, 2.31], 2.32)]
ax = Axes(s, AX + 90, AY + AH - 70, AW - 380, AH - 150, (0, 2.6), (0, 1))
for v in (0.5, 1.0, 1.5, 2.0, 2.5):
    yy = ax.y0 - ax.h * v / 2.6
    s.line(ax.x0, yy, ax.x0 + ax.w, yy, "#dee2e6" if v != 1.0 else GRAY, 1.0 if v != 1.0 else 1.8, rough=0.2, passes=1,
           dash="3 6" if v != 1.0 else "8 5")
    s.text(ax.x0 - 10, yy + 4.5, f"{v:.1f}x", 12.5, GRAY, "end")
s.line(ax.x0, ax.y0, ax.x0 + ax.w, ax.y0, INK, 1.6, rough=0.4)
s.line(ax.x0, ax.y0, ax.x0, ax.y0 - ax.h, INK, 1.4, rough=0.4)
s.text(ax.x0 - 62, ax.y0 - ax.h / 2, "speedup", 14, INK, "middle", bold=True, rot=-90)
gw = ax.w / len(shapes)
bw = (gw - 22) / len(cfgs)
for j, sh in enumerate(shapes):
    gx = ax.x0 + j * gw + 11
    for i, (lab, col, vals, gm) in enumerate(cfgs):
        v = vals[j]
        top = ax.y0 - ax.h * v / 2.6
        s.rect(gx + i * bw, top, bw - 2, ax.y0 - top, STROKE[col], FILL[col], 1.2, r=1, rough=0.3, hatch=LIGHT[col], hatch_gap=6)
    s.text(ax.x0 + j * gw + gw / 2, ax.y0 + 22, sh, 13, INK, "middle", bold=True)
    if j == 7:
        s.text(ax.x0 + j * gw + gw / 2, ax.y0 + 40, "(production)", 12, GRAY, "middle")
s.text(ax.x0 + 8.0 * gw + gw, ax.y0 + 40, "b 1, s_q 4096", 12, GRAY, "middle")
# legend with geomeans
lx = ax.x0 + ax.w + 34
s.text(lx, AY + 60, "configuration (geomean)", 14, INK, bold=True)
for i, (lab, col, vals, gm) in enumerate(reversed(cfgs)):
    y = AY + 92 + i * 52
    s.rect(lx, y - 14, 26, 18, STROKE[col], FILL[col], 1.4, r=2, rough=0.4, hatch=LIGHT[col], hatch_gap=6)
    s.text(lx + 36, y, lab, 13, INK)
    s.text(lx + 36, y + 20, f"{gm:.2f}x", 14, STROKE[col], bold=True)
s.lines(lx, AY + 404, [("dashed line: bf16 (exact) = 1.00x", GRAY, True), ("exact modes: fp8x2, mxfp4x4", PURPLE, True),
                       ("approximate: mxfp8", TEAL, True),
                       "4-bit caches are compared with", "P1 on the caches they replace"], 13, lh=1.4, maxw=300)

# ============================================================================================ B: timeline
BY = AY + AH + 16
BH = H - 34 - BY
BW = 1150
s.group(AX, BY, BW, BH, "B  How the FP8-cache kernels got there (b = 512, FP8 extra; kernel time in us)", stroke=PURPLE, size=17)
steps = ["P1", "block-scaled\nMMAs", "producer\nwarpgroup", "adjacent\nhead CTAs", "consumer\npairs", "gather4", "today"]
series = {
    128: {"fp8x2": [1813, 1455, 1379.8, 1369.7, 1257.3, 1248.1, 1260.2], "mxfp8": [1813, 1224.5, 1011.8, 951.5, 923.5, 868.2, 872.2]},
    64: {"fp8x2": [946.3, 765.1, 722.8, None, 671.4, 663.6, 667.3], "mxfp8": [946.3, 631.7, 518.7, None, 507.3, 471.6, 474.4]},
}
sw = (BW - 60) / 2
for k, h in enumerate((128, 64)):
    x0 = AX + 70 + k * sw
    ymax = 1900 if h == 128 else 1000
    a = Axes(s, x0, BY + BH - 92, sw - 70, BH - 170, (-0.4, 6.4), (0, ymax))
    for v in ([500, 1000, 1500] if h == 128 else [250, 500, 750]):
        yy = a.Y(v)
        s.line(a.x0, yy, a.x0 + a.w, yy, "#dee2e6", 1.0, rough=0.2, passes=1, dash="3 6")
        s.text(a.x0 - 8, yy + 4.5, str(v), 12, GRAY, "end")
    s.line(a.x0, a.y0, a.x0 + a.w, a.y0, INK, 1.5, rough=0.4)
    s.line(a.x0, a.y0, a.x0, a.y0 - a.h, INK, 1.3, rough=0.4)
    s.text(a.x0 + a.w / 2, BY + 52, f"h_q = {h}", 15, INK, "middle", bold=True)
    for i, st in enumerate(steps):
        for li, part in enumerate(st.split("\n")):
            s.text(a.X(i), a.y0 + 18 + li * 15, part, 11.5, GRAY, "middle")
    for mode, col in (("fp8x2", "purple"), ("mxfp8", "teal")):
        vals = series[h][mode]
        pts = [(a.X(i), a.Y(v)) for i, v in enumerate(vals) if v is not None]
        s.polyline(pts, STROKE[col], 2.4, rough=0.4, passes=2)
        for i, v in enumerate(vals):
            if v is None:
                continue
            dot(s, a.X(i), a.Y(v), col if i else "gray", "o", r=6)
        v = vals[1]
        if mode == "fp8x2":
            s.text(a.X(1) + 8, a.Y(v) - 12, f"{v:g}", 12.5, STROKE[col], bold=True)
        else:
            s.text(a.X(1) - 10, a.Y(v) + 16, f"{v:g}", 12.5, STROKE[col], "end", bold=True)
        v = vals[6]
        s.text(a.X(6), a.Y(v) + (-12 if mode == "fp8x2" else 22), f"{v:g}", 12.5, STROKE[col], "middle", bold=True)
    s.text(a.X(0) + 10, a.Y(series[h]["fp8x2"][0]) + 4, f"{series[h]['fp8x2'][0]:g}", 12.5, GRAY, bold=True)
    s.text(a.X(6), a.Y(series[h]["fp8x2"][6]) - 30, "fp8x2", 13, PURPLE, "middle", bold=True)
    s.text(a.X(6), a.Y(series[h]["mxfp8"][6]) + 42, "mxfp8", 13, TEAL, "middle", bold=True)
s.text(AX + 24, BY + BH - 18, "Each step was timed in its own session (a few % session to session); h_q = 64 has no number for the adjacency step.",
       13, GRAY)

# ============================================================================================ C: FlashInfer
CX = AX + BW + 16
CW = W - 24 - CX
s.group(CX, BY, CW, BH, "C  Against FlashInfer 0.7.1 (sparse_mla_sm120)", stroke=ORANGE, size=17)
rows = [("FP8 extra, h64 b64", 2.3, 3.8), ("FP8 extra, h64 b512", 2.9, 4.8), ("FP8 extra, h128 b64", 2.3, None),
        ("FP8 extra, h128 b512", 3.0, None), ("FP4 extra, h64 b64", 2.4, 3.6), ("FP4 extra, h64 b512", 3.1, 4.6),
        ("FP4 extra, h128 b64", 2.5, None), ("FP4 extra, h128 b512", 3.2, None)]
fx0 = CX + 220
fsc = (CW - 300) / 5.0
s.text(fx0, BY + 50, "how many times faster than FlashInfer", 13.5, INK, bold=True)
for i, (lab, ex, df) in enumerate(rows):
    y = BY + 72 + i * 40
    s.text(fx0 - 10, y + 22, lab, 13, INK, "end")
    s.rect(fx0, y + 6, ex * fsc, 14, GRAY, FILL["gray"], 1.2, r=1, rough=0.3, hatch=LIGHT["gray"], hatch_gap=5)
    s.text(fx0 + ex * fsc + 6, y + 18, f"{ex}x", 12, GRAY, bold=True)
    if df:
        s.rect(fx0, y + 21, df * fsc, 14, TEAL, FILL["teal"], 1.2, r=1, rough=0.3, hatch=LIGHT["teal"], hatch_gap=5)
        s.text(fx0 + df * fsc + 6, y + 33, f"{df}x", 12, TEAL, bold=True)
    else:
        s.text(fx0 + 6, y + 33, "FlashInfer's FP8 mode raises at h128", 11.5, RED)
legend(s, fx0, BY + 72 + 8 * 40 + 18, [("exact vs exact (bf16)", "gray", LIGHT["gray"]), ("our mxfp8 vs its default FP8 mode", "teal",
                                                                                           LIGHT["teal"])], size=12.5)
s.lines(CX + 22, BY + 72 + 8 * 40 + 46, ["Same data, same timing; ours from an earlier build (within 2%",
                                         "of panel A). At these sizes FlashInfer runs its prefill kernel.",
                                         ("Small batches: at T = b x s_q = 1 (and 4 with h64) FlashInfer is", INK, True),
                                         ("1.1-1.4x faster; from T = 16 our exact mode is 1.7-3.1x faster.", INK, True)], 13, lh=1.4, maxw=CW - 40)

footer(s, "One RTX 5090 (170 SMs), clocks not locked (2.8-2.9 GHz under load), nothing else on the GPU. Panel A: 4 samples per cell, never more "
          "than 0.8% apart. 'prefill' = b 1, s_q 4096.")
s.save("../fig7_results.svg")
print("fig7 ok")
