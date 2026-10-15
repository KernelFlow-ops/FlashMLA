"""Fig. 6 - KV cache formats: byte layouts, what 4 bits cost in storage, writer choices, and accuracy against speed."""
import math

from figcommon import (BLUE, GRAY, GREEN, LIGHT, ORANGE, PINK, PURPLE, RED, TEAL, YELLOW, FILL, INK, STROKE, Axes, Sketch, dot, footer,
                       header, legend, text_width)

W, H = 2000, 1290
s = Sketch(W, H, seed=61)
header(s, 6, "KV cache formats: three ways to store a 512-dim token, and what each costs in accuracy",
       "The kernels read every format as stored. 4-bit formats halve the bytes; how much they lose is decided when the cache is written, "
       "not in the kernel.")

# ============================================================================================ A: byte layouts
AX, AY, AW, AH = 24, 112, 1130, 520
s.group(AX, AY, AW, AH, "A  Bytes per token (K = V: one row serves both)", stroke=BLUE, size=17)
fmts = [("V4.1 FP8 (upstream)", 512, "512 x e4m3", 16, "16 x ue8m0", "one power-of-two scale per 32 dims", 528, "blue"),
        ("V4.1 FP4 (upstream, NVFP4-style)", 256, "512 x e2m1", 32, "32 x e4m3", "one e4m3 scale per 16 dims", 288, "green"),
        ("MXFP4 (new, sm120)", 256, "512 x e2m1", 16, "16 x ue8m0", "one power-of-two scale per 32 dims", 272, "orange")]
bx0 = AX + 30
bsc = 760 / 528
for i, (name, db, dlab, sb, slab, gran, tot, col) in enumerate(fmts):
    y = AY + 60 + i * 112
    s.text(bx0, y, name, 15.5, STROKE[col], bold=True)
    s.text(bx0 + text_width(name, 15.5, True) + 14, y, gran, 13.5, INK)
    s.rect(bx0, y + 12, db * bsc, 44, STROKE[col], FILL[col], 1.6, r=3, rough=0.5, hatch=LIGHT[col], hatch_gap=8)
    s.text(bx0 + db * bsc / 2, y + 40, f"{dlab}  ({db} B)", 14, STROKE[col], "middle", bold=True)
    sw = sb * bsc
    s.rect(bx0 + db * bsc, y + 12, sw, 44, STROKE["yellow"], FILL["yellow"], 1.6, r=2, rough=0.4)
    s.text(bx0 + db * bsc + sw + 10, y + 32, f"+ {slab} ({sb} B)", 13.5, STROKE["yellow"], bold=True)
    s.text(bx0 + db * bsc + sw + 10, y + 52, f"= {tot} B", 15, INK, bold=True)
# value grids of the element types
gy = AY + 404
s.line(AX + 20, gy - 24, AX + AW - 20, gy - 24, "#dee2e6", 1.0, rough=0.2, passes=1, dash="4 6")
s.text(bx0, gy, "element and scale types", 14.5, INK, bold=True)
types = [("e4m3", "4 significant bits, |x| <= 448", BLUE), ("e2m1", "values 0, 0.5, 1, 1.5, 2, 3, 4, 6", GREEN),
         ("ue8m0", "2^k only: the block max is rounded up to a power of two", ORANGE),
         ("e4m3 scale", "4 significant bits: the scale follows amax / 6 closely (NVFP4)", GREEN)]
for i, (t, d, c) in enumerate(types):
    x = bx0 + (i % 2) * 540
    y = gy + 28 + (i // 2) * 30
    s.text(x, y, t, 13.5, c, bold=True)
    s.text(x + 104, y, d, 13.5, INK)
s.text(bx0, AY + AH - 18, "The format is recognized by the last dimension of the cache tensor (528 / 288 / 272), as upstream does.", 13, GRAY)

# ============================================================================================ B: storage error
BX = AX + AW + 16
BW = W - 24 - BX
s.group(BX, AY, BW, AH, "B  Storage error alone (fp64 attention, cache vs bf16 KV)", stroke=GREEN, size=17)
ax = Axes(s, BX + 90, AY + 280, 300, 210, (0, 0.13), (0, 1))
for v in (0.05, 0.10):
    s.line(ax.x0, ax.Y(v / 0.13), ax.x0 + ax.w, ax.Y(v / 0.13), "#dee2e6", 1.0, rough=0.2, passes=1, dash="3 6")
    s.text(ax.x0 - 10, ax.Y(v / 0.13) + 4, f"{v:.2f}", 12.5, GRAY, "end")
s.line(ax.x0, ax.y0, ax.x0 + ax.w, ax.y0, INK, 1.6, rough=0.4)
s.line(ax.x0, ax.y0, ax.x0, ax.y0 - ax.h, INK, 1.4, rough=0.4)
bars = [("FP8", 0.026, "blue"), ("V4.1 FP4", 0.095, "green"), ("MXFP4", 0.115, "orange")]
for i, (lab, v, col) in enumerate(bars):
    x = ax.x0 + 22 + i * 96
    top = ax.Y(v / 0.13)
    s.rect(x, top, 64, ax.y0 - top, STROKE[col], FILL[col], 1.6, r=2, rough=0.4, hatch=LIGHT[col], hatch_gap=7)
    s.text(x + 32, top - 8, f"{v:.3f}".rstrip("0"), 13.5, STROKE[col], "middle", bold=True)
    s.text(x + 32, ax.y0 + 20, lab, 13, INK, "middle", bold=True)
s.text(ax.x0 - 64, ax.y0 - ax.h / 2, "out rel. RMS error", 13.5, INK, "middle", bold=True, rot=-90)
# right: NVFP4 vs MXFP4 by data class
rx = BX + 450
s.text(rx, AY + 66, "V4.1 FP4 (NVFP4-style) is more accurate", 14, GREEN, bold=True)
s.text(rx, AY + 86, "than MXFP4, for 5.9% more bytes:", 14, GREEN, bold=True)
cls = [("upstream test data", 17, 18), ("high-variance logits", 17, 18), ("spread of token amplitudes", 12, 12), ("outlier channels", 56, 56)]
csc = (BW - 470 - 70) / 60
for i, (lab, lo, hi) in enumerate(cls):
    y = AY + 118 + i * 58
    s.text(rx, y, lab, 13.5, INK)
    s.rect(rx, y + 8, hi * csc, 24, GREEN, FILL["green"], 1.4, r=2, rough=0.4, hatch=LIGHT["green"], hatch_gap=7)
    s.text(rx + hi * csc + 8, y + 26, f"{lo}-{hi}%" if lo != hi else f"{hi}%", 13.5, GREEN, bold=True)
s.lines(BX + 22, AY + 340, ["Smaller blocks (16 vs 32 dims) and scales with a mantissa both help; outlier channels",
                            "gain most because a 32-dim block shares its scale with the outlier.",
                            ("The kernel adds little on top: the exact modes nothing measurable, mxfp8 0.1-4.5%", INK, True),
                            ("on the V4.1 FP4 cache. FP8 storage is 3.6x more accurate than V4.1 FP4.", INK, True)], 13.5, lh=1.4,
        maxw=BW - 40)
s.text(BX + 22, AY + AH - 18, "Synthetic data (tests/eval-sm120-storage.py): it ranks formats, it does not predict model accuracy.", 13, GRAY)

# ============================================================================================ C: writer choices + two-level
CY = AY + AH + 16
CH = H - 34 - CY
CW = 820
s.group(AX, CY, CW, CH, "C  The cache writer decides the 4-bit error", stroke=ORANGE, size=17)
wr = [("block scale amax / 6 rounded to nearest e4m3", "better than rounding it up (upstream rounds to nearest)", (5, 11), "green"),
      ("Four Over Six: per block, largest magnitude 6 or 4", "on top of that; same bytes, only the writer changes", (8, 15), "green"),
      ("second, per-tensor 2^g scale (two-level NVFP4)", "nothing on data of normal range", None, "gray")]
for i, (t, sub, v, col) in enumerate(wr):
    y = CY + 58 + i * 72
    s.text(AX + 24, y, t, 14, INK, bold=True)
    s.text(AX + 24, y + 22, sub, 13.5, STROKE[col])
    if v:
        s.rect(AX + 570, y - 6, v[0] * 11, 22, STROKE[col], FILL[col], 1.4, r=2, rough=0.4, hatch=LIGHT[col], hatch_gap=7)
        s.rect(AX + 570 + v[0] * 11, y - 6, (v[1] - v[0]) * 11, 22, STROKE[col], "#ffffff", 1.2, r=2, rough=0.4, dash="4 3")
        s.text(AX + 578 + v[1] * 11, y + 11, f"{v[0]}-{v[1]}%", 13.5, STROKE[col], bold=True)
    else:
        s.text(AX + 570, y + 11, "0%", 13.5, GRAY, bold=True)
# range axis
ry = CY + 400
rx0, rx1 = AX + 60, AX + CW - 60
lo2, hi2 = -16, 16


def RX(e):
    return rx0 + (e - lo2) / (hi2 - lo2) * (rx1 - rx0)


s.text(AX + 24, ry - 76, "Where the second level matters: the typical magnitude of the cached values (log2)", 14, INK, bold=True)
s.rect(RX(-6), ry - 50, RX(11) - RX(-6), 40, GREEN, FILL["green"], 1.4, r=4, rough=0.4, hatch=LIGHT["green"], hatch_gap=8)
s.text((RX(-6) + RX(11)) / 2, ry - 24, "block scales fit e4m3: one level is enough (schematic)", 13, GREEN, "middle", bold=True)
s.line(rx0, ry, rx1, ry, INK, 1.6, rough=0.4)
for e in range(lo2, hi2 + 1, 4):
    s.line(RX(e), ry, RX(e), ry + 6, INK, 1.2, rough=0.2, passes=1)
    s.text(RX(e), ry + 22, f"2^{e}", 12, GRAY, "middle")
pts = [(-12, "x 2^-12", "one level: error 1.0 (scales underflow to 0)", "two-level: 0.226", -1), (0, "tests", "", "", 0),
       (14, "x 2^14", "one level: 0.421 (scales saturate)", "two-level: 0.225", 1)]
for e, lab, l1, l2, side in pts:
    dot(s, RX(e), ry, "red" if side else "gray", "d", r=7)
    s.text(RX(e), ry + 44, lab if side else "the tests' data", 13, RED if side else GRAY, "middle", bold=True)
    if side:
        x = RX(e) + (10 if side < 0 else -10)
        anc = "start" if side < 0 else "end"
        s.text(x - (40 if side < 0 else -40), ry + 70, l1, 13, RED, anc)
        s.text(x - (40 if side < 0 else -40), ry + 90, l2, 13, GREEN, anc, bold=True)
s.text(AX + 24, CY + CH - 18, "So the second level protects the range, not the precision; the API has no argument for it (open item).", 13,
       GRAY)

# ============================================================================================ D: accuracy vs speed
DX = AX + CW + 16
DW = W - 24 - DX
s.group(DX, CY, DW, CH, "D  Kernel error against speed (geomean over the 10 shapes of Fig. 7)", stroke=TEAL, size=17)
ax = Axes(s, DX + 110, CY + CH - 76, DW - 400, CH - 140, (0.9, 2.5), (3e-7, 1e-3), ylog=True)
ax.frame(xticks=[(1.0, "1.0x"), (1.5, "1.5x"), (2.0, "2.0x"), (2.5, "2.5x")],
         yticks=[(1e-6, "1e-6"), (1e-5, "1e-5"), (1e-4, "1e-4"), (1e-3, "1e-3")],
         xlabel="speedup over the P1 baseline (geomean)", ylabel="out cos_diff vs tests/ref.py")
ty = ax.Y(5e-6)
s.line(ax.x0, ty, ax.x0 + ax.w, ty, RED, 2.0, rough=0.3, passes=1, dash="8 5")
s.text(ax.x0 + ax.w - 6, ty - 8, "upstream tolerance 5e-6", 13, RED, "end", bold=True)
fy0, fy1 = ax.Y(4.2e-4), ax.Y(2.2e-4)
s.rect(ax.x0 + 2, fy0, ax.w - 4, fy1 - fy0, GRAY, "#f1f3f5", 1.0, r=2, rough=0.2, dash="4 4")
s.text(ax.x0 + 10, fy1 - 6, "FlashInfer default FP8 mode", 12.5, GRAY)
s.line(ax.X(1.0), ax.y0, ax.X(1.0), ax.y0 - ax.h, GRAY, 1.8, rough=0.3, passes=1, dash="6 4")
s.text(ax.X(1.0) + 8, ax.Y(6e-7), "bf16: exact", 13, GRAY, bold=True)
pts = [(1.45, 9e-7, "purple", "o", "fp8x2"), (1.47, 1.2e-6, "pink", "d", "mxfp4x4"), (2.06, 2.1e-4, "teal", "o", "mxfp8"),
       (1.87, 3.65e-4, "teal", "o", "mxfp8, FP4 extra"), (2.17, 4.2e-4, "teal", "s", "mxfp8"), (2.32, 2.0e-4, "teal", "d", "mxfp8")]
for x, y, col, sh, lab in pts:
    dot(s, ax.X(x), ax.Y(y), col, sh, r=8)
lbl = [(1.45, 9e-7, "fp8x2, FP8", -14, 22, "end"), (1.47, 1.2e-6, "mxfp4x4, MXFP4", 14, -10, "start"),
       (2.06, 2.1e-4, "mxfp8, FP8", -14, 24, "end"), (1.87, 3.65e-4, "mxfp8, FP8 + V4.1 FP4", -14, -12, "end"),
       (2.17, 4.2e-4, "mxfp8, V4.1 FP4", 14, -12, "start"), (2.32, 2.0e-4, "mxfp8, MXFP4", 14, 22, "start")]
for x, y, t, dx, dy, anc in lbl:
    s.text(ax.X(x) + dx, ax.Y(y) + dy, t, 12.5, INK, anc, bold=True)
# key
kx = ax.x0 + ax.w + 34
s.text(kx, CY + 64, "shape = main cache", 13.5, INK, bold=True)
for i, (sh, lab, col) in enumerate([("o", "FP8 (528 B)", "blue"), ("s", "V4.1 FP4 (288 B)", "green"), ("d", "MXFP4 (272 B)", "orange")]):
    dot(s, kx + 10, CY + 92 + i * 28, "gray", sh, r=7)
    s.text(kx + 28, CY + 97 + i * 28, lab, 13, INK)
s.text(kx, CY + 196, "colour = mode", 13.5, INK, bold=True)
for i, (lab, col) in enumerate([("fp8x2", "purple"), ("mxfp4x4", "pink"), ("mxfp8", "teal")]):
    dot(s, kx + 10, CY + 224 + i * 28, col, "o", r=7)
    s.text(kx + 28, CY + 229 + i * 28, lab, 13, INK)
s.lines(kx, CY + 330, ["Kernel error only: it is", "measured against the", "same quantized cache.", "A 4-bit cache adds its",
                       "storage error (panel B)", "on top."], 13, lh=1.38, maxw=DW - (kx - DX) - 16)

footer(s, "Storage errors: out relative RMS on synthetic data, 6 cases per class. Kernel errors: the upstream test's data; MXFP4 mxfp8 about 2e-4, "
          "mxfp4x4 the worst case (<= 1.2e-6).")
s.save("../fig6_formats.svg")
print("fig6 ok")
