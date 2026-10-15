"""Fig. 2 - measured hardware limits of the RTX 5090 and the roofline of this operator."""
import math

from figcommon import (BLUE, GRAY, GREEN, LIGHT, ORANGE, PINK, PURPLE, RED, TEAL, FILL, INK, STROKE, Axes, Sketch, dot, footer,
                       header, text_width)

W, H = 2000, 1262
s = Sketch(W, H, seed=23)
header(s, 2, "The RTX 5090 measured: tensor-core ceilings, memory, latencies, and where the kernels sit",
       "All peaks come from micro-benchmarks in benchmarks/sm120/micro on this card (2.65-2.90 GHz). Kernel points: b = 512, s_q = 4, "
       "topk 128 + 512, times of Fig. 7.")

# ============================================================================================ A: roofline
AX, AY, AW, AH = 24, 112, 1110, 1112
s.group(AX, AY, AW, AH, "A  Roofline of sparse MLA decoding (log-log)", stroke=BLUE, size=17)
ax = Axes(s, 150, AY + AH - 120, 930, 880, (60, 3000), (60, 3000), xlog=True, ylog=True)
ax.frame(xticks=[(v, str(v)) for v in (100, 200, 500, 1000, 2000)], yticks=[(v, str(v)) for v in (100, 200, 500, 1000, 2000)],
         xlabel="FLOP per byte of gathered KV cache (Q, O and index traffic not counted)", ylabel="TFLOPS (upstream test's FLOP count)")
BW_TB = 1.53
ceil = [("FP4 block-scaled MMA  1959", 1959, GREEN), ("MXFP8 block-scaled MMA  963", 963, TEAL), ("BF16 MMA  252 (half rate)", 252, GRAY)]
for lab, pk, col in ceil:
    ridge = pk / BW_TB
    s.line(ax.X(ridge), ax.Y(pk), ax.X(3000), ax.Y(pk), col, 2.6 if col != GRAY or pk == 252 else 1.6, rough=0.4,
           dash="7 6" if pk == 495 else None)
    if pk == 963:
        s.text(ax.X(ridge) + 8, ax.Y(pk) + 22, lab + " TFLOPS", 14, col, bold=True)
    else:
        s.text(ax.X(3000) - 6, ax.Y(pk) - 9, lab + " TFLOPS", 14, col, "end", bold=True)
    if pk != 495:
        s.line(ax.X(ridge), ax.Y(pk) + 6, ax.X(ridge), ax.y0, col, 1.0, rough=0.3, passes=1, dash="2 5")
        s.text(ax.X(ridge) + 5, ax.y0 - 10, f"ridge {ridge:.0f}", 12, col)
s.line(ax.X(60), ax.Y(60 * BW_TB), ax.X(1959 / BW_TB), ax.Y(1959), ORANGE, 2.6, rough=0.4)
s.text(ax.X(75), ax.Y(75 * BW_TB) - 16, "gather limit 1.53 TB/s", 14, ORANGE, bold=True, rot=-43.5)
# intensity of each cache format x h_q
inten = [("FP8 h64", 248, "blue", 0), ("FP8 h128", 496, "blue", 0), ("V4.1 FP4 h64", 455, "green", 1), ("V4.1 FP4 h128", 910, "green", 1),
         ("MXFP4 h64", 482, "orange", 2), ("MXFP4 h128", 964, "orange", 2)]
for lab, x, col, lvl in inten:
    s.line(ax.X(x), ax.y0, ax.X(x), ax.Y(2500), STROKE[col], 1.2, rough=0.3, passes=1, dash="4 6", opacity=0.8)
    s.text(ax.X(x) + (-6 if lvl == 1 else 6), ax.Y(2900) + lvl * 20, lab, 12.5, STROKE[col], "end" if lvl == 1 else "start", bold=True)
# measured points (b = 512): (name, intensity h64, TFLOPS h64, intensity h128, TFLOPS h128, colour, h64 shape, h128 shape)
pts = [("exact (bf16), FP8 cache", 248, 181, 496, 189, "gray", "o", "s"),
       ("fp8x2, FP8 cache", 248, 257, 496, 273, "purple", "o", "s"),
       ("mxfp8, FP8 cache", 248, 362, 496, 394, "teal", "o", "s"),
       ("mxfp8, FP8 + V4.1 FP4 extra", 390, 338, 780, 367, "teal", "d", "d"),
       ("mxfp8, V4.1 FP4 main + extra", 455, 387, 910, 410, "green", "o", "s"),
       ("mxfp8, MXFP4 main + extra", 482, 412, 964, 438, "orange", "o", "s"),
       ("mxfp4x4, MXFP4 main + extra", 482, 264, 964, 279, "pink", "o", "s")]
for name, x1, y1, x2, y2, col, sh1, sh2 in pts:
    dot(s, ax.X(x1), ax.Y(y1), col, sh1, r=7)
    dot(s, ax.X(x2), ax.Y(y2), col, sh2, r=7)
# legend with the values
tx, ty, tw = 676, 836, 404
s.rect(tx, ty, tw, 232, GRAY, "#ffffff", 1.3, r=8, rough=0.4)
s.text(tx + 14, ty + 24, "kernel points, b = 512", 14, INK, bold=True)
s.text(tx + tw - 14, ty + 24, "TFLOPS  h64 / h128", 13, GRAY, "end")
for i, (name, x1, y1, x2, y2, col, sh1, sh2) in enumerate(pts):
    yy = ty + 50 + i * 25
    dot(s, tx + 22, yy - 5, col, sh1, r=6)
    dot(s, tx + 40, yy - 5, col, sh2, r=6)
    s.text(tx + 56, yy, name, 13, INK, maxw=250)
    s.text(tx + tw - 14, yy, f"{y1} / {y2}", 13, STROKE[col], "end", bold=True)
s.text(tx + 14, ty + 50 + 7 * 25 + 4, "first marker: h_q = 64, second: h_q = 128", 12, GRAY)
# take-aways
s.note(165, 330, 320, 140, ["Every format lies right of the", "BF16 ridge: the exact kernel is", "bound by the half-rate BF16 MMA",
                                        "(71-75% of 252 TFLOPS)."], title="exact kernel", size=13.5, rot=-1.0)
s.note(165, 498, 300, 150, ["FP8 at h_q = 64 lies left of the", "MXFP8 ridge: memory-bound, at", "91% of the gather limit by the",
                                       "upstream byte count (+ Q, O)."], title="mxfp8, FP8, h_q = 64", size=13.5, rot=0.8, fill="#e3fafc", stroke=TEAL)
s.note(874, 336, 206, 124, ["h_q = 128, 4-bit caches:", "no bound reached; rebase,", "softmax and handoffs", "take the rest."],
       title="the rest", size=13.5, rot=-0.8, fill="#fff4e6", stroke=ORANGE)

# ============================================================================================ B: memory
BX = AX + AW + 16
BW = W - 24 - BX
BY, BH = AY, 348
s.group(BX, BY, BW, BH, "B  Random-row gather into shared memory", stroke=ORANGE, size=17)
bars = [("528 B rows, 1 bulk copy / row, 16.5 KB in flight", 1282, 1282, "orange"),
        ("528 B rows, 1 bulk copy / row, >= 33 KB in flight", 1491, 1496, "orange"),
        ("288 B rows, 1 bulk copy / row, >= 33 KB in flight", 820, 1091, "red")]
gx0, gsc = BX + 24, (BW - 110) / 1800
for i, (lab, lo, hi, col) in enumerate(bars):
    y = BY + 48 + i * 58
    s.text(gx0, y + 4, lab, 13.5, INK, bold=True)
    s.rect(gx0, y + 12, hi * gsc, 22, STROKE[col], FILL[col], 1.5, r=2, rough=0.5, hatch=LIGHT[col], hatch_gap=7)
    txt = f"{lo} GB/s" if lo == hi else f"{lo}-{hi} GB/s"
    s.out.append(f'<rect x="{gx0 + 8}" y="{y + 15}" width="{text_width(txt, 13.5, True) + 10:.1f}" height="17" fill="{FILL[col]}" opacity="0.93"/>')
    s.text(gx0 + 13, y + 29, txt, 13.5, STROKE[col], bold=True)
for v, lab, col in ((1792, "DRAM 1792", RED), (1530, "limit 1530", ORANGE)):
    xx = gx0 + v * gsc
    s.line(xx, BY + 40, xx, BY + 210, col, 1.5, rough=0.3, passes=1, dash="5 5")
    s.text(xx + (-4 if v == 1530 else 4), BY + 52, lab, 12, col, "end" if v == 1530 else "start")
s.lines(gx0, BY + 236, ["Random rows are as fast as sequential ones. A bulk copy has a fixed cost, so short",
                         "rows lose bandwidth; tile::gather4 (4 rows per instruction) moves 288 B rows",
                         ("38% faster than one bulk copy per row.", GREEN, True)], 13.5, lh=1.4, maxw=BW - 40)

# ============================================================================================ C: latencies
CY, CH = BY + BH + 16, 418
s.group(BX, CY, BW, CH, "C  Inside one SM: what a warp waits for (clock cycles)", stroke=PURPLE, size=17)
lat = [("FADD / FFMA", 4.4, 4.4, "gray"), ("MUFU ex2", 18, 18, "gray"), ("warp shuffle", 29, 29, "gray"), ("block-scaled MMA", 29, 29, "teal"),
       ("shared-memory load", 50, 50, "gray"), ("named barrier, 2 warps", 66, 66, "purple"), ("one bulk copy issued (serial / SM)", 65, 95, "orange"),
       ("mbarrier arrive -> waiter runs", 165, 165, "purple"), ("32-row tile by bulk copies, SM idle", 3000, 3000, "orange"),
       ("same, consumers busy", 5100, 5100, "red"), ("same by 8 gather4, consumers busy", 1640, 1640, "green")]
lx0 = BX + 300
lsc = (BW - 300 - 110) / math.log10(8000)
for i, (lab, lo, hi, col) in enumerate(lat):
    y = CY + 38 + i * 32
    s.text(lx0 - 10, y + 17, lab, 13.5, INK, "end")
    wv = lsc * math.log10(hi) + 4
    s.rect(lx0, y + 5, wv, 18, STROKE[col], FILL[col], 1.4, r=2, rough=0.4)
    s.text(lx0 + wv + 8, y + 19, f"{lo:g}" if lo == hi else f"{lo:g}-{hi:g}", 13.5, STROKE[col], bold=True)
s.text(BX + 24, CY + CH - 14, "log scale. A softmax step is a latency chain; a tile of bulk copies keeps one warp busy for thousands of cycles.", 12.5,
       GRAY, maxw=BW - 40)

# ============================================================================================ D: utilisation
DY = CY + CH + 16
DH = AY + AH - DY
s.group(BX, DY, BW, DH, "D  How close each kernel gets (b = 512)", stroke=GREEN, size=17)
gauges = [("exact (bf16): 180-190 TFLOPS", "71-75% of the BF16 MMA peak", 0.71, 0.75, "gray"),
          ("mxfp8, FP8, h_q = 64: 1397 GB/s", "91% of the measured gather limit", 0.91, 0.91, "teal"),
          ("fp8x2: 510-540 TFLOPS of MMA work", "about half of the MXFP8 peak", 0.53, 0.56, "purple"),
          ("mxfp8, h_q = 128: 388 TFLOPS", "40% of the MXFP8 peak", 0.40, 0.40, "teal")]
gx, gw = BX + 24, BW - 100
for i, (t, sub, lo, hi, col) in enumerate(gauges):
    y = DY + 40 + i * 62
    s.text(gx, y, t, 13.5, INK, bold=True)
    s.rect(gx, y + 8, gw, 22, GRAY, "#f1f3f5", 1.2, r=3, rough=0.4)
    s.rect(gx, y + 8, gw * hi, 22, STROKE[col], FILL[col], 1.5, r=3, rough=0.5, hatch=LIGHT[col], hatch_gap=7)
    s.out.append(f'<rect x="{gx + 8}" y="{y + 11}" width="{text_width(sub, 13) + 10:.1f}" height="17" fill="{FILL[col]}" opacity="0.93"/>')
    s.text(gx + 13, y + 24, sub, 13, INK)
    s.text(gx + gw + 8, y + 24, "100%", 12.5, GRAY)

footer(s, "TFLOPS = 2048 h_q FLOP per attended token / kernel time, as tests/lib.py counts them. Panel D: the numbers of build a006306 "
          "(within 2% of today's times).")
s.save("../fig2_hardware.svg")
print("fig2 ok")
