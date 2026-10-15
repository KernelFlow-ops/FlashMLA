"""Fig. 4 - numerics: how Q, K, P and V get the scales of the block-scaled MMAs, and the error of every mode."""
import math

from figcommon import (BLUE, GRAY, GREEN, LIGHT, ORANGE, PINK, PURPLE, RED, TEAL, YELLOW, FILL, INK, STROKE, Axes, Sketch, footer,
                       header, text_width)

W, H = 2000, 1268
s = Sketch(W, H, seed=41)
header(s, 4, "Numerics: both GEMMs on block-scaled tensor cores, and what each mode costs in accuracy",
       "mma.sync m16n8k32 kind::mxf8f6f4.block_scale: A (16 x 32) and B (32 x 8) in e4m3, one ue8m0 scale per row of A and per column "
       "of B for every 32 values of k.")


def grid(x, y, rows, cols, cw, ch, col, fills=None, lw=1.2):
    for r in range(rows):
        for c in range(cols):
            f = fills(r, c) if fills else FILL[col]
            s.rect(x + c * cw, y + r * ch, cw, ch, STROKE[col], f, lw, r=0, rough=0.25, passes=1)


# ============================================================================================ A: QK
AX, AY, AW, AH = 24, 112, 950, 420
s.group(AX, AY, AW, AH, "A  QK: the cached K bytes are the B operand, as stored", stroke=BLUE, size=17)
# A operand: Q
grid(60, 190, 4, 8, 16, 16, "purple")
s.text(124, 176, "Q  (A: 16 heads x 32 dims)", 13.5, PURPLE, "middle", bold=True)
grid(196, 190, 4, 1, 16, 16, "yellow")
s.text(204, 268, "SF_A", 12.5, STROKE["yellow"], "middle", bold=True)
s.lines(60, 290, ["quantized once per CTA:", "e4m3 + one ue8m0 per", "32 dims (fp8x2: two terms)"], 13, lh=1.35)
s.text(240, 226, "x", 22, INK, "middle", bold=True)
# B operand: K rows
kx = 270
for r in range(8):
    s.rect(kx, 182 + r * 18, 200, 16, BLUE, FILL["blue"], 1.1, r=0, rough=0.25, passes=1)
    s.rect(kx + 200, 182 + r * 18, 16, 16, STROKE["yellow"], FILL["yellow"], 1.1, r=0, rough=0.25, passes=1)
s.text(kx + 100, 172, "8 cached K rows: 32 of 512 dims", 13.5, BLUE, "middle", bold=True)
s.text(kx + 208, 344, "SF_B", 12.5, STROKE["yellow"], "middle", bold=True)
s.lines(kx, 368, ["a token stores 512 e4m3 + 16 ue8m0:", "one scale per (token, 32 dims) =", "SF_B, one per column n and 32 of k.",
                  "ldmatrix reads the raw bytes."], 13, lh=1.35, maxw=320)
s.arrow([(530, 226), (590, 226)], INK, 2.0)
s.box(600, 170, 350, 116, "QMMA.SF.16832.F32.E4M3.E4M3.E8", ["block-scaled MMA, 4x the BF16 rate,", "FP32 accumulation (full FP32,", "measured)"],
      fill=FILL["teal"], stroke=TEAL, tsize=14, size=13.5)
s.lines(600, 312, [("Nothing is dequantized and K is never copied:", INK, True), "QK costs one MMA per 32 dims and 8 tokens.",
                   ("FP4 tiles: OMMA.SF.16864 with K's e2m1 bytes", GREEN, True), ("and e4m3 (V4.1 FP4) or ue8m0 (MXFP4) scales; Q as", GREEN),
                   ("two e2m1 terms (four in mxfp4x4).", GREEN)], 13.5, lh=1.38, maxw=360)
s.text(60, AY + AH - 16, "The same bytes serve as K here and, after a rebase, as V (K = V in MLA).", 13, GRAY)

# ============================================================================================ B: PV and the rebase
BX = AX + AW + 16
BW = W - 24 - BX
s.group(BX, AY, BW, AH, "B  PV: V's scales point the wrong way, so the producers rebase each tile once", stroke=GREEN, size=17)
s.lines(BX + 22, AY + 44, ["O = P V contracts over tokens: SF_B must be one scale per output dim and 32 tokens.",
                           "The cache has one scale per token and 32 dims: the wrong axis."], 13.5, lh=1.4, maxw=BW - 40)
# before
gx, gy = BX + 40, AY + 120
exps = [[3, 1, 4, 2], [5, 2, 4, 1], [2, 2, 3, 3], [4, 1, 5, 2], [3, 3, 4, 1], [5, 2, 2, 2]]
shade = {1: "#e7f5ff", 2: "#a5d8ff", 3: "#74c0fc", 4: "#4dabf7", 5: "#339af0"}
for r, row in enumerate(exps):
    for c, e in enumerate(row):
        s.rect(gx + c * 46, gy + r * 26, 46, 26, BLUE, shade[e], 1.1, r=0, rough=0.25, passes=1)
        s.text(gx + c * 46 + 23, gy + r * 26 + 18, f"e={e}", 11.5, INK, "middle")
s.text(gx + 92, gy - 12, "tile: 32 tokens x 16 groups", 13, BLUE, "middle", bold=True)
s.text(gx + 92, gy + 6 * 26 + 18, "scale per (token, group); 6 x 4 shown", 12.5, GRAY, "middle")
s.arrow([(gx + 200, gy + 78), (gx + 268, gy + 78)], GREEN, 2.4)
s.text(gx + 234, gy + 64, "rebase", 13.5, GREEN, "middle", bold=True)
# after
ax2 = gx + 280
emax = [max(r[c] for r in exps) for c in range(4)]
for r, row in enumerate(exps):
    for c, e in enumerate(row):
        s.rect(ax2 + c * 46, gy + r * 26, 46, 26, GREEN, FILL["green"], 1.1, r=0, rough=0.25, passes=1)
        d = emax[c] - e
        s.text(ax2 + c * 46 + 23, gy + r * 26 + 18, ">>" + str(d) if d else "=", 11.5, INK, "middle")
for c in range(4):
    s.rect(ax2 + c * 46, gy - 34, 46, 22, STROKE["yellow"], FILL["yellow"], 1.2, r=0, rough=0.25, passes=1)
    s.text(ax2 + c * 46 + 23, gy - 18, f"2^{emax[c]}", 12, STROKE["yellow"], "middle", bold=True)
s.text(ax2 + 92, gy + 6 * 26 + 18, "one SF_B per (tile, group)", 12.5, GRAY, "middle")
s.lines(ax2 + 214, gy - 20, ["per (tile, 32-dim group): e_max over the", "valid tokens becomes the column scale;",
                             "each byte's exponent drops by e_max - e_t", ("exact unless a value becomes subnormal", INK, True),
                             ("in e4m3 (then rounded to nearest)", INK, True), "integer instructions only, once per tile",
                             "for all 64 or 128 heads, by producer", "warps 1-3, into a separate V buffer"], 13, lh=1.36, maxw=BW - 520)
s.note(BX + 30, AY + AH - 86, BW - 60, 64, ["FlashInfer instead folds V's scales into P and runs PV on the unscaled e4m3 MMA, at half the block-scaled rate."],
       size=13.5, rot=-0.4, fill="#f8f9fa", stroke=GRAY)

# ============================================================================================ C, D, E
MY, MH = AY + AH + 16, 316
CW3 = (W - 48 - 32) / 3
panels = [("C  P: one e4m3 term, rowsum in FP32", PURPLE), ("D  fp8x2: two e4m3 terms = 8 bits", PURPLE), ("E  4-bit tiles: convert V once", ORANGE)]
for i, (t, col) in enumerate(panels):
    s.group(AX + i * (CW3 + 16), MY, CW3, MH, t, stroke=col, size=17)
px = AX + 22
s.lines(px, MY + 46, ["P = exp2(S - m) with the lazy rescale of upstream:",
                      "m only moves when the row max grows past the",
                      "threshold, so P stays below 2^6.",
                      ("e4m3(4 P) <= 256 < 448;  SF_A = 2^-2 undoes the 4", PURPLE, True),
                      "",
                      "The row sum (hence lse) uses the FP32 P: summing",
                      "the quantized P misses the upstream lse tolerance",
                      "(measured in the float64 emulator).",
                      "",
                      ("fp8x2 adds P_lo = e4m3(16 (4P - hi)), SF_A = 2^-6.", PURPLE)], 13.5, lh=1.38, maxw=CW3 - 40)
dx = AX + CW3 + 16 + 22
s.lines(dx, MY + 46, ["x  =  hi + lo,   hi = e4m3(x),   lo = e4m3(16 (x - hi)) / 16",
                      "4 significant bits each -> 8 bits, as many as bf16.", ""], 13.5, lh=1.38, maxw=CW3 - 40)
terms = [("S", "Q_hi K  +  Q_lo K"), ("O", "P_hi V  +  P_lo V")]
for k, (lhs, rhs) in enumerate(terms):
    y = MY + 118 + k * 46
    s.rect(dx, y - 22, CW3 - 44, 34, PURPLE, FILL["purple"], 1.4, r=6, rough=0.5)
    s.text(dx + 14, y, f"{lhs}  =  {rhs}", 15, PURPLE, bold=True)
s.lines(dx, MY + 226, ["K and V stay single terms: they are the stored values.",
                       "Two block-scaled MMAs per product still take half the",
                       "cycles of one BF16 MMA.",
                       ("Passes the upstream test, 1534 / 1534 (cos_diff 9e-7).", PURPLE, True)], 13.5, lh=1.38, maxw=CW3 - 40)
ex = AX + 2 * (CW3 + 16) + 22
s.lines(ex, MY + 46, [("QK: real FP4 MMAs (both operands e2m1).", GREEN, True),
                      "V4.1 FP4: kind::mxf4nvf4, ue4m3 per 16 dims;",
                      "MXFP4: kind::mxf4, ue8m0 per 32 dims.",
                      "",
                      ("PV: the producers turn e2m1 x scale into e4m3", ORANGE, True),
                      "with one power of two per (tile, group), then",
                      "the PV path of panel B.",
                      "MXFP4: an exponent shift, exact up to subnormals.",
                      "V4.1 FP4: e2m1 x e4m3 rounds once to e4m3 -",
                      "why fp8x2 runs the bf16 kernel on these tiles."], 13.5, lh=1.38, maxw=CW3 - 40)

# ============================================================================================ F: error ladder
FY = MY + MH + 16
FH = H - 30 - FY
s.group(AX, FY, W - 48, FH, "F  Output error of every mode (cos_diff against tests/ref.py, which dequantizes the same cache; log scale)",
        stroke=RED, size=17)
ax = Axes(s, AX + 520, FY + FH - 56, W - 48 - 560, FH - 110, (3e-7, 1e-3), (0, 1), xlog=True)
ticks = [(1e-6, "1e-6"), (1e-5, "1e-5"), (1e-4, "1e-4"), (1e-3, "1e-3")]
for v, lab in ticks:
    s.line(ax.X(v), ax.y0, ax.X(v), ax.y0 - ax.h, "#dee2e6", 1.0, rough=0.2, passes=1, dash="3 6")
    s.text(ax.X(v), ax.y0 + 20, lab, 12.5, GRAY, "middle")
s.line(ax.x0, ax.y0, ax.x0 + ax.w, ax.y0, INK, 1.6, rough=0.4)
rows = [("fp8x2, FP8 cache", 8.6e-7, 9.0e-7, "purple", None),
        ("mxfp4x4, MXFP4 cache", 3e-7, 1.2e-6, "pink", "<= 1.2e-6; lse of 1-token rows 3.4e-6"),
        ("mxfp8, FP8 cache", 2.0e-4, 2.2e-4, "teal", None),
        ("mxfp8, MXFP4 cache", 2.0e-4, 2.0e-4, "teal", "about 2e-4"),
        ("mxfp8, FP8 + V4.1 FP4 extra", 3.6e-4, 3.7e-4, "teal", None),
        ("mxfp8, V4.1 FP4 main", 4.2e-4, 4.2e-4, "teal", "median 4.2e-4"),
        ("FlashInfer default (FP8)", 2.2e-4, 4.2e-4, "gray", "2.2-4.2e-4")]
rh = (ax.h - 10) / len(rows)
for i, (lab, lo, hi, col, note) in enumerate(rows):
    y = ax.y0 - ax.h + 6 + i * rh
    s.text(ax.x0 - 14, y + rh / 2 + 5, lab, 13.5, INK, "end", bold=True)
    x1, x2 = ax.X(lo), ax.X(hi)
    if x2 - x1 < 10:
        x1, x2 = x1 - 5, x2 + 5
    s.rect(x1, y + 5, x2 - x1, rh - 10, STROKE[col], FILL[col], 1.5, r=4, rough=0.4, hatch=LIGHT[col], hatch_gap=6)
    s.text(x2 + 10, y + rh / 2 + 5, note or (f"{lo:.1e}" if lo == hi else f"{lo:.1e} - {hi:.1e}").replace("e-0", "e-"), 13,
           STROKE[col], bold=True)
tol = ax.X(5e-6)
s.line(tol, ax.y0 + 4, tol, ax.y0 - ax.h - 6, RED, 2.2, rough=0.3, passes=1, dash="8 5")
s.text(tol + 6, ax.y0 - ax.h + 2, "upstream tolerance 5e-6", 13.5, RED, bold=True)
s.lines(AX + 22, FY + 44, [("exact (bf16): the arithmetic of the", INK, True), "reference; 1534 / 1534 upstream cases",
                           "and 860 / 860 V4.1 FP4 main-cache cases.", "",
                           ("mxfp8 fails the 5e-6 out tolerance by", INK, True), "design (its lse passes); it is in the class",
                           "of FlashInfer's default FP8 mode."], 13.5, lh=1.38, maxw=440)

footer(s, "Significant bits: e4m3 4, e2m1 2; ue8m0 is a power of two. Errors on the upstream test's data; fp8x2 also stays within the "
          "tolerance on the high-variance data of tests/eval-sm120-modes.py.")
s.save("../fig4_numerics.svg")
print("fig4 ok")
