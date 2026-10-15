"""Fig. 5 - the memory path: TMA tile::gather4, row layouts in shared memory, bank positions, measured gains."""
from figcommon import (BLUE, GRAY, GREEN, LIGHT, ORANGE, PINK, PURPLE, RED, TEAL, YELLOW, FILL, INK, STROKE, Sketch, footer, header,
                       text_width)

W, H = 2000, 1250
s = Sketch(W, H, seed=53)
header(s, 5, "Feeding the tensor cores from a sparse, paged cache: TMA tile::gather4 and bank-aware row layouts",
       "Every 32-token tile is 32 rows at random pages. One producer warp issues all copies, so the cost per instruction, "
       "not DRAM, limited the fast modes.")

# ============================================================================================ A: issue cost
AX, AY, AW, AH = 24, 112, 1100, 420
s.group(AX, AY, AW, AH, "A  Issue cost of one 32-row tile for the gather warp (clock cycles, 528 B rows)", stroke=ORANGE, size=17)
x0, sc = AX + 210, (AW - 320) / 5200
ticks = [0, 1000, 2000, 3000, 4000, 5000]
base = AY + 330
s.line(x0, base, x0 + 5100 * sc, base, INK, 1.6, rough=0.4)
for t in ticks:
    s.line(x0 + t * sc, base, x0 + t * sc, base + 6, INK, 1.2, rough=0.2, passes=1)
    s.text(x0 + t * sc, base + 22, str(t), 12.5, GRAY, "middle")
lanes = [("32 x cp.async.bulk", "SM otherwise idle", 3000, 32, "orange"), ("32 x cp.async.bulk", "consumers busy", 5100, 32, "red"),
         ("8 x tile::gather4", "consumers busy", 1640, 8, "green")]
for i, (lab, sub, total, n, col) in enumerate(lanes):
    y = AY + 64 + i * 82
    s.text(x0 - 12, y + 16, lab, 14, INK, "end", bold=True)
    s.text(x0 - 12, y + 35, sub, 13, GRAY, "end")
    w = total * sc / n
    for k in range(n):
        s.rect(x0 + k * w, y + 4, w - 1.5, 36, STROKE[col], FILL[col], 1.0, r=1, rough=0.25, passes=1)
    s.text(x0 + total * sc + 10, y + 28, f"{total} clk", 14, STROKE[col], bold=True)
s.text(x0 + 2550 * sc, base + 46, "a bulk copy costs 65-95 clk whatever its size, serialized in the SM; a gather4 costs about as much for 4 rows",
       13.5, INK, "middle")

# ============================================================================================ B: one gather4
BX = AX + AW + 16
BW = W - 24 - BX
s.group(BX, AY, BW, AH, "B  One gather4: 4 random rows -> one 128 B-aligned group", stroke=GREEN, size=17)
gx, gy = BX + 28, AY + 58
for r in range(9):
    sel = r in (1, 4, 6, 8)
    s.rect(gx, gy + r * 24, 220, 20, STROKE["yellow"] if not sel else GREEN, FILL["yellow"] if not sel else FILL["green"], 1.1, r=0,
           rough=0.25, passes=1)
s.text(gx + 110, gy - 10, "KV cache rows (global)", 13, STROKE["yellow"], "middle", bold=True)
s.text(gx + 230, gy + 24 + 15, "row 7031", 12, GREEN)
s.text(gx + 230, gy + 96 + 15, "row 112", 12, GREEN)
s.text(gx + 230, gy + 144 + 15, "row -1: zeros", 12, RED, bold=True)
s.text(gx + 230, gy + 192 + 15, "row 50417", 12, GREEN)
s.box(gx + 352, gy + 6, 410, 74, "cp.async.bulk.tensor.2d ... tile::gather4", ["tensor map [rows][row bytes / 4] UINT32,", "box {width / 4, 1}, 4 row coordinates"],
      fill=FILL["teal"], stroke=TEAL, tsize=13.5, size=13)
s.arrow([(gx + 320, gy + 90), (gx + 380, gy + 120)], GREEN, 2.0)
for r in range(4):
    s.rect(gx + 400, gy + 116 + r * 22, 300, 20, GREEN, FILL["green"] if r != 2 else "#ffffff", 1.2, r=0, rough=0.25, passes=1)
s.rect(gx + 700, gy + 116, 50, 88, GRAY, "#f1f3f5", 1.1, r=0, rough=0.25, passes=1)
s.text(gx + 725, gy + 166, "gap", 12, GRAY, "middle")
s.text(gx + 550, gy + 228, "shared memory: one group, start 128 B aligned", 13, GREEN, "middle", bold=True)
s.lines(BX + 24, AY + AH - 100, ["Row coordinate -1 (an invalid token) delivers zeros and still counts toward the mbarrier's",
                                 "complete_tx bytes: the cache's NaN rows are never read and nothing hangs.",
                                 ("Rows whose page stride is not a whole number of rows fall back to bulk copies.", GRAY)],
        13.5, lh=1.4, maxw=BW - 40)

# ============================================================================================ C: layouts and banks
CY, CH = AY + AH + 16, 664
CW = 1320
s.group(AX, CY, CW, CH, "C  Row layouts in shared memory and the 16 B bank position of each row (128 B line = 8 positions)", stroke=BLUE, size=17)
fmts = [("FP8 (528 B rows)", "group = 4 x 528 + 64 = 2176 B (17 x 128)", 528, 528, 2176, 0, "blue",
         "4 positions, the same in every group (2-way conflicts); the rebase reads a group's two halves in opposite order"),
        ("V4.1 FP4 main cache (288 B rows)", "box 352 B per row, group 1408 B (11 x 128), odd groups 16 B later", 288, 352, 1408, 16, "green",
         "8 different positions: one ldmatrix (8 rows of two groups) is conflict-free"),
        ("MXFP4 (272 B rows, mxfp8 mode)", "box 288 B per row, group 1152 B (9 x 128), odd groups 16 B later", 272, 288, 1152, 16, "orange",
         "8 different positions: conflict-free")]
rowcols = ["#1864ab", "#2b8a3e", "#e67700", "#c92a2a", "#5f3dc4", "#0b7285", "#a61e4d", "#495057"]
for i, (name, sub, rb, stride, gstride, shift, col, verdict) in enumerate(fmts):
    y = CY + 46 + i * 204
    s.text(AX + 24, y + 6, name, 15.5, STROKE[col], bold=True)
    s.text(AX + 24 + text_width(name, 15.5, True) + 16, y + 6, sub, 13.5, INK)
    # byte strip of two groups
    bsc = 760 / (2 * gstride)
    sx = AX + 24
    starts = []
    for g in range(2):
        g0 = g * gstride + (shift if g % 2 else 0)
        s.rect(sx + g * gstride * bsc, y + 22, gstride * bsc, 52, GRAY, "#f8f9fa", 1.0, r=0, rough=0.25, passes=1, dash="4 4")
        for r in range(4):
            st = g0 + r * stride
            starts.append(st)
            s.rect(sx + st * bsc, y + 28, rb * bsc, 40, rowcols[g * 4 + r], FILL[col], 1.3, r=0, rough=0.3, passes=1)
            s.text(sx + (st + rb / 2) * bsc, y + 53, f"r{g * 4 + r}", 12, rowcols[g * 4 + r], "middle", bold=True)
    s.text(sx, y + 92, "0", 11.5, GRAY)
    s.text(sx + gstride * bsc, y + 92, str(gstride), 11.5, GRAY, "middle")
    s.text(sx + 2 * gstride * bsc, y + 92, str(2 * gstride), 11.5, GRAY, "end")
    # bank positions
    bx = AX + 850
    s.text(bx, y + 14, "start mod 128 (16 B slots)", 12.5, GRAY)
    slots = {}
    for k, st in enumerate(starts):
        slots.setdefault((st % 128) // 16, []).append(k)
    for p in range(8):
        rows = slots.get(p, [])
        clash = len(rows) > 1
        s.rect(bx + p * 52, y + 24, 48, 48, RED if clash else GRAY, "#fff5f5" if clash else ("#ffffff" if not rows else FILL[col]), 1.4,
               r=3, rough=0.35)
        s.text(bx + p * 52 + 24, y + 86, str(p * 16), 11.5, GRAY, "middle")
        for j, k in enumerate(rows):
            s.text(bx + p * 52 + 24, y + 42 + j * 16, f"r{k}", 12, rowcols[k], "middle", bold=True)
    s.text(AX + 24, y + 118, verdict, 13.5, RED if i == 0 else STROKE[col], bold=True)
    s.text(AX + 24, y + 140, {0: "The V buffer gets an XOR swizzle instead of padding (no room for a padded V).",
                              1: "The box reads past the row (zeros); the 16 B shift comes from starting odd groups at column -4.",
                              2: "mxfp4x4 keeps per-row copies: its four Q terms leave no shared memory for the padded groups."}[i],
           13, INK)

# ============================================================================================ D: measured gains
DX = AX + CW + 16
DW = W - 24 - DX
s.group(DX, CY, DW, CH, "D  Measured gain of gather4 (kernel time)", stroke=TEAL, size=17)
gains = [("FP8 tiles, mxfp8", 3.0, 7.0, "blue", "-3 to -7%"), ("FP8 tiles, fp8x2", 0.6, 1.3, "blue", "-1%"),
         ("MXFP4 cache, mxfp8", 7.5, 9.1, "orange", "-7.5 to -9.1%"), ("V4.1 FP4 main, mxfp8", 5.3, 6.8, "green", "-5.3 to -6.8%"),
         ("FP4 extra next to FP8,", -1.7, 3.0, "red", "at most -1 to -3%,"), ]
gx0 = DX + 30
gsc = (DW - 200) / 10
s.line(gx0, CY + 60, gx0, CY + 60 + 5 * 74 - 10, INK, 1.4, rough=0.3)
for i, (lab, lo, hi, col, txt) in enumerate(gains):
    y = CY + 60 + i * 74
    s.text(gx0 + 4, y + 2, lab, 14, INK, bold=True)
    x1, x2 = gx0 + max(lo, 0) * gsc, gx0 + hi * gsc
    s.rect(gx0, y + 12, x2 - gx0, 26, STROKE[col], FILL[col], 1.4, r=2, rough=0.4, hatch=LIGHT[col] if col != "red" else None, hatch_gap=7,
           dash="6 4" if col == "red" else None)
    s.text(x2 + 8, y + 31, txt, 13.5, STROKE[col], bold=True)
s.text(gx0 + 4, CY + 60 + 4 * 74 + 56, "upper bound only, +1.7% at h_q 128 b 512: not used", 13.5, RED)
s.lines(DX + 22, CY + 470, ["Faster = shorter kernel time at the production",
                            "shapes, against the same kernel with one bulk",
                            "copy per row, measured in one session.",
                            "",
                            "After gather4 the gather warp mostly waits; the",
                            "fast modes are then bound by the rebase and",
                            "the softmax (Fig. 2 D, Fig. 7)."], 13.5, lh=1.4, maxw=DW - 40)

footer(s, "Row starts in panel C are byte offsets inside the raw tile: FP8 r*528 + group*2176; V4.1 FP4 r*352 + group*1408 (+16 for odd groups); "
          "MXFP4 r*288 + group*1152 (+16).")
s.save("../fig5_gather.svg")
print("fig5 ok")
