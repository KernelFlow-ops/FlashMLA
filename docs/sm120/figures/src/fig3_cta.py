"""Fig. 3 - one CTA of the block-scaled kernels: warp roles, shared-memory ring, consumer pairs, per-tile pipeline, grid."""
from figcommon import (BLUE, GRAY, GREEN, LIGHT, ORANGE, PINK, PURPLE, RED, TEAL, YELLOW, FILL, INK, STROKE, Sketch, footer, header,
                       text_width)

W, H = 2000, 1330
s = Sketch(W, H, seed=37)
header(s, 3, "One CTA of the block-scaled kernels (mxfp8.cu, mxfp4.cu): warp specialization around a three-tile ring",
       "CTA = one query token x 64 heads, 384 threads = 4 producer warps + 8 consumer warps. Tiles of 32 KV tokens stream through "
       "shared memory; every hand-off is an mbarrier or a pair barrier.")

# ============================================================================================ A: the CTA
AX, AY, AW, AH = 24, 112, 1262, 1186
s.group(AX, AY, AW, AH, "A  Data flow and roles inside the CTA", stroke=BLUE, size=17)
# global memory row
s.box(50, 152, 520, 74, "paged KV cache  (global memory, 96 MB L2)", ["640 rows per query token, at random pages (sparse top-k)"],
      fill=FILL["yellow"], stroke=STROKE["yellow"], tsize=15.5, size=13.5)
s.box(592, 152, 250, 74, "row indices", ["read one tile ahead"], fill=FILL["gray"], stroke=GRAY, tsize=15.5, size=13.5)
s.box(864, 152, 396, 74, "Q (bf16) in, O (bf16) + lse out", ["Q quantized once per CTA; O at the end"], fill=FILL["blue"], stroke=BLUE,
      tsize=15.5, size=13.5)

# producers
PY = 262
s.group(40, PY, 1230, 214, "producer warps 0-3  (40 / 56 registers)", stroke=ORANGE, size=15)
s.box(56, PY + 26, 520, 172, "warp 0: gather", ["per 32-token tile: 8 x TMA tile::gather4 (FP8 rows,",
                                                 "4-bit main caches) or 32 x cp.async.bulk (FP4 extra rows)",
                                                 "completion counted by mbarrier complete_tx -> bar_raw",
                                                 "an invalid token reads row -1: zeros, never the cache's NaN",
                                                 "waits on bar_qk_done + bar_rebased to reuse a raw slot"],
      fill=FILL["orange"], stroke=ORANGE, align="left", size=13.5)
s.box(596, PY + 26, 660, 172, "warps 1-3: rebase", ["6 / 5 / 5 groups of 32 dims per tile, as soon as the tile lands",
                                                     "per (tile, group): e_max over the valid tokens -> column scale 2^e_max",
                                                     "FP8 rows: shift each byte's exponent down by e_max - e_token",
                                                     "FP4 rows: e2m1 x scale -> e4m3 with one scale per tile and group",
                                                     "write the V buffer, arrive bar_rebased; wait bar_v_free first"],
      fill=FILL["green"], stroke=GREEN, align="left", size=13.5)

# shared memory
SY = 506
s.group(40, SY, 1230, 196, "shared memory: 101,376 B", stroke=PURPLE, size=15)
ring = [("raw tile t", "in use: QK + rebase", BLUE), ("raw tile t+1", "landed", BLUE), ("raw tile t+2", "in flight", GRAY)]
for i, (t, sub, col) in enumerate(ring):
    x = 56 + i * 210
    s.box(x, SY + 28, 196, 84, t, [sub, "17,408 B"], fill=FILL["blue" if col == BLUE else "gray"], stroke=col, tsize=15, size=13)
s.text(56, SY + 136, "raw ring: rows as stored (528 / 288 / 272 B) in groups of 4 rows; mbarriers and masks in the 64 B gaps", 13, INK)
s.text(56, SY + 158, "V buffers: e4m3 + one scale per 32 dims; with two, a rebase never waits for the PV of the tile before it", 13, INK)
vb = [("V buffer A", "tile t: PV reads it"), ("V buffer B", "tile t+1: rebase writes")]
for i, (t, sub) in enumerate(vb):
    x = 700 + i * 196
    s.box(x, SY + 28, 184, 84, t, [sub, "16,384 B e4m3"], fill=FILL["green"], stroke=GREEN, tsize=15, size=13)
s.box(1094, SY + 28, 160, 84, "S exchange", ["half of S", "16,384 B"], fill=FILL["purple"], stroke=PURPLE, tsize=15, size=13)

# consumers
CY = 732
s.group(40, CY, 1230, 540, "consumer warps 4-11  (232 / 224 registers)", stroke=TEAL, size=15)
# pair 0 in detail
s.rect(56, CY + 26, 760, 496, TEAL, "#f8f9fa", 1.6, r=12, rough=0.6)
s.text(72, CY + 54, "pair m = warps 2m and 2m+1, on different SM sub-partitions: heads 16m .. 16m+15", 14.5, TEAL, bold=True)
wx = [72, 448]
for k, x in enumerate(wx):
    s.badge(250 + 360 * k, CY + 84, f"warp 2m{'+1' if k else ''}: dims {256 * k}-{256 * k + 255}, out columns {256 * k}-{256 * k + 255}",
            FILL["teal"], TEAL, 12.5)
steps = [("1  QK on its half of the dims", "K = the raw bytes (ldmatrix x4), cache scales as SF_B: block-scaled MMA", BLUE),
         ("2  swap the halves of S", "through S exchange, one pair barrier (no CTA-wide sync)", PURPLE),
         ("3  softmax, identical in both warps", "row max, exp2, lazy rescale; P stays in registers as the A operand", PURPLE),
         ("4  PV on its 256 output columns", "V = rebased buffer (ldmatrix.trans), P as e4m3 terms; O: 128 fp32 regs", TEAL),
         ("5  release", "arrive bar_qk_done (raw slot) and bar_v_free (V buffer)", GRAY)]
for i, (t, sub, col) in enumerate(steps):
    y = CY + 110 + i * 80
    key = {BLUE: "blue", PURPLE: "purple", TEAL: "teal", GRAY: "gray"}[col]
    s.box(80, y, 712, 66, t, [sub], fill=FILL[key], stroke=col, tsize=15, size=13.5, align="left")
    if i:
        s.arrow([(436, y - 14), (436, y)], INK, 1.6)
# the other pairs
for i in range(3):
    y = CY + 30 + i * 112
    s.box(862, y, 392, 96, f"pair {i + 1}: heads {16 * (i + 1)}-{16 * (i + 1) + 15}", ["same steps on the same tile, own S exchange",
                                                                                      "slot and barrier; drifts against pair 0"],
          fill=FILL["teal"], stroke=TEAL, tsize=15, size=13.5)
s.note(866, CY + 372, 384, 142, ["The pairs on one sub-partition drift apart,", "so one pair's softmax overlaps the other's",
                                  "MMAs. Pairs instead of one CTA barrier:", "fp8x2 -7 to -9%, mxfp8 -1 to -5%."],
       title="why pairs", size=13.5, rot=-1.0)

# arrows of panel A
s.arrow([(450, 226), (450, PY + 26)], ORANGE, 2.2)
s.text(458, 250, "rows", 13, ORANGE, bold=True)
s.arrow([(716, 226), (548, PY + 26)], GRAY, 1.6, dash="5 4")
s.arrow([(520, PY + 198), (574, SY + 28)], ORANGE, 2.2)
s.text(588, SY - 12, "bar_raw", 12.5, ORANGE, bold=True)
s.arrow([(364, SY + 28), (680, PY + 198)], BLUE, 1.8, dash="6 4")
s.arrow([(990, PY + 198), (990, SY + 28)], GREEN, 2.2)
s.text(1000, SY - 12, "bar_rebased", 12.5, GREEN, bold=True)
s.arrow([(56, SY + 70), (47, SY + 70), (47, CY + 145), (80, CY + 145)], BLUE, 2.0)
s.text(52, CY + 132, "K", 14, BLUE, bold=True)
s.arrow([(800, SY + 112), (830, SY + 190), (830, CY + 1115 - CY), (794, CY + 1115 - CY)], GREEN, 2.0)
s.text(836, CY + 380, "V", 15, GREEN, bold=True)
s.arrow([(1174, SY + 112), (1174, SY + 186), (848, SY + 186), (848, CY + 940 - CY + 6), (794, CY + 940 - CY + 6)], PURPLE, 1.7, both=True)
s.arrow([(1060, 226), (1060, 244), (1278, 244), (1278, CY + 20)], BLUE, 1.4, dash="6 5")

# ============================================================================================ B: shared memory
BX = AX + AW + 16
BW = W - 24 - BX
BY, BH = AY, 290
s.group(BX, BY, BW, BH, "B  Shared-memory budget (FP8 instance)", stroke=PURPLE, size=17)
parts = [("raw ring 3 x 17,408", 52224, "blue"), ("V 2 x 16,384", 32768, "green"), ("S exchange", 16384, "purple")]
x = BX + 24
tw = BW - 48
for name, v, col in parts:
    w = tw * v / 101376
    s.rect(x, BY + 50, w, 46, STROKE[col], FILL[col], 1.6, r=3, rough=0.5, hatch=LIGHT[col], hatch_gap=8)
    s.text(x + w / 2, BY + 79, f"{v:,} B", 14, STROKE[col], "middle", bold=True)
    s.text(x + w / 2, BY + 118, name, 13.5, INK, "middle")
    x += w
s.text(BX + BW - 24, BY + 40, "= 101,376 B", 14, INK, "end", bold=True)
s.lines(BX + 24, BY + 156, ["A 32-token FP8 tile is 32 x 528 = 16,896 B; gather4 needs 128 B aligned",
                            "groups of 4 rows, so a group takes 2176 B (+64 B, where the barriers live).",
                            "Not one byte is left: every later idea had to fit in these gaps or replace",
                            "something (Fig. 5, 6)."], 13.5, lh=1.4, maxw=BW - 40)

# ============================================================================================ C: per-tile pipeline
CY2, CH2 = BY + BH + 16, 446
s.group(BX, CY2, BW, CH2, "C  Steady state: three tiles in flight", stroke=GREEN, size=17)
lanes = ["warp 0 (gather)", "warps 1-3 (rebase)", "pair 0", "pair 1"]
lx0 = BX + 168
unit = (BW - 168 - 30) / 6.2
for i, lab in enumerate(lanes):
    y = CY2 + 64 + i * 66
    s.text(lx0 - 10, y + 26, lab, 13.5, INK, "end", bold=True)
    s.line(lx0, y + 46, lx0 + 6.1 * unit, y + 46, "#dee2e6", 1.0, rough=0.2, passes=1)


def blk(lane, t0, t1, lab, col, hatch=None):
    y = CY2 + 64 + lane * 66
    s.rect(lx0 + t0 * unit, y + 8, (t1 - t0) * unit - 3, 34, STROKE[col], FILL[col], 1.4, r=4, rough=0.5, hatch=hatch, hatch_gap=7)
    s.text(lx0 + (t0 + t1) / 2 * unit, y + 31, lab, 12.5, STROKE[col], "middle", bold=True)


for k in range(4):
    blk(0, k * 1.5, k * 1.5 + 0.7, f"t+{k}", "orange")
for k in range(4):
    blk(1, 0.7 + k * 1.5, 0.7 + k * 1.5 + 1.2, f"t+{k}", "green")
for k in range(3):
    b = 1.9 + k * 1.5
    blk(2, b, b + 0.5, "QK", "blue")
    blk(2, b + 0.5, b + 0.8, "sm", "purple")
    blk(2, b + 0.8, b + 1.4, "PV", "teal")
    b2 = b + 0.35
    blk(3, b2, b2 + 0.5, "QK", "blue")
    blk(3, b2 + 0.5, b2 + 0.8, "sm", "purple")
    if b2 + 1.4 < 6.1:
        blk(3, b2 + 0.8, b2 + 1.4, "PV", "teal")
s.lines(BX + 24, CY2 + 340, ["Gather runs two tiles ahead, rebase one tile ahead; the consumers never wait",
                             "for V in steady state. The pairs drift, so softmax (sm) of one pair overlaps",
                             "the MMAs of another. Times are schematic; measured stalls are in Fig. 7."], 13.5, lh=1.4, maxw=BW - 40)

# ============================================================================================ D: grid
DY = CY2 + CH2 + 16
DH = AY + AH - DY
s.group(BX, DY, BW, DH, "D  The grid and small batches", stroke=ORANGE, size=17)
gx, gy = BX + 28, DY + 44
for j in range(2):
    for i in range(6):
        s.rect(gx + i * 46, gy + j * 44, 40, 36, BLUE if j == 0 else PURPLE, FILL["blue" if j == 0 else "purple"], 1.3, r=4, rough=0.4)
        s.text(gx + i * 46 + 20, gy + j * 44 + 23, f"h{j}", 12, BLUE if j == 0 else PURPLE, "middle", bold=True)
s.text(gx, gy + 104, "grid (h_q / 64, s_q, b): the two head blocks", 13.5, INK)
s.text(gx, gy + 124, "of h_q = 128 launch side by side", 13.5, INK)
s.lines(BX + 330, gy + 10, ["the second CTA finds the same rows in L2:", ("h_q = 128, mxfp8: -3.5 to -6.0%", ORANGE, True)], 13.5, lh=1.4,
        maxw=BW - 340)
s.lines(BX + 24, gy + 166, [("small batches: split over the SMs", INK, True),
                            "when b x s_q x h_q/64 x 2 <= 170 (not with batch invariance):",
                            "partial results + the upstream combine kernel.",
                            ("h_q = 64, b = 1, s_q = 1, exact: 71.8 -> 17.8 us per layer (graph)", ORANGE, True),
                            "production shapes are not split: forcing it was 3-7% slower",
                            "at b >= 256 (the combine eats the saved tail wave)."], 13.5, lh=1.4, maxw=BW - 40)

footer(s, "Register split by setmaxnreg: 8 x 232 + 4 x 40 = 64,512 registers, all a 384-thread block gets (232 + 48 hangs in setmaxnreg.inc). "
          "No spills in any instance.")
s.save("../fig3_cta.svg")
print("fig3 ok")
