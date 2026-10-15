"""Shared palette and chart helpers of the sm120 figures (built on sketch.py)."""
import math

from sketch import FILL, INK, STROKE, Sketch, text_width

BLUE, GREEN, ORANGE, RED, GRAY, PURPLE, TEAL, YELLOW, PINK = (STROKE[k] for k in
                                                             ("blue", "green", "orange", "red", "gray", "purple", "teal", "yellow", "pink"))
# One colour per cache format and per arithmetic mode, the same in every figure
FMT = {"fp8": "blue", "nvfp4": "green", "mxfp4": "orange"}
MODE = {"bf16": "gray", "fp8x2": "purple", "mxfp8": "teal", "mxfp4x4": "pink"}
LIGHT = {"blue": "#74c0fc", "green": "#8ce99a", "orange": "#ffa94d", "red": "#ffa8a8", "purple": "#b197fc", "teal": "#66d9e8",
         "gray": "#adb5bd", "pink": "#faa2c1", "yellow": "#ffd43b"}


def header(s, num, title, sub):
    s.text(28, 48, f"Fig. {num}  {title}", 28, bold=True)
    s.text(30, 80, sub, 16, GRAY)


def footer(s, text, y=None):
    s.text(30, (y or s.h - 16), text, 13.5, GRAY, italic=True)


def legend(s, x, y, items, size=13.5, gap=28, box=24, per_row=None, col_w=None):
    """items: (label, colour key, hatch?) ; laid out left to right, wrapping after per_row items."""
    cx, cy = x, y
    for i, it in enumerate(items):
        label, col = it[0], it[1]
        hatch = it[2] if len(it) > 2 else None
        if per_row and i and i % per_row == 0:
            cx, cy = x, cy + gap
        s.rect(cx, cy - 13, box, 15, STROKE[col], FILL[col], 1.4, r=2, rough=0.4, hatch=hatch, hatch_gap=6)
        s.text(cx + box + 8, cy, label, size, INK)
        cx += col_w if col_w else box + 22 + text_width(label, size)


class Axes:
    """Plot area with linear or log10 axes. (x0, y0) is the bottom-left corner in px."""

    def __init__(self, s, x0, y0, w, h, xlim, ylim, xlog=False, ylog=False):
        self.s, self.x0, self.y0, self.w, self.h = s, x0, y0, w, h
        self.xlim, self.ylim, self.xlog, self.ylog = xlim, ylim, xlog, ylog

    def _f(self, v, lim, log):
        if log:
            return (math.log10(v) - math.log10(lim[0])) / (math.log10(lim[1]) - math.log10(lim[0]))
        return (v - lim[0]) / (lim[1] - lim[0])

    def X(self, v):
        return self.x0 + self.w * self._f(v, self.xlim, self.xlog)

    def Y(self, v):
        return self.y0 - self.h * self._f(v, self.ylim, self.ylog)

    def frame(self, xticks=(), yticks=(), xlabel=None, ylabel=None, grid=True, size=12.5):
        s = self.s
        s.line(self.x0 - 6, self.y0, self.x0 + self.w, self.y0, INK, 1.8, rough=0.5)
        s.line(self.x0, self.y0 + 6, self.x0, self.y0 - self.h - 6, INK, 1.6, rough=0.5)
        for v, lab in yticks:
            ty = self.Y(v)
            if grid:
                s.line(self.x0, ty, self.x0 + self.w, ty, "#ced4da", 1.0, rough=0.3, passes=1, dash="3 6")
            s.line(self.x0 - 6, ty, self.x0, ty, INK, 1.2, rough=0.2, passes=1)
            s.text(self.x0 - 10, ty + 4.5, lab, size, GRAY, "end")
        for v, lab in xticks:
            tx = self.X(v)
            if grid:
                s.line(tx, self.y0, tx, self.y0 - self.h, "#e9ecef", 1.0, rough=0.3, passes=1, dash="3 6")
            s.line(tx, self.y0, tx, self.y0 + 6, INK, 1.2, rough=0.2, passes=1)
            s.text(tx, self.y0 + 21, lab, size, GRAY, "middle")
        if xlabel:
            s.text(self.x0 + self.w / 2, self.y0 + 44, xlabel, 14, INK, "middle", bold=True)
        if ylabel:
            s.text(self.x0 - 58, self.y0 - self.h / 2, ylabel, 14, INK, "middle", bold=True, rot=-90)


def dot(s, x, y, col, shape="o", r=7.5, label=None, lsize=12.5, dx=11, dy=4, anchor="start", bold=True, lcol=None):
    if shape == "o":
        s.ellipse(x, y, r, r, STROKE[col], FILL[col], 1.8, rough=0.4)
    elif shape == "s":
        s.rect(x - r, y - r, 2 * r, 2 * r, STROKE[col], FILL[col], 1.8, r=1.5, rough=0.4)
    else:  # diamond
        s.polyline([(x, y - r * 1.25), (x + r * 1.25, y), (x, y + r * 1.25), (x - r * 1.25, y)], STROKE[col], 1.8, rough=0.3, closed=True)
    if label:
        s.text(x + dx, y + dy, label, lsize, lcol or STROKE[col], anchor, bold=bold)
