"""Generate the GitHub social preview (1280x640) for this repo.

Usage:
    python tools/generate_social_preview.py

Writes assets/social-preview.png. Pure matplotlib, no external assets.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from matplotlib import font_manager

W, H = 1280, 640

# Prefer a modern sans if present, else default.
def _pick_font():
    prefer = ["Segoe UI", "Microsoft YaHei", "DejaVu Sans", "Arial", "Helvetica"]
    have = {f.name for f in font_manager.fontManager.ttflist}
    for p in prefer:
        if p in have:
            return p
    return "DejaVu Sans"

FONT = _pick_font()
MONO = "Consolas" if "Consolas" in {f.name for f in font_manager.fontManager.ttflist} else "DejaVu Sans Mono"


def gradient_bg(ax):
    """Vertical gradient: deep navy -> teal."""
    top = np.array([0x0b, 0x12, 0x20]) / 255
    mid = np.array([0x0f, 0x2a, 0x4a]) / 255
    bot = np.array([0x0e, 0x74, 0x90]) / 255
    n = H
    g = np.zeros((n, 1, 3))
    half = n // 2
    for i in range(half):
        t = i / max(half - 1, 1)
        g[i, 0] = top * (1 - t) + mid * t
    for i in range(half, n):
        t = (i - half) / max(n - half - 1, 1)
        g[i, 0] = mid * (1 - t) + bot * t
    ax.imshow(g, extent=[0, W, 0, H], origin="upper", aspect="auto", zorder=0)


def chip(ax, x, y, w, h, label, value, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.0,rounding_size=10",
                                fc="#0e1c30", ec=color, lw=1.8, zorder=3))
    ax.text(x + 18, y + h - 24, value, fontsize=22, color=color,
            family=FONT, fontweight="bold", va="center", zorder=4)
    ax.text(x + 18, y + 22, label, fontsize=13, color="#cbd5e1",
            family=FONT, va="center", zorder=4)


def main():
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    gradient_bg(ax)

    # top accent bar
    ax.add_patch(FancyBboxPatch((0, H - 8), W, 8, boxstyle="square,pad=0",
                               fc="#22d3ee", ec="none", zorder=2))

    # Title block
    ax.text(64, 520, "Remote Sensing Image Segmentation", fontsize=40,
            color="#f8fafc", family=FONT, fontweight="bold", va="center", zorder=4)
    ax.text(64, 470, "Models  ·  Papers  ·  Official Code  ·  Datasets  ·  Runnable Baselines",
            fontsize=17, color="#7dd3fc", family=FONT, va="center", zorder=4)

    # accent line
    ax.add_patch(FancyBboxPatch((64, 440), 120, 4, boxstyle="square,pad=0",
                               fc="#22d3ee", ec="none", zorder=4))

    # Subtitle / value prop
    ax.text(64, 405,
            "A curated collection: 40+ RS segmentation methods organized by family,",
            fontsize=15, color="#cbd5e1", family=FONT, va="center", zorder=4)
    ax.text(64, 378,
            "with real runnable baselines (U-Net / SegFormer / SAM) that train offline.",
            fontsize=15, color="#cbd5e1", family=FONT, va="center", zorder=4)

    # Family chips
    cy, ch = 232, 92
    chip(ax, 64, cy, 262, ch, "Classic CNN baselines", "CNN · 8", "#38bdf8")
    chip(ax, 344, cy, 262, ch, "Transformer / attention", "Transformer · 8", "#a78bfa")
    chip(ax, 624, cy, 286, ch, "Foundation models / SAM", "SAM family · 14", "#34d399")
    chip(ax, 928, cy, 288, ch, "Weakly / point-supervised", "Weak-sup · 6", "#fbbf24")

    # Code strip
    ax.add_patch(FancyBboxPatch((64, 118), 620, 88,
                                boxstyle="round,pad=0.0,rounding_size=10",
                                fc="#08111f", ec="#1e3a5f", lw=1.5, zorder=3))
    ax.text(86, 178, "$ pip install torch && python train.py --demo",
            fontsize=15, color="#a5f3fc", family=MONO, va="center", zorder=4)
    ax.text(86, 142, "no dataset · no GPU · runs in seconds", fontsize=13,
            color="#64748b", family=FONT, va="center", zorder=4)

    # Star CTA
    ax.add_patch(FancyBboxPatch((720, 118), 496, 88,
                                boxstyle="round,pad=0.0,rounding_size=10",
                                fc="#111827", ec="#f43f5e", lw=1.8, zorder=3))
    ax.text(748, 178, "Star if it saves you time", fontsize=19,
            color="#fb7185", family=FONT, fontweight="bold", va="center", zorder=4)
    ax.text(748, 142, "github.com/qi-fg/Remote-Sensing-Image-Segmentation-Models",
            fontsize=11.5, color="#94a3b8", family=MONO, va="center", zorder=4)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "social-preview.png")
    fig.savefig(out, dpi=100, facecolor="#0b1220")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
