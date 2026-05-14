"""
Schematic diagram showing the four lamp configurations as seen from the front
(specimen's point of view). Filament drawn as a bright line inside each lamp.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

OUTPUT_PNG = config.BOSON_ROOT + r"\filament_diagram.png"

fig, axes = plt.subplots(1, 4, figsize=(20, 7))
fig.patch.set_facecolor("#111111")
fig.suptitle("Four configurations — as seen from the specimen\n"
             "Lamp rectangle = housing  |  Yellow line = filament  |  Arrow = beam direction toward specimen",
             color="white", fontsize=12, y=1.02)

CONFIGS = [
    ("Side lamps\nVertical filament\n(current)",   "#ff6b6b", "side",   "v"),
    ("Side lamps\nHorizontal filament\n(BEST)",    "#ffd93d", "side",   "h"),
    ("Top/Bottom lamps\nVertical filament",        "#6bcb77", "topbot", "v"),
    ("Top/Bottom lamps\nHorizontal filament",      "#4d96ff", "topbot", "h"),
]

for ax, (title, col, pos, fil) in zip(axes, CONFIGS):
    ax.set_facecolor("#0d0d0d")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(col); sp.set_linewidth(2.5)

    # ── specimen (centre, lime border) ──────────────────────────────────────
    spec_w, spec_h = 3.2, 1.75
    ax.add_patch(mpatches.Rectangle(
        (-spec_w/2, -spec_h/2), spec_w, spec_h,
        fc="#1a3a1a", ec="lime", lw=2, zorder=3))
    ax.text(0, 0, "SPECIMEN\n320×175mm", ha="center", va="center",
            color="lime", fontsize=8, fontweight="bold", zorder=4)

    # ── camera (oval above specimen centre) ─────────────────────────────────
    cam = mpatches.Ellipse((0, 0), 0.7, 0.5, fc="#334455", ec="white", lw=1.5, zorder=5)
    ax.add_patch(cam)
    ax.text(0, 0, "CAM", ha="center", va="center", color="white", fontsize=7, zorder=6)

    # ── lamp geometry ────────────────────────────────────────────────────────
    lw, lh = 1.4, 0.7   # lamp width, height in axis units

    if pos == "side":
        lamp_centres = [(-4.0, 0), (4.0, 0)]   # left and right
        for cx, cy in lamp_centres:
            # lamp box
            ax.add_patch(mpatches.Rectangle(
                (cx - lw/2, cy - lh/2), lw, lh,
                fc="#1a2a3a", ec=col, lw=2, zorder=3))
            # filament
            if fil == "v":   # vertical line in lamp
                ax.plot([cx, cx], [cy - lh/2 + 0.08, cy + lh/2 - 0.08],
                        color="yellow", lw=3, zorder=5)
                ax.text(cx, cy - lh/2 - 0.25, "vertical\nfilament",
                        ha="center", color="yellow", fontsize=7)
            else:             # horizontal line in lamp
                ax.plot([cx - lw/2 + 0.1, cx + lw/2 - 0.1], [cy, cy],
                        color="yellow", lw=3, zorder=5)
                ax.text(cx, cy - lh/2 - 0.25, "horizontal\nfilament",
                        ha="center", color="yellow", fontsize=7)
            # beam arrow toward specimen centre
            dx = -np.sign(cx) * 1.5
            ax.annotate("", xy=(cx + dx*0.85, cy),
                        xytext=(cx + dx*0.05, cy),
                        arrowprops=dict(arrowstyle="-|>", color=col,
                                        lw=1.8, mutation_scale=14), zorder=4)

        # angle arc label
        ax.annotate("", xy=(-2.3, 0.0), xytext=(-3.2, 0.0),
                    arrowprops=dict(arrowstyle="-", color="#888888", lw=1))
        ax.text(-2.8, 0.25, "angle θ\n(horizontal)", ha="center",
                color="#aaaaaa", fontsize=7)

        # sketch of beam spread on specimen (wider in X if v, wider in Y if h)
        if fil == "v":
            bw, bh = 1.8, 0.5   # beam footprint: wide X, narrow Y
        else:
            bw, bh = 0.9, 1.2   # beam footprint: narrower X, taller Y
        for cx_b in [-0.6, 0.6]:
            ax.add_patch(mpatches.Ellipse(
                (cx_b, 0), bw, bh,
                fc=col, alpha=0.12, ec=col, lw=0.8, ls="--", zorder=2))
        ax.text(0, -spec_h/2 - 0.55,
                "beam wider in X" if fil == "v" else "beam wider in Y",
                ha="center", color=col, fontsize=8)

    else:  # topbot
        lamp_centres = [(0, 3.5), (0, -3.5)]   # top and bottom
        for cx, cy in lamp_centres:
            # lamp box (landscape: wide)
            ax.add_patch(mpatches.Rectangle(
                (cx - lw/2, cy - lh/2), lw, lh,
                fc="#1a2a3a", ec=col, lw=2, zorder=3))
            if fil == "v":   # vertical filament
                ax.plot([cx, cx], [cy - lh/2 + 0.08, cy + lh/2 - 0.08],
                        color="yellow", lw=3, zorder=5)
                side = "right"
                ax.text(cx + lw/2 + 0.12, cy, "vertical\nfilament",
                        ha="left", color="yellow", fontsize=7, va="center")
            else:
                ax.plot([cx - lw/2 + 0.1, cx + lw/2 - 0.1], [cy, cy],
                        color="yellow", lw=3, zorder=5)
                ax.text(cx + lw/2 + 0.12, cy, "horizontal\nfilament",
                        ha="left", color="yellow", fontsize=7, va="center")
            # beam arrow
            dy = -np.sign(cy) * 1.5
            ax.annotate("", xy=(cx, cy + dy*0.85),
                        xytext=(cx, cy + dy*0.05),
                        arrowprops=dict(arrowstyle="-|>", color=col,
                                        lw=1.8, mutation_scale=14), zorder=4)

        ax.text(1.6, 2.8, "angle φ\n(vertical)", ha="center",
                color="#aaaaaa", fontsize=7)

        if fil == "v":
            bw, bh = 1.8, 0.5
        else:
            bw, bh = 0.9, 1.2
        for cy_b in [-0.3, 0.3]:
            ax.add_patch(mpatches.Ellipse(
                (0, cy_b), bw, bh,
                fc=col, alpha=0.12, ec=col, lw=0.8, ls="--", zorder=2))
        ax.text(0, -spec_h/2 - 0.55,
                "beam wider in X" if fil == "v" else "beam wider in Y",
                ha="center", color=col, fontsize=8)

    ax.set_title(title, color=col, fontsize=10, fontweight="bold", pad=10)

    # ── axis labels ──────────────────────────────────────────────────────────
    ax.text( 4.7,  0, "X →", color="#666666", fontsize=8, va="center")
    ax.text( 0,  4.7, "Y ↑", color="#666666", fontsize=8, ha="center")

plt.tight_layout(pad=2.5)
fig.savefig(OUTPUT_PNG, dpi=140, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved: {OUTPUT_PNG}")
