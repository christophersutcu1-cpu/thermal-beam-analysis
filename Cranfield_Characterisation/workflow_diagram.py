"""Render a 1-sentence-per-box flowchart of the Cranfield SEQ -> drone-rig pipeline."""
import os, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUTPUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workflow_diagram.png")

BOXES = [
    ("INPUT",   "3 .seq captures at 300 / 400 / 500 mm standoff from Cranfield.",                            "#9e9e9e"),
    ("STAGE 1", "Fit a 2D Gaussian per standoff and derive the beam model sigma_x(d), sigma_y(d), peak(d).", "#6bcb77"),
    ("STAGE 2", "Compute the Boson+ (18 mm / 24 deg) camera FOV at each design standoff.",                   "#ffd93d"),
    ("STAGE 3", "Sweep lamp tilt and within-pair offset for Configs A and B to minimise CoV on the FOV.",    "#4d96ff"),
    ("OUTPUT",  "Recommended drone-rig design: lamp arrangement, tilt, offset and standoff for min CoV.",    "#ff6b6b"),
]

WRAP_WIDTH = 58       # characters per line — keeps the longest box to <=2 lines

# pre-wrap every sentence so we can size boxes consistently
wrapped = [(tag, textwrap.fill(s, width=WRAP_WIDTH), col) for (tag, s, col) in BOXES]
max_lines = max(w[1].count("\n") + 1 for w in wrapped)

# layout constants
box_w   = 8.6
line_h  = 0.42                                # axes units per line of text
tag_h   = 0.55
pad_v   = 0.30
box_h   = tag_h + max_lines * line_h + pad_v  # height = tag + N text lines + padding
gap     = 0.55
x_left  = (10 - box_w) / 2

total_h = len(wrapped) * box_h + (len(wrapped) - 1) * gap
y_top0  = total_h + 0.5
fig_h   = total_h + 1.2

fig, ax = plt.subplots(figsize=(11, fig_h))
fig.patch.set_facecolor("#111111")
ax.set_facecolor("#111111")
ax.set_xlim(0, 10)
ax.set_ylim(0, total_h + 1.0)
ax.set_aspect("equal")
ax.axis("off")

for i, (tag, text, col) in enumerate(wrapped):
    y_top = y_top0 - i * (box_h + gap)
    y_bot = y_top - box_h

    ax.add_patch(FancyBboxPatch(
        (x_left, y_bot), box_w, box_h,
        boxstyle="round,pad=0.05,rounding_size=0.18",
        fc="#1a1a1a", ec=col, lw=2.5, zorder=2))

    # tag near top of box
    ax.text(x_left + 0.35, y_top - tag_h * 0.55, tag,
            color=col, fontsize=14, fontweight="bold",
            va="center", ha="left")

    # wrapped sentence; vertical-centred in the remaining space below the tag
    text_centre_y = y_bot + (box_h - tag_h) / 2
    ax.text(x_left + 0.35, text_centre_y, text,
            color="white", fontsize=11, va="center", ha="left",
            linespacing=1.25)

    # arrow to next box
    if i < len(wrapped) - 1:
        next_y_top = y_top0 - (i + 1) * (box_h + gap)
        ax.add_patch(FancyArrowPatch(
            (5, y_bot - 0.05), (5, next_y_top + 0.05),
            arrowstyle="-|>", mutation_scale=22,
            color="#aaaaaa", lw=2, zorder=3))

fig.savefig(OUTPUT_PNG, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved: {OUTPUT_PNG}")
