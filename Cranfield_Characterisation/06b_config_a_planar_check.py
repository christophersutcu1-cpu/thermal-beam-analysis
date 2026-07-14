"""
Understanding check for Config A — PLANAR fixture (corrected).

My earlier 06b model was wrong: it put the two bulbs on each arm at DIFFERENT
standoffs (an arm receding into depth). This diagram shows the corrected model
for the user to confirm:

  * The whole fixture is PLANAR — all 4 bulbs lie in one vertical plane,
    at the SAME standoff d from the target.
  * d1 = horizontal separation of the bulbs (unchanged meaning).
  * d2 = VERTICAL separation of the bulbs, in-plane.
  * Each bulb sits on a small swivel; the swivel angle is LOCKED at 45 deg.
    For Config A the swivel tilts in the vertical plane (pitch): the upper
    bulbs aim down, the lower bulbs aim up, toward the FOV.

Two views:
  LEFT  — front view (looking along the optical axis): the single vertical
          plane with the 2x2 bulb layout, d1 (horizontal) and d2 (vertical).
  RIGHT — side view (y-z): all bulbs at the same standoff d, each beam at 45 deg.

This is a SCHEMATIC for confirmation only (normalised units) — no sweep numbers,
because the sweep geometry must be rebuilt once the model is agreed.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch, Arc

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_PNG   = os.path.join(REPO_ROOT, "06b_config_a_planar_check.png")

ACCENT = "#ff6b6b"      # Config A colour
INV2   = 1.0/np.sqrt(2.0)

# representative (schematic) dimensions in normalised units
D1H = 0.95              # half of d1 (horizontal half-separation)
D2H = 0.55             # half of d2 (vertical half-separation)

fig = plt.figure(figsize=(17, 8.5))
fig.patch.set_facecolor("#111111")
gs = mgridspec.GridSpec(1, 2, figure=fig, left=0.05, right=0.97,
                        top=0.80, bottom=0.20, wspace=0.18)

def style(ax):
    ax.set_facecolor("#0d0d0d"); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(ACCENT); sp.set_linewidth(2.0)

def bulb(ax, x, y, r=0.075):
    ax.add_patch(Circle((x, y), r, fc="#1a2a3a", ec=ACCENT, lw=2.0, zorder=6))
    ax.add_patch(Circle((x, y), r*0.4, fc="#fff2a0", ec="none", alpha=0.95, zorder=7))

# ============================ LEFT: FRONT VIEW ===============================
ax1 = fig.add_subplot(gs[0, 0]); style(ax1)
ax1.set_title("FRONT VIEW — one vertical plane (all 4 bulbs coplanar)",
              color=ACCENT, fontsize=12, fontweight="bold", pad=10)

# FOV box (camera target footprint, centred)
ax1.add_patch(Rectangle((-0.5, -0.375), 1.0, 0.75, fc="#16282b",
                        ec="cyan", lw=2.0, zorder=3))
ax1.text(0, 0, "FOV", ha="center", va="center", color="cyan",
         fontsize=10, fontweight="bold", zorder=4)

# vertical mounting rails at x = +-d1/2
for sx in (-1, +1):
    ax1.plot([sx*D1H, sx*D1H], [-D2H, D2H], color="#888888", lw=2.5,
             zorder=2, solid_capstyle="round")

# 4 bulbs at (+-d1/2, +-d2/2) + in-plane swivel arrows toward FOV centre
for sx in (-1, +1):
    for sy in (-1, +1):
        bx, by = sx*D1H, sy*D2H
        bulb(ax1, bx, by)
        # in-plane aim arrow toward FOV centre (just indicative)
        ax1.add_patch(FancyArrowPatch((bx, by), (bx*0.35, by*0.2),
                                      arrowstyle="-|>", color=ACCENT,
                                      mutation_scale=12, lw=1.2, alpha=0.7, zorder=5))

# d1 dimension (horizontal)
ax1.annotate("", xy=(D1H, -0.95), xytext=(-D1H, -0.95),
             arrowprops=dict(arrowstyle="<->", color="white", lw=1.5))
ax1.text(0, -1.02, "d1  =  horizontal separation  (unchanged)",
         ha="center", va="top", color="white", fontsize=10, fontweight="bold")

# d2 dimension (vertical)
ax1.annotate("", xy=(-1.28, D2H), xytext=(-1.28, -D2H),
             arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.5))
ax1.text(-1.33, 0, "d2 = vertical\nseparation\n(in-plane)", ha="right", va="center",
         color=ACCENT, fontsize=9.5, fontweight="bold")

ax1.text(0, 1.15, "the page = the single vertical plane the fixture lives in",
         ha="center", va="bottom", color="#aaaaaa", fontsize=9, style="italic")

ax1.set_xlim(-1.75, 1.45)
ax1.set_ylim(-1.35, 1.35)

# ============================ RIGHT: SIDE VIEW ===============================
ax2 = fig.add_subplot(gs[0, 1]); style(ax2)
ax2.set_title("SIDE VIEW (y–z) — all bulbs at the SAME standoff d  ⇒  'planar'",
              color=ACCENT, fontsize=12, fontweight="bold", pad=10)

Z_PANEL = 1.6                      # standoff (depth) of the bulb plane
# target (FOV height) at z = 0
ax2.plot([0, 0], [-0.4, 0.4], color="#39d353", lw=6, zorder=4,
         solid_capstyle="butt")
ax2.text(-0.06, 0, "target\n(FOV)", ha="right", va="center", color="#39d353",
         fontsize=9, fontweight="bold")

# the planar fixture (dashed vertical line at z = d) with the 2 bulb heights
ax2.plot([Z_PANEL, Z_PANEL], [-D2H-0.15, D2H+0.15], color="#888888", lw=2.0,
         ls="--", zorder=2)
ax2.text(Z_PANEL, D2H+0.22, "bulb plane", ha="center", va="bottom",
         color="#aaaaaa", fontsize=9, style="italic")

for sy in (+1, -1):                # upper / lower bulb
    by = sy*D2H
    bulb(ax2, Z_PANEL, by)
    # 45-deg beam from bulb to target centre
    ax2.add_patch(FancyArrowPatch((Z_PANEL, by), (0.04, 0.0),
                                  arrowstyle="-|>", color=ACCENT,
                                  mutation_scale=14, lw=1.6, alpha=0.85, zorder=5))
    # reference: panel-normal (straight to target, horizontal)
    ax2.plot([Z_PANEL, Z_PANEL-0.5], [by, by], color="#555555", lw=1.0,
             ls=":", zorder=3)
    # 45-deg arc between normal and beam
    ang = np.degrees(np.arctan2(0.0-by, 0.04-Z_PANEL))
    a1, a2 = (180, ang) if sy > 0 else (ang, 180)
    ax2.add_patch(Arc((Z_PANEL, by), 0.55, 0.55, theta1=a1, theta2=a2,
                      color="orange", lw=1.6, zorder=6))
ax2.text(Z_PANEL-0.33, 0.0, "45°\n(locked\nswivel)", ha="center", va="center",
         color="orange", fontsize=8.5, fontweight="bold")

# d2 (vertical) on the panel
ax2.annotate("", xy=(Z_PANEL+0.18, D2H), xytext=(Z_PANEL+0.18, -D2H),
             arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.4))
ax2.text(Z_PANEL+0.24, 0, "d2", color=ACCENT, fontsize=10, fontweight="bold",
         va="center", ha="left")

# standoff d
ax2.annotate("", xy=(Z_PANEL, -0.78), xytext=(0, -0.78),
             arrowprops=dict(arrowstyle="<->", color="white", lw=1.4))
ax2.text(Z_PANEL/2, -0.85, "standoff  d  (same for every bulb)",
         ha="center", va="top", color="white", fontsize=9.5, fontweight="bold")

ax2.set_xlim(-0.7, Z_PANEL+0.7)
ax2.set_ylim(-1.05, 1.15)

# ============================ header + footer ================================
fig.suptitle("Config A — corrected PLANAR model.  Have I understood this right?",
             color="white", fontsize=15, fontweight="bold", y=0.95)

fig.text(0.5, 0.115,
         "My understanding now:  (1) all 4 bulbs lie in ONE vertical plane at the same standoff d (planar);   "
         "(2) d1 = horizontal bulb separation (unchanged);   (3) d2 = vertical bulb separation, in-plane;\n"
         "(4) each bulb on a swivel LOCKED at 45° — for Config A the swivel tilts vertically (pitch), upper bulbs aim down / lower bulbs aim up toward the FOV.        "
         "Config B would be the same plane but swivel tilts horizontally (yaw).",
         ha="center", va="top", color="#ffd93d", fontsize=10, family="monospace")

fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved: {OUT_PNG}")
