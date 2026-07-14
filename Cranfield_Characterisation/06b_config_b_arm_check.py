"""
Understanding check for Config B — ANGLED-ARM V-trough (corrected).

Config B is NOT the planar swivel fixture of Config A. It is the original
angled-arm model:

  * All 4 bulbs sit on the SAME horizontal axis (y = 0, the camera centreline).
  * There is a fixed angled ARM on each side; the bulbs are mounted on the arm.
  * The arm is at a FIXED 45 deg — there is NO swivel.
  * d1 = horizontal baseline (separation of the two arm roots) — SAME as Config A.
  * d2 = offset between the two bulbs measured ALONG the angled arm, so the inner
         bulb is closer to the target and the outer bulb is further (the two
         bulbs are at DIFFERENT standoffs).

Because the arm (45 deg) is parallel to each bulb's fixed beam axis (45 deg),
both bulbs on a side aim at the same target line (+-d1/2); d2 changes only their
relative standoff -> inner = smaller/brighter footprint, outer = larger/dimmer.

Two views:
  LEFT  — TOP view (x-z), looking down: the V-trough in the horizontal plane —
          green target strip (d1), two 45 deg arms, 2 bulbs each spaced d2 along
          the arm at different standoffs.
  RIGHT — FRONT view (x-y): shows all 4 bulbs lying on the same y = 0 axis.

Schematic only (normalised units).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch, Arc

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_PNG   = os.path.join(REPO_ROOT, "06b_config_b_arm_check.png")

ACCENT = "#4d96ff"      # Config B colour
INV2   = 1.0/np.sqrt(2.0)

# schematic dimensions (normalised units)
D1H = 0.70              # half of d1 (arm-root half-separation, horizontal)
S0  = 0.28              # inner bulb distance up the arm from the root
D2  = 0.55              # along-arm offset between the two bulbs

fig = plt.figure(figsize=(17, 8.5))
fig.patch.set_facecolor("#111111")
gs = mgridspec.GridSpec(1, 2, figure=fig, left=0.05, right=0.97,
                        top=0.80, bottom=0.20, wspace=0.18)

def style(ax):
    ax.set_facecolor("#0d0d0d"); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(ACCENT); sp.set_linewidth(2.0)

def bulb(ax, x, y, r=0.07):
    ax.add_patch(Circle((x, y), r, fc="#1a2a3a", ec=ACCENT, lw=2.0, zorder=6))
    ax.add_patch(Circle((x, y), r*0.4, fc="#fff2a0", ec="none", alpha=0.95, zorder=7))

# bulb positions along each arm (x, z) for the TOP view ----------------------
def arm_bulbs(side):
    """side = +1 (right) / -1 (left). Returns [(inner), (outer)] in (x, z)."""
    root = (side*D1H, 0.0)
    out  = []
    for s in (S0, S0 + D2):
        out.append((root[0] + side*s*INV2, root[1] + s*INV2))
    return root, out

# ============================ LEFT: TOP VIEW =================================
ax1 = fig.add_subplot(gs[0, 0]); style(ax1)
ax1.set_title("TOP VIEW (x–z), looking down — angled-arm V-trough (horizontal plane)",
              color=ACCENT, fontsize=11.5, fontweight="bold", pad=10)

# target / FOV strip at z = 0
ax1.plot([-0.55, 0.55], [0, 0], color="#39d353", lw=6, zorder=4,
         solid_capstyle="butt")
ax1.text(0, -0.10, "target / FOV (camera looks up the page)", ha="center",
         va="top", color="#39d353", fontsize=8.5, fontweight="bold")

for side in (+1, -1):
    root, (inner, outer) = arm_bulbs(side)
    # the fixed angled arm (root -> beyond outer bulb)
    tip = (root[0] + side*(S0+D2+0.18)*INV2, (S0+D2+0.18)*INV2)
    ax1.plot([root[0], tip[0]], [root[1], tip[1]], color="#cfcfcf", lw=3.2,
             zorder=2, solid_capstyle="round")
    # bulbs + beams back to the common aim point (the root) at fixed 45 deg
    for (bx, bz) in (inner, outer):
        ax1.add_patch(FancyArrowPatch((bx, bz), root, arrowstyle="-|>",
                                      color=ACCENT, mutation_scale=12, lw=1.3,
                                      alpha=0.8, zorder=4))
        bulb(ax1, bx, bz)
    # d2 along the arm (right side only)
    if side == +1:
        ax1.annotate("", xy=outer, xytext=inner,
                     arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.4))
        mx, mz = (inner[0]+outer[0])/2, (inner[1]+outer[1])/2
        ax1.text(mx+0.07, mz, "d2  (offset ALONG the arm\n→ different standoffs)",
                 color=ACCENT, fontsize=8.5, fontweight="bold", va="center", ha="left")
    # 45 deg fixed arc (left side only) between the target normal (+z) and arm
    if side == -1:
        ax1.add_patch(Arc(root, 0.34, 0.34, theta1=90, theta2=135,
                          color="orange", lw=1.8, zorder=6))
        ax1.text(root[0]-0.04, 0.22, "45°\nfixed\n(no swivel)", color="orange",
                 fontsize=8.5, fontweight="bold", ha="right", va="bottom")

# d1 baseline (between the two arm roots)
ax1.annotate("", xy=(D1H, 0.13), xytext=(-D1H, 0.13),
             arrowprops=dict(arrowstyle="<->", color="white", lw=1.5))
ax1.text(0, 0.30, "d1 = baseline (same as Config A)",
         ha="center", va="bottom", color="white", fontsize=9, fontweight="bold")

ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-0.45, 1.05)

# ============================ RIGHT: FRONT VIEW ==============================
ax2 = fig.add_subplot(gs[0, 1]); style(ax2)
ax2.set_title("FRONT VIEW (x–y) — all 4 bulbs on the SAME y = 0 axis",
              color=ACCENT, fontsize=11.5, fontweight="bold", pad=10)

# FOV box centred
ax2.add_patch(Rectangle((-0.5, -0.375), 1.0, 0.75, fc="#16222b",
                        ec="cyan", lw=2.0, zorder=3))
ax2.text(0, 0.0, "FOV", ha="center", va="center", color="cyan",
         fontsize=10, fontweight="bold", zorder=4)

# the y = 0 centreline
ax2.axhline(0, color="#666666", lw=1.2, ls="--", zorder=2)
ax2.text(1.32, 0.05, "y = 0", color="#999999", fontsize=9, va="bottom", ha="right")

# 4 bulbs projected onto y = 0 at their x positions (inner + outer, both sides)
for side in (+1, -1):
    _, (inner, outer) = arm_bulbs(side)
    for (bx, _bz), tag in ((inner, "inner"), (outer, "outer")):
        bulb(ax2, bx, 0.0)
        ax2.text(bx, -0.13, tag, ha="center", va="top", color="#aaaaaa",
                 fontsize=7.5)

ax2.text(0, -0.62, "the two bulbs per side differ only in x here;\n"
                   "their standoff difference (d2) is into the page",
         ha="center", va="top", color="#aaaaaa", fontsize=8.5, style="italic")

ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(-0.95, 0.85)

# ============================ header + footer ================================
fig.suptitle("Config B — corrected ANGLED-ARM model (NOT planar).  Have I got it now?",
             color="white", fontsize=15, fontweight="bold", y=0.95)

fig.text(0.5, 0.115,
         "Config B:  (1) all 4 bulbs on the same horizontal axis (y = 0);   "
         "(2) a FIXED 45° angled arm each side (no swivel);   (3) d1 = baseline arm-root separation (same as Config A);\n"
         "(4) d2 = offset ALONG the arm → the two bulbs on a side are at DIFFERENT standoffs (inner closer/brighter, outer further/dimmer), both aiming at ±d1/2.",
         ha="center", va="top", color="#ffd93d", fontsize=10, family="monospace")

fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved: {OUT_PNG}")
