"""
Bracket-design explainer for the 45deg-locked max-CoV sweep (06b).

Draws, for Config A and Config B, the physical bracket cross-section and marks
EXACTLY what the optimiser is allowed to move:

    SWEPT  : d1  (baseline / aim separation) and d2 (along-arm lamp spacing)
    LOCKED : lamp tilt = 45 deg
    OBJECTIVE : maximise CoV of irradiance on the camera FOV

Config A — bracket arms ABOVE + BELOW the FOV  -> baseline d1 runs along the
            VERTICAL FOV (fov_h); each lamp pitched 45 deg in the vertical plane.
Config B — bracket arms LEFT + RIGHT of the FOV -> baseline d1 runs along the
            HORIZONTAL FOV (fov_w); each lamp yawed 45 deg in the horizontal plane.

Each bracket arm carries 2 lamps spaced d2 along the 45 deg arm, so the inner
lamp sits at standoff d and the outer at d + d2 (the outer is further -> larger,
dimmer footprint). Both lamps on an arm aim at the same target line (+-d1/2).

Reads the optimum from 06b_lamp_config_maxcov45.json (run that first).
Geometry is drawn to scale in the FOV plane (d1, d2, FOV) but the full standoff
is compressed with a break mark — the arm is physically ~d long.
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch, Arc

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
JSON_IN   = os.path.join(REPO_ROOT, "06b_lamp_config_maxcov45.json")
OUT_PNG   = os.path.join(REPO_ROOT, "06b_bracket_design.png")
REP_D     = 400                      # representative standoff for the drawn geometry

with open(JSON_IN) as f:
    data = json.load(f)
results = data["results"]
sweep   = data["sweep_settings"]
cfg_col = {"A": "#ff6b6b", "B": "#4d96ff"}
INV2    = 1.0 / np.sqrt(2.0)

def rep(cfg, d=REP_D):
    return next(r for r in results if r["config"] == cfg and r["standoff_mm"] == d)

# -----------------------------------------------------------------------------
def draw_bracket(ax, cfg):
    """One cross-section panel. In-plane axis 'u' horizontal on the page
    (= vertical FOV for A, horizontal FOV for B); standoff axis 'z' vertical."""
    accent = cfg_col[cfg]
    r   = rep(cfg)
    F   = r["fov_dim_mm"]            # relevant FOV dimension (vertical for A)
    d1  = r["best_d1_mm"]
    d2  = r["best_d2_mm"]
    cov = r["max_cov_pct"]

    ax.set_facecolor("#0d0d0d")
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(accent); sp.set_linewidth(2.5)

    # ---- target / FOV (green strip at z = 0) --------------------------------
    ax.plot([-F/2, F/2], [0, 0], color="#39d353", lw=6, zorder=4,
            solid_capstyle="butt")
    fov_axis = "VERTICAL FOV" if cfg == "A" else "HORIZONTAL FOV"
    ax.annotate("", xy=(F/2, -22), xytext=(-F/2, -22),
                arrowprops=dict(arrowstyle="<->", color="#39d353", lw=1.3))
    ax.text(0, -30, f"{fov_axis} = {F:.0f} mm", ha="center", va="top",
            color="#39d353", fontsize=9, fontweight="bold")
    # camera at FOV centre
    ax.add_patch(Rectangle((-12, -10), 24, 12, fc="#223344", ec="cyan",
                            lw=1.3, zorder=5))
    ax.text(0, -4, "CAM", ha="center", va="center", color="white",
            fontsize=6.5, fontweight="bold", zorder=6)

    # ---- the two bracket arms (45 deg) + 2 lamps each -----------------------
    z_top   = 1.15 * F                       # visual top of the arm (standoff compressed)
    z_inner = z_top - d2 * INV2              # inner lamp one d2-step below outer (to scale)
    z_inner = max(z_inner, 0.45 * F)
    for side in (+1, -1):                    # +1 = top/right arm, -1 = bottom/left arm
        ap = (side * d1/2, 0.0)              # aim point on the target
        # arm direction up-and-out at 45 deg
        outer = (ap[0] + side * z_top   , z_top)
        inner = (ap[0] + side * z_inner , z_inner)
        # arm rail
        ax.plot([ap[0], outer[0]], [ap[1], outer[1]], color="#cfcfcf",
                lw=3.5, zorder=2, solid_capstyle="round")
        # break mark on the rail (standoff not to scale)
        bz = 0.22 * F
        bx = ap[0] + side * bz
        ax.plot([bx - side*7, bx + side*4], [bz - 4, bz + 2], color="#0d0d0d",
                lw=6, zorder=3)
        ax.plot([bx - side*7, bx + side*4], [bz - 8, bz - 2], color="#cfcfcf",
                lw=1.2, zorder=3)
        ax.plot([bx - side*7 + side*5, bx + side*4 + side*5],
                [bz - 4, bz + 2], color="#cfcfcf", lw=1.2, zorder=3)
        # lamps (inner + outer) + beam arrows to the common aim point
        for (lx, lz), tag in ((inner, "inner"), (outer, "outer")):
            ax.add_patch(FancyArrowPatch((lx, lz), ap, arrowstyle="-|>",
                                         color=accent, mutation_scale=12,
                                         lw=1.3, alpha=0.8, zorder=4))
            ax.add_patch(Circle((lx, lz), 7.5, fc="#1a2a3a", ec=accent,
                                lw=2.0, zorder=6))
            ax.add_patch(Circle((lx, lz), 3.0, fc="#fff2a0", ec="none",
                                alpha=0.95, zorder=7))
        # d2 dimension between the two lamps (to scale, along the arm)
        ax.annotate("", xy=outer, xytext=inner,
                    arrowprops=dict(arrowstyle="<->", color=accent, lw=1.4))
        if side == +1:
            mx, mz = (inner[0]+outer[0])/2, (inner[1]+outer[1])/2
            ax.text(mx + 12, mz, f"d2 = {d2:.0f} mm", color=accent,
                    fontsize=9, fontweight="bold", va="center", ha="left")
        # 45 deg arc between the vertical (target normal) and the arm
        if side == -1:
            ax.add_patch(Arc(ap, 46, 46, theta1=90, theta2=135,
                             color="orange", lw=1.8, zorder=6))
            ax.text(ap[0] - 6, 30, "45°\n(locked)", color="orange", fontsize=8.5,
                    fontweight="bold", ha="right", va="bottom")

    # ---- d1 baseline dimension (between the two aim points) ------------------
    ax.annotate("", xy=(d1/2, 14), xytext=(-d1/2, 14),
                arrowprops=dict(arrowstyle="<->", color="white", lw=1.4))
    ax.text(0, 18, f"d1 = {d1:.0f} mm  ({r['best_d1_frac']:.2f} x FOV)",
            ha="center", va="bottom", color="white", fontsize=9, fontweight="bold")

    title = ("Config A — arms ABOVE + BELOW FOV  (45° pitch, vertical plane)"
             if cfg == "A" else
             "Config B — arms LEFT + RIGHT of FOV  (45° yaw, horizontal plane)")
    ax.set_title(title, color=accent, fontsize=12, fontweight="bold", pad=10)

    ax.text(0.5, -0.10,
            f"optimum @ d = {REP_D} mm  →  d1 = {d1:.0f} mm,  d2 = {d2:.0f} mm,  "
            f"max CoV = {cov:.1f}%",
            transform=ax.transAxes, ha="center", va="top",
            color="#dddddd", fontsize=9.5, family="monospace")

    span = d1/2 + z_top + 20
    ax.set_xlim(-span, span)
    ax.set_ylim(-55, z_top + 35)

# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(18, 11))
fig.patch.set_facecolor("#111111")
gs = mgridspec.GridSpec(2, 2, figure=fig, left=0.04, right=0.97,
                        top=0.84, bottom=0.16, hspace=0.30, wspace=0.18,
                        height_ratios=[1.0, 0.32])

for ci, cfg in enumerate(("A", "B")):
    draw_bracket(fig.add_subplot(gs[0, ci]), cfg)

# per-config optimum tables along the bottom
for ci, cfg in enumerate(("A", "B")):
    axt = fig.add_subplot(gs[1, ci]); axt.axis("off")
    axt.set_title(f"Config {cfg} optimum per standoff",
                  color=cfg_col[cfg], fontsize=10, fontweight="bold", pad=2)
    rows = [["d (mm)", "d1 (mm)", "d1 / FOV", "d2 (mm)", "max CoV"]]
    for r in sorted([x for x in results if x["config"] == cfg],
                    key=lambda x: x["standoff_mm"]):
        rows.append([f"{r['standoff_mm']}", f"{r['best_d1_mm']:.0f}",
                     f"{r['best_d1_frac']:.2f}", f"{r['best_d2_mm']:.0f}",
                     f"{r['max_cov_pct']:.1f}%"])
    colx = [0.04, 0.26, 0.46, 0.68, 0.86]
    for ri, row in enumerate(rows):
        y = 0.80 - ri * 0.18
        for cx, val in zip(colx, row):
            axt.text(cx, y, val, transform=axt.transAxes,
                     color="white" if ri else cfg_col[cfg],
                     fontsize=9.5, fontweight="bold" if ri == 0 else "normal",
                     va="top")

# header: what is being optimised
d1_lo, d1_hi = sweep["d1_fracs_of_fov"][0], sweep["d1_fracs_of_fov"][-1]
d2_lo, d2_hi = sweep["d2_mm"][0], sweep["d2_mm"][-1]
fig.suptitle("What the 45°-locked sweep optimises — bracket design (Config A vs B)",
             color="white", fontsize=15, fontweight="bold", y=0.965)
fig.text(0.5, 0.895,
         f"OBJECTIVE: maximise CoV on FOV     "
         f"SWEPT:  d1 ∈ [{d1_lo:.1f}, {d1_hi:.1f}] × FOV   |   "
         f"d2 ∈ [{d2_lo:.0f}, {d2_hi:.0f}] mm along the arm     "
         f"LOCKED:  lamp tilt = 45°",
         ha="center", va="top", color="#ffd93d", fontsize=11,
         family="monospace")

fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved: {OUT_PNG}")
