"""
Two-head (left+right) min-CoV design at 300 mm lamp->plane standoff.

Model (per the user's diagram, 2026-06-23):
  * 2 lamp HEADS, one each side of the camera, on the camera centreline.
  * Each head = 2 bulbs treated as ONE combined footprint, beam centres 25 mm apart.
  * standoff = lamp -> viewing plane = 300 mm (lamps aimed ~normal at the FOV).
  * The two heads are separated horizontally; we find the separation that
    MINIMISES CoV on the FOV, then map it onto the fixture's d1 / d2.

Beam model: beam_derived_combined.json (refined 300/500/700 mm capture).
At 300 mm each footprint (sigma ~86 mm) is WIDER than the 128x102 mm FOV, so the
heads only need a modest separation to flatten the centre.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch, Arc

REPO = os.path.dirname(os.path.abspath(__file__))
beam = json.load(open(os.path.join(REPO, "beam_derived_combined.json")))
bm = beam["derived_beam_model"]
SXm, SXb = bm["sigma_x_vs_d"]["slope_mm_per_mm"], bm["sigma_x_vs_d"]["intercept_mm"]
SYm, SYb = bm["sigma_y_vs_d"]["slope_mm_per_mm"], bm["sigma_y_vs_d"]["intercept_mm"]

D = 300.0
SX = SXm*D + SXb
SY = SYm*D + SYb
BULB_GAP = 25.0
HFOV, VFOV = 24.0, 19.3
FW = 2*D*np.tan(np.radians(HFOV/2)); FH = 2*D*np.tan(np.radians(VFOV/2))

xs = np.linspace(-FW/2, FW/2, 181); ys = np.linspace(-FH/2, FH/2, 181)
XX, YY = np.meshgrid(xs, ys)

def field(sep):
    centres = [(h*sep/2 + b*BULB_GAP/2, 0.0) for h in (-1, 1) for b in (-1, 1)]
    irr = np.zeros_like(XX)
    for cx, cy in centres:
        irr += np.exp(-0.5*(((XX-cx)/SX)**2 + ((YY-cy)/SY)**2))
    return irr, centres

def cov(sep):
    irr, _ = field(sep); return 100*irr.std()/irr.mean()

seps = np.arange(0, 260, 1.0)
covs = np.array([cov(s) for s in seps])
i = int(np.argmin(covs)); SEP = float(seps[i]); CMIN = float(covs[i])
band = seps[covs <= CMIN + 0.1]
yv = np.exp(-0.5*((ys/SY)**2)); yfloor = 100*yv.std()/yv.mean()

# ---- figure -----------------------------------------------------------------
fig = plt.figure(figsize=(19, 7.6)); fig.patch.set_facecolor("#111111")
gs = mgridspec.GridSpec(1, 3, figure=fig, left=0.05, right=0.985, top=0.86,
                        bottom=0.12, wspace=0.28, width_ratios=[1.0, 1.0, 1.25])
def dk(ax):
    ax.set_facecolor("#0d0d0d")
    for sp in ax.spines.values(): sp.set_edgecolor("#444444")
    ax.tick_params(colors="#aaaaaa", labelsize=8)

# Panel 1: CoV vs separation
ax = fig.add_subplot(gs[0]); dk(ax); ax.grid(True, alpha=0.15, color="white")
ax.plot(seps, covs, color="#ffd93d", lw=2.2)
ax.axvspan(band.min(), band.max(), color="#4dffb8", alpha=0.18)
ax.plot(SEP, CMIN, "*", color="#ff3b3b", ms=18, mec="white", mew=1)
ax.axhline(yfloor, color="#7fd4ff", ls=":", lw=1.3)
ax.text(243, yfloor+0.12, f"sigma_y floor {yfloor:.2f}%", color="#7fd4ff",
        ha="right", fontsize=8)
ax.annotate(f"min CoV {CMIN:.2f}%\n@ sep {SEP:.0f} mm", (SEP, CMIN),
            (SEP+8, CMIN+1.4), color="white", fontsize=9,
            arrowprops=dict(arrowstyle="->", color="white"))
ax.text(band.max()+4, CMIN+0.35, f"flat band\n{band.min():.0f}-{band.max():.0f} mm",
        color="#4dffb8", fontsize=8)
ax.set_xlabel("head-centre separation [mm]", color="#aaa", fontsize=9.5)
ax.set_ylabel("CoV on FOV [%]", color="#aaa", fontsize=9.5)
ax.set_title("CoV vs head separation", color="white", fontsize=11, fontweight="bold")

# Panel 2: optimum irradiance map
ax = fig.add_subplot(gs[1]); dk(ax)
irr, centres = field(SEP)
im = ax.imshow(irr/irr.max(), origin="lower", cmap="inferno", aspect="equal",
               extent=[xs[0], xs[-1], ys[0], ys[-1]])
ax.add_patch(Rectangle((-FW/2, -FH/2), FW, FH, fill=False, ec="#4dffb8", lw=1.8, ls="--"))
for cx, cy in centres:
    ax.plot(cx, cy, "+", color="#7fd4ff", ms=10, mew=1.7)
plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046).ax.tick_params(labelsize=7, colors="white")
ax.set_xlabel("x [mm]", color="#aaa", fontsize=9.5); ax.set_ylabel("y [mm]", color="#aaa", fontsize=9.5)
ax.set_title(f"Optimum field  (sep {SEP:.0f} mm)\n4 bulb centres (+), FOV dashed",
             color="white", fontsize=11, fontweight="bold")

# Panel 3: corrected plan-view fixture sketch
ax = fig.add_subplot(gs[2]); ax.set_facecolor("#0d0d0d")
for sp in ax.spines.values(): sp.set_edgecolor("#444444")
ax.tick_params(colors="#888", labelsize=7)
# example split: short arms, baseline carries the separation
A = 45.0; ca = np.cos(np.radians(A))
d2_ex = 60.0
d1_ex = SEP - np.sqrt(2)*d2_ex
# plan coords: x horizontal, y toward plane (up)
y_lamp = d2_ex*np.sin(np.radians(A)); y_plane = D + y_lamp
ax.plot([-260, 260], [y_plane, y_plane], color="#888", lw=1.2)
ax.plot([-FW/2, FW/2], [y_plane, y_plane], color="#4dffb8", lw=5, solid_capstyle="butt")
ax.text(0, y_plane+12, f"viewing plane (FOV {FW:.0f} mm)", color="#4dffb8", ha="center", fontsize=8)
# camera + arm roots
ax.add_patch(Rectangle((-18, -18), 36, 22, fc="#222", ec="#ccc", lw=1.2))
ax.text(0, -30, "camera", color="#ccc", ha="center", va="top", fontsize=8)
for s in (-1, 1):
    root = (s*d1_ex/2, 0.0)
    lamp = (s*(d1_ex/2 + d2_ex*ca), y_lamp)
    bc   = (s*SEP/2, y_plane)        # beam centre on plane
    ax.plot([root[0], s*(d1_ex/2 + d2_ex*ca*1.5)], [0, y_lamp*1.5],
            color="#666", lw=5, alpha=0.5, solid_capstyle="round")   # 45 arm
    ax.add_patch(FancyArrowPatch(lamp, bc, arrowstyle="-|>", mutation_scale=11,
                                 color="#7fd4ff", lw=1.3, ls="--"))
    ax.add_patch(Circle(lamp, 11, fc="#ffd11a", ec="white", lw=1))
    ax.plot(bc[0], y_plane, "x", color="#ff7b7b", ms=10, mew=2.2)
# dims
ax.annotate("", xy=(-d1_ex/2, -8), xytext=(d1_ex/2, -8),
            arrowprops=dict(arrowstyle="<->", color="#ff7b7b", lw=1.3))
ax.text(0, -12, f"d1 = {d1_ex:.0f} mm", color="#ff7b7b", ha="center", va="top", fontsize=8.5)
ax.annotate("", xy=(d1_ex/2, 0), xytext=(d1_ex/2 + d2_ex*ca, y_lamp),
            arrowprops=dict(arrowstyle="<->", color="#ffd93d", lw=1.3))
ax.text(d1_ex/2 + d2_ex*ca + 8, y_lamp/2, f"d2 = {d2_ex:.0f} mm", color="#ffd93d", fontsize=8.5)
ax.annotate("", xy=(-SEP/2, y_plane-26), xytext=(SEP/2, y_plane-26),
            arrowprops=dict(arrowstyle="<->", color="#4dffb8", lw=1.3))
ax.text(0, y_plane-40, f"beam-centre sep = {SEP:.0f} mm", color="#4dffb8", ha="center", fontsize=8.5)
ax.text(-250, D*0.5, "standoff\n300 mm", color="#9dd6ff", fontsize=8)
ax.set_title("Fixture (plan view) — one example d1/d2 split",
             color="white", fontsize=11, fontweight="bold")
ax.set_xlim(-270, 270); ax.set_ylim(-55, y_plane+45); ax.set_aspect("equal")

fig.suptitle("Two-head min-CoV design @ 300 mm  |  each head = 2 bulbs (25 mm apart) combined  |  "
             f"OPTIMUM: heads {SEP:.0f} mm apart -> CoV {CMIN:.2f}% (floored by sigma_y)",
             color="white", fontsize=12.5, y=0.965)
OUT = os.path.join(REPO, "06d_twohead_mincov.png")
fig.savefig(OUT, dpi=135, facecolor=fig.get_facecolor()); plt.close(fig)

json.dump({"standoff_mm": D, "sigma_x_mm": SX, "sigma_y_mm": SY,
           "fov_w_mm": FW, "fov_h_mm": FH, "bulb_gap_mm": BULB_GAP,
           "optimal_head_sep_mm": SEP, "min_cov_pct": CMIN,
           "flat_band_mm": [float(band.min()), float(band.max())],
           "sigma_y_cov_floor_pct": yfloor,
           "mapping": "head_sep = d1 + sqrt(2)*d2  (lamps aimed ~normal at FOV)"},
          open(os.path.join(REPO, "06d_twohead_mincov.json"), "w"), indent=2)
print(f"Saved {OUT}")
print(f"optimum sep {SEP:.0f} mm  CoV {CMIN:.2f}%  band {band.min():.0f}-{band.max():.0f}  yfloor {yfloor:.2f}%")
