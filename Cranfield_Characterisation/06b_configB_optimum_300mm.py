"""
Most-uniform (MIN CoV) Config B case at 300 mm standoff, shown graphically.

Config B geometry (angled-arm V-trough, confirmed):
  * 4 bulbs on y = 0; two sides aiming at x = +-d1/2.
  * fixed 45deg arm each side; inner bulb at standoff d, outer at d + d2.
  * sigma_x stretched by 1/cos45 (horizontal tilt); sigma_y unchanged.

This sweeps a fine (d1, d2) grid at d = 300 mm and finds the pair that MINIMISES
CoV on the FOV (= the design optimum, most uniform), then draws:
  (1) the irradiance map at that optimum,
  (2) the CoV(d1, d2) landscape with the optimum marked,
  (3) horizontal + vertical irradiance cross-sections through the FOV centre.
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
from matplotlib.patches import Rectangle

REPO_ROOT  = os.path.dirname(os.path.abspath(__file__))
BEAM_JSON  = os.path.join(REPO_ROOT, "beam_derived_combined.json")
LENS_JSON  = os.path.join(REPO_ROOT, "05_lens_fov_sweep.json")
OUT_PNG    = os.path.join(REPO_ROOT, "06b_configB_optimum_300mm.png")

D        = 300                       # standoff (mm)
LENS_F   = 18.0
THETA    = 45.0
COS_T    = np.cos(np.radians(THETA))
GRID_PX  = 161

# fine design sweep (NOT the constrained max-CoV ranges — we want the true min)
D1_FRAC  = np.linspace(0.30, 5.00, 140)  # d1 as fraction of horizontal FOV
D2_MM    = np.linspace(50.0, 900.0, 70)  # along-arm offset (mm)

with open(BEAM_JSON) as f: beam = json.load(f)
with open(LENS_JSON) as f: lens_sweep = json.load(f)
bm   = beam["derived_beam_model"]
SX_M = bm["sigma_x_vs_d"]["slope_mm_per_mm"];  SX_B = bm["sigma_x_vs_d"]["intercept_mm"]
SY_M = bm["sigma_y_vs_d"]["slope_mm_per_mm"];  SY_B = bm["sigma_y_vs_d"]["intercept_mm"]
PK_K = bm["peak_vs_d_powerlaw"]["K_amp_K_mm_to_power"]
PK_E = bm["peak_vs_d_powerlaw"]["exponent"]
def sigma_x(d): return SX_M*d + SX_B
def sigma_y(d): return SY_M*d + SY_B
def peak(d):    return PK_K * d**PK_E

cell = next(g for g in lens_sweep["grid"]
            if g["lens_f_mm"] == LENS_F and g["standoff_mm"] == D)
FOV_W, FOV_H = cell["fov_w_mm"], cell["fov_h_mm"]

pad = 0.3*max(FOV_W, FOV_H)
xs  = np.linspace(-FOV_W/2-pad, FOV_W/2+pad, GRID_PX)
ys  = np.linspace(-FOV_H/2-pad, FOV_H/2+pad, GRID_PX)
XX, YY = np.meshgrid(xs, ys)
IN_FOV = (np.abs(XX) <= FOV_W/2) & (np.abs(YY) <= FOV_H/2)

def irr_map(d1, d2):
    irr = np.zeros_like(XX); centres = []
    for side in (+1, -1):
        for d_eff in (D, D + d2):
            sx_eff = sigma_x(d_eff)/COS_T
            sy_eff = sigma_y(d_eff)
            pk     = peak(d_eff)*COS_T
            cx, cy = side*d1/2, 0.0
            irr += pk*np.exp(-0.5*(((XX-cx)/sx_eff)**2 + ((YY-cy)/sy_eff)**2))
            centres.append((cx, cy))
    return irr, centres

def cov_of(irr):
    v = irr[IN_FOV]; m = v.mean()
    return 100*v.std()/m if m > 0 else np.nan

# ---- sweep for the MIN CoV --------------------------------------------------
COV = np.full((len(D1_FRAC), len(D2_MM)), np.nan)
for i, f1 in enumerate(D1_FRAC):
    for j, d2 in enumerate(D2_MM):
        COV[i, j] = cov_of(irr_map(f1*FOV_W, d2)[0])
i_b, j_b = np.unravel_index(np.nanargmin(COV), COV.shape)
D1_OPT   = D1_FRAC[i_b]*FOV_W
D2_OPT   = D2_MM[j_b]
COV_OPT  = COV[i_b, j_b]
irr_opt, centres = irr_map(D1_OPT, D2_OPT)

print(f"Config B @ {D} mm — most uniform (MIN CoV):")
print(f"  d1 = {D1_OPT:.1f} mm ({D1_FRAC[i_b]:.2f} x FOV_w)")
print(f"  d2 = {D2_OPT:.1f} mm")
print(f"  CoV = {COV_OPT:.2f} %   (range over grid: {np.nanmin(COV):.2f}–{np.nanmax(COV):.2f} %)")

# ---- figure -----------------------------------------------------------------
fig = plt.figure(figsize=(18, 8.5))
fig.patch.set_facecolor("#111111")
gs = mgridspec.GridSpec(2, 2, figure=fig, left=0.06, right=0.96,
                        top=0.86, bottom=0.10, wspace=0.28, hspace=0.42,
                        width_ratios=[1.15, 1.0])

def dark(ax):
    ax.set_facecolor("#0a0a0a")
    for sp in ax.spines.values(): sp.set_edgecolor("#444444")
    ax.tick_params(colors="#bbbbbb", labelsize=8)

# (1) irradiance map at optimum (spans left column)
ax1 = fig.add_subplot(gs[:, 0]); dark(ax1); ax1.set_aspect("equal")
im = ax1.imshow(irr_opt, cmap="inferno", origin="lower",
                extent=[xs[0], xs[-1], ys[0], ys[-1]], vmin=0)
ax1.add_patch(Rectangle((-FOV_W/2, -FOV_H/2), FOV_W, FOV_H, fc="none",
                        ec="cyan", lw=2.2, zorder=4))
for cx, cy in centres:
    ax1.plot(cx, cy, "+", color="white", ms=14, mew=2, zorder=5)
ax1.set_title(f"Config B — most uniform irradiance @ d = {D} mm\n"
              f"d1 = {D1_OPT:.0f} mm,  d2 = {D2_OPT:.0f} mm  →  CoV = {COV_OPT:.2f}%",
              color="white", fontsize=11.5, fontweight="bold", pad=8)
ax1.set_xlabel("x (mm)", color="#bbbbbb"); ax1.set_ylabel("y (mm)", color="#bbbbbb")
cb = plt.colorbar(im, ax=ax1, pad=0.02, fraction=0.046)
cb.ax.tick_params(labelsize=7, colors="white")
cb.set_label("Irradiance (relative)", color="white", fontsize=8)
ax1.text(0, FOV_H/2+pad*0.5, "cyan box = FOV   +  = beam centres (±d1/2)",
         ha="center", va="bottom", color="#888888", fontsize=8)

# (2) CoV(d1, d2) landscape
ax2 = fig.add_subplot(gs[0, 1]); dark(ax2)
ext = [D2_MM[0], D2_MM[-1], D1_FRAC[0]*FOV_W, D1_FRAC[-1]*FOV_W]
cm = ax2.imshow(COV, origin="lower", aspect="auto", extent=ext, cmap="viridis")
ax2.plot(D2_OPT, D1_OPT, "*", color="#ff4444", ms=20, mec="white", mew=1.2, zorder=5)
ax2.text(D2_OPT, D1_OPT, f"  min {COV_OPT:.2f}%", color="white",
         fontsize=9, fontweight="bold", va="center")
ax2.set_title("CoV landscape over (d1, d2)  —  ★ = most uniform",
              color="white", fontsize=10.5, fontweight="bold", pad=6)
ax2.set_xlabel("d2 — along-arm offset (mm)", color="#bbbbbb")
ax2.set_ylabel("d1 — baseline (mm)", color="#bbbbbb")
cb2 = plt.colorbar(cm, ax=ax2, pad=0.02, fraction=0.046)
cb2.ax.tick_params(labelsize=7, colors="white"); cb2.set_label("CoV (%)", color="white", fontsize=8)

# (3) cross-sections through FOV centre
ax3 = fig.add_subplot(gs[1, 1]); dark(ax3); ax3.set_facecolor("#161616")
iy0 = np.argmin(np.abs(ys)); ix0 = np.argmin(np.abs(xs))
prof_x = irr_opt[iy0, :]; prof_y = irr_opt[:, ix0]
ax3.plot(xs, prof_x/prof_x.max(), color="#4d96ff", lw=2, label="horizontal (y=0)")
ax3.plot(ys, prof_y/prof_y.max(), color="#ffb84d", lw=2, label="vertical (x=0)")
ax3.axvspan(-FOV_W/2, FOV_W/2, color="cyan", alpha=0.07)
ax3.axvline(-FOV_W/2, color="cyan", lw=1, ls=":"); ax3.axvline(FOV_W/2, color="cyan", lw=1, ls=":")
# show the in-FOV min/max band for the horizontal profile
infx = (np.abs(xs) <= FOV_W/2)
ax3.axhspan(prof_x[infx].min()/prof_x.max(), prof_x[infx].max()/prof_x.max(),
            color="#4d96ff", alpha=0.10)
ax3.set_title("Irradiance cross-sections through FOV centre (normalised)",
              color="white", fontsize=10.5, fontweight="bold", pad=6)
ax3.set_xlabel("position (mm)", color="#bbbbbb")
ax3.set_ylabel("relative irradiance", color="#bbbbbb")
ax3.legend(fontsize=8.5, facecolor="#222222", labelcolor="white", edgecolor="#555555")
ax3.set_ylim(0, 1.05)

fig.suptitle("Config B optimum (most uniform) — 18 mm Boson+ lens, 300 mm standoff, 45° locked",
             color="white", fontsize=14, fontweight="bold", y=0.955)

fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved: {OUT_PNG}")
