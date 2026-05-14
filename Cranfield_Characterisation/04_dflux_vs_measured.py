"""
Compare the DFLUX-formulated beam shape in experimentalpulse.f against the
characterised beam from beam_shape_summary_seq.json (300/400/500 mm @ 70 deg).

Shape-only comparison. No flux magnitudes -- both profiles normalised to
peak=1. We compare:
  - sigma(d)  -- divergence slope and virtual-source intercept
  - normalised horizontal / vertical / radial profiles
  - ellipse footprint vs the circular DFLUX ellipse
  - aspect ratio (measured) vs DFLUX (1:1 by construction)

Convention conversion:
  - beam_shape_summary_seq.py uses the standard Gaussian convention
        I(r) = exp(-0.5*(r/sigma_std)**2)
  - experimentalpulse.f uses (no 0.5):
        I(r) = exp(-(r/sigma_D)**2)
  - Same physical beam: sigma_D = sqrt(2) * sigma_std
  This script does the conversion explicitly so the comparison is fair.
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
from matplotlib.patches import Ellipse, Rectangle

REPO_ROOT  = os.path.dirname(os.path.abspath(__file__))
INPUT_JSON = os.path.join(REPO_ROOT, "beam_shape_summary_seq.json")
OUTPUT_PNG = os.path.join(REPO_ROOT, "dflux_vs_measured.png")
OUTPUT_JSON = os.path.join(REPO_ROOT, "dflux_vs_measured.json")

# ---- DFLUX reference (from experimentalpulse.f) ----------------------------
# I(r) = PEAK * exp(-(r/SIGMA)**n)   (no 0.5 factor; circular; n = 2)
# SIGMA(d) = SIGMA_PER_MM * d + SIGMA_VIRT  [mm]
DFLUX_SIGMA_PER_MM = 0.1751
DFLUX_SIGMA_VIRT   = 2.79
DFLUX_SG_N         = 2.0
SQ2 = np.sqrt(2.0)

# ---- Specimen rectangle for context ---------------------------------------
SPEC_W_MM, SPEC_H_MM = 320.0, 175.0

# ---- Load measured data ----------------------------------------------------
with open(INPUT_JSON) as f:
    M = json.load(f)

ds      = np.array(M["standoffs_mm"], dtype=float)
sx_std  = np.array([p["sigma_x_mm"] for p in M["per_standoff"]])
sy_std  = np.array([p["sigma_y_mm"] for p in M["per_standoff"]])

# Convert measured to DFLUX convention so both speak the same language.
sx_D = sx_std * SQ2
sy_D = sy_std * SQ2

# Linear fits in DFLUX convention
cx = np.polyfit(ds, sx_D, 1)   # slope, intercept
cy = np.polyfit(ds, sy_D, 1)
half_div_x = np.degrees(np.arctan(cx[0]))
half_div_y = np.degrees(np.arctan(cy[0]))
# DFLUX equivalents
half_div_D = np.degrees(np.arctan(DFLUX_SIGMA_PER_MM))

# Stats
sigma_geom_D = np.sqrt(sx_D * sy_D)
aspect = sx_D / sy_D

print("== Shape comparison (DFLUX convention: exp(-(r/sigma)^2)) ==")
print(f"  d (mm):                  {ds.astype(int).tolist()}")
print(f"  measured sigma_x_D:      {[f'{v:.1f}' for v in sx_D]}  mm")
print(f"  measured sigma_y_D:      {[f'{v:.1f}' for v in sy_D]}  mm")
print(f"  DFLUX sigma:             "
      f"{[f'{DFLUX_SIGMA_PER_MM*d + DFLUX_SIGMA_VIRT:.1f}' for d in ds]}  mm")
print(f"  aspect (sx/sy):          {[f'{a:.3f}' for a in aspect]}")
print()
print(f"  measured sigma_x(d) = {cx[0]:+.4f} d + {cx[1]:+.2f}   "
      f"(half-div = {half_div_x:+.2f} deg)")
print(f"  measured sigma_y(d) = {cy[0]:+.4f} d + {cy[1]:+.2f}   "
      f"(half-div = {half_div_y:+.2f} deg)")
print(f"  DFLUX    sigma(d)   = {DFLUX_SIGMA_PER_MM:.4f} d + {DFLUX_SIGMA_VIRT:.2f}   "
      f"(half-div = {half_div_D:.2f} deg)")
slope_err_x = (cx[0] - DFLUX_SIGMA_PER_MM)/DFLUX_SIGMA_PER_MM*100
slope_err_y = (cy[0] - DFLUX_SIGMA_PER_MM)/DFLUX_SIGMA_PER_MM*100
print(f"  slope error vs DFLUX: x={slope_err_x:+.1f}%,  y={slope_err_y:+.1f}%")

# =============================================================================
# Figure: 3 rows
#  Row 0: sigma(d) measured vs DFLUX | divergence cone | aspect-ratio
#  Row 1: per-standoff overlays (measured ellipse + DFLUX ellipse + spec)
#  Row 2: normalised radial profiles -- measured Gaussian vs DFLUX

fig = plt.figure(figsize=(20, 18))
fig.patch.set_facecolor("#101010")
gs = mgridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.30,
                        left=0.05, right=0.97, top=0.94, bottom=0.05,
                        height_ratios=[1.0, 1.05, 1.0])

def dark(ax, grid=True):
    ax.set_facecolor("#1a1a1a")
    for sp in ax.spines.values(): sp.set_edgecolor("#444")
    ax.tick_params(colors="#aaa", labelsize=8)
    if grid: ax.grid(True, alpha=0.12, color="white")

COLS = ["#6bcb77", "#ffd93d", "#ff6b6b"]

# --- Row 0, col 0: sigma(d) -------------------------------------------------
ax = fig.add_subplot(gs[0, 0]); dark(ax)
d_line = np.linspace(min(ds.min(), 250)-10, max(ds.max(), 550)+10, 400)
ax.plot(d_line, DFLUX_SIGMA_PER_MM*d_line + DFLUX_SIGMA_VIRT,
        color="cyan", lw=2.2, label=f"DFLUX: {DFLUX_SIGMA_PER_MM:.4f} d + {DFLUX_SIGMA_VIRT:.2f}")
ax.plot(d_line, np.polyval(cx, d_line), color="#ffd93d", lw=1.7, ls="--",
        label=f"meas $\\sigma_x$: {cx[0]:+.4f} d + {cx[1]:+.2f}")
ax.plot(d_line, np.polyval(cy, d_line), color="#4d96ff", lw=1.7, ls="--",
        label=f"meas $\\sigma_y$: {cy[0]:+.4f} d + {cy[1]:+.2f}")
for d, sx, sy, col in zip(ds, sx_D, sy_D, COLS):
    ax.scatter(d, sx, s=160, marker="*", color=col, ec="white", lw=0.5, zorder=5)
    ax.scatter(d, sy, s=120, marker="s", color=col, ec="white", lw=0.5, zorder=5)
    ax.text(d+5, sx+1.5, f"{sx:.0f}", color=col, fontsize=8.5, fontweight="bold")
    ax.text(d+5, sy-3.0, f"{sy:.0f}", color=col, fontsize=8.5, fontweight="bold")
ax.scatter([], [], s=110, color="white", marker="*", label="meas $\\sigma_x$")
ax.scatter([], [], s=80,  color="white", marker="s", label="meas $\\sigma_y$")
ax.set_xlabel("standoff d (mm)", color="#aaa", fontsize=10)
ax.set_ylabel("$\\sigma$ (mm, DFLUX convention)", color="#aaa", fontsize=10)
ax.set_title("$\\sigma$(d) — measured vs DFLUX  |  same beam shape, different size",
             color="white", fontsize=11)
ax.legend(facecolor="#111", labelcolor="white", edgecolor="#555",
          fontsize=8.5, loc="upper left")

# --- Row 0, col 1: divergence cone diagram ----------------------------------
ax = fig.add_subplot(gs[0, 1]); dark(ax, grid=False)
d_show = np.linspace(0, max(ds.max() + 50, 720), 200)
sig_D_path  = DFLUX_SIGMA_PER_MM*d_show + DFLUX_SIGMA_VIRT
sig_x_path  = np.polyval(cx, d_show)
sig_y_path  = np.polyval(cy, d_show)
# DFLUX cone (cyan)
ax.fill_between(d_show, -sig_D_path, sig_D_path, color="cyan", alpha=0.12)
ax.plot(d_show,  sig_D_path, color="cyan", lw=1.8)
ax.plot(d_show, -sig_D_path, color="cyan", lw=1.8, label="DFLUX 1$\\sigma$ envelope")
# Meas x cone
ax.plot(d_show,  sig_x_path, color="#ffd93d", lw=1.5, ls="--", label="meas $\\sigma_x$ envelope")
ax.plot(d_show, -sig_x_path, color="#ffd93d", lw=1.5, ls="--")
ax.plot(d_show,  sig_y_path, color="#4d96ff", lw=1.5, ls=":", label="meas $\\sigma_y$ envelope")
ax.plot(d_show, -sig_y_path, color="#4d96ff", lw=1.5, ls=":")
for d, col in zip(ds, COLS):
    ax.axvline(d, color=col, lw=1, alpha=0.5)
    ax.text(d, ax.get_ylim()[1]*0.92 if ax.get_ylim()[1] else 100,
            f"{int(d)}", color=col, ha="center", fontsize=8)
ax.set_xlabel("standoff d (mm)", color="#aaa", fontsize=10)
ax.set_ylabel("offset from beam centre (mm)", color="#aaa", fontsize=10)
ax.set_title(f"Divergence cones  |  half-div: meas x={half_div_x:.1f}$^\\circ$  "
             f"y={half_div_y:.1f}$^\\circ$  vs  DFLUX={half_div_D:.1f}$^\\circ$",
             color="white", fontsize=11)
ax.legend(facecolor="#111", labelcolor="white", edgecolor="#555",
          fontsize=8.5, loc="upper left")
ax.set_xlim(0, max(ds.max() + 50, 720))

# --- Row 0, col 2: aspect ratio --------------------------------------------
ax = fig.add_subplot(gs[0, 2]); dark(ax)
for d, a, col in zip(ds, aspect, COLS):
    ax.scatter(d, a, s=160, marker="D", color=col, ec="white", lw=0.5, zorder=5)
    ax.text(d+6, a+0.005, f"{a:.3f}", color=col, fontsize=9)
mean_a = aspect.mean()
ax.axhline(mean_a, color="white", lw=1, ls="--", alpha=0.5,
           label=f"mean = {mean_a:.3f}")
ax.axhline(1.0, color="cyan", lw=2, label="DFLUX (1.000, circular)")
ax.set_xlabel("standoff d (mm)", color="#aaa", fontsize=10)
ax.set_ylabel("aspect $\\sigma_x / \\sigma_y$", color="#aaa", fontsize=10)
ax.set_title("Ellipticity (measured vs DFLUX circular)",
             color="white", fontsize=11)
ax.legend(facecolor="#111", labelcolor="white", edgecolor="#555",
          fontsize=9, loc="best")
ax.set_ylim(0.95, max(1.10, aspect.max() + 0.03))

# --- Row 1: per-standoff ellipse comparison --------------------------------
for j, (d, sx, sy, col) in enumerate(zip(ds, sx_D, sy_D, COLS)):
    ax = fig.add_subplot(gs[1, j]); dark(ax, grid=False)
    sig_D = DFLUX_SIGMA_PER_MM*d + DFLUX_SIGMA_VIRT
    # measured ellipses (1s, 2s)
    for sc, ls in [(1, "-"), (2, "--")]:
        ax.add_patch(Ellipse((0,0), 2*sx*sc, 2*sy*sc, fc="none",
                              ec=col, lw=1.8, ls=ls,
                              label="meas 1$\\sigma$" if sc==1 else "meas 2$\\sigma$"))
    # DFLUX circular ellipses
    for sc, ls in [(1, "-"), (2, "--")]:
        ax.add_patch(Ellipse((0,0), 2*sig_D*sc, 2*sig_D*sc, fc="none",
                              ec="cyan", lw=1.5, ls=ls,
                              label="DFLUX 1$\\sigma$" if sc==1 else "DFLUX 2$\\sigma$"))
    # specimen
    ax.add_patch(Rectangle((-SPEC_W_MM/2, -SPEC_H_MM/2), SPEC_W_MM, SPEC_H_MM,
                            fc="none", ec="lime", lw=1.5, ls="--",
                            label="Specimen 320x175mm"))
    ax.plot(0, 0, "+", color="white", ms=10, mew=2)
    lim = max(sx, sy, sig_D) * 2.4
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim*0.7, lim*0.7)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm, board frame)", color="#aaa", fontsize=9)
    ax.set_ylabel("y (mm)", color="#aaa", fontsize=9)
    delta_pct = ((sx + sy)/2 - sig_D) / sig_D * 100
    ax.set_title(f"{int(d)} mm  |  meas $\\sigma_x$={sx:.0f}  $\\sigma_y$={sy:.0f}  "
                 f"|  DFLUX $\\sigma$={sig_D:.0f}  |  $\\Delta$={delta_pct:+.1f}%",
                 color=col, fontsize=10)
    if j == 0:
        ax.legend(facecolor="#111", labelcolor="white", edgecolor="#555",
                  fontsize=8, loc="lower right")

# --- Row 2: normalised profiles --------------------------------------------
# Row 2 col 0: horizontal profiles
ax_h = fig.add_subplot(gs[2, 0]); dark(ax_h)
# Row 2 col 1: vertical profiles
ax_v = fig.add_subplot(gs[2, 1]); dark(ax_v)
# Row 2 col 2: radial overlay (per-standoff normalised + DFLUX)
ax_r = fig.add_subplot(gs[2, 2]); dark(ax_r)

r_plot = np.linspace(0, 280, 600)
for d, sx, sy, col in zip(ds, sx_D, sy_D, COLS):
    sig_D = DFLUX_SIGMA_PER_MM*d + DFLUX_SIGMA_VIRT
    # Measured profile (DFLUX convention): exp(-(r/sigma)^2)
    h_meas = np.exp(-(r_plot / sx)**2)
    h_dflx = np.exp(-(r_plot / sig_D)**2)
    ax_h.plot( r_plot, h_meas, color=col, lw=1.8,
              label=f"meas {int(d)} mm ($\\sigma_x$={sx:.0f})")
    ax_h.plot( r_plot, h_dflx, color=col, lw=1.0, ls=":")
    ax_h.plot(-r_plot, h_meas, color=col, lw=1.8)
    ax_h.plot(-r_plot, h_dflx, color=col, lw=1.0, ls=":")

    v_meas = np.exp(-(r_plot / sy)**2)
    ax_v.plot( r_plot, v_meas, color=col, lw=1.8,
              label=f"meas {int(d)} mm ($\\sigma_y$={sy:.0f})")
    ax_v.plot( r_plot, h_dflx, color=col, lw=1.0, ls=":")  # DFLUX is circular
    ax_v.plot(-r_plot, v_meas, color=col, lw=1.8)
    ax_v.plot(-r_plot, h_dflx, color=col, lw=1.0, ls=":")

    # radial: use geometric mean sigma
    sig_g = np.sqrt(sx*sy)
    r_meas = np.exp(-(r_plot/sig_g)**2)
    ax_r.plot(r_plot, r_meas, color=col, lw=1.8,
              label=f"meas {int(d)} mm ($\\sigma$$_g$={sig_g:.0f})")
    ax_r.plot(r_plot, h_dflx, color=col, lw=1.0, ls=":",
              label=f"DFLUX {int(d)} mm ($\\sigma$={sig_D:.0f})")

for ax, ttl, xl in [(ax_h, "Horizontal normalised profile  (solid: measured, dotted: DFLUX)",
                     "distance from centre (mm)"),
                    (ax_v, "Vertical normalised profile  (solid: measured, dotted: DFLUX)",
                     "distance from centre (mm)"),
                    (ax_r, "Radial profile  (solid: measured, dotted: DFLUX)",
                     "radius (mm)")]:
    ax.set_xlabel(xl, color="#aaa", fontsize=9.5)
    ax.set_ylabel("normalised intensity", color="#aaa", fontsize=9.5)
    ax.set_title(ttl, color="white", fontsize=10)
    ax.axhline(np.exp(-1), color="white", lw=0.6, ls=":", alpha=0.4)
    ax.axhline(0.5,         color="white", lw=0.6, ls=":", alpha=0.4)
    ax.legend(facecolor="#111", labelcolor="white", edgecolor="#555",
              fontsize=7.5, loc="upper right")
    ax.set_ylim(-0.02, 1.05)

ax_h.set_xlim(-260, 260)
ax_v.set_xlim(-260, 260)
ax_r.set_xlim(0,    260)

fig.suptitle(
    "Beam shape comparison  |  measured (300/400/500 mm) vs DFLUX from experimentalpulse.f\n"
    f"DFLUX: $\\sigma$(d) = {DFLUX_SIGMA_PER_MM:.4f} d + {DFLUX_SIGMA_VIRT:.2f} mm,  n=2, circular"
    f"   |   measured: $\\sigma_x$(d) = {cx[0]:+.4f} d + {cx[1]:+.2f},  "
    f"$\\sigma_y$(d) = {cy[0]:+.4f} d + {cy[1]:+.2f}  "
    f"(slope error vs DFLUX: x={slope_err_x:+.1f}%, y={slope_err_y:+.1f}%)",
    color="white", fontsize=12, y=0.985)

fig.savefig(OUTPUT_PNG, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"\nSaved figure -> {OUTPUT_PNG}")

# ---- JSON output -----------------------------------------------------------
out = {
    "convention_note": ("All sigma values in DFLUX convention exp(-(r/sigma)^2). "
                         "Measured values from beam_shape_summary_seq.json were "
                         "in standard convention exp(-0.5*(r/sigma)^2); converted "
                         "via sigma_DFLUX = sqrt(2)*sigma_std."),
    "DFLUX_reference": {
        "SIGMA_PER_MM": DFLUX_SIGMA_PER_MM,
        "SIGMA_VIRT":   DFLUX_SIGMA_VIRT,
        "SG_N":         DFLUX_SG_N,
        "circular":     True,
        "half_divergence_deg": half_div_D,
    },
    "measured": {
        "standoffs_mm": [int(d) for d in ds],
        "sigma_x_mm":   [float(v) for v in sx_D],
        "sigma_y_mm":   [float(v) for v in sy_D],
        "aspect_x_over_y": [float(v) for v in aspect],
        "sigma_x_vs_d": {"slope_mm_per_mm": float(cx[0]),
                          "intercept_mm":   float(cx[1]),
                          "half_div_deg":   float(half_div_x)},
        "sigma_y_vs_d": {"slope_mm_per_mm": float(cy[0]),
                          "intercept_mm":   float(cy[1]),
                          "half_div_deg":   float(half_div_y)},
    },
    "DFLUX_vs_measured": {
        "slope_error_x_pct":     float(slope_err_x),
        "slope_error_y_pct":     float(slope_err_y),
        "virtual_offset_diff_x_mm": float(cx[1] - DFLUX_SIGMA_VIRT),
        "virtual_offset_diff_y_mm": float(cy[1] - DFLUX_SIGMA_VIRT),
        "delta_sigma_pct_per_standoff": [
            float(((sx + sy)/2 - (DFLUX_SIGMA_PER_MM*d + DFLUX_SIGMA_VIRT))
                   / (DFLUX_SIGMA_PER_MM*d + DFLUX_SIGMA_VIRT) * 100)
            for d, sx, sy in zip(ds, sx_D, sy_D)
        ],
    },
}
with open(OUTPUT_JSON, "w") as f:
    json.dump(out, f, indent=2)
print(f"Saved JSON   -> {OUTPUT_JSON}")
