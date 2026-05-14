"""
Shows how the two-beam irradiance pattern looks on the specimen
as seen by the Boson 640 with a 9.2mm lens, for beam angles 30-60 deg.

Each panel is the actual camera pixel map (640x512) with the specimen
highlighted and the simulated heat flux from two symmetric beams.
"""

import os
import config
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter

OUTPUT_PNG   = config.BOSON_ROOT + r"\beam_on_specimen_9p2mm.png"
BEST_SESSION = config.BOSON_ROOT + r"\700 mm\summary.json"

# Camera: 9.2mm lens
FOCAL_MM   = 9.2
SENSOR_W   = 640
SENSOR_H   = 512
PIXEL_PITCH = 0.012          # mm
SENSOR_W_MM = SENSOR_W * PIXEL_PITCH   # 7.68mm
SENSOR_H_MM = SENSOR_H * PIXEL_PITCH   # 6.144mm

STANDOFF   = max(320 * FOCAL_MM / SENSOR_W_MM,
                 175 * FOCAL_MM / SENSOR_H_MM)   # 383mm
MM_PER_PX  = STANDOFF * PIXEL_PITCH / FOCAL_MM   # ~0.5mm/px

SPEC_W_MM  = 320.0
SPEC_H_MM  = 175.0
SPEC_W_PX  = int(SPEC_W_MM / MM_PER_PX)   # 640px
SPEC_H_PX  = int(SPEC_H_MM / MM_PER_PX)   # 350px

# Specimen top-left corner in sensor (centred)
SPEC_OX    = (SENSOR_W - SPEC_W_PX) // 2   # 0px (fills width)
SPEC_OY    = (SENSOR_H - SPEC_H_PX) // 2   # 81px

ANGLES     = [30, 40, 45, 50, 60]


def load_beam():
    scale_700 = 2 * 700 * np.tan(np.radians(
        2 * np.degrees(np.arctan(SENSOR_W_MM / (2 * 18.0))) / 2)) / SENSOR_W
    with open(BEST_SESSION) as f:
        b = json.load(f)["beam"]
    return b["sigma_x"] * scale_700, b["sigma_y"] * scale_700, b["amplitude"]


def make_irradiance_map(sx_free, sy_free, peak, theta_deg):
    """
    Returns a (SENSOR_H, SENSOR_W) irradiance array in pixel coordinates.
    Two beams at +-theta, optimal offset, centred on specimen.
    """
    theta_r  = np.radians(theta_deg)
    sx_s     = sx_free  / np.cos(theta_r)   # px equivalent
    sy_s     = sy_free                        # px equivalent
    surf_peak = peak * np.cos(theta_r)

    # Convert sigma from mm to pixels
    sx_s_px = sx_s / MM_PER_PX
    sy_s_px = sy_s / MM_PER_PX

    # Optimal offset: minimise CoV across specimen width
    best_cov, best_off = 1e9, 0.0
    x_spec = np.arange(SPEC_W_PX)
    cx     = SPEC_W_PX / 2
    for off in np.linspace(0, sx_s_px * 2, 150):
        ix = (np.exp(-0.5*((x_spec - cx - off)/sx_s_px)**2) +
              np.exp(-0.5*((x_spec - cx + off)/sx_s_px)**2))
        c = ix.std() / ix.mean() * 100
        if c < best_cov:
            best_cov, best_off = c, off

    # Build full sensor map
    irr = np.zeros((SENSOR_H, SENSOR_W), dtype=np.float32)
    yy, xx = np.mgrid[0:SENSOR_H, 0:SENSOR_W]
    cx_sensor = SPEC_OX + SPEC_W_PX / 2
    cy_sensor = SPEC_OY + SPEC_H_PX / 2

    for sign in [+1, -1]:
        bx = cx_sensor + sign * best_off
        irr += surf_peak * np.exp(
            -0.5 * (((xx - bx) / sx_s_px)**2 +
                    ((yy - cy_sensor) / sy_s_px)**2)).astype(np.float32)

    return irr, best_off * MM_PER_PX, best_cov


def uniformity_on_specimen(irr):
    spec = irr[SPEC_OY:SPEC_OY+SPEC_H_PX, SPEC_OX:SPEC_OX+SPEC_W_PX]
    m = spec.mean()
    cov = spec.std() / m * 100 if m > 0 else 0
    p2v = (spec.max() - spec.min()) / m * 100 if m > 0 else 0
    return cov, p2v


# =========================================================================
# FIGURE: top row = irradiance maps, bottom row = cross-sections
# =========================================================================
sx_free, sy_free, peak = load_beam()
sx_free_px = sx_free / MM_PER_PX
sy_free_px = sy_free / MM_PER_PX

n   = len(ANGLES)
fig = plt.figure(figsize=(5*n, 12))
fig.patch.set_facecolor("#111111")
fig.suptitle(
    f"Two-beam irradiance on specimen  |  9.2mm lens @ {STANDOFF:.0f}mm standoff  |  "
    f"{MM_PER_PX:.2f} mm/px  |  Beam angles 30-60°",
    fontsize=12, color="white", y=1.01)

gs = plt.GridSpec(3, n, figure=fig, hspace=0.5, wspace=0.15,
                  left=0.05, right=0.98, top=0.97, bottom=0.04)

CMAP = "inferno"
maps, offsets, covs = [], [], []

# Pre-compute all maps for shared colour scale
for theta in ANGLES:
    irr, off, cov_x = make_irradiance_map(sx_free_px, sy_free_px, peak, theta)
    maps.append(irr); offsets.append(off); covs.append(cov_x)

vmax = max(m.max() for m in maps)
angle_colors = plt.cm.plasma(np.linspace(0.15, 0.85, n))

for i, (theta, irr, off, cov_x) in enumerate(zip(ANGLES, maps, offsets, covs)):
    cov_2d, p2v_2d = uniformity_on_specimen(irr)
    spec_patch = irr[SPEC_OY:SPEC_OY+SPEC_H_PX, SPEC_OX:SPEC_OX+SPEC_W_PX]

    # ----- Row 0: full sensor pixel map -------------------------------------
    ax_map = fig.add_subplot(gs[0, i])
    ax_map.set_facecolor("#0a0a0a")
    ax_map.imshow(irr, cmap=CMAP, origin="upper",
                  vmin=0, vmax=vmax, aspect="equal",
                  extent=[0, SENSOR_W, SENSOR_H, 0])

    # specimen border
    ax_map.add_patch(mpatches.Rectangle(
        (SPEC_OX, SPEC_OY), SPEC_W_PX, SPEC_H_PX,
        ec="lime", fc="none", lw=2, ls="--", zorder=5))

    # beam centre markers
    cx_s = SPEC_OX + SPEC_W_PX/2
    cy_s = SPEC_OY + SPEC_H_PX/2
    off_px = off / MM_PER_PX
    for sign in [+1, -1]:
        ax_map.plot(cx_s + sign*off_px, cy_s, "+",
                    color="white", ms=10, mew=1.5, zorder=6)

    ax_map.set_xlim(0, SENSOR_W); ax_map.set_ylim(SENSOR_H, 0)
    ax_map.set_xticks([0, 320, 640]); ax_map.set_yticks([0, 256, 512])
    ax_map.tick_params(colors="#666666", labelsize=7)
    for sp in ax_map.spines.values():
        sp.set_edgecolor(angle_colors[i]); sp.set_linewidth(2)
    ax_map.set_title(
        f"{theta}°  |  offset={off:.0f}mm\nCoV={cov_2d:.0f}%  P2V={p2v_2d:.0f}%",
        color=angle_colors[i], fontsize=9, fontweight="bold")
    if i == 0:
        ax_map.set_ylabel("px", color="#aaaaaa", fontsize=8)

    # ----- Row 1: specimen close-up -----------------------------------------
    ax_spec = fig.add_subplot(gs[1, i])
    ax_spec.set_facecolor("#0a0a0a")
    ax_spec.imshow(spec_patch, cmap=CMAP, origin="upper",
                   vmin=0, vmax=vmax, aspect="auto",
                   extent=[0, SPEC_W_MM, SPEC_H_MM, 0])
    # uniform reference lines
    mean_val = spec_patch.mean()
    ax_spec.set_xlim(0, SPEC_W_MM); ax_spec.set_ylim(SPEC_H_MM, 0)
    ax_spec.tick_params(colors="#666666", labelsize=7)
    for sp in ax_spec.spines.values():
        sp.set_edgecolor("lime"); sp.set_linewidth(1.5)
    ax_spec.set_title("Specimen (mm)", color="lime", fontsize=8)
    if i == 0:
        ax_spec.set_ylabel("mm", color="#aaaaaa", fontsize=8)
    ax_spec.set_xlabel("mm", color="#aaaaaa", fontsize=8)

    # ----- Row 2: cross-sections --------------------------------------------
    ax_xc = fig.add_subplot(gs[2, i])
    ax_xc.set_facecolor("#1a1a1a")
    for sp in ax_xc.spines.values():
        sp.set_edgecolor("#444444")
    ax_xc.tick_params(colors="#aaaaaa", labelsize=7)

    x_mm  = np.linspace(0, SPEC_W_MM, SPEC_W_PX)
    y_mm  = np.linspace(0, SPEC_H_MM, SPEC_H_PX)
    mid_h = SPEC_H_PX // 2
    mid_w = SPEC_W_PX // 2

    h_profile = spec_patch[mid_h, :]
    v_profile = spec_patch[:, mid_w]

    ax_xc.plot(x_mm, h_profile / vmax, color=angle_colors[i], lw=2,
               label="Horizontal")
    ax_xc.plot(v_profile / vmax, y_mm / SPEC_H_MM,  # inset y-profile
               color="white", lw=1.5, ls="--", alpha=0.7, label="Vertical (right axis)")

    # +-10% band
    ax_xc.axhline(mean_val/vmax * 1.10, color="lime", lw=0.6, ls=":", alpha=0.5)
    ax_xc.axhline(mean_val/vmax * 0.90, color="lime", lw=0.6, ls=":", alpha=0.5)
    ax_xc.axhspan(mean_val/vmax * 0.90, mean_val/vmax * 1.10,
                  color="lime", alpha=0.05)

    ax_xc.set_xlim(0, SPEC_W_MM)
    ax_xc.set_ylim(0, 1.05)
    ax_xc.set_xlabel("X along specimen (mm)", color="#aaaaaa", fontsize=8)
    if i == 0:
        ax_xc.set_ylabel("Norm. irradiance", color="#aaaaaa", fontsize=8)
    ax_xc.set_title(f"H cross-section\n(shaded = ±10% band)",
                    color="white", fontsize=8)
    ax_xc.grid(True, alpha=0.1, color="white")
    if i == 0:
        ax_xc.legend(fontsize=7, facecolor="#222222",
                     labelcolor="white", edgecolor="#555555")

# Shared colourbar
cax = fig.add_axes([0.92, 0.38, 0.008, 0.25])
cax.set_facecolor("#111111")
sm  = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, vmax))
sm.set_array([])
cb  = fig.colorbar(sm, cax=cax)
cb.set_label("Irradiance (a.u.)", color="white", fontsize=8)
cb.ax.yaxis.set_tick_params(color="white", labelsize=7)
plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved: {OUTPUT_PNG}")

print(f"\nSetup: 9.2mm lens @ {STANDOFF:.0f}mm  |  {MM_PER_PX:.3f} mm/px")
print(f"Specimen on sensor: {SPEC_W_PX}x{SPEC_H_PX} px  (fills sensor width)\n")
print(f"{'Angle':>6}  {'Offset':>10}  {'CoV':>8}  {'P2V':>8}")
print("-" * 40)
for theta, irr, off in zip(ANGLES, maps, offsets):
    cov, p2v = uniformity_on_specimen(irr)
    print(f"  {theta:3.0f}°   {off:8.1f}mm   {cov:7.1f}%  {p2v:7.1f}%")
