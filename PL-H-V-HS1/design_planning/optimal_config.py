"""
For the two lamp pairings (side = left/right, topbot = top/bottom),
sweep angle 20-70 deg, find optimal beam offset at each angle,
and report the best CoV/P2V on the 320x175mm specimen.

Filament orientation is also optimised per configuration:
  - Side lamps : horizontal filament (wider in Y) is better
  - Top/bottom : vertical filament  (wider in X) is better
Both filament options are shown so the user can compare.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from beam_on_specimen import load_beam, MM_PER_PX, SPEC_W_PX, SPEC_H_PX, SPEC_W_MM, SPEC_H_MM
from beam_on_specimen import SENSOR_W, SENSOR_H, SPEC_OX, SPEC_OY

OUTPUT_PNG = config.BOSON_ROOT + r"\optimal_config.png"

sx_mm, sy_mm, peak = load_beam()   # sx > sy  (vertical filament, wider in X)
print(f"Beam free-space: sigma_x={sx_mm:.1f}mm  sigma_y={sy_mm:.1f}mm")
print(f"  FWHM_x={sx_mm*2.355:.1f}mm  FWHM_y={sy_mm*2.355:.1f}mm")
print(f"  Specimen: {SPEC_W_MM:.0f} x {SPEC_H_MM:.0f} mm\n")

ANGLES = np.arange(20, 71, 2, dtype=float)

# ─── helpers ──────────────────────────────────────────────────────────────────

def optimal_offset(profile_len_px, sigma_px):
    coords  = np.arange(profile_len_px)
    cx      = profile_len_px / 2
    best_c, best_off = 1e9, 0.0
    for off in np.linspace(0, sigma_px * 2.5, 300):
        ix = (np.exp(-0.5 * ((coords - cx - off) / sigma_px)**2) +
              np.exp(-0.5 * ((coords - cx + off) / sigma_px)**2))
        c = ix.std() / ix.mean() * 100
        if c < best_c:
            best_c, best_off = c, off
    return best_off, best_c


def build_irr_map(sx_surf_mm, sy_surf_mm, surf_peak, offset_mm, axis):
    """axis: 'x' -> side lamps (offset in X),  'y' -> top/bot (offset in Y)"""
    sx_px  = sx_surf_mm / MM_PER_PX
    sy_px  = sy_surf_mm / MM_PER_PX
    off_px = offset_mm  / MM_PER_PX
    irr    = np.zeros((SENSOR_H, SENSOR_W), dtype=np.float32)
    yy, xx = np.mgrid[0:SENSOR_H, 0:SENSOR_W]
    cx_s   = SPEC_OX + SPEC_W_PX / 2
    cy_s   = SPEC_OY + SPEC_H_PX / 2
    for sign in [+1, -1]:
        bx = cx_s + sign * off_px if axis == "x" else cx_s
        by = cy_s + sign * off_px if axis == "y" else cy_s
        irr += surf_peak * np.exp(
            -0.5 * (((xx - bx) / sx_px)**2 + ((yy - by) / sy_px)**2)
        ).astype(np.float32)
    return irr


def uniformity_2d(irr):
    spec = irr[SPEC_OY:SPEC_OY+SPEC_H_PX, SPEC_OX:SPEC_OX+SPEC_W_PX]
    m = spec.mean()
    return spec.std()/m*100, (spec.max()-spec.min())/m*100


# ─── sweep ────────────────────────────────────────────────────────────────────
# Four curves:
#   A = side  + horizontal filament  (sy_free = sx_mm, sx_free = sy_mm)
#   B = side  + vertical filament    (sx_free = sx_mm, sy_free = sy_mm)  [current]
#   C = topbot + vertical filament   (sx_free = sx_mm, sy_free = sy_mm)
#   D = topbot + horizontal filament (sx_free = sy_mm, sy_free = sx_mm)

curves = {
    "Side — horizontal filament":  ("x", sy_mm, sx_mm),  # (axis, sx_free, sy_free)
    "Side — vertical filament\n(current)": ("x", sx_mm, sy_mm),
    "Top/Bot — vertical filament": ("y", sx_mm, sy_mm),
    "Top/Bot — horizontal filament": ("y", sy_mm, sx_mm),
}
colors = ["#ffd93d", "#ff6b6b", "#6bcb77", "#4d96ff"]

results = {}   # label -> (angles, covs, p2vs, offsets_mm)

for label, (axis, sx_f, sy_f) in curves.items():
    covs, p2vs, offs = [], [], []
    for theta in ANGLES:
        tr = np.radians(theta)
        sx_s = sx_f / np.cos(tr) if axis == "x" else sx_f
        sy_s = sy_f / np.cos(tr) if axis == "y" else sy_f
        sp   = peak * np.cos(tr)
        # optimise offset along the separation axis
        ref_sigma_px = (sx_s if axis == "x" else sy_s) / MM_PER_PX
        ref_len_px   = SPEC_W_PX if axis == "x" else SPEC_H_PX
        off_px, _    = optimal_offset(ref_len_px, ref_sigma_px)
        off_mm       = off_px * MM_PER_PX
        irr          = build_irr_map(sx_s, sy_s, sp, off_mm, axis)
        cov, p2v     = uniformity_2d(irr)
        covs.append(cov); p2vs.append(p2v); offs.append(off_mm)
    results[label] = (ANGLES, np.array(covs), np.array(p2vs), np.array(offs))

# ─── find overall best ────────────────────────────────────────────────────────
best_cov, best_label, best_angle = 1e9, "", 0
for label, (angs, covs, _, offs) in results.items():
    idx = np.argmin(covs)
    if covs[idx] < best_cov:
        best_cov   = covs[idx]
        best_label = label
        best_angle = angs[idx]
        best_off   = offs[idx]

print(f"Best configuration : {best_label.strip()}")
print(f"Best angle         : {best_angle:.0f} deg")
print(f"Best offset        : {best_off:.1f} mm")
print(f"Best CoV           : {best_cov:.1f} %\n")

# ─── figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor("#111111")
gs  = plt.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.3,
                   left=0.07, right=0.97, top=0.91, bottom=0.07)

# Row 0: CoV vs angle, P2V vs angle
ax_cov = fig.add_subplot(gs[0, :2])
ax_p2v = fig.add_subplot(gs[0, 2])

for ax in [ax_cov, ax_p2v]:
    ax.set_facecolor("#1a1a1a")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444444")
    ax.tick_params(colors="#aaaaaa", labelsize=9)
    ax.grid(True, alpha=0.15, color="white")

for (label, (angs, covs, p2vs, offs)), col in zip(results.items(), colors):
    lbl = label.replace("\n", " ")
    ax_cov.plot(angs, covs, color=col, lw=2.5, label=lbl)
    ax_p2v.plot(angs, p2vs, color=col, lw=2.5)

ax_cov.axhline(5,  color="lime", lw=0.8, ls=":", alpha=0.6)
ax_cov.axhline(10, color="lime", lw=0.8, ls=":", alpha=0.4)
ax_cov.text(ANGLES[-1]+0.5, 5,  "5% target", color="lime", fontsize=8, va="center")
ax_cov.text(ANGLES[-1]+0.5, 10, "10%",        color="lime", fontsize=8, va="center", alpha=0.7)
ax_cov.set_xlabel("Beam angle (deg)", color="#aaaaaa", fontsize=10)
ax_cov.set_ylabel("2D CoV on specimen (%)", color="#aaaaaa", fontsize=10)
ax_cov.set_title("Uniformity vs angle  —  all configurations", color="white", fontsize=11)
ax_cov.legend(fontsize=8.5, facecolor="#222222", labelcolor="white",
              edgecolor="#555555", loc="upper right")

ax_p2v.set_xlabel("Beam angle (deg)", color="#aaaaaa", fontsize=10)
ax_p2v.set_ylabel("P2V (%)", color="#aaaaaa", fontsize=10)
ax_p2v.set_title("Peak-to-valley vs angle", color="white", fontsize=11)

# Row 1: irradiance maps for the best angle of each config
CMAP = "inferno"
for ci, ((label, (angs, covs, p2vs, offs)), col) in enumerate(zip(results.items(), colors)):
    if ci >= 3:
        break
    idx    = np.argmin(covs)
    theta  = angs[idx]
    tr     = np.radians(theta)
    axis, sx_f, sy_f = list(curves.values())[ci]
    sx_s   = sx_f / np.cos(tr) if axis == "x" else sx_f
    sy_s   = sy_f / np.cos(tr) if axis == "y" else sy_f
    sp     = peak * np.cos(tr)
    off_mm = offs[idx]
    irr    = build_irr_map(sx_s, sy_s, sp, off_mm, axis)
    cov, p2v = uniformity_2d(irr)

    ax = fig.add_subplot(gs[1, ci])
    ax.set_facecolor("#0a0a0a")
    ax.imshow(irr, cmap=CMAP, origin="upper", vmin=0,
              vmax=irr.max()*1.02, aspect="equal",
              extent=[0, SENSOR_W, SENSOR_H, 0])
    ax.add_patch(mpatches.Rectangle(
        (SPEC_OX, SPEC_OY), SPEC_W_PX, SPEC_H_PX,
        ec="lime", fc="none", lw=2, ls="--", zorder=5))
    ax.set_xlim(0, SENSOR_W); ax.set_ylim(SENSOR_H, 0)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(col); sp.set_linewidth(2)
    short = label.split("\n")[0]
    ax.set_title(
        f"{short}\nBest: {theta:.0f}deg  off={off_mm:.0f}mm\nCoV={cov:.1f}%  P2V={p2v:.1f}%",
        color=col, fontsize=9, fontweight="bold")

# 4th map: best overall config at its optimal angle
ax4 = fig.add_subplot(gs[1, 2]) if len(results) < 4 else None
# (already 3 maps for 3 configs in row 1, 4th curve shown in top plots only)

fig.suptitle(
    f"Optimal angle & offset for each lamp configuration  |  9.2mm lens  |  "
    f"Specimen {SPEC_W_MM:.0f}x{SPEC_H_MM:.0f}mm",
    color="white", fontsize=12)

fig.savefig(OUTPUT_PNG, dpi=140, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved: {OUTPUT_PNG}")

# ─── console table ────────────────────────────────────────────────────────────
print(f"\n{'Config':<40}  {'Best angle':>10}  {'Min CoV':>8}  {'P2V@best':>9}  {'Offset':>9}")
print("-" * 82)
for (label, (angs, covs, p2vs, offs)), col in zip(results.items(), colors):
    idx = np.argmin(covs)
    print(f"{label.replace(chr(10),' '):<40}  {angs[idx]:>10.0f}deg  "
          f"{covs[idx]:>7.1f}%  {p2vs[idx]:>8.1f}%  {offs[idx]:>7.1f}mm")
