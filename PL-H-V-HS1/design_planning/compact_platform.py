"""
For the best configuration (side lamps + horizontal filament),
sweep lamp offset L (distance from camera axis) and angle theta.

For each (L, theta): beam centre on specimen = L - d*tan(theta)
CoV is computed for that exact beam placement — offset is NOT free.

d = 383mm (fixed by 9.2mm lens)
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

OUTPUT_PNG = config.BOSON_ROOT + r"\compact_platform.png"

sx_mm, sy_mm, peak = load_beam()
# Best config: side lamps, horizontal filament -> sx_free=sy_mm (narrow), sy_free=sx_mm (wide)
sx_free = sy_mm   # ~59mm  (horizontal, in X)
sy_free = sx_mm   # ~108mm (vertical, in Y)

D_STANDOFF = 383.0   # mm, fixed by 9.2mm lens

ANGLES = np.arange(10, 75, 1.0)
L_VALUES = np.arange(50, 501, 5.0)   # lamp lateral distance from camera axis

print(f"Beam (horizontal filament): sigma_x={sx_free:.1f}mm  sigma_y={sy_free:.1f}mm")
print(f"Standoff d = {D_STANDOFF:.0f}mm  (9.2mm lens)\n")

# ── helpers ──────────────────────────────────────────────────────────────────

def build_irr(sx_surf_mm, sy_surf_mm, surf_peak, offset_mm):
    """Two side beams at +-offset_mm from specimen centre."""
    sx_px  = sx_surf_mm / MM_PER_PX
    sy_px  = sy_surf_mm / MM_PER_PX
    off_px = offset_mm  / MM_PER_PX
    irr    = np.zeros((SENSOR_H, SENSOR_W), dtype=np.float32)
    yy, xx = np.mgrid[0:SENSOR_H, 0:SENSOR_W]
    cx_s   = SPEC_OX + SPEC_W_PX / 2
    cy_s   = SPEC_OY + SPEC_H_PX / 2
    for sign in [+1, -1]:
        bx = cx_s + sign * off_px
        irr += surf_peak * np.exp(
            -0.5 * (((xx - bx) / sx_px)**2 + ((yy - cy_s) / sy_px)**2)
        ).astype(np.float32)
    return irr


def uniformity_2d(irr):
    spec = irr[SPEC_OY:SPEC_OY+SPEC_H_PX, SPEC_OX:SPEC_OX+SPEC_W_PX]
    m    = spec.mean()
    if m == 0: return 100.0, 100.0
    return spec.std()/m*100, (spec.max()-spec.min())/m*100


# ── compute CoV grid ──────────────────────────────────────────────────────────
COV_GRID   = np.full((len(L_VALUES), len(ANGLES)), np.nan)
OFFSET_GRID = np.full_like(COV_GRID, np.nan)

for li, L in enumerate(L_VALUES):
    for ti, theta in enumerate(ANGLES):
        tr      = np.radians(theta)
        offset  = L - D_STANDOFF * np.tan(tr)   # signed: +ve = beam right of centre
        # We need the magnitude; if negative, beam has crossed centre (still symmetric)
        offset_abs = abs(offset)
        OFFSET_GRID[li, ti] = offset_abs

        sx_surf = sx_free / np.cos(tr)
        sy_surf = sy_free   # unchanged for horizontal tilt
        sp      = peak * np.cos(tr)

        irr = build_irr(sx_surf, sy_surf, sp, offset_abs)
        cov, _ = uniformity_2d(irr)
        COV_GRID[li, ti] = cov

# ── find best (L, theta) for each L ──────────────────────────────────────────
best_per_L = []
for li, L in enumerate(L_VALUES):
    idx = np.nanargmin(COV_GRID[li, :])
    best_per_L.append((L, ANGLES[idx], COV_GRID[li, idx], OFFSET_GRID[li, idx]))

# overall best
best_per_L.sort(key=lambda x: x[2])
print(f"{'L (mm)':>8}  {'Best angle':>10}  {'Min CoV':>8}  {'Offset on specimen':>20}")
print("-" * 55)
for L, ang, cov, off in best_per_L[:20]:
    marker = " <-- most compact" if L == best_per_L[0][0] else ""
    print(f"{L:>8.0f}mm  {ang:>10.0f}deg  {cov:>7.1f}%  {off:>18.1f}mm{marker}")

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor("#111111")
gs  = plt.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                   left=0.07, right=0.97, top=0.91, bottom=0.07)

# ── Panel A: CoV heatmap (L vs theta) ────────────────────────────────────────
ax_heat = fig.add_subplot(gs[:, 0:2])
ax_heat.set_facecolor("#0a0a0a")

im = ax_heat.contourf(ANGLES, L_VALUES, COV_GRID,
                      levels=np.arange(0, 51, 2),
                      cmap="RdYlGn_r", extend="max")
cb = fig.colorbar(im, ax=ax_heat, pad=0.02)
cb.set_label("2D CoV on specimen (%)", color="white", fontsize=10)
cb.ax.yaxis.set_tick_params(color="white"); plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

# contour lines at key thresholds
cs = ax_heat.contour(ANGLES, L_VALUES, COV_GRID,
                     levels=[5, 10, 15, 20], colors=["white","white","white","white"],
                     linewidths=[2, 1.5, 1, 0.8], linestyles=["--",":",":","--"])
ax_heat.clabel(cs, fmt="%d%%", colors="white", fontsize=9)

# best-angle line per L
best_angles = [b[1] for b in sorted(best_per_L, key=lambda x: x[0])]
ax_heat.plot(best_angles, L_VALUES, color="cyan", lw=2, ls="--", label="Optimal angle for each L")

# mark compact zone
ax_heat.axhline(200, color="lime", lw=1.5, ls=":", alpha=0.7)
ax_heat.text(11, 205, "L = 200mm  (compact)", color="lime", fontsize=9)

ax_heat.set_xlabel("Beam angle θ (deg)", color="#aaaaaa", fontsize=11)
ax_heat.set_ylabel("L — lamp distance from camera axis (mm)", color="#aaaaaa", fontsize=11)
ax_heat.set_title(
    f"CoV vs angle & lamp position  |  d={D_STANDOFF:.0f}mm standoff  |  "
    f"Side lamps + horizontal filament\n"
    f"Beam: σ_x={sx_free:.0f}mm  σ_y={sy_free:.0f}mm  |  Specimen {SPEC_W_MM:.0f}×{SPEC_H_MM:.0f}mm",
    color="white", fontsize=11)
ax_heat.tick_params(colors="#aaaaaa", labelsize=9)
ax_heat.legend(fontsize=9, facecolor="#222222", labelcolor="white", edgecolor="#555555")

# ── Panel B: Min CoV vs L (how compact can you go?) ──────────────────────────
ax_cov = fig.add_subplot(gs[0, 2])
ax_cov.set_facecolor("#1a1a1a")
for sp in ax_cov.spines.values(): sp.set_edgecolor("#444444")
ax_cov.tick_params(colors="#aaaaaa", labelsize=9)
ax_cov.grid(True, alpha=0.15, color="white")

Ls_sorted = [b[0] for b in sorted(best_per_L, key=lambda x: x[0])]
covs_min  = [b[2] for b in sorted(best_per_L, key=lambda x: x[0])]
ax_cov.plot(Ls_sorted, covs_min, color="#ffd93d", lw=2.5)
ax_cov.axhline(10, color="lime", lw=1, ls=":", alpha=0.7)
ax_cov.axhline(5,  color="lime", lw=1, ls="--", alpha=0.7)
ax_cov.text(510, 10.5, "10%", color="lime", fontsize=8)
ax_cov.text(510, 5.5,  " 5%", color="lime", fontsize=8)
ax_cov.axvline(200, color="lime", lw=1, ls=":", alpha=0.5)
ax_cov.set_xlabel("L — lamp distance from camera axis (mm)", color="#aaaaaa", fontsize=9)
ax_cov.set_ylabel("Best achievable CoV (%)", color="#aaaaaa", fontsize=9)
ax_cov.set_title("Most compact L for target CoV", color="white", fontsize=10)
ax_cov.set_xlim(50, 500)

# annotate minimum L for <10% and <15%
for target, c in [(10, "#ffd93d"), (15, "#ff6b6b")]:
    hits = [L for L, cov in zip(Ls_sorted, covs_min) if cov <= target]
    if hits:
        Lmin = min(hits)
        ax_cov.axvline(Lmin, color=c, lw=1.5, ls="--", alpha=0.8)
        ax_cov.text(Lmin+5, target+1, f"L≥{Lmin:.0f}mm\nfor <{target}%",
                    color=c, fontsize=8)

# ── Panel C: best angle vs L ──────────────────────────────────────────────────
ax_ang = fig.add_subplot(gs[1, 2])
ax_ang.set_facecolor("#1a1a1a")
for sp in ax_ang.spines.values(): sp.set_edgecolor("#444444")
ax_ang.tick_params(colors="#aaaaaa", labelsize=9)
ax_ang.grid(True, alpha=0.15, color="white")

best_angs_sorted = [b[1] for b in sorted(best_per_L, key=lambda x: x[0])]
best_offs_sorted = [b[3] for b in sorted(best_per_L, key=lambda x: x[0])]
ax_ang.plot(Ls_sorted, best_angs_sorted, color="#6bcb77", lw=2.5, label="Optimal angle")
ax2 = ax_ang.twinx()
ax2.plot(Ls_sorted, best_offs_sorted, color="#4d96ff", lw=2, ls="--", label="Beam offset on specimen")
ax2.set_ylabel("Beam offset on specimen (mm)", color="#4d96ff", fontsize=8)
ax2.tick_params(colors="#4d96ff", labelsize=8)
ax_ang.axvline(200, color="lime", lw=1, ls=":", alpha=0.5)
ax_ang.set_xlabel("L — lamp distance from camera axis (mm)", color="#aaaaaa", fontsize=9)
ax_ang.set_ylabel("Optimal angle θ (deg)", color="#6bcb77", fontsize=9)
ax_ang.set_title("Optimal angle & beam offset vs L", color="white", fontsize=10)
lines1, labels1 = ax_ang.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax_ang.legend(lines1+lines2, labels1+labels2, fontsize=8,
              facecolor="#222222", labelcolor="white", edgecolor="#555555")

fig.suptitle("Compact platform design  —  Side lamps + horizontal filament  |  9.2mm lens @ 383mm",
             color="white", fontsize=12)
fig.savefig(OUTPUT_PNG, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"\nSaved: {OUTPUT_PNG}")
