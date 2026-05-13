"""
Y-axis irradiance distribution on the specimen for each beam angle.
9.2mm lens, two symmetric beams with optimal X offset.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from beam_on_specimen import (
    make_irradiance_map, sx_free_px, sy_free_px, peak,
    SPEC_OX, SPEC_OY, SPEC_W_PX, SPEC_H_PX, SPEC_H_MM, MM_PER_PX, ANGLES
)

OUTPUT_PNG = config.BOSON_ROOT + r"\y_profile_vs_angle.png"

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor("#111111")
for ax in axes:
    ax.set_facecolor("#1a1a1a")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444444")
    ax.tick_params(colors="#aaaaaa", labelsize=9)
    ax.grid(True, alpha=0.15, color="white")

colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(ANGLES)))
y_mm = np.linspace(0, SPEC_H_MM, SPEC_H_PX)

for i, theta in enumerate(ANGLES):
    irr, off, _ = make_irradiance_map(sx_free_px, sy_free_px, peak, theta)
    spec = irr[SPEC_OY:SPEC_OY+SPEC_H_PX, SPEC_OX:SPEC_OX+SPEC_W_PX]

    mid_w    = SPEC_W_PX // 2
    v_profile = spec[:, mid_w]
    v_norm    = v_profile / v_profile.max()

    # Left panel: normalised
    axes[0].plot(y_mm, v_norm, color=colors[i], lw=2, label=f"{theta}°")

    # Right panel: absolute (arbitrary units)
    axes[1].plot(y_mm, v_profile, color=colors[i], lw=2, label=f"{theta}°")

# ±10% band on normalised panel
axes[0].axhspan(0.90, 1.10, color="lime", alpha=0.07, label="±10% band")
axes[0].axhline(0.90, color="lime", lw=0.8, ls=":")
axes[0].axhline(1.10, color="lime", lw=0.8, ls=":")
axes[0].set_ylim(0, 1.2)
axes[0].set_xlabel("Y position along specimen height (mm)", color="#aaaaaa", fontsize=10)
axes[0].set_ylabel("Normalised irradiance", color="#aaaaaa", fontsize=10)
axes[0].set_title("Y profile — normalised to peak", color="white", fontsize=11)
axes[0].legend(fontsize=9, facecolor="#222222", labelcolor="white", edgecolor="#555555")
axes[0].axvline(SPEC_H_MM/2, color="white", lw=0.6, ls="--", alpha=0.4)
axes[0].text(SPEC_H_MM/2 + 2, 0.05, "centre", color="white", fontsize=8, alpha=0.5)

# Specimen edges
for ax in axes:
    ax.axvline(0,          color="lime", lw=1, ls="--", alpha=0.5)
    ax.axvline(SPEC_H_MM,  color="lime", lw=1, ls="--", alpha=0.5)
    ax.set_xlim(0, SPEC_H_MM)

axes[1].set_xlabel("Y position along specimen height (mm)", color="#aaaaaa", fontsize=10)
axes[1].set_ylabel("Irradiance (a.u.)", color="#aaaaaa", fontsize=10)
axes[1].set_title("Y profile — absolute intensity", color="white", fontsize=11)
axes[1].legend(fontsize=9, facecolor="#222222", labelcolor="white", edgecolor="#555555")

fig.suptitle(
    "Vertical irradiance distribution on specimen  |  9.2mm lens @ 383mm  |  "
    "Two symmetric beams with optimal X offset",
    color="white", fontsize=11, y=1.02)

plt.tight_layout(pad=2.0)
fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved: {OUTPUT_PNG}")

# Print edge-to-centre ratio for each angle
print(f"\n{'Angle':>6}  {'Edge/Centre':>12}  {'Edge irr (a.u.)':>16}  {'Centre irr (a.u.)':>18}")
print("-" * 58)
for i, theta in enumerate(ANGLES):
    irr, _, _ = make_irradiance_map(sx_free_px, sy_free_px, peak, theta)
    spec = irr[SPEC_OY:SPEC_OY+SPEC_H_PX, SPEC_OX:SPEC_OX+SPEC_W_PX]
    mid_w = SPEC_W_PX // 2
    v = spec[:, mid_w]
    centre = v[len(v)//2]
    edge   = (v[0] + v[-1]) / 2
    print(f"  {theta:3.0f}deg   {edge/centre*100:10.1f}%   {edge:16.1f}   {centre:18.1f}")
