"""
Compares four excitation source configurations for a 320x175mm specimen.

Side lamps   = two lamps left/right, angle in horizontal (X) plane
Top/bot lamps = two lamps above/below, angle in vertical (Y) plane

Filament vertical   -> beam wider in X (sigma_x > sigma_y)  [measured]
Filament horizontal -> beam wider in Y (sigma_y > sigma_x)  [rotated 90]

For each config the offset between the two beams is optimised for the
axis along which they are separated.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from beam_on_specimen import (
    load_beam, MM_PER_PX,
    SENSOR_W, SENSOR_H,
    SPEC_OX, SPEC_OY, SPEC_W_PX, SPEC_H_PX,
    SPEC_W_MM, SPEC_H_MM,
)

OUTPUT_PNG = config.BOSON_ROOT + r"\config_comparison.png"

sx_mm, sy_mm, peak = load_beam()   # sigma in mm, vertical filament orientation

ANGLES = [30, 40, 45, 50, 60]
CMAP   = "inferno"

# ── helpers ──────────────────────────────────────────────────────────────────

def optimal_offset_1d(profile_len, sigma_px):
    """Sweep offset along one axis, return (best_offset_px, best_cov)."""
    coords = np.arange(profile_len)
    cx = profile_len / 2
    best_cov, best_off = 1e9, 0.0
    for off in np.linspace(0, sigma_px * 2.5, 200):
        ix = (np.exp(-0.5 * ((coords - cx - off) / sigma_px) ** 2) +
              np.exp(-0.5 * ((coords - cx + off) / sigma_px) ** 2))
        c = ix.std() / ix.mean() * 100
        if c < best_cov:
            best_cov, best_off = c, off
    return best_off, best_cov


def make_map(sx_surf_mm, sy_surf_mm, surf_peak, config):
    """
    Build (SENSOR_H, SENSOR_W) irradiance map.
    config: 'side'   -> beams offset in X
            'topbot' -> beams offset in Y
    """
    sx_px = sx_surf_mm / MM_PER_PX
    sy_px = sy_surf_mm / MM_PER_PX

    if config == "side":
        off_px, _ = optimal_offset_1d(SPEC_W_PX, sx_px)
        off_mm    = off_px * MM_PER_PX
    else:
        off_px, _ = optimal_offset_1d(SPEC_H_PX, sy_px)
        off_mm    = off_px * MM_PER_PX

    irr = np.zeros((SENSOR_H, SENSOR_W), dtype=np.float32)
    yy, xx = np.mgrid[0:SENSOR_H, 0:SENSOR_W]
    cx_s = SPEC_OX + SPEC_W_PX / 2
    cy_s = SPEC_OY + SPEC_H_PX / 2

    for sign in [+1, -1]:
        bx = cx_s + sign * off_px if config == "side"   else cx_s
        by = cy_s + sign * off_px if config == "topbot" else cy_s
        irr += surf_peak * np.exp(
            -0.5 * (((xx - bx) / sx_px) ** 2 +
                    ((yy - by) / sy_px) ** 2)).astype(np.float32)

    return irr, off_mm


def uniformity(irr):
    spec = irr[SPEC_OY:SPEC_OY + SPEC_H_PX, SPEC_OX:SPEC_OX + SPEC_W_PX]
    m = spec.mean()
    return spec.std() / m * 100, (spec.max() - spec.min()) / m * 100


# ── four configurations ───────────────────────────────────────────────────────
# Each entry: (label, tilt_axis, filament)
# tilt_axis: 'x' = lamps left/right (side), 'y' = lamps top/bot
# filament:  'v' = vertical (sx>sy), 'h' = horizontal (sy>sx)

CONFIGS = [
    ("Side\nVertical filament\n(current)",   "x", "v"),
    ("Side\nHorizontal filament",             "x", "h"),
    ("Top/Bottom\nVertical filament",         "y", "v"),
    ("Top/Bottom\nHorizontal filament",       "y", "h"),
]

# ── figure: one column per config, rows = angles ─────────────────────────────
n_cfg = len(CONFIGS)
n_ang = len(ANGLES)

fig_w = 4 * n_cfg
fig_h = 3.5 * n_ang + 2.5
fig = plt.figure(figsize=(fig_w, fig_h))
fig.patch.set_facecolor("#111111")

gs = plt.GridSpec(n_ang + 1, n_cfg, figure=fig,
                  hspace=0.55, wspace=0.25,
                  left=0.05, right=0.97, top=0.94, bottom=0.04)

cfg_colors = ["#ff6b6b", "#ffd93d", "#6bcb77", "#4d96ff"]

# summary table rows
summary = []   # (config_label, angle, cov, p2v, off)

for ci, (cfg_label, tilt_axis, filament) in enumerate(CONFIGS):

    # Column header (row n_ang = bottom summary)
    ax_hdr = fig.add_subplot(gs[n_ang, ci])
    ax_hdr.set_facecolor("#0d0d0d")
    ax_hdr.axis("off")

    col_rows = []   # collect per-angle results for this config

    for ri, theta in enumerate(ANGLES):
        theta_r = np.radians(theta)

        # beam sigmas on specimen surface
        if filament == "v":
            sx_free, sy_free = sx_mm, sy_mm          # wider in X
        else:
            sx_free, sy_free = sy_mm, sx_mm          # wider in Y

        if tilt_axis == "x":   # side lamps — X stretched
            sx_surf = sx_free / np.cos(theta_r)
            sy_surf = sy_free
            surf_pk = peak * np.cos(theta_r)
            cfg_key = "side"
        else:                  # top/bot lamps — Y stretched
            sx_surf = sx_free
            sy_surf = sy_free / np.cos(theta_r)
            surf_pk = peak * np.cos(theta_r)
            cfg_key = "topbot"

        irr, off_mm = make_map(sx_surf, sy_surf, surf_pk, cfg_key)
        cov, p2v    = uniformity(irr)
        col_rows.append((theta, cov, p2v, off_mm))
        summary.append((cfg_label.replace("\n", " "), theta, cov, p2v, off_mm))

        # ── subplot ────────────────────────────────────────────────────────
        ax = fig.add_subplot(gs[ri, ci])
        ax.set_facecolor("#0a0a0a")
        ax.imshow(irr, cmap=CMAP, origin="upper", vmin=0,
                  vmax=irr.max() * 1.05, aspect="equal",
                  extent=[0, SENSOR_W, SENSOR_H, 0])

        # specimen border
        ax.add_patch(mpatches.Rectangle(
            (SPEC_OX, SPEC_OY), SPEC_W_PX, SPEC_H_PX,
            ec="lime", fc="none", lw=1.5, ls="--", zorder=5))

        ax.set_xlim(0, SENSOR_W); ax.set_ylim(SENSOR_H, 0)
        ax.set_xticks([]); ax.set_yticks([])

        for sp in ax.spines.values():
            sp.set_edgecolor(cfg_colors[ci]); sp.set_linewidth(2)

        title_str = f"{theta}°  CoV={cov:.1f}%  P2V={p2v:.1f}%"
        if ri == 0:
            title_str = cfg_label.replace("\n", "  ") + "\n" + title_str
        ax.set_title(title_str, color=cfg_colors[ci], fontsize=7.5,
                     fontweight="bold", pad=3)

    # ── summary mini-table in bottom row ──────────────────────────────────
    ax_hdr.set_xlim(0, 1); ax_hdr.set_ylim(0, 1)
    lines = [cfg_label.replace("\n", " / "),
             f"{'Ang':>4}  {'CoV':>6}  {'P2V':>6}  {'Offset':>8}"]
    for (theta, cov, p2v, off) in col_rows:
        lines.append(f"{theta:>4}deg  {cov:>5.1f}%  {p2v:>5.1f}%  {off:>6.1f}mm")
    ax_hdr.text(0.05, 0.95, "\n".join(lines),
                va="top", ha="left", fontsize=7,
                color=cfg_colors[ci], fontfamily="monospace",
                transform=ax_hdr.transAxes)

fig.suptitle(
    "Excitation config comparison  |  9.2mm lens @ 383mm  |  "
    "Specimen 320x175mm  |  Two symmetric beams",
    color="white", fontsize=11)

fig.savefig(OUTPUT_PNG, dpi=130, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved: {OUTPUT_PNG}\n")

# ── console summary ────────────────────────────────────────────────────────
print(f"{'Configuration':<35}  {'Angle':>5}  {'CoV':>7}  {'P2V':>7}  {'Offset':>9}")
print("-" * 72)
for (lbl, theta, cov, p2v, off) in summary:
    print(f"{lbl:<35}  {theta:>5}deg  {cov:>6.1f}%  {p2v:>6.1f}%  {off:>7.1f}mm")
