"""
Fair comparison of 4 lamp configurations at 45 degrees, optimum offset for each.
Specimen 320x175mm, 9.2mm lens @ 383mm standoff.

Configs:
  1. Side   + vertical filament   (current) — lamps left/right, filament vertical
  2. Side   + horizontal filament           — lamps left/right, filament horizontal
  3. Top/Bot + horizontal filament          — lamps above/below, filament horizontal
  4. Top/Bot + vertical filament            — lamps above/below, filament vertical
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

OUTPUT_PNG = config.BOSON_ROOT + r"\four_config_comparison.png"

sx_mm, sy_mm, peak = load_beam()
# Corrected orientation:
#   Horizontal filament (runs left-right) -> reflector fans light horizontally -> beam WIDER in X
#   Vertical filament   (runs up-down)    -> reflector fans light vertically   -> beam TALLER in Y
# From measurements: sx_mm=108mm (wide), sy_mm=59mm (narrow)
# So sx_mm belongs to horizontal filament, sy_mm to vertical.

THETA = 45.0
tr    = np.radians(THETA)

CONFIGS = [
    # (label, axis, sx_free, sy_free, colour)
    # vertical filament -> beam taller: sx=sy_mm(59mm), sy=sx_mm(108mm)
    ("1. Side — Vertical filament\n(CURRENT)",    "x", sy_mm, sx_mm, "#ff6b6b"),
    # horizontal filament -> beam wider: sx=sx_mm(108mm), sy=sy_mm(59mm)
    ("2. Side — Horizontal filament",             "x", sx_mm, sy_mm, "#ffd93d"),
    ("3. Top/Bottom — Horizontal filament",       "y", sx_mm, sy_mm, "#4d96ff"),
    ("4. Top/Bottom — Vertical filament",         "y", sy_mm, sx_mm, "#6bcb77"),
]

CMAP = "inferno"

# ── helpers ──────────────────────────────────────────────────────────────────

def optimal_offset(spec_len_px, sigma_px):
    coords = np.arange(spec_len_px)
    cx = spec_len_px / 2
    best_c, best_off = 1e9, 0.0
    for off in np.linspace(0, sigma_px * 2.5, 400):
        ix = (np.exp(-0.5*((coords - cx - off)/sigma_px)**2) +
              np.exp(-0.5*((coords - cx + off)/sigma_px)**2))
        c = ix.std() / ix.mean() * 100
        if c < best_c:
            best_c, best_off = c, off
    return best_off * MM_PER_PX   # return in mm


def build_irr(sx_surf_mm, sy_surf_mm, surf_peak, offset_mm, axis):
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
            -0.5*(((xx-bx)/sx_px)**2 + ((yy-by)/sy_px)**2)
        ).astype(np.float32)
    return irr


def uniformity_2d(irr):
    spec = irr[SPEC_OY:SPEC_OY+SPEC_H_PX, SPEC_OX:SPEC_OX+SPEC_W_PX]
    m    = spec.mean()
    return spec.std()/m*100, (spec.max()-spec.min())/m*100

# ── pre-compute ───────────────────────────────────────────────────────────────
computed = []
vmax_all = 0

for label, axis, sx_f, sy_f, col in CONFIGS:
    sx_s = sx_f / np.cos(tr) if axis == "x" else sx_f
    sy_s = sy_f / np.cos(tr) if axis == "y" else sy_f
    sp   = peak * np.cos(tr)

    ref_sigma_px = (sx_s if axis == "x" else sy_s) / MM_PER_PX
    ref_len_px   = SPEC_W_PX if axis == "x" else SPEC_H_PX
    off_mm       = optimal_offset(ref_len_px, ref_sigma_px)

    irr          = build_irr(sx_s, sy_s, sp, off_mm, axis)
    cov, p2v     = uniformity_2d(irr)
    spec_patch   = irr[SPEC_OY:SPEC_OY+SPEC_H_PX, SPEC_OX:SPEC_OX+SPEC_W_PX]

    vmax_all = max(vmax_all, irr.max())
    computed.append(dict(label=label, axis=axis, col=col,
                         sx_s=sx_s, sy_s=sy_s, off_mm=off_mm,
                         irr=irr, spec=spec_patch, cov=cov, p2v=p2v))

    print(f"{label.replace(chr(10),' '):<40}  "
          f"sx_surf={sx_s:.0f}mm  sy_surf={sy_s:.0f}mm  "
          f"offset={off_mm:.0f}mm  CoV={cov:.1f}%  P2V={p2v:.1f}%")

# ── figure  (4 columns x 3 rows) ─────────────────────────────────────────────
# Row 0: lamp position schematic
# Row 1: irradiance map on full sensor + specimen outline
# Row 2: H and V cross-sections on specimen

fig = plt.figure(figsize=(22, 14))
fig.patch.set_facecolor("#111111")
gs  = plt.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.25,
                   left=0.05, right=0.96, top=0.93, bottom=0.05)

for ci, d in enumerate(computed):
    col = d["col"]

    # ── Row 0: schematic (front view) ────────────────────────────────────────
    ax_s = fig.add_subplot(gs[0, ci])
    ax_s.set_facecolor("#0d0d0d")
    ax_s.set_xlim(-6, 6); ax_s.set_ylim(-4, 4)
    ax_s.set_aspect("equal"); ax_s.set_xticks([]); ax_s.set_yticks([])
    for sp in ax_s.spines.values():
        sp.set_edgecolor(col); sp.set_linewidth(2)

    # specimen
    sw, sh = 3.2, 1.75
    ax_s.add_patch(mpatches.Rectangle((-sw/2,-sh/2), sw, sh,
                   fc="#1a3a1a", ec="lime", lw=2, zorder=3))
    ax_s.text(0, 0, f"{SPEC_W_MM:.0f}×{SPEC_H_MM:.0f}mm",
              ha="center", va="center", color="lime", fontsize=8, zorder=4)

    # camera
    ax_s.add_patch(mpatches.Ellipse((0,0), 0.6, 0.45,
                   fc="#223344", ec="white", lw=1.5, zorder=5))
    ax_s.text(0, 0, "CAM", ha="center", va="center",
              color="white", fontsize=6.5, zorder=6)

    # lamps + filament + beam arrow
    is_vertical_filament = "Vertical" in d["label"]
    # housing orientation matches filament: vertical filament = portrait box, horizontal = landscape
    if is_vertical_filament:
        lw_box, lh_box = 0.65, 1.3   # portrait (taller than wide)
    else:
        lw_box, lh_box = 1.3, 0.65   # landscape (wider than tall)

    if d["axis"] == "x":   # side lamps
        positions = [(-5.0, 0), (5.0, 0)]
    else:                  # top/bottom lamps
        positions = [(0, 3.2), (0, -3.2)]

    # beam ellipses on specimen (footprint)
    bw = d["sx_s"] / (SPEC_W_MM/sw)
    bh = d["sy_s"] / (SPEC_H_MM/sh)
    off_norm = d["off_mm"] / (SPEC_W_MM/sw if d["axis"]=="x" else SPEC_H_MM/sh)
    for sign in [+1, -1]:
        bx = sign*off_norm if d["axis"]=="x" else 0
        by = sign*off_norm if d["axis"]=="y" else 0
        ax_s.add_patch(mpatches.Ellipse((bx, by), bw*2, bh*2,
                       fc=col, alpha=0.15, ec=col, lw=1, ls="--", zorder=2))

    for px, py in positions:
        # lamp box — portrait for vertical filament, landscape for horizontal
        ax_s.add_patch(mpatches.Rectangle(
            (px-lw_box/2, py-lh_box/2), lw_box, lh_box,
            fc="#1a2a3a", ec=col, lw=2, zorder=3))
        # filament line inside box
        if is_vertical_filament:
            ax_s.plot([px, px], [py-lh_box/2+0.1, py+lh_box/2-0.1],
                      color="yellow", lw=3, zorder=5)
        else:
            ax_s.plot([px-lw_box/2+0.1, px+lw_box/2-0.1], [py, py],
                      color="yellow", lw=3, zorder=5)
        # beam arrow toward specimen
        if d["axis"] == "x":
            dx = -np.sign(px)*1.8
            ax_s.annotate("", xy=(px+dx*0.9, py), xytext=(px+dx*0.05, py),
                          arrowprops=dict(arrowstyle="-|>", color=col,
                                          lw=1.8, mutation_scale=13), zorder=4)
        else:
            dy = -np.sign(py)*1.5
            ax_s.annotate("", xy=(px, py+dy*0.9), xytext=(px, py+dy*0.05),
                          arrowprops=dict(arrowstyle="-|>", color=col,
                                          lw=1.8, mutation_scale=13), zorder=4)

    ax_s.set_title(d["label"], color=col, fontsize=9, fontweight="bold", pad=6)

    # ── Row 1: specimen close-up with offset annotation ──────────────────────
    ax_m = fig.add_subplot(gs[1, ci])
    ax_m.set_facecolor("#0a0a0a")

    # show only the specimen patch in physical mm coordinates
    ax_m.imshow(d["spec"], cmap=CMAP, origin="upper", vmin=0, vmax=vmax_all,
                aspect="auto", extent=[0, SPEC_W_MM, SPEC_H_MM, 0])
    ax_m.set_xlim(0, SPEC_W_MM); ax_m.set_ylim(SPEC_H_MM, 0)
    ax_m.tick_params(colors="#666666", labelsize=7)
    for sp in ax_m.spines.values():
        sp.set_edgecolor(col); sp.set_linewidth(2)
    ax_m.set_xlabel("mm", color="#aaaaaa", fontsize=8)
    ax_m.set_title(f"CoV={d['cov']:.1f}%   P2V={d['p2v']:.1f}%",
                   color=col, fontsize=9, fontweight="bold")

    # specimen centre in mm
    cx_mm = SPEC_W_MM / 2   # 160mm
    cy_mm = SPEC_H_MM / 2   # 87.5mm
    off   = d["off_mm"]

    if d["axis"] == "x":
        # beam centres at cx_mm ± off_mm  (may be outside specimen — clip to edge)
        b1 = min(cx_mm + off, SPEC_W_MM)
        b2 = max(cx_mm - off, 0)

        # arrow from centre to right beam centre (clipped)
        ax_m.annotate("", xy=(b1, cy_mm), xytext=(cx_mm, cy_mm),
                      arrowprops=dict(arrowstyle="-|>" if b1 < SPEC_W_MM else "-",
                                      color="white", lw=2, mutation_scale=13), zorder=8)
        # left beam centre
        ax_m.annotate("", xy=(b2, cy_mm), xytext=(cx_mm, cy_mm),
                      arrowprops=dict(arrowstyle="-|>" if b2 > 0 else "-",
                                      color="white", lw=2, mutation_scale=13), zorder=8)

        # beam centre markers (only if inside specimen)
        for bx in [cx_mm + off, cx_mm - off]:
            if 0 <= bx <= SPEC_W_MM:
                ax_m.plot(bx, cy_mm, "+", color="white", ms=12, mew=2.5, zorder=9)
            else:
                # arrow at edge pointing outward
                edge = SPEC_W_MM if bx > SPEC_W_MM else 0
                ax_m.annotate("→" if bx > SPEC_W_MM else "←",
                              xy=(edge, cy_mm), ha="center", va="center",
                              color="white", fontsize=12, zorder=9)

        ax_m.text(cx_mm, cy_mm - 12,
                  f"← offset {off:.0f}mm →",
                  ha="center", color="white", fontsize=8, fontweight="bold",
                  bbox=dict(fc="#000000", alpha=0.65, pad=2, ec="none"), zorder=10)

    else:
        b1 = min(cy_mm + off, SPEC_H_MM)
        b2 = max(cy_mm - off, 0)

        ax_m.annotate("", xy=(cx_mm, b1), xytext=(cx_mm, cy_mm),
                      arrowprops=dict(arrowstyle="-|>" if b1 < SPEC_H_MM else "-",
                                      color="white", lw=2, mutation_scale=13), zorder=8)
        ax_m.annotate("", xy=(cx_mm, b2), xytext=(cx_mm, cy_mm),
                      arrowprops=dict(arrowstyle="-|>" if b2 > 0 else "-",
                                      color="white", lw=2, mutation_scale=13), zorder=8)

        for by in [cy_mm + off, cy_mm - off]:
            if 0 <= by <= SPEC_H_MM:
                ax_m.plot(cx_mm, by, "+", color="white", ms=12, mew=2.5, zorder=9)

        ax_m.text(cx_mm + 10, cy_mm,
                  f"offset\n{off:.0f}mm",
                  ha="left", va="center", color="white", fontsize=8, fontweight="bold",
                  bbox=dict(fc="#000000", alpha=0.65, pad=2, ec="none"), zorder=10)

    # specimen centre dot
    ax_m.plot(cx_mm, cy_mm, "o", color="lime", ms=7, mew=0,
              mfc="lime", zorder=9, alpha=0.9)
    ax_m.text(cx_mm + 4, cy_mm + 8, "centre", color="lime", fontsize=7, zorder=10)

    # ── Row 2: cross-sections ─────────────────────────────────────────────────
    ax_x = fig.add_subplot(gs[2, ci])
    ax_x.set_facecolor("#1a1a1a")
    for sp in ax_x.spines.values(): sp.set_edgecolor("#444444")
    ax_x.tick_params(colors="#aaaaaa", labelsize=8)
    ax_x.grid(True, alpha=0.12, color="white")

    spec  = d["spec"]
    mean  = spec.mean()
    x_mm  = np.linspace(0, SPEC_W_MM, SPEC_W_PX)
    y_mm  = np.linspace(0, SPEC_H_MM, SPEC_H_PX)
    h_mid = spec[SPEC_H_PX//2, :]
    v_mid = spec[:, SPEC_W_PX//2]

    ax_x.plot(x_mm, h_mid/mean, color=col, lw=2, label="H (width)")
    # plot V profile on its own normalised x-axis scaled to specimen width
    v_x = np.linspace(0, SPEC_W_MM, SPEC_H_PX)
    ax_x.plot(v_x, v_mid/mean,
              color="white", lw=1.5, ls="--", alpha=0.7, label="V (height)")
    ax_x.axhspan(0.90, 1.10, color="lime", alpha=0.07)
    ax_x.axhline(0.90, color="lime", lw=0.7, ls=":")
    ax_x.axhline(1.10, color="lime", lw=0.7, ls=":")
    ax_x.set_xlim(0, SPEC_W_MM)
    ax_x.set_ylim(0, 1.6)
    ax_x.set_xlabel("Along specimen width (mm)", color="#aaaaaa", fontsize=8)
    if ci == 0:
        ax_x.set_ylabel("Norm. irradiance", color="#aaaaaa", fontsize=8)
        ax_x.legend(fontsize=7.5, facecolor="#222222",
                    labelcolor="white", edgecolor="#555555")
    ax_x.set_title("±10% band (green)", color="white", fontsize=8)

fig.suptitle(
    f"4 lamp configurations  |  45° beam angle  |  Optimum offset for each  |  "
    f"9.2mm lens @ 383mm  |  Specimen {SPEC_W_MM:.0f}×{SPEC_H_MM:.0f}mm\n"
    f"Yellow line = filament orientation   Dashed ellipses = beam footprint",
    color="white", fontsize=11)

fig.savefig(OUTPUT_PNG, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"\nSaved: {OUTPUT_PNG}")
