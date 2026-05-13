"""
Fixture geometry — side lamps, best configuration.

Right-angle triangle
  d = 383 mm  (standoff, vertical leg)
  L           (lateral lamp offset, horizontal leg)
  H = sqrt(d^2 + L^2)   fixture arm length (hypotenuse)

  Beam hits specimen at offset  x_s = L - d*tan(theta)
  Beam path length              R   = d / cos(theta)

CoV is computed analytically via Gaussian separability — no simulation.
  I(x,y) = G_x(x) * G_y(y)   (separable)
  where G_x is the sum of two offset Gaussians in X.
  mean, E[I^2] -> CoV without any pixel loop.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy.special import erf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrow
from beam_on_specimen import load_beam, MM_PER_PX, SPEC_W_MM, SPEC_H_MM

OUTPUT_PNG = config.BOSON_ROOT + r"\fixture_geometry.png"

sx_mm, sy_mm, peak = load_beam()
sx_free = sy_mm    # sigma in X free-space (~59 mm, narrow)
sy_free = sx_mm    # sigma in Y free-space (~108 mm, wide)

D = 383.0          # mm standoff

# ── grids ─────────────────────────────────────────────────────────────────────
ANGLES   = np.arange(10, 75, 1.0)        # degrees
L_VALUES = np.arange(50, 601, 5.0)       # mm

tr  = np.radians(ANGLES)                 # (n_theta,)
LL  = L_VALUES[:, None]                  # (n_L, 1)
TT  = tr[None, :]                        # (1, n_theta)

# geometry
H_GRID     = np.sqrt(D**2 + LL**2)                  # (n_L, n_theta) but L-only
H_VALUES   = np.sqrt(D**2 + L_VALUES**2)             # (n_L,)
OFFSET     = np.abs(LL - D * np.tan(TT))             # (n_L, n_theta)
SX_SURF    = sx_free / np.cos(TT)                    # (1, n_theta) — broadens with angle
SY_SURF    = sy_free                                  # scalar — unchanged

# ── analytical CoV (Gaussian separability) ────────────────────────────────────
# Specimen runs -W/2..W/2 in x, -H/2..H/2 in y
Wx = SPEC_W_MM;  Wy = SPEC_H_MM
xg = np.linspace(-Wx/2, Wx/2, 300)  # fine enough for accurate integrals
yg = np.linspace(-Wy/2, Wy/2, 150)

# G_y is fixed (sx_surf changes, sy_surf = const)
Gy    = np.exp(-0.5 * (yg / SY_SURF)**2)
mGy   = Gy.mean()
mGy2  = (Gy**2).mean()

# G_x for every (L, theta): shape (n_L, n_theta, n_x)
xg3   = xg[None, None, :]
off3  = OFFSET[:, :, None]
sx3   = SX_SURF[None, :, None]        # (1, n_theta, 1) -> broadcast

sx3   = (sx_free / np.cos(tr))[None, :, None]   # (1, n_theta, 1)
Gx    = (np.exp(-0.5 * ((xg3 - off3) / sx3)**2) +
         np.exp(-0.5 * ((xg3 + off3) / sx3)**2))  # (n_L, n_theta, n_x)

mGx   = Gx.mean(axis=2)   # (n_L, n_theta)
mGx2  = (Gx**2).mean(axis=2)

mean_I = mGx * mGy
E_I2   = mGx2 * mGy2
CoV    = np.sqrt(np.maximum(E_I2 - mean_I**2, 0)) / mean_I * 100  # (n_L, n_theta)

print(f"Beam: sx_free={sx_free:.1f}mm  sy_free={sy_free:.1f}mm  d={D:.0f}mm")
print(f"CoV grid: {CoV.shape[0]}L x {CoV.shape[1]}angles computed analytically\n")

# best theta per L
best_idx  = np.nanargmin(CoV, axis=1)             # (n_L,)
best_cov  = CoV[np.arange(len(L_VALUES)), best_idx]
best_th   = ANGLES[best_idx]
best_off  = OFFSET[np.arange(len(L_VALUES)), best_idx]
best_R    = D / np.cos(np.radians(best_th))

# minimum H for CoV targets
def min_H_for(target):
    mask = best_cov <= target
    if not mask.any(): return None
    i = np.where(mask)[0][0]
    return dict(H=H_VALUES[i], L=L_VALUES[i],
                theta=best_th[i], cov=best_cov[i], offset=best_off[i])

h15 = min_H_for(15);  h10 = min_H_for(10);  h5 = min_H_for(5)

print(f"{'Target':>8}  {'Min H':>8}  {'L':>8}  {'Angle':>8}  {'Offset':>10}  {'CoV':>7}")
for tgt, rec in [(15, h15), (10, h10), (5, h5)]:
    if rec:
        print(f"<{tgt:>6}%  {rec['H']:>7.0f}mm  {rec['L']:>7.0f}mm  "
              f"{rec['theta']:>7.1f}deg  {rec['offset']:>9.1f}mm  {rec['cov']:>6.1f}%")

# ── helper ────────────────────────────────────────────────────────────────────
def dark_ax(ax):
    ax.set_facecolor("#1a1a1a")
    for sp in ax.spines.values(): sp.set_edgecolor("#444444")
    ax.tick_params(colors="#aaaaaa", labelsize=9)
    ax.grid(True, alpha=0.12, color="white")

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 26))
fig.patch.set_facecolor("#111111")
gs  = mgridspec.GridSpec(4, 3, figure=fig, hspace=0.52, wspace=0.32,
                          left=0.07, right=0.97, top=0.94, bottom=0.04,
                          height_ratios=[1.2, 1, 1, 1])

# ══════════════════════════════════════════════════════════════════════════════
# ROW 0, Col 0 — geometry diagram (side view)
# ══════════════════════════════════════════════════════════════════════════════
ax_geo = fig.add_subplot(gs[0, 0])
ax_geo.set_facecolor("#0a0a0a")
for sp in ax_geo.spines.values(): sp.set_edgecolor("#333333")
ax_geo.tick_params(colors="#aaaaaa", labelsize=8)
ax_geo.set_aspect("equal")

# Specimen
ax_geo.fill_between([-30, SPEC_W_MM+10], -12, 0, color="#1a3a1a", alpha=0.7)
ax_geo.plot([-30, SPEC_W_MM+10], [0, 0], color="lime", lw=2.5)
ax_geo.text(SPEC_W_MM/2, -20, "Specimen (top edge cross-section)",
            ha="center", color="lime", fontsize=8)

# Camera box at (0, D)
ax_geo.add_patch(mpatches.FancyBboxPatch((-20, D-18), 40, 36,
    boxstyle="round,pad=3", fc="#1a1a3a", ec="#5555cc", lw=2, zorder=5))
ax_geo.text(0, D, "Camera", ha="center", va="center",
            color="#aaaacc", fontsize=8, fontweight="bold", zorder=6)

# d double-arrow (vertical)
ax_geo.annotate("", xy=(-50, 0), xytext=(-50, D),
                arrowprops=dict(arrowstyle="<->", color="white", lw=1.5))
ax_geo.text(-60, D/2, f"d={D:.0f}mm", ha="right", va="center",
            color="white", fontsize=9)

# Draw for three H distances
show_cases = [(h15, "#ff6b6b", "<15% CoV"), (h10, "#ffd93d", "<10% CoV")]
if h5:
    show_cases.append((h5, "#6bcb77", "<5% CoV"))

for rec, col, lbl in show_cases:
    if rec is None: continue
    L_r  = rec["L"]
    H_r  = rec["H"]
    th_r = rec["theta"]
    xs_r = rec["offset"]

    # right-angle triangle sides
    ax_geo.plot([0, L_r], [D, D], color=col, lw=1.2, ls=":")          # L (horizontal)
    ax_geo.plot([L_r, L_r], [0, D], color=col, lw=1.2, ls=":")        # d (vertical at lamp)
    ax_geo.plot([0, L_r], [D, 0], color=col, lw=2, ls="--",           # H (hypotenuse)
                label=f"H={H_r:.0f}mm ({lbl})")

    # beam path: lamp -> hit point
    ax_geo.annotate("", xy=(xs_r, 0), xytext=(L_r, D),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2,
                                   mutation_scale=12), zorder=7)

    # lamp dot
    ax_geo.plot(L_r, D, "o", color=col, ms=9, zorder=8)
    ax_geo.text(L_r+6, D+6, f"L={L_r:.0f}mm\nth={th_r:.0f}deg",
                color=col, fontsize=7.5)

    # offset arrow on specimen
    ax_geo.annotate("", xy=(xs_r, 0), xytext=(SPEC_W_MM/2, 0),
                    arrowprops=dict(arrowstyle="<->", color=col, lw=1.5))
    ax_geo.text((xs_r + SPEC_W_MM/2)/2, 8,
                f"x_s={xs_r:.0f}mm", ha="center", color=col, fontsize=7.5)

    # H label along hypotenuse
    ang_h = np.degrees(np.arctan2(D, L_r))
    ax_geo.text(L_r/2+10, D/2+10, f"H={H_r:.0f}",
                color=col, fontsize=7.5, rotation=-ang_h,
                rotation_mode="anchor", ha="center")

# right-angle mark at camera corner
ax_geo.add_patch(mpatches.Rectangle(
    (show_cases[1][0]["L"]-12, D-12), 12, 12,
    fc="none", ec="#888888", lw=1, alpha=0.7))

# optical axis
ax_geo.plot([0, 0], [0, D-18], color="#333377", lw=1, ls="--")
ax_geo.text(3, D*0.55, "optical\naxis", color="#445577", fontsize=7)

# centre of specimen
ax_geo.plot(SPEC_W_MM/2, 0, "x", color="lime", ms=10, mew=2, zorder=6)
ax_geo.text(SPEC_W_MM/2+5, 8, "centre", color="lime", fontsize=7.5)

ax_geo.set_xlim(-80, max(r[0]["L"] for r in show_cases if r[0]) + 60)
ax_geo.set_ylim(-30, D + 55)
ax_geo.set_xlabel("Horizontal (mm)", color="#aaaaaa", fontsize=9)
ax_geo.set_ylabel("Vertical from specimen (mm)", color="#aaaaaa", fontsize=9)
ax_geo.set_title("Side-view: right-angle triangle\nd (vertical) | L (lateral) | H (hypotenuse = fixture arm)",
                 color="white", fontsize=10)
ax_geo.legend(fontsize=8, facecolor="#222222", labelcolor="white",
              edgecolor="#555555", loc="upper left")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 0, Col 1-2 — CoV heatmap on (H, theta)
# ══════════════════════════════════════════════════════════════════════════════
ax_heat = fig.add_subplot(gs[0, 1:3])
ax_heat.set_facecolor("#0a0a0a")
ax_heat.tick_params(colors="#aaaaaa", labelsize=9)

# CoV grid has shape (n_L, n_theta); y-axis = H_VALUES (per L), not L
im = ax_heat.contourf(ANGLES, H_VALUES, CoV,
                      levels=np.arange(0, 51, 2), cmap="RdYlGn_r", extend="max")
cb = fig.colorbar(im, ax=ax_heat, pad=0.02)
cb.set_label("CoV on specimen (%)", color="white", fontsize=10)
cb.ax.yaxis.set_tick_params(color="white")
plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

cs = ax_heat.contour(ANGLES, H_VALUES, CoV,
                     levels=[5, 10, 15, 20],
                     colors=["white"]*4, linewidths=[2, 1.5, 1.2, 1])
ax_heat.clabel(cs, fmt="%d%%", colors="white", fontsize=9)

# optimal-angle ridge
ax_heat.plot(best_th, H_VALUES, color="cyan", lw=2, ls="--",
             label="Optimal theta at each H")

# H threshold lines
for rec, col, lbl in show_cases:
    if rec:
        ax_heat.axhline(rec["H"], color=col, lw=2, ls="--")
        ax_heat.text(11, rec["H"]+4, f"H={rec['H']:.0f}mm ({lbl})",
                     color=col, fontsize=9, fontweight="bold")

ax_heat.set_xlabel("Beam angle theta (degrees)", color="#aaaaaa", fontsize=11)
ax_heat.set_ylabel("Fixture arm  H = sqrt(d^2+L^2)  (mm)", color="#aaaaaa", fontsize=11)
ax_heat.set_title(
    f"Analytical CoV vs beam angle and fixture arm length\n"
    f"d={D:.0f}mm  |  sx_free={sx_free:.0f}mm  sy_free={sy_free:.0f}mm  |  "
    f"Specimen {SPEC_W_MM:.0f}x{SPEC_H_MM:.0f}mm",
    color="white", fontsize=11)
ax_heat.legend(fontsize=9, facecolor="#222222", labelcolor="white", edgecolor="#555555")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1, Col 0 — Min CoV vs fixture arm H
# ══════════════════════════════════════════════════════════════════════════════
ax_cov = fig.add_subplot(gs[1, 0])
dark_ax(ax_cov)

ax_cov.plot(H_VALUES, best_cov, color="#ffd93d", lw=2.5)

for tgt, col, rec in [(15,"#ff6b6b",h15),(10,"#ffd93d",h10),(5,"#6bcb77",h5)]:
    ax_cov.axhline(tgt, color=col, lw=1.2, ls="--", alpha=0.7)
    if rec:
        ax_cov.axvline(rec["H"], color=col, lw=1.5, ls="--", alpha=0.9)
        ax_cov.annotate(f"H={rec['H']:.0f}mm\nth={rec['theta']:.0f}deg",
                        xy=(rec["H"], tgt),
                        xytext=(rec["H"]+12, tgt+2.5),
                        color=col, fontsize=8, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color=col, lw=1))

ax_cov.set_xlabel("Fixture arm H (mm)", color="#aaaaaa", fontsize=9)
ax_cov.set_ylabel("Best achievable CoV (%)", color="#aaaaaa", fontsize=9)
ax_cov.set_title("Min CoV vs fixture arm\n(at optimal angle for each H)", color="white", fontsize=10)
ax_cov.set_xlim(H_VALUES[0], H_VALUES[-1])
ax_cov.set_ylim(0, 35)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1, Col 1 — Optimal angle & offset vs H
# ══════════════════════════════════════════════════════════════════════════════
ax_th = fig.add_subplot(gs[1, 1])
dark_ax(ax_th)

ax_th.plot(H_VALUES, best_th, color="#6bcb77", lw=2.5, label="Optimal angle theta")
ax2 = ax_th.twinx()
ax2.plot(H_VALUES, best_off, color="#ff9f43", lw=2, ls="--",
         label="Beam offset  x_s (mm)")
ax2.set_ylabel("Beam offset x_s (mm)", color="#ff9f43", fontsize=9)
ax2.tick_params(colors="#ff9f43", labelsize=8)

for rec, col in [(h15,"#ff6b6b"),(h10,"#ffd93d")]:
    if rec:
        ax_th.axvline(rec["H"], color=col, lw=1.2, ls="--", alpha=0.6)

ax_th.set_xlabel("Fixture arm H (mm)", color="#aaaaaa", fontsize=9)
ax_th.set_ylabel("Optimal angle (degrees)", color="#6bcb77", fontsize=9)
ax_th.set_title("Optimal beam angle and offset vs fixture arm",
                color="white", fontsize=10)
l1, lb1 = ax_th.get_legend_handles_labels()
l2, lb2 = ax2.get_legend_handles_labels()
ax_th.legend(l1+l2, lb1+lb2, fontsize=8, facecolor="#222222",
             labelcolor="white", edgecolor="#555555")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1, Col 2 — CoV vs angle at fixed H values (pattern sensitivity)
# ══════════════════════════════════════════════════════════════════════════════
ax_sens = fig.add_subplot(gs[1, 2])
dark_ax(ax_sens)

fixed_Hs = [h15, h10, h5] if h5 else [h15, h10]
cols_s   = ["#ff6b6b", "#ffd93d", "#6bcb77"]

for rec, col in zip(fixed_Hs, cols_s):
    if rec is None: continue
    li = np.argmin(np.abs(H_VALUES - rec["H"]))
    ax_sens.plot(ANGLES, CoV[li, :], color=col, lw=2,
                 label=f"H={rec['H']:.0f}mm  (best th={rec['theta']:.0f}deg)")
    ax_sens.axvline(rec["theta"], color=col, lw=1.2, ls="--", alpha=0.7)

ax_sens.axhline(10, color="white", lw=0.8, ls=":", alpha=0.5)
ax_sens.axhline(15, color="white", lw=0.8, ls=":", alpha=0.5)
ax_sens.set_xlabel("Beam angle theta (degrees)", color="#aaaaaa", fontsize=9)
ax_sens.set_ylabel("CoV (%)", color="#aaaaaa", fontsize=9)
ax_sens.set_title("CoV vs angle at fixed fixture distances\n(shows sensitivity to angle choice)",
                  color="white", fontsize=10)
ax_sens.legend(fontsize=8, facecolor="#222222", labelcolor="white", edgecolor="#555555")
ax_sens.set_ylim(0, 50)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2 — Analytical beam footprint on specimen at 3 fixture distances
# ══════════════════════════════════════════════════════════════════════════════
x_sp = np.linspace(-SPEC_W_MM/2, SPEC_W_MM/2, 400)
y_sp = np.linspace(-SPEC_H_MM/2, SPEC_H_MM/2, 200)
XX_sp, YY_sp = np.meshgrid(x_sp, y_sp)

draw_cases = [(h15, "#ff6b6b", "H={H:.0f}mm  th={theta:.0f}deg  x_s=+-{offset:.0f}mm\n<15% CoV target"),
              (h10, "#ffd93d", "H={H:.0f}mm  th={theta:.0f}deg  x_s=+-{offset:.0f}mm\n<10% CoV target")]
if h5:
    draw_cases.append((h5, "#6bcb77",
                       "H={H:.0f}mm  th={theta:.0f}deg  x_s=+-{offset:.0f}mm\n<5% CoV target"))

for ci, (rec, col, ttl_tmpl) in enumerate(draw_cases[:3]):
    if rec is None: continue
    ax_bp = fig.add_subplot(gs[2, ci])
    ax_bp.set_facecolor("#0a0a0a")
    for sp in ax_bp.spines.values(): sp.set_edgecolor(col); sp.set_linewidth(2)

    tr_r = np.radians(rec["theta"])
    sx_s = sx_free / np.cos(tr_r)
    xs   = rec["offset"]

    ZZ_bp = (np.exp(-0.5 * ((XX_sp - xs) / sx_s)**2 - 0.5 * (YY_sp / sy_free)**2) +
             np.exp(-0.5 * ((XX_sp + xs) / sx_s)**2 - 0.5 * (YY_sp / sy_free)**2))
    ZZ_bp /= ZZ_bp.max()

    im_bp = ax_bp.imshow(ZZ_bp, extent=[-SPEC_W_MM/2, SPEC_W_MM/2, -SPEC_H_MM/2, SPEC_H_MM/2],
                         origin="lower", cmap="inferno", aspect="equal",
                         vmin=0, vmax=1)
    fig.colorbar(im_bp, ax=ax_bp, fraction=0.04, pad=0.02).ax.yaxis.set_tick_params(color="white")

    # contours
    for lv, lc in [(np.exp(-0.5), "white"), (0.5, "#ffd93d")]:
        ax_bp.contour(x_sp, y_sp, ZZ_bp, levels=[lv], colors=[lc], linewidths=1.5)

    # beam centre markers
    for sign in [+1, -1]:
        ax_bp.plot(sign*xs, 0, "+", color="white", ms=16, mew=2.5, zorder=6)
        ax_bp.axvline(sign*xs, color="white", lw=0.8, ls=":", alpha=0.5)

    # specimen boundary
    ax_bp.add_patch(mpatches.Rectangle((-SPEC_W_MM/2, -SPEC_H_MM/2),
                    SPEC_W_MM, SPEC_H_MM,
                    fc="none", ec="lime", lw=2, ls="--", zorder=5))

    # CoV annotation
    li = np.argmin(np.abs(H_VALUES - rec["H"]))
    ti = np.argmin(np.abs(ANGLES - rec["theta"]))
    ax_bp.text(-SPEC_W_MM/2+5, SPEC_H_MM/2-12,
               f"CoV = {CoV[li,ti]:.1f}%", color="white", fontsize=10, fontweight="bold",
               bbox=dict(fc="#000000cc", ec="none", pad=2))

    # H and V cross-sections overlaid
    mid_y_idx = len(y_sp)//2
    mid_x_idx = len(x_sp)//2
    h_line = ZZ_bp[mid_y_idx, :] * (SPEC_H_MM * 0.35)
    v_line = ZZ_bp[:, mid_x_idx] * (SPEC_W_MM * 0.3)
    ax_bp.plot(x_sp, -SPEC_H_MM/2 + h_line, color="#ffd93d", lw=1.8, alpha=0.9)
    ax_bp.plot(-SPEC_W_MM/2 + v_line, y_sp, color="#4d96ff", lw=1.8, alpha=0.9)

    ax_bp.set_xlabel("X (mm)", color="#aaaaaa", fontsize=9)
    ax_bp.set_ylabel("Y (mm)", color="#aaaaaa", fontsize=9)
    ax_bp.tick_params(colors="#aaaaaa", labelsize=8)
    ax_bp.set_title(ttl_tmpl.format(**rec), color=col, fontsize=10, fontweight="bold")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 3 — How the smallest triangle side changes with angle
# ══════════════════════════════════════════════════════════════════════════════
# phi = triangle angle at camera (between H and d) = arctan(L/d)
# As phi increases: L grows, d stays fixed, H grows faster
# Smallest side = L when phi < 45deg, d when phi > 45deg

phi_deg = np.linspace(0.5, 89, 500)
phi_rad = np.radians(phi_deg)
L_phi   = D * np.tan(phi_rad)
H_phi   = D / np.cos(phi_rad)
d_phi   = np.full_like(phi_deg, D)
small_s = np.minimum(L_phi, d_phi)

# — Panel A: all three sides vs angle ————————————————————
ax_sides = fig.add_subplot(gs[3, 0:2])
dark_ax(ax_sides)

ax_sides.plot(phi_deg, H_phi,   color="#ff6b6b", lw=2.5, label="H = hypotenuse (fixture arm)")
ax_sides.plot(phi_deg, d_phi,   color="white",   lw=2.0, ls="--", label=f"d = {D:.0f}mm (standoff, fixed)")
ax_sides.plot(phi_deg, L_phi,   color="#4d96ff", lw=2.5, label="L = lateral offset")
ax_sides.plot(phi_deg, small_s, color="#ffd93d", lw=3.0, ls=":",
              label="Smallest side  min(d, L)")

# 45° transition
ax_sides.axvline(45, color="#6bcb77", lw=1.5, ls="--", alpha=0.8)
ax_sides.text(46, 50, "45deg\nL = d = H/sqrt(2)", color="#6bcb77", fontsize=8.5)
ax_sides.axhline(D, color="white", lw=0.8, ls=":", alpha=0.3)

# mark actual fixture angles for optimal configs
for si, (rec, col, lbl) in enumerate(show_cases):
    if rec is None: continue
    phi_r = np.degrees(np.arctan2(rec["L"], D))
    ax_sides.axvline(phi_r, color=col, lw=1.5, ls="--", alpha=0.8)
    ax_sides.text(phi_r+0.8, H_phi.max()*0.85 - si*70,
                 f"{lbl}\nphi={phi_r:.0f}deg  L={rec['L']:.0f}mm  H={rec['H']:.0f}mm",
                 color=col, fontsize=7.5)

ax_sides.set_xlabel("Triangle angle phi at camera  arctan(L/d)  (degrees)", color="#aaaaaa", fontsize=10)
ax_sides.set_ylabel("Side length (mm)", color="#aaaaaa", fontsize=10)
ax_sides.set_title(
    "How all three triangle sides change with angle\n"
    "d is fixed — L and H grow; smallest side is L below 45deg, d above 45deg",
    color="white", fontsize=10)
ax_sides.legend(fontsize=9, facecolor="#222222", labelcolor="white", edgecolor="#555555")
ax_sides.set_xlim(0, 89)
ax_sides.set_ylim(0, min(H_phi.max(), 1400))

# — Panel B: scaled triangle diagrams at 3 angles ———————
ax_tri = fig.add_subplot(gs[3, 2])
ax_tri.set_facecolor("#0a0a0a")
for sp in ax_tri.spines.values(): sp.set_edgecolor("#333333")
ax_tri.tick_params(colors="#aaaaaa", labelsize=8)
ax_tri.set_aspect("equal")
ax_tri.set_xlim(-0.1, 1.8)
ax_tri.set_ylim(-0.15, 1.25)
ax_tri.axis("off")
ax_tri.set_title("Triangle shape at 3 angles\n(scaled, d=1 unit)", color="white", fontsize=10)

for phi_v, col, yoff in [(20, "#ff6b6b", 0.0),
                          (45, "#ffd93d", 0.0),
                          (70, "#6bcb77", 0.0)]:
    xoff = {20: 0.0, 45: 0.55, 70: 1.15}[phi_v]
    L_n  = np.tan(np.radians(phi_v))          # normalised (d=1)
    H_n  = 1.0 / np.cos(np.radians(phi_v))

    # triangle vertices (right angle at bottom-left)
    vx = np.array([xoff,       xoff + L_n, xoff]) + 0.0
    vy = np.array([yoff + 1.0, yoff + 1.0, yoff])

    ax_tri.fill(vx, vy, color=col, alpha=0.12)
    ax_tri.plot(np.append(vx, vx[0]), np.append(vy, vy[0]),
                color=col, lw=2)

    # right-angle mark
    sq = 0.04
    ax_tri.add_patch(mpatches.Rectangle((xoff, yoff), sq, sq,
                     fc="none", ec=col, lw=1, alpha=0.7))

    # labels
    ax_tri.text(xoff - 0.05, yoff + 0.5, "d", ha="right",
                color="white", fontsize=9, fontweight="bold")
    ax_tri.text(xoff + L_n/2, yoff + 1.0 + 0.05, f"L={L_n:.2f}",
                ha="center", color=col, fontsize=8)
    ax_tri.text(xoff + L_n/2 + 0.04, yoff + 0.5,
                f"H={H_n:.2f}", ha="left", color=col, fontsize=8,
                rotation=-phi_v)

    # smallest-side highlight
    small_v = min(1.0, L_n)
    sm_lbl  = "d=1" if phi_v >= 45 else f"L={L_n:.2f}"
    ax_tri.text(xoff + L_n/2, yoff - 0.09,
                f"phi={phi_v}deg  smallest={sm_lbl}",
                ha="center", color=col, fontsize=7.5, fontweight="bold")

fig.suptitle(
    "Fixture geometry — Side lamps  |  H = sqrt(d^2 + L^2)  (fixture arm = hypotenuse)\n"
    f"d = {D:.0f}mm (standoff, fixed)  |  CoV computed analytically  |  "
    f"Beam: sx_free={sx_free:.0f}mm  sy_free={sy_free:.0f}mm",
    color="white", fontsize=12, y=0.97)

fig.savefig(OUTPUT_PNG, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"\nSaved: {OUTPUT_PNG}")
