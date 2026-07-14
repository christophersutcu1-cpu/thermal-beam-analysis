"""
Stage 3 (variant) — lamp configuration / CoV *maximisation* at locked 45deg.

Counterpart to 06_lamp_config_sweep.py. Locks each lamp's 45deg angle and sweeps
two rig dimensions (d1, d2), picking the pair that MAXIMISES the coefficient of
variation (CoV) of irradiance on the FOV, for two DIFFERENT 4-lamp mechanisms.
(Geometry confirmed against 06b_config_a_planar_check.py / 06b_config_b_arm_check.py.)

Config A — PLANAR swivel panel
  * All 4 bulbs lie in ONE vertical plane, at the SAME standoff d.
  * Bulb positions in the plane: (+-d1/2, +-d2/2)  ->  d1 = horizontal separation,
    d2 = vertical separation (in-plane).
  * Each bulb on a swivel LOCKED at 45deg pitch (vertical tilt) toward the FOV;
    the footprint centre shifts along y by d*tan(45deg) = d (upper bulbs aim down,
    lower bulbs aim up). sigma_y is stretched by 1/cos45.

Config B — ANGLED-ARM V-trough
  * All 4 bulbs on the same horizontal axis (y = 0).
  * A FIXED 45deg arm each side (no swivel); the 2 bulbs on a side are spaced d2
    ALONG the arm, so they sit at DIFFERENT standoffs (inner d, outer d + d2),
    both aiming at +-d1/2. sigma_x is stretched by 1/cos45.
  * d1 = baseline arm-root separation (same meaning as Config A's d1).

Constrained so the max can't collapse to a point:
  d1 >= FOV width (horizontal) ;  d2 in [50, 400] mm.

Beam model from beam_derived_combined.json:
  sigma_x(d) = ax*d + bx;  sigma_y(d) = ay*d + by;  peak(d) = K * d^exp.
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec

# ---- I/O --------------------------------------------------------------------
REPO_ROOT     = os.path.dirname(os.path.abspath(__file__))
BEAM_JSON     = os.path.join(REPO_ROOT, "beam_derived_combined.json")
LENS_JSON     = os.path.join(REPO_ROOT, "05_lens_fov_sweep.json")
OUTPUT_PNG    = os.path.join(REPO_ROOT, "06b_lamp_config_maxcov45.png")
OUTPUT_JSON   = os.path.join(REPO_ROOT, "06b_lamp_config_maxcov45.json")

# ---- Sweep settings ---------------------------------------------------------
LOCK_THETA_DEG = 45.0                      # lamp angle — LOCKED (not swept)
# Constrained max-CoV sweep (so the optimum can't collapse to a single point):
#   d1 >= FOV width (horizontal)  ;  d2 in [50, 400] mm
D1_FRAC = np.linspace(1.0, 2.0, 16)        # baseline d1 as fraction of the horizontal FOV
D2_MM   = np.linspace(50.0, 400.0, 15)     # A: vertical bulb separation; B: along-arm offset
GRID_PX = 121                              # target-plane sampling along each axis

# ---- Load Stage-1 + Stage-2 -------------------------------------------------
with open(BEAM_JSON) as f:
    beam = json.load(f)
with open(LENS_JSON) as f:
    lens_sweep = json.load(f)

bm   = beam["derived_beam_model"]
SX_M = bm["sigma_x_vs_d"]["slope_mm_per_mm"];  SX_B = bm["sigma_x_vs_d"]["intercept_mm"]
SY_M = bm["sigma_y_vs_d"]["slope_mm_per_mm"];  SY_B = bm["sigma_y_vs_d"]["intercept_mm"]
PK_K = bm["peak_vs_d_powerlaw"]["K_amp_K_mm_to_power"]
PK_E = bm["peak_vs_d_powerlaw"]["exponent"]

def sigma_x(d):  return SX_M * d + SX_B
def sigma_y(d):  return SY_M * d + SY_B
def peak(d):     return PK_K * d ** PK_E

COS_T = np.cos(np.radians(LOCK_THETA_DEG))
TAN_T = np.tan(np.radians(LOCK_THETA_DEG))

# ---- Single-config CoV calculation -----------------------------------------

def cov_for(config, lens_f, d, d1, d2, return_map=False):
    """CoV of irradiance on the camera FOV at the locked 45deg angle.
    config : 'A' (planar swivel panel) or 'B' (angled-arm V-trough).
    d1     : horizontal separation / baseline (mm).
    d2     : A -> vertical bulb separation (mm); B -> along-arm offset (mm).
    """
    cell = next(g for g in lens_sweep["grid"]
                if g["lens_f_mm"] == lens_f and g["standoff_mm"] == d)
    fov_w, fov_h = cell["fov_w_mm"], cell["fov_h_mm"]

    pad   = 0.3 * max(fov_w, fov_h)
    xs    = np.linspace(-fov_w/2 - pad, fov_w/2 + pad, GRID_PX)
    ys    = np.linspace(-fov_h/2 - pad, fov_h/2 + pad, GRID_PX)
    XX, YY = np.meshgrid(xs, ys)

    irr = np.zeros_like(XX)
    centres = []

    if config == "A":
        # PLANAR panel: 4 bulbs at (+-d1/2, +-d2/2), all at standoff d, each on a
        # 45deg pitch swivel toward the FOV centre (vertical tilt).
        sx_eff = sigma_x(d)
        sy_eff = sigma_y(d) / COS_T            # sigma_y stretched by the vertical tilt
        pk     = peak(d) * COS_T
        shift  = d * TAN_T                     # footprint shift along y toward centre
        for sx in (-1, +1):
            for sy in (-1, +1):
                cx = sx * d1 / 2.0
                cy = sy * d2 / 2.0 - sy * shift
                irr += pk * np.exp(-0.5 * (((XX - cx) / sx_eff) ** 2
                                             + ((YY - cy) / sy_eff) ** 2))
                centres.append((cx, cy))

    elif config == "B":
        # ANGLED-ARM V-trough: 2 sides at x = +-d1/2, all bulbs on y = 0; inner
        # bulb at standoff d, outer at d + d2 (offset along the fixed 45deg arm).
        for side in (+1, -1):
            for d_eff in (d, d + d2):
                sx_eff = sigma_x(d_eff) / COS_T    # sigma_x stretched by the horizontal tilt
                sy_eff = sigma_y(d_eff)
                pk     = peak(d_eff) * COS_T
                cx, cy = side * d1 / 2.0, 0.0
                irr += pk * np.exp(-0.5 * (((XX - cx) / sx_eff) ** 2
                                             + ((YY - cy) / sy_eff) ** 2))
                centres.append((cx, cy))
    else:
        raise ValueError(config)

    in_fov = (np.abs(XX) <= fov_w/2) & (np.abs(YY) <= fov_h/2)
    vals   = irr[in_fov]
    m, s   = vals.mean(), vals.std()
    cov    = 100 * s / m if m > 0 else np.nan

    if return_map:
        return cov, irr, xs, ys, fov_w, fov_h, centres
    return cov

# ---- Sweep grid -------------------------------------------------------------
STANDOFFS  = lens_sweep["standoffs"]

results = []   # one entry per (config, standoff)
for cfg in ("A", "B"):
    for L in lens_sweep["lenses"]:
        for d in STANDOFFS:
            cell  = next(g for g in lens_sweep["grid"]
                         if g["lens_f_mm"] == L["f_mm"] and g["standoff_mm"] == d)
            fov_w = cell["fov_w_mm"]                       # d1 keyed to horizontal FOV
            sweep = np.full((len(D1_FRAC), len(D2_MM)), np.nan)
            for i, f1 in enumerate(D1_FRAC):
                d1 = f1 * fov_w
                for j, d2 in enumerate(D2_MM):
                    sweep[i, j] = cov_for(cfg, L["f_mm"], d, d1, d2)
            i_best, j_best = np.unravel_index(np.nanargmax(sweep), sweep.shape)
            results.append({
                "config":          cfg,
                "lens_f_mm":       L["f_mm"],
                "lens_label":      L["label"],
                "standoff_mm":     d,
                "locked_theta_deg": LOCK_THETA_DEG,
                "fov_dim_mm":      float(fov_w),
                "best_d1_frac":    float(D1_FRAC[i_best]),
                "best_d1_mm":      float(D1_FRAC[i_best] * fov_w),
                "best_d2_mm":      float(D2_MM[j_best]),
                "max_cov_pct":     float(sweep[i_best, j_best]),
                "sweep_cov_pct":   sweep.tolist(),
            })
            print(f"cfg {cfg} | {L['label']:<28} | d={d:>4} mm | "
                  f"max CoV = {sweep[i_best, j_best]:>7.2f}% @ "
                  f"d1={D1_FRAC[i_best]*fov_w:>6.1f} mm ({D1_FRAC[i_best]:.2f}xFOV), "
                  f"d2={D2_MM[j_best]:>5.1f} mm")

# ---- Figure: 2 (configs) x [schematic + 4 standoffs] -----------------------
from matplotlib.patches import Rectangle, Circle, Ellipse, FancyArrowPatch, Arc

fig = plt.figure(figsize=(24, 12))
fig.patch.set_facecolor("#111111")

n_d = len(STANDOFFS)
gs_top = mgridspec.GridSpec(2, 1 + n_d, figure=fig,
                            left=0.04, right=0.97, top=0.90, bottom=0.36,
                            hspace=0.30, wspace=0.30,
                            width_ratios=[1.35] + [1.0]*n_d)

def dark_ax(ax):
    ax.set_facecolor("#0a0a0a")
    for sp in ax.spines.values(): sp.set_edgecolor("#444444")
    ax.tick_params(colors="#aaaaaa", labelsize=7)

# common vmax per row so heatmaps are comparable within a config
vmax_by_cfg = {}
maps_cache  = {}
for cfg in ("A", "B"):
    row_max = 0.0
    for d in STANDOFFS:
        best = next(r for r in results
                    if r["config"] == cfg and r["standoff_mm"] == d)
        _, irr, xs, ys, fw, fh, centres = cov_for(
            cfg, best["lens_f_mm"], d, best["best_d1_mm"], best["best_d2_mm"],
            return_map=True)
        XX, YY = np.meshgrid(xs, ys)
        in_fov = (np.abs(XX) <= fw/2) & (np.abs(YY) <= fh/2)
        row_max = max(row_max, float(irr[in_fov].max()))
        maps_cache[(cfg, d)] = (irr, xs, ys, fw, fh, centres, best)
    vmax_by_cfg[cfg] = row_max

# ---- schematic (correct per-config mechanism) -------------------------------
def draw_schematic(ax, cfg, accent):
    ax.set_facecolor("#0d0d0d"); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor(accent); sp.set_linewidth(2.5)
    rep = next(r for r in results if r["config"] == cfg and r["standoff_mm"] == 400)
    d1mm, d2mm = rep["best_d1_mm"], rep["best_d2_mm"]

    if cfg == "A":
        ax.set_title("Config A — PLANAR panel (front view)\n45° pitch swivel, all bulbs at standoff d",
                     color=accent, fontsize=10, fontweight="bold", pad=8)
        fw, fh = 1.0, 0.78
        ax.add_patch(Rectangle((-fw/2, -fh/2), fw, fh, fc="#16282b",
                               ec="cyan", lw=2, zorder=3))
        ax.text(0, 0, "FOV", ha="center", va="center", color="cyan",
                fontsize=8, fontweight="bold")
        d1n, d2n = 0.92, 0.62
        for sx in (-1, +1):
            ax.plot([sx*d1n/2, sx*d1n/2], [-d2n/2, d2n/2], color="#888888",
                    lw=2.0, zorder=2)
            for sy in (-1, +1):
                bx, by = sx*d1n/2, sy*d2n/2
                ax.add_patch(Circle((bx, by), 0.058, fc="#1a2a3a", ec=accent,
                                    lw=2, zorder=5))
                ax.add_patch(Circle((bx, by), 0.023, fc="#fff2a0", ec="none", zorder=6))
                ax.add_patch(FancyArrowPatch((bx, by), (bx, by*0.2),
                                             arrowstyle="-|>", color=accent,
                                             mutation_scale=10, lw=1.1, alpha=0.7, zorder=4))
        ax.annotate("", xy=(d1n/2, -0.60), xytext=(-d1n/2, -0.60),
                    arrowprops=dict(arrowstyle="<->", color="white", lw=1.3))
        ax.text(0, -0.66, f"d1 (horiz) = {d1mm:.0f} mm", ha="center", va="top",
                color="white", fontsize=8.5, fontweight="bold")
        ax.annotate("", xy=(-0.60, d2n/2), xytext=(-0.60, -d2n/2),
                    arrowprops=dict(arrowstyle="<->", color=accent, lw=1.3))
        ax.text(-0.65, 0, f"d2 (vert)\n{d2mm:.0f} mm", ha="right", va="center",
                color=accent, fontsize=8.5, fontweight="bold")
        ax.set_xlim(-1.08, 0.82); ax.set_ylim(-0.92, 0.72)
    else:
        ax.set_title("Config B — ANGLED ARM (top view)\nfixed 45°, 2 bulbs along each arm",
                     color=accent, fontsize=10, fontweight="bold", pad=8)
        half = 0.55; inv2 = 1/np.sqrt(2); S0 = 0.30; D2n = 0.45
        ax.plot([-half, half], [0, 0], color="#39d353", lw=5, zorder=3,
                solid_capstyle="round")
        ax.text(0, -0.12, "target / FOV", ha="center", va="top", color="#39d353",
                fontsize=8, fontweight="bold")
        for side in (+1, -1):
            root = (side*half, 0)
            tip  = (root[0] + side*(S0+D2n+0.15)*inv2, (S0+D2n+0.15)*inv2)
            ax.plot([root[0], tip[0]], [root[1], tip[1]], color="#cfcfcf",
                    lw=2.8, zorder=2, solid_capstyle="round")
            for s in (S0, S0+D2n):
                bx, bz = root[0] + side*s*inv2, s*inv2
                ax.add_patch(FancyArrowPatch((bx, bz), root, arrowstyle="-|>",
                                             color=accent, mutation_scale=10,
                                             lw=1.1, alpha=0.8, zorder=4))
                ax.add_patch(Circle((bx, bz), 0.05, fc="#1a2a3a", ec=accent, lw=2, zorder=5))
                ax.add_patch(Circle((bx, bz), 0.02, fc="#fff2a0", ec="none", zorder=6))
            if side == +1:
                a = (root[0]+side*S0*inv2, S0*inv2)
                b = (root[0]+side*(S0+D2n)*inv2, (S0+D2n)*inv2)
                ax.annotate("", xy=b, xytext=a,
                            arrowprops=dict(arrowstyle="<->", color=accent, lw=1.2))
                ax.text((a[0]+b[0])/2+0.05, (a[1]+b[1])/2, f"d2={d2mm:.0f}mm",
                        color=accent, fontsize=7.5, fontweight="bold", va="center", ha="left")
            if side == -1:
                ax.add_patch(Arc(root, 0.28, 0.28, theta1=90, theta2=135,
                                 color="orange", lw=1.5, zorder=6))
                ax.text(root[0]-0.03, 0.20, "45°", color="orange", fontsize=8,
                        fontweight="bold", ha="right")
        ax.annotate("", xy=(half, 0.15), xytext=(-half, 0.15),
                    arrowprops=dict(arrowstyle="<->", color="white", lw=1.3))
        ax.text(0, 0.32, f"d1 = {d1mm:.0f} mm", ha="center", va="bottom",
                color="white", fontsize=8.5, fontweight="bold")
        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.40, 1.0)

config_cols = {"A": "#ff6b6b", "B": "#4d96ff"}

for ci, cfg in enumerate(("A", "B")):
    draw_schematic(fig.add_subplot(gs_top[ci, 0]), cfg, config_cols[cfg])

    for di, d in enumerate(STANDOFFS):
        irr, xs, ys, fw, fh, centres, best = maps_cache[(cfg, d)]
        ax = fig.add_subplot(gs_top[ci, di + 1])
        dark_ax(ax)

        crop_w = fw * 0.65; crop_h = fh * 0.65
        ax.set_xlim(-fw/2 - crop_w*0.1, fw/2 + crop_w*0.1)
        ax.set_ylim(-fh/2 - crop_h*0.1, fh/2 + crop_h*0.1)
        ax.set_aspect("equal")

        im = ax.imshow(irr, cmap="inferno", origin="lower",
                       extent=[xs[0], xs[-1], ys[0], ys[-1]],
                       vmin=0, vmax=vmax_by_cfg[cfg])

        ax.add_patch(Rectangle((-fw/2, -fh/2), fw, fh,
                               fc="none", ec="cyan", lw=2, zorder=4))

        for cx, cy in centres:
            if (xs[0] <= cx <= xs[-1]) and (ys[0] <= cy <= ys[-1]):
                ax.plot(cx, cy, "+", color="white", ms=12, mew=2, zorder=5)

        ax.set_title(
            f"d = {d} mm  |  FOV {fw:.0f}x{fh:.0f} mm\n"
            f"d1={best['best_d1_mm']:.0f}mm  d2={best['best_d2_mm']:.0f}mm  "
            f"-> CoV={best['max_cov_pct']:.2f}%",
            color="white", fontsize=8.5, pad=4)
        ax.set_xlabel("x (mm)", color="#aaaaaa", fontsize=8)

        if di == n_d - 1:
            cb = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.045)
            cb.ax.tick_params(labelsize=7, colors="white")
            cb.set_label("Irradiance (relative)", color="white", fontsize=8)

# ---- bottom: CoV comparison + optimum table --------------------------------
gs_bot = mgridspec.GridSpec(1, 2, figure=fig,
                            left=0.07, right=0.97, top=0.30, bottom=0.06,
                            wspace=0.25)

ax_cov = fig.add_subplot(gs_bot[0])
dark_ax(ax_cov)
ax_cov.set_facecolor("#1a1a1a")
markers = {"A": "o", "B": "s"}
for cfg in ("A", "B"):
    rows = sorted([r for r in results if r["config"] == cfg],
                  key=lambda r: r["standoff_mm"])
    xs_ = [r["standoff_mm"] for r in rows]
    ys_ = [r["max_cov_pct"]  for r in rows]
    ax_cov.plot(xs_, ys_, marker=markers[cfg], color=config_cols[cfg],
                lw=2, ms=10, label=f"Config {cfg}")
    for x, y in zip(xs_, ys_):
        ax_cov.text(x, y + 0.5, f"{y:.1f}%", ha="center",
                    color=config_cols[cfg], fontsize=8, fontweight="bold")
ax_cov.set_xlabel("Standoff (mm)", color="#aaaaaa")
ax_cov.set_ylabel("MAX CoV across (d1, d2) sweep (%)", color="#aaaaaa")
ax_cov.set_title("Worst-case (max) CoV vs standoff — 18 mm lens, angle locked 45°",
                 color="white", fontsize=10)
ax_cov.legend(fontsize=8.5, facecolor="#222222", labelcolor="white",
              edgecolor="#555555", loc="upper left")
ax_cov.set_xticks(STANDOFFS)

ax_t = fig.add_subplot(gs_bot[1])
ax_t.set_facecolor("#111111"); ax_t.axis("off")
table_data = [["d (mm)",
               "A d1 (mm)", "A d2 (mm)", "A CoV",
               "B d1 (mm)", "B d2 (mm)", "B CoV"]]
for d in STANDOFFS:
    a = next(r for r in results if r["config"] == "A" and r["standoff_mm"] == d)
    b = next(r for r in results if r["config"] == "B" and r["standoff_mm"] == d)
    table_data.append([
        f"{d}",
        f"{a['best_d1_mm']:.0f}",  f"{a['best_d2_mm']:.0f}",  f"{a['max_cov_pct']:.1f}%",
        f"{b['best_d1_mm']:.0f}",  f"{b['best_d2_mm']:.0f}",  f"{b['max_cov_pct']:.1f}%",
    ])

col_x = [0.02, 0.16, 0.32, 0.47, 0.62, 0.78, 0.92]
row_h = 0.13
y0    = 0.92
for ri, row in enumerate(table_data):
    y = y0 - ri * row_h
    if ri == 0:
        col_for = ["white", "#ff6b6b", "#ff6b6b", "#ff6b6b",
                   "#4d96ff", "#4d96ff", "#4d96ff"]
        fw_ = "bold"
    else:
        col_for = ["white"] * 7
        fw_ = "normal"
        a_cov = float(row[3].rstrip("%"))
        b_cov = float(row[6].rstrip("%"))
        if a_cov > b_cov:                                 # green = HIGHER CoV
            col_for[1] = col_for[2] = col_for[3] = "lime"
        else:
            col_for[4] = col_for[5] = col_for[6] = "lime"
    for ci, (val, c) in enumerate(zip(row, col_for)):
        ax_t.text(col_x[ci], y, val, transform=ax_t.transAxes,
                  color=c, fontsize=9.5, fontweight=fw_, va="top")
ax_t.set_title("Optimum (d1, d2) per (config, standoff) — green = HIGHER CoV",
               color="white", fontsize=10, pad=4)

fig.suptitle(
    "Stage 3 (variant) — Lamp configs that MAXIMISE CoV, angle LOCKED at 45° (Cranfield, 18 mm lens)\n"
    "Config A: planar swivel panel — bulbs at (±d1/2, ±d2/2), same standoff   |   "
    "Config B: angled-arm V-trough — bulbs on y=0, d2 along the fixed 45° arm",
    color="white", fontsize=13, y=0.97)

fig.savefig(OUTPUT_PNG, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"\nSaved figure: {OUTPUT_PNG}")

# ---- JSON output ------------------------------------------------------------
overall_worst = max(results, key=lambda r: r["max_cov_pct"])
out = {
    "objective": "MAXIMISE CoV (worst-case uniformity)",
    "geometry": {
        "A": "planar swivel panel — bulbs at (+-d1/2, +-d2/2), same standoff d, 45deg pitch",
        "B": "angled-arm V-trough — bulbs on y=0, inner at d / outer at d+d2 along fixed 45deg arm",
    },
    "sweep_settings": {
        "locked_theta_deg": LOCK_THETA_DEG,
        "d1_fracs_of_fov_w": D1_FRAC.tolist(),
        "d2_mm":             D2_MM.tolist(),
        "grid_px":           GRID_PX,
    },
    "beam_source":       BEAM_JSON,
    "lens_sweep_source": LENS_JSON,
    "results":           results,
    "overall_max_cov":   {k: v for k, v in overall_worst.items() if k != "sweep_cov_pct"},
}
with open(OUTPUT_JSON, "w") as f:
    json.dump(out, f, indent=2)
print(f"Saved JSON:   {OUTPUT_JSON}")

print(f"\n=== OVERALL MAX CoV ===")
print(f"  Config:    {overall_worst['config']}")
print(f"  Standoff:  {overall_worst['standoff_mm']} mm")
print(f"  Angle:     {overall_worst['locked_theta_deg']:.0f} deg (locked)")
print(f"  d1:        {overall_worst['best_d1_mm']:.0f} mm ({overall_worst['best_d1_frac']:.2f} x FOV)")
print(f"  d2:        {overall_worst['best_d2_mm']:.0f} mm")
print(f"  CoV:       {overall_worst['max_cov_pct']:.2f}%")
