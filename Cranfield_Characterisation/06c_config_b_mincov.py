"""
Stage 3 (Config B only) — MIN-CoV sweep for the fixed-45deg angled-arm V-trough.

Counterpart to 06b_lamp_config_maxcov45.py, but:
  * Config B ONLY (the angled-arm V-trough the user is building).
  * MINIMISES CoV of irradiance on the camera FOV (the design goal).
  * Sweeps the two rig dimensions over ABSOLUTE mm ranges the user specified:
        d1 in [100, 500] mm   (horizontal arm-root baseline / separation)
        d2 in [ 50, 500] mm   (spacing of the 2 bulbs ALONG each fixed 45deg arm)
  * Standoffs 300 / 500 / 700 mm (the newly-measured set).

Config B — ANGLED-ARM V-trough (geometry identical to 06b_lamp_config_maxcov45.py):
  * 4 bulbs, 2 per side, all on the camera centreline y = 0.
  * Each side is a FIXED 45deg arm (no swivel, no up/down pitch). The bulb axis
    is locked at 45deg yaw toward the FOV.
  * The 2 bulbs on a side are spaced d2 along the arm -> inner bulb at standoff d,
    outer bulb at standoff d + d2. Both aim at x = +-d1/2 on the target plane.
  * The 45deg yaw stretches sigma_x by 1/cos45; sigma_y is unaffected (no pitch).

Beam model from beam_derived_combined.json (refined 2026-06-23 capture set):
  sigma_x(d) = ax*d + bx ;  sigma_y(d) = ay*d + by ;  peak(d) = K * d^exp.

FOV is computed directly from the Boson+ deployment camera geometry
(18 mm lens, 24deg HFOV / 19.3deg VFOV) so the 700 mm standoff (absent from
05_lens_fov_sweep.json) is handled without a re-run.
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
from matplotlib.patches import Rectangle

# ---- I/O --------------------------------------------------------------------
REPO_ROOT   = os.path.dirname(os.path.abspath(__file__))
BEAM_JSON   = os.path.join(REPO_ROOT, "beam_derived_combined.json")
OUTPUT_PNG  = os.path.join(REPO_ROOT, "06c_config_b_mincov.png")
OUTPUT_JSON = os.path.join(REPO_ROOT, "06c_config_b_mincov.json")

# ---- Design constraints (user-specified) ------------------------------------
LOCK_THETA_DEG = 45.0                       # arm angle — fixed (yaw only)
STANDOFFS      = [300, 500, 700]            # newly-measured standoffs (mm)
D1_MM = np.arange(100.0, 500.0 + 1e-6, 5.0)   # horizontal baseline sweep
D2_MM = np.arange( 50.0, 500.0 + 1e-6, 5.0)   # along-arm bulb spacing sweep
GRID_PX = 161                               # target-plane sampling per axis

# ---- Boson+ deployment camera (Stage 2 geometry) ----------------------------
HFOV_DEG, VFOV_DEG = 24.0, 19.3             # 18 mm lens on Boson+ 640x512
def fov_w_mm(d): return 2.0 * d * np.tan(np.radians(HFOV_DEG / 2.0))
def fov_h_mm(d): return 2.0 * d * np.tan(np.radians(VFOV_DEG / 2.0))

# ---- Load refined beam model ------------------------------------------------
with open(BEAM_JSON) as f:
    beam = json.load(f)
bm   = beam["derived_beam_model"]
SX_M = bm["sigma_x_vs_d"]["slope_mm_per_mm"];  SX_B = bm["sigma_x_vs_d"]["intercept_mm"]
SY_M = bm["sigma_y_vs_d"]["slope_mm_per_mm"];  SY_B = bm["sigma_y_vs_d"]["intercept_mm"]
PK_K = bm["peak_vs_d_powerlaw"]["K_amp_K_mm_to_power"]
PK_E = bm["peak_vs_d_powerlaw"]["exponent"]

def sigma_x(d): return max(SX_M * d + SX_B, 1.0)
def sigma_y(d): return max(SY_M * d + SY_B, 1.0)
def peak(d):    return PK_K * d ** PK_E

COS_T = np.cos(np.radians(LOCK_THETA_DEG))

# ---- Config B CoV on the FOV ------------------------------------------------

def cov_for(d, d1, d2, return_map=False):
    """CoV (%) of irradiance over the Boson+ FOV for the fixed-45deg V-trough."""
    fw, fh = fov_w_mm(d), fov_h_mm(d)
    pad = 0.30 * max(fw, fh)
    xs = np.linspace(-fw/2 - pad, fw/2 + pad, GRID_PX)
    ys = np.linspace(-fh/2 - pad, fh/2 + pad, GRID_PX)
    XX, YY = np.meshgrid(xs, ys)

    irr = np.zeros_like(XX)
    centres = []
    for side in (+1, -1):
        for d_eff in (d, d + d2):
            sx_eff = sigma_x(d_eff) / COS_T     # 45deg yaw stretches sigma_x
            sy_eff = sigma_y(d_eff)             # no pitch -> sigma_y unchanged
            pk     = peak(d_eff) * COS_T
            cx, cy = side * d1 / 2.0, 0.0
            irr += pk * np.exp(-0.5 * (((XX - cx) / sx_eff) ** 2
                                         + ((YY - cy) / sy_eff) ** 2))
            centres.append((cx, cy, d_eff))

    in_fov = (np.abs(XX) <= fw/2) & (np.abs(YY) <= fh/2)
    vals = irr[in_fov]
    m, s = vals.mean(), vals.std()
    cov = 100.0 * s / m if m > 0 else np.nan
    if return_map:
        return cov, irr, xs, ys, fw, fh, centres
    return cov

# ---- Sweep ------------------------------------------------------------------
results = []
for d in STANDOFFS:
    sweep = np.full((len(D1_MM), len(D2_MM)), np.nan)
    for i, d1 in enumerate(D1_MM):
        for j, d2 in enumerate(D2_MM):
            sweep[i, j] = cov_for(d, d1, d2)
    i_best, j_best = np.unravel_index(np.nanargmin(sweep), sweep.shape)
    best_d1, best_d2 = float(D1_MM[i_best]), float(D2_MM[j_best])
    edge = (i_best in (0, len(D1_MM)-1)) or (j_best in (0, len(D2_MM)-1))
    results.append({
        "config": "B", "lens_label": "18 mm (Boson+ 24deg HFoV)",
        "standoff_mm": d, "locked_theta_deg": LOCK_THETA_DEG,
        "fov_w_mm": float(fov_w_mm(d)), "fov_h_mm": float(fov_h_mm(d)),
        "best_d1_mm": best_d1, "best_d2_mm": best_d2,
        "min_cov_pct": float(sweep[i_best, j_best]),
        "optimum_on_sweep_edge": bool(edge),
        "d1_grid_mm": D1_MM.tolist(), "d2_grid_mm": D2_MM.tolist(),
        "sweep_cov_pct": sweep.tolist(),
    })
    print(f"d={d:>4} mm | min CoV = {sweep[i_best, j_best]:6.3f}% @ "
          f"d1={best_d1:6.1f} mm, d2={best_d2:6.1f} mm"
          + ("   [EDGE]" if edge else ""))

overall = min(results, key=lambda r: r["min_cov_pct"])

# ---- Figure -----------------------------------------------------------------
fig = plt.figure(figsize=(20, 13))
fig.patch.set_facecolor("#111111")
n = len(STANDOFFS)
gs = mgridspec.GridSpec(3, n, figure=fig, height_ratios=[1.0, 1.0, 0.45],
                        left=0.06, right=0.97, top=0.91, bottom=0.05,
                        hspace=0.42, wspace=0.35)

def dark_ax(ax):
    ax.set_facecolor("#0a0a0a")
    for sp in ax.spines.values(): sp.set_edgecolor("#444444")
    ax.tick_params(colors="#aaaaaa", labelsize=8)

# Row 0: CoV(d1, d2) heatmaps, min marked
for k, r in enumerate(results):
    ax = fig.add_subplot(gs[0, k]); dark_ax(ax)
    sweep = np.array(r["sweep_cov_pct"])
    im = ax.imshow(sweep, origin="lower", aspect="auto", cmap="viridis",
                   extent=[D2_MM[0], D2_MM[-1], D1_MM[0], D1_MM[-1]])
    ax.plot(r["best_d2_mm"], r["best_d1_mm"], "*", color="#ff3b3b", ms=20,
            mec="white", mew=1.2, zorder=5)
    cb = plt.colorbar(im, ax=ax, pad=0.02); cb.ax.tick_params(labelsize=7, colors="white")
    cb.set_label("CoV on FOV [%]", color="#cccccc", fontsize=8)
    ax.set_xlabel("d2  (along-arm bulb spacing) [mm]", color="#aaaaaa", fontsize=8.5)
    ax.set_ylabel("d1  (arm-root baseline) [mm]", color="#aaaaaa", fontsize=8.5)
    ax.set_title(f"{r['standoff_mm']} mm  —  min CoV = {r['min_cov_pct']:.2f}%\n"
                 f"@ d1={r['best_d1_mm']:.0f} mm, d2={r['best_d2_mm']:.0f} mm",
                 color="#ffd93d", fontsize=10, fontweight="bold")

# Row 1: optimum irradiance maps
for k, r in enumerate(results):
    ax = fig.add_subplot(gs[1, k]); dark_ax(ax)
    cov, irr, xs, ys, fw, fh, centres = cov_for(
        r["standoff_mm"], r["best_d1_mm"], r["best_d2_mm"], return_map=True)
    im = ax.imshow(irr, origin="lower", cmap="inferno", aspect="equal",
                   extent=[xs[0], xs[-1], ys[0], ys[-1]])
    ax.add_patch(Rectangle((-fw/2, -fh/2), fw, fh, fill=False,
                           ec="#4dffb8", lw=1.6, ls="--", zorder=4))
    for cx, cy, d_eff in centres:
        ax.plot(cx, cy, "+", color="#7fd4ff", ms=9, mew=1.6, zorder=5)
    cb = plt.colorbar(im, ax=ax, pad=0.02); cb.ax.tick_params(labelsize=7, colors="white")
    cb.set_label("rel. irradiance", color="#cccccc", fontsize=8)
    ax.set_xlabel("x [mm]", color="#aaaaaa", fontsize=8.5)
    ax.set_ylabel("y [mm]", color="#aaaaaa", fontsize=8.5)
    ax.set_title(f"{r['standoff_mm']} mm optimum  (FOV {fw:.0f}x{fh:.0f} mm)",
                 color="white", fontsize=10)

# Row 2: summary table
ax_t = fig.add_subplot(gs[2, :]); ax_t.axis("off")
rows = [["Standoff", "FOV (w x h) mm", "Optimum d1 [mm]", "Optimum d2 [mm]",
         "Min CoV [%]", "On edge?"]]
for r in results:
    rows.append([f"{r['standoff_mm']} mm",
                 f"{r['fov_w_mm']:.0f} x {r['fov_h_mm']:.0f}",
                 f"{r['best_d1_mm']:.0f}", f"{r['best_d2_mm']:.0f}",
                 f"{r['min_cov_pct']:.2f}",
                 "yes" if r["optimum_on_sweep_edge"] else "no"])
tbl = ax_t.table(cellText=rows, cellLoc="center", loc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.7)
for (rr, cc), cell in tbl.get_celld().items():
    cell.set_edgecolor("#555555")
    if rr == 0:
        cell.set_facecolor("#333333"); cell.set_text_props(color="white", fontweight="bold")
    else:
        is_best = results[rr-1]["standoff_mm"] == overall["standoff_mm"]
        cell.set_facecolor("#1f2d1f" if is_best else "#1a1a1a")
        cell.set_text_props(color="#9dff9d" if is_best else "white")

fig.suptitle(
    "Stage 3 — Config B (fixed 45deg angled-arm V-trough): MIN-CoV design sweep\n"
    f"d1 in [{D1_MM[0]:.0f}, {D1_MM[-1]:.0f}] mm  x  d2 in [{D2_MM[0]:.0f}, {D2_MM[-1]:.0f}] mm   |   "
    f"refined beam model (300/500/700 mm capture)   |   "
    f"OVERALL BEST: {overall['standoff_mm']} mm, d1={overall['best_d1_mm']:.0f}, "
    f"d2={overall['best_d2_mm']:.0f} -> CoV {overall['min_cov_pct']:.2f}%",
    color="white", fontsize=13, y=0.985)

fig.savefig(OUTPUT_PNG, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"\nSaved figure: {OUTPUT_PNG}")

# ---- Save JSON --------------------------------------------------------------
with open(OUTPUT_JSON, "w") as f:
    json.dump({
        "config": "B",
        "description": "Fixed-45deg angled-arm V-trough; CoV minimised over (d1,d2).",
        "locked_theta_deg": LOCK_THETA_DEG,
        "d1_range_mm": [float(D1_MM[0]), float(D1_MM[-1])],
        "d2_range_mm": [float(D2_MM[0]), float(D2_MM[-1])],
        "beam_source": BEAM_JSON,
        "camera": {"lens_mm": 18.0, "hfov_deg": HFOV_DEG, "vfov_deg": VFOV_DEG},
        "results": results,
        "overall_best": {k: overall[k] for k in
                         ("standoff_mm", "best_d1_mm", "best_d2_mm", "min_cov_pct")},
    }, f, indent=2)
print(f"Saved JSON:   {OUTPUT_JSON}")
