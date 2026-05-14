"""
Stage 3 — lamp configuration / CoV minimisation.

For each (lens, standoff) combination from Stage 2, evaluate two 4-lamp
arrangements and find the (tilt, offset) combination that minimises the
coefficient of variation (CoV) of irradiance across the camera FOV.

Configurations:
  A — top + bottom pairs (2 lamps above FOV, 2 lamps below)
        sweep: lamp tilt angle (pitch), within-pair horizontal offset
  B — left + right pairs (2 lamps left, 2 lamps right; lamps side-by-side on each side)
        sweep: lamp tilt angle (yaw), within-pair vertical offset

Geometry (target plane at z=0, optical axis up at z=d):
  Config A — lamp mounting at y = ±(FOV_H/2 + CLEARANCE), z = d
             each lamp pitched by θ toward FOV centre (top tilts down, bottom up).
             Footprint:
                centre_y = ±(H_mount - d·tan θ)       (moves toward 0 as θ grows)
                centre_x = ±(off/2)                   (within-pair horizontal offset)
                σ_x' = σ_x                            (perpendicular to tilt axis)
                σ_y' = σ_y / cos θ                    (parallel to tilt axis -> stretched)
                peak' = peak(d/cosθ) · cos θ          (1/d² fall-off + slant projection)
  Config B — all 4 lamps at y = 0 (same horizontal line as the camera), with 2
             lamps side-by-side on each side of the FOV at z = d.
             Lamp x-positions: inner pair at x = ±(FOV_W/2 + CLEARANCE),
                               outer pair at x = ±(FOV_W/2 + CLEARANCE + pair_off).
             Each lamp yawed by θ toward FOV centre.
             Footprint:
                centre_y = 0
                centre_x = x_lamp ∓ d·tan θ          (sign brings beam toward x=0)
                σ_x' = σ_x / cos θ                   (parallel to tilt axis -> stretched)
                σ_y' = σ_y                            (unchanged)

Beam model loaded from beam_derived_combined.json:
  σ_x(d) = ax·d + bx;  σ_y(d) = ay·d + by;  peak(d) = K · d^exp.
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
OUTPUT_PNG    = os.path.join(REPO_ROOT, "06_lamp_config_sweep.png")
OUTPUT_JSON   = os.path.join(REPO_ROOT, "06_lamp_config_sweep.json")

# ---- Sweep settings ---------------------------------------------------------
LAMP_CLEARANCE_MM = 50.0              # how far outside the FOV the lamp mount sits
THETA_DEGS  = np.arange(0, 65, 5)     # lamp tilt in deg
OFFSET_FRAC = np.linspace(0.0, 1.0, 11)  # within-pair offset as fraction of FOV in that axis
GRID_PX     = 121                     # target-plane sampling along each axis

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

# ---- Single-config CoV calculation -----------------------------------------

def cov_for(config, lens_f, d, theta_deg, offset_frac, return_map=False):
    """Compute CoV of irradiance on the camera FOV for the given parameters.
    config: 'A' (top+bot pairs) or 'B' (left+right pairs).
    offset_frac: within-pair offset as fraction of FOV width (A) or height (B).
    """
    # FOV at this (lens, d) from the Stage-2 grid
    cell = next(g for g in lens_sweep["grid"]
                if g["lens_f_mm"] == lens_f and g["standoff_mm"] == d)
    fov_w, fov_h = cell["fov_w_mm"], cell["fov_h_mm"]

    # Target-plane sampling — slightly oversize so beam centres outside FOV still contribute
    pad   = 0.3 * max(fov_w, fov_h)
    xs    = np.linspace(-fov_w/2 - pad, fov_w/2 + pad, GRID_PX)
    ys    = np.linspace(-fov_h/2 - pad, fov_h/2 + pad, GRID_PX)
    XX, YY = np.meshgrid(xs, ys)

    th       = np.radians(theta_deg)
    cos_t    = max(np.cos(th), 1e-3)
    d_slant  = d / cos_t                          # lamp-to-target along beam axis

    # Beam shape at d_slant, plus slant-projection cos factor on peak
    sx_b = sigma_x(d_slant)
    sy_b = sigma_y(d_slant)
    pk_b = peak(d_slant) * cos_t

    # Beam centres on target for the 4 lamps
    if config == "A":
        h_mount  = fov_h/2 + LAMP_CLEARANCE_MM
        off_half = (fov_w/2) * offset_frac           # half the within-pair horizontal spread
        cy_top   = +(h_mount - d * np.tan(th))
        cy_bot   = -(h_mount - d * np.tan(th))
        centres  = [(-off_half, cy_top), (+off_half, cy_top),
                    (-off_half, cy_bot), (+off_half, cy_bot)]
        sx_eff, sy_eff = sx_b, sy_b / cos_t          # σ_y stretched by tilt
    elif config == "B":
        # All 4 lamps at y = 0 (same line as camera); inner-pair lamps sit just
        # outside the FOV horizontally, outer-pair lamps step further out by
        # pair_off. All four share the same yaw angle θ.
        inner_x  = fov_w/2 + LAMP_CLEARANCE_MM
        pair_off = (fov_w/2) * offset_frac
        outer_x  = inner_x + pair_off
        # Where each beam centre lands on the target plane:
        bx_l_inner = -inner_x + d * np.tan(th)
        bx_l_outer = -outer_x + d * np.tan(th)
        bx_r_inner = +inner_x - d * np.tan(th)
        bx_r_outer = +outer_x - d * np.tan(th)
        centres  = [(bx_l_outer, 0), (bx_l_inner, 0),
                    (bx_r_inner, 0), (bx_r_outer, 0)]
        sx_eff, sy_eff = sx_b / cos_t, sy_b          # σ_x stretched by tilt
    else:
        raise ValueError(config)

    irr = np.zeros_like(XX)
    for cx, cy in centres:
        irr += pk_b * np.exp(-0.5 * (((XX - cx) / sx_eff) ** 2
                                       + ((YY - cy) / sy_eff) ** 2))

    # CoV restricted to the FOV rectangle
    in_fov = (np.abs(XX) <= fov_w/2) & (np.abs(YY) <= fov_h/2)
    vals   = irr[in_fov]
    m, s   = vals.mean(), vals.std()
    cov    = 100 * s / m if m > 0 else np.nan

    if return_map:
        return cov, irr, xs, ys, fov_w, fov_h, centres, sx_eff, sy_eff, pk_b
    return cov

# ---- Sweep grid -------------------------------------------------------------
LENSES_F   = [L["f_mm"] for L in lens_sweep["lenses"]]
STANDOFFS  = lens_sweep["standoffs"]

results = []   # one entry per (config, lens, standoff)
for cfg in ("A", "B"):
    for L in lens_sweep["lenses"]:
        for d in STANDOFFS:
            sweep = np.full((len(THETA_DEGS), len(OFFSET_FRAC)), np.nan)
            for ti, th in enumerate(THETA_DEGS):
                for oi, of in enumerate(OFFSET_FRAC):
                    sweep[ti, oi] = cov_for(cfg, L["f_mm"], d, th, of)
            ti_best, oi_best = np.unravel_index(np.nanargmin(sweep), sweep.shape)
            results.append({
                "config":           cfg,
                "lens_f_mm":        L["f_mm"],
                "lens_label":       L["label"],
                "standoff_mm":      d,
                "best_theta_deg":   float(THETA_DEGS[ti_best]),
                "best_offset_frac": float(OFFSET_FRAC[oi_best]),
                "min_cov_pct":      float(sweep[ti_best, oi_best]),
                "sweep_cov_pct":    sweep.tolist(),
            })
            print(f"cfg {cfg} | {L['label']:<32} | d={d:>4} mm | "
                  f"min CoV = {sweep[ti_best, oi_best]:>6.2f}% @ tilt={THETA_DEGS[ti_best]:>3} deg, "
                  f"off={OFFSET_FRAC[oi_best]:.2f}")

# ---- Figure: 2 (configs) x [schematic + 4 standoffs] -----------------------
from matplotlib.patches import Rectangle, Circle, Ellipse, FancyArrowPatch, Arc

fig = plt.figure(figsize=(24, 12))
fig.patch.set_facecolor("#111111")

n_d = len(STANDOFFS)
# col 0 = schematic; cols 1..n_d = irradiance maps at each standoff
gs_top = mgridspec.GridSpec(2, 1 + n_d, figure=fig,
                            left=0.04, right=0.97, top=0.90, bottom=0.36,
                            hspace=0.30, wspace=0.30,
                            width_ratios=[1.35] + [1.0]*n_d)

def dark_ax(ax):
    ax.set_facecolor("#0a0a0a")
    for sp in ax.spines.values(): sp.set_edgecolor("#444444")
    ax.tick_params(colors="#aaaaaa", labelsize=7)

# determine a common vmax per row so heatmaps are comparable within a config
vmax_by_cfg = {}
maps_cache  = {}
for ci, cfg in enumerate(("A", "B")):
    row_max = 0.0
    for d in STANDOFFS:
        best = next(r for r in results
                    if r["config"] == cfg and r["standoff_mm"] == d)
        _, irr, xs, ys, fw, fh, centres, sxe, sye, pkb = cov_for(
            cfg, best["lens_f_mm"], d,
            best["best_theta_deg"], best["best_offset_frac"], return_map=True)
        # measure peak only inside the FOV to scale the colormap
        XX, YY = np.meshgrid(xs, ys)
        in_fov = (np.abs(XX) <= fw/2) & (np.abs(YY) <= fh/2)
        row_max = max(row_max, float(irr[in_fov].max()))
        maps_cache[(cfg, d)] = (irr, xs, ys, fw, fh, centres, best)
    vmax_by_cfg[cfg] = row_max

# ---- helper: draw the config schematic in normalised "stage" coordinates ----
def draw_schematic(ax, cfg, accent_col):
    """Front view (x-y on target plane). FOV centred; circular lamp icons
    drawn at their mount positions; arrows show beam projection onto the FOV;
    dashed ellipses indicate where each beam lands on the target.
    Uses the median-standoff optimum (offset/tilt) for a representative geometry."""
    ax.set_facecolor("#0d0d0d")
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(accent_col); sp.set_linewidth(2.5)

    # representative geometry — use the middle standoff result
    rep = next(r for r in results if r["config"] == cfg and r["standoff_mm"] == 400)
    th     = np.radians(rep["best_theta_deg"])
    off_f  = rep["best_offset_frac"]

    # normalised "stage" coordinates (FOV is the reference 1.0 × 0.75 unit box)
    fov_w, fov_h = 1.0, 0.75
    half_w, half_h = fov_w/2, fov_h/2

    # FOV (camera footprint)
    ax.add_patch(Rectangle((-half_w, -half_h), fov_w, fov_h,
                            fc="#1a2d2a", ec="cyan", lw=2.0, zorder=3))
    # tiny camera icon at FOV centre
    ax.add_patch(Ellipse((0, 0), 0.18, 0.13, fc="#223344", ec="white",
                          lw=1.0, zorder=4))
    ax.text(0, 0, "CAM", ha="center", va="center", color="white",
             fontsize=7, fontweight="bold", zorder=5)

    # lamp mount + beam-landing centres in stage coords
    LAMP_R     = 0.07            # circular bulb radius for the icon
    MOUNT_GAP  = 0.12            # how far outside the FOV the lamp sits
    BEAM_R     = 0.30            # representative beam footprint radius on target

    if cfg == "A":
        h_mount = half_h + MOUNT_GAP
        off_h   = (fov_w/2) * off_f
        # 4 lamps: top-left, top-right, bot-left, bot-right
        lamps   = [(-off_h, +h_mount), (+off_h, +h_mount),
                   (-off_h, -h_mount), (+off_h, -h_mount)]
        # where each beam lands on the target (tilt brings it toward FOV centre)
        # for visual, scale the inward shift by tilt fraction of max
        shift   = (MOUNT_GAP + half_h) * np.sin(th)
        beams   = [(-off_h,        +h_mount - shift), (+off_h,        +h_mount - shift),
                   (-off_h,        -h_mount + shift), (+off_h,        -h_mount + shift)]
    else:  # cfg B — all bulbs at y = 0 (same horizontal line as the camera)
        inner_x = half_w + MOUNT_GAP
        pair_h  = half_w * off_f                       # within-pair x-spacing
        outer_x = inner_x + pair_h
        lamps   = [(-outer_x, 0), (-inner_x, 0),
                   (+inner_x, 0), (+outer_x, 0)]
        shift   = (MOUNT_GAP + half_w) * np.sin(th)
        beams   = [(-outer_x + shift, 0), (-inner_x + shift, 0),
                   (+inner_x - shift, 0), (+outer_x - shift, 0)]

    # draw beam footprints (dashed ellipses on target)
    for bx, by in beams:
        ax.add_patch(Ellipse((bx, by), 2*BEAM_R, 2*BEAM_R*0.85,
                              fc=accent_col, alpha=0.16, ec=accent_col,
                              lw=1.0, ls="--", zorder=2))

    # draw lamp icons + arrows + tilt angle annotation
    for (lx, ly), (bx, by) in zip(lamps, beams):
        # circular bulb
        ax.add_patch(Circle((lx, ly), LAMP_R, fc="#1a2a3a", ec=accent_col,
                             lw=2.0, zorder=5))
        # filament glow (just a small bright dot since lamp is circular)
        ax.add_patch(Circle((lx, ly), LAMP_R*0.35, fc="#fff2a0",
                             ec="none", alpha=0.95, zorder=6))
        # beam-axis arrow from lamp toward the FOV (toward beam-landing point)
        ax.add_patch(FancyArrowPatch((lx, ly), (bx, by),
                                       arrowstyle="-|>", color=accent_col,
                                       mutation_scale=14, lw=1.8, zorder=4))
        # tilt arc indicator near the lamp
        # axis from lamp straight toward FOV centre (for reference)
        ang_to_centre = np.degrees(np.arctan2(-ly, -lx))
        ang_to_beam   = np.degrees(np.arctan2(by - ly, bx - lx))
        arc_r         = LAMP_R * 1.9
        a1, a2 = min(ang_to_centre, ang_to_beam), max(ang_to_centre, ang_to_beam)
        if abs(a2 - a1) > 1:
            ax.add_patch(Arc((lx, ly), arc_r*2, arc_r*2,
                              theta1=a1, theta2=a2,
                              color="white", lw=1.2, alpha=0.6, zorder=6))

    # title and annotations
    title = ("Config A  —  2 lamps above + 2 below FOV"
              if cfg == "A" else
              "Config B  —  2 lamps left + 2 right  (side-by-side on each side)")
    ax.set_title(title, color=accent_col, fontsize=11, fontweight="bold", pad=8)

    # legend-ish annotations along the bottom
    annot = (f"Representative geometry @ d = 400 mm:\n"
              f"   tilt  θ = {rep['best_theta_deg']:.0f}°\n"
              f"   offset = {rep['best_offset_frac']:.2f} × FOV\n"
              f"Arrows = beam axis on target\n"
              f"Dashed ellipses = beam footprint")
    ax.text(0.02, -1.0, annot, transform=ax.transAxes,
             color="#cccccc", fontsize=8, va="top", family="monospace")

    # plot limits — make sure all lamps and beams fit comfortably
    margin = 0.20
    x_extents = [abs(lx) for (lx, _) in lamps] + [half_w]
    y_extents = [abs(ly) for (_, ly) in lamps] + [half_h]
    xmax = max(x_extents) + LAMP_R + margin
    ymax = max(y_extents) + LAMP_R + margin
    ax.set_xlim(-xmax, +xmax)
    ax.set_ylim(-ymax - 0.30, +ymax)

# ---- draw row 0 (Cfg A) + row 1 (Cfg B) -------------------------------------
config_cols = {"A": "#ff6b6b", "B": "#4d96ff"}

for ci, cfg in enumerate(("A", "B")):
    # col 0: schematic
    ax_sch = fig.add_subplot(gs_top[ci, 0])
    draw_schematic(ax_sch, cfg, config_cols[cfg])

    # cols 1..n_d: irradiance maps
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
            inside = (xs[0] <= cx <= xs[-1]) and (ys[0] <= cy <= ys[-1])
            if inside:
                ax.plot(cx, cy, "+", color="white", ms=12, mew=2, zorder=5)

        ax.set_title(
            f"d = {d} mm  |  FOV {fw:.0f}×{fh:.0f} mm\n"
            f"tilt={best['best_theta_deg']:.0f}°  off={best['best_offset_frac']:.2f}  "
            f"-> CoV={best['min_cov_pct']:.2f}%",
            color="white", fontsize=8.5, pad=4)
        ax.set_xlabel("x (mm)", color="#aaaaaa", fontsize=8)

        if di == n_d - 1:
            cb = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.045)
            cb.ax.tick_params(labelsize=7, colors="white")
            cb.set_label("Irradiance (relative)", color="white", fontsize=8)

# ── bottom: CoV comparison line / bar chart ─────────────────────────────────
gs_bot = mgridspec.GridSpec(1, 2, figure=fig,
                            left=0.07, right=0.97, top=0.30, bottom=0.06,
                            wspace=0.25)

# left: CoV vs standoff for A and B
ax_cov = fig.add_subplot(gs_bot[0])
dark_ax(ax_cov)
ax_cov.set_facecolor("#1a1a1a")
markers = {"A": "o", "B": "s"}
config_cols = {"A": "#ff6b6b", "B": "#4d96ff"}
for cfg in ("A", "B"):
    rows = sorted([r for r in results if r["config"] == cfg],
                   key=lambda r: r["standoff_mm"])
    xs_ = [r["standoff_mm"] for r in rows]
    ys_ = [r["min_cov_pct"]  for r in rows]
    ax_cov.plot(xs_, ys_, marker=markers[cfg], color=config_cols[cfg],
                 lw=2, ms=10, label=f"Config {cfg}")
    for x, y in zip(xs_, ys_):
        ax_cov.text(x, y + 0.25, f"{y:.2f}%", ha="center",
                     color=config_cols[cfg], fontsize=8, fontweight="bold")
ax_cov.axhline(5, color="lime", lw=1, ls=":", alpha=0.5)
ax_cov.text(STANDOFFS[-1], 5.2, "  5% reference", color="lime",
             fontsize=7.5, va="bottom")
ax_cov.set_xlabel("Standoff (mm)", color="#aaaaaa")
ax_cov.set_ylabel("Min CoV across (tilt, offset) sweep (%)", color="#aaaaaa")
ax_cov.set_title("CoV vs standoff — 13.1 mm lens fixed",
                  color="white", fontsize=10)
ax_cov.legend(fontsize=8.5, facecolor="#222222", labelcolor="white",
               edgecolor="#555555", loc="upper left")
ax_cov.set_xticks(STANDOFFS)

# right: optimum-parameter table
ax_t = fig.add_subplot(gs_bot[1])
ax_t.set_facecolor("#111111"); ax_t.axis("off")
table_data = [["d (mm)",
                "Cfg A tilt", "Cfg A offset", "Cfg A CoV",
                "Cfg B tilt", "Cfg B offset", "Cfg B CoV"]]
for d in STANDOFFS:
    a = next(r for r in results if r["config"] == "A" and r["standoff_mm"] == d)
    b = next(r for r in results if r["config"] == "B" and r["standoff_mm"] == d)
    table_data.append([
        f"{d}",
        f"{a['best_theta_deg']:.0f}°",  f"{a['best_offset_frac']:.2f}",
        f"{a['min_cov_pct']:.2f}%",
        f"{b['best_theta_deg']:.0f}°",  f"{b['best_offset_frac']:.2f}",
        f"{b['min_cov_pct']:.2f}%",
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
        # highlight whichever config has lower CoV in this row
        a_cov = float(row[3].rstrip("%"))
        b_cov = float(row[6].rstrip("%"))
        if a_cov < b_cov:
            col_for[1] = col_for[2] = col_for[3] = "lime"
        else:
            col_for[4] = col_for[5] = col_for[6] = "lime"
    for ci, (val, c) in enumerate(zip(row, col_for)):
        ax_t.text(col_x[ci], y, val, transform=ax_t.transAxes,
                   color=c, fontsize=9.5, fontweight=fw_, va="top")
ax_t.set_title("Optimum (tilt, offset) per (config, standoff) — green = lower CoV",
                color="white", fontsize=10, pad=4)

fig.suptitle(
    "Stage 3 — Optimum lamp configurations for the 13.1 mm lens (Cranfield setup)\n"
    "Config A: 2 lamps above + 2 below FOV    |    Config B: 2 lamps left + 2 right (side-by-side per side)",
    color="white", fontsize=13, y=0.97)

fig.savefig(OUTPUT_PNG, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"\nSaved figure: {OUTPUT_PNG}")

# ---- JSON output ------------------------------------------------------------
overall_best = min(results, key=lambda r: r["min_cov_pct"])
out = {
    "sweep_settings": {
        "theta_degs":         THETA_DEGS.tolist(),
        "offset_fracs":       OFFSET_FRAC.tolist(),
        "lamp_clearance_mm":  LAMP_CLEARANCE_MM,
        "grid_px":            GRID_PX,
    },
    "beam_source":       BEAM_JSON,
    "lens_sweep_source": LENS_JSON,
    "results":           results,
    "overall_best":      {k: v for k, v in overall_best.items() if k != "sweep_cov_pct"},
}
with open(OUTPUT_JSON, "w") as f:
    json.dump(out, f, indent=2)
print(f"Saved JSON:   {OUTPUT_JSON}")

print(f"\n=== OVERALL BEST ===")
print(f"  Config:    {overall_best['config']}")
print(f"  Lens:      {overall_best['lens_label']}")
print(f"  Standoff:  {overall_best['standoff_mm']} mm")
print(f"  Tilt:      {overall_best['best_theta_deg']:.0f} deg")
print(f"  Offset:    {overall_best['best_offset_frac']:.2f} x FOV")
print(f"  CoV:       {overall_best['min_cov_pct']:.2f}%")
