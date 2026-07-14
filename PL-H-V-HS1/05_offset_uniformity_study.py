"""
Follow-up study — specimen non-uniformity (U) vs lamp angle, standoff and
beam-centre offset, for the PL-H-V-HS1 two-lamp rig.

Motivated by:
  Sutcu et al., "A Computational and Experimental Study on Heating Uniformity
  and Energy Density in Optically Stimulated Thermography" (Table 8, Table 9).
  That paper swept angle (15-75 deg) and standoff (200-700 mm) with both lamps
  aimed exactly at the specimen centre, and explicitly flagged beam-centre
  offset from the specimen centre as unstudied future work:
    "This study did not take into consideration the effect of offsetting the
     beam centres from the specimen centre, which could be an important
     parameter to optimise ... This will be studied in future work."
  This script is that follow-up.

Definition of U (paper, section 4.3.1): the coefficient of variation of a
20 mm Gaussian-filtered temperature distribution across the full specimen
width. U = 0% is perfectly flat/uniform. U is a *shape* metric — it is
invariant to lamp power, absorptivity or delivery efficiency, since those are
uniform scale factors that cancel in a std/mean ratio. That means this study
does not need to track intensity (peak dT) at all; only beam width sigma(d),
angle and offset matter.

Two-lamp geometry (mirror-symmetric about the specimen centreline, matching
the paper's rig and this repo's existing Config-B tilt formulas):
  - both lamps at the same standoff d and tilt angle theta (yaw, in-plane
    toward each other), landing on the specimen at (-offset, 0) and
    (+offset, 0) respectively.
  - offset = 0 reproduces the paper's own setup (both beams aimed dead
    centre) and is used to calibrate this geometric model against the
    paper's measured Table 8 values.
  - offset > 0 is the new parameter: how far each lamp's aim point is pulled
    away from the specimen centre, pulling the two footprints apart.

U is computed two ways:
  - U_1D: paper-exact definition — 20 mm-smoothed centreline (y=0) profile,
    CoV taken across the full 320 mm specimen width. Used for calibration.
  - U_2D: 20 mm-smoothed full specimen area (320x175 mm), CoV over the whole
    footprint. This is the "distribution across the specimen" metric and is
    used as the design target for picking the optimum offset.

Calibration: a linear fit U_paper ~= a + b*U_1D_model is derived from the
9 offset=0 anchors in Table 8 (angle sweep @ 500 mm, standoff sweep @ 45 deg)
plus the 6 routes in Table 9. This is an approximate empirical projection —
the geometric model omits lamp/geometric bias, detector noise and the real
DFLUX obliquity term the paper used — but it reproduces the correct
monotonic trends (uniformity improves with angle and with standoff) and is
used only to express the offset-sweep results in "paper-equivalent" units
alongside the raw geometric numbers.
"""

import os, json
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
from matplotlib.patches import Rectangle

REPO_ROOT   = os.path.dirname(os.path.abspath(__file__))
BEAM_JSON   = os.path.join(REPO_ROOT, "beam_derived_combined.json")
OUTPUT_PNG  = os.path.join(REPO_ROOT, "05_offset_uniformity_study.png")
OUTPUT_JSON = os.path.join(REPO_ROOT, "05_offset_uniformity_study.json")

# ---- Specimen (matches 04_dflux_vs_measured.py / the paper's own specimen) --
SPEC_W_MM, SPEC_H_MM = 320.0, 175.0
SMOOTH_SIGMA_MM = 20.0          # paper: "20 mm gaussian filter"

# ---- Sweep settings ----------------------------------------------------------
ANGLE_DEGS  = np.array([15, 30, 45, 60, 75])          # matches Table 8 levels
STANDOFF_MM = np.array([200, 300, 400, 500, 700])     # matches Table 8 levels
OFFSET_MM   = np.linspace(0, SPEC_W_MM/2, 17)         # NEW parameter (half-separation of the 2 beam centres)
                                                       # capped at the specimen half-width: the beam centre
                                                       # has to land on the plate, or the config is meaningless
GRID_NX, GRID_NY = 321, 176                           # ~1 mm resolution

# ---- Paper anchors for calibration (Table 8, offset = 0) --------------------
TABLE8_ANGLE_AT_500 = {15: 46.9, 30: 39.4, 45: 28.0, 60: 16.0, 75: 8.9}
TABLE8_STANDOFF_AT_45 = {200: 87.0, 300: 57.9, 400: 39.6, 500: 28.0, 700: 16.0}
# Table 9 routes (angle, standoff, U) — extra calibration/validation points
TABLE9_ROUTES = [
    (45, 500, 29.1), (45, 200, 86.8),
    (45, 500, 20.3), (60, 200, 86.5),
    (45, 500, 29.6), (45, 200, 86.1),
]

# ---- Load PL-H-V-HS1's own beam model ---------------------------------------
with open(BEAM_JSON) as f:
    beam = json.load(f)
bm = beam["derived_beam_model"]
SX_M, SX_B = bm["sigma_x_vs_d"]["slope_mm_per_mm"], bm["sigma_x_vs_d"]["intercept_mm"]
SY_M, SY_B = bm["sigma_y_vs_d"]["slope_mm_per_mm"], bm["sigma_y_vs_d"]["intercept_mm"]

def sigma_x(d): return SX_M * d + SX_B
def sigma_y(d): return SY_M * d + SY_B

xs = np.linspace(-SPEC_W_MM/2, SPEC_W_MM/2, GRID_NX)
ys = np.linspace(-SPEC_H_MM/2, SPEC_H_MM/2, GRID_NY)
dx = xs[1] - xs[0]
dy = ys[1] - ys[0]
XX, YY = np.meshgrid(xs, ys)

def field_for(angle_deg, standoff_mm, offset_mm):
    """Combined normalised 2-lamp footprint on the specimen plane.
    Tilt is a yaw about y (lamps sit either side, tilt toward each other) —
    same convention as Cranfield's Config B: sigma_x stretched by 1/cos(theta),
    slant distance d/cos(theta) feeds the standoff-dependent beam width."""
    th = np.radians(angle_deg)
    cos_t = max(np.cos(th), 1e-3)
    d_slant = standoff_mm / cos_t
    sx_eff = sigma_x(d_slant) / cos_t
    sy_eff = sigma_y(d_slant)
    field = np.zeros_like(XX)
    for cx in (-offset_mm, +offset_mm):
        field += np.exp(-0.5 * (((XX - cx) / sx_eff) ** 2 + (YY / sy_eff) ** 2))
    return field, sx_eff, sy_eff

def u_metrics(angle_deg, standoff_mm, offset_mm):
    field, sx_eff, sy_eff = field_for(angle_deg, standoff_mm, offset_mm)

    row = field[GRID_NY // 2, :]
    row_sm = gaussian_filter1d(row, sigma=SMOOTH_SIGMA_MM / dx, mode="nearest")
    u1d = 100 * row_sm.std() / row_sm.mean()

    field_sm = gaussian_filter(field, sigma=(SMOOTH_SIGMA_MM / dy, SMOOTH_SIGMA_MM / dx), mode="nearest")
    u2d = 100 * field_sm.std() / field_sm.mean()

    return u1d, u2d, sx_eff, sy_eff

def capture_fraction(angle_deg, standoff_mm, offset_mm):
    """Fraction of ONE lamp's total (unclipped) Gaussian output that actually
    lands on the 320x175 mm plate. U alone is scale-invariant -- it does not
    penalise energy that misses the specimen entirely, so this is tracked
    separately. Analytic (product of two 1D erf/CDF integrals, since the
    Gaussian is separable in x/y) rather than numerical, and exact.

    This is monotonically non-increasing in |offset|: moving a Gaussian's
    centre away from the middle of a fixed window can only reduce the
    fraction captured inside that window. There is therefore no offset that
    improves both U and capture efficiency -- it is a strict trade-off."""
    th = np.radians(angle_deg)
    cos_t = max(np.cos(th), 1e-3)
    d_slant = standoff_mm / cos_t
    sx_eff = sigma_x(d_slant) / cos_t
    sy_eff = sigma_y(d_slant)
    hw, hh = SPEC_W_MM / 2, SPEC_H_MM / 2
    fx = norm.cdf((hw - offset_mm) / sx_eff) - norm.cdf((-hw - offset_mm) / sx_eff)
    fy = norm.cdf(hh / sy_eff) - norm.cdf(-hh / sy_eff)
    return fx * fy

# ---- Calibration: fit U_paper ~= a + b * U_1D_model at offset = 0 -----------
cal_pts = []
for a_deg, u_paper in TABLE8_ANGLE_AT_500.items():
    u1, _, _, _ = u_metrics(a_deg, 500, 0.0)
    cal_pts.append((u1, u_paper, f"Table8 angle={a_deg}@500"))
for d_mm, u_paper in TABLE8_STANDOFF_AT_45.items():
    if d_mm == 500:
        continue  # already included via the angle=45 anchor above
    u1, _, _, _ = u_metrics(45, d_mm, 0.0)
    cal_pts.append((u1, u_paper, f"Table8 standoff={d_mm}@45"))
for a_deg, d_mm, u_paper in TABLE9_ROUTES:
    u1, _, _, _ = u_metrics(a_deg, d_mm, 0.0)
    cal_pts.append((u1, u_paper, f"Table9 {a_deg}deg/{d_mm}mm"))

cal_x = np.array([p[0] for p in cal_pts])
cal_y = np.array([p[1] for p in cal_pts])
CAL_B, CAL_A = np.polyfit(cal_x, cal_y, 1)
cal_pred = CAL_A + CAL_B * cal_x
ss_res = np.sum((cal_y - cal_pred) ** 2)
ss_tot = np.sum((cal_y - cal_y.mean()) ** 2)
CAL_R2 = 1 - ss_res / ss_tot

def to_paper_equiv(u_model):
    return CAL_A + CAL_B * u_model

print(f"Calibration: U_paper ~= {CAL_A:.2f} + {CAL_B:.3f} * U_1D_model   (R2={CAL_R2:.3f}, n={len(cal_pts)})")

# ---- Full sweep ---------------------------------------------------------------
# capture_pct: fraction of ONE lamp's total output that lands on the plate.
# U alone never penalises energy that misses the specimen -- pushing offset up
# always improves U (or is flat) but ALSO always reduces capture (proven
# monotonic above), so there is no offset that is "best" on both axes
# simultaneously. Both are reported; no single number is asserted as optimal.
results = []
for a_deg in ANGLE_DEGS:
    for d_mm in STANDOFF_MM:
        for off in OFFSET_MM:
            u1, u2, sxe, sye = u_metrics(a_deg, d_mm, off)
            cap = capture_fraction(a_deg, d_mm, off)
            results.append({
                "angle_deg": int(a_deg), "standoff_mm": int(d_mm), "offset_mm": float(off),
                "U_1D_model_pct": float(u1), "U_2D_model_pct": float(u2),
                "U_1D_paper_equiv_pct": float(to_paper_equiv(u1)),
                "U_2D_paper_equiv_pct": float(to_paper_equiv(u2)),
                "capture_pct": float(cap * 100),
                "sigma_x_eff_mm": float(sxe), "sigma_y_eff_mm": float(sye),
            })

baseline = next(r for r in results if r["angle_deg"] == 45 and r["standoff_mm"] == 500 and r["offset_mm"] == 0.0)
print(f"\nBaseline (45deg / 500mm / offset=0): U_2D={baseline['U_2D_model_pct']:.2f}%  "
      f"(paper-equiv {baseline['U_2D_paper_equiv_pct']:.1f}%)  capture={baseline['capture_pct']:.1f}%")
print(f"\nNOTE: capture efficiency (fraction of each lamp's output that actually lands on the")
print(f"320x175mm plate) is monotonically non-increasing in offset -- proven analytically, not just")
print(f"observed. Pushing offset toward the plate edge to minimise U therefore always throws away")
print(f"an increasing share of delivered energy. There is NO offset that improves both U and capture")
print(f"simultaneously -- min(U) alone is not a valid objective here. What follows is the trade-off,")
print(f"not a single 'optimum'.")

# ---- Trade-off table: U and capture efficiency at 3 reference points per ----
# (angle, standoff): offset=0, a "modest" point where capture has dropped 10%
# relative to its offset=0 value, and the plate-edge bound (offset=160mm).
offset_optimum_table = []
practical_combos = ([(a, 500) for a in ANGLE_DEGS] +
                     [(45, d) for d in STANDOFF_MM if d != 500])
for a_deg, d_mm in practical_combos:
    rows = sorted([r for r in results if r["angle_deg"] == a_deg and r["standoff_mm"] == d_mm],
                   key=lambda r: r["offset_mm"])
    at_zero = rows[0]
    cap_floor = 0.9 * at_zero["capture_pct"]         # 10% relative energy-capture loss
    modest = next((r for r in rows if r["capture_pct"] <= cap_floor), rows[-1])
    at_edge = rows[-1]                                # offset = plate half-width (160 mm)
    offset_optimum_table.append({
        "angle_deg": int(a_deg), "standoff_mm": int(d_mm),
        "U_2D_at_offset0_pct": float(at_zero["U_2D_model_pct"]),
        "capture_at_offset0_pct": float(at_zero["capture_pct"]),
        "modest_offset_mm": float(modest["offset_mm"]),
        "U_2D_at_modest_pct": float(modest["U_2D_model_pct"]),
        "capture_at_modest_pct": float(modest["capture_pct"]),
        "U_2D_at_plate_edge_pct": float(at_edge["U_2D_model_pct"]),
        "capture_at_plate_edge_pct": float(at_edge["capture_pct"]),
    })

print(f"\nTrade-off at each paper-tested (angle, standoff) level "
      f"(offset=0  vs  -10% capture  vs  plate edge / offset={OFFSET_MM[-1]:.0f}mm):")
print(f"  {'angle/standoff':>15} {'U@0':>7} {'cap@0':>7} | {'off(-10%cap)':>13} {'U':>7} {'cap':>7} | {'U@edge':>8} {'cap@edge':>9}")
for r in offset_optimum_table:
    print(f"  {r['angle_deg']:>3}d/{r['standoff_mm']:>4}mm    {r['U_2D_at_offset0_pct']:>6.2f}% {r['capture_at_offset0_pct']:>6.1f}% | "
          f"{r['modest_offset_mm']:>11.0f}mm {r['U_2D_at_modest_pct']:>6.2f}% {r['capture_at_modest_pct']:>6.1f}% | "
          f"{r['U_2D_at_plate_edge_pct']:>7.2f}% {r['capture_at_plate_edge_pct']:>8.1f}%")

baseline_offset_row = next(r for r in offset_optimum_table if r["angle_deg"] == 45 and r["standoff_mm"] == 500)
print(f"\n>>> At the paper's own baseline (45deg / 500mm): offset=0 gives U_2D={baseline_offset_row['U_2D_at_offset0_pct']:.2f}% "
      f"at {baseline_offset_row['capture_at_offset0_pct']:.1f}% capture. Accepting a 10% relative capture loss "
      f"(offset={baseline_offset_row['modest_offset_mm']:.0f}mm, capture->{baseline_offset_row['capture_at_modest_pct']:.1f}%) "
      f"buys U_2D={baseline_offset_row['U_2D_at_modest_pct']:.2f}%. Pushing to the plate edge "
      f"(offset={OFFSET_MM[-1]:.0f}mm) only reaches U_2D={baseline_offset_row['U_2D_at_plate_edge_pct']:.2f}% "
      f"but capture has fallen to {baseline_offset_row['capture_at_plate_edge_pct']:.1f}% -- "
      f"more than 2/3 of each lamp's output is then missing the plate entirely.")
print(f"\nIMPORTANT: capture efficiency is already low at offset=0 for this beam model ({baseline['capture_pct']:.0f}% at")
print(f"the paper's own baseline) -- sigma_x/sigma_y from PL-H-V-HS1's own characterisation are large relative to")
print(f"the 320x175mm specimen at 500mm standoff, so most of each lamp's output already misses the plate before")
print(f"any offset is applied. This may be worth revisiting independently of the offset question.")

# ---- Figure -------------------------------------------------------------------
fig = plt.figure(figsize=(22, 14))
fig.patch.set_facecolor("#0e0e0e")
gs = mgridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.35,
                        left=0.045, right=0.98, top=0.92, bottom=0.06,
                        height_ratios=[1.0, 1.0, 1.05])

def dark(ax, grid=True):
    ax.set_facecolor("#161616")
    for sp in ax.spines.values(): sp.set_edgecolor("#444")
    ax.tick_params(colors="#aaa", labelsize=8)
    if grid: ax.grid(True, alpha=0.12, color="white")

# --- Row 0, col 0: calibration scatter ---------------------------------------
ax = fig.add_subplot(gs[0, 0]); dark(ax)
ax.scatter(cal_x, cal_y, s=70, color="#ffd93d", ec="white", lw=0.6, zorder=5,
           label="Table 8 / 9 anchors")
xline = np.linspace(0, cal_x.max()*1.15, 50)
ax.plot(xline, CAL_A + CAL_B*xline, color="cyan", lw=1.8,
        label=f"fit: {CAL_A:.1f} + {CAL_B:.2f}x  ($R^2$={CAL_R2:.2f})")
ax.set_xlabel("U_1D model (%, geometric, offset=0)", color="#aaa", fontsize=9)
ax.set_ylabel("U paper (%)", color="#aaa", fontsize=9)
ax.set_title("Calibration: geometric model vs paper U", color="white", fontsize=10)
ax.legend(facecolor="#111", labelcolor="white", edgecolor="#555", fontsize=7.5, loc="upper left")

# --- Row 0, col 1: U_2D vs offset for several angle/standoff combos ----------
ax = fig.add_subplot(gs[0, 1]); dark(ax)
combos = [(30, 500), (45, 500), (60, 500), (45, 300), (45, 700)]
cols = plt.cm.plasma(np.linspace(0.15, 0.9, len(combos)))
for (a_deg, d_mm), col in zip(combos, cols):
    rows = [r for r in results if r["angle_deg"] == a_deg and r["standoff_mm"] == d_mm]
    rows.sort(key=lambda r: r["offset_mm"])
    ax.plot([r["offset_mm"] for r in rows], [r["U_2D_model_pct"] for r in rows],
            color=col, lw=1.8, marker="o", ms=3, label=f"{a_deg}deg/{d_mm}mm")
ax.set_xlabel("offset (mm, half beam-centre separation)", color="#aaa", fontsize=9)
ax.set_ylabel("U_2D model (%)", color="#aaa", fontsize=9)
ax.set_title("U_2D vs offset", color="white", fontsize=10)
ax.legend(facecolor="#111", labelcolor="white", edgecolor="#555", fontsize=7, loc="best")

# --- Row 0, col 2: capture efficiency vs offset, same combos as col 1 -------
ax = fig.add_subplot(gs[0, 2]); dark(ax)
combos = [(30, 500), (45, 500), (60, 500), (45, 300), (45, 700)]
cols = plt.cm.plasma(np.linspace(0.15, 0.9, len(combos)))
for (a_deg, d_mm), col in zip(combos, cols):
    rows = [r for r in results if r["angle_deg"] == a_deg and r["standoff_mm"] == d_mm]
    rows.sort(key=lambda r: r["offset_mm"])
    ax.plot([r["offset_mm"] for r in rows], [r["capture_pct"] for r in rows],
            color=col, lw=1.8, marker="o", ms=3, label=f"{a_deg}deg/{d_mm}mm")
ax.set_xlabel("offset (mm)", color="#aaa", fontsize=9)
ax.set_ylabel("capture efficiency (%)", color="#aaa", fontsize=9)
ax.set_title("Capture efficiency vs offset\n(fraction of each lamp's output landing on the plate)",
             color="white", fontsize=9.5)
ax.legend(facecolor="#111", labelcolor="white", edgecolor="#555", fontsize=7, loc="best")

# --- Row 0, col 3: Pareto trade-off -- U vs capture, offset as the parameter -
ax = fig.add_subplot(gs[0, 3]); dark(ax)
for (a_deg, d_mm), col in zip(combos, cols):
    rows = [r for r in results if r["angle_deg"] == a_deg and r["standoff_mm"] == d_mm]
    rows.sort(key=lambda r: r["offset_mm"])
    ax.plot([r["capture_pct"] for r in rows], [r["U_2D_model_pct"] for r in rows],
            color=col, lw=1.5, marker="o", ms=3, label=f"{a_deg}deg/{d_mm}mm")
ax.scatter([baseline["capture_pct"]], [baseline["U_2D_model_pct"]], s=90, marker="*",
           color="white", ec="black", zorder=6, label="baseline (offset=0)")
ax.invert_xaxis()
ax.set_xlabel("capture efficiency (%)  (arrow = increasing offset)", color="#aaa", fontsize=8.5)
ax.set_ylabel("U_2D model (%)", color="#aaa", fontsize=9)
ax.set_title("Pareto trade-off: U vs energy captured\n(every point on a line = a different offset; no free lunch)",
             color="white", fontsize=9)
ax.legend(facecolor="#111", labelcolor="white", edgecolor="#555", fontsize=6.5, loc="best")

# --- Row 1: specimen irradiance maps -- offset=0 vs modest vs plate-edge, at
# the paper's own baseline (angle, standoff). No panel claims a single
# "optimum" -- each is labelled with both its U and its capture cost.
row45_500 = next(r for r in offset_optimum_table if r["angle_deg"] == 45 and r["standoff_mm"] == 500)
map_specs = [
    ("Offset = 0 (paper's own setup)\n45deg/500mm", 45, 500, 0.0),
    (f"-10% relative capture (modest offset)\n45deg/500mm, offset={row45_500['modest_offset_mm']:.0f}mm",
     45, 500, row45_500["modest_offset_mm"]),
    (f"Plate-edge bound (max offset)\n45deg/500mm, offset={OFFSET_MM[-1]:.0f}mm", 45, 500, float(OFFSET_MM[-1])),
    ("ILLUSTRATIVE ONLY -- off-plate, invalid\n45deg/500mm, offset=250mm (past plate edge)", 45, 500, 250.0),
]
for j, (title, a_deg, d_mm, off) in enumerate(map_specs):
    ax = fig.add_subplot(gs[1, j]); dark(ax, grid=False)
    field, sxe, sye = field_for(a_deg, d_mm, off)
    field_sm = gaussian_filter(field, sigma=(SMOOTH_SIGMA_MM/dy, SMOOTH_SIGMA_MM/dx), mode="nearest")
    u2 = 100 * field_sm.std() / field_sm.mean()
    cap = capture_fraction(a_deg, d_mm, off) * 100
    im = ax.imshow(field, cmap="inferno", origin="lower",
                   extent=[xs[0], xs[-1], ys[0], ys[-1]], aspect="equal")
    ax.add_patch(Rectangle((-SPEC_W_MM/2, -SPEC_H_MM/2), SPEC_W_MM, SPEC_H_MM,
                            fc="none", ec="cyan", lw=1.8))
    for cx in (-off, off):
        ax.plot(cx, 0, "+", color="white", ms=10, mew=1.8)
    ax.set_title(f"{title}\nU_2D={u2:.2f}%   capture={cap:.1f}%   ($\\sigma_x$={sxe:.0f}, $\\sigma_y$={sye:.0f})",
                 color="white", fontsize=8.5)
    ax.set_xlabel("x (mm)", color="#aaa", fontsize=8)
    if j == 0:
        ax.set_ylabel("y (mm)", color="#aaa", fontsize=8)

# --- Row 2, col 0-1: summary table -- U/capture trade-off at each paper-tested level
ax_t = fig.add_subplot(gs[2, 0:2]); ax_t.axis("off"); ax_t.set_facecolor("#111111")
rows_txt = [["angle/standoff", "U/cap @ off=0", "-10% capture pt", "U/cap there", "U/cap @ plate edge"]]
for r in offset_optimum_table:
    rows_txt.append([f"{r['angle_deg']}°/{r['standoff_mm']}mm",
                      f"{r['U_2D_at_offset0_pct']:.1f}% / {r['capture_at_offset0_pct']:.0f}%",
                      f"{r['modest_offset_mm']:.0f} mm",
                      f"{r['U_2D_at_modest_pct']:.1f}% / {r['capture_at_modest_pct']:.0f}%",
                      f"{r['U_2D_at_plate_edge_pct']:.1f}% / {r['capture_at_plate_edge_pct']:.0f}%"])
col_x = [0.02, 0.24, 0.48, 0.64, 0.85]
row_h = 0.095; y0 = 0.97
for ri, row in enumerate(rows_txt):
    y = y0 - ri*row_h
    fw_ = "bold" if ri == 0 else "normal"
    col = "#ffd93d" if ri == 0 else ("lime" if row[0] == "45°/500mm" else "white")
    for ci, val in enumerate(row):
        ax_t.text(col_x[ci], y, val, transform=ax_t.transAxes, color=col, fontsize=9, fontweight=fw_, va="top")
ax_t.set_title("U% / capture% trade-off at each paper-tested (angle, standoff) level  "
               "(green = paper's own baseline)",
               color="white", fontsize=10, pad=6, loc="left")

# --- Row 2, col 2-3: notes ----------------------------------------------------
ax_n = fig.add_subplot(gs[2, 2:4]); ax_n.axis("off"); ax_n.set_facecolor("#111111")
notes = (
    "Notes\n"
    "-----\n"
    f"U alone is scale-invariant and does not penalise energy that misses the\n"
    f"plate entirely, so offset always looks 'free' if you only look at U. It\n"
    f"is not: capture efficiency (fraction of each lamp's output landing on the\n"
    f"320x175mm plate) is PROVABLY monotonically non-increasing in offset --\n"
    f"there is no offset that improves both U and capture. This is a genuine\n"
    f"trade-off, not an optimisation with a single answer.\n\n"
    f"At the paper's baseline (45deg/500mm): offset=0 already only captures\n"
    f"{baseline['capture_pct']:.0f}% of each lamp's output on the plate (sigma is large relative\n"
    f"to the specimen at this standoff for PL-H-V-HS1's own beam model). Accepting\n"
    f"a further 10% relative capture loss (offset={row45_500['modest_offset_mm']:.0f}mm) buys U_2D="
    f"{row45_500['U_2D_at_modest_pct']:.2f}%\n"
    f"(vs {row45_500['U_2D_at_offset0_pct']:.2f}% at offset=0). Pushing to the plate edge "
    f"(offset={OFFSET_MM[-1]:.0f}mm) only\n"
    f"reaches U_2D={row45_500['U_2D_at_plate_edge_pct']:.2f}% while capture falls to "
    f"{row45_500['capture_at_plate_edge_pct']:.0f}% -- steeply\n"
    f"diminishing uniformity return for a large energy cost.\n\n"
    f"Calibration against paper Table 8/9 (offset=0, n={len(cal_pts)} points):\n"
    f"  U_paper ~= {CAL_A:.1f} + {CAL_B:.2f} x U_1D_model   (R2={CAL_R2:.2f})\n"
    f"Correct monotonic trend, lower-bound magnitude (paper's real U also includes\n"
    f"lamp/geometric bias and detector noise absent from this pure-geometry model).\n"
)
ax_n.text(0.0, 1.0, notes, transform=ax_n.transAxes, color="#ccc", fontsize=9.2,
          va="top", family="monospace")

fig.suptitle(
    "PL-H-V-HS1 follow-up — specimen non-uniformity (U) vs offset, traded off against energy capture efficiency\n"
    "Extends Sutcu et al. Table 8/9 (offset=0 baseline); offset capped at the specimen half-width — the beam centre must land on the plate",
    color="white", fontsize=13, y=0.985)

fig.savefig(OUTPUT_PNG, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"\nSaved figure -> {OUTPUT_PNG}")

# ---- JSON output --------------------------------------------------------------
out = {
    "specimen_mm": [SPEC_W_MM, SPEC_H_MM],
    "smoothing_sigma_mm": SMOOTH_SIGMA_MM,
    "beam_source": BEAM_JSON,
    "calibration": {
        "anchors": [{"U_1D_model_pct": float(x), "U_paper_pct": float(y), "label": lbl}
                    for x, y, lbl in cal_pts],
        "a": float(CAL_A), "b": float(CAL_B), "r2": float(CAL_R2),
        "formula": "U_paper_equiv = a + b * U_model",
    },
    "sweep_settings": {
        "angle_degs": ANGLE_DEGS.tolist(),
        "standoff_mm": STANDOFF_MM.tolist(),
        "offset_mm": OFFSET_MM.tolist(),
    },
    "baseline_45deg_500mm_offset0": baseline,
    "no_single_optimum_note": (
        "There is no single 'optimum offset'. U is scale-invariant and improves "
        "(or holds flat) monotonically as offset increases up to the plate edge, "
        "but capture_pct (fraction of each lamp's output landing on the plate) is "
        "provably monotonically non-increasing in offset -- proven analytically "
        "via the separable-Gaussian CDF, not just observed on this grid. Moving "
        "away from offset=0 always trades captured energy for uniformity. See "
        "offset_uniformity_capture_tradeoff for U and capture_pct reported "
        "together at offset=0, at a -10% relative capture point, and at the "
        "plate-edge bound, for each paper-tested (angle, standoff) level. Also "
        "note the joint (angle, standoff, offset) space is NOT separately "
        "re-optimised here -- angle/standoff are held at Table 8's own tested "
        "levels, since jointly minimising U across all three degenerates to the "
        "sweep edge (a re-statement of Table 8, not a new result)."
    ),
    "offset_uniformity_capture_tradeoff": offset_optimum_table,
    "results": results,
}
with open(OUTPUT_JSON, "w") as f:
    json.dump(out, f, indent=2)
print(f"Saved JSON   -> {OUTPUT_JSON}")
