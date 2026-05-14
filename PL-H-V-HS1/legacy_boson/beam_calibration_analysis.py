"""
Shows how the Gaussian beam parameters were derived from the four
standoff-distance calibration sessions (500, 600, 700, 800mm).

For each session:
  - sigma_x, sigma_y are in pixels (from 2D Gaussian fit)
  - Converted to mm using camera geometry (18mm lens, HFOV=24.09deg)
  - Linear divergence fit through 600-800mm (500mm excluded: beam misaligned)
  - 700mm session chosen as reference -> sx_mm=108mm, sy_mm=59mm used in simulations
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUTPUT_PNG = config.BOSON_ROOT + r"\beam_calibration_analysis.png"

# ── camera geometry (18mm lens, Boson 640) ────────────────────────────────────
FOCAL_MM    = 18.0
SENSOR_W_PX = 640
SENSOR_H_PX = 512
PIXEL_PITCH = 0.012   # mm
SENSOR_W_MM = SENSOR_W_PX * PIXEL_PITCH   # 7.68mm
HFOV_DEG    = 2 * np.degrees(np.arctan(SENSOR_W_MM / (2 * FOCAL_MM)))  # 24.09deg

def mm_per_px(standoff_mm):
    return 2 * standoff_mm * np.tan(np.radians(HFOV_DEG / 2)) / SENSOR_W_PX

# ── load all sessions ─────────────────────────────────────────────────────────
SESSIONS = {
    500: config.BOSON_ROOT + r"\500 mm\summary.json",
    600: config.BOSON_ROOT + r"\600 mm\summary.json",
    700: config.BOSON_ROOT + r"\700 mm\summary.json",
    800: config.BOSON_ROOT + r"\800 mm\summary.json",
}

standoffs, sx_px, sy_px, sx_mm_list, sy_mm_list = [], [], [], [], []
thetas, aspects, amplitudes = [], [], []

for d, path in SESSIONS.items():
    with open(path) as f:
        s = json.load(f)["beam"]
    mpp = mm_per_px(d)
    standoffs.append(d)
    sx_px.append(s["sigma_x"])
    sy_px.append(s["sigma_y"])
    sx_mm_list.append(s["sigma_x"] * mpp)
    sy_mm_list.append(s["sigma_y"] * mpp)
    thetas.append(s["theta_deg"])
    aspects.append(s["aspect_ratio"])
    amplitudes.append(s["amplitude"])

standoffs   = np.array(standoffs,   float)
sx_px       = np.array(sx_px,       float)
sy_px       = np.array(sy_px,       float)
sx_mm_arr   = np.array(sx_mm_list,  float)
sy_mm_arr   = np.array(sy_mm_list,  float)
thetas      = np.array(thetas,      float)
aspects     = np.array(aspects,     float)

print(f"{'Standoff':>9}  {'mm/px':>7}  {'sx_px':>7}  {'sy_px':>7}  "
      f"{'sx_mm':>7}  {'sy_mm':>7}  {'theta':>7}  {'aspect':>7}")
print("-" * 72)
for i, d in enumerate(standoffs):
    mpp = mm_per_px(d)
    excl = "  <-- EXCLUDED (misaligned)" if d == 500 else ""
    print(f"{d:>9.0f}mm  {mpp:>7.4f}  {sx_px[i]:>7.1f}  {sy_px[i]:>7.1f}  "
          f"{sx_mm_arr[i]:>7.1f}  {sy_mm_arr[i]:>7.1f}  "
          f"{thetas[i]:>7.1f}  {aspects[i]:>7.3f}{excl}")

# ── linear divergence fit: 600-800mm only ────────────────────────────────────
fit_mask = standoffs >= 600
d_fit    = standoffs[fit_mask]

cx = np.polyfit(d_fit, sx_mm_arr[fit_mask], 1)   # sigma_x = cx[0]*d + cx[1]
cy = np.polyfit(d_fit, sy_mm_arr[fit_mask], 1)   # sigma_y = cy[0]*d + cy[1]

d_line = np.linspace(400, 900, 200)
sx_fit = np.polyval(cx, d_line)
sy_fit = np.polyval(cy, d_line)

# reference values from 700mm session (used in simulations)
REF_D      = 700
ref_sx_mm  = sx_mm_arr[standoffs == REF_D][0]
ref_sy_mm  = sy_mm_arr[standoffs == REF_D][0]

print(f"\nLinear fit (600-800mm):")
print(f"  sigma_x = {cx[0]*1000:.3f}*d/1000 + {cx[1]:.1f}  mm")
print(f"  sigma_y = {cy[0]*1000:.3f}*d/1000 + {cy[1]:.1f}  mm")
print(f"\nReference (700mm): sigma_x={ref_sx_mm:.1f}mm  sigma_y={ref_sy_mm:.1f}mm")
print(f"  FWHM_x={ref_sx_mm*2.355:.1f}mm  FWHM_y={ref_sy_mm*2.355:.1f}mm")
print(f"  Aspect ratio (sx/sy) = {ref_sx_mm/ref_sy_mm:.2f}")

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor("#111111")
gs  = plt.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                   left=0.07, right=0.97, top=0.91, bottom=0.06)

COL_X    = "#ffd93d"   # sigma_x colour
COL_Y    = "#4d96ff"   # sigma_y colour
COL_EXCL = "#888888"   # excluded point
COL_REF  = "lime"      # reference session

def style_ax(ax):
    ax.set_facecolor("#1a1a1a")
    for sp in ax.spines.values(): sp.set_edgecolor("#444444")
    ax.tick_params(colors="#aaaaaa", labelsize=9)
    ax.grid(True, alpha=0.15, color="white")

# ── Panel A: sigma in mm vs standoff ─────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, :2])
style_ax(ax_a)

# excluded point (500mm)
ax_a.scatter([500], [sx_mm_arr[0]], s=100, color=COL_EXCL, marker="x",
             linewidths=2.5, zorder=5, label="500mm excluded (beam misaligned)")
ax_a.scatter([500], [sy_mm_arr[0]], s=100, color=COL_EXCL, marker="x",
             linewidths=2.5, zorder=5)

# fitted points (600-800mm)
ax_a.scatter(standoffs[fit_mask], sx_mm_arr[fit_mask],
             s=120, color=COL_X, marker="o", zorder=6, label=f"σ_x measured")
ax_a.scatter(standoffs[fit_mask], sy_mm_arr[fit_mask],
             s=120, color=COL_Y, marker="s", zorder=6, label=f"σ_y measured")

# linear fits
ax_a.plot(d_line, sx_fit, color=COL_X, lw=2, ls="--", alpha=0.8,
          label=f"σ_x fit: {cx[0]*1000:+.2f}mm/m·d {cx[1]:+.0f}mm")
ax_a.plot(d_line, sy_fit, color=COL_Y, lw=2, ls="--", alpha=0.8,
          label=f"σ_y fit: {cy[0]*1000:+.2f}mm/m·d {cy[1]:+.0f}mm")

# reference marker
ax_a.axvline(REF_D, color=COL_REF, lw=1.5, ls=":", alpha=0.7)
ax_a.scatter([REF_D], [ref_sx_mm], s=200, color=COL_REF, marker="*", zorder=8)
ax_a.scatter([REF_D], [ref_sy_mm], s=200, color=COL_REF, marker="*", zorder=8)
ax_a.text(REF_D+8, ref_sx_mm+3, f"σ_x={ref_sx_mm:.0f}mm\n(reference)",
          color=COL_REF, fontsize=8)
ax_a.text(REF_D+8, ref_sy_mm-10, f"σ_y={ref_sy_mm:.0f}mm\n(reference)",
          color=COL_REF, fontsize=8)

# FWHM secondary axis
ax_a2 = ax_a.twinx()
ax_a2.set_ylabel("FWHM (mm)", color="#aaaaaa", fontsize=9)
ax_a2.tick_params(colors="#aaaaaa", labelsize=8)
fwhm_lim = np.array(ax_a.get_ylim()) * 2.355
ax_a2.set_ylim(fwhm_lim)

ax_a.set_xlabel("Standoff distance (mm)", color="#aaaaaa", fontsize=10)
ax_a.set_ylabel("Gaussian sigma (mm)", color="#aaaaaa", fontsize=10)
ax_a.set_title(
    "Beam sigma vs standoff  |  18mm lens  |  Converted from pixels using camera FOV\n"
    "Linear divergence fit through 600–800mm  (500mm excluded: θ=−24.8°, beam misaligned)",
    color="white", fontsize=10)
ax_a.legend(fontsize=8.5, facecolor="#222222", labelcolor="white",
            edgecolor="#555555", loc="upper right")
ax_a.set_xlim(400, 900)

# ── Panel B: beam rotation angle ─────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 2])
style_ax(ax_b)

bar_cols = [COL_EXCL if d == 500 else COL_REF if d == 700 else "#aaaaaa"
            for d in standoffs]
ax_b.bar(standoffs, thetas, width=40, color=bar_cols, alpha=0.8, zorder=3)
ax_b.axhline(0, color="white", lw=0.8, ls="--", alpha=0.5)
ax_b.set_xlabel("Standoff (mm)", color="#aaaaaa", fontsize=9)
ax_b.set_ylabel("Beam rotation angle θ (deg)", color="#aaaaaa", fontsize=9)
ax_b.set_title("Beam tilt per session\n(|θ| > 5° → excluded from fit)",
               color="white", fontsize=10)
ax_b.tick_params(colors="#aaaaaa", labelsize=9)
for i, (d, th) in enumerate(zip(standoffs, thetas)):
    ax_b.text(d, th + (1 if th >= 0 else -3), f"{th:.1f}°",
              ha="center", color="white", fontsize=9, fontweight="bold")
ax_b.add_patch(mpatches.FancyBboxPatch((470, -6), 380, 12,
               boxstyle="round,pad=5", fc="#1a3a1a", ec="lime",
               lw=1.5, ls="--", zorder=1, alpha=0.3))
ax_b.text(660, -7.5, "±5° acceptable", color="lime", fontsize=8, ha="center")

# ── Panel C: sigma_x in pixels vs standoff (raw fit) ─────────────────────────
ax_c = fig.add_subplot(gs[1, 0])
style_ax(ax_c)

ax_c.scatter([500], [sx_px[0]], s=100, color=COL_EXCL, marker="x",
             linewidths=2.5, zorder=5)
ax_c.scatter([500], [sy_px[0]], s=100, color=COL_EXCL, marker="x",
             linewidths=2.5, zorder=5)
ax_c.scatter(standoffs[fit_mask], sx_px[fit_mask],
             s=120, color=COL_X, marker="o", zorder=6, label="σ_x (px)")
ax_c.scatter(standoffs[fit_mask], sy_px[fit_mask],
             s=120, color=COL_Y, marker="s", zorder=6, label="σ_y (px)")

# fit in px space
mpp_arr = np.array([mm_per_px(d) for d in standoffs])
cx_px = np.polyfit(d_fit, sx_px[fit_mask], 1)
cy_px = np.polyfit(d_fit, sy_px[fit_mask], 1)
ax_c.plot(d_line, np.polyval(cx_px, d_line), color=COL_X, lw=2, ls="--", alpha=0.8)
ax_c.plot(d_line, np.polyval(cy_px, d_line), color=COL_Y, lw=2, ls="--", alpha=0.8)
ax_c.axvline(REF_D, color=COL_REF, lw=1.5, ls=":", alpha=0.7)
ax_c.set_xlabel("Standoff (mm)", color="#aaaaaa", fontsize=9)
ax_c.set_ylabel("Sigma (pixels)", color="#aaaaaa", fontsize=9)
ax_c.set_title("Raw pixel sigmas\n(decrease with standoff — beam fills fewer px)",
               color="white", fontsize=10)
ax_c.legend(fontsize=8, facecolor="#222222", labelcolor="white", edgecolor="#555555")
ax_c.set_xlim(400, 900)

# ── Panel D: aspect ratio vs standoff ────────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 1])
style_ax(ax_d)

ax_d.scatter([500], [aspects[0]], s=100, color=COL_EXCL, marker="x",
             linewidths=2.5, zorder=5, label="500mm (excluded)")
ax_d.scatter(standoffs[fit_mask], aspects[fit_mask],
             s=140, color="#ff6b6b", marker="D", zorder=6, label="sx/sy")
ax_d.axhline(np.mean(aspects[fit_mask]), color="white", lw=1.5, ls="--", alpha=0.6)
ax_d.text(860, np.mean(aspects[fit_mask])+0.03,
          f"mean={np.mean(aspects[fit_mask]):.2f}", color="white", fontsize=9)
ax_d.set_xlabel("Standoff (mm)", color="#aaaaaa", fontsize=9)
ax_d.set_ylabel("Aspect ratio σ_x / σ_y", color="#aaaaaa", fontsize=9)
ax_d.set_title("Beam aspect ratio vs standoff\n(consistent → beam shape is intrinsic, not artefact)",
               color="white", fontsize=10)
ax_d.legend(fontsize=8, facecolor="#222222", labelcolor="white", edgecolor="#555555")
ax_d.set_xlim(400, 900)
for i, (d, asp) in enumerate(zip(standoffs, aspects)):
    ax_d.text(d, asp+0.04, f"{asp:.2f}", ha="center",
              color=COL_EXCL if d == 500 else "white", fontsize=9)

# ── Panel E: summary diagram ─────────────────────────────────────────────────
ax_e = fig.add_subplot(gs[1, 2])
ax_e.set_facecolor("#0d0d0d")
ax_e.set_xlim(-200, 200); ax_e.set_ylim(-160, 160)
ax_e.set_aspect("equal"); ax_e.set_xticks([]); ax_e.set_yticks([])
for sp in ax_e.spines.values(): sp.set_edgecolor("#444444")

# beam ellipse at 700mm (free-space)
ax_e.add_patch(mpatches.Ellipse((0, 0), ref_sx_mm*2, ref_sy_mm*2,
               fc=COL_X, alpha=0.15, ec=COL_X, lw=2, ls="--"))
# FWHM ellipse
ax_e.add_patch(mpatches.Ellipse((0, 0), ref_sx_mm*2.355*2, ref_sy_mm*2.355*2,
               fc="none", ec="white", lw=1.5, ls=":", alpha=0.5))
# specimen overlay
ax_e.add_patch(mpatches.Rectangle((-160, -87.5), 320, 175,
               fc="#1a3a1a", ec="lime", lw=2, alpha=0.4))

# sigma arrows
ax_e.annotate("", xy=(ref_sx_mm, 0), xytext=(0, 0),
              arrowprops=dict(arrowstyle="<->", color=COL_X, lw=2))
ax_e.text(ref_sx_mm/2, 8, f"σ_x={ref_sx_mm:.0f}mm", ha="center",
          color=COL_X, fontsize=9, fontweight="bold")

ax_e.annotate("", xy=(0, ref_sy_mm), xytext=(0, 0),
              arrowprops=dict(arrowstyle="<->", color=COL_Y, lw=2))
ax_e.text(18, ref_sy_mm/2, f"σ_y={ref_sy_mm:.0f}mm", ha="left",
          color=COL_Y, fontsize=9, fontweight="bold")

ax_e.text(0, -130, "Specimen 320×175mm", ha="center", color="lime", fontsize=8)
ax_e.text(0, 130, f"Beam at 700mm reference\nFWHM {ref_sx_mm*2.355:.0f}×{ref_sy_mm*2.355:.0f}mm",
          ha="center", color="white", fontsize=8)
ax_e.set_title("Free-space beam used in simulation", color="white", fontsize=10)

fig.suptitle(
    "Beam Gaussian calibration  |  18mm lens, Boson 640  |  "
    "4 standoff sessions: 500 / 600 / 700 / 800mm\n"
    "σ_x and σ_y fitted from 2D rotated Gaussian on differential thermal frames  "
    "→  reference values from 700mm session",
    color="white", fontsize=11)

fig.savefig(OUTPUT_PNG, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"\nSaved: {OUTPUT_PNG}")
