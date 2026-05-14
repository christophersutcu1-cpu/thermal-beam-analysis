"""
Stage 2 — lens / standoff / FOV sweep.

Given the beam fit produced by Stage 1 (beam_derived_combined.json), sweep camera
lens choices and camera-to-target standoffs to see how the field of view changes
and how much of the beam is actually captured by the FOV.

Lenses sweep (FLIR A655sc catalogue, one wider + current + one narrower):
  7.5 mm focal length  →  80° HFOV  (FOL7)
 13.1 mm focal length  →  45° HFOV  (FOL18, the current Cranfield setup)
 24.6 mm focal length  →  25° HFOV  (FOL25)

Standoffs sweep:  300 / 400 / 500 / 750 mm

Outputs:
  05_lens_fov_sweep.json   structured (lens × standoff) grid, consumed by Stage 3
  05_lens_fov_sweep.png    visual comparison
"""

import os, json
import numpy as np
from math import erf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse

# ---- I/O --------------------------------------------------------------------
REPO_ROOT    = os.path.dirname(os.path.abspath(__file__))
INPUT_JSON   = os.path.join(REPO_ROOT, "beam_derived_combined.json")
OUTPUT_PNG   = os.path.join(REPO_ROOT, "05_lens_fov_sweep.png")
OUTPUT_JSON  = os.path.join(REPO_ROOT, "05_lens_fov_sweep.json")

# ---- Camera (Teledyne FLIR Boson+ — the *deployment* camera) ----------------
# NB: Stage 1 (beam characterisation at Cranfield) was captured with a FLIR
# A655sc + 13.1 mm lens. The deployment camera for drone-mounted inspection is
# different, so Stage 2 (and downstream Stage 3) use these Boson+ figures.
SENSOR_W_PX = 640
SENSOR_H_PX = 512
PIXEL_PITCH = 0.012                              # mm  (12 µm VOx microbolometer)
SENSOR_W_MM = SENSOR_W_PX * PIXEL_PITCH          # 7.68 mm
SENSOR_H_MM = SENSOR_H_PX * PIXEL_PITCH          # 6.144 mm

# ---- Lens catalogue ---------------------------------------------------------
# 18 mm lens (24° HFOV) as fitted to the Boson+ unit.
# VFOV computed from sensor aspect: 2*atan((SENSOR_H_MM/2)/f) ≈ 19.3°.
LENSES = [
    {"f_mm": 18.0, "hfov": 24.0, "vfov": 19.3, "label": "18 mm (Boson+ 24° HFoV)", "col": "#ffd93d"},
]
STANDOFFS = [300, 400, 500, 750]

# ---- Load Stage-1 beam -----------------------------------------------------
with open(INPUT_JSON) as f:
    beam = json.load(f)

# beam_derived_combined.json from Stage 1 (03_derive_combined.py) gives us:
#   sigma_x_vs_d: {slope_mm_per_mm, intercept_mm}
#   sigma_y_vs_d: {slope_mm_per_mm, intercept_mm}
#   peak_at_d_500_K (or similar reference peak)
def get_sigma(stage1, axis, d_mm):
    """Return σ at standoff d from Stage-1's linear divergence model."""
    m = stage1["derived_beam_model"][f"sigma_{axis}_vs_d"]
    return m["slope_mm_per_mm"] * d_mm + m["intercept_mm"]

def get_peak(stage1, d_mm):
    """Return peak ΔT at d from Stage-1's power-law fit: peak(d) = K · d^exp."""
    p = stage1["derived_beam_model"]["peak_vs_d_powerlaw"]
    return p["K_amp_K_mm_to_power"] * d_mm ** p["exponent"]

def fov_mm(hfov_deg, vfov_deg, d_mm):
    fw = 2 * d_mm * np.tan(np.radians(hfov_deg / 2))
    fh = 2 * d_mm * np.tan(np.radians(vfov_deg / 2))
    return fw, fh

def mm_per_px(hfov_deg, d_mm):
    return 2 * d_mm * np.tan(np.radians(hfov_deg / 2)) / SENSOR_W_PX

def beam_energy_inside_fov(fw, fh, sx, sy):
    """Fraction of a centred 2D Gaussian inside a rectangular FOV.
    Assumes Stage-1 convention I ∝ exp(-r²/(2σ²)) so half-width to 1σ uses √2."""
    # ∫ exp(-x²/(2σ²)) dx from -L/2 to L/2 = σ√(2π)·erf(L/(2σ√2))
    return erf(fw / (2 * sx * np.sqrt(2))) * erf(fh / (2 * sy * np.sqrt(2)))

# ---- Build the sweep grid --------------------------------------------------
grid = []
for L in LENSES:
    for d in STANDOFFS:
        fw, fh = fov_mm(L["hfov"], L["vfov"], d)
        mpp    = mm_per_px(L["hfov"], d)
        sx     = float(get_sigma(beam, "x", d))
        sy     = float(get_sigma(beam, "y", d))
        peak   = float(get_peak(beam, d))
        in_fov = beam_energy_inside_fov(fw, fh, sx, sy)
        # FWHM beam diameter for an intuitive "fits / clips" check
        fwhm_x = 2.355 * sx
        fwhm_y = 2.355 * sy
        # σ_x in pixels at this configuration (useful for Stage 3)
        sx_px  = sx / mpp
        sy_px  = sy / mpp
        grid.append({
            "lens_f_mm":           L["f_mm"],
            "lens_label":          L["label"],
            "hfov_deg":            L["hfov"],
            "vfov_deg":            L["vfov"],
            "standoff_mm":         d,
            "fov_w_mm":            float(fw),
            "fov_h_mm":            float(fh),
            "mm_per_px":           float(mpp),
            "sigma_x_mm":          sx,
            "sigma_y_mm":          sy,
            "sigma_x_px":          float(sx_px),
            "sigma_y_px":          float(sy_px),
            "fwhm_x_mm":           float(fwhm_x),
            "fwhm_y_mm":           float(fwhm_y),
            "fwhm_fits_x":         bool(fwhm_x <= fw),
            "fwhm_fits_y":         bool(fwhm_y <= fh),
            "beam_energy_in_fov":  float(in_fov),
            "peak_dT_K":           peak,
        })

# ---- Console summary --------------------------------------------------------
print("Stage 2 -- Lens x Standoff sweep")
print(f"{'lens':<28}{'d (mm)':>8}{'FOV (mm)':>15}{'mm/px':>10}{'sx (mm)':>10}{'sy (mm)':>10}{'E_in_FOV':>11}")
print("-" * 92)
for r in grid:
    print(f"{r['lens_label']:<28}{r['standoff_mm']:>8}"
          f"{r['fov_w_mm']:>7.0f}×{r['fov_h_mm']:<7.0f}"
          f"{r['mm_per_px']:>10.3f}"
          f"{r['sigma_x_mm']:>10.1f}{r['sigma_y_mm']:>10.1f}"
          f"{r['beam_energy_in_fov']*100:>10.1f}%")

# ---- Figure -----------------------------------------------------------------
fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor("#111111")

gs_outer = mgridspec.GridSpec(2, 2, figure=fig,
                              left=0.05, right=0.97, top=0.92, bottom=0.06,
                              hspace=0.30, wspace=0.22,
                              height_ratios=[2.5, 1])

# === Top-left: 3×4 grid of FOV vs beam footprint at each (lens, d) ============
gs_fov = mgridspec.GridSpecFromSubplotSpec(len(LENSES), len(STANDOFFS),
                                            subplot_spec=gs_outer[0, :],
                                            hspace=0.45, wspace=0.20)

def dark_ax(ax, grid=True):
    ax.set_facecolor("#1a1a1a")
    for sp in ax.spines.values(): sp.set_edgecolor("#444444")
    ax.tick_params(colors="#aaaaaa", labelsize=7)
    if grid: ax.grid(True, alpha=0.10, color="white")

for li, L in enumerate(LENSES):
    for di, d in enumerate(STANDOFFS):
        ax = fig.add_subplot(gs_fov[li, di])
        dark_ax(ax, grid=False)
        ax.set_aspect("equal")
        r = next(g for g in grid if g["lens_f_mm"] == L["f_mm"] and g["standoff_mm"] == d)

        # axes in mm, centred on optical axis
        fov_w = r["fov_w_mm"]; fov_h = r["fov_h_mm"]
        sx = r["sigma_x_mm"]; sy = r["sigma_y_mm"]
        # plot half-window large enough to show both FOV and beam
        half_w = max(fov_w, 2.5 * sx) * 0.7
        half_h = max(fov_h, 2.5 * sy) * 0.7
        ax.set_xlim(-half_w, half_w)
        ax.set_ylim(-half_h, half_h)

        # beam: shaded 1σ + 2σ ellipses
        ax.add_patch(Ellipse((0, 0), 2*sx, 2*sy,
                             fc=L["col"], alpha=0.18, ec=L["col"], lw=1.4,
                             label=f"1σ: {sx:.0f}×{sy:.0f} mm"))
        ax.add_patch(Ellipse((0, 0), 4*sx, 4*sy,
                             fc="none", ec=L["col"], lw=1.0, ls="--", alpha=0.6,
                             label="2σ"))
        # FOV rectangle
        fov_col = "lime" if (r["fwhm_fits_x"] and r["fwhm_fits_y"]) else "#ff6b6b"
        ax.add_patch(mpatches.Rectangle((-fov_w/2, -fov_h/2), fov_w, fov_h,
                                         fc="none", ec=fov_col, lw=2.0,
                                         label=f"FOV {fov_w:.0f}×{fov_h:.0f} mm"))

        title_col = L["col"]
        ax.set_title(
            f"{L['f_mm']:.1f} mm @ {d} mm\n"
            f"mm/px={r['mm_per_px']:.3f}  E_in_FOV={r['beam_energy_in_fov']*100:.0f}%",
            color=title_col, fontsize=8, pad=4)
        if di == 0:
            ax.set_ylabel(L["label"].split(' (')[0], color=L["col"],
                          fontsize=9, fontweight="bold")
        if li == 0:
            ax.text(0.5, 1.18, f"d = {d} mm", transform=ax.transAxes,
                    ha="center", color="white", fontsize=10, fontweight="bold")

# === Bottom-left: bar chart of FOV area per lens & standoff ===================
ax_b1 = fig.add_subplot(gs_outer[1, 0])
dark_ax(ax_b1)
width = 0.25
x_pos = np.arange(len(STANDOFFS))
for li, L in enumerate(LENSES):
    fov_areas = [next(g for g in grid if g["lens_f_mm"] == L["f_mm"]
                      and g["standoff_mm"] == d)["fov_w_mm"] *
                 next(g for g in grid if g["lens_f_mm"] == L["f_mm"]
                      and g["standoff_mm"] == d)["fov_h_mm"] / 1e6
                 for d in STANDOFFS]
    bars = ax_b1.bar(x_pos + (li - 1) * width, fov_areas, width=width,
                      color=L["col"], label=L["label"])
    for b, fa in zip(bars, fov_areas):
        ax_b1.text(b.get_x() + b.get_width()/2, fa + 0.02, f"{fa:.2f}",
                   ha="center", va="bottom", color=L["col"], fontsize=7.5)
ax_b1.set_xticks(x_pos)
ax_b1.set_xticklabels([f"{d} mm" for d in STANDOFFS], color="#aaaaaa")
ax_b1.set_ylabel("FOV area (m²)", color="#aaaaaa", fontsize=9)
ax_b1.set_title("FOV area vs standoff per lens", color="white", fontsize=10)
ax_b1.legend(fontsize=7.5, facecolor="#222222", labelcolor="white",
              edgecolor="#555555", loc="upper left")

# === Bottom-right: beam energy capture vs standoff per lens ===================
ax_b2 = fig.add_subplot(gs_outer[1, 1])
dark_ax(ax_b2)
for L in LENSES:
    es = [next(g for g in grid if g["lens_f_mm"] == L["f_mm"]
               and g["standoff_mm"] == d)["beam_energy_in_fov"] * 100
          for d in STANDOFFS]
    ax_b2.plot(STANDOFFS, es, "o-", color=L["col"], lw=2, ms=8, label=L["label"])
    for d, e in zip(STANDOFFS, es):
        ax_b2.text(d, e + 1.5, f"{e:.0f}%",
                   ha="center", color=L["col"], fontsize=8)
ax_b2.axhline(95, color="lime", lw=1, ls="--", alpha=0.4)
ax_b2.text(STANDOFFS[-1], 96, "  95% target", color="lime", fontsize=7.5, va="bottom")
ax_b2.set_xlabel("Standoff (mm)", color="#aaaaaa", fontsize=9)
ax_b2.set_ylabel("Beam energy captured by FOV (%)", color="#aaaaaa", fontsize=9)
ax_b2.set_title("Beam-vs-FOV capture (centred beam)", color="white", fontsize=10)
ax_b2.set_ylim(0, 105)
ax_b2.legend(fontsize=7.5, facecolor="#222222", labelcolor="white",
              edgecolor="#555555", loc="lower right")

fig.suptitle(
    "Stage 2 — Lens × Standoff sweep\n"
    "Beam from Stage 1 (Cranfield beam_derived_combined.json)  |  FLIR A655sc 640×480 @ 17 µm pitch",
    color="white", fontsize=12, y=0.97)

fig.savefig(OUTPUT_PNG, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"\nSaved figure: {OUTPUT_PNG}")

# ---- JSON for Stage 3 -------------------------------------------------------
with open(OUTPUT_JSON, "w") as f:
    json.dump({
        "camera": {"sensor_w_px": SENSOR_W_PX, "sensor_h_px": SENSOR_H_PX,
                   "pixel_pitch_mm": PIXEL_PITCH},
        "lenses":     LENSES,
        "standoffs":  STANDOFFS,
        "beam_source": INPUT_JSON,
        "grid":       grid,
    }, f, indent=2)
print(f"Saved JSON:   {OUTPUT_JSON}")
