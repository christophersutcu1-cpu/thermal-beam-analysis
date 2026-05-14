"""
Multi-standoff beam characterisation.

Loads summary.json from each standoff folder, converts pixel measurements
to physical mm using the camera HFOV, fits a linear divergence model to
extrapolate the true beam profile, and produces a comparison figure.

Usage:
    python beam_comparison.py
"""

import os
import config
import json
import glob
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter

# -- Config -------------------------------------------------------------------
CAPTURES_ROOT  = config.BOSON_ROOT + r""
OUTPUT_PNG     = os.path.join(CAPTURES_ROOT, "beam_comparison.png")

# Boson 640 with 18mm lens: pixel_pitch=12um, sensor=7.68x6.144mm
# HFOV = 2*atan(7.68/36) = 24.09 deg, VFOV = 2*atan(6.144/36) = 19.37 deg
BOSON_HFOV_DEG = 24.09
BOSON_VFOV_DEG = 19.37
SENSOR_W_PX    = 640
SENSOR_H_PX    = 512
CAMERA_FPS     = 9.0
PRE_RELAY_SECS = 1.0

# Map folder name → standoff in mm
STANDOFFS = {
    "500 mm": 500,
    "600 mm": 600,
    "700 mm": 700,
    "800 mm": 800,
}

BEAM_COLOR  = "#00cfff"
COLORS      = ["#ff6b6b", "#ffd93d", "#6bcb77", "#4d96ff"]  # one per standoff
EXTRAPOLATE = [400, 500, 600, 700, 800, 900, 1000, 1200]    # mm


# -- Camera geometry ----------------------------------------------------------
def mm_per_px(standoff_mm, axis="x"):
    """Physical mm per pixel at a given standoff distance."""
    fov = BOSON_HFOV_DEG if axis == "x" else BOSON_VFOV_DEG
    sensor_px = SENSOR_W_PX if axis == "x" else SENSOR_H_PX
    return 2.0 * standoff_mm * np.tan(np.radians(fov / 2.0)) / sensor_px


# -- Frame loading ------------------------------------------------------------
def load_diff_image(folder):
    """Return baseline-subtracted mean image for a session folder."""
    tiff_dirs = sorted(glob.glob(os.path.join(folder, "boson_*")))
    tiff_dirs = [d for d in tiff_dirs if os.path.isdir(d)]
    if not tiff_dirs:
        return None
    tiff_dir = tiff_dirs[0]
    paths = sorted(glob.glob(os.path.join(tiff_dir, "frame_*.tiff")))
    if not paths:
        return None
    frames = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if len(img.shape) == 3:
            img = img[:, :, 0]
        frames.append(img.astype(np.float32))
    frames = np.array(frames)
    fps  = CAMERA_FPS
    i0   = int((PRE_RELAY_SECS + 2.0) * fps)
    i1   = min(len(frames), int((PRE_RELAY_SECS + 11.0) * fps))
    if i0 >= i1:
        i0, i1 = len(frames) // 4, len(frames)
    baseline_end = max(1, int((PRE_RELAY_SECS - 0.1) * fps))
    diff = frames[i0:i1].mean(axis=0) - frames[0:baseline_end].mean(axis=0)
    return diff


# -- Divergence model ---------------------------------------------------------
def linear(d, a, b):
    """FWHM_physical = a + b*d  (a=source size mm, b=divergence half-angle rad)"""
    return a + b * np.array(d)


# -- Main ---------------------------------------------------------------------
def main():
    records = []
    for folder_name, standoff in sorted(STANDOFFS.items(), key=lambda x: x[1]):
        folder = os.path.join(CAPTURES_ROOT, folder_name)
        json_path = os.path.join(folder, "summary.json")
        if not os.path.exists(json_path):
            print(f"  Missing: {json_path} — skipping")
            continue
        with open(json_path) as f:
            data = json.load(f)
        beam = data.get("beam", {})
        unif = data.get("uniformity", {})

        scale_x = mm_per_px(standoff, "x")
        scale_y = mm_per_px(standoff, "y")

        records.append({
            "label":     folder_name,
            "standoff":  standoff,
            "fwhm_x_px": beam.get("fwhm_x", 0),
            "fwhm_y_px": beam.get("fwhm_y", 0),
            "fwhm_x_mm": beam.get("fwhm_x", 0) * scale_x,
            "fwhm_y_mm": beam.get("fwhm_y", 0) * scale_y,
            "sigma_x_mm":beam.get("sigma_x", 0) * scale_x,
            "sigma_y_mm":beam.get("sigma_y", 0) * scale_y,
            "cx_px":     beam.get("cx", 0),
            "cy_px":     beam.get("cy", 0),
            "amplitude": beam.get("amplitude", 0),
            "theta_deg": beam.get("theta_deg", 0),
            "aspect":    beam.get("aspect_ratio", 1),
            "cov":       unif.get("cov_pct", 0),
            "p2v":       unif.get("p2v_pct", 0),
            "mean_dt":   unif.get("mean_dt", 0),
            "folder":    folder,
            "scale_x":   scale_x,
            "scale_y":   scale_y,
        })
        print(f"{folder_name}:  FWHM {beam.get('fwhm_x',0):.0f}×{beam.get('fwhm_y',0):.0f} px  "
              f"->  {beam.get('fwhm_x',0)*scale_x:.0f}x{beam.get('fwhm_y',0)*scale_y:.0f} mm  "
              f"CoV={unif.get('cov_pct',0):.1f}%")

    if len(records) < 2:
        print("Need at least 2 standoffs to fit divergence model.")
        return

    standoffs  = np.array([r["standoff"]  for r in records])
    fwhm_x_mm  = np.array([r["fwhm_x_mm"] for r in records])
    fwhm_y_mm  = np.array([r["fwhm_y_mm"] for r in records])
    amplitudes  = np.array([r["amplitude"] for r in records])
    covs        = np.array([r["cov"]       for r in records])

    # 500mm data excluded from fit — beam centre near frame edge, large rotation (-24.8 deg)
    fit_mask = standoffs >= 600
    s_fit    = standoffs[fit_mask]
    fx_fit   = fwhm_x_mm[fit_mask]
    fy_fit   = fwhm_y_mm[fit_mask]

    # Fit linear divergence: FWHM(d) = a + b*d
    (cx0, cx1), _ = curve_fit(linear, s_fit, fx_fit, p0=[50, 0.5])
    (cy0, cy1), _ = curve_fit(linear, s_fit, fy_fit, p0=[50, 0.3])

    # Fit inverse-square amplitude: A(d) = k / d^2
    def inv_sq(d, k):
        return k / np.array(d, dtype=float)**2
    (kA,), _ = curve_fit(inv_sq, standoffs, amplitudes, p0=[1e6])

    d_fit   = np.linspace(min(standoffs)*0.7, max(EXTRAPOLATE), 300)
    d_extra = np.array(EXTRAPOLATE)

    div_x_deg = np.degrees(np.arctan(cx1))
    div_y_deg = np.degrees(np.arctan(cy1))

    print(f"\nDivergence model (linear fit, 600-800mm):")
    print(f"  FWHM_x(d) = {cx0:.1f} + {cx1:.4f}*d mm   ({div_x_deg:.2f} deg half-angle)")
    print(f"  FWHM_y(d) = {cy0:.1f} + {cy1:.4f}*d mm   ({div_y_deg:.2f} deg half-angle)")
    print(f"\nNOTE: Physical FWHM is roughly flat — beam likely overflows the sensor FOV")
    print(f"      at all distances. True divergence angle requires the full beam in-frame.")
    print(f"\nExtrapolated beam FWHM:")
    for d in d_extra:
        print(f"  {d:4.0f} mm:  {linear(d,cx0,cx1):.0f} x {linear(d,cy0,cy1):.0f} mm")

    # =========================================================================
    # FIGURE  —  4 rows
    # Row 0: diff images at each standoff (4 panels)
    # Row 1: FWHM x & y vs standoff with fit + extrapolation (2 wide panels)
    # Row 2: amplitude vs standoff | CoV vs standoff | theta vs standoff
    # Row 3: extrapolation table
    # =========================================================================
    n = len(records)
    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor("#111111")
    fig.suptitle("Multi-Standoff Beam Characterisation", fontsize=14, color="white", y=0.99)

    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.5, wspace=0.35,
                           left=0.05, right=0.97, top=0.96, bottom=0.04)

    def dark_ax(ax):
        ax.set_facecolor("#1a1a1a")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        ax.xaxis.label.set_color("#aaaaaa")
        ax.yaxis.label.set_color("#aaaaaa")
        ax.title.set_color("white")
        return ax

    # ----- Row 0: differential images ----------------------------------------
    for i, (rec, col) in enumerate(zip(records, COLORS)):
        ax = dark_ax(fig.add_subplot(gs[0, i]))
        diff = load_diff_image(rec["folder"])
        if diff is not None:
            vmin, vmax = np.percentile(diff, 2), np.percentile(diff, 98)
            ax.imshow(diff, cmap="inferno", origin="upper", vmin=vmin, vmax=vmax)
            # FWHM ellipse in pixels
            e = Ellipse(xy=(rec["cx_px"], rec["cy_px"]),
                        width=rec["fwhm_x_px"], height=rec["fwhm_y_px"],
                        angle=rec["theta_deg"],
                        edgecolor=col, facecolor="none", lw=1.8, ls="--", zorder=5)
            ax.add_patch(e)
            ax.plot(rec["cx_px"], rec["cy_px"], "+", color=col, ms=10, mew=2, zorder=6)
        ax.set_title(f"{rec['label']}\nFWHM {rec['fwhm_x_mm']:.0f}×{rec['fwhm_y_mm']:.0f} mm",
                     fontsize=9)
        ax.set_xlim(0, SENSOR_W_PX); ax.set_ylim(SENSOR_H_PX, 0)
        ax.set_xlabel("px", fontsize=7); ax.set_ylabel("px", fontsize=7)

    # ----- Row 1 left: FWHM vs standoff + fit --------------------------------
    ax_fw = dark_ax(fig.add_subplot(gs[1, :2]))
    ax_fw.scatter(standoffs, fwhm_x_mm, color=[COLORS[i] for i in range(n)],
                  s=80, zorder=5, label="_nolegend_")
    ax_fw.scatter(standoffs, fwhm_y_mm, color=[COLORS[i] for i in range(n)],
                  s=80, marker="s", zorder=5, label="_nolegend_")
    ax_fw.plot(d_fit, linear(d_fit, cx0, cx1), color=BEAM_COLOR, lw=1.5,
               label=f"FWHM-x fit  ({div_x_deg:.1f}° half-angle)")
    ax_fw.plot(d_fit, linear(d_fit, cy0, cy1), color="#ffaa00", lw=1.5, ls="--",
               label=f"FWHM-y fit  ({div_y_deg:.1f}° half-angle)")
    # shade extrapolation zone
    ax_fw.axvspan(max(standoffs), max(EXTRAPOLATE), color="white", alpha=0.04, label="Extrapolated")
    for d in d_extra:
        if d > max(standoffs):
            ax_fw.axvline(d, color="#444444", lw=0.6, ls=":")
    # label each measured point
    for rec, col in zip(records, COLORS):
        ax_fw.annotate(f"  {rec['label']}", (rec["standoff"], rec["fwhm_x_mm"]),
                       color=col, fontsize=7, va="center")
    ax_fw.set_xlabel("Standoff (mm)"); ax_fw.set_ylabel("Physical FWHM (mm)")
    ax_fw.set_title("Beam FWHM vs standoff  (● x-axis, ■ y-axis)  +  linear divergence fit")
    ax_fw.legend(fontsize=8, facecolor="#222222", labelcolor="white", edgecolor="#555555")
    ax_fw.grid(True, alpha=0.12, color="white")
    ax_fw.set_xlim(min(standoffs) * 0.85, max(EXTRAPOLATE) * 1.02)

    # ----- Row 1 right: aspect ratio & rotation vs standoff ------------------
    ax_ar = dark_ax(fig.add_subplot(gs[1, 2]))
    aspects = [r["aspect"] for r in records]
    thetas  = [abs(r["theta_deg"]) for r in records]
    ax_ar.plot(standoffs, aspects, color=BEAM_COLOR, lw=2, marker="o", ms=7, label="Aspect sx/sy")
    ax_ar.axhline(1.0, color="lime", lw=0.8, ls=":", label="Circular (1.0)")
    ax_ar.set_xlabel("Standoff (mm)"); ax_ar.set_ylabel("Aspect ratio (sx/sy)")
    ax_ar.set_title("Aspect ratio vs standoff")
    ax_ar.legend(fontsize=8, facecolor="#222222", labelcolor="white", edgecolor="#555555")
    ax_ar.grid(True, alpha=0.12, color="white")

    ax_th = dark_ax(fig.add_subplot(gs[1, 3]))
    ax_th.plot(standoffs, thetas, color="#ffaa00", lw=2, marker="o", ms=7)
    ax_th.set_xlabel("Standoff (mm)"); ax_th.set_ylabel("|Rotation angle| (deg)")
    ax_th.set_title("Beam tilt vs standoff")
    ax_th.grid(True, alpha=0.12, color="white")

    # ----- Row 2: amplitude, CoV, mean_dt ------------------------------------
    ax_amp = dark_ax(fig.add_subplot(gs[2, :2]))
    ax_amp.scatter(standoffs, amplitudes,
                   color=COLORS[:n], s=80, zorder=5, label="Measured peak")
    ax_amp.plot(d_fit, inv_sq(d_fit, kA), color=BEAM_COLOR, lw=1.5, ls="--",
                label="1/d² fit")
    ax_amp.set_xlabel("Standoff (mm)"); ax_amp.set_ylabel("Peak amplitude (counts)")
    ax_amp.set_title("Peak amplitude vs standoff  (inverse-square law fit)")
    ax_amp.legend(fontsize=8, facecolor="#222222", labelcolor="white", edgecolor="#555555")
    ax_amp.grid(True, alpha=0.12, color="white")

    ax_cov = dark_ax(fig.add_subplot(gs[2, 2]))
    bars = ax_cov.bar([r["label"] for r in records], covs,
                      color=COLORS[:n], edgecolor="#333333", linewidth=0.8)
    for bar, val in zip(bars, covs):
        ax_cov.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", color="white", fontsize=8)
    ax_cov.axhline(15, color="lime", lw=1, ls=":", label="15% target")
    ax_cov.set_ylabel("CoV (%)"); ax_cov.set_title("Beam uniformity (CoV)\nwithin 1-sigma ellipse")
    ax_cov.tick_params(axis="x", labelrotation=30, labelsize=7)
    ax_cov.legend(fontsize=8, facecolor="#222222", labelcolor="white", edgecolor="#555555")
    ax_cov.grid(True, alpha=0.12, color="white", axis="y")

    ax_dt = dark_ax(fig.add_subplot(gs[2, 3]))
    mean_dts = [r["mean_dt"] for r in records]
    ax_dt.plot(standoffs, mean_dts, color="#6bcb77", lw=2, marker="o", ms=7)
    ax_dt.set_xlabel("Standoff (mm)"); ax_dt.set_ylabel("Mean dT (counts)")
    ax_dt.set_title("Mean beam intensity vs standoff")
    ax_dt.grid(True, alpha=0.12, color="white")

    # ----- Row 3: extrapolation table ----------------------------------------
    ax_tab = fig.add_subplot(gs[3, :])
    ax_tab.set_facecolor("#111111"); ax_tab.axis("off")

    header = ["Standoff (mm)", "FWHM-x (mm)", "FWHM-y (mm)", "Aspect x/y",
              "Coverage at 600×400mm target\n(% of frame filled)"]
    rows_data = []
    for d in d_extra:
        fx = linear(d, cx0, cx1)
        fy = linear(d, cy0, cy1)
        aspect = fx / fy if fy > 0 else 0
        # what fraction of a 600×400mm target is covered by FWHM footprint
        cover = min(100, fx / 600 * 100) if fx > 0 else 0
        measured = "★ " if d in standoffs else ""
        rows_data.append([f"{measured}{d:.0f}", f"{fx:.0f}", f"{fy:.0f}",
                          f"{aspect:.2f}", f"{cover:.0f}%"])

    tab = ax_tab.table(cellText=rows_data, colLabels=header,
                       cellLoc="center", loc="center", bbox=[0.05, 0, 0.9, 1])
    tab.auto_set_font_size(False); tab.set_fontsize(9)
    for j in range(len(header)):
        tab[0, j].set_facecolor("#222244")
        tab[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows_data) + 1):
        for j in range(len(header)):
            is_measured = rows_data[i-1][0].startswith("★")
            bg = "#1e2e1e" if is_measured else ("#1a1a1a" if i % 2 == 0 else "#222222")
            tab[i, j].set_facecolor(bg)
            tab[i, j].set_text_props(color="lime" if is_measured else "white")
            tab[i, j].set_edgecolor("#333333")
    ax_tab.set_title(
        f"Extrapolated beam size  |  Divergence: {div_x_deg:.2f}° (x)  ×  {div_y_deg:.2f}° (y)  "
        f"|  ★ = measured  |  HFOV assumed {BOSON_HFOV_DEG}°",
        color="white", pad=8, fontsize=10)

    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nFigure saved: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
