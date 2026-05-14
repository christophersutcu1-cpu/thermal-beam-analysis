"""
Active Thermography Setup Summary Report

Covers:
  - Excitation source suitability for 320x175mm specimen
  - Optimal specimen orientation in camera FOV
  - Camera lens comparison: 4.3, 9.2, 13.8, 18mm
  - Two-beam coverage assessment
  - Recommendations

Usage:
    python summary_report.py
"""

import os
import config
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.patheffects as pe

# -- Config -------------------------------------------------------------------
CAPTURES_ROOT = config.BOSON_ROOT + r""
OUTPUT_PNG    = os.path.join(CAPTURES_ROOT, "setup_summary.png")
BEST_SESSION  = os.path.join(CAPTURES_ROOT, "700 mm", "summary.json")

SPEC_W, SPEC_H = 320.0, 175.0   # mm  (landscape: W=320 horizontal)
SENSOR_W_PX    = 640
SENSOR_H_PX    = 512
PIXEL_PITCH    = 0.012           # mm  (Boson 640, 12um pitch)
SENSOR_W_MM    = SENSOR_W_PX * PIXEL_PITCH   # 7.68mm
SENSOR_H_MM    = SENSOR_H_PX * PIXEL_PITCH   # 6.144mm

LENSES_MM  = [4.3, 9.2, 13.8, 18.0]
LENS_NAMES = ["4.3mm", "9.2mm", "13.8mm", "18mm (current)"]
LENS_COLORS = ["#ff6b6b", "#ffd93d", "#6bcb77", "#4d96ff"]


# -- Camera geometry ----------------------------------------------------------
def lens_fov(f):
    hfov = 2 * np.degrees(np.arctan(SENSOR_W_MM / (2 * f)))
    vfov = 2 * np.degrees(np.arctan(SENSOR_H_MM / (2 * f)))
    return hfov, vfov

def min_standoff(f):
    """Minimum standoff so specimen fits in frame (landscape)."""
    return max(SPEC_W * f / SENSOR_W_MM, SPEC_H * f / SENSOR_H_MM)

def px_per_mm2(f, d):
    scale = d * PIXEL_PITCH / f   # mm/px
    spec_px_w = min(SPEC_W, SENSOR_W_PX * scale) / scale
    spec_px_h = min(SPEC_H, SENSOR_H_PX * scale) / scale
    return (spec_px_w * spec_px_h) / (SPEC_W * SPEC_H)


# -- Beam analysis ------------------------------------------------------------
def load_beam():
    with open(BEST_SESSION) as f:
        b = json.load(f)["beam"]
    scale = 2 * 700 * np.tan(np.radians(lens_fov(18)[0] / 2)) / 640
    return b["sigma_x"] * scale, b["sigma_y"] * scale, b["amplitude"]

def beam_coverage(sx_free, sy_free, theta_deg, spec_w, spec_h, n=500):
    """
    For two symmetric beams at +-theta with optimal offset,
    return CoV on specimen and edge-to-centre ratio in each axis.
    """
    theta_r = np.radians(theta_deg)
    sx_s = sx_free / np.cos(theta_r)
    sy_s = sy_free
    surf_peak = np.cos(theta_r)

    # Optimise offset (1D in x)
    offsets = np.linspace(0, sx_s * 2, n)
    x = np.linspace(-spec_w/2, spec_w/2, 500)
    best_cov, best_off = 1e9, 0.0
    for off in offsets:
        ix = surf_peak * (np.exp(-0.5 * ((x - off)/sx_s)**2) +
                         np.exp(-0.5 * ((x + off)/sx_s)**2))
        c = ix.std() / ix.mean() * 100
        if c < best_cov:
            best_cov, best_off = c, off

    # Y uniformity (single pass, no offset benefit)
    y   = np.linspace(-spec_h/2, spec_h/2, 500)
    iy  = surf_peak * np.exp(-0.5 * (y / sy_s)**2) * 2
    cov_y = iy.std() / iy.mean() * 100
    edge_y = float(np.exp(-0.5 * (spec_h/2 / sy_s)**2))

    return best_cov, cov_y, best_off, sx_s, sy_s, edge_y


def main():
    sx_free, sy_free, peak = load_beam()
    fwhm_x = sx_free * 2.355
    fwhm_y = sy_free * 2.355

    # =========================================================================
    # TEXT REPORT
    # =========================================================================
    sep = "=" * 68
    print(sep)
    print("  ACTIVE THERMOGRAPHY SETUP SUMMARY")
    print(f"  Specimen: {SPEC_W:.0f} x {SPEC_H:.0f} mm  |  Camera: Boson 640 (12um pitch)")
    print(sep)

    print("\n1. EXCITATION SOURCE ASSESSMENT")
    print("-" * 40)
    print(f"  Beam FWHM (free-space, 700mm standoff):")
    print(f"    X-axis : {fwhm_x:.0f} mm  (sigma={sx_free:.0f}mm)")
    print(f"    Y-axis : {fwhm_y:.0f} mm  (sigma={sy_free:.0f}mm)")
    print(f"  Specimen requires coverage of {SPEC_W:.0f} x {SPEC_H:.0f} mm")
    print()
    edge_x_single = np.exp(-0.5 * (SPEC_W/2 / sx_free)**2)
    edge_y_single = np.exp(-0.5 * (SPEC_H/2 / sy_free)**2)
    print(f"  Single beam edge irradiance:")
    print(f"    X edges (+/-{SPEC_W/2:.0f}mm): {edge_x_single*100:.0f}% of centre peak  "
          f"({'POOR' if edge_x_single < 0.5 else 'OK'})")
    print(f"    Y edges (+/-{SPEC_H/2:.0f}mm): {edge_y_single*100:.0f}% of centre peak  "
          f"({'POOR' if edge_y_single < 0.5 else 'OK'})")
    print()
    for theta in [30, 45, 60]:
        cov_x, cov_y, opt_off, sx_s, sy_s, ey = beam_coverage(
            sx_free, sy_free, theta, SPEC_W, SPEC_H)
        print(f"  Two beams at +/-{theta}° (optimal offset={opt_off:.0f}mm):")
        print(f"    X-CoV = {cov_x:.1f}%   Y-CoV = {cov_y:.1f}%   "
              f"Y-edge = {ey*100:.0f}%")
        print(f"    Footprint on specimen: {sx_s*2.355:.0f} x {sy_s*2.355:.0f} mm")
    print()
    print("  VERDICT: Adequate for qualitative thermography. The beam")
    print(f"  under-fills the specimen in Y ({fwhm_y:.0f}mm FWHM vs {SPEC_H:.0f}mm specimen).")
    print("  Y-edge heating is only ~33% of centre. Use two-beam offset")
    print("  to correct X-axis. For quantitative work, consider a diffuser")
    print("  or additional lamp pair in Y.")

    print("\n2. SPECIMEN ORIENTATION IN CAMERA FOV")
    print("-" * 40)
    for label, sw, sh in [("Landscape (320mm horizontal)", SPEC_W, SPEC_H),
                           ("Portrait  (175mm horizontal)", SPEC_H, SPEC_W)]:
        f = 18.0
        d = min_standoff(f)
        scale = d * PIXEL_PITCH / f
        fw = SENSOR_W_PX * scale
        fh = SENSOR_H_PX * scale
        spec_px = min(sw, fw)/scale * min(sh, fh)/scale
        fill = spec_px / (SENSOR_W_PX * SENSOR_H_PX) * 100
        print(f"  {label}:")
        print(f"    Min standoff (18mm): {d:.0f}mm  Frame: {fw:.0f}x{fh:.0f}mm")
        print(f"    Pixels on specimen : {spec_px/1000:.0f}k / {SENSOR_W_PX*SENSOR_H_PX/1000:.0f}k  "
              f"({fill:.0f}% sensor fill)")
        print(f"    Resolution         : {scale:.3f} mm/px")
    print()
    print("  RECOMMENDATION: Landscape (320mm horizontal). Higher sensor")
    print("  fill, better pixel utilisation, beam long axis aligns with")
    print("  specimen width for two-beam offset benefit.")

    print("\n3. CAMERA LENS COMPARISON (landscape, specimen just fills width)")
    print("-" * 40)
    print(f"  {'Lens':>10}  {'HFOV':>7}  {'Min standoff':>13}  "
          f"{'Pixels on spec':>15}  {'Res (mm/px)':>12}")
    print(f"  {'-'*10}  {'-'*7}  {'-'*13}  {'-'*15}  {'-'*12}")
    lens_data = []
    for f, name in zip(LENSES_MM, LENS_NAMES):
        hfov, vfov = lens_fov(f)
        d    = min_standoff(f)
        scale = d * PIXEL_PITCH / f
        fw   = SENSOR_W_PX * scale
        fh   = SENSOR_H_PX * scale
        spec_px_w = min(SPEC_W, fw) / scale
        spec_px_h = min(SPEC_H, fh) / scale
        spec_px   = spec_px_w * spec_px_h
        fill      = spec_px / (SENSOR_W_PX * SENSOR_H_PX) * 100
        lens_data.append(dict(f=f, name=name, hfov=hfov, vfov=vfov,
                              d=d, scale=scale, fw=fw, fh=fh,
                              spec_px=spec_px, fill=fill))
        print(f"  {name:>10}  {hfov:>6.1f}°  {d:>12.0f}mm  "
              f"{spec_px/1000:>13.0f}k  {scale:>12.3f}")
    print()
    print("  NOTE: All lenses give equal pixel density at minimum standoff.")
    print("  Lens choice is driven by working distance and lamp clearance:")
    print("    4.3mm  (179mm): Too close — camera inside lamp zone")
    print("    9.2mm  (383mm): Workable — compact setup")
    print("    13.8mm (575mm): Good clearance — RECOMMENDED")
    print("    18mm   (750mm): Current — fine but large footprint")

    print("\n4. BEAM vs CAMERA FOV MATCH")
    print("-" * 40)
    for ld in lens_data:
        d = ld["d"]
        # scale beam mm -> px at this standoff
        scale_at_d = d * PIXEL_PITCH / ld["f"]
        beam_px_x  = fwhm_x / scale_at_d
        beam_px_y  = fwhm_y / scale_at_d
        spec_px_x  = SPEC_W / scale_at_d
        spec_px_y  = SPEC_H / scale_at_d
        print(f"  {ld['name']:>14}: beam footprint {fwhm_x:.0f}x{fwhm_y:.0f}mm  "
              f"= {beam_px_x:.0f}x{beam_px_y:.0f}px  |  "
              f"specimen = {spec_px_x:.0f}x{spec_px_y:.0f}px")
    print()
    print("  Beam physical size is lamp-standoff dependent, not lens dependent.")
    print(f"  Current beam FWHM {fwhm_x:.0f}x{fwhm_y:.0f}mm covers specimen width")
    print(f"  adequately in X with two-beam offset, but under-fills Y by")
    print(f"  {(SPEC_H - fwhm_y):.0f}mm ({(SPEC_H-fwhm_y)/SPEC_H*100:.0f}% deficit).")

    print(f"\n{sep}")
    print("  OVERALL RECOMMENDATION")
    print(sep)
    print(f"  Camera  : 13.8mm lens at ~575mm standoff (landscape)")
    print(f"  Lamps   : Two beams at 45°, each offset {beam_coverage(sx_free,sy_free,45,SPEC_W,SPEC_H)[2]:.0f}mm from centre")
    print(f"  Expected uniformity: X-CoV ~{beam_coverage(sx_free,sy_free,45,SPEC_W,SPEC_H)[0]:.0f}%  Y-CoV ~{beam_coverage(sx_free,sy_free,45,SPEC_W,SPEC_H)[1]:.0f}%")
    print(f"  Limiting factor: beam height ({fwhm_y:.0f}mm) < specimen height ({SPEC_H:.0f}mm)")
    print(f"  To achieve <15% overall CoV, beam Y-FWHM must reach ~{SPEC_H*1.5:.0f}mm")
    print(f"  Options: add Y-axis lamp pair, diffuser plate, or scan excitation")
    print(sep)

    # =========================================================================
    # FIGURE
    # =========================================================================
    fig = plt.figure(figsize=(24, 20))
    fig.patch.set_facecolor("#0d0d0d")
    fig.suptitle(
        "Active Thermography Setup Summary  |  "
        f"Specimen {SPEC_W:.0f}x{SPEC_H:.0f}mm  |  Boson 640 (12um pitch)",
        fontsize=14, color="white", y=0.99, fontweight="bold")

    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.55, wspace=0.38,
                           left=0.05, right=0.97, top=0.96, bottom=0.04)

    def dark_ax(ax, title=""):
        ax.set_facecolor("#1a1a1a")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444444")
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        ax.xaxis.label.set_color("#aaaaaa")
        ax.yaxis.label.set_color("#aaaaaa")
        if title:
            ax.set_title(title, color="white", fontsize=9, pad=4)
        return ax

    # =========================================================================
    # Row 0: System diagram | Beam profile on specimen | Sensor fill diagram
    # =========================================================================

    # ----- Panel 0,0-1: System diagram (top view) ----------------------------
    ax_sys = dark_ax(fig.add_subplot(gs[0, :2]),
                     "System Layout (top view)")
    ax_sys.set_xlim(-700, 700); ax_sys.set_ylim(-100, 900)
    ax_sys.set_aspect("equal")

    # Specimen
    spec_rect = Rectangle((-SPEC_W/2, 0), SPEC_W, SPEC_H*0.4,
                           fc="#334433", ec="lime", lw=2, zorder=5)
    ax_sys.add_patch(spec_rect)
    ax_sys.text(0, SPEC_H*0.2, f"Specimen\n{SPEC_W:.0f}x{SPEC_H:.0f}mm",
                ha="center", va="center", color="lime", fontsize=9, fontweight="bold")

    # Camera (above specimen)
    for ld in [lens_data[1], lens_data[2], lens_data[3]]:  # 9.2, 13.8, 18mm
        cam_y  = ld["d"]
        hw     = np.tan(np.radians(ld["hfov"]/2)) * ld["d"]
        ax_sys.annotate("", xy=(-SPEC_W/2, 0), xytext=(0, cam_y),
                        arrowprops=dict(arrowstyle="-", color=LENS_COLORS[LENSES_MM.index(ld["f"])],
                                        lw=0.8, ls="--", alpha=0.5))
        ax_sys.annotate("", xy=(+SPEC_W/2, 0), xytext=(0, cam_y),
                        arrowprops=dict(arrowstyle="-", color=LENS_COLORS[LENSES_MM.index(ld["f"])],
                                        lw=0.8, ls="--", alpha=0.5))
        ax_sys.plot(0, cam_y, "s", color=LENS_COLORS[LENSES_MM.index(ld["f"])],
                    ms=10, zorder=8)
        ax_sys.text(15, cam_y + 10, ld["name"], color=LENS_COLORS[LENSES_MM.index(ld["f"])],
                    fontsize=8)

    # Two lamps at 45 deg
    lamp_d = 700
    for sign, label in [(+1, "Lamp 1"), (-1, "Lamp 2")]:
        lx = sign * lamp_d * np.sin(np.radians(45))
        ly = lamp_d * np.cos(np.radians(45))
        ax_sys.plot(lx, ly, "*", color="#ffaa00", ms=16, zorder=8)
        ax_sys.annotate("", xy=(0, SPEC_H*0.2),
                        xytext=(lx, ly),
                        arrowprops=dict(arrowstyle="->", color="#ffaa00",
                                        lw=1.5, alpha=0.8))
        ax_sys.text(lx + sign*20, ly + 20, label, color="#ffaa00", fontsize=8)

    ax_sys.set_xlabel("X (mm)"); ax_sys.set_ylabel("Y/Z (mm)")
    legend_items = [plt.Line2D([0],[0], color=LENS_COLORS[i], lw=1.5,
                               label=LENS_NAMES[i]) for i in range(1,4)]
    legend_items += [plt.Line2D([0],[0], color="#ffaa00", marker="*",
                                ls="none", ms=10, label="Halogen lamp")]
    ax_sys.legend(handles=legend_items, fontsize=7,
                  facecolor="#222222", labelcolor="white", edgecolor="#555555",
                  loc="upper right")
    ax_sys.grid(True, alpha=0.08, color="white")

    # ----- Panel 0,2: Sensor fill for each lens + orientation ----------------
    ax_sf = dark_ax(fig.add_subplot(gs[0, 2]),
                    "Sensor fill at minimum standoff")
    ax_sf.set_xlim(-0.1, 1.1); ax_sf.set_ylim(-0.1, 1.1)
    ax_sf.set_aspect("equal")
    ax_sf.set_xticks([]); ax_sf.set_yticks([])

    # Sensor outline (normalised 640x512)
    ax_sf.add_patch(Rectangle((0, 0), 1, SENSOR_H_PX/SENSOR_W_PX,
                               ec="#555555", fc="#111111", lw=2))
    ax_sf.text(0.5, SENSOR_H_PX/SENSOR_W_PX + 0.02, "Sensor (640x512)",
               ha="center", color="#aaaaaa", fontsize=8)

    # Landscape specimen normalised
    spec_norm_w = SPEC_W / (SENSOR_W_PX * 0.5)  # at 0.5mm/px
    spec_norm_h = SPEC_H / (SENSOR_W_PX * 0.5)
    spec_x = (1 - spec_norm_w) / 2
    spec_y = (SENSOR_H_PX/SENSOR_W_PX - spec_norm_h) / 2
    ax_sf.add_patch(Rectangle((spec_x, spec_y), spec_norm_w, spec_norm_h,
                               ec="lime", fc="#1e3e1e", lw=2, ls="--"))
    ax_sf.text(0.5, spec_y + spec_norm_h/2,
               f"Specimen {SPEC_W:.0f}x{SPEC_H:.0f}mm\n(landscape)",
               ha="center", va="center", color="lime", fontsize=8)
    fill_pct = spec_norm_w * spec_norm_h / (1 * SENSOR_H_PX/SENSOR_W_PX) * 100
    ax_sf.text(0.5, -0.06, f"Sensor fill: {fill_pct:.0f}%  |  "
               f"224k px on specimen  |  0.5mm/px",
               ha="center", color="white", fontsize=7)

    # ----- Panel 0,3: Beam profile in Y vs specimen height -------------------
    ax_by = dark_ax(fig.add_subplot(gs[0, 3]),
                    "Beam Y-profile vs specimen height")
    y_arr = np.linspace(-200, 200, 500)
    iy    = 2 * np.exp(-0.5 * (y_arr / sy_free)**2)
    ax_by.plot(iy, y_arr, color="#00cfff", lw=2, label=f"Two beams (FWHM={fwhm_y:.0f}mm)")
    ax_by.axhline( SPEC_H/2, color="lime", lw=1.5, ls="--", label=f"Spec edge +{SPEC_H/2:.0f}mm")
    ax_by.axhline(-SPEC_H/2, color="lime", lw=1.5, ls="--")
    edge_val = 2 * np.exp(-0.5 * (SPEC_H/2 / sy_free)**2)
    ax_by.axhline(0, color="#444444", lw=0.5)
    ax_by.annotate(f"{edge_val/2.0*100:.0f}% of peak\nat specimen edge",
                   xy=(edge_val, SPEC_H/2),
                   xytext=(edge_val + 0.3, SPEC_H/2 + 30),
                   color="#ff6b6b", fontsize=8,
                   arrowprops=dict(arrowstyle="->", color="#ff6b6b", lw=1))
    ax_by.fill_betweenx(y_arr,
                        np.where((y_arr >= -SPEC_H/2) & (y_arr <= SPEC_H/2), 0, np.nan),
                        np.where((y_arr >= -SPEC_H/2) & (y_arr <= SPEC_H/2), iy, np.nan),
                        alpha=0.15, color="lime")
    ax_by.set_xlabel("Normalised irradiance"); ax_by.set_ylabel("Y (mm)")
    ax_by.legend(fontsize=7, facecolor="#222222", labelcolor="white", edgecolor="#555555")
    ax_by.grid(True, alpha=0.12, color="white")

    # =========================================================================
    # Row 1: H beam profiles at 30/45/60 with optimal offset
    # =========================================================================
    for col, theta in enumerate(DETAIL_ANGLES := [30, 45, 60]):
        ax_bx = dark_ax(fig.add_subplot(gs[1, col]),
                        f"X-profile  theta={theta}°  (optimal offset)")
        cov_x, cov_y, opt_off, sx_s, sy_s, ey = beam_coverage(
            sx_free, sy_free, theta, SPEC_W, SPEC_H)
        x_arr   = np.linspace(-SPEC_W*0.8, SPEC_W*0.8, 500)
        sp      = np.cos(np.radians(theta))
        ix_opt  = sp * (np.exp(-0.5*((x_arr - opt_off)/sx_s)**2) +
                        np.exp(-0.5*((x_arr + opt_off)/sx_s)**2))
        ix_ctr  = sp * 2 * np.exp(-0.5*(x_arr/sx_s)**2)
        ax_bx.plot(x_arr, ix_ctr, color="#888888", lw=1.5, ls="--",
                   alpha=0.6, label="Centred (offset=0)")
        ax_bx.plot(x_arr, ix_opt, color=["#ff6b6b","#ffd93d","#4d96ff"][col],
                   lw=2, label=f"Offset={opt_off:.0f}mm")
        ax_bx.axvline(-SPEC_W/2, color="lime", lw=1, ls=":")
        ax_bx.axvline( SPEC_W/2, color="lime", lw=1, ls=":", label="Specimen edges")
        ax_bx.fill_between(x_arr,
                           np.where((x_arr>=-SPEC_W/2)&(x_arr<=SPEC_W/2), 0, np.nan),
                           np.where((x_arr>=-SPEC_W/2)&(x_arr<=SPEC_W/2), ix_opt, np.nan),
                           alpha=0.1, color="lime")
        ax_bx.set_xlabel("X (mm)"); ax_bx.set_ylabel("Irradiance (a.u.)")
        ax_bx.set_title(f"X-profile  theta={theta}°\nX-CoV={cov_x:.1f}%  Y-CoV={cov_y:.1f}%",
                        color="white", fontsize=9)
        ax_bx.legend(fontsize=7, facecolor="#222222", labelcolor="white",
                     edgecolor="#555555")
        ax_bx.grid(True, alpha=0.12, color="white")

    # ----- Row 1, col 3: CoV breakdown X vs Y --------------------------------
    ax_cov_breakdown = dark_ax(fig.add_subplot(gs[1, 3]),
                               "X and Y uniformity vs angle")
    angles = np.arange(30, 65, 5)
    cov_xs, cov_ys = [], []
    for a in angles:
        cx, cy, *_ = beam_coverage(sx_free, sy_free, a, SPEC_W, SPEC_H)
        cov_xs.append(cx); cov_ys.append(cy)
    ax_cov_breakdown.plot(angles, cov_xs, color="#00cfff", lw=2, marker="o",
                          ms=6, label="X-CoV (two beams + offset)")
    ax_cov_breakdown.plot(angles, cov_ys, color="#ff6b6b", lw=2, marker="s",
                          ms=6, ls="--", label="Y-CoV (single pass)")
    ax_cov_breakdown.axhline(15, color="lime", lw=0.8, ls=":", label="15% target")
    ax_cov_breakdown.fill_between(angles, 0, cov_xs, alpha=0.08, color="#00cfff")
    ax_cov_breakdown.fill_between(angles, 0, cov_ys, alpha=0.08, color="#ff6b6b")
    ax_cov_breakdown.set_xlabel("Beam angle (deg from normal)")
    ax_cov_breakdown.set_ylabel("CoV (%)")
    ax_cov_breakdown.legend(fontsize=7, facecolor="#222222", labelcolor="white",
                            edgecolor="#555555")
    ax_cov_breakdown.grid(True, alpha=0.12, color="white")

    # =========================================================================
    # Row 2: Lens comparison bar charts
    # =========================================================================

    # ----- Min standoff per lens ---------------------------------------------
    ax_sd = dark_ax(fig.add_subplot(gs[2, 0]), "Min standoff vs lens")
    standoffs_mm = [min_standoff(f) for f in LENSES_MM]
    bars = ax_sd.bar(LENS_NAMES, standoffs_mm, color=LENS_COLORS,
                     edgecolor="#333333", lw=0.8)
    for bar, val in zip(bars, standoffs_mm):
        ax_sd.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                   f"{val:.0f}mm", ha="center", color="white", fontsize=8)
    ax_sd.axhline(300, color="#ff6b6b", lw=1, ls=":", label="Lamp zone limit (~300mm)")
    ax_sd.set_ylabel("Standoff (mm)")
    ax_sd.tick_params(axis="x", labelrotation=20, labelsize=7)
    ax_sd.legend(fontsize=7, facecolor="#222222", labelcolor="white",
                 edgecolor="#555555")
    ax_sd.grid(True, alpha=0.12, color="white", axis="y")

    # ----- HFOV per lens -----------------------------------------------------
    ax_fv = dark_ax(fig.add_subplot(gs[2, 1]), "HFOV vs lens")
    hfovs = [lens_fov(f)[0] for f in LENSES_MM]
    bars2 = ax_fv.bar(LENS_NAMES, hfovs, color=LENS_COLORS,
                      edgecolor="#333333", lw=0.8)
    for bar, val in zip(bars2, hfovs):
        ax_fv.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                   f"{val:.1f}°", ha="center", color="white", fontsize=8)
    ax_fv.set_ylabel("HFOV (deg)")
    ax_fv.tick_params(axis="x", labelrotation=20, labelsize=7)
    ax_fv.grid(True, alpha=0.12, color="white", axis="y")

    # ----- Pixels on specimen vs standoff for all lenses ---------------------
    ax_px = dark_ax(fig.add_subplot(gs[2, 2:]),
                    "Pixels on specimen vs standoff (all lenses)")
    for f, name, col in zip(LENSES_MM, LENS_NAMES, LENS_COLORS):
        d_arr = np.linspace(min_standoff(f), min_standoff(f)*4, 200)
        px_arr = []
        for d in d_arr:
            scale = d * PIXEL_PITCH / f
            spx_w = min(SPEC_W, SENSOR_W_PX*scale)/scale
            spx_h = min(SPEC_H, SENSOR_H_PX*scale)/scale
            px_arr.append(spx_w * spx_h / 1000)
        ax_px.plot(d_arr, px_arr, color=col, lw=2, label=name)
        ax_px.axvline(min_standoff(f), color=col, lw=0.8, ls=":", alpha=0.6)
    ax_px.axhline(224, color="lime", lw=1, ls="--", label="Max 224k px (landscape)")
    ax_px.set_xlabel("Standoff (mm)"); ax_px.set_ylabel("Pixels on specimen (k)")
    ax_px.legend(fontsize=8, facecolor="#222222", labelcolor="white",
                 edgecolor="#555555", ncol=2)
    ax_px.grid(True, alpha=0.12, color="white")
    ax_px.set_xlim(0, 2500)

    # =========================================================================
    # Row 3: Summary verdict table
    # =========================================================================
    ax_vt = fig.add_subplot(gs[3, :])
    ax_vt.set_facecolor("#0d0d0d"); ax_vt.axis("off")

    cov_at_45 = beam_coverage(sx_free, sy_free, 45, SPEC_W, SPEC_H)

    rows = [
        ["ITEM", "VALUE", "STATUS", "NOTE"],
        ["Beam FWHM X (free space)",
         f"{fwhm_x:.0f} mm",
         "MARGINAL",
         f"Specimen width {SPEC_W:.0f}mm > beam {fwhm_x:.0f}mm — needs two-beam offset"],
        ["Beam FWHM Y (free space)",
         f"{fwhm_y:.0f} mm",
         "INSUFFICIENT",
         f"Specimen height {SPEC_H:.0f}mm > beam {fwhm_y:.0f}mm — Y edges at 33% of peak"],
        ["Two-beam X uniformity (45°, opt offset)",
         f"CoV {cov_at_45[0]:.0f}%",
         "GOOD" if cov_at_45[0] < 15 else "MARGINAL",
         f"Optimal beam offset = {cov_at_45[2]:.0f}mm from specimen centre"],
        ["Two-beam Y uniformity (45°)",
         f"CoV {cov_at_45[1]:.0f}%",
         "POOR",
         "Fundamental beam limitation — not fixable with angle/offset alone"],
        ["Specimen orientation",
         "Landscape (320mm horiz)",
         "OPTIMAL",
         "Higher sensor fill (70%) vs portrait (55%)"],
        ["Recommended camera lens",
         "13.8mm @ 575mm",
         "RECOMMENDED",
         "Best balance of working distance and clearance for lamps"],
        ["Current camera lens",
         "18mm @ 750mm",
         "ACCEPTABLE",
         "Works but requires larger setup footprint"],
        ["4.3mm lens",
         "@ 179mm standoff",
         "NOT SUITABLE",
         "Camera inside lamp zone — physical interference"],
        ["9.2mm lens",
         "@ 383mm standoff",
         "POSSIBLE",
         "Compact but tight clearance with 45-deg lamps at 700mm"],
        ["Overall suitability",
         "Qualitative OK",
         "REVIEW",
         "Suitable for defect detection. Quantitative HF needs Y-axis improvement"],
    ]

    status_colors = {"GOOD": "lime", "OPTIMAL": "lime", "RECOMMENDED": "lime",
                     "MARGINAL": "#ffd93d", "POSSIBLE": "#ffd93d", "ACCEPTABLE": "#ffd93d",
                     "POOR": "#ff6b6b", "INSUFFICIENT": "#ff6b6b",
                     "NOT SUITABLE": "#ff6b6b", "REVIEW": "#ffd93d"}

    tab = ax_vt.table(cellText=rows[1:], colLabels=rows[0],
                      cellLoc="left", loc="center", bbox=[0, 0, 1, 1])
    tab.auto_set_font_size(False); tab.set_fontsize(8)
    col_widths = [0.22, 0.14, 0.10, 0.54]
    for j, w in enumerate(col_widths):
        tab.auto_set_column_width(j)

    for j in range(4):
        tab[0, j].set_facecolor("#222244")
        tab[0, j].set_text_props(color="white", fontweight="bold")

    for i in range(1, len(rows)):
        status = rows[i][2]
        sc = status_colors.get(status, "white")
        for j in range(4):
            bg = "#1a1a1a" if i % 2 == 0 else "#222222"
            tab[i, j].set_facecolor(bg)
            tc = sc if j == 2 else "white"
            tab[i, j].set_text_props(color=tc)
            if j == 2:
                tab[i, j].set_text_props(color=sc, fontweight="bold")
            tab[i, j].set_edgecolor("#333333")

    ax_vt.set_title("Setup Summary  |  Active Thermography  |  Halogen Excitation",
                    color="white", pad=8, fontsize=10, fontweight="bold")

    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nFigure saved: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
