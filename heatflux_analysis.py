"""
Single-beam heat-flux characterisation.

Loads a Boson thermal TIFF sequence, subtracts a pre-lamp baseline,
fits a single rotated 2D Gaussian, and produces a visual report:
differential image, 3-D surface, residuals, H/V cross-sections,
radial profile, encircled energy, and a parameter table.

Usage:
    python heatflux_analysis.py [tiff_dir] [session_dir]
    python heatflux_analysis.py            # picks latest in OUTPUT_DIR
"""

import sys
import os
import config
import glob
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter

# -- Config -------------------------------------------------------------------
OUTPUT_DIR       = config.BOSON_ROOT + r""
CAMERA_FPS       = 9.0
PRE_RELAY_SECS   = 1.0
RELAY_PULSE_SECS = 1.0
POST_RELAY_SECS  = 10.0

STEADY_START_SECS = PRE_RELAY_SECS + 2.0
STEADY_END_SECS   = PRE_RELAY_SECS + RELAY_PULSE_SECS + POST_RELAY_SECS - 1.0

BEAM_COLOR = "#00cfff"


# -- Model --------------------------------------------------------------------
def rotated_gaussian(xy, A, x0, y0, sx, sy, theta, bg):
    x, y = xy
    ct, st = np.cos(theta), np.sin(theta)
    xr =  ct * (x - x0) + st * (y - y0)
    yr = -st * (x - x0) + ct * (y - y0)
    return A * np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2)) + bg

def fwhm(sigma):
    return 2.3548 * sigma


# -- Data loading -------------------------------------------------------------
def load_tiff_sequence(directory):
    paths = sorted(glob.glob(os.path.join(directory, "frame_*.tiff")))
    if not paths:
        raise RuntimeError(f"No TIFF frames found in {directory}")
    frames = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if len(img.shape) == 3:
            img = img[:, :, 0]
        frames.append(img.astype(np.float32))
    return np.array(frames)

def load_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if len(f.shape) == 3 else f
        frames.append(gray.astype(np.float32))
    cap.release()
    return np.array(frames)

def steady_mean(frames, fps):
    i0 = int(STEADY_START_SECS * fps)
    i1 = min(int(STEADY_END_SECS * fps), len(frames))
    if i0 >= i1:
        i0, i1 = len(frames) // 4, len(frames)
    print(f"Steady-state: frames {i0}-{i1-1}  ({i1-i0} frames, {(i1-i0)/fps:.1f} s)")
    return frames[i0:i1].mean(axis=0)

def baseline_mean(frames, fps):
    i1 = max(1, int(PRE_RELAY_SECS * fps) - 1)
    print(f"Baseline: frames 0-{i1-1}  ({i1} frames)")
    return frames[0:i1].mean(axis=0)


# -- Fitting ------------------------------------------------------------------
def fit_single_gaussian(img):
    H, W = img.shape
    xx, yy = np.meshgrid(np.arange(W, dtype=np.float64),
                         np.arange(H, dtype=np.float64))
    xy = (xx.ravel(), yy.ravel())
    z  = img.ravel().astype(np.float64)

    blurred = gaussian_filter(img, sigma=15)
    peak_idx = np.unravel_index(np.argmax(blurred), blurred.shape)
    x0_s, y0_s = float(peak_idx[1]), float(peak_idx[0])
    vmax = float(img.max())
    bg0  = float(np.percentile(img, 5))
    s0   = min(W, H) / 5.0

    p0 = [vmax - bg0, x0_s, y0_s, s0, s0, 0.0, bg0]
    lo = [0,      0, 0, 1, 1, -np.pi / 2, -abs(bg0) * 3]
    hi = [vmax*3, W, H, W, H,  np.pi / 2,  vmax]

    popt, pcov = curve_fit(rotated_gaussian, xy, z,
                           p0=p0, bounds=(lo, hi), maxfev=20000)
    perr   = np.sqrt(np.diag(pcov))
    fitted = rotated_gaussian(xy, *popt).reshape(H, W)
    resid  = img.astype(np.float64) - fitted
    return popt, perr, fitted, resid


# -- Beam metrics -------------------------------------------------------------
def radial_profile(img, cx, cy, max_r):
    """Azimuthally averaged radial profile centred on (cx, cy)."""
    H, W = img.shape
    yy, xx = np.ogrid[:H, :W]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).ravel()
    v = img.ravel()
    bins   = np.arange(0, max_r + 1, 1.0)
    counts, _ = np.histogram(r, bins=bins)
    total,  _ = np.histogram(r, bins=bins, weights=v)
    mean_r = np.where(counts > 0, total / counts, 0.0)
    return bins[:-1], mean_r

def encircled_energy(img, cx, cy, max_r):
    """Fraction of total (positive) beam energy within radius r."""
    H, W = img.shape
    yy, xx = np.ogrid[:H, :W]
    r   = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    pos = np.clip(img, 0, None)
    tot = pos.sum()
    radii = np.arange(0, max_r + 1, 1.0)
    ee = np.array([(pos[r <= ri]).sum() / tot for ri in radii])
    return radii, ee

def beam_mask_1sigma(H, W, x0, y0, sx, sy, theta):
    yy, xx = np.ogrid[:H, :W]
    ct, st = np.cos(theta), np.sin(theta)
    xr =  ct * (xx - x0) + st * (yy - y0)
    yr = -st * (xx - x0) + ct * (yy - y0)
    return (xr / sx) ** 2 + (yr / sy) ** 2 <= 1.0


# -- Plot helpers -------------------------------------------------------------
def dark_ax(ax):
    ax.set_facecolor("#1a1a1a")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    ax.xaxis.label.set_color("#aaaaaa")
    ax.yaxis.label.set_color("#aaaaaa")
    ax.title.set_color("white")
    return ax


# -- Main ---------------------------------------------------------------------
def analyse(path, fps=CAMERA_FPS, session_dir=None):
    print(f"\n{'='*60}")
    print(f"Analysing: {path}")
    print(f"{'='*60}")

    frames = load_tiff_sequence(path) if os.path.isdir(path) else load_video(path)
    if len(frames) == 0:
        raise RuntimeError("No frames read.")
    print(f"Loaded {len(frames)} frames  ({frames.shape[2]}x{frames.shape[1]} px)")

    mean_img = steady_mean(frames, fps)
    base_img = baseline_mean(frames, fps)
    diff_img = mean_img - base_img

    print(f"Differential stats:  min={diff_img.min():.2f}  "
          f"max={diff_img.max():.2f}  mean={diff_img.mean():.2f}")

    fit_input = diff_img - diff_img.min()

    print("Fitting single-Gaussian model ...")
    popt, perr, fitted, residuals = fit_single_gaussian(fit_input)
    A, x0, y0, sx, sy, theta, bg = popt

    H, W   = diff_img.shape
    fwhm_x = fwhm(sx)
    fwhm_y = fwhm(sy)

    print(f"\n-- Beam --------------------------------------------------")
    print(f"  Centre (px)        : ({x0:.1f}, {y0:.1f})")
    print(f"  Peak amplitude     : {A:.2f}  (background={bg:.2f})")
    print(f"  sigma_x            : {sx:.1f} px   FWHM {fwhm_x:.1f} px")
    print(f"  sigma_y            : {sy:.1f} px   FWHM {fwhm_y:.1f} px")
    print(f"  Rotation           : {np.degrees(theta):.1f} deg")
    print(f"  Aspect ratio sx/sy : {sx/sy:.3f}")

    rms    = np.sqrt((residuals ** 2).mean())
    signal = float(fit_input.max())
    print(f"\n-- Fit quality -------------------------------------------")
    print(f"  Residual RMS       : {rms:.2f} counts")
    print(f"  RMS / peak         : {rms/max(signal,1)*100:.2f}%")

    max_r = max(1, int(min(x0, y0, W - x0, H - y0)) - 1)
    r_arr, rp_data = radial_profile(diff_img, x0, y0, max_r)
    _,     rp_fit  = radial_profile(fitted,   x0, y0, max_r)
    r_ee,  ee      = encircled_energy(diff_img, x0, y0, max_r)

    d50_r = float(r_ee[np.searchsorted(ee, 0.50)]) if ee[-1] >= 0.50 else float(r_ee[-1])
    d86_r = float(r_ee[np.searchsorted(ee, 0.86)]) if ee[-1] >= 0.86 else float(r_ee[-1])
    print(f"  D50 radius         : {d50_r:.1f} px")
    print(f"  D86 radius (1/e²)  : {d86_r:.1f} px")

    mask      = beam_mask_1sigma(H, W, x0, y0, sx, sy, theta)
    beam_vals = diff_img[mask]
    bv_mean   = float(beam_vals.mean())
    cov = beam_vals.std() / bv_mean * 100.0 if bv_mean != 0 else 0.0
    p2v = (beam_vals.max() - beam_vals.min()) / bv_mean * 100.0 if bv_mean != 0 else 0.0
    print(f"\n-- Beam uniformity (within 1-sigma ellipse) --------------")
    print(f"  Mean dT            : {bv_mean:.2f} counts")
    print(f"  CoV                : {cov:.2f}%  (lower = more uniform)")
    print(f"  P2V                : {p2v:.2f}%")

    # =========================================================================
    # FIGURE  —  3 rows × 3 cols
    # Row 0: diff + ellipses | 3-D surface | residuals
    # Row 1: H cross-section (2 cols wide) | V cross-section
    # Row 2: radial profile  | encircled energy | parameter table
    # =========================================================================
    c = BEAM_COLOR
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor("#111111")
    fig.suptitle(
        f"Beam Characterisation  |  {os.path.basename(path)}",
        fontsize=13, color="white", y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35,
                           left=0.06, right=0.97, top=0.94, bottom=0.05)
    kw  = dict(cmap="inferno", origin="upper")
    kwr = dict(cmap="RdBu_r",  origin="upper")

    # ----- Panel 0,0 : Differential image + beam ellipses -------------------
    ax0 = dark_ax(fig.add_subplot(gs[0, 0]))
    im0 = ax0.imshow(diff_img, **kw)
    cb0 = plt.colorbar(im0, ax=ax0, fraction=0.04)
    cb0.set_label("dT (counts)", color="#aaaaaa", fontsize=8)
    cb0.ax.yaxis.set_tick_params(color="#aaaaaa", labelsize=7)

    e_fwhm = Ellipse(xy=(x0, y0), width=fwhm_x, height=fwhm_y,
                     angle=np.degrees(theta),
                     edgecolor=c, facecolor="none", lw=2, ls="--", zorder=5)
    e_1sig = Ellipse(xy=(x0, y0), width=2*sx, height=2*sy,
                     angle=np.degrees(theta),
                     edgecolor="lime", facecolor="none", lw=1.2, ls=":", zorder=5)
    ax0.add_patch(e_fwhm)
    ax0.add_patch(e_1sig)
    ax0.plot(x0, y0, "+", color=c, ms=14, mew=2.5, zorder=6)
    ax0.legend(handles=[
        mpatches.Patch(ec=c,      fc="none", ls="--", lw=2,   label=f"FWHM  ({fwhm_x:.0f}×{fwhm_y:.0f} px)"),
        mpatches.Patch(ec="lime", fc="none", ls=":",  lw=1.2, label="1-sigma"),
    ], fontsize=7, facecolor="#222222", labelcolor="white", edgecolor="#555555", loc="lower right")
    ax0.set_title("Differential  +  beam ellipses")
    ax0.set_xlim(0, W); ax0.set_ylim(H, 0)

    # ----- Panel 0,1 : 3-D surface -------------------------------------------
    ax3d = fig.add_subplot(gs[0, 1], projection="3d")
    ax3d.set_facecolor("#1a1a1a")
    ax3d.tick_params(colors="#aaaaaa", labelsize=7)
    for attr in ("xaxis", "yaxis", "zaxis"):
        getattr(ax3d, attr).label.set_color("#aaaaaa")
    step = 8
    xs = np.arange(0, W, step); ys = np.arange(0, H, step)
    Xs, Ys = np.meshgrid(xs, ys)
    ax3d.plot_surface(Xs, Ys, fitted[::step, ::step],
                      cmap="inferno", linewidth=0, antialiased=True, alpha=0.92)
    z_peak = float(fitted[int(np.clip(y0, 0, H-1)), int(np.clip(x0, 0, W-1))])
    ax3d.scatter([x0], [y0], [z_peak], color=c, s=60, zorder=10)
    ax3d.set_xlabel("X (px)", fontsize=8); ax3d.set_ylabel("Y (px)", fontsize=8)
    ax3d.set_zlabel("dT", fontsize=8)
    ax3d.set_title("3-D Gaussian fit", color="white")
    ax3d.view_init(elev=30, azim=-60)

    # ----- Panel 0,2 : Residuals ---------------------------------------------
    ax2 = dark_ax(fig.add_subplot(gs[0, 2]))
    lim = max(abs(residuals.min()), abs(residuals.max()))
    im2 = ax2.imshow(residuals, **kwr, vmin=-lim, vmax=lim)
    cb2 = plt.colorbar(im2, ax=ax2, fraction=0.04)
    cb2.set_label("residual (counts)", color="#aaaaaa", fontsize=8)
    cb2.ax.yaxis.set_tick_params(color="#aaaaaa", labelsize=7)
    ax2.set_title(f"Residuals  ({rms/max(signal,1)*100:.1f}% of peak)")
    ax2.text(0.02, 0.97, f"RMS={rms:.2f}", transform=ax2.transAxes,
             va="top", color="white", fontsize=8,
             bbox=dict(fc="black", alpha=0.5, pad=3))

    # ----- Panel 1,0-1 : Horizontal cross-section ----------------------------
    ax_h = dark_ax(fig.add_subplot(gs[1, :2]))
    row  = int(np.clip(y0, 0, H-1))
    ax_h.plot(diff_img[row, :], color=c, lw=1.8, alpha=0.85, label=f"Data (row {row})")
    ax_h.plot(fitted[row, :],   color=c, lw=1.4, ls="--", alpha=0.6, label="Gaussian fit")
    pk_h   = float(fitted[row, int(np.clip(x0, 0, W-1))])
    half_x = fwhm_x / 2.0
    ax_h.annotate("", xy=(x0 + half_x, pk_h * 0.5), xytext=(x0 - half_x, pk_h * 0.5),
                  arrowprops=dict(arrowstyle="<->", color=c, lw=1.8))
    ax_h.text(x0, pk_h * 0.55, f"FWHM={fwhm_x:.0f}px",
              ha="center", va="bottom", color=c, fontsize=8)
    ax_h.axhline(pk_h * 0.5, color=c, lw=0.8, ls=":", alpha=0.5)
    ax_h.axvline(x0, color=c, lw=0.8, ls=":", alpha=0.4)
    ax_h.set_xlabel("Column (px)"); ax_h.set_ylabel("dT (counts)")
    ax_h.set_title("Horizontal cross-section through beam centre  (dashed = Gaussian fit)")
    ax_h.legend(fontsize=8, facecolor="#222222", labelcolor="white", edgecolor="#555555")
    ax_h.grid(True, alpha=0.15, color="white")

    # ----- Panel 1,2 : Vertical cross-section --------------------------------
    ax_v = dark_ax(fig.add_subplot(gs[1, 2]))
    col  = int(np.clip(x0, 0, W-1))
    ax_v.plot(diff_img[:, col], np.arange(H), color=c, lw=1.8, alpha=0.85, label=f"Data (col {col})")
    ax_v.plot(fitted[:, col],   np.arange(H), color=c, lw=1.4, ls="--", alpha=0.6, label="Gaussian fit")
    half_y = fwhm_y / 2.0
    ax_v.axhline(y0 - half_y, color=c, lw=0.8, ls=":", alpha=0.5)
    ax_v.axhline(y0 + half_y, color=c, lw=0.8, ls=":", alpha=0.5)
    ax_v.annotate("", xy=(0, y0 + half_y), xytext=(0, y0 - half_y),
                  arrowprops=dict(arrowstyle="<->", color=c, lw=1.6))
    ax_v.invert_yaxis()
    ax_v.set_xlabel("dT (counts)"); ax_v.set_ylabel("Row (px)")
    ax_v.set_title("Vertical cross-section")
    ax_v.legend(fontsize=7, facecolor="#222222", labelcolor="white", edgecolor="#555555")
    ax_v.grid(True, alpha=0.15, color="white")

    # ----- Panel 2,0 : Radial profile ----------------------------------------
    ax_r = dark_ax(fig.add_subplot(gs[2, 0]))
    ax_r.plot(r_arr, rp_data, color=c,        lw=1.8, alpha=0.85, label="Data (azim. avg.)")
    ax_r.plot(r_arr, rp_fit,  color=c,        lw=1.4, ls="--", alpha=0.6, label="Gaussian fit")
    ax_r.axvline(d50_r, color="lime",    lw=1, ls=":", label=f"D50 = {d50_r:.0f} px")
    ax_r.axvline(d86_r, color="#ffaa00", lw=1, ls=":", label=f"D86 = {d86_r:.0f} px")
    ax_r.set_xlabel("Radius from centre (px)"); ax_r.set_ylabel("Mean dT (counts)")
    ax_r.set_title("Radial profile (azimuthal average)")
    ax_r.legend(fontsize=8, facecolor="#222222", labelcolor="white", edgecolor="#555555")
    ax_r.grid(True, alpha=0.15, color="white")

    # ----- Panel 2,1 : Encircled energy --------------------------------------
    ax_e = dark_ax(fig.add_subplot(gs[2, 1]))
    ax_e.plot(r_ee, ee * 100, color="#ffaa00", lw=2)
    ax_e.axhline(50, color="lime",    lw=0.8, ls=":", alpha=0.7)
    ax_e.axhline(86, color="#ffaa00", lw=0.8, ls=":", alpha=0.7)
    ax_e.axvline(d50_r, color="lime",    lw=1, ls=":", label=f"D50 = {d50_r:.0f} px")
    ax_e.axvline(d86_r, color="#ffaa00", lw=1, ls=":", label=f"D86 = {d86_r:.0f} px  (1/e²)")
    ax_e.set_xlabel("Radius from centre (px)"); ax_e.set_ylabel("Encircled energy (%)")
    ax_e.set_title("Encircled energy")
    ax_e.set_ylim(0, 105)
    ax_e.legend(fontsize=8, facecolor="#222222", labelcolor="white", edgecolor="#555555")
    ax_e.grid(True, alpha=0.15, color="white")

    # ----- Panel 2,2 : Parameter table ---------------------------------------
    ax_tab = fig.add_subplot(gs[2, 2])
    ax_tab.set_facecolor("#111111"); ax_tab.axis("off")

    rows = [
        ["Parameter",          "Value"],
        ["Centre X (px)",      f"{x0:.1f}"],
        ["Centre Y (px)",      f"{y0:.1f}"],
        ["Peak amplitude",     f"{A:.2f} counts"],
        ["Background",         f"{bg:.2f} counts"],
        ["FWHM x",             f"{fwhm_x:.1f} px"],
        ["FWHM y",             f"{fwhm_y:.1f} px"],
        ["Rotation",           f"{np.degrees(theta):.1f} deg"],
        ["Aspect ratio sx/sy", f"{sx/sy:.3f}"],
        ["D50 radius",         f"{d50_r:.1f} px"],
        ["D86 radius (1/e²)",  f"{d86_r:.1f} px"],
        ["Beam CoV",           f"{cov:.2f}%"],
        ["Beam P2V",           f"{p2v:.2f}%"],
        ["Residual RMS",       f"{rms:.2f} ({rms/max(signal,1)*100:.1f}% peak)"],
    ]

    tab = ax_tab.table(cellText=rows[1:], colLabels=rows[0],
                       cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tab.auto_set_font_size(False); tab.set_fontsize(8.5)
    for j in range(2):
        tab[0, j].set_facecolor("#222244")
        tab[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows)):
        for j in range(2):
            tab[i, j].set_facecolor("#1a1a1a" if i % 2 == 0 else "#222222")
            tab[i, j].set_text_props(color=c if j == 1 else "white")
            tab[i, j].set_edgecolor("#333333")
    ax_tab.set_title("Beam parameters", color="white", pad=8, fontsize=10)

    # -- Save -----------------------------------------------------------------
    out_dir  = session_dir if session_dir else os.path.dirname(path)
    base     = os.path.splitext(os.path.basename(path))[0]
    out_png  = os.path.join(out_dir, f"{base}_heatflux_analysis.png")
    out_json = os.path.join(out_dir, "summary.json")

    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nFigure saved : {out_png}")

    results = {
        "video":   path,
        "session": out_dir,
        "beam": dict(
            cx=round(x0, 1), cy=round(y0, 1),
            amplitude=round(float(A), 3),
            background=round(float(bg), 3),
            sigma_x=round(float(sx), 1),   sigma_y=round(float(sy), 1),
            fwhm_x=round(fwhm_x, 1),       fwhm_y=round(fwhm_y, 1),
            theta_deg=round(float(np.degrees(theta)), 1),
            aspect_ratio=round(float(sx / sy), 3),
            d50_px=round(d50_r, 1),
            d86_px=round(d86_r, 1),
        ),
        "uniformity": dict(
            cov_pct=round(cov, 2),
            p2v_pct=round(p2v, 2),
            mean_dt=round(bv_mean, 2),
        ),
        "fit": dict(
            residual_rms=round(rms, 3),
            rms_frac_pct=round(rms / max(signal, 1) * 100, 2),
        ),
    }

    class _NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.integer):  return int(obj)
            return super().default(obj)

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, cls=_NpEncoder)
    print(f"Summary saved: {out_json}")
    return results


if __name__ == "__main__":
    path        = sys.argv[1] if len(sys.argv) > 1 else None
    session_dir = sys.argv[2] if len(sys.argv) > 2 else None

    if path is None:
        dirs = sorted(glob.glob(os.path.join(OUTPUT_DIR, "**", "boson_*"), recursive=True))
        dirs = [d for d in dirs if os.path.isdir(d)]
        if dirs:
            path = dirs[-1]
        else:
            files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "**", "boson_*.avi"), recursive=True))
            if not files:
                print(f"No captures found under {OUTPUT_DIR}")
                sys.exit(1)
            path = files[-1]
        print(f"Auto-selected: {path}")

    analyse(path, session_dir=session_dir)
