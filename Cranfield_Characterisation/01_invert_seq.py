"""
Beam shape analysis for FLIR A655sc .seq captures at 300, 400, 500 mm.

Adapted from beam_shape_analysis.py (Boson TIFF pipeline). Reads FLIR .seq via
flirpy, converts radiometric counts to Kelvin (count * 0.04), subtracts baseline,
fits a 2D rotated Gaussian to dT(x,y), and produces a multi-panel figure
mirroring the existing analysis style.

Camera:  FLIR A655sc, 640x480, 17 um pitch
Lens:    FOL18 / T199105 -> 45 deg x 33.7 deg HFOV/VFOV  (=> ~13.1 mm effective focal
         from sensor geometry; FLIR nomenclature uses "18 mm" but the FOV
         spec is what governs mm/px scaling)
"""

import os, glob, tempfile, json, shutil
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
from matplotlib.patches import Ellipse
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from flirpy.io.seq import Splitter

# ---- I/O --------------------------------------------------------------------
REPO_ROOT  = os.path.dirname(os.path.abspath(__file__))
SEQ_ROOT   = os.path.join(REPO_ROOT, "wetransfer_300mm-seq_2026-05-13_1259")
OUTPUT_PNG = os.path.join(REPO_ROOT, "beam_shape_analysis_seq.png")
RESULTS_JSON = os.path.join(REPO_ROOT, "beam_shape_analysis_seq.json")
CACHE_DIR = os.path.join(REPO_ROOT, "_seq_cache")  # extracted frames cached here

SESSIONS = [
    (300, os.path.join(SEQ_ROOT, "300mm.seq")),
    (400, os.path.join(SEQ_ROOT, "400mm.seq")),
    (500, os.path.join(SEQ_ROOT, "500mm.seq")),
]
SESSION_COLS = ["#6bcb77", "#ffd93d", "#ff6b6b"]

# ---- Camera geometry --------------------------------------------------------
SENSOR_W_PX = 640
SENSOR_H_PX = 480
HFOV_DEG    = 45.0                # FOL18 / T199105 on A655sc
VFOV_DEG    = 33.7

def mm_per_px(d_mm):
    return 2 * d_mm * np.tan(np.radians(HFOV_DEG / 2)) / SENSOR_W_PX

# ---- FLIR radiometric scaling ----------------------------------------------
COUNT_TO_K = 0.04                  # T_K = count * 0.04 (T_K*25 stored as uint16)
FPS_DEFAULT = 8.0                  # from metadata Frame Rate
N_BASELINE = 5                     # frames averaged for baseline
N_HOT      = 5                     # frames averaged for hot (last N)

# ---- .seq loading via flirpy + cache ---------------------------------------

def load_seq_frames(seq_path, cache_root):
    """Returns frames as float32 array in Kelvin, shape (N, H, W).
    Uses on-disk cache to avoid re-running flirpy on each call."""
    base = os.path.splitext(os.path.basename(seq_path))[0]
    cache_dir = os.path.join(cache_root, base)
    cache_npy = os.path.join(cache_dir, "frames_K.npy")
    if os.path.exists(cache_npy):
        return np.load(cache_npy)
    os.makedirs(cache_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        sp = Splitter(tmp, split_folders=False)
        sp.process([seq_path])
        rad_dir = os.path.join(tmp, "radiometric")
        tiffs = sorted(glob.glob(os.path.join(rad_dir, "*.tiff")))
        frames = []
        for t in tiffs:
            img = cv2.imread(t, cv2.IMREAD_UNCHANGED)
            if img is not None:
                frames.append(img.astype(np.float32) * COUNT_TO_K)
        frames = np.array(frames, dtype=np.float32)
    np.save(cache_npy, frames)
    return frames

# ---- 2D rotated Gaussian and fitter ----------------------------------------

def rotated_gaussian(xy, A, x0, y0, sx, sy, theta, bg):
    x, y = xy
    ct, st = np.cos(theta), np.sin(theta)
    xr =  ct*(x-x0) + st*(y-y0)
    yr = -st*(x-x0) + ct*(y-y0)
    return A*np.exp(-0.5*((xr/sx)**2 + (yr/sy)**2)) + bg

def fit_gaussian(img):
    H, W = img.shape
    xx, yy = np.meshgrid(np.arange(W, dtype=np.float64),
                         np.arange(H, dtype=np.float64))
    blurred = gaussian_filter(img, sigma=10)
    pk = np.unravel_index(np.argmax(blurred), blurred.shape)
    x0s, y0s = float(pk[1]), float(pk[0])
    vmax = float(img.max()); bg0 = float(np.percentile(img, 5))
    s0 = min(W, H) / 6
    p0 = [max(vmax - bg0, 1e-3), x0s, y0s, s0, s0, 0.0, bg0]
    lo = [0, 0, 0, 1, 1, -np.pi/2, -abs(bg0)*3 - 1]
    hi = [vmax*3 + 1, W, H, W, H, np.pi/2, vmax + 1]
    popt, pcov = curve_fit(rotated_gaussian, (xx.ravel(), yy.ravel()),
                           img.ravel().astype(np.float64),
                           p0=p0, bounds=(lo, hi), maxfev=20000)
    fitted = rotated_gaussian((xx.ravel(), yy.ravel()), *popt).reshape(H, W)
    return popt, np.sqrt(np.diag(pcov)), fitted, img - fitted

def radial_profile(img, cx, cy, max_r):
    H, W = img.shape
    yy, xx = np.ogrid[:H, :W]
    r = np.sqrt((xx-cx)**2 + (yy-cy)**2).ravel()
    v = img.ravel()
    bins = np.arange(0, max_r + 1, 1.0)
    cnt, _ = np.histogram(r, bins=bins)
    tot, _ = np.histogram(r, bins=bins, weights=v)
    return bins[:-1], np.where(cnt > 0, tot/cnt, 0.0)

def encircled_energy(img, cx, cy, max_r):
    H, W = img.shape
    yy, xx = np.ogrid[:H, :W]
    r = np.sqrt((xx-cx)**2 + (yy-cy)**2)
    pos = np.clip(img, 0, None); tot = pos.sum()
    rad = np.arange(0, max_r + 1, 1.0)
    ee = np.array([(pos[r <= ri]).sum()/tot for ri in rad])
    return rad, ee

def dark_ax(ax, grid=True):
    ax.set_facecolor("#1a1a1a")
    for sp in ax.spines.values(): sp.set_edgecolor("#444444")
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    if grid: ax.grid(True, alpha=0.12, color="white")

# ---- Process all sessions ---------------------------------------------------
results = []
for standoff, seq_path in SESSIONS:
    mpp = mm_per_px(standoff)
    print(f"\n{'='*60}\n{standoff} mm  ({mpp:.4f} mm/px, FOV {mpp*SENSOR_W_PX:.0f} x {mpp*SENSOR_H_PX:.0f} mm)")
    print(f"  loading {seq_path}")
    frames_K = load_seq_frames(seq_path, CACHE_DIR)
    N = len(frames_K)
    print(f"  frames={N}, shape={frames_K[0].shape}, fps={FPS_DEFAULT}")

    baseline = frames_K[:N_BASELINE].mean(axis=0)
    hot      = frames_K[-N_HOT:].mean(axis=0)
    diff     = hot - baseline                      # delta T in K
    fit_in   = diff - diff.min()                   # positive-definite for fit

    # frame-mean time series for diagnostics
    fm = frames_K.reshape(N, -1).mean(axis=1) - frames_K[0].reshape(-1).mean()
    print(f"  baseline mean = {baseline.mean():.2f} K  ({baseline.mean()-273.15:.1f} C)")
    print(f"  hot mean      = {hot.mean():.2f} K       ({hot.mean()-273.15:.1f} C)")
    print(f"  peak dT       = {diff.max():.3f} K")

    popt, perr, fitted, resid = fit_gaussian(fit_in)
    A, x0, y0, sx, sy, theta, bg = popt
    if sy > sx:
        sx, sy = sy, sx
        theta += np.pi/2

    sx_mm  = sx * mpp
    sy_mm  = sy * mpp
    fwhm_x = sx * 2.355 * mpp
    fwhm_y = sy * 2.355 * mpp
    peak_dT = float(diff.max())
    fit_peak_dT = float(A)
    print(f"  centre  = ({x0:.1f}, {y0:.1f}) px")
    print(f"  sigma_x = {sx:.1f}px = {sx_mm:.1f}mm  FWHM_x = {fwhm_x:.1f}mm")
    print(f"  sigma_y = {sy:.1f}px = {sy_mm:.1f}mm  FWHM_y = {fwhm_y:.1f}mm")
    print(f"  theta   = {np.degrees(theta):.1f} deg     aspect = {sx/sy:.3f}")

    max_r = int(min(SENSOR_W_PX, SENSOR_H_PX) / 2)
    r_bins, r_mean = radial_profile(fit_in, x0, y0, max_r)
    r_rad, r_ee   = encircled_energy(fit_in, x0, y0, max_r)

    results.append(dict(
        standoff=standoff, mpp=mpp,
        diff=diff, fit_in=fit_in, fitted=fitted, resid=resid,
        A=A, x0=x0, y0=y0, sx=sx, sy=sy, theta=theta, bg=bg,
        sx_mm=sx_mm, sy_mm=sy_mm, fwhm_x=fwhm_x, fwhm_y=fwhm_y,
        peak_dT=peak_dT, fit_peak_dT=fit_peak_dT,
        r_bins=r_bins, r_mean=r_mean, r_rad=r_rad, r_ee=r_ee,
        time_series=fm,
    ))

# ---- Figure -----------------------------------------------------------------
fig = plt.figure(figsize=(22, 28))
fig.patch.set_facecolor("#111111")

gs_outer = mgridspec.GridSpec(4, 1, figure=fig,
                               hspace=0.55, wspace=0.3,
                               left=0.06, right=0.97,
                               top=0.96, bottom=0.04,
                               height_ratios=[1, 1, 1, 0.7])

CMAP_HOT  = "inferno"
CMAP_DIFF = "RdBu_r"

for si, (r, col) in enumerate(zip(results, SESSION_COLS)):
    d = r["standoff"]
    gs_s = mgridspec.GridSpecFromSubplotSpec(3, 4, subplot_spec=gs_outer[si],
                                              hspace=0.55, wspace=0.3)

    fig.text(0.5, gs_outer[si].get_position(fig).y1 + 0.005,
             f"-- {d} mm standoff  |  {r['mpp']:.4f} mm/px  |  "
             f"peak dT = {r['peak_dT']:.2f} K  |  "
             f"sigma_x={r['sx_mm']:.1f} mm  sigma_y={r['sy_mm']:.1f} mm  |  "
             f"FWHM {r['fwhm_x']:.0f} x {r['fwhm_y']:.0f} mm  |  "
             f"theta={np.degrees(r['theta']):.1f} deg",
             ha="center", color=col, fontsize=10, fontweight="bold")

    vmax_d = np.percentile(r["diff"], 99.5)
    vmin_d = np.percentile(r["diff"], 0.5)
    vmax_f = r["fit_in"].max()

    # col 0: raw dT
    ax = fig.add_subplot(gs_s[0, 0])
    ax.set_facecolor("#0a0a0a")
    im = ax.imshow(r["diff"], cmap=CMAP_DIFF, vmin=vmin_d, vmax=vmax_d,
                   origin="upper", aspect="equal")
    plt.colorbar(im, ax=ax, pad=0.02).ax.tick_params(labelsize=7, colors="white")
    ax.set_title("Raw dT  [K]", color="white", fontsize=8)
    ax.set_xlabel("px", color="#aaaaaa", fontsize=7)
    ax.set_ylabel("px", color="#aaaaaa", fontsize=7)
    ax.tick_params(colors="#666666", labelsize=7)

    # col 1: dT + fit ellipses
    ax = fig.add_subplot(gs_s[0, 1])
    ax.set_facecolor("#0a0a0a")
    ax.imshow(r["fit_in"], cmap=CMAP_HOT, vmin=0, vmax=vmax_f,
              origin="upper", aspect="equal")
    for scale, ls, lbl in [(1, "-", "1s"), (2, "--", "2s"),
                            (2.355/2, ":", "FWHM")]:
        ax.add_patch(Ellipse((r["x0"], r["y0"]),
                             2*r["sx"]*scale, 2*r["sy"]*scale,
                             angle=np.degrees(r["theta"]),
                             fc="none", ec=col, lw=1.5, ls=ls, zorder=5))
    ax.plot(r["x0"], r["y0"], "+", color="white", ms=10, mew=2, zorder=6)
    ax.set_title("dT + fit ellipses (1s / 2s / FWHM)", color="white", fontsize=8)
    ax.tick_params(colors="#666666", labelsize=7)
    ax.set_xlabel("px", color="#aaaaaa", fontsize=7)

    # col 2: Gaussian fit
    ax = fig.add_subplot(gs_s[0, 2])
    ax.set_facecolor("#0a0a0a")
    im2 = ax.imshow(r["fitted"], cmap=CMAP_HOT, vmin=0, vmax=vmax_f,
                    origin="upper", aspect="equal")
    plt.colorbar(im2, ax=ax, pad=0.02).ax.tick_params(labelsize=7, colors="white")
    ax.set_title("2D Gaussian fit", color="white", fontsize=8)
    ax.tick_params(colors="#666666", labelsize=7)
    ax.set_xlabel("px", color="#aaaaaa", fontsize=7)

    # col 3: residuals
    ax = fig.add_subplot(gs_s[0, 3])
    ax.set_facecolor("#0a0a0a")
    rlim = np.percentile(np.abs(r["resid"]), 98)
    im3 = ax.imshow(r["resid"], cmap="RdBu_r", vmin=-rlim, vmax=rlim,
                    origin="upper", aspect="equal")
    plt.colorbar(im3, ax=ax, pad=0.02).ax.tick_params(labelsize=7, colors="white")
    rms = float(np.sqrt((r["resid"]**2).mean()))
    ax.set_title(f"Residuals  (RMS={rms:.3f} K)", color="white", fontsize=8)
    ax.tick_params(colors="#666666", labelsize=7)
    ax.set_xlabel("px", color="#aaaaaa", fontsize=7)

    # row 1: cross-sections
    mid_x = int(round(r["x0"]))
    mid_y = int(round(r["y0"]))
    H, W = r["fit_in"].shape
    x_px = np.arange(W)
    y_px = np.arange(H)
    h_data = r["fit_in"][mid_y, :]
    v_data = r["fit_in"][:, mid_x]
    h_fit  = r["fitted"][mid_y, :]
    v_fit  = r["fitted"][:, mid_x]
    x_mm_ax = (x_px - r["x0"]) * r["mpp"]
    y_mm_ax = (y_px - r["y0"]) * r["mpp"]

    ax_h = fig.add_subplot(gs_s[1, 0:2])
    dark_ax(ax_h)
    ax_h.plot(x_mm_ax, h_data, color=col, lw=1.5, alpha=0.7, label="Data")
    ax_h.plot(x_mm_ax, h_fit,  color="white", lw=2, ls="--", label="Gaussian fit")
    ax_h.axvline(-r["sx_mm"], color=col, lw=1, ls=":", alpha=0.6)
    ax_h.axvline( r["sx_mm"], color=col, lw=1, ls=":", alpha=0.6)
    ax_h.axvspan(-r["sx_mm"], r["sx_mm"], color=col, alpha=0.06)
    ax_h.axvline(-r["fwhm_x"]/2, color="lime", lw=0.8, ls="--", alpha=0.5)
    ax_h.axvline( r["fwhm_x"]/2, color="lime", lw=0.8, ls="--", alpha=0.5)
    ax_h.set_xlabel("Distance from centre (mm)", color="#aaaaaa", fontsize=8)
    ax_h.set_ylabel("dT (K)", color="#aaaaaa", fontsize=8)
    ax_h.set_title(f"Horizontal cross-section  (sigma_x = {r['sx_mm']:.1f} mm)",
                   color="white", fontsize=8)
    ax_h.legend(fontsize=7.5, facecolor="#222222", labelcolor="white",
                edgecolor="#555555", loc="upper right")

    ax_v = fig.add_subplot(gs_s[1, 2:4])
    dark_ax(ax_v)
    ax_v.plot(y_mm_ax, v_data, color=col, lw=1.5, alpha=0.7, label="Data")
    ax_v.plot(y_mm_ax, v_fit,  color="white", lw=2, ls="--", label="Gaussian fit")
    ax_v.axvline(-r["sy_mm"], color=col, lw=1, ls=":", alpha=0.6)
    ax_v.axvline( r["sy_mm"], color=col, lw=1, ls=":", alpha=0.6)
    ax_v.axvspan(-r["sy_mm"], r["sy_mm"], color=col, alpha=0.06)
    ax_v.axvline(-r["fwhm_y"]/2, color="lime", lw=0.8, ls="--", alpha=0.5)
    ax_v.axvline( r["fwhm_y"]/2, color="lime", lw=0.8, ls="--", alpha=0.5)
    ax_v.set_xlabel("Distance from centre (mm)", color="#aaaaaa", fontsize=8)
    ax_v.set_ylabel("dT (K)", color="#aaaaaa", fontsize=8)
    ax_v.set_title(f"Vertical cross-section  (sigma_y = {r['sy_mm']:.1f} mm)",
                   color="white", fontsize=8)
    ax_v.legend(fontsize=7.5, facecolor="#222222", labelcolor="white",
                edgecolor="#555555", loc="upper right")

    # row 2: radial profile + time series
    r_mm    = r["r_bins"] * r["mpp"]
    r_ee_mm = r["r_rad"]  * r["mpp"]

    ax_r = fig.add_subplot(gs_s[2, 0:2])
    dark_ax(ax_r)
    peak_r = r["r_mean"].max() if r["r_mean"].max() > 0 else 1.0
    ax_r.plot(r_mm, r["r_mean"]/peak_r, color=col, lw=2, label="Azimuthal mean")
    r_theory = np.linspace(0, r_mm[-1], 300)
    sigma_eff = np.sqrt(r["sx_mm"]*r["sy_mm"])
    ax_r.plot(r_theory, np.exp(-0.5*(r_theory/sigma_eff)**2),
              color="white", lw=1.5, ls="--", alpha=0.7,
              label=f"Gaussian sigma_eff={sigma_eff:.0f} mm")
    ax_r.axvline(sigma_eff, color="white", lw=0.8, ls=":", alpha=0.5)
    ax_r.axvline(sigma_eff*2.355/2, color="lime", lw=0.8, ls="--", alpha=0.5)
    ax_r.set_xlabel("Radius from centre (mm)", color="#aaaaaa", fontsize=8)
    ax_r.set_ylabel("Normalised dT", color="#aaaaaa", fontsize=8)
    ax_r.set_title("Azimuthally averaged radial profile", color="white", fontsize=8)
    ax_r.set_xlim(0, min(250, r_mm[-1]))
    ax_r.legend(fontsize=7.5, facecolor="#222222", labelcolor="white", edgecolor="#555555")

    # time series of frame-mean dT (diagnostic)
    ax_t = fig.add_subplot(gs_s[2, 2:4])
    dark_ax(ax_t)
    t_s = np.arange(len(r["time_series"])) / FPS_DEFAULT
    ax_t.plot(t_s, r["time_series"], color=col, lw=1.8)
    ax_t.axhline(0, color="white", lw=0.5, alpha=0.4)
    ax_t.set_xlabel("Time (s)", color="#aaaaaa", fontsize=8)
    ax_t.set_ylabel("Frame-mean dT vs frame-0 (K)", color="#aaaaaa", fontsize=8)
    ax_t.set_title("Heating time-series diagnostic", color="white", fontsize=8)

# bottom summary
gs_bot = mgridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_outer[3], wspace=0.35)
standoffs_fit = np.array([r["standoff"] for r in results], float)
sx_mm_fit     = np.array([r["sx_mm"]    for r in results], float)
sy_mm_fit     = np.array([r["sy_mm"]    for r in results], float)
peak_dT_fit   = np.array([r["peak_dT"]  for r in results], float)

cx = np.polyfit(standoffs_fit, sx_mm_fit, 1)
cy = np.polyfit(standoffs_fit, sy_mm_fit, 1)
d_line = np.linspace(250, 550, 200)

ax_s = fig.add_subplot(gs_bot[0])
dark_ax(ax_s)
for r, col in zip(results, SESSION_COLS):
    ax_s.scatter(r["standoff"], r["sx_mm"], s=120, color=col, marker="o", zorder=5)
    ax_s.scatter(r["standoff"], r["sy_mm"], s=120, color=col, marker="s", zorder=5)
ax_s.plot(d_line, np.polyval(cx, d_line), color="#ffd93d", lw=2, ls="--",
          label=f"sigma_x  slope={cx[0]:.3f} mm/mm   intercept={cx[1]:.1f} mm")
ax_s.plot(d_line, np.polyval(cy, d_line), color="#4d96ff", lw=2, ls="--",
          label=f"sigma_y  slope={cy[0]:.3f} mm/mm   intercept={cy[1]:.1f} mm")
ax_s.scatter([], [], s=80, color="white", marker="o", label="sigma_x measured")
ax_s.scatter([], [], s=80, color="white", marker="s", label="sigma_y measured")
ax_s.set_xlabel("Standoff (mm)", color="#aaaaaa", fontsize=9)
ax_s.set_ylabel("Sigma (mm)", color="#aaaaaa", fontsize=9)
ax_s.set_title("sigma_x / sigma_y vs standoff -- linear fit",
               color="white", fontsize=9)
ax_s.legend(fontsize=7.5, facecolor="#222222", labelcolor="white",
            edgecolor="#555555", loc="best")

# peak dT * d^2 check (inverse-square test)
ax_p = fig.add_subplot(gs_bot[1])
dark_ax(ax_p)
peak_d2 = peak_dT_fit * standoffs_fit**2
for r, col, pd2 in zip(results, SESSION_COLS, peak_d2):
    ax_p.scatter(r["standoff"], pd2, s=140, color=col, marker="o", zorder=5)
    ax_p.text(r["standoff"]+5, pd2, f"  {pd2/1e3:.1f}",
              color=col, fontsize=8, va="center")
mean_p = peak_d2.mean()
ax_p.axhline(mean_p, color="white", lw=1.2, ls="--", alpha=0.5,
             label=f"mean = {mean_p:.0f}")
ax_p.set_xlabel("Standoff (mm)", color="#aaaaaa", fontsize=9)
ax_p.set_ylabel("peak dT x d^2  (K mm^2)", color="#aaaaaa", fontsize=9)
ax_p.set_title("Inverse-square check (flat -> 1/d^2 holds)",
               color="white", fontsize=9)
ax_p.legend(fontsize=7.5, facecolor="#222222", labelcolor="white",
            edgecolor="#555555", loc="best")

# aspect ratio
ax_a = fig.add_subplot(gs_bot[2])
dark_ax(ax_a)
for r, col in zip(results, SESSION_COLS):
    ax_a.scatter(r["standoff"], r["sx_mm"]/r["sy_mm"], s=140, color=col,
                 marker="D", zorder=5)
    ax_a.text(r["standoff"]+5, r["sx_mm"]/r["sy_mm"]+0.02,
              f"{r['sx_mm']/r['sy_mm']:.2f}", color=col, fontsize=9)
mean_asp = np.mean([r["sx_mm"]/r["sy_mm"] for r in results])
ax_a.axhline(mean_asp, color="white", lw=1.5, ls="--", alpha=0.6)
ax_a.text(510, mean_asp+0.02, f"mean={mean_asp:.2f}", color="white", fontsize=8)
ax_a.set_xlabel("Standoff (mm)", color="#aaaaaa", fontsize=9)
ax_a.set_ylabel("Aspect ratio sigma_x / sigma_y", color="#aaaaaa", fontsize=9)
ax_a.set_title("Beam aspect ratio across standoffs",
               color="white", fontsize=9)

fig.suptitle(
    "Beam shape analysis  |  FLIR A655sc, FOL18 (45 deg) lens  |  300 / 400 / 500 mm standoff\n"
    "2D Gaussian fit to baseline-subtracted dT (K), from .seq @ 8 fps, 99 frames",
    color="white", fontsize=12, y=0.975)

fig.savefig(OUTPUT_PNG, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"\nSaved figure: {OUTPUT_PNG}")

# ---- Results JSON -----------------------------------------------------------
results_clean = []
for r in results:
    results_clean.append({
        "standoff_mm": int(r["standoff"]),
        "mm_per_px":   float(r["mpp"]),
        "peak_dT_K":   float(r["peak_dT"]),
        "fit_peak_dT_K": float(r["fit_peak_dT"]),
        "centroid_px": [float(r["x0"]), float(r["y0"])],
        "sigma_x_mm":  float(r["sx_mm"]),
        "sigma_y_mm":  float(r["sy_mm"]),
        "fwhm_x_mm":   float(r["fwhm_x"]),
        "fwhm_y_mm":   float(r["fwhm_y"]),
        "theta_deg":   float(np.degrees(r["theta"])),
        "aspect_ratio": float(r["sx_mm"]/r["sy_mm"]),
    })

cross_fit = {
    "sigma_x_vs_d":  {"slope_mm_per_mm": float(cx[0]), "intercept_mm": float(cx[1])},
    "sigma_y_vs_d":  {"slope_mm_per_mm": float(cy[0]), "intercept_mm": float(cy[1])},
    "mean_aspect_x_over_y": float(np.mean([r["sx_mm"]/r["sy_mm"] for r in results])),
    "peak_dT_d2_mean": float(mean_p),
    "peak_dT_d2_std_over_mean": float(peak_d2.std()/peak_d2.mean()),
}

with open(RESULTS_JSON, 'w') as f:
    json.dump({"per_standoff": results_clean, "cross_standoff": cross_fit},
              f, indent=2)
print(f"Saved JSON:   {RESULTS_JSON}")

print("\n--- Summary ---")
for rc in results_clean:
    print(f"  {rc['standoff_mm']} mm:  sx={rc['sigma_x_mm']:6.1f} mm  sy={rc['sigma_y_mm']:6.1f} mm  "
          f"peakdT={rc['peak_dT_K']:.2f} K  aspect={rc['aspect_ratio']:.2f}")
print(f"\n  sigma_x(d)  = {cross_fit['sigma_x_vs_d']['slope_mm_per_mm']:.4f} * d + {cross_fit['sigma_x_vs_d']['intercept_mm']:.2f}")
print(f"  sigma_y(d)  = {cross_fit['sigma_y_vs_d']['slope_mm_per_mm']:.4f} * d + {cross_fit['sigma_y_vs_d']['intercept_mm']:.2f}")
print(f"  peak dT * d^2  mean = {cross_fit['peak_dT_d2_mean']:.0f}  K*mm^2   (CV = {cross_fit['peak_dT_d2_std_over_mean']*100:.1f}%)")
