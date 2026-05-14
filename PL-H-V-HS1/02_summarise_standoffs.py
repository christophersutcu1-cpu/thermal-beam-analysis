"""
Beam shape summary -- SEQ / FLIR A655sc version, same layout as beam_shape_summary.py.

Story: three standoff measurements (300/400/500 mm) -> overlaid cross-sections
       -> consistent shape -> linear fit -> derived consensus beam.

Layout (5 rows):
  Row 0 (images)      : dT image + ellipses for 300 / 400 / 500 mm
  Row 1 (overlay)     : H cross-sections | V cross-sections | radial profiles
  Row 2 (derivation)  : sigma vs standoff fit | consensus beam diagram | encircled energy
  Row 3 (derived)     : derived Gaussian 2D top-down | 3D surface
  Row 4 (projections) : 2D projection | H profile w/ FWHM | V profile w/ FWHM
"""

import os, glob, json, tempfile
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
from matplotlib.patches import Ellipse, Rectangle
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from flirpy.io.seq import Splitter

# ---- I/O --------------------------------------------------------------------
SEQ_ROOT   = r"C:\Users\cs1d25\Downloads\OneDrive_1_14-05-2026"
REPO_ROOT  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PNG = os.path.join(REPO_ROOT, "beam_shape_summary_seq.png")
RESULTS_JSON = os.path.join(REPO_ROOT, "beam_shape_summary_seq.json")
CACHE_DIR  = os.path.join(REPO_ROOT, "_seq_cache")

SESSIONS = [
    # NOTE: filenames mis-labelled in source folder. Actual standoffs are
    # 300/500/700 mm (user confirmed 2026-05-14).
    (300, os.path.join(SEQ_ROOT, "300mm.seq")),
    (500, os.path.join(SEQ_ROOT, "400mm.seq")),
    (700, os.path.join(SEQ_ROOT, "500mm.seq")),
]
COLS = ["#6bcb77", "#ffd93d", "#ff6b6b"]

# ---- Camera (FLIR A655sc + FOL18 lens) -------------------------------------
SENSOR_W_PX, SENSOR_H_PX = 640, 480
HFOV_DEG = 45.0
CAM_TILT_DEG = 15.0          # camera 15 deg off-normal in x; lamp normal.
COS_TILT = np.cos(np.radians(CAM_TILT_DEG))

def mpp(d): return 2*d*np.tan(np.radians(HFOV_DEG/2)) / SENSOR_W_PX

# ---- Specimen footprint (from DFLUX XC_PLATE=160, YC_PLATE=87.5) -----------
SPEC_W_MM, SPEC_H_MM = 320.0, 175.0

# ---- FLIR scale -------------------------------------------------------------
COUNT_TO_K = 0.04
FPS = 62.0 / 3.0             # ~20.67 Hz effective (from metadata timestamps)
N_BASE = 5
N_HOT  = 3                   # last clean heating frames before cool-down

# ---- Loader -----------------------------------------------------------------

def load_seq_frames(seq_path, cache_root):
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

# ---- Pulse-window detection (freeze-aware) ---------------------------------

def select_baseline_and_hot_frames(frames_K, hot_n_pix=200, n_base=N_BASE, n_hot=N_HOT):
    """Return (baseline_img, hot_img) using only valid pre-pulse and peak frames.
    Handles USB-stall freezes (500 mm) by skipping frames whose temporal diff
    is below 0.01 K (frozen) after the pulse starts.
    """
    N = frames_K.shape[0]
    flat = frames_K.reshape(N, -1)
    final_avg = flat[-3:].mean(axis=0)
    hot_idx = np.argsort(final_avg)[-hot_n_pix:]
    pk = flat[:, hot_idx].mean(axis=1)

    # pulse start: sustained rise >= 0.5 K above 5-frame baseline
    baseline_T = pk[:5].mean()
    rose = pk - baseline_T > 0.5
    i_start = None
    for i in range(N - 3):
        if rose[i] and rose[i+1] and rose[i+2]:
            i_start = i
            break
    if i_start is None:
        i_start = int(np.argmax(rose)) if rose.any() else 5

    # frozen-frame mask AFTER pulse start
    dpk = np.abs(np.diff(pk))
    frozen = np.zeros(N, dtype=bool)
    i = i_start + 1
    while i < N - 2:
        if dpk[i] < 0.01 and dpk[i+1] < 0.01 and dpk[i+2] < 0.01:
            j = i
            while j < N - 1 and dpk[j] < 0.01:
                frozen[j+1] = True
                j += 1
            i = j + 1
        else:
            i += 1

    # peak frame: global argmax of trace
    i_peak = int(np.argmax(pk))

    # baseline: average of first n_base pre-pulse frames
    n_pre = min(n_base, max(i_start, 1))
    baseline = frames_K[:n_pre].mean(axis=0)

    # hot: last n_hot non-frozen heating frames up to and including peak
    valid_heat = [i for i in range(i_start, i_peak + 1) if not frozen[i]]
    if len(valid_heat) >= n_hot:
        hot_frames = valid_heat[-n_hot:]
    elif len(valid_heat) >= 1:
        hot_frames = valid_heat[-min(n_hot, len(valid_heat)):]
    else:
        hot_frames = [i_peak]
    hot = frames_K[hot_frames].mean(axis=0)

    print(f"     pulse: start=f{i_start} peak=f{i_peak} frozen={int(frozen.sum())} "
          f"hot_frames={hot_frames}")
    return baseline, hot

# ---- Robust 2D rotated Gaussian fitter -------------------------------------

def rotated_gaussian(xy, A, x0, y0, sx, sy, theta, bg):
    x, y = xy
    ct, st = np.cos(theta), np.sin(theta)
    xr =  ct*(x-x0) + st*(y-y0)
    yr = -st*(x-x0) + ct*(y-y0)
    return A*np.exp(-0.5*((xr/sx)**2 + (yr/sy)**2)) + bg

def weighted_centroid(img):
    pos = np.clip(img, 0, None)
    if pos.sum() <= 0:
        return img.shape[1]/2, img.shape[0]/2
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    return float((xx*pos).sum()/pos.sum()), float((yy*pos).sum()/pos.sum())

def fit_gaussian(img):
    """Wider centroid bounds so a beam clipped outside the FOV can still
    resolve a meaningful sigma."""
    H, W = img.shape
    xx, yy = np.meshgrid(np.arange(W, dtype=np.float64),
                         np.arange(H, dtype=np.float64))
    smoothed = gaussian_filter(img, sigma=25)
    x0s, y0s = weighted_centroid(smoothed)
    pos = np.clip(smoothed - np.percentile(smoothed, 10), 0, None)
    if pos.sum() > 0:
        sx_init = np.sqrt(max((((xx - x0s)**2 * pos).sum()/pos.sum()), 1.0))
        sy_init = np.sqrt(max((((yy - y0s)**2 * pos).sum()/pos.sum()), 1.0))
    else:
        sx_init = sy_init = min(W, H)/5.0
    vmax = float(img.max())
    bg0  = float(np.percentile(img, 5))
    p0 = [max(vmax - bg0, 1e-3), x0s, y0s, sx_init, sy_init, 0.0, bg0]
    # allow centroid to extrapolate outside the frame and sigma up to 3x dim
    lo = [0,         -W*0.6, -H*0.6, 5,    5,    -np.pi/2, -abs(bg0)*3 - 1]
    hi = [vmax*3+1,  W*1.6,  H*1.6,  W*3,  H*3,  np.pi/2,  vmax + 1]
    popt, _ = curve_fit(rotated_gaussian, (xx.ravel(), yy.ravel()),
                        img.ravel().astype(np.float64),
                        p0=p0, bounds=(lo, hi), maxfev=30000)
    fitted = rotated_gaussian((xx.ravel(), yy.ravel()), *popt).reshape(H, W)
    return popt, fitted

def radial_profile(img, cx, cy, max_r):
    H, W = img.shape
    yy, xx = np.ogrid[:H, :W]
    r = np.sqrt((xx-cx)**2 + (yy-cy)**2).ravel()
    v = img.ravel()
    bins = np.arange(0, max_r+1, 1.0)
    cnt,_ = np.histogram(r, bins=bins)
    tot,_ = np.histogram(r, bins=bins, weights=v)
    return bins[:-1], np.where(cnt>0, tot/cnt, 0.0)

def encircled_energy(img, cx, cy, max_r):
    H, W = img.shape
    yy, xx = np.ogrid[:H, :W]
    r = np.sqrt((xx-cx)**2 + (yy-cy)**2)
    pos = np.clip(img, 0, None); tot = pos.sum()
    rad = np.arange(0, max_r+1, 1.0)
    ee = np.array([(pos[r<=ri]).sum()/tot for ri in rad])
    return rad, ee

def dark_ax(ax, grid=True):
    ax.set_facecolor("#1a1a1a")
    for sp in ax.spines.values(): sp.set_edgecolor("#444444")
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    if grid: ax.grid(True, alpha=0.12, color="white")

# ---- Process all sessions ---------------------------------------------------
sessions = []
for (d, seq_path), col in zip(SESSIONS, COLS):
    mpp_d = mpp(d)                  # uniform mm/px AFTER tilt-stretch
    print(f"\n{d} mm:")
    frames = load_seq_frames(seq_path, CACHE_DIR)
    base, hot = select_baseline_and_hot_frames(frames)
    dT_raw = hot - base

    # mask bad pixels via 5x5 median outlier test
    med5 = cv2.medianBlur(dT_raw.astype(np.float32), 5)
    tol = np.maximum(3.0, 0.5 * np.abs(med5))
    bad = (np.abs(dT_raw - med5) > tol) | (~np.isfinite(dT_raw))
    if bad.any():
        dT_raw = np.where(bad, med5, dT_raw)
        print(f"     masked {int(bad.sum())} bad pixels")

    # ---- Camera-tilt correction: stretch image in x by 1/cos(15deg) -------
    # so the resulting image is in undistorted board-frame geometry, with
    # uniform mm/px = mpp(d) in both directions. Original image was
    # foreshortened in x (camera 15 deg off-normal); stretch undoes that.
    H, W = dT_raw.shape
    new_W = int(round(W / COS_TILT))
    dT = cv2.resize(dT_raw, (new_W, H), interpolation=cv2.INTER_LINEAR)

    fit_in = dT - dT.min()
    popt, fitted = fit_gaussian(fit_in)
    A, x0, y0, sx, sy, theta, bg = popt
    if sy > sx:                              # convention sx >= sy
        sx, sy = sy, sx
        theta += np.pi/2

    sx_mm = sx * mpp_d
    sy_mm = sy * mpp_d
    fwhm_x = sx * 2.355 * mpp_d
    fwhm_y = sy * 2.355 * mpp_d

    # radial profile and encircled energy using fitted centroid (pixel-space)
    max_r = int(min(new_W, SENSOR_H_PX))
    rb, rp = radial_profile(fit_in, x0, y0, max_r)
    re, ee = encircled_energy(fit_in, x0, y0, max_r)

    sessions.append(dict(
        d=d, col=col, mpp=mpp_d, new_W=new_W,
        dT=dT, fit_in=fit_in, fitted=fitted,
        A=A, x0=x0, y0=y0, sx=sx, sy=sy, theta=theta, bg=bg,
        sx_mm=sx_mm, sy_mm=sy_mm, fwhm_x=fwhm_x, fwhm_y=fwhm_y,
        peak_dT=float(dT.max()),
        rb=rb, rp=rp, re=re, ee=ee,
    ))
    print(f"     sigma_x={sx_mm:.1f} mm, sigma_y={sy_mm:.1f} mm, "
          f"FWHM {fwhm_x:.0f}x{fwhm_y:.0f} mm, "
          f"centroid=({x0:.0f},{y0:.0f}) px (in stretched image of {new_W}x{H}), "
          f"peak dT={dT.max():.2f} K")

# ---- Cross-standoff linear fits + consensus -------------------------------
ds   = np.array([s["d"]     for s in sessions], float)
sx_m = np.array([s["sx_mm"] for s in sessions], float)
sy_m = np.array([s["sy_mm"] for s in sessions], float)
cx_lin = np.polyfit(ds, sx_m, 1)    # sx_mm = cx_lin[0]*d + cx_lin[1]
cy_lin = np.polyfit(ds, sy_m, 1)
sx_consensus = float(np.mean(sx_m))
sy_consensus = float(np.mean(sy_m))
fwhm_x_cons  = sx_consensus * 2.355
fwhm_y_cons  = sy_consensus * 2.355
aspect_cons  = sx_consensus / sy_consensus

half_div_x = np.degrees(np.arctan(cx_lin[0]))
half_div_y = np.degrees(np.arctan(cy_lin[0]))

print(f"\nConsensus: sigma_x = {sx_consensus:.1f} mm, sigma_y = {sy_consensus:.1f} mm, "
      f"FWHM {fwhm_x_cons:.0f}x{fwhm_y_cons:.0f} mm, aspect {aspect_cons:.2f}:1")
print(f"sigma_x slope = {cx_lin[0]:+.4f} mm/mm  (half-div = {half_div_x:+.2f} deg)")
print(f"sigma_y slope = {cy_lin[0]:+.4f} mm/mm  (half-div = {half_div_y:+.2f} deg)")

# ---- Figure -----------------------------------------------------------------
fig = plt.figure(figsize=(22, 32))
fig.patch.set_facecolor("#111111")
gs = mgridspec.GridSpec(5, 6, figure=fig,
                        hspace=0.55, wspace=0.45,
                        left=0.05, right=0.97, top=0.96, bottom=0.04,
                        height_ratios=[1.15, 0.85, 1.0, 1.4, 1.0])
CMAP = "inferno"

# ---- Row 0: 3 dT images with sigma ellipses + specimen footprint ----------
for i, s in enumerate(sessions):
    ax = fig.add_subplot(gs[0, i*2:(i+1)*2])
    ax.set_facecolor("#0a0a0a")
    H, W = s["fit_in"].shape
    # mm extent centred on the fitted centroid
    x_ext = ((0 - s["x0"]) * s["mpp"], (W - s["x0"]) * s["mpp"])
    y_ext = ((H - s["y0"]) * s["mpp"], (0 - s["y0"]) * s["mpp"])
    im = ax.imshow(s["fit_in"], cmap=CMAP, origin="upper", aspect="equal",
                   extent=[x_ext[0], x_ext[1], y_ext[0], y_ext[1]])
    # sigma ellipses
    for sc, ls in [(1, "-"), (2, "--"), (2.355/2, ":")]:
        ax.add_patch(Ellipse((0, 0), 2*s["sx_mm"]*sc, 2*s["sy_mm"]*sc,
                             angle=-np.degrees(s["theta"]),
                             fc="none", ec=s["col"], lw=1.8, ls=ls, zorder=5))
    # specimen footprint
    ax.add_patch(Rectangle((-SPEC_W_MM/2, -SPEC_H_MM/2), SPEC_W_MM, SPEC_H_MM,
                            fc="none", ec="lime", lw=1.5, ls="--", alpha=0.8))
    # centroid marker
    ax.plot(0, 0, "+", color="white", ms=12, mew=2)
    # sigma annotations
    ax.annotate("", xy=(s["sx_mm"], 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="<-", color="white", lw=1.5))
    ax.text(s["sx_mm"]/2, 4, f"sigma_x={s['sx_mm']:.0f}mm",
            color="white", fontsize=9, ha="center")
    ax.annotate("", xy=(0, -s["sy_mm"]), xytext=(0, 0),
                arrowprops=dict(arrowstyle="<-", color="white", lw=1.5))
    ax.text(4, -s["sy_mm"]/2, f"sigma_y={s['sy_mm']:.0f}mm",
            color="white", fontsize=9, ha="left", va="center")
    ax.set_title(f"{s['d']}mm standoff  |  {s['mpp']:.4f} mm/px\n"
                 f"FWHM  {s['fwhm_x']:.0f} x {s['fwhm_y']:.0f} mm",
                 color=s["col"], fontsize=10, fontweight="bold")
    ax.set_xlabel("x [mm]", color="#aaaaaa", fontsize=8)
    ax.set_ylabel("y [mm]", color="#aaaaaa", fontsize=8)
    ax.tick_params(colors="#888888", labelsize=7)
    plt.colorbar(im, ax=ax, pad=0.02, fraction=0.04,
                 label="dT [K]").ax.tick_params(labelsize=7, colors="white")

# ---- Row 1: H cross / V cross / radial profile (all 3 overlaid) -----------
ax_h = fig.add_subplot(gs[1, 0:2]); dark_ax(ax_h)
ax_v = fig.add_subplot(gs[1, 2:4]); dark_ax(ax_v)
ax_r = fig.add_subplot(gs[1, 4:6]); dark_ax(ax_r)
for s in sessions:
    midy = int(np.clip(round(s["y0"]), 0, s["fit_in"].shape[0]-1))
    midx = int(np.clip(round(s["x0"]), 0, s["fit_in"].shape[1]-1))
    x_mm = (np.arange(s["fit_in"].shape[1]) - s["x0"]) * s["mpp"]
    y_mm = (np.arange(s["fit_in"].shape[0]) - s["y0"]) * s["mpp"]
    h = s["fit_in"][midy, :]; v = s["fit_in"][:, midx]
    h_n = h/h.max() if h.max() > 0 else h
    v_n = v/v.max() if v.max() > 0 else v
    ax_h.plot(x_mm, h_n, color=s["col"], lw=1.4,
              label=f"{s['d']}mm  sigma_x={s['sx_mm']:.0f}mm")
    ax_v.plot(y_mm, v_n, color=s["col"], lw=1.4,
              label=f"{s['d']}mm  sigma_y={s['sy_mm']:.0f}mm")
    # radial profile normalised
    r_mm = s["rb"] * s["mpp"]
    rp_n = s["rp"]/s["rp"].max() if s["rp"].max() > 0 else s["rp"]
    ax_r.plot(r_mm, rp_n, color=s["col"], lw=1.5, label=f"{s['d']}mm")

for ax, ttl, xl in [(ax_h, "Horizontal cross-section -- all 3 standoffs overlaid",
                     "Distance from beam centre (mm)"),
                    (ax_v, "Vertical cross-section -- all 3 standoffs overlaid",
                     "Distance from beam centre (mm)"),
                    (ax_r, "Azimuthal radial profile -- all 3 overlaid vs theory",
                     "Radius from centre (mm)")]:
    ax.set_xlabel(xl, color="#aaaaaa", fontsize=9)
    ax.set_ylabel("Normalised dT", color="#aaaaaa", fontsize=9)
    ax.set_title(ttl, color="white", fontsize=10)
    ax.axhline(0.5, color="white", lw=0.7, ls=":", alpha=0.4)
    ax.axhline(np.exp(-0.5), color="white", lw=0.7, ls=":", alpha=0.4)
    ax.legend(fontsize=7.5, facecolor="#222222", labelcolor="white",
              edgecolor="#555555")

# theoretical Gaussian on radial plot
sigma_eff = np.sqrt(sx_consensus * sy_consensus)
r_theory = np.linspace(0, ax_r.get_xlim()[1] or 250, 300)
ax_r.plot(r_theory, np.exp(-0.5*(r_theory/sigma_eff)**2),
          color="white", lw=1.3, ls="--", alpha=0.8,
          label=f"Theory sigma_eff={sigma_eff:.0f}mm")
ax_r.legend(fontsize=7.5, facecolor="#222222", labelcolor="white",
            edgecolor="#555555")

# ---- Row 2: sigma vs standoff, consensus beam, encircled energy -----------
ax_sig = fig.add_subplot(gs[2, 0:2]); dark_ax(ax_sig)
d_line = np.linspace(250, 550, 200)
for s in sessions:
    ax_sig.scatter(s["d"], s["sx_mm"], s=180, color=s["col"],
                   marker="*", zorder=6)
    ax_sig.text(s["d"], s["sx_mm"]+1.5, f"{s['sx_mm']:.0f}",
                color=s["col"], ha="center", fontsize=9, fontweight="bold")
    ax_sig.scatter(s["d"], s["sy_mm"], s=140, color=s["col"],
                   marker="s", zorder=6)
    ax_sig.text(s["d"], s["sy_mm"]-3.0, f"{s['sy_mm']:.0f}",
                color=s["col"], ha="center", fontsize=9, fontweight="bold")
ax_sig.plot(d_line, np.polyval(cx_lin, d_line), color="#ffd93d", lw=1.8, ls="--",
            label=f"sigma_x  ({cx_lin[0]:+.4f} mm/mm)")
ax_sig.plot(d_line, np.polyval(cy_lin, d_line), color="#4d96ff", lw=1.8, ls="--",
            label=f"sigma_y  ({cy_lin[0]:+.4f} mm/mm)")
ax_sig.scatter([], [], s=120, color="white", marker="*", label="sigma_x measured")
ax_sig.scatter([], [], s=80, color="white", marker="s", label="sigma_y measured")
ax_sig.set_xlabel("Standoff (mm)", color="#aaaaaa", fontsize=9)
ax_sig.set_ylabel("Sigma (mm)", color="#aaaaaa", fontsize=9)
slope_label = "diverging" if abs(cx_lin[0]) > 0.02 else "near-zero -> collimated"
ax_sig.set_title(f"sigma_x / sigma_y vs standoff -> linear fit\n"
                 f"slope {cx_lin[0]:+.3f} mm/mm  ({slope_label}; size set by lamp+divergence)",
                 color="white", fontsize=10)
ax_sig.legend(fontsize=8, facecolor="#222222", labelcolor="white", edgecolor="#555555",
              loc="lower right")

# Consensus beam diagram
ax_cb = fig.add_subplot(gs[2, 2:4]); dark_ax(ax_cb)
for sc, ls, lbl in [(1, "-", "1sigma"), (2, "--", "2sigma"),
                     (2.355/2, ":", "FWHM")]:
    ax_cb.add_patch(Ellipse((0, 0), 2*sx_consensus*sc, 2*sy_consensus*sc,
                             fc="none", ec="white", lw=1.5, ls=ls, label=lbl))
ax_cb.add_patch(Rectangle((-SPEC_W_MM/2, -SPEC_H_MM/2), SPEC_W_MM, SPEC_H_MM,
                          fc="none", ec="lime", lw=1.8, ls="--", alpha=0.8,
                          label=f"Specimen {int(SPEC_W_MM)}x{int(SPEC_H_MM)}mm"))
ax_cb.plot(0, 0, "+", color="white", ms=12, mew=2)
ax_cb.annotate("", xy=(sx_consensus, 0), xytext=(0, 0),
               arrowprops=dict(arrowstyle="<-", color="cyan", lw=1.5))
ax_cb.text(sx_consensus/2, 3, f"sigma_x={sx_consensus:.0f}mm",
           color="cyan", fontsize=10, ha="center", fontweight="bold")
ax_cb.annotate("", xy=(0, -sy_consensus), xytext=(0, 0),
               arrowprops=dict(arrowstyle="<-", color="orange", lw=1.5))
ax_cb.text(4, -sy_consensus/2, f"sigma_y={sy_consensus:.0f}mm",
           color="orange", fontsize=10, ha="left", va="center", fontweight="bold")
lim = max(sx_consensus, sy_consensus)*2.2
ax_cb.set_xlim(-lim, lim); ax_cb.set_ylim(-lim*0.7, lim*0.7)
ax_cb.set_aspect("equal")
ax_cb.set_xlabel("x [mm]", color="#aaaaaa", fontsize=9)
ax_cb.set_ylabel("y [mm]", color="#aaaaaa", fontsize=9)
ax_cb.set_title(f"Consensus beam shape  (mean 300-500mm)\n"
                f"FWHM  {fwhm_x_cons:.0f} x {fwhm_y_cons:.0f} mm  |  aspect {aspect_cons:.2f}:1",
                color="white", fontsize=10)
ax_cb.legend(fontsize=8, facecolor="#222222", labelcolor="white",
             edgecolor="#555555", loc="lower right")

# Encircled energy
ax_ee = fig.add_subplot(gs[2, 4:6]); dark_ax(ax_ee)
for s in sessions:
    r_mm = s["re"] * s["mpp"]
    ax_ee.plot(r_mm, s["ee"]*100, color=s["col"], lw=1.8, label=f"{s['d']}mm")
    for pct, lc in [(50, "white"), (86, "lime"), (95, "#ff6b6b")]:
        idx = np.searchsorted(s["ee"]*100, pct)
        if 0 < idx < len(r_mm):
            ax_ee.axvline(r_mm[idx], color=s["col"], lw=0.5, ls=":", alpha=0.5)
for pct, lc, lbl in [(50, "white", "D50"), (86, "lime", "D86"), (95, "#ff6b6b", "D95")]:
    ax_ee.axhline(pct, color=lc, lw=0.7, ls="--", alpha=0.5)
    ax_ee.text(ax_ee.get_xlim()[1]*0.85, pct-3, lbl, color=lc, fontsize=8)
ax_ee.set_xlabel("Radius from centre (mm)", color="#aaaaaa", fontsize=9)
ax_ee.set_ylabel("Encircled energy (%)", color="#aaaaaa", fontsize=9)
ax_ee.set_title("Encircled energy -- D50 / D86 / D95\n(consistent across 3 standoffs)",
                color="white", fontsize=10)
ax_ee.set_ylim(0, 105)
ax_ee.legend(fontsize=8, facecolor="#222222", labelcolor="white", edgecolor="#555555",
             loc="lower right")

# ---- Row 3: derived Gaussian 2D + 3D ---------------------------------------
ax_dg2 = fig.add_subplot(gs[3, 0:3]); ax_dg2.set_facecolor("#0a0a0a")
xg = np.linspace(-1.5*sx_consensus, 1.5*sx_consensus, 400)
yg = np.linspace(-1.5*sy_consensus, 1.5*sy_consensus, 300)
Xg, Yg = np.meshgrid(xg, yg)
Ig = np.exp(-0.5*((Xg/sx_consensus)**2 + (Yg/sy_consensus)**2))
im = ax_dg2.imshow(Ig, extent=[xg[0], xg[-1], yg[0], yg[-1]],
                   origin="lower", cmap="inferno", aspect="equal", vmin=0, vmax=1)
levels = [0.135, 0.50, 0.607]
ax_dg2.contour(Xg, Yg, Ig, levels=levels,
               colors=["white", "yellow", "lightgreen"],
               linestyles=["--", "-", "-"], linewidths=1.4)
ax_dg2.add_patch(Rectangle((-SPEC_W_MM/2, -SPEC_H_MM/2), SPEC_W_MM, SPEC_H_MM,
                            fc="none", ec="lime", lw=1.8, ls="--",
                            label="Specimen 320x175mm"))
ax_dg2.plot(0, 0, "+", color="cyan", ms=14, mew=2)
ax_dg2.annotate(f"sigma_x={sx_consensus:.0f}mm", xy=(sx_consensus, 0),
                color="cyan", fontsize=10, ha="left", va="center", fontweight="bold")
ax_dg2.annotate(f"sigma_y={sy_consensus:.0f}mm", xy=(0, sy_consensus+2),
                color="orange", fontsize=10, ha="center", fontweight="bold")
ax_dg2.set_xlabel("X distance from centre (mm)", color="#aaaaaa", fontsize=9)
ax_dg2.set_ylabel("Y distance from centre (mm)", color="#aaaaaa", fontsize=9)
ax_dg2.set_title(f"Derived Gaussian -- top-down 2D view\n"
                 f"sigma_x={sx_consensus:.0f}mm  sigma_y={sy_consensus:.0f}mm  "
                 f"aspect {aspect_cons:.2f}:1",
                 color="white", fontsize=10)
ax_dg2.tick_params(colors="#888888", labelsize=7)
plt.colorbar(im, ax=ax_dg2, pad=0.02, fraction=0.04,
             label="Normalised irradiance").ax.tick_params(labelsize=7, colors="white")
ax_dg2.legend(fontsize=8, facecolor="#222222", labelcolor="white",
              edgecolor="#555555", loc="upper right")

ax_dg3 = fig.add_subplot(gs[3, 3:6], projection="3d")
ax_dg3.set_facecolor("#0a0a0a")
xg3 = np.linspace(-1.4*sx_consensus, 1.4*sx_consensus, 80)
yg3 = np.linspace(-1.4*sy_consensus, 1.4*sy_consensus, 60)
Xg3, Yg3 = np.meshgrid(xg3, yg3)
Ig3 = np.exp(-0.5*((Xg3/sx_consensus)**2 + (Yg3/sy_consensus)**2))
surf = ax_dg3.plot_surface(Xg3, Yg3, Ig3, cmap="inferno", linewidth=0,
                           antialiased=True, alpha=0.95)
# overlay specimen
spec_z = np.zeros((2, 2))
ax_dg3.plot([-SPEC_W_MM/2, SPEC_W_MM/2, SPEC_W_MM/2, -SPEC_W_MM/2, -SPEC_W_MM/2],
            [-SPEC_H_MM/2, -SPEC_H_MM/2, SPEC_H_MM/2, SPEC_H_MM/2, -SPEC_H_MM/2],
            [0, 0, 0, 0, 0], color="lime", lw=2, ls="--")
ax_dg3.set_xlabel("X (mm)", color="#aaaaaa", fontsize=9)
ax_dg3.set_ylabel("Y (mm)", color="#aaaaaa", fontsize=9)
ax_dg3.set_zlabel("Normalised irradiance", color="#aaaaaa", fontsize=9)
ax_dg3.set_title(f"Derived Gaussian -- 3D surface\n"
                 f"FWHM {fwhm_x_cons:.0f} x {fwhm_y_cons:.0f} mm  |  Peak normalised to 1.0",
                 color="white", fontsize=10)
ax_dg3.tick_params(colors="#888888", labelsize=7)
ax_dg3.view_init(elev=25, azim=-55)
fig.colorbar(surf, ax=ax_dg3, pad=0.10, fraction=0.04, shrink=0.7,
             label="Normalised irradiance").ax.tick_params(labelsize=7, colors="white")

# ---- Row 4: 2D projection + H profile + V profile -------------------------
ax_p = fig.add_subplot(gs[4, 0:2]); ax_p.set_facecolor("#0a0a0a")
ax_p.imshow(Ig, extent=[xg[0], xg[-1], yg[0], yg[-1]],
            origin="lower", cmap="inferno", aspect="equal", vmin=0, vmax=1)
for sc, ls, lbl in [(1, "-", "1sigma"), (2, "--", "2sigma"),
                     (2.355/2, ":", "FWHM")]:
    ax_p.add_patch(Ellipse((0, 0), 2*sx_consensus*sc, 2*sy_consensus*sc,
                            fc="none", ec="white", lw=1.4, ls=ls))
ax_p.add_patch(Rectangle((-SPEC_W_MM/2, -SPEC_H_MM/2), SPEC_W_MM, SPEC_H_MM,
                          fc="none", ec="lime", lw=1.8, ls="--",
                          label="Specimen 320x175mm"))
ax_p.axhline(0, color="orange", lw=0.8, ls=":", alpha=0.6, label="V profile (x=0)")
ax_p.axvline(0, color="cyan",   lw=0.8, ls=":", alpha=0.6, label="H profile (y=0)")
ax_p.set_xlabel("X (mm)", color="#aaaaaa", fontsize=9)
ax_p.set_ylabel("Y (mm)", color="#aaaaaa", fontsize=9)
ax_p.set_title("2D beam projection -- top-down view\nwith FWHM / 1sigma / 2sigma contours",
               color="white", fontsize=10)
ax_p.tick_params(colors="#888888", labelsize=7)
ax_p.legend(fontsize=8, facecolor="#222222", labelcolor="white",
            edgecolor="#555555", loc="upper right")

ax_xp = fig.add_subplot(gs[4, 2:4]); dark_ax(ax_xp)
x_prof = np.exp(-0.5*(xg/sx_consensus)**2)
ax_xp.plot(xg, x_prof, color="#ffd93d", lw=2.2)
ax_xp.fill_between(xg, 0, x_prof, color="#ffd93d", alpha=0.18)
ax_xp.axhline(0.5, color="white", lw=0.6, ls="--", alpha=0.5)
ax_xp.axhline(np.exp(-0.5), color="white", lw=0.6, ls=":", alpha=0.5)
fwhm_half = sx_consensus * 2.355 / 2
sigma_2 = 2*sx_consensus
ax_xp.axvline(-fwhm_half, color="orange", lw=0.8, ls="--", alpha=0.7)
ax_xp.axvline( fwhm_half, color="orange", lw=0.8, ls="--", alpha=0.7)
ax_xp.text(0, 0.55, f"{int(2*fwhm_half)}mm", ha="center",
           color="orange", fontsize=10)
ax_xp.text(0, 0.65, "FWHM", ha="center", color="orange", fontsize=9)
ax_xp.text(0, 0.13, f"2sigma = {int(2*sigma_2)}mm", ha="center",
           color="white", fontsize=8)
ax_xp.set_xlabel("Distance from centre (mm)", color="#aaaaaa", fontsize=9)
ax_xp.set_ylabel("Normalised irradiance", color="#aaaaaa", fontsize=9)
ax_xp.set_title(f"Horizontal (X) profile  --  sigma_x = {sx_consensus:.1f}mm\n"
                f"FWHM = {fwhm_x_cons:.1f}mm  |  2sigma width = {2*sigma_2:.1f}mm",
                color="white", fontsize=10)
ax_xp.set_ylim(-0.05, 1.1)

ax_yp = fig.add_subplot(gs[4, 4:6]); dark_ax(ax_yp)
y_prof = np.exp(-0.5*(yg/sy_consensus)**2)
ax_yp.plot(yg, y_prof, color="#4d96ff", lw=2.2)
ax_yp.fill_between(yg, 0, y_prof, color="#4d96ff", alpha=0.18)
ax_yp.axhline(0.5, color="white", lw=0.6, ls="--", alpha=0.5)
ax_yp.axhline(np.exp(-0.5), color="white", lw=0.6, ls=":", alpha=0.5)
fwhm_y_half = sy_consensus * 2.355/2
ax_yp.axvline(-fwhm_y_half, color="orange", lw=0.8, ls="--", alpha=0.7)
ax_yp.axvline( fwhm_y_half, color="orange", lw=0.8, ls="--", alpha=0.7)
ax_yp.text(0, 0.55, f"{int(2*fwhm_y_half)}mm", ha="center",
           color="orange", fontsize=10)
ax_yp.text(0, 0.65, "FWHM", ha="center", color="orange", fontsize=9)
ax_yp.text(0, 0.13, f"2sigma = {int(4*sy_consensus)}mm", ha="center",
           color="white", fontsize=8)
ax_yp.set_xlabel("Distance from centre (mm)", color="#aaaaaa", fontsize=9)
ax_yp.set_ylabel("Normalised irradiance", color="#aaaaaa", fontsize=9)
ax_yp.set_title(f"Vertical (Y) profile  --  sigma_y = {sy_consensus:.1f}mm\n"
                f"FWHM = {fwhm_y_cons:.1f}mm  |  2sigma width = {4*sy_consensus:.1f}mm",
                color="white", fontsize=10)
ax_yp.set_ylim(-0.05, 1.1)

fig.suptitle(
    f"Beam shape derivation  |  FOL18 (45 deg) lens, FLIR A655sc  |  "
    f"300 / 400 / 500 mm standoff\n"
    f"Three independent measurements -> consensus  "
    f"sigma_x={sx_consensus:.0f}mm  sigma_y={sy_consensus:.0f}mm  "
    f"|  FWHM {fwhm_x_cons:.0f}x{fwhm_y_cons:.0f}mm  |  aspect {aspect_cons:.2f}:1",
    color="white", fontsize=13, y=0.985)

fig.savefig(OUTPUT_PNG, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"\nSaved figure: {OUTPUT_PNG}")

# ---- Save JSON --------------------------------------------------------------
out = {
    "camera": "FLIR A655sc + FOL18 (45 deg HFOV)",
    "standoffs_mm": [int(s["d"]) for s in sessions],
    "per_standoff": [{
        "d_mm": int(s["d"]), "sigma_x_mm": float(s["sx_mm"]),
        "sigma_y_mm": float(s["sy_mm"]),
        "fwhm_x_mm": float(s["fwhm_x"]), "fwhm_y_mm": float(s["fwhm_y"]),
        "peak_dT_K": float(s["peak_dT"]),
        "centroid_px": [float(s["x0"]), float(s["y0"])],
        "theta_deg": float(np.degrees(s["theta"])),
    } for s in sessions],
    "consensus": {
        "sigma_x_mm": sx_consensus, "sigma_y_mm": sy_consensus,
        "fwhm_x_mm": fwhm_x_cons, "fwhm_y_mm": fwhm_y_cons,
        "aspect_x_over_y": aspect_cons,
    },
    "linear_fit": {
        "sigma_x":  {"slope_mm_per_mm": float(cx_lin[0]),
                     "intercept_mm":    float(cx_lin[1]),
                     "half_div_deg":    float(half_div_x)},
        "sigma_y":  {"slope_mm_per_mm": float(cy_lin[0]),
                     "intercept_mm":    float(cy_lin[1]),
                     "half_div_deg":    float(half_div_y)},
    },
    "specimen_footprint_mm": [SPEC_W_MM, SPEC_H_MM],
}
with open(RESULTS_JSON, "w") as f:
    json.dump(out, f, indent=2)
print(f"Saved JSON:   {RESULTS_JSON}")
