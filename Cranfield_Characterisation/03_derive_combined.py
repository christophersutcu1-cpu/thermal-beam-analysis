"""
Combined beam analysis: fits 300/400/500 mm FLIR A655sc captures jointly
and derives one parametric beam model suitable for DFLUX.

Outputs a single figure:
  Row 1 (3 panels): raw dT(x,y) at 300/400/500 mm with fit ellipses on a
                    common physical-mm scale
  Row 2 (2 panels): horizontal + vertical cross-sections overlaid
  Row 3 (1 panel):  normalised radial profile overlay (self-similarity test)
  Row 4 (3 panels): sigma_x,sigma_y vs d ; peak dT vs d w/ 1/d^2 fit ; derived
                    beam model summary card
"""

import os, glob, json, tempfile
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
# Refined run (2026-06-23): new FLIR A655sc capture set, CORRECTLY labelled at the
# true standoffs 300/500/700 mm (unlike the 2026-05-13 set — see PROVENANCE_STANDOFFS.md).
SEQ_ROOT   = os.path.join(REPO_ROOT, "wetransfer_thermography-seq-files_2026-06-19_1302")
OUTPUT_PNG = os.path.join(REPO_ROOT, "beam_derived_combined.png")
RESULTS_JSON = os.path.join(REPO_ROOT, "beam_derived_combined.json")
# Fresh cache dir: _seq_cache is keyed by .seq basename, and the new 300mm/500mm
# basenames would otherwise collide with the OLD set's cache (whose 500mm holds
# true-700 mm frames). Keep the new captures in their own cache.
CACHE_DIR  = os.path.join(REPO_ROOT, "_seq_cache_0619")

SESSIONS = [
    (300, os.path.join(SEQ_ROOT, "300mm.seq"), "#6bcb77"),
    (500, os.path.join(SEQ_ROOT, "500mm.seq"), "#ffd93d"),
    (700, os.path.join(SEQ_ROOT, "700mm.seq"), "#ff6b6b"),
]

# ---- Camera -----------------------------------------------------------------
SENSOR_W_PX, SENSOR_H_PX = 640, 480
HFOV_DEG = 45.0    # FOL18 / T199105
def mm_per_px(d): return 2*d*np.tan(np.radians(HFOV_DEG/2))/SENSOR_W_PX

# ---- FLIR scale -------------------------------------------------------------
COUNT_TO_K = 0.04
FPS = 8.0
N_BASE = 5
N_HOT  = 5

# ---- Load -------------------------------------------------------------------

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

# ---- Fit --------------------------------------------------------------------

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

def fit_gaussian_robust(img):
    """Smooth heavily, init from weighted centroid, then fit on the raw image."""
    H, W = img.shape
    xx, yy = np.meshgrid(np.arange(W, dtype=np.float64),
                         np.arange(H, dtype=np.float64))
    smoothed = gaussian_filter(img, sigma=25)
    x0_w, y0_w = weighted_centroid(smoothed)
    # robust width init from variance of the smoothed positive part
    pos = np.clip(smoothed - np.percentile(smoothed, 10), 0, None)
    if pos.sum() > 0:
        var_x = (((xx - x0_w)**2 * pos).sum()/pos.sum())
        var_y = (((yy - y0_w)**2 * pos).sum()/pos.sum())
        sx_init = np.sqrt(max(var_x, 1.0))
        sy_init = np.sqrt(max(var_y, 1.0))
    else:
        sx_init = sy_init = min(W, H)/6.0
    vmax = float(img.max())
    bg0 = float(np.percentile(img, 5))
    p0 = [max(vmax - bg0, 1e-3), x0_w, y0_w, sx_init, sy_init, 0.0, bg0]
    lo = [0, W*0.1, H*0.1, 5, 5, -np.pi/2, -abs(bg0)*3 - 1]
    hi = [vmax*3 + 1, W*0.9, H*0.9, W, H, np.pi/2, vmax + 1]
    popt, pcov = curve_fit(rotated_gaussian, (xx.ravel(), yy.ravel()),
                           img.ravel().astype(np.float64),
                           p0=p0, bounds=(lo, hi), maxfev=30000)
    fitted = rotated_gaussian((xx.ravel(), yy.ravel()), *popt).reshape(H, W)
    return popt, np.sqrt(np.diag(pcov)), fitted, img - fitted, (x0_w, y0_w)

# ---- Process sessions -------------------------------------------------------
sessions = []
for d, seq_path, col in SESSIONS:
    mpp = mm_per_px(d)
    frames = load_seq_frames(seq_path, CACHE_DIR)
    base   = frames[:N_BASE].mean(axis=0)
    hot    = frames[-N_HOT:].mean(axis=0)
    dT     = hot - base
    popt, perr, fitted, resid, (xw, yw) = fit_gaussian_robust(dT)
    A, x0, y0, sx, sy, theta, bg = popt
    if sy > sx:                       # convention sx >= sy
        sx, sy = sy, sx
        theta += np.pi/2
    sessions.append(dict(
        d=d, col=col, mpp=mpp, dT=dT, fitted=fitted, resid=resid,
        A=A, x0=x0, y0=y0, sx=sx, sy=sy, theta=theta, bg=bg,
        sx_mm=sx*mpp, sy_mm=sy*mpp,
        fwhm_x=sx*2.355*mpp, fwhm_y=sy*2.355*mpp,
        peak_dT=float(dT.max()), fit_peak=float(A),
        peak_dT_fit_above_bg=float(A),
        weighted_centroid=(xw, yw),
    ))

# ---- Joint cross-standoff fits ---------------------------------------------
ds   = np.array([s["d"]      for s in sessions], float)
sx_m = np.array([s["sx_mm"]  for s in sessions], float)
sy_m = np.array([s["sy_mm"]  for s in sessions], float)
pks  = np.array([s["peak_dT_fit_above_bg"] for s in sessions], float)

cx = np.polyfit(ds, sx_m, 1)   # sx_mm = cx[0]*d + cx[1]
cy = np.polyfit(ds, sy_m, 1)
# divergence half-angles (1-sigma) from slope:
half_div_x = np.degrees(np.arctan(cx[0]))
half_div_y = np.degrees(np.arctan(cy[0]))
# virtual source position behind aperture (where sigma -> 0):
d_virt_x = -cx[1] / cx[0]
d_virt_y = -cy[1] / cy[0]

# peak dT scaling: fit Q(d) = K * d^p  ->  log fit
log_d = np.log(ds); log_p = np.log(np.maximum(pks, 1e-6))
pcoef = np.polyfit(log_d, log_p, 1)   # log p = pcoef[0]*log d + pcoef[1]
power_exp = pcoef[0]
K_log     = pcoef[1]
K_amp     = np.exp(K_log)
def Q_model(d): return K_amp * d**power_exp

# ---- Self-similarity / normalised overlay ----------------------------------
# resample each radial profile on rho = r/sigma_eff(d) axis
rho_grid = np.linspace(0, 4, 200)
overlay = []
for s in sessions:
    H, W = s["dT"].shape
    yy, xx = np.ogrid[:H, :W]
    r_px = np.sqrt((xx-s["x0"])**2 + (yy-s["y0"])**2).ravel()
    r_mm = r_px * s["mpp"]
    v = (s["dT"] - s["bg"]).ravel()
    sigma_eff_mm = np.sqrt(s["sx_mm"] * s["sy_mm"])
    rho = r_mm / sigma_eff_mm
    peak = s["peak_dT_fit_above_bg"]
    # azimuthal mean in rho bins
    bins = np.linspace(0, 4, 41)
    cnt,_ = np.histogram(rho, bins=bins)
    tot,_ = np.histogram(rho, bins=bins, weights=v)
    mean_v = np.where(cnt>0, tot/cnt, np.nan)
    overlay.append((bins[:-1]+np.diff(bins)/2, mean_v/peak, s))

# Gaussian reference
gauss_ref = np.exp(-0.5 * rho_grid**2)

# ---- Figure -----------------------------------------------------------------
fig = plt.figure(figsize=(20, 22))
fig.patch.set_facecolor("#111111")
gs = mgridspec.GridSpec(4, 6, figure=fig,
                        hspace=0.55, wspace=0.45,
                        left=0.06, right=0.97, top=0.96, bottom=0.05,
                        height_ratios=[1.0, 0.7, 0.8, 0.8])

def dark_ax(ax, grid=True):
    ax.set_facecolor("#1a1a1a")
    for sp in ax.spines.values(): sp.set_edgecolor("#444444")
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    if grid: ax.grid(True, alpha=0.12, color="white")

# determine common dT scale for row 1 so images are comparable
v_global = max(s["peak_dT_fit_above_bg"] for s in sessions)

# Row 1: 3 dT maps in MM axes (common-extent display)
extent_max_mm = max((sensor_size := mm_per_px(s["d"])) * max(SENSOR_W_PX, SENSOR_H_PX)/2 for s in sessions)
for i, s in enumerate(sessions):
    ax = fig.add_subplot(gs[0, i*2:(i+1)*2])
    ax.set_facecolor("#0a0a0a")
    # extent in mm centred on the fit centroid so all panels share the same physical scale
    W = s["dT"].shape[1]; H = s["dT"].shape[0]
    x_extent = ((0 - s["x0"]) * s["mpp"], (W - s["x0"]) * s["mpp"])
    y_extent = ((H - s["y0"]) * s["mpp"], (0 - s["y0"]) * s["mpp"])
    im = ax.imshow(s["dT"] - s["bg"], cmap="inferno", vmin=0, vmax=v_global,
                   extent=[x_extent[0], x_extent[1], y_extent[0], y_extent[1]],
                   origin="upper", aspect="equal")
    # overlay sigma ellipses in mm
    for sc, ls in [(1, "-"), (2, "--"), (2.355/2, ":")]:
        ax.add_patch(Ellipse((0, 0),
                             2*s["sx_mm"]*sc, 2*s["sy_mm"]*sc,
                             angle=-np.degrees(s["theta"]),
                             fc="none", ec=s["col"], lw=1.5, ls=ls, zorder=5))
    ax.plot(0, 0, "+", color="white", ms=10, mew=2, zorder=6)
    ax.set_xlim(-extent_max_mm, extent_max_mm)
    ax.set_ylim(-extent_max_mm*0.75, extent_max_mm*0.75)
    plt.colorbar(im, ax=ax, pad=0.02, label="dT - bg [K]"
                 ).ax.tick_params(labelsize=7, colors="white")
    ax.set_title(f"{s['d']} mm   sigma_x={s['sx_mm']:.0f} mm   sigma_y={s['sy_mm']:.0f} mm   "
                 f"peak dT={s['peak_dT_fit_above_bg']:.2f} K",
                 color=s["col"], fontsize=10, fontweight="bold")
    ax.set_xlabel("x [mm]", color="#aaaaaa", fontsize=8)
    ax.set_ylabel("y [mm]", color="#aaaaaa", fontsize=8)
    ax.tick_params(colors="#888888", labelsize=7)

# Row 2: cross-sections overlay
ax_h = fig.add_subplot(gs[1, 0:3]); dark_ax(ax_h)
ax_v = fig.add_subplot(gs[1, 3:6]); dark_ax(ax_v)
for s in sessions:
    midy = int(round(s["y0"])); midx = int(round(s["x0"]))
    x_mm = (np.arange(s["dT"].shape[1]) - s["x0"]) * s["mpp"]
    y_mm = (np.arange(s["dT"].shape[0]) - s["y0"]) * s["mpp"]
    h_data = s["dT"][midy, :] - s["bg"]
    v_data = s["dT"][:, midx] - s["bg"]
    h_fit  = s["fitted"][midy, :] - s["bg"]
    v_fit  = s["fitted"][:, midx] - s["bg"]
    ax_h.plot(x_mm, h_data, color=s["col"], lw=1.4, alpha=0.7,
              label=f"{s['d']} mm  (sigma_x={s['sx_mm']:.0f} mm)")
    ax_h.plot(x_mm, h_fit, color=s["col"], lw=1.6, ls="--", alpha=0.9)
    ax_v.plot(y_mm, v_data, color=s["col"], lw=1.4, alpha=0.7,
              label=f"{s['d']} mm  (sigma_y={s['sy_mm']:.0f} mm)")
    ax_v.plot(y_mm, v_fit, color=s["col"], lw=1.6, ls="--", alpha=0.9)
ax_h.set_xlim(-extent_max_mm*0.9, extent_max_mm*0.9)
ax_v.set_xlim(-extent_max_mm*0.7, extent_max_mm*0.7)
for ax, ttl in [(ax_h, "Horizontal cross-section (solid=data, dashed=fit)"),
                (ax_v, "Vertical cross-section (solid=data, dashed=fit)")]:
    ax.set_xlabel("Offset from centroid [mm]", color="#aaaaaa", fontsize=9)
    ax.set_ylabel("dT - bg  [K]", color="#aaaaaa", fontsize=9)
    ax.set_title(ttl, color="white", fontsize=10)
    ax.legend(fontsize=8, facecolor="#222222", labelcolor="white", edgecolor="#555555")

# Row 3: normalised radial overlay (self-similarity test) full width
ax_norm = fig.add_subplot(gs[2, :]); dark_ax(ax_norm)
for rho_c, vals, s in overlay:
    ax_norm.plot(rho_c, vals, "o-", color=s["col"], lw=1.5, ms=6,
                 label=f"{s['d']} mm  (sigma_eff={np.sqrt(s['sx_mm']*s['sy_mm']):.0f} mm)")
ax_norm.plot(rho_grid, gauss_ref, color="white", ls="--", lw=1.4, alpha=0.7,
             label="exp(-0.5 rho^2) reference")
ax_norm.set_xlabel("rho = r / sigma_eff(d)", color="#aaaaaa", fontsize=9)
ax_norm.set_ylabel("dT / peak", color="#aaaaaa", fontsize=9)
ax_norm.set_title("Normalised radial profile -- overlay tests beam self-similarity",
                  color="white", fontsize=10)
ax_norm.set_xlim(0, 3.5); ax_norm.set_ylim(-0.1, 1.1)
ax_norm.legend(fontsize=8.5, facecolor="#222222", labelcolor="white", edgecolor="#555555")

# Row 4 (3 panels): sigma vs d, peak vs d, summary card
ax_sig = fig.add_subplot(gs[3, 0:2]); dark_ax(ax_sig)
d_line = np.linspace(0, 600, 200)
for s in sessions:
    ax_sig.scatter(s["d"], s["sx_mm"], s=130, color=s["col"], marker="o", zorder=6)
    ax_sig.scatter(s["d"], s["sy_mm"], s=130, color=s["col"], marker="s", zorder=6)
ax_sig.plot(d_line, np.polyval(cx, d_line), color="#ffd93d", lw=2, ls="--",
            label=f"sigma_x  slope={cx[0]:.4f}  half-div={half_div_x:.2f} deg\n"
                  f"intercept={cx[1]:.1f} mm  virt source @ d={d_virt_x:.0f} mm")
ax_sig.plot(d_line, np.polyval(cy, d_line), color="#4d96ff", lw=2, ls="--",
            label=f"sigma_y  slope={cy[0]:.4f}  half-div={half_div_y:.2f} deg\n"
                  f"intercept={cy[1]:.1f} mm  virt source @ d={d_virt_y:.0f} mm")
ax_sig.scatter([], [], s=80, color="white", marker="o", label="sigma_x")
ax_sig.scatter([], [], s=80, color="white", marker="s", label="sigma_y")
ax_sig.set_xlim(0, 600)
ax_sig.set_xlabel("Standoff d [mm]", color="#aaaaaa", fontsize=9)
ax_sig.set_ylabel("Sigma [mm]", color="#aaaaaa", fontsize=9)
ax_sig.set_title("Beam-width divergence  -- linear joint fit",
                 color="white", fontsize=10)
ax_sig.legend(fontsize=8, facecolor="#222222", labelcolor="white",
              edgecolor="#555555", loc="lower right")

ax_pk = fig.add_subplot(gs[3, 2:4]); dark_ax(ax_pk)
for s in sessions:
    ax_pk.scatter(s["d"], s["peak_dT_fit_above_bg"], s=140, color=s["col"],
                  marker="D", zorder=6)
d_fine = np.linspace(200, 600, 200)
ax_pk.plot(d_fine, Q_model(d_fine), color="white", lw=1.8, ls="--",
           label=f"Q(d) = {K_amp:.2g} * d^{power_exp:.2f}")
ideal = pks[0] * (ds[0]/d_fine)**2
ax_pk.plot(d_fine, ideal, color="cyan", lw=1.2, ls=":",
           alpha=0.7, label=f"1/d^2 anchored at {int(ds[0])} mm")
ax_pk.set_xlabel("Standoff d [mm]", color="#aaaaaa", fontsize=9)
ax_pk.set_ylabel("Fitted peak dT - bg [K]", color="#aaaaaa", fontsize=9)
ax_pk.set_title("Peak dT vs standoff -- empirical power-law fit",
                color="white", fontsize=10)
ax_pk.set_xlim(200, 600)
ax_pk.legend(fontsize=8, facecolor="#222222", labelcolor="white", edgecolor="#555555")

# Summary card
ax_sum = fig.add_subplot(gs[3, 4:6]); ax_sum.axis("off")
ax_sum.set_facecolor("#1a1a1a")
ax_sum.add_patch(plt.Rectangle((0,0),1,1, transform=ax_sum.transAxes,
                                fc="#1a1a1a", ec="#555555", lw=1))
mean_asp = float(np.mean([s["sx_mm"]/s["sy_mm"] for s in sessions]))
text = (
    "DERIVED BEAM MODEL\n"
    "------------------\n"
    f"Camera: FLIR A655sc + FOL18 (45 deg)\n"
    f"Source: unidentified (this dataset)\n\n"
    "Spatial profile (lamp-frame, elliptical Gaussian):\n"
    f"  dT(x,y;d) = peak(d) * exp(-0.5 * ((x/sigma_x(d))^2 + (y/sigma_y(d))^2))\n\n"
    f"  sigma_x(d) = {cx[0]:.4f} * d + {cx[1]:.2f}  mm\n"
    f"  sigma_y(d) = {cy[0]:.4f} * d + {cy[1]:.2f}  mm\n\n"
    f"Half-divergence angles (1-sigma):\n"
    f"  theta_x = {half_div_x:.2f} deg     theta_y = {half_div_y:.2f} deg\n\n"
    f"Virtual source position behind aperture:\n"
    f"  z_virt_x = {-d_virt_x:.0f} mm     z_virt_y = {-d_virt_y:.0f} mm\n\n"
    f"Aspect ratio sigma_x/sigma_y mean = {mean_asp:.3f}\n\n"
    f"Peak amplitude scaling:\n"
    f"  Q(d) = {K_amp:.3g} * d^({power_exp:.3f})\n"
    f"  (ideal 1/d^2 -> exponent = -2)\n"
)
ax_sum.text(0.04, 0.96, text, color="white", fontsize=9.5, family="monospace",
            va="top", ha="left", transform=ax_sum.transAxes)

fig.suptitle(
    "Combined beam analysis  |  300 / 500 / 700 mm @ FLIR A655sc + FOL18\n"
    "Joint fit of dT(x,y) across all standoffs -> single derived beam model for DFLUX",
    color="white", fontsize=13, y=0.985)

fig.savefig(OUTPUT_PNG, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved figure: {OUTPUT_PNG}")

# ---- Save derived beam model ------------------------------------------------
derived = {
    "camera": "FLIR A655sc + FOL18 (45 deg HFOV)",
    "standoffs_mm": [int(s["d"]) for s in sessions],
    "per_standoff": [
        {
            "d_mm": int(s["d"]),
            "mm_per_px": float(s["mpp"]),
            "sigma_x_mm": float(s["sx_mm"]),
            "sigma_y_mm": float(s["sy_mm"]),
            "fwhm_x_mm": float(s["fwhm_x"]),
            "fwhm_y_mm": float(s["fwhm_y"]),
            "peak_dT_K": float(s["peak_dT"]),
            "peak_dT_above_bg_K": float(s["peak_dT_fit_above_bg"]),
            "theta_deg": float(np.degrees(s["theta"])),
            "centroid_px": [float(s["x0"]), float(s["y0"])],
        }
        for s in sessions
    ],
    "derived_beam_model": {
        "profile_form": "dT(x,y;d) = peak(d) * exp(-0.5*((x/sigma_x(d))^2 + (y/sigma_y(d))^2))",
        "sigma_x_vs_d": {"slope_mm_per_mm": float(cx[0]), "intercept_mm": float(cx[1]),
                          "half_div_deg": float(half_div_x),
                          "virtual_source_d_mm": float(d_virt_x)},
        "sigma_y_vs_d": {"slope_mm_per_mm": float(cy[0]), "intercept_mm": float(cy[1]),
                          "half_div_deg": float(half_div_y),
                          "virtual_source_d_mm": float(d_virt_y)},
        "aspect_ratio_x_over_y_mean": float(mean_asp),
        "peak_vs_d_powerlaw": {"K_amp_K_mm_to_power": float(K_amp),
                               "exponent": float(power_exp),
                               "exponent_minus_2_residual": float(power_exp + 2)},
    },
}
with open(RESULTS_JSON, "w") as f:
    json.dump(derived, f, indent=2)
print(f"Saved JSON:   {RESULTS_JSON}")

# ---- Console summary --------------------------------------------------------
print("\n=== Derived beam model ===")
print(f"  sigma_x(d) = {cx[0]:.4f} * d + {cx[1]:.2f} mm   "
      f"(half-div = {half_div_x:.2f} deg, virt-src @ d = {d_virt_x:.0f} mm)")
print(f"  sigma_y(d) = {cy[0]:.4f} * d + {cy[1]:.2f} mm   "
      f"(half-div = {half_div_y:.2f} deg, virt-src @ d = {d_virt_y:.0f} mm)")
print(f"  aspect sigma_x/sigma_y mean = {mean_asp:.3f}")
print(f"  Q(d) = {K_amp:.3g} * d^({power_exp:.3f})   "
      f"(ideal -2; residual = {power_exp+2:+.2f})")
print("\nPer standoff:")
for s in sessions:
    print(f"  {s['d']} mm: sigma_x={s['sx_mm']:.1f}, sigma_y={s['sy_mm']:.1f}, "
          f"peak={s['peak_dT_fit_above_bg']:.2f} K, "
          f"centroid=({s['x0']:.0f},{s['y0']:.0f}) px")
