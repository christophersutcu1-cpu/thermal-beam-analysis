"""
Comprehensive beam shape analysis for 600, 700, 800mm standoff sessions.

For each session shows:
  Row 0 : raw differential thermal image  |  Gaussian fit overlay  |  residuals
  Row 1 : H cross-section  |  V cross-section  |  2D beam ellipses (all 3 overlaid)
  Row 2 : radial profile  |  encircled energy  |  parameter summary table

Bottom panel: sigma_x / sigma_y in mm vs standoff with linear fit.
"""

import os, json, glob
import config
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter

OUTPUT_PNG = config.BOSON_ROOT + r"\beam_shape_analysis.png"

# ── camera geometry ───────────────────────────────────────────────────────────
FOCAL_MM    = 18.0
SENSOR_W_PX = 640
SENSOR_H_PX = 512
SENSOR_W_MM = SENSOR_W_PX * 0.012
HFOV_DEG    = 2 * np.degrees(np.arctan(SENSOR_W_MM / (2 * FOCAL_MM)))

def mm_per_px(d): return 2*d*np.tan(np.radians(HFOV_DEG/2)) / SENSOR_W_PX

# ── capture timing ────────────────────────────────────────────────────────────
FPS              = 9.0
PRE_RELAY_SECS   = 1.0
RELAY_PULSE_SECS = 1.0
POST_RELAY_SECS  = 10.0
STEADY_START     = PRE_RELAY_SECS + 2.0
STEADY_END       = PRE_RELAY_SECS + RELAY_PULSE_SECS + POST_RELAY_SECS - 1.0

SESSIONS = [
    (600, config.BOSON_ROOT + r"\600 mm\boson_20260422_113853"),
    (700, config.BOSON_ROOT + r"\700 mm\boson_20260422_114608"),
    (800, config.BOSON_ROOT + r"\800 mm\boson_20260422_115822"),
]
SESSION_COLS = ["#6bcb77", "#ffd93d", "#ff6b6b"]

# ── helpers ───────────────────────────────────────────────────────────────────

def load_tiffs(directory):
    paths  = sorted(glob.glob(os.path.join(directory, "frame_*.tiff")))
    frames = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is not None:
            frames.append((img[:,:,0] if img.ndim==3 else img).astype(np.float32))
    return np.array(frames)

def rotated_gaussian(xy, A, x0, y0, sx, sy, theta, bg):
    x, y = xy
    ct, st = np.cos(theta), np.sin(theta)
    xr =  ct*(x-x0) + st*(y-y0)
    yr = -st*(x-x0) + ct*(y-y0)
    return A*np.exp(-0.5*((xr/sx)**2+(yr/sy)**2)) + bg

def fit_gaussian(img):
    H, W = img.shape
    xx, yy = np.meshgrid(np.arange(W,dtype=np.float64), np.arange(H,dtype=np.float64))
    blurred = gaussian_filter(img, sigma=15)
    pk = np.unravel_index(np.argmax(blurred), blurred.shape)
    x0s, y0s = float(pk[1]), float(pk[0])
    vmax = float(img.max()); bg0 = float(np.percentile(img,5))
    s0 = min(W,H)/5
    p0 = [vmax-bg0, x0s, y0s, s0, s0, 0.0, bg0]
    lo = [0, 0, 0, 1, 1, -np.pi/2, -abs(bg0)*3]
    hi = [vmax*3, W, H, W, H, np.pi/2, vmax]
    popt, pcov = curve_fit(rotated_gaussian, (xx.ravel(),yy.ravel()),
                           img.ravel().astype(np.float64),
                           p0=p0, bounds=(lo,hi), maxfev=20000)
    fitted = rotated_gaussian((xx.ravel(),yy.ravel()), *popt).reshape(H,W)
    return popt, np.sqrt(np.diag(pcov)), fitted, img-fitted

def radial_profile(img, cx, cy, max_r):
    H,W = img.shape
    yy,xx = np.ogrid[:H,:W]
    r = np.sqrt((xx-cx)**2+(yy-cy)**2).ravel()
    v = img.ravel()
    bins = np.arange(0, max_r+1, 1.0)
    cnt,_ = np.histogram(r, bins=bins)
    tot,_ = np.histogram(r, bins=bins, weights=v)
    return bins[:-1], np.where(cnt>0, tot/cnt, 0.0)

def encircled_energy(img, cx, cy, max_r):
    H,W = img.shape
    yy,xx = np.ogrid[:H,:W]
    r   = np.sqrt((xx-cx)**2+(yy-cy)**2)
    pos = np.clip(img,0,None); tot = pos.sum()
    rad = np.arange(0, max_r+1, 1.0)
    ee  = np.array([(pos[r<=ri]).sum()/tot for ri in rad])
    return rad, ee

def dark_ax(ax, grid=True):
    ax.set_facecolor("#1a1a1a")
    for sp in ax.spines.values(): sp.set_edgecolor("#444444")
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    if grid: ax.grid(True, alpha=0.12, color="white")

# ── process all sessions ──────────────────────────────────────────────────────
results = []
for standoff, tiff_dir in SESSIONS:
    mpp = mm_per_px(standoff)
    print(f"\n{'='*50}\n{standoff}mm  ({mpp:.4f} mm/px)")

    frames  = load_tiffs(tiff_dir)
    i0_base = 0
    i1_base = max(1, int(PRE_RELAY_SECS*FPS)-1)
    i0_hot  = int(STEADY_START*FPS)
    i1_hot  = min(int(STEADY_END*FPS), len(frames))

    baseline = frames[i0_base:i1_base].mean(axis=0)
    hot      = frames[i0_hot:i1_hot].mean(axis=0)
    diff     = hot - baseline
    fit_in   = diff - diff.min()

    print(f"  Baseline frames: {i0_base}-{i1_base-1}  Hot frames: {i0_hot}-{i1_hot-1}")

    popt, perr, fitted, resid = fit_gaussian(fit_in)
    A, x0, y0, sx, sy, theta, bg = popt

    # enforce sx >= sy convention (sx = wider axis)
    if sy > sx:
        sx, sy = sy, sx
        theta += np.pi/2

    sx_mm  = sx * mpp
    sy_mm  = sy * mpp
    fwhm_x = sx * 2.355 * mpp
    fwhm_y = sy * 2.355 * mpp

    print(f"  Centre: ({x0:.1f}, {y0:.1f}) px")
    print(f"  sigma_x={sx:.1f}px = {sx_mm:.1f}mm   FWHM={fwhm_x:.1f}mm")
    print(f"  sigma_y={sy:.1f}px = {sy_mm:.1f}mm   FWHM={fwhm_y:.1f}mm")
    print(f"  theta={np.degrees(theta):.1f}deg   aspect={sx/sy:.3f}")

    max_r = int(min(SENSOR_W_PX, SENSOR_H_PX)/2)
    r_bins, r_mean = radial_profile(fit_in, x0, y0, max_r)
    r_rad,  r_ee   = encircled_energy(fit_in, x0, y0, max_r)

    results.append(dict(
        standoff=standoff, mpp=mpp,
        diff=diff, fit_in=fit_in, fitted=fitted, resid=resid,
        A=A, x0=x0, y0=y0, sx=sx, sy=sy, theta=theta, bg=bg,
        sx_mm=sx_mm, sy_mm=sy_mm, fwhm_x=fwhm_x, fwhm_y=fwhm_y,
        r_bins=r_bins, r_mean=r_mean, r_rad=r_rad, r_ee=r_ee,
    ))

# ── figure ────────────────────────────────────────────────────────────────────
n = len(results)
fig = plt.figure(figsize=(22, 28))
fig.patch.set_facecolor("#111111")

# GridSpec: 3 session blocks (4 rows each) + 1 summary row
gs_outer = mgridspec.GridSpec(4, 1, figure=fig,
                               hspace=0.55, wspace=0.3,
                               left=0.06, right=0.97,
                               top=0.96, bottom=0.04,
                               height_ratios=[1,1,1,0.7])

CMAP_HOT  = "inferno"
CMAP_DIFF = "RdBu_r"

for si, (r, col) in enumerate(zip(results, SESSION_COLS)):
    d = r["standoff"]
    gs_s = mgridspec.GridSpecFromSubplotSpec(3, 4, subplot_spec=gs_outer[si],
                                              hspace=0.55, wspace=0.3)

    # title strip
    fig.text(0.5, gs_outer[si].get_position(fig).y1 + 0.005,
             f"── {d}mm standoff  |  {r['mpp']:.4f} mm/px  |  "
             f"σ_x={r['sx_mm']:.1f}mm  σ_y={r['sy_mm']:.1f}mm  |  "
             f"FWHM {r['fwhm_x']:.0f}×{r['fwhm_y']:.0f}mm  |  "
             f"θ={np.degrees(r['theta']):.1f}°",
             ha="center", color=col, fontsize=10, fontweight="bold")

    vmax_d = np.percentile(r["diff"], 99)
    vmin_d = np.percentile(r["diff"], 1)
    vmax_f = r["fit_in"].max()

    # ── col 0: raw differential image ────────────────────────────────────────
    ax = fig.add_subplot(gs_s[0, 0])
    ax.set_facecolor("#0a0a0a")
    im = ax.imshow(r["diff"], cmap=CMAP_DIFF, vmin=vmin_d, vmax=vmax_d,
                   origin="upper", aspect="equal")
    plt.colorbar(im, ax=ax, pad=0.02).ax.tick_params(labelsize=7, colors="white")
    ax.set_title("Raw ΔT (hot − baseline)", color="white", fontsize=8)
    ax.set_xlabel("px", color="#aaaaaa", fontsize=7)
    ax.set_ylabel("px", color="#aaaaaa", fontsize=7)
    ax.tick_params(colors="#666666", labelsize=7)

    # ── col 1: differential + Gaussian fit ellipses ───────────────────────────
    ax = fig.add_subplot(gs_s[0, 1])
    ax.set_facecolor("#0a0a0a")
    ax.imshow(r["fit_in"], cmap=CMAP_HOT, vmin=0, vmax=vmax_f,
              origin="upper", aspect="equal")
    # 1-sigma, 2-sigma, FWHM ellipses
    for scale, ls, lbl in [(1, "-", "1σ"), (2, "--", "2σ"),
                            (2.355/2, ":", "FWHM")]:
        ax.add_patch(Ellipse((r["x0"], r["y0"]),
                             2*r["sx"]*scale, 2*r["sy"]*scale,
                             angle=np.degrees(r["theta"]),
                             fc="none", ec=col, lw=1.5, ls=ls, zorder=5))
    ax.plot(r["x0"], r["y0"], "+", color="white", ms=10, mew=2, zorder=6)
    ax.set_title("ΔT + fit ellipses (1σ / 2σ / FWHM)", color="white", fontsize=8)
    ax.tick_params(colors="#666666", labelsize=7)
    ax.set_xlabel("px", color="#aaaaaa", fontsize=7)

    # ── col 2: Gaussian fit ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs_s[0, 2])
    ax.set_facecolor("#0a0a0a")
    im2 = ax.imshow(r["fitted"], cmap=CMAP_HOT, vmin=0, vmax=vmax_f,
                    origin="upper", aspect="equal")
    plt.colorbar(im2, ax=ax, pad=0.02).ax.tick_params(labelsize=7, colors="white")
    ax.set_title("2D Gaussian fit", color="white", fontsize=8)
    ax.tick_params(colors="#666666", labelsize=7)
    ax.set_xlabel("px", color="#aaaaaa", fontsize=7)

    # ── col 3: residuals ───────────────────────────────────────────────────────
    ax = fig.add_subplot(gs_s[0, 3])
    ax.set_facecolor("#0a0a0a")
    rlim = np.percentile(np.abs(r["resid"]), 98)
    im3 = ax.imshow(r["resid"], cmap="RdBu_r", vmin=-rlim, vmax=rlim,
                    origin="upper", aspect="equal")
    plt.colorbar(im3, ax=ax, pad=0.02).ax.tick_params(labelsize=7, colors="white")
    rms = float(np.sqrt((r["resid"]**2).mean()))
    ax.set_title(f"Residuals  (RMS={rms:.2f})", color="white", fontsize=8)
    ax.tick_params(colors="#666666", labelsize=7)
    ax.set_xlabel("px", color="#aaaaaa", fontsize=7)

    # ── row 1: cross-sections ─────────────────────────────────────────────────
    mid_x = int(round(r["x0"]))
    mid_y = int(round(r["y0"]))
    H, W  = r["fit_in"].shape

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
    ax_h.text(r["sx_mm"]+2, h_data.max()*0.5, f"±σ_x\n{r['sx_mm']:.0f}mm",
              color=col, fontsize=7)
    ax_h.text(r["fwhm_x"]/2+2, h_data.max()*0.2, f"FWHM\n{r['fwhm_x']:.0f}mm",
              color="lime", fontsize=7)
    ax_h.set_xlabel("Distance from centre (mm)", color="#aaaaaa", fontsize=8)
    ax_h.set_ylabel("ΔT (counts)", color="#aaaaaa", fontsize=8)
    ax_h.set_title(f"Horizontal cross-section through beam centre  (σ_x={r['sx_mm']:.1f}mm)",
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
    ax_v.text(r["sy_mm"]+2, v_data.max()*0.5, f"±σ_y\n{r['sy_mm']:.0f}mm",
              color=col, fontsize=7)
    ax_v.text(r["fwhm_y"]/2+2, v_data.max()*0.2, f"FWHM\n{r['fwhm_y']:.0f}mm",
              color="lime", fontsize=7)
    ax_v.set_xlabel("Distance from centre (mm)", color="#aaaaaa", fontsize=8)
    ax_v.set_ylabel("ΔT (counts)", color="#aaaaaa", fontsize=8)
    ax_v.set_title(f"Vertical cross-section through beam centre  (σ_y={r['sy_mm']:.1f}mm)",
                   color="white", fontsize=8)
    ax_v.legend(fontsize=7.5, facecolor="#222222", labelcolor="white",
                edgecolor="#555555", loc="upper right")

    # ── row 2: radial profile + encircled energy ──────────────────────────────
    r_mm    = r["r_bins"] * r["mpp"]      # length = max_r  (radial profile)
    r_ee_mm = r["r_rad"]  * r["mpp"]      # length = max_r+1 (encircled energy)

    ax_r = fig.add_subplot(gs_s[2, 0:2])
    dark_ax(ax_r)
    peak_r = r["r_mean"].max()
    ax_r.plot(r_mm, r["r_mean"]/peak_r, color=col, lw=2, label="Azimuthal mean")
    # theoretical Gaussian radial profile
    r_theory = np.linspace(0, r_mm[-1], 300)
    sigma_eff = np.sqrt(r["sx_mm"]*r["sy_mm"])   # geometric mean sigma
    ax_r.plot(r_theory, np.exp(-0.5*(r_theory/sigma_eff)**2),
              color="white", lw=1.5, ls="--", alpha=0.7,
              label=f"Gaussian  σ_eff={sigma_eff:.0f}mm")
    ax_r.axvline(sigma_eff,       color="white", lw=0.8, ls=":", alpha=0.5)
    ax_r.axvline(sigma_eff*2.355/2, color="lime", lw=0.8, ls="--", alpha=0.5)
    ax_r.text(sigma_eff+1, 0.65, f"σ_eff\n{sigma_eff:.0f}mm", color="white", fontsize=7)
    ax_r.text(sigma_eff*2.355/2+1, 0.45, f"FWHM/2\n{sigma_eff*2.355/2:.0f}mm",
              color="lime", fontsize=7)
    ax_r.set_xlabel("Radius from centre (mm)", color="#aaaaaa", fontsize=8)
    ax_r.set_ylabel("Norm. mean irradiance", color="#aaaaaa", fontsize=8)
    ax_r.set_title("Azimuthally averaged radial profile", color="white", fontsize=8)
    ax_r.set_xlim(0, min(200, r_mm[-1]))
    ax_r.legend(fontsize=7.5, facecolor="#222222", labelcolor="white", edgecolor="#555555")

    ax_e = fig.add_subplot(gs_s[2, 2:4])
    dark_ax(ax_e)
    ax_e.plot(r_ee_mm, r["r_ee"]*100, color=col, lw=2)
    for pct, lc in [(50,"white"), (86,"lime"), (95,"#ff6b6b")]:
        idx = np.searchsorted(r["r_ee"]*100, pct)
        if idx < len(r_ee_mm):
            ax_e.axhline(pct, color=lc, lw=0.8, ls="--", alpha=0.6)
            ax_e.axvline(r_ee_mm[idx], color=lc, lw=0.8, ls="--", alpha=0.6)
            ax_e.text(r_ee_mm[idx]+1, pct-4, f"D{pct}={r_ee_mm[idx]:.0f}mm",
                      color=lc, fontsize=7)
    ax_e.set_xlabel("Radius from centre (mm)", color="#aaaaaa", fontsize=8)
    ax_e.set_ylabel("Encircled energy (%)", color="#aaaaaa", fontsize=8)
    ax_e.set_title("Encircled energy — D50 / D86 / D95", color="white", fontsize=8)
    ax_e.set_xlim(0, min(200, r_ee_mm[-1]))
    ax_e.set_ylim(0, 105)

# ── bottom panel: sigma vs standoff summary ───────────────────────────────────
gs_bot = mgridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_outer[3],
                                            wspace=0.35)

standoffs_fit = np.array([r["standoff"] for r in results], float)
sx_mm_fit     = np.array([r["sx_mm"]    for r in results], float)
sy_mm_fit     = np.array([r["sy_mm"]    for r in results], float)
cx = np.polyfit(standoffs_fit, sx_mm_fit, 1)
cy = np.polyfit(standoffs_fit, sy_mm_fit, 1)
d_line = np.linspace(550, 850, 200)

# sigma vs standoff
ax_s = fig.add_subplot(gs_bot[0])
dark_ax(ax_s)
for r, col in zip(results, SESSION_COLS):
    ax_s.scatter(r["standoff"], r["sx_mm"], s=120, color=col, marker="o", zorder=5)
    ax_s.scatter(r["standoff"], r["sy_mm"], s=120, color=col, marker="s", zorder=5)
ax_s.plot(d_line, np.polyval(cx, d_line), color="#ffd93d", lw=2, ls="--",
          label=f"σ_x fit  ({cx[0]*1e3:+.1f}mm/m)")
ax_s.plot(d_line, np.polyval(cy, d_line), color="#4d96ff", lw=2, ls="--",
          label=f"σ_y fit  ({cy[0]*1e3:+.1f}mm/m)")
ax_s.scatter([], [], s=80, color="white", marker="o", label="σ_x measured")
ax_s.scatter([], [], s=80, color="white", marker="s", label="σ_y measured")
ax_s.set_xlabel("Standoff (mm)", color="#aaaaaa", fontsize=9)
ax_s.set_ylabel("Sigma (mm)", color="#aaaaaa", fontsize=9)
ax_s.set_title("σ_x / σ_y in mm vs standoff\n(600–800mm, nearly constant → collimated beam)",
               color="white", fontsize=9)
ax_s.legend(fontsize=7.5, facecolor="#222222", labelcolor="white",
            edgecolor="#555555", loc="right")

# FWHM vs standoff
ax_f = fig.add_subplot(gs_bot[1])
dark_ax(ax_f)
for r, col in zip(results, SESSION_COLS):
    ax_f.scatter(r["standoff"], r["fwhm_x"], s=120, color=col, marker="o", zorder=5)
    ax_f.scatter(r["standoff"], r["fwhm_y"], s=120, color=col, marker="s", zorder=5)
    ax_f.text(r["standoff"]+5, r["fwhm_x"]+2, f"{r['fwhm_x']:.0f}",
              color=col, fontsize=8)
    ax_f.text(r["standoff"]+5, r["fwhm_y"]-8, f"{r['fwhm_y']:.0f}",
              color=col, fontsize=8)
ax_f.axhline(320, color="lime", lw=1.2, ls="--", alpha=0.7, label="Specimen width 320mm")
ax_f.axhline(175, color="lime", lw=1.2, ls=":",  alpha=0.7, label="Specimen height 175mm")
ax_f.scatter([], [], s=80, color="white", marker="o", label="FWHM_x")
ax_f.scatter([], [], s=80, color="white", marker="s", label="FWHM_y")
ax_f.set_xlabel("Standoff (mm)", color="#aaaaaa", fontsize=9)
ax_f.set_ylabel("FWHM (mm)", color="#aaaaaa", fontsize=9)
ax_f.set_title("FWHM vs standoff vs specimen dimensions",
               color="white", fontsize=9)
ax_f.legend(fontsize=7.5, facecolor="#222222", labelcolor="white",
            edgecolor="#555555", loc="right")

# aspect ratio
ax_a = fig.add_subplot(gs_bot[2])
dark_ax(ax_a)
for r, col in zip(results, SESSION_COLS):
    ax_a.scatter(r["standoff"], r["sx_mm"]/r["sy_mm"],
                 s=140, color=col, marker="D", zorder=5)
    ax_a.text(r["standoff"]+5, r["sx_mm"]/r["sy_mm"]+0.02,
              f"{r['sx_mm']/r['sy_mm']:.2f}", color=col, fontsize=9)
mean_asp = np.mean([r["sx_mm"]/r["sy_mm"] for r in results])
ax_a.axhline(mean_asp, color="white", lw=1.5, ls="--", alpha=0.6)
ax_a.text(815, mean_asp+0.02, f"mean={mean_asp:.2f}", color="white", fontsize=8)
ax_a.set_xlabel("Standoff (mm)", color="#aaaaaa", fontsize=9)
ax_a.set_ylabel("Aspect ratio σ_x / σ_y", color="#aaaaaa", fontsize=9)
ax_a.set_title("Beam aspect ratio\n(consistent → intrinsic lamp property)",
               color="white", fontsize=9)

fig.suptitle(
    "Beam shape analysis  |  18mm lens, Boson 640  |  600 / 700 / 800mm standoff\n"
    "2D Gaussian fitted to baseline-subtracted differential thermal image",
    color="white", fontsize=12, y=0.975)

fig.savefig(OUTPUT_PNG, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"\nSaved: {OUTPUT_PNG}")
