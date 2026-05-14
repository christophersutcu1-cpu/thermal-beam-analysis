"""
Digestable beam shape summary.

Story: three standoff measurements → overlaid cross-sections → consistent shape
       → linear fit → consensus beam used in all simulations.

Layout
------
Row 0 (images)   : differential image + ellipses for 600 / 700 / 800mm
Row 1 (overlay)  : H cross-sections all 3 | V cross-sections all 3 | radial profiles all 3
Row 2 (derivation): sigma vs standoff + linear fit | consensus beam diagram | encircled energy
"""

import os, glob, json
import config
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter

OUTPUT_PNG = config.BOSON_ROOT + r"\beam_shape_summary.png"

# ── camera / timing ───────────────────────────────────────────────────────────
FOCAL_MM    = 18.0
SENSOR_W_PX = 640
SENSOR_W_MM = SENSOR_W_PX * 0.012
HFOV_DEG    = 2 * np.degrees(np.arctan(SENSOR_W_MM / (2 * FOCAL_MM)))
FPS         = 9.0

def mpp(d): return 2*d*np.tan(np.radians(HFOV_DEG/2)) / SENSOR_W_PX

SESSIONS = [
    (600, config.BOSON_ROOT + r"\600 mm\boson_20260422_113853"),
    (700, config.BOSON_ROOT + r"\700 mm\boson_20260422_114608"),
    (800, config.BOSON_ROOT + r"\800 mm\boson_20260422_115822"),
]
COLS = ["#6bcb77", "#ffd93d", "#ff6b6b"]

# ── helpers ───────────────────────────────────────────────────────────────────

def load_tiffs(d):
    paths = sorted(glob.glob(os.path.join(d, "frame_*.tiff")))
    frames = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is not None:
            frames.append((img[:,:,0] if img.ndim==3 else img).astype(np.float32))
    return np.array(frames)

def rotated_gaussian(xy, A, x0, y0, sx, sy, theta, bg):
    x, y = xy
    ct, st = np.cos(theta), np.sin(theta)
    xr =  ct*(x-x0)+st*(y-y0)
    yr = -st*(x-x0)+ct*(y-y0)
    return A*np.exp(-0.5*((xr/sx)**2+(yr/sy)**2))+bg

def fit_gaussian(img):
    H, W = img.shape
    xx, yy = np.meshgrid(np.arange(W,dtype=np.float64), np.arange(H,dtype=np.float64))
    blurred = gaussian_filter(img, sigma=15)
    pk  = np.unravel_index(np.argmax(blurred), blurred.shape)
    x0s, y0s = float(pk[1]), float(pk[0])
    vmax = float(img.max()); bg0 = float(np.percentile(img,5))
    p0 = [vmax-bg0, x0s, y0s, min(W,H)/5, min(W,H)/5, 0.0, bg0]
    lo = [0,0,0,1,1,-np.pi/2,-abs(bg0)*3]
    hi = [vmax*3,W,H,W,H,np.pi/2,vmax]
    popt, _ = curve_fit(rotated_gaussian, (xx.ravel(),yy.ravel()),
                        img.ravel().astype(np.float64),
                        p0=p0, bounds=(lo,hi), maxfev=20000)
    fitted = rotated_gaussian((xx.ravel(),yy.ravel()), *popt).reshape(H,W)
    return popt, fitted

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
    rad = np.arange(0, max_r, 1.0)
    ee  = np.array([(pos[r<=ri]).sum()/tot for ri in rad])
    return rad, ee

def dark_ax(ax):
    ax.set_facecolor("#1a1a1a")
    for sp in ax.spines.values(): sp.set_edgecolor("#444444")
    ax.tick_params(colors="#aaaaaa", labelsize=9)
    ax.grid(True, alpha=0.12, color="white")

# ── process ───────────────────────────────────────────────────────────────────
results = []
for standoff, tiff_dir in SESSIONS:
    scale = mpp(standoff)
    frames = load_tiffs(tiff_dir)
    baseline = frames[0:max(1,int(1.0*FPS)-1)].mean(axis=0)
    hot      = frames[int(3.0*FPS):min(int(11.0*FPS),len(frames))].mean(axis=0)
    diff     = hot - baseline
    fit_in   = diff - diff.min()
    popt, fitted = fit_gaussian(fit_in)
    A, x0, y0, sx, sy, theta, bg = popt
    if sy > sx:
        sx, sy = sy, sx; theta += np.pi/2
    max_r = int(min(fit_in.shape)/2)
    r_bins, r_mean = radial_profile(fit_in, x0, y0, max_r)
    r_rad,  r_ee   = encircled_energy(fit_in, x0, y0, max_r)
    results.append(dict(
        standoff=standoff, scale=scale,
        diff=diff, fit_in=fit_in, fitted=fitted,
        x0=x0, y0=y0, sx=sx, sy=sy, theta=theta,
        sx_mm=sx*scale, sy_mm=sy*scale,
        fwhm_x=sx*2.355*scale, fwhm_y=sy*2.355*scale,
        r_bins=r_bins, r_mean=r_mean, r_rad=r_rad, r_ee=r_ee,
    ))
    print(f"{standoff}mm  sigma_x={sx*scale:.1f}mm  sigma_y={sy*scale:.1f}mm  "
          f"FWHM {sx*2.355*scale:.0f}x{sy*2.355*scale:.0f}mm")

# consensus values (mean of 600-800mm)
sx_vals = np.array([r["sx_mm"] for r in results])
sy_vals = np.array([r["sy_mm"] for r in results])
sx_mean = sx_vals.mean(); sy_mean = sy_vals.mean()
print(f"\nConsensus: sigma_x={sx_mean:.1f}mm  sigma_y={sy_mean:.1f}mm")

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 34))
fig.patch.set_facecolor("#111111")
gs = mgridspec.GridSpec(5, 3, figure=fig, hspace=0.5, wspace=0.32,
                        left=0.06, right=0.97, top=0.96, bottom=0.02,
                        height_ratios=[1, 1, 1, 1.1, 1.1])

CMAP = "inferno"
vmax_global = max(r["fit_in"].max() for r in results)

# ════════════════════════════════════════════════════════
# ROW 0 — differential image + fit ellipses per session
# ════════════════════════════════════════════════════════
for ci, (r, col) in enumerate(zip(results, COLS)):
    ax = fig.add_subplot(gs[0, ci])
    ax.set_facecolor("#0a0a0a")
    ax.imshow(r["fit_in"], cmap=CMAP, origin="upper", aspect="equal",
              vmin=0, vmax=vmax_global)
    # 1sigma and FWHM ellipses
    for scale_e, ls, label in [(1,"-","1sigma"), (2.355/2,"--","FWHM")]:
        ax.add_patch(Ellipse((r["x0"],r["y0"]),
                             2*r["sx"]*scale_e, 2*r["sy"]*scale_e,
                             angle=np.degrees(r["theta"]),
                             fc="none", ec=col, lw=2, ls=ls, zorder=5))
    ax.plot(r["x0"], r["y0"], "+", color="white", ms=12, mew=2, zorder=6)

    # sigma arrows in image
    # Gaussian: xr = cos(t)*(x-x0)+sin(t)*(y-y0), yr = -sin(t)*(x-x0)+cos(t)*(y-y0)
    # sigma_x axis in data (y-down) coords: (cos t, +sin t)
    # sigma_y axis in data (y-down) coords: (-sin t, +cos t)
    sa = r["sx"]; ca = r["sy"]
    ct = np.cos(r["theta"]); st = np.sin(r["theta"])
    ax.annotate("", xy=(r["x0"]+sa*ct, r["y0"]+sa*st),
                xytext=(r["x0"], r["y0"]),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.5,
                                mutation_scale=12), zorder=7)
    ax.text(r["x0"]+sa*ct/2+4, r["y0"]+sa*st/2-10,
            f"sigma_x={r['sx_mm']:.0f}mm", color=col, fontsize=8, fontweight="bold")
    ax.annotate("", xy=(r["x0"]-ca*st, r["y0"]+ca*ct),
                xytext=(r["x0"], r["y0"]),
                arrowprops=dict(arrowstyle="-|>", color="#aaaaaa", lw=1.5,
                                mutation_scale=12), zorder=7)
    ax.text(r["x0"]-ca*st+4, r["y0"]+ca*ct+4,
            f"sigma_y={r['sy_mm']:.0f}mm", color="#aaaaaa", fontsize=8)

    ax.set_xlim(0,640); ax.set_ylim(512,0)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_edgecolor(col); sp.set_linewidth(2.5)
    ax.set_title(f"{r['standoff']}mm standoff  |  {r['scale']:.4f} mm/px\n"
                 f"FWHM  {r['fwhm_x']:.0f} x {r['fwhm_y']:.0f} mm",
                 color=col, fontsize=10, fontweight="bold")

# ════════════════════════════════════════════════════════
# ROW 1 — overlaid cross-sections and radial profiles
# ════════════════════════════════════════════════════════

# H cross-sections
ax_h = fig.add_subplot(gs[1, 0])
dark_ax(ax_h)
for r, col in zip(results, COLS):
    H, W = r["fit_in"].shape
    mid_y = int(round(r["y0"]))
    h_data = r["fit_in"][mid_y, :]
    x_mm   = (np.arange(W) - r["x0"]) * r["scale"]
    ax_h.plot(x_mm, h_data / h_data.max(), color=col, lw=2,
              label=f"{r['standoff']}mm  sigma_x={r['sx_mm']:.0f}mm")
    ax_h.axvline( r["sx_mm"], color=col, lw=0.8, ls=":", alpha=0.5)
    ax_h.axvline(-r["sx_mm"], color=col, lw=0.8, ls=":", alpha=0.5)

ax_h.axhline(0.5, color="white", lw=0.8, ls="--", alpha=0.4, label="FWHM level (0.5)")
ax_h.axhline(np.exp(-0.5), color="white", lw=0.8, ls=":", alpha=0.4, label="1sigma level (0.607)")
ax_h.set_xlabel("Distance from beam centre (mm)", color="#aaaaaa", fontsize=9)
ax_h.set_ylabel("Normalised ΔT", color="#aaaaaa", fontsize=9)
ax_h.set_title("Horizontal cross-section — all 3 standoffs overlaid",
               color="white", fontsize=10)
ax_h.legend(fontsize=8.5, facecolor="#222222", labelcolor="white", edgecolor="#555555")
ax_h.set_xlim(-300, 300)

# V cross-sections
ax_v = fig.add_subplot(gs[1, 1])
dark_ax(ax_v)
for r, col in zip(results, COLS):
    H, W = r["fit_in"].shape
    mid_x = int(round(r["x0"]))
    v_data = r["fit_in"][:, mid_x]
    y_mm   = (np.arange(H) - r["y0"]) * r["scale"]
    ax_v.plot(y_mm, v_data / v_data.max(), color=col, lw=2,
              label=f"{r['standoff']}mm  sigma_y={r['sy_mm']:.0f}mm")
    ax_v.axvline( r["sy_mm"], color=col, lw=0.8, ls=":", alpha=0.5)
    ax_v.axvline(-r["sy_mm"], color=col, lw=0.8, ls=":", alpha=0.5)

ax_v.axhline(0.5, color="white", lw=0.8, ls="--", alpha=0.4)
ax_v.axhline(np.exp(-0.5), color="white", lw=0.8, ls=":", alpha=0.4)
ax_v.set_xlabel("Distance from beam centre (mm)", color="#aaaaaa", fontsize=9)
ax_v.set_ylabel("Normalised ΔT", color="#aaaaaa", fontsize=9)
ax_v.set_title("Vertical cross-section — all 3 standoffs overlaid",
               color="white", fontsize=10)
ax_v.legend(fontsize=8.5, facecolor="#222222", labelcolor="white", edgecolor="#555555")
ax_v.set_xlim(-200, 200)

# Radial profiles
ax_r = fig.add_subplot(gs[1, 2])
dark_ax(ax_r)
for r, col in zip(results, COLS):
    r_mm_ax = r["r_bins"] * r["scale"]
    norm    = r["r_mean"].max()
    ax_r.plot(r_mm_ax, r["r_mean"]/norm, color=col, lw=2,
              label=f"{r['standoff']}mm")
    sigma_eff = np.sqrt(r["sx_mm"]*r["sy_mm"])
    ax_r.axvline(sigma_eff, color=col, lw=0.8, ls=":", alpha=0.5)

# theoretical Gaussian with consensus sigma
sig_eff_mean = np.sqrt(sx_mean*sy_mean)
r_theory = np.linspace(0, 250, 400)
ax_r.plot(r_theory, np.exp(-0.5*(r_theory/sig_eff_mean)**2),
          color="white", lw=2.5, ls="--",
          label=f"Theory  sigma_eff={sig_eff_mean:.0f}mm")
ax_r.set_xlabel("Radius from centre (mm)", color="#aaaaaa", fontsize=9)
ax_r.set_ylabel("Normalised mean irradiance", color="#aaaaaa", fontsize=9)
ax_r.set_title("Azimuthal radial profile — all 3 overlaid vs theory",
               color="white", fontsize=10)
ax_r.legend(fontsize=8.5, facecolor="#222222", labelcolor="white", edgecolor="#555555")
ax_r.set_xlim(0, 250)

# ════════════════════════════════════════════════════════
# ROW 2 — derivation: sigma fit → consensus beam diagram
# ════════════════════════════════════════════════════════

# sigma vs standoff
ax_sig = fig.add_subplot(gs[2, 0])
dark_ax(ax_sig)
d_arr = np.array([r["standoff"] for r in results], float)
cx    = np.polyfit(d_arr, sx_vals, 1)
cy    = np.polyfit(d_arr, sy_vals, 1)
d_line = np.linspace(550, 850, 200)

for r, col in zip(results, COLS):
    ax_sig.scatter(r["standoff"], r["sx_mm"], s=150, color=col,
                   marker="o", zorder=6)
    ax_sig.scatter(r["standoff"], r["sy_mm"], s=150, color=col,
                   marker="s", zorder=6)
    ax_sig.text(r["standoff"]+5, r["sx_mm"]+1.5,
                f"{r['sx_mm']:.0f}", color=col, fontsize=8.5, fontweight="bold")
    ax_sig.text(r["standoff"]+5, r["sy_mm"]-5,
                f"{r['sy_mm']:.0f}", color=col, fontsize=8.5)

ax_sig.plot(d_line, np.polyval(cx, d_line), color="#ffd93d", lw=2, ls="--",
            label=f"sigma_x  ({cx[0]*1e3:+.1f} mm/m·d)")
ax_sig.plot(d_line, np.polyval(cy, d_line), color="#4d96ff", lw=2, ls="--",
            label=f"sigma_y  ({cy[0]*1e3:+.1f} mm/m·d)")
ax_sig.scatter([],[],s=100,color="white",marker="o",label="sigma_x measured")
ax_sig.scatter([],[],s=100,color="white",marker="s",label="sigma_y measured")

# reference star at 700mm
ax_sig.scatter([700],[results[1]["sx_mm"]], s=300, color="lime",
               marker="*", zorder=8)
ax_sig.scatter([700],[results[1]["sy_mm"]], s=300, color="lime",
               marker="*", zorder=8)
ax_sig.text(707, results[1]["sx_mm"]+2, "reference\n700mm",
            color="lime", fontsize=8)

ax_sig.set_xlabel("Standoff (mm)", color="#aaaaaa", fontsize=9)
ax_sig.set_ylabel("Sigma (mm)", color="#aaaaaa", fontsize=9)
ax_sig.set_title("sigma_x / sigma_y vs standoff → linear fit\n"
                 "Near-zero slope = collimated beam (size set by lamp, not distance)",
                 color="white", fontsize=10)
ax_sig.legend(fontsize=8, facecolor="#222222", labelcolor="white",
              edgecolor="#555555", loc="right")
ax_sig.set_xlim(550, 850)

# consensus beam diagram
ax_beam = fig.add_subplot(gs[2, 1])
ax_beam.set_facecolor("#0a0a0a")
ax_beam.set_xlim(-200, 200); ax_beam.set_ylim(-160, 160)
ax_beam.set_aspect("equal")
for sp in ax_beam.spines.values(): sp.set_edgecolor("#444444")
ax_beam.tick_params(colors="#aaaaaa", labelsize=8)
ax_beam.grid(True, alpha=0.08, color="white")

# specimen
ax_beam.add_patch(mpatches.Rectangle((-160,-87.5),320,175,
                  fc="#1a3a1a", ec="lime", lw=2, alpha=0.5, zorder=1))
ax_beam.text(0, 0, "Specimen\n320x175mm", ha="center", va="center",
             color="lime", fontsize=9, zorder=3)

# individual session ellipses (1sigma, faded)
for r, col in zip(results, COLS):
    ax_beam.add_patch(Ellipse((0,0), r["sx_mm"]*2, r["sy_mm"]*2,
                              fc="none", ec=col, lw=1.5, ls="--",
                              alpha=0.5, zorder=4))

# consensus ellipses
for scale_e, lw_e, ls_e, lbl in [(1,2.5,"-","1sigma"), (2,1.5,"--","2sigma"),
                                   (2.355/2,2,":","FWHM")]:
    ax_beam.add_patch(Ellipse((0,0), sx_mean*2*scale_e, sy_mean*2*scale_e,
                              fc="none", ec="white", lw=lw_e, ls=ls_e, zorder=5,
                              label=lbl))

# sigma dimension arrows
ax_beam.annotate("", xy=(sx_mean,0), xytext=(0,0),
                 arrowprops=dict(arrowstyle="<->", color="#ffd93d", lw=2))
ax_beam.text(sx_mean/2, 8, f"sigma_x={sx_mean:.0f}mm", ha="center",
             color="#ffd93d", fontsize=9, fontweight="bold")
ax_beam.annotate("", xy=(0,-sy_mean), xytext=(0,0),
                 arrowprops=dict(arrowstyle="<->", color="#4d96ff", lw=2))
ax_beam.text(18, -sy_mean/2, f"sigma_y={sy_mean:.0f}mm", ha="left",
             color="#4d96ff", fontsize=9, fontweight="bold")

ax_beam.set_title(
    f"Consensus beam shape (mean 600–800mm)\n"
    f"FWHM  {sx_mean*2.355:.0f} x {sy_mean*2.355:.0f} mm  |  aspect {sx_mean/sy_mean:.2f}:1",
    color="white", fontsize=10)
leg = ax_beam.legend(fontsize=8, facecolor="#222222", labelcolor="white",
                     edgecolor="#555555", loc="lower right")
for r, col in zip(results, COLS):
    ax_beam.plot([],[], color=col, lw=1.5, ls="--",
                 label=f"{r['standoff']}mm")

# encircled energy all 3
ax_ee = fig.add_subplot(gs[2, 2])
dark_ax(ax_ee)
for r, col in zip(results, COLS):
    r_ee_mm = r["r_rad"] * r["scale"]
    ax_ee.plot(r_ee_mm, r["r_ee"]*100, color=col, lw=2,
               label=f"{r['standoff']}mm")

for pct, lc in [(50,"white"), (86,"lime"), (95,"#ff6b6b")]:
    ax_ee.axhline(pct, color=lc, lw=0.8, ls="--", alpha=0.5)
    # mark on 700mm curve
    r700 = results[1]
    r_ee_mm_700 = r700["r_rad"] * r700["scale"]
    idx = np.searchsorted(r700["r_ee"]*100, pct)
    if idx < len(r_ee_mm_700):
        ax_ee.axvline(r_ee_mm_700[idx], color=lc, lw=0.8, ls="--", alpha=0.5)
        ax_ee.text(r_ee_mm_700[idx]+2, pct-5,
                   f"D{pct}={r_ee_mm_700[idx]:.0f}mm",
                   color=lc, fontsize=8)

ax_ee.set_xlabel("Radius from centre (mm)", color="#aaaaaa", fontsize=9)
ax_ee.set_ylabel("Encircled energy (%)", color="#aaaaaa", fontsize=9)
ax_ee.set_title("Encircled energy — D50 / D86 / D95\n(all 3 standoffs agree → reliable)",
                color="white", fontsize=10)
ax_ee.legend(fontsize=8.5, facecolor="#222222", labelcolor="white", edgecolor="#555555")
ax_ee.set_xlim(0, 200); ax_ee.set_ylim(0, 105)

# ════════════════════════════════════════════════════════
# ROW 3 — consensus Gaussian: 2D map and 3D surface
# ════════════════════════════════════════════════════════

# build the consensus Gaussian on a mm grid
x_g = np.linspace(-300, 300, 601)
y_g = np.linspace(-220, 220, 441)
XX, YY = np.meshgrid(x_g, y_g)
ZZ = np.exp(-0.5 * ((XX / sx_mean)**2 + (YY / sy_mean)**2))

# — Panel A: 2D irradiance map ——————————————————————————
ax_2d = fig.add_subplot(gs[3, 0])
ax_2d.set_facecolor("#0a0a0a")
im2d = ax_2d.imshow(ZZ, extent=[-300, 300, -220, 220],
                    origin="lower", cmap="inferno", aspect="equal",
                    vmin=0, vmax=1)
cb2 = fig.colorbar(im2d, ax=ax_2d, fraction=0.046, pad=0.04)
cb2.set_label("Normalised irradiance", color="white", fontsize=8)
cb2.ax.yaxis.set_tick_params(color="white")
plt.setp(cb2.ax.yaxis.get_ticklabels(), color="white")

# contour lines at key irradiance levels
levels_2d = [np.exp(-0.5), np.exp(-2), 0.5, 0.1]
lbls_2d   = ["1sigma (0.61)", "2sigma (0.14)", "FWHM (0.50)", "0.10"]
line_cols  = ["white", "#aaaaaa", "#ffd93d", "#4d96ff"]
for lv, lc in zip(levels_2d, line_cols):
    cs2 = ax_2d.contour(x_g, y_g, ZZ, levels=[lv], colors=[lc], linewidths=1.5)
    ax_2d.clabel(cs2, fmt=f"{lv:.2f}", colors=lc, fontsize=7)

# specimen rectangle
ax_2d.add_patch(mpatches.Rectangle((-160, -87.5), 320, 175,
                fc="none", ec="lime", lw=2.5, ls="--", zorder=5))
ax_2d.text(0, -105, "Specimen 320x175mm", ha="center",
           color="lime", fontsize=8.5, fontweight="bold")

# sigma arrows
ax_2d.annotate("", xy=(sx_mean, 0), xytext=(-sx_mean, 0),
               arrowprops=dict(arrowstyle="<->", color="#ffd93d", lw=2))
ax_2d.text(0, 14, f"2*sigma_x = {2*sx_mean:.0f}mm",
           ha="center", color="#ffd93d", fontsize=9, fontweight="bold")
ax_2d.annotate("", xy=(0, sy_mean), xytext=(0, -sy_mean),
               arrowprops=dict(arrowstyle="<->", color="#4d96ff", lw=2))
ax_2d.text(190, 0, f"2*sigma_y\n={2*sy_mean:.0f}mm",
           ha="left", va="center", color="#4d96ff", fontsize=9, fontweight="bold")

ax_2d.set_xlabel("X distance from centre (mm)", color="#aaaaaa", fontsize=9)
ax_2d.set_ylabel("Y distance from centre (mm)", color="#aaaaaa", fontsize=9)
ax_2d.set_title(
    f"Derived Gaussian — top-down 2D view\n"
    f"sigma_x={sx_mean:.0f}mm  sigma_y={sy_mean:.0f}mm  aspect {sx_mean/sy_mean:.2f}:1",
    color="white", fontsize=10)
for sp in ax_2d.spines.values(): sp.set_edgecolor("#444444")
ax_2d.tick_params(colors="#aaaaaa", labelsize=9)

# legend patches
patches_2d = [mpatches.Patch(color=lc, label=lb)
              for lc, lb in zip(line_cols, lbls_2d)]
patches_2d.append(mpatches.Patch(color="lime", label="Specimen footprint"))
ax_2d.legend(handles=patches_2d, fontsize=7.5, facecolor="#222222",
             labelcolor="white", edgecolor="#555555", loc="upper right")

# — Panel B: 3D surface (spanning 2 columns) ——————————
ax_3d = fig.add_subplot(gs[3, 1:3], projection="3d")
ax_3d.set_facecolor("#0a0a0a")
ax_3d.patch.set_facecolor("#0a0a0a")

# downsample for speed
step = 4
Xs = XX[::step, ::step]; Ys = YY[::step, ::step]; Zs = ZZ[::step, ::step]
surf = ax_3d.plot_surface(Xs, Ys, Zs, cmap="inferno",
                          linewidth=0, antialiased=True, alpha=0.88,
                          vmin=0, vmax=1)
fig.colorbar(surf, ax=ax_3d, shrink=0.55, pad=0.08,
             label="Normalised irradiance").ax.yaxis.set_tick_params(color="white")

# specimen footprint at z=0
spec_x = np.array([-160, 160, 160, -160, -160])
spec_y = np.array([-87.5, -87.5, 87.5, 87.5, -87.5])
ax_3d.plot(spec_x, spec_y, zs=0, zdir="z", color="lime", lw=2.5,
           ls="--", label="Specimen 320x175mm")

# 1sigma contour ring projected onto z=0
theta_ring = np.linspace(0, 2*np.pi, 360)
ring_x = sx_mean * np.cos(theta_ring)
ring_y = sy_mean * np.sin(theta_ring)
ax_3d.plot(ring_x, ring_y, zs=0, zdir="z",
           color="white", lw=1.5, ls=":", label="1sigma ellipse")

# cross-sections at z surface
x_cs  = np.linspace(-300, 300, 300)
ax_3d.plot(x_cs, np.zeros_like(x_cs),
           np.exp(-0.5*(x_cs/sx_mean)**2),
           color="#ffd93d", lw=2, label=f"H profile  sigma_x={sx_mean:.0f}mm")
y_cs  = np.linspace(-220, 220, 220)
ax_3d.plot(np.zeros_like(y_cs), y_cs,
           np.exp(-0.5*(y_cs/sy_mean)**2),
           color="#4d96ff", lw=2, label=f"V profile  sigma_y={sy_mean:.0f}mm")

ax_3d.set_xlabel("X (mm)", color="#aaaaaa", fontsize=9, labelpad=8)
ax_3d.set_ylabel("Y (mm)", color="#aaaaaa", fontsize=9, labelpad=8)
ax_3d.set_zlabel("Normalised irradiance", color="#aaaaaa", fontsize=9, labelpad=8)
ax_3d.tick_params(colors="#aaaaaa", labelsize=8)
ax_3d.xaxis.pane.fill = False
ax_3d.yaxis.pane.fill = False
ax_3d.zaxis.pane.fill = False
ax_3d.xaxis.pane.set_edgecolor("#333333")
ax_3d.yaxis.pane.set_edgecolor("#333333")
ax_3d.zaxis.pane.set_edgecolor("#333333")
ax_3d.view_init(elev=28, azim=-55)
ax_3d.set_title(
    f"Derived Gaussian — 3D surface\n"
    f"FWHM {sx_mean*2.355:.0f}x{sy_mean*2.355:.0f}mm  |  Peak normalised to 1.0",
    color="white", fontsize=10, pad=10)
ax_3d.legend(fontsize=8, facecolor="#222222", labelcolor="white",
             edgecolor="#555555", loc="upper left")

# ════════════════════════════════════════════════════════
# ROW 4 — 2D Gaussian projection + annotated H/V profiles
# ════════════════════════════════════════════════════════

# shared grids
x_p   = np.linspace(-320, 320, 641)
y_p   = np.linspace(-230, 230, 461)
XX_p, YY_p = np.meshgrid(x_p, y_p)
ZZ_p  = np.exp(-0.5 * ((XX_p / sx_mean)**2 + (YY_p / sy_mean)**2))

H_prof = np.exp(-0.5 * (x_p / sx_mean)**2)   # horizontal 1D profile
V_prof = np.exp(-0.5 * (y_p / sy_mean)**2)   # vertical 1D profile

fwhm_x  = sx_mean * 2.3548
fwhm_y  = sy_mean * 2.3548
sig2_x  = sx_mean * 2
sig2_y  = sy_mean * 2

# key irradiance levels
LVL_1S = np.exp(-0.5)   # 0.6065  (1 sigma)
LVL_2S = np.exp(-2.0)   # 0.1353  (2 sigma)
LVL_HW = 0.5            # FWHM level

# — Panel A: 2D projection with crosshairs and definitions ——
ax_proj = fig.add_subplot(gs[4, 0])
ax_proj.set_facecolor("#0a0a0a")

im_p = ax_proj.imshow(ZZ_p, extent=[-320, 320, -230, 230],
                       origin="lower", cmap="inferno", aspect="equal",
                       vmin=0, vmax=1)
cb_p = fig.colorbar(im_p, ax=ax_proj, fraction=0.046, pad=0.04)
cb_p.set_label("Normalised irradiance", color="white", fontsize=8)
cb_p.ax.yaxis.set_tick_params(color="white")
plt.setp(cb_p.ax.yaxis.get_ticklabels(), color="white")

# filled contours at 1sigma and 2sigma
ax_proj.contourf(x_p, y_p, ZZ_p, levels=[LVL_2S, LVL_1S],
                 colors=["#ffffff"], alpha=0.07)
ax_proj.contourf(x_p, y_p, ZZ_p, levels=[LVL_1S, 1.0],
                 colors=["#ffffff"], alpha=0.10)

# contour outlines
for lv, lc, lw_c, lbl in [
        (LVL_HW, "#ffd93d", 2.0, f"FWHM  {fwhm_x:.0f}x{fwhm_y:.0f}mm"),
        (LVL_1S, "white",   2.0, f"1sigma  {sx_mean*2:.0f}x{sy_mean*2:.0f}mm"),
        (LVL_2S, "#aaaaaa", 1.5, f"2sigma  {sig2_x*2:.0f}x{sig2_y*2:.0f}mm")]:
    cs = ax_proj.contour(x_p, y_p, ZZ_p, levels=[lv],
                         colors=[lc], linewidths=lw_c)
    ax_proj.clabel(cs, fmt=f"{lbl}", colors=lc, fontsize=7, inline=True)

# specimen rectangle
ax_proj.add_patch(mpatches.Rectangle((-160, -87.5), 320, 175,
                  fc="none", ec="lime", lw=2.5, ls="--", zorder=5))
ax_proj.text(0, -105, "Specimen  320x175mm", ha="center",
             color="lime", fontsize=8, fontweight="bold")

# crosshair lines showing profile positions
ax_proj.axhline(0, color="#ffd93d", lw=1.5, ls="--", alpha=0.9,
                label="H profile (y=0)")
ax_proj.axvline(0, color="#4d96ff", lw=1.5, ls="--", alpha=0.9,
                label="V profile (x=0)")

# sigma tick marks on axes
for sign in [+1, -1]:
    ax_proj.plot([sign*sx_mean, sign*sx_mean], [-8, 8],
                 color="#ffd93d", lw=2.5, zorder=6)
    ax_proj.plot([-8, 8], [sign*sy_mean, sign*sy_mean],
                 color="#4d96ff", lw=2.5, zorder=6)

ax_proj.set_xlabel("X (mm)", color="#aaaaaa", fontsize=9)
ax_proj.set_ylabel("Y (mm)", color="#aaaaaa", fontsize=9)
ax_proj.set_title("2D beam projection — top-down view\nwith FWHM / 1sigma / 2sigma contours",
                  color="white", fontsize=10)
for sp in ax_proj.spines.values(): sp.set_edgecolor("#444444")
ax_proj.tick_params(colors="#aaaaaa", labelsize=9)
ax_proj.legend(fontsize=8, facecolor="#222222", labelcolor="white",
               edgecolor="#555555", loc="upper right")


# helper to draw beam definition annotations on a 1D profile
def annotate_profile(ax, x_arr, y_arr, sigma, is_horiz=True):
    col_s  = "#ffd93d" if is_horiz else "#4d96ff"
    lbl    = "sigma_x" if is_horiz else "sigma_y"
    fwhm   = sigma * 2.3548
    s2     = sigma * 2

    ax.plot(x_arr, y_arr, color=col_s, lw=2.5, zorder=5)

    # horizontal level lines
    for lv, lc, ls, txt in [
            (LVL_1S, "white",   ":", f"e^-0.5 = {LVL_1S:.3f}  (1-sigma level)"),
            (LVL_HW, "#ffd93d", "--", "0.500  (FWHM level)"),
            (LVL_2S, "#888888", ":", f"e^-2   = {LVL_2S:.3f}  (2-sigma level)")]:
        ax.axhline(lv, color=lc, lw=1.0, ls=ls, alpha=0.7)
        ax.text(x_arr[-1]*0.98, lv+0.02, txt, color=lc,
                fontsize=7, ha="right", va="bottom")

    # vertical sigma tick lines
    for s_val, s_col, s_lbl, y_lv in [
            (sigma,  "white",   f"{lbl}={sigma:.0f}mm",  LVL_1S),
            (fwhm/2, "#ffd93d", f"FWHM/2={fwhm/2:.0f}mm", LVL_HW),
            (s2,     "#888888", f"2{lbl}={s2:.0f}mm",    LVL_2S)]:
        for sign in [+1, -1]:
            ax.axvline(sign*s_val, color=s_col, lw=1.2, ls="--", alpha=0.6)
        # double-headed span arrow
        ax.annotate("", xy=(s_val, y_lv), xytext=(-s_val, y_lv),
                    arrowprops=dict(arrowstyle="<->", color=s_col,
                                   lw=1.5, mutation_scale=10))
        ax.text(0, y_lv + 0.03, f"{s_val*2:.0f}mm", ha="center",
                color=s_col, fontsize=8, fontweight="bold")

    ax.set_ylim(-0.05, 1.18)
    ax.set_xlim(x_arr[0], x_arr[-1])
    ax.set_xlabel("Distance from centre (mm)", color="#aaaaaa", fontsize=9)
    ax.set_ylabel("Normalised irradiance", color="#aaaaaa", fontsize=9)


# — Panel B: H profile ————————————————————————————————————
ax_hp = fig.add_subplot(gs[4, 1])
dark_ax(ax_hp)
annotate_profile(ax_hp, x_p, H_prof, sx_mean, is_horiz=True)
ax_hp.set_title(
    f"Horizontal (X) profile  —  sigma_x = {sx_mean:.1f}mm\n"
    f"FWHM = {fwhm_x:.1f}mm  |  2-sigma width = {sig2_x*2:.1f}mm",
    color="white", fontsize=10)

# — Panel C: V profile ————————————————————————————————————
ax_vp = fig.add_subplot(gs[4, 2])
dark_ax(ax_vp)
annotate_profile(ax_vp, y_p, V_prof, sy_mean, is_horiz=False)
ax_vp.set_title(
    f"Vertical (Y) profile  —  sigma_y = {sy_mean:.1f}mm\n"
    f"FWHM = {fwhm_y:.1f}mm  |  2-sigma width = {sig2_y*2:.1f}mm",
    color="white", fontsize=10)

fig.suptitle(
    "Beam shape derivation  |  18mm lens, Boson 640  |  600 / 700 / 800mm standoff\n"
    "Three independent measurements agree → consensus: "
    f"sigma_x={sx_mean:.0f}mm  sigma_y={sy_mean:.0f}mm  |  "
    f"FWHM {sx_mean*2.355:.0f}x{sy_mean*2.355:.0f}mm  |  aspect {sx_mean/sy_mean:.2f}:1",
    color="white", fontsize=12, y=0.978)

fig.savefig(OUTPUT_PNG, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"\nSaved: {OUTPUT_PNG}")
