"""
Resolution-constrained fixture optimisation.
Fix spatial resolution (mm/px) → d computed per lens.
Sweep (L, θ) to find optimal fixture for each lens.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

OUTPUT_PNG = config.BOSON_ROOT + r"\resolution_constrained_optimisation.png"

# ── constants ──────────────────────────────────────────────────────────────────
PIXEL_PITCH = 0.012        # mm  (Boson 640, 12 µm)
SENSOR_W_PX = 640
SENSOR_H_PX = 512
SPEC_W      = 320.0        # mm
SPEC_H      = 175.0        # mm
SX          =  59.0        # mm  σ_x free-space (vertical filament — narrow in X)
SY          = 108.0        # mm  σ_y free-space (vertical filament — wide in Y)
RES_TARGET  = 0.5          # mm/px  — fixed resolution constraint

LENSES = [
    {"f":  9.2, "label": "9.2 mm",  "col": "#ffd93d"},
    {"f": 13.8, "label": "13.8 mm", "col": "#4d96ff"},
    {"f": 18.0, "label": "18 mm",   "col": "#ff6b6b"},
]

# d = (resolution × focal_length) / pixel_pitch
for lens in LENSES:
    lens["d"] = RES_TARGET * lens["f"] / PIXEL_PITCH

# ── CoV function ───────────────────────────────────────────────────────────────
NX, NY = 160, 88

def compute_cov(L_lamp, theta_deg, d_lamp):
    tr      = np.radians(theta_deg)
    x_s     = L_lamp - d_lamp * np.tan(tr)
    sx_surf = SX / np.cos(tr)
    xs = np.linspace(-SPEC_W / 2, SPEC_W / 2, NX)
    ys = np.linspace(-SPEC_H / 2, SPEC_H / 2, NY)
    X, Y = np.meshgrid(xs, ys)
    I1 = np.exp(-0.5 * ((X - x_s) / sx_surf) ** 2 - 0.5 * (Y / SY) ** 2)
    I2 = np.exp(-0.5 * ((X + x_s) / sx_surf) ** 2 - 0.5 * (Y / SY) ** 2)
    I  = I1 + I2
    mu = I.mean()
    return I.std() / mu * 100 if mu > 1e-10 else np.nan

# sweep grids
L_range     = np.linspace(50,  600, 100)
theta_range = np.linspace(15,  70,  100)
LL, TT = np.meshgrid(L_range, theta_range)

# ── pre-compute CoV grids ──────────────────────────────────────────────────────
print(f"Resolution target: {RES_TARGET} mm/px\n")
print(f"{'Lens':>8}  {'d (mm)':>8}  {'opt L':>8}  {'opt th':>8}  {'CoV':>8}")
print("-" * 48)

opt_results = []
cov_grids   = []

for lens in LENSES:
    d = lens["d"]
    COV = np.vectorize(compute_cov)(LL, TT, d)
    cov_grids.append(COV)

    idx   = np.nanargmin(COV)
    oi, oj = np.unravel_index(idx, COV.shape)
    r = dict(lens=lens["label"], col=lens["col"], d=d,
             opt_L=L_range[oj], opt_theta=theta_range[oi], opt_cov=COV[oi, oj])
    opt_results.append(r)
    print(f"{lens['label']:>8}  {d:>8.0f}  {r['opt_L']:>8.0f}  "
          f"{r['opt_theta']:>7.0f}deg  {r['opt_cov']:>7.1f}%")

# ── resolution sensitivity (fixed lens = 9.2 mm) ──────────────────────────────
res_range  = np.linspace(0.3, 1.2, 40)
best_covs  = []
for res in res_range:
    d_r = res * LENSES[0]["f"] / PIXEL_PITCH
    COV_r = np.vectorize(compute_cov)(LL, TT, d_r)
    best_covs.append(np.nanmin(COV_r))

# ── figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 13), facecolor="#0d0d0d")
gs  = GridSpec(2, 4, figure=fig, hspace=0.42, wspace=0.32,
               left=0.06, right=0.97, top=0.88, bottom=0.07,
               width_ratios=[1, 1, 1, 0.95])

fig.suptitle(
    f"Resolution-constrained fixture optimisation  —  target {RES_TARGET} mm/px\n"
    f"Beam  σ_x = {SX} mm (horiz, narrow)   σ_y = {SY} mm (vert, wide)  |  "
    f"Specimen {SPEC_W:.0f} × {SPEC_H:.0f} mm  |  Vertical filament, side lamps  |  "
    f"d = (res × f) / pixel_pitch  per lens",
    color="white", fontsize=12, fontweight="bold")

def dark_ax(ax):
    ax.set_facecolor("#111111")
    for sp in ax.spines.values(): sp.set_edgecolor("#333333")
    ax.tick_params(colors="#888888", labelsize=9)
    ax.grid(color="#222222", lw=0.5, zorder=0)

cov_levels  = np.linspace(0, 50, 51)
label_levs  = [5, 10, 15, 20, 30]

# ── row 0: CoV contour maps ────────────────────────────────────────────────────
for ci, (lens, COV, r) in enumerate(zip(LENSES, cov_grids, opt_results)):
    ax = fig.add_subplot(gs[0, ci])
    ax.set_facecolor("#111111")
    for sp in ax.spines.values(): sp.set_edgecolor("#333333")
    ax.tick_params(colors="#888888", labelsize=9)

    cf = ax.contourf(LL, TT, np.clip(COV, 0, 50), levels=cov_levels,
                     cmap="RdYlGn_r", vmin=0, vmax=50)
    cs = ax.contour(LL, TT, COV, levels=label_levs,
                    colors="white", linewidths=0.9, alpha=0.55)
    ax.clabel(cs, fmt="%d%%", colors="white", fontsize=8)

    # optimum marker
    ax.plot(r["opt_L"], r["opt_theta"], "*", color="white", ms=15, zorder=10,
            label=f"Opt: L={r['opt_L']:.0f}mm  θ={r['opt_theta']:.0f}°  CoV={r['opt_cov']:.1f}%")
    ax.legend(fontsize=8, facecolor="#1a1a1a", labelcolor="white",
              edgecolor=lens["col"], loc="upper right")

    cb = plt.colorbar(cf, ax=ax, shrink=0.88, pad=0.02)
    cb.set_label("CoV (%)", color="#888888", fontsize=8)
    cb.ax.tick_params(colors="#888888", labelsize=8)

    ax.set_xlabel("L — lamp lateral offset (mm)", color="#888888", fontsize=9)
    ax.set_ylabel("θ — beam angle (°)", color="#888888", fontsize=9)
    ax.set_title(
        f"{lens['label']}  |  d = {r['d']:.0f} mm",
        color=lens["col"], fontsize=11, fontweight="bold")

# ── row 0 col 3: resolution sensitivity ───────────────────────────────────────
ax_rs = fig.add_subplot(gs[0, 3])
dark_ax(ax_rs)
ax_rs.plot(res_range, best_covs, color=LENSES[0]["col"], lw=2.5)
ax_rs.axvline(RES_TARGET, color="white", lw=1.5, ls="--",
              label=f"Current target\n{RES_TARGET} mm/px")
ax_rs.axhline(10, color="#ff9f43", lw=1.2, ls=":", label="10% CoV")
ax_rs.axhline(5,  color="#6bcb77", lw=1.2, ls=":", label="5% CoV")
ax_rs.fill_between(res_range, best_covs, 5, where=[c > 5 for c in best_covs],
                   color="#ff4444", alpha=0.12, label=">5% CoV region")
ax_rs.set_xlabel("Resolution target (mm/px)", color="#888888", fontsize=9)
ax_rs.set_ylabel("Best achievable CoV (%)", color="#888888", fontsize=9)
ax_rs.set_title("CoV vs resolution target\n(9.2 mm lens, vertical filament, optimal L & th)",
                color=LENSES[0]["col"], fontsize=10, fontweight="bold")
ax_rs.legend(fontsize=8, facecolor="#1a1a1a", labelcolor="white", edgecolor="#333333")
ax_rs.set_xlim(res_range[0], res_range[-1])

# ── row 1: summary bars ────────────────────────────────────────────────────────
labels = [r["lens"] for r in opt_results]
cols   = [r["col"]  for r in opt_results]
x      = np.arange(len(opt_results))
w      = 0.45

# col 0: standoff d
ax_d = fig.add_subplot(gs[1, 0])
dark_ax(ax_d)
ds   = [r["d"] for r in opt_results]
bars = ax_d.bar(labels, ds, color=cols, alpha=0.85, width=w)
for bar, val in zip(bars, ds):
    ax_d.text(bar.get_x() + bar.get_width() / 2, val + 4,
              f"{val:.0f} mm", ha="center", color="white", fontsize=9, fontweight="bold")
ax_d.set_ylabel("Camera standoff d (mm)", color="#888888", fontsize=9)
ax_d.set_title(f"Required standoff @ {RES_TARGET} mm/px", color="white", fontsize=10)
ax_d.set_ylim(0, max(ds) * 1.25)

# col 1: optimal L
ax_L = fig.add_subplot(gs[1, 1])
dark_ax(ax_L)
opt_Ls = [r["opt_L"] for r in opt_results]
bars   = ax_L.bar(labels, opt_Ls, color=cols, alpha=0.85, width=w)
for bar, val in zip(bars, opt_Ls):
    ax_L.text(bar.get_x() + bar.get_width() / 2, val + 4,
              f"{val:.0f} mm", ha="center", color="white", fontsize=9, fontweight="bold")
ax_L.set_ylabel("Optimal lamp offset L (mm)", color="#888888", fontsize=9)
ax_L.set_title("Optimal L per lens", color="white", fontsize=10)
ax_L.set_ylim(0, max(opt_Ls) * 1.25)

# col 2: optimal theta
ax_th = fig.add_subplot(gs[1, 2])
dark_ax(ax_th)
opt_ths = [r["opt_theta"] for r in opt_results]
bars    = ax_th.bar(labels, opt_ths, color=cols, alpha=0.85, width=w)
for bar, val in zip(bars, opt_ths):
    ax_th.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
               f"{val:.0f}°", ha="center", color="white", fontsize=9, fontweight="bold")
ax_th.set_ylabel("Optimal beam angle θ (°)", color="#888888", fontsize=9)
ax_th.set_title("Optimal θ per lens", color="white", fontsize=10)
ax_th.set_ylim(0, max(opt_ths) * 1.25)

# col 3: best CoV
ax_cv = fig.add_subplot(gs[1, 3])
dark_ax(ax_cv)
opt_cvs = [r["opt_cov"] for r in opt_results]
bars    = ax_cv.bar(labels, opt_cvs, color=cols, alpha=0.85, width=w)
for bar, val in zip(bars, opt_cvs):
    ax_cv.text(bar.get_x() + bar.get_width() / 2, val + 0.2,
               f"{val:.1f}%", ha="center", color="white", fontsize=9, fontweight="bold")
ax_cv.axhline(10, color="#ff9f43", lw=1.5, ls="--", alpha=0.8, label="10% threshold")
ax_cv.axhline(5,  color="#6bcb77", lw=1.5, ls="--", alpha=0.8, label="5% threshold")
ax_cv.set_ylabel("Best achievable CoV (%)", color="#888888", fontsize=9)
ax_cv.set_title("Best CoV per lens", color="white", fontsize=10)
ax_cv.legend(fontsize=8, facecolor="#1a1a1a", labelcolor="white", edgecolor="#333333")
ax_cv.set_ylim(0, max(opt_cvs) * 1.35)

plt.savefig(OUTPUT_PNG, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"\nSaved: {OUTPUT_PNG}")
