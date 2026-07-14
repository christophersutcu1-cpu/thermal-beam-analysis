"""
Standoff intensity trade-off — companion picture to 06_dual_source_specimen_map.py.

For each candidate standoff, re-optimise (angle, aim offset) with the same
width-weighted objective + high-angle bias, then render the resulting
two-lamp irradiance map on a SHARED ABSOLUTE colour scale (K-equivalent on
cardboard, 5 s pulse, both lamps at the characterisation operating point),
so the intensity cost of standing further back is visible directly.

Outputs: 06b_standoff_intensity_tradeoff.{json,png}
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BEAM_JSON = os.path.join(HERE, "beam_shape_summary_seq.json")
OUT_JSON = os.path.join(HERE, "06b_standoff_intensity_tradeoff.json")
OUT_PNG = os.path.join(HERE, "06b_standoff_intensity_tradeoff.png")

SPEC_W, SPEC_H = 320.0, 175.0
GRID_STEP = 2.5
WIDTH_WEIGHT = 2.0
ANGLE_BIAS_TOL = 0.02

STANDOFFS = [300.0, 400.0, 500.0, 600.0, 750.0]
TH_SWEEP = np.arange(0.0, 60.1, 2.0)
A_SWEEP = np.arange(-160.0, 160.1, 5.0)

with open(BEAM_JSON) as f:
    beam = json.load(f)
AX = beam["linear_fit"]["sigma_x"]["slope_mm_per_mm"]
BX = beam["linear_fit"]["sigma_x"]["intercept_mm"]
AY = beam["linear_fit"]["sigma_y"]["slope_mm_per_mm"]
BY = beam["linear_fit"]["sigma_y"]["intercept_mm"]
anchors = {s["d_mm"]: s["peak_dT_K"] * 2 * np.pi * s["sigma_x_mm"] * s["sigma_y_mm"]
           for s in beam["per_standoff"]}
P_LAMP = float(np.mean([anchors[300], anchors[500]]))

xs = np.arange(-SPEC_W / 2 + GRID_STEP / 2, SPEC_W / 2, GRID_STEP)
ys = np.arange(-SPEC_H / 2 + GRID_STEP / 2, SPEC_H / 2, GRID_STEP)
X, Y = np.meshgrid(xs, ys)


def lamp_field(side, d, th_deg, a):
    th = np.radians(th_deg)
    u = np.array([-side * np.sin(th), 0.0, -np.cos(th)])
    pos = np.array([side * a, 0.0, 0.0]) - d * u
    ex = np.array([np.cos(th), 0.0, -side * np.sin(th)])
    vx, vy, vz = X - pos[0], Y - pos[1], -pos[2]
    z = vx * u[0] + vy * u[1] + vz * u[2]
    rx = vx * ex[0] + vy * ex[1] + vz * ex[2]
    ry = vy
    sx = AX * z + BX
    sy = AY * z + BY
    cos_inc = pos[2] / np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    return (P_LAMP / (2 * np.pi * sx * sy)
            * np.exp(-0.5 * ((rx / sx) ** 2 + (ry / sy) ** 2)) * cos_inc)


def evaluate(d, th, a):
    I = lamp_field(-1, d, th, a) + lamp_field(+1, d, th, a)
    cov_face = I.std() / I.mean()
    px = I.mean(axis=0)
    cov_x = px.std() / px.mean()
    return I, dict(d_mm=d, angle_deg=th, offset_mm=a, cov_face=cov_face,
                   cov_width=cov_x, score=cov_face + WIDTH_WEIGHT * cov_x,
                   mean_K=I.mean(), peak_K=I.max(), min_K=I.min())


results = []
for d in STANDOFFS:
    rows = [evaluate(d, th, a)[1] for th in TH_SWEEP for a in A_SWEEP]
    best = min(r["score"] for r in rows)
    band = [r for r in rows if r["score"] <= best * (1 + ANGLE_BIAS_TOL)]
    opt = max(band, key=lambda r: (r["angle_deg"], -r["score"]))
    I, _ = evaluate(opt["d_mm"], opt["angle_deg"], opt["offset_mm"])
    results.append((opt, I))
    print(f"d={d:.0f}: th={opt['angle_deg']:.0f} deg, a={opt['offset_mm']:+.0f} mm | "
          f"mean {opt['mean_K']:.1f} K, CoV face {opt['cov_face']*100:.1f}%, "
          f"width {opt['cov_width']*100:.2f}%")

vmax = max(I.max() for _, I in results)

fig = plt.figure(figsize=(19, 8.6), constrained_layout=True)
gs = fig.add_gridspec(2, len(STANDOFFS) + 1, width_ratios=[1] * len(STANDOFFS) + [1.15],
                      height_ratios=[1.15, 1])
fig.suptitle("PL-H-V-HS1 dual-source: what each standoff actually buys you — "
             "same absolute colour scale (K-equiv, cardboard 5 s pulse, both lamps)",
             fontsize=13, fontweight="bold")

for i, (opt, I) in enumerate(results):
    ax = fig.add_subplot(gs[0, i])
    im = ax.imshow(I, origin="lower", cmap="inferno", vmin=0, vmax=vmax,
                   extent=[-SPEC_W/2, SPEC_W/2, -SPEC_H/2, SPEC_H/2], aspect="equal")
    ax.contour(X, Y, I / I.mean(), levels=[0.9, 1.1], colors="cyan", linewidths=0.7)
    ax.set_title(f"d = {opt['d_mm']:.0f} mm  |  {opt['angle_deg']:.0f}°, "
                 f"aim ±{abs(opt['offset_mm']):.0f} mm\n"
                 f"mean {opt['mean_K']:.1f} K   peak {opt['peak_K']:.1f} K\n"
                 f"CoV face {opt['cov_face']*100:.1f} %  ·  "
                 f"width CoV {opt['cov_width']*100:.2f} %",
                 fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    if i == 0:
        ax.set_ylabel("175 mm")

cax = fig.add_subplot(gs[0, -1])
cax.axis("off")
fig.colorbar(im, ax=cax, fraction=0.6, label="K-equivalent (5 s, cardboard)")

# absolute width profiles
ax = fig.add_subplot(gs[1, 0:3])
colors = plt.cm.plasma(np.linspace(0.05, 0.8, len(results)))
for (opt, I), c in zip(results, colors):
    ax.plot(xs, I.mean(axis=0), lw=2, color=c,
            label=f"d={opt['d_mm']:.0f} mm  (mean {opt['mean_K']:.1f} K)")
ax.set_title("Width profile in ABSOLUTE units — flatness vs intensity", fontsize=10)
ax.set_xlabel("x (mm)"); ax.set_ylabel("K-equivalent")
ax.grid(alpha=0.3); ax.legend(fontsize=8)

# vertical profiles, normalised (droop comparison)
ax = fig.add_subplot(gs[1, 3])
for (opt, I), c in zip(results, colors):
    pv = I.mean(axis=1)
    ax.plot(ys, pv / pv.max() * 100, lw=1.8, color=c)
ax.axhline(90, color="0.6", lw=0.8, ls=":")
ax.set_title("Vertical droop\n(% of own centre)", fontsize=10)
ax.set_xlabel("y (mm)"); ax.set_ylabel("%")
ax.set_ylim(50, 103); ax.grid(alpha=0.3)

# intensity + CoV vs standoff
ax = fig.add_subplot(gs[1, 4:])
dd = [o["d_mm"] for o, _ in results]
ax.plot(dd, [o["mean_K"] for o, _ in results], "^-", color="#2ca02c", lw=2,
        label="mean intensity (K-equiv)")
ax.plot(dd, [o["min_K"] for o, _ in results], "v--", color="#2ca02c", lw=1.2,
        label="worst corner (K-equiv)")
ax.set_xlabel("standoff (mm)"); ax.set_ylabel("K-equivalent", color="#2ca02c")
ax2 = ax.twinx()
ax2.plot(dd, [o["cov_face"] * 100 for o, _ in results], "o-", color="#d62728",
         label="CoV face %")
ax2.plot(dd, [o["cov_width"] * 100 for o, _ in results], "s-", color="#1f77b4",
         label="CoV width %")
ax2.set_ylabel("CoV (%)")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
ax.set_title("Intensity vs uniformity trade-off", fontsize=10)
ax.grid(alpha=0.3)

fig.savefig(OUT_PNG, dpi=130)
print(f"Wrote {OUT_PNG}")

with open(OUT_JSON, "w") as f:
    json.dump({"per_standoff_optima": [
        {k: float(v) for k, v in o.items()} for o, _ in results]}, f, indent=2)
print(f"Wrote {OUT_JSON}")
