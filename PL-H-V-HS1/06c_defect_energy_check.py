"""
Defect-resolving check — which dual-source config best resolves ALL defects
in the stepped 320 x 175 mm specimen (Figure 4 schematic, Energy Density paper).

Defect map below is in FLAT-FACE (illuminated-side) coordinates: the paper
schematic is drawn looking at the stepped back face, so x is mirrored here —
the 3.00 mm-thick section with the deepest defects (2.4 mm) lands bottom-RIGHT,
as confirmed by the user.

Logic: required excitation energy rises steeply with defect depth (taken as
demand ~ depth^2, transparent diffusion-length argument), so the binding
constraint is the delivered intensity at the 2.4 mm defects. The width profile
is already flat at the 06b optima; the free lever is a common DOWNWARD shift
of the lamp aim line (dy), which moves the vertical Gaussian onto the deep row
at the expense of the (easy, 0.3 mm) top row.

Outputs: 06c_defect_energy_check.{json,png}
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BEAM_JSON = os.path.join(HERE, "beam_shape_summary_seq.json")
OUT_JSON = os.path.join(HERE, "06c_defect_energy_check.json")
OUT_PNG = os.path.join(HERE, "06c_defect_energy_check.png")

SPEC_W, SPEC_H = 320.0, 175.0

# (x_mm, y_mm, depth_mm) — flat-face view, origin at specimen centre
DEFECTS = [
    # top row, 0.3 mm deep
    (+110, 20, 0.3), (+77, 20, 0.3), (+45, 20, 0.3), (+1, 20, 0.3),
    (-40, 20, 0.3), (-70, 20, 0.3), (-120, 20, 0.3),
    # middle row
    (+110, -22, 1.2), (+77, -22, 1.2), (+45, -22, 0.9), (+1, -22, 0.9),
    (-40, -22, 0.9), (-70, -22, 0.9), (-122, -22, 0.6),
    # bottom row — deepest bottom-right
    (+110, -64, 2.4), (+77, -64, 2.4), (+45, -64, 1.8), (+1, -64, 1.8),
    (-122, -64, 0.9),
]

# candidate configs: 06b per-standoff optima (all 60 deg)
CONFIGS = [(300.0, 60.0, 75.0), (350.0, 60.0, 72.0),
           (400.0, 60.0, 70.0), (500.0, 60.0, 65.0),
           (600.0, 60.0, 65.0), (750.0, 60.0, 65.0)]
DY_SWEEP = np.arange(0.0, -71.0, -5.0)     # common downward aim-line shift

with open(BEAM_JSON) as f:
    beam = json.load(f)
AX = beam["linear_fit"]["sigma_x"]["slope_mm_per_mm"]
BX = beam["linear_fit"]["sigma_x"]["intercept_mm"]
AY = beam["linear_fit"]["sigma_y"]["slope_mm_per_mm"]
BY = beam["linear_fit"]["sigma_y"]["intercept_mm"]
anch = {s["d_mm"]: s["peak_dT_K"] * 2 * np.pi * s["sigma_x_mm"] * s["sigma_y_mm"]
        for s in beam["per_standoff"]}
P_LAMP = float(np.mean([anch[300], anch[500]]))


def field(Xq, Yq, side, d, th_deg, a, dy):
    th = np.radians(th_deg)
    u = np.array([-side * np.sin(th), 0.0, -np.cos(th)])
    pos = np.array([side * a, dy, 0.0]) - d * u
    ex = np.array([np.cos(th), 0.0, -side * np.sin(th)])
    vx, vy, vz = Xq - pos[0], Yq - pos[1], -pos[2]
    z = vx * u[0] + vz * u[2]
    rx = vx * ex[0] + vz * ex[2]
    sx, sy = AX * z + BX, AY * z + BY
    ci = pos[2] / np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    return (P_LAMP / (2 * np.pi * sx * sy)
            * np.exp(-0.5 * ((rx / sx) ** 2 + (vy / sy) ** 2)) * ci)


def total(Xq, Yq, d, th, a, dy):
    return field(Xq, Yq, -1, d, th, a, dy) + field(Xq, Yq, +1, d, th, a, dy)


dx_, dy_, dz_ = (np.array(v) for v in zip(*DEFECTS))
demand = (dz_ / dz_.min()) ** 2            # relative energy demand ~ depth^2

# ---- sweep configs x aim-line shift; pick by worst depth-margin -----------
rows = []
for (d, th, a) in CONFIGS:
    for dy in DY_SWEEP:
        I_def = total(dx_, dy_, d, th, a, dy)
        margin = I_def / demand
        rows.append(dict(d_mm=d, angle_deg=th, offset_mm=a, aimline_mm=dy,
                         I_deepest=float(I_def[dz_ == 2.4].min()),
                         I_min_all=float(I_def.min()),
                         worst_margin=float(margin.min()),
                         top_row_K=float(I_def[dz_ == 0.3].min())))
best = max(rows, key=lambda r: r["worst_margin"])

print(" d(mm) | aim dy | I@2.4mm | top-row I | worst margin (rel)")
for (d, th, a) in CONFIGS:
    for dy in (0.0, -20.0, -40.0, -60.0):
        r = next(r for r in rows if r["d_mm"] == d and r["aimline_mm"] == dy)
        tag = "  <-- BEST" if r is best else ""
        print(f"  {d:4.0f} | {dy:+5.0f} |  {r['I_deepest']:5.2f}  |   "
              f"{r['top_row_K']:5.2f}   |  {r['worst_margin']*1000:.2f}{tag}")

# ---- figure ---------------------------------------------------------------
gx = np.arange(-SPEC_W/2 + 1.25, SPEC_W/2, 2.5)
gy = np.arange(-SPEC_H/2 + 1.25, SPEC_H/2, 2.5)
GX, GY = np.meshgrid(gx, gy)
I_best = total(GX, GY, best["d_mm"], best["angle_deg"], best["offset_mm"],
               best["aimline_mm"])
I_flat = total(GX, GY, best["d_mm"], best["angle_deg"], best["offset_mm"], 0.0)

fig = plt.figure(figsize=(17, 8.5), constrained_layout=True)
gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1])
fig.suptitle(
    f"Resolving all defects (deepest 2.4 mm, bottom-right): "
    f"standoff {best['d_mm']:.0f} mm, {best['angle_deg']:.0f}°, "
    f"aim ±{best['offset_mm']:.0f} mm, aim line {best['aimline_mm']:.0f} mm "
    f"below centre", fontsize=13, fontweight="bold")

ax = fig.add_subplot(gs[:, 0])
im = ax.imshow(I_best, origin="lower", cmap="inferno",
               extent=[-SPEC_W/2, SPEC_W/2, -SPEC_H/2, SPEC_H/2], aspect="equal")
sc = ax.scatter(dx_, dy_, s=180, c=dz_, cmap="Blues", vmin=0, vmax=2.6,
                edgecolors="w", linewidths=1.4, marker="s")
for x, y, z in DEFECTS:
    ax.annotate(f"{z}", (x, y), color="w", fontsize=7, ha="center",
                textcoords="offset points", xytext=(0, 9))
ax.axhline(best["aimline_mm"], color="cyan", lw=1, ls="--")
ax.text(-155, best["aimline_mm"] + 3, "lamp aim line", color="cyan", fontsize=8)
ax.set_title("Irradiance (K-equiv) with defects (labels = depth mm) — flat-face view")
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
fig.colorbar(im, ax=ax, shrink=0.75, label="K-equiv (5 s, cardboard)")
fig.colorbar(sc, ax=ax, shrink=0.75, label="defect depth (mm)")

# per-defect delivered vs demand
ax = fig.add_subplot(gs[0, 1])
I_def_best = total(dx_, dy_, best["d_mm"], best["angle_deg"],
                   best["offset_mm"], best["aimline_mm"])
I_def_flat = total(dx_, dy_, best["d_mm"], best["angle_deg"],
                   best["offset_mm"], 0.0)
order = np.argsort(dz_)
ax.plot(dz_[order] + np.linspace(-.02, .02, len(dz_)), I_def_flat[order], "o",
        color="0.6", label="aim line centred (dy=0)")
ax.plot(dz_[order] + np.linspace(-.02, .02, len(dz_)), I_def_best[order], "o",
        color="#d62728", label=f"aim line {best['aimline_mm']:.0f} mm")
ax.set_xlabel("defect depth (mm)"); ax.set_ylabel("delivered K-equiv")
ax.set_title("Delivered energy per defect — deepest defects gain, "
             "shallow row gives it up")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# aim-line sweep at each standoff: I at deepest defects
ax = fig.add_subplot(gs[1, 1])
for (d, th, a), c in zip(CONFIGS, plt.cm.plasma(np.linspace(0.05, 0.8, len(CONFIGS)))):
    rr = sorted((r for r in rows if r["d_mm"] == d),
                key=lambda r: r["aimline_mm"])
    ax.plot([r["aimline_mm"] for r in rr], [r["I_deepest"] for r in rr],
            "o-", color=c, label=f"d={d:.0f} mm")
ax.plot(best["aimline_mm"], best["I_deepest"], "k*", ms=16)
ax.set_xlabel("aim-line shift below centre (mm)")
ax.set_ylabel("K-equiv at 2.4 mm defects")
ax.set_title("Energy on the deepest defects vs aim-line drop")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.savefig(OUT_PNG, dpi=140)
print(f"\nBest: d={best['d_mm']:.0f} mm, aim line {best['aimline_mm']:.0f} mm, "
      f"I@2.4mm = {best['I_deepest']:.2f} K-equiv "
      f"(centred aim gives {next(r for r in rows if r['d_mm']==best['d_mm'] and r['aimline_mm']==0)['I_deepest']:.2f})")
print(f"Wrote {OUT_PNG}")

with open(OUT_JSON, "w") as f:
    json.dump({"defects_flat_face_xy_depth": DEFECTS,
               "demand_model": "relative required energy ~ depth^2",
               "best": best,
               "sweep": rows}, f, indent=2)
print(f"Wrote {OUT_JSON}")
