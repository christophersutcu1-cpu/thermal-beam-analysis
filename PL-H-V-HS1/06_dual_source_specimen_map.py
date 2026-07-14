"""
Dual-source specimen mapping — PL-H-V-HS1.

Finds the (standoff, angle-from-normal, aim offset) for a mirrored two-lamp
setup (lamps left + right of the IR camera in the horizontal plane) that gives
the flattest irradiance over a 320 x 175 mm specimen, with priority on
flatness across the 320 mm width.

Beam model (CANONICAL — see ../PROVENANCE_STANDOFFS.md):
  beam_shape_summary_seq.json from 02_summarise_standoffs.py
    sigma_x(z), sigma_y(z) linear in axial distance z  [std convention,
    I = I0 * exp(-0.5*(rx^2/sx^2 + ry^2/sy^2))]
    aspect x/y ~ 1.03 (approximately circular)
  Absolute scale: energy conservation I0(z) = P / (2*pi*sx*sy), with P
  anchored on the 300 & 500 mm shots (they agree to 0.2%; the 700 mm shot
  reads ~40% low — flagged in the JSON caveats).

Per specimen point, per lamp: true axial distance z along the lamp axis,
radial offset in the beam frame, sigma growth with z, 1/(sx*sy) intensity
scaling and cos(incidence) projection onto the specimen plane.

Objective:  score = CoV(face) + WIDTH_WEIGHT * CoV(width profile)
Angle bias: among all sweep points within ANGLE_BIAS_TOL of the best score,
pick the HIGHEST angle (user request — higher angle clears the camera and
costs intensity, not uniformity; intensity is recovered with power/pulse).

Outputs (next to this script):
  06_dual_source_specimen_map.json
  06_dual_source_specimen_map.png
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- constants
HERE = os.path.dirname(os.path.abspath(__file__))
BEAM_JSON = os.path.join(HERE, "beam_shape_summary_seq.json")
OUT_JSON = os.path.join(HERE, "06_dual_source_specimen_map.json")
OUT_PNG = os.path.join(HERE, "06_dual_source_specimen_map.png")

SPEC_W, SPEC_H = 320.0, 175.0          # specimen, mm
GRID_STEP = 2.5                        # evaluation grid, mm

WIDTH_WEIGHT = 2.0                     # extra penalty on horizontal non-flatness
ANGLE_BIAS_TOL = 0.02                  # near-tie band for the high-angle bias

# coarse sweep
D_SWEEP = np.arange(300.0, 751.0, 50.0)        # standoff along lamp axis, mm
TH_SWEEP = np.arange(0.0, 60.1, 2.0)           # angle from normal, deg
A_SWEEP = np.arange(-160.0, 160.1, 5.0)        # aim offset from centre, mm
                                               # a>0: each lamp aims at its own
                                               # side; a<0: cross-fire

# ---------------------------------------------------------------- beam model
with open(BEAM_JSON) as f:
    beam = json.load(f)

AX = beam["linear_fit"]["sigma_x"]["slope_mm_per_mm"]
BX = beam["linear_fit"]["sigma_x"]["intercept_mm"]
AY = beam["linear_fit"]["sigma_y"]["slope_mm_per_mm"]
BY = beam["linear_fit"]["sigma_y"]["intercept_mm"]

# power anchor P = peak * 2*pi*sx*sy  (K*mm^2, cardboard-5s-pulse units)
anchors = {s["d_mm"]: s["peak_dT_K"] * 2 * np.pi * s["sigma_x_mm"] * s["sigma_y_mm"]
           for s in beam["per_standoff"]}
P_LAMP = float(np.mean([anchors[300], anchors[500]]))
P700_DEV = anchors[700] / P_LAMP - 1.0

print(f"Beam model: sx(z) = {AX:.4f} z + {BX:.2f} mm | "
      f"sy(z) = {AY:.4f} z + {BY:.2f} mm")
print(f"Power anchor P = {P_LAMP:.0f} K mm^2  "
      f"(300/500 mm agree to {abs(anchors[300]/anchors[500]-1)*100:.1f}%; "
      f"700 mm reads {P700_DEV*100:+.0f}% — treated as outlier)")

# ---------------------------------------------------------------- geometry
xs = np.arange(-SPEC_W / 2 + GRID_STEP / 2, SPEC_W / 2, GRID_STEP)
ys = np.arange(-SPEC_H / 2 + GRID_STEP / 2, SPEC_H / 2, GRID_STEP)
X, Y = np.meshgrid(xs, ys)             # specimen plane z = 0, lamps at z > 0


def lamp_field(side, d, th_deg, a):
    """Irradiance map (K-equivalent) of one lamp on the specimen grid.

    side = -1 (left lamp) / +1 (right lamp). The lamp aims at (side*a, 0, 0)
    with its axis tilted th_deg from the specimen normal in the horizontal
    plane, standoff d measured along the axis to the aim point.
    """
    th = np.radians(th_deg)
    # axis unit vector lamp -> aim point
    u = np.array([-side * np.sin(th), 0.0, -np.cos(th)])
    aim = np.array([side * a, 0.0, 0.0])
    pos = aim - d * u                          # lamp position, z = d*cos(th)
    # beam-frame transverse axes (ex horizontal-ish, ey vertical)
    ex = np.array([np.cos(th), 0.0, -side * np.sin(th)])
    ey = np.array([0.0, 1.0, 0.0])

    vx, vy, vz = X - pos[0], Y - pos[1], -pos[2]
    z = vx * u[0] + vy * u[1] + vz * u[2]      # axial distance
    rx = vx * ex[0] + vy * ex[1] + vz * ex[2]  # transverse offsets
    ry = vx * ey[0] + vy * ey[1] + vz * ey[2]

    sx = AX * z + BX
    sy = AY * z + BY
    dist = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    cos_inc = pos[2] / dist                    # incidence on the flat specimen

    return (P_LAMP / (2 * np.pi * sx * sy)
            * np.exp(-0.5 * ((rx / sx) ** 2 + (ry / sy) ** 2)) * cos_inc)


def evaluate(d, th_deg, a):
    """Summed two-lamp map + uniformity metrics."""
    I = lamp_field(-1, d, th_deg, a) + lamp_field(+1, d, th_deg, a)
    mean = I.mean()
    cov_face = I.std() / mean
    prof_x = I.mean(axis=0)                    # y-averaged width profile
    cov_x = prof_x.std() / prof_x.mean()
    score = cov_face + WIDTH_WEIGHT * cov_x
    return I, dict(d_mm=d, angle_deg=th_deg, offset_mm=a,
                   cov_face=cov_face, cov_width=cov_x, score=score,
                   mean_K=mean, min_K=I.min(), max_K=I.max(),
                   width_ripple_pct=(prof_x.max() - prof_x.min())
                                    / prof_x.mean() * 100)


def sweep(ds, ths, aas):
    rows = []
    for d in ds:
        for th in ths:
            for a in aas:
                rows.append(evaluate(d, th, a)[1])
    return rows


def pick_biased(rows):
    """Best score, then highest angle within ANGLE_BIAS_TOL of it."""
    best = min(r["score"] for r in rows)
    band = [r for r in rows if r["score"] <= best * (1 + ANGLE_BIAS_TOL)]
    return max(band, key=lambda r: (r["angle_deg"], -r["score"])), best


# ---------------------------------------------------------------- sweeps
print(f"Coarse sweep: {len(D_SWEEP)} standoffs x {len(TH_SWEEP)} angles x "
      f"{len(A_SWEEP)} offsets = {len(D_SWEEP)*len(TH_SWEEP)*len(A_SWEEP)} combos")
coarse = sweep(D_SWEEP, TH_SWEEP, A_SWEEP)
pick, coarse_best = pick_biased(coarse)
print(f"  coarse best score {coarse_best:.4f}; biased pick: "
      f"d={pick['d_mm']:.0f} th={pick['angle_deg']:.1f} a={pick['offset_mm']:.0f} "
      f"(score {pick['score']:.4f})")

# fine refinement around the biased pick
fine = sweep(
    np.clip(np.arange(pick["d_mm"] - 50, pick["d_mm"] + 51, 10.0), 300, 750),
    np.clip(np.arange(pick["angle_deg"] - 3, pick["angle_deg"] + 3.1, 0.5), 0, 60),
    np.arange(pick["offset_mm"] - 7.5, pick["offset_mm"] + 7.6, 1.0),
)
opt, fine_best = pick_biased(fine + [pick])
I_opt = evaluate(opt["d_mm"], opt["angle_deg"], opt["offset_mm"])[0]

# per-standoff bests (same bias rule) for the standoff trade-off curve
per_d = []
for d in D_SWEEP:
    rows_d = [r for r in coarse if r["d_mm"] == d]
    b, _ = pick_biased(rows_d)
    per_d.append(b)

th_rad = np.radians(opt["angle_deg"])
lamp_x = opt["offset_mm"] + opt["d_mm"] * np.sin(th_rad)
lamp_z = opt["d_mm"] * np.cos(th_rad)

print("\n=== OPTIMUM (angle-biased) ===")
print(f"  standoff along axis : {opt['d_mm']:.0f} mm  "
      f"(lamp plane distance {lamp_z:.0f} mm from specimen)")
print(f"  angle from normal   : {opt['angle_deg']:.1f} deg")
print(f"  aim offset          : {opt['offset_mm']:+.0f} mm from centre "
      f"({'own side' if opt['offset_mm'] >= 0 else 'CROSS-FIRE'})")
print(f"  lamp positions      : x = ±{lamp_x:.0f} mm, z = {lamp_z:.0f} mm")
print(f"  CoV over face       : {opt['cov_face']*100:.2f} %")
print(f"  CoV across width    : {opt['cov_width']*100:.2f} %  "
      f"(peak-to-valley {opt['width_ripple_pct']:.1f} %)")
print(f"  mean level          : {opt['mean_K']:.1f} K-equiv "
      f"(cardboard, 5 s, both lamps at characterisation power)")

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(17, 9.5), constrained_layout=True)
gs = fig.add_gridspec(2, 3)
fig.suptitle(
    f"PL-H-V-HS1 dual-source mapping onto {SPEC_W:.0f} x {SPEC_H:.0f} mm specimen — "
    f"optimum: standoff {opt['d_mm']:.0f} mm, {opt['angle_deg']:.1f}° from normal, "
    f"aim offset ±{abs(opt['offset_mm']):.0f} mm"
    f"{' (cross-fire)' if opt['offset_mm'] < 0 else ''}",
    fontsize=13, fontweight="bold")

# (0,0) irradiance map
ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(I_opt, origin="lower",
               extent=[-SPEC_W/2, SPEC_W/2, -SPEC_H/2, SPEC_H/2],
               cmap="inferno", aspect="equal")
ax.contour(X, Y, I_opt / I_opt.mean(), levels=[0.9, 0.95, 1.05, 1.1],
           colors="cyan", linewidths=0.8)
for s in (-1, 1):
    ax.plot(s * opt["offset_mm"], 0, "wx", ms=9, mew=2)
ax.set_title(f"Summed irradiance (K-equiv)\nCoV(face) = {opt['cov_face']*100:.2f} %  |  "
             f"cyan: ±5/±10 % of mean")
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
fig.colorbar(im, ax=ax, shrink=0.8)

# (0,1) width profiles
ax = fig.add_subplot(gs[0, 1])
prof_mean = I_opt.mean(axis=0) / I_opt.mean(axis=0).mean()
mid = I_opt[np.argmin(np.abs(ys)), :]
prof_mid = mid / mid.mean()
ax.plot(xs, prof_mean * 100, lw=2, color="#d62728", label="y-averaged")
ax.plot(xs, prof_mid * 100, lw=1.2, ls="--", color="#1f77b4", label="centreline y=0")
for lv, c in [(95, "0.6"), (105, "0.6"), (100, "0.35")]:
    ax.axhline(lv, color=c, lw=0.8, ls=":" if lv != 100 else "-")
ax.set_ylim(80, 115)
ax.set_title(f"Horizontal flatness across width\n"
             f"CoV = {opt['cov_width']*100:.2f} %, peak-to-valley = "
             f"{opt['width_ripple_pct']:.1f} %")
ax.set_xlabel("x (mm)"); ax.set_ylabel("% of mean")
ax.legend(); ax.grid(alpha=0.3)

# (0,2) vertical profile
ax = fig.add_subplot(gs[0, 2])
pv_mean = I_opt.mean(axis=1) / I_opt.mean(axis=1).mean()
midv = I_opt[:, np.argmin(np.abs(xs))]
ax.plot(ys, pv_mean * 100, lw=2, color="#d62728", label="x-averaged")
ax.plot(ys, midv / midv.mean() * 100, lw=1.2, ls="--", color="#1f77b4",
        label="centreline x=0")
for lv, c in [(95, "0.6"), (105, "0.6"), (100, "0.35")]:
    ax.axhline(lv, color=c, lw=0.8, ls=":" if lv != 100 else "-")
ax.set_ylim(80, 115)
pv_rip = (pv_mean.max() - pv_mean.min()) * 100
ax.set_title(f"Vertical profile over 175 mm height\npeak-to-valley = {pv_rip:.1f} %")
ax.set_xlabel("y (mm)"); ax.set_ylabel("% of mean")
ax.legend(); ax.grid(alpha=0.3)

# (1,0) top-view geometry
ax = fig.add_subplot(gs[1, 0])
ax.plot([-SPEC_W/2, SPEC_W/2], [0, 0], "k-", lw=5, solid_capstyle="butt",
        label="specimen (320 mm)")
ax.plot(0, -opt["d_mm"], "s", color="0.3", ms=10)
ax.annotate("IR camera", (0, -opt["d_mm"]), textcoords="offset points",
            xytext=(0, -16), ha="center")
ax.plot([0, 0], [0, -opt["d_mm"]], color="0.6", lw=0.8, ls="-.")
for s, col in [(-1, "#d62728"), (1, "#1f77b4")]:
    px, pz = s * lamp_x, -lamp_z
    ax.plot(px, pz, "o", color=col, ms=12)
    ax.plot([px, s * opt["offset_mm"]], [pz, 0], color=col, lw=1.2, ls="--")
    ax.annotate(f"lamp {'L' if s < 0 else 'R'}\n({px:+.0f}, {abs(pz):.0f})",
                (px, pz), textcoords="offset points",
                xytext=(s * 30, -6), ha="center", color=col, fontsize=9)
ax.set_title(f"Top view — angle {opt['angle_deg']:.1f}° from normal, "
             f"standoff {opt['d_mm']:.0f} mm along axis")
ax.set_xlabel("x (mm)"); ax.set_ylabel("z toward camera (mm)")
ax.set_aspect("equal"); ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=8)

# (1,1) score map angle x offset at optimum standoff
ax = fig.add_subplot(gs[1, 1])
rows_d = [r for r in coarse if r["d_mm"] ==
          min(D_SWEEP, key=lambda d: abs(d - opt["d_mm"]))]
S = np.full((len(A_SWEEP), len(TH_SWEEP)), np.nan)
for r in rows_d:
    S[np.argmin(np.abs(A_SWEEP - r["offset_mm"])),
      np.argmin(np.abs(TH_SWEEP - r["angle_deg"]))] = r["score"]
im = ax.imshow(S * 100, origin="lower", aspect="auto", cmap="viridis_r",
               extent=[TH_SWEEP[0], TH_SWEEP[-1], A_SWEEP[0], A_SWEEP[-1]],
               vmax=np.nanpercentile(S, 60) * 100)
ax.plot(opt["angle_deg"], opt["offset_mm"], "r*", ms=16, mec="w")
ax.contour(TH_SWEEP, A_SWEEP, S, levels=[fine_best * (1 + ANGLE_BIAS_TOL)],
           colors="w", linewidths=1.2, linestyles="--")
ax.set_title(f"Score at standoff {opt['d_mm']:.0f} mm\n"
             f"white dashes: within {ANGLE_BIAS_TOL*100:.0f}% of best "
             f"(high-angle bias applied)")
ax.set_xlabel("angle from normal (deg)"); ax.set_ylabel("aim offset (mm)")
fig.colorbar(im, ax=ax, shrink=0.8, label="score x100")

# (1,2) standoff trade-off + parameter table
ax = fig.add_subplot(gs[1, 2])
dd = [r["d_mm"] for r in per_d]
ax.plot(dd, [r["cov_face"] * 100 for r in per_d], "o-", color="#d62728",
        label="CoV face %")
ax.plot(dd, [r["cov_width"] * 100 for r in per_d], "s-", color="#1f77b4",
        label="CoV width %")
ax2 = ax.twinx()
ax2.plot(dd, [r["mean_K"] for r in per_d], "^--", color="0.4",
         label="mean level (K-equiv)")
ax2.set_ylabel("mean K-equiv (right)")
ax.axvline(opt["d_mm"], color="r", lw=1, ls=":")
ax.set_title("Best-per-standoff trade-off\n(angle & offset re-optimised at each d)")
ax.set_xlabel("standoff (mm)"); ax.set_ylabel("CoV (%)")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8); ax.grid(alpha=0.3)

fig.savefig(OUT_PNG, dpi=140)
print(f"\nWrote {OUT_PNG}")

# ---------------------------------------------------------------- json
out = {
    "beam_model_source": os.path.basename(BEAM_JSON),
    "beam_model": {
        "sigma_x_mm(z)": f"{AX:.5f}*z + {BX:.3f}",
        "sigma_y_mm(z)": f"{AY:.5f}*z + {BY:.3f}",
        "profile": "I0(z)*exp(-0.5*(rx^2/sx^2+ry^2/sy^2)), I0 = P/(2*pi*sx*sy)",
        "P_anchor_K_mm2": P_LAMP,
        "caveat_700mm_anchor_deviation": f"{P700_DEV*100:+.0f}% vs energy "
                                         "conservation — absolute levels at long "
                                         "standoff may be optimistic; geometry/"
                                         "uniformity results unaffected",
    },
    "specimen_mm": [SPEC_W, SPEC_H],
    "objective": f"CoV(face) + {WIDTH_WEIGHT}*CoV(width profile); highest angle "
                 f"within {ANGLE_BIAS_TOL*100:.0f}% of best score preferred",
    "sweep": {"standoff_mm": [float(D_SWEEP[0]), float(D_SWEEP[-1])],
              "angle_deg": [float(TH_SWEEP[0]), float(TH_SWEEP[-1])],
              "offset_mm": [float(A_SWEEP[0]), float(A_SWEEP[-1])]},
    "optimum": {**{k: float(v) for k, v in opt.items()},
                "lamp_positions_mm": {"left": [-lamp_x, 0.0, lamp_z],
                                      "right": [lamp_x, 0.0, lamp_z]},
                "aim_points_mm": {"left": [-opt["offset_mm"], 0.0],
                                  "right": [opt["offset_mm"], 0.0]},
                "unbiased_best_score": float(fine_best)},
    "per_standoff_best": [{k: float(v) for k, v in r.items()} for r in per_d],
    "notes": [
        "offset > 0: each lamp aims at its own side of centre; < 0: cross-fire",
        "standoff = lamp-to-aim-point distance along the lamp axis",
        "mean/min/max K are cardboard-absorber dT-equivalents for a 5 s pulse "
        "at the characterisation operating point, both lamps",
    ],
}
with open(OUT_JSON, "w") as f:
    json.dump(out, f, indent=2)
print(f"Wrote {OUT_JSON}")
