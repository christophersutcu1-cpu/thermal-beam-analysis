"""
06d — Experiment design: resolve all 19 Teflon inserts with the dual-lamp
2 kW setup (follow-up to the Energy-Density paper).

Physics (per PL-H-V-HS1/CLAUDE.md, updated from the draft paper):
  Beam    : DFLUX convention I = I0*exp(-(r/sigma)^2), sigma(d) = 0.185 d + 9.1
            I0 = P*eta/(pi*sigma^2), P = 1000 W/lamp, eta = 0.708.
            Absolute predictions derated by the paper's +12.7% validated
            over-prediction. Absorptivity 0.945.
  Defects : 13 um Teflon inserts, 10x10 mm, gap conductance hc = 7390 W/m2K
            (R = 1/hc), at depth z inside the local section thickness L.
  Contrast: 1D implicit FD slab model per (z, L): front flux + linearised
            convection/radiation losses both faces, insert as interface
            resistance. Contrast = front-face T(defect) - T(sound), linear in
            absorbed flux -> geometry and contrast factorise:
            C_j = q_abs(x_j, y_j) * g(z_j, L_j, tau).
  3D note : 1D over-predicts peak contrast for 10 mm inserts once the in-plane
            diffusion length approaches the insert half-width; a lateral-loss
            factor erf(w/2 / sqrt(4*a11*t_pk))^2 is applied as the reported
            (conservative) estimate.

Optimiser: sweep (standoff, angle, mirrored aim offset, aim-line drop),
maximise the minimum defect contrast; pulse duration picked from the
contrast-vs-tau knee. Uniformity reported as face CoV + 2x width CoV.

Outputs: 06d_experiment_design.{json,png}
"""

import json
import os

import numpy as np
from scipy.linalg import solve_banded
from scipy.special import erf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, "06d_experiment_design.json")
OUT_PNG = os.path.join(HERE, "06d_experiment_design.png")

# ---------------- constants ------------------------------------------------
SPEC_W, SPEC_H = 320.0, 175.0
P_LAMP, ETA = 1000.0, 0.708            # W electrical per lamp, delivery eff.
DERATE = 1.0 / 1.127                   # paper: model over-predicts +12.7 %
ABSORPT = 0.945
SIG_A, SIG_B = 0.185, 9.1              # sigma(d) mm, DFLUX convention

RHO, CP, K33, K11 = 1533.0, 979.0, 2.9, 5.8
RHOC = RHO * CP
A11 = K11 / RHOC * 1e6                 # mm^2/s in-plane
R_INS = 1.0 / 7390.0                   # m2K/W insert interface resistance
H_LOSS = 10.5                          # W/m2K conv (5) + linearised rad (~5.5)

NETD = 0.03
THRESH = 3 * NETD                      # raw detectability criterion, K
DT_SURF_LIMIT = 80.0                   # K, front-face rise flag
INSERT_W = 10.0                        # mm

TAUS = [1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0]
T_END, DT = 60.0, 0.02

# sections in FLAT-FACE coords (x from, x to, thickness mm)
SECTIONS = [(-160, -100, 1.14), (-100, -20, 1.75), (-20, 20, 2.35), (20, 160, 3.00)]

DEFECTS = [  # (x, y, depth) flat-face view
    (+110, 20, 0.3), (+77, 20, 0.3), (+45, 20, 0.3), (+1, 20, 0.3),
    (-40, 20, 0.3), (-70, 20, 0.3), (-120, 20, 0.3),
    (+110, -22, 1.2), (+77, -22, 1.2), (+45, -22, 0.9), (+1, -22, 0.9),
    (-40, -22, 0.9), (-70, -22, 0.9), (-122, -22, 0.6),
    (+110, -64, 2.4), (+77, -64, 2.4), (+45, -64, 1.8), (+1, -64, 1.8),
    (-122, -64, 0.9),
]


def thickness_at(x):
    for x0, x1, L in SECTIONS:
        if x0 <= x <= x1:
            return L
    raise ValueError(x)


# ---------------- 1D FD contrast model ------------------------------------
def front_response(L_mm, z_def_mm, tau, q_abs=1000.0):
    """Front-face T(t) for a slab with optional insert (z_def_mm=None: sound).
    Square pulse q_abs [W/m2] of duration tau. Returns t, T_front."""
    n_int = max(12, int(round(L_mm / 0.05)))
    dz = L_mm / n_int * 1e-3                      # m
    N = n_int + 1
    G = np.full(n_int, K33 / dz)                  # inter-node conductance
    if z_def_mm is not None:
        i = int(round(z_def_mm / (L_mm / n_int)))
        i = min(max(i, 1), n_int - 1)
        G[i - 1] = 1.0 / (dz / K33 + R_INS)
    C = np.full(N, RHOC * dz); C[0] *= 0.5; C[-1] *= 0.5

    # banded matrix for implicit Euler: (C/dt + K) T+ = C/dt T + b
    ab = np.zeros((3, N))
    ab[0, 1:] = -G                                 # upper
    ab[2, :-1] = -G                                # lower
    ab[1, :] = C / DT
    ab[1, :-1] += G
    ab[1, 1:] += G
    ab[1, 0] += H_LOSS
    ab[1, -1] += H_LOSS

    steps = int(T_END / DT)
    T = np.zeros(N)
    out = np.empty(steps)
    for s in range(steps):
        t = (s + 1) * DT
        rhs = C / DT * T
        if t <= tau:
            rhs[0] += q_abs
        T = solve_banded((1, 1), ab, rhs)
        out[s] = T[0]
    return np.arange(1, steps + 1) * DT, out


print("Building unit-contrast factors g(z, L, tau) ...")
pairs = sorted({(z, thickness_at(x)) for x, y, z in DEFECTS})
Ls = sorted({L for _, L in pairs})
g_peak, g_tpk, surf_unit = {}, {}, {}
for tau in TAUS:
    sound = {L: front_response(L, None, tau) for L in Ls}
    for (z, L) in pairs:
        t, Td = front_response(L, z, tau)
        c = Td - sound[L][1]
        j = int(np.argmax(c))
        g_peak[(z, L, tau)] = c[j] / 1000.0        # K per (W/m2)
        g_tpk[(z, L, tau)] = t[j]
    for L in Ls:
        surf_unit[(L, tau)] = sound[L][1].max() / 1000.0
print(f"  done ({len(pairs)} (z,L) pairs x {len(TAUS)} pulse durations)")


# ---------------- geometry: absorbed-flux field ----------------------------
def q_abs_points(X, Y, d, th_deg, a, dy):
    """Absorbed flux [W/m2] at points (X, Y) on the flat face."""
    th = np.radians(th_deg)
    total = 0.0
    for side in (-1, 1):
        u = np.array([-side * np.sin(th), 0.0, -np.cos(th)])
        pos = np.array([side * a, dy, 0.0]) - d * u
        ex = np.array([np.cos(th), 0.0, -side * np.sin(th)])
        vx, vy, vz = X - pos[0], Y - pos[1], -pos[2]
        zax = vx * u[0] + vz * u[2]                # axial distance, mm
        rx = vx * ex[0] + vz * ex[2]
        sig = SIG_A * zax + SIG_B                  # mm
        I0 = P_LAMP * ETA / (np.pi * (sig * 1e-3) ** 2)
        cinc = pos[2] / np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
        total = total + I0 * np.exp(-(rx ** 2 + vy ** 2) / sig ** 2) * cinc
    return total * ABSORPT * DERATE


dxs = np.array([d[0] for d in DEFECTS], float)
dys = np.array([d[1] for d in DEFECTS], float)
dzs = np.array([d[2] for d in DEFECTS], float)
dLs = np.array([thickness_at(x) for x in dxs])

TAU_RANK = 15.0                                    # tau used for geometry ranking
g_rank = np.array([g_peak[(z, L, TAU_RANK)] for z, L in zip(dzs, dLs)])

print("Sweeping geometry (standoff x angle x offset x aim-line drop) ...")
best = None
for d in [300., 350., 400., 450., 500., 600., 750.]:
    for th in [30., 40., 45., 50., 55., 60.]:
        for a in np.arange(-160., 160.1, 10.):
            for dy in np.arange(0., -70.1, -10.):
                q = q_abs_points(dxs, dys, d, th, a, dy)
                m = (q * g_rank).min()
                if best is None or m > best[0]:
                    best = (m, d, th, a, dy)
minC_rank, D, TH, A, DY = best
print(f"  best geometry: d={D:.0f} mm, {TH:.0f} deg, aim ±{A:.0f} mm, "
      f"aim line {DY:.0f} mm  (min contrast {minC_rank:.3f} K @ tau={TAU_RANK:.0f}s)")

# refine
for d in np.arange(max(300, D - 40), D + 41, 10.):
    for th in np.arange(max(30, TH - 4), min(60, TH + 4) + .1, 1.):
        for a in np.arange(A - 8, A + 8.1, 2.):
            for dy in np.arange(max(-70, DY - 8), min(0, DY + 8) + .1, 2.):
                q = q_abs_points(dxs, dys, d, th, a, dy)
                m = (q * g_rank).min()
                if m > best[0]:
                    best = (m, d, th, a, dy)
minC_rank, D, TH, A, DY = best
q_def = q_abs_points(dxs, dys, D, TH, A, DY)

# ---------------- pulse duration choice ------------------------------------
tau_curve = []
for tau in TAUS:
    g = np.array([g_peak[(z, L, tau)] for z, L in zip(dzs, dLs)])
    jmin = int(np.argmin(q_def * g))
    # surface-temperature worst case: max q within each section vs its L
    gx = np.linspace(-SPEC_W/2, SPEC_W/2, 129)
    gy = np.linspace(-SPEC_H/2, SPEC_H/2, 71)
    GX, GY = np.meshgrid(gx, gy)
    Q = q_abs_points(GX, GY, D, TH, A, DY)
    dTs = max(Q[:, (gx >= x0) & (gx <= x1)].max() * surf_unit[(L, tau)]
              for x0, x1, L in SECTIONS)
    tau_curve.append(dict(tau=tau, min_contrast=float((q_def * g).min()),
                          limiting=int(jmin), max_surf_dT=float(dTs)))

for r in tau_curve:
    print(f"  tau={r['tau']:5.1f} s  min contrast {r['min_contrast']:.3f} K  "
          f"max surface rise {r['max_surf_dT']:.1f} K")

# knee: among pulses within the surface-temperature cap, the shortest one that
# reaches 90 % of the best feasible min-contrast; if nothing is within the cap,
# take the shortest pulse (least heat) and flag it.
feasible = [r for r in tau_curve if r["max_surf_dT"] <= DT_SURF_LIMIT]
if feasible:
    c_best = max(r["min_contrast"] for r in feasible)
    row_star = next(r for r in feasible if r["min_contrast"] >= 0.9 * c_best)
else:
    row_star = tau_curve[0]
    print(f"  WARNING: no pulse keeps surface rise <= {DT_SURF_LIMIT:.0f} K; "
          f"using shortest pulse — consider a larger standoff")
TAU_STAR = row_star["tau"]

g_star = np.array([g_peak[(z, L, TAU_STAR)] for z, L in zip(dzs, dLs)])
t_star = np.array([g_tpk[(z, L, TAU_STAR)] for z, L in zip(dzs, dLs)])
C_1d = q_def * g_star
lat = erf(INSERT_W / 2 / np.sqrt(4 * A11 * t_star)) ** 2   # 3D lateral loss
C_3d = C_1d * lat

# uniformity of the chosen field (kept metric: face CoV + 2x width CoV)
gx = np.linspace(-SPEC_W/2, SPEC_W/2, 129); gy = np.linspace(-SPEC_H/2, SPEC_H/2, 71)
GX, GY = np.meshgrid(gx, gy)
Q = q_abs_points(GX, GY, D, TH, A, DY)
cov_face = Q.std() / Q.mean()
px = Q.mean(axis=0); cov_w = px.std() / px.mean()

th_r = np.radians(TH)
lamp_x, lamp_z = A + D * np.sin(th_r), D * np.cos(th_r)
capture = TAU_STAR + float(t_star.max()) + 5.0

print(f"\n=== EXPERIMENT RECIPE ===")
print(f"  standoff (lamp aperture -> aim point) : {D:.0f} mm")
print(f"  angle from normal                     : {TH:.0f} deg")
print(f"  aim points                            : (±{A:.0f}, {DY:.0f}) mm")
print(f"  lamp positions                        : x=±{lamp_x:.0f}, y={DY:.0f}, "
      f"z={lamp_z:.0f} mm")
print(f"  power / pulse                         : 2000 W total, {TAU_STAR:.0f} s")
print(f"  capture                               : >= {capture:.0f} s from pulse start")
print(f"  min defect contrast (1D / 3D-corr)    : {C_1d.min():.3f} / "
      f"{C_3d.min():.3f} K vs threshold {THRESH:.2f} K")
print(f"  max front-face rise                   : {row_star['max_surf_dT']:.1f} K")
print(f"  field CoV face / width                : {cov_face*100:.1f}% / {cov_w*100:.2f}%")
print("\n  defect (x, y)   z(mm) L(mm)  q_abs(kW/m2)  C_1d(K)  C_3d(K)  t_pk(s)")
for j in np.argsort(C_3d):
    print(f"   ({dxs[j]:+4.0f},{dys[j]:+4.0f})  {dzs[j]:4.1f}  {dLs[j]:4.2f}"
          f"   {q_def[j]/1000:6.2f}      {C_1d[j]:6.3f}  {C_3d[j]:6.3f}"
          f"   {TAU_STAR + 0*t_star[j]:.0f}->{t_star[j]:5.1f}")

# ---------------- figure ----------------------------------------------------
fig = plt.figure(figsize=(17, 9), constrained_layout=True)
gs = fig.add_gridspec(2, 3)
fig.suptitle(
    f"Experiment design — 2 kW dual-source, resolve all 19 Teflon inserts:  "
    f"standoff {D:.0f} mm · {TH:.0f}° · aim (±{A:.0f}, {DY:.0f}) mm · "
    f"pulse {TAU_STAR:.0f} s", fontsize=13, fontweight="bold")

ax = fig.add_subplot(gs[0, 0:2])
im = ax.imshow(Q / 1000, origin="lower", cmap="inferno",
               extent=[-SPEC_W/2, SPEC_W/2, -SPEC_H/2, SPEC_H/2], aspect="equal")
for x0, x1, L in SECTIONS[:-1]:
    ax.axvline(x1, color="w", lw=0.6, ls=":")
sc = ax.scatter(dxs, dys, s=170, c=C_3d, cmap="RdYlGn",
                vmin=0, vmax=max(3*THRESH, C_3d.max()),
                edgecolors="k", linewidths=1.2, marker="s")
for j in range(len(DEFECTS)):
    ax.annotate(f"{dzs[j]}", (dxs[j], dys[j]), color="k", fontsize=7,
                ha="center", textcoords="offset points", xytext=(0, 10))
ax.set_title(f"Absorbed flux (kW/m²), defect markers coloured by predicted "
             f"contrast — face CoV {cov_face*100:.1f}%, width CoV {cov_w*100:.2f}%")
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
fig.colorbar(im, ax=ax, shrink=0.75, label="kW/m²")
fig.colorbar(sc, ax=ax, shrink=0.75, label="C_3d (K)")

ax = fig.add_subplot(gs[0, 2])
order = np.argsort(C_3d)
lbl = [f"z{dzs[j]:.1f}/L{dLs[j]:.2f} ({dxs[j]:+.0f})" for j in order]
ax.barh(np.arange(len(order)), C_3d[order], color="#1f77b4", label="3D-corrected")
ax.barh(np.arange(len(order)), C_1d[order], fill=False, edgecolor="0.5",
        label="1D upper bound")
ax.axvline(THRESH, color="r", ls="--", lw=1.2, label=f"3×NETD = {THRESH:.2f} K")
ax.axvline(NETD, color="r", ls=":", lw=0.8, label="NETD")
ax.set_yticks(np.arange(len(order))); ax.set_yticklabels(lbl, fontsize=6.5)
ax.set_xlabel("peak contrast (K)"); ax.set_xscale("log")
ax.set_title(f"Per-defect predicted contrast, pulse {TAU_STAR:.0f} s")
ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="x")

ax = fig.add_subplot(gs[1, 0])
ax.plot([r["tau"] for r in tau_curve], [r["min_contrast"] for r in tau_curve],
        "o-", color="#d62728", label="min defect contrast (1D)")
ax.axvline(TAU_STAR, color="k", ls=":", lw=1)
ax.axhline(THRESH, color="r", ls="--", lw=0.8)
ax.set_xlabel("pulse duration (s)"); ax.set_ylabel("K")
ax2 = ax.twinx()
ax2.plot([r["tau"] for r in tau_curve], [r["max_surf_dT"] for r in tau_curve],
         "s--", color="0.4", label="max surface rise")
ax2.axhline(DT_SURF_LIMIT, color="0.4", ls=":", lw=0.8)
ax2.set_ylabel("max front-face ΔT (K)")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8)
ax.set_title(f"Pulse-duration choice (knee at {TAU_STAR:.0f} s)")
ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[1, 1])
tt, Td = front_response(3.00, 2.4, TAU_STAR, q_abs=q_def[dzs == 2.4].min())
_, Ts = front_response(3.00, None, TAU_STAR, q_abs=q_def[dzs == 2.4].min())
ax.plot(tt, Td - Ts, lw=2, color="#1f77b4",
        label="2.4 mm insert (1D contrast)")
ax.axhline(THRESH, color="r", ls="--", lw=0.8)
ax.axvline(TAU_STAR, color="k", ls=":", lw=0.8)
ax.set_xlabel("time from pulse start (s)"); ax.set_ylabel("contrast (K)")
ax.set_title(f"Deepest defect contrast history — capture ≥ {capture:.0f} s")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[1, 2])
ax.plot([-SPEC_W/2, SPEC_W/2], [0, 0], "k-", lw=5, solid_capstyle="butt")
ax.plot(0, -420, "s", color="0.3", ms=9)
ax.annotate("camera", (0, -420), textcoords="offset points", xytext=(0, -14),
            ha="center", fontsize=8)
for s, col in [(-1, "#d62728"), (1, "#1f77b4")]:
    ax.plot(s*lamp_x, -lamp_z, "o", color=col, ms=11)
    ax.plot([s*lamp_x, s*A], [-lamp_z, 0], color=col, lw=1, ls="--")
    ax.annotate(f"({s*lamp_x:+.0f}, {lamp_z:.0f})\ny={DY:.0f}", (s*lamp_x, -lamp_z),
                textcoords="offset points", xytext=(s*28, -4), ha="center",
                fontsize=8, color=col)
ax.set_aspect("equal"); ax.grid(alpha=0.3)
ax.set_title("Top view (aim line dropped %.0f mm below centre)" % -DY)
ax.set_xlabel("x (mm)"); ax.set_ylabel("z (mm)")

fig.savefig(OUT_PNG, dpi=140)
print(f"\nWrote {OUT_PNG}")

out = dict(
    model=dict(beam="I0=P*eta/(pi*sigma^2), sigma=0.185d+9.1, exp(-(r/sigma)^2)",
               P_lamp_W=P_LAMP, eta=ETA, derate_vs_paper=DERATE, absorptivity=ABSORPT,
               material=dict(rho=RHO, cp=CP, k33=K33, k11=K11),
               insert_R_m2K_W=R_INS, h_loss=H_LOSS,
               note="1D FD slab per (depth, section thickness); 3D lateral "
                    "correction erf(w/2/sqrt(4*a11*t_pk))^2"),
    recipe=dict(standoff_mm=D, angle_deg=TH, aim_offset_mm=A, aim_line_mm=DY,
                lamp_positions=dict(left=[-lamp_x, DY, lamp_z],
                                    right=[lamp_x, DY, lamp_z]),
                power_total_W=2000, pulse_s=TAU_STAR, capture_s=capture,
                min_contrast_1d_K=float(C_1d.min()),
                min_contrast_3d_K=float(C_3d.min()),
                max_surface_dT_K=row_star["max_surf_dT"],
                cov_face=float(cov_face), cov_width=float(cov_w)),
    per_defect=[dict(x=float(dxs[j]), y=float(dys[j]), depth=float(dzs[j]),
                     thickness=float(dLs[j]), q_abs_W_m2=float(q_def[j]),
                     C_1d_K=float(C_1d[j]), C_3d_K=float(C_3d[j]),
                     t_peak_s=float(t_star[j])) for j in range(len(DEFECTS))],
    tau_sweep=tau_curve,
)
with open(OUT_JSON, "w") as f:
    json.dump(out, f, indent=2)
print(f"Wrote {OUT_JSON}")
