"""
Triangle side breakdown for side-lamp fixture.
Fixed: d = 383mm (standoff), x_s = 105mm (beam offset, Config 1 best).

For each beam angle theta:
  L = x_s + d*tan(theta)   horizontal leg  (lamp lateral distance)
  d = 383mm                 vertical leg    (always fixed)
  R = d/cos(theta)          beam path lamp → specimen
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import matplotlib.patches as mpatches
from matplotlib.patches import Arc
import matplotlib.patheffects as pe

OUTPUT_PNG = config.BOSON_ROOT + r"\lamp_distance_vs_angle.png"

D  = 383.0   # mm
XS = 105.0   # mm

ANGLES = [20, 30, 40, 45, 55, 65]
COLS   = ["#4d96ff", "#6bcb77", "#a29bfe", "#ffd93d", "#ff9f43", "#ff6b6b"]

def sides(theta_deg):
    tr = np.radians(theta_deg)
    L  = XS + D * np.tan(tr)
    R  = D / np.cos(tr)
    return dict(theta=theta_deg, L=L, R=R, d=D)

DATA = [sides(a) for a in ANGLES]

# continuous sweep for curves
th_sweep = np.linspace(5, 72, 600)
tr_sweep = np.radians(th_sweep)
L_sweep  = XS + D * np.tan(tr_sweep)
R_sweep  = D / np.cos(tr_sweep)

print(f"d = {D:.0f}mm (fixed)   x_s = {XS:.0f}mm (fixed)\n")
print(f"{'Angle':>7}  {'d (vert)':>10}  {'L (horiz)':>11}  {'R (beam)':>10}")
print("-" * 45)
for v in DATA:
    print(f"{v['theta']:>6}deg  {v['d']:>9.1f}mm  {v['L']:>10.1f}mm  {v['R']:>9.1f}mm")

# ── helpers ────────────────────────────────────────────────────────────────────
def dark_ax(ax):
    ax.set_facecolor("#111111")
    for sp in ax.spines.values(): sp.set_edgecolor("#333333")
    ax.tick_params(colors="#888888", labelsize=9)
    ax.grid(True, alpha=0.10, color="white", lw=0.5)

# ── figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 20))
fig.patch.set_facecolor("#0d0d0d")
gs  = mgridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.28,
                          left=0.06, right=0.97, top=0.93, bottom=0.04,
                          height_ratios=[1.4, 1, 1])

# ══════════════════════════════════════════════════════════════════════════════
# ROW 0 (full width) — All lamp positions at true scale
# ══════════════════════════════════════════════════════════════════════════════
ax_fan = fig.add_subplot(gs[0, :])
ax_fan.set_facecolor("#080808")
for sp in ax_fan.spines.values(): sp.set_edgecolor("#222222")
ax_fan.tick_params(colors="#777777", labelsize=9)
ax_fan.set_aspect("equal")

max_L = DATA[-1]["L"]

# specimen base
ax_fan.fill_between([-40, max_L + 60], -22, 0, color="#0a1a0a", alpha=0.8)
ax_fan.plot([-40, max_L + 60], [0, 0], color="lime", lw=2.5)
ax_fan.text(XS, -13, f"x_s = {XS:.0f}mm", ha="center",
            color="lime", fontsize=9, fontweight="bold")
ax_fan.plot(XS, 0, "^", color="lime", ms=12, zorder=10)

# camera box
ax_fan.add_patch(mpatches.FancyBboxPatch((-26, D - 20), 52, 40,
    boxstyle="round,pad=4", fc="#0e0e2a", ec="#4444cc", lw=2.5, zorder=8))
ax_fan.text(0, D, "Camera", ha="center", va="center",
            color="#8888ee", fontsize=10, fontweight="bold", zorder=9)

# shared vertical leg d (drawn once, white)
ax_fan.plot([0, 0], [0, D - 20], color="#555555", lw=2, ls="--", zorder=3)
ax_fan.annotate("", xy=(-38, 0), xytext=(-38, D),
                arrowprops=dict(arrowstyle="<->", color="white", lw=1.8))
ax_fan.text(-48, D / 2, f"d = {D:.0f}mm\n(fixed)", ha="right", va="center",
            color="white", fontsize=10, fontweight="bold")

# right-angle mark
sq = 14
ax_fan.plot([sq, sq, 0], [D, D - sq, D - sq], color="#555555", lw=1.5)

# draw each angle
t_vals = [0.62, 0.56, 0.50, 0.44, 0.38, 0.32]   # R-label positions along beam
for i, (v, col) in enumerate(zip(DATA, COLS)):
    L = v["L"];  a = v["theta"]

    # horizontal leg L (at height D)
    ax_fan.plot([0, L], [D, D], color=col, lw=3.0, solid_capstyle="round", zorder=5)

    # beam path: lamp to specimen
    ax_fan.plot([L, XS], [D, 0], color=col, lw=1.8, ls="--", alpha=0.65, zorder=4)

    # lamp dot
    ax_fan.plot(L, D, "o", color=col, ms=14, zorder=11,
                path_effects=[pe.withStroke(linewidth=4, foreground="#000000")])

    # arc from horizontal leg to beam, labelled with angle
    beam_dir = np.degrees(np.arctan2(-D, XS - L))   # ~250° for small a, ~205° for large a
    arc_r = 42
    ax_fan.add_patch(Arc((L, D), 2 * arc_r, 2 * arc_r,
                         angle=0, theta1=180, theta2=beam_dir,
                         color=col, lw=1.8, zorder=5))
    arc_mid = np.radians((180 + beam_dir) / 2)
    lbl_r = arc_r + 20
    ax_fan.text(L + lbl_r * np.cos(arc_mid), D + lbl_r * np.sin(arc_mid),
                f"{a}°", ha="center", va="center",
                color=col, fontsize=9, fontweight="bold", zorder=6)

    # L label — stacked double-headed arrows above the horizontal leg, one per angle
    y_arr = D + 28 + i * 22
    ax_fan.annotate("", xy=(L, y_arr), xytext=(0, y_arr),
                    arrowprops=dict(arrowstyle="<->", color=col, lw=1.5))
    ax_fan.text(L / 2, y_arr + 6, f"L = {L:.0f} mm",
                ha="center", va="bottom", color=col, fontsize=8.5, fontweight="bold",
                bbox=dict(fc="#000000cc", ec="none", pad=1))

    # R label — centered on beam, rotated to follow it, staggered along beam
    t = t_vals[i]
    rx = L + t * (XS - L)
    ry = D * (1 - t)
    beam_rot = np.degrees(np.arctan2(D, L - XS))
    ax_fan.text(rx, ry, f"R = {v['R']:.0f} mm",
                ha="center", va="center", color=col, fontsize=8, fontweight="bold",
                rotation=beam_rot, rotation_mode="anchor", zorder=6,
                bbox=dict(fc="#000000bb", ec="none", pad=1.5))

ax_fan.set_xlim(-70, max_L + 200)
ax_fan.set_ylim(-30, D + 28 + 6 * 22 + 30)   # accommodate stacked L arrows
ax_fan.set_xlabel("Horizontal distance from camera axis (mm)", color="#888888", fontsize=10)
ax_fan.set_ylabel("Height above specimen (mm)", color="#888888", fontsize=10)
ax_fan.set_title(
    f"Lamp geometry — true scale  |  d = {D:.0f} mm (vertical, fixed)  |  x_s = {XS:.0f} mm\n"
    "Solid = L (horizontal)   Dashed = R (beam path lamp → specimen)   Arc = beam angle from horizontal",
    color="white", fontsize=11)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1, Col 0 — Grouped bars: d and L per angle
# ══════════════════════════════════════════════════════════════════════════════
ax_bar = fig.add_subplot(gs[1, 0])
ax_bar.set_facecolor("#111111")
for sp in ax_bar.spines.values(): sp.set_edgecolor("#333333")
ax_bar.tick_params(colors="#888888", labelsize=9)
ax_bar.yaxis.grid(True, alpha=0.12, color="white", lw=0.5)
ax_bar.set_axisbelow(True)

n     = len(ANGLES)
x_pos = np.arange(n)
width = 0.35

ax_bar.bar(x_pos - width / 2, [D] * n,               width, color="white", alpha=0.75, label="d (standoff, fixed)")
ax_bar.bar(x_pos + width / 2, [v["L"] for v in DATA], width, color=COLS,   alpha=0.90, label="L (horizontal leg)")

for i, v in enumerate(DATA):
    ax_bar.text(i - width / 2, D + 6,      f"{D:.0f}",     ha="center", color="white",  fontsize=7.5, fontweight="bold")
    ax_bar.text(i + width / 2, v["L"] + 6, f"{v['L']:.0f}", ha="center", color=COLS[i], fontsize=7.5, fontweight="bold")

ax_bar.set_xticks(x_pos)
ax_bar.set_xticklabels([f"{a}°" for a in ANGLES], color="#aaaaaa")
ax_bar.set_ylabel("Side length (mm)", color="#888888", fontsize=9)
ax_bar.set_title("d and L per angle\nd (white) | L (solid colour)", color="white", fontsize=10)
ax_bar.legend(fontsize=8, facecolor="#1a1a1a", labelcolor="white",
              edgecolor="#333333", loc="upper left")
ax_bar.set_ylim(0, max(v["L"] for v in DATA) * 1.20)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1, Col 1 — d, L and R vs theta continuous curves
# ══════════════════════════════════════════════════════════════════════════════
ax_crv = fig.add_subplot(gs[1, 1])
dark_ax(ax_crv)

ax_crv.plot(th_sweep, [D] * len(th_sweep), color="white",   lw=2.5, ls=":",  label=f"d = {D:.0f}mm  (vertical, fixed)")
ax_crv.plot(th_sweep, L_sweep,              color="#4d96ff", lw=2.5,          label="L = x_s + d·tan(θ)  (horizontal)")
ax_crv.plot(th_sweep, R_sweep,              color="#ff9f43", lw=1.8, ls="-.", label="R = d/cos(θ)  (beam path length)")

for v, col in zip(DATA, COLS):
    ax_crv.plot(v["theta"], v["d"], "o", color="white", ms=7,  zorder=7)
    ax_crv.plot(v["theta"], v["L"], "o", color=col,     ms=8,  zorder=7)
    ax_crv.plot(v["theta"], v["R"], "D", color=col,     ms=6,  zorder=7)

ax_crv.set_xlabel("Beam angle θ (degrees)", color="#888888", fontsize=9)
ax_crv.set_ylabel("Length (mm)", color="#888888", fontsize=9)
ax_crv.set_title("d, L and beam path R vs angle\n(d flat, L diverges at high angles)",
                 color="white", fontsize=10)
ax_crv.legend(fontsize=8, facecolor="#1a1a1a", labelcolor="white",
              edgecolor="#333333", loc="upper left")
ax_crv.set_xlim(5, 72)
ax_crv.set_ylim(0, max(L_sweep.max(), R_sweep.max()) * 1.10)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1, Col 2 — Summary table: theta, d, L, R
# ══════════════════════════════════════════════════════════════════════════════
ax_tbl = fig.add_subplot(gs[1, 2])
ax_tbl.set_facecolor("#111111")
ax_tbl.axis("off")

col_x = [0.05, 0.30, 0.55, 0.78]
row_h = 0.12
y0    = 0.92
hdrs  = ["Angle", "d (mm)", "L (mm)", "R (mm)"]
hcols = ["white", "white", "#4d96ff", "#ff9f43"]

for ci, (hdr, hcol) in enumerate(zip(hdrs, hcols)):
    ax_tbl.text(col_x[ci], y0, hdr, transform=ax_tbl.transAxes,
                color=hcol, fontsize=10, fontweight="bold", va="top")
ax_tbl.plot([0.02, 0.98], [y0 - 0.04, y0 - 0.04],
            color="#444444", lw=1, transform=ax_tbl.transAxes, clip_on=False)

for ri, (v, col) in enumerate(zip(DATA, COLS)):
    y  = y0 - (ri + 1) * row_h
    bg = "#1e1e1e" if ri % 2 == 0 else "#161616"
    ax_tbl.add_patch(mpatches.Rectangle(
        (0.02, y - 0.02), 0.96, row_h - 0.01,
        transform=ax_tbl.transAxes, fc=bg, ec="none", zorder=0))
    ax_tbl.text(col_x[0], y, f"{v['theta']}°",  transform=ax_tbl.transAxes,
                color=col,     fontsize=9.5, va="top", fontweight="bold")
    ax_tbl.text(col_x[1], y, f"{v['d']:.0f}",   transform=ax_tbl.transAxes,
                color="white", fontsize=9.5, va="top")
    ax_tbl.text(col_x[2], y, f"{v['L']:.0f}",   transform=ax_tbl.transAxes,
                color="#4d96ff", fontsize=9.5, va="top", fontweight="bold")
    ax_tbl.text(col_x[3], y, f"{v['R']:.0f}",   transform=ax_tbl.transAxes,
                color="#ff9f43", fontsize=9.5, va="top", fontweight="bold")

ax_tbl.set_title("Angle → d, L, R (beam path)", color="white",
                 fontsize=10, fontweight="bold", pad=6)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2 — Six individual to-scale diagrams (one per angle)
# ══════════════════════════════════════════════════════════════════════════════
gs2 = mgridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=gs[2, :], wspace=0.08)

for i, (v, col) in enumerate(zip(DATA, COLS)):
    ax_t = fig.add_subplot(gs2[i])
    ax_t.set_facecolor("#080808")
    for sp in ax_t.spines.values(): sp.set_edgecolor(col); sp.set_linewidth(2)
    ax_t.set_aspect("equal")
    ax_t.set_xticks([]); ax_t.set_yticks([])

    L = v["L"];  a = v["theta"]
    sc  = 1.0 / D
    Ln  = L * sc;  XSn = XS * sc

    # shaded area (no hypotenuse edge)
    ax_t.fill([0, Ln, 0], [0, 1, 1], color=col, alpha=0.08)

    # vertical leg d and horizontal leg L only — no hypotenuse
    ax_t.plot([0, 0],  [0, 1],  color=col, lw=2.5)
    ax_t.plot([0, Ln], [1, 1],  color=col, lw=2.5)

    # right-angle mark
    sq = 0.05
    ax_t.plot([sq, sq, 0], [1, 1 - sq, 1 - sq], color="#555555", lw=1.5)

    # beam path
    ax_t.plot([Ln, XSn], [1, 0], color=col, lw=1.8, ls="--", alpha=0.70, zorder=4)
    ax_t.plot(XSn, 0, "^", color="lime", ms=8, zorder=6)

    # R label — midpoint of beam, rotated to follow it
    beam_rot_t = np.degrees(np.arctan2(1, Ln - XSn))
    rx_t = Ln + 0.50 * (XSn - Ln)
    ry_t = 0.50
    ax_t.text(rx_t + 0.06, ry_t, f"R={v['R']:.0f}mm",
              ha="left", va="center", color=col, fontsize=7, fontweight="bold",
              rotation=beam_rot_t, rotation_mode="anchor", zorder=7,
              bbox=dict(fc="#000000dd", ec="none", pad=1))

    # camera
    ax_t.add_patch(mpatches.FancyBboxPatch((-0.09, 0.92), 0.18, 0.16,
        boxstyle="round,pad=0.01", fc="#0e0e2a", ec="#4444cc", lw=1.5, zorder=5))
    ax_t.text(0, 1.0, "Cam", ha="center", va="center",
              color="#8888ee", fontsize=7, fontweight="bold", zorder=6)

    # lamp
    ax_t.plot(Ln, 1, "o", color=col, ms=11, zorder=7,
              path_effects=[pe.withStroke(linewidth=3, foreground="#000000")])

    # specimen
    ax_t.fill_between([-0.05, Ln + 0.05], -0.07, 0, color="#0a1a0a", alpha=0.8)
    ax_t.plot([-0.05, Ln + 0.05], [0, 0], color="lime", lw=2)

    # labels — d on the left, L above
    ax_t.text(-0.10, 0.5, f"d\n{D:.0f}mm", ha="right", va="center",
              color="white", fontsize=8, fontweight="bold")
    ax_t.text(Ln / 2, 1.10, f"L = {L:.0f}mm",
              ha="center", color=col, fontsize=8, fontweight="bold")

    ax_t.set_xlim(-0.22, Ln + 0.25)
    ax_t.set_ylim(-0.12, 1.30)
    ax_t.set_title(f"θ = {a}°", color=col, fontsize=10, fontweight="bold")

fig.suptitle(
    f"Fixture geometry — lamp horizontal distance L  |  d = {D:.0f}mm (standoff, fixed)  |  "
    f"x_s = {XS:.0f}mm (beam offset)  |  L = {XS:.0f} + {D:.0f}·tan(θ)",
    color="white", fontsize=12, y=0.97)

fig.savefig(OUTPUT_PNG, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved: {OUTPUT_PNG}")
