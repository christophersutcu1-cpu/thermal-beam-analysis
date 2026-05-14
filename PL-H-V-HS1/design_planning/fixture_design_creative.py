"""
Creative fixture design analysis.
Fixed offset x_s = 105mm (Config 1 best).  Standoff d = 383mm.

L(theta) = 105 + 383*tan(theta)   lateral lamp distance
H(theta) = sqrt(d^2 + L^2)        fixture arm (hypotenuse)
R(theta) = d / cos(theta)          beam path length

Three creative views:
  1. Fan of overlaid triangles — shows how fixture expands with angle
  2. Top-down plan view with H-range rings — shows WHERE the lamp lives
  3. Design trade-off: H (size you pay) vs L, R, sx_surface (what you get)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import matplotlib.patches as mpatches
from matplotlib.patches import Arc, FancyArrow, Circle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patheffects as pe

OUTPUT_PNG = config.BOSON_ROOT + r"\fixture_design_creative.png"

D    = 383.0   # mm standoff
XS   = 105.0   # mm beam offset (Config 1 optimum)
SX_F = 58.8    # mm sigma_x free-space
SY_F = 108.1   # mm sigma_y free-space

theta_deg = np.linspace(5, 72, 600)
theta_rad = np.radians(theta_deg)

L_arr  = XS + D * np.tan(theta_rad)
H_arr  = np.sqrt(D**2 + L_arr**2)
R_arr  = D / np.cos(theta_rad)
SX_arr = SX_F / np.cos(theta_rad)   # beam broadens in X at angle

# key angles to highlight
KEY = [20, 30, 45, 60]
cmap_key = plt.cm.plasma
norm_key = Normalize(vmin=10, vmax=70)

def key_col(a): return cmap_key(norm_key(a))

def kv(a):
    tr = np.radians(a)
    Lv = XS + D*np.tan(tr)
    return dict(theta=a, L=Lv, H=np.sqrt(D**2+Lv**2),
                R=D/np.cos(tr), sx=SX_F/np.cos(tr))

KVDATA = {a: kv(a) for a in KEY}

# ── figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 18))
fig.patch.set_facecolor("#0d0d0d")
gs  = mgridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.30,
                          left=0.05, right=0.97, top=0.93, bottom=0.05,
                          height_ratios=[1.15, 1])

def dark_ax(ax):
    ax.set_facecolor("#111111")
    for sp in ax.spines.values(): sp.set_edgecolor("#333333")
    ax.tick_params(colors="#888888", labelsize=9)
    ax.grid(True, alpha=0.10, color="white", lw=0.5)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL A — Fan of triangles (side view)  [row 0, col 0]
# ══════════════════════════════════════════════════════════════════════════════
ax_fan = fig.add_subplot(gs[0, 0])
ax_fan.set_facecolor("#080808")
for sp in ax_fan.spines.values(): sp.set_edgecolor("#222222")
ax_fan.tick_params(colors="#666666", labelsize=8)

# draw triangles for a range of angles, colour-mapped
fan_angles = np.arange(15, 72, 3)
cmap_fan   = plt.cm.coolwarm
norm_fan   = Normalize(vmin=fan_angles.min(), vmax=fan_angles.max())

for a in fan_angles:
    v  = kv(a)
    c  = cmap_fan(norm_fan(a))
    Lv = v["L"]
    # triangle: right-angle at camera (0, D), base at (0,0), lamp at (Lv, D)
    ax_fan.plot([0, Lv], [D, D], color=c, lw=1.2, alpha=0.7)          # L (horizontal)
    ax_fan.plot([0, Lv], [D, 0], color=c, lw=1.5, alpha=0.8)          # H (hypotenuse)
    ax_fan.plot(Lv, D, "o", color=c, ms=4, zorder=5)

# specimen
ax_fan.fill_between([-20, max(kv(a)["L"] for a in fan_angles)+30],
                    -18, 0, color="#0a1a0a", alpha=0.9)
ax_fan.plot([-20, max(kv(a)["L"] for a in fan_angles)+30], [0,0],
            color="lime", lw=2)
ax_fan.text(XS, -10, f"x_s={XS:.0f}mm", ha="center",
            color="lime", fontsize=8, fontweight="bold")

# camera
ax_fan.add_patch(mpatches.FancyBboxPatch((-22, D-18), 44, 36,
    boxstyle="round,pad=3", fc="#0e0e2a", ec="#3a3acc", lw=2, zorder=8))
ax_fan.text(0, D, "Camera", ha="center", va="center",
            color="#7777ee", fontsize=8, fontweight="bold", zorder=9)

# vertical axis (d)
ax_fan.plot([0, 0], [0, D-18], color="#333333", lw=1, ls="--")
ax_fan.annotate("", xy=(-35, 0), xytext=(-35, D),
                arrowprops=dict(arrowstyle="<->", color="#888888", lw=1.2))
ax_fan.text(-42, D/2, f"d={D:.0f}mm", ha="right", va="center",
            color="#aaaaaa", fontsize=8)

# highlight 4 key angles
for a in KEY:
    v  = kv(a)
    c  = key_col(a)
    ax_fan.plot([0, v["L"]], [D, D], color=c, lw=2.5, zorder=6)
    ax_fan.plot([0, v["L"]], [D, 0], color=c, lw=2.5, ls="--", zorder=6)
    ax_fan.plot(v["L"], D, "o", color=c, ms=10, zorder=10,
                path_effects=[pe.withStroke(linewidth=3, foreground="#000000")])
    ax_fan.text(v["L"]+6, D+8,
                f"{a}deg\nL={v['L']:.0f}\nH={v['H']:.0f}",
                color=c, fontsize=7.5, fontweight="bold", va="bottom")
    # beam line to hit point
    ax_fan.plot([v["L"], XS], [D, 0], color=c, lw=1.2, ls=":", alpha=0.6, zorder=5)
    ax_fan.plot(XS, 0, "^", color=c, ms=7, zorder=7)

# colourbar
sm = ScalarMappable(cmap=cmap_fan, norm=norm_fan)
sm.set_array([])
cb = fig.colorbar(sm, ax=ax_fan, fraction=0.035, pad=0.02)
cb.set_label("Beam angle theta (deg)", color="#888888", fontsize=8)
cb.ax.yaxis.set_tick_params(color="#888888")
plt.setp(cb.ax.yaxis.get_ticklabels(), color="#888888")

ax_fan.set_xlim(-55, max(kv(a)["L"] for a in KEY)+80)
ax_fan.set_ylim(-25, D+60)
ax_fan.set_aspect("equal")
ax_fan.set_xlabel("Horizontal from camera axis (mm)", color="#888888", fontsize=9)
ax_fan.set_ylabel("Height above specimen (mm)", color="#888888", fontsize=9)
ax_fan.set_title("Fan of fixture triangles — side view\nEach line = one beam angle, all at x_s=105mm offset",
                 color="white", fontsize=10)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL B — Top-down plan view with H-range rings  [row 0, col 1-2]
# ══════════════════════════════════════════════════════════════════════════════
ax_plan = fig.add_subplot(gs[0, 1:3])
ax_plan.set_facecolor("#080808")
for sp in ax_plan.spines.values(): sp.set_edgecolor("#222222")
ax_plan.tick_params(colors="#666666", labelsize=9)
ax_plan.set_aspect("equal")

# H-range rings (in plan view: ring radius r = sqrt(H^2 - d^2))
H_rings = [400, 450, 500, 550, 600, 650, 700, 750, 800]
for H_r in H_rings:
    r = np.sqrt(max(H_r**2 - D**2, 0))
    ring = Circle((0, 0), r, fc="none", ec="#222222", lw=0.8, ls="--", zorder=1)
    ax_plan.add_patch(ring)
    ax_plan.text(r + 6, 6, f"H={H_r:.0f}", color="#444444", fontsize=7,
                 va="bottom", ha="left")

# specimen rectangle
spec_w = 320; spec_h = 175
ax_plan.add_patch(mpatches.Rectangle((-spec_w/2, -spec_h/2), spec_w, spec_h,
    fc="#0a1a0a", ec="lime", lw=2.5, zorder=3))
ax_plan.text(0, 0, f"Specimen\n{spec_w}×{spec_h}mm",
             ha="center", va="center", color="lime", fontsize=9,
             fontweight="bold", zorder=4)

# camera at origin
ax_plan.plot(0, 0, "+", color="#5555ff", ms=18, mew=2.5, zorder=8)
ax_plan.add_patch(Circle((0, 0), 12, fc="#0e0e2a", ec="#3a3acc", lw=2, zorder=7))
ax_plan.text(0, 14, "Camera", ha="center", color="#7777ee", fontsize=8,
             fontweight="bold", zorder=9)

# beam hit points (symmetric, ±105mm)
for sign in [+1, -1]:
    ax_plan.plot(sign*XS, 0, "D", color="lime", ms=8, zorder=8,
                 path_effects=[pe.withStroke(linewidth=2, foreground="#000000")])

ax_plan.annotate("", xy=(XS, -spec_h/2-18), xytext=(-XS, -spec_h/2-18),
                 arrowprops=dict(arrowstyle="<->", color="lime", lw=1.5))
ax_plan.text(0, -spec_h/2-30, f"2 × x_s = {2*XS:.0f}mm beam spacing",
             ha="center", color="lime", fontsize=8.5)

# lamp positions at key angles (RIGHT side, +x direction)
# In plan view lamp appears at (L, 0) — same y as camera, x=L
for a in KEY:
    v   = kv(a)
    c   = key_col(a)
    Lv  = v["L"]
    # right lamp
    ax_plan.plot(Lv, 0, "o", color=c, ms=12, zorder=10,
                 path_effects=[pe.withStroke(linewidth=3, foreground="#000000")])
    ax_plan.text(Lv+8, 12, f"theta={a}deg\nL={Lv:.0f}mm\nH={v['H']:.0f}mm",
                 color=c, fontsize=7.5, fontweight="bold", va="bottom")
    # line: lamp → beam hit point on specimen (+x side)
    ax_plan.plot([Lv, XS], [0, 0], color=c, lw=1.5, ls="--", alpha=0.7, zorder=5)
    # bracket showing L
    ax_plan.annotate("", xy=(Lv, -spec_h/2-52 - KEY.index(a)*18),
                     xytext=(0, -spec_h/2-52 - KEY.index(a)*18),
                     arrowprops=dict(arrowstyle="<->", color=c, lw=1.2))
    ax_plan.text(Lv/2, -spec_h/2-50 - KEY.index(a)*18,
                 f"L={Lv:.0f}mm", ha="center", color=c, fontsize=7.5)

    # left lamp (mirror)
    ax_plan.plot(-Lv, 0, "o", color=c, ms=12, zorder=10,
                 path_effects=[pe.withStroke(linewidth=3, foreground="#000000")])
    ax_plan.plot([-Lv, -XS], [0, 0], color=c, lw=1.5, ls="--", alpha=0.7, zorder=5)

# axis arrows
ax_fan_max_L = max(kv(a)["L"] for a in KEY)
ax_plan.set_xlim(-ax_fan_max_L - 60, ax_fan_max_L + 200)
ax_plan.set_ylim(-spec_h/2 - 130, spec_h/2 + 60)
ax_plan.set_xlabel("Horizontal X from camera axis (mm) — lamps along this axis", color="#888888", fontsize=9)
ax_plan.set_ylabel("Horizontal Y (mm)", color="#888888", fontsize=9)
ax_plan.set_title(
    "Top-down plan view — where does the lamp live?\n"
    "Dashed rings = constant fixture-arm length H  |  Dots = lamp position at each angle",
    color="white", fontsize=11)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL C — H, L, R vs theta with beam width  [row 1, col 0-1]
# ══════════════════════════════════════════════════════════════════════════════
ax_dist = fig.add_subplot(gs[1, 0:2])
dark_ax(ax_dist)

ax_dist.plot(theta_deg, H_arr,  color="#ffd93d", lw=3.0,
             label="H = fixture arm (hypotenuse)")
ax_dist.plot(theta_deg, L_arr,  color="#4d96ff", lw=2.5,
             label="L = lateral distance (horizontal leg)")
ax_dist.plot(theta_deg, R_arr,  color="#ff9f43", lw=2.0, ls="--",
             label="R = beam path  d/cos(theta)")
ax_dist.axhline(D, color="white", lw=1.2, ls=":", alpha=0.5,
                label=f"d = {D:.0f}mm (standoff, fixed)")

# secondary axis: beam sigma_x on specimen
ax2 = ax_dist.twinx()
ax2.plot(theta_deg, SX_arr, color="#ff6b6b", lw=2.0, ls="-.",
         label="sigma_x on specimen  (broadens with angle)")
ax2.axhline(spec_w/2, color="#ff6b6b", lw=0.8, ls=":", alpha=0.5)
ax2.set_ylabel("Beam sigma_x on specimen (mm)", color="#ff6b6b", fontsize=9)
ax2.tick_params(colors="#ff6b6b", labelsize=8)
ax2.set_ylim(0, 500)

# key angle markers
for a in KEY:
    v = kv(a); c = key_col(a)
    for arr, val in [(H_arr, v["H"]), (L_arr, v["L"])]:
        ax_dist.plot(a, val, "o", color=c, ms=9, zorder=7,
                     path_effects=[pe.withStroke(linewidth=2, foreground="#000000")])
    ax_dist.axvline(a, color=c, lw=1.2, ls="--", alpha=0.5)
    ax_dist.text(a+0.5, v["H"]+15,
                 f"{a}deg\nH={v['H']:.0f}\nL={v['L']:.0f}",
                 color=c, fontsize=7.5, fontweight="bold")

ax_dist.set_xlabel("Beam angle theta (degrees)", color="#888888", fontsize=10)
ax_dist.set_ylabel("Distance (mm)", color="#888888", fontsize=10)
ax_dist.set_title(
    f"All distances vs beam angle  |  x_s={XS:.0f}mm fixed\n"
    "H grows faster than L — the hypotenuse penalty of tilting the lamp",
    color="white", fontsize=10)
l1, b1 = ax_dist.get_legend_handles_labels()
l2, b2 = ax2.get_legend_handles_labels()
ax_dist.legend(l1+l2, b1+b2, fontsize=8.5, facecolor="#1a1a1a",
               labelcolor="white", edgecolor="#333333", loc="upper left")
ax_dist.set_xlim(5, 72)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL D — H vs useful gain (beam coverage on specimen)  [row 1, col 2]
# ══════════════════════════════════════════════════════════════════════════════
ax_trade = fig.add_subplot(gs[1, 2])
dark_ax(ax_trade)

# colour the H vs sx curve by angle
points = np.array([H_arr, SX_arr]).T.reshape(-1, 1, 2)
segs   = np.concatenate([points[:-1], points[1:]], axis=1)

from matplotlib.collections import LineCollection
lc = LineCollection(segs, cmap="plasma",
                    norm=Normalize(vmin=theta_deg.min(), vmax=theta_deg.max()),
                    linewidth=3)
lc.set_array(theta_deg[:-1])
ax_trade.add_collection(lc)
fig.colorbar(lc, ax=ax_trade, label="Beam angle theta (deg)").ax.yaxis.set_tick_params(color="white")

# specimen half-width reference
ax_trade.axvline(spec_w/2, color="lime", lw=1.5, ls="--", alpha=0.7,
                 label=f"Specimen half-width {spec_w//2}mm")
ax_trade.axhline(spec_w/2, color="lime", lw=0.8, ls=":", alpha=0.4)

# mark key angles
for a in KEY:
    v = kv(a); c = key_col(a)
    ax_trade.plot(v["H"], v["sx"], "o", color=c, ms=11, zorder=8,
                  path_effects=[pe.withStroke(linewidth=3, foreground="#000000")])
    ax_trade.annotate(f"{a}deg\nH={v['H']:.0f}mm\nsx={v['sx']:.0f}mm",
                      xy=(v["H"], v["sx"]),
                      xytext=(v["H"]+15, v["sx"]-15),
                      color=c, fontsize=7.5, fontweight="bold",
                      arrowprops=dict(arrowstyle="->", color=c, lw=1))

ax_trade.set_xlabel("H — fixture arm length (mm)", color="#888888", fontsize=10)
ax_trade.set_ylabel("sigma_x on specimen (mm) — beam coverage", color="#888888", fontsize=10)
ax_trade.set_title(
    "Design trade-off: fixture size vs beam coverage\n"
    "Larger H = farther lamp = wider beam on specimen",
    color="white", fontsize=10)
ax_trade.legend(fontsize=8.5, facecolor="#1a1a1a",
                labelcolor="white", edgecolor="#333333")
ax_trade.set_xlim(H_arr.min()-20, H_arr.max()+30)
ax_trade.set_ylim(SX_arr.min()-10, min(SX_arr.max(), 400))

fig.suptitle(
    f"Fixture design — creative analysis  |  x_s = {XS:.0f}mm (Config 1 optimum)  |  d = {D:.0f}mm\n"
    "L = 105 + 383·tan(theta)   |   H = sqrt(d²+L²)   |   R = d/cos(theta)",
    color="white", fontsize=13, y=0.97)

fig.savefig(OUTPUT_PNG, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved: {OUTPUT_PNG}")
