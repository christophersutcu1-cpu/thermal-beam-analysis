"""
Sketch of the OPTIMUM Config B fixture at the 300 mm standoff.

Reads the Config B min-CoV optimum (06c_config_b_mincov.json) and draws the
buildable fixture: a fixed-45deg angled-arm V-trough, 2 bulbs per side, both
firing along parallel 45deg rays onto a point +-d1/2 off the FOV centreline.

Geometry (slant-distance convention, consistent with the optical model where the
beam-width model sigma(d) is evaluated at the bulb-to-target SLANT distance and
the 45deg incidence stretches the footprint by 1/cos45):

  aim point on target (per side)      : x = +- d1/2 ,  y = 0
  inner bulb slant distance to aim    : s_in  = d        (=300 mm)
  outer bulb slant distance to aim    : s_out = d + d2   (=540 mm)
  a bulb at slant s, 45deg yaw sits at : x = aim_x + s*sin45 ,  perp standoff = s*cos45
  => the two bulbs on a side lie on a 45deg rail, spaced d2 along it.

Two panels:
  (left)  plan view (top-down, x vs perpendicular standoff) — the fixture
  (right) front view (what the camera/target sees) — FOV box + beam centres
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Arc, Circle

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(REPO_ROOT, "06c_config_b_mincov.json")) as f:
    data = json.load(f)
best = next(r for r in data["results"] if r["standoff_mm"] == 300)

D      = best["standoff_mm"]          # 300 mm camera standoff
D1     = best["best_d1_mm"]           # 275 mm beam-centre baseline
D2     = best["best_d2_mm"]           # 240 mm along-arm spacing (slant)
FW, FH = best["fov_w_mm"], best["fov_h_mm"]
COV    = best["min_cov_pct"]
ANG    = 45.0
s45    = np.sin(np.radians(ANG)); c45 = np.cos(np.radians(ANG))

aim_x  = D1 / 2.0                      # 137.5
s_in, s_out = D, D + D2                # 300, 540 slant distances
# bulb positions (right side): (x, perpendicular standoff z)
in_x,  in_z  = aim_x + s_in  * s45, s_in  * c45
out_x, out_z = aim_x + s_out * s45, s_out * c45

# ---- figure ----------------------------------------------------------------
fig = plt.figure(figsize=(17, 8.4))
fig.patch.set_facecolor("#111111")
axP = fig.add_axes([0.055, 0.09, 0.56, 0.82])   # plan view
axF = fig.add_axes([0.68, 0.09, 0.30, 0.82])    # front view
for ax in (axP, axF):
    ax.set_facecolor("#0d0d0d")
    for sp in ax.spines.values(): sp.set_edgecolor("#444444")
    ax.tick_params(colors="#aaaaaa", labelsize=8)

LAMP = "#ffd11a"; RAY = "#7fd4ff"; FOVC = "#4dffb8"; DIMC = "#ff7b7b"; CAMC = "#cccccc"

# ============================ PLAN VIEW (top-down) ===========================
# axes: x horizontal, perpendicular standoff z vertical, target at z=0 on top
axP.set_title("PLAN VIEW (top-down)  —  optimum Config B fixture @ 300 mm standoff",
              color="white", fontsize=12, fontweight="bold", pad=12)

# target plane + FOV
axP.plot([-700, 700], [0, 0], color="#888888", lw=1.2)
axP.plot([-FW/2, FW/2], [0, 0], color=FOVC, lw=5, solid_capstyle="butt", zorder=4)
axP.text(0, -16, f"target plane  (FOV width {FW:.0f} mm)", color=FOVC,
         ha="center", va="top", fontsize=8.5)

# beam-centre / aim points (outside the FOV)
for sgn in (+1, -1):
    axP.plot(sgn*aim_x, 0, "x", color=DIMC, ms=11, mew=2.4, zorder=6)
axP.text(aim_x, 14, "beam centre\n(aim point)", color=DIMC, ha="center",
         va="bottom", fontsize=8)

# camera
axP.add_patch(Rectangle((-22, D-13), 44, 26, fc="#222222", ec=CAMC, lw=1.4, zorder=5))
axP.text(0, D+26, "Boson+ camera\n(standoff 300 mm)", color=CAMC, ha="center",
         va="bottom", fontsize=8)
axP.add_patch(FancyArrowPatch((0, D-13), (0, 6), arrowstyle="-|>",
              mutation_scale=13, color=CAMC, lw=1.0, ls=":", zorder=3))

# bulbs + rails + aim rays, both sides
for sgn in (+1, -1):
    bx_in,  bz_in  = sgn*in_x,  in_z
    bx_out, bz_out = sgn*out_x, out_z
    # 45deg rail (extended slightly past the bulbs)
    rx = np.array([sgn*(aim_x + 250*s45), sgn*(aim_x + 600*s45)])
    rz = np.array([250*c45, 600*c45])
    axP.plot(rx, rz, color="#666666", lw=6, alpha=0.5, solid_capstyle="round", zorder=2)
    # aim rays (bulb -> aim point), both at 45deg
    for bx, bz in [(bx_in, bz_in), (bx_out, bz_out)]:
        axP.add_patch(FancyArrowPatch((bx, bz), (sgn*aim_x, 0), arrowstyle="-|>",
                      mutation_scale=12, color=RAY, lw=1.3, ls="--", alpha=0.9, zorder=3))
    # bulbs
    for bx, bz, lab in [(bx_in, bz_in, "inner"), (bx_out, bz_out, "outer")]:
        axP.add_patch(Circle((bx, bz), 15, fc=LAMP, ec="white", lw=1.2, zorder=7))
        axP.text(bx + sgn*26, bz, f"{lab}\nbulb", color=LAMP, ha="left" if sgn>0 else "right",
                 va="center", fontsize=7.5)

# --- dimensions ---
# d1 between the two aim points (just below target line)
axP.annotate("", xy=(-aim_x, -42), xytext=(aim_x, -42),
             arrowprops=dict(arrowstyle="<->", color=DIMC, lw=1.4))
axP.text(0, -50, f"d1 = {D1:.0f} mm  (beam-centre baseline)", color=DIMC,
         ha="center", va="top", fontsize=9)
# d2 along the right rail
mx, mz = (in_x+out_x)/2, (in_z+out_z)/2
axP.annotate("", xy=(in_x, in_z), xytext=(out_x, out_z),
             arrowprops=dict(arrowstyle="<->", color="#ffd93d", lw=1.4))
axP.text(mx+30, mz, f"d2 = {D2:.0f} mm\n(along 45° arm)", color="#ffd93d",
         ha="left", va="center", fontsize=8.5)
# 45deg angle arc at inner-right bulb (between rail toward target and the vertical normal)
axP.add_patch(Arc((in_x, in_z), 120, 120, angle=0, theta1=180+45, theta2=270,
                  color="white", lw=1.3))
axP.text(in_x-58, in_z-44, "45°", color="white", fontsize=10, ha="center")
# standoff dimension (camera to target) on the centreline
axP.annotate("", xy=(0, 0), xytext=(0, D),
             arrowprops=dict(arrowstyle="<->", color="#9dd6ff", lw=1.1, alpha=0.7))
axP.text(-12, D/2, "300 mm", color="#9dd6ff", ha="right", va="center", fontsize=8, rotation=90)

axP.set_xlabel("x  [mm]", color="#aaaaaa", fontsize=9)
axP.set_ylabel("perpendicular standoff from target  [mm]", color="#aaaaaa", fontsize=9)
axP.set_xlim(-760, 760)
axP.set_ylim(420, -90)         # target at top, invert so rig is below
axP.set_aspect("equal")
axP.grid(True, alpha=0.12, color="white")

# ============================ FRONT VIEW ====================================
axF.set_title("FRONT VIEW (target plane,\nas the camera sees it)", color="white",
              fontsize=11, fontweight="bold", pad=10)
axF.add_patch(Rectangle((-FW/2, -FH/2), FW, FH, fill=False, ec=FOVC, lw=2, zorder=4))
axF.text(0, FH/2+6, f"FOV  {FW:.0f} × {FH:.0f} mm", color=FOVC, ha="center",
         va="bottom", fontsize=9)
axF.axhline(0, color="#555555", lw=0.8, ls=":")
for sgn in (+1, -1):
    axF.plot(sgn*aim_x, 0, "x", color=DIMC, ms=14, mew=3, zorder=6)
axF.text(aim_x, -14, "beam centre", color=DIMC, ha="center", va="top", fontsize=8)
gap = aim_x - FW/2
axF.annotate("", xy=(FW/2, FH/2*0.55), xytext=(aim_x, FH/2*0.55),
             arrowprops=dict(arrowstyle="<->", color="#ffd93d", lw=1.2))
axF.text((FW/2+aim_x)/2, FH/2*0.62, f"{gap:.0f} mm\noutside edge", color="#ffd93d",
         ha="center", va="bottom", fontsize=7.5)
axF.set_xlabel("x  [mm]", color="#aaaaaa", fontsize=9)
axF.set_ylabel("y  [mm]", color="#aaaaaa", fontsize=9)
axF.set_xlim(-aim_x-55, aim_x+55)
axF.set_ylim(-FH/2-45, FH/2+45)
axF.set_aspect("equal")
axF.grid(True, alpha=0.12, color="white")

# ---- spec note -------------------------------------------------------------
spec = (f"OPTIMUM @ 300 mm  ·  min CoV = {COV:.2f}%\n"
        f"arm angle 45°  ·  4 bulbs (2 per side)\n"
        f"d1 = {D1:.0f} mm   d2 = {D2:.0f} mm (along arm)\n"
        f"aim points:  x = ±{aim_x:.1f} mm , y = 0\n"
        f"inner bulb: ±{in_x:.0f} mm x, {in_z:.0f} mm standoff (slant {s_in:.0f})\n"
        f"outer bulb: ±{out_x:.0f} mm x, {out_z:.0f} mm standoff (slant {s_out:.0f})\n"
        f"overall rig width ≈ {2*out_x:.0f} mm")
fig.text(0.055, 0.965, spec, color="white", fontsize=8.6, family="monospace",
         va="top", ha="left",
         bbox=dict(boxstyle="round", fc="#1b1b1b", ec="#555555"))

OUT = os.path.join(REPO_ROOT, "06c_fixture_sketch_300mm.png")
fig.savefig(OUT, dpi=140, facecolor=fig.get_facecolor())
plt.close(fig)
print("Saved:", OUT)
print(f"inner bulb (right): x={in_x:.1f}  z={in_z:.1f}")
print(f"outer bulb (right): x={out_x:.1f}  z={out_z:.1f}")
print(f"aim points: x=±{aim_x:.1f}  ; rig width ≈ {2*out_x:.0f} mm")
