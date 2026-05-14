"""
Two-beam irradiance simulation on specimen plane.

Uses measured single-beam parameters (from summary.json) to simulate two
symmetric beams at +-theta hitting a specimen. The key variable is the
beam OFFSET on the specimen surface: each beam can be aimed to land its
peak at a different point, so that the two overlapping Gaussians create
a flat-top irradiance profile.

Optimal offset for two Gaussians: x_offset ~ sigma_x_surface
(each beam peak at +-sigma from centre gives the flattest combined profile)

Usage:
    python two_beam_simulation.py
"""

import os
import config
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, Ellipse

# -- Config -------------------------------------------------------------------
CAPTURES_ROOT  = config.BOSON_ROOT + r""
OUTPUT_PNG     = os.path.join(CAPTURES_ROOT, "two_beam_simulation.png")
BEST_SESSION   = os.path.join(CAPTURES_ROOT, "700 mm", "summary.json")

SPEC_W_MM = 320.0
SPEC_H_MM = 175.0
GRID_MM   = 1.0

# Beam tilt axis: "x" = beams sweep left/right across specimen width
TILT_AXIS = "x"

# Pixel mm for Boson 640 18mm lens at 700mm standoff
PIXEL_MM  = 2 * 700 * np.tan(np.radians(24.09 / 2)) / 640

ANGLES_DEG     = [30, 40, 45, 50, 60]
DETAIL_ANGLES  = [30, 45, 60]


# -- Helpers ------------------------------------------------------------------
def load_beam(json_path):
    with open(json_path) as f:
        d = json.load(f)
    b = d["beam"]
    return b["sigma_x"] * PIXEL_MM, b["sigma_y"] * PIXEL_MM, b["amplitude"]


def surface_sigma(sigma_x_free, sigma_y_free, theta_rad, tilt_axis):
    """Project free-space beam sigma onto specimen surface."""
    if tilt_axis == "x":
        return sigma_x_free / np.cos(theta_rad), sigma_y_free
    else:
        return sigma_x_free, sigma_y_free / np.cos(theta_rad)


def two_beam_irradiance(sigma_x_free, sigma_y_free, peak, theta_deg,
                        offset_x=0.0, offset_y=0.0, tilt_axis=TILT_AXIS):
    """
    Simulate combined irradiance of two symmetric beams on specimen plane.

    offset_x: how far each beam peak is offset from specimen centre in x (mm).
              Beam 1 at +offset_x, Beam 2 at -offset_x.
    offset_y: same for y axis.
    """
    theta_rad = np.radians(theta_deg)
    sx_s, sy_s = surface_sigma(sigma_x_free, sigma_y_free, theta_rad, tilt_axis)
    surf_peak  = peak * np.cos(theta_rad)

    pad  = max(SPEC_W_MM, SPEC_H_MM)
    x_arr = np.arange(-(SPEC_W_MM/2 + pad), (SPEC_W_MM/2 + pad), GRID_MM)
    y_arr = np.arange(-(SPEC_H_MM/2 + pad), (SPEC_H_MM/2 + pad), GRID_MM)
    gx, gy = np.meshgrid(x_arr, y_arr)

    def gaussian(cx, cy):
        return surf_peak * np.exp(-0.5 * (((gx-cx)/sx_s)**2 + ((gy-cy)/sy_s)**2))

    irr  = gaussian(+offset_x, +offset_y) + gaussian(-offset_x, -offset_y)
    mask = (np.abs(gx) <= SPEC_W_MM/2) & (np.abs(gy) <= SPEC_H_MM/2)
    return irr, x_arr, y_arr, mask, sx_s, sy_s


def uniformity(irr, mask):
    vals = irr[mask]
    if vals.size == 0 or vals.mean() == 0:
        return np.nan, np.nan
    return float(vals.std() / vals.mean() * 100), \
           float((vals.max() - vals.min()) / vals.mean() * 100)


def optimal_offset(sigma_x_free, sigma_y_free, peak, theta_deg,
                   tilt_axis=TILT_AXIS, n=100):
    """Find beam offset that minimises CoV over the specimen."""
    theta_rad = np.radians(theta_deg)
    sx_s, _ = surface_sigma(sigma_x_free, sigma_y_free, theta_rad, tilt_axis)
    offsets  = np.linspace(0, sx_s * 2.0, n)
    best_cov, best_off = 1e9, 0.0
    for off in offsets:
        irr, _, _, mask, _, _ = two_beam_irradiance(
            sigma_x_free, sigma_y_free, peak, theta_deg,
            offset_x=off, tilt_axis=tilt_axis)
        cov, _ = uniformity(irr, mask)
        if cov < best_cov:
            best_cov, best_off = cov, off
    return best_off, best_cov


# -- Main ---------------------------------------------------------------------
def main():
    sx_free, sy_free, peak = load_beam(BEST_SESSION)
    print(f"Beam (700mm perpendicular):  sigma_x={sx_free:.1f}mm  sigma_y={sy_free:.1f}mm")
    print(f"Specimen: {SPEC_W_MM:.0f} x {SPEC_H_MM:.0f} mm\n")

    # -- Find optimal offset at each angle -----------------------------------
    print(f"{'Angle':>6} | {'sx_surf':>8} | {'sy_surf':>8} | "
          f"{'Opt offset':>10} | {'Best CoV':>9} | {'CoV@0':>8}")
    print("-" * 65)
    angle_results = []
    for theta in ANGLES_DEG:
        opt_off, best_cov = optimal_offset(sx_free, sy_free, peak, theta)
        irr0, _, _, mask, sx_s, sy_s = two_beam_irradiance(
            sx_free, sy_free, peak, theta, offset_x=0)
        cov0, _ = uniformity(irr0, mask)
        irr_opt, xa, ya, mask, sx_s, sy_s = two_beam_irradiance(
            sx_free, sy_free, peak, theta, offset_x=opt_off)
        cov_opt, p2v_opt = uniformity(irr_opt, mask)
        angle_results.append(dict(
            theta=theta, sx_s=sx_s, sy_s=sy_s,
            opt_off=opt_off, best_cov=cov_opt, p2v=p2v_opt,
            cov0=cov0, irr_opt=irr_opt, xa=xa, ya=ya, mask=mask))
        print(f"  {theta:4.0f}  | {sx_s:7.1f}  | {sy_s:7.1f}  | "
              f"{opt_off:9.1f}mm | {cov_opt:8.1f}%  | {cov0:7.1f}%")

    best = min(angle_results, key=lambda r: r["best_cov"])
    print(f"\nBest: theta={best['theta']}°  offset={best['opt_off']:.1f}mm  "
          f"CoV={best['best_cov']:.1f}%  P2V={best['p2v']:.1f}%")

    # -- Offset sweep at 45 deg to illustrate the effect ---------------------
    theta_demo = 45
    sx_demo, sy_demo = surface_sigma(sx_free, sy_free,
                                     np.radians(theta_demo), TILT_AXIS)
    offsets_demo = np.linspace(0, sx_demo * 2, 60)
    covs_demo    = []
    for off in offsets_demo:
        irr, _, _, mask, _, _ = two_beam_irradiance(
            sx_free, sy_free, peak, theta_demo, offset_x=off)
        c, _ = uniformity(irr, mask)
        covs_demo.append(c)

    # =========================================================================
    # FIGURE
    # Row 0: irradiance maps at detail angles — centred vs optimal offset
    # Row 1: H cross-sections at detail angles (centred vs optimal)
    # Row 2: CoV vs angle (opt vs centred) | CoV vs offset (45deg) | table
    # =========================================================================
    detail_res = [r for r in angle_results if r["theta"] in DETAIL_ANGLES]
    vmax_g = max(r["irr_opt"].max() for r in detail_res)

    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor("#111111")
    fig.suptitle(
        f"Two-Beam Simulation  |  Specimen {SPEC_W_MM:.0f}x{SPEC_H_MM:.0f}mm  "
        f"|  Tilt axis: {TILT_AXIS.upper()}  |  Beams symmetric at +-theta",
        fontsize=13, color="white", y=0.99)

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.35,
                           left=0.05, right=0.97, top=0.95, bottom=0.04)

    def dark_ax(ax):
        ax.set_facecolor("#1a1a1a")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444444")
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        ax.xaxis.label.set_color("#aaaaaa")
        ax.yaxis.label.set_color("#aaaaaa")
        ax.title.set_color("white")
        return ax

    # ----- Row 0: irradiance maps at optimal offset -------------------------
    for col, res in enumerate(detail_res):
        ax = dark_ax(fig.add_subplot(gs[0, col]))
        ext = [res["xa"].min(), res["xa"].max(),
               res["ya"].min(), res["ya"].max()]
        im  = ax.imshow(res["irr_opt"], extent=ext, cmap="inferno",
                        origin="lower", vmin=0, vmax=vmax_g, aspect="equal")
        plt.colorbar(im, ax=ax, fraction=0.04).ax.yaxis.set_tick_params(
            color="#aaaaaa", labelsize=6)

        # specimen
        ax.add_patch(Rectangle((-SPEC_W_MM/2, -SPEC_H_MM/2),
                                SPEC_W_MM, SPEC_H_MM,
                                edgecolor="lime", facecolor="none",
                                lw=2, ls="--", zorder=5))
        # beam 1 FWHM ellipse
        for sign in [+1, -1]:
            ax.add_patch(Ellipse(
                (sign * res["opt_off"], 0),
                res["sx_s"] * 2.355, res["sy_s"] * 2.355,
                edgecolor="#00cfff", facecolor="none", lw=1.5, ls=":", zorder=5))

        ax.set_xlim(-SPEC_W_MM*0.85, SPEC_W_MM*0.85)
        ax.set_ylim(-SPEC_H_MM*1.3,  SPEC_H_MM*1.3)
        ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
        ax.set_title(f"theta={res['theta']}°  offset={res['opt_off']:.0f}mm\n"
                     f"CoV={res['best_cov']:.1f}%   P2V={res['p2v']:.0f}%",
                     fontsize=9)

    # ----- Row 0, col 3: V cross-section at optimal offset for all angles ---
    ax_vc = dark_ax(fig.add_subplot(gs[0, 3]))
    palette = plt.cm.plasma(np.linspace(0.1, 0.9, len(angle_results)))
    for res, col_c in zip(angle_results, palette):
        cx_idx = np.argmin(np.abs(res["xa"]))
        ax_vc.plot(res["irr_opt"][:, cx_idx], res["ya"],
                   color=col_c, lw=1.5, label=f"{res['theta']}°")
    ax_vc.axhline( SPEC_H_MM/2, color="lime", lw=0.8, ls=":")
    ax_vc.axhline(-SPEC_H_MM/2, color="lime", lw=0.8, ls=":")
    ax_vc.set_xlabel("Irradiance (a.u.)"); ax_vc.set_ylabel("Y (mm)")
    ax_vc.set_title("Vertical cross-section\n(optimal offset, all angles)")
    ax_vc.legend(fontsize=7, facecolor="#222222", labelcolor="white",
                 edgecolor="#555555", title="Angle", title_fontsize=7, ncol=2)
    ax_vc.grid(True, alpha=0.12, color="white")

    # ----- Row 1: H cross-sections centred vs optimal offset ----------------
    h_colors = ["#ff6b6b", "#ffd93d", "#4d96ff"]
    for col, (res, hc) in enumerate(zip(detail_res, h_colors)):
        ax_h = dark_ax(fig.add_subplot(gs[1, col]))
        cy_idx = np.argmin(np.abs(res["ya"]))
        # centred (offset=0)
        irr0, xa, ya, mask, _, _ = two_beam_irradiance(
            sx_free, sy_free, peak, res["theta"], offset_x=0)
        ax_h.plot(xa, irr0[cy_idx, :], color=hc, lw=1.5, ls="--",
                  alpha=0.6, label="Centred (offset=0)")
        # optimal offset
        ax_h.plot(res["xa"], res["irr_opt"][cy_idx, :], color=hc, lw=2,
                  label=f"Optimal offset={res['opt_off']:.0f}mm")
        ax_h.axvline(-SPEC_W_MM/2, color="lime", lw=1, ls=":")
        ax_h.axvline( SPEC_W_MM/2, color="lime", lw=1, ls=":",
                     label="Specimen edges")
        # mark beam peak positions
        ax_h.axvline( res["opt_off"], color="white", lw=0.7, ls=":", alpha=0.5)
        ax_h.axvline(-res["opt_off"], color="white", lw=0.7, ls=":", alpha=0.5)
        ax_h.set_xlabel("X (mm)"); ax_h.set_ylabel("Irradiance (a.u.)")
        ax_h.set_title(f"H cross-section  theta={res['theta']}°\n"
                       f"CoV: {res['cov0']:.1f}% -> {res['best_cov']:.1f}% with offset",
                       fontsize=9)
        ax_h.set_xlim(-SPEC_W_MM*0.85, SPEC_W_MM*0.85)
        ax_h.legend(fontsize=7, facecolor="#222222", labelcolor="white",
                    edgecolor="#555555")
        ax_h.grid(True, alpha=0.12, color="white")

    # ----- Row 1, col 3: CoV vs offset at 45 deg ----------------------------
    ax_off = dark_ax(fig.add_subplot(gs[1, 3]))
    opt45 = next(r for r in angle_results if r["theta"] == 45)
    ax_off.plot(offsets_demo, covs_demo, color="#00cfff", lw=2)
    ax_off.axvline(opt45["opt_off"], color="lime", lw=1, ls="--",
                   label=f"Optimal = {opt45['opt_off']:.0f}mm")
    ax_off.axvline(opt45["sx_s"], color="#ffaa00", lw=1, ls=":",
                   label=f"1-sigma = {opt45['sx_s']:.0f}mm")
    ax_off.set_xlabel("Beam offset from centre (mm)")
    ax_off.set_ylabel("CoV on specimen (%)")
    ax_off.set_title("CoV vs beam offset\n(theta=45°, vary each beam's x-centre)")
    ax_off.legend(fontsize=8, facecolor="#222222", labelcolor="white",
                  edgecolor="#555555")
    ax_off.grid(True, alpha=0.12, color="white")

    # ----- Row 2, col 0-1: CoV vs angle (centred vs optimal) ----------------
    ax_uni = dark_ax(fig.add_subplot(gs[2, :2]))
    angles  = [r["theta"]    for r in angle_results]
    cov_opt = [r["best_cov"] for r in angle_results]
    cov_ctr = [r["cov0"]     for r in angle_results]
    ax_uni.plot(angles, cov_ctr, color="#ff6b6b", lw=2, marker="o", ms=7,
                label="Centred beams (offset=0)")
    ax_uni.plot(angles, cov_opt, color="#00cfff", lw=2, marker="o", ms=7,
                label="Optimal offset")
    ax_uni.axhline(15, color="lime", lw=0.8, ls=":", label="15% target")
    ax_uni.axvline(best["theta"], color="lime", lw=1, ls="--",
                   label=f"Best: {best['theta']}°  CoV={best['best_cov']:.1f}%")
    ax_uni.fill_between(angles, cov_ctr, cov_opt, alpha=0.12, color="#00cfff",
                        label="Gain from offset")
    ax_uni.set_xlabel("Beam angle from normal (deg)")
    ax_uni.set_ylabel("CoV on specimen (%)")
    ax_uni.set_title("Uniformity vs angle  (blue = with optimal beam offset)")
    ax_uni.legend(fontsize=8, facecolor="#222222", labelcolor="white",
                  edgecolor="#555555")
    ax_uni.grid(True, alpha=0.12, color="white")

    # ----- Row 2, col 2: optimal offset vs angle ----------------------------
    ax_oa = dark_ax(fig.add_subplot(gs[2, 2]))
    opt_offs = [r["opt_off"] for r in angle_results]
    sx_surfs = [r["sx_s"]    for r in angle_results]
    ax_oa.plot(angles, opt_offs, color="#00cfff", lw=2, marker="o", ms=7,
               label="Optimal offset")
    ax_oa.plot(angles, sx_surfs, color="#ffaa00", lw=2, marker="s", ms=7,
               ls="--", label="1-sigma surface")
    ax_oa.set_xlabel("Beam angle from normal (deg)")
    ax_oa.set_ylabel("Distance (mm)")
    ax_oa.set_title("Optimal beam offset vs angle\n(optimal ~ 1-sigma on surface)")
    ax_oa.legend(fontsize=8, facecolor="#222222", labelcolor="white",
                 edgecolor="#555555")
    ax_oa.grid(True, alpha=0.12, color="white")

    # ----- Row 2, col 3: summary table --------------------------------------
    ax_tab = fig.add_subplot(gs[2, 3])
    ax_tab.set_facecolor("#111111"); ax_tab.axis("off")
    hdr   = ["Angle", "sx surf", "Opt off", "CoV", "P2V"]
    rows  = [[f"{r['theta']}",
              f"{r['sx_s']:.0f}mm",
              f"{r['opt_off']:.0f}mm",
              f"{r['best_cov']:.1f}%",
              f"{r['p2v']:.0f}%"] for r in angle_results]
    tab   = ax_tab.table(cellText=rows, colLabels=hdr,
                         cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tab.auto_set_font_size(False); tab.set_fontsize(9)
    for j in range(5):
        tab[0, j].set_facecolor("#222244")
        tab[0, j].set_text_props(color="white", fontweight="bold")
    for i, res in enumerate(angle_results, 1):
        is_best = res["theta"] == best["theta"]
        for j in range(5):
            tab[i, j].set_facecolor("#1e2e1e" if is_best else
                                    ("#1a1a1a" if i % 2 == 0 else "#222222"))
            tab[i, j].set_text_props(color="lime" if is_best else "white")
            tab[i, j].set_edgecolor("#333333")
    ax_tab.set_title("Optimal offset results\n(green = best CoV)",
                     color="white", pad=6, fontsize=9)

    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nFigure saved: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
