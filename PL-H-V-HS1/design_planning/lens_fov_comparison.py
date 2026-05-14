"""
Fixed-standoff lens comparison: 9.2 mm vs 18 mm at 383 mm standoff.
Shows sensor fill, resolution, and what is visible of the specimen.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe

OUTPUT_PNG  = config.BOSON_ROOT + r"\lens_fov_comparison.png"

SENSOR_W_PX = 640
SENSOR_H_PX = 512
PIXEL_PITCH = 0.012      # mm  (Boson 640, 12 µm pitch)
SENSOR_W_MM = SENSOR_W_PX * PIXEL_PITCH   # 7.68 mm
SENSOR_H_MM = SENSOR_H_PX * PIXEL_PITCH   # 6.144 mm

STANDOFF    = 383.0      # mm  — fixed
SPEC_W      = 320.0      # mm
SPEC_H      = 175.0      # mm

LENSES = [
    {"f": 9.2,  "label": "9.2 mm",  "col": "#ffd93d"},
    {"f": 18.0, "label": "18 mm",   "col": "#4d96ff"},
]

# ── derived quantities ────────────────────────────────────────────────────────
for L in LENSES:
    f = L["f"]
    mpp          = STANDOFF * PIXEL_PITCH / f        # mm per pixel
    fov_w        = SENSOR_W_PX * mpp                  # mm
    fov_h        = SENSOR_H_PX * mpp
    spec_px_w    = SPEC_W / mpp
    spec_px_h    = SPEC_H / mpp
    fits         = spec_px_w <= SENSOR_W_PX and spec_px_h <= SENSOR_H_PX
    fill_pct     = min(spec_px_w, SENSOR_W_PX) * min(spec_px_h, SENSOR_H_PX) \
                   / (SENSOR_W_PX * SENSOR_H_PX) * 100
    # how much of the specimen is visible
    vis_w        = min(SPEC_W, fov_w)
    vis_h        = min(SPEC_H, fov_h)
    vis_pct      = vis_w * vis_h / (SPEC_W * SPEC_H) * 100
    # standoff needed to fit specimen with this lens
    d_needed     = max(SPEC_W * f / SENSOR_W_MM, SPEC_H * f / SENSOR_H_MM)
    L.update(dict(mpp=mpp, fov_w=fov_w, fov_h=fov_h,
                  spec_px_w=spec_px_w, spec_px_h=spec_px_h,
                  fits=fits, fill_pct=fill_pct,
                  vis_w=vis_w, vis_h=vis_h, vis_pct=vis_pct,
                  d_needed=d_needed))

# ── figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14), facecolor="#111111")
gs  = GridSpec(2, 3, figure=fig,
               left=0.07, right=0.97, top=0.91, bottom=0.08,
               hspace=0.42, wspace=0.35,
               width_ratios=[1, 1, 1.1])

fig.suptitle(
    f"Boson 640  —  9.2 mm vs 18 mm lens  @  {STANDOFF:.0f} mm fixed standoff\n"
    f"Specimen: {SPEC_W:.0f} × {SPEC_H:.0f} mm",
    fontsize=14, color="white", fontweight="bold")

# ── Row 0: sensor pixel-map panels ───────────────────────────────────────────
for col_idx, L in enumerate(LENSES):
    ax = fig.add_subplot(gs[0, col_idx])
    ax.set_facecolor("#1a1a1a")
    ax.set_xlim(0, SENSOR_W_PX)
    ax.set_ylim(0, SENSOR_H_PX)
    ax.set_aspect("equal")
    ax.tick_params(colors="#666666", labelsize=7)

    for spine in ax.spines.values():
        spine.set_edgecolor(L["col"])
        spine.set_linewidth(2.5)

    # sensor background
    ax.add_patch(mpatches.Rectangle(
        (0, 0), SENSOR_W_PX, SENSOR_H_PX,
        fc="#222222", ec="none", zorder=1))

    if L["fits"]:
        # specimen centred, fully inside sensor
        ox = (SENSOR_W_PX - L["spec_px_w"]) / 2
        oy = (SENSOR_H_PX - L["spec_px_h"]) / 2
        ax.add_patch(mpatches.Rectangle(
            (ox, oy), L["spec_px_w"], L["spec_px_h"],
            fc="#1a3a2a", ec="lime", lw=2, zorder=3))
        # grid lines
        for i in range(1, 7):
            ax.axvline(ox + i * L["spec_px_w"] / 6,
                       ymin=oy/SENSOR_H_PX, ymax=(oy+L["spec_px_h"])/SENSOR_H_PX,
                       color="lime", lw=0.5, alpha=0.4, zorder=4)
            ax.axhline(oy + i * L["spec_px_h"] / 6,
                       xmin=ox/SENSOR_W_PX, xmax=(ox+L["spec_px_w"])/SENSOR_W_PX,
                       color="lime", lw=0.5, alpha=0.4, zorder=4)
        ax.text(SENSOR_W_PX/2, SENSOR_H_PX/2,
                f"Specimen fully visible\n"
                f"{L['spec_px_w']:.0f} × {L['spec_px_h']:.0f} px\n"
                f"({SPEC_W:.0f} × {SPEC_H:.0f} mm)",
                ha="center", va="center",
                color="lime", fontsize=9, fontweight="bold", zorder=5)
        status_txt = "FITS"
        status_col = "lime"
    else:
        # specimen overflows — show the visible (clipped) portion centred
        # visible portion in pixels on sensor
        vis_px_w = L["vis_w"] / L["mpp"]
        vis_px_h = L["vis_h"] / L["mpp"]
        ox = (SENSOR_W_PX - vis_px_w) / 2
        oy = (SENSOR_H_PX - vis_px_h) / 2
        # grey overflow indicator
        ax.add_patch(mpatches.Rectangle(
            (0, 0), SENSOR_W_PX, SENSOR_H_PX,
            fc="#2a1a1a", ec="red", lw=1, ls="--", zorder=2))
        # visible central portion
        ax.add_patch(mpatches.Rectangle(
            (ox, oy), vis_px_w, vis_px_h,
            fc="#3a1a1a", ec="red", lw=2, zorder=3))
        ax.text(SENSOR_W_PX/2, SENSOR_H_PX/2,
                f"Only centre visible\n"
                f"{L['vis_w']:.0f} × {L['vis_h']:.0f} mm\n"
                f"({L['vis_pct']:.0f}% of specimen)",
                ha="center", va="center",
                color="#ff6b6b", fontsize=9, fontweight="bold", zorder=5)
        # overflow arrows showing how much is cut
        clip_w_mm = SPEC_W - L["vis_w"]
        clip_h_mm = SPEC_H - L["vis_h"]
        if clip_w_mm > 0:
            ax.annotate("", xy=(0, SENSOR_H_PX/2),
                        xytext=(ox, SENSOR_H_PX/2),
                        arrowprops=dict(arrowstyle="<-", color="#ff6b6b", lw=1.5))
            ax.text(ox/2, SENSOR_H_PX/2 + 15,
                    f"−{clip_w_mm/2:.0f}mm", ha="center",
                    color="#ff6b6b", fontsize=7)
            ax.annotate("", xy=(SENSOR_W_PX, SENSOR_H_PX/2),
                        xytext=(ox+vis_px_w, SENSOR_H_PX/2),
                        arrowprops=dict(arrowstyle="<-", color="#ff6b6b", lw=1.5))
            ax.text((ox+vis_px_w+SENSOR_W_PX)/2, SENSOR_H_PX/2 + 15,
                    f"−{clip_w_mm/2:.0f}mm", ha="center",
                    color="#ff6b6b", fontsize=7)
        status_txt = "CLIPS"
        status_col = "#ff6b6b"

    # standoff needed annotation
    ax.text(SENSOR_W_PX/2, SENSOR_H_PX - 12,
            f"Needs {L['d_needed']:.0f} mm standoff to fit",
            ha="center", color="#aaaaaa", fontsize=7.5, style="italic")

    ax.set_title(
        f"{L['label']}  @  {STANDOFF:.0f} mm  [{status_txt}]\n"
        f"{L['mpp']:.3f} mm/px  |  FOV: {L['fov_w']:.0f} × {L['fov_h']:.0f} mm",
        color=L["col"], fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel(f"← {SENSOR_W_PX} px ({SENSOR_W_PX*L['mpp']:.0f} mm) →",
                  color="#aaaaaa", fontsize=8)
    ax.set_ylabel(f"← {SENSOR_H_PX} px ({SENSOR_H_PX*L['mpp']:.0f} mm) →",
                  color="#aaaaaa", fontsize=8)

# ── Row 0 col 2: key metrics comparison table ─────────────────────────────────
ax_t = fig.add_subplot(gs[0, 2])
ax_t.set_facecolor("#111111")
ax_t.axis("off")

metrics = [
    ("Focal length",        "9.2 mm",                    "18 mm"),
    ("Standoff",            f"{STANDOFF:.0f} mm",         f"{STANDOFF:.0f} mm"),
    ("mm / pixel",          f"{LENSES[0]['mpp']:.4f}",    f"{LENSES[1]['mpp']:.4f}"),
    ("FOV width",           f"{LENSES[0]['fov_w']:.0f} mm", f"{LENSES[1]['fov_w']:.0f} mm"),
    ("FOV height",          f"{LENSES[0]['fov_h']:.0f} mm", f"{LENSES[1]['fov_h']:.0f} mm"),
    ("Spec px (W)",         f"{LENSES[0]['spec_px_w']:.0f} px", f"{LENSES[1]['spec_px_w']:.0f} px"),
    ("Spec px (H)",         f"{LENSES[0]['spec_px_h']:.0f} px", f"{LENSES[1]['spec_px_h']:.0f} px"),
    ("Specimen fits?",      "YES",                        "NO"),
    ("Visible area",        "100%",                       f"{LENSES[1]['vis_pct']:.0f}%"),
    ("Standoff to fit",     f"{LENSES[0]['d_needed']:.0f} mm", f"{LENSES[1]['d_needed']:.0f} mm"),
]

col_x = [0.05, 0.45, 0.78]
row_h  = 0.082
y0     = 0.94
hdrs   = ["Metric", "9.2 mm", "18 mm"]
hcols  = ["white", LENSES[0]["col"], LENSES[1]["col"]]
for ci, (hdr, hcol) in enumerate(zip(hdrs, hcols)):
    ax_t.text(col_x[ci], y0, hdr, transform=ax_t.transAxes,
              color=hcol, fontsize=10, fontweight="bold", va="top")
ax_t.plot([0.02, 0.98], [y0 - 0.025, y0 - 0.025],
          color="#444444", lw=1, transform=ax_t.transAxes, clip_on=False)

highlight_rows = {7, 8}   # "Specimen fits?" and "Visible area"
for ri, (metric, v1, v2) in enumerate(metrics):
    y = y0 - (ri + 1) * row_h
    bg = "#1e1e1e" if ri % 2 == 0 else "#161616"
    ax_t.add_patch(mpatches.Rectangle(
        (0.02, y - 0.01), 0.96, row_h - 0.005,
        transform=ax_t.transAxes, fc=bg, ec="none", zorder=0))
    c1 = "lime"   if ri in highlight_rows and v1 in ("YES", "100%") else "white"
    c2 = "#ff6b6b" if ri in highlight_rows and v1 == "YES" and v2 != "YES" else "white"
    ax_t.text(col_x[0], y, metric, transform=ax_t.transAxes,
              color="#aaaaaa", fontsize=9, va="top")
    ax_t.text(col_x[1], y, v1, transform=ax_t.transAxes,
              color=c1, fontsize=9, va="top", fontweight="bold")
    ax_t.text(col_x[2], y, v2, transform=ax_t.transAxes,
              color=c2, fontsize=9, va="top", fontweight="bold")
ax_t.set_title("Metrics @ 383 mm standoff", color="white",
               fontsize=10, fontweight="bold", pad=6)

# ── Row 1 col 0: mm/px bar chart ──────────────────────────────────────────────
ax_res = fig.add_subplot(gs[1, 0])
ax_res.set_facecolor("#1a1a1a")
labels = [L["label"] for L in LENSES]
mpps   = [L["mpp"]   for L in LENSES]
cols   = [L["col"]   for L in LENSES]
bars   = ax_res.bar(labels, mpps, color=cols, width=0.45, zorder=3)
ax_res.set_facecolor("#1a1a1a")
ax_res.tick_params(colors="#aaaaaa")
ax_res.set_ylabel("mm / pixel", color="#aaaaaa")
ax_res.set_title("Spatial resolution", color="white", fontsize=10, fontweight="bold")
ax_res.spines[["top","right"]].set_visible(False)
for sp in ["left","bottom"]: ax_res.spines[sp].set_color("#444444")
ax_res.yaxis.label.set_color("#aaaaaa")
# finer = better label
ax_res.text(0.5, 0.92, "← finer is better", ha="center",
            transform=ax_res.transAxes, color="#888888", fontsize=8, style="italic")
for bar, mpp in zip(bars, mpps):
    ax_res.text(bar.get_x() + bar.get_width()/2, mpp + 0.003,
                f"{mpp:.4f}\nmm/px", ha="center", va="bottom",
                color="white", fontsize=9, fontweight="bold")
ax_res.set_ylim(0, max(mpps) * 1.35)
ax_res.tick_params(axis="x", colors="white")
ax_res.grid(axis="y", color="#333333", lw=0.5, zorder=0)

# ── Row 1 col 1: FOV comparison ───────────────────────────────────────────────
ax_fov = fig.add_subplot(gs[1, 1])
ax_fov.set_facecolor("#1a1a1a")
ax_fov.set_aspect("equal")
ax_fov.tick_params(colors="#aaaaaa")
ax_fov.set_title("FOV footprint vs specimen", color="white",
                 fontsize=10, fontweight="bold")
ax_fov.spines[["top","right"]].set_visible(False)
for sp in ["left","bottom"]: ax_fov.spines[sp].set_color("#444444")

# specimen rectangle
ax_fov.add_patch(mpatches.Rectangle(
    (-SPEC_W/2, -SPEC_H/2), SPEC_W, SPEC_H,
    fc="#1a3a2a", ec="lime", lw=2, zorder=2, label=f"Specimen {SPEC_W:.0f}×{SPEC_H:.0f}mm"))

# FOV rectangles
for L in LENSES:
    ax_fov.add_patch(mpatches.Rectangle(
        (-L["fov_w"]/2, -L["fov_h"]/2), L["fov_w"], L["fov_h"],
        fc="none", ec=L["col"], lw=2, ls="--", zorder=3,
        label=f"{L['label']} FOV ({L['fov_w']:.0f}×{L['fov_h']:.0f}mm)"))

ax_fov.set_xlim(-350, 350)
ax_fov.set_ylim(-230, 230)
ax_fov.set_xlabel("mm", color="#aaaaaa", fontsize=8)
ax_fov.set_ylabel("mm", color="#aaaaaa", fontsize=8)
ax_fov.axhline(0, color="#333333", lw=0.5, zorder=1)
ax_fov.axvline(0, color="#333333", lw=0.5, zorder=1)
ax_fov.legend(fontsize=7.5, loc="upper right",
              facecolor="#222222", edgecolor="#555555",
              labelcolor="white")
ax_fov.grid(color="#222222", lw=0.5, zorder=0)

# ── Row 1 col 2: standoff-needed comparison ───────────────────────────────────
ax_sd = fig.add_subplot(gs[1, 2])
ax_sd.set_facecolor("#1a1a1a")

# continuous curve: standoff needed to fit specimen vs focal length
f_range = np.linspace(4, 25, 300)
d_needed = np.maximum(SPEC_W * f_range / SENSOR_W_MM,
                       SPEC_H * f_range / SENSOR_H_MM)
ax_sd.plot(f_range, d_needed, color="#888888", lw=2, zorder=2,
           label="Standoff to fit specimen")
ax_sd.axhline(STANDOFF, color="white", lw=1.5, ls="--", zorder=3,
              label=f"Your standoff: {STANDOFF:.0f} mm")

for L in LENSES:
    ax_sd.plot(L["f"], L["d_needed"], "o", color=L["col"],
               ms=10, zorder=5)
    offset_y = 20 if L["d_needed"] > STANDOFF else -35
    ax_sd.annotate(
        f"{L['label']}\nneeds {L['d_needed']:.0f} mm",
        xy=(L["f"], L["d_needed"]),
        xytext=(L["f"] + 1.2, L["d_needed"] + offset_y),
        color=L["col"], fontsize=8, fontweight="bold",
        arrowprops=dict(arrowstyle="-", color=L["col"], lw=1))

# shade: region where specimen fits at 383mm
f_fits_max = SENSOR_W_MM / SPEC_W * STANDOFF  # = 7.68/320*383 = 9.2mm
ax_sd.axvspan(4, f_fits_max, alpha=0.12, color="lime", zorder=1)
ax_sd.text(f_fits_max/2 + 2, STANDOFF + 40,
           f"Fits at\n{STANDOFF:.0f} mm\n(f ≤ {f_fits_max:.1f} mm)",
           color="lime", fontsize=8, ha="center", style="italic")

ax_sd.set_xlabel("Focal length (mm)", color="#aaaaaa")
ax_sd.set_ylabel("Standoff needed to fit specimen (mm)", color="#aaaaaa")
ax_sd.set_title("Required standoff vs focal length", color="white",
                fontsize=10, fontweight="bold")
ax_sd.tick_params(colors="#aaaaaa")
ax_sd.spines[["top","right"]].set_visible(False)
for sp in ["left","bottom"]: ax_sd.spines[sp].set_color("#444444")
ax_sd.legend(fontsize=8, facecolor="#222222", edgecolor="#555555", labelcolor="white")
ax_sd.grid(color="#333333", lw=0.5, zorder=0)
ax_sd.set_xlim(4, 22)
ax_sd.set_ylim(0, max(d_needed[-1], 1200))

plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved: {OUTPUT_PNG}")
