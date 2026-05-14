# Cranfield Characterisation

Beam-characterisation campaign on a halogen-class lamp. The campaign follows the three-stage workflow defined in `../CLAUDE.md`. **Two distinct cameras are in play** — keep them separated:

- **Characterisation camera** (Stage 1 — already captured): FLIR A655sc + 13.1 mm lens (FOL18, 45° HFOV). This is what produced the `.seq` files used to fit the beam shape.
- **Deployment camera** (Stages 2 & 3 — the drone payload): Teledyne FLIR Boson+ 640 × 512 with 18 mm lens (24° HFOV, 12 µm pitch, shutterless LWIR). This is what Stage 2 designs the FOV for and Stage 3 minimises CoV across.

## Hardware

| Stage | Component | Notes |
|---|---|---|
| 1 (capture) | FLIR A655sc | 640×480, 17 µm pitch, LWIR |
| 1 (capture) | A655sc lens | 13.1 mm focal length, 45° HFOV (FOL18) |
| 1 (target) | Absorber | (assumed) plain cardboard — same `α / ρcδ` as PL-H-V-HS1 unless told otherwise |
| 1 (dataset) | Standoffs | 300, 400, 500 mm (filenames in `wetransfer_300mm-seq_2026-05-13_1259/`) |
| 2-3 (deploy) | **Teledyne FLIR Boson+** | **640 × 512, 12 µm pitch, shutterless LWIR, 21×21×11 mm core** |
| 2-3 (deploy) | **Boson+ lens** | **18 mm focal length, 24° HFOV (≈19.3° VFOV)** |

## Operating point — open questions

The dataset shows pulse-rise temperatures around 1–2 K (vs PL-H-V-HS1's protocol-expected 50–70 K), so either:
- Pulse duration was much shorter than the 5 s used at Southampton,
- Lamp power setting was lower,
- Or the absorber is different.

Stage 1 still runs end-to-end on this data; the numbers are physically valid for the rig as captured but the absolute peak flux can't be compared to PL-H-V-HS1 without knowing pulse duration and lamp model.

## Three-stage pipeline

| Stage | Script | Purpose |
|---|---|---|
| 1 | `01_invert_seq.py` → `02_summarise_standoffs.py` → `03_derive_combined.py` | Beam shape `{σ_x, σ_y, peak}` per standoff + linear divergence fit. Output: `beam_derived_combined.json`. |
| 1 (optional) | `04_dflux_vs_measured.py` | Compares fitted beam against the Abaqus SteppedPlate DFLUX subroutine. Not part of the drone workflow — kept for reference against the PL-H-V-HS1 campaign. |
| 2 | `05_lens_fov_sweep.py` | Computes Boson+ (18 mm / 24° HFoV) camera FOV, mm/px, and beam-vs-FOV coverage at each standoff (300 / 400 / 500 / 750 mm). |
| 3 | `06_lamp_config_sweep.py` | For each standoff, sweep lamp tilt and within-pair offset for two 4-lamp configurations (A = top + bottom pairs, B = 4 lamps on the camera's y-line). Output is a 2 × 4 grid of irradiance maps with the schematic of each config on the left, plus a CoV-vs-standoff summary and an optimum-parameter table. |

Run order for the drone workflow: `python 01_invert_seq.py → 02_summarise_standoffs.py → 03_derive_combined.py → 05_lens_fov_sweep.py → 06_lamp_config_sweep.py`.

The end-to-end data flow is also drawn in `workflow_diagram.png` (regenerable via `python workflow_diagram.py`).

## Stage-1 issues to be aware of (data quality)

Captured during the first analysis (2026-05-14):
- Beam centroid sits above the camera FOV at all three standoffs — `centroid_y ≈ 0` in `01`'s JSON. σ_y / FWHM_y are therefore unreliable (bottom half of the Gaussian is partially missing).
- Peak ΔT is non-monotonic with standoff (300 → 2.30 K, 400 → 1.46 K, 500 → 1.91 K). Either captures weren't at the same operating point, or there was baseline drift between shots.
- 500 mm session has 18 frozen frames (USB stalls) and only 96 frames vs the other sessions' 99.
- `02_summarise_standoffs.py` was patched for an off-by-one bug exposed by the short 500 mm session (`while i < N-2` → `while i < len(dpk)-2`).
- `02` is configured with `CAM_TILT_DEG = 15` from the PL-H-V-HS1 rig. If the Cranfield setup didn't tilt the camera 15° off normal, set `CAM_TILT_DEG = 0` and re-run.

These propagate into Stages 2 and 3 — the methodology is correct but the absolute numbers should be re-evaluated once the capture is redone with the lamp aimed into the centre of the FOV.

## Operational lens (fixed) — used by Stage 2 + 3

| Focal length | HFOV / VFOV | Sensor | Notes |
|---|---|---|---|
| **18 mm** | 24° / 19.3° | Boson+ 640×512 @ 12 µm | Deployment camera + lens |

Lens-sweep code paths in `05_lens_fov_sweep.py` are kept (just edit the `LENSES` list back to multiple entries) but the design conclusion for this campaign is built around the 18 mm / Boson+ combination only. The 13.1 mm A655sc setup was only used to capture the Stage 1 beam shape data.

## Headline result (current run)

Overall best from the latest Stage 3 sweep:

| Config | Standoff | Tilt | Offset (× FOV) | Min CoV |
|---|---|---|---|---|
| **A — 2 above + 2 below FOV** | 300 mm | 35° | 1.00 | **3.20 %** |

Config A wins at every standoff (3.20 – 4.03 %). Config B (4 lamps on the camera's y-line) is worse at every standoff (4.04 – 9.67 %) because nothing tilts in y, so σ_y must alone cover FOV height.

Caveats: Stage 1 absolute peak values are unreliable (beam was above the FOV during capture, σ_y under-estimated). Several Stage 3 optima are pegged at the sweep edges (offset = 1.00, tilt = 60° for Config B) — widen `THETA_DEGS` / `OFFSET_FRAC` in `06_lamp_config_sweep.py` if you want to confirm there's nothing better past the boundary.

## Files in this folder

| File | Purpose |
|---|---|
| `01_invert_seq.py` – `04_dflux_vs_measured.py` | Stage 1 pipeline (beam characterisation from `.seq`). |
| `05_lens_fov_sweep.py` | Stage 2 — Boson+ FOV per standoff. |
| `06_lamp_config_sweep.py` | Stage 3 — lamp-configuration CoV sweep. |
| `workflow_diagram.py` / `.png` | End-to-end flowchart of the SEQ → drone-rig pipeline. |
| `beam_*.json` / `beam_*.png` | Stage 1 outputs (per-standoff fits + derived model). |
| `05_lens_fov_sweep.json` / `.png` | Stage 2 outputs. |
| `06_lamp_config_sweep.json` / `.png` | Stage 3 outputs — the design-decision figure. |
| `dflux_vs_measured.json` / `.png` | Optional Stage 1 sub-output (Abaqus comparison). |
| `_probe_seq.py`, `_probe_meta.py`, `_check_dT.py` | Ad-hoc diagnostics for inspecting `.seq` files / cache. |
| `_seq_cache/` *(gitignored)* | Per-`.seq` frame cache (numpy `.npy`). Not tracked because it's large and regenerable. |
| `wetransfer_*/` *(gitignored)* | Source `.seq` files. Not tracked — get them from the original wetransfer link or supplier. |
