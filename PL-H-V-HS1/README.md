# PL-H-V-HS1 Campaign

Beam characterisation of the Visioo PL-H-V-HS1 halogen lamp using a FLIR A655sc. Full project context, hardware spec, operating point, physics decisions, and DFLUX upgrade plan: see `CLAUDE.md` in this folder.

## Run order — the four pipeline scripts at this folder's root

| # | Script | What it does | Reads | Writes |
|---|---|---|---|---|
| 1 | `01_invert_seq.py` | Inverts each `.seq` to `dT(x,y)`; fits a 2D rotated Gaussian per standoff; caches extracted frames. | `.seq` files from `SEQ_ROOT` (hardcoded at top of script) | `beam_shape_analysis_seq.{json,png}`, `_seq_cache/<session>/frames_K.npy` |
| 2 | `02_summarise_standoffs.py` | Multi-standoff summary (cross-sections + linear `σ(d) = a·d + b` divergence fit). | Same `.seq` files (uses the same `_seq_cache/`) | `beam_shape_summary_seq.{json,png}` |
| 3 | `03_derive_combined.py` | Consolidated parametric beam model across all standoffs. | Same cache | `beam_derived_combined.{json,png}` |
| 4 | `04_dflux_vs_measured.py` | Compares measured beam vs the current DFLUX in `experimentalpulse.f`. | `beam_shape_summary_seq.json` + hardcoded DFLUX params | `dflux_vs_measured.{json,png}` |

**TL;DR — run `python 01_invert_seq.py` → `02` → `03` → `04` in this folder, in order.**

Before running for a new dataset, edit the `SEQ_ROOT` and `SESSIONS` lists at the top of `01_invert_seq.py` (and similar block in `02`/`03`) to point at the new `.seq` files and standoffs.

## Folder map

| Path | Purpose |
|---|---|
| `01_…`–`04_…` | Pipeline scripts (run in order). |
| `*.json`, `*.png` | Latest outputs from the pipeline, named after the script that wrote them. |
| `_seq_cache/` | Extracted radiometric frames (numpy `.npy`) cached per `.seq` so flirpy doesn't re-run. |
| `_probe_seq.py`, `_probe_meta.py` | Ad-hoc inspection of a `.seq` file (frame count, metadata, dimensions). |
| `_check_dT.py` | Visualise raw dT images from the cache to sanity-check beam containment before fitting. Outputs `_dT_check.png`. |
| `CLAUDE.md` | Full campaign documentation: physics decisions, open questions, DFLUX upgrade plan. |
| `design_planning/` | Pre-campaign analysis: fixture geometry, lens FOV trade-offs, near/far-field, optimal lamp configuration, two-beam superposition simulation. **Self-contained** — has its own `config.py` + `beam_on_specimen.py` helper used by many scripts. |
| `hardware_control/` | Lamp + camera control: `LampTest.py` (telnet flash control), `cameratest.py` (Boson SDK capture), `relayinterfacer.py` (USB relay driver — Boson-era, kept for reference). |
| `legacy_boson/` | Older Boson-TIFF workflow superseded by the FLIR `.seq` pipeline at this root. Kept for historical comparison; do not run for the new campaign. |

## Outstanding issues (from `CLAUDE.md`)

- 500 W vs 1000 W mismatch in DFLUX (`POWER_ELEC`).
- Cardboard `ρcδ = 2100 J/m²·K` assumed for 2 mm corrugated; needs weighing.
- Beam diameter exceeds 25° lens FOV at 750–1000 mm standoff.
