# Thermal Beam Analysis — Workspace

Per-lamp characterisation campaigns whose ultimate purpose is to design a multi-lamp irradiation rig (drone-mounted, in the current programme) that delivers the **most uniform irradiance** over the camera FOV — i.e. minimum coefficient of variation (CoV) on the target.

## The three-stage workflow

Every campaign — `PL-H-V-HS1/`, `Cranfield_Characterisation/`, anything added later — runs the same three stages in order. Each stage is downstream of the previous: you can't pick the lens until you know the beam, and you can't optimise the lamp arrangement until you've picked the lens.

### Stage 1 — Beam shape characterisation
Measure the lamp's spatial heat-flux distribution `q″(x,y)` from FLIR `.seq` captures on a cardboard absorber and fit a 2D Gaussian per standoff. Output: `{σ_x, σ_y, peak, x_c, y_c}` per standoff + a linear divergence model `σ(d) = a·d + b`.

Scripts (in campaign folder, run in order):
```
01_invert_seq.py            .seq → dT(x,y) → per-standoff Gaussian fit
02_summarise_standoffs.py   multi-standoff cross-sections + divergence fit
03_derive_combined.py       consolidated parametric beam model → beam_derived_combined.json
04_dflux_vs_measured.py     (optional) compare against current Abaqus DFLUX
```

The contract for downstream stages is the `beam_derived_combined.json` written by stage 03 — `σ_x(d)`, `σ_y(d)`, peak-vs-distance, aspect ratio.

### Stage 2 — FOV per standoff (lens fixed)
Given the beam from Stage 1, compute the camera FOV and what fraction of the beam fits inside it at each standoff. The lens is **fixed at 13.1 mm (FOL18, 45° HFOV)** — the one used at Cranfield — so this stage is now a small lookup rather than a sweep, but stays as Stage 2 because Stage 3 reads its JSON output to know FOV dimensions per standoff.
- Standoffs: `300, 400, 500, 750 mm` (drone working envelope)
- Lens: `13.1 mm` only

Script:
```
05_lens_fov_sweep.py        (d) → FOV mm, mm/px, beam-vs-FOV coverage → 05_lens_fov_sweep.{json,png}
```

### Stage 3 — Lamp configuration / CoV minimisation
With the beam shape (Stage 1) and the FOV (Stage 2) known, find the lamp arrangement that minimises CoV on the FOV. Two configurations are compared, both using 4 lamps:
- **Config A — top + bottom pairs**: two lamps above the FOV, two lamps below it. Sweep tilt angle (pitch) + within-pair horizontal offset.
- **Config B — left + right pairs**: two lamps left of the FOV, two lamps right (side-by-side per side). Sweep tilt angle (yaw) + within-pair vertical offset.

For each `(config, standoff)`, sweep tilt and offset to find the `(angle, offset)` that minimises CoV across the FOV. The output figure is a 2 × 4 grid of irradiance maps — one per `(config, standoff)` — each rendered at that cell's optimum, plus a CoV-vs-standoff comparison line and an optimum-parameter table.

Script:
```
06_lamp_config_sweep.py     (config, d, angle, offset) → CoV(FOV) → 06_lamp_config_sweep.{json,png}
```

The pipeline output of Stage 3 is the design recommendation for the drone rig: what camera-to-target standoff, which lamp pattern, what lamp tilt, and what offset between the lamps in each pair.

## Layout

```
thermal-beam-analysis/
├── CLAUDE.md                       (this file — workspace methodology)
├── PL-H-V-HS1/                     Campaign 1: Visioo PL-H-V-HS1 halogen
│   ├── CLAUDE.md / README.md
│   ├── 01_invert_seq.py … 04_dflux_vs_measured.py            Stage 1
│   ├── (planned) 05_lens_fov_sweep.py, 06_lamp_config_sweep.py   Stages 2-3
│   ├── _seq_cache/, *.json, *.png   outputs alongside scripts
│   ├── _probe_*.py, _check_*.py     ad-hoc diagnostics
│   ├── design_planning/             pre-existing planning figures (legacy 2-lamp comparisons)
│   ├── hardware_control/            camera + lamp control
│   └── legacy_boson/                older Boson-TIFF workflow
│
└── Cranfield_Characterisation/     Campaign 2
    ├── CLAUDE.md / README.md
    ├── 01–04_*.py                   Stage 1
    ├── 05_lens_fov_sweep.py         Stage 2
    ├── 06_lamp_config_sweep.py      Stage 3
    └── wetransfer_…/                source .seq data
```

## Per-campaign setup

When starting a new campaign:
1. Copy stages 01–06 from a working campaign (PL-H-V-HS1 once Stages 2-3 are migrated, otherwise Cranfield).
2. Repoint `SEQ_ROOT` in 01–03 at the new `.seq` folder.
3. Adjust camera-geometry constants (`HFOV_DEG`, `VFOV_DEG`, `SENSOR_W_PX`, `SENSOR_H_PX`, `COUNT_TO_K`) and absorber `ρcδ / α` if the rig differs.
4. Stages 2 and 3 read `beam_derived_combined.json` automatically — no manual repointing needed if the upstream pipeline ran.
5. Write a campaign-specific `CLAUDE.md` recording hardware, operating point, and any non-default assumptions.

## Conventions (shared across campaigns)

- Distances in mm. Heat flux in W/m² in Python; the Abaqus DFLUX subroutines use W/mm² internally.
- Beam profile convention: `I(r) = I₀ · exp(−(r/σ)ⁿ)` (no factor of 0.5 — matches the SteppedPlate DFLUX).
- Stage 1 outputs and Stage 2/3 design figures all live next to the script that wrote them, named after the script.
- Pipeline scripts use `os.path.dirname(os.path.abspath(__file__))` to locate I/O — moving a script moves its outputs with it.
- Stage 1 output `beam_derived_combined.json` is the public interface between Stage 1 and Stages 2/3.

## Downstream (Abaqus DFLUX)

The beam parameters from Stage 1 also feed the DFLUX subroutine at `C:\Users\cs1d25\Abaqus\SteppedPlate\experimentalpulse.f` for thermography simulation — see the campaign-specific `CLAUDE.md` for the Tier 1/2/3 DFLUX upgrades planned per campaign.
