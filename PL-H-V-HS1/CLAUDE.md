# PL-H-V-HS1 — OST Energy-Density Paper Campaign (bench, dual-lamp)

**Status (2026-07):** this campaign is the experimental + computational basis of the
draft paper *"A Computational and Experimental Study on Heating Uniformity and Energy
Density in Optically Stimulated Thermography"* (Sutcu, Avdelidis, Meo,
Ibarra-Castanedo, Maldague) —
`C:\Users\cs1d25\OneDrive - University of Southampton\Postdoc\Paper writing\Energy Density\...docx`.
The draft is complete: experiments done, model validated, 77-config parametric study
done. Current work = the paper's declared **future work**: optimising lamp aim
offsets / angle / standoff for the follow-up experiment at full power (the 06-series
scripts here). This campaign is a *bench* dual-lamp setup — the workspace's drone-rig
Stage 2/3 scripts do not apply.

## Hardware (as used in the paper)

| Component | Notes |
|---|---|
| FLIR A655sc | 640×480 LWIR microbolometer, 17 µm, 7.5–14 µm, 50 Hz, NETD <30 mK; triggered from the lamp controller for timing |
| 2× Visioo PL-H-V-HS1 | 1000 W FEL QTH halogen, lens-free non-crossover reflector, beam knob set 70° |
| Visioo PL-C-V-H controller | Ethernet/Telnet control; also triggers the camera |
| Emissivity for radiometric conversion | 0.945 |

## The completed experimental campaign (paper §2)

- 9 configurations: angle 30/45/60° (from normal) × standoff 300/400/500 mm
  (standoff measured **from the lamp aperture**, angle from specimen normal).
- **Pulse 1 s, total power 1000 W (500 W per lamp)** — the old "500 W vs 1000 W"
  open question is RESOLVED: the lamps were deliberately driven at half rating.
- Both lamps aimed at the plate centre (no offsets), beam centre height = plate
  centre; positioned with template + laser level.
- 3 repeats per config: (1) control, (2) lamps + cables swapped (lamp bias),
  (3) positioning template flipped (geometric bias). Run order randomised.
- Data: centreline A-A ΔT profile (through the 0.3 mm defect row), ΔT = T_peak −
  T_init frames from ResearchIR.
- Headline results: mean lateral asymmetry −4.1 % (−9.5 % without template flip);
  shot-to-shot peak-ΔT CoV mean 26 %, worst at short standoff.

## Specimen

Stepped CFRP **plain-weave** epoxy laminate, 320 × 175 mm, ply 0.3 mm.
Sections (viewed from the stepped BACK face, left→right): 140 mm @ 3.00 mm,
40 mm @ 2.35 mm, 80 mm @ 1.75 mm, 60 mm @ 1.14 mm. **Viewed from the illuminated
flat face the thick 3.00 mm section is on the RIGHT** — the deepest defects
(2.4 mm below the flat face) are bottom-right in lamp/camera coordinates.
19 defects: square **Teflon inserts, 10 × 10 mm, 13 µm thick** at depths
0.3–2.4 mm from the flat face (NOT flat-bottom holes — contrast is set by the
interface resistance, modelled as gap conductance **h_c = 7390 W/m²K**
= 13 µm PTFE + 1 µm air). Schematic: paper Fig. 4 / `Plots\Paper Figures\Figure 4
Specimen Schematic.png` (drawn in back-face view — mirror x for lamp-side work).

Material properties (paper Table 1, representative, refs [31,32]):

| ρ (kg/m³) | Cp (J/kg·K) | k11, k22, k33 (W/m·K) |
|---|---|---|
| 1533 | 979 | 5.8, 4.3, 2.9 |

Derived: through-thickness diffusivity α₃₃ = 1.93 mm²/s; in-plane α₁₁ = 3.86 mm²/s;
through-thickness effusivity e = √(k₃₃ρCp) ≈ 2086 W·s½/m²K.

## Beam model — which file is canonical

- **`beam_shape_summary_seq.json` (from `02_summarise_standoffs.py`) is CANONICAL.**
  True standoffs 300/500/700 mm (raw `.seq` filenames mislabelled — see
  `../PROVENANCE_STANDOFFS.md`), 1/cos(15°) camera-tilt corrected, aspect
  σx/σy ≈ 1.04 → beam treated as circular.
- Two equivalent conventions for the same distribution:
  - DFLUX / paper: `I = I₀·exp(−(r/σ)²)`, **σ(d) = 0.185·d + 9.1 mm**
  - std-Gaussian: `exp(−0.5 r²/σ²)`, σx = 0.1291·d + 8.39, σy = 0.1324·d + 4.51
- Absolute flux: **I₀ = P·η/(πσ²)** per lamp with **η = 0.708** (Bédard 0.735 ×
  Jenkins temperature derating) — inverse-square law emerges from σ(d).
  Do NOT anchor absolute flux on the `peak_dT_K` values in the JSON (capture
  operating point/absorber not traceable); use the DFLUX anchor.
- **DISOWNED outputs** (do not use): `beam_shape_analysis_seq.json` and
  `beam_derived_combined.json` (01-preview: mislabelled standoffs, no tilt
  correction, spurious aspect 1.42), and `05_offset_uniformity_study.*`
  (user-declared nonsensical).

## Digital twin (paper §3) — Abaqus + DFLUX

`C:\Users\cs1d25\Abaqus\SteppedPlate\experimentalpulse.f`. Key constants (paper
Table 3): n = 2, a = 0.185, σᵥ = 9.1 mm, P = 500 W/lamp (paper campaign),
η = 0.708, α = ε = 0.945, h = 5 W/m²K, h_c = 7390 W/m²K, per-lamp obliquity
cosφ and lamp-to-point distance per Eqs. 6–8. Mesh: DC3D20, 539k elements,
3 elements through the 0.3 mm surface ply, 1 mm in-plane.
Validation across all 9 configs: mean r = 0.985, RMSE = 1.03 K, systematic
**+12.7 % peak over-prediction** (attributed to η/α uncertainty).

## Key paper findings that steer design work

- Angle ↑ ⇒ uniformity ↑, intensity ↓ (U = 46.9 % @15° → 8.9 % @75°); standoff is
  the more aggressive lever (U = 87 % @200 mm → 16 % @700 mm).
- Equal energy density ≠ equal response: within an ED band, peak ΔT varies up to
  2× and defect contrast up to 3.6× depending on delivery route — standoff
  dominates. ED alone is not a sufficient experiment descriptor.
- Uniformity metric **U** = CoV of the centreline ΔT profile after a 20 mm
  Gaussian smooth (removes defect signatures), over the full specimen width.
- Declared future work: optimise the offset between beam centres and specimen
  centre — implemented here as the 06-series scripts.

## Current follow-up work (06-series, 2026-07)

| Script | Purpose |
|---|---|
| `06_dual_source_specimen_map.py` | (standoff, angle, mirrored aim offset) sweep minimising width-weighted CoV over the 320×175 face |
| `06b_standoff_intensity_tradeoff.py` | Same optimiser per standoff on a shared absolute scale — intensity vs uniformity picture |
| `06c_defect_energy_check.py` | Per-defect delivered energy incl. aim-line drop toward the deep row. **Caveat:** its absolute "K-equiv" scale uses the disowned cardboard anchor and FBH-style reasoning — geometry ranking valid, absolute values superseded by 06d |
| `06d_experiment_design.py` | **The experiment recipe.** DFLUX absolute anchor (1 kW/lamp, η=0.708, ×1/1.127 derate), 1D FD Teflon-insert contrast model (interface R=1/7390) per (depth, section), 3D lateral-loss correction. Maximises the minimum defect contrast s.t. front-face rise ≤ 60 K. Decisions on record: no Abaqus in this study; metric = face CoV + 2× width CoV |

Planned next experiment: **2 kW total (1 kW per lamp — full rating)**; pulse
duration to be chosen from thermal-diffusion / contrast analysis (diffusion time
to the deepest 2.4 mm defect: z²/α₃₃ ≈ 3.0 s); configuration from the 06-series
optimisation. Full Paper-2 protocol (coverage validation → depth optimisation →
A655sc-vs-Boson + SEQpost post-processing comparison): `PAPER2_EXPERIMENT_PLAN.md`.
Post-processing lives in `...\Desktop\SEQpost` (PIM Parker-fit denoise → TSR/PCT/PPT).

## Conventions

- Distances mm; W/m² in Python; DFLUX internally W/mm².
- Beam profile in DFLUX convention has **no ½ factor**: `I(r) = I₀·exp(−(r/σ)ⁿ)`.
- Positive lamp angle tilts toward +X; lamp 2 mirrored.
- Outputs live next to the script that wrote them, named after it.
