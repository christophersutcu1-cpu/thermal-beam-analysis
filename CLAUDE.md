# Beam Characterisation Campaign — PL-H-V-HS1 → SteppedPlate DFLUX

## Purpose

Experimental campaign to measure the spatial heat-flux distribution `q″(x,y)` of the Visioo PL-H-V-HS1 halogen excitation lamp using the FLIR A655sc, then feed those measurements into the DFLUX subroutine of the SteppedPlate Abaqus pulse-thermography (PT) simulation. The two halves connect:

- **Experimental side** (this repo): `thermal-beam-analysis/` — Python pipeline for capturing FLIR `.seq`, inverting to `q″`, fitting beam-shape parameters.
- **Simulation side**: `C:\Users\cs1d25\Abaqus\SteppedPlate\experimentalpulse.f` — DFLUX consuming the fitted beam parameters.

Goal output: per-standoff fits of `{q″_peak, σ_x, σ_y, n, x_c, y_c}` from the experimental campaign, ready to drop into the DFLUX subroutine.

## Hardware

| Component | Notes |
|---|---|
| **FLIR A655sc** | LWIR microbolometer, 640×480, 7.5–14 µm, NETD ≤30 mK, 50 Hz full / 200 Hz windowed, radiometric. Uncooled (TEC-stabilised). |
| **2× Visioo PL-H-V-HS1** | 1000 W FEL halogen, "lens-free non-crossover" reflector, beam adjustable 20°–70°, reflector aperture ~150 mm, FEL bulb life 300 h. |
| **Visioo PL-C-V-H controller** | Ethernet (Telnet) control of flash / lockin / linear modes. |
| **Absorber** | Plain cardboard (no coating — α ≈ 0.7–0.8 in halogen band, ρcδ ≈ 2100 J/m²·K for 2 mm corrugated). |

Hardware spec sheet (quote): `C:\Users\cs1d25\Documents\Postdoc\MISC\Q20250823-1-SOU.pdf`.

## Operating point for this campaign

- **Beam knob: 70° only** (defocused / wide-flood mode — bare-filament-dominated, less near-field structure than 20° focused setting).
- **Standoffs**: 250, 500, 750, 1000 mm (4× range, covers historical 300–500 mm working zone plus far-field reference).
- **Single standard pulse duration: 5 s** across all four standoffs. Matches typical CFRP PT step-heating protocol for 1–5 mm depths.
- **Single lamp at a time** for the characterisation; two-lamp superposition validity verified separately at end of campaign.

## Experimental protocol (per shot)

1. **Lamp warm-up**: cycle the lamp once before the first measurement of the day (30 s on, cool, then start). Do not leave continuously on between shots — FEL bulb life is short.
2. **A655sc setup**: 50 Hz full-frame, default 0–150 °C range (extended range only if needed at 250 mm), fixed object distance / atmospheric transmission in radiometric setup, **shutter NUC immediately before every shot**.
3. **Capture window**: `t = −1 s` (baseline) → `t = 0` (pulse on) → `t = +10 s`. Pulse duration = 5 s, controller flash mode.
4. **Per-pixel analysis**: linear fit of `ΔT(x,y,t)` over `t = 0.2 s to 2.0 s` → `dT/dt(x,y)` → `q″_absorbed(x,y) = ρcδ · dT/dt(x,y) / α`.
5. **Between shots**: wait ~5 min for cardboard to return within 0.5 K of ambient, **or swap cards** (preferred — avoids accumulated discolouration drift in α).
6. **No active cooling** of cardboard, lamp, or camera. Forced airflow would corrupt the energy balance. Lamp housing is passively cooled by design.

## Standoff-specific expectations

| d [mm] | Beam Ø at plate | Peak q″ (est.) | ΔT after 5 s | Notes |
|---|---|---|---|---|
| 250 | ~350 mm | 20–30 kW/m² | 50–70 K | Confirm with 2 s test shot first; safe margin to char (~180 K) |
| 500 | ~700 mm | 5–8 kW/m² | 12–20 K | Sweet spot |
| 750 | ~1050 mm | 2–3.5 kW/m² | 5–8 K | Camera will need to step back for FOV |
| 1000 | ~1400 mm | 1–2 kW/m² | 3–5 K | Lowest SNR but >100× NETD |

## Physics decisions on record

- **5×D textbook near-field rule was rejected for this lamp at 70°.** Rationale: at the wide knob setting the reflector is defocused, bare-filament radiation dominates, and the "non-crossover" geometry suppresses focal artefacts. User's existing 300–500 mm data shows no ring / three-lobed structure. The four-standoff campaign will *verify* far-field behaviour empirically (self-similarity of normalised profiles, flatness of `q″_peak · d²`) rather than assume it from a conservative textbook threshold.
- **Cardboard chosen over a calibrated coating** because no coating is available. Acceptance is that absolute α has ~±10 % uncertainty; the absolute scaling can be cross-anchored later via total-power balance if needed.
- **Lateral conduction is not a constraint** at 5 s pulse — diffusion length L = √(α_t·t) ≈ 0.9 mm vs cm-scale beam features.

## Analysis pipeline (planned)

Per standoff, the script will output to a single results table:

| field | meaning |
|---|---|
| `d` | standoff [mm] |
| `q_peak` | peak absorbed flux [W/m²] |
| `sigma_x`, `sigma_y` | beam-perpendicular 1/e radii [mm] |
| `n` | super-Gaussian exponent (start fit free; expect 2–3.5) |
| `x_c`, `y_c` | centroid offset from optical axis [mm] |
| `total_power` | ∫∫ q″ dA — for power-balance closure |

Cross-standoff fits then yield:
- `σ_x(d) = a_x · d + b_x`, `σ_y(d) = a_y · d + b_y` → divergence half-angles and virtual-source offsets
- `q″_peak(d) · d²` constancy check → confirms or quantifies departure from 1/d²
- Mean of fitted `n` across standoffs → single `SG_N` value for DFLUX
- Centroid linearity in d per lamp → per-lamp aim offsets

## Downstream — SteppedPlate DFLUX upgrades (planned, post-analysis)

Current state of `experimentalpulse.f`:
- Pure Gaussian (`SG_N = 2`), circularly symmetric (single σ)
- `SIGMA(d) = 0.1751·d + 2.79 mm` (calibrated against 7.55 K peak at 45°/500 mm/1 s — knob setting unknown, suspected 20°)
- Both lamps aim exactly at plate centre
- 1/d² scaling baked in via `PEAK_I ∝ 1/σ²`
- No time dependence inside the subroutine — pulse comes from input-deck amplitude curve

**Tier 1 upgrades after this campaign** (drop-in):
1. Replace `SIGMA_PER_MM`, `SIGMA_VIRT` with refitted values from the new four-standoff sweep at 70°.
2. Add `SIGMA_PER_MM_X / _Y` and `SIGMA_VIRT_X / _Y` — split the circular σ into elliptical `σ_x, σ_y`. Update `R_GAUSS` and the integral normalisation accordingly.
3. Set `SG_N` and `GAMMA_2N = Γ(2/n)` from the fitted exponent (currently `GAMMA_2N = 1` because `SG_N = 2`).
4. If `q″_peak · d²` is not flat, fold a small empirical `f(d)` correction into PEAK_I.

**Tier 2** (geometric refinements):
- Per-lamp centroid offsets (`XC1`, `XC2`, `YC1`, `YC2`)
- Full 3D `n̂_beam · r̂_pixel` cosine projection (not just X-tilt)
- Stepped-geometry local normal projection — riser faces currently get wrong cosine
- Self-shadowing on step risers at 30° tilt

**Tier 3** (model upgrades):
- Time-resolved DFLUX with measured `g(t)` lamp on/off envelope
- Gridded `q″(x,y)` lookup replacing parametric Gaussian entirely

## Known issues / open questions

- **`POWER_ELEC = 500 W` in DFLUX vs 1000 W on the spec sheet** — comment says "500 W FEL lamps" but PL-H-V-HS1 is a 1000 W lamp. Either (a) lamps were being driven at 50 % via controller (legitimate but record it), or (b) latent bug. Product `POWER_ELEC × RAD_EFF = 164.5 W` is what's tied to the calibration; if 1000 W is correct, `RAD_EFF` should be 0.1645 not 0.329. **Resolve before re-calibrating with new data.**
- **Cardboard ρcδ assumed at 2100 J/m²·K for 2 mm corrugated.** Measure actual thickness and weigh a known area to nail this — drives absolute `q″` scaling directly.
- **Existing 300–500 mm data**: not yet decided whether to incorporate into the new fit or treat the new campaign as standalone. The previous data was at unknown knob setting; safer to treat as separate validation.
- **A655sc lens FOV vs beam diameter at long standoff**: at 750–1000 mm the beam (1.0–1.4 m diameter) exceeds the 25° lens FOV. Either raster-stitch or accept partial-skirt capture and fit only the inner ~70 % of the beam.

## Files & locations

| Path | Purpose |
|---|---|
| `./` (this repo) | Python pipeline for `.seq` → `q″` → fits |
| `./beam_shape_analysis.py` and siblings | Pre-existing fitting scripts — to be extended |
| `./near_far_field_explainer.py` + `.png`s | Didactic figures on near/far-field for 70° beam |
| `C:\Users\cs1d25\Abaqus\SteppedPlate\experimentalpulse.f` | DFLUX subroutine to be updated post-campaign |
| `C:\Users\cs1d25\Documents\Postdoc\MISC\Q20250823-1-SOU.pdf` | Visioo quote — authoritative hardware spec |

## Conventions

- **Distances** in mm throughout (matches DFLUX convention).
- **Heat flux** in W/m² (DFLUX uses W/mm² internally — `POWER_ELEC` is in mW per code; units already consistent there).
- **Beam profile convention**: `I(r) = I₀ · exp(−(r/σ)ⁿ)` (no factor of 0.5 in the exponent — matches existing DFLUX code).
- **Lamp angle**: positive `ANGLE_RAD` tilts lamp toward +X; lamp 2 mirrored at −ANGLE_RAD.
- **Plate frame origin** at the back-bottom corner; aim point `(XC_PLATE, YC_PLATE) = (160, 87.5) mm`.

## Workflow stages

- [x] Hardware confirmed, experimental design agreed
- [ ] Acquire 4-standoff dataset (250 / 500 / 750 / 1000 mm @ 70°, 5 s pulse)
- [ ] Run `.seq` → `q″` inversion pipeline
- [ ] Fit per-standoff `{q_peak, σ_x, σ_y, n, x_c, y_c}`
- [ ] Cross-standoff fits → divergence + virtual source
- [ ] Validate self-similarity and `q″·d²` constancy
- [ ] Resolve the 500 W / 1000 W question in DFLUX
- [ ] Apply Tier 1 upgrades to `experimentalpulse.f`
- [ ] Validate Abaqus sim against an independent CFRP PT shot
