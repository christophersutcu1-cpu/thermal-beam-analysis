# Paper 2 — Experimental Protocol

## Excitation (identical for both shots — the 06d optimum)

Mirrored lamp pair, 300 mm standoff, 36° from normal, aimed at (±72, −64) mm
(the deep-defect row). Filaments at (±248, −64, 243) mm, mounted level.
1000 W per lamp (2 kW total). Pulse 3 s. Record from 1 s before the pulse to
30 s after pulse start. Position with template + laser through the filament
centre onto the marked aim points. Origin = specimen centre, +x right facing
the flat face, +y up, +z toward the camera; 2.4 mm inserts bottom-right.

## Cameras — both on axis, perpendicular, full plate, matched 1.04 mm/px

| Camera | Position (x, y, z) mm | mm/px | Frame |
|---|---|---|---|
| Boson 320 (8.3 mm lens, 60 Hz) | (0, 0, 720) | 1.04 | 333 × 266 mm — full plate |
| A655sc (45° lens, 50 Hz) | (0, 0, 800) | 1.04 | plate centred in frame |

One camera mounted at a time; lamps untouched between shots. NUC (A655sc) /
FFC (Boson) immediately before each shot. Fiducial tape at (±150, +40),
(0, +40), (±150, −80), (0, −80) for registration.

## Shots — 2 total

1. A655sc shot.
2. Swap camera, Boson shot.

Before each: plate within 0.5 K of ambient, centreline pattern-free.

Requirements:
- Both cameras on pre-set mounts before shot 1 — the swap must not touch the lamp rig.
- Boson automatic FFC **disabled during capture** (a mid-record FFC corrupts every technique).
- A655sc object-distance parameter set to 0.8 m.
- Live surface monitor: abort if the rise exceeds 100 K (prediction 71 K).
- The ratio of the two shots' early-time flux maps is the no-touch repeatability
  measurement — report it alongside the results.
- CNR is reported as continuous values; the ≥ 3 threshold is a secondary summary.
  For the 2.4 mm pair, average frames over t = 2.9–3.9 s (the predicted peak
  window) before computing CNR.

## Analysis — identical chain on both datasets (SEQpost)

1. Register to the mm grid via fiducials.
2. Flux-normalise each shot by its own early-time map (per-pixel dT/dt over
   0.10–0.40 s, 20 mm smooth) — removes shot-to-shot lamp variance and
   verifies the aim (asymmetry ≤ 5 %).
3. Techniques: raw ΔT · PIM · TSR 1st/2nd derivatives · PCT EOFs 2–4 · PPT
   phase at f_b = α₃₃/(πz²) per depth: 6.8 / 1.7 / 0.76 / 0.43 / 0.19 / 0.11 Hz
   at k₃₃ = 2.9 W/mK, or ×0.86 (5.9 … 0.09 Hz) at k₃₃ = 2.5. Recompute from the
   settled k₃₃ at analysis time; the recipe itself is insensitive to this choice.
4. Per defect: CNR = |ROI_def − ROI_sound| / σ_sound (6×6 mm insert centre vs
   10–20 mm same-section annulus). Detected = CNR ≥ 3.

## Outputs

- Detection matrix: 19 defects × 6 techniques × 2 cameras.
- Per-depth CNR gap A655sc / Boson at matched mm/px — the camera-spec cost.
- Recovery table: defects the Boson misses in raw ΔT, and the technique that
  lifts each to CNR ≥ 3, if one does.
- Depth-estimation error per camera (TSR 2nd-derivative peak time, PPT blind
  frequency).
- Best technique + observation time per depth, per camera.
