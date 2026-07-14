# Beam characterisation — standoff provenance (READ BEFORE TOUCHING THE DATA)

## The raw `.seq` filenames are mis-labelled

The three FLIR A655sc captures arrived (WeTransfer, 2026-05-13) named:

| `.seq` filename | TRUE standoff |
|---|---|
| `300mm.seq` | **300 mm** |
| `400mm.seq` | **500 mm** |
| `500mm.seq` | **700 mm** |

The true standoffs are **300 / 500 / 700 mm** (user confirmed 2026-05-14). The
filenames are wrong by one step from `400mm.seq` onward.

## Why the files are NOT renamed

`_seq_cache/<basename>/frames_K.npy` is keyed by the `.seq` **basename**. Renaming
`400mm.seq → 500mm.seq` would make the loader read the existing `500mm` cache —
which holds the *true-700 mm* frames — silently pairing the wrong data. The
scripts also cannot cleanly regenerate the cache (their historical `SEQ_ROOT`
pointed at a `Downloads\` path). Renaming would therefore *create* a corruption,
not fix one. **Leave the filenames as-is; rely on the mapping above.**

## Which script is canonical

- **`02_summarise_standoffs.py`** — CANONICAL. Uses the correct standoffs
  (300/500/700) **and** the `1/cos(15°)` camera-tilt correction. Produces
  `beam_shape_summary_seq.json`, from which `04_dflux_vs_measured.py` derives the
  beam model **σ(d) = 0.185 d + 9.1 mm** that feeds the DFLUX (`experimentalpulse.f`).
- **`01_invert_seq.py`** — SUPERSEDED preview. Its standoff *labels* were corrected
  2026-06-12, but it still lacks the tilt correction. **Do not use its σ outputs.**

Verified: `02` outputs (σ ≈ 48/71/100 mm std-convention; ×√2 → DFLUX 0.185 d + 9.1)
match the constants hard-coded in `experimentalpulse.f`.

This note applies to both `PL-H-V-HS1\` and `Cranfield_Characterisation\` (identical
pipelines / data).
