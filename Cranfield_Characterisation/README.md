# Cranfield Characterisation

Placeholder for the next beam-characterisation campaign at Cranfield.

When starting:
1. Fill out hardware + operating point in a new `CLAUDE.md` in this folder (template: `../PL-H-V-HS1/CLAUDE.md`).
2. Copy the four pipeline scripts (`01_invert_seq.py` → `04_dflux_vs_measured.py`) from `../PL-H-V-HS1/`.
3. Update `SEQ_ROOT`, `SESSIONS`, and camera-geometry constants (`HFOV_DEG`, `VFOV_DEG`, `SENSOR_W_PX`, `SENSOR_H_PX`, `COUNT_TO_K`) at the top of each pipeline script to match the Cranfield hardware.
4. Adjust absorber `ρcδ` and `α` if a different target material is used.

See `../CLAUDE.md` for the workspace-wide methodology.
