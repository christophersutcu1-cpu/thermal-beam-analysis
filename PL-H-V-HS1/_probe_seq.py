"""Quick probe of a FLIR .seq file - reports dims, frame count, temp range, dt."""
import tempfile, os, glob, sys
import numpy as np
import cv2
from flirpy.io.seq import Splitter

src = r"C:\Users\cs1d25\Downloads\wetransfer_300mm-seq_2026-05-13_1259\300mm.seq"

with tempfile.TemporaryDirectory() as tmp:
    sp = Splitter(tmp, split_folders=False)
    sp.process([src])
    # discover output structure
    print("Top-level entries in tmp:")
    for e in sorted(os.listdir(tmp)):
        full = os.path.join(tmp, e)
        if os.path.isdir(full):
            print(f"  DIR  {e}/  -> {len(os.listdir(full))} entries")
        else:
            print(f"  FILE {e}  ({os.path.getsize(full)} B)")

    # Splitter writes preview/ radiometric/ raw/ in the output folder
    rad_dir = os.path.join(tmp, 'radiometric')
    raw_dir = os.path.join(tmp, 'raw')
    print(f"\nradiometric/ contents (first 5):")
    for f in sorted(os.listdir(rad_dir))[:5]:
        print(f"  {f}")
    print(f"raw/ contents (first 5):")
    for f in sorted(os.listdir(raw_dir))[:5]:
        print(f"  {f}")

    rad_files = sorted(glob.glob(os.path.join(rad_dir, '*')))
    print(f"\nTotal radiometric files: {len(rad_files)}")
    if rad_files:
        first = cv2.imread(rad_files[0], cv2.IMREAD_UNCHANGED)
        if first is not None:
            print(f"First frame: shape={first.shape}, dtype={first.dtype}")
            print(f"  min={first.min()}, max={first.max()}, mean={first.mean():.1f}")
            # try last
            last = cv2.imread(rad_files[-1], cv2.IMREAD_UNCHANGED)
            print(f"Last frame: min={last.min()}, max={last.max()}, mean={last.mean():.1f}")
            # frame 50 ish
            mid = cv2.imread(rad_files[len(rad_files)//2], cv2.IMREAD_UNCHANGED)
            print(f"Mid frame:  min={mid.min()}, max={mid.max()}, mean={mid.mean():.1f}")
        else:
            print(f"cv2 returned None for {rad_files[0]}")
            print(f"File size: {os.path.getsize(rad_files[0])} bytes")
    raw_files = sorted(glob.glob(os.path.join(raw_dir, '*')))
    print(f"\nTotal raw files: {len(raw_files)}")
    if raw_files:
        # Show extensions
        exts = set(os.path.splitext(f)[1] for f in raw_files)
        print(f"  extensions: {exts}")
        # Try exiftool-like extraction from .fff files for metadata
        for ext in exts:
            sample = next(f for f in raw_files if f.endswith(ext))
            print(f"  Sample {ext}: {sample}, size={os.path.getsize(sample)} B")
