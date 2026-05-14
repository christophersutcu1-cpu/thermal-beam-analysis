"""Read metadata + frame-mean evolution from a .seq file."""
import tempfile, os, glob
import numpy as np
import cv2
from flirpy.io.seq import Splitter

src = r"C:\Users\cs1d25\Downloads\wetransfer_300mm-seq_2026-05-13_1259\300mm.seq"

with tempfile.TemporaryDirectory() as tmp:
    sp = Splitter(tmp, split_folders=False)
    sp.process([src])
    # metadata of first frame
    txt = os.path.join(tmp, 'raw', 'frame_000000.txt')
    if os.path.exists(txt):
        print("=== frame_000000.txt (full) ===")
        with open(txt, 'r', errors='ignore') as f:
            for line in f:
                line = line.rstrip()
                if any(k in line.lower() for k in
                       ['rate', 'time', 'temp', 'emissivity', 'distance', 'lens',
                        'planck', 'object', 'reflected', 'atmos', 'window', 'humidity']):
                    print(f"  {line}")
        print()

    rad_dir = os.path.join(tmp, 'radiometric')
    tiffs = sorted(glob.glob(os.path.join(rad_dir, '*.tiff')))
    means = []
    maxes = []
    for t in tiffs:
        img = cv2.imread(t, cv2.IMREAD_UNCHANGED).astype(np.float64)
        means.append(img.mean())
        maxes.append(img.max())
    means = np.array(means); maxes = np.array(maxes)
    print(f"Frame count: {len(means)}")
    print(f"Mean count - first 5: {means[:5].round(1)}")
    print(f"Mean count - last 5:  {means[-5:].round(1)}")
    print(f"Max count - first 5:  {maxes[:5].round(0)}")
    print(f"Max count - last 5:   {maxes[-5:].round(0)}")
    # find frame index of peak mean
    pk_idx = int(np.argmax(means))
    pk_max_idx = int(np.argmax(maxes))
    print(f"Frame at peak MEAN: {pk_idx}  (mean={means[pk_idx]:.1f})")
    print(f"Frame at peak MAX:  {pk_max_idx}  (max={maxes[pk_max_idx]:.0f})")
    # detect a step: difference between first 5 and last 5 means
    print(f"Baseline mean (frames 0-4): {means[:5].mean():.1f}")
    print(f"Plateau mean (frames -5:):  {means[-5:].mean():.1f}")
    print(f"Mean rise (counts):         {means[-5:].mean() - means[:5].mean():.2f}")
    print(f"Max rise (counts):          {maxes.max() - maxes[:5].mean():.2f}")
    # assume FLIR conversion count -> T K (typical FLIR R/J0 calibration is custom; many users use *0.04)
    print(f"\nAssuming x0.04 K/count:")
    print(f"  Peak DT (mean): {(means.max()-means[:5].mean())*0.04:.3f} K")
    print(f"  Peak DT (max):  {(maxes.max()-maxes[:5].mean())*0.04:.3f} K")
