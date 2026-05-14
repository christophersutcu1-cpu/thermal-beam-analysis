"""Visualise raw dT for each session to check beam containment."""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_seq_cache")
sessions = [("300mm", 300), ("400mm", 400), ("500mm", 500)]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, d) in zip(axes, sessions):
    frames = np.load(os.path.join(CACHE_DIR, name, "frames_K.npy"))
    base = frames[:5].mean(axis=0)
    hot  = frames[-5:].mean(axis=0)
    dT   = hot - base
    im = ax.imshow(dT, cmap="inferno", aspect="equal")
    # intensity-weighted centroid
    pos = np.clip(dT, 0, None)
    if pos.sum() > 0:
        yy, xx = np.mgrid[:dT.shape[0], :dT.shape[1]]
        cx_w = (xx*pos).sum()/pos.sum(); cy_w = (yy*pos).sum()/pos.sum()
    else:
        cx_w, cy_w = -1, -1
    pk = np.unravel_index(np.argmax(dT), dT.shape)
    ax.plot(pk[1], pk[0], "x", color="cyan", ms=12, mew=2, label=f"argmax ({pk[1]},{pk[0]})")
    ax.plot(cx_w, cy_w, "+", color="lime", ms=12, mew=2,
            label=f"weighted cen ({cx_w:.0f},{cy_w:.0f})")
    ax.set_title(f"{name}: peak dT = {dT.max():.2f} K, dim = {dT.shape}")
    ax.legend(loc="lower right", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.04, label="dT [K]")
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "_dT_check.png"), dpi=110, bbox_inches="tight")
print("Saved _dT_check.png")
