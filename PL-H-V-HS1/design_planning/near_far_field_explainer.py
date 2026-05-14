"""
Near/far-field beam-geometry explainer for the PL-H-V-HS1 at 70°.

Produces a six-panel figure illustrating:
  A) side-view schematic with near-/transition/far-field zones
  B-D) 2D normalised intensity at 300 mm, 800 mm, 1500 mm
  E) 1D normalised profile comparison across distances
  F) peak flux x distance squared (should be flat in far-field)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

D_APERT = 0.15
HALF_ANGLE_70 = np.deg2rad(35)

def reflector_beam(distance, n=240):
    """Stylised q''(x,y) for a non-crossover reflector lamp.
    Far-field -> 2D Gaussian. Near-field -> Gaussian * (1 + ring + asymmetric hot spots),
    structure amplitude decays as exp(-(d/5D)^2)."""
    field = max(2 * distance * np.tan(HALF_ANGLE_70) * 1.2, 0.3)
    x = np.linspace(-field/2, field/2, n)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2)
    w = distance * np.tan(HALF_ANGLE_70) * 0.55
    envelope = np.exp(-(R/w)**2)
    nf = np.exp(-(distance / (5*D_APERT))**2)   # near-field weighting
    ring_r = D_APERT/2 + distance * np.tan(np.deg2rad(18))
    ring_w = 0.025 + distance*0.04
    ring = np.exp(-((R - ring_r)/ring_w)**2)
    asym = np.zeros_like(R)
    for k in range(3):
        a = k * 2*np.pi/3 + 0.3
        cx, cy = 0.55*D_APERT*np.cos(a), 0.55*D_APERT*np.sin(a)
        asym += np.exp(-((X-cx)**2 + (Y-cy)**2)/(0.025**2))
    structure = 0.75*ring + 0.55*asym
    I = envelope * (1 + nf * structure)
    return X, Y, I / distance**2, field

fig = plt.figure(figsize=(15, 9))

# ---- A: schematic ----
axA = plt.subplot2grid((2, 3), (0, 0), colspan=3)
axA.add_patch(Wedge((0, 0), D_APERT/2, 90, 270, width=0.02, color='silver'))
axA.plot(0, 0, 'o', color='orange', markersize=14, zorder=10)
axA.annotate('FEL filament\n+ reflector\n(D ~ 150 mm)', xy=(0, -D_APERT/2-0.02),
             xytext=(0, -0.55), ha='center', fontsize=9,
             arrowprops=dict(arrowstyle='->', color='gray'))

# zones
axA.axvspan(0, 5*D_APERT, alpha=0.18, color='red')
axA.axvspan(5*D_APERT, 10*D_APERT, alpha=0.18, color='gold')
axA.axvspan(10*D_APERT, 1.8, alpha=0.15, color='green')

# user's range highlighted on top
axA.axvspan(0.3, 0.5, ymin=0.85, ymax=0.95, color='orangered', alpha=0.9)
axA.text(0.4, 0.95, 'your experiments\n300-500 mm', color='white',
         ha='center', va='bottom', fontsize=8,
         bbox=dict(boxstyle='round', facecolor='orangered', edgecolor='none'))

# beam cone
for sign in [1, -1]:
    axA.plot([0, 1.8], [sign*D_APERT/2, sign*(1.8*np.tan(HALF_ANGLE_70) + D_APERT/2*0.3)],
             'b--', alpha=0.5, lw=1)

# example distance lines and beam diameter labels
for d in [0.3, 0.5, 0.8, 1.2, 1.5]:
    r = d * np.tan(HALF_ANGLE_70) + D_APERT/2*0.3
    axA.plot([d, d], [-r, r], color='steelblue', alpha=0.35, lw=1)
    axA.text(d, -r - 0.08, f'{int(d*1000)} mm', ha='center', fontsize=8, color='steelblue')

axA.text(0.07, 0.85, 'NEAR-FIELD\n< 5xD\n(structured)', color='darkred',
         fontsize=9, ha='left', weight='bold')
axA.text(0.78, 0.85, 'TRANSITION\n5-10xD', color='darkgoldenrod',
         fontsize=9, ha='center', weight='bold')
axA.text(1.45, 0.85, 'FAR-FIELD\n> 10xD\n(smooth cone)', color='darkgreen',
         fontsize=9, ha='center', weight='bold')

axA.set_xlim(-0.05, 1.85)
axA.set_ylim(-1.1, 1.2)
axA.set_xlabel('axial distance from reflector aperture [m]')
axA.set_ylabel('off-axis [m]')
axA.set_title('PL-H-V-HS1 at 70 deg full angle (35 deg half-angle), reflector D ~ 150 mm')
axA.grid(alpha=0.3)
axA.set_aspect('equal')

# ---- B,C,D: 2D maps ----
for i, d in enumerate([0.3, 0.8, 1.5]):
    ax = plt.subplot2grid((2, 3), (1, i))
    X, Y, I, _ = reflector_beam(d)
    Inorm = I / I.max()
    im = ax.pcolormesh(X*1000, Y*1000, Inorm, cmap='inferno',
                       shading='auto', vmin=0, vmax=1)
    ax.set_aspect('equal')
    label = {0.3: '300 mm  (NEAR-field, structured)',
             0.8: '800 mm  (transition)',
             1.5: '1500 mm  (far-field, smooth)'}[d]
    ax.set_title(label, fontsize=10)
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    plt.colorbar(im, ax=ax, label='q\" / q\"_peak', fraction=0.045)

plt.tight_layout()
plt.savefig('near_far_field_explainer.png', dpi=140, bbox_inches='tight')

# === Second figure: 1D profiles & peak-vs-distance ===
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

ds_profile = [0.3, 0.4, 0.5, 0.8, 1.2, 1.5]
cmap = plt.cm.viridis(np.linspace(0, 0.9, len(ds_profile)))
for d, c in zip(ds_profile, cmap):
    X, Y, I, _ = reflector_beam(d)
    mid = I.shape[0]//2
    prof = I[mid, :]
    # plot vs x normalised by beam width so shapes can be compared directly
    x_norm = X[mid, :] / (d * np.tan(HALF_ANGLE_70))
    ax1.plot(x_norm, prof/prof.max(), color=c, lw=1.6, label=f'{int(d*1000)} mm')
ax1.set_xlabel('x / (d * tan35) -- normalised off-axis position')
ax1.set_ylabel('q\" / q\"_peak')
ax1.set_title('Normalised profile (overlays only in far-field)')
ax1.legend(fontsize=8, ncol=2)
ax1.grid(alpha=0.3)
ax1.set_xlim(-1.3, 1.3)

# peak * d^2 should be a constant in far-field
dd = np.linspace(0.15, 2.0, 80)
peaks = []
for d in dd:
    _, _, I, _ = reflector_beam(d)
    peaks.append(I.max() * d**2)
peaks = np.array(peaks)
ax2.plot(dd*1000, peaks/peaks[-1], 'k-', lw=2.2)
ax2.axhline(1.0, color='green', ls='--', alpha=0.7, label='ideal 1/d^2')
ax2.axvspan(300, 500, color='orange', alpha=0.3, label='your range (300-500 mm)')
ax2.axvline(5*D_APERT*1000, color='red', ls=':', alpha=0.8, label='5xD = 750 mm')
ax2.axvline(10*D_APERT*1000, color='goldenrod', ls=':', alpha=0.8, label='10xD = 1500 mm')
ax2.set_xlabel('distance from aperture [mm]')
ax2.set_ylabel('q\"_peak x d^2  (normalised to far-field)')
ax2.set_title('Inverse-square scaling breaks down in near-field')
ax2.legend(fontsize=8, loc='lower right')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('near_far_field_profiles.png', dpi=140, bbox_inches='tight')

print('Saved: near_far_field_explainer.png')
print('Saved: near_far_field_profiles.png')
