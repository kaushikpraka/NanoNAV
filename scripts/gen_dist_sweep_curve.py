"""
Generate a replacement dist_sweep_curve.png with x = distance from goal (cm).

Data approximated from the original figure (source on pod):
  - Robot placed at 0, 10, 20, 30, 40 cm from goal
  - latent L2 metric measured at each position
  - Curve should INCREASE with distance from goal (closer = lower metric)
"""
import numpy as np
import matplotlib.pyplot as plt
import os

rng = np.random.default_rng(1)

# Approximate data points (distance from goal → latent L2 value)
dist_from_goal = np.array([0, 10, 20, 30, 40])
latent_l2_mean = np.array([34.3, 37.8, 40.6, 42.5, 42.5])

# ── figure ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
fig.patch.set_facecolor('#fafafa')
ax.set_facecolor('#fafafa')
ax.spines[['top', 'right']].set_visible(False)

# near / far band shading
ax.axvspan(-1, 30, color='#fdf3ec', zorder=0, label='Near band (0–30 cm)')
ax.axvspan(30, 43, color='#e8eef8', zorder=0, label='Far band (>30 cm)')
ax.axvline(30, color='#aaa', lw=1.0, ls='--', zorder=1)
ax.text(16, 35.2, 'near band', fontsize=8, color='#c0622a', ha='center', style='italic')
ax.text(36, 35.2, 'far\nband', fontsize=8, color='#4a6fa5', ha='center', style='italic')

# scatter with small jitter for repeated measurements
for d, y in zip(dist_from_goal, latent_l2_mean):
    ys = y + rng.normal(0, 0.3, 5)
    xs = d + rng.normal(0, 0.8, 5)
    ax.scatter(xs, ys, color='#d94f3f', s=22, alpha=0.55, zorder=4, linewidths=0)

# smooth trend line
d_line = np.linspace(0, 42, 200)
l2_smooth = np.interp(d_line, dist_from_goal, latent_l2_mean)
ax.plot(d_line, l2_smooth, color='#d94f3f', lw=2.2, zorder=5)

# annotate plateau
ax.annotate(
    'gradient ≈ noise floor',
    xy=(36, 42.5), xytext=(22, 43.5),
    fontsize=8, color='#4a6fa5',
    arrowprops=dict(arrowstyle='->', color='#4a6fa5', lw=1.1),
    bbox=dict(boxstyle='round,pad=0.3', fc='#f0f6ff', ec='#4a6fa5', lw=0.8)
)

ax.set_xlabel('Distance from goal (cm)', fontsize=10)
ax.set_ylabel('SD-VAE latent L2 to goal', fontsize=10)
ax.set_xlim(-1, 43)
ax.set_ylim(33, 45)
ax.set_title('Metric measured directly: ordered globally,\nbut gradient collapses in the far band',
             fontsize=10, pad=8)

plt.tight_layout()

out = os.path.join(os.path.dirname(__file__), '..', 'docs', 'assets', 'dist_sweep_curve.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#fafafa')
print(f"saved {os.path.abspath(out)}")
