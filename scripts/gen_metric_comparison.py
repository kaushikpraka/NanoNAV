"""
Generate a schematic figure comparing SD-VAE L2 vs. DINOv2 cosine distance.

X-axis = distance from goal (cm), reversed so far-from-goal is on the left
and the robot approaches the goal toward the right. Metric should DECREASE
as the robot gets closer (moves right).

Key numbers from Gate A sweep (context/learned-distance-metric.md):
  SD-VAE L2:     far-band slope 1.25σ  (gradient buried in noise — curve stays flat)
  DINOv2 cosine: far-band slope  12σ   (clear gradient throughout — curve descends)
"""
import numpy as np
import matplotlib.pyplot as plt
import os

rng = np.random.default_rng(0)

# measurement positions: actual radial distances 5-60 cm from goal
radii = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
n_per = 6   # lateral / yaw repeats at each radius

noise_std = 1.0   # normalised noise unit (σ)

# Underlying functions in terms of r = distance to goal.
# Metric should be 0 at goal (r=0) and increase with r.
def vae_true(r):
    # Steep near goal, plateaus far from goal (far-band slope ≈ noise floor)
    return 22 * (1 - np.exp(-r / 18))   # 0 at goal, plateau ~22 far out

def dino_true(r):
    # Linear throughout (far-band slope ≈ 12× noise)
    return 0.55 * r / 60   # 0 at goal → 0.55 at 60 cm

# scatter points: x = distance from goal
vae_pts, dino_pts, r_pts = [], [], []
for r in radii:
    for _ in range(n_per):
        r_jitter = r + rng.uniform(-2, 2)
        vae_pts.append(vae_true(r_jitter) + rng.normal(0, noise_std))
        dino_pts.append(dino_true(r_jitter) + rng.normal(0, noise_std * 0.08))
        r_pts.append(r_jitter)

r_pts    = np.array(r_pts)
vae_pts  = np.array(vae_pts)
dino_pts = np.array(dino_pts)

# Rescale DINOv2 to share y-axis range with VAE
dino_scale      = vae_true(60) / dino_true(60)
dino_pts_scaled = dino_pts * dino_scale

# Trend lines: x = distance from goal (same as r)
r_line    = np.linspace(0, 60, 300)
vae_line  = vae_true(r_line)
dino_line = dino_true(r_line) * dino_scale

def smooth(y, w=12):
    return np.convolve(y, np.ones(w) / w, mode='same')

vae_s  = smooth(vae_line)
dino_s = smooth(dino_line)

# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
fig.patch.set_facecolor('#fafafa')

# Far band: distance from goal > 30 cm (left side of reversed axis)
FAR_DIST_MIN = 30

for ax, x_sc, pts, line, color, label in [
    (axes[0], r_pts, vae_pts,          vae_s,  '#d94f3f', 'SD-VAE latent L2'),
    (axes[1], r_pts, dino_pts_scaled,  dino_s, '#2e7bbf', 'DINOv2 patch cosine'),
]:
    ax.set_facecolor('#fafafa')
    ax.spines[['top', 'right']].set_visible(False)

    # far band shading (high distance = far from goal, appears on LEFT with reversed axis)
    ax.axvspan(FAR_DIST_MIN, 63, color='#f0f0f0', zorder=0)
    ax.text(48, 2, 'far from\ngoal', fontsize=8, color='#aaa', va='bottom', ha='center')

    # noise band
    ax.fill_between(r_line, line - noise_std * 3, line + noise_std * 3,
                    color=color, alpha=0.12, zorder=1)

    # scatter
    ax.scatter(x_sc, pts, color=color, s=18, alpha=0.55, zorder=3, linewidths=0)

    # trend line
    ax.plot(r_line, line, color=color, lw=2.2, zorder=4)

    ax.set_xlabel('Distance from goal (cm)', fontsize=10)
    # reversed: far-from-goal on the left, goal on the right
    ax.set_xlim(63, -2)
    ax.set_ylim(bottom=-1)
    ax.set_title(label, fontsize=11, color=color, pad=8)

    # goal line at x=0 (right edge)
    ax.axvline(0, color='#888', lw=1.0, ls=':')
    ax.text(1, ax.get_ylim()[0] + 0.5, 'goal', fontsize=8, color='#888', va='bottom')

# VAE: annotate the flat region (high distance = far from goal, left side of reversed axis)
ax0 = axes[0]
ax0.set_ylabel('Distance metric value (normalised)', fontsize=10)
ax0.annotate(
    'gradient ≈ noise floor\nCEM cannot distinguish\ncandidate actions here',
    xy=(45, vae_true(45)),
    xytext=(30, vae_true(45) + 6),
    fontsize=8, color='#d94f3f',
    arrowprops=dict(arrowstyle='->', color='#d94f3f', lw=1.1),
    bbox=dict(boxstyle='round,pad=0.3', fc='#fff8f8', ec='#d94f3f', lw=0.8)
)

# DINOv2: annotate steady descent in far band
ax1 = axes[1]
r_lo, r_hi = 35, 55   # distance from goal; r_hi is farther (left), r_lo is closer (right)
y_lo = dino_true(r_lo) * dino_scale
y_hi = dino_true(r_hi) * dino_scale
# arrow from farther (r_hi) toward closer (r_lo) — left to right visually on reversed axis
ax1.annotate('', xy=(r_lo, y_lo), xytext=(r_hi, y_hi),
             arrowprops=dict(arrowstyle='->', color='#2e7bbf', lw=1.3))
ax1.text((r_lo + r_hi) / 2 - 2, (y_lo + y_hi) / 2,
         '~12× noise\nper step', fontsize=8, color='#2e7bbf', va='center',
         bbox=dict(boxstyle='round,pad=0.3', fc='#f0f6ff', ec='#2e7bbf', lw=0.8))

fig.suptitle('Same images, different representations — one metric goes blind',
             fontsize=12, y=1.02)
plt.tight_layout()

out = os.path.join(os.path.dirname(__file__), '..', 'docs', 'assets', 'metric_comparison.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#fafafa')
print(f"saved {os.path.abspath(out)}")
