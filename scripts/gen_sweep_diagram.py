"""
Top-down schematic of the Gate A distance-metric sweep.

Shows how the robot was placed at different radial distances (10-60 cm),
lateral offsets (±60 cm), and yaw orientations (±30°) relative to a fixed
goal position, and an image was captured at each pose.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import os

fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor('#fafafa')
ax.set_facecolor('#fafafa')
ax.set_aspect('equal')
ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

# ── coordinate system ──────────────────────────────────────────
# Goal at (0, 0).  Robot approaches from below (negative y = farther away).
# x = lateral offset;  y = -radial distance (so closer = higher on page)

RADII   = [10, 20, 30, 40, 50, 60]   # cm
LATERALS = [-40, -20, 0, 20, 40]      # cm  (subset for clarity)
YAWS     = [-25, 0, 25]               # degrees relative to straight-on

FAR_BAND = 30   # cm

# ── far-band shading ───────────────────────────────────────────
far_rect = plt.Rectangle((-75, -65), 150, 65 - FAR_BAND,
                          color='#e8eef8', zorder=0, label=f'Far band (>{FAR_BAND} cm)')
near_rect = plt.Rectangle((-75, -FAR_BAND), 150, FAR_BAND,
                           color='#fdf3ec', zorder=0, label=f'Near band (0–{FAR_BAND} cm)')
ax.add_patch(far_rect)
ax.add_patch(near_rect)
ax.text(68, -(FAR_BAND + 17), 'far band', fontsize=8.5, color='#4a6fa5',
        va='center', style='italic')
ax.text(68, -(FAR_BAND / 2), 'near band', fontsize=8.5, color='#c0622a',
        va='center', style='italic')

# boundary between bands
ax.axhline(-FAR_BAND, color='#aaaaaa', lw=1.0, ls='--', zorder=1)

# ── radial distance rings (partial arcs) ──────────────────────
for r in RADII:
    theta = np.linspace(np.pi * 1.1, np.pi * 1.9, 120)
    ax.plot(r * np.cos(theta), -r + r * np.sin(theta) * 0,
            color='#cccccc', lw=0.0)   # placeholder; use horizontal guide instead
    ax.annotate('', xy=(-73, -r), xytext=(-80, -r),
                arrowprops=dict(arrowstyle='-', color='#cccccc', lw=0.8))
    ax.text(-81, -r, f'{r} cm', fontsize=7.5, color='#999', ha='right', va='center')

ax.axhline(0, xmin=0.05, xmax=0.95, color='#cccccc', lw=0.5, ls=':')

# ── robot icon helper ──────────────────────────────────────────
def draw_robot(cx, cy, yaw_deg, color, alpha=0.85, size=4.5):
    """Draw a small triangle (body) + arrow (heading)."""
    yaw = np.radians(yaw_deg)
    # triangle vertices centred at (cx,cy)
    front = np.array([ np.sin(yaw),  np.cos(yaw)]) * size
    left  = np.array([-np.cos(yaw),  np.sin(yaw)]) * size * 0.55
    right = np.array([ np.cos(yaw), -np.sin(yaw)]) * size * 0.55
    verts = np.array([
        [cx + front[0], cy + front[1]],
        [cx + left[0],  cy + left[1]],
        [cx + right[0], cy + right[1]],
    ])
    tri = plt.Polygon(verts, closed=True, color=color, alpha=alpha, zorder=4,
                      linewidth=0.5, edgecolor='white')
    ax.add_patch(tri)

# ── plot robot poses ───────────────────────────────────────────
# Only show yaw variation at a few positions to keep it readable
SHOW_YAWS_AT = {(0, 40), (0, 50), (-20, 30), (20, 30)}   # (lateral, radius) pairs

far_color  = '#3a6bbf'
near_color = '#d46020'

for r in RADII:
    for lat in LATERALS:
        is_far = r >= FAR_BAND
        color = far_color if is_far else near_color
        if (lat, r) in SHOW_YAWS_AT:
            yaws_here = YAWS
        else:
            yaws_here = [0]
        for yaw in yaws_here:
            draw_robot(lat, -r, yaw, color=color,
                       alpha=0.7 if len(yaws_here) > 1 else 0.85)

# ── goal marker ───────────────────────────────────────────────
ax.scatter([0], [0], s=260, marker='*', color='#f5c400', zorder=6,
           edgecolors='#c9940a', linewidths=1.2)
ax.text(0, 2.5, 'Goal image\ncaptured here', fontsize=8.5, ha='center',
        va='bottom', color='#7a5c00',
        bbox=dict(boxstyle='round,pad=0.35', fc='#fffce0', ec='#c9940a', lw=0.8))

# ── lateral offset dimension arrow ────────────────────────────
ax.annotate('', xy=(40, -68), xytext=(-40, -68),
            arrowprops=dict(arrowstyle='<->', color='#555', lw=1.2))
ax.text(0, -70.5, '±40 cm lateral', fontsize=8.5, ha='center',
        color='#555')

# ── yaw variation callout at (0, -50) ─────────────────────────
# draw arc to show yaw range
theta_arc = np.linspace(np.radians(65), np.radians(115), 40)
arc_r = 11
cx0, cy0 = 0, -50
ax.plot(cx0 + arc_r * np.cos(theta_arc), cy0 + arc_r * np.sin(theta_arc),
        color='#555', lw=1.1, zorder=5)
ax.annotate('', xy=(cx0 + arc_r * np.cos(np.radians(115)),
                     cy0 + arc_r * np.sin(np.radians(115))),
            xytext=(cx0 + arc_r * np.cos(np.radians(114)),
                     cy0 + arc_r * np.sin(np.radians(114))),
            arrowprops=dict(arrowstyle='->', color='#555', lw=1.0))
ax.text(cx0 + 13.5, cy0 + 5, '±25°\nyaw', fontsize=8, color='#555', va='center')

# ── legend / annotation box ────────────────────────────────────
legend_elems = [
    mpatches.Patch(color=near_color, alpha=0.8, label='Near band (0–30 cm)'),
    mpatches.Patch(color=far_color,  alpha=0.8, label='Far band (30–60 cm)'),
    mpatches.Patch(color='#f5c400',  label='Goal position'),
]
ax.legend(handles=legend_elems, loc='lower right', fontsize=8.5,
          framealpha=0.9, edgecolor='#ccc')

# ── axis labels & title ───────────────────────────────────────
ax.set_xlim(-85, 80)
ax.set_ylim(-76, 12)
ax.set_title('Gate A sweep: robot placed at 360 poses\nto grade each distance metric',
             fontsize=11, pad=10)

# lateral axis label
ax.text(0, -74, '← lateral offset →', fontsize=8.5, ha='center',
        color='#888', style='italic')

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'docs', 'assets', 'sweep_diagram.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#fafafa')
print(f"saved {os.path.abspath(out)}")
