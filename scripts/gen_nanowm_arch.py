"""
NanoWM architecture diagram: frozen VAE context latents + action chunk →
Transformer (AdaLN) → predicted next latent → optional token-to-RGB decoder.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import os

fig, ax = plt.subplots(figsize=(13, 5.5))
fig.patch.set_facecolor('#fafafa')
ax.set_facecolor('#fafafa')
ax.set_xlim(0, 13)
ax.set_ylim(0, 5.5)
ax.axis('off')

FROZEN   = '#d4e6f7'
TRAINED  = '#fde8c5'
VIZ_CLR  = '#dff2df'
B_FROZEN = '#2e6da4'
B_TRAIN  = '#b8720a'
B_VIZ    = '#2e7d32'
GRAY     = '#555555'


def rbox(ax, x, y, w, h, fc, ec, label, sub=None, fs=9.5):
    r = mpatches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.12',
                                 facecolor=fc, edgecolor=ec, linewidth=1.6, zorder=3)
    ax.add_patch(r)
    cy = y + h / 2
    if sub:
        ax.text(x + w/2, cy + 0.2,  label, ha='center', va='center', fontsize=fs,
                fontweight='bold', color='#111', zorder=4)
        ax.text(x + w/2, cy - 0.22, sub,   ha='center', va='center', fontsize=7.5,
                color='#555', style='italic', zorder=4)
    else:
        ax.text(x + w/2, cy, label, ha='center', va='center', fontsize=fs,
                fontweight='bold', color='#111', zorder=4)


def arr(ax, x1, y1, x2, y2, color='#555', lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw),
                zorder=5)


# ── Frozen VAE encoder ────────────────────────────────────────────
rbox(ax, 0.2, 3.8, 2.0, 0.75, FROZEN, B_FROZEN, 'Frozen VAE Encoder', 'SD pretrained', fs=9)
ax.text(1.2, 4.9, 'Camera frames', ha='center', fontsize=8.5, color=GRAY, style='italic')
arr(ax, 1.2, 4.85, 1.2, 4.55)   # camera → encoder

# ── Context latents ───────────────────────────────────────────────
labels = [r'$z_{t-3}$', r'$z_{t-2}$', r'$z_{t-1}$', r'$z_t$']
latent_xs = [0.2, 1.2, 2.2, 3.2]
for lx, lb in zip(latent_xs, labels):
    rbox(ax, lx, 2.55, 0.85, 0.75, FROZEN, B_FROZEN, lb, fs=10)

ax.text(2.05, 3.55, 'Context latents', ha='center', fontsize=8.5, color=GRAY, style='italic')
arr(ax, 2.2, 3.8, 2.05, 3.3)    # encoder → latents

# ── Transformer block ─────────────────────────────────────────────
rbox(ax, 4.4, 1.6, 2.6, 2.2, TRAINED, B_TRAIN,
     'Transformer  (160 M)', r'AdaLN:  $\gamma(a)\cdot\mathrm{LN}(x)+\beta(a)$', fs=10)

# latents → transformer
for lx in latent_xs:
    arr(ax, lx + 0.425, 2.55, 5.7, 3.8)

# ── Action chunk → embedding ──────────────────────────────────────
rbox(ax, 0.2, 0.5, 1.7, 0.75, TRAINED, B_TRAIN, 'Action Embedding', fs=9)
ax.text(1.05, 0.2, r'Chunk $({\Delta}x,\,{\Delta}\theta)$', ha='center',
        fontsize=8.5, color=GRAY, style='italic')
arr(ax, 1.05, 0.5, 1.05, 0.25)
arr(ax, 1.9, 0.875, 4.4, 2.15)  # embedding → transformer (AdaLN input)
ax.text(3.1, 1.3, r'$\gamma(a),\,\beta(a)$', fontsize=8, color=B_TRAIN,
        style='italic', ha='center')

# ── Predicted latent ──────────────────────────────────────────────
rbox(ax, 7.3, 2.35, 1.3, 1.1, TRAINED, B_TRAIN, r'$\hat{z}_{t+1}$', 'predicted', fs=12)
arr(ax, 7.0, 2.7, 7.3, 2.9)     # transformer → predicted

# ── Token-to-RGB decoder ──────────────────────────────────────────
rbox(ax, 9.0, 2.35, 2.2, 1.1, VIZ_CLR, B_VIZ, 'Token→RGB Decoder', 'visualization only', fs=9)
arr(ax, 8.6, 2.9, 9.0, 2.9)
ax.text(10.1, 3.7, 'Planner never decodes\n— scores in token space',
        ha='center', fontsize=7.8, color=B_VIZ, style='italic')

# ── Legend ────────────────────────────────────────────────────────
legend = [
    mpatches.Patch(facecolor=FROZEN, edgecolor=B_FROZEN, label='Frozen / pretrained'),
    mpatches.Patch(facecolor=TRAINED, edgecolor=B_TRAIN,  label='Trained on robot data'),
    mpatches.Patch(facecolor=VIZ_CLR, edgecolor=B_VIZ,    label='Visualization only'),
]
ax.legend(handles=legend, loc='lower right', fontsize=8.5,
          framealpha=0.92, edgecolor='#ccc')

ax.set_title('NanoWM Architecture', fontsize=13, fontweight='bold',
             color='#111', pad=6)

plt.tight_layout()
out = os.path.join(os.path.dirname(__file__), '..', 'docs', 'assets', 'nanowm_arch.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#fafafa')
print(f"saved {os.path.abspath(out)}")
