"""
Diffusion Forcing concept diagram.

Two panels:
  Top — Standard diffusion: all frames share one noise level.
  Bottom — Diffusion Forcing: each frame gets its own independent noise level,
           enabling causal autoregressive rollout at inference.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

fig, axes = plt.subplots(2, 1, figsize=(11, 6.5))
fig.patch.set_facecolor('#fafafa')

CONTEXT_CLR = '#d4e6f7'
TARGET_CLR  = '#fde8c5'
B_CTX       = '#2e6da4'
B_TGT       = '#b8720a'
NOISE_CLR   = '#e06060'
GRAY        = '#555'

N_FRAMES = 6


def draw_panel(ax, title, noise_levels, labels, is_df=False):
    ax.set_facecolor('#fafafa')
    ax.set_xlim(-0.5, N_FRAMES + 1.5)
    ax.set_ylim(-0.3, 2.8)
    ax.axis('off')
    ax.text(-0.3, 2.65, title, fontsize=11, fontweight='bold', color='#111', va='top')

    frame_w = 1.0
    gap = 0.3

    for i, (sigma, label) in enumerate(zip(noise_levels, labels)):
        x = i * (frame_w + gap)
        is_context = is_df and sigma == 0.0
        is_target  = is_df and sigma > 0.0

        fc = CONTEXT_CLR if is_context else (TARGET_CLR if is_target else '#e8e8e8')
        ec = B_CTX       if is_context else (B_TGT      if is_target else '#888')

        # frame box
        rect = mpatches.FancyBboxPatch((x, 1.2), frame_w, 1.0,
                                        boxstyle='round,pad=0.07',
                                        facecolor=fc, edgecolor=ec,
                                        linewidth=1.5, zorder=3)
        ax.add_patch(rect)
        ax.text(x + frame_w/2, 1.7, f'$f_{i+1}$', ha='center', va='center',
                fontsize=11, fontweight='bold', color='#222', zorder=4)

        # noise bar underneath
        bar_h = sigma * 0.9
        if bar_h > 0:
            bar = mpatches.FancyBboxPatch((x + 0.15, 0.25), frame_w - 0.3, bar_h,
                                           boxstyle='square,pad=0',
                                           facecolor=NOISE_CLR, edgecolor='none',
                                           alpha=0.75, zorder=2)
            ax.add_patch(bar)

        # sigma label
        ax.text(x + frame_w/2, 0.08, f'$\\sigma={sigma:.1f}$',
                ha='center', va='bottom', fontsize=8.5, color=GRAY)

        # role label (DF only)
        if is_df:
            role = 'context' if is_context else 'target'
            ax.text(x + frame_w/2, 2.38, role, ha='center', fontsize=8,
                    color=B_CTX if is_context else B_TGT, style='italic')

    # noise bar axis label
    ax.text(-0.45, 0.7, 'noise\nlevel', fontsize=8, color=NOISE_CLR,
            ha='center', va='center', style='italic')

    # arrow for inference rollout (DF panel only)
    if is_df:
        xpred = N_FRAMES * (frame_w + gap) - gap + 0.05
        rect2 = mpatches.FancyBboxPatch((xpred, 1.2), frame_w, 1.0,
                                         boxstyle='round,pad=0.07',
                                         facecolor='#f5f0ff', edgecolor='#7c4dbc',
                                         linewidth=1.8, linestyle='--', zorder=3)
        ax.add_patch(rect2)
        ax.text(xpred + frame_w/2, 1.7, r'$\hat{f}_7$', ha='center', va='center',
                fontsize=11, fontweight='bold', color='#7c4dbc', zorder=4)
        ax.text(xpred + frame_w/2, 0.08, r'$\sigma: 1\!\to\!0$',
                ha='center', va='bottom', fontsize=8.5, color='#7c4dbc')
        ax.text(xpred + frame_w/2, 2.38, 'predicted', ha='center', fontsize=8,
                color='#7c4dbc', style='italic')

        ax.annotate('', xy=(xpred + 0.05, 1.7),
                    xytext=((N_FRAMES - 1) * (frame_w + gap) + frame_w + 0.05, 1.7),
                    arrowprops=dict(arrowstyle='->', color='#7c4dbc', lw=1.5))
        ax.text(xpred - 0.35, 1.92, 'predict\nnext', fontsize=8,
                color='#7c4dbc', ha='center', style='italic')


# ── Standard diffusion (all same σ) ──────────────────────────────
shared_sigma = 0.6
draw_panel(axes[0],
           'Standard Diffusion — all frames share one noise level',
           [shared_sigma] * N_FRAMES,
           [f'f_{i+1}' for i in range(N_FRAMES)],
           is_df=False)

# ── Diffusion Forcing (independent σ per frame) ───────────────────
df_sigmas = [0.0, 0.0, 0.0, 0.0, 0.0, 0.7]  # first 5 context, last is target
draw_panel(axes[1],
           'Diffusion Forcing — each frame gets its own noise level',
           df_sigmas,
           [f'f_{i+1}' for i in range(N_FRAMES)],
           is_df=True)

# ── Shared legend ─────────────────────────────────────────────────
legend = [
    mpatches.Patch(facecolor=CONTEXT_CLR, edgecolor=B_CTX,  label='Context frame (σ = 0, clean)'),
    mpatches.Patch(facecolor=TARGET_CLR,  edgecolor=B_TGT,  label='Target frame (being denoised)'),
    mpatches.Patch(facecolor='#f5f0ff',   edgecolor='#7c4dbc', label='Predicted frame (next step)'),
]
fig.legend(handles=legend, loc='lower center', ncol=3, fontsize=8.8,
           framealpha=0.92, edgecolor='#ccc', bbox_to_anchor=(0.5, -0.01))

fig.suptitle('Diffusion Forcing vs Standard Diffusion', fontsize=13,
             fontweight='bold', color='#111', y=1.01)

plt.tight_layout(h_pad=1.2)
out = os.path.join(os.path.dirname(__file__), '..', 'docs', 'assets', 'diffusion_forcing.png')
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#fafafa')
print(f"saved {os.path.abspath(out)}")
