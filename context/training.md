# Training

## Model Configuration

| Parameter | Value | Source |
|---|---|---|
| Backbone | NanoWM-B/2 (160M params) | Paper default for planning |
| Latent space | SD-VAE (stabilityai/sd-vae-ft-mse) [4, 32, 32] | Action-sensitive (Finding #4) |
| Patch size | 2 → 16×16 = 256 spatial tokens/frame | Paper naming convention |
| Objective | v-prediction | Best FID (paper finding) |
| Schedule | Squared-cosine + ZTSNR | Paper default |
| Action injection | Additive | Best quality-per-param on PushT |
| Masking | Causal | Paper default |
| Context frames | 1 | Avoid multi-frame action decorrelation |
| Predicted frames (H) | 3 | Paper default for planning |
| Frame interval (f) | 5 | 167ms chunks at 30 Hz |
| Action dim | 2 (Δx, Δθ) | See [[action-representation]] |

## Training Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 0.01 |
| Training steps | 50-100K (start conservative) |
| Effective batch size | 64 |
| DDIM steps (eval) | 20 for planning, 250 for quality metrics |

## Latent Space Options in NanoWM

Three encoders available. SD-VAE is chosen for action-conditioned planning:

**SD-VAE [4, 32, 32]:** Reconstruction-oriented (MSE + perceptual + GAN). Encodes pixel-level detail. Small camera motions produce measurable latent changes → action branch stays alive. Decodable to RGB for free visualization. 25% PushT planning success, action RMS ~0.11.

**Web-DINO (DINOv2) [1024, 16, 16]:** Semantic/geometric features. Trained to be INVARIANT to viewpoint changes — the exact information the action branch needs. Action branch atrophies (RMS ~0.002), 0% planning success.

**V-JEPA 2.1 [1024, 16, 16]:** Video-pretrained predictive features. Same failure as DINO in NanoWM's framework (0% planning, RMS ~0.002), despite video pretraining. Note: V-JEPA 2.1 works for planning in other architectures (CDiT), so the failure is framework-specific, not inherent.

## Validation Gate: Table 5/6 Diagnostic

Run BEFORE building the planner. Takes one afternoon.

**Procedure:**
1. Take held-out episodes
2. Roll out under ground-truth actions, zero actions, and random actions
3. Measure final-latent L2 distance to actual next observation
4. Check action-embedding RMS

**Pass → proceed to planning:**
- GT actions clearly produce closer predictions than zero/random
- Action embedding RMS ~0.1+

**Fail → fix training first:**
- Random ≈ GT distance (all three conditions score similarly)
- RMS ~0.002 (action branch atrophied)
- No planner will save a model that ignores actions

> **⚠️ RMS threshold is mis-calibrated for the NanoNAV 2-D additive setup (Run 002, 2026-06-03).** The
> ~0.1 / ~0.002 numbers are the paper's PushT values. NanoNAV's healthy, action-sensitive Run 002 model
> reports **RMS ≈ 0.0089** — essentially identical to the *failed* Run 001 (0.0088) across two very
> different checkpoints — i.e. RMS appears **architecturally pinned** by the 2-D additive embedder
> (`x = x + action_emb`), not a live training signal. **Use the rollout signals as the real gate:** the
> magnitude of the **gt < zero < random** separation (does corrupting the action clearly hurt?) and
> **motion-tracking fidelity** on real-motion chunks (`motion_rollout_viz.py`). See [[training-runs]]
> (Run 002) and [[open-questions]].

## Visual Evaluation (richer than Table 5/6)

Decode predicted latents from GT-action rollouts and compare side-by-side with real video:

**Frames 1-5 (training window):** should be sharp, spatially accurate. Robot body fixed, floor slides correctly, objects approach/recede at right rate.

**Frames 5-10:** texture degradation expected (Finding #5). Scene layout should still be correct. Acceptable for CEM scoring.

**Beyond frame 15:** coarse geometry starts drifting. MPC never operates here — replans every step.

**Key comparison:** plot latent L2 to ground truth for all three conditions on the same axis. GT curve should be clearly below zero/random for the first ~10 frames.

## SD-VAE Exploration Tool

`explore_sdvae_latents.py` — script for visualizing the latent space:
- `channels`: visualize 4 latent channels of an image
- `compare`: L2 distance + difference heatmap between two frames
- `trajectory`: distance matrix + drift over a frame sequence
- `interpolate`: linear interpolation in latent space, decoded
- `roundtrip`: encode → decode quality check

Key experiment: `compare` two frames separated by one chunk (167ms of driving). If L2 distance is clearly above noise, the action branch has signal to learn from.

## Compute

Training: ThunderCompute / 4×H100 (from prior LeHome Challenge work).
Eval/sim: GCP g2-standard-12 (L40S) or AWS g6e.2xlarge (L40S).
