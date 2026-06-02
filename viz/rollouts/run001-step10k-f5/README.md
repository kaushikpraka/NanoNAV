# GT-action rollout — Run 001, step-10K checkpoint (f=5)

Model rollout visualizations produced by `external/nanowm/src/sample/gt_rollout_viz.py`.
Relocated here from the (gitignored) pod run dir
`/workspace/results/20260602_023906-NanoWM-B-2-F4S5-lekiwi/gt_rollout_10k/`.

## Source
- **Run:** 001 (wandb `x3ub`) — see [[../../context/training-runs.md]].
- **Checkpoint:** `epoch=16-step=10000.ckpt` (the least-overfit survivor; val_loss bottomed ~step 1.75K).
- **Config:** NanoWM-B/2, SD-VAE, v-pred, `integrate_se2`, `frame_interval=5`, `action_dim=2`,
  `n_context_frames=1`, rollout `horizon=5`.

## Files
- `sample<i>_grid.png` — top row **GT** (real frame → SD-VAE → decode), bottom row **Pred**
  (world-model rollout under GT actions). Columns: context, t+1 … t+H. Annotated with per-frame
  latent-L2(pred, GT).
- `sample<i>_cmp.mp4` — same, as a side-by-side GT | Pred video over time.

## What they show (the Run 001 failure, visually)
The prediction barely diverges from the context frame: pred-vs-GT latent-L2 ≈ 29–31, which is
≈ the *real* per-chunk latent change (≈30.6 from `chunk_motion_stats.json`). The model is close to
"predict no change" — consistent with the FAILED action diagnostic (action-embed RMS 0.0088 ≪ 0.1,
GT 37.8 vs zero 42.0 / random 42.4). Root cause is weak action SNR (per-chunk motion ~1.67 cm sits
below the non-action latent noise floor). See `context/training-runs.md` and `context/open-questions.md`.
</content>
</invoke>
