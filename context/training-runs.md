# Training Runs

Append-only telemetry log for **training runs**, maintained primarily by the pod-side agent (see
[[runpod-operator-guide]]). This is distinct from [[experiment-log]] (design-decision chronology):
this file is the operational record of *what was trained, on what, and how it went*.

Add a new entry at the top for each run, using the template below. Keep entries factual; link wandb.

---

## Entry template (copy for each run)

```
## Run <id> — <YYYY-MM-DD>

**Status:** running | completed | failed | aborted

### Setup
- NanoWM fork SHA: <git sha>
- NanoNAV SHA: <git sha>
- Dataset: kaushikpraka/wm-smallarea_nav30 (LeRobot v2.1, 30 Hz)  |  build SHA: <sha>
- Model: nanowm_b2 (B/2, SD-VAE, v-pred, additive)
- frame_interval: 5   action_aggregation: integrate_se2   action_dim: 2
- Effective batch: 64   batch_size: <n>   grad_accum: <n>   lr: 1e-4
- max_steps: <n>   pod: 1× H100 80 GB
- wandb: <url>

### Progress
- step <n>: loss <v>, <steps/sec>, action-embed RMS <v>
- checkpoints: <paths in $RESULTS_DIR>

### Anomalies / interventions
- <timestamp> <what happened> → <action taken>

### Table 5/6 diagnostic (gate)
- GT latent-L2:     <v>
- zero latent-L2:   <v>
- random latent-L2: <v>
- action-embed RMS: <v>
- verdict: PASS | FAIL
- notes: <e.g. GT clearly < zero/random; RMS ~0.1+>

### Outcome / next
- <result, decision, next action>
```

---

<!-- New run entries go below this line, newest first. -->

## Run 001 — 2026-06-02

**Status:** aborted (stopped ~step 23K of 50K — overfitting; diagnostic FAILED at step 10K)

### Setup
- NanoWM fork SHA: 41f5c0c (+ uncommitted env/integration fixes — see below)
- NanoNAV SHA: 0e69b24 (+ uncommitted dataset-builder/script fixes)
- Dataset: kaushikpraka/wm-smallarea_nav30 (LeRobot v2.1, 30 Hz) — built locally to
  `/workspace/data/lekiwi`, 50 episodes / 44,926 frames. Built via a parallel decode-once +
  sharded-encode pipeline (~6 min vs ~45–60 min sequential); shards merged + verified to load.
- Model: nanowm_b2 (B/2, SD-VAE, v-pred, additive)
- frame_interval: 5   action_aggregation: integrate_se2   action_dim: 2
- Effective batch: 64   batch_size: 16   grad_accum: 4   lr: 1e-4
- max_steps: 50000   pod: 1× H100 80 GB
- Env: **uv venv** at `/workspace/nanowm-venv` (not conda). The upstream `environment.yml` could not
  build as written; required fixes (now in the fork working tree): lerobot 0.3.3 (the bogus
  `lerobot-datasets==2.1.0` pin); Python 3.11; torch/vision/codec 2.6.0/0.21.0/0.2.1 **+cu124**;
  diffusers 0.32.2; transformers 4.46.3; huggingface-hub <1.0; **pytorch-lightning 2.5.2** (code uses
  PL 2.x APIs; the 1.9.5 pin was stale); system ffmpeg. Plus integration fixes: factory routes
  `lekiwi` → LeRobot loader; data source uses pyav video backend (system FFmpeg 4.4 makes torchcodec
  flaky on AV1) and reads action/state from parquet (avoids per-frame video decode that made action
  stats take ~47 min).
- wandb: https://wandb.ai/kaushikpprakash-personal/nanonav/runs/x3ub
- run dir: /workspace/results/20260602_023906-NanoWM-B-2-F4S5-lekiwi

### Progress
- integrate_se2 action stats (f=5): mean=[0.0111, -0.00029], std=[0.0071, 0.0355]
- train_loss decreasing: 0.727 (step 10) → 0.30 (~step 1.3K) → 0.259 (~step 2.3K)
- throughput ~1.90 batches/s, GPU ~98% util / ~30 GB. NOTE: max_steps=50000 is *optimizer*
  steps and grad_accum=4, so the run is ~200K batches ≈ **~81 epochs ≈ ~24–26 h** (2,473 batches/
  epoch ÷ 4 ≈ 618 opt-steps/epoch). The dataset is small (50 eps, overlapping stride-1 slices) — judge
  "trained" by val_loss flattening (watch for train/val divergence = overfitting), not the step count.
- checkpoints: under the run dir (`checkpoints/latest/` + `checkpoints/across_timesteps/`)
- diagnostic auto-runs on the pod when training finishes (tmux session `diag`,
  `/workspace/diag_watcher.sh` → `/workspace/results/diag_watcher.log`)

### Anomalies / interventions
- **Overfitting (early).** val_loss bottomed ~0.248 at **step ~1,750 (epoch ~3)** then climbed to
  ~0.43 by step 23K while train_loss kept dropping (~0.07). 50 episodes is tiny for B/2.
- **Checkpoint-retention gap.** The config keeps only `latest` (rolling, every 1K) + `across_timesteps`
  (every 10K) — **no `monitor=val_loss` best checkpoint**, so the val-optimum (~step 1.75K) was lost.
  Surviving checkpoints (10K/20K/23K) are all in the overfit regime. **Fix next run: add a best-val
  ModelCheckpoint + EarlyStopping + much lower max_steps.**
- Stopped training at ~step 23K to save pod cost once it was clearly overfitting.

### Table 5/6 diagnostic (gate) — ran on the step-10K checkpoint (least-overfit survivor)
- GT final-latent L2:   **37.77**
- zero final-latent L2: 41.998
- random latent-L2:     42.391
- action-embed RMS:     **0.0088**   (threshold 0.05; paper's SD-VAE 0.1119)
- verdict: **FAIL** — GT only ~10% better than zero/random; RMS far below 0.1. Action conditioning
  is weak/atrophied. (`action_diag_10k/`.)

### Eval findings (2026-06-02, step-10K ckpt) — root cause of the FAIL
- **GT-action rollout viz** (`gt_rollout_10k/`, `src/sample/gt_rollout_viz.py`): predictions reproduce
  the context with little action-driven change; pred-vs-GT latent L2 ≈31 ≈ the real per-chunk change
  (≈30.6), i.e. barely beats "predict no change."
- **Per-chunk motion vs frame change** (`chunk_motion/`, `src/sample/chunk_motion_viz.py`, 960 chunks):
  \|Δx\| bang-bang at 0 / ~1.67 cm (p50=p95=max=1.67); \|Δθ\|~1.5°; pixel change ~4.6%; latent L2 ~30.6;
  **corr(\|Δx\|, latentL2)=0.23**. Stationary chunks' latent L2 (~10–45) overlaps full-speed chunks'
  (~13–51): **action-driven change is below the non-action latent noise floor.** Low action SNR ⇒
  the model correctly learns to mostly ignore actions. See [[open-questions]] (answered: "will the
  action branch survive").

### Outcome / next
- First checkpoint trains and the pipeline works end-to-end, but the model does **not** usefully
  action-condition → **not plan-ready**. This is a data/representation issue, not training length.
- **Update (2026-06-02, frame-interval sweep — `viz/signal-fsweep/`):** the root cause is specifically
  **translation observability**. Across f=5/8/10/15/20 (previewed with no retraining), `corr(|Δx|,
  SD-VAE latentL2) ≈ 0` at every f (the elevated ~55° camera de-magnifies forward motion), while
  `corr(|Δθ|, latentL2) ≈ 0.64–0.70` (rotation is well observed). **⇒ the f=8–10 plan below is
  REFUTED** — raising f grows Δx 4× but leaves the correlation at ~0.
- **Next (revised):** ~~retrain at f=8–10~~ → instead, **change the camera/representation to restore
  translation observability** (re-tilt/relocate camera for parallax, and/or add pose/odometry auxiliary
  conditioning for Δx), raise capture SNR (exposure/WB lock, avoid lossy AV1), then re-collect/retrain
  and re-run the diagnostic. See [[experiment-log]], [[open-questions]]. Do NOT just train longer.
