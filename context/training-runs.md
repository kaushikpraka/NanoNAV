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

## Run 002 — 2026-06-03

**Status:** ✅ **completed** — trained the full **12,000 steps** (~20 epochs) at **f=10**. Survived
three separate crashes (wandb-key / FID-metric / CUDACallback), each fixed + pushed. The
**cross-checkpoint rollout evaluation** (step 4125 / 6K / 8K / 10K / 12K — gate + motion rollouts +
GT-vs-pred videos) is **in progress**; results to be appended.

**Goal:** pass the Table 5/6 action-conditioning gate. Run 001 failed for **training** reasons, not
observability (see Run 001 CORRECTION + `viz/stationary-vs-translation/`). Run 002 changes the two
things that caused the failure (plus a weak val monitor) and nothing else, so the result is interpretable.

### Setup (actual)
- NanoWM fork SHA **e55fc17**; NanoNAV SHA **5910af6**.
- Dataset: kaushikpraka/wm-smallarea_nav30 (LeRobot v2.1, 30 Hz), `/workspace/data/lekiwi` (50 eps).
- Model: nanowm_b2 (B/2). **SD-VAE `sd-vae-ft-mse` is frozen pretrained; the 160M transformer is
  trained from scratch** (`pretrained: null`). v-pred, **additive** injection (`x = x + action_emb`).
- frame_interval **10**, integrate_se2, action_dim 2. Eff-bs 64 (bs 16 × grad_accum 4), lr 1e-4,
  max_steps 12000, 1× H100, bf16.
- Val: 5 held-out eps, **seeded fixed subset of 256 windows**; validated every 250 *batches*
  (≈ every ~62 opt-steps — Lightning counts `val_check_interval` in batches, not opt-steps).
- Checkpointing: across_timesteps(1000) + latest(500) + **best_val(val_loss, top-3)**.
- wandb `jh56pz5q` (steps 0→5000) → `6c72` (final resume 5375→12000). Run dirs:
  `20260603_041515` (orig) → `_154312` (resume 1) → `_160326` (resume 2, completed).

### Progress
- val_loss bottomed **0.2047 at step 4125 (epoch 6)**, then rose to ~0.22–0.226 by ~step 10.5K
  (denoising-loss overfit). train_loss ~0.13–0.15. **For diffusion-forcing, val_loss is a weak
  rollout-quality proxy** — so we deliberately trained the full session and judge by *rollouts*
  (cross-checkpoint eval), not the val curve.
- best-val global optimum = `best-epoch=6-step=4125-val_loss=0.2047.ckpt` (in `_041515`).
- across_timesteps checkpoints 6K/7K/8K/9K/10K/11K/12K (in `_160326`).

### Anomalies / interventions (3 crashes — all fixed + pushed)
1. **Warmup crash — wandb "No API key".** The key lived in the root-FS `~/.netrc`, wiped by the pod
   restart (only `/workspace` persists). Fix: persist `WANDB_API_KEY` in `/workspace/secrets/env.sh`,
   sourced by `run_training.sh`. See [[persistent-secrets]].
2. **Step-5000 crash — FID metric.** `pytorch_fid.calculate_frechet_distance` → scipy ≥1.17 `sqrtm`
   `disp` deprecation → `ValueError` propagated out of `trainer.fit`. Fix: try/except guard around
   FVD/FID in `callbacks.py` (an auxiliary metric must never kill training). Commit `de85260`.
3. **First-resume crash (~step 5570) — CUDACallback.** Native resume drops in mid-epoch, so
   `on_train_epoch_start` never ran → `on_train_epoch_end` hit `AttributeError: start_time`. Fix:
   `hasattr` guard. Commit `e55fc17`.
- **Native resume added** to finish the run: `experiment.ckpt_path` + `trainer.fit(ckpt_path=...)`
  (restores optimizer/LR/global_step/loops) — distinct from `resume_from_checkpoint` (warm-start only,
  restarts the step counter). `run_training.sh` gained a `RESUME_CKPT` env var. Resumed to 12000.

### Table 5/6 diagnostic — on the step-4125 best-val checkpoint (n_batches 16, seed 42)
- GT final-latent L2 **36.12**, zero **40.72**, random **45.23**; action-embed RMS **0.0089**.
- verdict: **FAIL on the RMS>0.05 gate** — BUT a clean, widening **gt < zero < random** separation
  (random now distinctly *worse* than zero, unlike Run 001 where zero≈random): the model **responds to
  action content**. RMS 0.0089 ≈ Run 001's 0.0088 across two very different checkpoints ⇒ **RMS looks
  architecturally pinned / mis-calibrated for this 2-D additive embedder**, not a live training signal.
  The separation + motion-tracking are the meaningful signals.
- Motion rollouts (`motion_rollout_viz.py`, new tool): the model **tracks real translation (+10 cm),
  rotation (+28°), and arc** motion in the correct direction; error grows with the horizon and is
  largest for big rotations (whole-FOV sweep). The action branch is functioning.

### Outcome / next
- Pipeline is now robust (3 crash classes fixed + pushed). Action branch is **alive and
  action-sensitive** at the val-best checkpoint — materially better than Run 001 — though the legacy
  RMS gate still reads FAIL (now believed mis-calibrated).
- **In progress:** cross-checkpoint rollout eval (4125/6K/8K/10K/12K) to answer *does more training
  improve rollout quality* with a measured metric-vs-step curve + GT-vs-pred videos, and to pick the
  checkpoint for the CEM/MPC planner. Results to be appended.

### What changes from Run 001 (and why)
1. **`frame_interval` 5 → 10.** Translation's SNR over the non-action latent floor is only ~1:1 at f=5
   but **~1.6:1 at f=10** (AUC 0.94 → 0.98); rotation stays strongly observable. f=10 (Δx≈3.33 cm,
   333 ms/chunk) is the sweet spot — f=20 barely improves AUC for double the chunk duration (coarser
   control, worse Δy-negligible approximation). Also ~doubles CEM reach per step (H=3 → ~10 cm).
   *No dataset rebuild needed* — f is a dataloader knob (`integrate_se2`); only `dataset.frame_interval`
   changes.
2. **Capture the best-val checkpoint + stop early.** Run 001's val bottomed ~epoch 3 (~1.8K steps) then
   overfit for 20K+ more steps, and the config kept **no** best-val checkpoint, so the optimum was lost
   and an overfit (epoch-16) model was diagnosed. Fixes:
   - Add a **`val_loss`-monitored `ModelCheckpoint`** (`monitor=val_loss, mode=min, save_top_k=3`) —
     small change in `src/experiments/train_experiment.py` gated by a `checkpointing.best_val` config block.
   - Checkpoint much more often: `across_timesteps` 10000→1000, `latest` 1000→500.
   - `val_every_n_steps: 1000 → 250` (50 eps is tiny; epochs are ~600 steps, so sample val finely).
   - `max_steps: 50000 → ~12000` (~20 epochs) — enough to pass the val bottom with margin; best-val
     callback keeps the optimum regardless.
   - Keep eff-bs 64, lr 1e-4, B/2, SD-VAE, v-pred, additive injection — **unchanged** (isolate the f +
     checkpoint variables).
3. **Fix the val monitor itself (it was non-representative).** See the next subsection — without this,
   best-val selection keys off a weak signal.

### Validation set + monitor (NEW approach)
- **Val set:** `split_ratio 0.9` → **5 held-out episodes** (~896 frames ≈ 30 s each); ~**4,330**
  exhaustive stride-1 windows at f=10 (4,405 at f=5).
- **Run 001's flaw:** `lerobot/base.yaml` sets `validation_size: 32`, and `train_experiment.py:670–677`
  truncates the val set to the **first 32 slices in index order** — i.e. 32 near-duplicate windows from
  the opening ~0.5 s of *one* episode. So `val_loss` was non-representative and a poor basis for picking
  a "best" checkpoint.
- **Fix (use the existing `_apply_fixed_validation_subset`, lines 484–541):**
  ```yaml
  dataset.loader:
    validation_size: null                 # OFF — stop the first-32 in-order truncation (runs AFTER the
                                          #       fixed subset, so it would otherwise re-chop it to 32)
    validation_fixed_subset_size: 256      # seeded random slices spread across all 5 val episodes
    validation_fixed_subset_seed: 42       # reproducible; persisted to run_dir/validation_subset.json
  ```
  Gives a **stable** (same 256 windows scored every checkpoint), **representative** (random across all 5
  episodes + all motion types), **reproducible** (seed + JSON, comparable across runs) monitor — the
  precondition for `monitor=val_loss` to mean anything. Cost: 256/16 = **16 val batches**, so validating
  every 250 steps is negligible. (Requires `val_slice_mode: exhaustive` — lekiwi already is.)
  - Caveat: `val_loss` still has residual timestep/noise sampling variance (diffusion loss at random t);
    the fixed subset removes the dominant *data* variance and 256 slices damp the rest. If the val curve
    is too jittery to pick a clean min, seed per-val noise/timesteps (deeper change, only if needed).
  - Untouched: `evaluation.validation_size: 32` (FVD/FID + video sampling path) and the action
    diagnostic's own held-out rollouts — different mechanisms.

### Diagnostic upgrade (run on the **best-val** checkpoint, not latest)
Extend `src/sample/action_diagnostic.py` to report **per-component** sensitivity: GT vs **Δx-zeroed**
(rotation only) vs **Δθ-zeroed** (translation only) vs all-zero vs random. Confirms *which* component the
action branch grounds — we expect rotation healthy and want to verify translation is now non-trivial.
Pass requires overall action-embed **RMS ~0.1+** and GT clearly beating zero/random.

### Contingencies (only if Run 002 still fails the gate)
- Translation sensitivity still ~0 but rotation healthy → the near-field-floor signal isn't being used:
  try **cross-attention action injection** or larger action-embed dim (open-questions fallbacks #2/#3);
  consider an **action-balanced loss** (rotation dominates raw latent change 44 vs 31 vs 12 floor).
- Persistent overfit dominates the result → **collect more episodes** (open-questions: upper bound
  ~130–190K transitions/room) and/or add augmentation (fallback #4).
- Only then revisit **camera/odometry** changes (open-questions fallback #1) — now a last resort.

### Implementation checklist — ✅ DONE (commits de85260 / e55fc17, NanoNAV 5910af6)

> All of A–F below were executed. Retained as the original spec / record of what was built.

**A. Code — best-val checkpoint.** In `external/nanowm/src/experiments/train_experiment.py`, after the
`latest_checkpoint` append (~line 819), add (backward-compatible via `.get` → old configs still work):
```python
best_val_cfg = ckpt_cfg.get("best_val", None)
if best_val_cfg is not None and best_val_cfg.get("enable", True):
    best_val_checkpoint = ModelCheckpoint(
        dirpath=os.path.join(checkpoint_dir, "best_val"),
        filename=best_val_cfg.get("filename", "best-{epoch}-{step}-{val_loss:.4f}"),
        monitor=best_val_cfg.get("monitor", "val_loss"),      # logged at train_experiment.py:295
        mode=best_val_cfg.get("mode", "min"),
        save_top_k=best_val_cfg.get("save_top_k", 3),
        every_n_train_steps=best_val_cfg.get("every_n_train_steps", None),  # None → saves at each val end
        save_on_train_epoch_end=False,
        save_weights_only=best_val_cfg.get("save_weights_only", False),
    )
    callbacks_list.append(best_val_checkpoint)
```
Cadence = `val_every_n_steps` (already wired to Trainer `val_check_interval`, line 823).

**B. Config — `src/configs/dataset/lerobot/lekiwi.yaml`:** `frame_interval: 5 → 10`; in `loader:` add
`validation_size: null`, `validation_fixed_subset_size: 256` (seed 42 is inherited).

**C. Config — `src/configs/experiment/lekiwi_nav.yaml`:** set `max_steps: 12000`,
`val_every_n_steps: 250`, and a `checkpointing:` override block:
`across_timesteps.every_n_train_steps: 1000`, `latest.every_n_train_steps: 500`, and a new
`best_val: {enable: true, monitor: val_loss, mode: min, save_top_k: 3, filename: "best-{epoch}-{step}-{val_loss:.4f}"}`.

**D. Diagnostic upgrade (optional, do before re-gating):** per-component sensitivity in
`src/sample/action_diagnostic.py` (see "Diagnostic upgrade" above).

**E. Launch:** `bash scripts/run_training.sh` in tmux (env: uv venv `/workspace/nanowm-venv`,
`LEKIWI_DATA_ROOT=/workspace/data/lekiwi`). Monitor wandb `nanonav`. ~20 epochs on 1× H100.

**F. Re-gate:** run `action_diagnostic.py` (+ chunk/stationary probes) on the **best-val** checkpoint
(`<run>/checkpoints/best_val/`), NOT latest. Pass = action-embed RMS ~0.1+, GT clearly < zero/random,
and (new) non-trivial Δθ-zeroed (translation-only) sensitivity. Then fill the telemetry template above.

### Setup (pre-launch spec) — superseded by "Setup (actual)" at the top of this entry
- frame_interval: **10**   action_aggregation: integrate_se2   action_dim: 2
- Effective batch: 64 (batch_size 16 × grad_accum 4)   lr: 1e-4   max_steps: ~12000
- val: 5 eps, fixed seeded subset 256   val_every_n_steps: 250
- checkpointing: across_timesteps(1000) + latest(500) + **best_val(val_loss, top-3)**
- pod: 1× H100 80 GB   wandb: jh56pz5q → 6c72   fork SHA e55fc17 / NanoNAV 5910af6

---

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
- **⚠️ CORRECTION (2026-06-03):** the "translation observability" diagnosis above is **wrong**. A
  controlled stationary-vs-translation contrast (`viz/stationary-vs-translation/`,
  `src/sample/stationary_vs_translation.py`) shows pure-translation chunks change the SD-VAE latent ~2×
  more than stationary (AUC 0.94 @ f=5 → 0.98 @ f=10; clean dose-response; near-field-floor footprint).
  The `corr(|Δx|,·)≈0` and the `0.23` above were artifacts of bang-bang Δx + pooled rotation chunks.
  **The FAIL was a TRAINING artifact:** the step-10K (epoch-16) checkpoint was deep into overfitting
  (val bottomed ~epoch 3; no best-val ckpt was kept) **at f=5**, where translation's signal only ≈ the
  noise floor (~1:1; 1.6:1 at f=10). **⇒ retry as Run 002 (f=10, best-val checkpointing) — no camera
  change needed.** See the Run 002 plan below and [[experiment-log]] (2026-06-03).
