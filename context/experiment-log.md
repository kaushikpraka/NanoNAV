# Experiment Log

## 2026-06-01 — Initial Design Session

### Decisions Made

**Action representation:** Settled on body-frame pose delta (Δx, Δθ). Worked through the full derivation from unicycle kinematics. Key insight: heading invariance means the same physical motion always produces the same action vector regardless of world-frame orientation. Rejected world-frame delta (breaks composability), raw velocity (constant during cruise → decorrelation risk), and velocity-delta/acceleration (zero during constant-speed cruise).

**Δy dropping:** Traced through the integration math for combined v_x + ω inputs. At typical speeds and chunk durations (167ms), Δy ≈ Δx · sin(Δθ/2) ≈ 1-2mm. Visual significance is ~0.3° vs 6-8° for kept components. Breaks down only at very aggressive turn rates (ω > ~2 rad/s). Built interactive visualizer to confirm.

**Camera choice:** Elevated third-person view from overhead mount (~55° tilt). NOT a straight-down camera. Four depth zones (robot body, near floor, mid objects, far walls) provide rich parallax signal. Fixed robot body in frame acts as ego-motion reference, strengthening action conditioning.

**Data paradigm:** General exploratory driving, not task demonstrations. Following NanoWM/DINO-WM precedent — both train on random policy data. Task enters only at inference via goal image + CEM. Suboptimal trajectories are valuable because CEM needs to evaluate and reject bad candidates.

**Latent space:** SD-VAE chosen over DINO/V-JEPA. Finding #4 shows semantic latents fail at action conditioning in NanoWM (action RMS → 0.002, 0% planning success). SD-VAE preserves pixel-level detail that action branch needs to stay alive.

**Planning architecture:** Stop-and-plan MPC with CEM. ~1-2s per replan acceptable for prototype. Waypoint scaffold needed for long-range goals (CEM scoring is flat beyond ~30cm). Topological graph from data + DepthAnything3 reconstruction is the recommended approach.

### Artifacts Created

- `nanowm-lekiwi-nav.md` — consolidated design document
- `explore_sdvae_latents.py` — SD-VAE latent space exploration tool (channels, compare, trajectory, interpolate, roundtrip)
- `delta-y-visualizer.jsx` — interactive visualization of Δy dropping logic
- This `context/` directory, tracked in git (overview, action-representation, data-collection, training, planning, experiment-log, open-questions)

## 2026-06-01 — Pose Integration Validation

Built `scripts/nav_integration.py` (the single source of truth for SE(2) integration, to be mirrored
by the dataset builder and the `integrate_se2` dataloader patch) + `scripts/visualize_integration.py`,
and ran them on the real velocities from `wm-smallarea_merged` (the 835 KB tabular parquet, no video).
Figures in `viz/`. The integration is **validated**, and the visualization surfaced data
characteristics that matter downstream:

- **`theta.vel` units = degrees/second**, NOT rad/s. Decisive: integrating as deg/s yields smooth
  ~130°-total exploratory paths; as rad/s the same episode spirals to 7528° (21 rotations). The
  integrator converts deg→rad. (`y.vel` is all-zero — strafe confirmed absent.)
- **Δy is negligible** — max 0.58 mm, 99th-pct 0.44 mm across 8,982 chunks (f=5). Even smaller than
  the design's 1–2 mm estimate. The "drop Δy" decision is firmly justified (see [[action-representation]]).
- **World-frame trajectories are smooth and plausible** within a ~1–2 m extent (consistent with the
  2×2 m room) — coherent arcs/loops, the diverse exploration the collection plan intended.

Two findings with **planning implications** (flagged in [[open-questions]]):
- **Forward speed is near bang-bang.** Δx per chunk is strongly bimodal — a spike at 0 (stationary)
  and a spike at ~1.65 cm (full speed, x.vel≈0.1 m/s), with sparse intermediate values. Little
  fine-speed coverage → the low-Δx regime needed for near-goal approach is thin.
- **Reach is shorter than assumed.** Max Δx ≈ 1.65 cm/chunk (not the design's ~5 cm), so an H=3 rollout
  covers ~5 cm, not ~15 cm. Strengthens the case for the f=8–10 experiment and the waypoint scaffold.

## 2026-06-01 — Implementation: dataset builder, NanoWM patch, configs, diagnostic

Built the full Stage 3–5 toolchain. Validated everything testable without a GPU/torch (compile,
hydra-compose, numpy-equivalence); the rest is pod-run.

- **Fork + submodule:** `KaushikTheProgrammer/nano-world-model` added at `external/nanowm` (pinned).
- **`scripts/build_lekiwi_nav_dataset.py`** (NanoNAV): v3.0→v2.1, top camera, 2-D SI action
  `[x.vel, omega_rad]`, 30 Hz. Reads raw (pandas + PyAV) so only the writer needs lerobot 2.1.0 —
  no version clash. Single-pass decode validated against episode metadata (50 eps, one contiguous
  av1 file, 44,926 frames).
- **`integrate_se2` patch** (fork, `world_model_dataset.py` + `models/__init__.py`): additive,
  default stays `concat`. Integrates per-step velocities → `(Δx, Δθ)` (mirrors `nav_integration.py`,
  matched to ~1e-9), f-dependent stats computed fresh, model action_dim = 2. Threaded through all
  three dataset factories.
- **Configs** (fork): `dataset/lerobot/lekiwi.yaml` + `experiment/lekiwi_nav.yaml`. Full chain
  verified by hydra-compose (integrate_se2, action_dim 2, f=5, eff-bs 64, v-pred + ZTSNR).
- **`context/runpod-setup.md`**: bring-up runbook for the pod-side Claude (install prerequisites →
  clone+submodule → conda env → build dataset → launch under tmux+wandb). Markdown runbook rather than
  a rigid script, so the agent adapts to whatever the RunPod template provides.
- **`src/sample/action_diagnostic.py`** (fork): GT/zero/random rollouts, final-latent L2,
  action-embed RMS, PASS/FAIL. Reuses `DiffusionWorldModel.rollout`.

Pending (pod): run the dataset build, train, run the diagnostic. Fork changes must be committed +
pushed to GitHub before the pod clones them.

## 2026-06-02 — RunPod bring-up: env repair, dataset build, training launched (Run 001)

Brought up a fresh RunPod H100 and got NanoWM-B/2 training. The upstream `environment.yml` was
unbuildable and the LeKiwi path had integration gaps; the fixes are committed to the fork `main`
(`nano-world-model`) and NanoNAV `main`. Full operational detail in [[training-runs]] Run 001;
realized integration summary in [[nanowm-integration]]; env reality in [[runpod-setup]].

- **Env (uv, not conda).** Once torch went to pip cu124 wheels, conda only provided Python, so the
  env is a uv venv (`/workspace/nanowm-venv`). Repaired pins: `lerobot==0.3.3` (the
  `lerobot-datasets==2.1.0` pin is a non-existent package); `python=3.11`; torch/vision/codec
  `2.6.0/0.21.0/0.2.1+cu124`; diffusers `0.32.2`; transformers `4.46.3`; `huggingface-hub<1.0`;
  **`pytorch-lightning==2.5.2`** (the code uses PL 2.x APIs — the 1.9.5 pin was stale, like
  `lerobot-datasets`); system `ffmpeg`.
- **Dataset built** to `/workspace/data/lekiwi` (50 eps / 44,926 frames, loads + decodes). Builder
  needed two lerobot-0.3.3 fixes (`add_frame(task=...)`, tuple feature shapes). Added a parallel
  **decode-once → sharded-encode → merge** path (~6 min vs ~45–60 min; verified byte-identical).
- **Integration fixes:** factory routes `lekiwi` → LeRobot loader; data source forces the **pyav**
  video backend (system FFmpeg 4.4 makes torchcodec flaky on AV1) and reads action/state from the
  parquet (the old per-frame video decode made action stats take ~47 min → now seconds).
- **Training: Run 001 running** — NanoWM-B/2, `integrate_se2`, f=5, eff-bs 64, 50K steps, 1× H100,
  bf16. ~1.9 batches/s; 50K *optimizer* steps × grad_accum 4 ≈ ~81 epochs ≈ ~24–26 h. Loss
  decreasing (0.73 → 0.15 by epoch 6). wandb run `x3ub`.
- **Diagnostic scheduled on the pod** (tmux `diag`): waits for training to finish, then runs
  `action_diagnostic.py` on the final checkpoint (a remote `/schedule` agent can't reach the pod's
  checkpoint/GPU).

## 2026-06-02 — Eval session: overfitting, Table 5/6 FAIL, and the root cause

Stopped Run 001 early (cost + overfitting) and evaluated the step-10K checkpoint. Full numbers in
[[training-runs]] Run 001; design implications in [[open-questions]].

- **Overfitting, early.** val_loss bottomed ~0.248 at step ~1.75K (epoch ~3), rose to ~0.43 by 23K.
  The 50-episode set is tiny for B/2; the paper's analogous small domains used only 15–30K steps (vs
  our 50K = ~81 epochs). The checkpoint config kept no best-val checkpoint, so the optimum was lost
  → next run needs `monitor=val_loss` + EarlyStopping + lower max_steps.
- **Action diagnostic: FAIL.** RMS 0.0088 (need ~0.1+); GT 37.8 vs zero 42.0 / random 42.4.
- **Root cause (quantified).** Built `gt_rollout_viz.py` (decode GT-action rollouts) and
  `chunk_motion_viz.py` (per-chunk motion vs frame/latent change). Over 960 chunks: \|Δx\| is bang-bang
  at 0/1.67 cm, **corr(\|Δx\|, SD-VAE latentL2)=0.23**, and stationary chunks change the latent (~10–45)
  about as much as full-speed chunks (~13–51). The action signal sits **below the non-action latent
  noise floor** → the model correctly learns to ignore actions. The world model's prediction error
  (latentL2≈31) ≈ the real per-chunk change (≈30.6).
- **Conclusion:** data/representation SNR problem, not training length. Highest-leverage fix is
  **frame_interval 8–10+** (more motion per chunk), re-running the diagnostic at each f.
- **Tooling fixes (committed to the fork):** `action_diagnostic.py` (missing `sys.path`; `${hydra:}`
  resolver stub so saved configs load standalone), `sampling_utils.py` (same resolver), and two new
  eval scripts.

## 2026-06-02 — Frame-interval sweep: translation is unobservable, rotation is — refutes "raise f"

Tested the roadmap's "retrain at f=8–10" hypothesis *before* spending GPU on it, by previewing the
per-chunk SD-VAE latent change across **f = 5/8/10/15/20** with no retraining (`chunk_motion_viz.py`
now takes `--frame-interval`; the checkpoint supplies only the frozen SD-VAE + config). Then, prompted
by the question "does the high camera mount dampen image change?", split the signal by **action
component** (`corr(|Δx|, latentL2)` vs `corr(|Δθ|, latentL2)`) and surfaced diverse drive/rotate/arc
example chunks (de-duped by episode+time). All measured in **SD-VAE latent space** (latL2 = ‖Δz‖_F),
the quantity v-prediction is trained on. Figures + numbers in `viz/signal-fsweep/` (README has the table).

- **Translation (Δx) is essentially invisible to this camera: `corr(|Δx|, latentL2) ≈ 0` at every f**
  (−0.04 … +0.04). A full-speed forward chunk (Δx=3.33 cm @ f=10) moves the latent ~latL2 27; raising
  f to 20 grows Δx 4× but leaves the correlation at ~0. The elevated ~55° downward mount geometrically
  **de-magnifies forward motion**.
- **Rotation (Δθ) is strongly observable: `corr(|Δθ|, latentL2) ≈ 0.64–0.70` at every f.** Pure-rotation
  chunks (Δx=0, Δθ≈9.5°) reach latL2 ~46; arcs ~51–54. Rotation sweeps the whole wide FOV.
- **So the Run 001 action-branch failure is specifically a *translation-observability* problem**, not a
  generic SNR/training-length problem — and **`frame_interval` cannot fix it** (the latent saturates and
  the non-action floor grows with the time window). This **refutes the f=8–10 plan** as the fix.
- Highest-leverage fixes now target translation: a **lower / more forward-facing camera** (or richer
  near-field floor texture) for parallax per cm; **auxiliary odometry/pose conditioning** for Δx;
  lower the non-action floor (exposure/white-balance lock, avoid lossy AV1). See [[open-questions]].
- Correction to the prior eval note: the earlier `corr(|Δx|,latentL2)≈0.23` was a noisy small in-order
  subset; the stable seed-42 / n_batches-40 estimate (~5–7k chunks/f) is ~0.
- **Tooling (fork `chunk_motion_viz.py`):** added `--frame-interval` (preview any f w/o retraining),
  `--seed` (sample scenes across episodes), `--example-mode {mixed,forward,rotate,arc}` with
  episode+time de-dup, a `corr(|Δθ|,·)` panel, and switched the example montage's 3rd column from a
  pixel |diff| to the **SD-VAE per-cell ‖Δz‖ map**.

### Next Steps

1. ~~Set up room environment (lighting, object positions, arm parking config)~~ ✅
2. ~~Verify lerobot-record logging pipeline (camera + velocity at 30 Hz, no v_y)~~ ✅
3. ~~Collect teleop episodes with PS5 controller~~ ✅ — merged to `kaushikpraka/wm-smallarea_merged`
4. ~~Build dataset: top-camera v2.1 + body-frame delta integration~~ ✅ — `/workspace/data/lekiwi`
5. ~~Train first NanoWM-B/2 checkpoint~~ ✅ Run 001 (overfit; stopped ~23K)
6. ~~Run Table 5/6 action diagnostic~~ ✅ **FAILED** (RMS 0.0088)
7. ~~Retrain at f=8–10~~ ❌ **refuted** by the f-sweep — translation is unobservable at all f; raising f
   won't revive the action branch (rotation already is observable).
8. ~~**Decide the camera/representation fix**~~ ← **SUPERSEDED by the 2026-06-03 entry below**: the
   stationary-vs-translation contrast shows translation *is* observable, so a camera change is not
   required. The fix is a better training run (Run 002). See the entry below.

## 2026-06-03 — Stationary vs pure-translation contrast: translation IS observable — overturns the f-sweep conclusion

Prompted by "compare the SD-VAE latents for a stationary robot vs a robot translating only", ran the
**controlled** test the f-sweep's pooled `corr(|Δx|, latentL2)` could not: hold rotation near zero and
contrast the latent-change distributions of STATIONARY (`|Δx|<0.3cm, |Δθ|<0.5°`) vs PURE-TRANSLATION
(`|Δx|>1.3cm, |Δθ|<0.5°`) chunks, with PURE-ROTATION as a positive control. New tool
`external/nanowm/src/sample/stationary_vs_translation.py`; figures + JSON in
`viz/stationary-vs-translation/{f05,f08,f10,f20}/`.

**Result — translation is clearly observable; the "geometrically unobservable / below the noise floor"
conclusion is WRONG.** latentL2 = `‖z(k+f)−z(k)‖_F` (the v-pred target), seed 42, n_batches 80:

| f | stationary μ | translation μ | rotation μ | signal/floor `(μt−μs)/μs` | AUC(trans>stat) |
|---|---|---|---|---|---|
| 5 | 12.0 | 23.5 | 38.5 | 0.96× | 0.942 |
| 8 | 11.9 | 27.8 | 42.6 | 1.34× | 0.964 |
| 10| 11.9 | 30.6 | 44.4 | 1.57× | 0.978 |
| 20| 12.6 | 37.0 | 51.4 | 1.93× | 0.980 |

- **AUC 0.94–0.98**: a random forward-driving chunk out-changes a random stationary chunk 94–98% of the
  time. That is *not* below the noise floor.
- **Dose-response proves causation**: as f grows (Δx 1.67→6.65 cm) the translation signal scales
  monotonically while the stationary floor stays flat (~12). A scene/content confound cannot do that.
- **Spatial footprint is physically correct** (the `latent_compare.png` heatmaps): translation lights up
  the **near-field floor (bottom)** — parallax; rotation lights up the **far-field horizon (top)** — FOV
  sweep. The robot body (bottom-center) is static in all classes (registration sanity check).

**Why the old metric misled.** `corr(|Δx|, latentL2)≈0` is the wrong estimator: (1) `|Δx|` is bang-bang
(≈0 or ≈1.67 cm at f=5) so there's no within-moving variance to correlate; (2) pure-rotation chunks
(large latentL2 at ~0 Δx) drag the correlation to zero. **This refutes the 2026-06-02 "translation is
unobservable / raising f can't help" conclusion** — in fact raising f from 5→10 lifts translation's SNR
over the floor from ~1:1 to ~1.6:1.

**Reinterpretation of the Run 001 action-branch failure (RMS 0.0088).** It is **not** an
observability/camera problem — the signal is in the latent. The real causes are training-side and
fixable without re-collecting data: **(a)** the diagnosed checkpoint (step 10K = epoch 16) was deep into
overfitting (val bottomed ~epoch 3; no best-val checkpoint was kept), so an overfit model was measured;
**(b)** at the trained **f=5** translation's signal only ≈ the noise floor (~1:1), trivially dropped
under overfitting — at f=8–10 it's 1.3–1.6× the floor and far more learnable.

⇒ Next is **Run 002, not a camera change**: retrain at **f=10**, add a **best-val checkpoint** + low
`max_steps` so the diagnostic runs on the *best-val* model, and extend the action diagnostic to report
**per-component** (Δx-only vs Δθ-only) sensitivity. See [[roadmap]] and [[training-runs]] (Run 002 plan).

## 2026-06-03 — Run 002 (f=10) trained to completion: action branch alive, RMS gate looks mis-calibrated

Executed the Run 002 plan: NanoWM-B/2 trained to the full **12,000 steps at f=10** with best-val
checkpointing on one H100. Operational detail + telemetry in [[training-runs]] (Run 002).

**Three crashes, each fixed + pushed** (the run is now reproducible):
- **wandb "No API key"** in warmup — the key lived in the root-FS `~/.netrc`, wiped by the pod restart
  (only `/workspace` persists). Fix: persist `WANDB_API_KEY` in `/workspace/secrets/env.sh`, sourced by
  `run_training.sh`. See [[persistent-secrets]].
- **FID metric at step 5000** — `pytorch_fid` → scipy ≥1.17 `sqrtm` `disp` deprecation → `ValueError`
  propagated out of `trainer.fit`. Fix: try/except guard around FVD/FID in `callbacks.py` (an auxiliary
  metric must never kill training).
- **CUDACallback at the first resume's epoch boundary** — native (`ckpt_path`) resume drops in
  mid-epoch, so `on_train_epoch_start` never ran → `on_train_epoch_end` hit `AttributeError:
  start_time`. Fix: `hasattr` guard. Also added **native Lightning resume** (`experiment.ckpt_path` +
  `trainer.fit(ckpt_path=...)`) to finish the run — distinct from the warm-start `resume_from_checkpoint`.

**Result — the action branch is alive and action-sensitive (much better than Run 001), but the RMS gate
mis-reads.** On the val-best step-4125 checkpoint: GT 36.1 / zero 40.7 / random 45.2, RMS 0.0089.
- The **gt < zero < random separation is clean and wide** — random is distinctly worse than zero, so the
  model uses action *content*. Run 001 had zero≈random (action ignored). Decoded **motion rollouts**
  (`motion_rollout_viz.py`, new — scans the val set for high-motion chunks) show the model tracks real
  translation (+10 cm), rotation (+28°) and arc motion in the right direction, error growing over the
  horizon (largest for big rotations — whole-FOV sweep).
- **RMS 0.0089 ≈ Run 001's 0.0088** across two very different checkpoints ⇒ the action-embed RMS looks
  **architecturally pinned** (injection is additive, `x = x + action_emb`) — a **mis-calibrated gate**,
  not a live signal. The separation + motion-tracking are the metrics that actually move.

**Methodology note (diffusion-forcing):** val_loss bottomed 0.2047 at step 4125 then rose, but the
denoising val_loss is a weak proxy for rollout quality — so we trained the *full* session (not
early-stopped on val) and judge by rollouts. **In progress:** a seeded **cross-checkpoint rollout eval**
(4125 / 6K / 8K / 10K / 12K — gate + motion + GT-vs-pred videos) to measure whether more training
improves rollouts and pick the planner checkpoint.

**Architecture clarification:** the SD-VAE perception (`sd-vae-ft-mse`) is **frozen pretrained**; the
160M transformer is trained **from scratch** (`pretrained: null`). So this is a scene-specific dynamics
model on a general perceptual backbone — it generalizes to novel trajectories/goals *within* the trained
room, not across environments (single-room scope; see [[open-questions]]).
