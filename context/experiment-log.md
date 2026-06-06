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
early-stopped on val) and judge by rollouts.

**Cross-checkpoint rollout eval — result (the diffusion-forcing caveat paid off).** Seeded gate +
motion rollouts at 4125/6K/8K/10K/12K (`results/eval_run002/`). **Rollout quality is U-shaped in step:
it improves *past* the val-best (4125) to a peak at ~6K–8K, then overfitting degrades it through 12K**
(GT latent-L2 36.15 → **35.30 @ 8K** → 37.11 @ 12K; same shape for translation/rotation/arc tracking).
So **val_loss mis-ranked the checkpoints** — it called 4125 optimal, but rollouts say ~8K, and 12K
overshoots. Action separation (random−GT) stays ~10 throughout and RMS only creeps 0.0089→0.0102 (still
≪ 0.05) — the action branch is robust; the RMS gate is mis-calibrated. ⇒ **carry step-8000 into the
CEM/MPC planner** (best GT accuracy + translation + arc; step-6000 best for rotation + separation), not
the val-best or the final checkpoint. Detail + table + plot in [[training-runs]] (Run 002).

**Architecture clarification:** the SD-VAE perception (`sd-vae-ft-mse`) is **frozen pretrained**; the
160M transformer is trained **from scratch** (`pretrained: null`). So this is a scene-specific dynamics
model on a general perceptual backbone — it generalizes to novel trajectories/goals *within* the trained
room, not across environments (single-room scope; see [[open-questions]]).

## 2026-06-04 — Stage 6a: offline CEM planning eval — PASS, 6b green-lit at DDIM=3

Built `src/sample/offline_planning_eval.py` (a standalone eval, NOT a registry env — LeKiwi has no
simulator, so the sim-coupled `PlanningExperiment._run_mpc` doesn't fit; follows the Run-002 eval-tool
pattern: load ckpt+dataset directly, run the REAL `CEMPlanner` + `DiffusionWorldModel`, grade against the
dataset as a built-in answer key) + `configs/planning/lekiwi.yaml` (record/scaffold for 6b). Reuses
unchanged: `CEMPlanner` (action_dim=2), `DiffusionWorldModel.rollout/encode_obs`, `create_objective_fn`,
the integrate_se2 action stats.

**Setup:** step-8000, **35 val scenes stratified by motion** (translation=9, pivot=8, arc=9, slow=9) across
**all 5 val episodes** (cap ≤2/episode; pivot shortfall 8/9 logged — only 190 pivot slices in val), each
goal `goal_H=3` chunks (~10 cm) ahead, swept over **DDIM ∈ {20,5,3}** at the cheap CEM config (32 samples ×
3 opt × top-10). H100, ~22 min. Metrics per scene: `do_nothing` (floor), `gt_ceiling` (WM accuracy under GT
actions), `cem_reached` (WM under CEM actions), `action_recovery` (denorm CEM vs GT (Δx,Δθ)), + decoded
montages. All latent-L2 (same convention as the motion-rollout eval, so numbers are comparable ~30).

**Result — all four acceptance gates pass:**
1. **CEM beats `do_nothing` 100%** and lands near-WM-optimal: `reached_ratio = cem_reached/gt_ceiling`
   0.99–1.11 in every bucket/DDIM. The residual gap to the goal is **WM prediction error, not planner
   failure** (pivot/arc carry the larger gap, as predicted — still ≤1.11).
2. **Action recovery:** forward/turn **sign 100%** (one DDIM=5 translation mis-signed a ~4° turn → 89% in
   that cell; sign is nulled when the GT component is near-zero so a pivot's ~0 Δx isn't scored as noise),
   magnitudes small (**dxErr ~0.6–2.0 cm, dθErr ~1.1–3.4°**) — CEM re-derives the true commands.
3. **Decoded montages** (8, 2/bucket) show the CEM-planned WM rollout landing on the goal frame, including
   arc (drive+turn) and pivot (pure rotation).
4. **Cheap-sampler hold — decisive.** DDIM=3 does NOT degrade goal-reaching in any bucket — `cem_reached`
   is *slightly lower* at DDIM=3 (overall 36.5→34.4, pivot 41.9→37.2). The pivot-softening risk flagged from
   the controllability eval **did not show up in closed planning accuracy** (`gt_ceiling` also tightens at
   fewer eta=0 DDIM steps; near-deterministic futures captured in 3 steps), so `reached_ratio` stays ~1.0.
   ⇒ **the ~7 s/replan DDIM=3 / 32×3 regime is confirmed for 6b** (DDIM=5 fallback only if a turn-heavy
   on-robot task regresses).

**Caveat (honest):** val holds only 5 episodes, so spatial/landmark coverage is the dataset ceiling, not a
sampling choice; and these are **open-loop** numbers on reachable dataset goals — closed-loop success
(compounding execution error, real-robot dynamics) is 6b. Artifacts: `results/offline_planning_step8000/`
(`offline_planning_eval.json` per-scene rows + aggregates, `montages/`, `run.log`). Detail + full table in
[[planning]] "6a — RESULTS". **Stage 6a passes; the planner engine is validated; 6b (closed-loop on LeKiwi)
is green-lit.**

## 2026-06-04 — Stage 6b.0: LeKiwi transport + units bring-up — PASS (the (Δx,Δθ)→velocity contract is pinned)

Ran `scripts/lekiwi_transport_check.py` (Mac as lerobot `LeKiwiClient`, local LAN, no GPU/WM) against the Pi
host at **10.0.0.125** — connect/contract/frame/RTT, then a wheels-up pass and a decisive **on-ground** pass.
**All checks pass; the robot-facing contract the live controller depends on is now empirically confirmed:**

- **Transport:** `LeKiwiClient(remote_ip=10.0.0.125, id=lekiwi)` connects over ZMQ; **import path
  `lerobot.robots.lekiwi`**. `get_observation()` RTT **~14–16 ms** (p95 < 22 ms) — network is a non-issue for
  stop-and-plan.
- **Contract:** action = 6 arm `.pos` + `x.vel` / `y.vel` / `theta.vel`; observation adds cameras
  `front` / `wrist` / **`top`** (bare key, **480×640×3 uint8**, matches the dataset's native res). Controller
  holds the 6 arm `.pos` at their observed values every step; `y.vel`=0 (strafe unused).
- **`x.vel` = m/s, `+x` = FORWARD** (commanded +0.05 → drove forward, readback 0.0465). → `x.vel = Δx/(f·Δt)`,
  no conversion.
- **`theta.vel` = DEG/S, `+theta` = LEFT/CCW** (commanded +15 deg/s → body turned CCW, readback 13.48). The
  WM's ω is rad/s (the build script converted deg/s→rad/s for training), so the controller **must convert**
  `theta.vel = (Δθ/(f·Δt))·(180/π)`. Sign **matches** the dataset (unicycle `+ω = CCW`) → **no negation**.
  Units confirmed two ways: the dataset build established raw deg/s, and a `12 deg/s` command read back a tidy
  `11.13` (rad/s would mean ~690°/s — motors would saturate, not report ~12).
- **`f·Δt = 10/30 = 0.333 s`** (the Run-002 chunk). So: `x.vel = Δx/0.333`; `theta.vel = (Δθ/0.333)·57.296`.
- **Low-speed rotation deadband (new finding):** `theta.vel=0.3` deg/s produced **no motion** (readback −0.586,
  encoder noise); `12–15` deg/s engaged cleanly. A typical chunk turns ~12 deg/s (Δθ≈0.07 rad/0.333 s) — in
  the controllable band — but **sub-deadband fine pivots may be a no-op**, so the controller likely needs a
  minimum-|theta| floor (or to accept tiny Δθ as no-turn). Minor cross-axis noise too (pure forward reported a
  spurious ~−1.2 deg/s; pure turn ~0.002 m/s) — watch for slight veer.

**Method note (caught a test-design bug):** wheels-up CANNOT show body rotation — LeKiwi's 3-omni-wheel base
spins the wheels tangentially but the body is fixed on the stand, and omni spin reads as "no rotation"
visually. The first wheels-up pass therefore *looked* like "no rotation at any theta"; the readback
(`12→11.13`) proved the motors did spin, and the **on-ground** pass gave the real body-turn direction. Added an
`--on-ground` mode + fixed the misleading wheels-up messaging. **6b.0 passes; transport + units + signs are
pinned → 6b.1 (open-loop replay) can convert recorded `(Δx,Δθ)` chunks to velocity with confidence.** See
[[planning]] "6b — RESULTS (6b.0)".

## 2026-06-04 — Stage 6b.1: open-loop replay — PASS (chunk approximation faithful, timing pinned)

Built `scripts/lekiwi_replay.py` + `scripts/lekiwi_common.py` (the 6b.0 `(Δx,Δθ)→velocity` contract in one
importable place). Converts a chunk sequence — **synthetic** patterns (forward/pivot/arc/square) or a
**recorded episode's** integrated `(Δx,Δθ)` — to base velocities and drives them **open-loop** (no WM/CEM/GPU),
with a dry-run that dead-reckons + plots (heading arrows + heading-vs-chunk) and an optional recorded-frame
filmstrip. Outcomes:

- **Trajectories match on hardware** (user-confirmed): synthetic and dataset episodes traced the dead-reckoned
  plots in shape, turn direction, and extent.
- **Constant-velocity-per-chunk approximation is faithful** — dead-reckon gap between the recorded fine 30 Hz
  path and the chunked-command path is **~0.0 cm even through a 117° pivot-arc** (ep44). Teleop is smooth at
  the 0.333 s chunk scale, so the collapse loses ~nothing ⇒ **6b.3's per-chunk velocity-hold won't add
  meaningful error.** (A phantom 6.2 cm "gap" turned out to be a clamp bug, see below.)
- **Per-chunk timing pinned.** Each chunk is now held for **exactly `CHUNK_DT`** (335–338 ms vs 333 ms target,
  ~1.5%), down from up-to-19% overshoot — the old loop checked the deadline at the top then ran a full
  `get_obs+sleep` iteration past it (~10–19% systematic over-travel at constant velocity). Fix: capture the
  arm-hold once, send a precomputed action (no `get_observation` in the hot loop), pace against a fixed
  deadline with a final partial sleep. The execute loop prints measured ms/target per chunk.
- **Action range corrected** (measured across all 50 eps): `x.vel∈[0,0.10] m/s`, `theta.vel∈±30°/s`
  (±0.5236 rad/s = ±π/6). The earlier ±0.34 rad/s undercounted the max; the safety clamp now uses ±30°/s.
- **Dataset access fixed + version-proofed:** created the missing **`v2.1` codebase-version tag** on
  `kaushikpraka/wm-smallarea_nav30` (it was untagged → `LeRobotDataset` refused to load). But a **recent
  lerobot (v3.0) can't read v2.1** (`BackwardCompatibilityError`), so the script reads the **parquet
  (`action`) + mp4 (`top`) directly** via `huggingface_hub`+`pyav` — no lerobot version gate. Confirmed the
  stored `theta.vel` is **rad/s** as assumed.

Artifacts: `viz/lekiwi_6b1/` (trajectory plots + filmstrips). **6b.1 passes — the `(Δx,Δθ)→velocity→robot`
pipeline is grounded end-to-end on hardware.** Remaining 6b is the GPU-side live CEM: **6b.2** (shared engine
module wrapping the 6a planner) → **6b.3** (closed-loop), resumed on the pod. Detail in [[planning]]
"6b — RESULTS".

## 2026-06-05 — Stage 6b.2: live engine smoke-test on the pod — PASS (LekiwiPlanner validated end-to-end)

Ran the authored `lekiwi_engine.LekiwiPlanner` (fork `4720053`) end-to-end on an **H100** with **step-8000**,
clearing the "engine authored, pod-test pending" flag. Drove the engine *directly* (no robot, no
`scripts/lekiwi_mpc.py`) with raw `top` frames pulled from the dataset mp4 via pyav — **480×640×3 uint8**,
exactly what `LeKiwiClient.get_observation()` returns — so the full live path executes: letterbox-preprocess
→ SD-VAE encode → CEM (32×3×top-10, DDIM=3, H=3) → WM rollout → decode → `PlanResult`. Harness +
artifacts: `results/smoke_6b2/` (`smoke_6b2.py`, PNGs) and `results/smoke_6b2_motion/`.

**All four gates pass, on a static AND a moving goal:**
1. **Action stats** = the integrate_se2 f=10 values (`mean=[0.0221,-0.0006]`, `std=[0.0141,0.0707]`) — match
   6a's `run.log` exactly.
2. **do_nothing sanity** (`plan(frame, frame)`): `dist_to_goal = 0.007–0.010 ≈ 0` (latent-L2 of a frame
   against itself; confirms encode + objective wiring).
3. **Goal is distinguishable & sign recovery is correct.** First pair (ep44 0→+30) happened to be
   near-static, so CEM correctly returned ≈no motion (`vx≈0`, `+3.1°/s`) — honest but not a motion test. So
   re-ran on a **moving** pair selected by scanning the parquet action stream for the largest 30-frame
   window: **ep11 frame 504→534**, GT first chunk `vx=+0.100 m/s, θ=−24.4°/s` (forward + right turn, a 6a
   "arc"). CEM recovered **`vx=+0.067 m/s, θ=−15.6°/s` — signs match exactly** (forward + CW), magnitudes
   conservative (CEM under-drives large motion, consistent with WM under-prediction). `dist_to_goal=43.8` vs
   do_nothing `0.007`.
4. **Decoded `imagined` is a coherent top-view** (std≈55.6, not noise) — robot body / curtain / floor / lamp
   all legible; on the moving pair the WM rollout under the plan visibly **advances + rotates right**, tracking
   the goal's direction (residual goal gap = WM prediction error, per 6a).

**The explicit-stats requirement (important for the 6b.3 launch).** The engine has two ways to obtain the
`(Δx,Δθ)` denormalization stats it needs to convert CEM's normalized action into metric `(m, rad)`:
its `__init__` first branch takes `action_mean`/`action_std` directly; otherwise it *reconstructs the val
dataset* via `create_train_val_datasets` and reads `val._raw_action_mean/std`. **The reconstruction path is
dead on the pod and must not be used for the live robot:**
- `LeRobotDataset.__init__` calls `get_safe_version()` → `list_repo_refs()`, which **hits the HF Hub even with
  a local `root`** to resolve the dataset's version ref. The source `kaushikpraka/wm-smallarea_nav30` is
  **private → 401 Unauthorized** without a token (and the smoke box has none wired into the venv).
- Even *with* a token it fails differently: the dataset is **codec v2.1**, and the installed **lerobot is v3.0,
  which refuses to read v2.1** (`BackwardCompatibilityError`) — the exact wall 6b.1 hit and worked around by
  reading parquet+mp4 directly. So the dataloader is not a viable stats source on this stack at all.
- The stats are also **not stored in the checkpoint** (the cfg carries the dataset *name*, not the computed
  normalization), so there is no offline fallback inside the ckpt.

⇒ **the live controller MUST pass `action_mean=[0.022110389545559883, -0.0005879045929759741]`,
`action_std=[0.014105414971709251, 0.07071184366941452]` explicitly** (the f=10 integrate_se2 values 6a
derived and printed; the engine prints them back with an `expect ~[0.0221,-0.0006]/[0.0141,0.0707]` check).
This is not a workaround — it's the intended on-robot config: the robot has **no dataset present**, so stats
*must* be injected. The smoke-test was run in exactly this configuration, so it validates the real deployment
path. **Action item for 6b.3:** `scripts/lekiwi_mpc.py --planner wm` (and `configs/planning/lekiwi.yaml`)
must thread these two vectors into `LekiwiPlanner(...)`; getting a wrong/zero stat silently rescales every
command (e.g. a missing `std` would zero the action) — so this is a hard precondition, not advisory.

A wrong-sign or wrong-scale stat is the one mistake that would pass every cheap check and still drive the
robot wrong, so it's pinned here and in [[roadmap]]/[[planning]]. **6b.2 passes — the engine module is
validated on real GPU + real frames; 6b.3 (closed-loop on LeKiwi) is unblocked, gated only on the robot.**

## 2026-06-05 — Interactive WM driver + first closed-loop run on the real robot (6b.3) + rerun live-viz fix

**Interactive WM "driving" evaluator** (`external/nanowm/src/sample/interactive_wm.py`, new). Browser tool
to drive the world model open-loop with the keyboard (WASD → one action-chunk/keypress → decode the predicted
frame) plus a CEM overlay (full imagined trajectory + elite endpoints toward a loaded goal). Headless-safe
(stdlib `http.server`, no Flask); reuses `LekiwiPlanner`/`DiffusionWorldModel`. Smoke-tested on step-8000:
do-nothing latent-L2 ≈ 0.015, open-loop step ≈ 0.2 s, CEM overlay ≈ 7.7 s @ DDIM=3, decoded frames are
coherent top-views; far-goal demo (6 chunks, horizon 6) showed CEM closing only ~11 of 57 latent units with
the imagined frames degrading past the 3-chunk train window — i.e. far goals need MPC replanning / waypoints,
not a one-shot plan. Encodes in the **training pixel range ([-1,1])** to match the validated 6a path — note
the 6b.2 engine's `_preprocess` feeds [0,1], a latent range mismatch worth revisiting.

**First closed-loop run on the LeKiwi (`scripts/lekiwi_mpc.py --planner wm`, full speed, goal
`goals/nearfan.png`).** Planning worked end-to-end on the real robot: engine loaded with the correct
integrate_se2 stats, CEM produced sane first-chunk commands (~7.4–7.6 s/plan @ DDIM=3), robot executed the
stop-and-plan loop. **But it did not converge** — `dist_to_goal` hovered ~44–46 over 22 steps (reach-thresh
35), and the **Pi-side robot host dropped mid-run** (the SSH tunnel went down with it — all of 5555/5556/9876
closed at once). So: motion + planning validated on hardware; goal-reaching convergence + tunnel stability
are **open**. Full telemetry captured to `/workspace/results/mpc_nearfan.rrd` (48 MB, 22 steps; on the
persistent volume).

**Rerun live telemetry — root-caused and fixed.** Live `--rerun-addr 127.0.0.1:9876` failed every time with
`re_grpc_client … transport error`, reproducible with a 3-line probe (so not our code). Cause: the
`-R 9876` reverse tunnel delivers to **Mac:9876, which VS Code Remote-SSH holds** → bytes hit VS Code, not a
viewer. (Rerun also demands viewer==SDK version, 0.22.1.) Fix: added **`--rerun-web`** to `lekiwi_mpc.py` —
the pod hosts a version-matched web viewer (`rr.serve_web`, HTTP 9090 + WS 9877); forward those two with
`-L` and open a browser, no Mac-side rerun at all. Verified the pod serves it (HTTP 200, both ports). Also
made `rr_init` **tee** telemetry to independent RecordingStreams so live + `.rrd` record run simultaneously
(rerun 0.22 is single-sink per recording). Runbook updated in [[tailscale-setup]] ("Live rerun telemetry").
**Next session:** bring tunnel + `--rerun-web` up, redo the run, watch why `dist` plateaus (WM under-drive vs
goal too far for horizon 3 vs tunnel-drop truncation).

## 2026-06-06 — Pixel-range bug found + fixed; convergence still open (range was necessary, not sufficient)

**Live rerun — switched to the NATIVE viewer on a clean port (not the web viewer).** The `--rerun-web`
path works (pod serves 9090/9877, browser only), but the user preferred the native rerun app. Root issue
was never the app — it was the *port*: we'd pointed live at 9876, which collides with VS Code Remote-SSH.
Fix is just a clean port: Mac runs `rerun --port 9999`, reverse-tunnels `ssh -N -R 9999:localhost:9999`,
pod runs `--rerun-addr 127.0.0.1:9999`. (`connect_grpc` accepts bare `host:port`; confirmed both that and
the `rerun+http://…/proxy` URL parse.) Added **`scripts/rerun_web_smoke.py`** — a standalone telemetry
generator (moving scalar + live image) that exercises the exact serve/connect path with NO robot, so live
viz can be validated independently; `--rerun-addr host:port` uses the native path, else serves web.

**Root-caused the non-convergence to a pixel-range mismatch — and fixed it.** Training normalizes pixels to
**[-1,1]** (`wm_datasets/world_model_dataset.py:664`, `video = video*2-1`, `normalize_pixel=True` default),
but the on-robot engine `sample/lekiwi_engine.py:_preprocess` fed the VAE **[0,1]** (its comment matched the
lerobot loader's [0,1] output but missed the `*2-1` the dataset applies on top). So every observed frame and
the goal were encoded in a range the VAE/WM never trained on → `z0`/`z_goal` off-distribution → `dist_to_goal`
meaningless and CEM had no real descent direction. **Fix:** `_preprocess` now pads in [0,1] (black borders
stay 0) then applies `*2-1` last — matching the dataset's pad-then-normalize order (borders → -1). Both `z0`
(plan) and `z_goal` (`_goal`) flow through it; decode (`decode_latents`, `(x+1)/2`) is the unaffected inverse.

**Re-ran nearfan (full speed, execute, fixed range) — STILL does not converge.** Over ~13 steps `dist`
sat at **51 ± 0.5, completely insensitive to the (varied) commands**; θ oscillated sign every step (robot
wiggles in place rather than committing to a heading). The range fix only shifted the absolute scale
(~47 → ~51) — it was **necessary but not sufficient**. Two structural notes feeding the diagnosis: per-chunk
motion is *tiny* (x≈0.05 m/s × CHUNK_DT 0.333 s ≈ 1.6 cm; θ a few deg/step), so even correct planning moves
the scene very little per step; and the **execution horizon is 1** (`lekiwi_engine.py:179` "FIRST chunk only,
execute-one replan"), planning **H=3**. Conclusion reached with the user: bumping *execution* horizon won't
help convergence (less feedback, not more reach); bumping *planning* H past the ~3-chunk train window is free
(`lekiwi_engine.py:84` rolls out autoregressively) but optimizes against degrading WM predictions — reliable
long-horizon planning needs **retraining**. The flat, action-insensitive `dist` now points at the WM not
giving CEM a usable gradient (goal beyond H=3 reach and/or under-responsive dynamics), NOT at preprocessing.

**Prefs:** user set "always run execute at full `--speed-scale 1.0`" (saved to memory). Telemetry captured to
`/workspace/results/mpc_nearfan_fix.rrd` (range-fixed, 13 steps) + earlier `mpc_nearfan_exec_full.rrd`.

**Open / next:** convergence is the live question. Probe whether CEM's *imagined* `dist` actually drops for
any action (is there a descent direction at all, or is the loss flat?); check if nearfan is simply beyond
H=3 reach (try a goal 1–2 chunks away, or larger per-chunk action magnitude / step-dx); consider waypoints or
a longer-horizon retrain. The `--reach-thresh` also needs recalibration to the new [-1,1] `dist` scale.
