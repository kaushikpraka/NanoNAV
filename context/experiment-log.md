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

## 2026-06-08 — Convergence root-caused: flat latent landscape from a wide-angle overhead camera (camera ⊗ objective conditioning)

Settled the closed-loop non-convergence. The world model and CEM are **fine** — the bottleneck is the
**objective landscape**, and it traces upstream to the **camera**. Headline lesson: **camera FOV and the
planning objective are a JOINT design choice**, not independent. A wide-angle view is great for
perception/obstacles but poorly *conditioned* for goal-distance planning.

**Offline probe (the vindication) — `sample/offline_planning_eval.py`, step-12000, 12 scenes × DDIM {20,3}:**
- **12/12 beat the do-nothing floor**, every motion bucket (translation/pivot/arc/slow).
- `wm_drop` (do_nothing − gt_ceiling) mean **+15–16** → the WM strongly predicts goal-reaching motion under
  the *true* actions. Dynamics are not broken — often dramatic (pivots +21 to +34).
- `reached_ratio` (cem_reached / gt_ceiling) ≈ **1.0** → CEM hits the WM's ceiling; search works.
- **DDIM=3 ≈ DDIM=20** → the robot's low sampling budget is NOT the bottleneck (rules out the cheap fix).
- Crucial caveat that explains the live gap: offline goals are always placed **exactly `goal_H=3` chunks
  ahead** — i.e. already *inside* the basin where the gradient exists. The probe never tested far goals.

**Live runs (step-12000, full speed, nearfan): same non-convergence, both checkpoints + positions.** CEM
commands **turns when straight is obviously needed** (θ +9…+20 from the start), `dist` flat ~42, robot
wanders/drifts away. The "place it slightly behind / should just drive straight" setup still failed — the
start was ~17 chunks (~46 cm) away, i.e. in the **flat region**, not the basin.

**The decisive diagnostic — `--drive-straight` (new flag): bypass CEM, drive a fixed forward vx (θ=0), still
encode + log WM `dist`.** Drove ~46 cm straight toward nearfan:
- `dist` **flat 40.5–44.4 for 16 steps** (no trend), then the **operator nudged a slight heading error** and
  `dist` snapped **44.4 → 32.8 → REACHED (<35)** in one step.
- So the goal IS reachable and on-distribution (earlier "off-distribution" hypothesis was **wrong**); the
  latent metric *does* track pose — but only on the **precise approach line**. Off it, flat.

**The root finding — flat-far / narrow-basin objective, and it's in RAW PIXELS:**
- pixel-L1(frame, goal): step0 **25.8** → step16 **26.1** (≈46 cm of driving, ~unchanged) → step17 **15.7**
  (after the heading correction). The flatness is present *before the WM* — the camera images themselves
  barely change under large motion. The WM faithfully encodes inputs that genuinely don't move.
- Why (all visible in the decoded frames, `results/drive_straight_frames.png`): **wide-angle egocentric
  overhead camera** → (1) low parallax from distant content (plant/back wall fill the wide FOV; parallax ∝
  1/depth), (2) the **robot's own body is fixed** in the lower frame (motion-invariant, eats latent capacity),
  (3) large low-texture floor/wall regions, (4) **barrel distortion** → position-dependent action→pixel map
  (sharp at center, flat at periphery). Net: a "flat far, narrow basin near" objective. CEM (H=3) outside the
  basin sees no gradient → flails into turns; blind straight-driving works only by stumbling into the basin,
  and only if the open-loop heading doesn't drift off the line first.

**Generalization (camera ⊗ objective):** this recurs for **image-distance objectives + distant/low-texture
scenes + translation goals + short-horizon samplers** — a conditioning trap, well known in image-goal nav /
visual servoing ("perceptual aliasing", "vanishing gradient far from goal"). It is NOT "wide-angle is bad":
near-field tasks (manipulation) use fisheye happily (big parallax), rotation is fine even wide-angle, and a
better-conditioned objective (learned value / relative-pose / feature-matching) extracts a gradient where
latent-L2 is flat. Change any one factor and the trap loosens.

**Fixes (cheapest first):** (a) **waypoints** — sub-goals ≤2–3 chunks apart so every plan starts inside the
basin (zero retrain; predicted to work); (b) **undistort + center-crop** the view to trade FOV for motion
sensitivity (likely needs a VAE/WM retrain on the cropped view); (c) **mask the robot body**; (d) **denser
near-field texture**; (e) a **denser/learned objective** to widen the basin. Decisive test for the camera's
role: retrain (or re-encode) on a distortion-corrected center crop and re-measure the latent-dist-vs-
displacement curve — if it steepens, the camera was a primary cause.

**Diagnostics added this session (committed):** live per-scene `do_nothing/gt_ceiling/cem_reached` print in
`offline_planning_eval.py`; `--drive-straight VX` open-loop flag in `lekiwi_mpc.py`; imagined-rollout viz fix
— the `imagined` panel now shows the **+1 chunk the robot actually executes** (was wrongly the +H endpoint,
the most autoregressively-degraded frame) plus a `rollout/h1..hH` filmstrip; flat single-row rerun blueprint
(the nested 2-row layout wedged the web viewer). See [[lekiwi-wm-camera-objective-conditioning]].

## 2026-06-08 (later) — CORRECTION: radial conditioning is FINE; the camera is NOT the bottleneck (controlled sweep refutes the "flat landscape" claim)

The camera-aliasing / "flat latent landscape" conclusion above is **WRONG** — it was built on a confounded
number. Re-measured with a controlled tool (`scripts/measure_dist_sweep.py`: hand-place the robot at marked
displacements along the goal axis, read latent-L2 + pixel-L1 to goal, plus a same-pose noise burst; NO
motion). Results (`/workspace/results/dist_sweep/curve.png`):

| displacement | latent_L2 | pixel_L1 | same-pose noise σ |
|---|---|---|---|
| 0 cm | 42.47 | 26.36 | 0.09 / 0.008 |
| 10 cm | 42.47 | 25.87 | 0.13 / 0.017 |
| 20 cm | 40.72 | 23.07 | 0.13 / 0.045 |
| 30 cm | 37.84 | 20.07 | 0.10 / 0.012 |
| 40 cm | 34.46 | 17.82 | 0.15 / 0.029 |

- **−8.0 latent / −8.5 pixel over 40 cm, monotonic**, noise σ only ~0.12 latent / ~0.02 pixel → **SNR ≈ 17/10 cm
  (latent), ≈ 97/10 cm (pixel)**. The objective is **well-conditioned along the radial approach axis** — not
  flat, not aliased. The wide-angle camera encodes pose just fine here.
- **Why the earlier "46 cm → 0.3 change" was an artifact:** the `--drive-straight` robot was drifting
  *off-course*, so those 46 cm were path-length while it stayed ~equidistant — never a radial approach. When
  the operator nudged it *onto* the axis, dist fell straight into the steep part of this curve (44→32.8). So
  "flat far / camera information-limited" is **retracted**.
- **NEW anomaly (matters):** the operator moved *away* from the believed goal, yet dist *decreased*, and the
  minimum (34.5 @ 40 cm) never reached the ~32 "reached" value → **`goals/nearfan.png` corresponds to a pose
  ~50 cm BEHIND the operator's "0 cm/at-goal" reference.** Likely a **goal-image ↔ intended-pose mismatch**:
  closed-loop may have been correctly driving toward the nearfan-capture pose, not where we thought the goal
  was. Verify by re-capturing the goal *at* the intended pose (or checking what nearfan.png actually depicts).

**Revised diagnosis:** camera radial info is good → the closed-loop failure is **off-axis**: the robot can't
*stay on* the radial axis (heading drift + CEM commanding turns push it laterally, where distance-to-goal is
geometrically ~flat — that's the "flat ~42" we kept seeing), and/or a goal-pose mismatch. **Next:** yaw sweep
(robot self-rotates in place in fixed increments, measure dist vs angle) + lateral sweep, to test the
heading/lateral conditioning the robot actually wanders in. The general **camera ⊗ objective** principle still
holds as a design lesson, but for THIS rig the camera is not the limiter.

## 2026-06-08 (resolution) — FIRST closed-loop convergence (REACHED ×2); both prior "root causes" were over-reach; the remaining open question is basin catchment, NOT the camera

**Headline: the closed-loop WM controller converges on the real robot.** Two clean runs toward a freshly
captured goal (`goals/nearfan2/goal.png`), step-12000, full speed, `--planner wm`:
- reach-thresh 35 → monotone-ish descent 41 → **REACHED 34.99** in 10 steps (`mpc_nearfan2_execute.rrd`).
- reach-thresh 25 → 40 → **REACHED 21.82** in 14 steps, with a sharp final dive 28.7→21.8 into the basin
  (near the ~16 floor) (`mpc_nearfan2_thresh25.rrd`).
The WM dynamics, CEM search, latent-L2 objective, the wide-angle camera, and DDIM=3 are all **vindicated** —
nothing in the stack is broken. Both big mid-session conclusions (camera-aliasing; then global
basin-of-attraction) were over-reach.

**Diagnostic chain that got here:**
1. Radial sweep (`scripts/measure_dist_sweep.py`, hand-placed, no motion): latent −8/40 cm monotonic, SNR
   ~17–97 → metric well-conditioned radially (killed camera-aliasing).
2. Yaw sweep (`--yaw-sweep`, robot self-rotates in place): a **clean sharp basin** in heading exists — the
   metric senses heading fine — but for the OLD goal the basin was shallow (depth 14, min 38) and sat at a
   pose offset from where we aimed.
3. Re-captured the goal at the robot's actual pose (`nearfan2`): yaw basin became **deep + sharp** — min
   **latent 16.3 / pixel 3.6** (near-perfect match), depth 33 (vs 14). pixel_L1 3.6 ⇒ the live frame ≈ the
   goal frame at that pose.
4. Closed loop toward `nearfan2` → REACHED ×2 (above).

**What is CONFIRMED:** the metric/WM/CEM/camera are healthy; the closed loop converges when the goal is
**within the basin catchment**. The objective has a sharp deep basin (steep within ~±10° heading / the
near-field radius) surrounded by a flat ~40–50 plateau.

**What is NOT settled (operator flag — do not overclaim "mislocated goal"):** `nearfan2` was captured *at the
robot's own pose*, so it is an **easy/close** goal already near the catchment. `nearfan` is still a **valid**
goal — its non-convergence may be because it is genuinely **farther / outside the catchment** (the real, open
basin-of-attraction limit), NOT because it was "wrong." The sweeps showed `nearfan`'s best-match pose is
offset from where we *thought* the goal was, but that is consistent with either (a) a capture-pose mismatch OR
(b) a legitimately farther goal. **Unresolved.**

**Open question → next experiments:** (a) re-run `nearfan` (unchanged) with the robot started *near nearfan's
actual basin* (low starting dist) — if it converges, `nearfan` is fine and the issue was just start-outside-
catchment; (b) **map the catchment radius** — how far / how misaligned a start still converges; (c) for starts
*outside* the catchment (far goals), the flat plateau is the real blocker → a **learned temporal-distance
metric + model-imagined subgoals** is the lever to extend reach (the "plan fully in the WM, no manual
waypoints" path). `--reach-thresh` recalibrated: with a correct goal the basin floor is ~16, plateau ~45, so
25–30 is a sensible threshold (35 only grazes the basin edge).

**Tooling added + committed this session:** `measure_dist_sweep.py` (+`--yaw-sweep`), `--drive-straight`,
imagined `+1`/filmstrip viz, flat single-row blueprint, offline-eval live-metrics print. Sweeps:
`/workspace/results/{dist_sweep,yaw_sweep,yaw_sweep2}/curve.png`.

## 2026-06-09 — Far-goal still stalls (H=5, var-scale, vx-max all no help); ⭐ KEY NEXT STEP = replace the raw latent-L2 objective with a learned/temporal distance

Spent the session probing the **far-start (outside-catchment) plateau** and ruling out the cheap knobs.
None of them help when the start is on the plateau, because **the objective itself has no gradient there** —
that's now the clearly-identified blocker, and the fix is a better *distance metric*, not more search.

**⭐ KEY NEXT STEP (flagged by operator) — the convergence objective is the limiter.** Convergence is
measured as **raw flattened latent-L2** between the current top-frame latent and the goal-frame latent
(`lekiwi_engine.plan`: `dist_to_goal = ‖z0 − zg‖`, `_flat_l2 = torch.norm((a-b).reshape(-1))`; REACHED when
`< --reach-thresh`). This is correct *as a convergence readout* and well-conditioned **near** the goal (sweeps:
basin floor ~16, SNR ~17–97), but **flat far out** — every spatial latent cell is weighted equally and most
encode "generic floor/wall," so two far-but-different poses look ~equidistant. **Replace raw latent-L2 with a
denser / learned objective — a self-supervised temporal-distance / quasimetric (frames k apart → distance ≈ k,
trainable on the data we already have)** — to put gradient on the plateau. This *also* enables
**model-imagined subgoals** (imagine reachable futures → score by learned distance → hop → repeat), i.e. the
"plan fully in the WM, no manual waypoints" path. This is the single highest-leverage next step for far goals.

**Knobs tested today (all NO help on the plateau):**
- **`--horizon 5`** (vs 3): only changes *planning*, not execution (see below); CEM looks 5 chunks ahead but
  +4/+5 are autoregressive past the train window (H_train=3). Far start 44→~49, drifted away. Lookahead into a
  *flat* region creates no gradient for the first step. (~14.5 s/plan, ~2× H=3.)
- **`--var-scale 2.0`**: visibly stronger turn sampling but same plateau stall → sampling width isn't the limit
  (the flat loss doesn't select the bigger turns). [[lekiwi-wm-camera-objective-conditioning]].
- **`--vx-max 0.12`** (0.04 m/chunk vs 0.033): CEM didn't even use the bigger cap on the plateau (no gradient
  to exploit it). Helps only *inside* the catchment.

**Execution semantics confirmed (execute-one, replan):** each plan executes **only chunk #1** of the H-chunk
plan (`lekiwi_engine.plan` returns `raw[0]`; `lekiwi_mpc` streams `(vx,θ)` for exactly one `CHUNK_DT`=0.333 s,
then STOP→OBSERVE→re-PLAN). Chunks 2…H are **imagined-only** (so CEM can score the +H endpoint, `mode="last"`)
and discarded. So a longer planning horizon ≠ the robot committing further; it still moves ~3 cm/step.

**Camera USB-enumeration swap (real, intermittent) + durable fix.** After ~4 Pi-host crashes, the `top` camera
*name* re-mapped to the front-facing device (verified: `front` key held the overhead/robot-body view the WM
trains on; `top` held a low floor view). This fed the WM an **out-of-distribution camera** and invalidated the
mid-session `nearchair1`/`vx-max` runs. A Pi restart swapped it back (non-deterministic). **Durable fix: a
udev rule on the Pi pinning each camera to a stable `/dev/lekiwi_top` by USB serial**, then point the host at
it. Always re-probe (`results/cam_probe*/`) after a host restart before trusting a run.

**Rollout audit (no bug; sequential scheduling explains the "+1 looks poor" viz).** `scheduling_mode:
sequential` (self-forcing): frames generate serially — **+1 = f(z0, a0)** conditioned *directly* on the start
latent, then +2 = f(+1, a1), +3 = f(+2, a2). So **+1 is the only frame tied straight to z0**: if the live z0 is
off-distribution, +1 inherits it and looks rough while +2/+3 regress toward the model's prior and look cleaner.
The interactive driver seeds from **val frames** (in-distribution) → looks clean; MPC seeds from **live**
frames → +1 reveals the live-distribution gap. Decode indexing verified correct. **CEM is NOT corrupted** —
its objective scores the +H endpoint, not +1. (Implication: live-frame ↔ training-distribution gap is worth
closing — `_preprocess` parity check or fine-tune on live frames.)

**New tooling this session (uncommitted unless noted):** `--var-scale` (committed), `--vx-max` (uncommitted,
needs a clean robot run), `--max-steps` default 30→**100**, interactive-driver **start-frame switcher**
(prev/next/random/jump over the 4405 val slices; committed, submodule 8f78848).

**⭐ Operator's synthesis (end of session) — the planner/objective is the highest-value next work; the WM is a
good-enough foundation.** On the H=5 run the operator watched the planner **correctly prioritize rotation and
turn the bot to face the chair** (overshot, ended a bit close) — a *qualitative* success — yet `dist` sat ~48
the whole time. So **the latent-L2 objective under-credited a real success**: CEM *chose* the right behavior
but the metric barely rewarded the alignment, so nothing locked it in → overshoot/drift. This is the same
failure as the flat plateau: **raw `‖z0−zg‖` doesn't track real reachability/pose-progress.** Conclusion:
"improve the planner" ≈ "improve the *distance objective*"; the search itself is fine (offline CEM hits the WM
ceiling). The WM has **generally mapped the room** (poses distinguishable near goals per the sweeps) — a solid
base to build a better objective on rather than retraining dynamics. Caveat acknowledged: **decoded frames are
still blurry** = the **VAE/WM reconstruction is lossy / latents are smooth** — but a learned distance operates
**on the latents (never decodes)**, so blur doesn't block it, and the smoothness is *part of why* raw L2 is
flat. **Concrete build (the ⭐ key next step):** a self-supervised **temporal-distance / quasimetric** head on
the existing latents — sample frame pairs from the dataset, label by temporal gap (k frames apart → dist ≈ k),
train a small head to predict reachability-distance; swap `dist_to_goal = ‖z0−zg‖` for `d_learned(z0,zg)` in
both the readout and CEM's objective. Trains on data we already have (no new collection); also unlocks
**model-imagined subgoals** ("plan fully in the WM, no manual waypoints"). See [[open-questions]] "Scoring
function alternatives". Operator is moving to off-pod planning from here.

## 2026-06-09 (later) — 🚨 NEW TOP BLOCKER: the WM HALLUCINATES from live frames (off-distribution z0) — upstream of the objective

A `nearhamper1` eval surfaced a more serious problem than the objective. With the bot **in front of the
hamper** (hamper clearly in the live `camera` panel), the **imagined rollout shows a completely DIFFERENT side
of the room** (a chair + the robot's own gripper from elsewhere), not the hamper. Pulled the actual frames from
`mpc_nearhamper1_pos2.rrd` (montages: `context/figures/live-distribution-gap_*.png`,
`results/hamper_debug/`): step 0 `imagined +1` is a **washed-out haze**; by `+2/+3` it **snaps to a different
training-common scene**. The `camera` panel (preprocessed live pixels) is correct — so the failure is in
**live-frame → latent → WM prediction**, not the camera or decode.

**Mechanism (sequential scheduling makes it visible):** `+1 = f(z0, a0)` is conditioned *directly* on the live
latent. When `z0` is off-distribution the model can't stay anchored — `+1` degrades, then the autoregressive
`+2/+3` **regress to a familiar training mode elsewhere in the room.** So the imagined "future" is not a
prediction of the live scene at all; it's a hallucination. The **interactive driver looked clean only because
it seeds from VAL frames (in-distribution)**; live frames trigger this. (Test in progress: seeding the
interactive driver from the live **goal image** to see if live-captured frames hallucinate there too.)

**Why this is THE top blocker (above the learned-distance objective):** if `z0` and the imagined rollouts are
garbage, then BOTH `dist_to_goal` and what CEM optimizes are garbage → the confidently-wrong commands (turning
away, rotating to the wrong side) all follow. A better distance metric can't help if it's measuring a
hallucinated latent. **Order: close the live↔training distribution gap FIRST, then the learned objective.**

**It's uneven by pose/coverage — reconciles the whole session:** `nearfan2` converged (REACHED ×2) because its
start was near a **well-covered** training pose → in-distribution `z0`; `nearhamper` from a far/off-axis/
under-covered pose → off-distribution `z0` → hallucination. So the WM "mapped the room" only where training
data was dense.

**Off-pod investigation order:** (1) **`_preprocess` ↔ dataset-pipeline byte parity** (cheapest, likely
culprit) — letterbox pad value, interpolation mode, color channel order, normalize range, JPEG/AV1-vs-raw; any
divergence puts `z0` off the manifold despite correct-looking pixels. The interactive driver uses *val tensors
directly* and is clean; MPC runs live frames through `_preprocess` → that's the suspect path. (2) capture-
condition match (exposure/WB lock, avoid lossy AV1 — already flagged in [[open-questions]]). (3) if parity is
exact → **data coverage**: more coverage of under-visited poses, or a light **fine-tune on live frames**.
See montages + `[[open-questions]]` "Live-frame distribution gap".

**RESOLUTION (same session) — it's OOD coverage, not a preprocess bug; fix = recollect + retrain.** Seeded the
**interactive driver from the nearhamper goal image** (a clean, settled live capture, loaded via the SAME
`load_image_rgb`+`_to_model` path as the `nearfan2`/`nearchair` goals that *converged*) → **it hallucinates
there too.** Since other goal images through the identical path work, this **rules out `_preprocess`/format
parity and motion-blur** and confirms the nearhamper view is a **genuinely under-covered region** of the room
(the WM "mapped" only densely-visited poses). So: **`_preprocess` parity check is DOWNGRADED**; the real fix is
**more data coverage + further training** (operator's plan). **Eval guidance until then: use goals in
well-covered regions** (e.g. `nearfan2`-style, near where the bot drove during collection); avoid OOD goals like
`nearhamper`. Operator is recollecting data + training further off-pod.

## 2026-06-09 (design session) — Learned distance objective (quasimetric) design + build order settled

Off-pod design session converging the **#2 next step** (after the live-frame distribution fix): replace the
raw SD-VAE latent-L2 planning objective with a **learned quasimetric distance**. Full design captured in
[[learned-distance-metric]]; this is the chronology pointer.

- **What it learns:** "≈ chunks-to-drive" (drivable shortest-path / optimal cost-to-go), NOT appearance —
  keys on *drivability* (two spots near on open floor = near; near across a wall = far), the thing raw L2
  can't express and the reason it's flat far-out.
- **Approach = Quasimetric RL (QRL, arXiv:2304.01203) + IQE head (arXiv:2211.15120).** Objective = two
  opposing forces + structure: local cap (adjacent chunks `d≤1`, the only ground truth, self-supervised
  from the video timeline) + **push random pairs apart** (anti-collapse) + IQE triangle-inequality
  *by construction*. Maximizing against the local caps makes the metric structure compute shortest-path
  (beads-and-strings). Cross-trajectory pairs are handled by stitching through shared places — answered the
  "physically-nearby-but-different-trajectory" worry: the Δ-inequality caps them at the true short distance
  if a connecting drive exists; over-estimates (safe pessimism) only on genuine coverage gaps.
- **Architecture:** frozen SD-VAE → **CNN φ** (encoder, *not* itself a quasimetric) → **IQE head** (supplies
  the geometry), trained jointly; single shared encoder, asymmetry from the head. = QRL's image recipe with
  φ reading the `[4,32,32]` latent instead of pixels. SD-VAE-specific handling: CNN-not-MLP (disentangle
  pose from appearance on the spatial grid), normalize to the WM convention, stay in WM latent space
  (no decode), train φ on *imagined* latents too.
- **Planner wiring:** cost on the **generated** latents `d(ẑ,zg)` (CEM ranks actions by imagined outcome —
  the only action-dependent quantity); termination on the **current** state `d(z0,zg)`. A good metric lets
  the WM rollout stay short (lean on the metric for the far horizon), dodging far-horizon WM degradation.
- **VLM:** ruled OUT as a runtime cost (latency × ~96 candidates, forces a decode into blurry/hallucinated
  pixels, too coarse); viable only as an **offline teacher** for cross-episode place labels (deferred).
- **Pitfalls recorded:** wormholes (Δ-inequality amplifies latent aliasing globally — "confidently wrong"
  vs raw L2's "flat but honest"), fragile min-max training, asymmetry mis-estimation, reachability ≠
  controllability (our under-drive), bounded by WM fidelity (not a hallucination fix), coverage/verification
  gaps.
- **Environment (`context/figures/room.jpg`):** small carpet area, **landmark-rich perimeter** (fan/TV/
  lamp/hamper/purifier/bed → low aliasing, softens wormholes) + low-texture centre (localized flat trap),
  small but tens-of-chunks deep (far goals real).
- **BUILD ORDER (decided):** (0) **simple temporal-distance MLP baseline** *with cross-trajectory negatives*
  (the variable that decides flat-vs-not, more than the head; ViNG trick = cross-traj pairs → max-distance) →
  (1) **QRL/IQE quasimetric** only where the baseline plateaus → (2) **topological graph** over real frames
  (edges = `d_learned<τ`, Dijkstra, nearest node = CEM subgoal; real-frame nodes dodge hallucination + the
  graph is a wormhole guard). The metric is a **hard prerequisite** for the graph; both rungs share ~80%
  scaffold so the baseline isn't throwaway and de-risks QRL's fragile training. **Next concrete action =
  encode-cache + sweep eval (radial/yaw/lateral) + rung-0 baseline; the sweep grade is the GO/NO-GO gate.**
  All buildable/validatable **offline now**, independent of the retrain (which only gates the on-robot
  far-goal payoff). See [[learned-distance-metric]], [[roadmap]] 6d, [[open-questions]] "Scoring function
  alternatives".

## 2026-06-09 (research session) — Plan stress-tested vs the literature: rung-1 head REVISED (QRL → contrastive), zero-training frozen-embedding arms added, phased plan with Gates A/B

Ran a multi-agent literature review over the planner-improvement plan ("is MLP → quasimetric right, or
are there better routes?"). **Direction confirmed; one component replaced; three cheap tracks added.**
All design changes folded into [[learned-distance-metric]]; execution into [[roadmap]] 6d.

**Confirmed by the literature:**
- **Temporal-distance head + topological graph is THE field-proven recipe** for room-scale image-goal
  nav from small offline data — ViNG (arXiv:2012.09812) ran real robots on exactly our rung-0 recipe
  (temporal regression + cross-trajectory negative mining + graph); GNM/ViNT/NoMaD kept the distance
  head at much larger scale. Nobody plans room-scale on raw latent distance: NWM (arXiv:2412.03572)
  uses LPIPS + reports our exact OOD mode-collapse at 1B params; DINO-WM / V-JEPA-2-AC keep horizons
  short. Flat-far-from-goal + OOD hallucination are *acknowledged, unsolved* properties of pure WM
  planning — the field's answer is external structure (graph), i.e. our 6d shape is right.
- **Graph promoted in framing:** not just "for far goals" — it's the component that keeps every module
  in its comfort zone (WM: short rollouts from real frames; metric: short well-covered pairs; CEM:
  in-basin). Effort goes to **edge hygiene** (conservative τ, ensemble-disagreement filtering, SGM-style
  edge deletion on failed hops — SGM: a *single* bad edge wrecks planning), not metric structure.

**REVISED — rung-1 head: QRL/IQE dual-ascent is OUT, contrastive / MC-quasimetric is the escalation.**
OGBench (arXiv:2410.20092): QRL ~0% on **every visual task** (dominates only state-based mazes); four
2025 follow-ups (TMD/Eikonal-QRL/ProQ/MQL) exist specifically to fix the dual-ascent instability ⇒
structural, not tuning; no real-robot/image-nav use found. And the property QRL is bought for —
triangle-inequality stitching — is exactly what the rung-2 graph + Dijkstra provides explicitly.
Replacements (in order): **CRL** InfoNCE on temporal proximity (94% where QRL gets 0%; Stable-CRL
arXiv:2306.03346 has the real-robot recipe) → **Contrastive Successor Features** (arXiv:2406.17098) or
**MQL** (arXiv:2511.07730, MC-fitted quasimetric, no min-max). IQE survives only as a head
parameterization if the rung-0 symmetric-vs-asymmetric ablation shows directionality matters.

**ADDED — Phase-0 zero-training frozen-embedding arms** on a distance-agnostic sweep harness:
**DINOv2 *patch* distance is the serious candidate** — DINO-WM (arXiv:2411.04983) plans CEM on exactly
that cost on top-down nav with the agent in frame (Wall 0.96), and its ablation kills pooled features
(CLS 0.58); corroborated by arXiv:2507.01667 (global embeddings ≈ no relative pose). **No paper has
measured monotonicity at 0–60 cm robot scale** (VPR is 10–25 m retrieval; DINOv2 rotation-invariance is
a heading risk) — so our sweep measures what the literature hasn't. **VIP expected to lose** (double-OOD;
GVL finds embedding-distance values near-random OOD; pooled). Mechanistic prior for why patch-DINO ≠
flat where spatial SD-VAE-L2 is: VAE cells encode appearance stats (carpet/wall cells dominate), DINO
patches encode landmark *identity* at grid positions. V-JEPA 2.1 token arm included (codec already in
nanowm); **the arm ranking doubles as codec selection for any semantic-latent retrain.**

**ADDED — complementary tracks:** (1) **GCBC proposal prior for CEM** (GCSL/RvS; hindsight relabeling →
millions of tuples from 45K frames; gradient where L2 is flat + biases CEM toward in-distribution action
sequences = less WM hallucination exposure); (2) **MASt3R-SLAM pose oracle, eval-only**
(arXiv:2412.12392; poses for all 45K frames → grade d_learned globally, catch wormholes at graph build —
closes pitfall #8's verification gap); (3) **recollection co-design** for WM+metric+graph (perimeter
coverage, loop closures through the rug centre, both approach directions, slow approaches, exposure/WB
lock + udev camera pin first).

**Finding #4 REINTERPRETED (JEPA question):** the DINO/V-JEPA action-conditioning atrophy is evidence
about **diffusion-forcing ⊗ semantic latents**, not the latents — every published semantic-latent WM
success uses deterministic regression (DINO-WM MSE / V-JEPA-2-AC L1+rollout / DINO-world) or
x0-prediction (V-JEPA 2.1 nav WM). Mechanism: temporally-smooth features → denoiser copies context →
action starves (SD-VAE's texture flicker kept ours alive). Bonus directly relevant to the live-frame
blocker: regression predictors **degrade to benign averaging OOD** instead of snapping to vivid wrong
scenes (our nearhamper failure; arXiv:2605.06388 also shows semantic latents preserve action geometry
*better* under regression: IDM corr 0.83 vs 0.51 VAE). Retrain decision point = the coverage retrain;
recorded in [[open-questions]] "Semantic-latent WM retrain".

**Phased plan with numeric gates** (full detail [[learned-distance-metric]] "Sequencing"): Phase 0 =
latent cache + harness + frozen arms → **Gate A** (ρ>0.9 to 60 cm radial+lateral, slope>3σ at 40–60 cm,
yaw basin preserved); Phase 1 = rung-0 (+distillation variant if DINO wins Gate A) → **Gate B**
(GO/NO-GO); Phase 2 = swap into `lekiwi_engine` + on-robot A/B on well-covered goals; Phase 3 = graph;
parallel = GCBC / SLAM oracle / recollection. **Order of attack: Phase 0 first — highest
information-per-hour; everything downstream branches on its result.**

## 2026-06-09 (build session) — Phase 0 CODE BUILT: sweep harness + frozen candidates + latent cache + imagined arm (smoke-tested)

Implemented all of Phase 0 (five tools, `scripts/`), smoke-tested locally end-to-end. **No training
anywhere — all candidates are formulas or frozen pretrained nets; the work is measurement.**

- **`sweep_common.py`** — the single source of truth three consumers import: (1) **letterbox/normalize
  exactly replicating `LekiwiPlanner._preprocess`** (torch bilinear `align_corners=False`, pad 0 in
  [0,1], then `*2−1`) — the 6b pixel-range-bug class is closed by construction; (2) the **sweep label
  grammar** (labels carry ground-truth pose: `r<cm>`, `lat±cm`, `yaw±deg`, `yawd±deg@r<cm>`,
  `g_r<cm>_b±deg`, `fork_<site>_<move>`, `noise`; legacy `<N>cm`/`yaw+10` accepted); (3) sweep-dir
  manifest IO (legacy `measure_dist_sweep.py` dirs auto-ingest, pod-absolute paths rebase — verified).
- **`capture_sweep.py`** — **GPU-free** robot-side capture (decouples capture from scoring; the old
  tool needed the WM+pod just to capture). Interactive labeled captures + bursts + goal snapshot +
  protocol/coverage helpers + the motorized yaw arm ported from `measure_dist_sweep.py`. Protocol
  documents the full Gate-A session (~50–60 placements incl. **lateral out to ±60 cm** — the far band
  is gated, not just small offsets).
- **`dist_harness.py`** — distance-agnostic grader. Per candidate × arm: Spearman ρ, far-band (40–60 cm)
  slope-per-10cm/σ, yaw-basin (argmin at center + depth/σ), yaw-at-distance (the under-credited-rotation
  check), grid ρ, fork rankings (margin/σ), noise σ as the universal denominator → **Gate A verdict** +
  `gate_report.md` + `metrics.csv` + per-candidate curve CSVs + clean-vs-imagined overlay plots.
  **Smoke-tested both paths on a synthetic sweep:** a saturating (plateau-like) signal correctly FAILs
  (ρ 0.54), a monotone one PASSes (ρ 1.0, fork winner correct, yawd basin at center).
- **`dist_candidates.py`** — frozen arms: `pixel_l1`; **`sdvae_l2` = the current planning objective**
  (diffusers `sd-vae-ft-mse`, engine-parity flat-L2; **posterior MODE, deterministic** — found that the
  engine's `encode_first_stage` default **samples**, so the engine's noise floor ≥ this one; documented);
  **`dinov2_mse`/`dinov2_cos`** (patch tokens per DINO-WM, vits14/vitb14); `vip_l2` + `vjepa21`
  (optional deps, graceful skip). sdvae+dinov2 **verified running locally on CPU**. Rung-0/1 learned
  heads will register here later → every metric ever built is graded on the same rig.
- **`build_latent_cache.py`** *(pod)* — direct parquet+mp4 read (lerobot-version-proof), chunk-boundary
  frames → exact preprocess → checkpoint-codec encode (deterministic) → `latents.npy` [N,4,32,32]
  (~75 MB) + `index.csv` RGB pointers + `meta.json` with every convention pinned. `--hf-vae` fallback.
- **`wm_imagined_arm.py`** *(pod)* — the 0d validate-first arm: rolls the WM from each radial capture
  with known straight chunks → combined sweep dir; imagined rows carry **raw WM latents** which
  `sdvae_l2` scores directly (`feature_is_wm_latent`, no decode→re-encode roundtrip) + decoded frames
  for image-space candidates; the harness overlays imagined vs clean. At an OOD start pose this arm
  quantifies the live-frame hallucination as a curve divergence.

**Remaining to close Phase 0 (needs hardware):** (1) robot capture session — re-probe the camera first
(USB-swap trap), `capture_sweep.py` protocol at a well-covered goal (`nearfan2`-style); (2) pod:
`build_latent_cache.py --ckpt step-8000`; (3) pod: `wm_imagined_arm.py` on the captured radial arm;
(4) `dist_harness.py` over all arms × all candidates → **the Gate A table**. See [[roadmap]] 6d,
[[learned-distance-metric]] "Sequencing".

## 2026-06-09 (paper review) — RAE-NWM (arXiv:2603.09241): semantic-retrain branch de-risked; Finding-#4 claim narrowed; dinov2 arm prior raised

Deep-read RAE-NWM (*Navigation World Model in Dense Visual Representation Space*, code
`github.com/20robo/raenwm`) — it is the "switch the WM latent space to DINOv2" experiment executed
end-to-end: **flow-matching CDiT over frozen DINOv2 patch tokens** (CLS dropped, 256×768d), CEM planning
with a **token-space DINO cost (no decode)**, decisively beating VAE-latent NWM (Habitat SR 78.95% vs
43.33%, ATE 2.91 vs 4.12, LPIPS@16s 0.349 vs 0.470). Takeaways folded into [[learned-distance-metric]]
"If the retrain switches latent space" + refs, and [[open-questions]] "Semantic-latent WM retrain":

- **Finding-#4 claim NARROWED:** their generative (flow-matching velocity) predictor over semantic
  latents works — action branch did not starve (AdaLN-gated conditioning). So "generative ⊗ semantic"
  is published-working; the unproven combination is *diffusion-forcing* specifically.
- **`dinov2_cos` Phase-0 arm prior raised:** two nav planners (DINO-WM + RAE-NWM) now succeed with the
  token-space patch-DINO cost. But the paper never measures far-field flatness (8 m episodes, 1 m
  success radius, 8-step rollouts) → **Gate A stands; the sweep still measures what the literature
  hasn't.** No metric / subgoal / graph machinery in the paper → build order untouched.
- **Steal-list for the semantic retrain:** frozen RAE decoder for viz (answers the lost-decodability
  cost), AdamW lr 2e-4→2e-6 / wd 0 / eff-batch 96 / 50 epochs / 2×A800 ~2 days, CEM 120×8-step.
  Scale: single-env 1K-trajectory regime demonstrated (~20× our 50 eps → recollection matters
  regardless of codec). Their DINOv2 failure mode (high-freq stochastic texture, grass) is benign for
  our featureless-carpet room.

Next: unchanged — close Phase 0 (capture session + pod cache/imagined runs → Gate A table).

## 2026-06-10 — GATE A RUN (sweep_nearchair): sdvae_l2 FAILS the far band as diagnosed; frozen DINOv2-patch PASSES; clean↔imagined weld LOOSE (23σ) — distillation variant + +1-weighted cost activated

Phase 0 closed on real data. Capture session at the `nearchair` goal (guided `capture_sweep.py`,
40 placements: noise×8, radial r10–r60, lateral ±10–±60, yaw ±30°, yawd±20°@r40, grid ×6; **forks not
captured** — appendable later, guided mode auto-skips done arms). Pod side: latent cache rebuilt
(4,500 × [4,32,32], step-12000 codec, deterministic mode), `wm_imagined_arm.py` rolled 18 imagined
latents (6 radial starts × 3 straight chunks), `dist_harness.py` graded 4 frozen candidates.
Artifacts: `results/sweep_nearchair{,_imagined}/`, `results/dist_harness_nearchair/` (gate_report.md,
metrics.csv, per-candidate curves). Placement was tape-measure-approximate — fine for the verdicts:
Spearman ρ needs only ordinal correctness and the pass/fail margins are 2–7×.

**The Gate A table:**

| candidate | radial ρ | far-band slope/σ (radial / lateral) | yaw basin | verdict |
|---|---|---|---|---|
| pixel_l1 | 1.00 | 706 / 386 | centered, 5686σ | FAIL (lateral ρ 0.878) |
| **sdvae_l2 (current objective)** | 1.00 | **1.25 / 0.80** | centered, 9.3σ | **FAIL** |
| dinov2_mse (vits14 patch) | 0.943 | 11.8 / 20.2 | centered, 101σ | **PASS** |
| dinov2_cos (vits14 patch) | 0.943 | 12.0 / 21.5 | centered, 102σ | **PASS** |

**Conclusions:**
1. **The objective bottleneck is now a number.** sdvae_l2 is perfectly *ordered* to 60 cm (ρ=1.0) but
   its 40–60 cm gradient is **below the same-pose noise floor** (1.25σ/0.80σ vs the 3σ gate): one chunk
   (~3 cm) moves the cost by ~0.3 against σ=0.77 standing still. CEM cannot see far-field progress —
   the measured mechanism behind the 6b far-goal stalls and the under-credited rotate-to-face.
2. **Information is in the pixels; the failure is the representation.** pixel_l1 and DINOv2 have huge
   far-band SNR on the *same frames*; yaw basin sharp + exactly centered for every candidate; lateral
   monotone to ±60. No perceptual-aliasing wall at room scale — SD-VAE-L2 buries the pose signal,
   as designed-for in [[learned-distance-metric]].
3. **Frozen DINOv2-patch is a working room-scale distance with zero training** — the measurement no
   paper had (DINO-WM/RAE-NWM never graded 0–60 cm flatness). Heading stays sharp (the
   rotation-invariance worry didn't materialize). ⇒ **rung-0 patch-DINO distillation variant
   ACTIVATED** (per the Gate-A trigger in [[learned-distance-metric]] "Rung-0 additions") and the
   **semantic-retrain branch gets its empirical prior** (codec-selection signal).
4. **Clean↔imagined weld is LOOSE — the VALIDATE-FIRST gate fired.** On the raw-latent path (no
   decode): imagined latents sit **+17.7 above the clean curve (23σ)**, ρ collapses to 0.29, and
   within every rollout d *increases* as the robot nominally approaches (+1→+2→+3 worse at all six
   start ranges) — rollout-step degradation outruns approach signal. ⇒ **fold WM-rolled-out latents
   into φ's training/validation set** (decision resolved YES) and **+1-weighted cost** strongly
   supported over endpoint-only. (dinov2's weld number is confounded by decode blur — it scores
   decoded imagined frames; only sdvae_l2 reads raw ẑ.)
5. **Why near goals converged anyway:** the sdvae_l2 basin is deep/sharp near-field (9–12σ yaw). The
   system was never broken near goals — it is blind far from them.

**Not established:** wormholes/cross-region behavior (one goal, one region — needs the distance-field
viz over the latent cache + a second goal's sweep), action-ranking margins (fork arm uncaptured).

**Harness fixes found by this run (committed 7c03385):** (1) Gate A now grades **clean captures only**
— the first run mixed the 18 imagined rows into the radial curve and failed everything spuriously;
imagined rows get their own `radial_imagined` grade + **weld_offset(/σ)** metric. (2)
`wm_imagined_arm.py` now reshapes the engine's **flattened** rollout latents ([1,1+H,4096] not
[1,1+H,C,h,w]) to [4,32,32]. Also: guided protocol mode added to `capture_sweep.py` (c215d72 — walks
all 52 placements with placement instructions; resumable).

**⚠️ Ops incident (resolved):** a Mac→pod results sync deleted everything in `/workspace/results/`
except the synced sweep dirs (checkpoints included; pattern = `rsync --delete`-like). **Restored in
full from the RunPod NFS snapshot** `/workspace/.snapshot/big_catalog_2026-06-10_01_42_17_UTC` (taken
20 min pre-deletion); latent cache (built in the gap) rebuilt from scratch. Backstop discovered:
`kaushikpraka/nanowm-lekiwi-b2-f10-step8000` on HF. **Lesson: never `--delete` toward
/workspace/results; checkpoints exist nowhere else** (wandb has metrics only) — consider uploading
step-12000 to HF.

**Next:** Phase 1 (rung-0 learned distance) with patch-DINO distillation as the lead variant; bar to
beat = ρ>0.94 / far-slope>12σ. Optionally append forks + a second-goal sweep first. See
[[learned-distance-metric]] "Sequencing" Gates A/B.

## 2026-06-10 (decision + launch) — OPTION C: retrain the WM over frozen DINOv2 tokens; C0 probe matrix launched

Design session (operator + Gate A data) settled the route for using the DINO signal: **NOT distilling
DINO distances into a φ that reads SD-VAE latents (Option B — stacks two approximations: φ ≈
DINO∘decode, on 23σ-off rollouts), but retraining the WM to predict frozen DINOv2 patch tokens**
(flow-matching/x0, NOT diffusion-forcing) so the rollout space IS the validated distance space and the
CEM cost (token cosine = the DINO-WM/RAE-NWM cost) needs zero training. Data-demand analysis: not
significantly more (frozen encoder ⇒ dynamics-only learning; same 256-token sequence at B/1; semantic
latents MORE action-recoverable under regression; all small-data precedents semantic-side) — and the
OOD failure mode flips from vivid wrong-room hallucination to benign averaging (attacks blocker #1).
Full plan + rationale: [[semantic-wm-retrain]] (roadmap 6e). Decoder for viz now in scope (C0.5,
eval-only). Subgoal graph (C3) stays in scope, simplified: first edges from calibrated frozen token
distance, learned head only if drivability gaps appear.

**Prep (same day):** encoder parity confirmed — `facebook/dinov2-small` == Gate-A's `dinov2_vits14`
(224px, 16×16×384 tokens). **`latent_codec.latent_scale` added to nanowm (69fe01b)** — measured DINO
token std 2.4 on lekiwi → scale to ~unit for FM targets (SD-VAE scaling_factor analog; verified
std 0.999 post-scale). Decoder-less validation already supported (latent-only metrics path). 12-step
smoke run of the C0a config PASSED (flow loss 1.8→1.74, no decoder crash).

**C0 probe matrix launched** (`scripts/run_c0_probes.sh`, sequential, ~3k steps ≈ 3 h each):
C0a flow+adaln_fuse (the RAE-NWM-shaped bet) · C0b flow+cross_attention · C0c x0+adaln_fuse ·
C0d flow+additive (control — expected to reproduce Finding #4). Per-run kill-switch =
`action_diagnostic` (PASS: action-emb RMS > 0.05 AND GT-action latent-L2 < zero/random baselines);
verdicts accumulate in `results/c0_probe_summary.md`.

## 2026-06-10 (C0 RESULT) — action branch ALIVE over frozen DINOv2 tokens; injection is the decisive variable; winner = x0 + adaln_fuse; Finding #4 reproduced on demand

The C0 probe matrix (4 × ~3k steps, NanoWM-B/1 over facebook/dinov2-small tokens, latent_scale 2.4,
existing 50 episodes) is complete and unambiguous. Kill-switch = `action_diagnostic` (PASS: action-emb
RMS > 0.05 AND GT-action rollout latent-L2 < zero/random-action baselines):

| probe | recipe | action-emb RMS | GT vs zero margin | verdict |
|---|---|---|---|---|
| C0a | flow + **adaln_fuse** | 0.207 | 12.4 (216.7 vs 229.1) | **PASS** |
| C0b | flow + cross_attention | 0.029 | 2.0 (224.9 vs 226.9) | FAIL |
| **C0c** | **x0 + adaln_fuse** | **0.182** | **21.3 (204.1 vs 225.4)** | **PASS — WINNER** |
| C0d | flow + additive (control) | **0.0028** | 4.1 | FAIL — **Finding #4 reproduced exactly** |

Reads:
1. **Finding #4 is the injection, not the latents.** `additive` (the original SD-VAE run's setting)
   atrophies to RMS 0.0028 — the precise documented signature — while the SAME objective with
   `adaln_fuse` holds RMS 0.207. The published narrowing (diffusion-forcing ⊗ semantic ⊗ weak
   injection) is confirmed on our data in one controlled comparison.
2. **adaln_fuse ≫ cross_attention** here (0.207/0.182 vs 0.029) — the lit's "documented fix" is the
   weaker arm at our scale; RAE-NWM's AdaLN-gated choice is the right one.
3. **x0 > flow on action-conditioned rollout accuracy** holding injection fixed (margin 21.3 vs 12.4)
   — the V-JEPA-2.1-nav-shaped recipe wins at 3k steps. (Both pass; flow stays the fallback.)
4. SD-VAE's high per-step variance was never necessary — the action branch lives on semantic tokens
   when the conditioning path is right.

Artifacts: `results/c0_diag_C0*/action_diagnostic.{png,json}`, run dirs `results/*-C0?-dinoB1-*`,
`results/c0_probe_summary.md`. **Next: C0-ext — x0+adaln_fuse to 12k steps (launched), then the
Gate C ladder (offline CEM action recovery / weld re-test / nearhamper roll). C0.5 viz decoder
training in parallel.** See [[semantic-wm-retrain]].

## 2026-06-10 (C1 offline) — token-space planner integrated + smoke-tested on the 3k probe ckpt; three-way metric parity

While C0-ext trains, the C1 offline pieces landed (nanowm 00b2e76, parent ff47c56/a6413a6):
- **Cosine-objective layout bug fixed before it bit:** `objective.py`'s token-cosine assumed
  token-major flattening, but `DiffusionWorldModel.rollout` flattens `[C,h,w]` channel-major — the
  reshape would have silently mixed channels across tokens. Added `channels_first` (tokens =
  `reshape(C, hw).T`), plus **`mode="first"`** (+1-chunk cost — least WM-degraded, the chunk actually
  executed) alongside last/all.
- **Engine** (`lekiwi_engine.py`): `cost_metric` auto (mse for sd_vae, token-cosine for semantic),
  `cost_mode`, decoder-less viz gating, optional `token_decoder` (C0.5 checkpoint) for imagined
  strips; termination `dist_to_goal` now uses the SAME metric as the cost. `lekiwi_mpc.py` grew
  `--cost-metric/--cost-mode/--token-decoder`.
- **Smoke test on the C0c 3k-step probe ckpt PASSED**: boot `latent=[384,16,16] (webdino),
  cost=cosine/first`; `dist_to_goal` reproduces the Gate-A dinov2_cos curve on real sweep frames —
  noise 0.010 ≪ r10 **0.139** < r40 0.254 < yaw+30 0.352 (Gate A: 0.134 @10cm, plateau 0.28–0.42).
  CEM returns sane commands. → `--reach-thresh` for the semantic stack ≈ **0.05–0.10**.
- **`wm_token_cos` harness candidate added** (the deployment cost in the WM's own codec space;
  scores raw WM token latents directly). Parity: harness 0.1394 == engine 0.1394 on the same
  r10-vs-goal pair. The Gate-C weld re-test will grade `wm_imagined_arm` output with it.

Running: C0-ext (x0+adaln_fuse → 12k, `results/C0ext.log`, loss ~0.21 at this writing) and the C0.5
viz decoder. Next (blocked on those): Gate C ladder — action_diagnostic @12k, offline CEM action
recovery, weld re-test (beat SD-VAE's +23σ / d must FALL within rollouts), nearhamper roll.
On-robot A/B remains the operator handoff point.

## 2026-06-10 (C0.5 + C1 viz) — token→RGB decoder trained (val 0.045); full imagined-strip path verified

C0.5 decoder done (15k steps, `results/token_decoder/decoder.pt`): GT-vs-decode val grids nearly
indistinguishable (`val_015000.png`) — landmarks, robot arm, viewpoint all faithful. Engine smoke with
`token_decoder=` on the C0c probe ckpt produced the live|+1+2+3|goal strip
(`results/c1_smoke_strip.png`): imagined frames are SOFT-BUT-CORRECT renderings of the same scene —
the benign-averaging OOD behavior Option C was chosen for, visible already at 3k steps. C1 offline
scope complete; remaining C1 = on-robot A/B (operator). Everything now blocks on C0-ext → Gate C.

## 2026-06-10 (GATE C — PASSED) — the 12k semantic WM is planner-ready; 6b hallucination FIXED at the source; on-robot A/B is the handoff

C0-ext (x0+adaln_fuse, 12k steps, val 0.0435 still improving) graded on the full ladder:

1. **Kill-switch @12k: PASS, strengthening.** action-emb RMS **0.333** (3k: 0.182), GT-vs-zero margin
   **43.4** (3k: 21.3) — the action branch gains authority with training, the opposite of atrophy.
2. **Offline CEM action recovery (24 scenes, DDIM=3, token-cosine objective): at the WM ceiling.**
   reached_ratio **1.01** overall (translation 0.96 / pivot 1.10 / arc 1.04 / slow 0.94), **100%
   beat-do-nothing, 100% sign agreement on Δx AND Δθ** in every bucket, dxErr 1.4 cm / dθErr 2.2°.
   The planner can invert this model. (`results/offline_planning_c0ext/`)
3. **Weld re-test (wm_imagined_arm @12k + wm_token_cos): the SD-VAE pathology is mostly gone.**
   Imagined-vs-nominal ρ **0.876** (SD-VAE: 0.29) — imagined tokens preserve position ordering across
   starts; offset vs clean curve is ~constant +0.22 (harmless for argmin ranking); within-rollout
   degradation:signal ≈ **1.7:1** (SD-VAE: 10:1) — still favors `--cost-mode first`, as designed.
   Clean-capture grades unchanged (wm_token_cos passes Gate A: ρ .943/.976, slope 11–21σ, yaw 91σ).
   (`results/dist_harness_dino/`, `results/sweep_nearchair_imagined_dino/`)
4. **Hallucination re-test: FIXED.** Rolling from the nearhamper goal — the exact image that produced
   the wrong-room hallucination in 6b — the imagined strip stays in the SAME scene (hamper + dark
   object + arm, soft regression blur), token-dist-to-start 0.28–0.36 ≈ the in-dist nearfan2 control
   (0.31–0.38). (`results/hamper_retest_*.png`) ⚠️ One flag for the operator: the nearfan2 control
   strip renders soft with curtain-like streaks — plausibly the same scene dimly rendered, but
   ambiguous to the eye; judge on-robot.

**The semantic stack (6e) now beats the SD-VAE stack on every offline axis** — metric (Gate A),
action conditioning (43.4 margin), weld (ρ 0.876 vs 0.29), OOD behavior (same-scene vs different-room)
— using only the existing 50 episodes. Tooling patched along the way: `offline_planning_eval.py` grew
`--visual_metric auto` + decoder-less guard; `wm_imagined_arm.py` grew `--token-decoder`.

**HANDOFF → on-robot A/B (needs operator).** From the pod, tunnel up, then:
```
export WEBDINO_MODEL_PATH=facebook/dinov2-small   # REQUIRED — config default is webssl, fails loudly without this
/workspace/nanowm-venv/bin/python scripts/lekiwi_mpc.py execute --planner wm \
  --nanowm-src external/nanowm/src \
  --ckpt "/workspace/results/20260610_112629-C0ext-dinoB1-x0-adalnfuse-F4S10-lekiwi/checkpoints/across_timesteps/epoch=19-step=12000.ckpt" \
  --goal goals/nearfan2/goal.png --ip 127.0.0.1 \
  --token-decoder /workspace/results/token_decoder/decoder.pt \
  --reach-thresh 0.08 --rerun
```
(token-cosine scale: noise floor ~0.01, 10 cm ≈ 0.14, plateau ~0.28–0.42 → start `--reach-thresh`
0.06–0.10. A/B vs the SD-VAE step-12000 stack on the same goals; then `--cost-mode first` as the
second arm. Camera re-probe first if the Pi host restarted.)

## 2026-06-11 (ON-ROBOT) — semantic stack first closed-loop session: 3/3 physical arrivals, incl. ×2 on the goal the SD-VAE stack failed; floor ≈ 0.2 with cross-session goal images

First on-robot runs of the Gate-C stack (C0ext 12k x0+adaln_fuse, token-cosine cost, cost-mode
**last** this session, reach-thresh 0.08, full speed, ~7.2 s/replan). `.rrd`s:
`results/mpc_semantic_{nearfan2,nearchair1,nearchair1_pos2}.rrd`.

| run | goal | trace | physical outcome (operator) |
|---|---|---|---|
| 1 | nearfan2 | hovered 0.23–0.28 by step ~34–45 | "approximately the right position" |
| 2 | nearchair1 | **0.32 → 0.21 monotone ~20 steps**, vx pegged 0.10 mid-run | **reached the chair** (slightly left) |
| 3 | nearchair1, new start | 0.30 → **0.19** | **reached the chair again** |

- **The Gate-A prediction held on hardware**: nearchair1 is the goal where flat-L2 sat on a
  plateau and once commanded a hard wrong-direction turn; the token-cosine stack descended
  monotonically from the 0.3 band and arrived, from two different starts. Far-band gradient →
  committed full-clamp driving; near the floor → millimeter commands (healthy near/far behavior).
- **Termination never fired** — the floor is ~0.19–0.28 on all three runs vs reach-thresh 0.08.
  All three goals were **cross-session captures**; calibration's 0.01 noise floor was same-session.
  Hypotheses: (a) stale goal images lift the floor ~0.2 (⇒ recapture goals per session), vs
  (b) genuine basin floor (⇒ raise thresh to **0.2–0.25**, operator's working estimate).
  **Discriminating test (deferred): park at goal, re-snapshot, rerun — floor ≲0.05 ⇒ (a).**
- Within-floor position error is unconstrained (the "slightly left") — consistent with the floor
  being mostly capture-condition offset.
- **NOT yet run**: the SD-VAE baseline leg on the same starts (historical 6b behavior is the
  implicit baseline); the `--cost-mode first` arm; the fresh-goal floor test.

**Infra found+fixed this session (commits 87d1994…e0c4685):**
- **Host lifecycle mystery SOLVED**: lerobot's LeKiwi host serves for `connection_time_s` (default
  **30 s**!) then exits — every "connect timeout" of the day was the MPC's ~60 s checkpoint load
  losing this race (the GPU-free probe always won it). **Launch the host with
  `--connection_time_s 7200`** for a session. Client-side: `lekiwi_mpc` now retries connect ×3.
- **Viewer correctness**: planner freshness verified end-to-end (ZMQ CONFLATE + observe-before-plan
  + sweep evidence) — the "stale frame" was display semantics. Final design (operator-driven):
  **atomic per-timestep panels** (camera | imagined +1..+H | goal land together post-plan = one
  consistent timestep; raw `live` recorded at observe time but undisplayed), **dist trace in a row
  below** (2-row blueprint renders fine in today's web viewer; `--viewer-flat` fallback kept),
  status line announces `observed → PLANNING` so plan latency is visible, HUD/trace/threshold all
  in token-cosine units. Engine: token-decoder viz under `no_grad` (crashed the interactive driver).
- Post-reboot camera probes clean ×2 (no USB swap this time).

**Future avenues (operator):** multi-camera training; reverse driving (data + `VX_MIN=0` clamp +
CEM range) — logged in [[semantic-wm-retrain]] with C2 scoping.

**Next:** fresh-goal floor test → set reach-thresh; `--cost-mode first` arm (+ the deferred
`--gen-frames` truncation: 7.2 s → ~2.5 s replans); optional formal SD-VAE baseline legs; then C2
recollection (multi-cam + reverse capture co-design) + retrain.

## 2026-06-11 (C3 SUBGOAL GRAPH — BUILT + OFFLINE-VALIDATED)

**The token-space subgoal graph is built, audited, and integrated into the MPC** (on-robot test
pending). Far goals no longer ask the metric a far question: Dijkstra over real frames routes the
room; CEM only ever chases a waypoint ~one reach away — the regime that went 3/3 on-robot.

**Substrate** — `results/token_cache/`: all 4,500 chunk-boundary frames (50 eps, stride 10)
re-encoded with the 12k semantic ckpt's frozen DINOv2 codec → [384,16,16] fp16 + per-row JPEGs
(`--save-frames`, new). Metric = exact `lekiwi_engine._dist` parity (channel-major reshape →
256 tokens, 1 − mean per-token cos).

**Calibration** (`results/subgoal_graph/calibration.csv`, within-episode pairs at chunk gap k):
k=1 → 0.092, k=3 → 0.182, k=10 → 0.330 (medians). Far-pair median 0.454 (the plateau, again).
**τ = 0.182** = one CEM reach (H=3 chunks), read off the curve, not guessed.

**Graph** (`scripts/build_subgoal_graph.py` → `results/subgoal_graph/graph.npz`):
- 4,500 nodes; **4,450 temporal edges** (certified — the robot drove them) + **10,061 shortcut
  welds** (8,973 cross-episode, 1,088 loop-closure; d < τ, per-node degree cap 8, same-episode
  pairs within 5 chunks excluded).
- Connectivity: 50 disconnected threads → **ONE component, 100% coverage**.
- **Wormhole audit PASS**: shortcuts ranked by temporal-only endpoint separation (highest-leverage
  = most suspicious); top-leverage + random montages (`audit/*.png`) all show the same physical
  place at near-identical pose. The low-texture-rug wormhole risk did not materialize at τ=0.182.

**Offline route validation** (`scripts/subgoal_graph.py --route`, simulated replan loop, filmstrips
in `results/subgoal_graph/route_*.png`): 3/3 routes reach ENDGAME — row0→nearchair (16 waypoints,
stitches ≥3 episodes), row2253→nearfan2 (3), row4498→**nearhamper** (16 waypoints across ≥6
episodes into the formerly-OOD region, ends graph_dist 0.075). Filmstrips read as one continuous
drive; per-step progress ~0.13–0.15 token-cos ≈ one CEM reach. Goal-image localization d_loc
0.19–0.27 — the cross-session offset again, exactly as on-robot.

**Design decision (offset-driven):** waypoint lookahead = **route progress** (graph_dist[src] −
graph_dist[node] < τ, within-session calibrated units) — NOT live-frame distance, which the ~+0.2
cross-session offset would collapse to 1-hop (~2 cm) crawling.

**Runtime** (`lekiwi_mpc.py --graph results/subgoal_graph`): per replan localize live frame
(k-NN over cached tokens, **11 ms** measured — free vs the 7 s plan) → walk the goal-rooted
Dijkstra tree → waypoint's REAL cached frame becomes the plan goal through the unchanged
`planner.plan` path. ENDGAME (graph_dist < τ) hands CEM the actual `--goal` image, so the final
approach keeps the validated reach-thresh semantics; **reach-thresh termination is gated to
endgame only** (dist_to_goal is to-waypoint while routing). Viewer: goal panel = current waypoint;
second time-series `graph_dist` = calibrated route distance to the FINAL goal (the monotone
progress signal `dist_to_goal` never was). GPU smoke test green (WAYPOINT plan + ENDGAME trigger).

**Open before on-robot:** (a) the localization floor on LIVE frames is the same cross-session
offset question (C1's fresh-goal test now covers both); (b) waypoint switch cadence under real
(imperfect) motion — replan-every-step recomputes everything, so failure mode is dithering, not
lock-in; (c) stuck-hop edge removal (SGM-style) deferred until observed.

**Next:** on-robot graph run (same goals, cross-room starts the flat planner could never do);
then C2 recollection feeds straight into a denser graph (the builder is data-driven end-to-end).

### Addendum (same day) — the graph must be DIRECTED (operator catch)

The chair→hamper route test exposed it: the undirected build routed **backwards through episode
threads** (node runs 28→27→26, 731→…→728, 358→…→344 — temporal edges against the driving
direction). The robot has **no reverse** (VX_MIN=0, no backward training data); an
against-the-flow waypoint is behind the robot at a heading CEM's forward-only H=3 plans cannot
re-acquire (turn-drive-turn for a ~2 cm hop is unplannable and out-of-distribution).

**Fix:** temporal edges traversable ONLY in the driving direction (`graph.npz` already stored
them in driving order — only the adjacency symmetrized them); shortcut welds stay bidirectional
(pose-identifications, "same place+heading" by the sharp yaw basin — not motion). `set_goal`
Dijkstra now runs over the REVERSED edges, so dist-to-goal is true forward-drivable distance.

**Directed connectivity (the honest metric):** 92 SCCs; largest SCC 4,333/4,500 = **96.3%**;
can-reach-core **98.2%**, core-can-reach **98.1%**. Stragglers = episode tails (chunks after a
thread's last weld are one-way outlets) — expected, harmless.

**Re-validation:** chair→hamper now traverses every within-episode run in ascending chunk order —
48 waypoints / graph_dist 7.75 vs the illegal 41 / 6.56 (the price of looping around instead of
backing up); final waypoint vs goal image = same hamper scene, near-identical pose
(`route_nearchair_nearhamper_directed.png`). The three earlier routes re-pass unchanged
(16/3/17 steps to ENDGAME) — they were already forward-flowing.

**C2 tie-in:** reverse-driving capture (already a C2 future avenue) would make temporal edges
bidirectional where reverse data exists — directly shrinking route lengths.

### Addendum 2 (same day) — welds must be DIRECTION-CERTIFIED too (operator catch #2)

One-way temporal edges weren't enough: the row2253→nearfan2 route still ended with the goal
BEHIND the arrival node. Cause: welds admitted at τ=0.182 span up to ~3 chunks of pose in ANY
direction — each bidirectional weld can silently move the virtual pose ~10 cm backward, and the
endgame then needs reverse.

**Threshold sweep ruled out the simple fix**: tightening welds to identification radius collapses
the graph (τ_weld=0.10 → largest SCC 32%, 3/4 routes UNREACHABLE) — 50 episodes don't revisit
poses precisely enough.

**Fix: motion-parallax direction certification** (zero new data; uses the threads we already
trust). For candidate weld i→j: **fwd** = some 1–3-chunk temporal successor of i gets closer to j
(j provably ahead; margin 0.015); **soft** = only the approach history certifies (predecessors
were farther) — j may be ≤2 chunks behind, so soft edges carry a +0.15 Dijkstra penalty and are
used only when no fwd alternative exists; **ident** = d < τ_id (k=1 median 0.092, same pose) —
bidirectional. Result: **17,796 directed welds (1,760 ident / 6,636 fwd / 9,400 soft)**, largest
SCC **94.5%**, can-reach-core **97.4%** — connectivity essentially recovered with direction
guarantees where the data supports them.

**Re-validation:** fan route restructured — three fwd-certified welds, the single soft edge only
at the terminal hop into the goal node (where ENDGAME takes over; residual ≤ ~2 chunks ≈ the
existing endgame slack, recoverable by replan + legal rotation). All other routes re-pass
(nearchair 14 steps, nearhamper 16, chair→hamper 50). `graph.npz` format bumped
(`directed_welds=True`; GraphNav rejects the old format with a rebuild message).

**Honest residuals:** soft edges bound backward error to ~2 chunks but don't eliminate it; the
final-cm placement question stays with the C1 floor work and the visual-servo endgame idea
(which can use reverse — it bypasses the WM, so VX_MIN=0 doesn't apply). C2 recollection with
reverse segments turns temporal edges bidirectional where reverse was actually driven.

## 2026-06-11 (evening) — FIRST ON-ROBOT GRAPH A/B — INCOMPLETE, resume tomorrow

Goal: nearhamper1 from a far start; run 1 = flat semantic MPC, run 2 = `--graph`. **Test cut short
by operator (pod shutdown); both `.rrd`s saved** (`mpc_semantic_nograph_nearhamper2.rrd` 16 steps,
`mpc_semantic_graph_nearhamper.rrd` 92 steps).

**Run 1 (no graph) — baseline confirmed the far-field failure:** dist pinned 0.45–0.50 for all 16
steps, theta oscillating ±20–26°/s — plateau wandering, no descent. Exactly the behavior the graph
exists to fix.

**Run 2 (--graph) — partial:** goal → node 705 (52-hop route, graph_dist 7.90). Real progress:
**graph_dist 7.90 → 7.07 (52 → 47 hops) over ~90 steps (~11 min)**, full-clamp segments, viewer
showed route strip + indexed waypoint banner + monotone route-dist trace working as designed.
**Then a STALL:** 7+ consecutive replans localized to the same node (src 4087 ep45 ck37, wp 2/47 =
node 4089 two chunks ahead, dist-to-waypoint pinned 0.29 ≈ d_loc 0.245 + ~1 hop). Candidates, in
order of prior: (a) localization stickiness under the ~+0.23 cross-session offset (live frame
between nodes keeps snapping to 4087); (b) CEM gradient too weak at 0.29-to-waypoint (the offset
eats the basin — cf. the C1 floor question); (c) a bad weld/route segment near ep45 ck37–39;
(d) physical obstruction (operator to confirm from the room). **First diagnostics tomorrow:**
review the .rrd around steps 85–92 (live vs waypoint panels side by side), check commanded vs
observed motion, try `--graph-lookahead 0.3` (waypoint farther ahead = stronger gradient through
the offset), and the fresh-goal/cross-session-offset test (C1) which this stall may share a root
cause with.

**Ops traps found+fixed this session:**
- **Wrong-camera suspicion after the day's first run** → Pi reboot + probe protocol: `top` verified
  correct post-reboot (`results/cam_probe5/`). Always probe after Pi restarts (USB swap trap).
- **Zombie MPC streamed velocity through the tunnel** after a wrapper-level kill — survived the Pi
  reboot (ZMQ re-attach) = phantom motion. Stop procedure: SIGINT the **python** PID (not the nohup
  wrapper), verify `ss` shows 9090/9877 freed + pgrep sweep empty.
- **Pod restart → new RunPod IP/SSH port** → Mac `tunnel_up.sh` must be re-pointed; MPC connects to
  `--ip 127.0.0.1` (reverse tunnel), NOT the Pi LAN IP.

**Resume-tomorrow checklist:** (1) Pi host with `--connection_time_s 7200`; (2) Mac tunnel to the
pod's NEW address (+ `-L 9090/-L 9877` for the viewer); (3) `--ip 127.0.0.1`; (4) camera probe;
(5) re-place bot at the same far start for run 2 retry; commands in this entry's runs are exact.

---

## 2026-06-12 — C3 GRAPH NAV: FIRST FULL ON-ROBOT SUCCESS (REACHED) + the three fixes that got it there

**Headline: `--graph` run REACHED nearpurifier — dist 0.08 < reach-thresh 0.08 after 129 steps**
(`results/mpc_semantic_graph_nearpurifier4.rrd`, 542 MB): 40-hop route from node 214 (ep2 ck34),
localization `[tracked]` essentially throughout (one clean same-thread reroute), hops burned
40→12→11→2, ENDGAME at step 116 (graph_dist < τ → real goal image), then CEM closed 0.30→0.08 in
13 endgame steps. First time the full pipeline (token graph → directed certified welds → sticky
localization → waypoint chain → endgame) worked end-to-end on the robot.

**A/B basin bracket (baseline runs, no graph):**
- nearpurifier from start-dist 0.35: **arrived without the graph** — 0.35→0.10 in 52 steps, classic
  endgame signature (x.vel→0, micro heading corrections), floor ~0.10 (cross-session offset;
  stopped above the 0.08 thresh). `mpc_semantic_nograph_nearpurifier.rrd`.
- nearhamper from start-dist 0.45–0.47: plateau-wander again (16-step replication of yesterday).
- ⇒ live CEM basin edge is between 0.35 (in) and 0.45 (out) — matches the offline calibration below.

**Fix 1 — calibrated waypoint spacing (e3f1e49).** Offline gradient-basin measurement (4,400
within-episode pairs): one-step descent reliability toward a target k chunks ahead = 95.8% (k=2),
94.3% (k=3), 90.9% (k=4), 75.8% (k=10), ~60% (k=20, noise). Route-progress is ~0.092/chunk, so
lookahead 2τ ≈ 4 chunks ≈ the 90% point → new default. ALSO: soft-weld +0.15 penalties no longer
count as lookahead "progress" (they're routing costs, not distance — one soft weld ate 40% of the
budget and pinned waypoints 1 hop ahead). Offline waypoint counts: purifier 51→29, hamper 9→5.

**Fix 2 — localization hysteresis + route stickiness (b63c64c).** First graph purifier run (32
steps, stopped): goal node 823 verified CORRECT (frame = the purifier approach), but per-replan
global-argmin localization flip-flopped across episode threads (ep18→27→23→27→29 at d_loc
0.22–0.28), the route re-rolled every replan (65/66/67/73 hops), waypoints chased unrelated
targets, graph_dist ROSE 10.26→11.67, theta oscillated ±28°/s. Since the next-hop tree is fixed
per goal, ALL route instability is src jumps ⇒ track src along the committed path: localize within
the first 12 path nodes; a global match must beat the on-path match by 0.03 for 2 consecutive
replans before a reroute is accepted. Offline alias replay (the run's actual lookalike nodes):
holds the thread, graph_dist monotone.

**Fix 3 — waypoint floor (operator call: "more difference between live frame and target").**
wp = at least path[3] regardless of budget → CEM gets a visibly different target frame and commits
(decisive turns instead of timid near-zero corrections). This was the config that REACHED.
Sparsified further post-success: **floor 3→5 hops + lookahead 2.5τ (a773ef7) — NOT yet tested
on-robot** (offline: purifier 9, neardesk 12, hamper 2 waypoints).

**neardesk graph run (147 steps, stopped; pre-sparsify config):** 59-hop route, tracked
throughout, ENDGAME at ~step 115, but endgame HOVERED at dist ~0.30 for 35 steps without crossing
0.08 (`mpc_semantic_graph_neardesk.rrd`). Purifier closed to 0.08; neardesk didn't ⇒ endgame
convergence is goal-image dependent (cross-session offset + view match) — same C1 floor question.
Watch: if endgame hovers ≥0.3, consider re-snapshotting the goal or reach-thresh 0.1–0.15.

**Viewer (addce3e):** route strip + goal banner now show the ACTUAL subgoal sequence —
`GraphNav._pick()` (shared selection rule) + `subgoal_chain()` replayed down the committed path;
strip tiles = exactly the frames CEM will be handed (`subgoal j/M (hop k/N)` + GOAL image); banner
`SUBGOAL 1/M  hop k/N`. Offline: live pick sequence == chain prefix on both demo routes.

**Ops — the evening blocker (UNRESOLVED):** `connection_time_s` is NOT a CLI flag —
`lekiwi_host.main()` hardcodes `LeKiwiHostConfig()` (default 30 s). Fix = edit the dataclass
default on the Pi: `sed -i 's/connection_time_s: int = .*/connection_time_s: int = 86400/'
$(<host-python> -c "import lerobot.robots.lekiwi.config_lekiwi as m; print(m.__file__)")` — but
the evening relaunches still died at connect (engine loads ~90 s; a 30 s host window is always
dead by then), so the sed likely hit a different install than the host's venv. **Verify on the Pi
with the host's exact python:** `<host-python> -c "from lerobot.robots.lekiwi.config_lekiwi import
LeKiwiHostConfig; print(LeKiwiHostConfig().connection_time_s)"` must print 86400 before relaunch.

**Resume checklist (next session):** (1) verify host window prints 86400 (above) + run host in
tmux; (2) tunnel up (`-L 9090 -L 9877` too); (3) relaunch neardesk graph run = first on-robot test
of the 5-hop sparse subgoals + subgoal-strip viewer (exact command in `mpc_graph_neardesk2.log`
header / this entry's sibling); (4) demo set for the write-up: nearhamper-from-far A/B (strongest
story: baseline provably fails on tape), 2nd purifier from a new corner, neardesk. **Project
direction (operator): write up as website + LinkedIn post from current data; full-room C2
recollect is the published next chapter, NOT a blocker. Inference-speedup plan (7.3 s→~1 s:
bf16+flash → compile/CUDA-graphs → CEM warm-start → DDIM/pyramid; TRT assessed not-worth-it for
the WM, DINO-TRT only if MPC goes continuous) saved in the plan file for later.**

---

## 2026-06-13 — Write-up scaffold + nearhamper graph attempts + recordings published

**Website scaffold (4ba686c).** Started the project write-up as a hand-rolled static site under
`docs/` for GitHub Pages (decision: long-form technical blog post, distill-style). `docs/index.html`
= masthead + hero-video slot (YouTube embed placeholder for phone footage, to be filmed) + written
TL;DR + 15-section build-log outline (problem → data → WM → run-001 failure → signal debugging →
run-002 → CEM → hardware → flat-landscape failure → first arrival → semantic pivot → on-robot
semantic → subgoal graph → the three fixes + REACHED → limitations/part-2); each unwritten section
carries a `draft` note with its beat + numbers + which figure. `docs/style.css` = serif/44rem
distill styling. `scripts/build_site_assets.py` = manifest-driven copy+downsize of 14 figures/videos
into `docs/assets/` (3.9 MB; PIL, no ImageMagick/ffmpeg needed). **Prose is the remaining work** —
sections are scaffolded, not written. To preview: `cd docs && python3 -m http.server 8000`. To
deploy: repo Settings → Pages → `main`/`docs`.

**nearhamper graph runs (no success).** Several `--graph` attempts at nearhamper1 across the day.
The run that progressed: step 1 dist 0.47 / graph_dist 7.14 / 52 hops → step 62 dist 0.48 /
graph_dist 6.12 / 40 hops (`[tracked]`, real route progress) → but by step 108 it had **regressed**
to graph_dist 7.20 / 48 hops, dist hovering ~0.5; stopped without reaching
(`mpc_semantic_graph_nearhamper3.rrd`, 423 MB; an earlier interrupted attempt =
`mpc_semantic_graph_nearhamper2.rrd`, 417 MB). nearhamper from start-dist ~0.45 remains the hard
case (outside the live CEM basin edge of 0.35–0.45) — it's the **strongest A/B story** for the
write-up precisely because the no-graph baseline provably fails here, but the graph run still needs
to land it. The 5-hop sparse-subgoal + subgoal-strip-viewer config (a773ef7/addce3e) got its first
robot exposure here but no clean reach yet.

**Ops — host blocker recurred AGAIN.** Same `DeviceNotConnectedError` at `robot.connect()`: engine
+ graph load fine (goal → node, 4384/4500 routable) but the LeKiwi client times out waiting for the
Pi host. The `connection_time_s` dataclass edit still hasn't stuck on the host's actual venv —
**verify before every session** with the host's python:
`<host-python> -c "from lerobot.robots.lekiwi.config_lekiwi import LeKiwiHostConfig; print(LeKiwiHostConfig().connection_time_s)"`
must print 86400. After a manual host restart one relaunch did connect and ran ~18 min before being
stopped.

**Run-launch style (memory).** Built then reverted a `run_mpc.sh` preset launcher
(bba7e0c→2b415d1): operator prefers **plain foreground python one-liners**, no wrapper scripts, no
nohup/log piping. Three canonical commands (semantic-graph / semantic-nograph / sdvae) recorded;
only `--goal` and `--rerun-save` change per run.

**Recordings published off-machine.** `.rrd` files can't go in git (up to 565 MB each, 5.6 GB total,
over GitHub's 100 MB/file limit; LFS free tier is 1 GB). Curated **keepers** (successes + A/B demo
material, ~2.1 GB / 8 files) uploaded as **GitHub Release `recordings-v1`**:
https://github.com/KaushikTheProgrammer/NanoNAV/releases/tag/recordings-v1 — index +
what-each-shows in [[RECORDINGS]]. Full set stays on `/workspace/results` (survives a pod stop).
