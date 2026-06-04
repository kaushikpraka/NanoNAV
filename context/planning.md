# Planning

## Stage 6 — Implementation Plan (eval-grounded, 2026-06-03)

> The sections below this one are the original (Run 001 / f=5) design. **This section is the current
> Run 002 / f=10 plan**, grounded in the cross-checkpoint + long-rollout evals and mapped onto the
> CEM/MPC code that **already exists** in the fork.

### What already exists (reuse, don't rebuild)
- `src/planning/cem_planner.py` — **`CEMPlanner`**: generic CEM (sample → batched WM rollout → score →
  refit elites). Args: `action_dim, horizon, num_samples, topk, opt_steps, var_scale, eval_every,
  sigma_min, action_low/high, rollout_batch_size`; has `.plan()`.
- `src/planning/diffusion_world_model.py` — **`DiffusionWorldModel.rollout`** (autoregressive, batched;
  verified to extend past H=3 via sliding context — see the long-rollout eval).
- `src/planning/objective.py` — latent-distance objective; `src/planning/preprocessor.py` — action
  (de)normalization (`denormalize_action` → velocity space).
- `src/experiments/planning_experiment.py` — **`PlanningExperiment.planning()`** wires checkpoint → WM →
  env → goals → CEM → rollout/score, with `_sample_dset_goals` (replay GT actions `goal_H*f` steps to
  pick a reachable goal frame). CEM config: `src/configs/planning/planner/cem.yaml`.

### What's missing for LeKiwi (the actual Stage 6 work)
1. **A LeKiwi planning "env".** `src/planning/envs/` has only a registry; the existing envs are sims
   (pusht/pointmaze/…). LeKiwi has no simulator, so:
   - **6a (offline, GPU):** a **dataset-replay harness** — no robot. Encode a held-out start frame, pick a
     goal frame `goal_H` chunks ahead (`_sample_dset_goals`), run CEM to recover an action sequence whose
     predicted rollout matches `z_goal`, then **decode-and-visualize** the planned rollout vs the GT path.
     Metrics: final predicted-vs-goal latent-L2, and recovered-vs-GT action error.
   - **6b (closed-loop, on-robot):** a real-robot env via the lerobot robot interface — capture frame →
     encode → CEM → execute first chunk's velocity → re-observe → repeat (stop-and-plan MPC).
2. **integrate_se2 action wiring**: CEM `action_dim=2`; `action_low/high` from the f=10 action stats
   (mean [0.0221, −0.0006], std [0.0141, 0.0707]); `denormalize_action` → `(Δx, Δθ)` → base velocity
   `v_x = Δx/(f·Δt)`, `ω = Δθ/(f·Δt)` with **`f·Δt = 10/30 = 0.333 s`** (Run 002 chunk; was 0.167 s at f=5).
3. **A `src/configs/planning/lekiwi.yaml`** (checkpoint, env=lekiwi-replay, CEM params below).

### Eval-grounded parameters
- **Checkpoint: step-8000** — `…/20260603_160326-NanoWM-B-2-F4S10-lekiwi/checkpoints/across_timesteps/epoch=13-step=8000.ckpt`
  (best rollout quality in the cross-checkpoint eval).
- **Horizon H = 3–5 chunks.** Long-rollout eval: pred-vs-GT latent-L2 stays low (~25–32) through ~step 5
  then accelerates → keep MPC lookahead ≤5 and **replan every chunk**. At f=10 a chunk ≈ 3.3 cm / ~6°, so
  H=3 reaches ~10 cm, H=5 ~17 cm.
- **CEM:** ~64 samples × 5 opt-steps × top-10 elites (start from `cem.yaml`, tune).
- **DDIM:** 20 for planning, 50 for final viz.

### ⏱ Measured inference latency (step-8000, H100, **uncompiled**, sequential DDIM — 2026-06-03)
| config | wall-time | per-sample |
|---|---|---|
| batch 1, H=3, DDIM=20 | 0.82 s | 818 ms |
| batch 64, H=3, DDIM=20 | **29.1 s** | 454 ms |
| batch 64, H=5, DDIM=20 | 58.1 s | 908 ms |
| **CEM replan** 64×5×H3, DDIM20 | **~146 s** | |
| CEM replan 64×5×H5 | ~291 s | |
| CEM replan 64×3×H3 | ~87 s | |

**A default CEM replan is ~2.4 min — the old "~1 s" estimate was ~100× optimistic.** Sequential
diffusion-forcing (3 frames/chunk × DDIM each = 60 forwards/chunk), **compute-bound at batch 64**
(attention already uses `scaled_dot_product_attention`/flash — no structural win there).

**`torch.compile` measured (default mode):** only **~1.13× at batch 64** (1.26× at batch 1) — because
the CEM batch is compute-bound, not overhead-bound, so compile barely helps. It is NOT the big lever.
(`reduce-overhead`/CUDA-graphs won't add much either — launch overhead is negligible at batch 64.)

**The real levers reduce compute** (latency is ~linear in each): DDIM steps, num_samples, opt_steps, H.
**Measured** (see "Few-step sampling" below): DDIM=3 + 32 samples + 3 opt-steps + H=3 → **~6.9 s/replan**
(~6 s compiled), a **~21× win** over the DDIM-20 / 64×5 default (146 s), with no quality loss. Below ~5 s
needs **step-distillation** (DDIM→1–2) or a smaller model; real-time (167 ms) is out of reach for the
prototype. **6a (offline) is unaffected** — latency irrelevant; just optionally DDIM=10 to speed the
sweep. **6b (closed-loop)** targets ~10 s/replan stop-and-plan. (The "Runtime Analysis" section below is
the original optimistic estimate — superseded.)

### Few-step sampling — DDIM=3 validated (2026-06-04); distillation NOT needed
Cheap-settings eval on step-8000 (`results/cheap_ddim{5,3}_step8000/`) — few-step vs the DDIM-20/50
baselines on the planning-critical signals. **DDIM=5 and DDIM=3 both hold:**
| metric | DDIM 50/20 | DDIM 5 | DDIM 3 |
|---|---|---|---|
| gate separation (random−GT) | 10.4 | 10.8 | 10.5 |
| motion trans / rot / arc latentL2 | 30.4 / 35.0 / 37.4 | 28.6 / 32.5 / 36.2 | 27.9 / 33.6 / 35.8 |
| controllability L-vs-R / straight-vs-stop | ~62 / ~37 | ~57.5 / ~34 | ~57.8 / ~34.9 |
| controllability pivot-L-vs-R (pure rotation) | ~64 | ~58.8 | ~54.6 ⚠ |

Gate separation + motion-tracking + arc/turn controllability hold flat; the only signal that softens is
**pure-rotation (pivot) control** (64→58.8→54.6) — still ≫ the ~12 noise floor, but it's what degrades
first, so **DDIM=3 is ~the floor** before distillation. (Futures are low-entropy/near-deterministic, so
few steps capture the mode.)

**Measured latency (uncompiled, H100), cheap CEM config 32 samples × 3 opt-steps × H=3:**
| DDIM | per-sample | replan |
|---|---|---|
| 20 (default-ish) | 454 ms | ~45 s |
| 5 | 115 ms | 11.3 s |
| **3 (recommended)** | **70 ms** | **6.9 s** (~6 s compiled) |

Latency is ~linear in DDIM steps (compute-bound). ⇒ **Run the planner at DDIM=3, 32 samples, 3 opt-steps,
H=3 → ~7 s/replan with no quality loss — a ~21× win over the DDIM-20/64×5 default (146 s).** LCM
distillation (DDIM→1–2) stays a back-pocket option only if real-time is later needed (and would mainly
buy back pivot control at very low steps). On a cheaper GPU these replans are ~2–3× longer (fine for 6a
offline; for 6b closed-loop the H100 is the better box).
- **Scoring:** latent-L2 (`objective.py`), valid **<~30 cm**; beyond that the landscape flattens → use the
  **waypoint scaffold** (Solution 1 below) for longer routes.

### 6a — Offline Planning Eval (implementation spec, 2026-06-04)

**The one question 6a answers:** *given a goal image, can the planner recover steering commands that
reach it* — and does that hold at the cheap sampler settings (DDIM=3) that make 6b's ~7 s/replan viable?
This is the gate before any robot/network work. **The planner validated here is the exact engine 6b wraps
behind an HTTP endpoint — 6a is not a throwaway test, it builds + bench-tests the engine.**

**Design decision — standalone eval script, NOT a registry "env".** `PlanningExperiment._run_mpc` /
`_sample_dset_goals` are built for *steppable sim envs* (pusht/point_maze/wall: `env.prepare/step/
eval_state`, a `states.pth`/`actions.pth` tensor layout). LeKiwi has **no simulator and no way to execute
an arbitrary CEM action offline**, so we do **not** fake a LeKiwi env (its `step()` would have to use the
WM as its own dynamics — circular/dishonest). Instead follow the Run 002 eval-tool pattern
(`src/sample/motion_rollout_viz.py`, `long_rollout_viz.py`): load ckpt + dataset directly, run CEM, grade
against the dataset as a built-in answer key.

**What 6a measures = open-loop planning accuracy (a necessary precondition), NOT closed-loop success.**
Offline there is no ground truth for "where the robot ends up if it executes CEM's actions" — that is 6b.
6a proves CEM can *invert the world model* to recover goal-reaching actions.

**The test (per sample):** from a held-out **val** episode, take a start frame z₀ and a **goal frame
`goal_H` chunks ahead** (a real reachable goal — we also know the *true* commands that produced it). Hide
the commands; give CEM only the two frames; let it plan (imagine 32 candidate command sequences → WM
rollout → keep the ones whose imagined future looks most like the goal → refit). Grade against the answer
key. Metrics, per (init,goal) pair × DDIM setting:

| metric | meaning |
|---|---|
| `do_nothing` = L2(z₀, z_goal) | floor CEM must beat (distance if robot never moves) |
| `gt_ceiling` = L2(WM.rollout(z₀, **GT actions**)₋₁, z_goal) | WM accuracy ceiling — CEM can't beat the WM's own prediction error; splits "planner failed" from "WM wrong" |
| `cem_reached` = L2(WM.rollout(z₀, **CEM actions**)₋₁, z_goal) | did CEM drive the WM prediction to the goal? want ≈ `gt_ceiling`, ≪ `do_nothing` |
| `action_recovery` = ‖denorm(CEM a) − GT (Δx,Δθ)‖ | **strongest offline signal** — did CEM re-derive ≈ the true commands (sign + magnitude on forward Δx, turn Δθ)? |
| decoded montage | z₀ / z_goal / CEM-planned rollout decoded through SD-VAE → eyeball whether the planned future resembles the goal |

**Headline deliverable — the DDIM sweep.** Run the whole battery at **DDIM ∈ {20, 5, 3}** (+ 50 for the
final montage). If DDIM=3 metrics ≈ DDIM=20 with no goal-reaching collapse, the ~7 s/replan cheap regime
(DDIM=3, 32 samples, 3 opt-steps) is green-lit for 6b. Watch turn-heavy goals — pivot-rotation control is
what softens first at DDIM=3 (see "Few-step sampling" above). If it collapses → fall back to DDIM=5 or flag
LCM distillation. Learned here for free, not live on the robot.

**Scene coverage — test across a variety of dataset scenes (required, not optional).** A single seed over
one episode is not a valid 6a result; aggregate-only numbers hide motion-specific failure. Sample
(init, goal) pairs to span the dataset and **stratify the report**:
- **Across episodes / space:** draw `init` frames from **many distinct val episodes** (cap ~1–2 pairs per
  episode), spanning the room's spatial coverage and starting headings — different landmarks in view
  (desk, tripod/easel, bowl/box, bare wall, near-field floor), not all from one corner.
- **Across motion type (stratify + report per-stratum, don't just average):** classify each goal by its GT
  integrated `(Δx, Δθ)` into **translation-dominant**, **rotation/pivot-dominant**, **arc (combined)**, and
  **slow / near-stationary**. Each stresses the WM differently and **pivot-dominant is the one that softens
  first at DDIM=3** — so it must be its own reported bucket, not buried in the mean. Aim for rough balance
  across buckets rather than whatever the random draw gives (the data is bang-bang, so translation goals
  dominate a naive sample).
- **Non-trivial goals only:** reject pairs where `do_nothing` is already tiny (goal ≈ start) — those inflate
  apparent success without testing planning. Require a minimum goal displacement.
- **Reproducible + localizable:** fix `--seed`, but log per-scene rows (episode id, offset, motion bucket,
  all metrics) so a failure points at a specific scene/motion, not just a worse average.
- **Scale `n_evals` for coverage:** ~30–40 pairs (≈8–10 per motion bucket) — still only minutes on the H100.

**Deliverables**
- **New (fork): `src/sample/offline_planning_eval.py`** — built on the `motion_rollout_viz.py` template
  (already loads ckpt+dataset+WM+decode); adds CEM + goal-sampling + the metric battery. Reuses
  **unchanged**: `CEMPlanner` (`action_dim=2`, `action_low/high` from f=10 stats), `DiffusionWorldModel`
  `.rollout`/`.encode_obs`, `create_objective_fn(mode="last", visual_metric="mse")`, `Preprocessor` with
  the integrate_se2 action stats (mean `[0.0221,−0.0006]`, std `[0.0141,0.0707]`), and the lekiwi val split
  for frames + GT integrated `(Δx,Δθ)`. CLI: `--ckpt --out --n_evals --goal_H --horizon --ddim <list>
  --num_samples --opt_steps --seed`. **Goal sampling stratifies by motion bucket + draws across episodes
  (see "Scene coverage")** — classify each candidate goal by GT `(Δx,Δθ)`, balance across
  translation/pivot/arc/slow, cap ~1–2 per episode, reject near-trivial goals. Outputs
  `offline_planning_eval.json` (per-scene rows tagged with episode/offset/**motion bucket**, plus aggregate
  **and per-bucket** summaries for each DDIM) + a montage PNG/MP4 per sample.
- **Optional `src/configs/planning/lekiwi.yaml`** — not needed for the standalone script (CLI suffices),
  but 6b will need it; add a minimal one for the record.

**Action-wiring traps (why we skip the experiment harness — integrate_se2 mode):**
- `planning_experiment.py:108-109` does `action_dim_total = action_dim * frame_interval` → with
  integrate_se2 that's `2×10 = 20`, a wrong 20-D CEM search. The standalone script sets `action_dim=2`.
- `_run_mpc` (~`:641`) does `act.reshape(frame_interval, -1)` to un-pack concat actions — meaningless for
  integrated deltas. Both need a conditional fix **only if 6b later routes through `PlanningExperiment`**;
  out of 6a's path.

**Run (GPU pod; ckpt + dataset already on `/workspace`; ~a few minutes of H100):**
```
python src/sample/offline_planning_eval.py \
  --ckpt /workspace/results/…/epoch=13-step=8000.ckpt \
  --out results/offline_planning_step8000 \
  --n_evals 36 --goal_H 3 --horizon 3 \
  --ddim 20 5 3 --num_samples 32 --opt_steps 3 --seed 42
```

**Acceptance criteria (gate to 6b) — must hold across scenes, reported per motion bucket:**
1. CEM **beats `do_nothing`** and `cem_reached` ≈ `gt_ceiling` (planner near-optimal *given* the WM) —
   **in every motion bucket**, not just on average (translation goals will pass easily; pivot/arc are the
   real test).
2. **Action recovery** has correct sign + comparable magnitude on the dominant forward/turn components,
   across translation-, pivot-, and arc-dominant goals.
3. **Decoded montages** show the planned future resembling the goal, sampled from *different* scenes/buckets.
4. **Cheap-sampler hold:** DDIM=3/5 ≈ DDIM=20 with no goal-reaching collapse **in any bucket** (watch
   pivot-dominant — it softens first) → confirms the 6b replan setting.

**Risks:** latent-L2 flattens **>~30 cm** → keep `goal_H` short (3 chunks ≈ 10 cm at f=10); long range is
6c (waypoints). Goals are sampled from real forward trajectories so they *are* reachable — a failure is the
planner/WM, not infeasibility.

### Milestones
- **6a — offline CEM eval (next; GPU). Spec'd — see "6a — Offline Planning Eval" above.** Standalone
  `src/sample/offline_planning_eval.py` (NOT a registry env — LeKiwi has no simulator): CEM recovers
  goal-reaching actions from held-out val frames, graded against the dataset answer key (`do_nothing` /
  `gt_ceiling` / `cem_reached` / `action_recovery`) + decoded montages, swept over DDIM ∈ {20,5,3} to
  confirm the cheap-sampler regime that makes 6b's ~7 s replan viable. Proves planner + WM + scoring before
  any hardware; the validated planner is the engine 6b wraps.
- **6b — closed-loop on LeKiwi.** Real-robot env, stop-and-plan MPC, goal-image tasks. Needs the robot.
- **6c — long-range.** Topological waypoint graph (Solution 1) once short-range MPC works.

### Develop vs run (cheap-box workflow)
- **Develop/plan anywhere (no GPU):** all of 6a's code (env, config, CEM wiring, action math) is just the
  **repo** — write + unit-test it on a cheap box; no checkpoint/dataset/GPU needed to author it.
- **Run 6a (GPU):** needs a CUDA GPU + step-8000 + dataset + venv. On *this* RunPod volume they're at
  `/workspace/results/…/epoch=13-step=8000.ckpt`, `/workspace/data/lekiwi`, `/workspace/nanowm-venv`.
  On *any other* machine, pull the **HF backups** (private, namespace `kaushikpraka`):
  `load_checkpoint("kaushikpraka/nanowm-lekiwi-b2-f10-step8000")` for the model (config + safetensors)
  and `kaushikpraka/wm-smallarea_nav30` for the dataset (the repo_id the loader already references).
  Spin the H100 (or any cheaper GPU — stop-and-plan is light) up on demand.
- **Run 6b:** the LeKiwi robot + a small edge GPU.

---

## MPC Outer Loop (Stop-and-Plan)

1. **Stop.** Robot is stationary.
2. **Observe.** Capture camera frame, encode through SD-VAE → z_current [4, 32, 32].
3. **Plan.** Run CEM (~1-2s) to find best action sequence of length H=3.
4. **Execute first action only.** Convert (Δx₁, Δθ₁) → velocity: v_x = Δx₁/(f·Δt), ω = Δθ₁/(f·Δt). Send to base for 167ms.
5. **Stop.** Return to step 2.

Actions 2 and 3 from the plan are discarded. Only one action executes before re-observing from ground truth. This bounds model error to a single step.

Stop-and-plan is acceptable for prototype — robot drives for 167ms, pauses ~1-2s for planning, drives again. Not smooth, but proves the concept.

## CEM Search (Per Replan)

Search space: H × 2 = 6 dimensions (three (Δx, Δθ) pairs).

**Initialization:** Gaussian over action sequences, mean=0, variance=dataset variance.

**Each of 5 iterations:**
1. Sample 64 candidates from current Gaussian
2. Roll out each through world model for H=3 steps (batched on GPU):
   z₁ = WM(z_current, a₁), z₂ = WM(z₁, a₂), z₃ = WM(z₂, a₃)
3. Score: -‖z₃ - z_goal‖² (L2 in SD-VAE latent space)
4. Take top-10 elites, refit Gaussian to their mean and variance

**Output:** final Gaussian mean = planned action sequence.

## Runtime Analysis

Per replan: 5 CEM iters × 3 rollout steps × 20 DDIM steps = 300 sequential forward passes, each at batch 64 through NanoWM-B/2 (160M params). At ~3ms per forward pass on L40S/A100: ~0.9-1.5 seconds.

**Latency reduction knobs (not needed for stop-and-plan prototype):**
- DDIM steps 20→5: 4× speedup (biggest lever)
- CEM iterations 5→2 with warm-starting: 2.5× speedup
- CEM samples 64→32: ~2× per forward pass
- Stacked: ~60ms, within real-time 167ms budget

## Scoring Function

L2 distance between final predicted latent and goal latent in SD-VAE space. Measures visual similarity.

Works at short range: "close to the bowl" vs "at the bowl" produces large latent differences. Fails at long range: 15cm of travel when the goal is 2m away produces negligible latent change. All candidates score approximately equally → CEM chooses randomly.

## Long-Range Problem and Solutions

With H=3, f=5, each rollout covers ~15cm. Goals beyond ~30cm produce flat scoring landscapes.

### Solution 1: Topological Waypoint Graph (recommended start)

Decompose route into nearby sub-goal images. Each MPC problem targets the next waypoint — always short-range.

**Graph construction from data:**
- Subsample frames from episodes at spatial intervals (~30cm of odometry)
- Connect nodes whose spatial proximity is below threshold
- Use DepthAnything3 reconstruction for metric edges (avoids visual aliasing)
- Path planning: A*/Dijkstra on metric graph

**Waypoint following:**
1. Localize current observation in graph
2. Find path to goal node
3. Target first waypoint, run CEM against it
4. When latent distance to waypoint drops below threshold, advance to next
5. Repeat until final goal is active target

**Known challenges:**
- Visual aliasing (carpet looks same everywhere) → mitigated by metric graph from DA3
- Directionality (same location, different heading = different latent) → need multi-heading waypoints or heading-invariant scoring
- Waypoint switching threshold tuning
- Graph coverage gaps in unvisited areas
- Timeout/recovery when MPC gets stuck

### Solution 2: Hierarchical World Model (HWM)

Use the world model at two levels: high-level CEM generates subgoals in latent space at coarser temporal resolution, low-level CEM plans actions to reach each subgoal. Eliminates external graph — subgoals are synthesized, not retrieved.

HWM paper reports Push-T success 17% → 61% with DINO-WM, and enables non-greedy behavior (temporarily moving away from goal). More complex to implement.

### Solution 3: Learned Distance Function

Replace raw latent L2 with a trained navigational distance predictor. Train on temporal distances in episodes (frames 10 steps apart → distance of 10). Stays informative at any range. Could eliminate waypoints entirely. Requires training a separate model.

### Solution 4: DepthAnything3 + Metric Scoring

Use DA3 reconstruction as collision map and/or for pose-based scoring. Score CEM candidates by metric distance to goal pose rather than latent visual similarity. Smooth scoring at any range. Risk: localizing hallucinated predicted frames in the reconstruction may be fragile.

## Rollout Horizon

H=3 with f=5 covers ~15cm/500ms. Sufficient for waypoint-based planning (each waypoint is ~30cm away).

Increasing H is possible via sliding-window autoregressive generation (paper demonstrates 50-frame rollouts). Costs:
- H×2 CEM dimensions (6D→12D→20D — CEM coverage degrades exponentially)
- H × 20 DDIM steps of sequential compute
- Prediction quality degrades over rollout (Finding #5)

Alternative: increase f instead of H. f=10 doubles reach per step (same 6D search space) at the cost of coarser temporal resolution and larger Δx per step.

## Visualization

All CEM rollouts happen in SD-VAE latent space. For debugging: decode predicted latents through the frozen VAE decoder → [3, 256, 256] RGB. Visualize the winning candidate's predicted trajectory side-by-side with actual camera frames after execution. Build this from the start.
