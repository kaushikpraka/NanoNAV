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

**The real levers reduce compute:** DDIM 20→5 (**~4×**, biggest) × 64→32 samples (**~2×**, linear when
compute-bound) × 5→3 opt-steps (**~1.7×**) ≈ 13× (× compile 1.13× ≈ **15×**) → **~10 s/replan**. Below
that needs **step-distillation** (DDIM→1–4) or a smaller model; real-time (167 ms) is out of reach for the
prototype. **6a (offline) is unaffected** — latency irrelevant; just optionally DDIM=10 to speed the
sweep. **6b (closed-loop)** targets ~10 s/replan stop-and-plan. (The "Runtime Analysis" section below is
the original optimistic estimate — superseded.)
- **Scoring:** latent-L2 (`objective.py`), valid **<~30 cm**; beyond that the landscape flattens → use the
  **waypoint scaffold** (Solution 1 below) for longer routes.

### Milestones
- **6a — offline CEM eval (next; GPU).** Build the dataset-replay env + `lekiwi.yaml`; run
  `experiment=planning … ckpt_path=<step-8000>`; report goal-reaching latent-L2 + decoded planned-vs-GT
  rollouts. Proves planner + WM + scoring before any hardware. **Crucially, validate planning quality at
  the cheap sampler settings (DDIM≈5, ~32 samples) that make 6b's ~10 s replan viable** — 6a must confirm
  CEM still reaches goals at them (if naive few-step sampling degrades too much, the fix is distillation —
  see "few-step sampling" note below).
- **6b — closed-loop on LeKiwi.** Real-robot env, stop-and-plan MPC, goal-image tasks. Needs the robot.
- **6c — long-range.** Topological waypoint graph (Solution 1) once short-range MPC works.

### Develop vs run (cheap-box workflow)
- **Develop/plan anywhere (no GPU):** all of 6a's code (env, config, CEM wiring, action math) is just the
  **repo** — write + unit-test it on a cheap box; no checkpoint/dataset/GPU needed to author it.
- **Run 6a (GPU):** needs a CUDA GPU + the step-8000 ckpt + dataset (`/workspace/data/lekiwi`) + venv
  (`/workspace/nanowm-venv`) — **all on this RunPod volume**. Spin the H100 (or any cheaper GPU —
  stop-and-plan is light) up on demand; the data stays here, nothing to move.
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
