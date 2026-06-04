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

### 6a — RESULTS (ran 2026-06-04; **PASS — all four gates met, 6b green-lit at DDIM=3**)

`src/sample/offline_planning_eval.py` built + run on step-8000 (H100, ~22 min). **35 scenes** stratified
across **all 5 val episodes** — translation=9, pivot=8 (only 190 pivot slices exist in val; logged
shortfall 8/9), arc=9, slow=9 — each planned at **DDIM ∈ {20, 5, 3}** (32 samples × 3 opt-steps × top-10,
the cheap CEM config). Action stats confirm the integrate_se2 wiring (`mean [0.0221, −0.0006]`,
`std [0.0141, 0.0707]`). Full per-scene rows + montages in `results/offline_planning_step8000/`.

Latent-L2 (lower `cem_reached` better; `reached_ratio = cem_reached/gt_ceiling → 1` = CEM is WM-optimal):

| DDIM | bucket | do_nothing | gt_ceiling | cem_reached | ratio | beats-floor | recov(norm) | dxErr cm | dθErr ° |
|---|---|---|---|---|---|---|---|---|---|
| **20** | translation | 43.7 | 30.8 | 30.1 | 0.99 | 100% | 0.66 | 0.8 | 1.1 |
| | pivot | 55.6 | 38.3 | 41.9 | 1.10 | 100% | 1.69 | 1.9 | 3.4 |
| | arc | 57.2 | 35.9 | 39.5 | 1.11 | 100% | 1.20 | 1.2 | 3.0 |
| | slow | 48.0 | 35.3 | 35.2 | 1.00 | 100% | 1.04 | 1.1 | 2.2 |
| | **overall** | **51.0** | **35.0** | **36.5** | **1.05** | **100%** | 1.13 | 1.2 | 2.4 |
| **3** | translation | 43.7 | 29.2 | 29.4 | 1.02 | 100% | 0.62 | 0.6 | 1.4 |
| | pivot | 55.6 | 36.0 | 37.2 | 1.05 | 100% | 1.50 | 1.7 | 3.0 |
| | arc | 57.2 | 37.7 | 37.1 | 1.00 | 100% | 1.12 | 1.0 | 3.1 |
| | slow | 48.0 | 35.5 | 34.2 | 1.00 | 100% | 1.12 | 1.2 | 2.3 |
| | **overall** | **51.0** | **34.5** | **34.4** | **1.01** | **100%** | 1.08 | 1.1 | 2.5 |

(DDIM=5 omitted for space — sits between, overall ratio 1.04. Full numbers in `run.log`.)

**Gate-by-gate verdict:**
1. **Beats `do_nothing` + near WM-optimal — PASS in every bucket.** `cem_reached < do_nothing` 100% of
   scenes (e.g. translation 30 vs 44, pivot 42 vs 56). `reached_ratio` 0.99–1.11 across all buckets/DDIM
   — CEM recovers essentially the best actions the WM allows; the residual gap to the goal is **WM
   prediction error, not planner failure** (pivot/arc carry the largest gap, as predicted, but still ≤1.11).
2. **Action recovery — PASS.** Forward/turn **sign match 100%** (one DDIM=5 translation scene mis-signed a
   tiny ~4° turn → 89% in that one cell; sign is nulled when the GT component is near-zero so pivot Dx isn't
   scored as noise). Magnitude errors small: **dxErr ~0.6–2.0 cm, dθErr ~1.1–3.4°** — CEM re-derives the
   true commands, not just any goal-reaching ones.
3. **Decoded montages — PASS.** 8 montages (2 per bucket) show the WM rollout under the CEM plan landing on
   the goal frame, including arc (drive+turn) and pivot (pure rotation). `results/offline_planning_step8000/montages/`.
4. **Cheap-sampler hold — PASS, decisively.** DDIM=3 does **not** degrade goal-reaching in any bucket — in
   fact `cem_reached` is *slightly lower* at DDIM=3 everywhere (overall 36.5→34.4, **pivot 41.9→37.2**). The
   pivot-softening risk flagged from the controllability eval **did not materialize in closed planning
   accuracy** — `gt_ceiling` also tightens at fewer steps (eta=0 deterministic DDIM; the near-deterministic
   futures are captured fine in 3 steps), so `reached_ratio` stays ~1.0. **⇒ the ~7 s/replan DDIM=3 / 32×3
   regime is confirmed for 6b** (keep DDIM=5 as the fallback only if a turn-heavy on-robot task regresses).

**Caveats (honest):** val holds only **5 episodes**, so spatial/landmark coverage is the dataset ceiling,
not a sampling choice (motion-bucket balance is good *within* those 5). All numbers are **open-loop**
planning accuracy on reachable dataset goals — closed-loop success (compounding execution error, real-robot
dynamics) is 6b. `goal_H=3` (~10 cm) stays inside the reliable rollout window by design; longer range is 6c.

**Bottom line:** the planner + world model + latent-L2 scoring form a working goal-reaching engine on
held-out data, and it holds at the cheap sampler settings — **Stage 6a passes; 6b (closed-loop on LeKiwi)
is green-lit** with step-8000, DDIM=3, 32 samples, 3 opt-steps, H=3, replan-every-chunk.

### 6b — Closed-Loop MPC on LeKiwi (implementation spec, 2026-06-04)

**The one question 6b answers:** *does the 6a-validated planner actually drive the real robot to a goal
view* — stop-and-plan MPC on hardware, closing the loop 6a left open (open-loop accuracy → closed-loop
arrival). The engine is fixed; 6b is the robot/network/safety wrapper around it.

**Architecture decision — RunPod is the lerobot *client*; the Pi keeps running the host (teleop already
works).** Original framing was "Pi → custom RunPod inference API → Pi". Because lerobot LeKiwi teleop is
already working, the cleaner shape reuses that tested transport:

```
┌────────── Raspberry Pi (on LeKiwi) ──────────┐        ┌────────── RunPod H100 (one process) ──────────┐
│  LeKiwiHost (lerobot, ALREADY RUNNING)        │        │  lekiwi_mpc.py:                                │
│   • ZMQ PUB  → observation.images.top + state │◄──────►│   • LeKiwiClient (lerobot)                    │
│   • ZMQ PULL ← base velocity + arm hold       │ Tail-  │   • DiffusionWorldModel + CEMPlanner (6a core)│
│   • host watchdog: no cmd → motors stop       │ scale  │   • stop-and-plan MPC loop + rerun telemetry  │
└───────────────────────────────────────────────┘ (WG)  └───────────────────────────────────────────────┘
```

**No bespoke HTTP inference API, no separate inference server.** The LeKiwiClient runs *on RunPod*, so
`get_observation()`/`send_action()` ARE the obs/command transport (lerobot ZMQ over Tailscale) and CEM
inference is a local function call in the same process — strictly less code than a custom API, and the
engine that runs is exactly the one 6a validated. Tailscale (WireGuard mesh) RTT home↔datacenter ~20–60 ms
is negligible vs the ~7 s plan, and the robot is stopped during planning anyway. **Connectivity = Tailscale**
(both Pi and RunPod on the same tailnet; RunPod dials the Pi's tailnet IP — survives IP changes, no router
config, encrypted; fallbacks SSH reverse-tunnel / port-forward if ever needed).

**The control loop (RunPod-side, per cycle ≈ 8–9 s):**
```
load engine ONCE: load_checkpoint → DiffusionWorldModel, CEMPlanner factory, action stats  (reuse 6a core)
load goal image → pad-to-256 + normalize → obs_goal ; zg = encode_obs(obs_goal)
connect LeKiwiClient over Tailscale ; capture arm pose once → hold it constant every step
repeat until reached or step == max_steps:
  1. STOP        send_action(zero base vel); settle ~0.5 s        (guarantee stationary)
  2. OBSERVE     obs = get_observation(); frame = obs['observation.images.top'] → obs_0; z0 = encode_obs
  3. TERMINATE?  latent_L2(z0, zg) < reach_thresh → success, break
  4. PLAN        mu, info = planner.plan(obs_0, obs_goal)          # DDIM=3, 32×3 elites, ~7 s
  5. EXECUTE 1st chunk only:
        (Δx, Δθ) = denorm(mu[0,0])                                # m, rad
        v_x = Δx / (f·Δt) ; ω = Δθ / (f·Δt)   with f·Δt = 10/30 = 0.333 s
        clamp → v_x∈[0, 0.1] m/s, ω∈[−0.32, 0.34] rad/s           (dataset range = safety envelope)
        send_action({x.vel:v_x, y.vel:0, theta.vel:ω, arm:hold}); sleep(0.333 s); send_action(zero)
  6. LOG (rerun) live frame · goal · decoded WINNING rollout · decoded TOP-K ELITE rollouts (CEM's
                 selected candidates) · latent-dist-to-goal · (v_x,ω) · CEM loss
```
Params carried straight from 6a: **step-8000, DDIM=3, 32 samples, 3 opt-steps, top-10, H=3,
replan-every-chunk, chunk = 0.333 s.** Executing only the first chunk bounds model error to one step.

**CEM is unchanged from 6a — elites + iteration retained.** Per replan: 3 opt-steps × (sample 32 → score by
latent-L2 → keep top-10 → refit μ,σ). The cheap regime came from **DDIM 20→3** (the dominant cost lever,
inside each rollout), NOT from gutting CEM (64→32 samples / 5→3 opt-steps is the only outer-loop trim). If a
turn-heavy on-robot task looks under-optimized, add quality back via opt-steps 3→5 or samples 32→64 (~linear
cost, still <~15 s) *before* touching DDIM.

**Sub-step decomposition (each gates the next — front-loads the cheap hardware-grounding before any CEM
touches the robot):**

| # | step | proves | new code |
|---|---|---|---|
| **6b.0** | **Transport + units bring-up** (no planning) | lerobot client ↔ Pi host (**Mac/local-LAN to dev; RunPod/Tailscale on the pod**); `get_observation()` returns a decodable `top` frame; `send_action()` drives the base; RTT measured; **sign/units of `x.vel` & `theta.vel` confirmed empirically** | tiny smoke script |
| **6b.1** | **Open-loop replay** (velocity conversion, no CEM) | replay a recorded val episode's GT `(Δx,Δθ)` chunks → velocities → execute; eyeball it traces the recorded path (translation + a turn) | `replay_chunks.py` |
| **6b.2** | **Engine module** | factor 6a's load + `DiffusionWorldModel` + `CEMPlanner` + action-stat helpers into one importable module so the live loop runs the *exact* validated path (no eval-vs-robot drift) | refactor of `offline_planning_eval.py` core |
| **6b.3** | **Closed-loop MPC** | the stop-and-plan loop above; terminate on latent threshold or max_steps; per-step logging | `lekiwi_mpc.py` |
| **6b.4** | **Goal capture + run harness** | `capture_goal.py` (drive-and-snapshot the `top` frame to a goal file; pre-staged photos use the same file interface) + run wrapper | `capture_goal.py` |
| **6b.5** | **Telemetry (rerun) + success criteria** | per-step montage (live frame · goal · winning rollout · **the selected top-K elite rollouts**) + latent-distance-descent curve; success = robot visibly arrives within max_steps on ≥3 short tasks | rerun viz + small `CEMPlanner` elite-surfacing patch |

**Develop locally (free), run inference on the pod.** The GPU/cost boundary falls between authoring + the
no-model robot checks (local, free) and live CEM (pod) — so the pod only spins up for inference:
- **Local (Mac, no GPU, no pod, no Tailscale):** all *authoring* — 6b.2 engine module, 6b.3 loop, 6b.4
  capture, 6b.5 viz, the `CEMPlanner` elite patch — **plus the robot-grounded checks that need no model**,
  6b.0 transport+units and 6b.1 open-loop replay, run with the **Mac as the lerobot client on the local LAN**
  (the existing teleop path; no Tailscale needed). These are the highest-risk sim-to-real checks (the
  `theta.vel` deg/s↔rad/s trap, the velocity math) and they cost nothing. Exercise the full
  control/viz/termination path end-to-end on CPU with a **stub planner** (canned action sequences) + mock
  latents, so everything but live CEM is validated locally.
- **Pod (H100, GPU-only — resume here):** the live closed-loop run. Swap stub→real `DiffusionWorldModel`,
  and swap Mac-client/local-LAN → **RunPod-client/Tailscale** (the Tailscale setup is part of resuming on the
  pod). Run 6b.3 at DDIM=3 (~7 s/replan) + the elite-rollout decode viz with real latents.
- **Make the swap a config change, not a rewrite:** inject the **planner** (real CEM vs stub) and the
  **robot client** (same `LeKiwiClient`, different endpoint — LAN IP vs tailnet IP) as dependencies. Then
  "resume on the pod" = flip the device string, the endpoint, and the planner impl; the loop, velocity math,
  clamps, termination, and rerun logging are identical and already validated locally. (Model + dataset pull
  from the HF backups on the pod — see 6a "Develop vs run".)

**Goal spec:** real in-distribution `top` frames — **drive-and-snapshot** (teleop to the spot, capture the
`top` frame, drive back to a start pose, run) and/or **pre-staged photos** from the same camera/mount/
exposure. Both flow through one goal-image-file interface; `capture_goal.py` is the drive-and-snapshot helper.

**Live telemetry — rerun over Tailscale.** Rerun's SDK (logging) and Viewer (rendering) are separate
processes that talk over the network — built for exactly this. Add the **Mac to the same tailnet** (already
up for the robot) and the cloud↔Mac link is LAN: run the **Viewer as a server on the Mac**, RunPod's
`lekiwi_mpc.py` does `rr.init(...)` then connects to the Mac's tailnet IP and streams (the Mac-as-server
direction survives RunPod restarts; both ends are directly addressable on the tailnet so direction is a
convenience, not a NAT constraint). Bandwidth is trivial (one frame + a few imagined frames + scalars per
~8 s). Exact connect call is rerun-version-dependent (`rr.connect`/`connect_tcp`/`connect_grpc`) — pin to the
installed version at implementation. **Zero-network fallback:** log to a `.rrd` on RunPod, `rsync` down, open
in the Mac viewer (same data, not live).

**Selected (elite) rollouts in the viz — requires a small `CEMPlanner` patch.** The visualization must show
not just the final winning plan but **the top-K trajectories CEM actually selected** (the elites it refits to
each iteration) — decoded through the VAE and logged to rerun as a candidate fan. That's what reveals whether
CEM is converging on a sensible, peaked candidate set vs scattering. `CEMPlanner.plan()` currently returns
only `mu` + `info={losses, final_loss, num_iterations}` and **discards** the per-iteration `topk_actions`
(`cem_planner.py:194-195`). Patch — backward-compatible, gated by a `return_elites` flag so 6a behavior stays
byte-identical: on the **final** opt-step, retain the elites and expose `info["elite_actions"]` [topk, H, 2]
(+ `elite_losses`). **Efficiency:** those elite rollouts are *already computed* in the final scoring pass —
retain their latent rollouts (`info["elite_latents"]` [topk, 1+H, D]) instead of re-rolling, so the viz pays
only the VAE *decode* of the selected elites, not a second WM rollout. The MPC loop decodes the winning +
top-K elite latents and logs them as a fan; cap the rendered count behind a flag (e.g. top-3..10) since extra
decodes cost wall-time — which matters on the slow Mac-MPS path.

**Traps that bite on hardware (not present in 6a):**
1. **`theta.vel` units — deg/s vs rad/s (highest-risk silent bug).** The build script converted raw LeKiwi
   `theta.vel` (deg/s) → rad/s for training, so the **model's ω is rad/s**; if lerobot `send_action` expects
   deg/s, convert back (`ω·180/π`) — a 57× scale error on every turn otherwise. 6b.0 confirms with a known
   command; 6b.1 confirms sign+scale on a real path before any CEM.
2. **Constant-velocity-within-chunk approximation.** Model trained on the *integrated* delta of per-step
   velocities that varied within a chunk; we execute one constant velocity for 0.333 s. Mild for bang-bang
   data but real — watch for systematic under/over-shoot in 6b.1.
3. **Arm joints in the action dict.** LeKiwi's action includes 6 arm joints; navigate the base only — capture
   the arm pose once and re-send it constant, or the arm sags/drifts.
4. **Camera/mount/exposure match.** Goal frame and live frames must be the same `top` mount/exposure as
   training — drive-and-snapshot guarantees it; pre-staged photos are the risk case (re-shoot if the mount
   moved).
5. **Reach-threshold calibration.** From 6a, real-frame `do_nothing` ~44–57, `gt_ceiling` ~30–38 (latent-L2).
   Start `reach_thresh` ≈ **35** and tune.
6. **Heading drift.** Small ω sign/scale bias compounds across steps; replan-every-chunk from a fresh
   observation bounds it, but a *systematic* bias won't self-correct — flag if 6b.1 shows consistent curl.

**Safety (build in from 6b.0):** per-command **velocity clamp** to the dataset envelope (guards OOD CEM
proposals); the Pi **host watchdog** is a free fail-stop (no commands → motors stop, so a Tailscale drop or
RunPod crash halts the robot — verify by killing the network mid-run); a **global speed scale** (~0.5×) on
first closed-loop runs; **Ctrl-C e-stop** (zero + exit) plus a physical kill within reach; **`max_steps`** cap
(~30).

**Acceptance criteria (gate 6b → 6c):**
1. **6b.0:** `top` frame decodes on RunPod; base moves in the commanded direction; RTT < 1 s; `x.vel`/
   `theta.vel` sign+units documented.
2. **6b.1:** open-loop replay visibly traces a recorded path including a turn (gross tolerance).
3. **6b.3:** on **≥3 short drive-and-snapshot tasks** (one translation-dominant, one turn-heavy, one arc),
   latent-distance-to-goal descends and the robot **visibly arrives at the goal view** within max_steps; the
   actual-vs-imagined montage shows the imagined rollout tracking reality.
4. **Safety verified live:** clamp + watchdog (network-kill → stop) both confirmed.

**Out of scope (stays future):** long-range goals >~30 cm where latent-L2 flattens → **6c** waypoint graph;
on-board/edge-GPU autonomy (cutting RunPod out) → Stage 8; smooth non-stop-and-plan control → later (DDIM=3's
~7 s plan makes continuous control infeasible without step-distillation).

### 6b — RESULTS (6b.0)

**6b.0 transport + units bring-up — ✅ DONE (2026-06-04, PASS).** `scripts/lekiwi_transport_check.py` (Mac as
lerobot client, local LAN, no GPU) vs the Pi host at **10.0.0.125**. The `(Δx,Δθ)→velocity` contract is now
empirically pinned (full detail in [[experiment-log]] "Stage 6b.0"):

| fact | value | controller rule |
|---|---|---|
| transport / import | `LeKiwiClient`, `lerobot.robots.lekiwi`, RTT ~14–16 ms | — |
| action contract | 6 arm `.pos` + `x.vel`/`y.vel`/`theta.vel`; cam `top` 480×640×3 uint8 | hold arm `.pos`, `y.vel`=0 |
| `x.vel` | **m/s, +x = forward** | `x.vel = Δx / 0.333` |
| `theta.vel` | **deg/s, +theta = LEFT/CCW** (matches dataset +ω=CCW) | `theta.vel = (Δθ / 0.333)·57.296`, **no sign flip** |
| chunk | `f·Δt = 10/30 = 0.333 s` | — |
| **deadband (watch-out)** | `0.3` deg/s = no motion; `12–15` deg/s engages | min-\|theta\| floor; tiny Δθ may be a no-op |

⇒ the `theta.vel` deg/s↔rad/s trap (the #1 6b risk) is **resolved**: convert ω→deg/s, no negation. 6b.1
(open-loop replay) can now convert recorded `(Δx,Δθ)` chunks to velocity with confidence.

### Milestones
- **6a — offline CEM eval — ✅ DONE (2026-06-04, PASS).** `src/sample/offline_planning_eval.py` +
  `configs/planning/lekiwi.yaml` (6b scaffold). 35 stratified val scenes × DDIM {20,5,3}: CEM beats
  `do_nothing` 100%, `reached_ratio` ~1.0–1.1 (WM-optimal) in every bucket, action sign 100% / dxErr
  ~1 cm / dθErr ~2.5°, montages land on goal, and **DDIM=3 holds** (no pivot collapse — see "6a — RESULTS").
  ⇒ engine validated, 6b green-lit.
- **6b — closed-loop on LeKiwi — SPEC'D (2026-06-04), ready to implement; needs the robot.** RunPod runs
  the lerobot `LeKiwiClient` (Pi keeps the working host) over **Tailscale**; stop-and-plan MPC wraps the
  6a-validated engine; goals are real `top` frames (drive-and-snapshot / pre-staged); **rerun-over-Tailscale**
  live telemetry to the Mac viewer. Sub-steps 6b.0 transport+units (✅ DONE — see "6b — RESULTS (6b.0)") → 6b.1 open-loop replay → 6b.2 shared
  engine module → 6b.3 closed-loop loop → 6b.4 goal capture → 6b.5 telemetry/success. Full spec + traps
  (theta.vel deg/s↔rad/s, constant-vel-within-chunk, arm hold, reach-threshold ~35) + safety + acceptance
  criteria above in "6b — Closed-Loop MPC on LeKiwi".
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
