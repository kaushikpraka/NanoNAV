# Planning

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
