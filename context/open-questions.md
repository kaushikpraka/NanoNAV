# Open Questions

## Immediate (Pre-Training)

### Dataset size sufficiency
50 episodes at 30-60s = ~13K-18K transitions. The 43K target from the original doc may be needed. Train first checkpoint on initial collection, check diagnostic, scale from there. Upper bound for single room: ~4-6 hours (~130-190K transitions).

### Frame interval f tuning
f=5 (167ms chunks) matches PushT baseline. Navigation may benefit from larger f (more ground covered per model step, same CEM dimensionality). But larger f → larger Δx → coarser body-frame delta approximation. Test f=5 first, try f=8-10 if CEM reach is insufficient.

~~**Resolved for the action-conditioning purpose (2026-06-02):** raising f does not improve action
observability...~~ **REVERSED (2026-06-03).** That conclusion rode on the broken `corr(|Δx|, latentL2)`
metric. The controlled stationary-vs-translation contrast (`viz/stationary-vs-translation/`) shows
**raising f DOES improve translation observability**: translation's SNR over the noise floor goes
0.96× (f=5) → 1.34× (f=8) → 1.57× (f=10) → 1.93× (f=20), AUC 0.94→0.98. **f=10 is chosen for Run 002**
(best signal/floor without over-coarsening the chunk). f also still helps CEM reach. See the corrected
"weak action SNR" update below + `viz/stationary-vs-translation/README.md`.

### Trajectory validation tool
Need to build offline visualization: take raw velocity logs, integrate body-frame deltas, plot world-frame trajectory, verify against odometry. Confirms the integration math before training.

**Update (2026-06-01):** The collected dataset (`wm-smallarea_merged`) has **no logged global pose** — `observation.state` mirrors the action (arm joints + base velocity), not pose. So there is no *independent* odometry to validate the SE(2) integration against; the world-frame trajectory we'd plot is itself derived from the same velocities. Validation is therefore **visual-flow consistency**: SD-VAE `compare` of frame *k* vs *k+f*, checking that flow direction/magnitude matches the sign/scale of the integrated `(Δx, Δθ)`. This is also exactly what training cares about. See [[nanowm-integration]].

### Forward-speed coverage (bang-bang data) — found 2026-06-01
Integration validation showed `x.vel` is near bang-bang: per-chunk Δx is bimodal (≈0 or ≈1.65 cm at
full speed), with few intermediate values. The slow/low-Δx regime needed for fine near-goal approach
is sparsely covered. Options if near-goal CEM struggles: collect a few deliberately-slow episodes
(as the original plan intended but the data under-delivered), or down-weight reliance on fine speed
control near the goal. See [[experiment-log]].

### Reach per step shorter than assumed — found 2026-06-01
Max Δx ≈ 1.65 cm/chunk (x.vel ≤ 0.1 m/s), so H=3 covers ~5 cm, not the design's ~15 cm. The
flat-scoring threshold (~30 cm) is hit after very few chunks → reinforces both the f=8–10 experiment
(below) and the waypoint scaffold ([[planning]]). Consider whether f=8–10 is needed from the start.

## Training Phase

### Will action branch survive with real-world data?
PushT action conditioning works in sim with clean renders. Real-world camera noise, slight lighting variations, and visual complexity might make unconditional prediction harder — which could either help (model NEEDS the action to predict) or hurt (too much visual noise drowns the action signal). Table 5/6 diagnostic is the gate.

**PARTLY ANSWERED, THEN REINTERPRETED (see 2026-06-03 correction at the end of this item).** Table 5/6
FAILED on Run 001, but the cause is a **training** problem (overfit + low-SNR f=5), not a fundamental
"visual noise drowns the signal" problem — translation is in fact observable.

**ANSWERED — Table 5/6 FAILED on Run 001 (2026-06-02). The action branch did NOT survive on this
checkpoint — original (later-corrected) reading: visual noise drowns the action signal.** See [[training-runs]] Run 001 and
[[experiment-log]] (eval session). Diagnostic on the step-10K checkpoint: action-embedding **RMS
0.0088** (need ~0.1+; paper's SD-VAE 0.1119), GT final-latent L2 37.8 vs zero 42.0 / random 42.4 (GT
only ~10% better). Root cause, from `chunk_motion_viz.py` over 960 chunks:
- Forward motion is **bang-bang and tiny**: \|Δx\| bimodal at 0 and **~1.67 cm/chunk** (p50=p95=max),
  rotation ~1.5°/chunk. (Confirms "bang-bang data" + "reach shorter than assumed" above.)
- **corr(\|Δx\|, SD-VAE latent-L2) = 0.23** — motion barely predicts the latent change.
- **Stationary chunks (Δx=0) have latent L2 ~10–45, essentially the SAME range as full-speed chunks
  (~13–51)** — moving the robot changes the latent about as much as not moving does. The action-driven
  signal sits below the non-action latent noise floor (sensor/lighting/exposure/SD-VAE sensitivity).
- The world model's prediction error (latent L2 ≈31) ≈ the actual per-chunk change (≈30.6): it barely
  beats "predict no change."

**This is a data/representation SNR problem, not a training-length problem — more steps will not fix
it.** Highest-leverage fixes: (1) ~~**frame_interval 8–10+**~~ — see the update below; (2) raise capture
SNR (controlled lighting/exposure, more deliberate/longer translations); then the fallback options below.

**UPDATE (2026-06-02, frame-interval sweep) — it's specifically TRANSLATION that's unobservable, and
raising `f` does NOT fix it.** Previewed per-chunk SD-VAE latent change at f = 5/8/10/15/20 with no
retraining, split by action component (`chunk_motion_viz.py`; figures in `viz/signal-fsweep/`):
- **`corr(|Δx|, latentL2) ≈ 0` at every f** (−0.04 … +0.04). Forward motion is geometrically
  de-magnified by the elevated ~55° camera — a full-speed forward chunk barely moves the latent, and
  growing Δx 4× (f=5→20) leaves the correlation at ~0. **So `frame_interval` is refuted as the fix.**
- **`corr(|Δθ|, latentL2) ≈ 0.64–0.70` at every f** — rotation sweeps the whole FOV and *is* well
  observed. The action-branch failure is a **translation-observability** problem, not a generic one.
- Correction: the earlier `corr(|Δx|,·)≈0.23` was a noisy small subset; stable estimate is ~0.

⇒ The fix must restore *translation* observability: a **lower / more forward-facing camera** (or
richer near-field floor texture) for parallax per cm, and/or **auxiliary pose/odometry conditioning**
for Δx (fallback #1 below), plus lowering the non-action floor (exposure/WB lock, avoid lossy AV1).

**⚠️ CORRECTION (2026-06-03) — translation IS observable; the above "translation-observability" framing
is WRONG.** The `corr(|Δx|, latentL2)≈0` statistic was a bad estimator (bang-bang Δx → no within-moving
variance; pooled pure-rotation chunks → high latentL2 at ~0 Δx, dragging corr to ~0). The **controlled
stationary-vs-translation contrast** (`stationary_vs_translation.py`, `viz/stationary-vs-translation/`)
holds rotation near zero and finds **pure-translation chunks change the SD-VAE latent ~2× more than
stationary** (AUC 0.94 @ f=5 → 0.98 @ f=10/20), with a monotonic dose-response (signal scales with Δx,
floor flat) and a near-field-floor parallax footprint. So the Run 001 failure is **training-side**:
the diagnosed checkpoint was **overfit** (step-10K = epoch 16; val bottomed ~epoch 3, no best-val ckpt)
**at f=5** where translation's signal only ≈ the noise floor (~1:1; it's 1.6:1 at f=10). **The fix is
Run 002** (retrain at f=10 + best-val checkpointing), NOT a camera change — fallbacks #1–4 below are now
contingencies if Run 002 still fails. See [[experiment-log]] (2026-06-03), [[training-runs]] (Run 002 plan).

**✅ RUN 002 OUTCOME (2026-06-03) — action branch is alive; legacy RMS gate is mis-calibrated.** Run 002
trained to 12K at f=10. On the val-best step-4125 checkpoint the gate gives GT 36.1 / zero 40.7 / random
45.2, RMS 0.0089: the **gt < zero < random separation is clean and wide** (random distinctly worse than
zero ⇒ the model uses action *content*; Run 001 had zero≈random), and decoded motion rollouts visibly
track real translation/rotation/arc. But **RMS 0.0089 ≈ Run 001's 0.0088** across two very different
checkpoints ⇒ the action-embed RMS is **architecturally pinned** (additive injection `x = x + action_emb`)
— a **mis-calibrated gate**, not the live signal. ⇒ The Table-5/6 pass criterion is being **re-based on
rollout health** (action separation + motion-tracking fidelity), measured across a cross-checkpoint
rollout eval (4125/6K/8K/10K/12K). Fallback #2 (cross-attention injection) / #3 (larger
action-embed) are still relevant if the *separation* proves too weak for the planner. See
[[experiment-log]] (Run 002), [[training-runs]] (Run 002).

**✅ RESOLVED (2026-06-04) — the action branch is strong enough for planning; no fallback needed.** The
cross-checkpoint rollout eval finished: rollout quality is **U-shaped, peaks ~6K–8K then overfits ⇒
step-8000**. Then **Stage 6a (offline CEM planning eval) confirmed the separation is sufficient for the
planner directly**: on step-8000, CEM inverts the world model to **near-WM-optimal** action sequences
(`cem_reached/gt_ceiling` ~1.0–1.1) in *every* motion bucket and recovers the true commands (sign 100%,
dxErr ~1 cm, dθErr ~2.5°). So the worry "if the separation proves too weak for the planner" is **answered
empirically — it isn't**; fallbacks #1–4 are now dead unless 6b (closed-loop) regresses. The
mis-calibrated RMS gate is moot. See [[planning]] "6a — RESULTS", [[experiment-log]] (2026-06-04).

### If Table 5/6 fails — fallback options
1. Add absolute global pose as auxiliary conditioning (environment-specific but maximally informative)
2. Try different action injection mechanism (cross-attention instead of additive — most expressive, most params)
3. Increase action embedding dimension
4. Data augmentation to force action sensitivity

## Planning Phase

### 🚨 Live-frame distribution gap — THE #1 GATING BLOCKER (2026-06-09, on-robot)
**The WM hallucinates when rolled out from live camera frames.** With the bot in front of a goal object,
the imagined rollout shows a *completely different side of the room* (montages:
`context/figures/live-distribution-gap_*.png`). The live `camera` pixels are correct, but `z0 = encode(live)`
lands **off-distribution**, so the sequential rollout degrades (`+1` haze) then **regresses to a familiar
training scene elsewhere**. This is **upstream of the scoring objective**: if `z0` + imagined futures are
hallucinated, both `dist_to_goal` and what CEM optimizes are garbage → the confidently-wrong commands follow.
**So fix this BEFORE the learned-distance objective below.** Evidence it's distribution, not a code bug: the
interactive driver seeds from **val tensors** (in-distribution) and looks clean; MPC runs **live frames through
`_preprocess`** and hallucinates; and `nearfan2` converged only because its start was a *well-covered* pose.
**REFINED (same session): it's OOD data coverage, NOT a preprocess bug.** Seeding the interactive driver from
the nearhamper **goal image** (a clean settled capture, loaded via the SAME path as the `nearfan2`/`nearchair`
goals that converged) **also hallucinates** → rules out `_preprocess`/format parity and motion-blur; the
nearhamper view is a **genuinely under-covered region**. **Fix = recollect more data + train further** (the WM
only "mapped" densely-visited poses). `_preprocess` byte-parity is a quick sanity check but DOWNGRADED. **Eval
guidance until retrained: use goals in well-covered regions** (`nearfan2`-style); avoid OOD goals like
`nearhamper`. See [[experiment-log]] 2026-06-09 (later). *(The learned-distance objective under "Scoring
function alternatives" is #2 — it can't help until `z0`/rollouts are faithful, which needs the data/training
fix here.)*

### Waypoint graph construction details
- Spatial sampling interval (~30cm proposed — tune based on CEM scoring range)
- DepthAnything3 reconstruction quality on overhead camera frames
- Graph connectivity: metric threshold vs k-nearest-neighbors
- Localization: how to place current observation in the graph at runtime

### Waypoint switching
- Threshold on latent distance for advancing to next waypoint
- Timeout mechanism if MPC gets stuck
- Handling approach from unexpected direction (directionality problem)

### Scoring function alternatives
- Raw SD-VAE L2 (simplest, appearance-based)
- DINO feature distance for scoring only (semantic, heading-invariant) while predicting in SD-VAE (decoupled generation/scoring latents)
- Learned navigational distance predictor
- Pose-based scoring via DA3 localization in reconstruction

**⭐ NOW THE #1 PRIORITY (empirically motivated, 2026-06-09).** On-robot closed-loop 6b confirmed the
**objective is the bottleneck, not the WM or the CEM search.** Raw SD-VAE latent-L2 (`‖z0−zg‖`, equal weight
on every latent cell) is well-conditioned **near** the goal (sweeps: basin floor ~16, SNR ~17–97) but **flat
far out** — most cells encode generic floor/wall, so distant poses look ~equidistant → no gradient on the
"plateau." It also **under-credits real progress**: the planner rotated the bot to face the chair yet `dist`
stayed ~48 (it didn't reward the alignment, so the bot overshot). `--horizon 5`, `--var-scale 2`,
`--vx-max 0.12` all fail to help because they don't change *what's being optimized*. Offline CEM already hits
the WM's accuracy ceiling (Stage 6a), and the WM has generally mapped the room — so the fix is the **distance
metric**, not more dynamics training or search.
**Build:** a self-supervised **temporal-distance / quasimetric** head on the *latents* (no decode, so the
known decode-blur is irrelevant): sample frame pairs from the existing dataset, label by temporal gap
(k frames apart → dist ≈ k; or contrastive/quasimetric), train a small head to predict reachability-distance.
Then swap `dist_to_goal = ‖z0−zg‖` → `d_learned(z0,zg)` in the readout **and** CEM's objective. Gives gradient
on the plateau + credits pose-progress, and unlocks **model-imagined subgoals** (imagine reachable futures →
score by learned distance → hop → repeat = "plan fully in the WM, no manual waypoints"). Trainable on data we
already have. See [[experiment-log]] 2026-06-09, [[roadmap]] 6b.3.

**Design settled (2026-06-09) → [[learned-distance-metric]].** Chosen approach = **quasimetric RL (QRL)**:
learns "≈ chunks-to-drive" (drivable shortest-path distance, not appearance) from temporal adjacency
(neighbours=1) + push-apart + an IQE triangle-inequality head — gradient on the plateau by construction,
and asymmetric (directionality). Wiring: **cost on the *generated* (imagined) latents** `d(ẑ,zg)` (CEM
ranks actions by imagined outcome — the only action-dependent quantity), **termination/readout on the
*current* state** `d(z0,zg)`; lean on the metric so the WM rollout can stay short. Validate offline on the
`measure_dist_sweep` ground-truth-displacement curves first. Subgoals = a **topological graph over real
dataset frames** (real-frame nodes dodge the WM hallucination that imagined subgoals would trigger);
WM-imagined subgoals + an optional offline VLM-teacher are deferred. Tracked as [[roadmap]] **6d**.

## Future Extensions

### Pattern B (goal-conditioned video generation + IDM)
Same data, same encoder. Replace action conditioning with goal-image conditioning. Train separate IDM on action-labeled data. Compare Pattern A vs Pattern B on same scene. Analytic optical-flow IDM baseline (AVDC-style, zero labels).

### Hierarchical World Model (HWM)
High-level CEM generates latent subgoals, low-level CEM plans actions. Eliminates external graph. Push-T success 17% → 61% with DINO-WM. More complex but more elegant.

### Mobile manipulation
Extend action space to include arm joints. Use both overhead and wrist cameras. The world model predicts visual consequences of both base motion and arm motion jointly.

### Multi-room / environment transfer
Current setup is single-room, single-lighting. Generalization requires diverse environments. Consider sim-to-real transfer via Isaac Sim LeKiwi nav environments.

### Latent actions (LAPA/CLAM)
Continuous latent action model reduces action-label requirement. Enables training on action-free video. Natural scaling path.

### Real-time planning
DDIM steps 20→5, CEM warm-starting, reduced samples. Target: replan within 167ms chunk duration for smooth continuous navigation.
