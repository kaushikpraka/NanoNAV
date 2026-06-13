# Semantic WM Retrain (Option C) — Plan & Status

**Status: ON-ROBOT VALIDATED (2026-06-11) — 3/3 physical arrivals, incl. ×2 on nearchair1 (the goal flat-L2 failed): monotone 0.32→0.19 token-cos descent, full-clamp driving, healthy near/far behavior. (Gate C passed 2026-06-10: kill-switch RMS 0.333/margin 43.4; CEM ratio 1.01; weld ρ 0.876; nearhamper fixed.) OPEN in C1: reach-thresh floor ≈0.2 on cross-session goal images — 0.08 never fired; fresh-goal floor test decides recapture-goals vs raise-thresh-to-0.2–0.25; cost-mode-first arm; optional SD-VAE baseline legs. OPS: Pi host `connection_time_s` is NOT a CLI flag — sed the dataclass default in the HOST venv's config_lekiwi.py to 86400 and VERIFY it prints back (evening 2026-06-12 relaunches died at connect; sed likely hit the wrong install). **C3 ON-ROBOT SUCCESS (2026-06-12): graph run REACHED nearpurifier — dist 0.08 < thresh after 129 steps, 40-hop route, [tracked] localization throughout, ENDGAME at step 116 (`mpc_semantic_graph_nearpurifier4.rrd`). Got there via 3 fixes: calibrated 2τ waypoint spacing + raw-progress budget (e3f1e49); localization hysteresis/route stickiness — track src along committed path (b63c64c); ≥3-hop waypoint floor. A/B basin bracket: purifier (0.35 start) arrives WITHOUT graph, hamper (0.45) plateau-wanders ⇒ basin edge 0.35–0.45, matches offline calibration (90% descent at 4 chunks). neardesk graph run: ENDGAME fine but hovered ~0.30 for 35 steps (goal-image-dependent endgame floor — C1 question). PENDING: sparser 5-hop floor + 2.5τ (a773ef7) + subgoal-strip viewer (addce3e) not yet robot-tested; demo set for write-up (hamper A/B, purifier #2, neardesk).** Then C2 recollection (+multi-cam/reverse capture). 2026-06-13: write-up scaffolded under `docs/` (GitHub Pages), keeper recordings published as Release `recordings-v1` ([[RECORDINGS]]); nearhamper graph runs made route progress but no clean REACH (host blocker recurred). See [[experiment-log]] 2026-06-12 / 2026-06-13.** The decision that
supersedes the SD-VAE planning stack's objective problem at the root: **retrain the world model to
predict frozen DINOv2 patch tokens** (flow-matching / x0 objective) instead of SD-VAE latents under
diffusion-forcing, so the WM's native rollout space *is* the Gate-A-validated distance space.
Tracked as roadmap **6e**; supersedes the Option-B "distill DINO into φ(SD-VAE)" variant of
[[learned-distance-metric]] rung 0 (the temporal/chunk-units metric + rung-2 graph remain in scope,
retargeted to token space).

## Why (decision rationale, 2026-06-10)

Gate A ([[experiment-log]] 2026-06-10) measured: **sdvae_l2** (current cost) has *sub-noise* far-band
gradient (1.25σ/0.80σ vs the 3σ gate) while **frozen DINOv2-patch distance PASSES everywhere**
(ρ 0.943, 12–21σ far-band, sharp yaw basin) — and the **weld check fired** (WM-imagined ẑ +23σ off
the clean curve; d *rises* within rollouts). CEM cost = `distance(rollout(z0,a), zg)`: with SD-VAE
both factors are broken (flat metric × off-manifold rollouts). Option C fixes both with one move:

- The cost CEM ranks on becomes **token-space DINO distance on predicted tokens — already validated,
  zero metric-training** (our `dinov2_cos` arm = DINO-WM / RAE-NWM's published planning cost).
- **Data demand is NOT significantly higher**: frozen encoder ⇒ the WM learns only dynamics (no
  appearance modeling); same 256-token sequence as today (B/1 over 16×16 tokens vs B/2 over 32×32
  SD-VAE); action signal *more* recoverable in semantic latents under regression (arXiv:2605.06388:
  0.83 vs 0.51); small-data precedents are all semantic-side (PLDM, DINO-WM 1–2K trajs, RAE-NWM 1K).
- **OOD failure mode flips from malignant to benign**: regression/FM predictors degrade to blurry
  averaging (honest mediocre cost) instead of the vivid wrong-room hallucination (nearhamper, blocker
  #1). The frozen encoder has seen millions of rooms — live frames can't be OOD *for the encoder*.
- Option B (distill DINO distances into φ reading SD-VAE latents) stacks two approximations (φ ≈
  DINO∘decode, plus 23σ-off rollouts) to imitate what C gets exactly. Rejected as the destination;
  its scaffold (pair sampler, harness) transfers.

**The one open scientific question = Finding #4**: does the action branch stay alive over semantic
tokens at our scale? Published failures used diffusion-forcing + additive injection; published
successes use flow-matching/x0 + AdaLN/cross-attn (RAE-NWM, V-JEPA-2.1 nav). C0 answers it on our
data for ~one overnight of H100.

## Key implementation facts (recon 2026-06-10)

- nanowm already has: `latent_codec=webdino` (any HF `Dinov2Model`), `pred_name: flow`
  (`flow_matching.py`) and `x` (x0), pluggable action injection
  (`additive|adaln_fuse|adaln|film|cross_attention`), decoder-less validation (latent-only metrics
  path), and the kill-switch tool `action_diagnostic.py` (PASS = action-emb RMS > 0.05 AND GT-action
  latent-L2 < zero/random-action baselines; FAIL = RMS ≈ 0.002).
- **Encoder parity**: `facebook/dinov2-small` == the Gate-A-validated torch-hub `dinov2_vits14`
  weights; both consume 224px → 16×16 patches × 384d. (Gate A graded exactly this space.)
- **`latent_codec.latent_scale` added** (nanowm 69fe01b): DINO post-LN tokens measure elementwise
  std **2.4** on lekiwi → scale 2.4 ⇒ ~unit-std diffusion/FM targets (the SD-VAE `scaling_factor`
  analog). Cosine cost is scale-invariant; recorded in every run config.
- Model: `NanoWM-B/1` (patch 1 over the 16×16 token grid → 256 model tokens, per-DINO-token
  correspondence, RAE-NWM's choice). `model.latent_size=16 model.latent_channels=384`.
- Planning stack is latent-shape-agnostic (rollout, CEM objective, engine encode); only decode/viz
  calls need `has_decoder` gates.

## Phases

| Phase | What | Data | Gate |
|---|---|---|---|
| **C0** | 4 probes (~3k steps each, sequential overnight): C0a flow+adaln_fuse (RAE-NWM-shaped bet) · C0b flow+cross_attention · C0c x0+adaln_fuse · C0d flow+additive (control — expected Finding-#4 repro) | existing 50 eps | action-RMS > 0.05 + GT-rollout beats zero/random (per-run `action_diagnostic`) |
| **C0.5** ∥ | token→RGB decoder, [384,16,16]→256², MSE+perceptual on dataset frames | existing frames | eval-only, NEVER in the cost path (viz for rerun viewer / rollout strips) |
| **C0-ext** | winner → 12k steps | existing 50 eps | Gate C ladder: 6a-style offline CEM action-recovery (token cost) ≥ SD-VAE 6a; weld re-test ≪ +23σ with d *falling* within rollouts; nearhamper roll = benign blur not a different room |
| **C1** | ✅ planner swap merged; **on-robot 2026-06-11: 3/3 arrivals (semantic arm)** — formal SD-VAE baseline legs + cost-mode-first arm + fresh-goal floor test still open | — | reach-thresh: floor ≈0.2 on cross-session goals; 0.08 never fired (operator estimate 0.2–0.25 pending fresh-goal test) |
| **C2** | recollection (co-design per [[learned-distance-metric]]; sized **150–250 eps** given 20× gap to RAE-NWM's regime) + full retrain | new data | full Gate A harness re-run + hallucination test |
| **C3** | ✅ **BUILT + OFFLINE-VALIDATED 2026-06-11** — token cache (4,500 × [384,16,16] + JPEGs), calibrated τ=0.182 (k=3 chunks), 4,450 temporal + 10,061 shortcut edges. **DIRECTED** (operator catches ×2: backward thread traversal, then backward-pointing welds): temporal = driving direction only; welds **direction-certified by motion parallax** (ident 1,760 / fwd 6,636 / soft 9,400 with +0.15 Dijkstra penalty) → largest SCC 94.5%, can-reach-core 97.4%. Wormhole audit clean; 4/4 offline routes re-pass on the certified graph (chair→hamper 50 forward waypoints); runtime = `lekiwi_mpc --graph` (raw-route-progress lookahead — offset- and penalty-immune; sticky on-path localization, margin 0.03/patience 2; ≥5-hop waypoint floor; endgame falls back to the real goal image; reach-thresh gated to endgame). Learned φ head still only if drivability gaps appear. **✅ ON-ROBOT SUCCESS 2026-06-12: REACHED nearpurifier (129 steps, 40 hops, dist 0.08); neardesk ENDGAME-but-hovered (~0.30); 5-hop sparse config not yet robot-tested.** | cache | cross-room goals via waypoint chains |

**Sharpenings from the design discussion:** action atrophy doesn't break rollout, it breaks
*planning* (action-agnostic futures ⇒ all CEM candidates tie — silent failure, hence the
instrumented kill-switch). RMS-alive is necessary-not-sufficient (hence the C0-ext ladder). The
prototype stays coverage-bound — C2's recollection is still what produces the deployable model; C0
de-risks the *recipe*. Far goals still need the graph: DINO cosine is appearance, not drivability.

## Launch artifacts

- `scripts/run_c0_probes.sh` — the matrix driver (env, overrides, per-run `action_diagnostic`,
  verdicts → `results/c0_probe_summary.md`, ntfy on completion).
- Probe runs land at `results/<ts>-C0{a,b,c,d}-dinoB1-*-F4S10-lekiwi/`; diagnostics at
  `results/c0_diag_C0*/`.
- wandb: project `nanonav`, run names = model.name (C0a-dinoB1-flow-adalnfuse, …).

## Deferred improvements

- **Viz decoder quality (logged 2026-06-10, operator: not critical).** Current C0.5 decoder is
  L1+MSE-only -> mean-seeking blur (the SD-VAE comparison is unfair: its decoder is reconstruction-
  trained with LPIPS+PatchGAN at web scale, and DINO tokens discard appearance by design — ceiling is
  "scene/layout right, texture approximate"). Upgrade ladder when viz matters: (1) +LPIPS perceptual
  loss (~5 lines, biggest win), (2) 2-4x decoder capacity + 30-50k steps, (3) light PatchGAN,
  (4) check RAE-NWM's frozen RAE decoder for dinov2-small at C2, (5) diffusion decoder (sampling
  kills regression blur; days). Decoder is NEVER in the cost path — purely operator UX.
- **Rollout truncation for cost-mode first (logged 2026-06-10, operator: keep full horizon for now).**
  With `first`, frames +2/+3 don't affect the score, and sequential-causal generation makes a
  `--gen-frames 1` truncation exactly equivalent for +1 ⇒ ~2-3x planning-latency win (~7s → ~2-3s
  per replan). Implementation when wanted: truncate candidate scoring, roll the full horizon once
  for the winning elite (viz strip survives); verify +1 bit-parity on fixed seed + re-run the
  offline CEM probe. Kept un-truncated for now for first-vs-last A/B flexibility.

## Future avenues (operator, 2026-06-11 on-robot session)

- **Multi-camera training** — train the WM on multiple views, not just the overhead `top` camera.
  Implications: per-view DINO token sets (concat as extra model tokens or a view-axis), goal
  specification becomes per-view or view-agnostic, and the cost must aggregate views. Natural
  decision point = C2 retrain (collect all views during the recollection session regardless —
  cameras are free at capture time even if v1 trains on `top` only).
- **Reverse driving in the training data** — the dataset (and the velocity clamp: `VX_MIN=0`)
  is forward-only today, so the planner literally cannot back out of an overshoot; it must turn
  around. Needs: teleop segments driving in reverse during recollection, clamp widened to
  negative vx, and the CEM sampling range opened up. Cheap to capture, high value for
  close-quarters goal approach. Also pairs with the asymmetry question in
  [[learned-distance-metric]] (forward vs U-turn cost).
- **Visual-servoing endgame for fine final placement (idea logged 2026-06-10).** Once the planner
  gets near the goal (`d(z0,zg)` under a handoff threshold ≈ the basin floor), switch to a
  **keypoint-based visual-servoing mode**: match keypoints between the live frame and the goal
  image (classic ORB/LightGlue, or DINOv2 patch-feature correspondences — the tokens are already
  computed for the cost), then drive the image error to zero with a classic IBVS control law
  emitting **(x.vel, y.vel, theta.vel)** directly. Three properties make this attractive:
  (1) it runs **below the learned stack** — no WM rollout, no CEM, just image error → velocity —
  so it can use the **strafe DOF (`y.vel`)** that was deliberately dropped from the WM action
  space, and small reverse corrections, without any retraining; (2) it **decouples final precision
  from the token-cosine floor** — the open reach-thresh problem (~0.2 floor on cross-session
  goals) stops mattering for *placement* because termination becomes "pixel error small", which
  also absorbs the observed within-floor position slack (the "slightly left" arrivals); (3) it
  completes a clean three-tier architecture: **graph (room scale) → CEM+WM (basin, ~10–20 cm) →
  servo (the last few cm/degrees)** — each layer handing off to one tuned for a finer regime.
  Risks/needs: keypoint-poor views (carpet-facing goals), cross-session lighting robustness
  (matchers are typically more robust than raw cosine), a rough depth/homography assumption for
  the interaction matrix, and gain tuning on the real base. Lit anchor: ViT-VS (arXiv:2503.04545,
  ViT-feature visual servoing — already in the [[learned-distance-metric]] refs). Natural slot:
  after C1's reach-thresh resolution; independent of C2/C3 (pure runtime addition).

## Risks

Finding-#4 repro on all arms (escalation: film injection → aux action-decode loss → V-JEPA 2.1
codec, also wired); bicubic-vs-bilinear 224px resize between codec and Gate-A grader (cosmetic;
unify in C1); lost decodability (retrieval-viz from the latent cache day-one; C0.5 decoder after);
B/1 sequence cost at 384d tokens (smoke-tested OK on the H100, batch 16×accum 4).

See [[experiment-log]] 2026-06-10 (Gate A + the Option-C decision entry), [[roadmap]] 6e,
[[learned-distance-metric]] (metric/graph design — still the substrate for C3).
