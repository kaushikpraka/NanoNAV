# Semantic WM Retrain (Option C) — Plan & Status

**Status: GATE C PASSED (2026-06-10) — the 12k x0+adaln_fuse model is planner-ready: kill-switch RMS 0.333 / margin 43.4; offline CEM at the WM ceiling (ratio 1.01, 100% sign agreement); weld ρ 0.876 (SD-VAE 0.29); nearhamper hallucination FIXED (same-scene rolls). C0/C0.5/C1-offline all done. **NEXT = C1 on-robot A/B (operator)** — exact command in [[experiment-log]] 2026-06-10 GATE C. Then C2 recollection + C3 graph.** The decision that
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
| **C1** | planner swap: token-cosine cost (+1-weighted option) + termination on real-frame tokens; reach-thresh recalibration from Gate A curves; viewer strips via decoder | — | on-robot A/B vs SD-VAE stack on well-covered goals **(needs operator)** |
| **C2** | recollection (co-design per [[learned-distance-metric]]; sized **150–250 eps** given 20× gap to RAE-NWM's regime) + full retrain | new data | full Gate A harness re-run + hallucination test |
| **C3** | subgoal graph in token space: nodes = real-frame tokens from the cache; temporal edges free; shortcut edges at **calibrated frozen token distance** (cosine-at-k-chunks stats from the cache) — learned φ head only if drivability gaps (wall wormholes) appear in forks/distance-field | cache + new | cross-room goals via waypoint chains |

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

## Risks

Finding-#4 repro on all arms (escalation: film injection → aux action-decode loss → V-JEPA 2.1
codec, also wired); bicubic-vs-bilinear 224px resize between codec and Gate-A grader (cosmetic;
unify in C1); lost decodability (retrieval-viz from the latent cache day-one; C0.5 decoder after);
B/1 sequence cost at 384d tokens (smoke-tested OK on the H100, batch 16×accum 4).

See [[experiment-log]] 2026-06-10 (Gate A + the Option-C decision entry), [[roadmap]] 6e,
[[learned-distance-metric]] (metric/graph design — still the substrate for C3).
