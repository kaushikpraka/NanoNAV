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

### Next Steps

1. ~~Set up room environment (lighting, object positions, arm parking config)~~ ✅
2. ~~Verify lerobot-record logging pipeline (camera + velocity at 30 Hz, no v_y)~~ ✅
3. ~~Collect teleop episodes with PS5 controller~~ ✅ — merged to `kaushikpraka/wm-smallarea_merged`
4. Build dataset: SD-VAE encoding + body-frame delta integration ← **current**
5. Validate integration with trajectory visualization tool
6. Train first NanoWM-B/2 checkpoint
7. Run Table 5/6 diagnostic
