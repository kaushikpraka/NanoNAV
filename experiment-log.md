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
- This Obsidian vault (overview, action-representation, data-collection, training, planning, experiment-log, open-questions)

### Next Steps

1. Set up room environment (lighting, object positions, arm parking config)
2. Verify lerobot-record logging pipeline (camera + velocity at 30 Hz, no v_y)
3. Collect ~50 teleop episodes with PS5 controller
4. Build dataset: SD-VAE encoding + body-frame delta integration
5. Validate integration with trajectory visualization tool
6. Train first NanoWM-B/2 checkpoint
7. Run Table 5/6 diagnostic
