# NanoWM × LeKiwi Navigation — Overview

## Project Goal

Use NanoWM (a diffusion-forcing world model) for goal-conditioned navigation on a LeKiwi mobile manipulator. The world model predicts future camera frames given current observations and actions. At inference, CEM planning searches for actions whose predicted futures match a goal image. The model learns general environment dynamics; tasks are specified at inference via goal images.

## Current Phase

**Phase 2: Data Collection** — Preparing to record ~50 teleop episodes in a controlled room subset with PS5 controller.

## Key Decisions (Settled)

- **Architecture:** Pattern A — action-conditioned forward model + CEM/MPC planning (not Pattern B video-as-plan + IDM)
- **Backbone:** NanoWM-B/2 (160M params)
- **Latent space:** SD-VAE [4, 32, 32] — action-sensitive, decodable for visualization. DINO/V-JEPA fail at action conditioning (Finding #4)
- **Action representation:** Body-frame pose delta (Δx, Δθ), 2D. Integrated from f=5 velocity commands per chunk. Δy dropped (negligible at chunk timescales)
- **Camera:** Elevated third-person (~55° tilt) mounted on robot arm structure. Captures robot body (fixed reference), near floor, mid objects, far walls
- **Control:** Unicycle model, 2-DOF (v_x, ω). No strafe. Exact kinematics for pose tracking
- **Data paradigm:** General environment exploration, not task demonstrations. Random/diverse driving. Task enters only at inference via goal image
- **Objective:** v-prediction, cosine + ZTSNR schedule
- **Planning:** Stop-and-plan MPC. ~1-2s per replan with full 20 DDIM steps. Not real-time, acceptable for prototype

## File Index

- [[action-representation]] — Body-frame delta construction, normalization, camera geometry
- [[data-collection]] — Episode structure, controller setup, logging, dataset sizing
- [[training]] — Model config, hyperparameters, diagnostic protocol
- [[planning]] — CEM/MPC pipeline, scoring, waypoint scaffold, long-range solutions
- [[experiment-log]] — Chronological record of decisions and results
- [[open-questions]] — Unresolved items and future directions
