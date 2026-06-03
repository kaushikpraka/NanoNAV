# NanoWM × LeKiwi Navigation — Overview

## Project Goal

Use NanoWM (a diffusion-forcing world model) for goal-conditioned navigation on a LeKiwi mobile manipulator. The world model predicts future camera frames given current observations and actions. At inference, CEM planning searches for actions whose predicted futures match a goal image. The model learns general environment dynamics; tasks are specified at inference via goal images.

## Current Phase

**Phase 5: Run 002 (f=10) trained to completion; re-gating the action branch via rollouts.** NanoWM-B/2
**Run 002** trained the full **12,000 steps at f=10** on a RunPod H100 (after fixing three crashes —
wandb-key / FID-metric / CUDACallback — all pushed). Architecture note: **the SD-VAE perception is frozen
pretrained; the 160M transformer is trained from scratch** (`pretrained: null`). val_loss bottomed 0.2047
at step 4125 then rose (denoising-loss overfit) — but for diffusion-forcing val_loss is a weak
rollout-quality proxy, so we trained the full session and judge by *rollouts*.

The action branch is now **alive and action-sensitive**: at the val-best (step-4125) checkpoint the gate
shows a clean **gt < zero < random** separation (36.1 / 40.7 / 45.2) and the model visibly tracks real
translation / rotation / arc motion — materially better than Run 001 (where zero≈random). The legacy
**action-embed RMS gate still reads FAIL** (0.0089 ≈ Run 001's 0.0088 across two very different
checkpoints), now believed **mis-calibrated / architecturally pinned** for the 2-D additive embedder
rather than a live training signal — the separation + motion-tracking are the meaningful signals.

Run 002 also confirmed (via `viz/stationary-vs-translation/`) that the earlier "translation is
geometrically unobservable" claim was wrong: translation IS observable (AUC 0.94 @ f=5 → 0.98 @ f=10);
the camera was never the problem. **In progress: a cross-checkpoint rollout eval (4125 / 6K / 8K / 10K /
12K)** to measure whether more training improves rollout quality and to pick the checkpoint for the
CEM/MPC planner. See [[training-runs]], [[experiment-log]], [[open-questions]].

## Project Tracking

Progress, experiments, and decisions are tracked in **git** (this repo), not Obsidian. The `context/` notes are the living design record; the chronological record lives in [[experiment-log]].

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
- [[experiment-log]] — Chronological record of design decisions
- [[open-questions]] — Unresolved items and future directions
- [[roadmap]] — Staged execution plan with current status
- [[nanowm-integration]] — How the dataset plugs into NanoWM (concat vs integrate, v3.0→v2.1, the patch)
- [[runpod-setup]] — Bring-up runbook for the pod-side agent (install, clone, build, launch)
- [[runpod-operator-guide]] — Runbook for the pod-side agent that babysits training
- [[training-runs]] — Per-run training telemetry log
