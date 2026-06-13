# NanoWM × LeKiwi Navigation — Overview

## Project Goal

Use NanoWM (a diffusion-forcing world model) for goal-conditioned navigation on a LeKiwi mobile manipulator. The world model predicts future camera frames given current observations and actions. At inference, CEM planning searches for actions whose predicted futures match a goal image. The model learns general environment dynamics; tasks are specified at inference via goal images.

## Current Phase

**Phase 6: Stage 6a (offline CEM planning eval) PASSED — planner engine validated, 6b (closed-loop on
LeKiwi) green-lit.** The CEM/MPC planner + NanoWM + latent-L2 scoring were run end-to-end on **step-8000**
over **35 held-out val scenes stratified by motion** (translation/pivot/arc/slow), swept over DDIM {20,5,3}.
**All four acceptance gates pass:** CEM beats the no-move floor 100% of scenes and lands **near-WM-optimal**
(`cem_reached/gt_ceiling` ~1.0–1.1) in *every* motion bucket; it recovers the true commands (turn/forward
sign 100%, dxErr ~1 cm, dθErr ~2.5°); decoded montages land on the goal; and the cheap **DDIM=3** regime
holds with **no pivot collapse** (~7 s/replan confirmed for 6b). The residual goal gap is WM prediction
error, not planner failure. Open-loop accuracy only — closed-loop success is 6b (needs the robot). See
[[planning]] "6a — RESULTS", [[roadmap]], `results/offline_planning_step8000/`.

**How we got here (Run 002 → step-8000).** NanoWM-B/2 Run 002 trained the full **12,000 steps at f=10** on
a RunPod H100 (after fixing three crashes — wandb-key / FID-metric / CUDACallback). Architecture note: **the
SD-VAE perception is frozen pretrained; the 160M transformer is trained from scratch** (`pretrained: null`).
For diffusion-forcing val_loss is a weak rollout proxy (it bottomed at step-4125 then rose), so checkpoints
were judged by *rollouts*: the action branch is **alive and action-sensitive** (clean **gt < zero < random**
separation + visible translation/rotation/arc tracking; the legacy action-embed RMS gate reads FAIL but is
**mis-calibrated / architecturally pinned** for the 2-D additive embedder). The **cross-checkpoint rollout
eval** found rollout quality is **U-shaped — peaks at ~6K–8K then overfits** ⇒ **step-8000 is the planner
checkpoint**. See [[training-runs]], [[experiment-log]], [[open-questions]].

## Project Tracking

Progress, experiments, and decisions are tracked in **git** (this repo), not Obsidian. The `context/` notes are the living design record; the chronological record lives in [[experiment-log]].

## Key Decisions (Settled)

- **Architecture:** Pattern A — action-conditioned forward model + CEM/MPC planning (not Pattern B video-as-plan + IDM)
- **Backbone:** NanoWM-B/2 (160M params)
- **Latent space:** SD-VAE [4, 32, 32] — action-sensitive, decodable for visualization. DINO/V-JEPA fail at action conditioning (Finding #4 — **reinterpreted 2026-06-09:** the failure is diffusion-forcing ⊗ semantic latents, not the latents; semantic latents + a *regression/x0* predictor is a credible retrain option, see [[open-questions]] "Semantic-latent WM retrain" + [[learned-distance-metric]])
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
- [[tailscale-setup]] — Pod↔LeKiwi bridge for closed-loop 6b.3 (SSH reverse tunnel [recommended], Tailscale paths, TUN blocker)
- [[runpod-operator-guide]] — Runbook for the pod-side agent that babysits training
- [[training-runs]] — Per-run training telemetry log
- [[RECORDINGS]] — On-robot Rerun (.rrd) recordings index + GitHub Release link (not in git)
- `docs/` — long-form write-up (GitHub Pages); build figures with `scripts/build_site_assets.py`
