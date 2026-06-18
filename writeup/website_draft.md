<!--
  WEBSITE DRAFT — working source for the NanoNAV write-up.
  This markdown is the editing surface; docs/index.html is the deploy target.
  Section numbering matches the TOC in docs/index.html so porting is mechanical.

  Conventions:
    [TODO: ...]      → something you need to supply or decide
    [FIGURE: file]   → image/video goes here; caption follows in *italics*
    assets/...       → path is relative to docs/ (where the deployed page lives)

  Asset status legend:
    ✅ already in docs/assets/      ⏳ needs to be generated/pulled from the pod      🆕 needs to be created
-->

# NanoNAV: Real-Robot Navigation with [Nano World Models](https://arxiv.org/abs/2605.23993)

**Subtitle:** Latent-space planning drives a small robot to goal images

**Byline:** Kaushik Prakash · June 2026 · code: coming soon

---

## Hero video

[FIGURE: ✅ assets/plan-demo.mp4 — PLACEHOLDER headline demo, autoplay/loop/muted in the hero slot]
*[TODO: replace with the filmed demo set when ready — strongest single asset is the nearhamper A/B (baseline wanders, graph reaches). Original source kept at docs/assets/plan-demo-source.mov.]*

---

## TL;DR

I taught a LeKiwi mobile manipulator to drive to a **photograph**: show it an image taken somewhere in the room, and it finds its way there. The whole stack is learned from **50 tele-operated episodes (~25 minutes, ~45K frames)** through a single overhead camera: a diffusion world model imagines candidate futures, a sampling-based planner picks the actions whose imagined outcome looks most like the goal, and a 4,500-node graph of moments from the training data carries the robot to goals beyond the model's local horizon. No pre-built map, no depth sensor, no external localization.

This post is the full build log, including all of the failures that led to success. These failures tell the full story: a world model that ignored its own actions, two confident wrong diagnoses I had to retract, a latent space that hallucinated, and a route planner that tried to drive the robot backwards. Key finding: **the search was never broken — the objective was blind**, and most of the work was proving that with a tape measure and then fixing it.

---

## Background — why I built this

I've been fascinated by world model research lately — a field making competing bets on what the right output is, what the right training signal is, and where the real leverage lies. Dr. Fei-Fei Li describes the current landscape best in [**A Functional Taxonomy of World Models**](https://www.worldlabs.ai/blog/taxonomy-of-world-models). She splits them by *what they output*: a **Renderer** outputs pixels meant for human eyes, where visual fidelity is what matters; a **Simulator** outputs state — a geometrically and physically faithful representation that programs can compute on; and a **Planner** outputs *actions* — given observations and a goal, it decides what the agent should do next, closing the perception–action loop. She argues these eventually converge into unified models, with simulation as the linchpin.

This project lives squarely in the **Planner** corner of that taxonomy. It is not trying to render a beautiful world or to be a faithful physics engine. The decoded frames are, frankly, blurry. It is trying to use a small, imperfect imagination as the inner loop of a controller: propose an action, imagine its consequence, score that consequence against a goal image, act. The whole story that follows is what happens when you take the Planner ambition seriously on cheap hardware and a small dataset.

I was inspired by [**Nano World Models**](https://arxiv.org/abs/2605.23993) (Huang et al., 2026), a minimalist, diffusion-forcing world-model codebase released recently. The authors remark that "the broader research community still lacks compact, reproducible, and easily extensible implementations" of modern world models and set out to change that. More than the code, that framing is what grabbed me: a call to *democratize* world-model research, to show the ideas don't require a frontier lab's compute or data to be worth building on. That resonated, and it set the constraint that defines everything here: do this small.

The Nano World Models project evaluates across three domains: simple control environments, game simulation, and **real-robot data (RT-1)**. If the recipe held on real robot data at that scale, it might hold on *mine*. And I had the robot: a **LeKiwi** left over from earlier imitation-learning work. So the plan was simple: collect my own data, train a nano-scale world model on it, and see if I could plan with it on the real machine.
---

## 1 · The Problem Statement

The classical way to make a robot go somewhere is a stack: build a map (SLAM), localize yourself in it, plan a path, follow the path. It works, and it is heavy — it wants depth sensors, careful calibration, and a metric model of the world maintained over time.

The bet here is that a world model trained on raw experience can replace that entire stack. The task is deliberately stark: the robot gets its **current camera frame** and a **target image**, and it outputs **body-frame velocities**. No pre-built map, no external localization or depth sensor, no GPS, no reward function, no task demonstrations. The goal is specified at *inference* time, by an image, and the model has never been told that goal exists.

[TODO: optionally a one-line spec box here — camera, 2-D action (forward velocity + yaw rate), goal-as-photo.]

---

## 2 · Robot Hardware

The **LeKiwi** is an open-source mobile manipulator from the LeRobot ecosystem: a low-cost SO-ARM-style arm bolted onto a **three-omniwheel "kiwi drive" base**, driven by an onboard **Raspberry Pi** and a handful of inexpensive serial-bus servos. (The Pi does no heavy lifting — it just streams camera frames and accepts velocity commands; the world model itself runs on a rented cloud GPU bridged to the robot over a tunnel, which we'll get to when we go closed-loop.) For navigation I use only the base; the arm stays parked in a fixed pose throughout. The stock LeKiwi looks out from a low front-facing webcam; I swapped in wider-angle USB cameras. I also added a **third overhead spatial camera on a custom mount that looks down over the robot from above at roughly a 55° tilt**. That overhead vantage captures four depth zones at once: the robot's own body, the near floor, mid-room objects, and the far walls.

*Hardware at a glance: LeKiwi (LeRobot) · holonomic 3-omniwheel base · SO-ARM arm, parked · Raspberry Pi host · low-cost serial-bus servos · overhead USB camera on a custom ~55° mount. [TODO: confirm exact Pi model / servo model / camera model and how the mount was fabricated.]*

[FIGURE: 🆕 assets/lekiwi-mount.jpg — photo of the LeKiwi with the custom overhead camera mount]
*The rig. The LeKiwi base with the arm parked and the custom mount holding the overhead camera that everything downstream depends on. [TODO: drop the photo at docs/assets/lekiwi-mount.jpg.]*

## 3 · Data

Collection was entirely manual. The recording side is **LeRobot's `record` pipeline**, which timestamps and synchronizes the overhead camera with the commanded base velocity at 30 Hz. For teleoperation, I configured a PS5 DualSense controller plugged in over USB and mapped through LeRobot's teleop interface. The left stick was for forward velocity and right stick for yaw rate, with no strafe binding so sideways velocity is zero by construction. The entire data-collection setup is a game controller and a laptop. The dataset is on HuggingFace at [kaushikpraka/wm-smallarea_merged](https://huggingface.co/datasets/kaushikpraka/wm-smallarea_merged), viewable in the [LeRobot visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset).

The dataset is **50 teleoperated episodes, 44,926 frames at 30 Hz** of deliberate *exploratory driving, not goal demonstrations*. The model's job is to learn the latent-space transitions the scene undergoes given *any* action, because at inference the planner will propose dozens of candidate actions per decision, including bad ones, and the model has to predict what all of them would do in order to rank them. Train only on clean, goal-directed trajectories and the model never learns what a *bad* action looks like, so it can't tell the planner which candidates to reject. So I drove to cover the space, not to accomplish anything. My driving trajectories included: arcs and curves, pure forward runs, pure rotations both directions, the occasional stationary pause as a clean identity anchor where the action is exactly `(0, 0)`. The environment was a subsection of my room, roughly 2 m × 5 m carpeted area, blinds closed and room lights on for stable illumination, furniture left in fixed positions as landmarks. I intentionally kept conditions consistent across episodes, while varying the positions and headings for each new run. Driving was kept to entirely forward motion without any reverse commands. This becomes important later on when constructing the navigation graph.

[FIGURE: ✅ assets/world_trajectories.png]
*A sample of the driving. Dead-reckoned paths from a handful of the 50 tele-operated episodes — a glimpse of the exploratory driving that makes up the ~25-minute dataset, all in one section of a room.*

### The action space: dead reckoning

The robot is controlled with two floats: forward velocity and yaw rate. But I don't hand those to the model directly. Instead I **dead-reckon**: integrate each short window of about five control steps (~167 ms) into a single body-frame pose change, a displacement **(Δx, Δθ)** chunk. Some advantages of this representation:

- It's a **body-frame** displacement, so "drive forward 5 cm" is the same vector `(0.05, 0)` no matter which way the robot is facing, providing the model with heading invariance instead of having to learn it.
- It's **low-dimensional**, keeping planner's search small.
- It's an **integrated displacement, not a raw velocity**.

During steady cruising the velocity is constant for many frames while the image keeps changing, so a model trained on velocities learns the action is uninformative and quietly stops listening to it. A displacement is nonzero exactly when the robot moves and scales with how far, so it stays coupled to what the camera sees.

Dead reckoning has one assumption baked in: **no significant slip**, that a commanded centimeter is a real centimeter across the floor. For a light, slow robot on flat carpet that holds well enough. Since the planner re-observes a fresh frame every chunk, any small error is corrected rather than accumulated. The rest I checked empirically: the dropped Δy never exceeds ~0.58 mm, and an open-loop replay on the real robot traced the dead-reckoned path to ~0 cm even through a 117° arc.

[FIGURE: ✅ assets/chunk_deltas.png]
*The action distribution per chunk, dead-reckoned from the raw logs. Forward motion is nearly bang-bang — stopped or full speed — and the per-chunk reach is short (~1.65 cm). The lateral drift Δy that I drop sits in the sub-millimeter range.*

---

## 4 · The World Model

The world model I used is **NanoWM**, a ~160M-parameter diffusion-forcing transformer. It does not work in pixels directly, but instead in a compressed *latent* space (initially a frozen Stable-Diffusion VAE). Given a few context frames and a candidate action chunk, it predicts a latent future frame. Stack those predictions and you get a *rollout*: a short imagined sequence of what latent driving would look like.

One critical knob is the **frame interval**, the temporal stride between the frames the model is trained to connect. Too short and each step barely moves the scene, so the action signal is swamped by noise; too long and the prediction gets hard. I return to this parameter in later sections.

The architecture choice worth stating plainly: the perception backbone (the VAE) is **frozen and pretrained**; the 160M transformer is trained **from scratch** on my 50 episodes. So this is a scene-specific latent dynamics model riding on a general perceptual backbone — it learns the physics of *this* room, and generalizes to new trajectories and goals within it, not across environments.

Training uses [**Diffusion Forcing**](https://arxiv.org/abs/2407.01392) (Chen et al., 2024): instead of corrupting every frame to one shared noise level, each frame gets its own independent noise level. With causal masking, that's what lets the network roll itself out autoregressively at inference — predict a frame, treat it as clean context, predict the next — which is exactly the loop CEM drives. So the transformer learns to **denoise the next frame's latent** given recent frames and the action chunk (the action enters through a small **additive embedding**). The rest is unremarkable: AdamW, effective batch 64, bf16, ~**12,000 steps on a single rented H100** — an overnight run.

**"But won't 160M parameters on 50 episodes overfit?"** Yes, fast — and it's the wrong worry, for three reasons. Specializing to *this one room* is the goal, not the failure: the frozen backbone carries the cross-scene generalization, so the trained part only learns one room's dynamics. The metric that screams "overfit" — denoising validation loss — is the wrong dial: planning quality keeps *improving* past where val-loss bottoms out (the U-shaped curve next section), so early-stopping on it gives a *worse* planner. And the real tax of tiny data wasn't memorization but **coverage** — crisp where I drove, blurry-to-hallucinatory where I didn't — fixed by more of the room on tape, not regularization, and exactly the failure a few sections from now.

[FIGURE: ✅ assets/long_0_cmp.mp4 — autoplay/loop/muted]
*Imagined vs. real. Left: a world-model rollout from 4 context frames and a recorded action sequence. Right: what the camera actually saw. It genuinely imagines driving — blurry, but directionally right.*

---

## 5 · Road to a working planner

Everything above is setup. What follows is the build log: each attempt, what broke, and what it taught.

### Planning: MPC + CEM in latent space

How does planning latent space work? It's turns out to be quite simple, as a search over action sequences. Here is the process:

1. Encode the current frame and goal image once.
2. Sample candidate action sequences from a normal distribution and roll each sequence through the world model.
3. Score each imagined endpoint by **distance to the goal in latent space**.
4. Keep the top fraction, resample around them. Repeat a few iterations.
5. Execute the first chunk of the winning plan, then replan from the new observation.

At this point, the score was the **L2 over the flattened VAE latent**. This followed the [**Nano World Models**](https://arxiv.org/abs/2605.23993) paper (Huang et al., 2026), which had reported 25% planning success on PushT using exactly this representation, making it the natural starting point.

---

### Inference setup

The world model and CEM planner run on a rented H100 (Runpod). On the robot, a Raspberry Pi runs LeRobot's LeKiwi host, a ZMQ server that streams the overhead frame and accepts velocity commands. An SSH reverse tunnel connects the two. Each plan drives the base for one chunk (~0.33 s), then the robot stops and waits while the next plan computes (~7 s). This is **stop-and-plan, not real-time control**, which makes network latency irrelevant.

---

### Run 001

This first model was trained on a frame interval of f=5 (~167 ms chunks). It failed a basic action test: roll out with the ground-truth action, zero action, and a random action. Ground-truth should land closest to what really happened. The model predicted nearly the same future regardless. Action-embedding RMS: **0.0088**. Zero and random rollouts were indistinguishable.

---

### Does translation exist in the latent?

If the frame interval is too short, each step barely moves the scene and the action signal drowns in noise. I chose to test this on the recorded frames without retraining: encode them, hold rotation near zero, and compare stationary chunks against pure-translation chunks directly. Translation AUC: **0.94–0.98**. The signal was there. The frame interval just needed to be wider.

[FIGURE: ✅ assets/stationary_latent_compare.png]
*Where motion lives in the latent. Translation lights up the near-field floor (parallax) and rotation lights up the far horizon (the FOV sweeping). The robot's own body stays put, a built-in registration check.*

[FIGURE: ✅ assets/fsweep_chunk_distributions.png]
*Latent change per chunk, across frame intervals. Raising the temporal stride lifts translation's signal above the noise floor.*

---

### Run 002

Retrained at f=10. The action branch came alive: clean **true < zero < random** separation, random now distinctly worse than zero (the model uses the action's *content*, not just its presence). Decoded rollouts visibly track translation, rotation, and arcs.

[FIGURE: ✅ assets/action_diagnostic.png]
*The action test, passed. True-action rollouts clearly beat zero- and random-action. The model now responds to what it's told the robot did.*

[FIGURE: ✅ assets/rotation_0_cmp.mp4 + assets/translation_0_cmp.mp4 — side by side]
*Motion tracking. The world model follows a real rotation (left) and a real translation (right), error growing over the horizon as expected.*

---


### First closed-loop run

[TODO: add goal image, start position, and video of this run]

The robot wandered. Distance-to-goal hovered around 45 for 22 steps, yaw command flip-flopping every step. Hand-placing the robot at measured distances and recording latents directly gave a clear result: **monotone descent over 40 cm with healthy signal-to-noise**.

Close to a goal, the metric has a sharp basin and the robot converges. Far from it, the metric goes flat. Every direction looks equidistant and the robot wanders. The problem wasn't the camera or the planner. It was the **distance metric**.

---

### The semantic pivot

Far from the goal, the **objective was blind**. CEM had no gradient and every candidate action looked equidistant from the goal.

To find out why, each candidate metric was measured at tape-marked positions across the room (10–60 cm out, ±60 cm lateral, ±30° yaw). Two things were tested for each: whether it correctly orders positions by distance at all (ρ, the correlation with true distance), and whether its gradient in the far band stays above the standing-still noise floor — the number CEM actually needs. A metric can score ρ = 1.00 globally and still be useless if the gradient per step is buried in noise.

[FIGURE: ✅ assets/sweep_diagram.png]
*The measurement setup. The robot was hand-placed at each grid position and a frame was captured. Orange poses are the near band (0–30 cm), blue are the far band where CEM needs to plan. At select positions, three yaw orientations were tested (±25°).*

| Metric | Global ordering (ρ) | Far-field gradient / noise (radial, lateral) | Verdict |
|---|---|---|---|
| Pixel L1 | 1.00 | 706×, 386× | fail (lateral ordering wrong) |
| **SD-VAE latent L2** | 1.00 | **1.25×, 0.80×** | **FAIL** (gradient below noise floor) |
| **Frozen DINOv2 patch cosine** | 0.94 | **12×, 21×** | **PASS** |

[FIGURE: ✅ assets/metric_comparison.png]
*Both metrics use the same images. The VAE metric (left) plateaus after ~25 cm — in the far band, one step forward is indistinguishable from standing still, so CEM has nothing to minimize. DINOv2 cosine (right) maintains a clear gradient across the full range.*

The information was in the images all along. The VAE representation was burying it.

The fix was to **retrain the world model to predict frozen DINOv2 patch tokens**, so the imagined space and the distance space are the same. The score becomes one minus average per-patch cosine similarity, requiring no additional training.

This is essentially **DINO-WM** (Zhou et al., 2024), arrived at through measurement. Two differences: a generative diffusion-forcing backbone instead of a deterministic predictor, and closed-loop operation on a real robot.

On the robot, the retrained model drove distance down from 0.32 → 0.19, committing from far out where the VAE objective had been flat.

One practical note: DINOv2 tokens are not directly human-readable. To watch the model think, a small **token-to-RGB decoder** was trained separately on the same episodes — it learns to map predicted token grids back to approximate pixel frames. The planner itself never uses it; it scores entirely in token space. The decoder exists purely for visualization, and is what produces the filmstrip figures in this post.

The retrain also answered the Run 001 question. The dead action wasn't an inherent incompatibility with semantic latents — it was the injection method. Additive injection adds the action as a small residual to each latent; with a strong semantic signal already dominating, the model learns to ignore it (RMS atrophied to 0.0028). AdaLN injection has the action modulate the scale and shift of the entire feature map, a stronger signal the model can't as easily tune out. On the same semantic latents, AdaLN held at 0.2 RMS.

[FIGURE: ✅ assets/c1_smoke_strip.png]
*Imagining in semantic space. The world model predicts DINOv2 tokens; a small decoder renders them back to pixels for visualization (the planner scores in token space and never decodes). Soft but correct, and from a previously-hallucinated viewpoint, it stays in the right room.*

[FIGURE: ⏳ hallucination before/after — pull results/hamper_retest_*.png and the old live-distribution-gap montage from the pod]
*[TODO: same goal frame. Old model snaps to a different room, new model stays put.]*

---

### Building a waypoint graph

The new metric is good for ~40 cm. Beyond that, the goal is out of range and the planner has no signal — start 180° rotated from the goal and CEM has nothing to descend. Every frame in the training data is a place the robot demonstrably reached, so the training data becomes the map.

To build it, DINOv2 tokens are cached for ~4,500 frames (one per chunk boundary), each becoming a **node**. **Temporal edges** connect consecutive frames within each episode. **Weld edges** connect frames from different episodes that pass through the same view, detected by token distance. Fifty disconnected episode threads fuse into one connected map.

At runtime, the live frame is localized against the cache, Dijkstra finds the path to the goal node, and the planner receives the next **waypoint**. The waypoint is always a real remembered frame, about one reach away. CEM only ever sees the local, solvable problem.

Edge thresholds and waypoint spacing are both calibrated from data. The weld threshold comes from inter-frame distance distributions. The waypoint spacing comes from a measured reliability curve where one-step descent succeeds 96% of the time at 2 chunks, falling off beyond that, so waypoints are placed at the 90% reliability point.

[FIGURE: ✅ assets/route_montage.png (wide)]
*A route is a film strip. Dijkstra returns a sequence of remembered frames; the planner chases them one at a time.*

[FIGURE: ✅ assets/subgoal-graph-anim.mp4 — wide, controls]
*Building and routing the graph. Episodes become nodes, shared views weld threads together, Dijkstra hands the planner one waypoint at a time.*

---

### Three failures on the way to the first graph success

**1. The graph must be directed.** First routes sent waypoints backwards along episode threads. This robot doesn't reverse (yet) and data is forward-only. Temporal edges became one-way.

**2. Welds also encode direction.** A weld can silently place a waypoint ~10 cm behind the robot, and tightening the threshold to prevent this collapsed map connectivity. The fix is **motion-parallax certification**: for a weld from frame i to frame j, verify that i's time-successors get closer to j. If they do, j is provably ahead. No new data required. This produced ~17,800 directed welds with 94.5% strong connectivity.

**3. Localization and waypoint tuning.** On the robot, localization flip-flopped between look-alike frames in different episodes, causing the route to re-roll every step. The fix was hysteresis: commit to a path and require strong evidence before re-routing. Waypoints placed too close gave CEM a nearly-identical target, producing near-zero commands, fixed by enforcing a minimum waypoint spacing.

**REACHED nearpurifier.** 129 steps, 40-hop route, localization tracked the whole way, metric closed 0.30 → 0.08. First full end-to-end run on the robot.

Without the graph, the flat planner succeeds from a start distance of 0.35 but wanders from 0.45. The graph crosses exactly that threshold.

[FIGURE: ✅ assets/route_strip_subgoals.png]
*Live routing view: the planner's current subgoal and the planned chain of waypoints ahead.*

[FIGURE: ⏳ on-robot success capture from mpc_semantic_graph_nearpurifier4.rrd — screen-record or trace]
*[TODO: headline run. A dist-to-goal + graph-distance trace, or a screen recording of the viewer.]*

---

## 6 · Honest limitations, and part 2

What this is not: it's one corner of one room, one camera, and stop-and-plan motion — the robot pauses ~7 seconds to think between moves, so it drives in deliberate hops, not smoothly. The goal-image offset between sessions puts a floor under the distance metric, so "arrived" needs a tolerance. Convergence in the final centimeters is goal-dependent — one goal closes to 0.08, another hovers at 0.30. The graph's nodes come from the same data the world model trained on, so the map is exactly as big as where I happened to drive.

Two honest disclaimers. First, none of the pieces are novel. The blunter name for the graph is **teach-and-repeat**: the nodes *are* training frames, localization is nearest-neighbor against them, and a route just replays stitched pieces of drives I've already done. Predicting frozen-DINO features is DINO-WM, the experience graph is ViNG, the planner is textbook sampling MPC, and the world model is specialized to one room by design. If you came for a new method or a deployable nav system, this isn't it — and if you needed dependable indoor navigation tomorrow, classic SLAM-and-plan (or a depth camera and a few libraries) would beat it.

The claim isn't the system — it's the **measurement**. The bricks are off the shelf; the contribution is the debugging: turning "it doesn't work" into a number with a tape measure, killing two confident wrong diagnoses with controlled experiments, and showing the search was never broken while the *representation* was blind — on a real robot, from 25 minutes of data, with the wrong turns left in.

But consider what 25 minutes of driving bought: a robot that drives to a photograph across a room it has no map of, using a world model small enough to train overnight, a distance metric that costs zero training, and a graph built offline in minutes. The architecture that emerged is three layers, each keeping the next inside its comfort zone — a **graph** (topological memory, routes the room) feeding **CEM + world model** (the local planner, ~40 cm of vision), with a **visual-servo endgame** (the final centimeters) as the named next piece.

**Part 2** is the obvious continuation: recollect the full room (more coverage, multiple cameras, and reverse driving — which literally adds edges to the graph); a visual-servo final approach that can strafe and reverse because it bypasses the world model entirely; and an inference speedup from ~7 s toward ~1 s to make the motion continuous.

The lessons, each earned above and worth saying plainly:

- **The objective is part of the planner.** The search was never broken; the metric was blind.
- **Make the bottleneck a number before you change the architecture.** One afternoon with a tape measure redirected the whole project.
- **Judge a world model by its rollouts, not its validation loss.**
- **Topology is cheaper than capability:** a graph fixed what no planning knob could.
- **Your map encodes your robot's physics:** no reverse means a directed graph.

---

*Code coming soon. Built on [LeRobot](https://github.com/huggingface/lerobot) (LeKiwi), [Nano World Models](https://arxiv.org/abs/2605.23993), and frozen [DINOv2](https://github.com/facebookresearch/dinov2) features.*

<!--
==================== OPEN DECISIONS (not for the page) ====================
1. nearhamper A/B: wait for a clean graph landing to make the single best before/after,
   or ship with nearpurifier as headline + hamper framed as the open hard case? (§14/§15)
2. Finding-#4 / C0 probe depth: kept as one paragraph in §11. Expand to its own box for the
   ML audience, or leave inline? (your call from the last review)
3. Failure-heavy backbone: §§4–11 are the wrong-diagnosis arc. Keep full, or compress §§4–6?
4. Assets still to produce/pull before deploy:
   - 🆕 hero video (demo set), system/loop diagram (§7)
   - ⏳ from pod: Gate A curve, hallucination before/after, nearpurifier success trace
   - copy context/figures/subgoal-graph-anim.mp4 into docs/assets/
5. Exact numbers I rounded for readability (full precision in context/experiment-log.md):
   RMS 0.0088 / 0.0028 / 0.207 / 0.333; reached_ratio ~1.0; τ=0.182; 17,796 welds; 94.5% SCC.
==========================================================================
-->
