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

I taught a LeKiwi mobile manipulator to drive to a **photograph**: show it an image taken somewhere in the room, and it finds its way there. The whole stack is learned from **50 teleoperated episodes (~25 minutes, ~45K frames)** through a single overhead camera: a diffusion world model imagines candidate futures, a sampling-based planner picks the actions whose imagined outcome looks most like the goal, and a 4,500-node graph of moments from the training data carries the robot to goals beyond the model's local horizon.

---

## Background — why I built this

I've been fascinated by world model research lately. The field is a collection of competing bets on what the right output is, what the right training signal is, and where the real leverage lies. Dr. Fei-Fei Li describes the current landscape in [**A Functional Taxonomy of World Models**](https://www.worldlabs.ai/blog/taxonomy-of-world-models). She organizes them by what they output. A **Renderer** outputs pixels meant for human eyes, where visual fidelity is what matters. A **Simulator** outputs state, a geometrically and physically faithful representation that programs can compute on. A **Planner** outputs *actions*. Given observations and a goal, it decides what the agent should do next, closing the perception–action loop. She argues these eventually converge into unified models, with simulation as the linchpin.

This project lives squarely in the **Planner** corner of that taxonomy. It is not trying to render a beautiful world or to be a faithful physics engine. The decoded frames are, frankly, blurry. It is trying to use a small, imperfect imagination as the inner loop of a controller. Propose an action, imagine its consequence, score that consequence against a goal image, act. The whole story that follows is what happens when you take the Planner ambition seriously on cheap hardware and a small dataset.

I was inspired by [**Nano World Models**](https://arxiv.org/abs/2605.23993) (Huang et al., 2026), a minimalist, diffusion-forcing world-model codebase released recently. The authors remark that "the broader research community still lacks compact, reproducible, and easily extensible implementations" of modern world models and set out to change that. More than the code, that framing is what grabbed me. It was a call to *democratize* world-model research, to show the ideas don't require a frontier lab's compute or data to be worth building on. That resonated, and it set the constraint that defines everything here: do this small.

The Nano World Models project evaluates across simple control environments, game simulation, and **real-robot data (RT-1)**. If the recipe held on real robot data at that scale, it might hold on *mine*. I had a [**LeKiwi**](https://github.com/SIGRobotics-UIUC/LeKiwi) mobile manipulator left over from earlier imitation-learning work. So the plan was simple. Collect my own data, train a nano-scale world model on it, and see if I could plan with it on real hardware.

---

## 1 · The Problem Statement

The classical way to make a robot go somewhere is a pipeline where you build a map with SLAM, localize yourself in it, plan a path, and follow it.

The bet here is that a world model trained on raw experience can replace all of that, so the task is kept deliberately stark. The robot receives its **current camera frame** and a **target image**, and outputs **body-frame velocities**, with no pre-built map, no external localization, no depth sensor, no reward function, and no task demonstrations.

[TODO: optionally a one-line spec box here — camera, 2-D action (forward velocity + yaw rate), goal-as-photo.]

---

## 2 · Robot Hardware

The [**LeKiwi**](https://github.com/SIGRobotics-UIUC/LeKiwi) is an open-source mobile manipulator from the LeRobot ecosystem, built around a low-cost SO-101 arm on a three-omniwheel kiwi drive base, driven by an onboard Raspberry Pi and inexpensive serial-bus servos. For navigation I use only the base, with the arm parked in a fixed pose throughout. The Pi stays light on the robot side, streaming camera frames and accepting velocity commands over the network while the world model itself runs on a rented cloud GPU connected through an SSH tunnel.

The stock LeKiwi uses a low front-facing webcam, which I replaced with wider-angle USB cameras and supplemented with a [third overhead camera](https://www.amazon.com/dp/B0C289GYVZ?ref=ppx_yo2ov_dt_b_product_details&th=1) on a custom mount, angled down at roughly 55°. That overhead vantage is what everything downstream depends on, capturing the scene across four depth zones at once, from the robot's own body and the near floor out to the mid-room objects and far walls.

*Hardware at a glance: LeKiwi (LeRobot) · holonomic 3-omniwheel base · SO-ARM arm, parked · Raspberry Pi host · low-cost serial-bus servos · overhead USB camera on a custom ~55° mount. [TODO: confirm exact Pi model / servo model / camera model and how the mount was fabricated.]*

[FIGURE: ✅ assets/lekiwi-mount.jpg — photo of the LeKiwi with the custom overhead camera mount]
*The rig. The LeKiwi base with the arm parked and the custom mount holding the overhead camera that everything downstream depends on.*

## 3 · Data

The most important constraint on data collection was what the model actually needs to learn. At inference time the [planner](#road-to-a-working-planner) proposes dozens of candidate action sequences, including bad ones, and the world model has to predict what each would produce in order to rank them. Training only on clean, goal-directed demonstrations would leave the model blind to wrong actions and unable to help the planner reject them. So the collection goal was coverage, not demonstration: fill the space, vary headings, and make sure the model has seen the consequences of driving in different directions.

Collection was entirely manual, using **LeRobot's `record` pipeline** to timestamp and synchronize the overhead camera with commanded base velocities at 30 Hz. For teleoperation, I connected a PS5 DualSense over USB to drive forward velocity on the left stick and yaw rate on the right, with no strafe binding so sideways velocity is zero by construction.

[FIGURE: 🆕 assets/ps5-controller.jpg — photo of the PS5 DualSense used for teleoperation]
*The full data-collection rig. A PS5 DualSense over USB mapped to forward velocity and yaw rate. 25 minutes of driving with this produced the entire dataset.*

The resulting dataset is **50 teleoperated episodes, 44,926 frames at 30 Hz**, available on HuggingFace at [kaushikpraka/wm-smallarea_merged](https://huggingface.co/datasets/kaushikpraka/wm-smallarea_merged) and viewable in the [LeRobot visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset). It covers a roughly 2 m × 5 m carpeted area with blinds closed and room lights on for consistent illumination, furniture in fixed positions as landmarks. Trajectories included arcs, pure forward runs, rotations in both directions, and occasional stationary pauses as identity anchors where the action is exactly `(0, 0)`. I varied start positions and headings across episodes while keeping environmental conditions stable, and drove only forward with no reverse commands. That last choice matters later when building the navigation graph.

[FIGURE: ✅ assets/world_trajectories.png]
*A sample of the driving. Dead-reckoned paths from a handful of the 50 teleoperated episodes, showing the exploratory coverage that makes up the ~25-minute dataset.*

### Dead reckoning the action space

The robot is controlled with two floats, forward velocity and yaw rate, but I don't hand those directly to the model. Instead I **dead-reckon** them, integrating each short window of about five control steps (~167 ms) into a single body-frame pose change, a displacement **(Δx, Δθ)** chunk. This representation has a few key advantages:

- It's a **body-frame** displacement, so "drive forward 5 cm" is the same vector `(0.05, 0)` no matter which way the robot is facing, giving the model heading invariance rather than requiring it to learn it.
- It's **low-dimensional**, keeping the planner's search small.
- It's an **integrated displacement, not a raw velocity**. During steady cruising the velocity is constant for many frames while the image keeps changing, so a model trained on velocities learns the action is uninformative and quietly stops listening. A displacement is nonzero exactly when the robot moves and scales with how far, so it stays coupled to what the camera sees.

Dead reckoning assumes **no significant slip**, meaning a commanded centimeter is a real centimeter across the floor. For a light, slow robot on flat carpet that holds well enough. Since the planner re-observes a fresh frame every chunk, any small accumulated error is corrected rather than compounded. The dropped Δy never exceeds ~0.58 mm, and an open-loop replay on the real robot traced the dead-reckoned path to ~0 cm even through a 117° arc.

[FIGURE: ✅ assets/chunk_deltas.png]
*The action distribution per chunk, dead-reckoned from the raw logs. Forward motion is nearly bang-bang, stopped or full speed, and the per-chunk reach is short (~1.65 cm). The lateral drift Δy that I drop sits in the sub-millimeter range.*

---

## 4 · The World Model

The world model I used is **NanoWM**, a ~160M-parameter diffusion-forcing transformer that works not in pixels but in a compressed *latent* space produced by a frozen Stable-Diffusion VAE. Given a few context frames and a candidate action chunk, it predicts a latent future frame, and stacking those predictions gives a *rollout*, a short imagined sequence of what latent driving would look like.

One critical knob is the **frame interval**, the temporal stride between the frames the model is trained to connect. If it is too short, each step barely moves the scene and the action signal drowns in noise, while if it is too long, the prediction itself becomes hard. I return to this parameter in the next section.

The key architecture choice is that the perception backbone, the VAE, is **frozen and pretrained**, while the 160M transformer is trained **from scratch** on my 50 episodes, making it a scene-specific latent dynamics model riding on a general perceptual backbone. It learns the physics of *this* room and generalizes to new trajectories and goals within it, not across environments.

Training uses [**Diffusion Forcing**](https://arxiv.org/abs/2407.01392) (Chen et al., 2024), which gives each frame its own independent noise level rather than corrupting every frame to one shared level. With causal masking, this lets the network roll itself out autoregressively at inference, predicting a frame, treating it as clean context, and using that to predict the next, which is exactly the loop CEM drives. The transformer learns to **denoise the next frame's latent** given recent frames and the action chunk. The action enters through a small **additive embedding**, a choice that turns out to matter and that I revisit in §5. Training ran for roughly **12,000 steps on a single rented H100** using AdamW with effective batch 64 in bf16, completing in a single overnight run.

[FIGURE: ✅ assets/long_0_cmp.mp4 — autoplay/loop/muted]
*Imagined vs. real. Left: a world-model rollout from 4 context frames and a recorded action sequence. Right: what the camera actually saw. Blurry, but directionally right.*

---

## 5 · Road to Planning

Everything above is setup. What follows is the build log: each attempt, what broke, and what it taught.

### Planning: MPC + CEM in latent space

Planning in latent space turns out to be a straightforward search over action sequences. The loop has five steps.

1. Encode the current frame and goal image once.
2. Sample candidate action sequences from a normal distribution and roll each sequence through the world model.
3. Score each imagined endpoint by **distance to the goal in latent space**.
4. Keep the top fraction, resample around them. Repeat a few iterations.
5. Execute the first chunk of the winning plan, then replan from the new observation.

The initial score was the **L2 over the flattened VAE latent**. The Nano World Models paper had reported 25% planning success on PushT using exactly this representation, making it the natural starting point.

---

### Inference setup

The world model and CEM planner run on a rented H100 (Runpod). On the robot, a Raspberry Pi runs LeRobot's LeKiwi host, a ZMQ server that streams the overhead frame and accepts velocity commands. An SSH reverse tunnel connects the two. Each plan drives the base for one chunk (~0.33 s), then the robot stops and waits while the next plan computes (~7 s). This is **stop-and-plan, not real-time control**, which makes network latency irrelevant.

---

### Run 001

This first model was trained on a frame interval of f=5 (~167 ms chunks). It failed a basic action test. Rolled out with the ground-truth action, a zero action, and a random action, the model predicted nearly the same future for all three, with an action-embedding RMS of **0.0088**. Zero and random rollouts were indistinguishable from ground-truth.

---

### Does translation exist in the latent?

If the frame interval is too short, each step barely moves the scene and the action signal drowns in noise. Rather than retrain, I tested this directly on the recorded frames by encoding them at f=5 and f=10, holding rotation near zero, and comparing stationary chunks against pure-translation chunks. At f=5 the distributions overlap significantly (AUC=0.942, SNR=2.57σ). At f=10 they separate cleanly (AUC=0.978, SNR=4.15σ), confirming the signal was there and the stride just needed to be wider. **AUC** measures how separable two distributions are: 0.5 means a classifier drawing from both does no better than random, and 1.0 means they are perfectly separable. **SNR** is the gap between the two distribution means expressed in standard deviations, where higher means a cleaner separation.

[FIGURE: ✅ assets/stationary_latent_compare_f05.png]
*At f=5 (167 ms), stationary and translation distributions overlap heavily. The near-floor parallax signal is there but buried.*

[FIGURE: ✅ assets/stationary_latent_compare.png]
*At f=10 (333 ms), the distributions separate cleanly. Translation lights up the near-field floor and rotation lights up the far horizon. The robot's own body stays put, acting as a built-in registration check.*

---

### Run 002

Retrained at f=10. The action branch came alive with clean **true < zero < random** separation, random now distinctly worse than zero, confirming the model uses the action's *content*, not just its presence. Decoded rollouts visibly track translation, rotation, and arcs.

[FIGURE: ✅ assets/action_diagnostic.png]
*The action test, passed. True-action rollouts clearly beat zero- and random-action. The model now responds to what it's told the robot did.*

[FIGURE: ✅ assets/rotation_0_cmp.mp4 + assets/translation_0_cmp.mp4 — side by side]
*Motion tracking. The world model follows a real rotation (left) and a real translation (right), error growing over the horizon as expected.*

---


### First closed-loop run

[TODO: add goal image, start position, and video of this run]

The robot wandered. The VAE latent L2 distance-to-goal hovered around 45 (a raw latent distance, not centimetres) for 22 steps, with the yaw command flip-flopping every step. The world model rollouts looked reasonable and the planner was sampling correctly, which pointed the blame at the **distance metric** rather than either of them.

To check, I hand-placed the robot at measured distances from the goal, varied its yaw orientation at each position, and recorded the latents for three candidate metrics: pixel L1, SD-VAE latent L2, and frozen DINOv2 patch cosine. Each metric decreased monotonically with distance, which looked reassuring. What that test missed was the rate of decrease: in the far band the gradient was so shallow it was buried in the robot's own standing-still noise, so CEM could not distinguish one candidate action from another. That failure only became clear through the systematic sweep described in the next section.

---

### The semantic pivot

Far from the goal, the **objective was blind**, giving CEM no gradient and making every candidate action look equidistant from the goal.

To find out why, I measured three candidate metrics at tape-marked positions across the room, ranging from 10–60 cm out along the approach axis and ±60 cm to either side. I graded each on two requirements for CEM to work.

**Requirement 1: global ordering.** Does the metric rank positions in the correct order by distance? I measured this with ρ, the Spearman rank correlation with ground-truth tape distance, where ρ = 1.0 means every closer position scores closer than every farther one. This is the test most people run, and nearly everything passes it.

**Requirement 2: far-field sensitivity.** In the far band (30–60 cm out), is the per-step change large enough for CEM to detect progress above the robot's own standing-still jitter? I report this as a multiple of the noise floor, where 1.0× means one step forward changes the metric by exactly the same amount as the robot's natural variation at rest. Below that, CEM cannot distinguish "I moved toward the goal" from "I am still." This is the test that matters for planning, and it is where the failure hides.

[FIGURE: ✅ assets/sweep_diagram.png]
*The measurement grid. I hand-placed the robot at each position and captured a frame. Orange poses are the near band (0–30 cm), blue are the far band where CEM needs to plan. Three yaw orientations were tested at select positions.*

| Metric | Ordering (ρ) | Far-field signal / noise | Why it fails |
|---|---|---|---|
| Pixel L1 | 1.00 | 706×, 386× | Strong signal, but lateral ordering breaks. The metric increases when moving sideways even when that sideways position is closer, so CEM would overcorrect. |
| **SD-VAE latent L2** | 1.00 | **1.25×, 0.80×** | Perfect global ordering, but the far-field gradient is at or below the noise floor. CEM cannot see progress more than ~25 cm out. |
| **Frozen DINOv2 patch cosine** | 0.94 | **12×, 21×** | Passes both. |

[FIGURE: ✅ assets/metric_comparison.png]
*Both metrics use the same images. The VAE metric (left) plateaus after ~25 cm, where one step forward is indistinguishable from standing still and CEM has nothing to minimize. DINOv2 cosine (right) maintains a clear gradient across the full range.*

[FIGURE: ✅ assets/fsweep_chunk_distributions.png]
*Why the VAE latent fails as a planning metric. At f=10, rotation strongly predicts latent displacement (corr=0.70), while translation magnitude barely registers (corr=0.03). Turning sweeps the far horizon across the frame and dominates the latent; driving forward produces only near-floor parallax that the VAE compresses away. A metric built on this representation cannot see translational progress.*

The information was in the images all along and the VAE representation was burying it.

The fix was to **retrain the world model to predict frozen DINOv2 patch tokens**, so the imagined space and the distance space are the same.

The new score works as follows. DINOv2 divides an image into a grid of small patches and computes a feature vector for each one. Given two images, you compare their corresponding patches using cosine similarity, which measures how aligned two vectors are in direction regardless of magnitude. A value of 1.0 means the patches are semantically identical and 0 means nothing in common. Average those similarities across the whole grid and subtract from 1, so the score is 0 at the goal and grows as the images diverge. No training is needed for any of this, since DINOv2 is frozen and pretrained and the score is just arithmetic on the output token grids.

$$\text{score} = 1 - \frac{1}{N} \sum_{i=1}^{N} \cos\!\left(\mathbf{f}_i,\, \mathbf{g}_i\right)$$

*N is the number of patches. **f**_i and **g**_i are the DINOv2 patch token vectors for patch i of the imagined frame and the goal image.*

This is essentially **DINO-WM** (Zhou et al., 2024), arrived at through measurement. The main differences are a generative diffusion-forcing backbone instead of a deterministic predictor and closed-loop operation on a real robot.

On the robot, the retrained model drove the DINOv2 cosine distance to goal down from 0.32 to 0.19 (on a 0–1 scale), committing from far out where the VAE objective had been flat.

DINOv2 tokens are not directly human-readable, so to watch the model think, I trained a small **token-to-RGB decoder** separately to map predicted token grids back to approximate pixel frames. The planner itself never uses it, scoring entirely in token space, and the decoder exists purely for visualization.

The retrain also answered the Run 001 question. Switching the prediction target from VAE latents to DINOv2 tokens meant training from scratch, which opened a clean opportunity to probe what had actually caused the dead action. Run 001 used VAE latents with additive injection, leaving two explanations open. Either the VAE representation was too weak for action conditioning to matter, or the additive injection method was too easy to ignore regardless of representation. Running both injection methods against the same semantic latents settled it. The dead action was not an inherent incompatibility with semantic latents, but a problem with the injection method.

**Additive injection** adds the action embedding as a residual to each transformer layer. The model can neutralize that influence by learning to make the residual contribution small. With a strong semantic signal already dominating the latents, that is exactly what happened, and the action embedding RMS atrophied to 0.0028.

**AdaLN injection** (Adaptive Layer Normalization) works differently. Standard LayerNorm normalizes activations and applies a fixed learned scale $\gamma$ and shift $\beta$. AdaLN instead predicts those values dynamically from the conditioning signal, in this case the action embedding $\mathbf{a}$.

$$\text{output} = \gamma(\mathbf{a}) \cdot \text{LayerNorm}(x) + \beta(\mathbf{a})$$

Because the action now multiplicatively controls the scale of the entire feature map at every layer, the model cannot reduce its influence by tuning a weight toward zero. On the same semantic latents where additive injection collapsed to 0.0028 RMS, AdaLN held at 0.2 RMS.

[FIGURE: ✅ assets/c1_smoke_strip.png]
*Imagining in semantic space. The world model predicts DINOv2 tokens, and a small decoder renders them back to pixels for visualization. The planner scores in token space and never decodes.*

[FIGURE: ⏳ assets/dinov2_planner_demo.mp4 — on-robot demo of the DINOv2 flat planner reaching a nearby goal]
*[TODO: short clip of the DINOv2 planner working on the robot without the graph — goal within ~40 cm, metric descending, robot converging. Shows the metric works before introducing the graph as the solution to its range limit.]*

---

### Building a waypoint graph

The new metric is good for ~40 cm. Beyond that, the goal is out of range and the planner has no signal. If you start 180° rotated from the goal, CEM has nothing to descend. Every frame in the training data is a place the robot demonstrably reached, so the training data becomes the map.

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

**3. Localization and waypoint tuning.** On the robot, localization flip-flopped between look-alike frames in different episodes, causing the route to re-roll every step. The fix was hysteresis, committing to a path and requiring strong evidence before re-routing. Waypoints placed too close gave CEM a nearly-identical target, producing near-zero commands, fixed by enforcing a minimum waypoint spacing.

[FIGURE: ⏳ TODO — insert additional on-robot success run videos here (e.g. screen recordings or .rrd traces of multiple goal-reach runs)]

The robot reached the **nearpurifier** goal in 129 steps along a 40-hop route, with localization tracked the whole way and the metric closing from 0.30 to 0.08. First full end-to-end success.

Without the graph, the flat planner succeeds from a start distance of 0.35 but wanders from 0.45. The graph crosses exactly that threshold.

[FIGURE: ✅ assets/route_strip_subgoals.png]
*Live routing view: the planner's current subgoal and the planned chain of waypoints ahead.*

[FIGURE: ⏳ on-robot success capture from mpc_semantic_graph_nearpurifier4.rrd — screen-record or trace]
*[TODO: headline run. A dist-to-goal + graph-distance trace, or a screen recording of the viewer.]*

---

## 6 · Limitations

The scope here is deliberately narrow, and that narrowness is a tradeoff worth naming. The system covers one corner of one room through a single overhead camera, with goals drawn from the same space the robot drove through. Navigation is stop-and-plan, meaning the robot pauses roughly 7 seconds between moves rather than driving continuously, and the graph covers exactly the ground the training data crossed. These bounds were chosen, not stumbled into. The question this project set out to answer did not require a general solution, and trying to build one would have buried the interesting parts under engineering scaffolding.

None of the algorithms here are original, and that too is intentional. The graph is a form of teach-and-repeat, where nodes are training frames, localization is nearest-neighbor lookup, and routes replay stitched segments of prior drives. The distance metric is frozen DINOv2 cosine similarity, which is DINO-WM. The experience graph mirrors ViNG. The planner is textbook sampling MPC. Using proven building blocks was a choice, not a shortcut, because the interesting question was never whether a new algorithm could navigate a room. It was whether these specific pieces could be combined and tuned to work reliably together on real hardware from 25 minutes of data. The difficulty was in the integration, and that is where all the effort went.

## 7 · What comes next

The most immediate constraint is coverage. Collecting more trajectories around the room adds nodes and edges to the graph. Multiple cameras would widen the field of view. Reverse driving adds backward edges without retraining. A visual-servo final approach that can strafe and reverse, bypassing the world model entirely, would handle the last centimeters more reliably than CEM. Faster inference, from roughly 7 s toward 1 s per step, would make motion continuous rather than hop-by-hop.

The broader architecture already has a natural three-layer shape. A graph handles topological memory and long-range routing. CEM and the world model handle local planning across roughly 40 cm of horizon. A visual-servo endgame handles the final approach. Each layer keeps the next one inside its comfort zone.

What 25 minutes of driving produced: a robot that navigates to a photograph across a room it has never explicitly mapped, using a world model small enough to train overnight, a distance metric requiring no additional training, and a graph built offline in minutes.

The lessons, each earned above:

- **The objective is part of the planner.** The search was never broken; the metric was blind.
- **Make the bottleneck a number before you change the architecture.** One afternoon with a tape measure redirected the whole project.
- **Judge a world model by its rollouts, not its validation loss.**
- **Topology is cheaper than capability.** A graph fixed what no planning knob could.
- **Your map encodes your robot's physics.** No reverse means a directed graph.

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
