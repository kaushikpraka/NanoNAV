# Learned Distance Metric (Quasimetric) for Planning — Design

**Status: PLANNED (design settled 2026-06-09; not yet implemented).** The chosen replacement for the
raw SD-VAE latent-L2 planning objective. This is the durable design home for the work flagged as the
**#1 planner priority** in [[open-questions]] "Scoring function alternatives" and [[roadmap]] 6d.
**Build order decided:** simple temporal-MLP baseline → QRL/IQE quasimetric → topological graph (see
"Build order"); the metric is a hard prerequisite for the graph. **Next concrete action = encode-cache +
sweep eval + rung-0 baseline + sweep grade (the GO/NO-GO gate).**

> **Dependency note.** This does *not* fix the [[open-questions]] "Live-frame distribution gap" (the WM
> hallucinates from OOD `z0`) — that is upstream and needs the recollect + retrain. A better distance
> can't help if it scores hallucinated latents. So: build + validate the metric **offline now** (it only
> reads cached latents), but expect the on-robot win to land **after** the coverage/hallucination fix.

---

## Why — the problem it solves

On-robot 6b confirmed the **objective is the bottleneck, not the WM or the CEM search** (offline CEM
already hits the WM's accuracy ceiling, Stage 6a). The current cost — raw flattened SD-VAE latent-L2
`‖z0 − zg‖`, equal weight on every latent cell — measures **how different two frames *look***, which is
a bad proxy for **how far apart they are to *drive***:

- Well-conditioned **near** the goal (sweeps: basin floor ~16, SNR ~17–97/10 cm radially) but **flat far
  out**: most latent cells encode generic floor/wall, so two distant poses look ~equidistant → no
  gradient on the "plateau" → CEM flails / commands turns when straight is needed.
- **Under-credits real progress:** the planner rotated the bot to face the chair, yet `dist` stayed ~48,
  so nothing locked the alignment in → overshoot. `--horizon 5`, `--var-scale 2`, `--vx-max 0.12` all
  fail because they don't change *what is optimized*.

See [[experiment-log]] 2026-06-09, [[open-questions]] "Scoring function alternatives".

## What it learns

A **driving-distance estimator**: given "where I am" and "the goal" (two latents), return *≈ how many
action-chunks of driving separate them*. Concretely it approximates the **optimal cost-to-go** `d*(a,g)`
= minimum chunks to drive from the pose in frame `a` to the pose in frame `g` — a **geodesic on the
manifold of drivable poses**, i.e. shortest-path length on the graph whose edges are "one chunk of
driving." Units = action-chunks (one chunk = f=10 = 0.333 s). It keys on **drivability, not
appearance**: two spots 20 cm apart on open floor are *near*; 20 cm apart with a wall/object between is
correctly *far* — the thing raw L2 cannot distinguish, and exactly what a planner needs.

The learned encoder's job is to **read pose/place out of the latent and ignore decoration** (floor
grain, wall colour, lighting) — the directions that make raw L2 flat.

## Environment (the actual room) — `context/figures/room.jpg`

A small open carpet area ringed by **visually distinct landmarks** (black standing fan, TV + desk +
office chair, floor lamp + framed picture, cream laundry hamper, white air purifier, grey curtains, bed
with a teal blanket); the LeKiwi sits on the rug mid-frame. Three consequences for the metric:

- **Landmark-rich perimeter → low perceptual-aliasing risk.** Distinct anchors per region mean distinct
  poses rarely *look* identical — the biggest factor *in our favour*: it lets even a simple method learn
  "which region / which heading," and it **softens the quasimetric's worst pitfall (wormholes from
  aliasing)**.
- **Low-texture beige carpet centre (+ translucent chair-mat) → the flat-appearance trap, but localized**
  to floor-facing / centre poses, not the whole room.
- **Small but many-chunks-deep** (~3 cm/chunk → crossing the rug is tens of chunks), so
  far-goal-outside-one-plan-basin is a real regime here (consistent with the `nearfan` plateau).

Net: a **forgiving** environment for distance learning — which is *why* the simple baseline (see "Build
order") is worth running first. (The original attachment `room.png` is an **HEIC photo misnamed `.png`**;
`sips -s format jpeg room.png --out x.jpg` to view — the committed `context/figures/room.jpg` is the
converted copy.)

## The objective (Quasimetric RL — plain version)

A quasimetric is a distance obeying the triangle inequality but allowed to be **asymmetric**
(`d(a,b) ≠ d(b,a)` — driving forward vs having to U-turn back cost differently). We learn `d` from two
opposing forces plus one structural guarantee:

1. **Local cap (the only ground truth):** back-to-back chunks in an episode are 1 step apart →
   `d(zₜ, zₜ₊₁) ≤ 1`. Free, self-supervised from the video timeline. No reward, no pose.
2. **Push apart (anti-collapse):** *maximize* `d` over random latent pairs.
3. **Structure:** the head (IQE — Interval Quasimetric Embeddings) **guarantees the triangle inequality
   by construction**, so `d` can't cheat on consistency. (A plain MLP regressor provably cannot learn a
   quasimetric reliably — the structure is load-bearing; see learnability ref below.)

**Why maximize (the counterintuitive part).** With only the local cap, the trivial useless solution is
`d ≡ 0` (everything is distance zero — obeys the cap). The push-apart force prevents that collapse. And
maximizing *as hard as possible* doesn't overshoot, because the triangle inequality + unit caps bound
how far any pair can be stretched — to exactly the **shortest chain of unit steps connecting them**.
Beads-and-strings: real transitions are strings of length ≤1; pull all beads apart; strings go taut;
taut separation = shortest path = `d*`. Maximizing against local caps makes the metric structure do the
shortest-path computation. The practical loss is a Lagrangian relaxation (push-apart term + soft
`relu(d−1)²` constraint penalty + a dual multiplier `λ` learned by gradient ascent); exact form in the
QRL paper §5.

**Cross-trajectory stitching (why it beats naive temporal regression).** Different episodes get welded
wherever they pass through the **same place** (a shared near-identical view → near-identical latent =
shared graph node). The triangle inequality then routes cross-episode pairs through shared nodes, so
distances between different-trajectory frames are correctly capped at the true short driving distance —
the push-apart force *cannot* inflate a pair that has a real connecting chain. Our **loopy exploratory
data**, which *poisons* time-gap regression (same spot revisited 50 chunks later ≠ 50 steps away), is
turned into useful loop-closure evidence here. The one genuine failure is a **coverage gap** (two
nearby spots with *no* connecting drive in the data → over-estimated as far) — which is **safe
pessimism** for a planner, rare in a small crisscrossed room, and shrinks with the retrain.

## Architecture

```
frozen SD-VAE → z [4,32,32]
   │  φ : small CNN over the [4,32,32] grid → flatten → MLP → e ∈ ℝ²⁵⁶   (trainable)
   │  IQE quasimetric head → d(z_a,z_b) = IQE(φ(z_a), φ(z_b))   (asymmetric, triangle-ineq by construction)
```
Train an **ensemble (2–4 heads)**; use the **pessimistic (max)** distance for any edge/subgoal admission
— guards against single-head "wormhole" false shortcuts. VAE stays **frozen**; only φ + head train
(minutes–~1 h on the H100). **Reads latents, never decodes** → the known decode-blur and pixel-space
hallucination never touch the cost path. MRN (Metric Residual Network) is a simpler alternative head.

**Encoder vs head — what's actually trained.** φ (the CNN) is a plain *encoder*: it maps a latent to an
embedding and is **not itself a quasimetric**. The quasimetric properties (asymmetry, triangle inequality,
`d(x,x)=0`) come entirely from the **IQE head** on top, which holds for *any* encoder weights. The two
train **jointly end-to-end** under the QRL loss — φ carries all the capacity (learns *which* latent
directions mean pose), IQE (≈parameter-free) supplies the geometry. A **single shared** encoder is used
for both source and goal; directionality comes from the head, not from separate encoders (contrast
contrastive RL's two encoders φ,ψ). Net: this is QRL's published image-observation recipe with φ reading
the `[4,32,32]` SD-VAE latent instead of raw pixels. `torch-quasimetric` provides IQE as a drop-in, so we
write only φ + the training loop.

## SD-VAE latent handling

The metric never treats the SD-VAE latent as a meaningful distance space — handling its quirks is the
whole job of φ, and the quasimetric lives **downstream of φ**, insulated from the raw-latent pathologies.
A frame → `[4,32,32]` = 4096-D, with three awkward properties and their handling:

1. **Spatial, not semantic ⇒ raw L2 ≈ pixel-MSE = appearance distance** (exactly why the current cost is
   flat — most cells encode generic floor/wall). **Handling: φ is a small *CNN* over the `[4,32,32]` grid**
   (not a flat MLP), whose explicit job is to **disentangle pose/place from appearance** — keep the
   directions that move when the robot drives (content shifting across the grid = the parallax/optical-flow
   signal), discard texture/lighting. A CNN is the right inductive bias *because* the latent is spatial; an
   MLP over 4096-D discards the layout that encodes pose. The IQE quasimetric operates on φ's ~256-D output,
   never the raw latent (the research flag "don't put IQE on raw 4096-D").
2. **Not L2-isotropic** (4 channels, different scales; SD `scaling_factor ≈ 0.182`; tiny KL ⇒ not unit
   Gaussian). **Handling: normalize to the WM/codec convention before φ** (scaling-factor or dataset
   per-channel mean/std) so φ's input is well-conditioned and the non-isotropy is absorbed at the input,
   not by the metric.
3. **Low-level + lossy** (blurry decode). Irrelevant to the metric — **we read latents, never decode**, and
   pose is still recoverable (the `measure_dist_sweep` curves confirm it).

Two consistency points specific to using SD-VAE latents:

- **Stay in the WM's latent space — no decode, no re-encode.** The cost scores *generated* latents `ẑ` from
  the WM rollout, and the WM rolls out *in SD-VAE latent space*. So φ consumes WM-space SD-VAE latents
  directly: `φ(ẑ)`, `φ(z0)`, `φ(zg)` are the same representation and compose without conversion. This is a
  deliberate choice over scoring in a different space (decode → DINO/CLIP, the [[open-questions]]
  "decoupled generation/scoring latents" idea), which would reintroduce the decode into blurry/hallucinated
  pixels.
- **Train φ on the WM's *imagined* latents, not only clean encoded ones.** At deploy time the cost scores
  `ẑ`, which are **approximate / slightly off-manifold** (the reason decodes blur). If φ only trained on
  clean VAE-encoded dataset latents it could misbehave on imagined ones. So φ's training set should
  **include WM-rolled-out latents** (or at least be validated on them) — the SD-VAE-latent version of the
  live-distribution-gap concern, one level down.
  - **Decision (2026-06-09): VALIDATE on WM latents first, ADD them to the training set later.**
    Rungs 0/1 train on clean encoded dataset latents (cheap, simple), but the sweep eval **must include a
    WM-imagined-`ẑ` arm** — feed φ generated latents (a short WM rollout to a known displacement) and check
    `d_learned(ẑ, zg)` tracks the clean-latent curve. If clean and imagined latents of the **same pose** map
    to materially different `d` (the clean↔imagined weld is loose), *then* fold WM-rolled-out latents into
    φ's training set (a third pair-type alongside adjacent-chunk and cross-episode). Don't pay that cost
    up front — gate it on the validation arm. This is the same "loose-weld under input shift" axis as the
    noise/near-pose concern below.

**Why SD-VAE's spatiality is actually an asset here:** because the latent preserves layout, heading/position
show up as *where* content sits in the grid — readable by a conv φ, and the reason pose is recoverable at
all. A global *semantic* embedding (CLIP/DINO) is often heading-*invariant* — good for place recognition,
bad for the fine pose/heading discrimination a planner needs. So a spatial SD-VAE latent + CNN φ is a
reasonable pairing for this job, not a liability. **The gate that this worked** is the sweep eval: `d_learned`
must be monotone + non-flat at 40–60 cm before the robot. If it isn't, the fallback is a richer φ or
distilling semantic structure into it (the VLM-teacher channel).

## Training

- "Step" = **one action chunk** (f=10), not one 30 Hz frame — pair spacing must match the planning step
  so `d≈1` means "one planning step."
- Batch = {adjacent-chunk pairs → constraint term} ∪ {random cross-episode pairs → push-apart term}.
- Dual-ascent on `λ`. No reward, no pose. Trains on the **existing** dataset (pre-encode all ~45K
  frames / ~4,490 chunks to latents once, cache).

## How it plugs into the planner — current vs generated states

Two distinct roles (both replace a `‖·‖` call):

- **Planning cost → on the GENERATED (imagined) latents.** CEM ranks candidate actions by where the WM
  *imagines* they land: `d_learned(ẑ, zg)` replaces `MSE(ẑ_H, zg)` in `objective.py`. This *must* be on
  generated states — `z0` is identical across candidates, so only the imagined outcome differentiates
  actions. This is the term that fixes the flat plateau.
- **Termination / readout → on the CURRENT state.** `dist_to_goal = d_learned(z0, zg)` (real observed
  frame vs goal). "Reached" → `d_learned(z0, zg) < ~1` step. Recalibrate `--reach-thresh` to the new
  (chunk-count) scale.

**Refinement — score the near horizon, lean on the metric for the far.** With a real steps-to-go metric,
minimizing `d_learned(ẑ₁, zg)` (the **+1** imagined chunk — least WM-degraded *and* the one actually
executed) means "take the chunk that most reduces *total* steps-to-go." The long-horizon reasoning lives
in the *metric*, so the WM rollout can stay **short** — sidestepping far-horizon compounding error and
the OOD hallucination. (Caveat: the cost still rides on generated states, so its quality is **bounded by
WM rollout fidelity** → another reason the retrain matters.) All in the same SD-VAE latent space → `ẑ`
feeds φ with no decode; `φ(z0)`, `φ(zg)` computed once/plan, `φ(ẑ)` per candidate — tiny head, cheap
inside CEM's ~96 rollouts.

## Evaluation (already half-built)

`scripts/measure_dist_sweep.py` gives **hand-placed ground-truth cm-displacement → latent** (radial +
`--yaw-sweep`; add a **lateral** sweep). That is the held-out test for any candidate distance:
*monotone in true displacement, and non-flat at 40–60 cm where raw L2 plateaus?* Rank **the rung-0
temporal-MLP, the rung-1 QRL quasimetric, and raw-L2** on this before touching the robot — the grade is
the GO/NO-GO gate (see "Build order"). (Ties into the broader
"rigorous eval" thread — this is the absolute, physically-grounded metric the WM-relative 6a
`reached_ratio` lacked.)

## Pitfalls / risks & mitigations

The quasimetric trades **"flat but honest" (raw L2) for "sharp but possibly confidently wrong."** Good for
planning *only if* the "confidently wrong" failure is actively policed. Split into QRL-specific risks and
ones shared by any learned distance but acute here.

**QRL-specific:**
1. **Wormholes — the triangle inequality amplifies errors globally.** QRL recovers *shortest-path* cost-to-go;
   if φ **aliases two distinct places** to nearby embeddings (likely — our latents are appearance-heavy in a
   low-texture room, the *same* condition that flattened L2), the metric routes everything through the
   spurious short link and **collapses distances across the whole graph**. Raw L2's aliasing errors are
   *local*; a quasimetric's **propagate**. Worst in under-covered regions (nothing pushes the aliased pair
   apart). *Mitigate:* ensemble + pessimistic-max; **ensemble disagreement as a wormhole detector**; the
   topological graph (below) filters implausible long-range edges.
2. **Min-max / dual-ascent training is fragile + hyperparameters won't transfer.** The constrained adversarial
   objective can collapse (`d→0`), inflate, or oscillate; published defaults are tuned for D4RL-style
   benchmarks, not SD-VAE-latent nav → expect to retune ε / λ-schedule / push-apart transform.
3. **Asymmetry can be mis-estimated.** Learning good directionality needs data exhibiting it; our
   near-holonomic base + exploratory teleop may yield spurious asymmetry from **coverage imbalance**
   (more-driven directions look cheaper). Adds expressiveness *and* a way to be subtly wrong.
4. **IQE inductive bias** — distance is max-over-components ⇒ bottleneck-dominated; underfits metrics not of
   that form. MRN is the fallback head.

**Shared but acute here:**
5. **Bounded by WM rollout fidelity — NOT a fix for hallucination.** The cost scores imagined `ẑ`; a perfect
   metric on a hallucinated `ẑ` is still garbage. Gated by the live-distribution gap, doesn't solve it.
6. **Reachability ≠ controllability.** `d*` assumes an *optimal* controller + *uniform* step cost; ours
   **under-drives**, has a rotation deadband, near-bang-bang speed, no strafe, and a chunk is ~0–3 cm. So the
   metric can hand CEM a gradient the controller can't follow, and "1 step" of metric ≠ 1 step of progress —
   it's *time-steps under the data's behavior*, not geometric distance (inherits the stop-heavy teleop).
7. **Coverage-bounded** — only knows the explored graph (50 episodes); unexplored links → safely over-far but
   un-routable. Same wall as the retrain.
8. **Verification gap** — no global pose ground truth (that's *why* it's self-supervised), so the metric can't
   be certified globally; the `measure_dist_sweep` curves are **local spot-checks** and won't catch a wormhole
   between unsampled regions until it causes a bad plan.
9. **Loose welds — the benign tail of the same SNR axis as #1.** The *opposite* end of wormholes: the **same**
   pose in two frames maps **slightly apart** (sub-chunk position jitter, exposure/WB flicker, AV1/sensor
   noise, or the clean↔imagined-`ẑ` gap). #1 is "different places too close"; this is "same place not close
   enough." It is **far more forgiving** — the error is **additive and bounded** (a loose weld `d(P,P')≈ε`
   adds ε of slack to a stitched path `d(A,B) ≤ d(A,P)+ε+d(P',B)`, bounded by φ's Lipschitzness), *not*
   globally propagated like a false weld. **It's largely handled by construction:** (a) the **local cap is
   itself the noise-robustness supervision** — adjacent chunks differ by ~one chunk of motion *plus exactly
   this nuisance*, so `d(zₜ,zₜ₊₁)≤1` literally trains φ to map slightly-perturbed near-same-pose views close;
   (b) **chunk-count units quantize it away** — a weld at d≈0.2 is sub-step, below reach-thresh / edge-τ / the
   k-NN localization margin; (c) **capture SNR is fixed upstream** (exposure+WB lock, avoid lossy AV1 — the
   recollect mitigations). *Where it stops being free is exactly the low-texture carpet centre* — there
   same-pose-plus-noise is indistinguishable from different-pose-similar-look, i.e. it **collapses back into
   #1**. They are **one knob** (φ's pose-vs-nuisance SNR), graded by the **lateral `measure_dist_sweep` arm**
   (d near-zero for small real displacement = weld tight enough, *without* going flat for large = not a
   wormhole) + ensemble disagreement. See the "VALIDATE on WM latents first" decision under SD-VAE latent
   handling — the clean↔imagined gap is this same loose-weld concern under input shift.

**Design-against-from-day-one:** wormholes (#1) and reachability≠controllability (#6) — the two with the
sharpest teeth (the metric amplifies the aliasing our latents are prone to; and we *already observe* the
under-drive, so an optimal-control metric over-promises). The loose-weld tail (#9) is benign on its own but
shares #1's knob, so the **lateral sweep + ensemble disagreement grade both ends at once**. All of the above
are **testable offline on the sweeps + ensemble disagreement before the robot**.

## Subgoal layer — topological graph (for far goals, AFTER the metric)

The metric fixes the gradient; the graph fixes the **reach**. One CEM plan covers ~10 cm (H=3), and the
metric is least reliable far-field — so for a goal across the room, **decompose it into a chain of nearby
subgoals, each one CEM can drive**, and do the long-range routing with discrete search instead of trusting
the raw continuous metric across the whole room. Mental model: the 50 episodes already traced a tangle of
paths; the graph **compresses that into a road-map** and you navigate by shortest-path on it, handing CEM
one short hop at a time.

**Anatomy (NanoNAV specifics):**
- **Nodes = real dataset frames.** Sparsify the ~4,490 chunk-latents into representative "places" (simple
  downsample, or SGM two-way-consistency merge of interchangeable frames). Critical: nodes are **real,
  in-distribution latents**, not imagined ones.
- **Edges = "reachable in ≈ one plan," weighted by the metric.** Directed edge A→B with weight
  `d_learned(A,B)` whenever `d_learned(A,B) < τ`, **τ ≈ one CEM reach (~3 chunks)** — admitted with the
  **ensemble-pessimistic** distance. **τ is the key knob (= SoRB `MaxDist`):** too large → admits wormholes
  / "teleports"; too small → graph fragments into disconnected within-episode chains and far goals have no
  path. Set it to one controller-reachable hop.
- **Goal insertion + localization:** encode goal → nearest node (goal node); each step encode the current
  frame → nearest node (source). Both are k-NN-in-`d_learned` lookups.

**Runtime loop:** localize → **Dijkstra** (directed, weights `d_learned`) source→goal → hand CEM the
**first waypoint** as `zg` → CEM+WM drive that short in-basin hop → on arrival
(`d_learned(now, waypoint) < switch-thresh`) **re-localize + re-plan** to the next node → recurse. So:
**graph = global topology, CEM+WM = local dynamics, and the metric is trusted only on the short
in-distribution pairs it's most reliable on** — never on a far raw gradient.

**Why this dodges our two bites:** (1) **hallucination** — every subgoal CEM gets is a *real* frame → clean
`zg`, WM rolls forward only a few chunks from the *real* current frame toward a *nearby* target; we never
ask it to imagine a far future or seed from an OOD pose (the reason the graph beats WM-imagined subgoals
*for us*). (2) **far-field metric error** — discrete shortest-path over **vetted short edges** can't be
silently misrouted by one bad long-range value; **the graph doubles as a wormhole guard** (a spurious
short metric estimate shows up as an implausible edge / ensemble disagreement you filter at build time).

**Failure modes:** coverage (50 eps → possibly poorly-connected graph; improves with retrain);
localization errors on OOD frames → mis-route (narrows but doesn't escape the live-distribution
dependence); **approach-heading / directionality** (a node is a frame at one heading; arriving from
another, the view may not match — the directed metric + switch-thresh must tolerate it); stuck/timeout →
SGM-style **self-supervised edge removal** (fail a hop repeatedly → delete that edge, re-route).

This is the evolution of the original **Stage 7** waypoint graph, but with **edges from `d_learned`**
instead of DepthAnything3 geometric reconstruction (no separate metric-reconstruction step; reuses the
distance head). Knobs map to [[open-questions]] "Waypoint graph construction details" / "Waypoint
switching".

**Deferred (post-retrain):** WM-imagined subgoals ("plan fully in the WM, no manual graph", Director /
LEXA / Subgoal-Diffuser) — revisit once coverage/hallucination is fixed.

## Optional — VLM as an offline teacher (not runtime)

A VLM is a poor *runtime* cost (latency × ~96 candidates/plan; forces a decode into blurry/hallucinated
pixels; too coarse for sub-chunk granularity; OOD top-down view). But it's a strong **offline teacher**:
run it once on the **real, sharp** dataset frames to produce **place-equivalence / relative-progress
labels** for *cross-episode* pairs — the exact blind spot self-supervised temporal distance has (no
time-index across episodes) — and distill into φ as a second supervision channel. Local scale stays
temporal-adjacency; VLM supplies coarse global anchoring. Optional; add only if the sweep eval shows
cross-episode regions too flat. (Lit: RL-VLM-F, VLM-RM, LIV, LM-Nav.)

## Build order — simple temporal-MLP baseline BEFORE the quasimetric

**Decision (2026-06-09): build a simple temporal-distance MLP first as a baseline, then escalate to QRL
only where it plateaus.** Not "build both at once," and not "dive straight into QRL." Rationale:

- **~80% shared scaffold, not throwaway.** Both need the same latent cache, pair sampler, **CNN φ**, and
  sweep eval; only the head + loss differ (MLP+MSE/Huber vs IQE+QRL min-max). The baseline stands up
  everything the quasimetric needs, **plus a permanent comparator** for the eval.
- **The room is forgiving enough that the baseline has a real shot** (landmark-rich perimeter → "which
  region / heading" is easy; see "Environment"). If it passes the sweeps, we may not need the fragile QRL
  min-max training for v1.
- **De-risks the quasimetric:** QRL's min-max / dual-ascent is the fragile part; debugging "broken or just
  hard?" is far easier with a working baseline on the same harness.

**The variable that actually decides flat-vs-not is the cross-trajectory negatives, NOT the head.** A
within-trajectory-only temporal regressor *will* be flat across regions (no labels there). The cheap
decisive upgrade is the **ViNG trick: label cross-trajectory pairs as max-distance** (or bucket
near/medium/far). Build the baseline *with* that, and ablate with/without it.

**Expected outcome (still informative either way):** likely **good in-region / near-goal** (landmarks
carry it), **flat or noisy cross-region + on the low-texture centre** (time≠space + multimodality bite).
That *localizes* where QRL is actually needed (the far-field / cross-region regime the graph edges depend
on) and gives a baseline number to beat.

**Rungs: (0) temporal-MLP + cross-traj negatives → (1) QRL/IQE quasimetric → (2) topological graph.** The
metric (rung 0/1) is a **hard prerequisite** for the graph — its edges *are* `d_learned` — and it delivers
value on its own (in-basin CEM), so it is strictly first; the graph cannot precede it.

## Sequencing

0. **Pre-encode dataset → latents (once, cache) + extend the sweep eval** (add lateral to radial+yaw).
   *(offline, now)*
1. **Rung 0 — simple temporal-distance MLP** (+ cross-trajectory negatives; ablate). Grade on the sweeps.
   *(offline, now)*
2. **Rung 1 — QRL/IQE quasimetric** (+ ensemble), only for the regime where rung 0 plateaus. Grade vs
   rung 0 + raw-L2; check ensemble disagreement (wormholes). **The sweep grade is the GO/NO-GO gate for
   the whole approach.** *(offline, now)*
3. **Swap the winner into CEM** — cost on the generated `ẑ`, termination on `z0`; recalibrate
   reach-thresh; test on a well-covered goal (`nearfan2`-style). *(improves single-goal planning alone)*
4. **Rung 2 — topological graph** over real frames → subgoal layer. *(far goals)*
5. *(later, post-retrain)* WM-imagined subgoals; optional offline VLM-teacher channel.

**Next concrete action = steps 0–2** (the smallest thing that answers "does a learned distance go non-flat
on our latents?"); treat the sweep grade as the gate before anything downstream. Steps 0–3 are
**independent of the retrain**; the on-robot far-goal payoff arrives once the live-frame distribution gap
is also closed.

## References

- **QRL** — *Optimal Goal-Reaching RL via Quasimetric Learning*, Wang, Torralba, Isola, Zhang, ICML 2023.
  arXiv:2304.01203. Code: `github.com/quasimetric-learning/quasimetric-rl`.
- **IQE** — *Interval Quasimetric Embeddings*, Wang & Isola, 2022. arXiv:2211.15120. Drop-in heads:
  `github.com/quasimetric-learning/torch-quasimetric`.
- **Learnability** — *On the Learning and Learnability of Quasimetrics*, Wang & Isola, ICLR 2022.
  arXiv:2206.15478 (why a quasimetric-structured model class is required).
- **MRN** — *Metric Residual Networks for Sample-Efficient GCRL*, Liu et al., AAAI 2023. arXiv:2208.08133.
- Subgoal/graph: SoRB (arXiv:1906.05253), SGM (arXiv:2003.06417), ViNG (arXiv:2012.09812),
  ViNT (arXiv:2306.14846, code `github.com/robodhruv/visualnav-transformer`), Director (arXiv:2206.04114),
  LEXA (arXiv:2110.09514), Subgoal Diffuser (arXiv:2403.13085).

See also [[planning]], [[roadmap]], [[experiment-log]], [[open-questions]], [[overview]].
