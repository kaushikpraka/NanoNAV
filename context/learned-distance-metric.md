# Learned Distance Metric (Quasimetric) for Planning — Design

**Status: PLANNED (design settled 2026-06-09; not yet implemented).** The chosen replacement for the
raw SD-VAE latent-L2 planning objective. This is the durable design home for the work flagged as the
**#1 planner priority** in [[open-questions]] "Scoring function alternatives" and [[roadmap]] 6d.

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
   │  IQE quasimetric head → d(z_a,z_b) = IQE(φ(z_a), φ(z_b))            (asymmetric, Δ-ineq by構造)
```
Train an **ensemble (2–4 heads)**; use the **pessimistic (max)** distance for any edge/subgoal admission
— guards against single-head "wormhole" false shortcuts. VAE stays **frozen**; only φ + head train
(minutes–~1 h on the H100). **Reads latents, never decodes** → the known decode-blur and pixel-space
hallucination never touch the cost path. MRN (Metric Residual Network) is a simpler alternative head.

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
*monotone in true displacement, and non-flat at 40–60 cm where raw L2 plateaus?* Rank **QRL vs a
ViNG-style bucketed classifier vs raw-L2** on this before touching the robot. (Ties into the broader
"rigorous eval" thread — this is the absolute, physically-grounded metric the WM-relative 6a
`reached_ratio` lacked.)

## Subgoal layer (for far goals — after the metric)

Far goals sit outside one plan's basin (~10 cm reach). **Recommended: a topological graph over the
offline buffer** (SGM / ViNG-style) — and crucially over **real dataset frames as nodes**, so every
subgoal is **in-distribution by construction → dodges the hallucination** that WM-*imagined* subgoals
would trigger (our confirmed live-distribution-gap). Nodes = sparsified dataset latents; edges = pairs
with `d_learned < τ` where **τ ≈ one CEM reach (~3 chunks)**, admitted with the ensemble-pessimistic
distance; Dijkstra to the goal node; hand CEM the **first waypoint** as its target; recurse. This
dissolves the plateau — CEM never sees a far goal, only an in-basin subgoal.

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

## Sequencing

1. Pre-encode dataset → latents (once) + extend the sweep eval (lateral). *(offline, now)*
2. Train QRL metric (+ ViNG-classifier baseline); rank on sweep monotonicity. *(offline, now)*
3. Swap into CEM cost + termination; recalibrate reach-thresh. *(improves single-goal planning alone)*
4. Topological graph over real frames → subgoal layer. *(far goals)*
5. *(later, post-retrain)* WM-imagined subgoals; optional VLM-teacher channel.

Steps 1–3 are **independent of the retrain** and give a planner win on their own; the on-robot far-goal
payoff arrives once the live-frame distribution gap is also closed.

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
