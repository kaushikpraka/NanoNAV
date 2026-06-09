# Learned Distance Metric for Planning — Design

**Status: PLANNED (design settled 2026-06-09; REVISED same day after a literature stress-test; not yet
implemented).** The chosen replacement for the raw SD-VAE latent-L2 planning objective. This is the
durable design home for the work flagged as the **#1 planner priority** in [[open-questions]] "Scoring
function alternatives" and [[roadmap]] 6d.
**Build order (revised 2026-06-09 research session):** Phase 0 = distance-agnostic sweep harness +
**zero-training frozen-embedding comparator arms** (DINOv2-patch, V-JEPA 2.1, VIP) → rung 0 =
temporal-MLP baseline (+ optional patch-DINO distillation) → rung 1 = **contrastive / MC-fitted
quasimetric head (NOT QRL dual-ascent — demoted on evidence, see "Rung-1 head, revised")** → rung 2 =
topological graph. The metric is a hard prerequisite for the graph. **Next concrete action = Phase 0
(latent cache + harness refactor + frozen-embedding arms); Gate A/Gate B in "Sequencing" are the
decision points.**

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

## Rung-1 head — REVISED 2026-06-09 (research session): contrastive / MC-quasimetric, NOT QRL dual-ascent

The QRL exposition above stands as the *conceptual* frame (local caps + push-apart + triangle
inequality = shortest-path), but a literature stress-test demoted **QRL's min-max / dual-ascent
training** from "fragile pitfall (#2)" to "likely the wrong rung 1":

- **OGBench** (the standard offline-GCRL benchmark, arXiv:2410.20092): QRL/IQE dominates *state-based*
  mazes but scores **~0% on every visual task**, and struggles on stitching from suboptimal data —
  exactly our regime (image obs, loopy teleop).
- A string of 2025 follow-ups exists specifically to fix the dual-ascent instability — **TMD**
  (arXiv:2509.20478), **Eikonal-QRL** (arXiv:2512.12046), **ProQ** (arXiv:2506.18847), **MQL**
  (arXiv:2511.07730) — i.e. the instability is **structural, not a tuning issue**. Essentially no
  published real-robot or image-based-navigation use of QRL was found.
- **The property QRL is bought for — cross-trajectory stitching via the triangle inequality — is
  exactly what rung 2's graph + Dijkstra provides explicitly anyway.** Paying QRL's fragility for
  something the next rung gives for free is a bad trade.

**Revised escalation if rung 0 plateaus (in order of preference):**
1. **Contrastive RL — InfoNCE on temporal proximity** (CRL, arXiv:2206.07568): on OGBench's visual
   mazes CRL gets ~94% where QRL gets ~0%. Known weakness = no stitching — covered by the graph.
   **Stable Contrastive RL** (arXiv:2306.03346) has the real-robot-image recipe (layernorm, cold init,
   ~2× success from those details alone).
2. **Contrastive Successor Features / temporal distances** (arXiv:2406.17098): same InfoNCE machinery,
   yields a proper triangle-inequality distance even under stochasticity. Or **MQL** (arXiv:2511.07730):
   fits a quasimetric head to Monte-Carlo returns — **no adversarial loop**, works on visual obs,
   demonstrated stitching on real-robot data. Both = "QRL's benefits without QRL's training."
3. **IQE/MRN survive only as head *parameterizations*** (MC- or contrastive-fitted, e.g. MQL-style) if
   the rung-0 symmetric-vs-asymmetric ablation shows directionality actually matters in our room.

Same φ, same latent cache, same sweep eval — the shared-scaffold argument survives the head swap intact.

## Architecture

```
frozen SD-VAE → z [4,32,32]
   │  φ : small CNN over the [4,32,32] grid → flatten → MLP → e ∈ ℝ²⁵⁶   (trainable)
   │  distance head → d(z_a,z_b) = head(φ(z_a), φ(z_b))
```
*(Head choice revised 2026-06-09: rung 0 = plain MLP/‖·‖ heads; the escalation head is contrastive /
MC-fitted — IQE survives only as a parameterization option, never trained by dual-ascent. The φ/head
split and everything below is unchanged. See "Rung-1 head — REVISED".)*
Train an **ensemble (2–4 heads)**; use the **pessimistic (max)** distance for any edge/subgoal admission
— guards against single-head "wormhole" false shortcuts. VAE stays **frozen**; only φ + head train
(minutes–~1 h on the H100). **Reads latents, never decodes** → the known decode-blur and pixel-space
hallucination never touch the cost path. MRN (Metric Residual Network) is a simpler alternative head.

**Encoder vs head — what's actually trained.** φ (the CNN) is a plain *encoder*: it maps a latent to an
embedding and is **not itself a quasimetric**. The quasimetric properties (asymmetry, triangle inequality,
`d(x,x)=0`) come entirely from the **IQE head** on top, which holds for *any* encoder weights. The two
train **jointly end-to-end** under whichever loss the chosen rung uses — φ carries all the capacity (learns *which* latent
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

## Evaluation (already half-built) — REVISED: distance-agnostic harness + zero-training comparator arms

`scripts/measure_dist_sweep.py` gives **hand-placed ground-truth cm-displacement → latent** (radial +
`--yaw-sweep`; add a **lateral** sweep — it grades wormholes *and* loose welds at once, pitfall #9).
**Refactor it so a candidate distance is just a callable `d(image_a, image_b) → float`** (internally it
may encode to SD-VAE latents, run DINOv2, whatever — the harness doesn't care). Output per candidate:
d-vs-displacement curves per axis + same-pose noise floor σ + two gate numbers: **Spearman ρ out to
60 cm** and **slope-to-σ ratio in the 40–60 cm band** (where raw L2's SNR collapses).

**Zero-training comparator arms (run on day one, before training anything — all on real sweep captures,
no decode, no training):**
- raw latent-L2 (current baseline) + pixel-L1
- **DINOv2 patch-feature distance** (mean patch MSE *and* per-patch cosine; ViT-S/14 + ViT-B/14). The
  serious candidate: **DINO-WM** (arXiv:2411.04983) plans CEM/MPC with exactly this cost on **top-down
  navigation with the agent body in frame** (Maze 0.98 / Wall 0.96), and its ablation shows the signal
  is in the *patch grid*, not pooled features (**patch 0.96 vs CLS 0.58** on Wall — pooled/CLS is
  contraindicated; corroborated by arXiv:2507.01667: global frozen embeddings carry ~no relative-pose
  info, patch-level does). V-JEPA-2-AC (arXiv:2506.09985) plans real arms on frozen-latent L1 with a
  "locally convex" energy. **Caveat: nobody has measured monotonicity at 0–60 cm robot scale** — VPR
  evidence (AnyLoc, arXiv:2308.00688) is 10–25 m *retrieval*, and DINOv2's rotation-invariance can blunt
  heading signal (ViT-VS, arXiv:2503.04545: ±90°/180° convergence failures) — so the sweep measures
  what no paper has. Mechanistic prior: SD-VAE cells encode local appearance stats (generic carpet/wall
  cells dominate the sum → measured flatness); DINOv2 patches encode *semantic identity* ("part of the
  fan" vs "part of the TV"), so distant poses putting different landmarks at different grid positions
  register strongly. Expected dead zone: rug-centre poses where only carpet fills the frame.
- **V-JEPA 2.1 image-tokenizer token distance** (nanowm already wires `latent_codec=vjepa2_1`). Whichever
  feature space goes non-flat is simultaneously the best metric candidate AND the strongest candidate WM
  target for the retrain (see "If the retrain switches latent space") — one measurement, two decisions.
- **VIP embedding L2** (arXiv:2210.00030) — included because it's ~20 lines, but **expected to lose**:
  double-OOD for us (viewpoint + navigation-vs-manipulation), independent out-of-domain measurements are
  negative (LIV needs in-domain fine-tuning for monotone rewards; GVL arXiv:2411.04549 finds
  embedding-distance values near-random OOD; BiMI arXiv:2409.15922 finds false-positive-dominated
  similarity rewards in navigation; R3M collapses to 0.34 on DINO-WM's Wall), and it's a global pooled
  embedding (the CLS failure mode).

**Plus the WM-imagined arm** (the validate-first decision above) and a **distance-field visualization**:
d from *every* cached chunk-latent to a goal frame — a healthy field is one basin around the goal's
region; a spurious second basin = wormhole (cross-check with ensemble disagreement).

**Gate A (numeric, not vibes):** a candidate passes if **ρ > 0.9 vs true displacement out to 60 cm on
radial AND lateral**, **slope > ~3× same-pose σ at 40–60 cm**, and the **yaw basin stays sharp** (raw L2
already has that — don't regress it). Rank rung-0, the frozen arms, and raw-L2 on this before touching
the robot. (Ties into the broader "rigorous eval" thread — this is the absolute, physically-grounded
metric the WM-relative 6a `reached_ratio` lacked.)

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
   **⇒ RESOLVED BY DESIGN (2026-06-09): the dual-ascent head is dropped from the build order entirely**
   (OGBench: QRL ~0% on all visual tasks; see "Rung-1 head — REVISED"). The escalation is contrastive /
   MC-fitted instead; this pitfall stays recorded as the reason why.
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

## Build order — simple temporal-MLP baseline BEFORE any quasimetric

**Decision (2026-06-09): build a simple temporal-distance MLP first as a baseline, then escalate (now:
to a contrastive/MC-quasimetric head, see "Rung-1 head — REVISED") only where it plateaus.** Not "build
both at once," and not "dive straight into the fancy head." Rung 0 is the **most field-proven piece of
the whole plan** — ViNG ran real robots on exactly this recipe (temporal-distance head + negatives +
graph), and GNM/ViNT/NoMaD kept the temporal-distance head at much larger scale. Rationale:

- **~80% shared scaffold, not throwaway.** Both need the same latent cache, pair sampler, **CNN φ**, and
  sweep eval; only the head + loss differ (MLP+MSE/Huber vs the contrastive / MC-quasimetric escalation).
  The baseline stands up everything the escalation needs, **plus a permanent comparator** for the eval.
- **The room is forgiving enough that the baseline has a real shot** (landmark-rich perimeter → "which
  region / heading" is easy; see "Environment"). If it passes the sweeps, we may not need a learned-RL
  head at all for v1.
- **De-risks the escalation:** the learned-head training is the fragile part; debugging "broken or just
  hard?" is far easier with a working baseline on the same harness.

**The variable that actually decides flat-vs-not is the cross-trajectory negatives, NOT the head.** A
within-trajectory-only temporal regressor *will* be flat across regions (no labels there). The cheap
decisive upgrade is the **ViNG trick: label cross-trajectory pairs as max-distance** (or bucket
near/medium/far). Build the baseline *with* that, and ablate with/without it. (ViNG's own reported
lessons name negative mining as one of its three key insights.)

**Rung-0 additions (2026-06-09 research session):**
- **Two heads as a built-in ablation:** symmetric `‖e_a−e_b‖` vs asymmetric `MLP(concat(e_a,e_b))` —
  the comparison answers *empirically* whether asymmetry matters in our room before paying for any
  quasimetric machinery.
- **Ensemble of 3–4 from day one** (minutes per model at this scale) — disagreement is the wormhole
  detector.
- **Optional distillation variant:** if Gate A shows DINOv2-patch is the signal that goes non-flat,
  train φ to *match patch-DINO distances* on real-frame pairs. This imports DINO's far-field structure
  into a head that **reads WM-space SD-VAE latents directly** — solving the wiring problem that the CEM
  cost must score *imagined* latents (a raw DINO cost would force decode→DINOv2 through blurry/
  hallucinated pixels, which stays rejected).
- **Loopy-data caveat:** within-episode time gaps *over*-estimate true distance on revisits; rung 0
  lives with it (like ViNG). Cheap robustification if it bites: treat labels as **upper bounds** (hinge:
  penalize only predictions above k) — note that drifts toward the QRL relaxation, so save it for the
  escalation decision.

**Expected outcome (still informative either way):** likely **good in-region / near-goal** (landmarks
carry it), **flat or noisy cross-region + on the low-texture centre** (time≠space + multimodality bite).
That *localizes* where the rung-1 head is actually needed (the far-field / cross-region regime the graph
edges depend on) and gives a baseline number to beat.

**Rungs: (0) temporal-MLP + cross-traj negatives → (1) contrastive / MC-quasimetric head → (2)
topological graph.** The metric (rung 0/1) is a **hard prerequisite** for the graph — its edges *are*
`d_learned` — and it delivers value on its own (in-basin CEM), so it is strictly first; the graph cannot
precede it.

## Complementary tracks (2026-06-09 research session) — cheap, parallel, underweighted before

- **GCBC proposal prior for CEM** (~days). Hindsight relabeling turns the 45K frames into millions of
  `(state, future-goal, action)` tuples; a small goal-conditioned BC policy (GCSL arXiv:1912.06088, RvS
  arXiv:2112.10751) **warm-starts CEM's sampling distribution** rather than replacing it. Attacks both
  top problems at once: it has gradient where L2 is flat (imitates "what did I do when heading toward
  views like this"), and it biases CEM toward **in-distribution action sequences**, keeping WM rollouts
  where they hallucinate less. Can't stitch — the graph covers that.
- **SLAM pose oracle for evaluation** (~days; addresses pitfall #8, the verification gap). Run
  MASt3R-SLAM (arXiv:2412.12392, robust to uncalibrated wide-angle cameras) offline over the existing
  45K frames → approximate poses for **every dataset frame**. Uses: grade `d_learned` against geodesic
  pose distance *everywhere* (not just the hand-placed sweeps), detect wormholes at graph-build time
  (a wormhole = one long edge cutting across the pose plot), and position graph nodes for debugging.
  **Explicitly NOT the planner** (would defeat the project's purpose) — eval-only.
- **Recollection co-design** (the retrain is happening anyway for the live-frame gap — make one session
  serve three consumers): dense coverage of nearhamper-class perimeter regions (WM), **loop closures
  through the rug centre + both approach directions on key paths** (metric stitching + honest
  asymmetry), slow deliberate goal approaches (near-goal control / bang-bang fix), exposure/WB lock +
  the udev camera-pin fix first.

## If the retrain switches latent space (Finding #4 reinterpreted — 2026-06-09 research session)

nanowm's Finding #4 ("DINO/V-JEPA fail at action conditioning", action-RMS → 0.002, 0% planning) is
evidence about **diffusion-forcing ⊗ semantic latents, not about semantic latents per se**. Every
published success with frozen semantic features uses a **deterministic teacher-forced regression
predictor** — DINO-WM (MSE, actions concatenated per patch token), V-JEPA 2-AC (L1 + 2-step rollout
loss), DINO-world (arXiv:2507.19468, smooth-L1) — and nobody has made plain diffusion-forcing into
semantic latents work. Mechanism: semantic features are temporally smooth → the denoiser scores well by
copying context → the action branch starves; SD-VAE's high per-step variance (texture flicker) is what
kept our action branch alive. Implications for the retrain decision point:
- **Semantic-latent WM = codec swap + objective swap together** (regression/x0-prediction predictor,
  actions per token); re-running `latent_codec=webdino|vjepa2_1` under diffusion-forcing would just
  reproduce Finding #4. V-JEPA 2.1's nav WM (arXiv:2603.14482) shows the diffusion-compatible variant:
  a DiT predicting **clean representations (x0), not noise**.
- **Why care:** (i) action-relevant signal is *more* recoverable in semantic spaces under regression
  (arXiv:2605.06388: IDM action-correlation 0.83 V-JEPA-2.1 vs 0.51 VAE; CEM action-recovery 0.42 vs
  0.61); (ii) **OOD behavior** — regression predictors degrade to averaged-but-task-faithful predictions
  (benign for an L1-to-goal cost), where generative diffusion WMs **snap to vivid plausible-wrong
  training scenes** — exactly our `nearhamper` hallucination (NWM arXiv:2412.03572 reports the same mode
  collapse at 1B params); (iii) a frozen DINOv2/V-JEPA encoder has seen millions of rooms — live frames
  can't land off-manifold *for the encoder* (the predictor stays coverage-limited). Small-data
  precedents: DINO-WM (1–2K trajectories/env), PLDM (arXiv:2502.14819, ~thousands of transitions,
  top-down nav, ~80%).
- **This stack transfers:** φ retargets to the new latent space, the pair sampler / sweep harness /
  graph machinery are latent-agnostic. Gate A's frozen-arm ranking doubles as the codec-selection
  signal. Plan a small separate decoder for viz (DINO-WM trained one) — SD-VAE's free decodability is
  lost.

## Sequencing — phased plan (revised 2026-06-09)

**Phase 0 — measuring stick + free candidates** (~1–2 days, mostly Mac; decides everything downstream):
**✅ CODE BUILT 2026-06-09 (all five tools; harness smoke-tested end-to-end on a synthetic sweep —
PASS and FAIL paths both verified). Remaining: run on real data (capture session + pod encodes).**
0a. `scripts/build_latent_cache.py` ✅ — chunk-boundary frames (direct parquet+mp4 read) → exact
    `_preprocess` letterbox → checkpoint codec encode (**deterministic posterior mode**, vs the
    engine's sampling — recorded in meta.json) → `latents.npy` + `index.csv` (RGB pointers) +
    `meta.json` (every convention pinned). `--hf-vae` checkpoint-free fallback. *(needs pod+dataset)*
0b. Harness ✅ — split into `scripts/capture_sweep.py` (**GPU-free** robot-side capture: label
    grammar carries ground-truth pose; arms radial/lateral/yaw/yawd/grid/fork/noise; motorized yaw
    mode; protocol + coverage built in) + `scripts/dist_harness.py` (distance-agnostic grader:
    per-arm ρ / far-slope-per-10cm/σ / yaw-basin / fork rankings → **Gate A verdict**, metrics.csv +
    gate_report.md + overlay plots) + `scripts/sweep_common.py` (THE single letterbox/normalize
    implementation — the 6b pixel-range-bug class is closed by construction; legacy
    `measure_dist_sweep.py` dirs auto-ingest). *(capture needs the robot; grading runs anywhere)*
0c. `scripts/dist_candidates.py` ✅ — pixel_l1, **sdvae_l2** (= the current objective, diffusers
    weights, engine-parity flat-L2), **dinov2_mse / dinov2_cos** (patch tokens, vits14/vitb14),
    vip_l2 + vjepa21 (optional deps, graceful skip). Rung-0/1 learned heads plug into the same
    registry later. *(sdvae+dinov2 verified running locally on CPU)*
0d. `scripts/wm_imagined_arm.py` ✅ — rolls the WM from radial captures with known straight chunks →
    combined sweep dir where imagined rows carry **raw WM latents** (sdvae_l2 scores them directly,
    no decode roundtrip — `feature_is_wm_latent`) + decoded frames for image-space candidates;
    harness overlays imagined vs clean curves. *(needs pod+ckpt; pass action stats explicitly)*
    **→ Gate A** (see Evaluation). Either way: GO/NO-GO rig + ranked comparator set exist.

**Phase 1 — rung-0 learned distance** (~2–4 days, brief pod GPU):
1a. Pair sampler (within-episode log-uniform k ≤ k_max≈30–50 + cross-episode negatives @ k_max;
    with/without-negatives ablation built in).
1b. CNN φ → 256-d; symmetric + asymmetric heads; Huber; ensemble 3–4; optional patch-DINO distillation
    variant (per Gate A).
1c. Grade on the harness vs all Phase-0 arms + distance-field visualization + ensemble disagreement.
    **→ Gate B (the GO/NO-GO):** monotone + non-flat where raw-L2 plateaus → Phase 2. Good in-region but
    flat cross-region → escalate the head (contrastive/CSF/MQL — see "Rung-1 head, REVISED") on the same
    scaffold, re-grade.

**Phase 2 — swap into the planner** (~1–2 days + robot session):
2a. Wire the winner into `lekiwi_engine`: cost on generated latents (try **+1-weighted** vs
    endpoint-only — with a real steps-to-go metric the +1 chunk is least WM-degraded), termination on
    `d(z0,zg)`, recalibrate `--reach-thresh` to chunk units.
2b. On-robot A/B **on well-covered goals only** (`nearfan2`-class, per the OOD guidance): raw-L2 vs
    learned, progressively farther starts. Success signal: the rotate-to-face maneuver L2 under-credited
    now shows monotone d-descent; catchment radius grows.

**Phase 3 — rung 2, topological graph** (after Gate B, ~3–5 days): per the "Subgoal layer" section —
stride-5 node subsample merged at `d<1` (~300–600 nodes), **temporal edges free/trusted vs metric
shortcut edges at pessimistic `d<τ≈3`**, Dijkstra → CEM gets the waypoint 2–3 edges ahead, edge deletion
on repeated failed hops, goal-insertion-as-OOD-detector.

**Parallel track:** GCBC proposal prior; SLAM pose oracle; recollection co-design + retrain (with the
latent-space decision point above). *(later, post-retrain)* WM-imagined subgoals; optional offline
VLM-teacher channel.

**Order of attack: Phase 0 items first** — the harness + frozen-embedding arms are the highest
information-per-hour in the plan; everything else branches on their result. Phases 0–3 are
**independent of the retrain**; the on-robot far-goal payoff arrives once the live-frame distribution
gap is also closed.

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

**Added 2026-06-09 (research session):**
- **OGBench** — offline-GCRL benchmark, Park et al. arXiv:2410.20092 (QRL ~0% on visual tasks — the
  rung-1 revision evidence).
- Contrastive line: **CRL** arXiv:2206.07568; **Stable Contrastive RL** arXiv:2306.03346 (real-robot
  image recipe); **Contrastive Successor Features / temporal distances** arXiv:2406.17098; **MQL**
  arXiv:2511.07730 (MC-fitted quasimetric, no min-max); TMD arXiv:2509.20478; Eikonal-QRL
  arXiv:2512.12046; ProQ arXiv:2506.18847; HIQL arXiv:2307.11949.
- Frozen-embedding cost evidence: **DINO-WM** arXiv:2411.04983 (patch-MSE CEM cost, top-down nav,
  patch-vs-CLS ablation); hierarchical follow-up HWM arXiv:2604.03208; **V-JEPA 2 / 2-AC**
  arXiv:2506.09985 (frozen-latent L1 CEM on real arms; camera-pose sensitivity caveat); **V-JEPA 2.1**
  arXiv:2603.14482 (image tokenizer, x0-prediction nav WM); image-goal-nav encoder study
  arXiv:2507.01667 (global embeddings ≈ no relative pose); ViT-VS arXiv:2503.04545 (rotation-invariance
  failure); AnyLoc arXiv:2308.00688 (VPR ≠ metric monotonicity); **VIP** arXiv:2210.00030 + negative
  OOD evidence (LIV arXiv:2306.00958, GVL arXiv:2411.04549, BiMI arXiv:2409.15922).
- Semantic-latent WM / Finding-#4 reinterpretation: **Reconstruction or Semantics?** arXiv:2605.06388
  (semantic > VAE latents on action-relevant metrics under regression); DINO-world arXiv:2507.19468;
  **PLDM** arXiv:2502.14819 (JEPA latent dynamics + MPC nav at thousands-of-transitions scale);
  NWM arXiv:2412.03572 (LPIPS CEM cost; OOD mode collapse at 1B); GameNGen arXiv:2408.14837
  (diffusion action-conditioning pathology context).
- Complementary tracks: **GCSL** arXiv:1912.06088, **RvS** arXiv:2112.10751 (GCBC proposal prior);
  **MASt3R-SLAM** arXiv:2412.12392 (pose oracle).

See also [[planning]], [[roadmap]], [[experiment-log]], [[open-questions]], [[overview]].
