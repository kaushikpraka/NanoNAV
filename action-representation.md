# Action Representation

## Training Sample Structure

Each sample spans one model-frame transition (f=5 env steps, ~167ms at 30 Hz):

| Component | Description | Shape |
|---|---|---|
| z_t (input) | SD-VAE encoding of camera frame at chunk start | [4, 32, 32] |
| a_t (input) | Body-frame pose delta (Δx, Δθ), normalized | [2] |
| z_{t+1} (target) | SD-VAE encoding of camera frame f steps later, noised per diffusion schedule | [4, 32, 32] |

The model learns: given this view and this displacement, predict the next view.

## Body-Frame Pose Delta Construction

At each chunk start, plant a local coordinate system on the robot (origin = current position, x-axis = forward, θ = 0). Integrate f=5 logged velocities offline:

```python
x, y, θ = 0.0, 0.0, 0.0
for i in range(f):
    x += v_x[i] * dt * cos(θ)
    y += v_x[i] * dt * sin(θ)
    θ += ω[i] * dt
a_t = (Δx, Δθ) = (x, θ)  # y is dropped
```

This is the SE(2) relative transform g_t⁻¹ · g_{t+1} with lateral component discarded.

## Why Body-Frame

Heading invariance: "drive forward 5cm" is always (0.05, 0.0) regardless of absolute heading. The visual effect depends on relative motion, not absolute pose. Composable across MPC replans — each replan anchors a fresh body frame.

World-frame deltas were considered but rejected: same physical motion produces different vectors at different headings, forcing the model to learn heading-dependent dynamics unnecessarily.

## Why Integrated Displacement (Not Raw Velocity)

Raw velocity is constant during cruise — identical action across consecutive samples while visuals change. The action branch can decorrelate (Finding #4 pathology). Integrated displacement stays nonzero whenever the robot moves.

Velocity-delta (acceleration) is worse: approximately zero during constant-velocity cruising, which is most of navigation.

## Why Δy Is Dropped

When v_x and ω are both nonzero, the robot traces a circular arc. The endpoint has lateral drift (Δy) in the body frame. At typical speeds and chunk durations (167ms), Δy ≈ Δx · sin(Δθ/2) ≈ 1-2mm — a second-order effect.

Visual significance: at 40cm camera height, Δy = 2mm produces ~0.3° angular shift. Δx and Δθ produce 6-8° each. The kept signal is ~20-30× larger than the dropped component.

Dropping Δy reduces CEM search from H×3 = 9D to H×2 = 6D for H=3.

Breaks down only with very aggressive turns (ω > ~2 rad/s) within a single chunk. MPC replans from real observations each step, so error never accumulates.

## Normalization

Per-channel zero-mean unit-variance across the dataset. Δx (~0-0.05m) and Δθ (~0-0.2 rad) live on different scales; without normalization, CEM sampling and the action embedding implicitly overweight one.

## Execution Reconversion

CEM outputs (Δx, Δθ). Convert to velocity: v_x = Δx / (f·Δt), ω = Δθ / (f·Δt). Send to base for one chunk duration (167ms). Robot traces the actual arc (including Δy). MPC replans from the real endpoint.

## Camera Geometry

Elevated third-person view (~55° from horizontal), mounted on robot arm structure.

**Depth zones in frame:**
- Robot body (bottom ~25%): fixed in frame — ego-motion reference anchor
- Near floor (20-50cm): fast uniform slide from Δx
- Mid-range objects (40cm-1m): moderate depth-dependent parallax
- Far walls (1-2m): slow creep from Δx, dominant lateral sweep from Δθ

The fixed robot body strengthens action conditioning: the model sees "everything above the robot moved, the robot didn't, so the action must be nonzero." Δx and Δθ produce geometrically independent flow patterns (depth-dependent parallax vs uniform angular sweep).

## Alternatives Considered

| Option | Dim | Verdict |
|---|---|---|
| Raw velocity (v_x, ω) | 2 | Rejected: constant during cruise |
| Body-frame delta (Δx, Δθ) | 2 | **Chosen** |
| World-frame delta (Δx_w, Δy_w, Δθ) | 3 | Rejected: breaks heading invariance |
| Absolute pose as auxiliary input | 3 | Fallback if Table 5/6 diagnostic fails |
| Multi-frame context | — | Risk: makes action branch MORE likely to atrophy |
