# Data Collection

## Paradigm

The world model learns general environment dynamics, not task-specific behavior. Following NanoWM and DINO-WM, training data is exploratory interaction — diverse driving through the environment. The task (navigate to target object) enters only at inference via a goal image.

Task-specific demonstrations are counterproductive: CEM proposes 64 candidate actions per iteration (most bad), and the model must accurately predict what ALL of them produce to rank them. Training only on "good" trajectories means the model can't distinguish good candidates from bad ones.

## Controller Setup

PS5 controller via LeRobot teleop interface:
- Left stick Y-axis → v_x (forward/back)
- Right stick X-axis → ω (left/right yaw)
- No strafe binding (v_y = 0 by construction)
- Combined inputs used naturally (both sticks simultaneously)

## Environment

Subset of room with controlled conditions:
- Blinds closed, room lighting on (consistent illumination)
- Target objects in fixed positions across all episodes (bowl, box, tripod, etc.)
- Carpet floor with furniture landmarks (desk, tripod/easel, wall edges)
- Space approximately 2m × 2m

## Episode Plan

~50 episodes, robot placed in varied starting positions AND headings between episodes.

**Driving style:** generally random exploration with diverse speed/turn combinations. Not task-directed. Cover all areas of the space from many angles.

**Include deliberately:**
- Combined v_x + ω (arcing/curving) — natural driving, needed because CEM will propose it
- Pure forward runs at various speeds
- Pure rotations (both directions)
- Stationary pauses mid-episode (2-3s) — identity anchor for (Δx, Δθ) = (0, 0)
- A handful (~5 episodes) of deliberately slow driving — fills in low-Δx regime for near-goal approach
- Backing up

**Episode length:** 30-60 seconds each. Natural rhythm: explore from current position, reposition robot, repeat.

**Speed variation:** don't fight the joystick's natural dynamics. Analog sticks naturally produce varied speeds during acceleration, deceleration, and maneuvering. No need to explicitly target speed ranges.

## Dataset Sizing

At 30 Hz env rate with f=5 subsampling → 6 model frames/second.

50 episodes × 45s average = ~37 minutes → ~13,500 transitions.
For ~43K target: ~80 episodes at 60s, or 50 episodes at ~2 minutes.

Start with 50 episodes, train first checkpoint, scale if diagnostic shows data is the bottleneck.

Reference: PushT used ~450-900K transitions (sim). Point Maze/Wall used much less. For a single small room, 13-43K transitions is a reasonable starting range.

## What Gets Logged (per env timestep at 30 Hz)

Handled by lerobot-record:
- Overhead camera image
- Commanded base velocity (v_x, ω) — verify no v_y leaking through
- Timestamp (synchronized)
- Wheel odometry / encoder readings (ground-truth check)
- Arm joint positions (parked, for verification)

## Dataset Builder Output (offline, per model-frame transition)

- SD-VAE encoded image at frame t → z_t [4, 32, 32]
- (Δx, Δθ) integrated from f velocity commands in body frame → a_t [2]
- Global pose (for diagnostics/visualization, not model input)
- Goal image (sampled for planning evaluation)

## Pre-Collection Checklist

- [ ] Park arm in consistent configuration, save joint positions
- [ ] Verify lerobot-record captures camera + velocities at expected rate
- [ ] Verify no v_y values leaking through controller mapping
- [ ] Fix target object positions, photograph arrangement
- [ ] Close blinds, set room lighting
- [ ] After first 2-3 episodes: sanity-check logs before continuing
- [ ] Take goal images from robot's perspective at target object positions
