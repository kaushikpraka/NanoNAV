"""
Shared LeKiwi client helpers for NanoNAV Stage 6b.

Encodes the (Δx, Δθ) → velocity contract PINNED IN 6b.0 (see context/planning.md
"6b — RESULTS (6b.0)" / experiment-log):

    x.vel     = Δx / (f·Δt)                      # m/s,   +x = forward
    theta.vel = (Δθ / (f·Δt)) · (180/π)          # deg/s, +theta = LEFT/CCW  (matches dataset +ω=CCW)
    f·Δt = 10/30 = 0.333 s ;  y.vel = 0 ;  arm .pos held

Pure lerobot (no nanowm / GPU). Imported by lekiwi_replay.py (6b.1) and, later, the
live controller (6b.3). The 6b.0 smoke script keeps its own inline copies for now.
"""

import time
from math import pi

import numpy as np

CHUNK_DT = 10.0 / 30.0           # f·Δt = 0.333 s  (Run-002 chunk, f=10 @ 30 Hz)
RAD2DEG = 180.0 / pi
CMD_HZ = 20.0                    # stream rate while a command is active (host watchdog)

# Safety envelope = the TRUE dataset action range, measured across all 50 episodes
# (scripts run 2026-06-04): vx∈[0,0.10] m/s, theta.vel∈[-30,+30]°/s (=±0.5236 rad/s = ±π/6,
# a clean LeKiwi base cap). NOTE: the earlier "|ω|≤0.34 rad/s ≈19.5°/s" figure was wrong.
VX_MIN, VX_MAX = 0.0, 0.10
THETA_MAX_DEG = 30.0
# Low-speed rotation deadband observed in 6b.0: 0.3°/s = no motion, 12–15°/s engages.
DEADBAND_WARN_DEG = 4.0


def import_lekiwi():
    """Version-dependent import (matches the 6b.0-verified path)."""
    from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig  # type: ignore
    return LeKiwiClient, LeKiwiClientConfig
    # fallback some versions use: from lerobot.common.robots.lekiwi import ...


def feature_keys(robot, which):
    for attr in (f"{which}_features", f"{which}_feature", "features"):
        feats = getattr(robot, attr, None)
        if isinstance(feats, dict) and feats:
            return list(feats.keys())
    return None


def classify_base_vel_keys(action_keys):
    out = {"x": None, "y": None, "theta": None}
    for k in action_keys:
        kl = k.lower()
        if ".vel" not in kl:
            continue
        if "theta" in kl or "rot" in kl or "yaw" in kl:
            out["theta"] = k
        elif kl.startswith("x") or ".x" in kl:
            out["x"] = k
        elif kl.startswith("y") or ".y" in kl:
            out["y"] = k
    return out


def build_action(obs, action_keys, base_overrides):
    """Hold every arm `.pos` at its observed value; set base `.vel` from overrides; zero the rest."""
    action = {}
    for k in action_keys:
        if k in base_overrides:
            action[k] = float(base_overrides[k])
        elif k in obs:
            action[k] = float(np.asarray(obs[k]).reshape(-1)[0])
        else:
            action[k] = 0.0
    return action


def chunk_to_velocity(dx_m, dth_rad):
    """One chunk's (Δx m, Δθ rad) → (x.vel m/s, theta.vel deg/s) per the 6b.0 contract."""
    vx = dx_m / CHUNK_DT
    theta_deg = (dth_rad / CHUNK_DT) * RAD2DEG
    return vx, theta_deg


def clamp_velocity(vx, theta_deg):
    return (float(np.clip(vx, VX_MIN, VX_MAX)),
            float(np.clip(theta_deg, -THETA_MAX_DEG, THETA_MAX_DEG)))


def capture_hold(robot, action_keys, base_keys):
    """
    The constant non-base part of every action: arm `.pos` held at the observed values, base vels
    zeroed. Capture ONCE per run (the arm doesn't move) so the hot send-loop never calls
    get_observation — keeping the command cadence steady and the chunk duration precise.
    """
    zero_base = {k: 0.0 for k in base_keys.values() if k}
    return build_action(robot.get_observation(), action_keys, zero_base)


def stream_velocity(robot, action_keys, base_keys, vx, theta_deg, duration, hold_action=None):
    """
    Hold a constant (vx m/s, theta_deg deg/s) for EXACTLY `duration` s, refreshing at CMD_HZ to keep
    the host watchdog alive. Sends a PRECOMPUTED action (no get_observation in the loop, so cadence is
    steady) and paces against a fixed deadline with a final partial sleep — so the chunk lasts `duration`
    to within ~one send latency, not overshot by a whole get_obs+sleep iteration. Caller stops afterwards.
    """
    xk, tk, yk = base_keys["x"], base_keys["theta"], base_keys["y"]
    if hold_action is None:
        hold_action = capture_hold(robot, action_keys, base_keys)
    action = dict(hold_action)
    action[xk] = float(vx)
    action[tk] = float(theta_deg)
    if yk:
        action[yk] = 0.0

    period = 1.0 / CMD_HZ
    t0 = time.monotonic()
    t_end = t0 + duration
    n = 0
    next_send = t0
    while True:
        robot.send_action(action)
        n += 1
        next_send += period
        if next_send >= t_end:
            rem = t_end - time.monotonic()      # final partial sleep to land exactly on t_end
            if rem > 0:
                time.sleep(rem)
            break
        slack = next_send - time.monotonic()
        if slack > 0:
            time.sleep(slack)
    return n


def stop(robot, action_keys, base_keys):
    xk, tk, yk = base_keys["x"], base_keys["theta"], base_keys["y"]
    base = {xk: 0.0, tk: 0.0}
    if yk:
        base[yk] = 0.0
    robot.send_action(build_action(robot.get_observation(), action_keys, base))


def dead_reckon(steps):
    """
    steps: iterable of (vx m/s, omega rad/s, dt s). Integrate a unicycle from the origin
    (body x = forward) and return world-frame (xs, ys, thetas) arrays incl. the start point.
    """
    x = y = th = 0.0
    xs, ys, ths = [0.0], [0.0], [0.0]
    for vx, om, dt in steps:
        sub = max(1, int(round(dt / 0.01)))
        h = dt / sub
        for _ in range(sub):
            x += vx * h * np.cos(th)
            y += vx * h * np.sin(th)
            th += om * h
        xs.append(x); ys.append(y); ths.append(th)
    return np.array(xs), np.array(ys), np.array(ths)
