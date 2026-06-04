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

# Safety envelope = the dataset action range (vx∈[0,0.10] m/s, |ω|≤0.34 rad/s ≈ 19.5°/s).
VX_MIN, VX_MAX = 0.0, 0.10
THETA_MAX_DEG = 19.5
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


def stream_velocity(robot, action_keys, base_keys, vx, theta_deg, duration):
    """Stream constant (vx m/s, theta_deg deg/s) for `duration` s at CMD_HZ, arm held. Caller stops."""
    xk, tk, yk = base_keys["x"], base_keys["theta"], base_keys["y"]
    base = {xk: vx, tk: theta_deg}
    if yk:
        base[yk] = 0.0
    period = 1.0 / CMD_HZ
    t_end = time.monotonic() + duration
    n = 0
    while time.monotonic() < t_end:
        obs = robot.get_observation()
        robot.send_action(build_action(obs, action_keys, base))
        n += 1
        time.sleep(period)
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
