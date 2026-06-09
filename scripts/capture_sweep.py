#!/usr/bin/env python3
"""
Phase-0 sweep capture (NanoNAV 6d) — GPU-FREE robot-side capture of ground-truth displacement sweeps.

Decouples CAPTURE (needs the robot, runs on the Mac/LAN like capture_goal.py) from SCORING
(scripts/dist_harness.py, runs anywhere) — unlike measure_dist_sweep.py, no WM/checkpoint/pod
is needed to capture, and every candidate ever built can be re-graded on the saved frames.

Writes a sweep dir for dist_harness.py:
    <out>/goal.png          full-res goal frame
    <out>/goal_model.png    256² letterboxed model view (exact engine letterbox)
    <out>/frames/NNN_<label>_full.png + _model.png
    <out>/manifest.csv      idx,label,arm,frame_full,frame_model,imagined,latent,note,t

Protocol (one session, ~50-60 placements; ~Gate A needs radial+lateral+yaw+noise at minimum
— see learned-distance-metric.md "Evaluation" + the sweep-design discussion):
  1. Place robot AT the goal pose. `g` -> snapshot goal.  Then `n noise 8` (same-pose noise burst).
  2. radial:  back straight away along the goal axis:  r10 r20 r30 r40 r50 r60
  3. lateral: from the goal pose, sidestep (re-place) keeping heading, OUT TO THE FAR BAND
              (Gate A grades lateral at 40-60 cm too): lat+10 lat+20 lat+40 lat+60 lat-10 ... lat-60
  4. yaw:     at the goal position rotate in place: yaw-30 yaw-20 yaw-10 yaw0(=r0) yaw+10 ...
              (or use --yaw-sweep to have the robot self-rotate and auto-capture)
  5. yaw-at-distance: at 40 cm out, sweep heading through goal-facing: yawd-20@r40 ... yawd+20@r40
  6. grid:    coarse polar field: g_r20_b+30, g_r40_b-30, ... (heading facing the goal)
  7. forks:   at a start pose, capture it (fork_a_start), then place at each ~one-chunk endpoint:
              fork_a_straight fork_a_pivl fork_a_pivr fork_a_arcl fork_a_arcr
              (sites: a = on-axis near, b = off-axis far, c = rug-centre/low-texture)
Labels are the ground truth — get them right; `?` prints this protocol, `ls` shows arm coverage.

Examples (Mac on the LAN, Pi host running):
  python scripts/capture_sweep.py --ip 10.0.0.125 --out results/sweep_nearfan2
  python scripts/capture_sweep.py --ip 10.0.0.125 --out results/sweep_x --goal goals/nearfan2/goal.png
  python scripts/capture_sweep.py --ip 10.0.0.125 --out results/sweep_x --yaw-sweep   # motorized yaw arm

SAFETY: capture is read-only (no motion). --yaw-sweep commands PURE ROTATION (confirm prompt,
clamped, Ctrl-C -> zero+disconnect). Re-probe the camera after any Pi-host restart (USB
enumeration swap — experiment-log 2026-06-09) BEFORE trusting a session.
"""

import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.append(_HERE)

import lekiwi_common as lk                      # noqa: E402
from capture_goal import get_top_frame, save_png  # noqa: E402
from sweep_common import ManifestWriter, letterbox_rgb, parse_label, imread_rgb  # noqa: E402

PROTOCOL = __doc__[__doc__.index("Protocol ("):__doc__.index("Examples (")]


def capture(robot, out, frames_dir, man, idx, label, note=""):
    frame = get_top_frame(robot)
    model = letterbox_rgb(frame)
    stem = f"{idx:03d}_{label}"
    fp_full = os.path.join("frames", f"{stem}_full.png")
    fp_model = os.path.join("frames", f"{stem}_model.png")
    save_png(frame, os.path.join(out, fp_full))
    save_png(model, os.path.join(out, fp_model))
    man.add(idx, label, frame_full=fp_full, frame_model=fp_model, note=note,
            t=f"{time.monotonic():.2f}")
    arm, params = parse_label(label)
    print(f"   #{idx:>3} [{label:>14}] arm={arm or 'UNPARSED!'} {params}")
    return frame


def coverage(man_path):
    import csv as _csv
    arms = {}
    if os.path.exists(man_path):
        with open(man_path, newline="") as f:
            for row in _csv.DictReader(f):
                arms[row["arm"] or "UNPARSED"] = arms.get(row["arm"] or "UNPARSED", 0) + 1
    need = ["radial", "lateral", "yaw", "noise"]
    print("[coverage] " + "  ".join(f"{a}:{arms.get(a, 0)}" for a in
                                    sorted(set(list(arms) + need))))
    missing = [a for a in need if not arms.get(a)]
    if missing:
        print(f"[coverage] Gate A still needs: {', '.join(missing)}")


def main():
    ap = argparse.ArgumentParser(description="GPU-free ground-truth displacement sweep capture")
    ap.add_argument("--ip", default="10.0.0.125")
    ap.add_argument("--id", default="lekiwi")
    ap.add_argument("--out", required=True, help="sweep dir to create/append")
    ap.add_argument("--goal", default=None, help="existing goal image (else snapshot one with 'g')")
    # motorized yaw arm (same contract as measure_dist_sweep --yaw-sweep, but GPU-free)
    ap.add_argument("--yaw-sweep", action="store_true", help="robot self-rotates in place + auto-captures")
    ap.add_argument("--yaw-theta", type=float, default=25.0, help="deg/s per step (+ = CCW)")
    ap.add_argument("--yaw-secs", type=float, default=0.4)
    ap.add_argument("--yaw-steps", type=int, default=13, help="captures across the sweep (odd -> center)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    frames_dir = os.path.join(args.out, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    man = ManifestWriter(args.out)
    idx = 0
    if os.path.exists(man.path):                       # append: continue idx numbering
        with open(man.path) as f:
            idx = max((int(line.split(",")[0]) for line in f.read().splitlines()[1:] if line), default=-1) + 1

    if args.goal:
        g = imread_rgb(args.goal)
        save_png(g, os.path.join(args.out, "goal.png"))
        save_png(letterbox_rgb(g), os.path.join(args.out, "goal_model.png"))
        print(f"[capture] goal copied from {args.goal}")

    print("[capture] connecting robot (read-only; `top` camera)...")
    LeKiwiClient, _ = lk.import_lekiwi()
    robot = LeKiwiClient(lk.make_client_config(args.ip, args.id, cameras=("top",)))
    robot.connect()
    print("[capture] connected. VERIFY the view is the overhead `top` camera (USB-swap trap).\n")

    # ===================== motorized yaw arm =====================
    if args.yaw_sweep:
        import signal
        act_keys = lk.feature_keys(robot, "action")
        base_keys = lk.classify_base_vel_keys(act_keys)
        if not base_keys.get("theta"):
            robot.disconnect(); sys.exit("[yaw] base theta key not mapped — cannot rotate.")
        hold = lk.capture_hold(robot, act_keys, base_keys)

        def _stop(*_):
            try:
                lk.stop(robot, act_keys, base_keys); robot.disconnect()
            except Exception:
                pass
            print("\n[yaw] ctrl-c: zeroed + disconnected."); sys.exit(130)
        signal.signal(signal.SIGINT, _stop)

        deg = args.yaw_theta * args.yaw_secs
        half = args.yaw_steps // 2
        print(f"[yaw] PURE ROTATION in place: {args.yaw_steps} captures, ~{deg:+.1f}°/step, "
              f"~±{half * deg:.0f}° about the start heading. Start AT the goal heading.")
        if input("[yaw] clear space, e-stop in reach — type 'go': ").strip().lower() != "go":
            robot.disconnect(); print("[yaw] aborted."); return

        def rotate(theta_deg):
            lk.stream_velocity(robot, act_keys, base_keys, 0.0, theta_deg, args.yaw_secs, hold_action=hold)
            lk.stop(robot, act_keys, base_keys); time.sleep(0.4)

        for _ in range(half):                                   # pre-rotate CW to one extreme
            rotate(-args.yaw_theta)
        for k in range(args.yaw_steps):
            yaw = (k - half) * deg
            lk.stop(robot, act_keys, base_keys); time.sleep(0.2)
            capture(robot, args.out, frames_dir, man, idx, f"yaw{yaw:+.0f}")
            idx += 1
            if k < args.yaw_steps - 1:
                rotate(args.yaw_theta)
        lk.stop(robot, act_keys, base_keys); robot.disconnect(); man.close()
        coverage(man.path)
        return

    # ===================== interactive labeled capture =====================
    print("Commands: <label>=capture | n <label> <count>=burst | g=snapshot goal | ?=protocol | ls=coverage | q=quit")
    try:
        while True:
            cmd = input("[capture] place robot, then command: ").strip()
            low = cmd.lower()
            if low == "q":
                break
            if low == "?":
                print(PROTOCOL); continue
            if low == "ls":
                coverage(man.path); continue
            if low == "g":
                frame = get_top_frame(robot)
                save_png(frame, os.path.join(args.out, "goal.png"))
                save_png(letterbox_rgb(frame), os.path.join(args.out, "goal_model.png"))
                print("[capture] goal snapshot saved (goal.png + goal_model.png)"); continue
            if not cmd:
                continue
            burst, label = 1, cmd
            if low.startswith("n "):
                parts = cmd.split()
                label = parts[1] if len(parts) > 1 else "noise"
                burst = int(parts[2]) if len(parts) > 2 else 8
            arm, _ = parse_label(label)
            if arm is None:
                if input(f"[capture] '{label}' doesn't parse to an arm — capture anyway? [y/N] ").lower() != "y":
                    continue
            for _ in range(burst):
                capture(robot, args.out, frames_dir, man, idx, label)
                idx += 1
                if burst > 1:
                    time.sleep(0.3)
    finally:
        try:
            robot.disconnect()
        except Exception:
            pass
        man.close()

    coverage(man.path)
    if not os.path.exists(os.path.join(args.out, "goal_model.png")):
        print("[capture] WARNING: no goal captured ('g') or provided (--goal) — harness needs one.")
    print(f"[capture] done -> {args.out}  (grade: python scripts/dist_harness.py --sweep {args.out})")


if __name__ == "__main__":
    main()
