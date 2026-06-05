#!/usr/bin/env python3
"""
6b.4 — LeKiwi goal-image capture (NanoNAV Stage 6b).

Snapshot the robot's `top` camera to a goal file for the closed-loop MPC controller
(`scripts/lekiwi_mpc.py --planner wm --goal <file>`). lerobot-only, NO GPU — runs wherever
the robot is reachable (today: the Mac on the LAN with the Pi at 10.0.0.125; later, the pod
over Tailscale). The SAME `top` frame the dataset and the live loop use (480x640x3 uint8).

Workflow (the "snapshot current view" method):
  1. place / drive the robot to the GOAL location by hand
  2. python scripts/capture_goal.py --out goals/run1        # snapshots the current top frame
  3. move the robot to a START pose (a short, in-distribution offset — ~10-20 cm / <~20 deg)
  4. (with Tailscale up, on the pod) lekiwi_mpc.py --planner wm --goal goals/run1/goal.png

Writes:
  <out>/goal.png            full-res top frame — pass THIS to --goal
  <out>/goal_preview.png    256x256 letterboxed exactly as LekiwiPlanner._preprocess sees it
  <out>/goal_meta.json      ip / id / shape / timestamp (UTC)

Optional `--drive "dx,dth; dx,dth; ..."` (metres, radians per chunk) drives the 6b.0 contract to a
goal pose before snapshotting, and `--return` reverses it afterwards — a programmatic alternative to
hand-placing. Default is a pure, motion-free snapshot.

Safety (only relevant with --drive): velocity clamp to the dataset envelope, e-stop confirm prompt,
Ctrl-C -> zero + disconnect, Pi host watchdog as a free fail-stop.
"""

import argparse
import datetime
import json
import os
import signal
import sys
import time

import numpy as np

import lekiwi_common as lk

DEFAULT_PI_IP = "10.0.0.125"
DEFAULT_ROBOT_ID = "lekiwi"
PREVIEW = 256  # LekiwiPlanner image_size — keep in sync with the engine's letterbox target


def get_top_frame(robot, top_hint="top"):
    """The dataset/live `top` frame as HWC uint8 RGB (mirrors lekiwi_mpc.get_top_frame)."""
    obs = robot.get_observation()
    for k in obs:
        if isinstance(k, str) and top_hint in k.lower() and "vel" not in k.lower():
            img = np.asarray(obs[k])
            if getattr(img, "ndim", 0) >= 2:
                if img.ndim == 3 and img.shape[0] in (1, 3):     # CHW -> HWC
                    img = np.transpose(img, (1, 2, 0))
                return np.ascontiguousarray(img[..., :3])
    raise RuntimeError(f"no `{top_hint}` image key in observation (keys: {list(obs)})")


def letterbox_256(frame):
    """
    Replicate LekiwiPlanner._preprocess letterboxing (PIL-only, no torch) so the preview is
    EXACTLY the planner's view: aspect-preserving resize to fit PREVIEW, zero(black)-padded, centered.
    """
    from PIL import Image
    img = Image.fromarray(frame.astype(np.uint8))
    W, H = img.size                                   # PIL size is (W, H)
    scale = min(PREVIEW / H, PREVIEW / W)
    new_w, new_h = max(1, int(W * scale)), max(1, int(H * scale))
    img = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (PREVIEW, PREVIEW), (0, 0, 0))
    canvas.paste(img, ((PREVIEW - new_w) // 2, (PREVIEW - new_h) // 2))
    return canvas


def save_png(arr_or_img, path):
    from PIL import Image
    img = arr_or_img if hasattr(arr_or_img, "save") else Image.fromarray(arr_or_img.astype(np.uint8))
    img.save(path)


def parse_chunks(spec):
    """'dx,dth; dx,dth' -> [(dx_m, dth_rad), ...]."""
    out = []
    for seg in spec.split(";"):
        seg = seg.strip()
        if not seg:
            continue
        dx, dth = (float(x) for x in seg.split(","))
        out.append((dx, dth))
    return out


def drive_chunks(robot, act_keys, base_keys, chunks, hold, speed_scale, label):
    """Drive a (Δx, Δθ) chunk sequence open-loop via the 6b.0 contract (clamped)."""
    print(f"[drive:{label}] {len(chunks)} chunk(s)")
    for i, (dx, dth) in enumerate(chunks):
        vx, th = lk.clamp_velocity(*lk.chunk_to_velocity(dx * speed_scale, dth * speed_scale))
        if abs(th) > 0 and abs(th) < lk.DEADBAND_WARN_DEG:
            print(f"  [warn] chunk {i}: |theta|={abs(th):.1f}deg/s < deadband {lk.DEADBAND_WARN_DEG} — may not turn")
        lk.stream_velocity(robot, act_keys, base_keys, vx, th, lk.CHUNK_DT, hold_action=hold)
        print(f"  chunk {i}: x.vel={vx:.3f} theta.vel={th:+.2f}")
    lk.stop(robot, act_keys, base_keys)


def main():
    ap = argparse.ArgumentParser(description="LeKiwi 6b.4 goal-image capture (snapshot the top frame)")
    ap.add_argument("--out", default="goals/goal", help="output dir (created); writes goal.png + preview + meta")
    ap.add_argument("--ip", default=DEFAULT_PI_IP)
    ap.add_argument("--id", default=DEFAULT_ROBOT_ID)
    ap.add_argument("--settle", type=float, default=0.6, help="seconds stationary before the snapshot")
    ap.add_argument("--warmup", type=int, default=3, help="frames to pull+discard before the keeper (camera AE/AGC)")
    # optional programmatic drive-to-goal (default: pure snapshot, no motion)
    ap.add_argument("--drive", default=None, help='chunks "dx,dth; dx,dth" (m, rad) to reach the goal pose first')
    ap.add_argument("--return", dest="return_", action="store_true", help="reverse the --drive sequence after snapshot")
    ap.add_argument("--speed-scale", type=float, default=1.0, help="scale on --drive velocity (<1 to start)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    chunks = parse_chunks(args.drive) if args.drive else []

    LeKiwiClient, LeKiwiClientConfig = lk.import_lekiwi()
    robot = LeKiwiClient(LeKiwiClientConfig(remote_ip=args.ip, id=args.id))

    def _stop(*_):
        try:
            ak = lk.feature_keys(robot, "action") or []
            lk.stop(robot, ak, lk.classify_base_vel_keys(ak))
            robot.disconnect()
        except Exception:
            pass
        print("\n[ctrl-c] sent zero + disconnected."); sys.exit(130)
    signal.signal(signal.SIGINT, _stop)

    print(f"[capture] connecting to {args.id} @ {args.ip} ...")
    robot.connect()
    act_keys = lk.feature_keys(robot, "action")
    base_keys = lk.classify_base_vel_keys(act_keys)

    try:
        if chunks:
            if not base_keys["x"] or not base_keys["theta"]:
                sys.exit("[capture] --drive needs base vel keys; none mapped.")
            try:
                if input("[capture] --drive will MOVE the robot. Clear space + e-stop in reach? type 'go': ").strip().lower() != "go":
                    print("[capture] aborted."); robot.disconnect(); return
            except EOFError:
                robot.disconnect(); return
            hold = lk.capture_hold(robot, act_keys, base_keys)
            drive_chunks(robot, act_keys, base_keys, chunks, hold, args.speed_scale, "to-goal")

        # settle (guarantee stationary) then snapshot
        if base_keys["x"] and base_keys["theta"]:
            lk.stop(robot, act_keys, base_keys)
        time.sleep(args.settle)
        for _ in range(max(0, args.warmup)):
            get_top_frame(robot)                      # discard warmup frames (let AE/AGC settle)
        frame = get_top_frame(robot)
        print(f"[capture] top frame {frame.shape} {frame.dtype}")

        goal_path = os.path.join(args.out, "goal.png")
        save_png(frame, goal_path)
        save_png(letterbox_256(frame), os.path.join(args.out, "goal_preview.png"))
        meta = {
            "ip": args.ip, "id": args.id,
            "shape": list(frame.shape), "dtype": str(frame.dtype),
            "preview_size": PREVIEW,
            "drive_chunks": chunks or None,
            "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        }
        with open(os.path.join(args.out, "goal_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        if args.return_ and chunks:
            rev = [(-dx, -dth) for dx, dth in reversed(chunks)]
            hold = lk.capture_hold(robot, act_keys, base_keys)
            drive_chunks(robot, act_keys, base_keys, rev, hold, args.speed_scale, "return")

    finally:
        if base_keys["x"] and base_keys["theta"]:
            lk.stop(robot, act_keys, base_keys)
        robot.disconnect()

    print(f"\n[done] goal saved:\n  {goal_path}   <- pass to lekiwi_mpc.py --goal\n"
          f"  {os.path.join(args.out, 'goal_preview.png')}   <- 256x256, exactly what the planner sees\n"
          f"  {os.path.join(args.out, 'goal_meta.json')}")


if __name__ == "__main__":
    main()
