#!/usr/bin/env python
"""
6b.3 diagnostic — latent/pixel distance vs controlled displacement (camera conditioning probe).

The closed-loop objective (latent-L2 to a goal frame) looked FLAT far from the goal, but the original
46 cm number was confounded: the robot drove OFF-COURSE, so that was path-length, not 46 cm of real
*approach*. This tool removes the confound by letting YOU place the robot at known displacements ALONG
the straight line to the goal and capturing a clean measurement at each pose. It also captures a
same-pose noise burst so the signal can be judged against the noise floor (the SNR that decides
"metric-limited, fixable by a learned distance" vs "information-limited, camera must change").

No robot motion is ever commanded — read-only. You physically position the robot between captures.

Per capture it records, for the live `top` frame vs the goal, BOTH:
  - latent_l2  = ||z_live - z_goal||         (the actual planning objective; engine _flat_l2)
  - pixel_l1   = mean|live_rgb - goal_rgb|    (raw pixels, model-view 256² letterboxed; pre-WM signal)
and saves the decoded model-view frame, so the dist-vs-displacement curve is reproducible offline.

Run it on the POD, interactively, in YOUR terminal (so you can press Enter between moves):
    ! LEKIWI_DATA_ROOT=/workspace/data/lekiwi /workspace/nanowm-venv/bin/python scripts/measure_dist_sweep.py \
        --ckpt <step-12000.ckpt> --nanowm-src external/nanowm/src --goal goals/nearfan.png --ip 127.0.0.1 \
        --out /workspace/results/dist_sweep

Suggested protocol:
  1. Place the robot AT the goal pose. Capture with label '0cm'. Then a noise burst: label 'noise', count 8.
  2. Back the robot straight away from the goal in marked steps (10, 20, 30, 40, 50, 60 cm), capturing each.
  3. Quit ('q'). Re-run scripts/plot_dist_sweep is not needed — the CSV + per-row prints give the curve.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np


def load_image(path):
    try:
        import imageio.v3 as iio
        return np.asarray(iio.imread(path))[..., :3]
    except Exception:
        from PIL import Image
        return np.asarray(Image.open(path).convert("RGB"))


def main():
    ap = argparse.ArgumentParser(description="latent/pixel dist vs controlled displacement (no motion)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--nanowm-src", required=True, help="path to <fork>/src")
    ap.add_argument("--goal", required=True)
    ap.add_argument("--ip", default="127.0.0.1")
    ap.add_argument("--id", default="lekiwi")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ddim", type=int, default=3)
    ap.add_argument("--out", default="/workspace/results/dist_sweep")
    # --- yaw-sweep mode: robot SELF-ROTATES in place in fixed increments (pure rotation, no translation),
    #     measuring dist-to-goal at each. Starts at the current heading, rotates to -range/2, then sweeps
    #     across to +range/2 capturing each point, so a basin (min at the aligned heading) shows in the middle.
    ap.add_argument("--yaw-sweep", action="store_true", help="auto yaw sweep (robot rotates in place; commands motion)")
    ap.add_argument("--yaw-theta", type=float, default=25.0, help="rotation command (deg/s, + = CCW) per step")
    ap.add_argument("--yaw-secs", type=float, default=0.4, help="seconds of rotation per step")
    ap.add_argument("--yaw-steps", type=int, default=13, help="number of capture points across the sweep (odd → center)")
    args = ap.parse_args()

    sys.path.append(os.path.join(args.nanowm_src, "sample"))
    sys.path.append(args.nanowm_src)
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # scripts/ for lekiwi_common
    from lekiwi_engine import LekiwiPlanner, _flat_l2
    import lekiwi_common as lk
    from lekiwi_mpc import get_top_frame
    from PIL import Image

    os.makedirs(args.out, exist_ok=True)
    frames_dir = os.path.join(args.out, "frames"); os.makedirs(frames_dir, exist_ok=True)
    csv_path = os.path.join(args.out, "measurements.csv")

    print(f"[sweep] loading WM ({os.path.basename(args.ckpt)})...")
    lp = LekiwiPlanner(args.ckpt, device=args.device, ddim=args.ddim)

    goal_img = load_image(args.goal)
    obs_g, zg = lp._goal(goal_img)
    goal_rgb = lp._denorm_view(obs_g["visual"])                 # 256² letterboxed, what the VAE encodes
    Image.fromarray(goal_rgb).save(os.path.join(args.out, "goal_modelview.png"))
    print(f"[sweep] goal loaded {goal_img.shape} -> model-view 256² saved")

    print("[sweep] connecting robot (read-only, NO motion; `top` camera)...")
    LeKiwiClient, _ = lk.import_lekiwi()
    robot = LeKiwiClient(lk.make_client_config(args.ip, args.id, cameras=("top",)))
    robot.connect()
    print("[sweep] connected.\n")

    def measure():
        frame = get_top_frame(robot)
        t = lp._preprocess(frame)
        z = lp._encode_last({"visual": t})
        live_rgb = lp._denorm_view(t)
        latent_l2 = _flat_l2(z, zg)
        pixel_l1 = float(np.abs(live_rgb.astype(np.int16) - goal_rgb.astype(np.int16)).mean())
        return latent_l2, pixel_l1, live_rgb

    rows = []
    new = not os.path.exists(csv_path)
    f = open(csv_path, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["idx", "label", "latent_l2", "pixel_l1", "frame", "t"])

    # ===================== YAW SWEEP (robot self-rotates; commands motion) =====================
    if args.yaw_sweep:
        import signal, time as _t
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

        deg_per_step = args.yaw_theta * args.yaw_secs          # commanded (uncalibrated) yaw increment
        half = args.yaw_steps // 2
        print(f"[yaw] PURE ROTATION in place, no translation. {args.yaw_steps} captures, "
              f"~{deg_per_step:+.1f}°/step (cmd θ={args.yaw_theta} deg/s × {args.yaw_secs}s) → "
              f"~±{half*deg_per_step:.0f}° sweep about the start heading.")
        try:
            if input("[yaw] clear space, e-stop in reach — type 'go' to rotate: ").strip().lower() != "go":
                lk.stop(robot, act_keys, base_keys); robot.disconnect(); print("[yaw] aborted."); return
        except EOFError:
            robot.disconnect(); return

        def rotate(theta_deg, secs):
            lk.stream_velocity(robot, act_keys, base_keys, 0.0, theta_deg, secs, hold_action=hold)
            lk.stop(robot, act_keys, base_keys); _t.sleep(0.4)        # settle before observing

        # 1) pre-rotate CW to one extreme (no capture), so the sweep crosses the start heading at the middle
        for _ in range(half):
            rotate(-args.yaw_theta, args.yaw_secs)
        # 2) sweep CCW across the full range, capturing at each point
        idx = 0
        for k in range(args.yaw_steps):
            yaw = (k - half) * deg_per_step                          # cumulative angle about start (deg)
            lk.stop(robot, act_keys, base_keys); _t.sleep(0.2)
            l2, l1, live_rgb = measure()
            label = f"yaw{yaw:+.0f}"
            fp = os.path.join(frames_dir, f"yaw_{idx:03d}_{label}.png")
            Image.fromarray(live_rgb).save(fp)
            w.writerow([idx, label, f"{l2:.4f}", f"{l1:.4f}", fp, f"{_t.monotonic():.2f}"]); f.flush()
            rows.append((label, l2, l1))
            print(f"   yaw≈{yaw:+5.0f}°   latent_L2={l2:7.3f}   pixel_L1={l1:6.2f}")
            if k < args.yaw_steps - 1:
                rotate(args.yaw_theta, args.yaw_secs)                # advance to next point
        lk.stop(robot, act_keys, base_keys); robot.disconnect(); f.close()
        ys = [(float(lbl.replace("yaw","")), l2, l1) for lbl, l2, l1 in rows]
        best = min(ys, key=lambda r: r[1])
        print(f"\n[yaw] done. latent_L2 min at yaw≈{best[0]:+.0f}° (={best[1]:.2f}); "
              f"range {min(y[1] for y in ys):.2f}–{max(y[1] for y in ys):.2f} "
              f"(Δ={max(y[1] for y in ys)-min(y[1] for y in ys):.2f}). "
              f"Big Δ + clear min = usable heading gradient; flat = poor yaw conditioning.")
        print(f"[yaw] wrote {csv_path}; frames in {frames_dir}")
        return

    print("Commands:  <label> + Enter = capture (e.g. '20cm')   |   'n <label> <count>' = noise burst")
    print("           Enter alone = capture unlabeled           |   'q' = quit\n")
    idx = 0
    try:
        while True:
            cmd = input("[sweep] place robot, then label/Enter (q=quit): ").strip()
            if cmd.lower() == "q":
                break
            burst = 1
            label = cmd
            if cmd.lower().startswith("n "):                    # noise burst: 'n <label> <count>'
                parts = cmd.split()
                label = parts[1] if len(parts) > 1 else "noise"
                burst = int(parts[2]) if len(parts) > 2 else 8
            for b in range(burst):
                l2, l1, live_rgb = measure()
                fp = os.path.join(frames_dir, f"{idx:03d}_{label or 'cap'}.png")
                Image.fromarray(live_rgb).save(fp)
                w.writerow([idx, label, f"{l2:.4f}", f"{l1:.4f}", fp, f"{time.monotonic():.2f}"]); f.flush()
                rows.append((label, l2, l1))
                print(f"   #{idx:>3} [{label or '-':>8}] latent_L2={l2:7.3f}   pixel_L1={l1:6.2f}")
                idx += 1
                if burst > 1:
                    time.sleep(0.3)
            # running summary of distinct labels (mean ± std for repeated/burst labels)
    finally:
        try:
            robot.disconnect()
        except Exception:
            pass
        f.close()

    # ---- summary: per-label mean/std, and a noise-floor vs signal read ----
    if rows:
        from collections import OrderedDict
        byl = OrderedDict()
        for label, l2, l1 in rows:
            byl.setdefault(label, []).append((l2, l1))
        print("\n[sweep] === summary (per label) ===")
        print(f"{'label':>10}  {'n':>2}  {'latent_L2 mean±std':>22}  {'pixel_L1 mean±std':>20}")
        for label, vals in byl.items():
            a = np.array(vals)
            print(f"{label or '-':>10}  {len(vals):>2}  {a[:,0].mean():9.3f} ± {a[:,0].std():6.3f}      "
                  f"{a[:,1].mean():7.2f} ± {a[:,1].std():5.2f}")
        print(f"\n[sweep] wrote {csv_path}  ({idx} captures, frames in {frames_dir})")
        print("[sweep] read: compare the spread ACROSS displacement labels to the 'noise' burst std —")
        print("        if displacement Δ ≫ noise std -> signal present (metric-limited, learnable);")
        print("        if displacement Δ ≲ noise std -> aliased (information-limited, camera must change).")


if __name__ == "__main__":
    main()
