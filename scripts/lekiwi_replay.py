#!/usr/bin/env python3
"""
6b.1 — LeKiwi open-loop replay (NanoNAV Stage 6b).

Converts a sequence of chunk-level (Δx, Δθ) commands to base velocities with the
6b.0-pinned contract and executes them OPEN-LOOP on the robot — no world model, no
CEM, no GPU. The last hardware-grounding step before live planning.

What it validates (the things 6b.0 couldn't, because 6b.0 was single commands):
  • the (Δx,Δθ)→velocity conversion + chunk timing over a multi-chunk SEQUENCE,
  • HEADING ACCUMULATION (systematic ω bias shows up as drift/curl — the `square`
    pattern is the acid test: if heading is right, it returns near the start),
  • the CONSTANT-VELOCITY-WITHIN-CHUNK approximation (dataset source only: real
    recorded velocities vary within a chunk; we collapse each to one constant vx/ω),
  • deadband behaviour over a real sequence (warns on sub-deadband turns).

NO ODOMETRY exists (dataset state is velocity, not pose), so "did it trace the path"
is eyeballed against the DEAD-RECKONED trajectory this script prints/plots — the robot
also starts from a different room pose than any recording, so only the SHAPE of the
motion (relative) is comparable, not an absolute path.

Two sources:
  --source synthetic  (default): hand-authored chunk patterns (forward / pivot / arc /
      square). Controlled + unambiguous; tests conversion+heading+deadband. Always works,
      no dataset needed. (Cannot test the within-chunk collapse — synthetic IS constant.)
  --source dataset:   integrate a recorded episode's per-frame velocities into chunks
      (the real action distribution; tests the within-chunk collapse). Needs the v2.1
      dataset reachable by lerobot; API path may need a tweak for your lerobot version.

SAFETY: --dry-run (default) does ZERO robot motion — it connects to nothing, just
integrates + dead-reckons + plots, and prints the per-chunk command table so you see
exactly what WOULD be sent. Pass --execute to drive the robot (clear space + e-stop;
a multi-chunk run can travel tens of cm — the dry-run prints the trajectory extent).
Ctrl-C sends zero + disconnects.
"""

import argparse
import json
import signal
import sys
from math import pi
from pathlib import Path

import numpy as np

import lekiwi_common as lk

DEFAULT_PI_IP = "10.0.0.125"
DEFAULT_ROBOT_ID = "lekiwi"
# Synthetic chunk magnitudes (near the dataset stats: Δx mean 2.2 cm, Δθ std 0.07 rad ≈ 4°).
SYNTH_DX = 0.025          # m per forward chunk  → vx ≈ 7.5 cm/s
SYNTH_DTH = 0.09          # rad per turn chunk    → ω ≈ 0.27 rad/s ≈ 15.5°/s (above deadband, in range)


# ----------------------------------------------------------------------------- sources

def synth_chunks(pattern, k):
    """Return (chunks, raw_steps) where chunks=[(dx,dth)...]; raw_steps=None (synthetic = constant)."""
    dx, dth = SYNTH_DX, SYNTH_DTH
    if pattern == "forward":
        chunks = [(dx, 0.0)] * k
    elif pattern == "pivot_left":
        chunks = [(0.0, +dth)] * k
    elif pattern == "pivot_right":
        chunks = [(0.0, -dth)] * k
    elif pattern == "arc_left":
        chunks = [(0.8 * dx, +0.8 * dth)] * k
    elif pattern == "arc_right":
        chunks = [(0.8 * dx, -0.8 * dth)] * k
    elif pattern == "square":
        turn = max(1, int(round((pi / 2) / dth)))      # chunks to pivot ~90°
        chunks = []
        for _ in range(4):
            chunks += [(dx, 0.0)] * k
            chunks += [(0.0, +dth)] * turn
    else:
        raise SystemExit(f"unknown synthetic pattern: {pattern}")
    return chunks, None


def _resolve(repo_id, relpath, root):
    """Local file under --root, else download the single file from HF (no lerobot version gate)."""
    if root:
        p = Path(root) / relpath
        if not p.exists():
            raise FileNotFoundError(f"{p} (check --root layout)")
        return str(p)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id, relpath, repo_type="dataset")


def dataset_chunks(repo_id, root, episode, start, n_chunks, f, video_key="observation.images.top"):
    """
    Integrate a recorded episode's per-frame (x.vel m/s, theta.vel rad/s) into f-window
    (Δx, Δθ) chunks via the same unicycle integration the dataloader uses (integrate_se2).
    Reads the parquet DIRECTLY (lerobot v3.0 can't read this v2.1 dataset) — version-proof.
    Returns (chunks, raw_steps) with raw_steps=[(vx,om,dt)...] at 30 Hz for the fine path.
    """
    import pandas as pd
    rel = f"data/chunk-{episode // 1000:03d}/episode_{episode:06d}.parquet"
    df = pd.read_parquet(_resolve(repo_id, rel, root)).sort_values("frame_index")
    acts = np.stack(df["action"].to_numpy()).astype(float)   # [T,2] = [x.vel m/s, theta.vel rad/s]
    print(f"[dataset] ep{episode}: {len(acts)} frames | "
          f"x.vel∈[{acts[:,0].min():.3f},{acts[:,0].max():.3f}] m/s | "
          f"theta.vel∈[{acts[:,1].min():+.3f},{acts[:,1].max():+.3f}] (rad/s expected, |ω|≲0.34)")
    acts = acts[start:]
    dt = 1.0 / 30.0
    chunks, raw = [], []
    while (n_chunks is None or len(chunks) < n_chunks) and (len(chunks) + 1) * f <= len(acts):
        win = acts[len(chunks) * f:(len(chunks) + 1) * f]
        x = th = 0.0
        for vx, om in win:
            x += vx * dt * np.cos(th)
            th += om * dt
            raw.append((float(vx), float(om), dt))
        chunks.append((x, th))
    if not chunks:
        raise SystemExit("[dataset] no full chunks in range — lower --start or --f, or pick a longer episode.")
    return chunks, raw


def save_recorded_frames(repo_id, root, episode, start, n_chunks, f, n_frames, out_png,
                         video_key="observation.images.top"):
    """Dump n_frames evenly-spaced recorded `top` frames as a filmstrip (decoded directly from the mp4)."""
    try:
        import av
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rel = f"videos/chunk-{episode // 1000:03d}/{video_key}/episode_{episode:06d}.mp4"
        path = _resolve(repo_id, rel, root)
        lo, hi = start, start + n_chunks * f
        idxs = np.linspace(lo, hi - 1, n_frames).round().astype(int)
        want = set(int(i) for i in idxs)

        grabbed = {}
        container = av.open(path)
        for fi, frame in enumerate(container.decode(video=0)):
            if fi in want:
                grabbed[fi] = frame.to_ndarray(format="rgb24")
            if fi >= max(want):
                break
        container.close()

        fig, axes = plt.subplots(1, n_frames, figsize=(2.4 * n_frames, 2.6))
        axes = np.atleast_1d(axes)
        for ax, i in zip(axes, idxs):
            ax.imshow(grabbed[int(i)]); ax.axis("off")
            ax.set_title(f"chunk ~{(int(i) - lo) // f}", fontsize=8)
        fig.suptitle(f"recorded `top` frames — ep{episode} (compare to the robot's live view)", fontsize=9)
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=110, bbox_inches="tight")
        print(f"[frames] recorded filmstrip → {out_png}")
    except Exception as e:
        print(f"[frames] could not save filmstrip ({e}) — skipped (the trajectory plot still stands).")


# ----------------------------------------------------------------------------- reporting

def cmd_table(chunks):
    """Per-chunk (vx, theta_deg) after conversion + clamp; print + flag deadband; return list."""
    print(f"\n  {'#':>3}  {'Δx cm':>7} {'Δθ °':>7} │ {'x.vel':>7} {'theta.vel':>10}  notes")
    cmds = []
    for i, (dx, dth) in enumerate(chunks):
        vx, th_deg = lk.chunk_to_velocity(dx, dth)
        cvx, cth = lk.clamp_velocity(vx, th_deg)
        note = []
        if (cvx, cth) != (vx, th_deg):
            note.append("CLAMPED")
        if 0 < abs(cth) < lk.DEADBAND_WARN_DEG:
            note.append(f"⚠deadband(<{lk.DEADBAND_WARN_DEG:.0f}°/s may not turn)")
        cmds.append((cvx, cth))
        print(f"  {i:>3}  {dx*100:>7.2f} {dth*180/pi:>7.1f} │ {cvx:>7.3f} {cth:>10.2f}  {' '.join(note)}")
    return cmds


def plot_paths(cmds, raw_steps, out_png, title):
    intended = lk.dead_reckon([(vx, cth * pi / 180.0, lk.CHUNK_DT) for vx, cth in cmds])
    fine = lk.dead_reckon(raw_steps) if raw_steps else None
    xs, ys, ths = intended
    extent = float(max(np.ptp(xs), np.ptp(ys)))
    end = (xs[-1], ys[-1], ths[-1] * 180 / pi)
    print(f"\n[dead-reckon] commanded path: end (x={end[0]*100:+.1f} cm, y={end[1]*100:+.1f} cm, "
          f"heading={end[2]:+.1f}°); bounding extent ≈ {extent*100:.0f} cm.")
    if fine is not None:
        fx, fy = fine[0][-1], fine[1][-1]
        gap = float(np.hypot(fx - xs[-1], fy - ys[-1]))
        print(f"[dead-reckon] recorded(fine 30Hz) path end (x={fx*100:+.1f}, y={fy*100:+.1f} cm); "
              f"chunk-approx endpoint gap ≈ {gap*100:.1f} cm  (the within-chunk-collapse error).")
    net_turn = (ths[-1] - ths[0]) * 180 / pi
    path_len = float(np.sum(np.hypot(np.diff(xs), np.diff(ys)))) * 100
    print(f"[dead-reckon] net forward {xs[-1]*100:+.1f} cm, net lateral {ys[-1]*100:+.1f} cm, "
          f"net turn {net_turn:+.1f}°, path length {path_len:.0f} cm.")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax, axh) = plt.subplots(1, 2, figsize=(11, 5.2),
                                      gridspec_kw={"width_ratios": [1.4, 1]})

        # ---- left: top-down path with HEADING ARROWS (orientation = what you watch for) ----
        if fine is not None:
            ax.plot(fine[0]*100, fine[1]*100, "-", color="0.6", lw=1.2, label="recorded (30 Hz)")
        ax.plot(xs*100, ys*100, "-o", ms=3, color="C0", label="chunked command (sent)")
        # heading arrows at up to ~14 evenly-spaced waypoints
        step = max(1, len(xs) // 14)
        idx = list(range(0, len(xs), step))
        alen = max(extent, 0.05) * 100 * 0.12        # arrow ≈ 12% of the figure extent
        ax.quiver(xs[idx]*100, ys[idx]*100, np.cos(ths[idx]), np.sin(ths[idx]),
                  color="C3", angles="xy", scale_units="xy", scale=1.0/alen,
                  width=0.006, label="heading")
        ax.plot([0], [0], "gs", ms=11, label="START (robot here, facing +x →)")
        ax.plot([xs[-1]*100], [ys[-1]*100], "r*", ms=16, label="expected END")
        ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="best")
        ax.set_xlabel("forward x (cm)"); ax.set_ylabel("lateral y (cm)  (+y = left)")
        ax.set_title(title)

        # ---- right: heading vs chunk (when does it turn, and which way) ----
        axh.plot(np.arange(len(ths)), ths * 180 / pi, "-o", ms=3, color="C3")
        axh.axhline(0, color="0.7", lw=.8)
        axh.grid(alpha=.3); axh.set_xlabel("chunk #"); axh.set_ylabel("heading (°, + = CCW/left)")
        axh.set_title("heading vs chunk — compare to where the robot turns")

        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=110, bbox_inches="tight")
        print(f"[dead-reckon] trajectory plot → {out_png}")
    except Exception as e:
        print(f"[dead-reckon] (matplotlib unavailable: {e} — skipped plot; numbers above stand)")
    return extent


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="LeKiwi 6b.1 open-loop replay")
    ap.add_argument("--source", choices=["synthetic", "dataset", "file"], default="synthetic")
    ap.add_argument("--pattern", default="square",
                    help="synthetic: forward|pivot_left|pivot_right|arc_left|arc_right|square")
    ap.add_argument("--k", type=int, default=5, help="synthetic: forward/turn chunks per leg")
    # dataset source
    ap.add_argument("--repo-id", default="kaushikpraka/wm-smallarea_nav30")
    ap.add_argument("--root", default=None, help="local dataset root (else lerobot cache/HF)")
    # file source (decouple dataset-read from robot-drive): read chunks where the data lives,
    # export the JSON, then --source file --commands <json> --execute on the Mac (no dataset needed).
    ap.add_argument("--commands", default=None, help="--source file: chunk JSON written by --export-commands")
    ap.add_argument("--export-commands", default=None, help="write the chunk sequence to this JSON")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--start", type=int, default=0, help="frame offset within the episode")
    ap.add_argument("--chunks", type=int, default=12, help="dataset: number of chunks to replay")
    ap.add_argument("--f", type=int, default=10, help="frame_interval (chunk size in frames)")
    # execution
    ap.add_argument("--execute", action="store_true", help="DRIVE THE ROBOT (else dry-run only)")
    ap.add_argument("--stop-between", action="store_true",
                    help="brief stop between chunks (more like stop-and-plan MPC; default = continuous)")
    ap.add_argument("--ip", default=DEFAULT_PI_IP)
    ap.add_argument("--id", default=DEFAULT_ROBOT_ID)
    ap.add_argument("--out", default="viz/lekiwi_6b1")
    ap.add_argument("--save-frames", type=int, default=0,
                    help="dataset: also dump N evenly-spaced recorded `top` frames as a filmstrip "
                         "(sanity-check the path against the room)")
    args = ap.parse_args()

    # ---- build the chunk sequence ----
    if args.source == "synthetic":
        chunks, raw = synth_chunks(args.pattern, args.k)
        title = f"6b.1 synthetic:{args.pattern} (k={args.k})"
        tag = args.pattern
    elif args.source == "dataset":
        chunks, raw = dataset_chunks(args.repo_id, args.root, args.episode, args.start, args.chunks, args.f)
        title = f"6b.1 dataset ep{args.episode}+{args.start} ({len(chunks)} chunks, f={args.f})"
        tag = f"ep{args.episode}"
    else:  # file — chunks precomputed where the dataset reads cleanly (e.g. the pod)
        if not args.commands:
            raise SystemExit("--source file needs --commands <json>")
        with open(args.commands) as fp:
            blob = json.load(fp)
        chunks = [tuple(c) for c in blob["chunks"]]
        raw = None
        title = blob.get("title", f"6b.1 file:{Path(args.commands).stem}")
        tag = Path(args.commands).stem
        if abs(blob.get("chunk_dt", lk.CHUNK_DT) - lk.CHUNK_DT) > 1e-6:
            print(f"[warn] exported chunk_dt {blob['chunk_dt']} != current {lk.CHUNK_DT}")
    print(f"[plan] {title}: {len(chunks)} chunks × {lk.CHUNK_DT:.3f}s ≈ {len(chunks)*lk.CHUNK_DT:.1f}s of motion")

    cmds = cmd_table(chunks)
    out = Path(args.out)
    extent = plot_paths(cmds, raw, out / f"{args.source}_{tag}.png", title)

    if args.export_commands:
        Path(args.export_commands).parent.mkdir(parents=True, exist_ok=True)
        with open(args.export_commands, "w") as fp:
            json.dump({"chunk_dt": lk.CHUNK_DT, "title": title,
                       "chunks": [[float(dx), float(dth)] for dx, dth in chunks]}, fp, indent=2)
        print(f"[export] {len(chunks)} chunks → {args.export_commands}  "
              f"(copy to the Mac, run: --source file --commands {Path(args.export_commands).name} --execute)")

    if args.save_frames and args.source == "dataset":
        save_recorded_frames(args.repo_id, args.root, args.episode, args.start,
                             len(chunks), args.f, args.save_frames, out / f"dataset_{tag}_frames.png")

    if not args.execute:
        print("\n[dry-run] no robot motion. Review the table + plot; re-run with --execute "
              "(clear space, e-stop in reach) to drive it.")
        return

    # ---- execute ----
    print(f"\n[execute] this will DRIVE the robot — needs ≈ {extent*100:.0f} cm of clear space (plus margin).")
    LeKiwiClient, LeKiwiClientConfig = lk.import_lekiwi()
    robot = LeKiwiClient(LeKiwiClientConfig(remote_ip=args.ip, id=args.id))

    base_keys = {}

    def _stop(*_):
        try:
            ak = lk.feature_keys(robot, "action") or []
            lk.stop(robot, ak, base_keys or lk.classify_base_vel_keys(ak))
            robot.disconnect()
        except Exception:
            pass
        print("\n[ctrl-c] sent zero + disconnected."); sys.exit(130)
    signal.signal(signal.SIGINT, _stop)

    robot.connect()
    act_keys = lk.feature_keys(robot, "action")
    base_keys = lk.classify_base_vel_keys(act_keys)
    if not base_keys["x"] or not base_keys["theta"]:
        robot.disconnect(); sys.exit("[execute] base vel keys not mapped.")

    try:
        if input("Clear space + e-stop in reach? type 'go' to drive: ").strip().lower() != "go":
            print("[execute] aborted."); robot.disconnect(); return
    except EOFError:
        robot.disconnect(); return

    for i, (vx, th_deg) in enumerate(cmds):
        n = lk.stream_velocity(robot, act_keys, base_keys, vx, th_deg, lk.CHUNK_DT)
        rep = robot.get_observation()
        rv = {k: round(float(np.asarray(rep[k]).reshape(-1)[0]), 3) for k in rep
              if isinstance(k, str) and k.lower().endswith(".vel")}
        print(f"  chunk {i:>3}/{len(cmds)}  sent x.vel={vx:.3f} theta.vel={th_deg:+.2f}  ({n} cmds)  reported {rv}")
        if args.stop_between:
            lk.stop(robot, act_keys, base_keys)

    lk.stop(robot, act_keys, base_keys)
    robot.disconnect()
    print("\n[done] replay complete. Eyeball: did the robot's gross motion match the dead-reckoned plot "
          "(shape, total turn, rough extent)? Note any systematic curl (heading bias) or short-fall (deadband).")


if __name__ == "__main__":
    main()
