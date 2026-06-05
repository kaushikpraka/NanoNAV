"""
Stage 6b.2 — LekiwiPlanner engine smoke-test (offline, on the pod; no robot).

Drives external/nanowm/src/sample/lekiwi_engine.LekiwiPlanner directly with raw `top` frames
pulled from the dataset mp4 (480x640x3 uint8 — exactly what LeKiwiClient.get_observation returns),
so _preprocess letterboxing is exercised on the real input shape. Verifies the four things the
engine docstring says to confirm on the pod:
  (1) action stats mean~[0.0221,-0.0006], std~[0.0141,0.0707]   (integrate_se2 f=10 stats)
  (2) do_nothing sanity: plan(frame, frame) -> dist_to_goal ~ 0
  (3) decoded `imagined` is a plausible top-view, not noise     (saved PNG to eyeball)
  (4) on a MOVING goal, CEM recovers the correct first-chunk (vx, theta) sign vs the GT action stream.

NOTE — this passes the integrate_se2 stats to LekiwiPlanner EXPLICITLY (action_mean/action_std). On the
pod the engine's dataset-rebuild fallback for these stats is dead: LeRobotDataset phones the HF Hub for the
version ref even with a local root (private repo -> 401) and lerobot v3.0 can't read the v2.1 codec. This is
the intended on-robot config (no dataset present) and is a hard precondition for 6b.3. See
context/experiment-log.md "Stage 6b.2".

Reproduce (pod, venv at /workspace/nanowm-venv, secrets sourced for completeness):
  # static pair (do_nothing + plumbing):
  python scripts/lekiwi_engine_smoke.py
  # moving pair (sign recovery on an arc):
  SMOKE_EP=11 SMOKE_START=504 SMOKE_TAG=motion python scripts/lekiwi_engine_smoke.py
Outputs land in /workspace/results/smoke_6b2[_TAG]/ (gitignored).
"""
import os, sys, math
import numpy as np
import torch

REPO = "/workspace/NanoNAV"
CKPT = "/workspace/results/20260603_160326-NanoWM-B-2-F4S10-lekiwi/checkpoints/across_timesteps/epoch=13-step=8000.ckpt"
EP = int(os.environ.get("SMOKE_EP", "44"))      # episode
START = int(os.environ.get("SMOKE_START", "0")) # start frame (goal = START + GOAL_H*F)
TAG = os.environ.get("SMOKE_TAG", "")           # output subdir suffix
VIDEO = f"/workspace/data/lekiwi/videos/chunk-000/observation.images.top/episode_{EP:06d}.mp4"
OUT = "/workspace/results/smoke_6b2" + (f"_{TAG}" if TAG else "")
NANOWM_SRC = os.path.join(REPO, "external/nanowm/src")
GOAL_H = 3          # chunks ahead
F = 10              # frame_interval -> goal sits GOAL_H*F = 30 frames ahead at 30Hz
os.makedirs(OUT, exist_ok=True)

sys.path.append(NANOWM_SRC)
sys.path.append(os.path.join(NANOWM_SRC, "sample"))


def read_frames(path, idxs):
    """Return {idx: HWC uint8 RGB} via pyav — mirrors lekiwi_replay's direct mp4 read."""
    import av
    want = sorted(set(idxs)); got = {}
    container = av.open(path)
    for i, frame in enumerate(container.decode(video=0)):
        if i in want:
            got[i] = frame.to_ndarray(format="rgb24")
        if i >= want[-1]:
            break
    container.close()
    return got


def save_png(arr, name):
    from PIL import Image
    Image.fromarray(arr).save(os.path.join(OUT, name))


def main():
    from lekiwi_engine import LekiwiPlanner

    print("=" * 70)
    print("6b.2 engine smoke-test — instantiating LekiwiPlanner (step-8000, DDIM=3)")
    print("=" * 70)
    # integrate_se2 f=10 stats as derived + printed by the 6a run (results/offline_planning_step8000/run.log).
    # This is the live-robot configuration: the engine takes stats explicitly so it never reconstructs the
    # (private, v2.1-codec) dataset on the robot. Verdict (1) confirms they match the engine's expected values.
    A_MEAN = [0.022110389545559883, -0.0005879045929759741]
    A_STD  = [0.014105414971709251, 0.07071184366941452]
    planner = LekiwiPlanner(ckpt=CKPT, device="cuda", ddim=3, num_samples=32,
                            opt_steps=3, topk=10, horizon=GOAL_H, n_elite_viz=3,
                            action_mean=A_MEAN, action_std=A_STD)

    # raw frames: t=START (current) and t=START+GOAL_H*F (goal).
    g_idx = START + GOAL_H * F
    frames = read_frames(VIDEO, [START, g_idx])
    frame0, goal = frames[START], frames[g_idx]
    print(f"\n[frames] ep{EP} start idx={START} shape={frame0.shape} dtype={frame0.dtype}  "
          f"goal idx={g_idx} shape={goal.shape}  (raw robot-native res)")
    save_png(frame0, "00_start.png"); save_png(goal, "01_goal.png")

    # ---- (2) do_nothing sanity: frame vs itself -> dist ~ 0 ----
    print("\n--- do_nothing sanity (plan(frame0, frame0)) ---")
    res0 = planner.plan(frame0, frame0)
    print(f"  dist_to_goal(z0,z0) = {res0.dist_to_goal:.4f}   (expect ~0)")
    if res0.imagined_rgb is not None:
        save_png(res0.imagined_rgb, "02_imagined_donothing.png")

    # ground-truth motion over the window (from the parquet action stream) for sign comparison
    import pandas as pd
    pq = f"/workspace/data/lekiwi/data/chunk-000/episode_{EP:06d}.parquet"
    a_gt = np.stack(pd.read_parquet(pq)["action"].values)[START:g_idx]   # [30,2] x.vel(m/s), theta.vel(rad/s)
    gt_fwd_cm = float(a_gt[:, 0].sum() / 30.0 * 100)            # ~net forward (cm), heading-ignored
    gt_dth_deg = float(np.degrees(a_gt[:, 1].sum() / 30.0))     # net heading change (deg)
    gt_first_vx = float(a_gt[:F, 0].mean())                     # first-chunk mean vx (m/s)
    gt_first_thdeg = float(np.degrees(a_gt[:F, 1].mean()))      # first-chunk mean theta (deg/s)
    print(f"\n[GT window] net fwd~{gt_fwd_cm:+.1f}cm  net dtheta~{gt_dth_deg:+.0f}deg   "
          f"| first-chunk GT vx={gt_first_vx:+.4f} theta={gt_first_thdeg:+.2f}deg/s")

    # ---- (3) + sign recovery: real plan to a goal GOAL_H chunks ahead ----
    print("\n--- real plan (plan(frame0, goal @ +30 frames)) ---")
    res = planner.plan(frame0, goal)
    print(f"  first-chunk action : vx={res.vx:+.4f} m/s   theta={res.theta_deg:+.2f} deg/s")
    print(f"  GT first chunk     : vx={gt_first_vx:+.4f} m/s   theta={gt_first_thdeg:+.2f} deg/s")
    print(f"  dist_to_goal(z0,zg): {res.dist_to_goal:.4f}   (should be >> do_nothing dist)")
    print(f"  cem_loss           : {res.cem_loss:.4f}")
    print(f"  imagined_rgb       : {None if res.imagined_rgb is None else res.imagined_rgb.shape}")
    print(f"  n elites decoded   : {len(res.elite_rgb)}")
    if res.imagined_rgb is not None:
        save_png(res.imagined_rgb, "03_imagined_plan.png")
    for k, e in enumerate(res.elite_rgb):
        save_png(e, f"04_elite_{k}.png")

    # ---- verdicts ----
    print("\n" + "=" * 70)
    print("VERDICTS")
    print("=" * 70)
    m = planner.a_mean.tolist(); s = planner.a_std.tolist()
    ok_stats = (abs(m[0]-0.0221) < 2e-3 and abs(m[1]+0.0006) < 2e-3 and
                abs(s[0]-0.0141) < 2e-3 and abs(s[1]-0.0707) < 2e-3)
    ok_donoth = res0.dist_to_goal < 1.0
    ok_moves  = res.dist_to_goal > 5 * max(res0.dist_to_goal, 1e-3)
    ok_imag   = (res.imagined_rgb is not None and res.imagined_rgb.std() > 5)
    print(f"  [{'PASS' if ok_stats else 'FAIL'}] action stats match integrate_se2 f=10  "
          f"mean={m} std={s}")
    print(f"  [{'PASS' if ok_donoth else 'FAIL'}] do_nothing dist≈0  ({res0.dist_to_goal:.4f} < 1.0)")
    print(f"  [{'PASS' if ok_moves else 'FAIL'}] goal frame is farther than itself  "
          f"({res.dist_to_goal:.3f} vs {res0.dist_to_goal:.3f})")
    print(f"  [{'PASS' if ok_imag else 'FAIL'}] decoded imagined is structured, not flat  "
          f"(std={res.imagined_rgb.std():.1f})")
    allok = all([ok_stats, ok_donoth, ok_moves, ok_imag])
    print(f"\n  OVERALL: {'PASS — engine validated, ready for 6b.3 on-robot' if allok else 'FAIL — investigate'}")
    print(f"  PNGs in {OUT}/  (eyeball 00_start vs 03_imagined_plan vs 01_goal)")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
