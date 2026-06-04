#!/usr/bin/env python3
"""
6b.3 — LeKiwi closed-loop stop-and-plan MPC controller (NanoNAV Stage 6b).

The stop-and-plan loop that wraps the 6a-validated planner behind an injected `Planner`
interface, so the SAME loop runs:
  • LOCALLY (free, no GPU) with `--planner stub` — drives a canned motion + fakes a
    descending distance, exercising the ENTIRE observe→plan→execute→terminate→telemetry +
    safety path on the real robot. This is how you validate the harness before paying for a pod.
  • ON THE POD with `--planner wm` — the real DiffusionWorldModel + CEMPlanner (Stage 6b.2
    `lekiwi_engine.LekiwiPlanner`, in the fork) plans against a goal image at DDIM=3.

"Resume on the pod" is therefore a planner swap + a device/endpoint change, NOT a rewrite —
the loop, the precise per-chunk timing, the velocity clamp, termination, and rerun telemetry
are identical and already validated locally.

The loop (per cycle ≈ 8–9 s with the WM; ~1 s with the stub):
  1. STOP + settle               (guarantee stationary before observing/planning)
  2. OBSERVE the `top` frame
  3. PLAN  res = planner.plan(frame, goal)        # (vx, theta_deg, dist_to_goal, +imagined/elites on pod)
  4. TELEMETRY → rerun           (live · goal · imagined rollout · elite fan · dist · cmd)
  5. TERMINATE? dist_to_goal < --reach-thresh → success
  6. EXECUTE the first chunk     (clamp → stream_velocity for exactly CHUNK_DT) → back to 1

Safety: velocity clamp to the dataset envelope, --speed-scale (start <1 on the real robot),
--max-steps cap, the Pi host watchdog as a free fail-stop (network drop → motors stop), and
Ctrl-C → zero + disconnect.

Reuses the 6b.0/6b.1-validated `lekiwi_common` (contract, precise streaming, clamp, hold).
"""

import argparse
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

import numpy as np

import lekiwi_common as lk

DEFAULT_PI_IP = "10.0.0.125"
DEFAULT_ROBOT_ID = "lekiwi"


# --------------------------------------------------------------------------- planner interface

@dataclass
class PlanResult:
    vx: float                 # m/s, first chunk (body forward)
    theta_deg: float          # deg/s, first chunk (+ = CCW/left, per 6b.0)
    dist_to_goal: float       # latent-L2 to goal (WM) or a proxy (stub) — drives termination + logging
    cem_loss: Optional[float] = None
    imagined_rgb: Optional[np.ndarray] = None        # decoded WM rollout's predicted goal frame (pod only)
    elite_rgb: List[np.ndarray] = field(default_factory=list)  # decoded top-K elite rollouts (pod only)


class StubPlanner:
    """
    No-GPU stand-in: returns a canned (vx, theta_deg) and a distance that descends to 0 over
    `steps_to_reach`, so the loop's observe→plan→execute→terminate→telemetry path runs end-to-end
    on the real robot WITHOUT a world model. Validates transport + timing + safety + rerun plumbing.
    """
    def __init__(self, vx=0.05, theta_deg=0.0, steps_to_reach=6, start_dist=50.0):
        self.vx, self.theta_deg = vx, theta_deg
        self.steps_to_reach, self.start_dist = max(1, steps_to_reach), start_dist
        self.k = 0

    def plan(self, frame, goal) -> PlanResult:
        self.k += 1
        dist = max(0.0, self.start_dist * (1.0 - self.k / self.steps_to_reach))
        return PlanResult(vx=self.vx, theta_deg=self.theta_deg, dist_to_goal=dist, cem_loss=dist)


def make_planner(args):
    if args.planner == "stub":
        return StubPlanner(vx=args.stub_vx, theta_deg=args.stub_theta, steps_to_reach=args.stub_steps)
    # --planner wm: the real engine (Stage 6b.2, in the fork) — runs on the pod with the ckpt + GPU.
    if not args.nanowm_src:
        sys.exit("--planner wm needs --nanowm-src <path to fork>/src and --ckpt (run on the pod).")
    sys.path.append(args.nanowm_src)
    try:
        from lekiwi_engine import LekiwiPlanner          # fork: src/planning/lekiwi_engine.py (6b.2)
    except Exception as e:
        sys.exit(f"[planner=wm] could not import lekiwi_engine from {args.nanowm_src}: {e}\n"
                 f"  (6b.2 engine module — author/run on the pod where nanowm + the checkpoint live.)")
    return LekiwiPlanner(
        ckpt=args.ckpt, device=args.device, ddim=args.ddim,
        num_samples=args.num_samples, opt_steps=args.opt_steps, topk=args.topk,
        horizon=args.horizon, n_elite_viz=args.elite_viz,
    )


# --------------------------------------------------------------------------- io helpers

def get_top_frame(robot, top_hint="top"):
    obs = robot.get_observation()
    for k in obs:
        if isinstance(k, str) and top_hint in k.lower() and "vel" not in k.lower():
            img = np.asarray(obs[k])
            if getattr(img, "ndim", 0) >= 2:
                if img.ndim == 3 and img.shape[0] in (1, 3):     # CHW → HWC
                    img = np.transpose(img, (1, 2, 0))
                return np.ascontiguousarray(img)
    raise RuntimeError("no `top` image key in observation")


def load_goal(path):
    try:
        import imageio.v3 as iio
        img = np.asarray(iio.imread(path))
    except Exception:
        from PIL import Image
        img = np.asarray(Image.open(path).convert("RGB"))
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    return np.ascontiguousarray(img[..., :3])


# --------------------------------------------------------------------------- rerun telemetry

def rr_init(args):
    if not args.rerun:
        return None
    try:
        import rerun as rr
        rr.init("nanonav_lekiwi_mpc", spawn=(args.rerun_addr is None))
        if args.rerun_addr:                              # connect to a viewer on the Mac over the tailnet/LAN
            for fn in ("connect_grpc", "connect_tcp", "connect"):   # version-dependent
                if hasattr(rr, fn):
                    getattr(rr, fn)(args.rerun_addr)
                    break
        return rr
    except Exception as e:
        print(f"[rerun] disabled ({e}) — continuing without live telemetry.")
        return None


def rr_log(rr, step, frame, goal, res: PlanResult, executed):
    if rr is None:
        return
    try:
        rr.set_time_sequence("step", step)
        rr.log("live", rr.Image(frame))
        rr.log("goal", rr.Image(goal))
        rr.log("dist_to_goal", rr.Scalar(res.dist_to_goal))
        rr.log("cmd/vx", rr.Scalar(executed[0]))
        rr.log("cmd/theta_deg", rr.Scalar(executed[1]))
        if res.cem_loss is not None:
            rr.log("cem_loss", rr.Scalar(res.cem_loss))
        if res.imagined_rgb is not None:
            rr.log("imagined", rr.Image(res.imagined_rgb))
        for i, e in enumerate(res.elite_rgb or []):
            rr.log(f"elite/{i}", rr.Image(e))
    except Exception as e:
        print(f"[rerun] log failed ({e})")


# --------------------------------------------------------------------------- main loop

def main():
    ap = argparse.ArgumentParser(description="LeKiwi 6b.3 closed-loop stop-and-plan MPC")
    ap.add_argument("--planner", choices=["stub", "wm"], default="stub")
    ap.add_argument("--goal", default=None, help="goal image (from capture_goal / pre-staged); required for wm")
    ap.add_argument("--reach-thresh", type=float, default=35.0,
                    help="terminate when dist_to_goal < this (latent-L2; calibrate ~35 from 6a)")
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--speed-scale", type=float, default=1.0, help="global scale on executed velocity (<1 to start)")
    ap.add_argument("--settle", type=float, default=0.4, help="seconds stationary after STOP before OBSERVE")
    # robot
    ap.add_argument("--ip", default=DEFAULT_PI_IP)
    ap.add_argument("--id", default=DEFAULT_ROBOT_ID)
    ap.add_argument("--no-execute", action="store_true", help="run the loop but never send motion (plumbing test)")
    # stub planner
    ap.add_argument("--stub-vx", type=float, default=0.05)
    ap.add_argument("--stub-theta", type=float, default=0.0)
    ap.add_argument("--stub-steps", type=int, default=6)
    # wm planner (pod)
    ap.add_argument("--nanowm-src", default=None, help="path to <fork>/src (for --planner wm on the pod)")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ddim", type=int, default=3)
    ap.add_argument("--num-samples", type=int, default=32)
    ap.add_argument("--opt-steps", type=int, default=3)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--elite-viz", type=int, default=3, help="top-K elite rollouts to decode for rerun")
    # telemetry
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--rerun-addr", default=None, help="viewer addr (Mac tailnet IP); default spawns local viewer")
    args = ap.parse_args()

    if args.planner == "wm" and not args.goal:
        sys.exit("--planner wm needs --goal <image>.")
    goal = load_goal(args.goal) if args.goal else None
    if goal is not None:
        print(f"[goal] {args.goal}  shape={goal.shape}")

    planner = make_planner(args)
    rr = rr_init(args)

    # connect
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
        robot.disconnect(); sys.exit("[mpc] base vel keys not mapped.")

    if not args.no_execute:
        try:
            if input(f"[{args.planner}] Clear space + e-stop in reach? type 'go' to run: ").strip().lower() != "go":
                print("[mpc] aborted."); robot.disconnect(); return
        except EOFError:
            robot.disconnect(); return

    hold = lk.capture_hold(robot, act_keys, base_keys)
    reached = False
    for step in range(args.max_steps):
        # 1. STOP + settle
        lk.stop(robot, act_keys, base_keys)
        time.sleep(args.settle)

        # 2. OBSERVE
        frame = get_top_frame(robot)

        # 3. PLAN
        t0 = time.monotonic()
        res = planner.plan(frame, goal)
        plan_ms = (time.monotonic() - t0) * 1000.0

        # 4. EXECUTE (clamp + scale) — compute now so telemetry logs what we actually send
        vx, th = lk.clamp_velocity(res.vx * args.speed_scale, res.theta_deg * args.speed_scale)

        # 5. TELEMETRY
        rr_log(rr, step, frame, goal if goal is not None else frame, res, (vx, th))
        print(f"  step {step:>2}/{args.max_steps}  dist={res.dist_to_goal:7.2f}  plan={plan_ms:6.0f}ms  "
              f"→ x.vel={vx:.3f} theta.vel={th:+.2f}")

        # 6. TERMINATE?
        if res.dist_to_goal < args.reach_thresh:
            reached = True
            print(f"[mpc] reached: dist {res.dist_to_goal:.2f} < {args.reach_thresh}")
            break

        # execute the first chunk (precise CHUNK_DT hold), then loop back to STOP
        if not args.no_execute:
            n = lk.stream_velocity(robot, act_keys, base_keys, vx, th, lk.CHUNK_DT, hold_action=hold)
        else:
            time.sleep(lk.CHUNK_DT)

    lk.stop(robot, act_keys, base_keys)
    robot.disconnect()
    print(f"\n[done] {'REACHED' if reached else 'max_steps'} after {step+1} steps. "
          f"{'(no-execute plumbing run)' if args.no_execute else ''}")


if __name__ == "__main__":
    main()
