#!/usr/bin/env python3
"""
6b.3 — LeKiwi closed-loop stop-and-plan MPC controller (NanoNAV Stage 6b).

The stop-and-plan loop that wraps the 6a-validated planner behind an injected `Planner`
interface, so the SAME loop runs:
  • LOCALLY (free, no GPU) with `--planner stub` — drives a canned motion + fakes a
    descending distance, exercising the ENTIRE observe→plan→execute→terminate→telemetry +
    safety path on the real robot. This is how you validate the harness before paying for a pod.
  • ON THE POD with `--planner wm` — the real DiffusionWorldModel + CEMPlanner (Stage 6b.2
    `lekiwi_engine.LekiwiPlanner`, in the fork) plans against a goal image at DDIM=3. The
    integrate_se2 (Δx,Δθ) denorm stats are wired in as `--action-mean/--action-std` defaults
    (the engine can't derive them on the pod — private-repo 401 + lerobot-v3-vs-v2.1; 6b.2).

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
import os
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

# integrate_se2 f=10 action stats — derived + printed by the 6a offline eval
# (results/offline_planning_step8000/run.log) and confirmed by the 6b.2 engine smoke-test.
# LekiwiPlanner denormalizes CEM's output to metric (Δx m, Δθ rad) with these, so they are a HARD
# precondition for --planner wm: the engine's only other source is rebuilding the val dataset, which
# is DEAD on the pod — LeRobotDataset phones the HF Hub for the version ref even with a local root
# (private repo → 401) and lerobot v3.0 can't read the v2.1 codec; the stats are not in the ckpt either.
# A wrong/zero std silently rescales (a zero std zeros every command). So they are wired in as the
# default here and must travel with the checkpoint. See context/experiment-log.md "Stage 6b.2".
INTEGRATE_SE2_ACTION_MEAN = [0.022110389545559883, -0.0005879045929759741]
INTEGRATE_SE2_ACTION_STD = [0.014105414971709251, 0.07071184366941452]


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
        sys.exit("--planner wm needs --nanowm-src <fork>/src and --ckpt (run on the pod).")
    sys.path.append(args.nanowm_src)                           # .../src
    sys.path.append(os.path.join(args.nanowm_src, "sample"))   # .../src/sample (where lekiwi_engine lives)
    try:
        from lekiwi_engine import LekiwiPlanner                # fork: src/sample/lekiwi_engine.py (6b.2)
    except Exception as e:
        sys.exit(f"[planner=wm] could not import lekiwi_engine from {args.nanowm_src}/sample: {e}\n"
                 f"  (6b.2 engine module — run on the pod where nanowm + the checkpoint live.)")
    # Explicit action stats are mandatory on the pod (see INTEGRATE_SE2_ACTION_* above): the engine's
    # dataset-rebuild fallback is dead here, and a zero/short std would silently zero or rescale every
    # command. Validate hard before we ever move the robot.
    a_mean, a_std = list(args.action_mean), list(args.action_std)
    if len(a_mean) != 2 or len(a_std) != 2 or any(not np.isfinite(s) or s == 0.0 for s in a_std):
        sys.exit(f"[planner=wm] need 2 finite, non-zero --action-std (got mean={a_mean}, std={a_std}); "
                 f"a zero std silently zeros every command. Defaults are the integrate_se2 f=10 stats.")
    print(f"[planner=wm] action stats mean={a_mean} std={a_std} "
          f"({'default integrate_se2 f=10' if a_mean == INTEGRATE_SE2_ACTION_MEAN and a_std == INTEGRATE_SE2_ACTION_STD else 'CLI override'})")
    return LekiwiPlanner(
        ckpt=args.ckpt, device=args.device, ddim=args.ddim,
        num_samples=args.num_samples, opt_steps=args.opt_steps, topk=args.topk,
        horizon=args.horizon, n_elite_viz=args.elite_viz,
        action_mean=a_mean, action_std=a_std, var_scale=args.var_scale,
        cost_metric=args.cost_metric, cost_mode=args.cost_mode,
        token_decoder=args.token_decoder,
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

APP_ID = "nanonav_lekiwi_mpc"


def _rr_connect(rr, rec, addr):
    """Point `rec` at a live viewer (version-tolerant); addr=None uses rerun's default grpc endpoint."""
    for fn in ("connect_grpc", "connect_tcp", "connect"):
        if not hasattr(rr, fn):
            continue
        try:
            getattr(rr, fn)(addr, recording=rec) if addr else getattr(rr, fn)(recording=rec)
            return True
        except TypeError:                                # older signature: positional addr, no recording=
            getattr(rr, fn)(addr) if addr else getattr(rr, fn)()
            return True
    return False


def rr_init(args):
    """
    Returns (rr, streams) or None. `streams` is a list of independent RecordingStreams we tee EVERY log
    to — so live (--rerun-addr) and a durable .rrd (--rerun-save) can run AT THE SAME TIME (rerun 0.22
    has a single global sink per recording, so two simultaneous sinks need two streams). `None` entries
    mean "the default recording" (a spawned local viewer).
    """
    if not (args.rerun or args.rerun_save or args.rerun_addr or args.rerun_web):
        return None
    try:
        import rerun as rr
    except Exception as e:
        print(f"[rerun] disabled ({e}) — continuing without telemetry.")
        return None
    streams = []
    viewer_recs = []                                     # live viewers (web/addr/spawn) get the simplified blueprint
    # 1) durable .rrd record (headless-safe; the always-on capture of EVERYTHING)
    if args.rerun_save:
        rec = rr.new_recording(application_id=APP_ID)
        rr.save(args.rerun_save, recording=rec)
        streams.append(rec)
        print(f"[rerun] recording -> {args.rerun_save}  (open later on the Mac: `rerun {os.path.basename(args.rerun_save)}`)")
    # 1b) pod-hosted WEB viewer (robust: no rerun install / version-match on the Mac — just a browser).
    #     Serves the viewer over HTTP (web_port) + a WebSocket data feed (ws_port). Forward BOTH from the
    #     Mac (ssh -L web_port + -L ws_port) and open http://127.0.0.1:<web_port>.
    if args.rerun_web:
        rec = rr.new_recording(application_id=APP_ID)
        rr.serve_web(open_browser=False, web_port=args.rerun_web_port, ws_port=args.rerun_ws_port,
                     recording=rec, server_memory_limit="25%")
        streams.append(rec)
        viewer_recs.append(rec)
        print(f"[rerun] web viewer -> http://127.0.0.1:{args.rerun_web_port}  (ws {args.rerun_ws_port})\n"
              f"        on the Mac: ssh -N -L {args.rerun_web_port}:localhost:{args.rerun_web_port} "
              f"-L {args.rerun_ws_port}:localhost:{args.rerun_ws_port} root@<POD_IP> -p <SSH_PORT>, then open the URL")
    # 2) live stream to a viewer over the tunnel (real-time) — composes with the file record above
    if args.rerun_addr:
        rec = rr.new_recording(application_id=APP_ID)
        ok = _rr_connect(rr, rec, args.rerun_addr)
        streams.append(rec)
        viewer_recs.append(rec)
        print(f"[rerun] live -> {args.rerun_addr}"
              + ("" if ok else "  [WARN] connect failed — is the viewer up + port forwarded?"))
    # 3) bare --rerun: spawn a local viewer (needs a display; guarded so a headless run can't crash)
    if args.rerun:
        try:
            rr.init(APP_ID, spawn=True)
            streams.append(None)                         # None == the default (spawned) recording
            viewer_recs.append(None)
            print("[rerun] spawned local viewer")
        except Exception as e:
            print(f"[rerun] local spawn skipped ({e}) — use --rerun-addr for a remote viewer on a headless host")
    # Simplified live layout: only the camera frame, the imagined rollout, and the goal — side by side.
    # The .rrd still records EVERYTHING (dist/cmd/cem_loss/elite) for diagnosis; this only declutters the
    # live viewer. auto_views=False suppresses auto-panels for the unlisted (scalar/elite) entities.
    if viewer_recs:
        try:
            import rerun.blueprint as rrb
            H = int(getattr(args, "horizon", 3))
            # Default: image row (camera | imagined +1..+H | goal — all the SAME timestep, atomic
            # post-plan) with the dist trace BELOW. Nested Vertical wedged an older web viewer, so
            # --viewer-flat keeps the single-row fallback if it recurs.
            views = [rrb.Spatial2DView(origin="model/live", name="camera (now)"),
                     rrb.Spatial2DView(origin="imagined", name="imagined +1 (executes next)")]
            for i in range(2, H + 1):
                views.append(rrb.Spatial2DView(origin=f"rollout/h{i}",
                                               name=f"imagined +{i}" + (" (CEM target)" if i == H else "")))
            graph_mode = bool(getattr(args, "graph", None))
            views.append(rrb.Spatial2DView(origin="model/goal",
                                           name="target (waypoint)" if graph_mode else "goal"))
            if getattr(args, "viewer_flat", False):
                bp = rrb.Blueprint(rrb.Horizontal(*views), auto_views=False, collapse_panels=True)
            else:
                ts = [rrb.TimeSeriesView(origin="dist_to_goal",
                                         name="dist (to waypoint)" if graph_mode else "dist")]
                if graph_mode:
                    ts.append(rrb.TimeSeriesView(origin="graph_dist", name="route dist to GOAL"))
                rows = [rrb.Horizontal(*views)]
                shares = [3]
                if graph_mode:                          # full planned trajectory, src -> goal
                    rows.append(rrb.Spatial2DView(origin="route", name="planned route -> goal"))
                    shares.append(1.4)
                rows.append(ts[0] if len(ts) == 1 else rrb.Horizontal(*ts))
                shares.append(1)
                bp = rrb.Blueprint(rrb.Vertical(*rows, row_shares=shares),
                                   auto_views=False, collapse_panels=True)
            for rec in viewer_recs:
                rr.send_blueprint(bp, recording=rec) if rec is not None else rr.send_blueprint(bp)
            mode = "flat" if getattr(args, "viewer_flat", False) else "2-row"
            tgt = "waypoint (indexed)" if graph_mode else "goal"
            below = "" if mode == "flat" else ("  /  planned route strip  /  dist + route-dist below"
                                               if graph_mode else "  /  dist below")
            print(f"[rerun] viewer blueprint ({mode}): camera | imagined +1..+{H} | {tgt}{below}")
        except Exception as e:
            print(f"[rerun] blueprint skipped ({e}) — viewer falls back to auto-layout")
    return (rr, streams) if streams else None


def _rr_set_time(rr, step, rec):
    if hasattr(rr, "set_time"):                      # rerun ≥ 0.23
        try:
            rr.set_time("step", sequence=step, recording=rec); return
        except TypeError:
            pass
    rr.set_time_sequence("step", step, recording=rec)   # 0.22


def _rr_scalar_cls(rr):
    return getattr(rr, "Scalars", None) or getattr(rr, "Scalar")   # 0.23+ Scalars, else Scalar


def _action_arrow(rr, vx, theta_deg, img_hw=256):
    """A 2D arrow (image-pixel space) for the executed command, to overlay on the imagined frame:
    length ∝ forward speed, tilt = turn rate (+θ = CCW/left → tilts left), origin at bottom-center.
    Labeled with the exact numbers so the glyph stays honest. Returns an Arrows2D archetype."""
    import math
    bx, by = img_hw / 2.0, img_hw - 18.0
    mag = min(max(abs(vx) * 1500.0, 18.0), 110.0)         # m/s -> px (clamped so it's always visible)
    fwd = mag if vx >= 0 else -mag                         # reverse -> point down
    thr = math.radians(theta_deg)
    dx = -fwd * math.sin(thr)                              # +θ (CCW/left) -> -x (left in image)
    dy = -fwd * math.cos(thr)                              # forward -> -y (up in image)
    return rr.Arrows2D(vectors=[[dx, dy]], origins=[[bx, by]],
                       labels=[f"vx={vx:+.3f} m/s  θ={theta_deg:+.1f}°/s"],
                       colors=[[255, 230, 0]], radii=[2.0])


def _banner(img, text, color=(255, 210, 80)):
    """Annotated DISPLAY copy (never fed to the planner): black strip + label on top."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(np.ascontiguousarray(img))
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, im.width, 16], fill=(0, 0, 0))
    dr.text((4, 2), text, fill=color)
    return np.asarray(im)


def route_strip(nav, path, goal_rgb, max_tiles=12):
    """The ENTIRE planned waypoint trajectory as one wide image (viewer 'route' row):
    subsampled route-node frames src+1..goal-node, then the real goal image. Labels carry
    the waypoint index within the current route."""
    from PIL import Image, ImageDraw
    nodes = path[1:]                                   # src itself is the camera panel
    if not nodes:
        return None
    sel = (np.unique(np.linspace(0, len(nodes) - 1, max_tiles).astype(int))
           if len(nodes) > max_tiles else np.arange(len(nodes)))
    tw, th = 160, 120
    n = len(sel) + 1
    canvas = Image.new("RGB", (n * (tw + 2) + 2, th + 18), (20, 20, 20))
    dr = ImageDraw.Draw(canvas)
    for c, k in enumerate(sel):
        nid = int(nodes[k])
        canvas.paste(Image.fromarray(nav.node_frame(nid)).resize((tw, th)), (2 + c * (tw + 2), 2))
        dr.text((4 + c * (tw + 2), th + 4), f"wp {k + 1}/{len(nodes)}", fill=(160, 220, 255))
    canvas.paste(Image.fromarray(np.ascontiguousarray(goal_rgb)).resize((tw, th)),
                 (2 + len(sel) * (tw + 2), 2))
    dr.text((4 + len(sel) * (tw + 2), th + 4), "GOAL image", fill=(255, 180, 120))
    return np.asarray(canvas)


def rr_log_route(ctx, step, strip):
    """Log the planned-route strip AT WAYPOINT-COMPUTE TIME (pre-plan) — the operator sees the
    whole trajectory before the first 7 s plan even finishes."""
    if ctx is None or strip is None:
        return
    rr, streams = ctx
    for rec in streams:
        try:
            _rr_set_time(rr, step, rec)
            rr.log("route", rr.Image(strip), recording=rec)
        except Exception as e:
            print(f"[rerun] route log failed ({e})")


def rr_log_observed(ctx, step, frame):
    """Log the live camera frame AT OBSERVE TIME (before the ~7 s plan), so the viewer reflects
    what the planner is actually planning from — not a frame that is plan_ms stale by the time
    the full telemetry lands (the perceived several-frames lag, 2026-06-10)."""
    if ctx is None:
        return
    rr, streams = ctx
    # Display panels (model/live, imagined, goal, dist) are logged ATOMICALLY post-plan so every
    # visible panel is the SAME timestep (operator decision 2026-06-10). Only the undisplayed raw
    # `live` entity + the status line land at observe time (phase clarity + timeline truth).
    for rec in streams:
        try:
            _rr_set_time(rr, step, rec)
            rr.log("live", rr.Image(frame), recording=rec)
            rr.log("status", rr.TextLog(f"step {step}: observed -> PLANNING (panels show step {step - 1} until the plan lands)"),
                   recording=rec)
        except Exception as e:
            print(f"[rerun] observe-time log failed ({e})")


def rr_log(ctx, step, frame, goal, res: PlanResult, executed, graph=None):
    if ctx is None:
        return
    rr, streams = ctx
    Scalar = _rr_scalar_cls(rr)
    arrow = None
    try:
        arrow = _action_arrow(rr, executed[0], executed[1])
    except Exception as e:
        print(f"[rerun] arrow build failed ({e})")
    status = f"step {step}: plan ready -> executing"
    if graph is not None:
        status += (f" | graph: {graph['status']} hops={graph.get('hops_left', '-')} "
                   f"graph_dist={graph['graph_dist']:.3f}"
                   + (f" wp {graph['wp_path_idx']}/{graph['hops_left']} (ep{graph['wp_ep']}ck{graph['wp_ck']})"
                      if graph["status"] == "WAYPOINT" else ""))
    for rec in streams:                                  # tee to every active sink (file and/or live)
        try:
            _rr_set_time(rr, step, rec)
            rr.log("live", rr.Image(frame), recording=rec)
            rr.log("goal", rr.Image(goal), recording=rec)
            # what the WM actually encodes (letterboxed 256² + black bars) — the viewer's primary panels
            if res.model_live_rgb is not None:
                rr.log("model/live", rr.Image(res.model_live_rgb), recording=rec)
            if res.model_goal_rgb is not None:
                gimg = res.model_goal_rgb
                if graph is not None:                  # mark WHAT the target panel is showing
                    gimg = (_banner(gimg, f"WAYPOINT {graph['wp_path_idx']}/{graph['hops_left']}"
                                          f"  node {graph['wp']} (ep{graph['wp_ep']} ck{graph['wp_ck']})")
                            if graph["status"] == "WAYPOINT"
                            else _banner(gimg, f"FINAL GOAL ({graph['status'].lower()})", (120, 255, 140)))
                rr.log("model/goal", rr.Image(gimg), recording=rec)
            rr.log("status", rr.TextLog(status), recording=rec)
            rr.log("dist_to_goal", Scalar(res.dist_to_goal), recording=rec)
            if graph is not None and np.isfinite(graph.get("graph_dist", np.inf)):
                rr.log("graph_dist", Scalar(graph["graph_dist"]), recording=rec)
            rr.log("cmd/vx", Scalar(executed[0]), recording=rec)
            rr.log("cmd/theta_deg", Scalar(executed[1]), recording=rec)
            if res.cem_loss is not None:
                rr.log("cem_loss", Scalar(res.cem_loss), recording=rec)
            # PRIMARY imagined panel = the +1-chunk frame the robot ACTUALLY executes toward (matches the
            # action arrow + comparable to the next live frame). The +H endpoint CEM scores and the full
            # +1..+H filmstrip (shows the forward-drift + autoregressive degradation) are separate entities.
            imagined_next = res.imagined_next_rgb if res.imagined_next_rgb is not None else res.imagined_rgb
            if imagined_next is not None:
                rr.log("imagined", rr.Image(imagined_next), recording=rec)
            if arrow is not None:                          # overlay the executed (+1) action on the +1 frame
                rr.log("imagined/action", arrow, recording=rec)
            for i, f in enumerate(res.imagined_seq_rgb or []):   # +1..+H filmstrip
                rr.log(f"rollout/h{i + 1}", rr.Image(f), recording=rec)
            if res.imagined_rgb is not None:               # +H endpoint (what CEM's objective minimizes)
                rr.log("imagined_endpoint", rr.Image(res.imagined_rgb), recording=rec)
            for i, e in enumerate(res.elite_rgb or []):
                rr.log(f"elite/{i}", rr.Image(e), recording=rec)
        except Exception as e:
            print(f"[rerun] log failed ({e})")


# --------------------------------------------------------------------------- main loop

def main():
    ap = argparse.ArgumentParser(description="LeKiwi 6b.3 closed-loop stop-and-plan MPC")
    ap.add_argument("--planner", choices=["stub", "wm"], default="stub")
    ap.add_argument("--goal", default=None, help="goal image (from capture_goal / pre-staged); required for wm")
    ap.add_argument("--reach-thresh", type=float, default=35.0,
                    help="terminate when dist_to_goal < this (latent-L2; calibrate ~35 from 6a)")
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--speed-scale", type=float, default=1.0, help="global scale on executed velocity (<1 to start)")
    ap.add_argument("--vx-max", type=float, default=None, metavar="M_S",
                    help="override velocity-clamp ceiling (default 0.10 m/s = dataset envelope; max dx/chunk = "
                         "vx_max·0.333). Raise to ~0.12 for ~0.04 m steps — bigger per-step motion/signal, mildly OOD.")
    ap.add_argument("--drive-straight", type=float, default=None, metavar="VX_MS",
                    help="DIAGNOSTIC: ignore CEM and drive a fixed forward vx (m/s), θ=0, each chunk — still "
                         "encodes/logs WM dist_to_goal. Tests if real straight-line motion reduces latent dist.")
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
    ap.add_argument("--cost-metric", choices=["auto", "mse", "cosine"], default="auto",
                    help="CEM cost + termination metric. auto = mse for sd_vae ckpts, token-cosine "
                         "for semantic (DINO) ckpts. NOTE: cosine dist is ~0-2 — recalibrate "
                         "--reach-thresh (Gate-A dinov2_cos basin floor ~0.13, plateau ~0.28-0.42).")
    ap.add_argument("--cost-mode", choices=["last", "first", "all"], default="last",
                    help="which imagined frame(s) the cost scores: last=+H endpoint (6b behavior), "
                         "first=+1 chunk (least WM-degraded, the chunk actually executed), all=exp-weighted")
    ap.add_argument("--graph", default=None, metavar="DIR",
                    help="C3 subgoal-graph dir (build_subgoal_graph.py output). Per replan: localize the "
                         "live frame in token space, route to the goal, hand CEM the waypoint ~one CEM "
                         "reach ahead (a REAL cached frame); the final hop falls back to the actual "
                         "--goal image, so endgame keeps the validated reach-thresh semantics.")
    ap.add_argument("--graph-lookahead", type=float, default=None,
                    help="waypoint route-progress budget in calibrated token-cos units (default: "
                         "2*tau ~= 6 chunks, halves the waypoint count vs one-CEM-reach spacing). "
                         "Route progress, not live distance — immune to the ~+0.2 cross-session offset.")
    ap.add_argument("--token-decoder", default=None,
                    help="train_token_decoder.py checkpoint for imagined-rollout viz with semantic ckpts")
    ap.add_argument("--var-scale", type=float, default=1.0,
                    help="CEM initial sampling std × dataset action-std (1.0=in-distribution; >1 widens "
                         "exploration incl. stronger rotations, at the cost of more out-of-distribution actions)")
    ap.add_argument("--elite-viz", type=int, default=3, help="top-K elite rollouts to decode for rerun")
    ap.add_argument("--action-mean", type=float, nargs=2, default=INTEGRATE_SE2_ACTION_MEAN,
                    metavar=("DX", "DTH"),
                    help="integrate_se2 (Δx,Δθ) denorm mean for --planner wm (default = the f=10 stats; "
                         "the engine can't derive these on the pod — see context/experiment-log.md 6b.2)")
    ap.add_argument("--action-std", type=float, nargs=2, default=INTEGRATE_SE2_ACTION_STD,
                    metavar=("DX", "DTH"),
                    help="integrate_se2 (Δx,Δθ) denorm std for --planner wm (default = the f=10 stats)")
    # telemetry
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--rerun-addr", default=None, help="live viewer addr (gRPC URL over the tunnel); default spawns local viewer")
    ap.add_argument("--rerun-save", default=None, metavar="PATH",
                    help="record telemetry to a .rrd FILE (robust, no live connection); open on the Mac with `rerun PATH`")
    ap.add_argument("--rerun-web", action="store_true",
                    help="serve a pod-hosted WEB viewer (no rerun install/version-match on the Mac — just forward the ports and open a browser)")
    ap.add_argument("--rerun-web-port", type=int, default=9090, help="HTTP port for --rerun-web (default 9090)")
    ap.add_argument("--viewer-flat", action="store_true",
                    help="single-row blueprint without the dist strip (fallback if the 2-row layout wedges the web viewer)")
    ap.add_argument("--rerun-ws-port", type=int, default=9877, help="WebSocket data port for --rerun-web (default 9877)")
    args = ap.parse_args()

    # Optional override of the velocity-clamp ceiling. Default 0.10 m/s = the measured dataset envelope
    # (max executed dx/chunk = VX_MAX·CHUNK_DT = 0.033 m). Raising it lets CEM's larger imagined forward
    # steps (up to ~0.05 m) actually EXECUTE — bigger per-step motion = bigger per-step dist signal, and
    # removes the imagine-vs-execute mismatch. >~0.12 is mildly out-of-distribution (the WM saw ≤0.033/chunk).
    if args.vx_max is not None:
        print(f"[clamp] VX_MAX {lk.VX_MAX} -> {args.vx_max} m/s  (max dx/chunk {lk.VX_MAX*lk.CHUNK_DT:.3f} "
              f"-> {args.vx_max*lk.CHUNK_DT:.3f} m)")
        lk.VX_MAX = float(args.vx_max)

    if args.planner == "wm" and not args.goal:
        sys.exit("--planner wm needs --goal <image>.")
    goal = load_goal(args.goal) if args.goal else None
    if goal is not None:
        print(f"[goal] {args.goal}  shape={goal.shape}")

    planner = make_planner(args)

    # C3 subgoal graph: goal -> Dijkstra tree once, then each replan localizes + picks a waypoint
    nav = None
    if args.graph:
        if args.planner != "wm":
            sys.exit("--graph needs --planner wm (localization runs in the ckpt's token space)")
        from subgoal_graph import GraphNav
        nav = GraphNav(args.graph, device=args.device)
        zg = planner._encode_last({"visual": planner._preprocess(goal)})
        nav.set_goal(zg.flatten())

    tel = rr_init(args)

    # connect — MUST request the `top` camera explicitly; lerobot's client default exposes only
    # front/wrist and silently drops top (the camera the WM uses). See lk.make_client_config.
    LeKiwiClient, _ = lk.import_lekiwi()
    robot = LeKiwiClient(lk.make_client_config(args.ip, args.id, cameras=("top",)))
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

    # ZMQ cold-start flake: the first connect after host/tunnel (re)start times out, the retry
    # lands (observed repeatedly 2026-06-10). Recreate the client per attempt — a timed-out
    # client object doesn't recover.
    for attempt in range(1, 4):
        try:
            robot.connect()
            break
        except Exception as e:
            if attempt == 3:
                raise
            print(f"[mpc] connect attempt {attempt} failed ({type(e).__name__}); retrying in 3 s ...")
            time.sleep(3.0)
            robot = LeKiwiClient(lk.make_client_config(args.ip, args.id, cameras=("top",)))
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

        # 2. OBSERVE — and surface it in the viewer immediately (planning takes ~7 s)
        frame = get_top_frame(robot)
        rr_log_observed(tel, step, frame)

        # 3. PLAN — graph mode plans toward the current waypoint (a REAL cached frame ~one CEM
        #    reach ahead on the route); ENDGAME (last hop) / UNREACHABLE fall back to the real goal.
        plan_goal, ginfo = goal, None
        if nav is not None:
            z_live = planner._encode_last({"visual": planner._preprocess(frame)})
            wp, wp_frame, ginfo = nav.waypoint(z_live.flatten(), args.graph_lookahead)
            if ginfo["status"] == "WAYPOINT":
                plan_goal = wp_frame
            print(f"  graph: {ginfo['status']}  src={ginfo['src']} (ep{ginfo['src_ep']} ck{ginfo['src_ck']}) "
                  f"d_loc={ginfo['d_loc']:.3f}  graph_dist={ginfo['graph_dist']:.3f}  "
                  f"hops={ginfo.get('hops_left', '-')}"
                  + (f"  -> wp {ginfo['wp_path_idx']}/{ginfo['hops_left']} node {ginfo['wp']} "
                     f"(ep{ginfo['wp_ep']} ck{ginfo['wp_ck']})"
                     if ginfo["status"] == "WAYPOINT" else "  -> REAL GOAL IMAGE"))
            # the whole planned trajectory, logged BEFORE the 7 s plan starts (step 0 = the
            # "view the route beforehand" panel; later steps show it shrinking as hops burn off)
            if ginfo.get("path"):
                rr_log_route(tel, step, route_strip(nav, ginfo["path"], goal))
        t0 = time.monotonic()
        res = planner.plan(frame, plan_goal)
        plan_ms = (time.monotonic() - t0) * 1000.0

        # 4. EXECUTE (clamp + scale) — compute now so telemetry logs what we actually send.
        # --drive-straight DIAGNOSTIC: ignore CEM's command, drive a FIXED forward vx (θ=0). Still
        # encodes/logs dist_to_goal + the imagined viz, so we can tell whether real straight-line motion
        # toward the goal actually reduces the latent dist (perception/goal OK) vs whether CEM's action
        # choice was the problem. cmd_src is logged so telemetry is honest about what's being sent.
        if args.drive_straight is not None:
            vx, th = lk.clamp_velocity(args.drive_straight, 0.0)
            cmd_src = "straight"
        else:
            vx, th = lk.clamp_velocity(res.vx * args.speed_scale, res.theta_deg * args.speed_scale)
            cmd_src = "cem"

        # 5. TELEMETRY — graph mode: the goal panel shows the CURRENT WAYPOINT (what CEM is
        #    actually chasing); graph_dist is the calibrated route distance to the FINAL goal
        #    (the monotone progress signal; dist_to_goal is only to the waypoint while routing).
        rr_log(tel, step, frame, plan_goal if plan_goal is not None else frame, res, (vx, th),
               graph=ginfo)
        print(f"  step {step:>2}/{args.max_steps}  dist={res.dist_to_goal:7.2f}  plan={plan_ms:6.0f}ms  "
              f"[{cmd_src}] → x.vel={vx:.3f} theta.vel={th:+.2f}"
              + (f"  (cem would: vx={res.vx:.3f} θ={res.theta_deg:+.1f})" if cmd_src == "straight" else ""))

        # 6. TERMINATE? — while routing, dist_to_goal is to the WAYPOINT; only the endgame
        #    (planning toward the real goal image) may terminate on reach-thresh.
        if res.dist_to_goal < args.reach_thresh and (ginfo is None or ginfo["status"] != "WAYPOINT"):
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
