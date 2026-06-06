#!/usr/bin/env python
"""
Live-telemetry smoke test for the pod-hosted rerun web viewer.

Mirrors the exact serving path in lekiwi_mpc.py rr_init() (rr.serve_web on web_port/ws_port,
logging to an explicit rr.new_recording stream). No robot, no tunnel-to-Pi needed — it just
streams a moving scalar + a live image so you can confirm updates arrive in the browser.

On the pod:
    /workspace/nanowm-venv/bin/python scripts/rerun_web_smoke.py
On the Mac:
    ssh -N -L 9090:localhost:9090 -L 9877:localhost:9877 root@<POD_IP> -p <SSH_PORT>
    open http://127.0.0.1:9090
"""
import argparse
import math
import time

import numpy as np

APP_ID = "nanonav_lekiwi_mpc"  # same app id as lekiwi_mpc.py so the viewer layout matches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--web-port", type=int, default=9090)
    ap.add_argument("--ws-port", type=int, default=9877)
    ap.add_argument("--rerun-addr", default=None,
                    help="connect to a NATIVE rerun viewer at host:port (e.g. 127.0.0.1:9999) "
                         "instead of serving the web viewer")
    ap.add_argument("--seconds", type=float, default=180.0, help="how long to stream")
    ap.add_argument("--hz", type=float, default=2.0, help="updates per second")
    args = ap.parse_args()

    import rerun as rr
    print(f"[rerun] version {rr.__version__}")

    rec = rr.new_recording(application_id=APP_ID)
    if args.rerun_addr:
        # NATIVE app path — mirrors lekiwi_mpc.py _rr_connect / --rerun-addr
        rr.connect_grpc(args.rerun_addr, recording=rec)
        print(f"[rerun] live -> {args.rerun_addr}  (native viewer; reverse-tunnel pod->Mac on that port)")
    else:
        rr.serve_web(open_browser=False, web_port=args.web_port, ws_port=args.ws_port,
                     recording=rec, server_memory_limit="25%")
        print(f"[rerun] web viewer -> http://127.0.0.1:{args.web_port}  (ws {args.ws_port})")
        print(f"        on the Mac: ssh -N -L {args.web_port}:localhost:{args.web_port} "
              f"-L {args.ws_port}:localhost:{args.ws_port} root@<POD_IP> -p <SSH_PORT>, then open the URL")
    print(f"[rerun] streaming {args.seconds:.0f}s at {args.hz:.0f} Hz — watch for live updates, then Ctrl-C")

    def set_time(step):
        if hasattr(rr, "set_time"):
            try:
                rr.set_time("step", sequence=step, recording=rec); return
            except TypeError:
                pass
        rr.set_time_sequence("step", step, recording=rec)

    n = int(args.seconds * args.hz)
    dt = 1.0 / args.hz
    h = w = 128
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for step in range(n):
        set_time(step)
        t = step * dt
        # a moving scalar (so the timeseries panel visibly advances)
        rr.log("telemetry/sine", rr.Scalar(math.sin(t)), recording=rec)
        rr.log("telemetry/step", rr.Scalar(float(step)), recording=rec)
        # a live image: a drifting gradient + a moving bright dot
        img = (0.5 + 0.5 * np.sin(0.05 * xx + t)) * 255.0
        img = np.stack([img, np.roll(img, 20, 1), 255 - img], -1).astype(np.uint8)
        cx = int((0.5 + 0.4 * math.cos(t)) * w)
        cy = int((0.5 + 0.4 * math.sin(t)) * h)
        img[max(0, cy-4):cy+4, max(0, cx-4):cx+4] = [255, 255, 0]
        rr.log("telemetry/live_image", rr.Image(img), recording=rec)
        if step % int(args.hz) == 0:
            print(f"  t={t:5.1f}s  step={step}/{n}  sine={math.sin(t):+.2f}", flush=True)
        time.sleep(dt)

    print("[rerun] done streaming. Leaving the server up for 60s so you can scrub the timeline (Ctrl-C to quit).")
    time.sleep(60)


if __name__ == "__main__":
    main()
