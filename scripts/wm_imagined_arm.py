#!/usr/bin/env python3
"""
Phase-0 step 0d — the WM-IMAGINED validation arm (NanoNAV 6d).

The planner's cost runs on GENERATED latents z-hat, which are approximate / slightly
off-manifold; a distance validated only on clean encoded frames could misbehave on them
(learned-distance-metric.md "SD-VAE latent handling" — the VALIDATE-FIRST decision).
This script manufactures imagined latents at KNOWN nominal displacements so the SAME
harness can overlay the imagined curve on the clean one:

  For each clean radial capture at r cm (robot facing the goal), roll the WM forward with
  straight chunks (dx per chunk, dtheta=0) for H chunks. Generated latent +k sits at a
  nominal r - k*dx*100 cm. If d(z-hat at ~30cm) tracks d(z_clean at 30cm), the clean<->
  imagined weld is tight; a big gap = fold WM-rolled-out latents into phi's training set
  (the gated decision) — and at OOD start poses, a wild imagined curve is the
  live-frame-hallucination signature made quantitative.

Writes a COMBINED harness-ready sweep dir:
  - clean rows copied from --sweep (frames referenced in place, imagined=0)
  - imagined rows: decoded frames (for image-space candidates; decode-blur caveat applies)
    + raw WM latents as .npy (latent column — sdvae_l2 scores these DIRECTLY, no decode)
  - goal.png / goal_model.png copied + goal_latent.npy (deterministic? NO — engine path,
    sampled posterior; recorded in imagined_meta.json)

Run on the POD (GPU + ckpt):
    /workspace/nanowm-venv/bin/python scripts/wm_imagined_arm.py \
        --ckpt <step-8000.ckpt> --nanowm-src external/nanowm/src \
        --sweep results/sweep_nearfan2 --out results/sweep_nearfan2_imagined \
        --action-mean 0.0221 -0.0006 --action-std 0.0141 0.0707

Then grade BOTH curves in one shot (overlay plots show clean o- vs imagined s--):
    python scripts/dist_harness.py --sweep results/sweep_nearfan2_imagined --candidates sdvae_l2,dinov2_mse
"""

import argparse
import csv
import json
import os
import shutil
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.append(_HERE)

from sweep_common import ManifestWriter, imread_rgb, load_sweep  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="roll the WM from radial sweep captures -> imagined-latent arm")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--nanowm-src", default="external/nanowm/src")
    ap.add_argument("--sweep", required=True, help="clean sweep dir (needs radial captures + goal)")
    ap.add_argument("--out", required=True, help="combined output sweep dir")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ddim", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--dx", type=float, default=0.03, help="m per straight chunk (<= 0.033 dataset max)")
    ap.add_argument("--action-mean", type=float, nargs=2, default=None,
                    help="integrate_se2 stats (REQUIRED on the pod — dataset fallback is dead there)")
    ap.add_argument("--action-std", type=float, nargs=2, default=None)
    args = ap.parse_args()

    sys.path.append(os.path.join(args.nanowm_src, "sample"))
    sys.path.append(args.nanowm_src)
    import torch
    from lekiwi_engine import LekiwiPlanner  # noqa: E402
    from PIL import Image

    sweep = load_sweep(args.sweep)
    radial = sorted(sweep.by_arm("radial", imagined=False), key=lambda c: c.params["r_cm"])
    if not radial:
        sys.exit(f"[imagined] {args.sweep}: no radial captures (labels r10/r20/... or '<N>cm')")

    lp = LekiwiPlanner(args.ckpt, device=args.device, ddim=args.ddim, horizon=args.horizon,
                       action_mean=args.action_mean, action_std=args.action_std, n_elite_viz=0)

    os.makedirs(args.out, exist_ok=True)
    frames_dir = os.path.join(args.out, "frames")
    lats_dir = os.path.join(args.out, "latents")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(lats_dir, exist_ok=True)

    # ---- goal: copy images, encode latent through the SAME engine path the planner uses ----
    goal_img = imread_rgb(sweep.goal_path)
    shutil.copy(sweep.goal_path, os.path.join(args.out, "goal_model.png" if sweep.goal_is_model_view else "goal.png"))
    _, zg = lp._goal(goal_img)
    np.save(os.path.join(args.out, "goal_latent.npy"),
            zg.reshape(lp.C_lat, lp.h_lat, lp.w_lat).cpu().numpy())

    man = ManifestWriter(args.out)
    idx = 0
    # ---- clean rows: reference the source sweep's frames in place ----
    for c in sweep.captures:
        if not c.image_path:
            continue
        man.add(idx, c.label, frame_model=os.path.abspath(c.image_path) if c.is_model_view else None,
                frame_full=None if c.is_model_view else os.path.abspath(c.image_path),
                imagined=False, note=f"clean from {args.sweep}")
        idx += 1

    # ---- imagined rows: straight rollout toward the goal from each radial capture ----
    raw = torch.zeros(1, args.horizon, 2, device=lp.device)
    raw[0, :, 0] = args.dx                                       # straight chunks, dtheta = 0
    mu = (raw - lp.a_mean) / lp.a_std                            # normalized actions for wm.rollout

    for c in radial:
        r0 = c.params["r_cm"]
        if r0 <= 0:
            continue
        frame = imread_rgb(c.image_path)
        obs_0 = {"visual": lp._preprocess(frame)}
        with torch.no_grad():
            z, _ = lp.wm.rollout(obs_0, mu, num_sampling_steps=args.ddim)
        vis = z["visual"]                                        # [1, 1+H, C_lat*h*w] (engine flattens)
        for k in range(1, vis.shape[1]):
            nominal = r0 - k * args.dx * 100.0
            if nominal < -1.0:
                break
            label = f"r{max(nominal, 0.0):g}"
            stem = f"img_{idx:03d}_from_r{r0:g}_plus{k}"
            lat = vis[0, k].cpu().numpy().reshape(lp.C_lat, lp.h_lat, lp.w_lat)
            lat_rel = os.path.join("latents", f"{stem}.npy")
            np.save(os.path.join(args.out, lat_rel), lat)
            dec_rel = os.path.join("frames", f"{stem}.png")
            Image.fromarray(lp._decode_last(vis[:, k:k + 1])).save(os.path.join(args.out, dec_rel))
            man.add(idx, label, frame_model=dec_rel, imagined=True, latent=lat_rel,
                    note=f"WM +{k} from clean r{r0:g} (nominal {nominal:.1f}cm, dx={args.dx})")
            print(f"[imagined] r{r0:g} +{k} -> nominal {nominal:5.1f} cm   ({stem})")
            idx += 1
    man.close()

    with open(os.path.join(args.out, "imagined_meta.json"), "w") as f:
        json.dump({"ckpt": os.path.abspath(args.ckpt), "ddim": args.ddim, "horizon": args.horizon,
                   "dx_m_per_chunk": args.dx, "source_sweep": os.path.abspath(args.sweep),
                   "note": "imagined latents via engine path (sampled VAE posterior on context, "
                           "WM-generated +1..+H); nominal displacement assumes on-axis straight drive",
                   "action_mean": args.action_mean, "action_std": args.action_std}, f, indent=2)
    print(f"[imagined] wrote {args.out} — grade with: python scripts/dist_harness.py --sweep {args.out}")


if __name__ == "__main__":
    main()
