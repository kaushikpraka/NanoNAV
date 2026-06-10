#!/usr/bin/env python3
"""
C0.5 — token->RGB decoder for VISUALIZATION ONLY (Option C / semantic-wm-retrain.md).

Trains a small upsampling conv decoder from frozen DINOv2 patch tokens [384,16,16] (in the
WM's scaled space, latent_scale applied) back to 256x256 RGB. Purpose: rerun-viewer imagined
strips + rollout sanity once the WM predicts tokens. It is NEVER in the planning cost path —
the cost stays in token space; decoder blur is cosmetic.

Trains on the existing dataset frames (stride-sampled), encoding tokens on the fly with the
same codec convention as training (facebook/dinov2-small, 224px, latent_scale 2.4).

Run on the POD (GPU, ~1-2 h):
    /workspace/nanowm-venv/bin/python scripts/train_token_decoder.py \
        --root /workspace/data/lekiwi --out /workspace/results/token_decoder
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.append(_HERE)

from sweep_common import preprocess_frame  # noqa: E402
from build_latent_cache import list_episodes, episode_paths, decode_strided_frames  # noqa: E402


def build_decoder(in_ch=384, base=512):
    import torch.nn as nn

    def up(cin, cout):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.GroupNorm(8, cout),
            nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.GroupNorm(8, cout),
            nn.SiLU(),
        )

    return nn.Sequential(                       # [B,384,16,16] -> [B,3,256,256]
        nn.Conv2d(in_ch, base, 1),
        nn.SiLU(),
        up(base, 256),                          # 32
        up(256, 128),                           # 64
        up(128, 64),                            # 128
        up(64, 32),                             # 256
        nn.Conv2d(32, 3, 3, padding=1),
        nn.Tanh(),                              # [-1,1], the frame convention
    )


def main():
    ap = argparse.ArgumentParser(description="train DINO-token -> RGB viz decoder")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="facebook/dinov2-small")
    ap.add_argument("--latent-scale", type=float, default=2.4,
                    help="MUST match the WM's latent_codec.latent_scale")
    ap.add_argument("--frame-stride", type=int, default=3, help="dataset frame subsample")
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--val-every", type=int, default=2000)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F

    nanowm_src = os.path.join(_HERE, "..", "external", "nanowm", "src")
    sys.path.append(nanowm_src)
    from latent_codecs.semantic import WebDINOLatentCodec  # noqa: E402
    from latent_codecs.base import LatentShape  # noqa: E402

    device = torch.device(args.device)
    codec = WebDINOLatentCodec(model_path=args.model,
                               latent_shape=LatentShape(channels=384, height=16, width=16),
                               input_size=224, patch_size=14, precision="fp32",
                               latent_scale=args.latent_scale)
    codec.to(device)

    # ---- load frames (uint8, letterboxed lazily per batch to save RAM) ----
    print("[decoder] loading frames ...")
    frames = []
    for ep in list_episodes(args.root):
        _, mp4 = episode_paths(args.root, ep)
        if mp4.exists():
            frames.extend(f for _, f in decode_strided_frames(mp4, args.frame_stride))
    print(f"[decoder] {len(frames)} frames (stride {args.frame_stride})")
    rng = np.random.default_rng(0)
    val_idx = rng.choice(len(frames), 8, replace=False)
    train_idx = np.setdiff1d(np.arange(len(frames)), val_idx)

    dec = build_decoder().to(device)
    opt = torch.optim.AdamW(dec.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    os.makedirs(args.out, exist_ok=True)

    def batch_tensors(idx):
        t = torch.stack([preprocess_frame(frames[i]) for i in idx]).to(device)  # [-1,1]
        with torch.no_grad():
            z = codec.encode(t)
        return t, z

    val_x, val_z = batch_tensors(val_idx)

    for step in range(1, args.steps + 1):
        idx = rng.choice(train_idx, args.batch, replace=False)
        x, z = batch_tensors(idx)
        pred = dec(z)
        loss = F.l1_loss(pred, x) + 0.5 * F.mse_loss(pred, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 200 == 0:
            print(f"[decoder] step {step}/{args.steps}  loss {loss.item():.4f}", flush=True)
        if step % args.val_every == 0 or step == args.steps:
            with torch.no_grad():
                vp = dec(val_z)
                vl = (F.l1_loss(vp, val_x) + 0.5 * F.mse_loss(vp, val_x)).item()
            grid = torch.cat([val_x, vp], dim=0)                  # top: GT, bottom: decode
            grid = ((grid.clamp(-1, 1) + 1) * 127.5).byte().cpu().permute(0, 2, 3, 1).numpy()
            rows = [np.concatenate(list(grid[i * 8:(i + 1) * 8]), axis=1) for i in range(2)]
            from PIL import Image
            Image.fromarray(np.concatenate(rows, axis=0)).save(
                os.path.join(args.out, f"val_{step:06d}.png"))
            torch.save({"state_dict": dec.state_dict(), "in_ch": 384,
                        "latent_scale": args.latent_scale, "model": args.model,
                        "step": step, "val_loss": vl},
                       os.path.join(args.out, "decoder.pt"))
            print(f"[decoder] step {step}  VAL loss {vl:.4f}  -> decoder.pt + val grid", flush=True)

    print(f"[decoder] done -> {args.out}/decoder.pt")


if __name__ == "__main__":
    main()
