#!/usr/bin/env python3
"""
Phase-0 latent cache builder (NanoNAV 6d step 0a) — encode the dataset ONCE, downstream is array math.

Reads the derived v2.1 LeRobot dataset DIRECTLY from parquet+mp4 (the lekiwi_replay.py-validated
path — recent lerobot can't load v2.1), takes every chunk-boundary frame (frame_index % f == 0,
f=10 -> ~4,490 chunk latents from 50 episodes / 44,926 frames), preprocesses with the EXACT
training/engine transform (sweep_common.preprocess_frame: letterbox [0,1], pad 0, *2-1), and
encodes with the frozen SD-VAE.

DETERMINISM NOTE: encodes the posterior MODE (sample=False), where training/engine
encode_first_stage default SAMPLES — a deterministic cache is what a distance metric wants;
the difference is sub-noise-floor but recorded in meta.json.

Consumers: rung-0/1 pair sampler (Phase 1), distance-field visualization, graph nodes (Phase 3),
GCBC states. See learned-distance-metric.md "Sequencing".

Output (<out>/):
    latents.npy   [N, C, h, w] float32 (or float16 with --fp16); SD-VAE -> [N,4,32,32], ~75 MB fp32
    index.csv     row,episode,chunk_idx,frame_idx,video_rel   (RGB pointers for pixel-side consumers)
    meta.json     every convention pinned: f, image_size, normalize, scaling_factor, encoder, ckpt

Run on the POD (GPU; ~minutes):
    /workspace/nanowm-venv/bin/python scripts/build_latent_cache.py \
        --root /workspace/data/lekiwi --ckpt <step-8000.ckpt> --nanowm-src external/nanowm/src \
        --out /workspace/results/latent_cache

  # checkpoint-free fallback (same frozen weights from HF; identical latents up to fp noise):
    python scripts/build_latent_cache.py --root /workspace/data/lekiwi --hf-vae --out ...
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.append(_HERE)

from sweep_common import MODEL_SIZE, preprocess_frame  # noqa: E402

VIDEO_KEY = "observation.images.top"


def episode_paths(root, episode):
    chunk = episode // 1000
    pq = Path(root) / f"data/chunk-{chunk:03d}/episode_{episode:06d}.parquet"
    mp4 = Path(root) / f"videos/chunk-{chunk:03d}/{VIDEO_KEY}/episode_{episode:06d}.mp4"
    return pq, mp4


def list_episodes(root):
    eps = []
    for pq in sorted(Path(root).glob("data/chunk-*/episode_*.parquet")):
        eps.append(int(pq.stem.split("_")[1]))
    return eps


def decode_strided_frames(mp4_path, stride):
    """Sequential mp4 decode keeping every `stride`-th frame -> [(frame_idx, HWC uint8 RGB)...]."""
    import av
    out = []
    with av.open(str(mp4_path)) as container:
        for fi, frame in enumerate(container.decode(video=0)):
            if fi % stride == 0:
                out.append((fi, frame.to_ndarray(format="rgb24")))
    return out


class CkptEncoder:
    """Exact-parity encoder: the checkpoint's own latent codec, deterministic (mode, no sampling)."""

    def __init__(self, ckpt, nanowm_src, device):
        import torch
        for p in (nanowm_src, os.path.join(nanowm_src, "sample")):
            if p not in sys.path:
                sys.path.append(p)
        from action_diagnostic import load_checkpoint           # noqa: E402
        _, latent_codec, _, train_cfg = load_checkpoint(ckpt, torch.device(device))
        self.codec = latent_codec
        self.device = torch.device(device)
        self.f = int(train_cfg.dataset.frame_interval)
        img = train_cfg.model.image_size
        self.image_size = img if isinstance(img, int) else int(img[0])
        self.desc = {"encoder": "ckpt_latent_codec", "ckpt": os.path.abspath(ckpt),
                     "codec_kind": getattr(latent_codec, "kind", "sd_vae"), "sample": False}
        if hasattr(latent_codec, "vae"):
            self.desc["scaling_factor"] = float(latent_codec.vae.config.scaling_factor)
        torch.set_grad_enabled(False)

    def encode(self, batch):                                    # batch: [B,3,S,S] in [-1,1]
        if hasattr(self.codec, "vae"):
            from utils.vae_ops import encode_first_stage        # noqa: E402
            return encode_first_stage(self.codec.vae, batch.to(self.device),
                                      precision=getattr(self.codec, "precision", "fp32"),
                                      sample=False).cpu()
        return self.codec.encode(batch.to(self.device)).cpu()   # semantic codecs: deterministic


class HFEncoder:
    """Checkpoint-free: same frozen SD-VAE weights from HF (stabilityai/sd-vae-ft-mse)."""

    def __init__(self, device, model_id="stabilityai/sd-vae-ft-mse"):
        import torch
        from diffusers import AutoencoderKL
        self.device = torch.device(device)
        self.vae = AutoencoderKL.from_pretrained(model_id).eval().to(self.device)
        self.scale = float(self.vae.config.scaling_factor)
        self.f = 10
        self.image_size = MODEL_SIZE
        self.desc = {"encoder": "hf_diffusers", "model_id": model_id,
                     "scaling_factor": self.scale, "sample": False}
        torch.set_grad_enabled(False)

    def encode(self, batch):
        return (self.vae.encode(batch.to(self.device)).latent_dist.mode() * self.scale).cpu()


def main():
    ap = argparse.ArgumentParser(description="encode dataset chunk-boundary frames -> latent cache")
    ap.add_argument("--root", required=True, help="derived v2.1 dataset root (/workspace/data/lekiwi)")
    ap.add_argument("--ckpt", default=None, help="WM checkpoint (exact-parity codec encode)")
    ap.add_argument("--nanowm-src", default="external/nanowm/src")
    ap.add_argument("--hf-vae", action="store_true", help="encode with HF sd-vae-ft-mse instead of --ckpt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stride", type=int, default=None,
                    help="frame stride (default = the ckpt's frame_interval, i.e. chunk boundaries; 1 = all 30 Hz frames)")
    ap.add_argument("--episodes", default=None, help="subset 'a:b' (default all)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--save-frames", action="store_true",
                    help="also write frames/{row:05d}.jpg per cache row (raw camera RGB; runtime "
                         "goal handoff + filmstrips re-letterbox through the normal engine path)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch

    if args.hf_vae:
        enc = HFEncoder(args.device)
    elif args.ckpt:
        enc = CkptEncoder(args.ckpt, args.nanowm_src, args.device)
    else:
        sys.exit("[cache] need --ckpt (exact parity) or --hf-vae")
    stride = args.stride or enc.f
    size = enc.image_size

    episodes = list_episodes(args.root)
    if not episodes:
        sys.exit(f"[cache] no episodes under {args.root}/data/chunk-*/")
    if args.episodes:
        a, b = (int(x) if x else None for x in args.episodes.split(":"))
        episodes = episodes[a:b]
    print(f"[cache] {len(episodes)} episodes, stride={stride} (chunk boundaries), "
          f"size={size}, encoder={enc.desc['encoder']}")

    os.makedirs(args.out, exist_ok=True)
    if args.save_frames:
        os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)
    lat_rows, idx_rows = [], []
    for ep in episodes:
        pq, mp4 = episode_paths(args.root, ep)
        if not mp4.exists():
            print(f"[cache] ep{ep}: missing {mp4} — SKIPPED")
            continue
        frames = decode_strided_frames(mp4, stride)
        tensors = torch.stack([preprocess_frame(f, size) for _, f in frames])
        lats = []
        for i in range(0, len(tensors), args.batch):
            lats.append(enc.encode(tensors[i:i + args.batch]))
        lats = torch.cat(lats)
        for k, (fi, rgb) in enumerate(frames):
            row = len(idx_rows)
            idx_rows.append({"row": row, "episode": ep, "chunk_idx": fi // stride,
                             "frame_idx": fi,
                             "video_rel": str(mp4.relative_to(args.root))})
            if args.save_frames:
                from PIL import Image
                Image.fromarray(rgb).save(os.path.join(args.out, "frames", f"{row:05d}.jpg"),
                                          quality=90)
        lat_rows.append(lats)
        print(f"[cache] ep{ep}: {len(frames)} frames -> latents {tuple(lats.shape[1:])}")

    all_lats = torch.cat(lat_rows).numpy()
    if args.fp16:
        all_lats = all_lats.astype(np.float16)
    np.save(os.path.join(args.out, "latents.npy"), all_lats)
    with open(os.path.join(args.out, "index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["row", "episode", "chunk_idx", "frame_idx", "video_rel"])
        w.writeheader(); w.writerows(idx_rows)
    meta = {"n": len(idx_rows), "latent_shape": list(all_lats.shape[1:]),
            "dtype": str(all_lats.dtype), "stride": stride, "image_size": size,
            "preprocess": "letterbox bilinear align_corners=False, pad 0 in [0,1], then *2-1 "
                          "(== LekiwiPlanner._preprocess / world_model_dataset normalize_pixel)",
            "video_key": VIDEO_KEY, "dataset_root": os.path.abspath(args.root),
            "episodes": episodes, **enc.desc}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    mb = all_lats.nbytes / 1e6
    print(f"[cache] wrote {args.out}: latents.npy [{len(idx_rows)}, {','.join(map(str, all_lats.shape[1:]))}] "
          f"({mb:.0f} MB), index.csv, meta.json")


if __name__ == "__main__":
    main()
