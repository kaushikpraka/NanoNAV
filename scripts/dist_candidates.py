"""
Phase-0 zero-training distance candidates (NanoNAV 6d Gate A).

Every candidate is a frozen, pretrained (or formula) distance d(image_a, image_b) -> float —
NO gradient steps anywhere. The harness (scripts/dist_harness.py) embeds each unique sweep
image once and compares pairs, so candidates expose:

    embed(imgs)  : list of HWC uint8 RGB -> feature tensor [N, ...]
    dist(fa, fb) : two feature rows -> float (lower = closer)

All candidates consume the MODEL-VIEW image (256² letterboxed, sweep_common.letterbox_rgb)
so every candidate sees the same pixels the planner's VAE sees — comparability over
per-candidate optimal crops.

Candidates (evidence + expectations: learned-distance-metric.md "Evaluation"):
    pixel_l1     mean |a-b| on the 256² letterbox — the pre-WM signal floor.
    sdvae_l2     flat L2 on SD-VAE latents = THE CURRENT PLANNING OBJECTIVE (baseline arm).
                 Loads diffusers stabilityai/sd-vae-ft-mse (same frozen weights as training).
                 Uses the posterior MODE (deterministic) where the engine's encode_first_stage
                 default SAMPLES — so the engine's same-pose noise floor is >= this one;
                 documented difference, deliberate (we grade the metric, not VAE noise).
                 Latents are x scaling_factor, matching utils/vae_ops.encode_first_stage.
    dinov2_mse   mean squared diff over DINOv2 PATCH tokens (the DINO-WM planning cost).
    dinov2_cos   1 - mean per-patch cosine over DINOv2 patch tokens.
                 (Patch-level on purpose: pooled/CLS is contraindicated — DINO-WM Wall 0.96
                 patch vs 0.58 CLS. --dinov2 vits14|vitb14.)
    vip_l2       L2 on VIP's 1024-d embedding (arXiv:2210.00030). OPTIONAL — needs the
                 `vip` package (github.com/facebookresearch/vip). Expected to lose
                 (double-OOD, global pooled embedding); included as the published baseline.
    vjepa21      mean squared diff over V-JEPA 2.1 image-tokenizer tokens (16x16x1024).
                 OPTIONAL — needs nanowm src on sys.path + VJEPA21_MODEL_PATH (and
                 VJEPA2_REPO_PATH for torch.hub), same env vars as nanowm's codec.

Usage: from dist_candidates import build_candidates
       cands = build_candidates(["pixel_l1", "sdvae_l2", "dinov2_mse"], device="cuda")
Missing optional deps -> the candidate is SKIPPED with a warning, never a crash.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.append(_HERE)

from sweep_common import MODEL_SIZE, letterbox_rgb, preprocess_frame  # noqa: E402

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _to_model_views(imgs):
    """list of HWC uint8 (any size) -> [N,256,256,3] uint8 model views."""
    out = []
    for im in imgs:
        im = np.asarray(im)
        if im.shape[:2] != (MODEL_SIZE, MODEL_SIZE):
            im = letterbox_rgb(im)
        out.append(im)
    return np.stack(out)


class Candidate:
    name = "base"
    batch = 16

    def embed(self, imgs):
        raise NotImplementedError

    def dist(self, fa, fb):
        raise NotImplementedError

    def _batched(self, x, fn):
        import torch
        outs = []
        for i in range(0, len(x), self.batch):
            outs.append(fn(x[i:i + self.batch]))
        return torch.cat(outs)


class PixelL1(Candidate):
    name = "pixel_l1"

    def embed(self, imgs):
        import torch
        return torch.as_tensor(_to_model_views(imgs)).float()

    def dist(self, fa, fb):
        return float((fa - fb).abs().mean())


class SDVAEL2(Candidate):
    """The current planning objective: flat L2 on scaled SD-VAE latents (deterministic mode)."""
    name = "sdvae_l2"
    # Features ARE WM-space scaled latents [4,32,32] — the harness feeds WM-imagined latents
    # (Capture.latent_path, from wm_imagined_arm.py) to dist() directly, skipping any decode.
    feature_is_wm_latent = True

    def __init__(self, device="cpu", model_id="stabilityai/sd-vae-ft-mse"):
        import torch
        from diffusers import AutoencoderKL
        self.device = torch.device(device)
        self.vae = AutoencoderKL.from_pretrained(model_id).eval().to(self.device)
        self.scale = float(self.vae.config.scaling_factor)
        torch.set_grad_enabled(False)

    def embed(self, imgs):
        import torch
        views = _to_model_views(imgs)
        t = torch.stack([preprocess_frame(v) for v in views]).to(self.device)  # [N,3,256,256] in [-1,1]

        def enc(b):
            return self.vae.encode(b).latent_dist.mode() * self.scale          # [B,4,32,32]
        return self._batched(t, enc).cpu()

    def dist(self, fa, fb):
        import torch
        return float(torch.norm((fa - fb).reshape(-1)))                        # matches engine _flat_l2


class Dinov2Patch(Candidate):
    """DINOv2 patch-token distance (DINO-WM cost). variant: vits14 | vitb14; mode: mse | cos."""

    def __init__(self, device="cpu", variant="vits14", mode="mse"):
        import torch
        self.name = f"dinov2_{mode}" + ("" if variant == "vits14" else f"_{variant}")
        self.device = torch.device(device)
        self.mode = mode
        self.model = torch.hub.load("facebookresearch/dinov2", f"dinov2_{variant}").eval().to(self.device)
        self.input = 224                                   # 16x16 patches at /14
        self._mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(self.device)
        self._std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(self.device)
        torch.set_grad_enabled(False)

    def embed(self, imgs):
        import torch
        import torch.nn.functional as F
        views = _to_model_views(imgs)
        t = torch.as_tensor(views).permute(0, 3, 1, 2).float().to(self.device) / 255.0
        t = F.interpolate(t, size=(self.input, self.input), mode="bilinear", align_corners=False)
        t = (t - self._mean) / self._std

        def enc(b):
            return self.model.forward_features(b)["x_norm_patchtokens"]        # [B,256,D]
        return self._batched(t, enc).cpu()

    def dist(self, fa, fb):
        import torch
        import torch.nn.functional as F
        if self.mode == "mse":
            return float(((fa - fb) ** 2).mean())
        return float(1.0 - F.cosine_similarity(fa, fb, dim=-1).mean())


class VIPL2(Candidate):
    """VIP (Value-Implicit Pre-training) embedding L2. Needs `pip install vip` (facebookresearch/vip)."""
    name = "vip_l2"

    def __init__(self, device="cpu"):
        import torch
        from vip import load_vip                            # raises ImportError if absent
        self.device = torch.device(device)
        m = load_vip()
        self.model = (m.module if hasattr(m, "module") else m).eval().to(self.device)
        torch.set_grad_enabled(False)

    def embed(self, imgs):
        import torch
        import torch.nn.functional as F
        views = _to_model_views(imgs)
        # VIP convention: [0,255] float, 224x224 (the repo's example transform).
        t = torch.as_tensor(views).permute(0, 3, 1, 2).float().to(self.device)
        t = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
        return self._batched(t, lambda b: self.model(b)).cpu()

    def dist(self, fa, fb):
        import torch
        return float(torch.norm(fa - fb))


class VJEPA21Tokens(Candidate):
    """V-JEPA 2.1 image-tokenizer token distance via nanowm's codec. Needs nanowm src + env vars."""
    name = "vjepa21"

    def __init__(self, device="cpu", nanowm_src=None, model_path=None, mode="mse"):
        import torch
        src = nanowm_src or os.environ.get("NANOWM_SRC")
        if src and src not in sys.path:
            sys.path.append(src)
        from latent_codecs.semantic import VJEPA21LatentCodec  # noqa: E402
        from latent_codecs.base import LatentShape             # noqa: E402
        mp = model_path or os.environ.get("VJEPA21_MODEL_PATH")
        if not mp:
            raise ImportError("VJEPA21_MODEL_PATH not set")
        self.device = torch.device(device)
        self.mode = mode
        # 16x16x1024 token grid at 256 input / 16px patches — nanowm's vjepa2_1 codec config.
        self.codec = VJEPA21LatentCodec(
            model_path=mp, latent_shape=LatentShape(1024, 16, 16),
            input_size=256, patch_size=16, precision="fp32").eval().to(self.device)
        torch.set_grad_enabled(False)

    def embed(self, imgs):
        import torch
        views = _to_model_views(imgs)
        t = torch.stack([preprocess_frame(v) for v in views]).to(self.device)  # [-1,1], codec denorms itself
        return self._batched(t, lambda b: self.codec.encode(b)).cpu()          # [B,1024,16,16]

    def dist(self, fa, fb):
        import torch.nn.functional as F
        if self.mode == "mse":
            return float(((fa - fb) ** 2).mean())
        # per-token cosine: [C,h,w] -> [h*w, C]
        ta, tb = fa.reshape(fa.shape[0], -1).T, fb.reshape(fb.shape[0], -1).T
        return float(1.0 - F.cosine_similarity(ta, tb, dim=-1).mean())


REGISTRY = {
    "pixel_l1": lambda device, kw: PixelL1(),
    "sdvae_l2": lambda device, kw: SDVAEL2(device=device),
    "dinov2_mse": lambda device, kw: Dinov2Patch(device=device, variant=kw.get("dinov2", "vits14"), mode="mse"),
    "dinov2_cos": lambda device, kw: Dinov2Patch(device=device, variant=kw.get("dinov2", "vits14"), mode="cos"),
    "vip_l2": lambda device, kw: VIPL2(device=device),
    "vjepa21": lambda device, kw: VJEPA21Tokens(device=device, nanowm_src=kw.get("nanowm_src"),
                                                model_path=kw.get("vjepa21_model_path")),
}

DEFAULT_SET = ["pixel_l1", "sdvae_l2", "dinov2_mse", "dinov2_cos"]


def build_candidates(names, device="cpu", **kw):
    """Instantiate candidates by name; optional deps that fail to import are skipped with a warning."""
    out = []
    for n in names:
        if n not in REGISTRY:
            print(f"[candidates] unknown candidate '{n}' — known: {sorted(REGISTRY)}")
            continue
        try:
            c = REGISTRY[n](device, kw)
            out.append(c)
            print(f"[candidates] {c.name} ready")
        except Exception as e:
            print(f"[candidates] {n} SKIPPED ({type(e).__name__}: {e})")
    return out
