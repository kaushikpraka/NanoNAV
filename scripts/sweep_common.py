"""
Phase-0 shared helpers for the distance-sweep stack (NanoNAV 6d).

Three consumers import this module (see context/learned-distance-metric.md "Sequencing"):
  - scripts/capture_sweep.py    (robot-side capture -> sweep dir with manifest)
  - scripts/dist_harness.py     (distance-agnostic grading of sweep dirs)
  - scripts/build_latent_cache.py / scripts/wm_imagined_arm.py (pod-side encoders)

It pins THREE contracts so every consumer agrees byte-for-byte:

1. LETTERBOX / PIXEL NORMALIZATION — identical to LekiwiPlanner._preprocess
   (external/nanowm/src/sample/lekiwi_engine.py) and the training transform
   (world_model_dataset.py): aspect-preserving bilinear resize (torch
   F.interpolate, align_corners=False) onto a black 256x256 canvas IN [0,1],
   THEN x*2-1. Pad value stays 0 in [0,1] (-> -1 after normalize), matching the
   dataset's order. The 6b pixel-range bug came from two code paths doing their
   own preprocessing — this module exists so there is exactly one.

2. SWEEP LABEL GRAMMAR — capture labels carry the ground-truth pose offset; the
   harness parses them back. Case-insensitive:
     r<cm>              radial displacement along the goal-facing axis (legacy '<N>cm' accepted)
     lat<+/-cm>         lateral offset, fixed heading                  e.g. lat+10, lat-20
     yaw<+/-deg>        heading offset AT the goal position            e.g. yaw+15 (legacy 'yaw+10')
     yawd<+/-deg>@r<cm> heading offset at radial distance              e.g. yawd+20@r40
     g_r<cm>_b<+/-deg>  polar-grid pose: distance + bearing, heading facing goal   e.g. g_r40_b-30
     fork_<site>_<move> fork-test endpoint; move in {start,straight,pivl,pivr,arcl,arcr}
                        'start' = the fork's reference pose            e.g. fork_a_arcl
     noise[<suffix>]    same-pose repeat (noise floor)
   Arms map 1:1 to the failure modes they grade — see learned-distance-metric.md
   "Evaluation" + the experiment-log 2026-06-09 research-session entry.

3. SWEEP DIR LAYOUT — a sweep dir contains:
     manifest.csv         idx,label,arm,frame_full,frame_model,imagined,note,t   (new tool)
       (legacy measurements.csv from measure_dist_sweep.py is auto-ingested:
        idx,label,latent_l2,pixel_l1,frame,t — `frame` is the model-view png)
     frames/...           the capture frames
     goal_model.png       256² letterboxed goal (legacy goal_modelview.png accepted)
     goal.png             full-res goal (optional)
"""

import csv
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

MODEL_SIZE = 256          # LekiwiPlanner image_size (train_cfg.model.image_size)
CHUNK_DT = 10.0 / 30.0    # f=10 @ 30 Hz — one action chunk (the metric's distance unit)


# --------------------------------------------------------------------------- images

def imread_rgb(path):
    """HWC uint8 RGB."""
    try:
        import imageio.v3 as iio
        return np.asarray(iio.imread(path))[..., :3]
    except Exception:
        from PIL import Image
        return np.asarray(Image.open(path).convert("RGB"))


def preprocess_unit(frame, size=MODEL_SIZE):
    """
    Raw HWC RGB (uint8 or float) -> torch float [3,size,size] in [0,1], letterbox-padded.
    EXACT torch replica of LekiwiPlanner._preprocess minus the final *2-1 (use
    preprocess_frame for the [-1,1] tensor the VAE/codec consumes).
    """
    import torch
    import torch.nn.functional as F

    t = torch.as_tensor(np.ascontiguousarray(frame))
    if t.ndim == 3 and t.shape[2] in (1, 3):          # HWC -> CHW
        t = t.permute(2, 0, 1)
    t = t.float()
    if float(t.max()) > 1.5:                          # uint8 -> [0,1]
        t = t / 255.0
    t = t.unsqueeze(0)                                # [1,C,H,W]
    _, _, H, W = t.shape
    if (H, W) != (size, size):
        scale = min(size / H, size / W)
        new_h, new_w = int(H * scale), int(W * scale)
        t = F.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)
        pad_h, pad_w = size - new_h, size - new_w
        pad_top, pad_left = pad_h // 2, pad_w // 2
        t = F.pad(t, (pad_left, pad_w - pad_left, pad_top, pad_h - pad_top), value=0.0)
    return t[0]                                       # [C,size,size] in [0,1]


def preprocess_frame(frame, size=MODEL_SIZE):
    """Raw HWC RGB -> torch float [3,size,size] in [-1,1] (the codec/WM input convention)."""
    return preprocess_unit(frame, size) * 2.0 - 1.0


def letterbox_rgb(frame, size=MODEL_SIZE):
    """Raw HWC RGB -> HWC uint8 letterboxed model view (what the VAE actually encodes)."""
    t = preprocess_unit(frame, size)
    return (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


# --------------------------------------------------------------------------- label grammar

_RE = [
    ("noise",       re.compile(r"^noise.*$"),                                lambda m: {}),
    ("yaw_at_dist", re.compile(r"^yawd(?P<yaw>[+-]?\d+\.?\d*)@r(?P<r>\d+\.?\d*)$"),
                    lambda m: {"yaw_deg": float(m["yaw"]), "r_cm": float(m["r"])}),
    ("yaw",         re.compile(r"^yaw(?P<yaw>[+-]?\d+\.?\d*)$"),             lambda m: {"yaw_deg": float(m["yaw"])}),
    ("lateral",     re.compile(r"^lat(?P<lat>[+-]?\d+\.?\d*)$"),             lambda m: {"lat_cm": float(m["lat"])}),
    ("grid",        re.compile(r"^g_r(?P<r>\d+\.?\d*)_b(?P<b>[+-]?\d+\.?\d*)$"),
                    lambda m: {"r_cm": float(m["r"]), "bearing_deg": float(m["b"])}),
    ("fork",        re.compile(r"^fork_(?P<site>[a-z0-9]+)_(?P<move>[a-z]+)$"),
                    lambda m: {"site": m["site"], "move": m["move"]}),
    ("radial",      re.compile(r"^r(?P<r>\d+\.?\d*)$"),                      lambda m: {"r_cm": float(m["r"])}),
    ("radial",      re.compile(r"^(?P<r>\d+\.?\d*)\s*cm$"),                  lambda m: {"r_cm": float(m["r"])}),  # legacy
]

FORK_MOVES = ("start", "straight", "pivl", "pivr", "arcl", "arcr")


def parse_label(label):
    """label -> (arm, params dict) or (None, {}) if unparseable."""
    s = (label or "").strip().lower().replace(" ", "")
    for arm, rx, fn in _RE:
        m = rx.match(s)
        if m:
            return arm, fn(m)
    return None, {}


# --------------------------------------------------------------------------- sweep dir IO

@dataclass
class Capture:
    idx: int
    label: str
    arm: Optional[str]
    params: Dict
    image_path: str            # model-view path if available, else full-res (harness letterboxes)
    is_model_view: bool
    imagined: bool = False     # True for WM-imagined frames (the 0d validation arm)
    latent_path: Optional[str] = None   # optional .npy [4,32,32] WM-space latent (imagined arm)


@dataclass
class Sweep:
    root: str
    goal_path: str
    goal_is_model_view: bool
    captures: List[Capture] = field(default_factory=list)
    goal_latent_path: Optional[str] = None

    def by_arm(self, arm, imagined=None):
        return [c for c in self.captures
                if c.arm == arm and (imagined is None or c.imagined == imagined)]


def _first_existing(root, names):
    for n in names:
        p = os.path.join(root, n)
        if os.path.exists(p):
            return p
    return None


def load_sweep(root):
    """Load a sweep dir (new manifest.csv, or legacy measurements.csv from measure_dist_sweep.py)."""
    goal_model = _first_existing(root, ["goal_model.png", "goal_modelview.png"])
    goal_full = _first_existing(root, ["goal.png"])
    goal = goal_model or goal_full
    if goal is None:
        raise FileNotFoundError(f"{root}: no goal_model.png / goal_modelview.png / goal.png")
    sweep = Sweep(root=root, goal_path=goal, goal_is_model_view=goal is goal_model)
    gl = _first_existing(root, ["goal_latent.npy"])
    sweep.goal_latent_path = gl

    man = _first_existing(root, ["manifest.csv"])
    legacy = _first_existing(root, ["measurements.csv"])
    if man:
        with open(man, newline="") as f:
            for row in csv.DictReader(f):
                arm, params = parse_label(row["label"])
                img = row.get("frame_model") or row.get("frame_full")
                if not img and not row.get("latent"):
                    print(f"[sweep] {root}: row idx={row.get('idx')} has no frame or latent — skipped")
                    continue
                is_model = bool(row.get("frame_model"))
                path = ""
                if img:
                    path = img if os.path.isabs(img) else os.path.join(root, img)
                lat = row.get("latent") or None
                if lat and not os.path.isabs(lat):
                    lat = os.path.join(root, lat)
                sweep.captures.append(Capture(
                    idx=int(row["idx"]), label=row["label"], arm=arm, params=params,
                    image_path=path, is_model_view=is_model,
                    imagined=str(row.get("imagined", "")).strip() in ("1", "true", "True"),
                    latent_path=lat))
    elif legacy:
        with open(legacy, newline="") as f:
            for row in csv.DictReader(f):
                arm, params = parse_label(row["label"])
                path = row["frame"]
                if not os.path.isabs(path):
                    path = os.path.join(root, path)
                if not os.path.exists(path):                     # pod-absolute paths -> rebase on frames/
                    alt = os.path.join(root, "frames", os.path.basename(row["frame"]))
                    if os.path.exists(alt):
                        path = alt
                sweep.captures.append(Capture(
                    idx=int(row["idx"]), label=row["label"], arm=arm, params=params,
                    image_path=path, is_model_view=True))        # legacy tool saved model-view frames
    else:
        raise FileNotFoundError(f"{root}: no manifest.csv or measurements.csv")

    n_bad = sum(1 for c in sweep.captures if c.arm is None)
    if n_bad:
        bad = sorted({c.label for c in sweep.captures if c.arm is None})
        print(f"[sweep] {root}: {n_bad} capture(s) with unparseable labels skipped from arms: {bad}")
    return sweep


MANIFEST_FIELDS = ["idx", "label", "arm", "frame_full", "frame_model", "imagined", "latent", "note", "t"]


class ManifestWriter:
    """Append-safe manifest.csv writer for capture tools."""

    def __init__(self, root):
        self.path = os.path.join(root, "manifest.csv")
        new = not os.path.exists(self.path)
        self._f = open(self.path, "a", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=MANIFEST_FIELDS)
        if new:
            self._w.writeheader()

    def add(self, idx, label, frame_full=None, frame_model=None, imagined=False,
            latent=None, note="", t=""):
        arm, _ = parse_label(label)
        self._w.writerow({"idx": idx, "label": label, "arm": arm or "",
                          "frame_full": frame_full or "", "frame_model": frame_model or "",
                          "imagined": int(bool(imagined)), "latent": latent or "",
                          "note": note, "t": t})
        self._f.flush()

    def close(self):
        self._f.close()
