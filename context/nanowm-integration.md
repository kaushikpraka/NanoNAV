# NanoWM Integration

How `kaushikpraka/wm-smallarea_merged` plugs into NanoWM
(`github.com/simchowitzlabpublic/nano-world-model`). Read alongside [[action-representation]],
[[training]], and [[roadmap]].

## Two facts that shape everything

1. **NanoWM concatenates, it does not integrate.** With `frame_interval=f`, the dataloader stacks
   the *f* per-step actions into one model-frame action: `action_dim = raw_dim × f`
   (`src/wm_datasets/world_model_dataset.py`, the `self.action_dim` line and the
   `reshape(num_frames, frame_interval, -1).reshape(num_frames, -1)` block). Intermediate frames are
   skipped (`step=frame_interval`), so the model only predicts the chunk **endpoint**.
2. **Version mismatch.** NanoWM's loader uses the v2.x `episode_data_index["from"/"to"]` API, and
   our source data is **v3.0** → not loadable as-is. (The upstream `lerobot-datasets==2.1.0` pin is a
   *non-existent package* — "v2.1" is the LeRobot dataset *codec* version, not a release. The actual
   installable release with the right loader is **`lerobot==0.3.3`**, the only one exposing the
   `lerobot.datasets.lerobot_dataset` path AND `episode_data_index` AND writing CODEBASE_VERSION v2.1.)

## Decisions

- **Action = integrated body-frame `(Δx, Δθ)` (2-D)**, not the per-step concatenated vector. Chosen
  for CEM-friendliness (6-D search at H=3 vs up to 30-D) and physical match to the endpoint-prediction
  target. Rationale and the trade-off table are in [[action-representation]].
- **Source stays 30 Hz; the dataloader integrates.** Rather than baking a 6 Hz dataset, the derived
  dataset keeps raw base velocities at 30 Hz and an **`integrate_se2`** dataloader mode integrates over
  `frame_interval`. This makes `f` (the reach knob) a config change, not a rebuild, and keeps the
  concat baseline one flag away for A/B.

## The pipeline

**`scripts/build_lekiwi_nav_dataset.py`** (one pass solves v3.0→v2.1, camera, action slicing):
- `top` camera only (`observation.images.top`, the elevated ~55° mount).
- `action = [x.vel (idx 6), theta.vel (idx 8)]` — drop the 6 arm joints and `y.vel` (strafe).
- `observation.state = [x.vel, theta.vel]` (mirror; satisfies the loader's hard requirement on
  `observation.state` + `meta.stats`; `normalize_state=False`).
- Store native 480×640; NanoWM resizes to 256² with **`resize_mode: pad`** (letterbox) to preserve
  geometry. Output → `kaushikpraka/wm-smallarea_nav30` (LeRobot v2.1, fps=30).

**NanoWM patch (in the fork `KaushikTheProgrammer/nano-world-model`, submodule `external/nanowm`):**
- `world_model_dataset.py`: add `action_aggregation="concat"|"integrate_se2"` (default concat,
  backward-compatible) + `action_dt` (=1/30). In integrate mode: integrate **un-normalized**
  velocities via unicycle kinematics → `(Δx, Δθ)`, then normalize by integrated-delta stats
  (f-dependent → cache key includes `frame_interval`). `action_dim = 2` (no `× frame_interval`).
- `models/__init__.py` (the `action_dim = spec.action_dim * frame_interval` line): respect the mode —
  use `spec.action_dim` (=2) when `integrate_se2`, else the model and dataset disagree (10 vs 2).

**Configs (fork):** `configs/dataset/lerobot/lekiwi.yaml` (`frame_interval: 5`, `action_dim: 2`,
`image_key: observation.images.top`, `action_aggregation: integrate_se2`, `resize_mode: pad`) and
`configs/experiment/lekiwi_nav.yaml`. Reuse `model=nanowm_b2`, `latent_codec=sd_vae`.

**Train:** `python src/main.py experiment=lekiwi_nav dataset=lerobot/lekiwi model=nanowm_b2`.

## Validation caveat

The dataset has **no logged global pose** (`observation.state` is velocity). There is no independent
odometry to check the SE(2) integration against → validate by visual-flow consistency (SD-VAE
`compare` of frame *k* vs *k+f*). This revises the "verify integration vs odometry" item in
[[open-questions]].

## Realized on RunPod (Run 001, 2026-06-02)

What it actually took to get this training, beyond the planned patch (all committed to the fork
`main` + NanoNAV `main`):

- **lerobot 0.3.3 builder fixes:** `add_frame(frame, task=...)` (task is a separate arg, not a frame
  key) and tuple feature shapes (`validate_frame` compares `value.shape != declared`, so list `[2]`
  ≠ tuple `(2,)` always fails).
- **factory routing:** `factory.py` now routes dataset name `lekiwi` → `LeRobotDataSource` (it only
  knew `rt1`).
- **video backend = pyav:** `LeRobotDataSource` forces `video_backend="pyav"`; lerobot's default
  torchcodec binds the *system* FFmpeg (4.4 on Ubuntu 22.04) and fails intermittently on AV1.
- **action stats off the parquet:** `_load_single_trajectory` reads action/state from `hf_dataset`
  instead of `self.dataset[i]` (which decoded a video frame per index — action stats took ~47 min;
  now seconds).
- **env stack:** uv venv, `pytorch-lightning==2.5.2` (code uses PL 2.x), diffusers 0.32.2,
  transformers 4.46.3, hf-hub<1.0, torch/vision/codec +cu124. See [[runpod-setup]].
