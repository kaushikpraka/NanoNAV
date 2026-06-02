"""
Build the derived LeKiwi navigation dataset for NanoWM.

Source : kaushikpraka/wm-smallarea_merged   (LeRobot v3.0, 30 Hz, 9-D action, 3 cameras)
Output : a LeRobot **v2.1** dataset that NanoWM (pinned lerobot-datasets==2.1.0) reads natively.

What this does (one pass, single environment):
  * Reads the source RAW — tabular via pandas/pyarrow, video via PyAV (`av`). It does NOT use
    lerobot to read, so no v3.0/v2.1 version clash: only the WRITER needs lerobot 2.1.0.
  * Keeps full **30 Hz** (no temporal subsampling, no integration baked in). The (Δx, Δθ)
    integration happens later in the NanoWM dataloader (`integrate_se2`), so `frame_interval`
    stays a config knob. See context/nanowm-integration.md.
  * Keeps only the **top** camera (the elevated ~55° mount).
  * Slices the base action to 2-D and stores it in **SI units**:
        action = [ x.vel (m/s) , omega (rad/s) ]
    i.e. action[6] and deg2rad(action[8]) of the source. Drops the 6 arm joints and y.vel (strafe,
    identically 0). `observation.state` mirrors the action (a meaningful 2-D stand-in that satisfies
    lerobot's hard requirement on observation.state + its stats).

Run this ON THE POD (needs `lerobot==2.1.0` and `av`). Quick check first:
    python scripts/build_lekiwi_nav_dataset.py --limit 2 --dry-run
Full build:
    python scripts/build_lekiwi_nav_dataset.py --out-root $LEKIWI_DATA_ROOT [--push kaushikpraka/wm-smallarea_nav30]
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

# Source action indices (9-D LeKiwi): [6 arm, x.vel, y.vel, theta.vel]
IDX_VX, IDX_VTHETA = 6, 8
SRC_REPO = "kaushikpraka/wm-smallarea_merged"
CAMERA = "observation.images.top"
FPS = 30


def to_2d_action_si(action_9d: np.ndarray) -> np.ndarray:
    """[N,9] source action -> [N,2] = [x.vel (m/s), omega (rad/s)]. theta.vel is deg/s in source."""
    a = np.asarray(action_9d, dtype=np.float32)
    vx = a[:, IDX_VX]
    omega = np.deg2rad(a[:, IDX_VTHETA])
    return np.stack([vx, omega], axis=1).astype(np.float32)


def load_source_meta(cache: Path):
    """Download (if absent) + load the source tabular parquet and per-episode metadata."""
    from huggingface_hub import hf_hub_download

    cache.mkdir(parents=True, exist_ok=True)
    data_pq = cache / "data.parquet"
    ep_pq = cache / "episodes.parquet"
    if not data_pq.exists():
        hf_hub_download(SRC_REPO, "data/chunk-000/file-000.parquet", repo_type="dataset",
                        local_dir=cache, local_dir_use_symlinks=False)
        # hf_hub_download preserves the subpath; normalize to flat name
        (cache / "data/chunk-000/file-000.parquet").replace(data_pq)
    if not ep_pq.exists():
        hf_hub_download(SRC_REPO, "meta/episodes/chunk-000/file-000.parquet", repo_type="dataset",
                        local_dir=cache, local_dir_use_symlinks=False)
        (cache / "meta/episodes/chunk-000/file-000.parquet").replace(ep_pq)
    return pd.read_parquet(data_pq), pd.read_parquet(ep_pq)


def video_path_for(cache: Path, chunk_index: int, file_index: int) -> Path:
    """Download (if absent) the shared top-camera video file and return its local path."""
    from huggingface_hub import hf_hub_download

    rel = f"videos/{CAMERA}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    local = cache / "video" / f"chunk{chunk_index:03d}_file{file_index:03d}.mp4"
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        p = hf_hub_download(SRC_REPO, rel, repo_type="dataset", local_dir=cache,
                            local_dir_use_symlinks=False)
        Path(p).replace(local)
    return local


def episode_plan(ep_df: pd.DataFrame):
    """Yield per-episode build records grouped by the top-camera video file they live in.

    Frame offset within a shared file = round(from_timestamp * fps); verified against `length`.
    """
    ck = f"videos/{CAMERA}/chunk_index"
    fk = f"videos/{CAMERA}/file_index"
    tk = f"videos/{CAMERA}/from_timestamp"
    plan = []
    for _, row in ep_df.iterrows():
        plan.append(dict(
            episode=int(row["episode_index"]),
            length=int(row["length"]),
            data_from=int(row["dataset_from_index"]),
            data_to=int(row["dataset_to_index"]),
            vid_chunk=int(row[ck]),
            vid_file=int(row[fk]),
            frame_offset=int(round(float(row[tk]) * FPS)),
            task=(row["tasks"][0] if len(row["tasks"]) else "Drive around"),
        ))
    # group by (vid_chunk, vid_file), ordered by frame_offset for sequential streaming
    plan.sort(key=lambda r: (r["vid_chunk"], r["vid_file"], r["frame_offset"]))
    return plan


def iter_file_frames(video: Path, recs: list):
    """Decode a (shared) video file ONCE and dispatch frames to their episode.

    Yields (record, local_index, rgb_uint8_hwc). `recs` are the episodes living in this file;
    they may be contiguous (the LeKiwi case: all 50 in one file) or have gaps. Single pass,
    O(1) frame memory — avoids re-decoding the whole file per episode.
    """
    import av

    recs = sorted(recs, key=lambda r: r["frame_offset"])
    container = av.open(str(video))
    stream = container.streams.video[0]
    gi, ri = 0, 0
    cur = recs[0] if recs else None
    for frame in container.decode(stream):
        while cur is not None and gi >= cur["frame_offset"] + cur["length"]:
            ri += 1
            cur = recs[ri] if ri < len(recs) else None
        if cur is None:
            break
        if cur["frame_offset"] <= gi < cur["frame_offset"] + cur["length"]:
            yield cur, gi - cur["frame_offset"], frame.to_ndarray(format="rgb24")
        gi += 1
    container.close()
    if ri < len(recs) - 1 or (cur is not None and gi < cur["frame_offset"] + cur["length"]):
        raise RuntimeError(f"{video}: ran out of frames at global index {gi}")


def group_by_file(plan: list) -> dict:
    """Group episode records by their (vid_chunk, vid_file)."""
    files: dict = {}
    for r in plan:
        files.setdefault((r["vid_chunk"], r["vid_file"]), []).append(r)
    return files


def build(args):
    cache = Path(args.cache)
    data_df, ep_df = load_source_meta(cache)
    A = np.stack(data_df["action"].to_numpy())  # [N, 9]
    plan = episode_plan(ep_df)
    if args.limit:
        plan = plan[: args.limit]
    if args.episode_slice:
        a, b = args.episode_slice
        plan = plan[a:b]   # contiguous source-episode range, for parallel sharded builds

    # Sanity: frame_offset + length must stay within file & match tabular slice length.
    for r in plan:
        n_tab = r["data_to"] - r["data_from"]
        if n_tab != r["length"]:
            raise RuntimeError(f"ep {r['episode']}: tabular rows {n_tab} != length {r['length']}")
    print(f"{len(plan)} episodes; total frames = {sum(r['length'] for r in plan)}")

    files = group_by_file(plan)

    if args.dry_run:
        # Decode-and-count only — verifies the offset math + units without needing lerobot.
        counts = {r["episode"]: 0 for r in plan}
        for (c, f), recs in files.items():
            vid = video_path_for(cache, c, f)
            for rec, _i, _img in iter_file_frames(vid, recs):
                counts[rec["episode"]] += 1
        for r in plan:
            act = to_2d_action_si(A[r["data_from"]:r["data_to"]])
            ok = "ok" if counts[r["episode"]] == r["length"] else "MISMATCH"
            print(f"  ep {r['episode']:2d}: frames {counts[r['episode']]}/{r['length']} {ok}  "
                  f"vx[{act[:,0].min():.3f},{act[:,0].max():.3f}] m/s  "
                  f"w[{act[:,1].min():.3f},{act[:,1].max():.3f}] rad/s")
        assert all(counts[r["episode"]] == r["length"] for r in plan), "frame-count mismatch"
        print("dry-run OK")
        return

    if args.extract_frames:
        # Phase 1 of the parallel build: decode the source ONCE and dump per-episode frames as .npy
        # to a fast scratch dir (e.g. /dev/shm). The sharded encode then reads these via
        # --frames-cache, so libdav1d is NOT run 25x concurrently (which oversubscribes all cores and
        # makes the redundant decode the bottleneck). See merge_lekiwi_shards.py for the merge step.
        outd = Path(args.extract_frames)
        for (c, f), recs in files.items():
            vid = video_path_for(cache, c, f)
            for rec, i, img in iter_file_frames(vid, recs):
                d = outd / f"ep{rec['episode']:06d}"
                if i == 0:
                    d.mkdir(parents=True, exist_ok=True)
                np.save(d / f"frame_{i:06d}.npy", img)
                if i == rec["length"] - 1:
                    print(f"extracted ep {rec['episode']} ({rec['length']} frames)", flush=True)
        print("extract done")
        return

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        CAMERA: {"dtype": "video", "shape": [480, 640, 3], "names": ["height", "width", "channel"]},
        # Shapes are tuples: lerobot 0.3.x validate_frame compares value.shape (a tuple) with `!=`
        # against this, and a list [2] != tuple (2,) always fails. Tuples match.
        "action": {"dtype": "float32", "shape": (2,), "names": ["x.vel", "theta.vel"]},
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["x.vel", "theta.vel"]},
    }
    out_root = Path(args.out_root)
    ds = LeRobotDataset.create(
        repo_id=args.push or "local/wm-smallarea_nav30",
        fps=FPS,
        features=features,
        root=out_root,
        robot_type="lekiwi_client",
        use_videos=True,
    )

    acts = {r["episode"]: to_2d_action_si(A[r["data_from"]:r["data_to"]]) for r in plan}

    if args.frames_cache:
        # Phase 2: read pre-decoded frames (no libdav1d) so many shards can encode in parallel.
        fdir = Path(args.frames_cache)
        def frame_source():
            for r in plan:
                for i in range(r["length"]):
                    yield r, i, np.load(fdir / f"ep{r['episode']:06d}" / f"frame_{i:06d}.npy")
    else:
        def frame_source():
            for (c, f), recs in files.items():
                vid = video_path_for(cache, c, f)
                for rec, i, img in iter_file_frames(vid, recs):
                    yield rec, i, img

    for rec, i, img in frame_source():
        act = acts[rec["episode"]]
        # lerobot 0.3.x: add_frame(frame, task, ...) — `task` is a separate arg, NOT a frame key.
        ds.add_frame({
            CAMERA: img,                       # HWC uint8 RGB
            "action": act[i],
            "observation.state": act[i],       # mirror (2-D meaningful stand-in)
        }, task=rec["task"])
        if i == rec["length"] - 1:             # episode complete
            ds.save_episode()
            print(f"  wrote ep {rec['episode']} ({rec['length']} frames)")

    # lerobot 2.1.x finalizes via finalize() or consolidate() depending on build.
    for closer in ("finalize", "consolidate"):
        fn = getattr(ds, closer, None)
        if callable(fn):
            fn()
            break

    if args.push:
        ds.push_to_hub()
        print(f"pushed to {args.push}")
    print(f"done -> {out_root}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="data/lekiwi", help="local dir for the v2.1 dataset")
    ap.add_argument("--cache", default="data/_cache", help="source download cache")
    ap.add_argument("--push", default=None, help="HF repo id to push to (e.g. kaushikpraka/wm-smallarea_nav30)")
    ap.add_argument("--limit", type=int, default=0, help="only first N episodes (smoke test)")
    ap.add_argument("--episode-slice", type=int, nargs=2, default=None, metavar=("START", "END"),
                    help="build only plan[START:END] (contiguous source-episode range) for sharded builds")
    ap.add_argument("--extract-frames", default=None, metavar="DIR",
                    help="Phase 1: decode source once, dump per-episode frames as .npy to DIR (no lerobot)")
    ap.add_argument("--frames-cache", default=None, metavar="DIR",
                    help="Phase 2: read frames from DIR (.npy) instead of decoding — enables parallel sharded encode")
    ap.add_argument("--dry-run", action="store_true", help="decode+count+slice only; no lerobot write")
    build(ap.parse_args())
