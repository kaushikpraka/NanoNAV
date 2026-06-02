"""
Merge sharded LeKiwi nav datasets (built in parallel via
`build_lekiwi_nav_dataset.py --episode-slice`) into one LeRobot v2.1 dataset.

Each shard is a standalone, valid v2.1 dataset whose episodes are numbered 0..n-1
locally. This concatenates the shards *in the given order* into one dataset:
  * videos are MOVED/COPIED as-is (no re-encode — that's the whole point of sharding),
  * each episode parquet has `episode_index` (-> global g) and `index` (-> global
    cumulative frame offset) rewritten; all other columns are byte-preserved,
  * meta/{episodes.jsonl, episodes_stats.jsonl, tasks.jsonl, info.json} are rebuilt.

Assumptions verified against a reference build: chunks_size=1000 (all 50 eps in
chunk-000), a single shared task (task_index always 0), no global meta/stats.json
(lerobot 0.3.3 aggregates per-episode stats on load).

    python scripts/merge_lekiwi_shards.py --shards _shards/shard_0 _shards/shard_1 ... \
        --out-root $LEKIWI_DATA_ROOT
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa

VIDEO_KEY = "observation.images.top"
DATA_REL = "data/chunk-000/episode_{:06d}.parquet"
VID_REL = "videos/chunk-000/{}/episode_{:06d}.mp4"


def read_jsonl(p: Path) -> list:
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(p: Path, rows: list):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def int_stat(values: np.ndarray) -> dict:
    """Per-episode stat block matching lerobot's format (lists of length 1)."""
    v = values.astype(np.float64)
    return {
        "min": [float(v.min())], "max": [float(v.max())],
        "mean": [float(v.mean())], "std": [float(v.std())],
        "count": [int(v.size)],
    }


def merge(shards: list, out_root: Path):
    out_root = Path(out_root)
    if out_root.exists():
        shutil.rmtree(out_root)
    (out_root / "data/chunk-000").mkdir(parents=True)
    (out_root / f"videos/chunk-000/{VIDEO_KEY}").mkdir(parents=True)
    (out_root / "meta").mkdir(parents=True)

    episodes_meta, episodes_stats = [], []
    g = 0                # global episode index
    frame_off = 0        # global cumulative frame offset (the `index` column)
    total_frames = 0

    for shard in shards:
        shard = Path(shard)
        local_eps = read_jsonl(shard / "meta/episodes.jsonl")
        local_stats = {s["episode_index"]: s for s in read_jsonl(shard / "meta/episodes_stats.jsonl")}
        for ep in local_eps:               # local episodes, in episode_index order
            l = ep["episode_index"]
            length = ep["length"]

            # --- parquet: rewrite episode_index (=g) and index (=frame_off + frame_index) ---
            t = pq.read_table(shard / DATA_REL.format(l))
            assert t.num_rows == length, f"{shard} ep{l}: rows {t.num_rows} != length {length}"
            frame_index = t.column("frame_index").to_numpy()
            t = t.set_column(t.schema.get_field_index("episode_index"),
                             "episode_index", pa.array(np.full(length, g, dtype=np.int64)))
            t = t.set_column(t.schema.get_field_index("index"),
                             "index", pa.array((frame_off + frame_index).astype(np.int64)))
            pq.write_table(t, out_root / DATA_REL.format(g))

            # --- video: copy as-is (no re-encode) ---
            shutil.copy2(shard / VID_REL.format(VIDEO_KEY, l),
                         out_root / VID_REL.format(VIDEO_KEY, g))

            # --- meta/episodes.jsonl ---
            episodes_meta.append({"episode_index": g, "tasks": ep["tasks"], "length": length})

            # --- meta/episodes_stats.jsonl: keep data-derived stats, refresh the two meta cols ---
            st = local_stats[l]["stats"]
            st["episode_index"] = int_stat(np.full(length, g))
            st["index"] = int_stat(np.arange(frame_off, frame_off + length))
            episodes_stats.append({"episode_index": g, "stats": st})

            g += 1
            frame_off += length
            total_frames += length

    # tasks.jsonl — single shared task, take it from the first shard
    shutil.copy2(Path(shards[0]) / "meta/tasks.jsonl", out_root / "meta/tasks.jsonl")
    tasks = read_jsonl(out_root / "meta/tasks.jsonl")

    write_jsonl(out_root / "meta/episodes.jsonl", episodes_meta)
    write_jsonl(out_root / "meta/episodes_stats.jsonl", episodes_stats)

    # info.json — start from shard 0, fix the totals
    info = json.load(open(Path(shards[0]) / "meta/info.json"))
    info["total_episodes"] = g
    info["total_frames"] = total_frames
    info["total_videos"] = g
    info["total_tasks"] = len(tasks)
    info["total_chunks"] = 1
    info["splits"] = {"train": f"0:{g}"}
    json.dump(info, open(out_root / "meta/info.json", "w"), indent=4)

    print(f"merged {g} episodes / {total_frames} frames -> {out_root}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True, help="shard roots, in global episode order")
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()
    merge(args.shards, Path(args.out_root))
