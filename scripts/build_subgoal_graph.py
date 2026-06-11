#!/usr/bin/env python3
"""
C3 step 1 — build the token-space subgoal graph from the DINO token cache.

Nodes  = real chunk-boundary frames (the token cache rows; ~4,500 from 50 episodes).
Edges  = temporal (consecutive chunks within an episode — CERTIFIED: the robot drove
         that hop, turns included) + shortcut (pairs that are far in time/episode but
         < tau in token-cosine — INFERRED welds that stitch the 50 threads into one map).
Metric = EXACT planner parity with lekiwi_engine._dist: tokens = lat.reshape(C,-1).T
         -> [256,384], d = 1 - mean per-token cosine. Scale-invariant, so the cache's
         latent_scale convention is irrelevant here.

tau is CALIBRATED, not guessed: within-episode pairs k chunks apart give the empirical
"k chunks of real driving = this much token-cos" curve; tau = the curve at --reach-chunks
(default 3 = one CEM plan, H=3). Too-large tau admits wormholes (visually-similar but
physically-distant welds, e.g. low-texture rug); too-small fragments the graph.

Wormhole audit (no ground-truth pose available): rank shortcuts by how far apart their
endpoints are in the TEMPORAL-ONLY graph — the highest-leverage welds are also the most
suspicious — and dump side-by-side frame-pair montages for human inspection.

Outputs (<out>/):
    graph.npz        nodes (episode, chunk_idx), edge arrays (i, j, w, kind), tau, config
    calibration.csv  k, n_pairs, p10, p25, median, p75, p90 token-cos at gap k
    report.md        graph stats: components, coverage, degree, audit verdict pointers
    audit/           montages: top-suspicion + random shortcut frame pairs
    embed.png        2D spectral embedding of the graph colored by episode (sanity viz)

Run (pod, GPU):
    /workspace/nanowm-venv/bin/python scripts/build_subgoal_graph.py \
        --cache /workspace/results/token_cache --out /workspace/results/subgoal_graph
"""

import argparse
import csv
import heapq
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_cache(cache_dir):
    meta = json.loads((Path(cache_dir) / "meta.json").read_text())
    lats = np.load(Path(cache_dir) / "latents.npy")            # [N,C,h,w]
    rows = list(csv.DictReader(open(Path(cache_dir) / "index.csv")))
    assert len(rows) == lats.shape[0], "index/latents mismatch"
    ep = np.array([int(r["episode"]) for r in rows])
    ck = np.array([int(r["chunk_idx"]) for r in rows])
    return lats, ep, ck, rows, meta


def token_distance_matrix(lats, device="cuda", block=512):
    """d(a,b) = 1 - mean_i cos(a_i, b_i) over aligned patch tokens (engine._dist parity)."""
    import torch
    N, C = lats.shape[0], lats.shape[1]
    t = torch.from_numpy(np.ascontiguousarray(lats)).to(device, torch.float32)
    t = t.reshape(N, C, -1).transpose(1, 2)                    # [N, hw, C] tokens
    t = torch.nn.functional.normalize(t, dim=-1)
    P = t.shape[1]
    D = torch.empty(N, N, dtype=torch.float32, device=device)
    for i in range(0, N, block):
        D[i:i + block] = 1.0 - torch.einsum("apd,bpd->ab", t[i:i + block], t) / P
    return D.cpu().numpy(), t


def calibrate(D, ep, ck, k_max=30):
    """Within-episode gap-k token-cos stats -> list of dict rows."""
    out = []
    order = np.lexsort((ck, ep))
    for k in range(1, k_max + 1):
        vals = []
        for e in np.unique(ep):
            idx = order[ep[order] == e]                        # cache rows of episode e, chunk order
            a, b = idx[:-k] if k else idx, idx[k:]
            ok = ck[b] - ck[a] == k                            # guard against gaps
            vals.append(D[a[ok], b[ok]])
        v = np.concatenate(vals)
        q = np.percentile(v, [10, 25, 50, 75, 90])
        out.append({"k": k, "n_pairs": len(v), "p10": q[0], "p25": q[1],
                    "median": q[2], "p75": q[3], "p90": q[4]})
    return out


def temporal_edges(D, ep, ck):
    edges = []
    order = np.lexsort((ck, ep))
    for a, b in zip(order[:-1], order[1:]):
        if ep[a] == ep[b] and ck[b] - ck[a] == 1:
            edges.append((int(a), int(b), float(D[a, b])))
    return edges


def shortcut_edges(D, ep, ck, tau, min_gap, max_deg):
    """Cross-episode (or same-episode loop-closure, |dchunk|>min_gap) pairs with d<tau.
    Per-node degree cap keeps junction neighborhoods from becoming cliques."""
    N = D.shape[0]
    cand = D < tau
    same = ep[:, None] == ep[None, :]
    near_t = same & (np.abs(ck[:, None] - ck[None, :]) <= min_gap)
    cand &= ~near_t                                            # temporal's job
    cand &= np.triu(np.ones_like(cand), k=1).astype(bool)
    ii, jj = np.nonzero(cand)
    order = np.argsort(D[ii, jj])                              # admit closest first
    deg = defaultdict(int)
    edges = []
    for t_ in order:
        i, j = int(ii[t_]), int(jj[t_])
        if deg[i] >= max_deg or deg[j] >= max_deg:
            continue
        deg[i] += 1
        deg[j] += 1
        edges.append((i, j, float(D[i, j])))
    return edges


def build_adj(n, *edge_lists):
    adj = defaultdict(list)
    for edges in edge_lists:
        for i, j, w in edges:
            w = max(w, 1e-6)
            adj[i].append((j, w))
            adj[j].append((i, w))
    return adj


def components(n, adj):
    seen = np.full(n, -1)
    comp = 0
    for s in range(n):
        if seen[s] >= 0:
            continue
        stack = [s]
        seen[s] = comp
        while stack:
            u = stack.pop()
            for v, _ in adj[u]:
                if seen[v] < 0:
                    seen[v] = comp
                    stack.append(v)
        comp += 1
    return seen, comp


def dijkstra_dist(adj, src, dst=None):
    dist = {src: 0.0}
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if dst is not None and u == dst:
            return d
        if d > dist.get(u, np.inf):
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist.get(v, np.inf):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist if dst is None else np.inf


def audit_montage(pairs, frames_dir, out_png, title):
    """Side-by-side frame pairs, one pair per grid cell."""
    from PIL import Image, ImageDraw
    cell_w, cell_h, pad = 320, 130, 6
    cols = 4
    rows = (len(pairs) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * (cell_w + pad) + pad,
                               rows * (cell_h + pad) + pad + 24), (24, 24, 24))
    dr = ImageDraw.Draw(canvas)
    dr.text((pad, 4), title, fill=(255, 255, 255))
    for n, (i, j, w, extra) in enumerate(pairs):
        r, c = divmod(n, cols)
        x0 = pad + c * (cell_w + pad)
        y0 = pad + 24 + r * (cell_h + pad)
        for s, idx in enumerate((i, j)):
            p = Path(frames_dir) / f"{idx:05d}.jpg"
            im = Image.open(p)
            im.thumbnail((cell_w // 2 - 2, cell_h - 14))
            canvas.paste(im, (x0 + s * (cell_w // 2), y0))
        dr.text((x0, y0 + cell_h - 13), f"{i}<->{j} d={w:.3f} {extra}", fill=(255, 210, 120))
    canvas.save(out_png)


def main():
    ap = argparse.ArgumentParser(description="build the C3 token-space subgoal graph")
    ap.add_argument("--cache", required=True, help="token cache dir (build_latent_cache.py output)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, default=None,
                    help="shortcut admission distance (default: calibration median at --reach-chunks)")
    ap.add_argument("--reach-chunks", type=int, default=3,
                    help="one CEM reach in chunks; sets tau from the calibration curve")
    ap.add_argument("--min-gap", type=int, default=5,
                    help="same-episode pairs closer than this many chunks are temporal-only")
    ap.add_argument("--max-shortcut-deg", type=int, default=8)
    ap.add_argument("--audit-n", type=int, default=24)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "audit"), exist_ok=True)
    lats, ep, ck, rows, meta = load_cache(args.cache)
    N = lats.shape[0]
    print(f"[graph] cache: {N} nodes, latent {tuple(lats.shape[1:])}, codec {meta.get('codec_kind')}")

    D, _ = token_distance_matrix(lats, args.device)
    print(f"[graph] D: [{N},{N}] min={D[~np.eye(N, dtype=bool)].min():.4f} "
          f"median={np.median(D):.4f} max={D.max():.4f}")

    cal = calibrate(D, ep, ck)
    with open(os.path.join(args.out, "calibration.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cal[0]))
        w.writeheader()
        w.writerows(cal)
    k_reach = next(r for r in cal if r["k"] == args.reach_chunks)
    tau = args.tau if args.tau is not None else float(k_reach["median"])
    print(f"[graph] calibration: k=1 median {cal[0]['median']:.4f} | "
          f"k={args.reach_chunks} median {k_reach['median']:.4f} (p25 {k_reach['p25']:.4f}) | "
          f"k=10 median {cal[9]['median']:.4f}  -> tau={tau:.4f}")

    t_edges = temporal_edges(D, ep, ck)
    s_edges = shortcut_edges(D, ep, ck, tau, args.min_gap, args.max_shortcut_deg)
    cross = sum(1 for i, j, _ in s_edges if ep[i] != ep[j])
    print(f"[graph] edges: {len(t_edges)} temporal, {len(s_edges)} shortcut "
          f"({cross} cross-episode, {len(s_edges) - cross} loop-closure)")

    adj_t = build_adj(N, t_edges)
    adj = build_adj(N, t_edges, s_edges)
    lab_t, n_t = components(N, adj_t)
    lab, n_full = components(N, adj)
    sizes = np.bincount(lab)
    big = sizes.max()
    print(f"[graph] components: temporal-only {n_t} (the 50 threads) -> full {n_full}; "
          f"largest covers {big}/{N} = {100 * big / N:.1f}%")

    # wormhole audit: shortcut leverage = endpoint distance in the temporal-only graph
    print("[graph] audit: ranking shortcuts by temporal-only endpoint separation...")
    lev = []
    for i, j, w in s_edges:
        if lab_t[i] != lab_t[j]:
            g = np.inf                                          # weld between threads
        else:
            g = dijkstra_dist(adj_t, i, j)
        lev.append((i, j, w, g))
    lev.sort(key=lambda x: (-(x[3] if np.isfinite(x[3]) else 1e9), x[2]))
    frames_dir = Path(args.cache) / "frames"
    if frames_dir.exists():
        top = [(i, j, w, "WELD" if not np.isfinite(g) else f"g={g:.2f}")
               for i, j, w, g in lev[:args.audit_n]]
        audit_montage(top, frames_dir, os.path.join(args.out, "audit", "top_leverage.png"),
                      f"TOP-LEVERAGE shortcuts (most suspicious; tau={tau:.3f}) — wormhole check: "
                      f"do both frames show the SAME place?")
        rng = np.random.default_rng(0)
        samp = [lev[k] for k in rng.choice(len(lev), size=min(args.audit_n, len(lev)),
                                           replace=False)]
        samp = [(i, j, w, "WELD" if not np.isfinite(g) else f"g={g:.2f}") for i, j, w, g in samp]
        audit_montage(samp, frames_dir, os.path.join(args.out, "audit", "random_sample.png"),
                      "RANDOM shortcut sample — typical weld quality")
        print(f"[graph] audit montages -> {args.out}/audit/")
    else:
        print("[graph] no frames/ in cache — rebuild cache with --save-frames for the audit")

    np.savez(os.path.join(args.out, "graph.npz"),
             episode=ep, chunk_idx=ck,
             frame_idx=np.array([int(r["frame_idx"]) for r in rows]),
             t_edges=np.array([(i, j, w) for i, j, w in t_edges], dtype=np.float64),
             s_edges=np.array([(i, j, w) for i, j, w in s_edges], dtype=np.float64),
             tau=tau, reach_chunks=args.reach_chunks,
             cache_dir=os.path.abspath(args.cache))

    deg = np.zeros(N, int)
    for i, j, _ in t_edges + s_edges:
        deg[i] += 1
        deg[j] += 1
    iso = int((deg == 0).sum())
    with open(os.path.join(args.out, "report.md"), "w") as f:
        f.write(f"""# C3 subgoal graph — build report

cache: `{args.cache}` ({N} nodes, codec {meta.get('codec_kind')})

## Calibration (within-episode token-cos at chunk gap k)
| k | median | p25 | p90 | n |
|---|--------|-----|-----|---|
""")
        for r in cal[:12]:
            f.write(f"| {r['k']} | {r['median']:.4f} | {r['p25']:.4f} | {r['p90']:.4f} | {r['n_pairs']} |\n")
        f.write(f"""
**tau = {tau:.4f}** (median at k={args.reach_chunks} chunks = one CEM reach{' — OVERRIDDEN' if args.tau else ''})

## Graph
- edges: **{len(t_edges)} temporal**, **{len(s_edges)} shortcut** ({cross} cross-episode welds, {len(s_edges) - cross} loop-closures)
- components: temporal-only {n_t} -> **{n_full} with shortcuts**; largest component {big}/{N} nodes ({100 * big / N:.1f}%)
- degree: median {int(np.median(deg))}, max {int(deg.max())}, isolated {iso}
- shortcut degree cap {args.max_shortcut_deg}; same-episode min gap {args.min_gap} chunks

## Wormhole audit
`audit/top_leverage.png` — shortcuts whose endpoints are FARTHEST in the temporal-only graph
(inter-thread WELDs first). These carry the routing; if any pair shows two different places,
that's a wormhole -> lower tau or raise the audit bar. `audit/random_sample.png` — typical quality.
""")
    print(f"[graph] wrote {args.out}/graph.npz, report.md, calibration.csv")


if __name__ == "__main__":
    main()
