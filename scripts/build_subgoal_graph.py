#!/usr/bin/env python3
"""
C3 step 1 — build the token-space subgoal graph from the DINO token cache.

Nodes  = real chunk-boundary frames (the token cache rows; ~4,500 from 50 episodes).
Edges  = temporal (consecutive chunks within an episode — CERTIFIED: the robot drove
         that hop, turns included) + shortcut (pairs that are far in time/episode but
         < tau in token-cosine — INFERRED welds that stitch the 50 threads into one map).
The graph is DIRECTED (operator catches 2026-06-11): temporal edges only in the driving
direction — the robot has no reverse (VX_MIN=0, no backward training data), so an
against-the-flow waypoint is unreachable for CEM's forward-only plans. Shortcut welds are
DIRECTION-CERTIFIED by motion parallax along the certified threads (second operator catch:
a weld at d<tau spans up to ~3 chunks of pose in ANY direction — bidirectional welds let
routes drift backward even with one-way temporal edges). Grades:
  ident  d < tau_id (~k=1: same pose within one chunk)        -> bidirectional, w = d
  fwd    a 1-3-chunk temporal successor of i gets CLOSER to j -> i->j only,     w = d
  soft   only the approach history certifies (pred moving     -> i->j only,
         toward i was farther from j); may be ~<=2 chunks        w = d + soft-penalty
         behind — recoverable by replan, but Dijkstra should     (prefer fwd routes)
         only use it when no fwd alternative exists
Connectivity is reported as STRONGLY-connected components + reach-to/from coverage.
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


def successor_chains(ep, ck, K=3):
    """succ/pred chains along temporal threads: S[k][i] = i's (k+1)-chunk successor (-1 = none)."""
    N = len(ep)
    succ = np.full(N, -1, int)
    pred = np.full(N, -1, int)
    order = np.lexsort((ck, ep))
    for a, b in zip(order[:-1], order[1:]):
        if ep[a] == ep[b] and ck[b] - ck[a] == 1:
            succ[a], pred[b] = b, a
    S, P = [succ], [pred]
    for _ in range(K - 1):
        S.append(np.where(S[-1] >= 0, succ[S[-1]], -1))
        P.append(np.where(P[-1] >= 0, pred[P[-1]], -1))
    return S, P


def shortcut_edges(D, ep, ck, tau, tau_id, margin, min_gap, max_deg, soft_penalty):
    """DIRECTED, direction-certified welds -> list of (i, j, w_effective, grade).

    Candidates: pairs far in time/episode with d < tau. Direction is certified by motion
    parallax along the certified threads (no pose ground truth needed):
      ident: d < tau_id (same pose within ~one chunk)            -> both directions, w=d
      fwd:   D[succ_k(i), j] < D[i,j] - margin for some k<=3     -> i->j, w=d
      soft:  D[pred_k(i), j] > D[i,j] + margin for some k<=3     -> i->j, w=d+soft_penalty
             (approach was converging on j; j may be ~<=2 chunks behind — Dijkstra-deprioritized)
    Admitted closest-first under per-node out/in degree caps."""
    N = D.shape[0]
    cand = D < tau
    same = ep[:, None] == ep[None, :]
    cand &= ~(same & (np.abs(ck[:, None] - ck[None, :]) <= min_gap))   # temporal's job
    cand &= np.triu(np.ones_like(cand), k=1).astype(bool)
    ii, jj = np.nonzero(cand)
    d_pair = D[ii, jj]
    S, P = successor_chains(ep, ck)

    def fwd_test(src, dst):
        base = D[src, dst]
        ok = np.zeros(len(src), bool)
        for Sk in S:
            s = Sk[src]
            v = s >= 0
            ok[v] |= D[s[v], dst[v]] < base[v] - margin
        return ok

    def soft_test(src, dst):
        base = D[src, dst]
        ok = np.zeros(len(src), bool)
        for Pk in P:
            p = Pk[src]
            v = p >= 0
            ok[v] |= D[p[v], dst[v]] > base[v] + margin
        return ok

    ident = d_pair < tau_id
    f_ij = fwd_test(ii, jj) & ~ident
    f_ji = fwd_test(jj, ii) & ~ident
    s_ij = soft_test(ii, jj) & ~ident & ~f_ij
    s_ji = soft_test(jj, ii) & ~ident & ~f_ji
    raw = []
    for a, b, mask, grade, pen in ((ii, jj, ident, "ident", 0.0), (jj, ii, ident, "ident", 0.0),
                                   (ii, jj, f_ij, "fwd", 0.0), (jj, ii, f_ji, "fwd", 0.0),
                                   (ii, jj, s_ij, "soft", soft_penalty),
                                   (jj, ii, s_ji, "soft", soft_penalty)):
        raw += [(int(x), int(y), float(D[x, y] + pen), grade)
                for x, y in zip(a[mask], b[mask])]
    raw.sort(key=lambda e: e[2])                               # admit closest (penalized) first
    outdeg, indeg = defaultdict(int), defaultdict(int)
    edges = []
    for i, j, w, grade in raw:
        if outdeg[i] >= max_deg or indeg[j] >= max_deg:
            continue
        outdeg[i] += 1
        indeg[j] += 1
        edges.append((i, j, w, grade))
    return edges


def build_adj(n, *edge_lists):
    adj = defaultdict(list)
    for edges in edge_lists:
        for i, j, w in edges:
            w = max(w, 1e-6)
            adj[i].append((j, w))
            adj[j].append((i, w))
    return adj


def build_directed_adj(t_edges, s_edges):
    """Runtime semantics: temporal one-way (driving direction); welds are already directed
    rows (ident pairs appear as two rows)."""
    adj = defaultdict(list)
    for i, j, w in t_edges:
        adj[i].append((j, max(w, 1e-6)))
    for e in s_edges:
        i, j, w = e[0], e[1], e[2]
        adj[i].append((j, max(w, 1e-6)))
    return adj


def scc(n, adj):
    """Iterative Kosaraju -> (labels, count), labels sorted by discovery."""
    order, seen = [], np.zeros(n, bool)
    for s in range(n):
        if seen[s]:
            continue
        seen[s] = True
        stack = [(s, 0)]
        while stack:
            u, k = stack[-1]
            if k < len(adj[u]):
                stack[-1] = (u, k + 1)
                v = adj[u][k][0]
                if not seen[v]:
                    seen[v] = True
                    stack.append((v, 0))
            else:
                order.append(u)
                stack.pop()
    radj = defaultdict(list)
    for u in range(n):
        for v, _ in adj[u]:
            radj[v].append(u)
    comp = np.full(n, -1)
    c = 0
    for s in reversed(order):
        if comp[s] >= 0:
            continue
        comp[s] = c
        stack = [s]
        while stack:
            u = stack.pop()
            for v in radj[u]:
                if comp[v] < 0:
                    comp[v] = c
                    stack.append(v)
        c += 1
    return comp, c


def reach_from(n, adj, seeds):
    seen = np.zeros(n, bool)
    seen[seeds] = True
    stack = list(seeds)
    while stack:
        u = stack.pop()
        for v, _ in adj[u]:
            if not seen[v]:
                seen[v] = True
                stack.append(v)
    return seen


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
    ap.add_argument("--tau-id", type=float, default=None,
                    help="same-pose identification radius (default: calibration median at k=1)")
    ap.add_argument("--margin", type=float, default=0.015,
                    help="motion-parallax certification margin for weld direction")
    ap.add_argument("--soft-penalty", type=float, default=0.15,
                    help="Dijkstra weight penalty on soft (pred-only certified) welds")
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

    tau_id = args.tau_id if args.tau_id is not None else float(cal[0]["median"])
    t_edges = temporal_edges(D, ep, ck)
    s_edges = shortcut_edges(D, ep, ck, tau, tau_id, args.margin, args.min_gap,
                             args.max_shortcut_deg, args.soft_penalty)
    cross = sum(1 for e in s_edges if ep[e[0]] != ep[e[1]])
    by_grade = {g: sum(1 for e in s_edges if e[3] == g) for g in ("ident", "fwd", "soft")}
    print(f"[graph] edges: {len(t_edges)} temporal (one-way), {len(s_edges)} directed welds "
          f"(ident {by_grade['ident']} / fwd {by_grade['fwd']} / soft {by_grade['soft']}; "
          f"{cross} cross-episode); tau_id={tau_id:.3f} margin={args.margin} "
          f"soft_penalty={args.soft_penalty}")

    adj_t = build_adj(N, t_edges)
    lab_t, n_t = components(N, adj_t)
    # directed semantics (runtime truth): temporal one-way, welds both ways
    dadj = build_directed_adj(t_edges, s_edges)
    radj = defaultdict(list)
    for u in range(N):
        for v, w in dadj[u]:
            radj[v].append((u, w))
    comp, n_scc = scc(N, dadj)
    sizes = np.bincount(comp)
    big_id = int(np.argmax(sizes))
    big = int(sizes[big_id])
    core = np.nonzero(comp == big_id)[0]
    to_core = reach_from(N, radj, core)      # nodes that can DRIVE TO the core
    from_core = reach_from(N, dadj, core)    # nodes the core can drive to
    print(f"[graph] DIRECTED connectivity: temporal-only {n_t} threads; {n_scc} SCCs; "
          f"largest SCC {big}/{N} = {100 * big / N:.1f}%; "
          f"can-reach-core {to_core.sum()}/{N} ({100 * to_core.mean():.1f}%), "
          f"core-can-reach {from_core.sum()}/{N} ({100 * from_core.mean():.1f}%)")

    # wormhole audit: shortcut leverage = endpoint distance in the temporal-only graph
    print("[graph] audit: ranking shortcuts by temporal-only endpoint separation...")
    lev, seen_pairs = [], set()
    for i, j, w, grade in s_edges:
        key = (min(i, j), max(i, j))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
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
             s_edges=np.array([(i, j, w) for i, j, w, _ in s_edges], dtype=np.float64),
             s_grade=np.array([g for _, _, _, g in s_edges]),
             directed_welds=True,
             tau=tau, tau_id=tau_id, margin=args.margin, soft_penalty=args.soft_penalty,
             reach_chunks=args.reach_chunks,
             cache_dir=os.path.abspath(args.cache))

    deg = np.zeros(N, int)
    for i, j, _ in t_edges:
        deg[i] += 1
        deg[j] += 1
    for i, j, _, _ in s_edges:
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

## Graph (DIRECTED: temporal = driving direction only; welds direction-certified)
- edges: **{len(t_edges)} temporal (one-way)**, **{len(s_edges)} directed welds** — ident {by_grade['ident']} (d<tau_id={tau_id:.3f}, bidirectional), fwd {by_grade['fwd']} (succ-certified ahead), soft {by_grade['soft']} (pred-only; +{args.soft_penalty} Dijkstra penalty); {cross} cross-episode
- strong connectivity: {n_t} temporal threads -> **{n_scc} SCCs**; largest SCC {big}/{N} nodes ({100 * big / N:.1f}%); can-reach-core {100 * to_core.mean():.1f}%, core-can-reach {100 * from_core.mean():.1f}%
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
