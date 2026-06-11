#!/usr/bin/env python3
"""
C3 runtime — GraphNav: localize -> shortest-path tree -> hand CEM one basin at a time.

Library (imported by lekiwi_mpc.py with --graph) + offline route-validation CLI.

Key choices:
- The graph is DIRECTED: temporal edges are traversable ONLY in the driving direction
  (the robot has no reverse — VX_MIN=0, no backward data; an against-the-flow waypoint
  is behind the robot at a heading CEM's forward-only H=3 plans cannot re-acquire).
  Shortcut welds stay bidirectional: they are pose-identifications ("same place+heading",
  d < tau with the sharp yaw basin), not motion. Found 2026-06-11: the first undirected
  build routed chair->hamper backwards through episode threads (operator catch).
- ONE Dijkstra from the GOAL node at set_goal() time, run over the REVERSED edges, giving
  every node its true forward-drivable dist-to-goal + a next-hop tree. Each replan is then
  just a k-NN localization + a tree walk: no per-step graph search.
- The waypoint handed to CEM is the FURTHEST tree node within --lookahead of ROUTE
  PROGRESS (graph_dist[src] - graph_dist[node] < lookahead, default tau = one CEM reach).
  Route progress is measured in within-session calibrated units, so the ~+0.2 cross-session
  offset on live-frame distances (on-robot 2026-06-11 finding) cancels entirely; a
  live-distance lookahead would degenerate to 1-hop (~2 cm) crawling under that offset.
  Always >= 1 hop ahead so we never chase our own node.
- ENDGAME: when the live frame is within one hop of the goal node (graph dist < tau),
  waypoint() returns None -> the MPC falls back to the actual goal image, i.e. the final
  approach is exactly the on-robot-validated 3/3 behavior (reach-thresh semantics intact).

Offline validation (no robot): --route simulates the replan loop by teleporting to each
waypoint (assume CEM succeeds at its validated job) and dumps the waypoint filmstrip:
    /workspace/nanowm-venv/bin/python scripts/subgoal_graph.py \
        --graph /workspace/results/subgoal_graph \
        --ckpt <12k semantic ckpt> --start-row 0 --goal-image goals/nearchair1/goal.png \
        --out /workspace/results/subgoal_graph/route_nearchair_from_row0.png
"""

import argparse
import heapq
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.append(_HERE)


class GraphNav:
    def __init__(self, graph_dir, device="cuda"):
        import torch
        g = np.load(Path(graph_dir) / "graph.npz", allow_pickle=True)
        self.episode, self.chunk_idx = g["episode"], g["chunk_idx"]
        self.tau = float(g["tau"])
        self.cache_dir = str(g["cache_dir"])
        self.frames_dir = Path(self.cache_dir) / "frames"
        self.N = len(self.episode)
        self.adj = defaultdict(list)      # DIRECTED u->[(v,w)]: hops the robot can drive
        self.radj = defaultdict(list)     # reversed, for the goal-rooted Dijkstra
        for i, j, w in g["t_edges"]:      # temporal: FORWARD ONLY (no reverse on the base)
            i, j, w = int(i), int(j), max(float(w), 1e-6)
            self.adj[i].append((j, w))
            self.radj[j].append((i, w))
        for i, j, w in g["s_edges"]:      # welds: same (pose, heading) -> free both ways
            i, j, w = int(i), int(j), max(float(w), 1e-6)
            self.adj[i].append((j, w))
            self.adj[j].append((i, w))
            self.radj[i].append((j, w))
            self.radj[j].append((i, w))
        lats = np.load(Path(self.cache_dir) / "latents.npy")            # [N,C,h,w]
        self.C = int(lats.shape[1])
        t = torch.from_numpy(np.ascontiguousarray(lats)).to(device, torch.float32)
        t = t.reshape(self.N, self.C, -1).transpose(1, 2)               # [N,hw,C]
        self.tokens = torch.nn.functional.normalize(t, dim=-1)
        self.device = device
        self.goal_node = None
        self.dist_to_goal = None      # [N] graph distance
        self.next_hop = None          # [N] next node toward goal (-1 = goal/unreachable)
        print(f"[graphnav] {self.N} nodes, tau={self.tau:.3f}, tokens on {device}")

    # ---- metric (engine._dist parity) ----
    def _tok(self, lat):
        """[C,h,w] or channel-major flat [C*h*w] (engine rollout layout) -> normalized [hw,C]."""
        import torch
        t = torch.as_tensor(lat, dtype=torch.float32, device=self.device)
        t = t.reshape(self.C, -1).T                                     # [hw,C]
        return torch.nn.functional.normalize(t, dim=-1)

    def dists_to_all(self, lat):
        import torch
        q = self._tok(lat)                                              # [hw,C]
        return (1.0 - torch.einsum("pd,npd->n", q, self.tokens) / q.shape[0]).cpu().numpy()

    def localize(self, lat, k=1):
        d = self.dists_to_all(lat)
        idx = np.argsort(d)[:k]
        return (int(idx[0]), float(d[idx[0]])) if k == 1 else [(int(i), float(d[i])) for i in idx]

    # ---- goal ----
    def set_goal(self, goal_lat):
        """Localize the goal image into the graph; Dijkstra over the REVERSED edges from it ->
        forward-drivable dist-to-goal + next-hop tree (relaxing reversed u->v = original v->u,
        so nxt[v]=u is v's next forward hop)."""
        gn, gd = self.localize(goal_lat)
        dist = np.full(self.N, np.inf)
        nxt = np.full(self.N, -1, dtype=int)
        dist[gn] = 0.0
        pq = [(0.0, gn)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, w in self.radj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    nxt[v] = u                                          # v's next hop toward goal
                    heapq.heappush(pq, (nd, v))
        self.goal_node, self.dist_to_goal, self.next_hop = gn, dist, nxt
        reach = np.isfinite(dist).sum()
        print(f"[graphnav] goal -> node {gn} (ep{self.episode[gn]} ck{self.chunk_idx[gn]}, "
              f"d_loc={gd:.3f}); {reach}/{self.N} nodes can route to it")
        return gn, gd, int(reach)

    def node_frame(self, i):
        from PIL import Image
        return np.asarray(Image.open(self.frames_dir / f"{i:05d}.jpg"))

    # ---- the per-replan call ----
    def waypoint(self, live_lat, lookahead=None):
        """-> (node_id, frame_rgb, info) or (None, None, info) = ENDGAME (use the real goal image).
        info: src node, d_loc, graph dist-to-goal, hops_left, waypoint d_live."""
        assert self.goal_node is not None, "set_goal first"
        lookahead = lookahead or self.tau
        d_all = self.dists_to_all(live_lat)
        s = int(np.argmin(d_all))
        info = {"src": s, "d_loc": float(d_all[s]),
                "graph_dist": float(self.dist_to_goal[s]),
                "src_ep": int(self.episode[s]), "src_ck": int(self.chunk_idx[s])}
        if not np.isfinite(self.dist_to_goal[s]):
            info["status"] = "UNREACHABLE"
            return None, None, info
        path = [s]
        while path[-1] != self.goal_node:
            path.append(int(self.next_hop[path[-1]]))
        info["hops_left"] = len(path) - 1
        if len(path) <= 1 or self.dist_to_goal[s] < self.tau:
            info["status"] = "ENDGAME"
            return None, None, info
        wp = path[1]
        for n in path[2:]:                # furthest path node ~one CEM reach of ROUTE PROGRESS ahead
            if n == self.goal_node:
                break                     # the goal node itself -> endgame next localize
            if self.dist_to_goal[s] - self.dist_to_goal[n] < lookahead:
                wp = n
            else:
                break
        info.update(status="WAYPOINT", wp=wp, wp_d_live=float(d_all[wp]),
                    wp_ep=int(self.episode[wp]), wp_ck=int(self.chunk_idx[wp]),
                    wp_graph_dist=float(self.dist_to_goal[wp]))
        return wp, self.node_frame(wp), info


# ---------------- offline route validation CLI ----------------

def _filmstrip(images, labels, out_png, title, cols=8):
    from PIL import Image, ImageDraw
    cw, chh, pad = 180, 150, 4
    rows = (len(images) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * (cw + pad) + pad, rows * (chh + pad) + pad + 22), (24, 24, 24))
    dr = ImageDraw.Draw(canvas)
    dr.text((pad, 4), title, fill=(255, 255, 255))
    for n, (im, lab) in enumerate(zip(images, labels)):
        r, c = divmod(n, cols)
        x0, y0 = pad + c * (cw + pad), pad + 22 + r * (chh + pad)
        p = Image.fromarray(im)
        p.thumbnail((cw, chh - 14))
        canvas.paste(p, (x0, y0))
        dr.text((x0, y0 + chh - 13), lab, fill=(160, 220, 255))
    canvas.save(out_png)
    print(f"[route] filmstrip -> {out_png}")


def main():
    ap = argparse.ArgumentParser(description="GraphNav offline route validation")
    ap.add_argument("--graph", required=True)
    ap.add_argument("--ckpt", required=True, help="WM ckpt (its codec encodes the start/goal images)")
    ap.add_argument("--nanowm-src", default="external/nanowm/src")
    ap.add_argument("--start-image", default=None)
    ap.add_argument("--start-row", type=int, default=None, help="use a cache row as the start instead")
    ap.add_argument("--goal-image", required=True)
    ap.add_argument("--lookahead", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from build_latent_cache import CkptEncoder
    from sweep_common import preprocess_frame
    from PIL import Image

    nav = GraphNav(args.graph, args.device)
    enc = CkptEncoder(args.ckpt, args.nanowm_src, args.device)

    def encode_img(path):
        rgb = np.asarray(Image.open(path).convert("RGB"))
        return enc.encode(preprocess_frame(rgb, enc.image_size).unsqueeze(0))[0].numpy(), rgb

    goal_lat, goal_rgb = encode_img(args.goal_image)
    nav.set_goal(goal_lat)

    if args.start_row is not None:
        lats = np.load(Path(nav.cache_dir) / "latents.npy")
        cur_lat, cur_rgb = lats[args.start_row].astype(np.float32), nav.node_frame(args.start_row)
        start_lab = f"start=row{args.start_row}"
    else:
        cur_lat, cur_rgb = encode_img(args.start_image)
        start_lab = f"start={Path(args.start_image).stem}"

    # simulate the replan loop: teleport to each waypoint (CEM assumed to do its validated job)
    lats = np.load(Path(nav.cache_dir) / "latents.npy")
    images, labels = [cur_rgb], [start_lab]
    for step in range(args.max_steps):
        wp, frame, info = nav.waypoint(cur_lat, args.lookahead)
        if info["status"] in ("ENDGAME", "UNREACHABLE"):
            print(f"[route] step {step}: {info['status']} at src={info['src']} "
                  f"(graph_dist={info['graph_dist']:.3f}, hops={info.get('hops_left', '-')})")
            break
        images.append(frame)
        labels.append(f"wp {info['wp']} ep{info['wp_ep']}ck{info['wp_ck']} "
                      f"g={info['wp_graph_dist']:.2f}")
        cur_lat = lats[wp].astype(np.float32)
        print(f"[route] step {step}: src={info['src']} -> wp={wp} "
              f"(d_live={info['wp_d_live']:.3f}, hops_left={info['hops_left']}, "
              f"graph_dist_left={info['wp_graph_dist']:.3f})")
    else:
        print(f"[route] WARNING: no endgame after {args.max_steps} waypoint hops")
    images.append(goal_rgb)
    labels.append("GOAL image")
    _filmstrip(images, labels, args.out,
               f"{start_lab} -> {Path(args.goal_image).parent.name or Path(args.goal_image).stem}: "
               f"{len(images) - 2} waypoints, then ENDGAME (real goal image to CEM)")


if __name__ == "__main__":
    main()
