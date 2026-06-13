# On-robot Rerun recordings

The `.rrd` files are Rerun recordings of on-robot MPC navigation runs (live `top` camera, planned
trajectories, imagined rollouts, distance-to-goal / graph-distance scalars, per-step action logs).
They are **not in git** — individual recordings run to ~0.5 GB and the full set is 5.6 GB, far over
GitHub's 100 MB/file limit. The curated keepers (successes + write-up A/B demos) are published as
**GitHub Release assets**; the full set lives on the pod's `/workspace/results` (persists across a
pod *stop*).

**Release:** https://github.com/KaushikTheProgrammer/NanoNAV/releases/tag/recordings-v1

**View:** `pip install rerun-sdk` then `rerun <file>.rrd`, or drag the file into
<https://rerun.io/viewer>.

## Keepers (in the release)

| File | What it shows |
|---|---|
| `mpc_semantic_graph_nearpurifier4.rrd` | **The headline.** First full on-robot graph success — REACHED nearpurifier, 129 steps, 40-hop route, `[tracked]` throughout, ENDGAME at step 116 closing 0.30→0.08. (2026-06-12) |
| `mpc_semantic_nograph_nearpurifier.rrd` | A/B baseline for the above — same goal **without** the graph: arrives from start-dist 0.35 (0.35→0.10 in 52 steps), floor ~0.10. Shows the basin-of-attraction limit the graph overcomes. |
| `mpc_semantic_graph_neardesk.rrd` | Graph run, full route + ENDGAME, but endgame **hovered ~0.30 for 35 steps** without crossing 0.08 — the goal-image-dependent endgame-floor open question (C1). |
| `mpc_semantic_graph_nearhamper3.rrd` | nearhamper graph run, 2026-06-13 session (stopped mid-run; did not reach — made route progress to step ~62 then regressed). |
| `mpc_semantic_nearchair1.rrd` | Semantic stack (no graph), 3/3 arrivals on nearchair1 — the goal that flat-L2 pixel-distance had failed. |
| `mpc_semantic_nearfan2.rrd` | Semantic stack arrival on nearfan2 (in-basin). |
| `mpc_nearfan2_execute.rrd` | **First-ever REACHED** (original SD-VAE WM, reach-thresh 35, 10 steps). |
| `mpc_nearfan2_thresh25.rrd` | Second SD-VAE REACHED (reach-thresh 25, 14 steps, sharp final dive into the basin). |

## Not released (remain on `/workspace/results`)

~35 additional recordings — earlier SD-VAE convergence attempts (`mpc_nearchair1_*`,
`mpc_nearfan_*`), camera/objective root-cause diagnostics (`mpc_drive_straight.rrd`,
`mpc_nearfan_execute_v2.rrd`), and superseded graph attempts (`mpc_semantic_graph_nearpurifier{,2,3}.rrd`,
`mpc_semantic_graph_nearhamper{,1,2}.rrd`). Grab any of these from the pod and add to the release if
needed.
