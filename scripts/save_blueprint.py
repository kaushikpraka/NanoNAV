"""
Save the NanoNAV viewer blueprint to a .rbl file so it can be applied
when opening existing .rrd recordings locally.

Usage:
    python3 scripts/save_blueprint.py                        # writes viewer_blueprint.rbl
    python3 scripts/save_blueprint.py --out my_layout.rbl
    python3 scripts/save_blueprint.py --no-graph            # flat-planner layout (no route strip)

Then open a recording with it:
    rerun viewer_blueprint.rbl live_run2.rrd
"""
import argparse
import rerun as rr
import rerun.blueprint as rrb

def build_blueprint(graph_mode: bool = True, horizon: int = 3, flat: bool = False) -> rrb.Blueprint:
    views = [
        rrb.Spatial2DView(origin="model/live", name="camera (now)"),
        rrb.Spatial2DView(origin="imagined",   name="imagined +1 (executes next)"),
    ]
    for i in range(2, horizon + 1):
        views.append(rrb.Spatial2DView(
            origin=f"rollout/h{i}",
            name=f"imagined +{i}" + (" (CEM target)" if i == horizon else ""),
        ))
    views.append(rrb.Spatial2DView(
        origin="model/goal",
        name="target (waypoint)" if graph_mode else "goal",
    ))

    if flat:
        return rrb.Blueprint(rrb.Horizontal(*views), auto_views=False, collapse_panels=True)

    ts = [rrb.TimeSeriesView(
        origin="dist_to_goal",
        name="dist (to waypoint)" if graph_mode else "dist",
    )]
    if graph_mode:
        ts.append(rrb.TimeSeriesView(origin="graph_dist", name="route dist to GOAL"))

    rows = [rrb.Horizontal(*views)]
    shares = [3]
    if graph_mode:
        rows.append(rrb.Spatial2DView(origin="route", name="planned route -> goal"))
        shares.append(1.4)
    rows.append(ts[0] if len(ts) == 1 else rrb.Horizontal(*ts))
    shares.append(1)

    return rrb.Blueprint(
        rrb.Vertical(*rows, row_shares=shares),
        auto_views=False,
        collapse_panels=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out",      default="viewer_blueprint.rbl", help="output .rbl path")
    parser.add_argument("--no-graph", action="store_true", help="flat-planner layout (no route strip or graph_dist)")
    parser.add_argument("--horizon",  type=int, default=3, help="number of imagined frames (default 3)")
    parser.add_argument("--flat",     action="store_true", help="single-row image strip only")
    args = parser.parse_args()

    bp = build_blueprint(graph_mode=not args.no_graph, horizon=args.horizon, flat=args.flat)
    bp.save("nanonav_lekiwi_mpc", path=args.out)
    print(f"saved {args.out}")
    print(f"open with:  rerun {args.out} <recording.rrd>")
