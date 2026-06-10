#!/usr/bin/env python3
"""
Phase-0 distance-agnostic sweep grader (NanoNAV 6d — the Gate A rig).

Grades any candidate distance d(image, goal) on hand-placed ground-truth displacement sweeps
(scripts/capture_sweep.py dirs, or legacy measure_dist_sweep.py dirs). A candidate is a frozen
formula/network from scripts/dist_candidates.py — and later, the rung-0/1 LEARNED heads plug into
the SAME registry, so every metric ever built is graded on the same rig against the same arms.

Per candidate x arm it computes (definitions in learned-distance-metric.md "Evaluation"):
  noise        sigma = std of d over same-pose repeats (the denominator for every signal claim)
  radial /     Spearman rho of d vs displacement; far-band slope per 10cm / sigma;
  lateral /    monotonic-step fraction
  yaw_at_dist  (yaw_at_dist uses |yaw| at fixed r — grades "heading credited at distance")
  yaw          basin check: is argmin at 0°? basin depth (mean d at |yaw|>=20° minus d at 0) / sigma
  grid         rho of d vs r pooled across bearings (the single-basin field check, tabular form)
  fork         per site: does the geometrically-correct move (lowest-d) win? margin / sigma
               (the fork's correct move is the one labeled by capture protocol — by convention
                the operator places the BEST move's endpoint; we grade "is d's argmin that move")

GATE A (numeric, from the design doc): radial AND lateral rho > 0.9 out to 60 cm,
far-band (40-60 cm) slope per 10 cm > 3 sigma, yaw basin min at center with depth > 3 sigma.

Runs anywhere (Mac OK): GPU only speeds up the embed pass. matplotlib optional (plots skipped).

Examples:
  # grade the frozen arms on a captured sweep
  python scripts/dist_harness.py --sweep results/sweep_nearfan2 \
      --candidates pixel_l1,sdvae_l2,dinov2_mse,dinov2_cos --device cuda --out results/dist_harness

  # legacy dir from measure_dist_sweep.py works too
  python scripts/dist_harness.py --sweep /workspace/results/dist_sweep --candidates dinov2_mse

  # overlay a WM-imagined arm dir (scripts/wm_imagined_arm.py) against its clean counterpart
  python scripts/dist_harness.py --sweep results/sweep_nearfan2 --sweep results/sweep_nearfan2_imagined
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.append(_HERE)

from sweep_common import imread_rgb, load_sweep  # noqa: E402
from dist_candidates import DEFAULT_SET, build_candidates  # noqa: E402

FAR_BAND = (40.0, 60.0)     # cm — where raw-L2 plateaus (experiment-log 2026-06-09)
YAW_FAR = 20.0              # deg — "clearly misaligned" for basin depth
GATE_RHO = 0.9
GATE_SLOPE_SIGMA = 3.0


def spearman(x, y):
    if len(x) < 3:
        return float("nan")
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(x, y).statistic)
    except Exception:
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        if rx.std() == 0 or ry.std() == 0:
            return float("nan")
        return float(np.corrcoef(rx, ry)[0, 1])


def _group_mean(pairs):
    """[(x, d)...] -> sorted [(x, mean_d, std_d, n)] grouped by x."""
    by = defaultdict(list)
    for x, d in pairs:
        by[x].append(d)
    return sorted((x, float(np.mean(v)), float(np.std(v)), len(v)) for x, v in by.items())


def grade_axis(pairs, sigma, far_band=FAR_BAND):
    """pairs = [(displacement, d)...] with displacement >= 0. Returns axis metrics dict."""
    if not pairs:
        return {}
    xs = np.array([p[0] for p in pairs], float)
    ds = np.array([p[1] for p in pairs], float)
    out = {"n": len(pairs), "rho": spearman(xs, ds)}
    g = _group_mean(pairs)
    gx = np.array([r[0] for r in g]); gd = np.array([r[1] for r in g])
    # monotonic-step fraction over grouped means
    if len(gx) >= 2:
        steps = np.sign(np.diff(gd))
        out["mono_frac"] = float((steps > 0).sum() / len(steps))
    # far-band slope per 10 cm (least squares over points in band, endpoints included)
    m = (gx >= far_band[0] - 1e-9) & (gx <= far_band[1] + 1e-9)
    if m.sum() >= 2:
        slope = float(np.polyfit(gx[m], gd[m], 1)[0]) * 10.0
        out["far_slope_per10cm"] = slope
        out["far_slope_over_sigma"] = slope / sigma if (sigma and sigma > 0) else float("nan")
    # near/far dynamic range
    out["d_at_min_x"] = float(gd[0]); out["d_at_max_x"] = float(gd[-1])
    return out


def grade_yaw(pairs, sigma):
    """pairs = [(yaw_deg signed, d)...] -> basin metrics."""
    if not pairs:
        return {}
    g = _group_mean(pairs)
    yaws = np.array([r[0] for r in g]); ds = np.array([r[1] for r in g])
    i_min = int(np.argmin(ds))
    far = np.abs(yaws) >= YAW_FAR
    out = {"n": len(pairs),
           "argmin_yaw_deg": float(yaws[i_min]),
           "min_at_center": bool(abs(yaws[i_min]) <= 5.0)}
    if far.any():
        depth = float(ds[far].mean() - ds[np.abs(yaws) <= 5.0].min()) if (np.abs(yaws) <= 5.0).any() else float("nan")
        out["basin_depth"] = depth
        out["basin_depth_over_sigma"] = depth / sigma if (sigma and sigma > 0) else float("nan")
    return out


def grade_forks(captures, dists, sigma):
    """Fork sites: argmin move + margin. The 'start' capture is the reference (excluded from ranking)."""
    sites = defaultdict(dict)
    for c in captures:
        sites[c.params["site"]][c.params["move"]] = dists[c.idx]
    rows = []
    for site, moves in sorted(sites.items()):
        ranked = sorted(((d, m) for m, d in moves.items() if m != "start"))
        if len(ranked) < 2:
            continue
        best_d, best_m = ranked[0]
        margin = ranked[1][0] - best_d
        rows.append({"site": site, "winner": best_m, "margin": margin,
                     "margin_over_sigma": margin / sigma if (sigma and sigma > 0) else float("nan"),
                     "ranking": " < ".join(m for _, m in ranked)})
    return rows


def gate_a(metrics):
    """Apply the Gate A thresholds to one candidate's metrics dict. Returns (verdict, reasons)."""
    reasons = []
    ok = True

    def check(name, val, thresh=None):
        nonlocal ok
        if val is None or (isinstance(val, float) and np.isnan(val)):
            reasons.append(f"{name}: MISSING (arm not captured?)")
            ok = False
            return
        if isinstance(val, bool):
            reasons.append(f"{name}: {val} {'PASS' if val else 'FAIL'} (required)")
            ok = ok and val
        else:
            passed = val > thresh
            reasons.append(f"{name}: {val:.3f} {'PASS' if passed else 'FAIL'} (need >{thresh})")
            ok = ok and bool(passed)

    rad, lat, yaw = metrics.get("radial", {}), metrics.get("lateral", {}), metrics.get("yaw", {})
    check("radial rho", rad.get("rho"), GATE_RHO)
    check("radial far-slope/sigma", rad.get("far_slope_over_sigma"), GATE_SLOPE_SIGMA)
    check("lateral rho", lat.get("rho"), GATE_RHO)
    check("lateral far-slope/sigma", lat.get("far_slope_over_sigma"), GATE_SLOPE_SIGMA)
    check("yaw min at center", yaw.get("min_at_center"))
    check("yaw basin depth/sigma", yaw.get("basin_depth_over_sigma"), GATE_SLOPE_SIGMA)
    return ("PASS" if ok else "FAIL"), reasons


def run(sweeps, candidates, out_dir, plots=True):
    os.makedirs(out_dir, exist_ok=True)
    all_rows = []          # flat metrics.csv rows
    report = ["# Gate A report", ""]

    for sweep in sweeps:
        tag = os.path.basename(os.path.normpath(sweep.root))
        goal_img = imread_rgb(sweep.goal_path)
        caps = [c for c in sweep.captures if c.image_path]
        report.append(f"## sweep `{tag}` — {len(caps)} captures, goal `{os.path.basename(sweep.goal_path)}`")

        for cand in candidates:
            # ---- features: WM-latent rows load directly (no decode->re-encode roundtrip);
            #      everything else embeds its image once ----
            use_lat = getattr(cand, "feature_is_wm_latent", False)
            import torch
            lat_caps = [c for c in caps if use_lat and c.latent_path]
            img_caps = [c for c in caps if c not in lat_caps]
            feats = {}
            if img_caps:
                emb = cand.embed([imread_rgb(c.image_path) for c in img_caps])
                for i, c in enumerate(img_caps):
                    feats[c.idx] = emb[i]
            for c in lat_caps:
                feats[c.idx] = torch.from_numpy(np.load(c.latent_path)).float()
            if use_lat and sweep.goal_latent_path:
                goal_f = torch.from_numpy(np.load(sweep.goal_latent_path)).float()
            else:
                goal_f = cand.embed([goal_img])[0]
            dists = {c.idx: cand.dist(feats[c.idx], goal_f) for c in caps}

            # ---- noise floor ----
            noise = [dists[c.idx] for c in caps if c.arm == "noise"]
            sigma = float(np.std(noise)) if len(noise) >= 3 else None
            metrics = {"noise": {"n": len(noise), "sigma": sigma,
                                 "mean": float(np.mean(noise)) if noise else None}}

            # ---- axes (Gate A grades CLEAN captures only — imagined rows are the separate
            #      weld-validation arm below, never mixed into the gate) ----
            metrics["radial"] = grade_axis([(c.params["r_cm"], dists[c.idx]) for c in sweep.by_arm("radial", imagined=False)], sigma)
            metrics["lateral"] = grade_axis([(abs(c.params["lat_cm"]), dists[c.idx]) for c in sweep.by_arm("lateral", imagined=False)], sigma)
            metrics["yaw"] = grade_yaw([(c.params["yaw_deg"], dists[c.idx]) for c in sweep.by_arm("yaw", imagined=False)], sigma)
            # yaw-at-distance: grade like a yaw basin, per radius
            yd = defaultdict(list)
            for c in sweep.by_arm("yaw_at_dist", imagined=False):
                yd[c.params["r_cm"]].append((c.params["yaw_deg"], dists[c.idx]))
            for r_cm, pairs in sorted(yd.items()):
                metrics[f"yaw_at_{r_cm:g}cm"] = grade_yaw(pairs, sigma)
            metrics["grid"] = grade_axis([(c.params["r_cm"], dists[c.idx]) for c in sweep.by_arm("grid", imagined=False)], sigma)

            # ---- WM-imagined arm (informational, not gated): same-axis grade + the
            #      clean<->imagined weld = mean offset of imagined d above/below the clean
            #      radial curve at the same nominal displacement (loose-weld check, the
            #      learned-distance-metric.md VALIDATE-FIRST decision) ----
            img_rad = sweep.by_arm("radial", imagined=True)
            if img_rad:
                metrics["radial_imagined"] = grade_axis(
                    [(c.params["r_cm"], dists[c.idx]) for c in img_rad], sigma)
                clean_pts = sorted((c.params["r_cm"], dists[c.idx])
                                   for c in sweep.by_arm("radial", imagined=False))
                if len(clean_pts) >= 2:
                    cx = np.array([p[0] for p in clean_pts], dtype=float)
                    cy = np.array([p[1] for p in clean_pts], dtype=float)
                    offs = [dists[c.idx] - float(np.interp(c.params["r_cm"], cx, cy))
                            for c in img_rad]
                    metrics["radial_imagined"]["weld_offset_mean"] = float(np.mean(offs))
                    if sigma:
                        metrics["radial_imagined"]["weld_offset_over_sigma"] = float(np.mean(offs) / sigma)
            forks = grade_forks(sweep.by_arm("fork"), dists, sigma)

            verdict, reasons = gate_a(metrics)

            # ---- report ----
            report.append(f"\n### {cand.name}  —  **Gate A: {verdict}**")
            for r in reasons:
                report.append(f"- {r}")
            if sigma is not None:
                report.append(f"- noise sigma = {sigma:.4f} (n={len(noise)})")
            for arm, m in metrics.items():
                if arm == "noise" or not m:
                    continue
                kv = ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in m.items())
                report.append(f"- `{arm}`: {kv}")
                all_rows.append({"sweep": tag, "candidate": cand.name, "arm": arm,
                                 **{k: v for k, v in m.items()}})
            for fr in forks:
                report.append(f"- `fork:{fr['site']}`: winner={fr['winner']} margin/sigma="
                              f"{fr['margin_over_sigma']:.2f} ({fr['ranking']})")
                all_rows.append({"sweep": tag, "candidate": cand.name, "arm": f"fork_{fr['site']}", **fr})

            # ---- curves csv (one per candidate x sweep: the raw d values for plotting/overlay) ----
            cpath = os.path.join(out_dir, f"curve_{tag}_{cand.name}.csv")
            with open(cpath, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["idx", "label", "arm", "imagined", "d"])
                for c in caps:
                    w.writerow([c.idx, c.label, c.arm or "", int(c.imagined), f"{dists[c.idx]:.6f}"])

            if plots:
                _plot(out_dir, tag, cand.name, sweep, dists)
            print(f"[harness] {tag} / {cand.name}: Gate A {verdict}")

    # ---- flat metrics.csv ----
    if all_rows:
        keys = sorted({k for r in all_rows for k in r})
        with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_rows)
    with open(os.path.join(out_dir, "gate_report.md"), "w") as f:
        f.write("\n".join(report) + "\n")
    print(f"[harness] wrote {out_dir}/gate_report.md + metrics.csv + curve_*.csv")


def _plot(out_dir, tag, cand_name, sweep, dists):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    arms = [("radial", "r_cm", "cm"), ("lateral", "lat_cm", "cm (signed)"), ("yaw", "yaw_deg", "deg")]
    fig, axes = plt.subplots(1, len(arms), figsize=(4.5 * len(arms), 3.5))
    for ax, (arm, key, unit) in zip(np.atleast_1d(axes), arms):
        for imagined, style in ((False, "o-"), (True, "s--")):
            pts = sorted((c.params[key], dists[c.idx]) for c in sweep.by_arm(arm, imagined=imagined))
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], style,
                        label="imagined" if imagined else "clean")
        noise = [dists[c.idx] for c in sweep.by_arm("noise")]
        if noise:
            ax.axhspan(np.mean(noise) - np.std(noise), np.mean(noise) + np.std(noise),
                       alpha=0.2, label="noise band")
        ax.set_title(f"{arm}"); ax.set_xlabel(unit); ax.set_ylabel("d"); ax.legend(fontsize=7)
    fig.suptitle(f"{tag} — {cand_name}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"curves_{tag}_{cand_name}.png"), dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="grade distance candidates on displacement sweeps (Gate A)")
    ap.add_argument("--sweep", action="append", required=True,
                    help="sweep dir (repeatable; capture_sweep.py or legacy measure_dist_sweep.py layout)")
    ap.add_argument("--candidates", default=",".join(DEFAULT_SET),
                    help=f"comma list (default {','.join(DEFAULT_SET)}; also vip_l2,vjepa21)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dinov2", default="vits14", choices=["vits14", "vitb14"])
    ap.add_argument("--nanowm-src", default=None, help="for the vjepa21 candidate")
    ap.add_argument("--vjepa21-model-path", default=None)
    ap.add_argument("--out", default="results/dist_harness")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    sweeps = [load_sweep(s) for s in args.sweep]
    cands = build_candidates([c.strip() for c in args.candidates.split(",") if c.strip()],
                             device=args.device, dinov2=args.dinov2,
                             nanowm_src=args.nanowm_src, vjepa21_model_path=args.vjepa21_model_path)
    if not cands:
        sys.exit("[harness] no usable candidates")
    run(sweeps, cands, args.out, plots=not args.no_plots)


if __name__ == "__main__":
    main()
