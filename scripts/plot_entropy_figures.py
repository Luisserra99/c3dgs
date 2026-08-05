#!/usr/bin/env python3
"""Regenerate the entropy-vs-quality figures and fit the empirical model
Q_drop = A * (H / log2 K) + B used for automatic codebook-size selection.

Input is the CSV produced by scripts/extract_metrics.py over the codebook-size
sweep (one run per scene per K, K in {4, ..., 4096}); the rows must carry
entropy_mean and color_codebook_size, which compress.py now always writes.

For each metric the script:
  1. normalizes each scene's value by that scene's K = --baseline-k value,
  2. fits the linear model by ordinary least squares over K >= --min-k,
  3. reports A, B, R2, 95% confidence intervals, the slope p-value under both
     naive and scene-clustered standard errors, and the within-scene R2
     (printed and saved to fit_results.json),
  4. draws a two-panel publication figure (single-column width, >= 9 pt fonts,
     colorblind-safe Okabe-Ito palette, one marker shape per dataset):
       (a) scatter + fit line + 95% confidence and prediction bands,
       (b) box plots of the same normalized metric grouped by K, over every
           scene and seed, which is what shows the run-to-run spread.

The clustered standard errors matter because the points are not independent:
each scene contributes one point per K, so the residuals are correlated within
a scene. The naive OLS p-value overstates the evidence; the cluster-robust one
(13 scenes = 13 clusters) does not. The within-scene R2 separates the part of
the trend that holds inside a scene from the between-scene level differences
that dilute the pooled R2.

Usage:

    python scripts/plot_entropy_figures.py metrics.csv -o figures/
    python scripts/plot_entropy_figures.py metrics.csv -o figures/ \
        --metrics LPIPS SSIM --min-k 16

Needs scipy for exact t quantiles and p-values -- the c3dgs conda env does not
have it, so run with /usr/bin/python3.

Outputs entropy_vs_<metric>.pdf (vector, for the paper) and .png per metric.
"""

import argparse
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None

# Okabe-Ito colorblind-safe hues, kept for any dataset-level plot.
DATASET_STYLE = {
    "MipNeRF360":      {"color": "#0072B2", "marker": "o"},
    "TanksAndTemples": {"color": "#E69F00", "marker": "s"},
    "DeepBlending":    {"color": "#009E73", "marker": "^"},
}
INK = "#333333"
METRIC_COLUMNS = {"LPIPS": "LPIPS", "SSIM": "SSIM", "PSNR": "PSNR", "size": "size_MB"}

# One colour and one marker per scene, so each scene reads as its own
# trajectory. 13 scenes exceed any single colourblind-safe palette, so the
# marker shape carries the identity as well as the hue. Only saturated hues are
# used: tab20's alternating light tints wash out at print size.
SCENE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#393b79", "#843c39",
    "#7b4173",
]
MARKER_CYCLE = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "p", "h", "*", "d"]


def scene_styles(scenes):
    return {
        scene: {"color": SCENE_COLORS[i % len(SCENE_COLORS)],
                "marker": MARKER_CYCLE[i % len(MARKER_CYCLE)]}
        for i, scene in enumerate(sorted(scenes))
    }


def set_font_size(size):
    """Scale every text element off one base size."""
    plt.rcParams.update({
        "font.size": size,
        "axes.labelsize": size,
        "axes.titlesize": size,
        "legend.fontsize": size * 0.8,
        "xtick.labelsize": size * 0.9,
        "ytick.labelsize": size * 0.9,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
    })


def load_sweep(csv_path, methods=None):
    """-> {(scene, K): {"dataset", "entropy", metric: [one value per seed]}}

    Per-seed values are kept rather than averaged: the box-plot panel needs the
    full spread, and the fit averages them itself.

    `methods` restricts which method labels contribute. This matters: a CSV that
    also holds the ablations has several methods at K = --baseline-k, and pooling
    them would build the normalization reference out of runs the sweep never
    used, silently biasing every ratio. Contributors per K are printed so a
    mismatch is visible.
    """
    acc = defaultdict(lambda: defaultdict(list))
    info = {}
    contributors = defaultdict(set)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("color_codebook_size") or not row.get("entropy_mean"):
                continue
            if methods and row["method"] not in methods:
                continue
            key = (row["scene"], int(row["color_codebook_size"]))
            info[key] = {"dataset": row["dataset"], "entropy": float(row["entropy_mean"])}
            contributors[int(row["color_codebook_size"])].add(row["method"])
            for metric, col in METRIC_COLUMNS.items():
                if row.get(col):
                    acc[key][metric].append(float(row[col]))
    for k in sorted(contributors):
        print(f"  K={k:<5d} from method(s): {', '.join(sorted(contributors[k]))}")
    out = {}
    for key, metrics in acc.items():
        out[key] = dict(info[key])
        for metric, vals in metrics.items():
            out[key][metric] = list(vals)
    return out


def build_points(sweep, metric, baseline_k, min_k, max_k):
    """Seed-averaged points, one per (scene, K).

    -> x = H/log2(K), y = metric normalized by the scene's baseline, plus the
    scene label (needed to cluster the standard errors) and the K value.

    The baseline K itself is included when it falls inside [min_k, max_k]: it
    contributes the anchor y = 1 that pins the right-hand end of every scene's
    trajectory. Excluding it would drop a real, measured operating point.
    """
    xs, ys, datasets, scenes_out, ks = [], [], [], [], []
    scenes = {scene for scene, _ in sweep}
    for scene in sorted(scenes):
        base = sweep.get((scene, baseline_k))
        if base is None or metric not in base:
            print(f"  warning: scene '{scene}' has no K={baseline_k} baseline, skipped")
            continue
        base_val = float(np.mean(base[metric]))
        for (s, k), entry in sweep.items():
            if s != scene or metric not in entry:
                continue
            if not (min_k <= k <= max_k):
                continue
            xs.append(entry["entropy"] / math.log2(k))
            ys.append(float(np.mean(entry[metric])) / base_val)
            datasets.append(entry["dataset"])
            scenes_out.append(scene)
            ks.append(k)
    return (np.array(xs), np.array(ys), datasets,
            np.array(scenes_out), np.array(ks))


def build_boxes(sweep, metric, baseline_k, max_k):
    """-> (sorted K values, per-K list of normalized values over scenes x seeds).

    Every seed is kept, so each box summarizes n_scenes * n_seeds runs.
    """
    per_k = defaultdict(list)
    scenes = {scene for scene, _ in sweep}
    for scene in sorted(scenes):
        base = sweep.get((scene, baseline_k))
        if base is None or metric not in base:
            continue
        base_val = float(np.mean(base[metric]))
        for (s, k), entry in sweep.items():
            if s != scene or metric not in entry or k > max_k:
                continue
            per_k[k].extend(v / base_val for v in entry[metric])
    ks = sorted(per_k)
    return ks, [per_k[k] for k in ks]


def fit_linear(x, y, groups=None):
    """OLS y = A*x + B with R2, 95% CIs, slope p-values and within-group R2.

    `groups` labels the cluster each point belongs to (here: the scene). When
    given, cluster-robust (CR1) standard errors and a within-group R2 are added.
    """
    n = len(x)
    A, B = np.polyfit(x, y, 1)
    resid = y - (A * x + B)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    dof = n - 2
    s2 = ss_res / dof if dof > 0 else float("nan")
    sxx = float(np.sum((x - np.mean(x)) ** 2))
    se_A = math.sqrt(s2 / sxx)
    se_B = math.sqrt(s2 * (1.0 / n + np.mean(x) ** 2 / sxx))
    t_crit = scipy_stats.t.ppf(0.975, dof) if scipy_stats is not None else 1.96
    t_A = A / se_A
    out = {
        "A": float(A), "B": float(B), "R2": r2, "n": n,
        "A_ci95": [float(A - t_crit * se_A), float(A + t_crit * se_A)],
        "B_ci95": [float(B - t_crit * se_B), float(B + t_crit * se_B)],
        "A_se": float(se_A), "A_t": float(t_A),
        "A_p": float(2 * scipy_stats.t.sf(abs(t_A), dof)) if scipy_stats else float("nan"),
        "RMSE": math.sqrt(s2),
        "_internals": (s2, sxx, float(np.mean(x)), t_crit),
    }

    if groups is not None:
        uniq = np.unique(groups)
        n_c = len(uniq)
        design = np.column_stack([x, np.ones(n)])
        xtx_inv = np.linalg.inv(design.T @ design)
        meat = np.zeros((2, 2))
        for g in uniq:
            sel = groups == g
            u = design[sel].T @ resid[sel]
            meat += np.outer(u, u)
        # CR1 small-sample correction, the usual default for clustered OLS
        scale = (n_c / (n_c - 1)) * ((n - 1) / (n - 2))
        v_clu = scale * xtx_inv @ meat @ xtx_inv
        se_A_c = math.sqrt(v_clu[0, 0])
        t_A_c = A / se_A_c
        t_crit_c = scipy_stats.t.ppf(0.975, n_c - 1) if scipy_stats is not None else 1.96
        out["n_clusters"] = int(n_c)
        out["A_se_cluster"] = float(se_A_c)
        out["A_ci95_cluster"] = [float(A - t_crit_c * se_A_c), float(A + t_crit_c * se_A_c)]
        out["A_p_cluster"] = (
            float(2 * scipy_stats.t.sf(abs(t_A_c), n_c - 1)) if scipy_stats else float("nan")
        )

        # Within-group (scene fixed-effects) fit: demean x and y per scene so
        # only the inside-a-scene trend remains.
        xw, yw = x.astype(float).copy(), y.astype(float).copy()
        for g in uniq:
            sel = groups == g
            xw[sel] -= x[sel].mean()
            yw[sel] -= y[sel].mean()
        a_w = float(np.polyfit(xw, yw, 1)[0])
        rw = yw - a_w * xw
        out["A_within"] = a_w
        out["R2_within"] = float(1.0 - (rw @ rw) / (yw @ yw)) if yw @ yw > 0 else float("nan")

        slopes = [float(np.polyfit(x[groups == g], y[groups == g], 1)[0])
                  for g in uniq if (groups == g).sum() > 2]
        out["per_scene_slopes"] = {"mean": float(np.mean(slopes)),
                                   "sd": float(np.std(slopes, ddof=1)),
                                   "min": float(np.min(slopes)),
                                   "max": float(np.max(slopes)),
                                   "n_same_sign": int(np.sum(np.sign(slopes) == np.sign(A)))}
    return out


def plot_metric(metric, x, y, scenes, ks, fit, box_ks, box_vals,
                out_stem, tolerance=None, figwidth=14.0, annotate_k=True,
                ymin=None, ymax=None):
    """Two stacked panels.

    (a) one trajectory per scene -- each scene's measured operating points
        joined in order of K and annotated with it -- over the fitted model,
        so a reader can see both the population trend and how far individual
        scenes sit from it. The fit itself uses the seed-mean of each
        (scene, K) cell, one point per cell.
    (b) box plots per K over every scene and seed, which is what shows the
        run-to-run dispersion behind those means.
    """
    styles = scene_styles(set(scenes))
    fig, (ax, axb) = plt.subplots(
        2, 1, figsize=(figwidth, figwidth * 0.96), constrained_layout=True,
        gridspec_kw={"height_ratios": [1.6, 1.0]},
    )

    # ---------------- panel (a): per-scene trajectories + fit ---------------
    ax.grid(True, color="#DDDDDD", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_axisbelow(True)

    s2, sxx, x_mean, t_crit = fit["_internals"]
    pad = 0.03 * (x.max() - x.min())
    grid = np.linspace(x.min() - pad, x.max() + pad, 200)
    line = fit["A"] * grid + fit["B"]
    se_mean = np.sqrt(s2 * (1.0 / fit["n"] + (grid - x_mean) ** 2 / sxx))
    ax.fill_between(grid, line - t_crit * se_mean, line + t_crit * se_mean,
                    color="#BBBBBB", alpha=0.45, linewidth=0, zorder=2)

    ax.axhline(1.0, color="#999999", linestyle="--", linewidth=1.4, zorder=1,
               label=f"baseline ($K={fit['baseline_k']}$)")

    for scene in sorted(set(scenes)):
        sel = np.array([s == scene for s in scenes])
        order = np.argsort(x[sel])
        xs, ys, kk = x[sel][order], y[sel][order], np.array(ks)[sel][order]
        st = styles[scene]
        ax.plot(xs, ys, color=st["color"], linewidth=1.6, alpha=0.85, zorder=3)
        ax.plot(xs, ys, linestyle="none", marker=st["marker"], markersize=10,
                markerfacecolor=st["color"], markeredgecolor=INK,
                markeredgewidth=0.6, zorder=4, label=scene)
        if annotate_k:
            for xi, yi, ki in zip(xs, ys, kk):
                ax.annotate(str(ki), (xi, yi), textcoords="offset points",
                            xytext=(4, 4), fontsize=plt.rcParams["font.size"] * 0.5,
                            color=st["color"], zorder=5)

    ax.plot(grid, line, color="black", linewidth=3.2, zorder=6,
            label=(f"fit: $A$={fit['A']:.4f}, $B$={fit['B']:.4f}, "
                   f"$R^2$={fit['R2']:.3f}"))
    ax.fill_between([], [], [], color="#BBBBBB", alpha=0.45, label="95% CI")
    if tolerance is not None:
        ax.axhline(tolerance, color="#C00000", linestyle=":", linewidth=2.2,
                   zorder=6,
                   label=f"tolerance $y={tolerance:g}$ (+{100 * (tolerance - 1):g}% {metric})")
    # The clustered slope p-value and the within-scene R2 are not drawn in the
    # legend: both are printed by main() and stored in fit_results.json, which
    # is where the paper cites them from.

    ax.set_xlim(grid[0], grid[-1])
    # The trajectories run along one diagonal, so the legend goes in a corner
    # off that diagonal and the extra room is added on the same side. Which
    # side depends on the sign of the trend: metrics where lower is better
    # (LPIPS) rise to the right, metrics where higher is better (SSIM, PSNR)
    # fall to the right.
    lower_is_better = fit["A"] > 0
    lo, hi = y.min(), y.max()
    span = hi - lo
    if lower_is_better:
        auto = (lo - 0.04 * span, hi + 0.32 * span)
        legend_loc, arrow = "upper left", r"$\leftarrow$ low distortion"
    else:
        auto = (lo - 0.32 * span, hi + 0.04 * span)
        legend_loc, arrow = "lower left", r"low distortion $\rightarrow$"
    # explicit limits win; the automatic headroom is generous because it has to
    # clear the legend, so tightening it is the usual reason to override
    ax.set_ylim(auto[0] if ymin is None else ymin,
                auto[1] if ymax is None else ymax)
    ax.set_xlabel(r"$\leftarrow$ high bitrate     $\bar{H}/\log_2 K$"
                  r"     low bitrate $\rightarrow$")
    ax.set_ylabel(f"{arrow}\n"
                  f"${metric}_K\\,/\\,{metric}_{{{fit['baseline_k']}}}$")
    ax.legend(loc=legend_loc, ncol=2, frameon=True, framealpha=0.92,
              edgecolor="#CCCCCC", handletextpad=0.5, borderaxespad=0.4,
              labelspacing=0.3, columnspacing=1.0)
    ax.set_title("(a)", loc="left")

    # ---------------- panel (b): dispersion per K ----------------
    axb.grid(True, axis="y", color="#DDDDDD", linewidth=0.8, linestyle="--", zorder=0)
    axb.set_axisbelow(True)
    pos = list(range(len(box_ks)))
    bp = axb.boxplot(box_vals, positions=pos, widths=0.6, showfliers=True,
                     patch_artist=True, zorder=3,
                     medianprops={"color": "black", "linewidth": 2.0},
                     whiskerprops={"color": INK, "linewidth": 1.2},
                     capprops={"color": INK, "linewidth": 1.2},
                     flierprops={"marker": ".", "markersize": 6,
                                 "markerfacecolor": "#888888",
                                 "markeredgecolor": "none"})
    for patch in bp["boxes"]:
        patch.set_facecolor("#0072B2")
        patch.set_alpha(0.8)
        patch.set_edgecolor(INK)
        patch.set_linewidth(1.2)
    axb.axhline(1.0, color="#999999", linestyle="--", linewidth=1.4, zorder=1)
    if tolerance is not None:
        axb.axhline(tolerance, color="#C00000", linestyle=":", linewidth=2.2, zorder=2)
    axb.set_xticks(pos)
    axb.set_xticklabels([str(k) for k in box_ks])
    axb.set_xlabel(r"Codebook size $K$")
    axb.set_ylabel(f"${metric}_K\\,/\\,{metric}_{{{fit['baseline_k']}}}$")
    n_runs = max(len(v) for v in box_vals) if box_vals else 0
    axb.set_title(f"(b) dispersion over {n_runs} runs per $K$ "
                  f"(13 scenes $\\times$ 5 seeds)", loc="left")

    for ext in ("pdf", "png"):
        fig.savefig(f"{out_stem}.{ext}", dpi=200)
    plt.close(fig)
    print(f"  wrote {out_stem}.pdf / .png")



def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("csv_file", help="CSV produced by extract_metrics.py (K sweep)")
    parser.add_argument("-o", "--outdir", default=".", help="output directory")
    parser.add_argument("--metrics", nargs="*", default=["LPIPS", "SSIM", "PSNR", "size"],
                        choices=list(METRIC_COLUMNS))
    parser.add_argument("--baseline-k", type=int, default=4096,
                        help="codebook size used as the quality reference")
    parser.add_argument("--min-k", type=int, default=4,
                        help="smallest K included in the fit and the plot")
    parser.add_argument("--max-k", type=int, default=4096,
                        help="largest K included. The default keeps the "
                             "K=--baseline-k anchor, whose normalized value is 1 "
                             "by construction but which is still a measured "
                             "operating point of every scene")
    parser.add_argument("--tolerance", type=float, default=1.02,
                        help="quality tolerance drawn on both panels")
    parser.add_argument("--font-size", type=float, default=26.0,
                        help="base font size; every other text element scales off it")
    parser.add_argument("--figwidth", type=float, default=14.0,
                        help="figure width in inches (height follows)")
    parser.add_argument("--no-annotate-k", action="store_true",
                        help="do not label each point with its codebook size")
    parser.add_argument("--ymin", type=float, default=None,
                        help="override the lower y limit of panel (a)")
    parser.add_argument("--ymax", type=float, default=None,
                        help="override the upper y limit of panel (a). The "
                             "automatic value leaves room for the legend, so "
                             "lowering it makes the data fill more of the panel. "
                             "Applies to every metric plotted, so pass one "
                             "--metrics at a time when using it")
    parser.add_argument(
        "--methods", nargs="*", default=["Ksweep", "C3DGS"],
        help="method labels allowed to contribute. Must name the sweep and the "
             "method that supplies the K=--baseline-k reference, and nothing "
             "else: any other method at that K would corrupt the normalization. "
             "Pass no values to accept every method.",
    )
    args = parser.parse_args()

    set_font_size(args.font_size)
    os.makedirs(args.outdir, exist_ok=True)
    print("Loading sweep:")
    sweep = load_sweep(args.csv_file, methods=set(args.methods) if args.methods else None)
    if not sweep:
        raise SystemExit("no usable rows (need entropy_mean and color_codebook_size)")

    fits = {}
    for metric in args.metrics:
        print(f"\n===== {metric} =====")
        # The fit uses the seed-mean of each (scene, K) cell, one point per cell.
        x, y, datasets, scenes, ks = build_points(
            sweep, metric, args.baseline_k, args.min_k, args.max_k)
        if len(x) < 3:
            print("  not enough points, skipped")
            continue
        fit = fit_linear(x, y, groups=scenes)
        fit["baseline_k"] = args.baseline_k
        print(f"  A = {fit['A']:.4f}  [{fit['A_ci95'][0]:.4f}, {fit['A_ci95'][1]:.4f}]"
              f"  (SE {fit['A_se']:.4f}, p = {fit['A_p']:.3e})")
        print(f"  B = {fit['B']:.4f}  [{fit['B_ci95'][0]:.4f}, {fit['B_ci95'][1]:.4f}]")
        print(f"  R2 = {fit['R2']:.3f}   within-scene R2 = {fit['R2_within']:.3f}"
              f"   RMSE = {fit['RMSE']:.4f}   (n = {fit['n']} pts, "
              f"{fit['n_clusters']} scenes)")
        print(f"  scene-clustered: SE_A = {fit['A_se_cluster']:.4f}, "
              f"CI [{fit['A_ci95_cluster'][0]:.4f}, {fit['A_ci95_cluster'][1]:.4f}], "
              f"p = {fit['A_p_cluster']:.3e}")
        pss = fit["per_scene_slopes"]
        print(f"  per-scene slopes: mean {pss['mean']:.4f}, sd {pss['sd']:.4f}, "
              f"range [{pss['min']:.4f}, {pss['max']:.4f}], "
              f"{pss['n_same_sign']}/{fit['n_clusters']} same sign as the pooled fit")
        box_ks, box_vals = build_boxes(sweep, metric, args.baseline_k, args.max_k)
        plot_metric(metric, x, y, scenes, ks, fit, box_ks, box_vals,
                    os.path.join(args.outdir, f"entropy_vs_{metric.lower()}"),
                    tolerance=args.tolerance if metric == "LPIPS" else None,
                    figwidth=args.figwidth, annotate_k=not args.no_annotate_k,
                    ymin=args.ymin, ymax=args.ymax)
        fits[metric] = {k: v for k, v in fit.items() if not k.startswith("_")}
        fits[metric]["min_k"] = args.min_k
        fits[metric]["max_k"] = args.max_k

    # Merge rather than overwrite: re-running a single --metrics entry (to
    # retune its axes, say) must not delete the other metrics' coefficients,
    # which the paper and arguments/__init__.py both cite.
    fit_path = os.path.join(args.outdir, "fit_results.json")
    merged = {}
    if os.path.exists(fit_path):
        try:
            with open(fit_path) as f:
                merged = json.load(f)
        except (ValueError, OSError):
            print(f"  warning: could not read {fit_path}, starting fresh")
    kept = [k for k in merged if k not in fits]
    merged.update(fits)
    with open(fit_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nFit coefficients written to {fit_path}"
          + (f" (updated {sorted(fits)}, kept {sorted(kept)})" if kept else ""))


if __name__ == "__main__":
    main()
