#!/usr/bin/env python3
"""Regenerate the entropy-vs-quality figures and fit the empirical model
Q_drop = A * (H / log2 K) + B used for automatic codebook-size selection.

Input is the CSV produced by scripts/extract_metrics.py over the codebook-size
sweep (one run per scene per K, K in {4, ..., 4096}); the rows must carry
entropy_mean and color_codebook_size, which compress.py now always writes.

For each metric the script:
  1. normalizes each scene's value by that scene's K = --baseline-k value,
  2. fits the linear model by ordinary least squares,
  3. reports A, B, R2 and 95% confidence intervals (printed and saved to
     fit_results.json),
  4. draws a publication figure (single-column width, >= 9 pt fonts,
     colorblind-safe Okabe-Ito palette, one marker shape per dataset,
     fit line with shaded 95% confidence band).

Usage:

    python scripts/plot_entropy_figures.py metrics.csv -o figures/
    python scripts/plot_entropy_figures.py metrics.csv -o figures/ \
        --metrics LPIPS SSIM --min-k 16

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

# Okabe-Ito colorblind-safe hues, fixed assignment order (validated: lightness
# band, chroma, CVD separation and normal-vision floor all pass on white).
DATASET_STYLE = {
    "MipNeRF360":      {"color": "#0072B2", "marker": "o"},
    "TanksAndTemples": {"color": "#E69F00", "marker": "s"},
    "DeepBlending":    {"color": "#009E73", "marker": "^"},
}
INK = "#333333"
METRIC_COLUMNS = {"LPIPS": "LPIPS", "SSIM": "SSIM", "PSNR": "PSNR", "size": "size_MB"}

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
})


def load_sweep(csv_path):
    """-> {(scene, K): {"dataset", "entropy", metric: mean-over-seeds value}}"""
    acc = defaultdict(lambda: defaultdict(list))
    info = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("color_codebook_size") or not row.get("entropy_mean"):
                continue
            key = (row["scene"], int(row["color_codebook_size"]))
            info[key] = {"dataset": row["dataset"], "entropy": float(row["entropy_mean"])}
            for metric, col in METRIC_COLUMNS.items():
                if row.get(col):
                    acc[key][metric].append(float(row[col]))
    out = {}
    for key, metrics in acc.items():
        out[key] = dict(info[key])
        for metric, vals in metrics.items():
            out[key][metric] = sum(vals) / len(vals)
    return out


def build_points(sweep, metric, baseline_k, min_k, max_k):
    """-> arrays x = H/log2(K), y = metric normalized by the scene's baseline."""
    xs, ys, datasets = [], [], []
    scenes = {scene for scene, _ in sweep}
    for scene in sorted(scenes):
        base = sweep.get((scene, baseline_k))
        if base is None or metric not in base:
            print(f"  warning: scene '{scene}' has no K={baseline_k} baseline, skipped")
            continue
        for (s, k), entry in sweep.items():
            if s != scene or k == baseline_k or metric not in entry:
                continue
            if not (min_k <= k <= max_k):
                continue
            xs.append(entry["entropy"] / math.log2(k))
            ys.append(entry[metric] / base[metric])
            datasets.append(entry["dataset"])
    return np.array(xs), np.array(ys), datasets


def fit_linear(x, y):
    """OLS y = A*x + B with R2 and 95% CIs on A and B."""
    n = len(x)
    A, B = np.polyfit(x, y, 1)
    pred = A * x + B
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    dof = n - 2
    s2 = ss_res / dof if dof > 0 else float("nan")
    sxx = float(np.sum((x - np.mean(x)) ** 2))
    se_A = math.sqrt(s2 / sxx)
    se_B = math.sqrt(s2 * (1.0 / n + np.mean(x) ** 2 / sxx))
    t_crit = scipy_stats.t.ppf(0.975, dof) if scipy_stats is not None else 1.96
    return {
        "A": float(A), "B": float(B), "R2": r2, "n": n,
        "A_ci95": [float(A - t_crit * se_A), float(A + t_crit * se_A)],
        "B_ci95": [float(B - t_crit * se_B), float(B + t_crit * se_B)],
        "_internals": (s2, sxx, float(np.mean(x)), t_crit),
    }


def plot_metric(metric, x, y, datasets, fit, out_stem):
    # 0.66 aspect matches the space reserved in the paper's single column
    fig, ax = plt.subplots(figsize=(3.45, 2.3), constrained_layout=True)
    ax.grid(True, color="#DDDDDD", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    # fit line + 95% confidence band for the mean response
    s2, sxx, x_mean, t_crit = fit["_internals"]
    grid = np.linspace(x.min(), x.max(), 100)
    line = fit["A"] * grid + fit["B"]
    se_line = np.sqrt(s2 * (1.0 / fit["n"] + (grid - x_mean) ** 2 / sxx))
    # shaded band = 95% CI of the mean response (described in the caption)
    ax.fill_between(grid, line - t_crit * se_line, line + t_crit * se_line,
                    color="#BBBBBB", alpha=0.4, linewidth=0, zorder=1)
    ax.plot(grid, line, color=INK, linewidth=1.2, zorder=2,
            label=f"fit: $A$={fit['A']:.3f}, $B$={fit['B']:.3f}, $R^2$={fit['R2']:.2f}")

    for dataset, style in DATASET_STYLE.items():
        mask = [d == dataset for d in datasets]
        if not any(mask):
            continue
        ax.scatter(x[mask], y[mask], s=22, marker=style["marker"],
                   facecolor=style["color"], edgecolor=INK, linewidth=0.4,
                   zorder=3, label=dataset)

    ax.set_xlabel(r"$\bar{H}/\log_2 K$")
    ax.set_ylabel(f"Normalized {metric}")
    # opaque background so the legend stays readable wherever "best" places it
    ax.legend(loc="best", frameon=True, framealpha=0.9, edgecolor="none",
              handletextpad=0.4, borderaxespad=0.2)

    for ext in ("pdf", "png"):
        fig.savefig(f"{out_stem}.{ext}", dpi=300)
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
                        help="smallest K included in the fit (raise to drop the "
                             "non-linear strong-degradation regime)")
    parser.add_argument("--max-k", type=int, default=2048)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    sweep = load_sweep(args.csv_file)
    if not sweep:
        raise SystemExit("no usable rows (need entropy_mean and color_codebook_size)")

    fits = {}
    for metric in args.metrics:
        print(f"\n===== {metric} =====")
        x, y, datasets = build_points(sweep, metric, args.baseline_k, args.min_k, args.max_k)
        if len(x) < 3:
            print("  not enough points, skipped")
            continue
        fit = fit_linear(x, y)
        print(f"  A = {fit['A']:.4f}  [{fit['A_ci95'][0]:.4f}, {fit['A_ci95'][1]:.4f}]")
        print(f"  B = {fit['B']:.4f}  [{fit['B_ci95'][0]:.4f}, {fit['B_ci95'][1]:.4f}]")
        print(f"  R2 = {fit['R2']:.3f}  (n = {fit['n']})")
        plot_metric(metric, x, y, datasets, fit,
                    os.path.join(args.outdir, f"entropy_vs_{metric.lower()}"))
        fits[metric] = {k: v for k, v in fit.items() if not k.startswith("_")}

    fit_path = os.path.join(args.outdir, "fit_results.json")
    with open(fit_path, "w") as f:
        json.dump(fits, f, indent=2)
    print(f"\nFit coefficients written to {fit_path}")


if __name__ == "__main__":
    main()
