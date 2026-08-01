#!/usr/bin/env python3
"""Statistical analysis of compression results: run-to-run variability and
paired significance tests against a baseline method.

Input is the CSV produced by scripts/extract_metrics.py, covering one or more
methods, the 13 benchmark scenes, and ideally several seeds per configuration
(run compress.py repeatedly with different --seed values).

For each method and metric the script reports:
  * per-dataset mean of the per-scene values (averaged over seeds), the
    cross-scene std, and the mean run-to-run std over seeds;
  * paired tests against the baseline (pairing per scene, pooled over all
    scenes): Wilcoxon signed-rank (primary) and paired t-test (check), with
    Cohen's d_z, a percentage change, and a bootstrap 95% CI on the mean
    difference;
  * Holm-Bonferroni adjusted p-values over the whole method x metric family.

Usage:

    python scripts/stats_significance.py metrics.csv --baseline C3DGS
    python scripts/stats_significance.py metrics.csv --baseline C3DGS \
        --metrics PSNR LPIPS --latex significance_table.tex

Wilcoxon requires scipy. The c3dgs conda env does NOT have it -- run this with
/usr/bin/python3 (or any interpreter with scipy); without scipy only descriptive
statistics are printed.
"""

import argparse
import csv
import math
import random
import statistics
from collections import defaultdict

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None

METRICS_DEFAULT = ["PSNR", "SSIM", "LPIPS", "size_MB"]
N_BOOTSTRAP = 20000


def load_rows(csv_path):
    with open(csv_path, newline="") as f:
        return [row for row in csv.DictReader(f)]


def scene_means(rows, method, metric):
    """Per-scene mean over seeds and per-scene run-to-run std, for one method."""
    per_scene = defaultdict(list)
    for row in rows:
        if row["method"] != method or not row.get(metric):
            continue
        per_scene[row["scene"]].append(float(row[metric]))
    means = {scene: statistics.mean(vals) for scene, vals in per_scene.items()}
    stds = {
        scene: (statistics.stdev(vals) if len(vals) > 1 else 0.0)
        for scene, vals in per_scene.items()
    }
    counts = {scene: len(vals) for scene, vals in per_scene.items()}
    return means, stds, counts


def dataset_of(rows, scene):
    for row in rows:
        if row["scene"] == scene:
            return row["dataset"]
    return "?"


def describe(rows, method, metric):
    means, stds, counts = scene_means(rows, method, metric)
    by_dataset = defaultdict(list)
    for scene, mean in means.items():
        by_dataset[dataset_of(rows, scene)].append((mean, stds[scene]))
    out = {}
    for dataset, pairs in sorted(by_dataset.items()):
        vals = [m for m, _ in pairs]
        run_stds = [s for _, s in pairs]
        out[dataset] = {
            "mean": statistics.mean(vals),
            "scene_std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "run_std": statistics.mean(run_stds),
            "n_scenes": len(vals),
        }
    n_runs = max(counts.values()) if counts else 0
    return out, n_runs


def bootstrap_ci(diffs, n_boot=N_BOOTSTRAP, seed=0):
    """Percentile bootstrap 95% CI on the mean paired difference.

    Resampling is over scenes, which is the unit the pairing is done on.
    """
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(n_boot):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[min(int(0.975 * n_boot), n_boot - 1)]
    return lo, hi


def holm_adjust(pvalues):
    """Holm-Bonferroni step-down adjustment, order preserved."""
    k = len(pvalues)
    order = sorted(range(k), key=lambda i: pvalues[i])
    adjusted = [0.0] * k
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (k - rank) * pvalues[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def paired_tests(rows, baseline, method, metric):
    base_means, _, _ = scene_means(rows, baseline, metric)
    meth_means, _, _ = scene_means(rows, method, metric)
    common = sorted(set(base_means) & set(meth_means))
    if len(common) < 3:
        return None
    base = [base_means[s] for s in common]
    meth = [meth_means[s] for s in common]
    diffs = [m - b for m, b in zip(meth, base)]
    mean_diff = statistics.mean(diffs)
    std_diff = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
    base_mean = statistics.mean(base)
    ci_lo, ci_hi = bootstrap_ci(diffs)
    result = {
        "n": len(common),
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        # d_z is the paired-sample effect size: mean difference in units of the
        # standard deviation of the differences.
        "cohen_dz": mean_diff / std_diff if std_diff > 0 else float("nan"),
        "pct_change": 100.0 * mean_diff / base_mean if base_mean else float("nan"),
        "ci95": (ci_lo, ci_hi),
    }
    if scipy_stats is not None:
        if any(d != 0 for d in diffs):
            w = scipy_stats.wilcoxon(meth, base)
            result["wilcoxon_p"] = w.pvalue
        else:
            result["wilcoxon_p"] = 1.0
        t = scipy_stats.ttest_rel(meth, base)
        result["ttest_p"] = t.pvalue
        result["t_stat"] = t.statistic
    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("csv_file", help="CSV produced by extract_metrics.py")
    parser.add_argument("--baseline", required=True, help="baseline method label")
    parser.add_argument("--metrics", nargs="*", default=METRICS_DEFAULT)
    parser.add_argument("--latex", default=None, help="write a LaTeX table fragment here")
    parser.add_argument(
        "--exclude", nargs="*", default=["Ksweep"],
        help="method labels to leave out of the paired tests. The codebook sweep "
             "is excluded by default: it holds several K per scene, which "
             "scene_means() would silently average into one meaningless value.",
    )
    parser.add_argument(
        "--codebook-size", type=int, default=None,
        help="keep only rows with this color_codebook_size (e.g. 4096), so a CSV "
             "that also contains a K sweep still yields a like-for-like comparison",
    )
    args = parser.parse_args()

    rows = load_rows(args.csv_file)
    if args.codebook_size is not None:
        rows = [r for r in rows
                if r.get("color_codebook_size") and
                int(r["color_codebook_size"]) == args.codebook_size]
        if not rows:
            parser.error(f"no rows with color_codebook_size={args.codebook_size}")
        print(f"Filtered to color_codebook_size={args.codebook_size}: {len(rows)} rows")
    methods = sorted({row["method"] for row in rows})
    if args.baseline not in methods:
        parser.error(f"baseline '{args.baseline}' not found; methods: {methods}")
    if scipy_stats is None:
        print("NOTE: scipy not available — significance tests skipped, "
              "only descriptive statistics reported. Re-run with an interpreter "
              "that has scipy (the c3dgs conda env does not).\n")

    # --- pass 1: every paired test, so Holm can correct over the full family ---
    tests = {}
    for metric in args.metrics:
        for method in methods:
            if method == args.baseline or method in args.exclude:
                continue
            test = paired_tests(rows, args.baseline, method, metric)
            if test is not None:
                tests[(method, metric)] = test
    if scipy_stats is not None and tests:
        keys = sorted(tests)
        adjusted = holm_adjust([tests[k]["wilcoxon_p"] for k in keys])
        for key, p_adj in zip(keys, adjusted):
            tests[key]["wilcoxon_p_holm"] = p_adj
        n_methods = len({m for m, _ in keys})
        print(f"Holm-Bonferroni correction applied over {len(keys)} tests "
              f"({n_methods} method(s) x {len(args.metrics)} metric(s)).")

    # --- pass 2: report ---
    for metric in args.metrics:
        print(f"\n===== {metric} =====")
        for method in methods:
            if method in args.exclude:
                continue
            desc, n_runs = describe(rows, method, metric)
            if not desc:
                continue
            print(f"\n{method}  ({n_runs} run(s) per scene)")
            for dataset, d in desc.items():
                print(
                    f"  {dataset:<16} mean={d['mean']:.4f}  "
                    f"cross-scene std={d['scene_std']:.4f}  "
                    f"run-to-run std={d['run_std']:.4f}  (n={d['n_scenes']})"
                )
            test = tests.get((method, metric))
            if method == args.baseline or test is None:
                continue
            print(
                f"  vs {args.baseline}: mean diff={test['mean_diff']:+.4f} "
                f"({test['pct_change']:+.2f}%), 95% CI "
                f"[{test['ci95'][0]:+.4f}, {test['ci95'][1]:+.4f}], "
                f"n={test['n']} scenes"
            )
            if "wilcoxon_p" in test:
                print(
                    f"    Wilcoxon p={test['wilcoxon_p']:.2e}  "
                    f"(Holm p={test['wilcoxon_p_holm']:.2e})  "
                    f"paired-t t({test['n'] - 1})={test['t_stat']:.2f} "
                    f"p={test['ttest_p']:.2e}  d_z={test['cohen_dz']:+.2f}"
                )

    if scipy_stats is not None and tests:
        # With n paired scenes the two-sided Wilcoxon p cannot go below
        # 2/2**n; flag it so a floored p is not read as an exact value.
        n_pairs = max(t["n"] for t in tests.values())
        print(f"\nNote: with n={n_pairs} pairs the smallest attainable two-sided "
              f"Wilcoxon p is {2 / 2 ** n_pairs:.2e}; values at that floor are "
              f"resolution-limited, not exact.")

    if args.latex and tests:
        header = (
            "% Auto-generated by stats_significance.py\n"
            "% columns: method & metric & mean diff & 95% CI & Wilcoxon p & "
            "Holm p & Cohen's d_z\n"
        )
        latex_lines = []
        for method, metric in sorted(tests, key=lambda k: (args.metrics.index(k[1]), k[0])):
            t = tests[(method, metric)]
            latex_lines.append(
                f"{method.replace('_', ' ')} & {metric.replace('_', ' ')} & "
                f"{t['mean_diff']:+.4f} & "
                f"[{t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f}] & "
                f"{t.get('wilcoxon_p', float('nan')):.1e} & "
                f"{t.get('wilcoxon_p_holm', float('nan')):.1e} & "
                f"{t['cohen_dz']:+.2f} \\\\"
            )
        with open(args.latex, "w") as f:
            f.write(header + "\n".join(latex_lines) + "\n")
        print(f"\nLaTeX fragment written to {args.latex}")


if __name__ == "__main__":
    main()
