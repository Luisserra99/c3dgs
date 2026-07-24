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
    scenes): Wilcoxon signed-rank and paired t-test.

Usage:

    python scripts/stats_significance.py metrics.csv --baseline C3DGS
    python scripts/stats_significance.py metrics.csv --baseline C3DGS \
        --metrics PSNR LPIPS --latex significance_table.tex

Wilcoxon requires scipy (already available in typical torch environments);
without scipy only descriptive statistics are printed.
"""

import argparse
import csv
import math
import statistics
from collections import defaultdict

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None

METRICS_DEFAULT = ["PSNR", "SSIM", "LPIPS", "size_MB"]


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


def paired_tests(rows, baseline, method, metric):
    base_means, _, _ = scene_means(rows, baseline, metric)
    meth_means, _, _ = scene_means(rows, method, metric)
    common = sorted(set(base_means) & set(meth_means))
    if len(common) < 3:
        return None
    base = [base_means[s] for s in common]
    meth = [meth_means[s] for s in common]
    diffs = [m - b for m, b in zip(meth, base)]
    result = {
        "n": len(common),
        "mean_diff": statistics.mean(diffs),
        "std_diff": statistics.stdev(diffs) if len(diffs) > 1 else 0.0,
    }
    if scipy_stats is not None:
        if any(d != 0 for d in diffs):
            w = scipy_stats.wilcoxon(meth, base)
            result["wilcoxon_p"] = w.pvalue
        else:
            result["wilcoxon_p"] = 1.0
        t = scipy_stats.ttest_rel(meth, base)
        result["ttest_p"] = t.pvalue
    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("csv_file", help="CSV produced by extract_metrics.py")
    parser.add_argument("--baseline", required=True, help="baseline method label")
    parser.add_argument("--metrics", nargs="*", default=METRICS_DEFAULT)
    parser.add_argument("--latex", default=None, help="write a LaTeX table fragment here")
    args = parser.parse_args()

    rows = load_rows(args.csv_file)
    methods = sorted({row["method"] for row in rows})
    if args.baseline not in methods:
        parser.error(f"baseline '{args.baseline}' not found; methods: {methods}")
    if scipy_stats is None:
        print("NOTE: scipy not available — significance tests skipped, "
              "only descriptive statistics reported.\n")

    latex_lines = []
    for metric in args.metrics:
        print(f"\n===== {metric} =====")
        for method in methods:
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
            if method == args.baseline:
                continue
            test = paired_tests(rows, args.baseline, method, metric)
            if test is None:
                print("  paired test: not enough common scenes")
                continue
            line = (
                f"  vs {args.baseline}: mean diff={test['mean_diff']:+.4f} "
                f"(std {test['std_diff']:.4f}, n={test['n']} scenes)"
            )
            if "wilcoxon_p" in test:
                line += (
                    f"  Wilcoxon p={test['wilcoxon_p']:.4f}"
                    f"  paired-t p={test['ttest_p']:.4f}"
                )
            print(line)
            if "wilcoxon_p" in test:
                latex_lines.append(
                    f"{method} & {metric} & {test['mean_diff']:+.3f} & "
                    f"{test['std_diff']:.3f} & {test['wilcoxon_p']:.3f} \\\\"
                )

    if args.latex and latex_lines:
        header = (
            "% Auto-generated by stats_significance.py\n"
            "% columns: method & metric & mean diff vs baseline & std & Wilcoxon p\n"
        )
        with open(args.latex, "w") as f:
            f.write(header + "\n".join(latex_lines) + "\n")
        print(f"\nLaTeX fragment written to {args.latex}")


if __name__ == "__main__":
    main()
