#!/usr/bin/env python3
"""Aggregate per-scene compression results into one tidy CSV.

Expected experiment layout (one root per method/configuration):

    <root>/<scene>/results.json                       written by compress.py
    <root>/<scene>/times.json                         written by compress.py
    <root>/<scene>/**/decode_eval_results.json        written by decode_and_eval.py

where <scene> is one of the 13 benchmark scenes (bicycle, bonsai, counter,
drjohnson, flowers, garden, kitchen, playroom, room, stump, train, treehill,
truck). Any nesting between <root> and results.json works — scenes are
recognized by directory name anywhere in the path, so layouts such as
<root>/<scene>/seed_1/results.json are also supported.

Usage:

    python scripts/extract_metrics.py /path/outputs_c3dgs /path/outputs_softmax \
        --labels C3DGS Softmax -o metrics.csv

Each row of the CSV holds: method, scene, dataset, seed, sensitivity_mode,
color_codebook_size, gaussian_codebook_size, PSNR, SSIM, LPIPS, size_MB,
entropy_mean, time_total, time_arithmetic_coding, time_without_ac,
time_ac_decode. Fields missing from older runs are left empty.
"""

import argparse
import csv
import json
import re
from pathlib import Path

SCENE_TO_DATASET = {
    "bicycle": "MipNeRF360",
    "bonsai": "MipNeRF360",
    "counter": "MipNeRF360",
    "flowers": "MipNeRF360",
    "garden": "MipNeRF360",
    "kitchen": "MipNeRF360",
    "room": "MipNeRF360",
    "stump": "MipNeRF360",
    "treehill": "MipNeRF360",
    "train": "TanksAndTemples",
    "truck": "TanksAndTemples",
    "drjohnson": "DeepBlending",
    "playroom": "DeepBlending",
}

FIELDS = [
    "method", "scene", "dataset", "seed", "sensitivity_mode", "ac_backend",
    "color_codebook_size", "gaussian_codebook_size",
    "PSNR", "SSIM", "LPIPS", "size_MB", "entropy_mean",
    "time_total", "time_arithmetic_coding", "time_without_ac", "time_ac_decode",
]


def scene_from_path(path: Path, root: Path):
    for part in path.relative_to(root).parts:
        if part in SCENE_TO_DATASET:
            return part
    return None


def codebook_size_from_cfg(run_dir: Path):
    """Fallback for runs whose results.json predates the codebook-size field:
    parse color_codebook_size out of the cfg_args_comp Namespace dump."""
    cfg = run_dir / "cfg_args_comp"
    if not cfg.exists():
        return None
    m = re.search(r"color_codebook_size=(\d+)", cfg.read_text())
    return int(m.group(1)) if m else None


def collect_run(results_file: Path, root: Path, method: str):
    scene = scene_from_path(results_file, root)
    if scene is None:
        print(f"  skipping {results_file}: no known scene name in path")
        return None

    with open(results_file) as f:
        data = json.load(f)
    # results.json holds a single top-level key such as "ours_35000"
    metrics = data[next(iter(data))]

    run_dir = results_file.parent
    times = {}
    times_file = run_dir / "times.json"
    if times_file.exists():
        with open(times_file) as f:
            times = json.load(f)

    decode = {}
    decode_files = sorted(run_dir.glob("**/decode_eval_results.json"))
    if decode_files:
        with open(decode_files[0]) as f:
            decode = json.load(f)

    def pick(*keys, sources=(metrics, times, decode)):
        for source in sources:
            for key in keys:
                if key in source:
                    return source[key]
        return ""

    row = {
        "method": method,
        "scene": scene,
        "dataset": SCENE_TO_DATASET[scene],
        "seed": pick("seed"),
        "sensitivity_mode": pick("sensitivity_mode"),
        "ac_backend": pick("ac_backend"),
        "color_codebook_size": pick("color_codebook_size"),
        "gaussian_codebook_size": pick("gaussian_codebook_size"),
        "PSNR": pick("PSNR"),
        "SSIM": pick("SSIM"),
        "LPIPS": pick("LPIPS"),
        "size_MB": pick("size"),
        "entropy_mean": pick("entropy_mean"),
        "time_total": pick("time_total", "total"),
        "time_arithmetic_coding": pick("time_arithmetic_coding", "arithmetic_coding"),
        "time_without_ac": pick("time_without_ac"),
        "time_ac_decode": pick("time_ac_decode"),
    }
    if row["color_codebook_size"] == "":
        cb = codebook_size_from_cfg(run_dir)
        if cb is not None:
            row["color_codebook_size"] = cb
    return row


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("roots", nargs="+", help="one output root per method/configuration")
    parser.add_argument(
        "--labels", nargs="*", default=None,
        help="method label per root (default: the root directory's basename)",
    )
    parser.add_argument("-o", "--output", default="metrics.csv", help="output CSV path")
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.roots):
        parser.error("--labels must have one entry per root")

    rows = []
    for i, root_str in enumerate(args.roots):
        root = Path(root_str)
        if not root.exists():
            print(f"Warning: {root} does not exist, skipping")
            continue
        method = args.labels[i] if args.labels else root.name
        print(f"Processing {root} as method '{method}'")
        for results_file in sorted(root.glob("**/results.json")):
            row = collect_run(results_file, root, method)
            if row is not None:
                rows.append(row)
                print(f"  {row['scene']:<10} seed={row['seed'] or '-'} OK")

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
