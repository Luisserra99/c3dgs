"""decode_and_eval.py — round-trip verification for C3DGS arithmetic coding.

Decodes a directory of arithmetic-coded Gaussian Splatting files produced by
compress.py and evaluates render quality (PSNR / SSIM / LPIPS) to confirm
that the compression is lossless end-to-end.

Expected directory layout (written by compress.py → write_ac_files):

    ac_dir/
        remaining.npz           # xyz, quant scale/zero-point, etc.
        metadata.json           # shapes and max-values of AC-encoded arrays
        features_rest.bin
        features_dc.bin
        opacity.bin
        scaling.bin
        scaling_factor.bin
        rotation.bin
        feature_indices.bin     (optional – present only with VQ color indexing)
        gaussian_indices.bin    (optional – present only with VQ gaussian indexing)

Usage:
    python decode_and_eval.py \\
        --ac_dir  /path/to/output_vq/ac_output \\
        --source_path /path/to/original/dataset \\
        [--sh_degree 3] \\
        [--output_dir /tmp/decoded_model] \\
        [--white_background]
"""

import argparse
import ast
import json
import os
import re
import time
from argparse import Namespace

import numpy as np
import torch

# Decode helpers defined in ac_gs.py
from ac_gs import (
    decode_features_rest_ac,
    decode_int8_array_ac,
    decode_vq_indices_ac,
)

# GPU chunk-parallel codec (FCGS-based); used when the archive was encoded
# with ac_backend == "gpu" (recorded in metadata.json)
import ac_gpu

# Re-use the evaluation loop from compress.py to keep results comparable
from compress import render_and_eval

from gaussian_renderer import GaussianModel
from scene import Scene


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------

def decode_ac_dir(ac_dir: str, sh_degree: int = 3) -> dict:
    """Decode all arithmetic-coded binary streams in *ac_dir*.

    Reads metadata.json for the exact shapes and max-values that were recorded
    during encoding, then decodes each .bin file with the matching parameters.

    Returns a dict with the same keys that GaussianModel.save_npz writes,
    suitable for passing to np.savez_compressed and then GaussianModel.load_npz.
    """
    meta_path = os.path.join(ac_dir, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"{meta_path} not found.\n"
            "Re-run compression with the updated compress.py that saves metadata.json."
        )
    with open(meta_path) as f:
        meta = json.load(f)

    # Backend the archive was encoded with; old archives predate the field.
    backend = meta.get("ac_backend", "cpu")
    print(f"  arithmetic-coding backend: {backend}")

    # Start with the arrays that were NOT arithmetic-coded (xyz, quant params …)
    remaining = np.load(os.path.join(ac_dir, "remaining.npz"), allow_pickle=True)
    out = {k: remaining[k] for k in remaining.files}

    # ---- features_rest  (one frequency model per SH×RGB slot) ---------------
    rest_bin = os.path.join(ac_dir, "features_rest.bin")
    if os.path.exists(rest_bin):
        feature_shape = tuple(meta["features_rest_feature_shape"])   # e.g. (15, 3)
        num_gaussians = meta["features_rest_num_gaussians"]
        print(f"  decoding features_rest  (feature_shape={feature_shape}, n={num_gaussians}) …")
        if backend == "gpu":
            out["features_rest"] = ac_gpu.decode_feature_rest_gpu(
                rest_bin, feature_shape=feature_shape, num_gaussians=num_gaussians
            )
        else:
            out["features_rest"] = decode_features_rest_ac(
                rest_bin, feature_shape=feature_shape, num_gaussians=num_gaussians
            )
        print(f"    → {out['features_rest'].shape}  {out['features_rest'].dtype}")

    # ---- int8 arrays  (single adaptive model each) --------------------------
    # NOTE: decode_int8_array_ac uses a fixed 257-symbol table (eof = 256).
    # This matches encode_int8_array_ac when all 256 int8 values appear in the
    # data (max_symbol == 255 after the +128 shift).  For large 3DGS models
    # this is virtually always true; the shape parameter provides an additional
    # early-stop guard for robustness.
    for key in ("features_dc", "opacity", "scaling", "scaling_factor", "rotation"):
        bin_path = os.path.join(ac_dir, f"{key}.bin")
        if os.path.exists(bin_path):
            if backend == "gpu":
                print(f"  decoding {key:<20} (gpu) …")
                out[key] = ac_gpu.decode_int_array_gpu(bin_path)
            else:
                shape = tuple(meta[f"{key}_shape"])
                # Read the eof_symbol saved by compress.py; fall back to 256 for old files
                # that were compressed before this metadata field was added.
                eof_symbol = meta.get(f"{key}_eof_symbol", 256)  # added: pass to decoder
                print(f"  decoding {key:<20} shape={list(shape)} eof={eof_symbol} …")
                out[key] = decode_int8_array_ac(bin_path, shape=shape, eof_symbol=eof_symbol)  # added eof_symbol
            print(f"    → {out[key].shape}  {out[key].dtype}")

    # ---- VQ index arrays  (flat adaptive model, only clustered values) ---------
    # The encoder split each index array into:
    #   • clustered values  [0, codebook_size-1]  → AC-encoded in the .bin file
    #   • direct values     [codebook_size, ...]  → reconstructed from mask + codebook_size
    # The boolean mask is stored in remaining.npz as "{key}_is_clustered".
    for key, cs_meta_key in (
        ("feature_indices",  "feature_indices_codebook_size"),
        ("gaussian_indices", "gaussian_indices_codebook_size"),
    ):
        bin_path = os.path.join(ac_dir, f"{key}.bin")
        if not os.path.exists(bin_path):
            continue

        max_value = meta[f"{key}_max"]
        print(f"  decoding {key:<20} max_value={max_value} …")
        if backend == "gpu":
            clustered_vals = ac_gpu.decode_int_array_gpu(bin_path)
        else:
            clustered_vals = decode_vq_indices_ac(bin_path, max_value=max_value)

        if cs_meta_key in meta:
            # Retrieve and remove both helper arrays from out — they came from remaining.npz
            # and are not keys that GaussianModel.load_npz understands.
            mask = out.pop(f"{key}_is_clustered").astype(bool)   # shape (N_gaussians,)
            # FIX: use the stored direct values instead of the broken arange() assumption.
            # Morton sort scrambles the gaussian order so direct indices are NOT sequential
            # in Morton-position order — we must store and restore the actual values.
            direct_vals = out.pop(f"{key}_direct")                # actual values ≥ codebook_size
            full = np.empty(len(mask), dtype=np.int32)
            full[mask] = clustered_vals                           # clustered: read from AC stream
            full[~mask] = direct_vals                             # direct: restored from remaining.npz
            out[key] = full
        else:
            # Fallback: encoder did not split (codebook_size not in metadata).
            out[key] = clustered_vals

        print(f"    → {out[key].shape}  {out[key].dtype}")

    return out


# ---------------------------------------------------------------------------
# Reconstruct the model on disk so Scene can load it normally
# ---------------------------------------------------------------------------

def save_reconstructed_npz(decoded: dict, output_dir: str) -> str:
    """Write *decoded* to the path that Scene expects and return the path."""
    npz_dir = os.path.join(output_dir, "point_cloud", "iteration_1")
    os.makedirs(npz_dir, exist_ok=True)
    npz_path = os.path.join(npz_dir, "point_cloud.npz")
    np.savez_compressed(npz_path, **decoded)
    return npz_path


# ---------------------------------------------------------------------------
# cfg_args loader
# ---------------------------------------------------------------------------

def load_cfg_args(path: str) -> dict:
    """Parse a cfg_args file saved by 3DGS training.

    The file contains a single line like:
        Namespace(eval=True, images='images_2', source_path='/...', ...)

    Returns a plain dict of {key: value} with proper Python types.
    """
    with open(path) as f:
        content = f.read().strip()
    # Strip the 'Namespace(...)' wrapper and convert to a dict literal
    inner = re.match(r"^Namespace\((.*)\)$", content, re.DOTALL)
    if not inner:
        raise ValueError(f"Unrecognised cfg_args format in {path}: {content!r}")
    # Turn  key=value, key=value  →  {"key": value, "key": value}  and eval safely
    dict_str = "{" + re.sub(r"(\w+)=", r'"\1":', inner.group(1)) + "}"
    return ast.literal_eval(dict_str)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode AC-compressed C3DGS model and evaluate render quality."
    )
    parser.add_argument(
        "--ac_dir", required=True,
        help="Directory produced by write_ac_files "
             "(contains *.bin, remaining.npz, metadata.json).",
    )
    # --cfg_args is the preferred way: reads all scene settings from the saved Namespace file.
    # Individual flags below are only needed when cfg_args is not available.
    parser.add_argument(
        "--cfg_args", default=None,
        help="Path to a cfg_args file saved by 3DGS training/compression (e.g. "
             "output_vq/cfg_args).  When provided, source_path / sh_degree / "
             "white_background / images / resolution are all read from it.",
    )
    parser.add_argument(
        "--source_path", default=None,
        help="Path to the original training dataset. Required when --cfg_args is not given.",
    )
    parser.add_argument(
        "--sh_degree", type=int, default=None,
        help="SH degree used during training. Overrides cfg_args when given.",
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Where to write the reconstructed model files. "
             "Defaults to <ac_dir>/decoded_model.",
    )
    parser.add_argument(
        "--white_background", action="store_true", default=None,
        help="Force white background. Overrides cfg_args when given.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ac_dir = os.path.abspath(args.ac_dir)
    output_dir = os.path.abspath(
        args.output_dir if args.output_dir else os.path.join(ac_dir, "decoded_model")
    )
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Resolve scene settings: prefer cfg_args, fall back to CLI flags
    # ------------------------------------------------------------------
    if args.cfg_args is not None:
        cfg = load_cfg_args(args.cfg_args)
        print(f"  Loaded cfg_args from {args.cfg_args}:")
        for k, v in cfg.items():
            print(f"    {k} = {v!r}")
        # CLI overrides win over cfg_args when explicitly provided
        source_path     = args.source_path     or cfg.get("source_path", "")
        sh_degree       = args.sh_degree       if args.sh_degree is not None else cfg.get("sh_degree", 3)
        white_background = args.white_background if args.white_background else cfg.get("white_background", False)
        images          = cfg.get("images", "images")
        resolution      = cfg.get("resolution", -1)
        data_device     = cfg.get("data_device", "cuda")
    else:
        if args.source_path is None:
            raise ValueError("--source_path is required when --cfg_args is not provided.")
        source_path      = args.source_path
        sh_degree        = args.sh_degree if args.sh_degree is not None else 3
        white_background = bool(args.white_background)
        images           = "images"
        resolution       = -1
        data_device      = "cuda"

    # ------------------------------------------------------------------
    # 1. Decode all binary streams
    # ------------------------------------------------------------------
    print("\n=== Decoding AC streams ===")
    # synchronize around the timer so pending GPU work is attributed correctly
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    decode_start = time.time()
    decoded = decode_ac_dir(ac_dir, sh_degree=sh_degree)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    time_ac_decode = time.time() - decode_start
    print(f"  AC decode time: {time_ac_decode:.2f} s")

    # ------------------------------------------------------------------
    # 2. Save reconstructed npz in the path layout that Scene expects:
    #      output_dir/point_cloud/iteration_1/point_cloud.npz
    # ------------------------------------------------------------------
    print("\n=== Saving reconstructed model ===")
    npz_path = save_reconstructed_npz(decoded, output_dir)
    print(f"  saved → {npz_path}")

    # ------------------------------------------------------------------
    # 3. Instantiate GaussianModel with matching quantization flag and
    #    build a Scene pointing at the reconstructed npz.
    #    Scene(load_iteration=1) will call gaussians.load_npz internally.
    # ------------------------------------------------------------------
    is_quantized = bool(decoded.get("quantization", False))
    gaussians = GaussianModel(sh_degree, quantization=is_quantized)

    model_params = Namespace(
        model_path=output_dir,
        source_path=source_path,
        images=images,              # read from cfg_args (e.g. 'images_2')
        resolution=resolution,      # read from cfg_args (e.g. 1)
        white_background=white_background,
        data_device=data_device,
        eval=True,
    )
    pipeline_params = Namespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
    )

    print("\n=== Loading Scene cameras and model ===")
    # load_iteration=1 matches the iteration_1 sub-directory we created above.
    # Scene will load the gaussians from disk via gaussians.load_npz.
    scene = Scene(model_params, gaussians, load_iteration=1, shuffle=False)

    # ------------------------------------------------------------------
    # 4. Evaluate render quality — should be identical to the pre-compression
    #    evaluation if the round-trip is truly lossless.
    # ------------------------------------------------------------------
    print("\n=== Evaluating render quality ===")
    metrics = render_and_eval(gaussians, scene, model_params, pipeline_params)

    print("\nResults:")
    print(f"  PSNR  = {metrics['PSNR']:.4f} dB")
    print(f"  SSIM  = {metrics['SSIM']:.4f}")
    print(f"  LPIPS = {metrics['LPIPS']:.4f}")

    metrics["time_ac_decode"] = time_ac_decode
    result_path = os.path.join(output_dir, "decode_eval_results.json")
    with open(result_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nResults written to {result_path}")


if __name__ == "__main__":
    main()
