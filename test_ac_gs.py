"""Utility script to exercise the arithmetic-coding helpers and collect metrics.

This is meant to be run *after* a point-cloud model has been written by
``compress.py`` (or any other code that produces an NPZ in the same format).
The script will:

1. load the NPZ and call the routines in ``ac_gs.py`` to produce separate
   bitstream files for each integer-valued array; the leftover arrays are
   written back into ``remaining.npz`` inside the same directory.
2. instantiate a ``GaussianModel``/``Scene`` pair, load the original NPZ into
   the model and run ``compress.render_and_eval`` (the same evaluation code used
   by ``compress.py``) to get SSIM/PSNR/LPIPS metrics.

The output directory thus contains two things: the AC bitstreams and the
remaining npz, and the console prints the evaluation scores that can be used to
compare against the original .npz file.
"""

import argparse
import os

import numpy as np
import torch

from ac_gs import decode_features_rest_ac, decode_vq_indices_ac
from compress import write_ac_files, render_and_eval
from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import GaussianModel
from scene import Scene


def main():
    parser = argparse.ArgumentParser(description="Test arithmetic coding on a saved model")
    parser.add_argument("--model_npz", required=True, help="path to the uncompressed point_cloud.npz")
    parser.add_argument("--ac_dir", required=True, help="directory where bitstreams and remaining.npz will be written")
    parser.add_argument("--run_eval", action="store_true", help="run render+eval after encode/decode verification")

    # reuse the parameter groups so the model/scene are constructed with the
    # same hyperparameters that were used during compression.  most of them have
    # sensible defaults so the user only needs to supply flags they care about.
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    optim = OptimizationParams(parser)

    args = parser.parse_args()

    # ensure ModelParams has a valid source_path/model_path so it can be extracted
    if getattr(args, "source_path", None) is None:
        args.source_path = args.model_npz
    if getattr(args, "model_path", None) is None:
        args.model_path = args.ac_dir

    os.makedirs(args.ac_dir, exist_ok=True)

    # arithmetic encode the integer fields and drop them from the numpy dict
    write_ac_files(args.model_npz, args.ac_dir)

    # verify decoding matches the original
    orig = np.load(args.model_npz)

    # features_rest
    decoded_rest = decode_features_rest_ac(
        os.path.join(args.ac_dir, "features_rest.bin"),
        feature_shape=orig["features_rest"].shape[1:],
        num_gaussians=orig["features_rest"].shape[0],
    )
    assert np.array_equal(decoded_rest, orig["features_rest"]), "features_rest mismatch after decode"

    # feature indices
    decoded_feature_indices = decode_vq_indices_ac(
        os.path.join(args.ac_dir, "feature_indices.bin"),
        max_value=int(orig["feature_indices"].max()),
    )
    assert decoded_feature_indices.shape == orig["feature_indices"].shape
    assert np.array_equal(decoded_feature_indices, orig["feature_indices"]), "feature_indices mismatch after decode"

    # gaussian indices
    decoded_gaussian_indices = decode_vq_indices_ac(
        os.path.join(args.ac_dir, "gaussian_indices.bin"),
        max_value=int(orig["gaussian_indices"].max()),
    )
    assert decoded_gaussian_indices.shape == orig["gaussian_indices"].shape
    assert np.array_equal(decoded_gaussian_indices, orig["gaussian_indices"]), "gaussian_indices mismatch after decode"

    print("AC encode/decode verification passed")

    if args.run_eval:
        # construct gaussian model + scene for evaluation
        model_params = model.extract(args)
        if getattr(model_params, "source_path", None) is None:
            model_params.source_path = args.model_npz
        if getattr(model_params, "model_path", None) in (None, ""):
            model_params.model_path = args.ac_dir

        optim_params = optim.extract(args)
        pipeline_params = pipeline.extract(args)

        gaussians = GaussianModel(model_params.sh_degree, quantization=not optim_params.not_quantization_aware)
        scene = Scene(model_params, gaussians, load_iteration=-1, shuffle=True)
        gaussians.load_npz(args.model_npz)

        print("running render + evaluation on loaded model...")
        metrics = render_and_eval(gaussians, scene, model_params, pipeline_params)
        print("evaluation metrics:", metrics)


if __name__ == "__main__":
    main()
