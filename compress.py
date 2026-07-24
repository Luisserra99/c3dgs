# %%
import gc
import json
import os
import random
import time
import uuid
from argparse import ArgumentParser, Namespace
from os import path
from shutil import copyfile
from typing import Dict, Tuple

# arithmetic coding helper (ac_gs.py provides routines to encode integer-valued
# arrays produced by the gaussian model).  we import the three functions that
# operate on the in‑memory dictionary, and later we will call them from
# run_vq when the user requests it.
from ac_gs import (
    encode_feature_rest,
    encode_int8_array_ac,
    encode_compress_features,
    encode_compress_gaussians,
)

# GPU chunk-parallel codec (FCGS-based); falls back to ac_gs if the
# 'arithmetic' CUDA extension is not installed.
import ac_gpu


import math
import torch
from tqdm import tqdm
from torchvision.utils import save_image
from PIL import Image
import numpy as np

# %%
from arguments import (
    CompressionParams,
    ModelParams,
    OptimizationParams,
    PipelineParams,
    get_combined_args,
)
from compression.vq import CompressionSettings, compress_gaussians
from gaussian_renderer import GaussianModel, render
from lpipsPyTorch import lpips
from scene import Scene
from finetune import finetune
from utils.image_utils import psnr
from utils.loss_utils import ssim


def unique_output_folder():
    if os.getenv("OAR_JOB_ID"):
        unique_str = os.getenv("OAR_JOB_ID")
    else:
        unique_str = str(uuid.uuid4())
    return os.path.join("./output_vq/", unique_str[0:10])


# -- arithmetic coding helpers ------------------------------------------------

# color_codebook_size / gaussian_codebook_size are passed in so we can
# separate clustered indices (freq > 1, compress well) from direct indices
# (freq = 1, pointless to AC-encode).
# backend: "gpu" uses the chunk-parallel CUDA codec (ac_gpu.py), "cpu" the
# original adaptive Python coder (ac_gs.py).
def write_ac_files(
    npz_path: str,
    ac_dir: str,
    color_codebook_size: int = None,    # added: threshold for feature_indices split
    gaussian_codebook_size: int = None, # added: threshold for gaussian_indices split
    backend: str = "cpu",
) -> None:

    os.makedirs(ac_dir, exist_ok=True)
    data = np.load(npz_path)
    save_dict = {k: data[k].copy() for k in data.files}

    # Collect shapes / max-values BEFORE each key is popped so the decode
    # script can reconstruct arrays without guessing.
    metadata = {"ac_backend": backend}

    def encode_int8(key):
        path = os.path.join(ac_dir, f"{key}.bin")
        metadata[f"{key}_shape"] = list(save_dict[key].shape)
        if backend == "gpu":
            ac_gpu.encode_int_array_gpu(save_dict.pop(key), path)
        else:
            # Save eof_symbol so the decoder can build an identical frequency
            # table: eof_symbol = max(shifted_array) + 1, mirroring the encoder.
            metadata[f"{key}_eof_symbol"] = int((save_dict[key].astype(np.int16) + 128).max()) + 1
            encode_int8_array_ac(save_dict, key, path)

    # encode each integer array that is present.  previously we only handled
    # the VQ-specific keys; extend support to the raw indices and features as
    # requested.
    if "features_rest" in save_dict:
        arr = save_dict["features_rest"]
        metadata["features_rest_feature_shape"] = list(arr.shape[1:])  # e.g. [15, 3]
        metadata["features_rest_num_gaussians"] = int(arr.shape[0])
        path = os.path.join(ac_dir, "features_rest.bin")
        if backend == "gpu":
            ac_gpu.encode_feature_rest_gpu(save_dict.pop("features_rest"), path)
        else:
            encode_feature_rest(save_dict, "features_rest", path)
    for key in ("features_dc", "opacity", "scaling", "scaling_factor", "rotation"):
        if key in save_dict:
            encode_int8(key)

    if "feature_indices" in save_dict:
        fi = save_dict["feature_indices"]
        if color_codebook_size is not None:
            # Split: values < codebook_size are clustered (high freq); the rest are
            # direct (each appears exactly once and compresses poorly with AC).
            is_clustered = fi < color_codebook_size
            save_dict["feature_indices_is_clustered"] = is_clustered   # mask → remaining.npz
            # FIX: store actual direct values — Morton sort scrambles their order so they
            # cannot be reconstructed from the mask alone with a simple arange() assumption.
            save_dict["feature_indices_direct"] = fi[~is_clustered]    # direct values → remaining.npz
            save_dict["feature_indices"] = fi[is_clustered]            # only clustered values for AC
            metadata["feature_indices_codebook_size"] = color_codebook_size  # decoder needs this
        # guard: empty clustered set has no max
        metadata["feature_indices_max"] = int(save_dict["feature_indices"].max()) if len(save_dict["feature_indices"]) > 0 else 0
        if backend == "gpu":
            ac_gpu.encode_int_array_gpu(save_dict.pop("feature_indices"), os.path.join(ac_dir, "feature_indices.bin"))
        else:
            encode_compress_features(save_dict, os.path.join(ac_dir, "feature_indices.bin"))
        # the encoder consumes "feature_indices"; mask and direct stay in save_dict

    if "gaussian_indices" in save_dict:
        gi = save_dict["gaussian_indices"]
        if gaussian_codebook_size is not None:
            # Same split for gaussian indices
            is_clustered_g = gi < gaussian_codebook_size
            save_dict["gaussian_indices_is_clustered"] = is_clustered_g   # mask → remaining.npz
            # FIX: same reason — Morton sort means direct index values are not sequential
            save_dict["gaussian_indices_direct"] = gi[~is_clustered_g]    # direct values → remaining.npz
            save_dict["gaussian_indices"] = gi[is_clustered_g]            # only clustered values for AC
            metadata["gaussian_indices_codebook_size"] = gaussian_codebook_size  # decoder needs this
        metadata["gaussian_indices_max"] = int(save_dict["gaussian_indices"].max()) if len(save_dict["gaussian_indices"]) > 0 else 0
        if backend == "gpu":
            ac_gpu.encode_int_array_gpu(save_dict.pop("gaussian_indices"), os.path.join(ac_dir, "gaussian_indices.bin"))
        else:
            encode_compress_gaussians(save_dict, os.path.join(ac_dir, "gaussian_indices.bin"))
        # the encoder consumes "gaussian_indices"; mask and direct stay in save_dict

    # Save metadata so the decoder can reconstruct shapes / max-values.
    with open(os.path.join(ac_dir, "metadata.json"), "w") as _f:
        json.dump(metadata, _f, indent=2)

    # save whatever is left (floating‑point or scale/zp information)
    np.savez_compressed(os.path.join(ac_dir, "remaining.npz"), **save_dict)


# -----------------------------------------------------------------------------



def calc_importance(
    gaussians: GaussianModel,
    scene,
    pipeline_params,
    output_dir=None,
    iteration=None,
    sensitivity_mode: str = "abs",
) -> Tuple[torch.Tensor, torch.Tensor]:
    # sensitivity_mode selects the scalar whose parameter gradients define the
    # sensitivity score: "energy" (original C3DGS image energy), "abs"
    # (absolute reconstruction error) or "sq" (squared reconstruction error).
    if sensitivity_mode not in ("energy", "abs", "sq"):
        raise ValueError(f"unknown sensitivity_mode: {sensitivity_mode}")
    scaling = gaussians.scaling_qa(
        gaussians.scaling_activation(gaussians._scaling.detach())
    )
    cov3d = gaussians.covariance_activation(
        scaling, 1.0, gaussians.get_rotation.detach(), True
    ).requires_grad_(True)
    scaling_factor = gaussians.scaling_factor_activation(
        gaussians.scaling_factor_qa(gaussians._scaling_factor.detach())
    )

    h1 = gaussians._features_dc.register_hook(lambda grad: grad.abs())
    h2 = gaussians._features_rest.register_hook(lambda grad: grad.abs())
    h3 = cov3d.register_hook(lambda grad: grad.abs())
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

    gaussians._features_dc.grad = None
    gaussians._features_rest.grad = None
    num_pixels = 0
    
    # Create output directory structure for heat maps if specified
    #heat_map_dir = None
    #if output_dir and iteration is not None:
    #    heat_map_dir = os.path.join(output_dir, "heat_map", f"iteration_{iteration}")
    #    os.makedirs(heat_map_dir, exist_ok=True)
    
    camera_idx = 0
    for camera in tqdm(scene.getTrainCameras(), desc="Calculating sensitivity"):
        cov3d_scaled = cov3d * scaling_factor.square()
        rendering = render(
            camera,
            gaussians,
            pipeline_params,
            background,
            clamp_color=False,
            cov3d=cov3d_scaled,
        )["render"]
        
        if sensitivity_mode == "energy":
            loss = rendering.sum()
        else:
            original_image = camera.original_image[0:3, :, :].unsqueeze(0)  # (1, 3, H, W)
            rendering_unsqueezed = rendering.unsqueeze(0)  # (1, 3, H, W)
            diff = (original_image - rendering_unsqueezed).abs()
            loss = diff.square().sum() if sensitivity_mode == "sq" else diff.sum()
        loss.backward()
        num_pixels += rendering.shape[1]*rendering.shape[2]
        
        # Save the 3 images if output directory is specified
        #if heat_map_dir:
                # Save original image
            #save_image(original_image, os.path.join(heat_map_dir, f"camera_{camera_idx}_original.png"))
                # Save rendered image
            #save_image(rendering_unsqueezed, os.path.join(heat_map_dir, f"camera_{camera_idx}_rendered.png"))
                # Save difference as heat map (normalized absolute difference)
                # Normalize for better visualization
        #    diff_normalized = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
        #    save_image(diff_normalized, os.path.join(heat_map_dir, f"camera_{camera_idx}_difference.png"))    
        #camera_idx += 1

    importance = torch.cat(
        [gaussians._features_dc.grad, gaussians._features_rest.grad],
        1,
    ).flatten(-2)/num_pixels
    cov_grad = cov3d.grad/num_pixels
    h1.remove()
    h2.remove()
    h3.remove()
    torch.cuda.empty_cache()
    return importance.detach(), cov_grad.detach()


def render_and_eval(
    gaussians: GaussianModel,
    scene: Scene,
    model_params: ModelParams,
    pipeline_params: PipelineParams,
) -> Dict[str, float]:
    with torch.no_grad():
        ssims = []
        psnrs = []
        lpipss = []

        views = scene.getTestCameras()

        bg_color = [1, 1, 1] if model_params.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        for view in tqdm(views, desc="Rendering progress"):
            rendering = render(view, gaussians, pipeline_params, background)[
                "render"
            ].unsqueeze(0)
            gt = view.original_image[0:3, :, :].unsqueeze(0)

            ssims.append(ssim(rendering, gt))
            psnrs.append(psnr(rendering, gt))
            lpipss.append(lpips(rendering, gt, net_type="vgg"))
            gc.collect()
            torch.cuda.empty_cache()

        return {
            "SSIM": torch.tensor(ssims).mean().item(),
            "PSNR": torch.tensor(psnrs).mean().item(),
            "LPIPS": torch.tensor(lpipss).mean().item(),
        }


def _quantize_to_int8_np(arr: np.ndarray) -> np.ndarray:
    max_abs = float(np.abs(arr).max())
    if max_abs == 0.0:
        return np.zeros_like(arr, dtype=np.int8)
    return np.clip(np.round(arr / (max_abs / 127.0)), -128, 127).astype(np.int8)


def _compute_entropy_np(arr_int: np.ndarray) -> float:
    _, counts = np.unique(arr_int.flatten(), return_counts=True)
    p = counts / float(counts.sum())
    return float(-np.sum(p * np.log2(p)))


def compute_scene_entropy(gaussians: GaussianModel) -> Tuple[Dict[str, float], float]:
    """Empirical Shannon entropy of the int8-quantized attributes of the
    uncompressed model. Returns (per-attribute entropies, mean entropy)."""
    arrays = {
        "features_dc":  gaussians._features_dc.detach().cpu().numpy(),
        "features_rest": gaussians._features_rest.detach().cpu().numpy(),
        "opacity":       gaussians._opacity.detach().cpu().numpy(),
        "scaling":       gaussians._scaling.detach().cpu().numpy(),
        "rotation":      gaussians._rotation.detach().cpu().numpy(),
    }
    per_attr = {
        name: _compute_entropy_np(_quantize_to_int8_np(a)) for name, a in arrays.items()
    }
    H = float(np.mean(list(per_attr.values())))
    print(f"Mean entropy H = {H:.4f} bits")
    return per_attr, H


def codebook_size_from_entropy(H: float, comp_params) -> int:
    """Codebook size from scene entropy using the LPIPS-based formula:

    codebook_size = 2 ** ceil( H * lpips_a / (lpips_loss - lpips_b) )
    """
    exponent = (H * comp_params.lpips_a) / (comp_params.lpips_loss - comp_params.lpips_b)
    cb_size = 2 ** math.ceil(exponent)
    print(f"Auto codebook size: {cb_size}  (2^{math.ceil(exponent)})")
    return cb_size


def run_vq(
    model_params: ModelParams,
    optim_params: OptimizationParams,
    pipeline_params: PipelineParams,
    comp_params: CompressionParams,
):
    # Seed all RNGs so repeated runs with different --seed values give the
    # independent samples needed for variance / significance analysis.
    random.seed(comp_params.seed)
    np.random.seed(comp_params.seed)
    torch.manual_seed(comp_params.seed)
    torch.cuda.manual_seed_all(comp_params.seed)

    gaussians = GaussianModel(
        model_params.sh_degree, quantization=not optim_params.not_quantization_aware
    )
    scene = Scene(
        model_params, gaussians, load_iteration=comp_params.load_iteration, shuffle=True
    )

    if comp_params.start_checkpoint:
        (checkpoint_params, first_iter) = torch.load(comp_params.start_checkpoint)
        gaussians.restore(checkpoint_params, optim_params)


    timings ={}

    # %%

    start_time = time.time()
    # Create heat_map directory under output_vq before calling calc_importance
    iteration = scene.loaded_iter
    color_importance, gaussian_sensitivity = calc_importance(
        gaussians,
        scene,
        pipeline_params,
        output_dir=comp_params.output_vq,
        iteration=iteration,
        sensitivity_mode=comp_params.sensitivity_mode,
    )
    end_time = time.time()
    timings["sensitivity_calculation"] = end_time-start_time

    # Entropy of the uncompressed model is always measured and stored in
    # results.json: the entropy-vs-quality analysis needs it for every run,
    # not only when auto_codebook is enabled.
    entropy_per_attr, mean_entropy = compute_scene_entropy(gaussians)

    if comp_params.auto_codebook:
        cb_size = codebook_size_from_entropy(mean_entropy, comp_params)
        comp_params.color_codebook_size = cb_size
        comp_params.gaussian_codebook_size = cb_size

    # %%
    print("vq compression..")
    with torch.no_grad():
        start_time = time.time()
        color_importance_n = color_importance.amax(-1)

        gaussian_importance_n = gaussian_sensitivity.amax(-1)

        torch.cuda.empty_cache()

        color_compression_settings = CompressionSettings(
            codebook_size=comp_params.color_codebook_size,
            importance_prune=comp_params.color_importance_prune,
            importance_include=comp_params.color_importance_include,
            steps=int(comp_params.color_cluster_iterations),
            decay=comp_params.color_decay,
            batch_size=comp_params.color_batch_size,
        )

        gaussian_compression_settings = CompressionSettings(
            codebook_size=comp_params.gaussian_codebook_size,
            importance_prune=None,
            importance_include=comp_params.gaussian_importance_include,
            steps=int(comp_params.gaussian_cluster_iterations),
            decay=comp_params.gaussian_decay,
            batch_size=comp_params.gaussian_batch_size,
        )

        compress_gaussians(
            gaussians,
            color_importance_n,
            gaussian_importance_n,
            color_compression_settings if not comp_params.not_compress_color else None,
            gaussian_compression_settings
            if not comp_params.not_compress_gaussians
            else None,
            comp_params.color_compress_non_dir,
            prune_threshold=comp_params.prune_threshold,
            use_softmax=not comp_params.no_softmax,
        )
        end_time = time.time()
        timings["clustering"]=end_time-start_time

    gc.collect()
    torch.cuda.empty_cache()
    os.makedirs(comp_params.output_vq, exist_ok=True)

    copyfile(
        path.join(model_params.model_path, "cfg_args"),
        path.join(comp_params.output_vq, "cfg_args"),
    )
    model_params.model_path = comp_params.output_vq

    with open(
        os.path.join(comp_params.output_vq, "cfg_args_comp"), "w"
    ) as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(comp_params))))

    iteration = scene.loaded_iter + comp_params.finetune_iterations
    if comp_params.finetune_iterations > 0:

        start_time = time.time()
        finetune(
            scene,
            model_params,
            optim_params,
            comp_params,
            pipeline_params,
            testing_iterations=[
                -1
            ],
            debug_from=-1,
        )
        end_time = time.time()
        timings["finetune"]=end_time-start_time

        # %%
    out_file = path.join(
        comp_params.output_vq,
        f"point_cloud/iteration_{iteration}/point_cloud.npz",
    )

    # When arithmetic coding is disabled we reproduce the original C3DGS output
    # (see default_compre.py): a Morton-sorted, DEFLATE-compressed .npz. Morton
    # sorting is what makes that .npz compressible, so refuse to drop it here.
    use_ac = not comp_params.no_ac
    if not use_ac and comp_params.not_sort_morton:
        raise ValueError(
            "--no_ac requires Morton sorting: the DEFLATE baseline is defined "
            "with sort_morton=True. Drop --not_sort_morton."
        )

    model_for_ac = None
    start_time = time.time()
    if comp_params.skip_save_npz and use_ac:
        temp_dir = os.path.join(comp_params.output_vq, "ac_output")
        os.makedirs(temp_dir, exist_ok=True)
        model_for_ac = path.join(temp_dir, "uncompressed_model.npz")
        gaussians.save_npz(model_for_ac, compress=False, sort_morton=not comp_params.not_sort_morton)
    else:
        gaussians.save_npz(out_file, sort_morton=not comp_params.not_sort_morton)
        model_for_ac = out_file
    end_time = time.time()
    timings["encode"] = end_time - start_time

    # compute the directory that will hold the bitstreams; it is always a
    # subfolder of ``output_vq``.
    ac_dir = os.path.join(comp_params.output_vq, "ac_output")
    ac_backend = comp_params.ac_backend if use_ac else "none"
    if use_ac and ac_backend == "gpu" and not ac_gpu.gpu_available():
        print("WARNING: 'arithmetic' CUDA extension unavailable "
              "(pip install ./submodules/arithmetic); falling back to CPU coder")
        ac_backend = "cpu"
    # synchronize around the timer so pending GPU work is attributed correctly
    torch.cuda.synchronize()
    start_time = time.time()
    if use_ac and model_for_ac is not None:
        write_ac_files(
            model_for_ac,
            ac_dir,
            # Pass codebook sizes so the encoder knows which index values are clustered.
            # When the corresponding VQ stage was skipped, pass None (fall back to full encoding).
            color_codebook_size=comp_params.color_codebook_size if not comp_params.not_compress_color else None,
            gaussian_codebook_size=comp_params.gaussian_codebook_size if not comp_params.not_compress_gaussians else None,
            backend=ac_backend,
        )
    torch.cuda.synchronize()
    timings["arithmetic_coding"] = time.time() - start_time

    # Three reporting categories: whole compression, AC stage alone, and
    # everything besides AC.
    timings["time_total"] = sum(
        timings[k]
        for k in ("sensitivity_calculation", "clustering", "finetune", "encode", "arithmetic_coding")
        if k in timings
    )
    timings["time_arithmetic_coding"] = timings["arithmetic_coding"]
    timings["time_without_ac"] = timings["time_total"] - timings["time_arithmetic_coding"]
    with open(f"{comp_params.output_vq}/times.json","w") as f:
        json.dump(timings,f,indent=4)

    file_size = 0.0
    if not use_ac:
        # DEFLATE baseline: the single Morton-sorted .npz *is* the model, so
        # measure just that file (default_compre.py).
        file_size = os.path.getsize(out_file) / (1024 ** 2)
        print(f"saved vq finetuned model to {out_file}")
        print(f"Compressed model size: {file_size:.2f} MB")
    else:
        # AC: the model is the set of bitstreams, so total the directory holding
        # them, excluding the uncompressed .npz that was only the encoder's input.
        model_dir = os.path.dirname(model_for_ac) if model_for_ac else None

        if model_dir and os.path.exists(model_dir):
            total_bytes = 0

            # Loop through everything in the directory
            for filename in os.listdir(model_dir):
                file_path = os.path.join(model_dir, filename)

                # Check if it is a file (ignore directories) AND not the uncompressed model
                if os.path.isfile(file_path) and filename != "uncompressed_model.npz":
                    total_bytes += os.path.getsize(file_path)

            # Convert the total byte sum to MB
            file_size = total_bytes / (1024 ** 2)

            if not comp_params.skip_save_npz:
                print(f"saved vq finetuned model to {out_file}")
                print(f"Compressed model size: {file_size:.2f} MB")

    print("evaluating...")
    if not use_ac:
        # Morton sorting reorders the primitives during serialization, so the
        # DEFLATE baseline is evaluated on the model that actually round-trips
        # through the file (default_compre.py).
        eval_gaussians = GaussianModel(
            model_params.sh_degree,
            quantization=not optim_params.not_quantization_aware,
        )
        eval_gaussians.load(out_file, override_quantization=True)
        metrics = render_and_eval(eval_gaussians, scene, model_params, pipeline_params)
    else:
        # AC is lossless, so the in-memory model matches what the decoder yields.
        metrics = render_and_eval(gaussians, scene, model_params, pipeline_params)
    metrics["size"] = file_size
    metrics["entropy_mean"] = mean_entropy
    metrics["entropy_per_attribute"] = entropy_per_attr
    metrics["seed"] = comp_params.seed
    metrics["sensitivity_mode"] = comp_params.sensitivity_mode
    metrics["ac_backend"] = ac_backend
    metrics["color_codebook_size"] = comp_params.color_codebook_size
    metrics["gaussian_codebook_size"] = comp_params.gaussian_codebook_size
    metrics["time_total"] = timings["time_total"]
    metrics["time_arithmetic_coding"] = timings["time_arithmetic_coding"]
    metrics["time_without_ac"] = timings["time_without_ac"]
    print(metrics)
    with open(f"{comp_params.output_vq}/results.json","w") as f:
        json.dump({f"ours_{iteration}":metrics},f,indent=4)

if __name__ == "__main__":
    parser = ArgumentParser(description="Compression script parameters")
    model = ModelParams(parser, sentinel=True)
    model.data_device = "cuda"
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    comp = CompressionParams(parser)
    args = get_combined_args(parser)

    if args.output_vq is None:
        args.output_vq = unique_output_folder()

    model_params = model.extract(args)
    optim_params = op.extract(args)
    pipeline_params = pipeline.extract(args)
    comp_params = comp.extract(args)

    run_vq(model_params, optim_params, pipeline_params, comp_params)
