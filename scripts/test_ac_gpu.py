#!/usr/bin/env python3
"""Round-trip self-test for the GPU arithmetic-coding backend (ac_gpu.py).

Run on a machine with a CUDA device after installing the codec:

    pip install ./submodules/arithmetic
    python scripts/test_ac_gpu.py

Encodes representative arrays (int8 attributes, skewed K=4096 VQ indexes,
degenerate cases, a features_rest tensor), decodes them back, asserts
bit-exact equality, and reports encoded size vs DEFLATE plus wall-clock time.
"""

import os
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ac_gpu  # noqa: E402


def roundtrip(name, arr, tmpdir, encode, decode):
    path = os.path.join(tmpdir, f"{name}.bin")
    t0 = time.time()
    encode(arr, path)
    t_enc = time.time() - t0
    t0 = time.time()
    out = decode(path)
    t_dec = time.time() - t0

    assert out.dtype == arr.dtype, f"{name}: dtype {out.dtype} != {arr.dtype}"
    assert out.shape == arr.shape, f"{name}: shape {out.shape} != {arr.shape}"
    assert np.array_equal(out, arr), f"{name}: decoded values differ"

    ac_size = os.path.getsize(path)
    deflate_size = len(zlib.compress(arr.tobytes(), 6))
    print(f"  {name:<22} n={arr.size:>9}  ac={ac_size/1024:8.1f} KiB  "
          f"deflate={deflate_size/1024:8.1f} KiB  "
          f"enc={t_enc:6.3f}s dec={t_dec:6.3f}s")


def main():
    if not ac_gpu.gpu_available():
        raise SystemExit(
            "FAIL: CUDA device and/or 'arithmetic' extension unavailable.\n"
            "Install with: pip install ./submodules/arithmetic"
        )
    rng = np.random.default_rng(0)
    tmpdir = os.path.join(os.path.dirname(__file__), "_ac_gpu_test")
    os.makedirs(tmpdir, exist_ok=True)

    n = 3_000_000

    print("single-array round trips:")
    cases = {
        # int8 attribute, roughly Laplacian like quantized SH coefficients
        "int8_laplacian": np.clip(
            rng.laplace(0, 12, n), -128, 127).astype(np.int8),
        # skewed clustered VQ indexes over a K=4096 alphabet (Zipf-like)
        "vq_indices_k4096": (rng.zipf(1.3, n) % 4096).astype(np.int32),
        # small alphabet
        "opacity_like": rng.integers(-128, 128, n).astype(np.int8),
        # degenerate cases
        "constant": np.full(12345, 7, dtype=np.int32),
        "single_symbol": np.array([42], dtype=np.int32),
        "empty": np.array([], dtype=np.int32),
    }
    for name, arr in cases.items():
        roundtrip(name, arr, tmpdir,
                  ac_gpu.encode_int_array_gpu, ac_gpu.decode_int_array_gpu)

    print("features_rest round trip (per-slot models):")
    fr = np.clip(rng.laplace(0, 8, (n // 4, 15, 3)), -128, 127).astype(np.int8)
    roundtrip(
        "features_rest", fr, tmpdir,
        ac_gpu.encode_feature_rest_gpu,
        lambda p: ac_gpu.decode_feature_rest_gpu(p, (15, 3), fr.shape[0]),
    )

    print("\nOK: all round trips are bit-exact")


if __name__ == "__main__":
    main()
