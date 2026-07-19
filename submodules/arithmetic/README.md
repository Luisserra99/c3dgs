# arithmetic — GPU chunk-parallel arithmetic codec

CUDA range coder (16-bit precision) used as the fast arithmetic-coding backend
of this pipeline (`ac_gpu.py`). One CUDA thread encodes/decodes each
10 000-symbol chunk, so chunks are processed concurrently; the per-symbol CDF
tables are evaluated fully in parallel.

Vendored unmodified from **FCGS** (Chen et al., "Fast Feedforward 3D Gaussian
Splatting Compression", ICLR 2025, https://github.com/YihangChen-ee/FCGS,
`submodules/arithmetic/`), which implements a CUDA version of
**torchac** (https://github.com/fab-jul/torchac). See the FCGS repository's
LICENSE.md for licensing terms.

## Install (required for `--ac_backend gpu`)

Compile in the same environment used to run `compress.py` (needs `nvcc`;
the c3dgs conda env already ships `cuda-toolkit`):

```bash
pip install ./submodules/arithmetic
```

Verify with:

```bash
python -c "import arithmetic; print('ok')"
python scripts/test_ac_gpu.py   # round-trip self-test (needs a GPU)
```

## API (pybind11 module `arithmetic`)

- `arithmetic_encode(sym, cdf, chunk_size, N, Lp) -> (bytes_u8, cnt_i32)`
  `sym`: int16 `[N]` CUDA, values in `[0, Lp-2]`;
  `cdf`: float32 `[N, Lp]` CUDA, monotone rows in `[0, 1]`.
- `arithmetic_decode(cdf, bytes_u8, cnt_i32, chunk_size, N, Lp) -> sym`
  The CDF must be rebuilt identically to the one used at encode time.
- `calculate_cdf(mean, scale, Q, min_value, max_value)` — Gaussian CDF helper
  (unused by this pipeline; `ac_gpu.py` builds empirical-histogram CDFs).
