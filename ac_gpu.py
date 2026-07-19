"""GPU arithmetic-coding backend based on the FCGS CUDA codec.

Encodes integer arrays with per-array *empirical* frequency tables (gathered in
a first pass and serialized in the bitstream header) using the chunk-parallel
CUDA range coder vendored in ``submodules/arithmetic`` (install with
``pip install ./submodules/arithmetic``; see its README).

Unlike ``ac_gs.py`` (pure-Python adaptive coder, kept as the CPU fallback),
this coder is static: the same CDF table must be rebuilt at decode time, which
is why the symbol counts are stored in the file. The CDF row is built once on
the CPU by ``_build_cdf_row`` — shared by encoder and decoder so both sides
quantize bit-identical tables.

File format (magic ``ACG1``): ``n_sections`` (i32), then per section:
dtype string (u8 length + ascii), ndim (u8), dims (i64 each), ``min_v`` (i64),
``span`` (i64, alphabet size), counts (i64[span]), ``n_chunks`` (i32), then per
chunk: ``n_rows`` (i64), cnt length (i64), cnt bytes (i32 array), stream
length (i64), stream bytes. Everything needed for decoding is in the file; a
section may be empty (span 0).

Requires a CUDA device. Run ``scripts/test_ac_gpu.py`` after installing the
extension to validate the round trip on the target machine.
"""

import struct

import numpy as np
import torch

try:
    import arithmetic  # CUDA extension from submodules/arithmetic
except ImportError:  # handled by callers (compress.py falls back to ac_gs)
    arithmetic = None

MAGIC = b"ACG1"
# matches FCGS: one CUDA thread codes each 10_000-symbol chunk
CHUNK_SIZE_CUDA = 10000
# cap on the broadcast [rows, Lp] float32 CDF table (2^28 floats = 1 GiB)
FLOAT_BUDGET = 2 ** 28
# symbols are int16 in the kernel
MAX_SPAN = 32767


def gpu_available() -> bool:
    return arithmetic is not None and torch.cuda.is_available()


def _require_gpu():
    if arithmetic is None:
        raise RuntimeError(
            "the 'arithmetic' CUDA extension is not installed; "
            "run: pip install ./submodules/arithmetic"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("ac_gpu requires a CUDA device")


def _build_cdf_row(counts: np.ndarray) -> torch.Tensor:
    """counts [span] -> CDF row [span+1] float32 on CPU (deterministic)."""
    pmf = torch.from_numpy(counts.astype(np.float32))
    pmf = pmf / pmf.sum()
    cdf = torch.cat([torch.zeros(1, dtype=torch.float32), torch.cumsum(pmf, dim=0)])
    return torch.clamp(cdf, 0.0, 1.0)


def _encode_section(f, arr: np.ndarray) -> None:
    dtype_str = arr.dtype.str.encode("ascii")
    f.write(struct.pack("<B", len(dtype_str)))
    f.write(dtype_str)
    f.write(struct.pack("<B", arr.ndim))
    f.write(struct.pack(f"<{arr.ndim}q", *arr.shape))

    flat = arr.reshape(-1).astype(np.int64)
    if flat.size == 0:
        f.write(struct.pack("<qqi", 0, 0, 0))  # min_v, span, n_chunks
        return

    min_v = int(flat.min())
    span = int(flat.max()) - min_v + 1
    if span - 1 > MAX_SPAN:
        raise ValueError(f"alphabet span {span} exceeds int16 symbol range")
    idx = flat - min_v
    counts = np.bincount(idx, minlength=span).astype(np.int64)

    Lp = span + 1
    cdf_row = _build_cdf_row(counts).cuda()
    sym_all = torch.from_numpy(idx.astype(np.int16)).cuda()

    rows_per_chunk = max(1, FLOAT_BUDGET // Lp)
    n_chunks = int(np.ceil(flat.size / rows_per_chunk))

    f.write(struct.pack("<qq", min_v, span))
    f.write(counts.tobytes())
    f.write(struct.pack("<i", n_chunks))
    for c in range(n_chunks):
        sym = sym_all[c * rows_per_chunk:(c + 1) * rows_per_chunk].contiguous()
        n_rows = int(sym.shape[0])
        cdf = cdf_row.unsqueeze(0).expand(n_rows, Lp).contiguous()
        stream, cnt = arithmetic.arithmetic_encode(sym, cdf, CHUNK_SIZE_CUDA, n_rows, Lp)
        cnt_bytes = cnt.cpu().numpy().astype(np.int32).tobytes()
        stream_bytes = stream.cpu().numpy().astype(np.uint8).tobytes()
        f.write(struct.pack("<qq", n_rows, len(cnt_bytes)))
        f.write(cnt_bytes)
        f.write(struct.pack("<q", len(stream_bytes)))
        f.write(stream_bytes)


def _decode_section(f) -> np.ndarray:
    (dtype_len,) = struct.unpack("<B", f.read(1))
    dtype = np.dtype(f.read(dtype_len).decode("ascii"))
    (ndim,) = struct.unpack("<B", f.read(1))
    shape = struct.unpack(f"<{ndim}q", f.read(8 * ndim))
    min_v, span = struct.unpack("<qq", f.read(16))
    if span == 0:
        (_,) = struct.unpack("<i", f.read(4))
        return np.zeros(shape, dtype=dtype)
    counts = np.frombuffer(f.read(8 * span), dtype=np.int64)
    (n_chunks,) = struct.unpack("<i", f.read(4))

    Lp = span + 1
    cdf_row = _build_cdf_row(counts).cuda()

    parts = []
    for _ in range(n_chunks):
        n_rows, len_cnt = struct.unpack("<qq", f.read(16))
        cnt = torch.from_numpy(
            np.frombuffer(f.read(len_cnt), dtype=np.int32).copy()
        ).cuda()
        (len_stream,) = struct.unpack("<q", f.read(8))
        stream = torch.from_numpy(
            np.frombuffer(f.read(len_stream), dtype=np.uint8).copy()
        ).cuda()
        cdf = cdf_row.unsqueeze(0).expand(n_rows, Lp).contiguous()
        sym = arithmetic.arithmetic_decode(cdf, stream, cnt, CHUNK_SIZE_CUDA, n_rows, Lp)
        parts.append(sym.cpu().numpy().astype(np.int64))

    flat = np.concatenate(parts) + min_v
    return flat.astype(dtype).reshape(shape)


def encode_arrays_gpu(arrays, path: str) -> None:
    """Encode a list of integer numpy arrays into one self-describing file."""
    _require_gpu()
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<i", len(arrays)))
        for arr in arrays:
            _encode_section(f, arr)


def decode_arrays_gpu(path: str):
    """Decode a file written by encode_arrays_gpu back into a list of arrays."""
    _require_gpu()
    with open(path, "rb") as f:
        if f.read(4) != MAGIC:
            raise ValueError(f"{path}: not an ACG1 file")
        (n_sections,) = struct.unpack("<i", f.read(4))
        return [_decode_section(f) for _ in range(n_sections)]


# --- convenience wrappers mirroring the ac_gs.py call sites ------------------

def encode_int_array_gpu(arr: np.ndarray, path: str) -> None:
    encode_arrays_gpu([arr], path)


def decode_int_array_gpu(path: str) -> np.ndarray:
    return decode_arrays_gpu(path)[0]


def encode_feature_rest_gpu(arr: np.ndarray, path: str) -> None:
    """features_rest [N, SH, C]: one section per SH*C slot, each with its own
    frequency table (the per-coefficient models of the paper)."""
    n = arr.shape[0]
    slots = arr.reshape(n, -1)
    encode_arrays_gpu([slots[:, i] for i in range(slots.shape[1])], path)


def decode_feature_rest_gpu(path: str, feature_shape, num_gaussians: int) -> np.ndarray:
    slots = decode_arrays_gpu(path)
    out = np.stack(slots, axis=1)
    return out.reshape(num_gaussians, *feature_shape)
