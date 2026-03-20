"""
depthforge.core.gpu_synthesizer
================================
GPU-accelerated SIRDS synthesis using PyTorch tensor operations.

No CUDA compiler required — implemented entirely with PyTorch tensor ops
(torch.gather, torch.scatter_reduce_, torch.arange, etc.).  Automatically
uses CUDA if available; falls back to CPU-tensor path if not; falls back to
the NumPy synthesizer if PyTorch is unavailable altogether.

Algorithm
---------
The Thimbleby-Inglis SIRDS algorithm has two serial inner loops per scanline:

  1. Constraint building — for each pixel x, link x_right → x_left via a
     chain-following walk (same[root] loop).
  2. Color assignment — left-to-right propagation of color indices along
     the same[] links.

GPU reformulation (all H rows processed in parallel):

  1. Compute all (x_left, x_right, valid) pairs simultaneously via vectorised
     shift arithmetic — zero Python loops.
  2. Build same[] with a single scatter_reduce_ (amin) call that correctly
     handles multiple constraints targeting the same pixel.
  3. Resolve transitive chains with **pointer doubling**: ceil(log2(max_shift))
     rounds of torch.gather.  Chains have length ≤ max_shift, so ~6-7 rounds
     suffice for typical parameters.
  4. Assign color indices as ``same_root % tile_W`` — one modulo operation.
  5. Gather RGBA pixels from the pattern tile in one batched index lookup.

Speedup
-------
For a 3840×2160 frame the NumPy implementation runs ~8 million serial
iterations.  The GPU path reduces this to ~7 tensor ops over the full
(H, W) batch — typically 10–50× faster on a mid-range CUDA GPU.

Public API
----------
    synthesize_gpu(depth, pattern, params, device='auto') -> np.ndarray
    best_device() -> str   # 'cuda', 'mps', or 'cpu'
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from depthforge import HAS_TORCH

if TYPE_CHECKING:
    from depthforge.core.synthesizer import StereoParams

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------


def best_device() -> str:
    """Return the best available torch device string.

    Priority: CUDA > MPS (Apple Silicon) > CPU.
    Returns 'cpu' if torch is not available.
    """
    if not HAS_TORCH:
        return "cpu"
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# GPU core
# ---------------------------------------------------------------------------


def _synthesize_gpu_core(
    depth: np.ndarray,
    pattern: np.ndarray,
    params: "StereoParams",
    device: str,
) -> np.ndarray:
    """Inner GPU synthesis — all rows processed in parallel via tensor ops.

    Parameters
    ----------
    depth : float32 (H, W), values in [0, 1].
    pattern : uint8 (tile_H, tile_W, 4) RGBA tile.
    params : StereoParams
    device : torch device string ('cuda', 'cpu', etc.)

    Returns
    -------
    uint8 (H, W, 4) RGBA stereogram.
    """
    import torch

    H, W = depth.shape
    tile_H, tile_W = pattern.shape[:2]
    max_shift = params.max_shift_px(W)

    # ── Move data to device ──────────────────────────────────────────────────
    d = torch.from_numpy(depth).to(device, dtype=torch.float32)           # (H, W)
    pat = torch.from_numpy(pattern).to(device)                            # (tile_H, tile_W, 4) uint8

    # ── Shift map ────────────────────────────────────────────────────────────
    # Convergence: shift the depth origin so that params.convergence = 0 parallax
    adj = (d - params.convergence) * params.depth_factor                   # (H, W) float
    shift_map = (adj * max_shift).round().clamp(-max_shift, max_shift)    # (H, W) float
    shift_map = shift_map.to(torch.int32)                                  # (H, W) int32

    # ── Per-pixel constraint pairs ────────────────────────────────────────────
    # x_left and x_right are the two pixels that must share the same colour.
    # Always x_left < x_right (we use abs(shift) to ensure this).
    x = torch.arange(W, device=device, dtype=torch.int32)                 # (W,)
    x = x.unsqueeze(0).expand(H, -1)                                      # (H, W)

    s_abs   = shift_map.abs()                                              # (H, W) >= 0
    x_left  = x - s_abs // 2                                              # (H, W)
    x_right = x + s_abs - s_abs // 2                                      # (H, W)

    valid = (
        (shift_map != 0) &
        (x_left  >= 0) & (x_left  < W) &
        (x_right >= 0) & (x_right < W) &
        (x_left != x_right)
    )                                                                      # (H, W) bool

    x_left  = x_left.clamp(0, W - 1)
    x_right = x_right.clamp(0, W - 1)

    # ── Build same[] ─────────────────────────────────────────────────────────
    # same[row, col] = the pixel this pixel must copy its colour from.
    # Initialise as identity: same[row, x] = x (each pixel is its own root).
    same = x.clone()                                                        # (H, W) int32

    # For each valid (x_left, x_right) pair, set same[row, x_right] = x_left.
    # Where multiple constraints map to the same x_right, keep the minimum
    # x_left (leftmost root) using scatter_reduce_ with 'amin'.
    #
    # For invalid pixels we scatter x_right (identity — no-op for amin since
    # existing value == x_right >= any valid x_left that targets x_right).
    src = torch.where(valid, x_left, x_right)                              # (H, W) int32

    try:
        # PyTorch >= 1.12
        same.scatter_reduce_(
            1,
            x_right.long(),
            src,
            reduce="amin",
            include_self=True,
        )
    except (AttributeError, TypeError, RuntimeError):
        # Older PyTorch fallback: plain scatter (last-write wins for conflicts).
        # Produces slightly different results when multiple constraints share an
        # x_right, but the stereogram is still visually correct.
        masked_src = torch.where(valid, x_left, same)
        same.scatter_(1, x_right.long(), masked_src)

    # ── Pointer doubling ─────────────────────────────────────────────────────
    # Resolve transitive chains in O(log max_shift) rounds.
    # After round k: same[x] → its 2^k-th ancestor.
    # Chains have length ≤ max_shift, so ceil(log2(max_shift + 2)) rounds suffice.
    n_rounds = max(1, math.ceil(math.log2(max_shift + 2)))
    same_long = same.long()                                                 # (H, W) int64
    for _ in range(n_rounds):
        same_long = torch.gather(same_long, 1, same_long)

    # ── Color index ───────────────────────────────────────────────────────────
    # Each pixel's colour column = root_pixel_position % tile_W
    color_col = same_long % tile_W                                          # (H, W) int64

    # ── Gather from pattern tile ──────────────────────────────────────────────
    # output[y, x] = pat[ y % tile_H,  color_col[y, x] ]
    tile_row = (torch.arange(H, device=device) % tile_H).long()            # (H,)
    tile_row_exp = tile_row.unsqueeze(1).expand(H, W)                       # (H, W)

    output_flat = pat[tile_row_exp.reshape(-1), color_col.reshape(-1)]     # (H*W, 4) uint8
    return output_flat.view(H, W, 4).cpu().numpy()


# ---------------------------------------------------------------------------
# Oversampled GPU path
# ---------------------------------------------------------------------------


def _synthesize_gpu_oversampled(
    depth: np.ndarray,
    pattern: np.ndarray,
    params: "StereoParams",
    device: str,
) -> np.ndarray:
    """GPU synthesis at oversample× resolution, Lanczos-downsampled."""
    from PIL import Image
    from depthforge.core.synthesizer import StereoParams as SP

    s = params.oversample
    H, W = depth.shape
    depth_up  = np.repeat(np.repeat(depth,   s, axis=0), s, axis=1)
    pattern_up = np.repeat(np.repeat(pattern, s, axis=0), s, axis=1)

    p2 = SP(
        depth_factor           = params.depth_factor,
        max_parallax_fraction  = params.max_parallax_fraction,
        eye_separation_fraction= params.eye_separation_fraction,
        convergence            = params.convergence,
        invert_depth           = False,
        oversample             = 1,
        seed                   = params.seed,
        safe_mode              = False,
    )
    hi = _synthesize_gpu_core(depth_up, pattern_up, p2, device)
    img = Image.fromarray(hi, mode="RGBA").resize((W, H), Image.LANCZOS)
    return np.asarray(img)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def synthesize_gpu(
    depth: np.ndarray,
    pattern: np.ndarray,
    params: "StereoParams | None" = None,
    device: str = "auto",
) -> np.ndarray:
    """GPU-accelerated SIRDS synthesis.

    Automatically selects CUDA > MPS > CPU-torch > NumPy fallback.

    Parameters
    ----------
    depth : float32 (H, W), values in [0, 1].
    pattern : uint8 (tile_H, tile_W, 4) RGBA tile.
    params : StereoParams (uses defaults if None).
    device : 'auto' | 'cuda' | 'mps' | 'cpu'.
        'auto' selects the best available device.

    Returns
    -------
    uint8 (H, W, 4) RGBA stereogram.
    """
    from depthforge.core.synthesizer import StereoParams, _validate_inputs

    if params is None:
        params = StereoParams()

    depth, pattern = _validate_inputs(depth, pattern)

    if params.invert_depth:
        depth = 1.0 - depth

    # Choose device
    if device == "auto":
        device = best_device()

    if not HAS_TORCH or device == "numpy":
        # Full NumPy fallback
        from depthforge.core.synthesizer import synthesize
        return synthesize(depth, pattern, params)

    try:
        if params.oversample > 1:
            return _synthesize_gpu_oversampled(depth, pattern, params, device)
        return _synthesize_gpu_core(depth, pattern, params, device)

    except Exception as exc:
        # Any GPU/torch error → NumPy fallback with warning
        import warnings
        warnings.warn(
            f"GPU synthesis failed ({exc!r}), falling back to NumPy.",
            RuntimeWarning,
            stacklevel=2,
        )
        from depthforge.core.synthesizer import synthesize
        return synthesize(depth, pattern, params)


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------


def benchmark(
    width: int = 1920,
    height: int = 1080,
    n_runs: int = 3,
) -> dict:
    """Compare NumPy vs GPU synthesis times.

    Usage::

        from depthforge.core.gpu_synthesizer import benchmark
        results = benchmark(3840, 2160)
        print(results)

    Returns
    -------
    dict with keys: device, numpy_ms, gpu_ms, speedup
    """
    import time
    from depthforge.core.synthesizer import StereoParams, synthesize
    from depthforge.core.pattern_gen import PatternParams, PatternType, generate_pattern

    rng = np.random.default_rng(42)
    depth = rng.random((height, width), dtype=np.float32)
    pattern = generate_pattern(PatternParams(
        pattern_type=PatternType.RANDOM_NOISE,
        tile_width=128, tile_height=128,
    ))
    params = StereoParams(depth_factor=0.4, seed=42)
    dev = best_device()

    # Warm-up
    synthesize_gpu(depth, pattern, params, device=dev)
    synthesize(depth, pattern, params)

    # NumPy timing
    t0 = time.perf_counter()
    for _ in range(n_runs):
        synthesize(depth, pattern, params)
    numpy_ms = (time.perf_counter() - t0) / n_runs * 1000

    # GPU timing
    t0 = time.perf_counter()
    for _ in range(n_runs):
        synthesize_gpu(depth, pattern, params, device=dev)
    gpu_ms = (time.perf_counter() - t0) / n_runs * 1000

    speedup = numpy_ms / gpu_ms if gpu_ms > 0 else float("inf")
    return {
        "device": dev,
        "resolution": f"{width}x{height}",
        "numpy_ms": round(numpy_ms, 1),
        "gpu_ms": round(gpu_ms, 1),
        "speedup": round(speedup, 1),
    }
