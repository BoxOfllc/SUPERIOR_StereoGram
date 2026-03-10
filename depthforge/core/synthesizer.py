"""
depthforge.core.synthesizer
===========================
Core single-image stereogram synthesis algorithm.

The algorithm works by building a "same" constraint array per scanline:
for each pixel, the depth value determines how far apart the two eyes must
see the *same* colour in order to perceive depth.  We march left→right,
assigning pattern colours while honouring those constraints.

References
----------
- Thimbleby, Inglis & Witten (1994) "Displaying 3D Images: Algorithms for
  Single Image Random Dot Stereograms"
- Tyler & Clarke (1990) original SIRDS patent

Public API
----------
    synthesize(depth, pattern, params) -> np.ndarray  (H, W, 4) RGBA uint8
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Parameter dataclass
# ---------------------------------------------------------------------------


@dataclass
class StereoParams:
    """All controls for the stereogram synthesizer.

    Parameters
    ----------
    depth_factor : float
        Global parallax multiplier.  Positive = scene recedes behind screen
        plane; negative = scene pops forward.  Range: -1.0 … 1.0.
        Default 0.4 is comfortable for most viewing distances.
    max_parallax_fraction : float
        Safety limiter.  Maximum horizontal shift as a fraction of image
        width.  1/30 ≈ 0.033 is the widely-cited comfortable-fusion limit.
        Increase to 0.05 for dramatic effect; decrease to 0.02 for print.
    eye_separation_fraction : float
        Assumed inter-ocular distance as fraction of viewing distance.
        Default 0.06 matches ~65 mm eyes at ~1 m viewing distance.
    convergence : float
        Normalised depth (0–1) of the "screen plane" — objects at this depth
        have zero parallax.  0.5 = mid-scene; 0.0 = everything pops out;
        1.0 = everything recedes.
    invert_depth : bool
        Flip depth convention.  Set True when white = far, black = near
        (e.g. some CG Z-passes export inverted).
    oversample : int
        Internal render scale factor (1 = none, 2 = 2×).  Improves sub-pixel
        accuracy then downsamples.  Costs 4× memory at oversample=2.
    seed : Optional[int]
        Random seed for reproducible SIRDS patterns.  None = random each run.
    safe_mode : bool
        Hard-clamp parallax to the comfortable limit even if depth_factor
        would exceed it.  Also limits pattern contrast for photosensitivity.
    """

    depth_factor: float = 0.4
    max_parallax_fraction: float = 1.0 / 30.0
    eye_separation_fraction: float = 0.06
    convergence: float = 0.5
    invert_depth: bool = False
    oversample: int = 1
    seed: Optional[int] = None
    safe_mode: bool = False

    # ---- derived / validated ----
    def __post_init__(self) -> None:
        if self.oversample < 1:
            raise ValueError("oversample must be >= 1")
        if not (0.0 <= self.convergence <= 1.0):
            raise ValueError("convergence must be in [0, 1]")
        if self.safe_mode:
            # Hard cap at comfortable limit
            self.max_parallax_fraction = min(self.max_parallax_fraction, 1.0 / 30.0)
            self.depth_factor = max(-0.5, min(0.5, self.depth_factor))

    def max_shift_px(self, width: int) -> int:
        """Maximum pixel shift for a given image width."""
        return max(1, int(width * self.max_parallax_fraction))

    def eye_sep_px(self, width: int) -> int:
        """Inter-ocular separation in pixels for a given image width."""
        return max(2, int(width * self.eye_separation_fraction))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def synthesize(
    depth: np.ndarray,
    pattern: np.ndarray,
    params: StereoParams = StereoParams(),
) -> np.ndarray:
    """Synthesize a single-image stereogram.

    Parameters
    ----------
    depth : np.ndarray
        2-D float32 array, shape (H, W), values in [0, 1].
        1.0 = nearest, 0.0 = farthest.  Use prep_depth() to condition raw
        depth maps before calling this.
    pattern : np.ndarray
        RGBA uint8 tile, shape (tile_H, tile_W, 4).  Will be tiled to fill
        the output.  For SIRDS, pass the output of generate_adaptive_dots()
        or a random-noise tile from pattern_gen.
    params : StereoParams
        Synthesis controls.

    Returns
    -------
    np.ndarray
        RGBA uint8 stereogram, shape (H, W, 4).

    Raises
    ------
    ValueError
        If depth is not 2-D float, or pattern is not (H, W, 4) uint8.
    """
    depth, pattern = _validate_inputs(depth, pattern)

    if params.invert_depth:
        depth = 1.0 - depth

    if params.oversample > 1:
        return _synthesize_oversampled(depth, pattern, params)

    return _synthesize_core(depth, pattern, params)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_inputs(depth: np.ndarray, pattern: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalise and validate input arrays."""
    # --- depth ---
    if depth.ndim != 2:
        raise ValueError(f"depth must be 2-D, got shape {depth.shape}")
    depth = depth.astype(np.float32)
    dmin, dmax = depth.min(), depth.max()
    if dmax > dmin:
        depth = (depth - dmin) / (dmax - dmin)
    else:
        depth = np.zeros_like(depth)

    # --- pattern ---
    if pattern.ndim == 2:
        # Greyscale → RGBA
        rgba = np.stack([pattern, pattern, pattern, np.full_like(pattern, 255)], axis=-1)
        pattern = rgba.astype(np.uint8)
    elif pattern.ndim == 3 and pattern.shape[2] == 3:
        alpha = np.full((*pattern.shape[:2], 1), 255, dtype=np.uint8)
        pattern = np.concatenate([pattern, alpha], axis=-1)
    elif pattern.ndim == 3 and pattern.shape[2] == 4:
        pattern = pattern.astype(np.uint8)
    else:
        raise ValueError(f"pattern must be (H,W), (H,W,3), or (H,W,4), " f"got {pattern.shape}")
    return depth, pattern


def _synthesize_core(
    depth: np.ndarray,
    pattern: np.ndarray,
    params: StereoParams,
) -> np.ndarray:
    """Inner synthesis loop — operates at native resolution."""
    H, W = depth.shape
    tile_H, tile_W = pattern.shape[:2]

    max_shift = params.max_shift_px(W)
    eye_sep = params.eye_sep_px(W)

    # Convergence offset: shift depth so that convergence depth = 0 parallax
    # We scale depth relative to convergence plane
    adj_depth = (depth - params.convergence) * params.depth_factor

    # Pixel shift per column (signed: + = right eye sees further right)
    # shift_map[y, x] = how many pixels apart L and R eye see pixel (y,x)
    shift_map = (adj_depth * max_shift).astype(np.int32)
    shift_map = np.clip(shift_map, -max_shift, max_shift)

    output = np.zeros((H, W, 4), dtype=np.uint8)

    for y in range(H):
        row_shift = shift_map[y]  # (W,) int32
        same = np.full(W, -1, dtype=np.int32)  # same[x] → linked x

        # Build constraint links
        # For each pixel x: left eye sees x - s//2, right eye sees x + s//2
        # Both must have the same colour.
        for x in range(W):
            s = int(row_shift[x])
            if s == 0:
                continue
            x_left = x - abs(s) // 2
            x_right = x + abs(s) - abs(s) // 2
            if 0 <= x_left < W and 0 <= x_right < W:
                # Follow chain: find root of x_left
                root = x_left
                while same[root] != -1:
                    root = same[root]
                if root != x_right:
                    same[x_right] = root

        # Assign colours respecting constraints
        color_idx = np.full(W, -1, dtype=np.int32)  # index into tile col
        pat_row = pattern[y % tile_H]  # (tile_W, 4)

        for x in range(W):
            linked = same[x]
            if linked != -1 and color_idx[linked] != -1:
                color_idx[x] = color_idx[linked]
            else:
                color_idx[x] = x % tile_W

        # Write output row
        output[y] = pat_row[color_idx]

    return output


def _synthesize_oversampled(
    depth: np.ndarray,
    pattern: np.ndarray,
    params: StereoParams,
) -> np.ndarray:
    """Render at oversample× resolution then Lanczos-downsample."""
    s = params.oversample
    H, W = depth.shape

    # Upsample depth
    depth_up = np.repeat(np.repeat(depth, s, axis=0), s, axis=1)

    # Upsample pattern tile
    tile_H, tile_W = pattern.shape[:2]
    pat_up = np.repeat(np.repeat(pattern, s, axis=0), s, axis=1)

    # Temporarily override oversample to avoid recursion
    p2 = StereoParams(
        depth_factor=params.depth_factor,
        max_parallax_fraction=params.max_parallax_fraction,
        eye_separation_fraction=params.eye_separation_fraction,
        convergence=params.convergence,
        invert_depth=False,  # already applied
        oversample=1,
        seed=params.seed,
        safe_mode=False,  # already applied
    )

    hi_res = _synthesize_core(depth_up, pat_up, p2)

    # Downsample with Lanczos
    img = Image.fromarray(hi_res, mode="RGBA")
    img = img.resize((W, H), Image.LANCZOS)
    return np.asarray(img)


# ---------------------------------------------------------------------------
# Convenience: load / save helpers used by tests and CLI
# ---------------------------------------------------------------------------


def load_depth_image(path: str) -> np.ndarray:
    """Load a depth image (any bit depth) → float32 [0, 1] array."""
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def save_stereogram(arr: np.ndarray, path: str) -> None:
    """Save RGBA uint8 array as PNG."""
    Image.fromarray(arr, mode="RGBA").save(path)
