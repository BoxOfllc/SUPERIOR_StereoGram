"""
depthforge.core.stereo_pair
===========================
Left/right stereo view synthesis with occlusion mask generation.

Produces the three outputs used by professional stereo pipelines:
  - Left view
  - Right view
  - Occlusion mask  (where background is newly exposed — needs inpainting)

Algorithm
---------
For each pixel (y, x) with depth d:
  horizontal shift = d * max_parallax_px * eye_balance
  left_x  = x - shift * left_fraction
  right_x = x + shift * right_fraction

Exposed occlusion regions are detected by finding pixels where no source
pixel maps — these are flagged in the occlusion mask for inpainting.

Layout modes
------------
SEPARATE     : Returns (left, right) as separate arrays.
SIDE_BY_SIDE : Concatenated horizontally  [left | right].
TOP_BOTTOM   : Concatenated vertically    [top=left / bottom=right].
ANAGLYPH     : Delegates to anaglyph.make_anaglyph().
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

import numpy as np
from PIL import Image

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class StereoLayout(Enum):
    SEPARATE     = auto()
    SIDE_BY_SIDE = auto()
    TOP_BOTTOM   = auto()
    ANAGLYPH     = auto()


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class StereoPairParams:
    """Controls for stereo pair synthesis.

    Parameters
    ----------
    max_parallax_fraction : float
        Max horizontal shift as fraction of image width.
        Default 1/30 = comfortable-fusion limit.
    eye_balance : float
        0.5 = symmetric (equal shift for both eyes).
        0.0 = right eye only (left = source).
        1.0 = left eye only (right = source).
    layout : StereoLayout
        Output format.
    feather_px : int
        Soften occlusion mask edges by this many pixels (Gaussian blur radius).
        0 = hard binary mask.
    invert_depth : bool
        Flip depth convention before processing.
    background_fill : str
        How to fill occluded regions before inpainting:
        "edge"   = repeat nearest edge pixel (default, fast)
        "mirror" = mirror-reflect
        "black"  = fill with zero
    """
    max_parallax_fraction: float        = 1.0 / 30.0
    eye_balance:           float        = 0.5
    layout:                StereoLayout = StereoLayout.SEPARATE
    feather_px:            int          = 3
    invert_depth:          bool         = False
    background_fill:       str          = "edge"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_stereo_pair(
    source: np.ndarray,
    depth:  np.ndarray,
    params: StereoPairParams = StereoPairParams(),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthesise a stereo pair from source image and depth map.

    Parameters
    ----------
    source : np.ndarray
        RGBA or RGB uint8, shape (H, W, 3|4).
    depth : np.ndarray
        float32, shape (H, W), values [0, 1].  1 = nearest.
    params : StereoPairParams

    Returns
    -------
    left  : np.ndarray  RGBA uint8 (H, W, 4)
    right : np.ndarray  RGBA uint8 (H, W, 4)
    occ   : np.ndarray  float32    (H, W)     0=visible, 1=occluded
    """
    if params.invert_depth:
        depth = 1.0 - depth

    source_rgba = _ensure_rgba(source)
    H, W = depth.shape

    max_shift = max(1, int(W * params.max_parallax_fraction))
    shift_map = (depth * max_shift).astype(np.float32)

    l_frac = params.eye_balance
    r_frac = 1.0 - params.eye_balance

    left,  left_occ  = _warp_view(source_rgba, shift_map, -l_frac, params)
    right, right_occ = _warp_view(source_rgba, shift_map,  r_frac, params)

    # Combined occlusion mask: any pixel exposed in either view
    occ = np.maximum(left_occ, right_occ)

    if params.feather_px > 0:
        occ = _feather_mask(occ, params.feather_px)

    if params.layout == StereoLayout.SIDE_BY_SIDE:
        combined = np.concatenate([left, right], axis=1)
        return combined, combined, occ   # same array twice for convenience

    if params.layout == StereoLayout.TOP_BOTTOM:
        combined = np.concatenate([left, right], axis=0)
        return combined, combined, occ

    if params.layout == StereoLayout.ANAGLYPH:
        from depthforge.core.anaglyph import make_anaglyph, AnaglyphParams
        ana = make_anaglyph(left, right, AnaglyphParams())
        return ana, ana, occ

    return left, right, occ


def compose_side_by_side(
    left:   np.ndarray,
    right:  np.ndarray,
    gap_px: int = 0,
    gap_color: Tuple[int,int,int,int] = (0,0,0,255),
) -> np.ndarray:
    """Compose L and R into a side-by-side image, optionally with a gap."""
    if gap_px > 0:
        H = left.shape[0]
        gap = np.full((H, gap_px, 4), gap_color, dtype=np.uint8)
        return np.concatenate([left, gap, right], axis=1)
    return np.concatenate([left, right], axis=1)


# ---------------------------------------------------------------------------
# Internal — view warping
# ---------------------------------------------------------------------------

def _warp_view(
    source:    np.ndarray,
    shift_map: np.ndarray,
    direction: float,
    params:    StereoPairParams,
) -> Tuple[np.ndarray, np.ndarray]:
    """Warp source by shift_map * direction → (warped_rgba, occlusion_mask)."""
    H, W = source.shape[:2]
    output  = np.zeros((H, W, 4), dtype=np.uint8)
    covered = np.zeros((H, W), dtype=bool)

    # Per-row forward warp
    for y in range(H):
        row_shift = (shift_map[y] * direction).astype(np.int32)
        xs        = np.arange(W, dtype=np.int32)
        xs_dest   = np.clip(xs + row_shift, 0, W - 1)

        # Write — later pixels (higher x) overwrite earlier for correct occlusion
        output[y, xs_dest] = source[y, xs]
        covered[y, xs_dest] = True

    # Fill uncovered pixels
    uncovered = ~covered
    if uncovered.any():
        output = _fill_uncovered(output, uncovered, source, params.background_fill)

    return output, uncovered.astype(np.float32)


def _fill_uncovered(
    output:    np.ndarray,
    mask:      np.ndarray,   # True where pixel was NOT written
    source:    np.ndarray,
    fill_mode: str,
) -> np.ndarray:
    """Fill occluded (uncovered) pixels in output."""
    H, W = output.shape[:2]

    if fill_mode == "black":
        output[mask] = 0
        output[mask, 3] = 255
        return output

    if fill_mode == "mirror":
        # Simple horizontal mirror at borders
        ys, xs = np.where(mask)
        xs_mir = np.clip(W - 1 - xs, 0, W - 1)
        output[ys, xs] = source[ys, xs_mir]
        return output

    # "edge" — propagate nearest filled pixel horizontally
    result = output.copy()
    for y in range(H):
        row_mask = mask[y]
        if not row_mask.any():
            continue
        row_out = result[y].copy()
        # Forward pass: propagate from left
        last_valid = source[y, 0]
        for x in range(W):
            if not row_mask[x]:
                last_valid = row_out[x]
            else:
                row_out[x] = last_valid
        # Backward pass: propagate from right (fill any leading holes)
        last_valid = source[y, W - 1]
        for x in range(W - 1, -1, -1):
            if not row_mask[x]:
                last_valid = row_out[x]
            else:
                if np.all(row_out[x] == 0):
                    row_out[x] = last_valid
        result[y] = row_out
    return result


# ---------------------------------------------------------------------------
# Internal — utilities
# ---------------------------------------------------------------------------

def _ensure_rgba(arr: np.ndarray) -> np.ndarray:
    """Ensure array is RGBA uint8 (H, W, 4)."""
    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr, np.full_like(arr, 255)], axis=-1)
    elif arr.shape[2] == 3:
        alpha = np.full((*arr.shape[:2], 1), 255, dtype=np.uint8)
        arr   = np.concatenate([arr, alpha], axis=-1)
    return arr


def _feather_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Gaussian feather the occlusion mask edges."""
    if _CV2:
        k = radius * 2 + 1
        return cv2.GaussianBlur(mask, (k, k), radius * 0.5)
    # NumPy fallback — box blur
    from depthforge.core.depth_prep import _box_blur
    return _box_blur(mask, radius)
