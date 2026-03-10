"""
depthforge.core.depth_prep
==========================
Depth map conditioning pipeline.

Converts raw depth inputs (Z-passes, monocular estimates, painted maps) into
clean, stereogram-ready float32 arrays.  Each stage is individually
controllable and can be bypassed.

Pipeline order
--------------
1. Normalise    → [0, 1] float32
2. Invert       → optional convention flip
3. Bilateral    → edge-aware spatial smooth (preserves object edges)
4. Dilate       → expand near regions to prevent background fringing
5. Falloff      → remap depth distribution via curve
6. Clamp        → near/far plane limits
7. Region masks → per-mask local depth multipliers

Dependency tiers
----------------
- Tier 0 (always): NumPy — basic normalise, clamp, simple box-smooth
- Tier 1 (OpenCV): proper bilateral filter, morphological dilation
- Tier 2 (SciPy):  Gaussian smooth, advanced morphology as fallbacks
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False

try:
    from scipy.ndimage import (
        gaussian_filter,
        binary_dilation,
        grey_dilation,
    )
    _SCIPY = True
except ImportError:
    _SCIPY = False


# ---------------------------------------------------------------------------
# Curve types
# ---------------------------------------------------------------------------

class FalloffCurve(Enum):
    """Built-in depth remapping curves."""
    LINEAR      = auto()   # identity — no remap
    GAMMA       = auto()   # power-law (gamma > 1 = compress near, expand far)
    S_CURVE     = auto()   # smooth ease-in/out (cosine)
    LOGARITHMIC = auto()   # emphasises near depth differences
    EXPONENTIAL = auto()   # emphasises far depth differences


# ---------------------------------------------------------------------------
# Region mask descriptor
# ---------------------------------------------------------------------------

@dataclass
class RegionMask:
    """A painted mask that locally scales depth.

    Parameters
    ----------
    mask : np.ndarray
        2-D float32 [0, 1].  1.0 = full override; 0.0 = no effect.
    multiplier : float
        Depth multiplier where mask == 1.  0.0 = flatten to zero parallax;
        0.5 = halve depth; 1.0 = no change; 2.0 = double depth.
    """
    mask:       np.ndarray
    multiplier: float = 0.0


# ---------------------------------------------------------------------------
# Parameter dataclass
# ---------------------------------------------------------------------------

@dataclass
class DepthPrepParams:
    """Controls for the depth conditioning pipeline.

    Parameters
    ----------
    normalise : bool
        Re-normalise to [0, 1] after every stage.  Recommended True.
    invert : bool
        Flip depth convention (white=near ↔ white=far).
    bilateral_sigma_space : float
        Spatial radius for bilateral filter.  Higher = smoother across
        larger areas.  0 = disabled.
    bilateral_sigma_color : float
        Color/intensity sigma for bilateral filter.  Higher = allows more
        blending across depth edges.
    dilation_px : int
        Morphological dilation radius in pixels.  Expands near regions to
        prevent background "peeking" at depth discontinuities.  0 = off.
    smooth_passes : int
        Number of smoothing passes (each pass runs the bilateral filter).
        Usually 1 is enough; 2–3 for very noisy depth maps.
    falloff_curve : FalloffCurve
        Depth remapping curve.
    falloff_gamma : float
        Gamma exponent used when falloff_curve == GAMMA.
    near_plane : float
        Clip depths below this value to 0.  0.0 = no clipping.
    far_plane : float
        Clip depths above this value to 1.  1.0 = no clipping.
    region_masks : List[RegionMask]
        Ordered list of region overrides applied after all other processing.
    edge_preserve : bool
        If True, use edge-preserving bilateral filter; if False, use faster
        Gaussian smooth.
    """

    normalise:             bool              = True
    invert:                bool              = False
    bilateral_sigma_space: float             = 5.0
    bilateral_sigma_color: float             = 0.1
    dilation_px:           int               = 3
    smooth_passes:         int               = 1
    falloff_curve:         FalloffCurve      = FalloffCurve.LINEAR
    falloff_gamma:         float             = 1.0
    near_plane:            float             = 0.0
    far_plane:             float             = 1.0
    region_masks:          List[RegionMask]  = field(default_factory=list)
    edge_preserve:         bool              = True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prep_depth(
    raw: np.ndarray,
    params: DepthPrepParams = DepthPrepParams(),
) -> np.ndarray:
    """Condition a raw depth map for stereogram synthesis.

    Parameters
    ----------
    raw : np.ndarray
        Input depth — any shape (H, W) or (H, W, 1), any numeric dtype.
        Multi-channel arrays are collapsed to luminance.
    params : DepthPrepParams

    Returns
    -------
    np.ndarray
        float32, shape (H, W), values in [near_plane, far_plane] ⊂ [0, 1].
    """
    d = _to_float(raw)

    # Stage 1 — Normalise
    if params.normalise:
        d = _normalise(d)

    # Stage 2 — Invert
    if params.invert:
        d = 1.0 - d

    # Stage 3 — Edge-aware smooth (bilateral or Gaussian)
    if params.bilateral_sigma_space > 0:
        for _ in range(params.smooth_passes):
            if params.edge_preserve:
                d = _bilateral(d,
                               params.bilateral_sigma_space,
                               params.bilateral_sigma_color)
            else:
                d = _gaussian(d, params.bilateral_sigma_space)

    # Stage 4 — Dilation (expand near regions)
    if params.dilation_px > 0:
        d = _dilate(d, params.dilation_px)

    # Stage 5 — Falloff curve
    d = _apply_curve(d, params.falloff_curve, params.falloff_gamma)

    # Stage 6 — Near/far clamp
    d = np.clip(d, params.near_plane, params.far_plane)
    if params.normalise:
        d = _normalise(d)

    # Stage 7 — Region masks
    for rm in params.region_masks:
        d = _apply_region_mask(d, rm)

    return d.astype(np.float32)


def normalise_depth(raw: np.ndarray) -> np.ndarray:
    """Quick single-step normalise; convenience wrapper."""
    return _normalise(_to_float(raw))


def depth_from_image(path: str, params: Optional[DepthPrepParams] = None) -> np.ndarray:
    """Load a depth image from disk and run the full prep pipeline.

    Accepts: PNG, TIFF (8/16-bit), EXR greyscale (if OpenEXR available),
    or any Pillow-readable format.
    """
    from PIL import Image
    img = Image.open(path)
    # Handle 16-bit TIFFs
    if img.mode == "I;16":
        arr = np.frombuffer(img.tobytes(), dtype=np.uint16).reshape(img.size[::-1])
        d   = arr.astype(np.float32) / 65535.0
    elif img.mode in ("RGB", "RGBA"):
        # Use luminance of colour depth maps
        d = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    else:
        d = np.asarray(img.convert("L"), dtype=np.float32) / 255.0

    p = params if params is not None else DepthPrepParams()
    return prep_depth(d, p)


def compute_vergence_map(
    depth: np.ndarray,
    eye_sep_fraction: float = 0.06,
    screen_distance:  float = 600.0,   # mm
) -> np.ndarray:
    """Estimate vergence angle (degrees) per pixel.

    Useful for the Vergence Comfort Analyzer — flag pixels where
    required vergence change is too fast across the image.

    Returns
    -------
    np.ndarray  float32 (H, W)  vergence angle in degrees, [0, ~6°]
    """
    # Simplified: vergence ≈ atan(eye_sep / viewing_distance_at_depth)
    # depth=1 → at screen, depth=0 → far background
    viewing_dist = screen_distance * (1.0 + (1.0 - depth) * 2.0)
    eye_sep_mm   = screen_distance * eye_sep_fraction
    vergence_rad = np.arctan2(eye_sep_mm, viewing_dist)
    return np.degrees(vergence_rad).astype(np.float32)


def detect_window_violations(
    depth:     np.ndarray,
    threshold: float = 0.05,
) -> np.ndarray:
    """Detect stereo window violations — regions where objects incorrectly
    break the frame edge (causing uncomfortable eye-strain artifacts).

    A violation occurs where a near object (depth > threshold from edge)
    is adjacent to the image border with insufficient depth falloff.

    Returns
    -------
    np.ndarray  bool (H, W)  True where a violation is likely.
    """
    H, W = depth.shape
    violation = np.zeros((H, W), dtype=bool)

    border = 5   # pixel border width to check
    near_t = 1.0 - threshold

    # Check all four edges: if near content extends to within 'border' px
    for edge_slice, interior_slice in [
        (depth[:border,  :],    depth[border:border*3,   :]),    # top
        (depth[-border:, :],    depth[-border*3:-border, :]),    # bottom
        (depth[:,  :border],    depth[:, border:border*3]),      # left
        (depth[:, -border:],    depth[:, -border*3:-border]),    # right
    ]:
        if edge_slice.max() > near_t:
            # Near object detected at edge — flag the border region
            violation |= _border_mask(H, W, border)
            break

    return violation


# ---------------------------------------------------------------------------
# Internal — normalise / convert
# ---------------------------------------------------------------------------

def _to_float(arr: np.ndarray) -> np.ndarray:
    """Collapse to 2-D float32."""
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
        else:
            # Weighted luminance
            w = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
            arr = (arr[:, :, :3].astype(np.float32) * w).sum(axis=2)
    arr = arr.astype(np.float32)
    return arr


def _normalise(d: np.ndarray) -> np.ndarray:
    lo, hi = d.min(), d.max()
    if hi > lo:
        return (d - lo) / (hi - lo)
    return np.zeros_like(d)


# ---------------------------------------------------------------------------
# Internal — smoothing
# ---------------------------------------------------------------------------

def _bilateral(d: np.ndarray, sigma_space: float, sigma_color: float) -> np.ndarray:
    """Edge-aware bilateral filter.  Prefers OpenCV; falls back to iterative."""
    if _CV2:
        # cv2 bilateral needs uint8 or float32; d is float32 [0,1]
        d8     = (d * 255.0).astype(np.uint8)
        d_size = max(3, int(sigma_space) * 2 + 1)
        smooth = cv2.bilateralFilter(d8, d_size,
                                     sigma_color * 255,
                                     sigma_space)
        return smooth.astype(np.float32) / 255.0

    if _SCIPY:
        # Gaussian is not bilateral but is a reasonable fallback
        return gaussian_filter(d, sigma=sigma_space * 0.5).astype(np.float32)

    # Pure NumPy box blur (fast, no edge preservation)
    return _box_blur(d, max(1, int(sigma_space)))


def _gaussian(d: np.ndarray, sigma: float) -> np.ndarray:
    """Simple Gaussian smooth."""
    if _SCIPY:
        return gaussian_filter(d, sigma=sigma * 0.5).astype(np.float32)
    return _box_blur(d, max(1, int(sigma)))


def _box_blur(d: np.ndarray, radius: int) -> np.ndarray:
    """Pure NumPy separable box blur — fallback when CV2/SciPy absent."""
    k = 2 * radius + 1
    kernel = np.ones(k, dtype=np.float32) / k
    d2 = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="same"), 1, d)
    d2 = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode="same"), 0, d2)
    return d2.astype(np.float32)


# ---------------------------------------------------------------------------
# Internal — dilation
# ---------------------------------------------------------------------------

def _dilate(d: np.ndarray, radius: int) -> np.ndarray:
    """Morphological grey-scale dilation (expand near / bright regions)."""
    if _CV2:
        k   = cv2.getStructuringElement(
                  cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        d8  = (d * 255).astype(np.uint8)
        out = cv2.dilate(d8, k)
        return out.astype(np.float32) / 255.0

    if _SCIPY:
        return grey_dilation(d, size=radius * 2 + 1).astype(np.float32)

    # Pure NumPy sliding-window max (slow but correct)
    from numpy.lib.stride_tricks import sliding_window_view
    pad = np.pad(d, radius, mode="edge")
    win = sliding_window_view(pad, (radius*2+1, radius*2+1))
    return win.max(axis=(-2, -1)).astype(np.float32)


# ---------------------------------------------------------------------------
# Internal — falloff curves
# ---------------------------------------------------------------------------

def _apply_curve(
    d: np.ndarray,
    curve: FalloffCurve,
    gamma: float,
) -> np.ndarray:
    d = np.clip(d, 0.0, 1.0)
    if curve == FalloffCurve.LINEAR:
        return d
    if curve == FalloffCurve.GAMMA:
        g = max(0.01, gamma)
        return np.power(d, g).astype(np.float32)
    if curve == FalloffCurve.S_CURVE:
        # Smooth step: 3t² − 2t³
        return (3.0 * d**2 - 2.0 * d**3).astype(np.float32)
    if curve == FalloffCurve.LOGARITHMIC:
        return (np.log1p(d * 9.0) / np.log(10.0)).astype(np.float32)
    if curve == FalloffCurve.EXPONENTIAL:
        return ((np.exp(d * 2.0) - 1.0) / (np.e**2 - 1.0)).astype(np.float32)
    return d


# ---------------------------------------------------------------------------
# Internal — region masks
# ---------------------------------------------------------------------------

def _apply_region_mask(d: np.ndarray, rm: RegionMask) -> np.ndarray:
    mask = np.clip(rm.mask.astype(np.float32), 0.0, 1.0)
    if mask.shape != d.shape:
        # Resize mask to match depth
        if _CV2:
            mask = cv2.resize(mask, (d.shape[1], d.shape[0]),
                              interpolation=cv2.INTER_LINEAR)
        else:
            from PIL import Image
            m_img = Image.fromarray((mask * 255).astype(np.uint8))
            m_img = m_img.resize((d.shape[1], d.shape[0]), Image.BILINEAR)
            mask  = np.asarray(m_img).astype(np.float32) / 255.0
    multiplied = d * rm.multiplier
    return (d * (1.0 - mask) + multiplied * mask).astype(np.float32)


def _border_mask(H: int, W: int, border: int) -> np.ndarray:
    m = np.zeros((H, W), dtype=bool)
    m[:border,  :] = True
    m[-border:, :] = True
    m[:,  :border] = True
    m[:, -border:] = True
    return m
