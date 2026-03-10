"""
depthforge.core.adaptive_dots
==============================
Adaptive dot density for SIRDS (Single Image Random Dot Stereogram) mode.

Standard SIRDS uses uniform random dots everywhere.  Adaptive dot density
varies the dot size and spacing based on local image complexity:

  - **High-complexity / edge regions** → smaller, denser dots
    → preserves fine hidden image detail
  - **Flat / low-complexity regions** → larger, sparser dots
    → reduces noise, cleaner appearance in empty areas

This significantly improves the legibility of hidden images with thin lines
or small text, and makes large smooth depth gradients look cleaner.

Algorithm
---------
1. Compute a complexity map from the depth map (edge magnitude + variance).
2. Discretise into N density levels.
3. For each density level, generate dots at the appropriate size/spacing.
4. Composite into a single tile.

The generated tile is then passed directly to synthesizer.synthesize()
as the pattern argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

try:
    import cv2

    _CV2 = True
except ImportError:
    _CV2 = False

try:
    from scipy.ndimage import generic_gradient_magnitude, sobel

    _SCIPY = True
except ImportError:
    _SCIPY = False


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveDotParams:
    """Controls for adaptive dot density generation.

    Parameters
    ----------
    tile_width, tile_height : int
        Output tile dimensions in pixels.
    min_dot_radius : int
        Smallest dot size (in high-complexity regions).
    max_dot_radius : int
        Largest dot size (in flat/low-complexity regions).
    min_spacing : int
        Minimum gap between dot centres (high-detail regions).
    max_spacing : int
        Maximum gap between dot centres (flat regions).
    n_levels : int
        Number of discrete complexity levels.  2 = binary; 4–8 = smooth.
    complexity_blur : float
        Gaussian blur sigma applied to the complexity map before
        discretisation.  Prevents isolated noisy speckles.
    dot_color : Tuple[int,int,int]
        RGB colour of dots.  (255, 255, 255) = white on black background.
    bg_color : Tuple[int,int,int]
        RGB background colour.
    seed : Optional[int]
        Random seed for reproducibility.
    jitter : float
        Random positional jitter as fraction of spacing.  0 = grid; 1 = full.
    """

    tile_width: int = 256
    tile_height: int = 256
    min_dot_radius: int = 1
    max_dot_radius: int = 4
    min_spacing: int = 3
    max_spacing: int = 12
    n_levels: int = 5
    complexity_blur: float = 2.0
    dot_color: Tuple[int, int, int] = (255, 255, 255)
    bg_color: Tuple[int, int, int] = (0, 0, 0)
    seed: Optional[int] = None
    jitter: float = 0.3

    def __post_init__(self) -> None:
        if self.min_dot_radius < 1:
            self.min_dot_radius = 1
        if self.max_dot_radius < self.min_dot_radius:
            self.max_dot_radius = self.min_dot_radius
        if self.min_spacing < self.min_dot_radius * 2:
            self.min_spacing = self.min_dot_radius * 2
        if self.max_spacing < self.min_spacing:
            self.max_spacing = self.min_spacing


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_adaptive_dots(
    depth: np.ndarray,
    params: AdaptiveDotParams = AdaptiveDotParams(),
) -> np.ndarray:
    """Generate an adaptive-density SIRDS dot tile driven by depth complexity.

    Parameters
    ----------
    depth : np.ndarray
        float32 (H, W) depth map — used to compute where fine detail lies.
        The output tile is the same size as the tile_width/tile_height params,
        NOT the same size as the depth map.  The tile is then passed to
        synthesize() which tiles it across the frame.
    params : AdaptiveDotParams

    Returns
    -------
    np.ndarray
        RGBA uint8 dot tile, shape (tile_H, tile_W, 4).
    """
    rng = np.random.default_rng(params.seed)
    cmap = _compute_complexity_map(depth, params)

    # Resize complexity map to tile size
    cmap_tile = _resize_map(cmap, params.tile_height, params.tile_width)

    # Discretise complexity into levels [0 … n_levels-1]
    levels = np.digitize(cmap_tile, bins=np.linspace(0, 1, params.n_levels + 1)[1:-1])

    # Build dot tile
    canvas = Image.new("RGB", (params.tile_width, params.tile_height), params.bg_color)
    draw = ImageDraw.Draw(canvas)

    for level in range(params.n_levels):
        # Higher level = more complex = smaller/denser dots
        t = level / max(1, params.n_levels - 1)  # 0 … 1
        radius = round(params.max_dot_radius - t * (params.max_dot_radius - params.min_dot_radius))
        spacing = round(params.max_spacing - t * (params.max_spacing - params.min_spacing))
        spacing = max(radius * 2 + 1, spacing)

        # Mask of pixels at this level
        level_mask = levels == level

        # Place dots on a jittered grid, only where level matches
        jitter_scale = spacing * params.jitter
        for y in range(0, params.tile_height, spacing):
            for x in range(0, params.tile_width, spacing):
                # Apply jitter
                jy = int(y + rng.uniform(-jitter_scale, jitter_scale))
                jx = int(x + rng.uniform(-jitter_scale, jitter_scale))
                jy = max(0, min(params.tile_height - 1, jy))
                jx = max(0, min(params.tile_width - 1, jx))

                # Only draw if this position matches the complexity level
                if level_mask[jy, jx]:
                    draw.ellipse(
                        [jx - radius, jy - radius, jx + radius, jy + radius],
                        fill=params.dot_color,
                    )

    # Convert to RGBA
    rgba = canvas.convert("RGBA")
    return np.asarray(rgba, dtype=np.uint8)


def complexity_from_depth(depth: np.ndarray) -> np.ndarray:
    """Convenience wrapper — compute normalised complexity map from depth only."""
    params = AdaptiveDotParams()
    return _compute_complexity_map(depth, params)


# ---------------------------------------------------------------------------
# Internal — complexity map
# ---------------------------------------------------------------------------


def _compute_complexity_map(
    depth: np.ndarray,
    params: AdaptiveDotParams,
) -> np.ndarray:
    """Compute a normalised [0, 1] complexity map from the depth map.

    Uses edge magnitude (Sobel) + local variance as complexity indicators.
    High values = high complexity = needs more dot detail.
    """
    d = depth.astype(np.float32)

    # Edge magnitude
    edges = _edge_magnitude(d)

    # Local variance (window 7×7)
    variance = _local_variance(d, window=7)

    # Combine: edges dominate, variance contributes
    complexity = 0.7 * edges + 0.3 * variance

    # Blur to avoid speckle
    if params.complexity_blur > 0:
        complexity = _gaussian_blur(complexity, params.complexity_blur)

    # Normalise
    lo, hi = complexity.min(), complexity.max()
    if hi > lo:
        complexity = (complexity - lo) / (hi - lo)
    else:
        complexity = np.zeros_like(complexity)

    return complexity.astype(np.float32)


def _edge_magnitude(d: np.ndarray) -> np.ndarray:
    """Compute edge magnitude using Sobel operator."""
    if _CV2:
        sx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(sx**2 + sy**2)
    elif _SCIPY:
        sx = sobel(d, axis=1)
        sy = sobel(d, axis=0)
        mag = np.hypot(sx, sy)
    else:
        # Pure NumPy finite difference
        sx = np.gradient(d, axis=1)
        sy = np.gradient(d, axis=0)
        mag = np.hypot(sx, sy)

    hi = mag.max()
    if hi > 0:
        mag = mag / hi
    return mag.astype(np.float32)


def _local_variance(d: np.ndarray, window: int = 7) -> np.ndarray:
    """Local variance in a sliding window via box filter trick."""
    w = window
    if _CV2:
        d2 = d**2
        mean = cv2.blur(d, (w, w))
        mean2 = cv2.blur(d2, (w, w))
        var = np.maximum(mean2 - mean**2, 0)
    else:
        # Pad and compute
        pad = w // 2
        dp = np.pad(d, pad, mode="reflect")
        from numpy.lib.stride_tricks import sliding_window_view

        wins = sliding_window_view(dp, (w, w))
        var = wins.var(axis=(-2, -1))

    hi = var.max()
    if hi > 0:
        var = var / hi
    return var.astype(np.float32)


def _gaussian_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur — OpenCV > SciPy > NumPy box blur."""
    if _CV2:
        k = int(sigma * 4) | 1
        return cv2.GaussianBlur(arr, (k, k), sigma)
    try:
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(arr, sigma=sigma).astype(np.float32)
    except ImportError:
        pass
    # Box blur fallback
    from depthforge.core.depth_prep import _box_blur

    return _box_blur(arr, max(1, int(sigma)))


def _resize_map(arr: np.ndarray, H: int, W: int) -> np.ndarray:
    """Resize 2-D array to (H, W)."""
    if _CV2:
        return cv2.resize(arr, (W, H), interpolation=cv2.INTER_LINEAR)
    img = Image.fromarray((arr * 255).astype(np.uint8))
    img = img.resize((W, H), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0
