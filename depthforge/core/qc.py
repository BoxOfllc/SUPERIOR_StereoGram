"""
depthforge.core.qc
==================
Quality-control visualisation overlays for depth maps and stereogram output.

Functions
---------
parallax_heatmap(depth, params)
    Full-colour heatmap showing parallax magnitude per pixel.

depth_band_preview(depth, n_bands)
    Divide depth into N discrete bands with distinct colours — useful for
    art-directing depth separation.

window_violation_overlay(depth, params)
    Highlight stereo window violations at frame edges.

comparison_grid(images, labels, cols)
    Arrange multiple images in a labelled grid for review.

depth_histogram(depth, bins)
    Return a histogram image of the depth distribution.

safe_zone_indicator(depth, params)
    Render a "traffic light" safe-zone band guide alongside the depth map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    from PIL import Image as _PILImage
    from PIL import ImageDraw as _Draw
    from PIL import ImageFont as _Font

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import cv2 as _cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------


@dataclass
class QCParams:
    """Parameters shared across QC functions.

    Parameters
    ----------
    frame_width : int
        Output frame width in pixels.
    depth_factor : float
        depth_factor used in synthesis (for parallax→pixel conversion).
    max_parallax_fraction : float
        Comfort limit as fraction of frame width.
    colormap : str
        Heatmap colormap: "jet", "inferno", "viridis", "plasma", "turbo".
        Default "inferno".
    label_font_size : int
        Font size for overlay labels. Default 18.
    """

    frame_width: int = 1920
    depth_factor: float = 0.35
    max_parallax_fraction: float = 1 / 30
    colormap: str = "inferno"
    label_font_size: int = 18


# ---------------------------------------------------------------------------
# Parallax heatmap
# ---------------------------------------------------------------------------


def parallax_heatmap(
    depth: np.ndarray,
    params: Optional[QCParams] = None,
    annotate: bool = True,
) -> np.ndarray:
    """Render a false-colour heatmap of parallax magnitude.

    Pixels near the comfort limit are shown in warm colours (yellow/red).
    Comfortable pixels are cool (blue/purple).

    Parameters
    ----------
    depth : np.ndarray
        Float32 depth map (H, W).
    params : QCParams, optional
    annotate : bool
        If True, draw a colourbar legend and comfort limit marker.

    Returns
    -------
    np.ndarray  uint8 RGBA (H, W, 4)
    """
    p = params or QCParams()
    H, W = depth.shape

    # Parallax in pixels
    limit_px = p.frame_width * p.max_parallax_fraction * p.depth_factor
    parallax = depth * limit_px * 2  # 0 to ~2× limit for visualisation range

    # Normalise to [0, 1] for colormap
    norm = np.clip(parallax / (limit_px * 1.5 + 1e-6), 0.0, 1.0)

    # Apply colormap
    rgb = _apply_colormap(norm, p.colormap)  # (H, W, 3) uint8
    alpha = np.full((H, W, 1), 255, dtype=np.uint8)
    out = np.concatenate([rgb, alpha], axis=-1)

    # Annotation: colourbar + limit line
    if annotate and HAS_PIL:
        out = _add_colourbar(out, limit_px, p)

    return out


# ---------------------------------------------------------------------------
# Depth band preview
# ---------------------------------------------------------------------------


def depth_band_preview(
    depth: np.ndarray,
    n_bands: int = 8,
    palette: Optional[Sequence[Tuple[int, int, int]]] = None,
) -> np.ndarray:
    """Divide depth into N discrete bands with distinct colours.

    Useful for quickly seeing how depth separates into distinct planes.
    Bands are ordered from far (band 0 = darkest) to near (band N-1 = brightest).

    Parameters
    ----------
    depth : np.ndarray
        Float32 depth map (H, W).
    n_bands : int
        Number of depth bands. Default 8.
    palette : list of (R,G,B) tuples, optional
        Custom palette. If None, generates a perceptually-spaced palette.

    Returns
    -------
    np.ndarray  uint8 RGBA (H, W, 4)
    """
    n_bands = max(2, n_bands)
    H, W = depth.shape

    if palette is None:
        palette = _generate_band_palette(n_bands)
    else:
        palette = list(palette)
        # Pad or truncate
        while len(palette) < n_bands:
            palette.append((128, 128, 128))
        palette = palette[:n_bands]

    # Quantise depth into bands
    band_idx = np.floor(np.clip(depth, 0, 1 - 1e-6) * n_bands).astype(np.int32)

    # Map bands to colours
    lut = np.array(palette, dtype=np.uint8)  # (n_bands, 3)
    rgb = lut[band_idx]  # (H, W, 3)
    alpha = np.full((H, W, 1), 255, dtype=np.uint8)
    out = np.concatenate([rgb, alpha], axis=-1)

    # Draw band boundaries
    if HAS_PIL:
        out = _draw_band_borders(out, band_idx, n_bands)

    return out


# ---------------------------------------------------------------------------
# Window violation overlay
# ---------------------------------------------------------------------------


def window_violation_overlay(
    depth: np.ndarray,
    threshold: float = 0.05,
    highlight_colour: Tuple[int, int, int] = (255, 60, 60),
    alpha: float = 0.7,
) -> np.ndarray:
    """Highlight stereo window violations with a coloured overlay.

    A stereo window violation occurs where a near object (high depth value)
    touches or crosses the frame edge without a depth falloff. This causes
    an uncomfortable "cut-off" effect for the viewer.

    Parameters
    ----------
    depth : np.ndarray
        Float32 depth map (H, W).
    threshold : float
        How close to 1.0 a depth value must be to count as "near". Default 0.05.
    highlight_colour : (R, G, B)
        Colour for violation regions. Default red.
    alpha : float
        Overlay opacity. Default 0.7.

    Returns
    -------
    np.ndarray  uint8 RGBA (H, W, 4)
    """
    from depthforge.core.comfort import _detect_window_violations

    H, W = depth.shape
    grey = (depth * 255).astype(np.uint8)
    out = np.stack([grey, grey, grey, np.full((H, W), 255, np.uint8)], axis=-1)

    violation = _detect_window_violations(depth, threshold=threshold)

    if violation.any():
        r, g, b = highlight_colour
        m = violation.astype(np.float32)[:, :, None]
        colour = np.array([r, g, b], dtype=np.float32)
        out[..., :3] = np.clip(
            out[..., :3].astype(np.float32) * (1 - m * alpha) + colour * m * alpha, 0, 255
        ).astype(np.uint8)

    return out


# ---------------------------------------------------------------------------
# Safe-zone indicator (side panel)
# ---------------------------------------------------------------------------


def safe_zone_indicator(
    depth: np.ndarray,
    params: Optional[QCParams] = None,
    indicator_width: int = 80,
) -> np.ndarray:
    """Append a "traffic light" safe-zone indicator to the right of the depth map.

    Shows three zones:
    - Green  : comfortable (below 60% of parallax limit)
    - Yellow : approaching limit (60–100%)
    - Red    : exceeds limit (> 100%)

    The indicator has a pointer showing the current max parallax position.

    Parameters
    ----------
    depth : np.ndarray
        Float32 depth map (H, W).
    params : QCParams, optional
    indicator_width : int
        Width of the appended indicator panel. Default 80.

    Returns
    -------
    np.ndarray  uint8 RGBA (H + iW, W, 4)  — depth greyscale + indicator
    """
    p = params or QCParams()
    H, W = depth.shape

    # Build depth image
    grey = (depth * 255).astype(np.uint8)
    depth_img = np.stack([grey, grey, grey, np.full((H, W), 255, np.uint8)], axis=-1)

    # Build indicator panel
    panel = np.zeros((H, indicator_width, 4), dtype=np.uint8)
    panel[..., 3] = 255

    # Traffic light zones
    green_top = 0
    green_bot = int(H * 0.6)
    yellow_top = green_bot
    yellow_bot = H
    red_top = int(H * 0.9)
    red_bot = H

    panel[green_top:green_bot, :, :3] = [50, 180, 50]
    panel[yellow_top:yellow_bot, :, :3] = [200, 180, 30]
    panel[red_top:red_bot, :, :3] = [200, 40, 40]

    # Pointer line showing current max parallax fraction
    limit_px = p.frame_width * p.max_parallax_fraction
    actual_max = float((depth * limit_px * p.depth_factor).max())
    pointer_frac = min(actual_max / (limit_px * 1.5 + 1e-6), 1.0)
    pointer_y = int(pointer_frac * H)
    panel[max(0, pointer_y - 2) : pointer_y + 3, :, :3] = [255, 255, 255]

    out = np.concatenate([depth_img, panel], axis=1)
    return out


# ---------------------------------------------------------------------------
# Depth histogram
# ---------------------------------------------------------------------------


def depth_histogram(
    depth: np.ndarray,
    bins: int = 64,
    width: int = 400,
    height: int = 200,
    bar_colour: Tuple[int, int, int] = (80, 160, 240),
    bg_colour: Tuple[int, int, int] = (20, 20, 28),
) -> np.ndarray:
    """Render a depth distribution histogram as an image.

    Parameters
    ----------
    depth : np.ndarray
        Float32 depth map (H, W).
    bins : int
        Number of histogram bins. Default 64.
    width, height : int
        Output image dimensions.

    Returns
    -------
    np.ndarray  uint8 RGBA (height, width, 4)
    """
    counts, edges = np.histogram(depth.ravel(), bins=bins, range=(0.0, 1.0))
    max_count = max(counts.max(), 1)

    out = np.full((height, width, 4), (*bg_colour, 255), dtype=np.uint8)

    bar_w = width / bins
    pad = 10  # top padding

    for i, c in enumerate(counts):
        bar_h = int((c / max_count) * (height - pad * 2))
        x0 = int(i * bar_w)
        x1 = int((i + 1) * bar_w) - 1
        y0 = height - pad - bar_h
        y1 = height - pad
        if x1 > x0 and y0 < y1:
            out[y0:y1, x0:x1, 0] = bar_colour[0]
            out[y0:y1, x0:x1, 1] = bar_colour[1]
            out[y0:y1, x0:x1, 2] = bar_colour[2]

    # Axis line
    out[height - pad, :, :3] = [80, 80, 80]

    return out


# ---------------------------------------------------------------------------
# Comparison grid
# ---------------------------------------------------------------------------


def comparison_grid(
    images: Sequence[np.ndarray],
    labels: Optional[Sequence[str]] = None,
    cols: int = 3,
    gap: int = 4,
    bg_colour: Tuple[int, int, int, int] = (15, 15, 20, 255),
    label_colour: Tuple[int, int, int] = (200, 200, 200),
) -> np.ndarray:
    """Arrange multiple RGBA images into a labelled grid.

    All images are resized to match the first image's dimensions.

    Parameters
    ----------
    images : list of np.ndarray
        RGBA uint8 arrays (H, W, 4).
    labels : list of str, optional
        One label per image.
    cols : int
        Number of columns. Default 3.

    Returns
    -------
    np.ndarray  uint8 RGBA (grid_H, grid_W, 4)
    """
    if not images:
        return np.zeros((100, 100, 4), dtype=np.uint8)

    H, W = images[0].shape[:2]
    rows = (len(images) + cols - 1) // cols
    label_h = 28 if labels else 0

    grid_h = rows * (H + gap + label_h) + gap
    grid_w = cols * (W + gap) + gap
    out = np.full((grid_h, grid_w, 4), bg_colour, dtype=np.uint8)

    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        y0 = row * (H + gap + label_h) + gap
        x0 = col * (W + gap) + gap

        # Resize if needed
        if img.shape[0] != H or img.shape[1] != W:
            if HAS_PIL:
                pil = _PILImage.fromarray(img).resize((W, H), _PILImage.LANCZOS)
                img = np.array(pil)
            else:
                # Simple nearest-neighbour
                fy = H / img.shape[0]
                fx = W / img.shape[1]
                yi = (np.arange(H) / fy).astype(int)
                xi = (np.arange(W) / fx).astype(int)
                img = img[np.ix_(yi, xi)]

        # Ensure RGBA
        if img.ndim == 2:
            img = np.stack([img, img, img, np.full_like(img, 255)], axis=-1)
        elif img.shape[2] == 3:
            img = np.concatenate([img, np.full((*img.shape[:2], 1), 255, np.uint8)], axis=-1)

        out[y0 : y0 + H, x0 : x0 + W] = img

        # Label
        if labels and idx < len(labels) and HAS_PIL:
            lbl_arr = out[y0 + H : y0 + H + label_h, x0 : x0 + W]
            if lbl_arr.size > 0:
                lbl_img = _PILImage.fromarray(lbl_arr)
                draw = _Draw.Draw(lbl_img)
                try:
                    font = _Font.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
                except Exception:
                    font = _Font.load_default()
                draw.text((4, 4), labels[idx], fill=(*label_colour, 255), font=font)
                out[y0 + H : y0 + H + label_h, x0 : x0 + W] = np.array(lbl_img)

    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _apply_colormap(norm: np.ndarray, name: str) -> np.ndarray:
    """Map normalised [0,1] float array to RGB uint8 using named colormap."""
    t = np.clip(norm, 0.0, 1.0)

    if name == "inferno":
        # Perceptually uniform dark-to-bright
        r = np.clip(3.0 * t - 0.5, 0, 1)
        g = np.clip(2.0 * t * t, 0, 1)
        b = np.clip(1.0 - 2.0 * t, 0, 1)
    elif name == "jet":
        r = np.clip(1.5 - np.abs(4 * t - 3), 0, 1)
        g = np.clip(1.5 - np.abs(4 * t - 2), 0, 1)
        b = np.clip(1.5 - np.abs(4 * t - 1), 0, 1)
    elif name == "viridis":
        r = np.clip(0.3 + 0.7 * t, 0, 1)
        g = np.clip(0.5 * t + 0.1, 0, 1)
        b = np.clip(0.8 - 0.8 * t, 0, 1)
    elif name == "plasma":
        r = np.clip(0.5 + 1.0 * t, 0, 1)
        g = np.clip(0.1 + 0.5 * t * t, 0, 1)
        b = np.clip(0.8 - 1.2 * t, 0, 1)
    elif name == "turbo":
        r = np.clip(np.sin(np.pi * t * 0.9 + 0.1), 0, 1)
        g = np.clip(np.sin(np.pi * t), 0, 1)
        b = np.clip(np.cos(np.pi * t * 0.8), 0, 1)
    else:
        # greyscale fallback
        r = g = b = t

    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def _generate_band_palette(n: int) -> list:
    """Generate n perceptually-spaced colours using HSV rotation."""
    palette = []
    for i in range(n):
        h = i / n
        # HSV → RGB (simple, no imports needed)
        h6 = h * 6
        hi = int(h6) % 6
        f = h6 - int(h6)
        p, q, t = 0.2, 0.2 + 0.8 * (1 - f), 0.2 + 0.8 * f
        v = 0.9
        cols = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]
        r, g, b = cols[hi]
        palette.append((int(r * 255), int(g * 255), int(b * 255)))
    return palette


def _add_colourbar(img: np.ndarray, limit_px: float, params: QCParams) -> np.ndarray:
    """Add a vertical colourbar with comfort-limit marker to the right of img."""
    H, W = img.shape[:2]
    bar_w = 30
    bar = np.zeros((H, bar_w, 4), dtype=np.uint8)
    for row in range(H):
        t = row / H
        rgb = _apply_colormap(np.array([[t]]), params.colormap)[0, 0]
        bar[row, :10, :3] = rgb
        bar[row, :10, 3] = 255

    # Draw limit line
    limit_frac = min(limit_px / (limit_px * 1.5 + 1e-6), 1.0)
    ly = int(limit_frac * H)
    bar[max(0, ly - 1) : ly + 2, :, :3] = [255, 255, 255]
    bar[max(0, ly - 1) : ly + 2, :, 3] = 255

    return np.concatenate([img, bar], axis=1)


def _draw_band_borders(img: np.ndarray, band_idx: np.ndarray, n_bands: int) -> np.ndarray:
    """Draw thin black lines at band boundaries."""
    if not HAS_PIL:
        return img
    pil = _PILImage.fromarray(img)
    draw = _Draw.Draw(pil)
    H, W = img.shape[:2]
    for b in range(1, n_bands):
        mask = (band_idx == b).astype(np.uint8)
        # Find top edge of this band
        rows = np.where(mask.any(axis=1))[0]
        if len(rows):
            y = rows[0]
            draw.line([(0, y), (W, y)], fill=(0, 0, 0, 180), width=1)
    return np.array(pil)
