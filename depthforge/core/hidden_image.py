"""
depthforge.core.hidden_image
=============================
Hidden image (magic-eye shape) encoding.

Unlike standard stereograms where the *depth map* encodes a continuous 3D
scene, hidden image mode uses a **binary mask** to embed a recognisable
shape at a single fixed depth plane.  The rest of the image appears flat.

This produces the classic "stare at it and a shape pops out" effect.

Usage
-----
    mask  = load_hidden_mask("logo.png")          # white=hidden, black=bg
    tile  = generate_pattern(PatternParams(...))
    stereo = encode_hidden_image(tile, mask, params)

The mask may be:
  - A binary B/W image (white = shape region)
  - A greyscale image (brighter = more depth = more prominent)
  - Any shape: text, logos, animals, icons

Algorithm
---------
We build a synthetic depth map from the mask:
  depth = background_depth * (1 - mask) + foreground_depth * mask

Then feed this into the standard synthesizer.  This keeps the hidden image
code thin and entirely re-uses the core engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from depthforge.core.synthesizer import StereoParams, synthesize

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass
class HiddenImageParams:
    """Controls for hidden image encoding.

    Parameters
    ----------
    foreground_depth : float
        Depth value (0–1) assigned to the hidden shape.
        Higher = closer = more parallax = easier to fuse.
        Default 0.8 (clearly in front of background).
    background_depth : float
        Depth value for the surrounding area.
        Default 0.0 (flat background / no depth).
    edge_soften_px : int
        Gaussian blur radius applied to the mask edges before encoding.
        Softening helps the eye fuse smoothly.  0 = sharp edges.
    depth_scale : float
        Additional multiplier on the shape's depth contrast.
        1.0 = normal; 1.5 = more pop; 0.5 = subtler.
    invert_mask : bool
        If True, the *dark* regions of the mask become the hidden shape.
    stereo_params : StereoParams
        Forwarded to the core synthesizer.
    """

    foreground_depth: float = 0.8
    background_depth: float = 0.0
    edge_soften_px: int = 4
    depth_scale: float = 1.0
    invert_mask: bool = False
    stereo_params: StereoParams = None  # type: ignore

    def __post_init__(self) -> None:
        if self.stereo_params is None:
            self.stereo_params = StereoParams(depth_factor=0.35)
        if not (0.0 <= self.foreground_depth <= 1.0):
            raise ValueError("foreground_depth must be in [0, 1]")
        if not (0.0 <= self.background_depth <= 1.0):
            raise ValueError("background_depth must be in [0, 1]")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def encode_hidden_image(
    pattern: np.ndarray,
    mask: np.ndarray,
    params: HiddenImageParams = HiddenImageParams(),
) -> np.ndarray:
    """Synthesise a stereogram with a hidden shape encoded in the dot field.

    Parameters
    ----------
    pattern : np.ndarray
        RGBA uint8 tile (tile_H, tile_W, 4) — the background dot pattern.
    mask : np.ndarray
        2-D array (H, W) — shape mask.  float [0,1] or uint8 [0,255].
        White/bright = shape region.
    params : HiddenImageParams

    Returns
    -------
    np.ndarray
        RGBA uint8 stereogram (H, W, 4) with the hidden image embedded.
    """
    depth = mask_to_depth(mask, params)
    return synthesize(depth, pattern, params.stereo_params)


def mask_to_depth(
    mask: np.ndarray,
    params: HiddenImageParams,
) -> np.ndarray:
    """Convert a binary/greyscale mask to a depth map for hidden-image encoding.

    Parameters
    ----------
    mask : np.ndarray
        Shape (H, W), uint8 [0,255] or float [0,1].

    Returns
    -------
    np.ndarray
        float32 (H, W) depth map ready for synthesize().
    """
    # Normalise mask to [0, 1]
    m = mask.astype(np.float32)
    if m.max() > 1.0:
        m = m / 255.0
    m = np.clip(m, 0.0, 1.0)

    if params.invert_mask:
        m = 1.0 - m

    # Soften edges
    if params.edge_soften_px > 0:
        m = _soften(m, params.edge_soften_px)

    # Map [0,1] mask → [background_depth, foreground_depth]
    fg = params.foreground_depth
    bg = params.background_depth
    depth = bg + m * (fg - bg) * params.depth_scale
    return np.clip(depth, 0.0, 1.0).astype(np.float32)


def load_hidden_mask(path: str, target_size: Optional[tuple] = None) -> np.ndarray:
    """Load an image as a hidden-image mask.

    Parameters
    ----------
    path : str
        Path to image file.  Any format Pillow supports.
    target_size : tuple (W, H) or None
        Resize to this size.  If None, use native resolution.

    Returns
    -------
    np.ndarray  float32 (H, W) in [0, 1].
    """
    img = Image.open(path).convert("L")
    if target_size is not None:
        img = img.resize(target_size, Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def text_to_mask(
    text: str,
    width: int,
    height: int,
    font_size: int = 0,
    font_path: Optional[str] = None,
    padding: int = 20,
    center: bool = True,
) -> np.ndarray:
    """Generate a hidden-image mask from a text string.

    Parameters
    ----------
    text : str
        Text to hide in the stereogram.
    width, height : int
        Output mask dimensions.
    font_size : int
        Font size in points.  0 = auto-fit.
    font_path : str or None
        Path to a .ttf font.  None = Pillow default font.
    padding : int
        Margin around text.

    Returns
    -------
    np.ndarray  float32 (height, width) in [0, 1].  Text = white (1.0).
    """
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)

    if font_path:
        try:
            fs = (
                font_size
                if font_size > 0
                else _auto_font_size(text, width, height, font_path, padding)
            )
            font = ImageFont.truetype(font_path, fs)
        except Exception:
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    if center:
        x = max(padding, (width - tw) // 2)
        y = max(padding, (height - th) // 2)
    else:
        x, y = padding, padding

    draw.text((x, y), text, fill=255, font=font)
    return np.asarray(canvas, dtype=np.float32) / 255.0


def shape_to_mask(
    shape: str,
    width: int,
    height: int,
    padding: int = 20,
) -> np.ndarray:
    """Generate a mask from a named primitive shape.

    Parameters
    ----------
    shape : str
        One of: "circle", "square", "triangle", "star", "diamond", "arrow".
    width, height : int
        Output mask dimensions.

    Returns
    -------
    np.ndarray  float32 (height, width) in [0, 1].
    """
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    p = padding
    cx, cy = width // 2, height // 2
    rx = width // 2 - p
    ry = height // 2 - p

    shape = shape.lower()

    if shape == "circle":
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)

    elif shape == "square":
        draw.rectangle([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)

    elif shape == "diamond":
        pts = [(cx, cy - ry), (cx + rx, cy), (cx, cy + ry), (cx - rx, cy)]
        draw.polygon(pts, fill=255)

    elif shape == "triangle":
        pts = [(cx, cy - ry), (cx + rx, cy + ry), (cx - rx, cy + ry)]
        draw.polygon(pts, fill=255)

    elif shape == "arrow":
        # Rightward arrow
        hw = rx // 3
        pts = [
            (cx - rx, cy - hw),
            (cx, cy - hw),
            (cx, cy - ry),
            (cx + rx, cy),
            (cx, cy + ry),
            (cx, cy + hw),
            (cx - rx, cy + hw),
        ]
        draw.polygon(pts, fill=255)

    elif shape == "star":
        pts = _star_points(cx, cy, rx, ry, 5)
        draw.polygon(pts, fill=255)

    else:
        raise ValueError(
            f"Unknown shape '{shape}'. " "Use: circle, square, triangle, star, diamond, arrow"
        )

    return np.asarray(canvas, dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _soften(mask: np.ndarray, radius: int) -> np.ndarray:
    """Gaussian blur via PIL for edge softening."""
    img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    img = img.filter(ImageFilter.GaussianBlur(radius))
    return np.asarray(img, dtype=np.float32) / 255.0


def _auto_font_size(text: str, width: int, height: int, font_path: str, padding: int) -> int:
    """Binary search for the largest font that fits within the canvas."""
    from PIL import ImageDraw, ImageFont

    lo, hi = 8, min(width, height) - padding * 2
    best = lo
    tmp = Image.new("L", (width, height))
    draw = ImageDraw.Draw(tmp)
    for _ in range(20):
        mid = (lo + hi) // 2
        font = ImageFont.truetype(font_path, mid)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if tw < width - padding * 2 and th < height - padding * 2:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _star_points(cx: int, cy: int, rx: int, ry: int, n: int) -> list:
    """Compute polygon points for an n-pointed star."""
    import math

    pts = []
    outer = (rx, ry)
    inner = (rx // 2, ry // 2)
    for i in range(n * 2):
        angle = math.pi / n * i - math.pi / 2
        r = outer if i % 2 == 0 else inner
        pts.append(
            (
                cx + int(r[0] * math.cos(angle)),
                cy + int(r[1] * math.sin(angle)),
            )
        )
    return pts
