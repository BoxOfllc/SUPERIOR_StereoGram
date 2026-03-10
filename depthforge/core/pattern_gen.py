"""
depthforge.core.pattern_gen
===========================
Procedural pattern generation and tile management for stereogram synthesis.

Generates tileable RGBA uint8 patterns that feed the synthesizer.
All generators return (tile_H, tile_W, 4) uint8 arrays.

Built-in procedural generators
-------------------------------
- random_noise      : classic SIRDS salt-and-pepper dot field
- perlin_noise      : smooth organic noise (OpenSimplex-compatible fallback)
- plasma            : classic demoscene plasma / lava-lamp
- voronoi           : cellular / Worley noise
- geometric_grid    : dots, hexes, checks, stripes
- mandelbrot        : fractal tile
- dot_matrix        : halftone / dot-screen style

All generators accept a `seed` parameter for reproducibility.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PatternType(Enum):
    RANDOM_NOISE    = auto()
    PERLIN          = auto()
    PLASMA          = auto()
    VORONOI         = auto()
    GEOMETRIC_GRID  = auto()
    MANDELBROT      = auto()
    DOT_MATRIX      = auto()
    CUSTOM_TILE     = auto()   # user supplies an image file


class GridStyle(Enum):
    DOTS    = auto()
    HEXES   = auto()
    CHECKS  = auto()
    STRIPES = auto()


class ColorMode(Enum):
    MONOCHROME  = auto()   # single hue
    PSYCHEDELIC = auto()   # full HSV rotation
    GREYSCALE   = auto()   # classic SIRDS
    CUSTOM      = auto()   # user palette


# ---------------------------------------------------------------------------
# Parameter dataclass
# ---------------------------------------------------------------------------

@dataclass
class PatternParams:
    """Controls for pattern generation.

    Parameters
    ----------
    pattern_type : PatternType
        Which generator to use.
    tile_width, tile_height : int
        Tile dimensions in pixels.  Smaller tiles = faster; larger = more detail.
        For SIRDS random noise, 64×64 is usually enough.
        For texture patterns, 256–512 gives better visual quality.
    color_mode : ColorMode
        Colour scheme applied to the generated pattern.
    hue : float
        Base hue [0, 1] for MONOCHROME mode.
    saturation : float
        Saturation [0, 1] for MONOCHROME mode.
    scale : float
        Feature scale multiplier.  1.0 = default; 2.0 = twice as large features.
    octaves : int
        Number of noise octaves (Perlin / plasma only).
    grid_style : GridStyle
        Sub-style for GEOMETRIC_GRID type.
    grid_spacing : int
        Grid cell size in pixels.
    seed : Optional[int]
        Random seed.  None = non-deterministic.
    safe_mode : bool
        Limit contrast to reduce photosensitivity risk.
    lut_path : Optional[str]
        Path to a .cube LUT file to apply after generation.
        (Phase 3 feature — ignored silently if lut_path is None)
    """

    pattern_type:  PatternType      = PatternType.RANDOM_NOISE
    tile_width:    int               = 128
    tile_height:   int               = 128
    color_mode:    ColorMode         = ColorMode.GREYSCALE
    hue:           float             = 0.0
    saturation:    float             = 0.8
    scale:         float             = 1.0
    octaves:       int               = 4
    grid_style:    GridStyle         = GridStyle.DOTS
    grid_spacing:  int               = 8
    seed:          Optional[int]     = None
    safe_mode:     bool              = False
    lut_path:      Optional[str]     = None

    def __post_init__(self) -> None:
        if self.tile_width  < 4:  raise ValueError("tile_width must be >= 4")
        if self.tile_height < 4:  raise ValueError("tile_height must be >= 4")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_pattern(params: PatternParams = PatternParams()) -> np.ndarray:
    """Generate a tileable RGBA uint8 pattern array.

    Returns
    -------
    np.ndarray  shape (tile_H, tile_W, 4)  dtype uint8
    """
    rng = np.random.default_rng(params.seed)

    if params.pattern_type == PatternType.RANDOM_NOISE:
        grey = _random_noise(params, rng)
    elif params.pattern_type == PatternType.PERLIN:
        grey = _perlin_noise(params, rng)
    elif params.pattern_type == PatternType.PLASMA:
        grey = _plasma(params, rng)
    elif params.pattern_type == PatternType.VORONOI:
        grey = _voronoi(params, rng)
    elif params.pattern_type == PatternType.GEOMETRIC_GRID:
        grey = _geometric_grid(params)
    elif params.pattern_type == PatternType.MANDELBROT:
        grey = _mandelbrot(params)
    elif params.pattern_type == PatternType.DOT_MATRIX:
        grey = _dot_matrix(params)
    else:
        grey = _random_noise(params, rng)

    rgba = _colourise(grey, params)

    if params.safe_mode:
        rgba = _apply_safe_contrast(rgba)

    return rgba


def load_tile(path: str, tile_width: int = 0, tile_height: int = 0) -> np.ndarray:
    """Load an image file as a pattern tile.

    If tile_width/tile_height are 0, uses the image's native size.
    Always returns RGBA uint8 (H, W, 4).
    """
    img = Image.open(path).convert("RGBA")
    if tile_width > 0 and tile_height > 0:
        img = img.resize((tile_width, tile_height), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def tile_to_frame(
    tile:   np.ndarray,
    height: int,
    width:  int,
) -> np.ndarray:
    """Tile a (tH, tW, 4) pattern to fill (height, width, 4)."""
    tH, tW = tile.shape[:2]
    reps_h  = math.ceil(height / tH)
    reps_w  = math.ceil(width  / tW)
    tiled   = np.tile(tile, (reps_h, reps_w, 1))
    return tiled[:height, :width]


# ---------------------------------------------------------------------------
# Internal — grey generators  (all return float32 [0,1], shape (H, W))
# ---------------------------------------------------------------------------

def _random_noise(
    params: PatternParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """Classic SIRDS random dot noise."""
    H, W = params.tile_height, params.tile_width
    return rng.random((H, W), dtype=np.float32)


def _perlin_noise(
    params: PatternParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """Layered Perlin-style noise via summed random phase gradients.

    Pure NumPy implementation — no external dependency required.
    Uses sum of sinusoids at octave frequencies (approximates Perlin).
    """
    H, W = params.tile_height, params.tile_width
    scale = max(0.1, params.scale)
    acc   = np.zeros((H, W), dtype=np.float64)
    amp   = 1.0
    freq  = 1.0 / (min(H, W) / 4.0 * scale)
    total = 0.0

    ys = np.arange(H)[:, None]
    xs = np.arange(W)[None, :]

    for _ in range(max(1, params.octaves)):
        phase_x = rng.uniform(0, 2 * math.pi)
        phase_y = rng.uniform(0, 2 * math.pi)
        angle   = rng.uniform(0, 2 * math.pi)
        acc  += amp * (
            np.sin(freq * (xs * math.cos(angle) + ys * math.sin(angle)) + phase_x) *
            np.cos(freq * (xs * math.sin(angle) - ys * math.cos(angle)) + phase_y)
        )
        total += amp
        amp   *= 0.5
        freq  *= 2.0

    out = (acc / total + 1.0) / 2.0
    return out.clip(0, 1).astype(np.float32)


def _plasma(
    params: PatternParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """Classic demoscene plasma — sum of sinusoidal waves."""
    H, W   = params.tile_height, params.tile_width
    scale  = max(0.1, params.scale)
    ys     = np.linspace(0, math.pi * 4 * scale, H)[:, None]
    xs     = np.linspace(0, math.pi * 4 * scale, W)[None, :]

    phases = rng.uniform(0, math.pi * 2, size=5)
    val    = (
        np.sin(xs + phases[0]) +
        np.sin(ys + phases[1]) +
        np.sin((xs + ys) * 0.5 + phases[2]) +
        np.sin(np.sqrt(np.maximum(xs**2 + ys**2, 0)) * 0.5 + phases[3]) +
        np.cos(xs * 0.3 - ys * 0.7 + phases[4])
    )
    val = (val - val.min()) / (val.max() - val.min() + 1e-8)
    return val.astype(np.float32)


def _voronoi(
    params: PatternParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """Worley / Voronoi cellular noise."""
    H, W   = params.tile_height, params.tile_width
    scale  = max(0.5, params.scale)
    n_pts  = max(4, int(16 * scale))

    pts_y = rng.uniform(0, H, n_pts)
    pts_x = rng.uniform(0, W, n_pts)

    ys = np.arange(H)[:, None]
    xs = np.arange(W)[None, :]

    min_dist = np.full((H, W), np.inf, dtype=np.float32)
    for py, px in zip(pts_y, pts_x):
        dy   = ys - py
        dx   = xs - px
        dist = np.sqrt(dy**2 + dx**2).astype(np.float32)
        min_dist = np.minimum(min_dist, dist)

    # Normalise
    d = min_dist / (min_dist.max() + 1e-8)
    return d.astype(np.float32)


def _geometric_grid(params: PatternParams) -> np.ndarray:
    """Geometric grid patterns: dots, hexes, checks, stripes."""
    H, W   = params.tile_height, params.tile_width
    sp     = max(2, params.grid_spacing)
    canvas = Image.new("L", (W, H), 0)
    draw   = ImageDraw.Draw(canvas)

    if params.grid_style == GridStyle.CHECKS:
        for y in range(0, H, sp):
            for x in range(0, W, sp):
                if ((y // sp) + (x // sp)) % 2 == 0:
                    draw.rectangle([x, y, x + sp - 1, y + sp - 1], fill=255)

    elif params.grid_style == GridStyle.STRIPES:
        for x in range(0, W, sp * 2):
            draw.rectangle([x, 0, x + sp - 1, H - 1], fill=255)

    elif params.grid_style == GridStyle.HEXES:
        r    = sp
        h_h  = int(r * math.sqrt(3))
        col  = 0
        x    = 0
        while x < W + r:
            y_off = (r // 2) if col % 2 else 0
            y     = y_off
            while y < H + r:
                _draw_hexagon(draw, x, y, r)
                y += h_h
            x  += int(r * 1.5)
            col += 1

    else:  # DOTS
        r = max(1, sp // 3)
        for y in range(sp // 2, H, sp):
            for x in range(sp // 2, W, sp):
                draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    return (np.asarray(canvas, dtype=np.float32) / 255.0)


def _mandelbrot(params: PatternParams) -> np.ndarray:
    """Mandelbrot set tile."""
    H, W   = params.tile_height, params.tile_width
    scale  = max(0.1, params.scale)
    max_it = 64

    xs = np.linspace(-2.5 / scale, 1.0 / scale, W)[None, :]
    ys = np.linspace(-1.25 / scale, 1.25 / scale, H)[:, None]
    c  = xs + 1j * ys
    z  = np.zeros_like(c)
    it = np.zeros((H, W), dtype=np.float32)

    for i in range(max_it):
        mask   = np.abs(z) <= 2
        z[mask] = z[mask]**2 + c[mask]
        it[mask] += 1

    return (it / max_it).astype(np.float32)


def _dot_matrix(params: PatternParams) -> np.ndarray:
    """Halftone / dot-matrix screen."""
    H, W   = params.tile_height, params.tile_width
    sp     = max(4, params.grid_spacing)
    canvas = np.zeros((H, W), dtype=np.float32)

    ys = np.arange(H)
    xs = np.arange(W)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")

    # Round to nearest cell centre
    cy = (yy // sp) * sp + sp // 2
    cx = (xx // sp) * sp + sp // 2
    d  = np.sqrt((yy - cy)**2 + (xx - cx)**2).astype(np.float32)
    r  = (sp // 2) * 0.7
    canvas[d < r] = 1.0
    return canvas


# ---------------------------------------------------------------------------
# Internal — colourisation
# ---------------------------------------------------------------------------

def _colourise(grey: np.ndarray, params: PatternParams) -> np.ndarray:
    """Convert greyscale [0,1] to RGBA uint8 per chosen ColorMode."""
    H, W = grey.shape

    if params.color_mode == ColorMode.GREYSCALE:
        g8   = (grey * 255).clip(0, 255).astype(np.uint8)
        rgba = np.stack([g8, g8, g8, np.full_like(g8, 255)], axis=-1)
        return rgba

    if params.color_mode == ColorMode.MONOCHROME:
        r8, g8, b8 = _hsv_to_rgb_single(params.hue, params.saturation, grey)
        rgba = np.stack([r8, g8, b8, np.full_like(r8, 255)], axis=-1)
        return rgba

    if params.color_mode == ColorMode.PSYCHEDELIC:
        # Rotate hue based on grey value — full rainbow
        h_arr = (grey + params.hue) % 1.0
        r8, g8, b8 = _hsv_to_rgb_array(h_arr, np.full_like(grey, 0.9), grey * 0.5 + 0.5)
        rgba = np.stack([r8, g8, b8, np.full_like(r8, 255)], axis=-1)
        return rgba

    # CUSTOM / fallback — greyscale
    g8   = (grey * 255).clip(0, 255).astype(np.uint8)
    return np.stack([g8, g8, g8, np.full_like(g8, 255)], axis=-1)


def _hsv_to_rgb_single(
    h: float, s: float, v_arr: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised HSV→RGB where only V varies per pixel."""
    h6   = h * 6.0
    i    = int(h6) % 6
    f    = h6 - math.floor(h6)
    p    = v_arr * (1.0 - s)
    q    = v_arr * (1.0 - s * f)
    t    = v_arr * (1.0 - s * (1.0 - f))

    lut = [(v_arr, t, p), (q, v_arr, p), (p, v_arr, t),
           (p, q, v_arr), (t, p, v_arr), (v_arr, p, q)]
    r, g, b = lut[i]
    to8 = lambda x: (x * 255).clip(0, 255).astype(np.uint8)
    return to8(r), to8(g), to8(b)


def _hsv_to_rgb_array(
    h: np.ndarray, s: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fully vectorised HSV→RGB (all channels vary per pixel)."""
    h6 = (h * 6.0).astype(np.float32)
    i  = h6.astype(np.int32) % 6
    f  = h6 - np.floor(h6)
    p  = v * (1.0 - s)
    q  = v * (1.0 - s * f)
    t  = v * (1.0 - s * (1.0 - f))

    R = np.select([i==0, i==1, i==2, i==3, i==4, i==5], [v, q, p, p, t, v])
    G = np.select([i==0, i==1, i==2, i==3, i==4, i==5], [t, v, v, q, p, p])
    B = np.select([i==0, i==1, i==2, i==3, i==4, i==5], [p, p, t, v, v, q])

    to8 = lambda x: (x * 255).clip(0, 255).astype(np.uint8)
    return to8(R), to8(G), to8(B)


def _apply_safe_contrast(rgba: np.ndarray) -> np.ndarray:
    """Reduce contrast for photosensitivity safety mode (max 50% contrast)."""
    rgb = rgba[:, :, :3].astype(np.float32) / 255.0
    mid = rgb.mean()
    rgb = mid + (rgb - mid) * 0.5
    rgba = rgba.copy()
    rgba[:, :, :3] = (rgb * 255).clip(0, 255).astype(np.uint8)
    return rgba


# ---------------------------------------------------------------------------
# Internal — drawing helpers
# ---------------------------------------------------------------------------

def _draw_hexagon(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: int) -> None:
    """Draw a filled hexagon outline."""
    pts = []
    for i in range(6):
        angle = math.pi / 3 * i
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    draw.polygon(pts, outline=200, fill=None)
