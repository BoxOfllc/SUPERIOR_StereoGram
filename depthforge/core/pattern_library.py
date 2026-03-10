"""
depthforge.core.pattern_library
================================
Curated library of 30+ named patterns organized by category.

Usage
-----
    from depthforge.core.pattern_library import get_pattern, list_patterns, list_categories

    pat = get_pattern("northern_lights", width=256, height=256, seed=42)
    pat = get_pattern("circuit_board", width=128, height=128)
    categories = list_categories()
    names = list_patterns(category="organic")

All functions return RGBA uint8 numpy arrays (H, W, 4).

Categories
----------
    organic      Noise, plasma, fluid, natural textures
    geometric    Grids, hexagons, circuits, crystalline patterns
    psychedelic  High-colour, kaleidoscope, fractal, trippy patterns
    minimal      Low-contrast, subtle, professional, print-safe
    animated     Patterns designed for video (temporal coherence)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass
class PatternEntry:
    name: str
    category: str
    description: str
    fn: Callable
    safe_mode: bool = True  # False = high contrast, not epilepsy-safe


_LIBRARY: dict[str, PatternEntry] = {}


def _register(name: str, category: str, description: str, safe_mode: bool = True):
    """Decorator to register a pattern generator function."""

    def decorator(fn):
        _LIBRARY[name] = PatternEntry(
            name=name, category=category, description=description, fn=fn, safe_mode=safe_mode
        )
        return fn

    return decorator


def list_patterns(category: Optional[str] = None) -> list[str]:
    """Return names of all registered patterns, optionally filtered by category."""
    if category:
        return [n for n, e in _LIBRARY.items() if e.category == category.lower()]
    return list(_LIBRARY.keys())


def list_categories() -> list[str]:
    """Return the list of all pattern categories."""
    return sorted({e.category for e in _LIBRARY.values()})


def get_pattern(
    name: str,
    width: int = 128,
    height: int = 128,
    seed: int = 42,
    scale: float = 1.0,
) -> np.ndarray:
    """Generate a named pattern tile.

    Parameters
    ----------
    name : str
        Pattern name. See ``list_patterns()`` for available names.
    width, height : int
        Output tile dimensions in pixels.
    seed : int
        Random seed for reproducibility.
    scale : float
        Pattern zoom (1.0 = default). >1 = zoomed in (larger features).

    Returns
    -------
    np.ndarray  uint8 RGBA (height, width, 4)
    """
    name = name.lower().strip()
    if name not in _LIBRARY:
        available = ", ".join(sorted(_LIBRARY.keys()))
        raise KeyError(f"Unknown pattern {name!r}. Available: {available}")
    rng = np.random.default_rng(seed)
    return _LIBRARY[name].fn(width, height, rng, scale)


def get_pattern_info(name: str) -> PatternEntry:
    """Return metadata for a named pattern."""
    name = name.lower().strip()
    if name not in _LIBRARY:
        raise KeyError(f"Unknown pattern: {name!r}")
    return _LIBRARY[name]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rgba(r, g, b, H, W) -> np.ndarray:
    """Combine float32 RGB channels (0–1) into RGBA uint8."""
    out = np.zeros((H, W, 4), dtype=np.uint8)
    out[..., 0] = np.clip(r * 255, 0, 255).astype(np.uint8)
    out[..., 1] = np.clip(g * 255, 0, 255).astype(np.uint8)
    out[..., 2] = np.clip(b * 255, 0, 255).astype(np.uint8)
    out[..., 3] = 255
    return out


def _grey(v, H, W) -> np.ndarray:
    """Single float32 value map (0-1) → RGBA uint8 greyscale."""
    u = np.clip(v * 255, 0, 255).astype(np.uint8)
    return np.stack([u, u, u, np.full((H, W), 255, np.uint8)], axis=-1)


def _hsv_to_rgb(h, s, v):
    """Vectorised HSV→RGB. All inputs float32 [0,1]."""
    h6 = h * 6.0
    i = np.floor(h6).astype(int) % 6
    f = h6 - np.floor(h6)
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r = np.where(
        i == 0,
        v,
        np.where(i == 1, q, np.where(i == 2, p, np.where(i == 3, p, np.where(i == 4, t, v)))),
    )
    g = np.where(
        i == 0,
        t,
        np.where(i == 1, v, np.where(i == 2, v, np.where(i == 3, q, np.where(i == 4, p, p)))),
    )
    b = np.where(
        i == 0,
        p,
        np.where(i == 1, p, np.where(i == 2, t, np.where(i == 3, v, np.where(i == 4, v, q)))),
    )
    return r.astype(np.float32), g.astype(np.float32), b.astype(np.float32)


def _smooth_noise(H, W, rng, scale, octaves=4) -> np.ndarray:
    """Multi-octave smooth noise (Perlin-like, NumPy-only)."""
    out = np.zeros((H, W), dtype=np.float32)
    amp = 1.0
    freq = 1.0 / scale
    total_amp = 0.0
    for _ in range(octaves):
        nx = max(2, int(W * freq))
        ny = max(2, int(H * freq))
        coarse = rng.random((ny, nx), dtype=np.float32)
        from PIL import Image as _PILImage

        out += (
            amp
            * np.array(
                _PILImage.fromarray((coarse * 255).astype(np.uint8)).resize(
                    (W, H), _PILImage.BILINEAR
                ),
                dtype=np.float32,
            )
            / 255.0
        )
        total_amp += amp
        amp *= 0.5
        freq *= 2.0
    return (out / total_amp).astype(np.float32)


# ===========================================================================
# ORGANIC patterns
# ===========================================================================


@_register("perlin_noise", "organic", "Classic multi-octave smooth noise")
def _perlin_noise(W, H, rng, scale):
    v = _smooth_noise(H, W, rng, scale * 2)
    return _grey(v, H, W)


@_register("colored_noise", "organic", "Smooth noise with HSV colour rotation", safe_mode=False)
def _colored_noise(W, H, rng, scale):
    v = _smooth_noise(H, W, rng, scale * 2)
    r, g, b = _hsv_to_rgb(v, np.ones_like(v) * 0.8, np.ones_like(v) * 0.9)
    return _rgba(r, g, b, H, W)


@_register("northern_lights", "organic", "Aurora-style flowing coloured curtains", safe_mode=False)
def _northern_lights(W, H, rng, scale):
    y = np.linspace(0, 1, H)[:, None]
    x = np.linspace(0, 1, W)[None, :]
    n = _smooth_noise(H, W, rng, scale * 3)
    hue = np.mod(n * 2.5 + y * 0.5, 1.0)
    sat = np.clip(0.7 + n * 0.3, 0, 1)
    val = np.clip(n * 1.2, 0, 1)
    r, g, b = _hsv_to_rgb(hue, sat, val)
    return _rgba(r, g, b, H, W)


@_register("water_ripples", "organic", "Concentric ripple interference pattern")
def _water_ripples(W, H, rng, scale):
    y = np.linspace(-1, 1, H)[:, None]
    x = np.linspace(-1, 1, W)[None, :]
    off_x = rng.uniform(-0.5, 0.5)
    off_y = rng.uniform(-0.5, 0.5)
    r1 = np.sqrt((x - off_x) ** 2 + (y - off_y) ** 2)
    r2 = np.sqrt((x + off_x) ** 2 + (y + off_y) ** 2)
    v = np.sin(r1 * 20 / scale) * 0.5 + np.sin(r2 * 15 / scale) * 0.5
    v = (v + 1) / 2
    return _grey(v, H, W)


@_register("marble", "organic", "Marble veining texture")
def _marble(W, H, rng, scale):
    n = _smooth_noise(H, W, rng, scale * 2)
    y = np.linspace(0, 1, H)[:, None] * np.ones((1, W))
    v = np.sin((y + n * 0.4) * np.pi * 6 / scale)
    v = (v + 1) / 2
    return _grey(v, H, W)


@_register("wood_grain", "organic", "Concentric wood grain rings")
def _wood_grain(W, H, rng, scale):
    y = np.linspace(-1, 1, H)[:, None]
    x = np.linspace(-1, 1, W)[None, :]
    n = _smooth_noise(H, W, rng, scale * 1.5)
    r = np.sqrt(x**2 + y**2) + n * 0.3
    v = (np.sin(r * 20 / scale) + 1) / 2
    # Apply warm wood tones
    r_c = np.clip(0.6 + v * 0.4, 0, 1)
    g_c = np.clip(0.35 + v * 0.3, 0, 1)
    b_c = np.clip(0.15 + v * 0.1, 0, 1)
    return _rgba(r_c, g_c, b_c, H, W)


@_register("lava", "organic", "Glowing lava / molten rock", safe_mode=False)
def _lava(W, H, rng, scale):
    n = _smooth_noise(H, W, rng, scale * 2)
    hue = np.clip(n * 0.15, 0, 0.12)  # red→orange range
    sat = np.ones_like(n) * 0.95
    val = np.clip(0.4 + n * 0.6, 0, 1)
    r, g, b = _hsv_to_rgb(hue, sat, val)
    return _rgba(r, g, b, H, W)


@_register("clouds", "organic", "Soft cumulus cloud texture")
def _clouds(W, H, rng, scale):
    n = _smooth_noise(H, W, rng, scale * 4, octaves=6)
    v = np.clip(n * 1.5 - 0.2, 0, 1)
    # White clouds on blue sky
    r_c = np.clip(0.4 + v * 0.6, 0, 1)
    g_c = np.clip(0.6 + v * 0.4, 0, 1)
    b_c = np.clip(0.9 + v * 0.1, 0, 1)
    return _rgba(r_c, g_c, b_c, H, W)


# ===========================================================================
# GEOMETRIC patterns
# ===========================================================================


@_register("hexgrid", "geometric", "Regular hexagonal grid")
def _hexgrid(W, H, rng, scale):
    y = np.linspace(0, H / scale, H)
    x = np.linspace(0, W / scale, W)
    xx, yy = np.meshgrid(x, y)
    # Hex grid formula
    hex_w = 2.0
    hex_h = math.sqrt(3)
    col = np.floor(xx / (hex_w * 0.75)).astype(int)
    row_off = (col % 2) * (hex_h / 2)
    row = np.floor((yy + row_off) / hex_h).astype(int)
    cx = col * hex_w * 0.75 + hex_w / 2
    cy = row * hex_h + hex_h / 2 - row_off
    dx = xx - cx
    dy = yy - cy
    dist = np.sqrt(dx**2 + dy**2)
    v = np.clip(1 - dist / 0.9, 0, 1)
    return _grey(v, H, W)


@_register("dotgrid", "geometric", "Evenly spaced dot array")
def _dotgrid(W, H, rng, scale):
    spacing = max(4, int(16 / scale))
    y = np.arange(H) % spacing
    x = np.arange(W) % spacing
    yy, xx = np.meshgrid(y, x, indexing="ij")
    cx = spacing / 2
    cy = spacing / 2
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    v = (r < spacing * 0.35).astype(np.float32)
    return _grey(v, H, W)


@_register("circuit_board", "geometric", "PCB circuit board trace pattern")
def _circuit_board(W, H, rng, scale):
    img = np.zeros((H, W), dtype=np.float32)
    cell = max(8, int(20 / scale))
    for gy in range(0, H, cell):
        for gx in range(0, W, cell):
            # Horizontal or vertical trace
            if rng.random() > 0.5:
                img[gy : gy + 2, gx : gx + cell] = 0.9
            else:
                img[gy : gy + cell, gx : gx + 2] = 0.9
            # Solder pad
            cy, cx = gy + cell // 2, gx + cell // 2
            yy, xx = np.ogrid[max(0, cy - 3) : min(H, cy + 3), max(0, cx - 3) : min(W, cx + 3)]
            dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            img[max(0, cy - 3) : min(H, cy + 3), max(0, cx - 3) : min(W, cx + 3)] = np.where(
                dist < 3, 1.0, img[max(0, cy - 3) : min(H, cy + 3), max(0, cx - 3) : min(W, cx + 3)]
            )
    # Green tint
    r = img * 0.1
    g = img * 0.8
    b = img * 0.2
    return _rgba(r, g, b, H, W)


@_register("checkerboard", "geometric", "High-contrast checkerboard", safe_mode=False)
def _checkerboard(W, H, rng, scale):
    cell = max(4, int(16 / scale))
    y = np.arange(H)[:, None] // cell
    x = np.arange(W)[None, :] // cell
    v = ((x + y) % 2).astype(np.float32)
    return _grey(v, H, W)


@_register("stripes", "geometric", "Diagonal stripes")
def _stripes(W, H, rng, scale):
    y = np.arange(H)[:, None]
    x = np.arange(W)[None, :]
    period = max(4, int(16 / scale))
    v = ((x + y) % period < period // 2).astype(np.float32)
    return _grey(v * 0.5 + 0.25, H, W)


@_register("crystalline", "geometric", "Voronoi crystal / stained glass", safe_mode=False)
def _crystalline(W, H, rng, scale):
    n_cells = max(4, int(25 / scale))
    pts_x = rng.uniform(0, W, n_cells)
    pts_y = rng.uniform(0, H, n_cells)
    colors = rng.random((n_cells, 3)).astype(np.float32)

    yy, xx = np.mgrid[0:H, 0:W]
    # Vectorised nearest-cell assignment
    dx = xx[:, :, None].astype(np.float32) - pts_x[None, None, :]
    dy = yy[:, :, None].astype(np.float32) - pts_y[None, None, :]
    dist = dx**2 + dy**2
    idx = dist.argmin(axis=2)

    r = colors[idx, 0]
    g = colors[idx, 1]
    b = colors[idx, 2]
    return _rgba(r, g, b, H, W)


@_register("triangles", "geometric", "Tiling isometric triangle mesh", safe_mode=False)
def _triangles(W, H, rng, scale):
    size = max(8, int(24 / scale))
    y = np.arange(H)[:, None]
    x = np.arange(W)[None, :]
    row = y // size
    col = x // size
    fy = (y % size) / size
    fx = (x % size) / size
    upper = (fx + fy < 1).astype(int)
    cell_id = (row * 100 + col * 3 + upper) % 360
    hue = cell_id.astype(np.float32) / 360.0
    r, g, b = _hsv_to_rgb(hue, np.full_like(hue, 0.7), np.full_like(hue, 0.85))
    return _rgba(r, g, b, H, W)


@_register("isometric", "geometric", "Isometric cube illusion", safe_mode=False)
def _isometric(W, H, rng, scale):
    size = max(6, int(20 / scale))
    yy = np.arange(H)[:, None]
    xx = np.arange(W)[None, :]
    # Isometric grid
    row = yy // (size // 2)
    col = xx // size
    face = (row + col) % 3
    faces = [
        (0.7, 0.7, 0.7),  # top
        (0.4, 0.4, 0.8),  # left
        (0.2, 0.2, 0.5),  # right
    ]
    r = np.where(face == 0, 0.7, np.where(face == 1, 0.4, 0.2)).astype(np.float32)
    g = np.where(face == 0, 0.7, np.where(face == 1, 0.4, 0.2)).astype(np.float32)
    b = np.where(face == 0, 0.7, np.where(face == 1, 0.8, 0.5)).astype(np.float32)
    return _rgba(r, g, b, H, W)


# ===========================================================================
# PSYCHEDELIC patterns
# ===========================================================================


@_register("plasma_wave", "psychedelic", "Classic demoscene plasma effect", safe_mode=False)
def _plasma_wave(W, H, rng, scale):
    y = np.linspace(0, 4 * np.pi, H)[:, None] / scale
    x = np.linspace(0, 4 * np.pi, W)[None, :] / scale
    off = rng.uniform(0, np.pi * 2)
    v = (
        np.sin(x + off)
        + np.sin(y + off)
        + np.sin((x + y) / 2)
        + np.sin(np.sqrt(x**2 + y**2) / 2 + off)
    )
    v = (v + 4) / 8
    r, g, b = _hsv_to_rgb(v, np.ones_like(v) * 0.9, np.ones_like(v) * 0.95)
    return _rgba(r, g, b, H, W)


@_register("rainbow_bands", "psychedelic", "Smooth rainbow colour bands", safe_mode=False)
def _rainbow_bands(W, H, rng, scale):
    n = _smooth_noise(H, W, rng, scale * 2)
    x = np.linspace(0, 1, W)[None, :]
    hue = np.mod(x + n * 0.3 + rng.uniform(), 1.0)
    r, g, b = _hsv_to_rgb(hue, np.ones_like(hue), np.ones_like(hue))
    return _rgba(r, g, b, H, W)


@_register("kaleidoscope", "psychedelic", "Kaleidoscope radial symmetry", safe_mode=False)
def _kaleidoscope(W, H, rng, scale):
    y = np.linspace(-1, 1, H)[:, None]
    x = np.linspace(-1, 1, W)[None, :]
    angle = np.arctan2(y, x)
    radius = np.sqrt(x**2 + y**2)
    n_fold = 6
    folded = np.mod(np.abs(angle) * n_fold / np.pi, 1.0)
    hue = np.mod(folded + radius / scale * 0.5, 1.0)
    sat = np.clip(1.0 - radius * 0.3, 0, 1)
    val = np.clip(0.6 + folded * 0.4, 0, 1)
    r, g, b = _hsv_to_rgb(hue, sat, val)
    return _rgba(r, g, b, H, W)


@_register(
    "neon_grid", "psychedelic", "Glowing neon grid lines on dark background", safe_mode=False
)
def _neon_grid(W, H, rng, scale):
    spacing = max(8, int(24 / scale))
    yy = np.arange(H)[:, None]
    xx = np.arange(W)[None, :]
    dy = yy % spacing
    dx = xx % spacing
    on_h = (dy < 2).astype(np.float32)
    on_v = (dx < 2).astype(np.float32)
    glow = np.maximum(on_h, on_v)
    # Neon cyan on black
    base_hue = rng.uniform(0.4, 0.7)
    r = glow * 0.0
    g = glow * 0.9
    b = glow * 1.0
    return _rgba(r, g, b, H, W)


@_register("tie_dye", "psychedelic", "Swirling tie-dye rings", safe_mode=False)
def _tie_dye(W, H, rng, scale):
    n = _smooth_noise(H, W, rng, scale * 3)
    y = np.linspace(-1, 1, H)[:, None]
    x = np.linspace(-1, 1, W)[None, :]
    r_dist = np.sqrt(x**2 + y**2) + n * 0.4
    hue = np.mod(r_dist * 3 / scale + n, 1.0)
    r, g, b = _hsv_to_rgb(hue, np.full_like(hue, 0.85), np.full_like(hue, 0.9))
    return _rgba(r, g, b, H, W)


@_register("fractal_flame", "psychedelic", "Fractal flame colour structure", safe_mode=False)
def _fractal_flame(W, H, rng, scale):
    y = np.linspace(-2, 2, H)[:, None] / scale
    x = np.linspace(-2, 2, W)[None, :] / scale
    # Julia-set-like iteration (simplified, no loop — just mathematical)
    z_r = x
    z_i = y
    for _ in range(3):
        new_r = z_r**2 - z_i**2 + 0.355 / scale
        new_i = 2 * z_r * z_i + 0.355 / scale
        z_r, z_i = new_r, new_i
    mag = np.sqrt(z_r**2 + z_i**2)
    hue = np.mod(np.log1p(mag) * 0.3, 1.0)
    sat = np.clip(1.0 - mag * 0.05, 0, 1)
    val = np.clip(mag * 0.2, 0, 1)
    r, g, b = _hsv_to_rgb(hue, sat, val)
    return _rgba(r, g, b, H, W)


@_register("electric", "psychedelic", "High-voltage electric arc texture", safe_mode=False)
def _electric(W, H, rng, scale):
    n1 = _smooth_noise(H, W, rng, scale * 0.5, octaves=6)
    n2 = _smooth_noise(H, W, np.random.default_rng(rng.integers(10000)), scale * 0.3, octaves=4)
    bolt = np.abs(n1 - n2)
    bolt = np.clip(1.0 - bolt * 5, 0, 1) ** 3
    r = bolt * 0.4
    g = bolt * 0.6
    b = bolt * 1.0
    return _rgba(r, g, b, H, W)


# ===========================================================================
# MINIMAL patterns
# ===========================================================================


@_register("fine_grain", "minimal", "Ultra-fine photographic grain — subtle, SIRDS-safe")
def _fine_grain(W, H, rng, scale):
    v = rng.random((H, W), dtype=np.float32)
    # Soft clip to 40–60% range for low-contrast appearance
    v = 0.4 + v * 0.2
    return _grey(v, H, W)


@_register("linen", "minimal", "Linen fabric weave texture")
def _linen(W, H, rng, scale):
    n = _smooth_noise(H, W, rng, scale * 0.5, octaves=2)
    yy = np.arange(H)[:, None]
    xx = np.arange(W)[None, :]
    thread_h = np.sin(xx * 0.8 / scale) * 0.05 + 0.5
    thread_v = np.sin(yy * 0.8 / scale) * 0.05 + 0.5
    v = thread_h * 0.5 + thread_v * 0.5 + n * 0.08
    return _grey(v, H, W)


@_register("paper", "minimal", "Textured white paper surface")
def _paper(W, H, rng, scale):
    n1 = _smooth_noise(H, W, rng, scale * 4, octaves=3) * 0.12
    n2 = rng.random((H, W), dtype=np.float32) * 0.04
    v = np.clip(0.88 + n1 + n2, 0, 1)
    return _grey(v, H, W)


@_register("subtle_dots", "minimal", "Very faint dot array — print-safe")
def _subtle_dots(W, H, rng, scale):
    spacing = max(6, int(14 / scale))
    yy = np.arange(H)[:, None] % spacing
    xx = np.arange(W)[None, :] % spacing
    cy = spacing // 2
    cx = spacing // 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    v = np.where(r < spacing * 0.25, 0.35, 0.65).astype(np.float32)
    return _grey(v, H, W)


@_register("crosshatch", "minimal", "Delicate crosshatch lines")
def _crosshatch(W, H, rng, scale):
    spacing = max(4, int(12 / scale))
    yy = np.arange(H)[:, None]
    xx = np.arange(W)[None, :]
    on_h = ((xx + yy) % spacing < 1).astype(np.float32)
    on_v = ((xx - yy) % spacing < 1).astype(np.float32)
    v = np.clip(0.7 - (on_h + on_v) * 0.35, 0, 1)
    return _grey(v, H, W)


@_register("sand", "minimal", "Desert sand texture — warm minimal tones")
def _sand(W, H, rng, scale):
    n = _smooth_noise(H, W, rng, scale * 1.5, octaves=4)
    grain = rng.random((H, W), dtype=np.float32) * 0.06
    v = np.clip(0.75 + n * 0.15 + grain, 0, 1)
    r = np.clip(v * 1.0, 0, 1)
    g = np.clip(v * 0.88, 0, 1)
    b = np.clip(v * 0.65, 0, 1)
    return _rgba(r, g, b, H, W)


@_register("brushed_metal", "minimal", "Brushed aluminium surface")
def _brushed_metal(W, H, rng, scale):
    # Horizontal streaks
    streak = _smooth_noise(H, W, rng, scale * 0.3, octaves=2)
    yy = np.arange(H)[:, None]
    horizontal = np.tile(np.sin(yy * 0.5 / scale) * 0.03, (1, W))
    v = np.clip(0.65 + streak * 0.15 + horizontal, 0, 1)
    return _grey(v, H, W)


@_register("soft_gradient", "minimal", "Smooth radial gradient — maximum fusability")
def _soft_gradient(W, H, rng, scale):
    y = np.linspace(0, 1, H)[:, None]
    x = np.linspace(0, 1, W)[None, :]
    cx, cy = rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    v = np.clip(1.0 - r * 1.2 / scale, 0.2, 0.9).astype(np.float32)
    return _grey(v, H, W)


# ===========================================================================
# Convenience alias for the full pattern_gen generator (by library name)
# ===========================================================================


def available_count() -> int:
    """Return the total number of registered patterns."""
    return len(_LIBRARY)
