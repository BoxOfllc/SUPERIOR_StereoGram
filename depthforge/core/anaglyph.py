"""
depthforge.core.anaglyph
========================
Anaglyph stereo image generation.

Takes either:
  (a) a source image + depth map  → synthesise L/R then composite, or
  (b) a pre-computed L/R stereo pair   → composite directly.

Supported anaglyph methods
---------------------------
TRUE_ANAGLYPH    : Simple R from left eye, GB from right eye.
                   Fast, low ghosting on greyscale.  Colour shift present.
GREY_ANAGLYPH    : Both eyes converted to grey before separation.
                   Eliminates retinal rivalry; no colour information.
COLOUR_ANAGLYPH  : Full colour — R from left, GB from right (alias for TRUE).
HALF_COLOUR      : Left eye grey → Red; right eye colour → Cyan.
                   Good compromise for most content.
OPTIMISED        : Dubois least-squares optimisation matrices.
                   Best colour fidelity; recommended for quality output.

References
----------
- Dubois (2001) "A Methodology for Developing Display and Image Processing
  Technology for Stereoscopic Video"
- Woods, Docherty & Koch (1993) anaglyph ghosting analysis
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------

class AnaglyphMode(Enum):
    TRUE_ANAGLYPH   = auto()
    GREY_ANAGLYPH   = auto()
    COLOUR_ANAGLYPH = auto()   # alias for TRUE
    HALF_COLOUR     = auto()
    OPTIMISED       = auto()


# ---------------------------------------------------------------------------
# Dubois matrices (sRGB, red/cyan)
# Reference: http://www.site.uottawa.ca/~edubois/anaglyph/
# ---------------------------------------------------------------------------

_DUBOIS_LEFT = np.array([
    [ 0.4561,  0.500484,  0.176381],
    [-0.0400822, -0.0378246, -0.0157589],
    [-0.0152161, -0.0205971, -0.00546856],
], dtype=np.float64)

_DUBOIS_RIGHT = np.array([
    [-0.0434706, -0.0879388, -0.00155529],
    [ 0.378476,  0.73364,   -0.0184503],
    [-0.0721527, -0.112961,   1.2264],
], dtype=np.float64)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class AnaglyphParams:
    """Controls for anaglyph compositing.

    Parameters
    ----------
    mode : AnaglyphMode
        Algorithm variant.
    parallax_px : int
        Horizontal pixel shift between L and R views.
        Derived from depth_factor and max_parallax when called via
        make_anaglyph_from_depth(); set manually when compositing pairs.
    swap_eyes : bool
        Swap left/right channels.  Useful if your depth convention is
        inverted relative to the expected viewing geometry.
    gamma : float
        Gamma correction applied before compositing (1.0 = no correction).
        2.2 is standard sRGB.
    """
    mode:        AnaglyphMode = AnaglyphMode.OPTIMISED
    parallax_px: int           = 20
    swap_eyes:   bool          = False
    gamma:       float         = 1.0


def make_anaglyph(
    left:   np.ndarray,
    right:  np.ndarray,
    params: AnaglyphParams = AnaglyphParams(),
) -> np.ndarray:
    """Composite a pre-computed L/R stereo pair into an anaglyph.

    Parameters
    ----------
    left, right : np.ndarray
        RGB or RGBA uint8 arrays, same shape (H, W, 3|4).

    Returns
    -------
    np.ndarray
        RGBA uint8 anaglyph, shape (H, W, 4).
    """
    L = _to_rgb_float(left)
    R = _to_rgb_float(right)

    if params.swap_eyes:
        L, R = R, L

    if params.gamma != 1.0:
        L = _apply_gamma(L, params.gamma)
        R = _apply_gamma(R, params.gamma)

    mode = params.mode
    if mode in (AnaglyphMode.TRUE_ANAGLYPH, AnaglyphMode.COLOUR_ANAGLYPH):
        out = _true_anaglyph(L, R)
    elif mode == AnaglyphMode.GREY_ANAGLYPH:
        out = _grey_anaglyph(L, R)
    elif mode == AnaglyphMode.HALF_COLOUR:
        out = _half_colour_anaglyph(L, R)
    else:  # OPTIMISED (Dubois)
        out = _optimised_anaglyph(L, R)

    # Undo gamma on output
    if params.gamma != 1.0:
        out = _apply_gamma(out, 1.0 / params.gamma)

    rgba = np.concatenate(
        [out, np.full((*out.shape[:2], 1), 1.0)], axis=-1
    )
    return (rgba * 255).clip(0, 255).astype(np.uint8)


def make_anaglyph_from_depth(
    source: np.ndarray,
    depth:  np.ndarray,
    params: AnaglyphParams = AnaglyphParams(),
) -> np.ndarray:
    """Generate an anaglyph directly from a source image and depth map.

    Synthesises L and R views by horizontally shifting the source by
    ±parallax_px/2 based on the depth map, then composites.

    Parameters
    ----------
    source : np.ndarray
        RGB or RGBA uint8, shape (H, W, 3|4).
    depth : np.ndarray
        float32, shape (H, W), values [0, 1].
    params : AnaglyphParams

    Returns
    -------
    np.ndarray
        RGBA uint8 anaglyph, shape (H, W, 4).
    """
    left, right = _warp_stereo_views(source, depth, params.parallax_px)
    return make_anaglyph(left, right, params)


# ---------------------------------------------------------------------------
# Internal — anaglyph compositing algorithms
# ---------------------------------------------------------------------------

def _true_anaglyph(L: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Red from left eye, Cyan (G+B) from right eye."""
    out = np.zeros_like(L)
    out[:, :, 0] = L[:, :, 0]   # R  ← left
    out[:, :, 1] = R[:, :, 1]   # G  ← right
    out[:, :, 2] = R[:, :, 2]   # B  ← right
    return out


def _grey_anaglyph(L: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Both eyes converted to luminance before separation."""
    Lg = _luminance(L)
    Rg = _luminance(R)
    out        = np.zeros_like(L)
    out[:, :, 0] = Lg
    out[:, :, 1] = Rg
    out[:, :, 2] = Rg
    return out


def _half_colour_anaglyph(L: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Left eye greyscale → Red; right eye full colour → Cyan."""
    Lg         = _luminance(L)
    out        = np.zeros_like(L)
    out[:, :, 0] = Lg
    out[:, :, 1] = R[:, :, 1]
    out[:, :, 2] = R[:, :, 2]
    return out


def _optimised_anaglyph(L: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Dubois least-squares optimised anaglyph — best colour fidelity."""
    H, W = L.shape[:2]
    L_flat = L.reshape(-1, 3)   # (N, 3)
    R_flat = R.reshape(-1, 3)

    # out = L @ M_left.T + R @ M_right.T
    out_flat = (L_flat @ _DUBOIS_LEFT.T + R_flat @ _DUBOIS_RIGHT.T)
    return out_flat.reshape(H, W, 3).clip(0.0, 1.0).astype(np.float64)


# ---------------------------------------------------------------------------
# Internal — view warping
# ---------------------------------------------------------------------------

def _warp_stereo_views(
    source:      np.ndarray,
    depth:       np.ndarray,
    parallax_px: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Warp source image into L and R views using depth-based horizontal shift.

    Simple per-row horizontal warp — background pixels filled with edge-repeat.
    For production quality, use the full stereo_pair module which handles
    occlusion fill properly.
    """
    src_f = _to_rgb_float(source)
    H, W  = src_f.shape[:2]

    left  = np.zeros_like(src_f)
    right = np.zeros_like(src_f)

    for y in range(H):
        row_depth = depth[y]   # (W,)
        shift     = (row_depth * parallax_px).astype(np.int32)

        xs = np.arange(W)

        # Left eye: shift left (scene appears further for deeper pixels)
        xs_l = np.clip(xs - shift // 2, 0, W - 1)
        left[y] = src_f[y, xs_l]

        # Right eye: shift right
        xs_r = np.clip(xs + shift - shift // 2, 0, W - 1)
        right[y] = src_f[y, xs_r]

    to_u8 = lambda x: (x * 255).clip(0, 255).astype(np.uint8)
    return to_u8(left), to_u8(right)


# ---------------------------------------------------------------------------
# Internal — colour utilities
# ---------------------------------------------------------------------------

def _to_rgb_float(arr: np.ndarray) -> np.ndarray:
    """→ float64 (H, W, 3) in [0, 1]."""
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float64) / 255.0
    else:
        arr = arr.astype(np.float64)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return arr


def _luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec.709 luminance from (H, W, 3) float array → (H, W) float."""
    w = np.array([0.2126, 0.7152, 0.0722])
    return (rgb * w).sum(axis=-1)


def _apply_gamma(rgb: np.ndarray, gamma: float) -> np.ndarray:
    return np.power(np.clip(rgb, 0.0, 1.0), gamma).astype(rgb.dtype)
