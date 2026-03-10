"""
depthforge.core.inpainting
===========================
Occlusion region inpainting for stereo view synthesis.

When stereo pairs are generated, depth discontinuities expose background
regions that were hidden in the source image.  These "occlusion holes" must
be filled for clean output.

This module provides two strategies:

1. **Patch-based fill** (Phase 1 — no AI required)
   Classic exemplar-based inpainting.  Samples patches from the surrounding
   image area and pastes the best-matching one into each hole.

2. **AI inpainting hook** (Phase 4 — ComfyUI DF_Inpaint node)
   Stub that accepts a callback function.  When wired to a Stable Diffusion
   inpainting model in ComfyUI (or any other model), the stub delegates to it.
   Falls back to patch-based if no callback is registered.

3. **Clean plate fallback**
   If a clean plate (background plate without the foreground subject) is
   provided, it is composited directly over occluded regions — the most
   accurate solution when available.

Usage
-----
    left, right, occ_mask = make_stereo_pair(source, depth)
    left_filled  = inpaint_occlusion(left,  occ_mask, params=InpaintParams())
    right_filled = inpaint_occlusion(right, occ_mask, params=InpaintParams())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

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

class InpaintMethod(Enum):
    PATCH_BASED  = auto()   # exemplar / patch-match
    CLEAN_PLATE  = auto()   # composite from clean plate
    EDGE_EXTEND  = auto()   # fast: extend nearest edge pixels (low quality)
    AI_CALLBACK  = auto()   # delegate to external AI model
    AUTO         = auto()   # choose best available method


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class InpaintParams:
    """Controls for occlusion inpainting.

    Parameters
    ----------
    method : InpaintMethod
        Which strategy to use.  AUTO picks the best available.
    patch_size : int
        Side length of each patch in the patch-based method.
        8–16 is typical.  Larger = smoother but slower.
    search_radius : int
        How far from the hole boundary to look for matching patches.
        0 = whole image.
    clean_plate : Optional[np.ndarray]
        If provided and method is CLEAN_PLATE or AUTO, use this as the
        background source for occluded regions.
    ai_callback : Optional[Callable]
        Function with signature:
          ai_callback(image_rgba: np.ndarray, mask: np.ndarray) -> np.ndarray
        Called when method is AI_CALLBACK.  Returns filled RGBA uint8.
    dilate_mask_px : int
        Expand the occlusion mask by this many pixels before inpainting.
        Helps cover sub-pixel fringing at depth edges.
    blend_px : int
        Feather blend the inpainted region back into the original over
        this many pixels.  0 = hard composite.
    """

    method:         InpaintMethod                          = InpaintMethod.AUTO
    patch_size:     int                                    = 8
    search_radius:  int                                    = 64
    clean_plate:    Optional[np.ndarray]                   = None
    ai_callback:    Optional[Callable]                     = None
    dilate_mask_px: int                                    = 2
    blend_px:       int                                    = 3


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def inpaint_occlusion(
    image:   np.ndarray,
    mask:    np.ndarray,
    params:  InpaintParams = InpaintParams(),
) -> np.ndarray:
    """Fill occluded (exposed background) regions in a stereo view.

    Parameters
    ----------
    image : np.ndarray
        RGBA uint8, shape (H, W, 4).
    mask : np.ndarray
        float32 (H, W), values [0, 1].  1.0 = occluded / needs fill.
    params : InpaintParams

    Returns
    -------
    np.ndarray
        RGBA uint8 (H, W, 4) with occluded regions filled.
    """
    if mask.max() < 0.01:
        return image   # nothing to fill

    image = _ensure_rgba(image)
    bin_mask = _prepare_mask(mask, params.dilate_mask_px)

    method = _resolve_method(params)

    if method == InpaintMethod.CLEAN_PLATE:
        filled = _inpaint_clean_plate(image, bin_mask, params.clean_plate)

    elif method == InpaintMethod.AI_CALLBACK:
        filled = _inpaint_ai(image, bin_mask, params.ai_callback)

    elif method == InpaintMethod.EDGE_EXTEND:
        filled = _inpaint_edge_extend(image, bin_mask)

    else:  # PATCH_BASED (default)
        filled = _inpaint_patch_based(image, bin_mask,
                                      params.patch_size,
                                      params.search_radius)

    if params.blend_px > 0:
        filled = _blend_boundary(image, filled, bin_mask, params.blend_px)

    return filled


def register_ai_inpaint_callback(
    callback: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> None:
    """Register a global AI inpainting callback.

    The ComfyUI DF_Inpaint node calls this during workflow execution to
    wire in the active SD inpaint model.

    Parameters
    ----------
    callback : Callable
        Function(image_rgba, mask_float) -> image_rgba.
        Both arrays are (H, W, 4) uint8 / (H, W) float32.
    """
    global _GLOBAL_AI_CALLBACK
    _GLOBAL_AI_CALLBACK = callback


_GLOBAL_AI_CALLBACK: Optional[Callable] = None


# ---------------------------------------------------------------------------
# Internal — method resolution
# ---------------------------------------------------------------------------

def _resolve_method(params: InpaintParams) -> InpaintMethod:
    if params.method != InpaintMethod.AUTO:
        return params.method
    if params.clean_plate is not None:
        return InpaintMethod.CLEAN_PLATE
    if params.ai_callback is not None or _GLOBAL_AI_CALLBACK is not None:
        return InpaintMethod.AI_CALLBACK
    return InpaintMethod.PATCH_BASED


# ---------------------------------------------------------------------------
# Internal — inpainting strategies
# ---------------------------------------------------------------------------

def _inpaint_clean_plate(
    image:  np.ndarray,
    mask:   np.ndarray,
    plate:  np.ndarray,
) -> np.ndarray:
    """Composite clean plate over occluded regions."""
    plate_rgba = _ensure_rgba(plate)
    if plate_rgba.shape != image.shape:
        # Resize plate to match
        p = Image.fromarray(plate_rgba).resize(
            (image.shape[1], image.shape[0]), Image.LANCZOS
        )
        plate_rgba = np.asarray(p, dtype=np.uint8)

    result = image.copy()
    result[mask] = plate_rgba[mask]
    return result


def _inpaint_ai(
    image:    np.ndarray,
    mask:     np.ndarray,
    callback: Optional[Callable],
) -> np.ndarray:
    """Delegate to AI inpainting callback."""
    fn = callback or _GLOBAL_AI_CALLBACK
    if fn is None:
        # Fallback gracefully
        return _inpaint_patch_based(image, mask, patch_size=8, search_radius=64)
    try:
        result = fn(image, mask.astype(np.float32))
        return _ensure_rgba(result)
    except Exception as exc:
        import warnings
        warnings.warn(f"AI inpaint callback failed ({exc}), falling back to patch-based.")
        return _inpaint_patch_based(image, mask, patch_size=8, search_radius=64)


def _inpaint_edge_extend(
    image: np.ndarray,
    mask:  np.ndarray,
) -> np.ndarray:
    """Fast edge-extension: propagate nearest non-masked pixel left→right."""
    result = image.copy()
    H, W = image.shape[:2]
    for y in range(H):
        row_mask = mask[y]
        if not row_mask.any():
            continue
        # Forward pass
        last = image[y, 0]
        for x in range(W):
            if not row_mask[x]:
                last = result[y, x]
            else:
                result[y, x] = last
        # Backward pass to fill left edge
        last = image[y, W - 1]
        for x in range(W - 1, -1, -1):
            if not row_mask[x]:
                last = result[y, x]
            elif np.all(result[y, x] == 0):
                result[y, x] = last
    return result


def _inpaint_patch_based(
    image:         np.ndarray,
    mask:          np.ndarray,
    patch_size:    int,
    search_radius: int,
) -> np.ndarray:
    """Exemplar-based patch inpainting.

    For each masked region, finds the best-matching non-masked patch nearby
    and copies it.  Uses OpenCV inpaint when available for speed;
    falls back to a pure NumPy implementation.
    """
    if _CV2:
        # OpenCV Telea or Navier-Stokes inpaint — excellent quality, fast
        mask8 = (mask * 255).astype(np.uint8)
        rgb   = image[:, :, :3]
        filled_rgb = cv2.inpaint(rgb, mask8, inpaintRadius=patch_size,
                                 flags=cv2.INPAINT_TELEA)
        result = image.copy()
        result[:, :, :3] = filled_rgb
        result[:, :, 3][mask] = 255
        return result

    # Pure NumPy patch-match (slower, but no external deps)
    return _numpy_patch_inpaint(image, mask, patch_size, search_radius)


def _numpy_patch_inpaint(
    image:         np.ndarray,
    mask:          np.ndarray,
    patch_size:    int,
    search_radius: int,
) -> np.ndarray:
    """Simple NumPy patch-match inpainting.

    Processes each masked pixel by finding the nearest non-masked patch
    (within search_radius) that minimises SSD difference at known pixels.
    This is O(holes × search_area × patch²) — fine for small holes.
    """
    result = image.copy().astype(np.float32)
    H, W   = image.shape[:2]
    ph     = patch_size // 2
    src    = image.astype(np.float32)

    # Get all hole pixels sorted by distance to boundary (onion-peel order)
    hole_ys, hole_xs = np.where(mask)
    if len(hole_ys) == 0:
        return image

    for idx in range(len(hole_ys)):
        py, px = int(hole_ys[idx]), int(hole_xs[idx])

        # Extract query patch around (py, px) — known pixels only
        best_ssd  = np.inf
        best_val  = result[py, px].copy()

        # Search neighbourhood
        y0 = max(ph, py - search_radius)
        y1 = min(H - ph, py + search_radius + 1)
        x0 = max(ph, px - search_radius)
        x1 = min(W - ph, px + search_radius + 1)

        for sy in range(y0, y1, patch_size):
            for sx in range(x0, x1, patch_size):
                if mask[sy, sx]:
                    continue   # candidate centre must not be in hole

                # Compare patch overlap (known pixels only)
                # Query patch
                qy0, qy1 = py - ph, py + ph + 1
                qx0, qx1 = px - ph, px + ph + 1
                # Candidate patch
                cy0, cy1 = sy - ph, sy + ph + 1
                cx0, cx1 = sx - ph, sx + ph + 1

                # Clamp
                qy0 = max(0, qy0); qy1 = min(H, qy1)
                qx0 = max(0, qx0); qx1 = min(W, qx1)
                cy0 = max(0, cy0); cy1 = min(H, cy1)
                cx0 = max(0, cx0); cx1 = min(W, cx1)

                h = min(qy1 - qy0, cy1 - cy0)
                w = min(qx1 - qx0, cx1 - cx0)
                if h <= 0 or w <= 0:
                    continue

                q_patch = result[qy0:qy0+h, qx0:qx0+w]
                c_patch = src  [cy0:cy0+h, cx0:cx0+w]
                q_mask  = mask [qy0:qy0+h, qx0:qx0+w]

                # SSD on known pixels only
                known   = ~q_mask[:h, :w]
                if not known.any():
                    continue
                diff    = (q_patch[known].astype(np.float32) -
                           c_patch[known].astype(np.float32))
                ssd     = (diff ** 2).mean()

                if ssd < best_ssd:
                    best_ssd = ssd
                    # Value to use is the candidate patch centre
                    best_val = src[sy, sx]

        result[py, px] = best_val

    return result.clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Internal — mask preparation and blending
# ---------------------------------------------------------------------------

def _prepare_mask(mask: np.ndarray, dilate_px: int) -> np.ndarray:
    """Convert float mask → bool, optionally dilated."""
    bin_mask = mask > 0.5
    if dilate_px > 0 and bin_mask.any():
        if _CV2:
            k    = dilate_px * 2 + 1
            kern = np.ones((k, k), np.uint8)
            m8   = bin_mask.astype(np.uint8) * 255
            m8   = cv2.dilate(m8, kern)
            bin_mask = m8 > 127
        else:
            try:
                from scipy.ndimage import binary_dilation
                bin_mask = binary_dilation(bin_mask,
                                           iterations=dilate_px)
            except ImportError:
                pass   # Skip dilation if neither available
    return bin_mask


def _blend_boundary(
    original: np.ndarray,
    filled:   np.ndarray,
    mask:     np.ndarray,
    blend_px: int,
) -> np.ndarray:
    """Feather the inpainted region boundary back into the original."""
    if not _CV2:
        return filled   # Skip feathering without CV2 (hard composite is fine)

    # Create a smooth alpha ramp at the mask boundary
    mask8   = mask.astype(np.uint8) * 255
    k       = blend_px * 2 + 1
    alpha   = cv2.GaussianBlur(mask8.astype(np.float32), (k, k),
                               blend_px * 0.5) / 255.0
    alpha4  = alpha[:, :, None]   # (H, W, 1)

    orig_f  = original.astype(np.float32)
    fill_f  = filled.astype(np.float32)
    result  = orig_f * (1.0 - alpha4) + fill_f * alpha4
    return result.clip(0, 255).astype(np.uint8)


def _ensure_rgba(arr: np.ndarray) -> np.ndarray:
    if arr.dtype != np.uint8:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr, np.full_like(arr, 255)], axis=-1)
    elif arr.shape[2] == 3:
        alpha = np.full((*arr.shape[:2], 1), 255, dtype=np.uint8)
        arr   = np.concatenate([arr, alpha], axis=-1)
    return arr
