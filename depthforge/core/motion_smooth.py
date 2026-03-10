"""
depthforge.core.motion_smooth
==============================
Motion-aware temporal depth smoothing for video stereogram sequences.

Builds on ``depthforge.core.optical_flow`` to implement a production-grade
IIR blending pipeline that:

1.  Estimates dense optical flow between consecutive source frames.
2.  Warps the previous smoothed depth into the current frame's coordinate
    space to avoid ghosting on moving subjects.
3.  Modulates the temporal blend weight per-pixel — fast-moving regions
    receive little or no smoothing, static regions receive full smoothing.
4.  Detects scene cuts and auto-resets state to prevent temporal smearing
    across discontinuities.

Graceful degradation
--------------------
- OpenCV available → Farnebäck dense flow (fast, high quality).
- SciPy available, no OpenCV → Horn-Schunck approximation (pure Python).
- Neither → zero flow (pure IIR blend, equivalent to Phase 4 behaviour).

Public API
----------
``MotionSmoothParams``
    Configuration dataclass.

``MotionSmoother(params)``
    ``.reset()``                               clear history
    ``.update(depth, source_frame) -> ndarray``  process one frame
    ``.process_sequence(depths, frames) -> list`` batch process

``motion_smooth_depth(prev_depth, curr_depth, prev_frame, curr_frame, params)``
    Stateless single-frame helper.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from depthforge import HAS_CV2, HAS_SCIPY
from depthforge.core.optical_flow import (
    detect_scene_cut,
)

if HAS_SCIPY:
    from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


class FlowBackend(Enum):
    AUTO = "auto"
    FARNEBACK = "farneback"
    HORN_SCHUNCK = "horn_schunck"
    NONE = "none"


@dataclass
class MotionSmoothParams:
    """
    Parameters for motion-aware temporal depth smoothing.

    Attributes
    ----------
    temporal_alpha : float
        Base IIR blend weight (previous → current). 0 = no smoothing,
        1 = fully locked. Typical: 0.25–0.45.
    motion_sensitivity : float
        Per-pixel blend reduction factor in high-motion regions.
        0 = ignore motion (pure IIR), 1 = zero blend on fast pixels.
    flow_scale : float
        Downsample factor for flow computation. 0.5 = half-res flow
        (2× faster, slightly lower quality).
    motion_blur_sigma : float
        Gaussian smoothing applied to the motion magnitude map.
        Prevents hard blending discontinuities at motion boundaries.
    scene_cut_threshold : float
        Frame-difference threshold above which state is auto-reset.
        Range 0–1; 0.15 is a good default.
    auto_reset_on_cut : bool
        Whether to auto-reset on detected scene cuts.
    farneback_levels : int
        Pyramid levels for Farnebäck flow (cv2 only).
    farneback_winsize : int
        Averaging window for Farnebäck (cv2 only).
    hs_iterations : int
        Iterations for Horn-Schunck fallback.
    hs_lambda : float
        Regularisation for Horn-Schunck.
    """

    temporal_alpha: float = 0.35
    motion_sensitivity: float = 0.8
    flow_scale: float = 0.5
    motion_blur_sigma: float = 3.0
    scene_cut_threshold: float = 0.15
    auto_reset_on_cut: bool = True
    farneback_levels: int = 3
    farneback_winsize: int = 15
    hs_iterations: int = 80
    hs_lambda: float = 0.1
    backend: FlowBackend = FlowBackend.AUTO


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _effective_backend(params: MotionSmoothParams) -> FlowBackend:
    if params.backend != FlowBackend.AUTO:
        return params.backend
    if HAS_CV2:
        return FlowBackend.FARNEBACK
    if HAS_SCIPY:
        return FlowBackend.HORN_SCHUNCK
    return FlowBackend.NONE


def _to_gray_u8(frame: np.ndarray) -> np.ndarray:
    """Convert any image array to uint8 greyscale."""
    if frame.ndim == 2:
        return frame.astype(np.uint8)
    rgb = frame[..., :3] if frame.shape[2] >= 3 else frame
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return gray.astype(np.uint8)


def _flow_hs(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    params: MotionSmoothParams,
) -> np.ndarray:
    """Horn-Schunck flow approximation. Returns (H, W, 2) float32."""
    if not HAS_SCIPY:
        H, W = prev_gray.shape
        return np.zeros((H, W, 2), dtype=np.float32)

    I1 = prev_gray.astype(np.float32) / 255.0
    I2 = curr_gray.astype(np.float32) / 255.0

    Ix = (
        np.roll(I1, -1, axis=1)
        - np.roll(I1, 1, axis=1)
        + np.roll(I2, -1, axis=1)
        - np.roll(I2, 1, axis=1)
    ) / 4.0
    Iy = (
        np.roll(I1, -1, axis=0)
        - np.roll(I1, 1, axis=0)
        + np.roll(I2, -1, axis=0)
        - np.roll(I2, 1, axis=0)
    ) / 4.0
    It = I2 - I1

    from scipy.ndimage import uniform_filter

    u = np.zeros_like(I1)
    v = np.zeros_like(I1)
    lam = params.hs_lambda

    for _ in range(params.hs_iterations):
        u_avg = uniform_filter(u, 3)
        v_avg = uniform_filter(v, 3)
        denom = lam + Ix * Ix + Iy * Iy + 1e-8
        update = (Ix * u_avg + Iy * v_avg + It) / denom
        u = u_avg - Ix * update
        v = v_avg - Iy * update

    return np.stack([u, v], axis=-1).astype(np.float32)


def _estimate_flow(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    params: MotionSmoothParams,
) -> Optional[np.ndarray]:
    """
    Return dense (H, W, 2) float32 flow or None if backend is NONE.
    """
    backend = _effective_backend(params)
    if backend == FlowBackend.NONE:
        return None

    H, W = prev_frame.shape[:2]
    prev_g = _to_gray_u8(prev_frame)
    curr_g = _to_gray_u8(curr_frame)

    # Downsample for speed
    scale = params.flow_scale
    if scale < 1.0 and HAS_CV2:
        import cv2

        sh, sw = max(1, int(H * scale)), max(1, int(W * scale))
        prev_g = cv2.resize(prev_g, (sw, sh), interpolation=cv2.INTER_AREA)
        curr_g = cv2.resize(curr_g, (sw, sh), interpolation=cv2.INTER_AREA)

    if backend == FlowBackend.FARNEBACK and HAS_CV2:
        import cv2

        flow_small = cv2.calcOpticalFlowFarneback(
            prev_g,
            curr_g,
            None,
            pyr_scale=0.5,
            levels=params.farneback_levels,
            winsize=params.farneback_winsize,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        ).astype(np.float32)
    else:
        flow_small = _flow_hs(prev_g, curr_g, params)

    # Upsample flow back to original resolution
    if scale < 1.0 and flow_small.shape[:2] != (H, W):
        if HAS_CV2:
            import cv2

            sx = W / flow_small.shape[1]
            sy = H / flow_small.shape[0]
            flow_small[..., 0] *= sx
            flow_small[..., 1] *= sy
            flow = cv2.resize(flow_small, (W, H), interpolation=cv2.INTER_LINEAR)
        elif HAS_SCIPY:
            from scipy.ndimage import zoom

            sx = W / flow_small.shape[1]
            sy = H / flow_small.shape[0]
            flow = np.stack(
                [
                    zoom(flow_small[..., 0], (sy, sx)) * sx,
                    zoom(flow_small[..., 1], (sy, sx)) * sy,
                ],
                axis=-1,
            )
        else:
            flow = flow_small
    else:
        flow = flow_small

    return flow.astype(np.float32)


def _warp(
    depth: np.ndarray,  # (H, W) float32
    flow: np.ndarray,  # (H, W, 2) float32
) -> np.ndarray:
    """Backward warp depth by flow. Falls back to nearest-neighbour."""
    H, W = depth.shape
    ys, xs = np.mgrid[0:H, 0:W]
    src_x = (xs + flow[..., 0]).clip(0, W - 1)
    src_y = (ys + flow[..., 1]).clip(0, H - 1)

    if HAS_SCIPY:
        from scipy.ndimage import map_coordinates

        warped = map_coordinates(depth, [src_y.ravel(), src_x.ravel()], order=1, mode="nearest")
        return warped.reshape(H, W).astype(np.float32)
    else:
        xi = np.round(src_x).astype(np.int32).clip(0, W - 1)
        yi = np.round(src_y).astype(np.int32).clip(0, H - 1)
        return depth[yi, xi].astype(np.float32)


def _blend_weight(
    flow: np.ndarray,  # (H, W, 2)
    alpha_base: float,
    sensitivity: float,
    blur_sigma: float,
) -> np.ndarray:
    """
    Per-pixel blend weight in [0, 1].
    High motion → low weight (trust current depth).
    Low motion  → high weight (smooth with previous).
    """
    mag = np.hypot(flow[..., 0], flow[..., 1])
    p95 = float(np.percentile(mag, 95)) or 1.0
    mag_norm = (mag / p95).clip(0.0, 1.0)

    if HAS_SCIPY and blur_sigma > 0:
        mag_norm = gaussian_filter(mag_norm, sigma=blur_sigma)

    weight = alpha_base * (1.0 - sensitivity * mag_norm).clip(0.0, 1.0)
    return weight.astype(np.float32)


# ---------------------------------------------------------------------------
# Stateless single-frame function
# ---------------------------------------------------------------------------


def motion_smooth_depth(
    prev_depth: np.ndarray,
    curr_depth: np.ndarray,
    prev_frame: Optional[np.ndarray],
    curr_frame: Optional[np.ndarray],
    params: Optional[MotionSmoothParams] = None,
) -> np.ndarray:
    """
    Motion-aware temporal blend of two depth maps.

    Parameters
    ----------
    prev_depth : (H, W) float32 — previous smoothed depth.
    curr_depth : (H, W) float32 — current raw depth.
    prev_frame : (H, W[, C]) uint8 — previous source colour frame (optional).
    curr_frame : (H, W[, C]) uint8 — current source colour frame (optional).
    params : MotionSmoothParams (uses defaults if None).

    Returns
    -------
    (H, W) float32 blended depth.
    """
    p = params or MotionSmoothParams()

    if p.temporal_alpha <= 0.0:
        return curr_depth.astype(np.float32)

    # Determine proxy frames for flow estimation
    if prev_frame is not None and curr_frame is not None:
        pf, cf = prev_frame, curr_frame
    else:
        # Use depth maps as greyscale proxy
        pf = (prev_depth * 255).astype(np.uint8)
        cf = (curr_depth * 255).astype(np.uint8)

    flow = _estimate_flow(pf, cf, p)

    if flow is not None:
        warped_prev = _warp(prev_depth, flow)
        alpha_map = _blend_weight(flow, p.temporal_alpha, p.motion_sensitivity, p.motion_blur_sigma)
    else:
        warped_prev = prev_depth.astype(np.float32)
        alpha_map = p.temporal_alpha

    return (alpha_map * warped_prev + (1.0 - alpha_map) * curr_depth).astype(np.float32)


# ---------------------------------------------------------------------------
# Stateful smoother
# ---------------------------------------------------------------------------


class MotionSmoother:
    """
    Stateful motion-aware depth smoother for video sequences.

    Call ``.update(depth, source_frame)`` once per frame in order.
    Call ``.reset()`` between shots/clips.
    """

    def __init__(self, params: Optional[MotionSmoothParams] = None):
        self.params = params or MotionSmoothParams()
        self._prev_depth: Optional[np.ndarray] = None
        self._prev_frame: Optional[np.ndarray] = None
        self._frame_index: int = 0

    def reset(self) -> None:
        """Clear all frame history."""
        self._prev_depth = None
        self._prev_frame = None
        self._frame_index = 0

    def update(
        self,
        depth: np.ndarray,
        source_frame: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Process one depth frame.

        Parameters
        ----------
        depth : (H, W) float32
        source_frame : (H, W[, C]) uint8, optional

        Returns
        -------
        (H, W) float32 smoothed depth.
        """
        depth = depth.astype(np.float32)

        if self._prev_depth is None:
            self._prev_depth = depth.copy()
            self._prev_frame = source_frame
            self._frame_index = 1
            return depth.copy()

        # Scene-cut detection
        if self.params.auto_reset_on_cut:
            proxy_prev = (
                self._prev_frame
                if self._prev_frame is not None
                else (self._prev_depth * 255).astype(np.uint8)
            )
            proxy_curr = (
                source_frame if source_frame is not None else (depth * 255).astype(np.uint8)
            )
            if detect_scene_cut(proxy_prev, proxy_curr, threshold=self.params.scene_cut_threshold):
                self._prev_depth = depth.copy()
                self._prev_frame = source_frame
                self._frame_index += 1
                return depth.copy()

        smoothed = motion_smooth_depth(
            self._prev_depth,
            depth,
            self._prev_frame,
            source_frame,
            self.params,
        )
        self._prev_depth = smoothed
        self._prev_frame = source_frame
        self._frame_index += 1
        return smoothed

    def process_sequence(
        self,
        depths: list[np.ndarray],
        frames: Optional[list[np.ndarray]] = None,
    ) -> list[np.ndarray]:
        """Process a full sequence. Resets state before starting."""
        self.reset()
        results = []
        for i, d in enumerate(depths):
            f = frames[i] if frames else None
            results.append(self.update(d, f))
        return results

    @property
    def frame_index(self) -> int:
        """Number of frames processed since last reset."""
        return self._frame_index
