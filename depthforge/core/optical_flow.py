"""
depthforge.core.optical_flow
============================
Optical flow-based depth proxy estimation from consecutive video frames.

Uses the Lucas-Kanade dense optical flow (Farneback method via OpenCV) to
estimate per-pixel motion magnitude and direction, then converts that motion
field to a depth proxy suitable for stereogram synthesis.

This is *not* true metric depth — it is a motion-parallax proxy. Objects
with large motion are treated as "near"; objects with small motion as "far".
The proxy is useful for:

- Animating stereograms that track scene motion
- Generating depth from video when no depth sensor is available
- Temporal depth coherence via flow-guided warping

API
---
::

    from depthforge.core.optical_flow import (
        FlowDepthEstimator, FlowDepthConfig, compute_flow, flow_to_depth
    )

    estimator = FlowDepthEstimator(FlowDepthConfig())
    depth     = estimator.estimate_from_pair(frame_a, frame_b)

    # Or process a sequence:
    depths    = estimator.estimate_sequence([frame0, frame1, frame2, ...])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class FlowDepthConfig:
    """Configuration for optical flow depth estimation.

    Attributes
    ----------
    pyr_scale : float
        Image pyramid scale for Farneback. 0.5 = half-size each level.
    levels : int
        Number of pyramid levels. More levels = captures larger motions.
    winsize : int
        Averaging window size. Larger = smoother but blurs motion edges.
    iterations : int
        Iterations per pyramid level.
    poly_n : int
        Pixel neighbourhood for polynomial expansion (5 or 7).
    poly_sigma : float
        Gaussian sigma for polynomial expansion smoothing.
    motion_scale : float
        Multiply motion magnitude by this before normalising to [0,1].
        Higher values amplify small motions.
    blur_sigma : float
        Gaussian blur applied to the raw flow magnitude before depth
        conversion. Reduces per-pixel noise in the depth proxy.
    invert : bool
        If True, large motion = far (0) instead of near (1).
    min_motion : float
        Motion magnitudes below this threshold are treated as zero.
    max_motion : float
        Clamp motion magnitudes to this ceiling before normalisation.
        0.0 = auto (use 95th percentile of the frame's motion).
    fallback_gradient : bool
        If cv2 is unavailable, generate a simple radial gradient depth
        proxy instead of raising an error.
    """

    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.1
    motion_scale: float = 1.0
    blur_sigma: float = 3.0
    invert: bool = False
    min_motion: float = 0.1
    max_motion: float = 0.0  # 0 = auto
    fallback_gradient: bool = True


# ---------------------------------------------------------------------------
# Low-level flow functions
# ---------------------------------------------------------------------------


def _to_grey_u8(frame: np.ndarray) -> np.ndarray:
    """Convert any frame format to uint8 greyscale (H, W)."""
    if frame.ndim == 3:
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]
        if frame.dtype != np.uint8:
            frame = (
                (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                if frame.max() <= 1.0
                else frame.astype(np.uint8)
            )
        if _HAS_CV2:
            return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            return (
                frame[:, :, 0] * 0.299 + frame[:, :, 1] * 0.587 + frame[:, :, 2] * 0.114
            ).astype(np.uint8)
    if frame.dtype != np.uint8:
        return (
            (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            if frame.max() <= 1.0
            else frame.astype(np.uint8)
        )
    return frame


def compute_flow(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    config: FlowDepthConfig,
) -> np.ndarray:
    """Compute dense optical flow from frame_a to frame_b.

    Parameters
    ----------
    frame_a, frame_b : np.ndarray
        Consecutive video frames. Accepted formats: uint8 (H,W,3), uint8
        (H,W,4), float32 (H,W,3) [0,1], or greyscale (H,W).
    config : FlowDepthConfig

    Returns
    -------
    np.ndarray  float32 (H, W, 2)
        ``flow[y, x, 0]`` = horizontal motion in pixels.
        ``flow[y, x, 1]`` = vertical motion in pixels.

    Raises
    ------
    ImportError
        If OpenCV is not available and fallback_gradient is False.
    """
    if not _HAS_CV2:
        if config.fallback_gradient:
            H, W = frame_a.shape[:2]
            return np.zeros((H, W, 2), dtype=np.float32)
        raise ImportError(
            "OpenCV (cv2) is required for optical flow computation. "
            "Install it with: pip install opencv-python"
        )

    ga = _to_grey_u8(frame_a)
    gb = _to_grey_u8(frame_b)

    flow = cv2.calcOpticalFlowFarneback(
        ga,
        gb,
        None,
        config.pyr_scale,
        config.levels,
        config.winsize,
        config.iterations,
        config.poly_n,
        config.poly_sigma,
        0,  # flags
    )
    return flow.astype(np.float32)


def flow_to_depth(
    flow: np.ndarray,
    config: FlowDepthConfig,
) -> np.ndarray:
    """Convert a dense flow field to a normalised depth proxy.

    Parameters
    ----------
    flow : np.ndarray
        float32 (H, W, 2) flow field from ``compute_flow``.
    config : FlowDepthConfig

    Returns
    -------
    np.ndarray  float32 (H, W) in [0, 1]
        1.0 = near (large motion), 0.0 = far (small/no motion).
        Inverted if ``config.invert == True``.
    """
    # Motion magnitude
    mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2).astype(np.float32)

    # Min-motion threshold — suppress background noise
    mag = np.where(mag < config.min_motion, 0.0, mag)

    # Scale
    mag = mag * config.motion_scale

    # Blur to smooth per-pixel noise
    if config.blur_sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter

            mag = gaussian_filter(mag, sigma=config.blur_sigma).astype(np.float32)
        except ImportError:
            if _HAS_CV2:
                ks = max(3, int(config.blur_sigma * 3) | 1)
                mag = cv2.GaussianBlur(mag, (ks, ks), config.blur_sigma)

    # Ceiling: user-specified or 95th percentile
    ceil = config.max_motion if config.max_motion > 0 else float(np.percentile(mag, 95))
    if ceil > 0:
        mag = np.clip(mag, 0.0, ceil)

    # Normalise to [0, 1]
    max_val = mag.max()
    if max_val > 0:
        depth = mag / max_val
    else:
        depth = np.zeros_like(mag)

    if config.invert:
        depth = 1.0 - depth

    return depth.astype(np.float32)


def flow_warp(
    frame: np.ndarray,
    flow: np.ndarray,
) -> np.ndarray:
    """Warp a frame forward by a flow field.

    Useful for depth-guided frame prediction and temporal blending.

    Parameters
    ----------
    frame : np.ndarray
        uint8 (H, W, C) source frame.
    flow : np.ndarray
        float32 (H, W, 2) flow field.

    Returns
    -------
    np.ndarray  uint8 (H, W, C)
    """
    H, W = flow.shape[:2]
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    map_x = (xs + flow[:, :, 0]).astype(np.float32)
    map_y = (ys + flow[:, :, 1]).astype(np.float32)

    if _HAS_CV2:
        return cv2.remap(
            frame.astype(np.uint8), map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )

    # Pure-numpy fallback: nearest-neighbour
    map_xi = np.clip(np.round(map_x).astype(np.int32), 0, W - 1)
    map_yi = np.clip(np.round(map_y).astype(np.int32), 0, H - 1)
    return frame[map_yi, map_xi]


# ---------------------------------------------------------------------------
# FlowDepthEstimator
# ---------------------------------------------------------------------------


class FlowDepthEstimator:
    """Stateful estimator that tracks the previous frame for streaming use.

    Example
    -------
    ::

        est = FlowDepthEstimator()
        for frame in video_frames:
            depth = est.feed(frame)   # None on first frame
            if depth is not None:
                stereo = synthesize(depth, pattern, params)
    """

    def __init__(self, config: Optional[FlowDepthConfig] = None):
        self.config = config or FlowDepthConfig()
        self._prev_grey: Optional[np.ndarray] = None
        self._prev_depth: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Clear the previous-frame state."""
        self._prev_grey = None
        self._prev_depth = None

    def feed(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Feed one frame. Returns depth proxy or None if first frame.

        Parameters
        ----------
        frame : np.ndarray
            Current video frame (H, W, C) uint8 or float32.

        Returns
        -------
        np.ndarray float32 (H, W) or None
        """
        grey = _to_grey_u8(frame)
        if self._prev_grey is None:
            self._prev_grey = grey
            return None

        flow = compute_flow(self._prev_grey, grey, self.config)
        depth = flow_to_depth(flow, self.config)

        self._prev_grey = grey
        self._prev_depth = depth
        return depth

    def estimate_from_pair(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
    ) -> np.ndarray:
        """Estimate depth from a single frame pair without state.

        Parameters
        ----------
        frame_a, frame_b : np.ndarray
            Consecutive frames.

        Returns
        -------
        np.ndarray  float32 (H, W) depth proxy.
        """
        flow = compute_flow(frame_a, frame_b, self.config)
        return flow_to_depth(flow, self.config)

    def estimate_sequence(
        self,
        frames: List[np.ndarray],
        pad_first: bool = True,
    ) -> List[np.ndarray]:
        """Estimate depth for every frame in a sequence.

        Parameters
        ----------
        frames : list of np.ndarray
            Ordered list of video frames.
        pad_first : bool
            If True, the first depth map is copied from frame 1→2 to fill
            the gap caused by the two-frame minimum. If False, the returned
            list is one element shorter than ``frames``.

        Returns
        -------
        list of np.ndarray  float32 (H, W), same length as ``frames`` if
        ``pad_first=True``, else ``len(frames) - 1``.
        """
        self.reset()
        depths = []

        for frame in frames:
            d = self.feed(frame)
            if d is not None:
                depths.append(d)

        if pad_first and depths:
            depths.insert(0, depths[0].copy())

        return depths


# ---------------------------------------------------------------------------
# Scene-cut detector (used by temporal.py)
# ---------------------------------------------------------------------------


def detect_scene_cut(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    threshold: float = 0.35,
) -> bool:
    """Detect a scene cut between two frames using mean absolute difference.

    Parameters
    ----------
    frame_a, frame_b : np.ndarray
        Consecutive frames.
    threshold : float
        MAD fraction [0, 1] above which a cut is declared.
        0.35 = 35% of max value difference.

    Returns
    -------
    bool  True if a scene cut is detected.
    """
    a = _to_grey_u8(frame_a).astype(np.float32) / 255.0
    b = _to_grey_u8(frame_b).astype(np.float32) / 255.0
    mad = float(np.mean(np.abs(a - b)))
    return mad > threshold
