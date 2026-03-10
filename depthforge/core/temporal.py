"""
depthforge.core.temporal
========================
Temporal coherence management for video stereogram synthesis.

Without temporal coherence, depth values can flicker frame-to-frame even
when the scene is static — both AI depth models and optical flow produce
noisy per-frame estimates. This module provides several strategies to
smooth depth over time:

Strategies
----------
ema         Exponential Moving Average — fast, low memory, slight lag.
windowed    Sliding window mean — more stable, higher memory usage.
flow_guided Flow-warped blending — warp previous frame forward then blend
            with current estimate. Best quality; requires OpenCV.
adaptive    Automatically selects between ema and flow_guided based on
            motion magnitude and scene-cut detection.

API
---
::

    from depthforge.core.temporal import TemporalSmoother, TemporalConfig

    smoother = TemporalSmoother(TemporalConfig(strategy="ema", alpha=0.3))

    for frame, raw_depth in video:
        smooth_depth = smoother.update(raw_depth, frame=frame)
        stereogram   = synthesize(smooth_depth, pattern, params)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TemporalConfig:
    """Configuration for temporal depth smoothing.

    Attributes
    ----------
    strategy : str
        Smoothing algorithm: ``"ema"``, ``"windowed"``, ``"flow_guided"``,
        ``"adaptive"``.
    alpha : float
        EMA blend factor in [0, 1]. Higher = more weight on current frame
        (less smoothing). Lower = heavier temporal averaging.
        Typical: 0.2–0.4 for video at 24fps.
    window_size : int
        Number of frames to average in windowed mode. Ignored for ema.
    flow_blend : float
        In flow_guided mode, blend weight for the warped previous frame.
        0.0 = use current depth only; 1.0 = use warped previous only.
    scene_cut_threshold : float
        Mean absolute difference fraction above which a scene cut is
        declared and state is reset. 0.35 is a sensible default.
    scene_cut_reset : bool
        If True, reset the temporal buffer on scene cut detection.
    min_motion_for_flow : float
        In adaptive mode, use flow_guided only when mean optical flow
        magnitude exceeds this threshold (pixels/frame). Otherwise use ema.
    depth_change_threshold : float
        If mean depth change between frames exceeds this value, reduce
        temporal smoothing strength to track fast depth changes.
    """

    strategy:               str     = "ema"
    alpha:                  float   = 0.3
    window_size:            int     = 5
    flow_blend:             float   = 0.3
    scene_cut_threshold:    float   = 0.35
    scene_cut_reset:        bool    = True
    min_motion_for_flow:    float   = 1.0
    depth_change_threshold: float   = 0.15

    def __post_init__(self):
        valid = {"ema", "windowed", "flow_guided", "adaptive"}
        if self.strategy not in valid:
            raise ValueError(f"strategy must be one of {valid}, got {self.strategy!r}")
        self.alpha      = float(np.clip(self.alpha, 0.01, 1.0))
        self.flow_blend = float(np.clip(self.flow_blend, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Per-strategy implementations
# ---------------------------------------------------------------------------

class _EMABuffer:
    """Exponential moving average depth buffer."""

    def __init__(self, alpha: float):
        self.alpha   = alpha
        self._smooth: Optional[np.ndarray] = None

    def reset(self):
        self._smooth = None

    def update(self, depth: np.ndarray) -> np.ndarray:
        if self._smooth is None:
            self._smooth = depth.copy()
            return self._smooth.copy()
        self._smooth = (1.0 - self.alpha) * self._smooth + self.alpha * depth
        return self._smooth.copy()

    def has_state(self) -> bool:
        return self._smooth is not None


class _WindowedBuffer:
    """Sliding-window mean depth buffer."""

    def __init__(self, window_size: int):
        self.window_size = max(1, window_size)
        self._buf: Deque[np.ndarray] = deque(maxlen=self.window_size)

    def reset(self):
        self._buf.clear()

    def update(self, depth: np.ndarray) -> np.ndarray:
        self._buf.append(depth.astype(np.float32))
        return np.mean(np.stack(list(self._buf), axis=0), axis=0).astype(np.float32)

    def has_state(self) -> bool:
        return len(self._buf) > 0


class _FlowGuidedBuffer:
    """Flow-guided temporal blending."""

    def __init__(self, blend: float, flow_config=None):
        self.blend        = blend
        self._prev_depth: Optional[np.ndarray] = None
        self._prev_frame: Optional[np.ndarray] = None
        self._flow_cfg    = flow_config

    def reset(self):
        self._prev_depth = None
        self._prev_frame = None

    def update(self, depth: np.ndarray, frame: Optional[np.ndarray] = None) -> np.ndarray:
        if self._prev_depth is None or frame is None or self._prev_frame is None:
            self._prev_depth = depth.copy()
            self._prev_frame = frame
            return depth.copy()

        # Compute flow and warp previous depth to current position
        try:
            from depthforge.core.optical_flow import compute_flow, flow_warp, FlowDepthConfig
            cfg  = self._flow_cfg or FlowDepthConfig()
            flow = compute_flow(self._prev_frame, frame, cfg)

            # Warp previous depth map using the flow
            prev_d_rgba = np.stack([
                (self._prev_depth * 255).astype(np.uint8),
                np.zeros_like(self._prev_depth, dtype=np.uint8),
                np.zeros_like(self._prev_depth, dtype=np.uint8),
                np.full(self._prev_depth.shape, 255, dtype=np.uint8),
            ], axis=-1)
            warped_rgba  = flow_warp(prev_d_rgba, flow)
            warped_depth = warped_rgba[:, :, 0].astype(np.float32) / 255.0

            # Blend: current depth (1-blend) + warped previous (blend)
            blended = (1.0 - self.blend) * depth + self.blend * warped_depth
        except Exception:
            # Fallback to simple EMA if flow fails
            if self._prev_depth is not None:
                blended = (1.0 - self.blend) * depth + self.blend * self._prev_depth
            else:
                blended = depth.copy()

        self._prev_depth = blended.copy()
        self._prev_frame = frame
        return blended.astype(np.float32)

    def has_state(self) -> bool:
        return self._prev_depth is not None


# ---------------------------------------------------------------------------
# TemporalSmoother
# ---------------------------------------------------------------------------

class TemporalSmoother:
    """Unified temporal depth smoother.

    Wraps one of four strategies behind a single ``update()`` call.
    Detects scene cuts and resets state automatically.

    Parameters
    ----------
    config : TemporalConfig
    """

    def __init__(self, config: Optional[TemporalConfig] = None):
        self.config       = config or TemporalConfig()
        self._frame_count = 0
        self._prev_frame: Optional[np.ndarray] = None
        self._ema         = _EMABuffer(self.config.alpha)
        self._windowed    = _WindowedBuffer(self.config.window_size)
        self._flow        = _FlowGuidedBuffer(self.config.flow_blend)

    def reset(self) -> None:
        """Reset all internal state (call on scene cut or at stream start)."""
        self._frame_count = 0
        self._prev_frame  = None
        self._ema.reset()
        self._windowed.reset()
        self._flow.reset()

    def update(
        self,
        depth: np.ndarray,
        frame: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply temporal smoothing to one depth frame.

        Parameters
        ----------
        depth : np.ndarray
            float32 (H, W) raw depth estimate for the current frame.
        frame : np.ndarray, optional
            The current video frame (H, W, C). Required for flow_guided
            and adaptive strategies.

        Returns
        -------
        np.ndarray  float32 (H, W) temporally smoothed depth.
        """
        from depthforge.core.optical_flow import detect_scene_cut

        # ── Scene-cut detection ──────────────────────────────────────────
        if (self.config.scene_cut_reset and
                self._prev_frame is not None and
                frame is not None):
            if detect_scene_cut(self._prev_frame, frame,
                                 threshold=self.config.scene_cut_threshold):
                self.reset()

        self._frame_count += 1
        strategy = self.config.strategy

        # ── Adaptive: choose strategy based on motion ────────────────────
        if strategy == "adaptive":
            strategy = self._adaptive_strategy(depth, frame)

        # ── Dispatch ─────────────────────────────────────────────────────
        if strategy == "ema":
            result = self._ema.update(depth)
        elif strategy == "windowed":
            result = self._windowed.update(depth)
        elif strategy == "flow_guided":
            result = self._flow.update(depth, frame)
        else:
            result = depth.copy()

        self._prev_frame = frame
        return result

    def _adaptive_strategy(
        self,
        depth: np.ndarray,
        frame: Optional[np.ndarray],
    ) -> str:
        """Pick ema or flow_guided based on depth change magnitude."""
        if not self._ema.has_state():
            return "ema"

        # Check how much depth changed
        if self._ema._smooth is not None:
            change = float(np.mean(np.abs(depth - self._ema._smooth)))
            if change > self.config.depth_change_threshold:
                # Fast depth change — use EMA with higher alpha for tracking
                self._ema.alpha = min(0.7, self.config.alpha * 2)
                return "ema"
            else:
                self._ema.alpha = self.config.alpha

        if frame is not None and self._prev_frame is not None:
            return "flow_guided"
        return "ema"

    @property
    def frame_count(self) -> int:
        """Number of frames processed since last reset."""
        return self._frame_count


# ---------------------------------------------------------------------------
# Depth history buffer for look-ahead smoothing
# ---------------------------------------------------------------------------

class DepthHistory:
    """Fixed-size sliding window of past depth frames.

    Useful for offline (non-realtime) processing where future frames are
    available and you can apply centred (non-causal) temporal filtering.

    Parameters
    ----------
    capacity : int
        Maximum number of frames to retain.

    Example
    -------
    ::

        hist = DepthHistory(capacity=7)
        for depth in depths:
            hist.push(depth)
            if hist.full:
                smoothed = hist.gaussian_smooth(sigma=1.5)
    """

    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self._buf: Deque[np.ndarray] = deque(maxlen=capacity)

    def push(self, depth: np.ndarray) -> None:
        self._buf.append(depth.astype(np.float32))

    @property
    def full(self) -> bool:
        return len(self._buf) == self.capacity

    @property
    def frames(self) -> List[np.ndarray]:
        return list(self._buf)

    def mean(self) -> Optional[np.ndarray]:
        if not self._buf:
            return None
        return np.mean(np.stack(list(self._buf), axis=0), axis=0).astype(np.float32)

    def gaussian_smooth(self, sigma: float = 1.5) -> Optional[np.ndarray]:
        """Apply 1D Gaussian weights along the temporal axis."""
        if not self._buf:
            return None
        frames = np.stack(list(self._buf), axis=0).astype(np.float32)  # (T, H, W)
        T      = frames.shape[0]

        # Gaussian weights centred on the middle frame
        centre  = (T - 1) / 2.0
        weights = np.exp(-0.5 * ((np.arange(T) - centre) / sigma) ** 2)
        weights = weights / weights.sum()

        # Weighted sum along temporal axis
        result = np.einsum('t,thw->hw', weights, frames)
        return result.astype(np.float32)

    def clear(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)
