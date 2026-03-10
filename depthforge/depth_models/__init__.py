"""
depthforge.depth_models
========================
AI depth estimation models for Phase 3.

Usage
-----
    from depthforge.depth_models import get_depth_estimator

    estimator = get_depth_estimator("midas")
    depth = estimator.estimate(rgb_image)   # float32 (H, W) in [0, 1]

Models
------
    midas       MiDaS v3.1 (DPT-Large) — best general-purpose quality
    zoedepth    ZoeDepth — metric depth with scale awareness

Both models require PyTorch. If PyTorch is unavailable, a
``DepthEstimationError`` is raised with a clear install message.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING

import numpy as np


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class DepthEstimationError(RuntimeError):
    """Raised when depth estimation is unavailable or fails."""
    pass


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class DepthEstimator(ABC):
    """Abstract base class for all depth estimators.

    Subclasses must implement:
    - ``name`` property
    - ``_load_model()``
    - ``_infer(rgb) -> np.ndarray``
    """

    _model_loaded: bool = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier string (e.g. 'midas')."""
        ...

    def estimate(self, rgb: np.ndarray) -> np.ndarray:
        """Estimate depth from an RGB image.

        Parameters
        ----------
        rgb : np.ndarray
            uint8 RGB image, shape (H, W, 3).

        Returns
        -------
        np.ndarray
            float32 depth map, shape (H, W), values in [0.0, 1.0].
            1.0 = near (closest to camera), 0.0 = far (farthest).
        """
        if not self._model_loaded:
            self._load_model()
            self._model_loaded = True

        if rgb.ndim == 2:
            rgb = np.stack([rgb, rgb, rgb], axis=-1)
        elif rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]

        depth = self._infer(rgb)

        # Normalise to [0, 1]
        lo, hi = depth.min(), depth.max()
        if hi > lo:
            depth = (depth - lo) / (hi - lo)
        else:
            depth = np.zeros_like(depth)

        return depth.astype(np.float32)

    @abstractmethod
    def _load_model(self) -> None:
        """Load model weights. Called once on first ``estimate()`` call."""
        ...

    @abstractmethod
    def _infer(self, rgb: np.ndarray) -> np.ndarray:
        """Run inference. Input is (H,W,3) uint8. Return float32 (H,W)."""
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type] = {}


def register_estimator(cls: type) -> type:
    """Decorator to register a DepthEstimator subclass."""
    _REGISTRY[cls.name.fget(None) if hasattr(cls.name, "fget") else cls().name] = cls
    return cls


def get_depth_estimator(name: str) -> DepthEstimator:
    """Instantiate a depth estimator by name.

    Parameters
    ----------
    name : str
        One of: "midas", "zoedepth"

    Raises
    ------
    DepthEstimationError
        If the model is unknown or required packages are missing.
    """
    name = name.lower().strip()

    # Import all model modules to trigger full registration (including mocks)
    try:
        from depthforge.depth_models import midas as _m    # noqa
    except Exception:
        pass
    try:
        from depthforge.depth_models import zoedepth as _z  # noqa
    except Exception:
        pass

    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise DepthEstimationError(
            f"Unknown depth model {name!r}. Available: {available}"
        )
    return _REGISTRY[name]()


def list_estimators() -> list[str]:
    """Return list of registered estimator names (only those whose deps are available)."""
    # Trigger registration of both models
    try:
        from depthforge.depth_models import midas as _      # noqa
    except Exception:
        pass
    try:
        from depthforge.depth_models import zoedepth as _   # noqa
    except Exception:
        pass
    return list(_REGISTRY.keys())
