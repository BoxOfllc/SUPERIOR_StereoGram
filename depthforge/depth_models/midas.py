"""
depthforge.depth_models.midas
==============================
MiDaS v3.1 monocular depth estimator.

Requires: torch, torchvision, timm
Install:  pip install "depthforge[ai]"

If PyTorch is not available this module imports cleanly but raises
``DepthEstimationError`` at instantiation time with a clear message.
"""

from __future__ import annotations

import numpy as np

from depthforge.depth_models import DepthEstimator, DepthEstimationError, _REGISTRY


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

try:
    import torch
    import torchvision
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

try:
    import timm  # noqa — needed by MiDaS DPT
    _TIMM_OK = True
except ImportError:
    _TIMM_OK = False


# ---------------------------------------------------------------------------
# MiDaS estimator
# ---------------------------------------------------------------------------

class MiDaSEstimator(DepthEstimator):
    """MiDaS v3.1 DPT-Large monocular depth estimator.

    On first call to ``estimate()``, downloads and caches the model weights
    using ``torch.hub`` (requires internet on first run). Subsequent calls
    use the local cache (~400MB).

    The model produces *relative* (not metric) depth. Output is normalised
    to [0, 1] with 1.0 = near.
    """

    _MODEL_NAME = "DPT_Large"
    _HUB_REPO   = "intel-isl/MiDaS"
    _INPUT_SIZE = 384

    def __init__(self):
        self._torch_model = None
        self._transform   = None
        self._device      = None
        super().__init__()

    @property
    def name(self) -> str:
        return "midas"

    def _load_model(self) -> None:
        if not _TORCH_OK:
            raise DepthEstimationError(
                "MiDaS requires PyTorch. Install it with:\n"
                "  pip install \"depthforge[ai]\"\n"
                "or: pip install torch torchvision timm"
            )
        if not _TIMM_OK:
            raise DepthEstimationError(
                "MiDaS requires timm. Install it with:\n"
                "  pip install timm"
            )

        import torch

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model via torch.hub
        self._torch_model = torch.hub.load(
            self._HUB_REPO,
            self._MODEL_NAME,
            pretrained=True,
        )
        self._torch_model.to(self._device)
        self._torch_model.eval()

        # Load transforms
        midas_transforms = torch.hub.load(self._HUB_REPO, "transforms")
        self._transform = midas_transforms.dpt_transform

    def _infer(self, rgb: np.ndarray) -> np.ndarray:
        import torch

        # Apply MiDaS preprocessing transform
        input_batch = self._transform(rgb).to(self._device)

        with torch.no_grad():
            prediction = self._torch_model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy().astype(np.float32)

        # MiDaS outputs inverse depth (larger = closer) — invert to standard
        # convention where larger = farther
        if depth.max() > 0:
            depth = 1.0 / (depth + 1e-6)

        return depth


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------

_REGISTRY["midas"] = MiDaSEstimator


# ---------------------------------------------------------------------------
# Mock estimator for testing (activated when PyTorch unavailable)
# ---------------------------------------------------------------------------

class _MockMiDaSEstimator(DepthEstimator):
    """Synthetic depth estimator for testing without PyTorch.

    Generates a plausible depth map from image luminance.
    Not suitable for production use.
    """

    @property
    def name(self) -> str:
        return "midas_mock"

    def _load_model(self) -> None:
        pass   # no model to load

    def _infer(self, rgb: np.ndarray) -> np.ndarray:
        """Generate depth from luminance (bright = near)."""
        lum = rgb.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722])

        # Apply Gaussian blur to create smooth depth
        try:
            import cv2
            depth = cv2.GaussianBlur(lum / 255.0, (31, 31), 8.0)
        except ImportError:
            # NumPy box blur
            from numpy.lib.stride_tricks import sliding_window_view
            lum_n = lum / 255.0
            pad = 8
            padded = np.pad(lum_n, pad, mode="edge")
            windows = sliding_window_view(padded, (pad * 2 + 1, pad * 2 + 1))
            depth = windows.mean(axis=(-2, -1)).astype(np.float32)

        return depth


_REGISTRY["midas_mock"] = _MockMiDaSEstimator
