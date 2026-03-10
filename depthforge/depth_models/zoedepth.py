"""
depthforge.depth_models.zoedepth
=================================
ZoeDepth metric monocular depth estimator.

Requires: torch, torchvision, timm
Install:  pip install "depthforge[ai]"

ZoeDepth produces metric (absolute) depth in metres, unlike MiDaS which
produces relative depth. The output is normalised to [0, 1] before return.

Reference: https://github.com/isl-org/ZoeDepth
"""

from __future__ import annotations

import numpy as np

from depthforge.depth_models import DepthEstimator, DepthEstimationError, _REGISTRY


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

try:
    import torch
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


# ---------------------------------------------------------------------------
# ZoeDepth estimator
# ---------------------------------------------------------------------------

class ZoeDepthEstimator(DepthEstimator):
    """ZoeDepth N (indoor/outdoor) metric depth estimator.

    Produces metric depth in metres then normalises to [0, 1].
    Uses torch.hub to download weights on first run (~700MB).

    For mixed indoor/outdoor scenes, ZoeDepth_N is used (NK model
    handles better scene-type transitions).
    """

    _HUB_REPO   = "isl-org/ZoeDepth"
    _MODEL_NAME = "ZoeD_N"

    def __init__(self):
        self._torch_model = None
        self._device      = None
        super().__init__()

    @property
    def name(self) -> str:
        return "zoedepth"

    def _load_model(self) -> None:
        if not _TORCH_OK:
            raise DepthEstimationError(
                "ZoeDepth requires PyTorch. Install it with:\n"
                "  pip install \"depthforge[ai]\"\n"
                "or: pip install torch torchvision timm"
            )

        import torch

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ZoeDepth requires torch.hub with trust_repo
        self._torch_model = torch.hub.load(
            self._HUB_REPO,
            self._MODEL_NAME,
            pretrained=True,
            trust_repo=True,
        )
        self._torch_model.to(self._device)
        self._torch_model.eval()

    def _infer(self, rgb: np.ndarray) -> np.ndarray:
        import torch
        from PIL import Image as _PILImage

        # ZoeDepth accepts PIL Image directly
        pil = _PILImage.fromarray(rgb)
        with torch.no_grad():
            depth_tensor = self._torch_model.infer_pil(pil)

        depth = depth_tensor.squeeze().cpu().numpy().astype(np.float32)

        # Invert — ZoeDepth produces metres where large = far
        # We want large = near
        if depth.max() > 0:
            depth = 1.0 / (depth + 1e-3)

        return depth


# ---------------------------------------------------------------------------
# Mock for testing
# ---------------------------------------------------------------------------

class _MockZoeDepthEstimator(DepthEstimator):
    """Synthetic estimator for testing without PyTorch.

    Generates a radial gradient depth map (sphere in the centre = near).
    """

    @property
    def name(self) -> str:
        return "zoedepth_mock"

    def _load_model(self) -> None:
        pass

    def _infer(self, rgb: np.ndarray) -> np.ndarray:
        H, W = rgb.shape[:2]
        y, x = np.mgrid[0:H, 0:W]
        cx, cy = W / 2, H / 2
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        depth = np.clip(1.0 - r / (min(W, H) * 0.5), 0.0, 1.0)
        return depth.astype(np.float32)


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------

_REGISTRY["zoedepth"]      = ZoeDepthEstimator
_REGISTRY["zoedepth_mock"] = _MockZoeDepthEstimator
