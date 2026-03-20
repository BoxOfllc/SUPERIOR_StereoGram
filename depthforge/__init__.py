"""
DepthForge — Professional Stereogram Engine
============================================
A production-grade stereogram synthesis library for VFX, motion graphics,
and creative pipelines.

Surfaces: CLI · OFX · Nuke · ComfyUI · Web Preview
"""

__version__ = "0.5.0"
__author__ = "DepthForge Contributors"

import importlib.util


def _has(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


HAS_CV2 = _has("cv2")
HAS_SCIPY = _has("scipy")
HAS_TORCH = _has("torch")
HAS_OCIO = _has("PyOpenColorIO")
HAS_OPENEXR = _has("OpenEXR")


def capability_report() -> str:
    """Return a human-readable string of available optional packages."""
    lines = []
    for name, flag in [
        ("OpenCV (cv2)", HAS_CV2),
        ("SciPy", HAS_SCIPY),
        ("PyTorch", HAS_TORCH),
        ("PyOpenColorIO", HAS_OCIO),
        ("OpenEXR", HAS_OPENEXR),
    ]:
        lines.append(f"  {name}: {'✓' if flag else '✗'}")
    return "DepthForge capabilities:\n" + "\n".join(lines)


# Flash / PSE safety
# Phase 1 core
from depthforge.core.comfort import (
    ComfortAnalyzer,
    ComfortAnalyzerParams,
    SafetyLimiter,
    SafetyLimiterParams,
    VergenceProfile,
    estimate_vergence_strain,
)
from depthforge.core.depth_prep import DepthPrepParams, FalloffCurve, RegionMask, prep_depth
from depthforge.core.flash_safety import (
    EPILEPSY_WARNING,
    EPILEPSY_WARNING_SHORT,
    EpilepticRisk,
    FlashSafetyConfig,
    FlashSafetyReport,
    check_frame_sequence,
    check_pattern_flash_risk,
    warn_if_unsafe,
)
from depthforge.core.pattern_gen import ColorMode, PatternParams, PatternType, generate_pattern

# Phase 2
from depthforge.core.presets import (
    Preset,
    get_preset,
    list_presets,
    load_preset_from_json,
    register_preset,
)
from depthforge.core.qc import (
    QCParams,
    comparison_grid,
    depth_band_preview,
    depth_histogram,
    parallax_heatmap,
    safe_zone_indicator,
    window_violation_overlay,
)
from depthforge.core.synthesizer import StereoParams, synthesize

# GPU synthesis (requires torch; graceful no-op if unavailable)
if HAS_TORCH:
    from depthforge.core.gpu_synthesizer import best_device, synthesize_gpu
else:
    def synthesize_gpu(depth, pattern, params=None, device="auto"):  # type: ignore[misc]
        """GPU synthesis unavailable — torch not installed. Using NumPy."""
        return synthesize(depth, pattern, params or StereoParams())

    def best_device() -> str:  # type: ignore[misc]
        return "cpu"
