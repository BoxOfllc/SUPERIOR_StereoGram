"""
DepthForge — Professional Stereogram Engine
============================================
A production-grade stereogram synthesis library for VFX, motion graphics,
and creative pipelines.

Surfaces: CLI · OFX · Nuke · ComfyUI · Web Preview
"""

__version__ = "0.5.0"
__author__  = "DepthForge Contributors"

import importlib.util

def _has(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None

HAS_CV2     = _has("cv2")
HAS_SCIPY   = _has("scipy")
HAS_TORCH   = _has("torch")
HAS_OCIO    = _has("PyOpenColorIO")
HAS_OPENEXR = _has("OpenEXR")


def capability_report() -> str:
    """Return a human-readable string of available optional packages."""
    lines = []
    for name, flag in [
        ("OpenCV (cv2)", HAS_CV2),
        ("SciPy",        HAS_SCIPY),
        ("PyTorch",      HAS_TORCH),
        ("PyOpenColorIO",HAS_OCIO),
        ("OpenEXR",      HAS_OPENEXR),
    ]:
        lines.append(f"  {name}: {'✓' if flag else '✗'}")
    return "DepthForge capabilities:\n" + "\n".join(lines)


# Phase 1 core
from depthforge.core.synthesizer import synthesize, StereoParams
from depthforge.core.depth_prep  import prep_depth, DepthPrepParams, FalloffCurve, RegionMask
from depthforge.core.pattern_gen import generate_pattern, PatternParams, PatternType, ColorMode

# Phase 2
from depthforge.core.presets import (
    get_preset, list_presets, register_preset, load_preset_from_json, Preset,
)
from depthforge.core.comfort import (
    ComfortAnalyzer, ComfortAnalyzerParams,
    SafetyLimiter, SafetyLimiterParams,
    VergenceProfile, estimate_vergence_strain,
)
from depthforge.core.qc import (
    parallax_heatmap, depth_band_preview, window_violation_overlay,
    safe_zone_indicator, depth_histogram, comparison_grid, QCParams,
)
