"""
depthforge.ofx.plugin_bridge
=============================
Python-side OFX plugin bridge.

In a full OFX deployment the C++ shell (plugin.cpp) embeds a Python
interpreter via pybind11 and calls into this module for every render
request. The C++ side handles all OpenFX API boilerplate (suite calls,
descriptor registration, image fetch/release), and delegates the actual
pixel work to Python.

This file also acts as the integration test harness: run it directly to
verify the bridge API works without any C++ host present.

Architecture
------------
C++ plugin.cpp
  ↓  pybind11 call  (frame params as dict)
depthforge.ofx.plugin_bridge
  ↓  converts param dict → dataclasses
depthforge.core.tiled_render.tile_synthesize
  ↓  multi-threaded SIRDS / texture synthesis
  ↑  RGBA uint8 numpy array
plugin_bridge
  ↑  returns flat bytes buffer to C++
plugin.cpp
  ↑  copies buffer to OFX output clip


Host API called from C++
------------------------
All functions accept / return only primitive Python types and bytes so that
pybind11 bindings stay trivial.

``describe()``              → dict   Plugin metadata for OFX descriptor.
``describe_in_context(ctx)``→ dict   Param descriptors for a given context.
``render(params_json)``     → bytes  RGBA uint8 flat buffer (H*W*4).
``get_preset_names()``      → list[str]
``get_pattern_names()``     → list[str]
``version()``               → str
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from depthforge import __version__
from depthforge.core.depth_prep import DepthPrepParams, prep_depth
from depthforge.core.export_profiles import get_profile, list_profiles
from depthforge.core.lut import apply_lut, get_builtin_lut, list_builtin_luts
from depthforge.core.pattern_cache import PatternCache
from depthforge.core.pattern_gen import PatternParams, PatternType
from depthforge.core.presets import get_preset, list_presets
from depthforge.core.synthesizer import StereoParams
from depthforge.core.tiled_render import tile_synthesize

# Module-level pattern cache shared across renders
_cache = PatternCache(max_tiles=32)


# ---------------------------------------------------------------------------
# Plugin metadata
# ---------------------------------------------------------------------------

PLUGIN_IDENTIFIER = "com.depthforge.StereogramSynthesizer"
PLUGIN_VERSION_MAJOR = 0
PLUGIN_VERSION_MINOR = 5


def version() -> str:
    return __version__


def describe() -> dict:
    """
    Return plugin-level metadata.

    Called once by the OFX host during plugin enumeration.
    """
    return {
        "identifier": PLUGIN_IDENTIFIER,
        "version_major": PLUGIN_VERSION_MAJOR,
        "version_minor": PLUGIN_VERSION_MINOR,
        "label": "DepthForge Stereogram",
        "short_label": "DepthForge",
        "long_label": "DepthForge — Professional Stereogram Synthesizer",
        "group_label": "DepthForge",
        "plugin_type": "Filter",
        "supported_contexts": ["filter", "general"],
        "supports_tiles": True,
        "supports_multithread": True,
        "temporal_access": False,
        "single_instance": False,
        "host_frame_threading": True,
    }


def describe_in_context(context: str) -> dict:
    """
    Return parameter descriptors for the given OFX context.

    The C++ side converts these dicts into OFX param descriptors and
    builds the UI. Each entry maps to an OFX ParamDescriptor.

    Parameters
    ----------
    context : str
        "filter" or "general"

    Returns
    -------
    dict with keys:
        "clips"  : list of clip descriptors
        "params" : list of param descriptors
    """
    clips = [
        {"name": "Source", "label": "Source", "optional": False},
        {"name": "Depth", "label": "Depth Map", "optional": False},
        {"name": "Pattern", "label": "Pattern", "optional": True},
        {"name": "Output", "label": "Output", "optional": False},
    ]

    params = [
        # ── Synthesis controls ───────────────────────────────────────────
        {
            "name": "preset_name",
            "type": "choice",
            "label": "Preset",
            "hint": "Named synthesis preset",
            "choices": ["none"] + list_presets(),
            "default": "none",
            "group": "Synthesis",
        },
        {
            "name": "depth_factor",
            "type": "double",
            "label": "Depth Factor",
            "hint": "Parallax strength (0–1)",
            "min": 0.0,
            "max": 1.0,
            "default": 0.35,
            "group": "Synthesis",
        },
        {
            "name": "max_parallax",
            "type": "double",
            "label": "Max Parallax",
            "hint": "Max parallax as fraction of frame width",
            "min": 0.005,
            "max": 0.1,
            "default": 0.033,
            "group": "Synthesis",
        },
        {
            "name": "oversample",
            "type": "int",
            "label": "Oversample",
            "hint": "Supersampling factor (1–4)",
            "min": 1,
            "max": 4,
            "default": 1,
            "group": "Synthesis",
        },
        {
            "name": "safe_mode",
            "type": "bool",
            "label": "Safe Mode",
            "hint": "Hard-clamp parallax to max_parallax",
            "default": False,
            "group": "Synthesis",
        },
        {
            "name": "invert_depth",
            "type": "bool",
            "label": "Invert Depth",
            "hint": "Flip near/far convention",
            "default": False,
            "group": "Synthesis",
        },
        {
            "name": "seed",
            "type": "int",
            "label": "Random Seed",
            "hint": "Pattern tiling seed",
            "min": 0,
            "max": 99999,
            "default": 42,
            "group": "Synthesis",
        },
        # ── Pattern controls ─────────────────────────────────────────────
        {
            "name": "pattern_source",
            "type": "choice",
            "label": "Pattern Source",
            "hint": "Where to get the pattern tile",
            "choices": ["clip", "library", "procedural"],
            "default": "procedural",
            "group": "Pattern",
        },
        {
            "name": "library_pattern",
            "type": "choice",
            "label": "Library Pattern",
            "choices": [
                "plasma_wave",
                "fine_grain",
                "linen",
                "marble",
                "perlin_noise",
                "water_ripples",
                "hexgrid",
                "neon_grid",
            ],
            "default": "plasma_wave",
            "group": "Pattern",
        },
        {
            "name": "pattern_type",
            "type": "choice",
            "label": "Procedural Type",
            "choices": [t.value for t in PatternType],
            "default": "random_noise",
            "group": "Pattern",
        },
        {
            "name": "tile_size",
            "type": "int",
            "label": "Tile Size",
            "hint": "Pattern tile size in pixels (square)",
            "min": 32,
            "max": 512,
            "default": 128,
            "group": "Pattern",
        },
        # ── Depth prep ───────────────────────────────────────────────────
        {
            "name": "bilateral_space",
            "type": "double",
            "label": "Smooth (Spatial)",
            "hint": "Bilateral filter spatial sigma",
            "min": 0.0,
            "max": 30.0,
            "default": 5.0,
            "group": "Depth Prep",
        },
        {
            "name": "dilation_px",
            "type": "int",
            "label": "Dilation (px)",
            "hint": "Morphological dilation radius",
            "min": 0,
            "max": 20,
            "default": 3,
            "group": "Depth Prep",
        },
        {
            "name": "near_plane",
            "type": "double",
            "label": "Near Plane",
            "hint": "Clip depth below this value",
            "min": 0.0,
            "max": 1.0,
            "default": 0.0,
            "group": "Depth Prep",
        },
        {
            "name": "far_plane",
            "type": "double",
            "label": "Far Plane",
            "hint": "Clip depth above this value",
            "min": 0.0,
            "max": 1.0,
            "default": 1.0,
            "group": "Depth Prep",
        },
        # ── LUT ──────────────────────────────────────────────────────────
        {
            "name": "lut_name",
            "type": "choice",
            "label": "Pattern LUT",
            "choices": ["none"] + list_builtin_luts(),
            "default": "none",
            "group": "Colour Grade",
        },
        {
            "name": "lut_strength",
            "type": "double",
            "label": "LUT Strength",
            "hint": "Blend between original and LUT (0–1)",
            "min": 0.0,
            "max": 1.0,
            "default": 1.0,
            "group": "Colour Grade",
        },
        # ── Export profile ───────────────────────────────────────────────
        {
            "name": "export_profile",
            "type": "choice",
            "label": "Export Profile",
            "choices": ["none"] + list_profiles(),
            "default": "none",
            "group": "Output",
        },
    ]

    return {"clips": clips, "params": params}


def get_preset_names() -> list:
    return ["none"] + list_presets()


def get_pattern_names() -> list:
    from depthforge.core.pattern_library import list_categories, list_patterns

    names = []
    for cat in list_categories():
        names.extend(list_patterns(cat))
    return names


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def render(params_json: str) -> bytes:
    """
    Main render entry point called by the C++ host.

    Parameters
    ----------
    params_json : str
        JSON-encoded dict of all param values plus image data. Must contain:
            "depth_bytes"   : base64-encoded or raw float32 bytes (H×W)
            "depth_width"   : int
            "depth_height"  : int
            "pattern_bytes" : base64-encoded RGBA uint8 bytes (optional)
            "pattern_width" : int (optional)
            "pattern_height": int (optional)
            + all param values from describe_in_context()

    Returns
    -------
    bytes
        Flat RGBA uint8 buffer of length H×W×4.
    """
    params: dict[str, Any] = json.loads(params_json)

    W = int(params["depth_width"])
    H = int(params["depth_height"])

    # ── Decode depth ──────────────────────────────────────────────────────
    import base64

    raw = base64.b64decode(params["depth_bytes"])
    depth = np.frombuffer(raw, dtype=np.float32).reshape(H, W)

    # ── Depth prep ────────────────────────────────────────────────────────
    dp = DepthPrepParams(
        invert=bool(params.get("invert_depth", False)),
        bilateral_space=float(params.get("bilateral_space", 5.0)),
        dilation_px=int(params.get("dilation_px", 3)),
        near_plane=float(params.get("near_plane", 0.0)),
        far_plane=float(params.get("far_plane", 1.0)),
    )
    depth = prep_depth(depth, dp)

    # ── Pattern ──────────────────────────────────────────────────────────
    pattern_source = params.get("pattern_source", "procedural")
    lut_name = params.get("lut_name", "none")
    lut_strength = float(params.get("lut_strength", 1.0))

    if pattern_source == "clip" and params.get("pattern_bytes"):
        pw = int(params.get("pattern_width", 128))
        ph = int(params.get("pattern_height", 128))
        raw_p = base64.b64decode(params["pattern_bytes"])
        pattern = np.frombuffer(raw_p, dtype=np.uint8).reshape(ph, pw, 4)
    elif pattern_source == "library":
        lib_name = params.get("library_pattern", "plasma_wave")
        tile_sz = int(params.get("tile_size", 128))
        pattern = _cache.get_library(lib_name, tile_sz, tile_sz, seed=int(params.get("seed", 42)))
    else:
        tile_sz = int(params.get("tile_size", 128))
        pp = PatternParams(
            pattern_type=PatternType(params.get("pattern_type", "random_noise")),
            tile_width=tile_sz,
            tile_height=tile_sz,
            seed=int(params.get("seed", 42)),
        )
        pattern = _cache.get(pp)

    # Apply LUT to pattern
    if lut_name and lut_name != "none":
        try:
            lut = get_builtin_lut(lut_name)
            pattern = apply_lut(pattern, lut, strength=lut_strength)
        except KeyError:
            pass  # unknown LUT → skip

    # ── Stereo params ─────────────────────────────────────────────────────
    preset_name = params.get("preset_name", "none")
    if preset_name and preset_name != "none":
        try:
            preset = get_preset(preset_name)
            sp = preset.stereo_params
        except KeyError:
            sp = _default_stereo_params(params)
    else:
        sp = _default_stereo_params(params)

    # ── Synthesize ────────────────────────────────────────────────────────
    result = tile_synthesize(depth, pattern, sp)

    # ── Export profile ────────────────────────────────────────────────────
    export_name = params.get("export_profile", "none")
    if export_name and export_name != "none":
        try:
            from depthforge.core.export_profiles import apply_profile

            ep = get_profile(export_name)
            result = apply_profile(result, ep)
        except (KeyError, Exception):
            pass

    return result.tobytes()


def _default_stereo_params(params: dict) -> StereoParams:
    return StereoParams(
        depth_factor=float(params.get("depth_factor", 0.35)),
        max_parallax_fraction=float(params.get("max_parallax", 0.033)),
        oversample=int(params.get("oversample", 1)),
        safe_mode=bool(params.get("safe_mode", False)),
        invert_depth=bool(params.get("invert_depth", False)),
        seed=int(params.get("seed", 42)),
    )


# ---------------------------------------------------------------------------
# Smoke test (run this file directly to verify the bridge)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import base64

    print("DepthForge OFX Python Bridge — smoke test")
    print(f"  version: {version()}")
    print(f"  plugin id: {PLUGIN_IDENTIFIER}")

    meta = describe()
    print(f"  label: {meta['label']}")

    ctx = describe_in_context("filter")
    print(f"  clips: {[c['name'] for c in ctx['clips']]}")
    print(f"  params: {len(ctx['params'])} descriptors")

    # Render a tiny 64×32 test frame
    H, W = 32, 64
    rng = np.random.default_rng(42)
    depth = rng.random((H, W), dtype=np.float32)
    payload = {
        "depth_bytes": base64.b64encode(depth.tobytes()).decode(),
        "depth_width": W,
        "depth_height": H,
        "pattern_source": "procedural",
        "pattern_type": "random_noise",
        "tile_size": 64,
        "depth_factor": 0.35,
        "seed": 42,
    }
    result_bytes = render(json.dumps(payload))
    result = np.frombuffer(result_bytes, dtype=np.uint8).reshape(H, W, 4)
    print(
        f"  render result: {result.shape} dtype={result.dtype} "
        f"min={result.min()} max={result.max()}"
    )
    print("  PASS")
