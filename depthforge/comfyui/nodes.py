"""
depthforge.comfyui.nodes
=========================
10 ComfyUI custom nodes for AI-powered stereogram workflows.

Install
-------
Copy the depthforge folder into ``ComfyUI/custom_nodes/`` and restart ComfyUI.
Or symlink: ``ln -s /path/to/depthforge ComfyUI/custom_nodes/depthforge``

Node catalog
------------
DF_DepthPrep        Condition a raw depth map (smooth, dilate, curve, clamp)
DF_PatternGen       Generate a procedural pattern tile (7 built-in generators)
DF_PatternLibrary   Access the 28-pattern named library
DF_Stereogram       Core SIRDS / texture stereogram synthesis
DF_AnaglyphOut      Red/cyan anaglyph output (5 modes)
DF_StereoPair       L/R stereo pair with occlusion mask output
DF_HiddenImage      Hidden shape/text stereogram
DF_SafetyLimiter    Clamp depth and stereo params to comfort limits
DF_QCOverlay        Parallax heatmap and violation overlay
DF_VideoSequence    Per-frame video stereogram (temporal coherence)

ComfyUI type conventions used
------------------------------
IMAGE           torch.Tensor (B, H, W, C) float32 [0,1]  or numpy passthrough
DEPTH_MAP       numpy float32 (H, W) [0,1]
STEREO_PARAMS   dict with synthesis parameters
DEPTH_PARAMS    dict with depth conditioning parameters
PATTERN         numpy uint8 (H, W, 4) RGBA
COMFORT_REPORT  depthforge.core.comfort.ComfortReport instance
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# ComfyUI tensor ↔ numpy conversion helpers
# ---------------------------------------------------------------------------


def _tensor_to_numpy(tensor) -> np.ndarray:
    """Convert ComfyUI IMAGE tensor (B,H,W,C float32) to numpy uint8 (H,W,4)."""
    try:
        import torch

        if isinstance(tensor, torch.Tensor):
            img = tensor.squeeze(0).cpu().numpy()
        else:
            img = np.array(tensor)
    except ImportError:
        img = np.array(tensor)

    # Squeeze leading batch dimension if present (B,H,W,C) -> (H,W,C)
    if img.ndim == 4 and img.shape[0] == 1:
        img = img[0]

    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

    if img.ndim == 2:
        img = np.stack([img, img, img, np.full_like(img, 255)], axis=-1)
    elif img.ndim == 3 and img.shape[2] == 3:
        alpha = np.full((*img.shape[:2], 1), 255, dtype=np.uint8)
        img = np.concatenate([img, alpha], axis=-1)

    return img


def _numpy_to_tensor(arr: np.ndarray):
    """Convert numpy RGBA uint8 (H,W,4) to ComfyUI IMAGE tensor (1,H,W,4)."""
    try:
        import torch

        t = torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0)
        return t
    except ImportError:
        return arr[None].astype(np.float32) / 255.0


def _depth_to_numpy(depth_input) -> np.ndarray:
    """Accept depth as tensor, numpy array, or list — return float32 (H,W)."""
    try:
        import torch

        if isinstance(depth_input, torch.Tensor):
            d = depth_input.squeeze().cpu().numpy()
        else:
            d = np.array(depth_input)
    except ImportError:
        d = np.array(depth_input)

    if d.ndim == 3:
        d = d.mean(axis=-1)
    elif d.ndim == 4:
        d = d[0].mean(axis=-1)

    return d.astype(np.float32)


def _numpy_depth_to_tensor(depth: np.ndarray):
    """Convert float32 depth (H,W) to ComfyUI IMAGE format (1,H,W,3)."""
    try:
        import torch

        d3 = np.stack([depth, depth, depth], axis=-1)
        return torch.from_numpy(d3).unsqueeze(0)
    except ImportError:
        d3 = np.stack([depth, depth, depth], axis=-1)
        return d3[None]


# ---------------------------------------------------------------------------
# Node base helpers
# ---------------------------------------------------------------------------


class _DFNodeBase:
    """Shared utilities for all DepthForge nodes."""

    CATEGORY = "DepthForge"
    FUNCTION = "execute"

    @staticmethod
    def _falloff_map():
        from depthforge.core.depth_prep import FalloffCurve

        return {
            "linear": FalloffCurve.LINEAR,
            "gamma": FalloffCurve.GAMMA,
            "s_curve": FalloffCurve.S_CURVE,
            "logarithmic": FalloffCurve.LOGARITHMIC,
            "exponential": FalloffCurve.EXPONENTIAL,
        }


# ===========================================================================
# Node 1 — DF_DepthPrep
# ===========================================================================


class DF_DepthPrep(_DFNodeBase):
    """Condition a raw depth map through the 7-stage DepthForge pipeline."""

    DESCRIPTION = (
        "Condition a raw depth map: smooth, dilate, apply a tone curve, and clamp "
        "near/far planes before feeding into any stereogram node."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth_image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Input depth map — white = near (close to camera), "
                            "black = far (background). Connect an AI depth estimator "
                            "or load a grayscale depth image."
                        )
                    },
                ),
            },
            "optional": {
                "invert": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Flip near/far — enable when your depth map uses the "
                            "black = near convention (e.g. some MiDaS outputs)."
                        ),
                    },
                ),
                "bilateral_space": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.0,
                        "max": 30.0,
                        "step": 0.5,
                        "tooltip": (
                            "Spatial smoothing radius in pixels for the bilateral filter. "
                            "Safe range: 3–15. Higher = smoother depth but slower. "
                            "0 = disabled."
                        ),
                    },
                ),
                "bilateral_color": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.01,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Edge-preserving strength of the bilateral filter. "
                            "Safe range: 0.05–0.3. Lower = sharper edge preservation. "
                            "Higher = more uniform smoothing across edges."
                        ),
                    },
                ),
                "dilation_px": (
                    "INT",
                    {
                        "default": 3,
                        "min": 0,
                        "max": 20,
                        "tooltip": (
                            "Expand foreground objects by this many pixels to reduce "
                            "stereo artifacts at depth edges. Safe range: 1–8. "
                            "0 = disabled."
                        ),
                    },
                ),
                "falloff_curve": (
                    ["linear", "gamma", "s_curve", "logarithmic", "exponential"],
                    {
                        "tooltip": (
                            "Tone curve applied to the depth values. "
                            "'linear' = neutral passthrough; "
                            "'gamma' = boost mid-depths; "
                            "'s_curve' = adds contrast in mid-tones; "
                            "'logarithmic' = compress highlights; "
                            "'exponential' = compress shadows."
                        )
                    },
                ),
                "near_plane": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Clip depth values below this threshold — useful to remove "
                            "noisy near-black background regions. "
                            "Typical: 0.0–0.1. Must be less than far_plane."
                        ),
                    },
                ),
                "far_plane": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Clip depth values above this threshold — useful to remove "
                            "blown-out near-white foreground noise. "
                            "Typical: 0.9–1.0. Must be greater than near_plane."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("DEPTH_MAP", "IMAGE")
    RETURN_NAMES = ("depth_map", "depth_preview")
    OUTPUT_NODE = False

    def execute(
        self,
        depth_image,
        invert=False,
        bilateral_space=5.0,
        bilateral_color=0.1,
        dilation_px=3,
        falloff_curve="linear",
        near_plane=0.0,
        far_plane=1.0,
    ):
        from depthforge.core.depth_prep import DepthPrepParams, prep_depth

        raw = _depth_to_numpy(depth_image)
        params = DepthPrepParams(
            invert=invert,
            bilateral_sigma_space=bilateral_space,
            bilateral_sigma_color=bilateral_color,
            dilation_px=dilation_px,
            falloff_curve=self._falloff_map()[falloff_curve],
            near_plane=near_plane,
            far_plane=far_plane,
        )
        conditioned = prep_depth(raw, params)
        preview = _numpy_depth_to_tensor(conditioned)
        return (conditioned, preview)


# ===========================================================================
# Node 2 — DF_PatternGen
# ===========================================================================


class DF_PatternGen(_DFNodeBase):
    """Generate a procedural pattern tile for SIRDS/texture stereograms."""

    DESCRIPTION = (
        "Generate a tileable procedural pattern tile using one of 7 built-in "
        "algorithms, ready to feed directly into DF_Stereogram or DF_HiddenImage."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pattern_type": (
                    [
                        "random_noise",
                        "perlin",
                        "plasma",
                        "voronoi",
                        "geometric_grid",
                        "mandelbrot",
                        "dot_matrix",
                    ],
                    {
                        "tooltip": (
                            "Algorithm used to generate the tile. "
                            "'random_noise' = fastest, classic SIRDS look; "
                            "'perlin'/'plasma' = organic, smooth textures; "
                            "'voronoi' = cellular/crystalline; "
                            "'mandelbrot' = fractal detail; "
                            "'dot_matrix' = halftone-style dots."
                        )
                    },
                ),
                "color_mode": (
                    ["greyscale", "monochrome", "psychedelic"],
                    {
                        "tooltip": (
                            "Colour palette applied to the generated pattern. "
                            "'greyscale' = neutral luminance only (recommended for most use); "
                            "'monochrome' = single saturated hue; "
                            "'psychedelic' = full multi-hue spectrum."
                        )
                    },
                ),
                "tile_width": (
                    "INT",
                    {
                        "default": 128,
                        "min": 32,
                        "max": 512,
                        "tooltip": (
                            "Width of the generated tile in pixels. "
                            "Must be smaller than the output stereogram width. "
                            "Larger tiles reduce visible repetition. Typical: 64–256."
                        ),
                    },
                ),
                "tile_height": (
                    "INT",
                    {
                        "default": 128,
                        "min": 32,
                        "max": 512,
                        "tooltip": (
                            "Height of the generated tile in pixels. "
                            "Must be smaller than the output stereogram height. "
                            "Larger tiles reduce visible repetition. Typical: 64–256."
                        ),
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 42,
                        "min": 0,
                        "max": 99999,
                        "tooltip": (
                            "Random seed for reproducible pattern generation. "
                            "Same seed always produces the same tile. "
                            "Change to explore different pattern variations."
                        ),
                    },
                ),
                "scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 8.0,
                        "step": 0.1,
                        "tooltip": (
                            "Feature size multiplier. "
                            "1.0 = default scale; <1.0 = finer detail; >1.0 = coarser/larger features. "
                            "Safe range: 0.5–4.0."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("PATTERN", "IMAGE")
    RETURN_NAMES = ("pattern", "pattern_preview")

    def execute(self, pattern_type, color_mode, tile_width, tile_height, seed, scale):
        from depthforge.core.pattern_gen import (
            ColorMode,
            PatternParams,
            PatternType,
            generate_pattern,
        )

        type_map = {
            "random_noise": PatternType.RANDOM_NOISE,
            "perlin": PatternType.PERLIN,
            "plasma": PatternType.PLASMA,
            "voronoi": PatternType.VORONOI,
            "geometric_grid": PatternType.GEOMETRIC_GRID,
            "mandelbrot": PatternType.MANDELBROT,
            "dot_matrix": PatternType.DOT_MATRIX,
        }
        color_map = {
            "greyscale": ColorMode.GREYSCALE,
            "monochrome": ColorMode.MONOCHROME,
            "psychedelic": ColorMode.PSYCHEDELIC,
        }
        pattern = generate_pattern(
            PatternParams(
                pattern_type=type_map[pattern_type],
                tile_width=tile_width,
                tile_height=tile_height,
                color_mode=color_map[color_mode],
                seed=seed,
                scale=scale,
            )
        )
        preview = _numpy_to_tensor(pattern)
        return (pattern, preview)


# ===========================================================================
# Node 3 — DF_PatternLibrary
# ===========================================================================


class DF_PatternLibrary(_DFNodeBase):
    """Access the DepthForge named pattern library (28+ patterns)."""

    DESCRIPTION = (
        "Select and generate a pattern tile from the DepthForge curated library "
        "of 28+ named production patterns, each tuned for stereogram use."
    )

    # Build name list at class definition time — safe to do since pattern_library
    # has no heavy imports
    _NAMES: list = []

    @classmethod
    def _get_names(cls):
        if not cls._NAMES:
            try:
                from depthforge.core.pattern_library import list_patterns

                cls._NAMES = list_patterns()
            except Exception:
                cls._NAMES = ["perlin_noise", "plasma_wave", "crystalline", "fine_grain"]
        return cls._NAMES

    @classmethod
    def INPUT_TYPES(cls):
        names = cls._get_names()
        return {
            "required": {
                "pattern_name": (
                    names,
                    {
                        "tooltip": (
                            "Select from the 28+ pre-designed production patterns. "
                            "Each pattern is tuned for stereogram use with appropriate "
                            "frequency, contrast, and tileability."
                        )
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 128,
                        "min": 32,
                        "max": 512,
                        "tooltip": (
                            "Tile width in pixels. Larger tiles reduce visible repetition "
                            "in the final stereogram. Typical: 64–256."
                        ),
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 128,
                        "min": 32,
                        "max": 512,
                        "tooltip": (
                            "Tile height in pixels. Larger tiles reduce visible repetition "
                            "in the final stereogram. Typical: 64–256."
                        ),
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 42,
                        "min": 0,
                        "max": 99999,
                        "tooltip": (
                            "Random seed for reproducible tile generation. "
                            "Same seed always produces the same pattern. Range: 0–99999."
                        ),
                    },
                ),
                "scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 8.0,
                        "step": 0.1,
                        "tooltip": (
                            "Feature size multiplier. "
                            "1.0 = designed default; 0.5 = finer detail; 2.0 = coarser. "
                            "Safe range: 0.5–3.0."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("PATTERN", "IMAGE", "STRING")
    RETURN_NAMES = ("pattern", "pattern_preview", "pattern_info")

    def execute(self, pattern_name, width, height, seed, scale):
        from depthforge.core.pattern_library import get_pattern, get_pattern_info

        pattern = get_pattern(pattern_name, width=width, height=height, seed=seed, scale=scale)
        info = get_pattern_info(pattern_name)
        info_str = f"{pattern_name} [{info.category}]: {info.description}"
        preview = _numpy_to_tensor(pattern)
        return (pattern, preview, info_str)


# ===========================================================================
# Node 4 — DF_Stereogram (core synthesis)
# ===========================================================================


class DF_Stereogram(_DFNodeBase):
    """Synthesize a stereogram from a depth map and pattern tile."""

    DESCRIPTION = (
        "Core stereogram synthesis node — converts a depth map and pattern tile "
        "into a SIRDS or texture stereogram using the DepthForge engine."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth_map": (
                    "DEPTH_MAP",
                    {
                        "tooltip": (
                            "Conditioned depth map (connect DF_DepthPrep output here "
                            "for best results). White = near, black = far."
                        )
                    },
                ),
                "pattern": (
                    "PATTERN",
                    {
                        "tooltip": (
                            "Pattern tile to repeat across the stereogram. "
                            "Connect DF_PatternGen or DF_PatternLibrary output here."
                        )
                    },
                ),
            },
            "optional": {
                "depth_factor": (
                    "FLOAT",
                    {
                        "default": 0.35,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Controls depth strength (stereo pop intensity). "
                            "0.35 = comfortable pop for most viewers; "
                            "0.5 = dramatic; above 0.6 risks eye strain. "
                            "Negative values invert depth. Safe range: 0.1–0.5."
                        ),
                    },
                ),
                "max_parallax": (
                    "FLOAT",
                    {
                        "default": 0.033,
                        "min": 0.005,
                        "max": 0.1,
                        "step": 0.001,
                        "tooltip": (
                            "Maximum pixel shift as a fraction of image width. "
                            "0.033 = comfortable (≈1/30 of width); "
                            "0.05 = strong; above 0.07 causes eye strain. "
                            "Safe range: 0.01–0.05."
                        ),
                    },
                ),
                "oversample": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 4,
                        "tooltip": (
                            "Anti-aliasing quality multiplier. "
                            "1 = fastest (no oversampling); "
                            "2 = smoother edges; "
                            "4 = maximum quality (4× slower). "
                            "Use 2 for print output, 1 for screen."
                        ),
                    },
                ),
                "safe_mode": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Auto-clamp depth to prevent eye strain violations. "
                            "Recommended for broadcast, public display, or print. "
                            "Overrides depth_factor if it exceeds safe limits."
                        ),
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 42,
                        "min": 0,
                        "max": 99999,
                        "tooltip": (
                            "Random seed for SIRDS dot pattern generation. "
                            "Change to explore different dot arrangements "
                            "with the same depth and pattern inputs."
                        ),
                    },
                ),
                "invert_depth": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Swap near/far depth interpretation. "
                            "Enable if the stereogram appears to pop inward "
                            "(hollow instead of raised) when cross-fusing."
                        ),
                    },
                ),
                "preset_name": (
                    ["none", "shallow", "medium", "deep", "cinema", "print", "broadcast"],
                    {
                        "tooltip": (
                            "Apply a named production preset. When set (not 'none'), "
                            "overrides depth_factor and max_parallax with tuned values: "
                            "'shallow' = subtle; 'cinema' = max comfortable; "
                            "'broadcast' = ITU-R safe limits."
                        )
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("stereogram",)
    OUTPUT_NODE = True

    def execute(
        self,
        depth_map,
        pattern,
        depth_factor=0.35,
        max_parallax=0.033,
        oversample=1,
        safe_mode=False,
        seed=42,
        invert_depth=False,
        preset_name="none",
    ):
        from depthforge.core.synthesizer import StereoParams, synthesize

        if preset_name != "none":
            from depthforge.core.presets import get_preset

            p = get_preset(preset_name)
            sp = p.stereo_params
        else:
            sp = StereoParams(
                depth_factor=depth_factor,
                max_parallax_fraction=max_parallax,
                oversample=oversample,
                safe_mode=safe_mode,
                seed=seed,
                invert_depth=invert_depth,
            )

        depth = _depth_to_numpy(depth_map)
        result = synthesize(depth, pattern, sp)
        return (_numpy_to_tensor(result),)


# ===========================================================================
# Node 5 — DF_AnaglyphOut
# ===========================================================================


class DF_AnaglyphOut(_DFNodeBase):
    """Generate a red/cyan anaglyph for 3D glasses viewing."""

    DESCRIPTION = (
        "Convert a source image and depth map into a red/cyan anaglyph "
        "for viewing with standard 3D glasses, with 5 algorithm modes."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Source colour image to convert to anaglyph. "
                            "Connect any RGB image output here."
                        )
                    },
                ),
                "depth_map": (
                    "DEPTH_MAP",
                    {
                        "tooltip": (
                            "Depth map controlling per-pixel stereo shift. "
                            "White = near (maximum shift), black = far (no shift)."
                        )
                    },
                ),
            },
            "optional": {
                "mode": (
                    ["true", "grey", "colour", "half_colour", "optimised"],
                    {
                        "tooltip": (
                            "Anaglyph algorithm. "
                            "'optimised' = best colour retention for red/cyan glasses (recommended); "
                            "'grey' = no colour ghosting, works with all glasses; "
                            "'true' = simple channel split; "
                            "'half_colour' = compromise between colour and ghosting."
                        )
                    },
                ),
                "depth_factor": (
                    "FLOAT",
                    {
                        "default": 0.35,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Stereo strength multiplier. "
                            "0.35 = comfortable; 0.5 = strong. "
                            "Above 0.5 may cause ghosting with cheap glasses. "
                            "Safe range: 0.1–0.5."
                        ),
                    },
                ),
                "parallax_px": (
                    "INT",
                    {
                        "default": 30,
                        "min": 1,
                        "max": 200,
                        "tooltip": (
                            "Base eye separation in pixels at full depth. "
                            "30 px = natural for 1080p viewing distance; "
                            "60 px = strong for 4K. "
                            "Scale proportionally with resolution."
                        ),
                    },
                ),
                "swap_eyes": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Reverse left/right eye assignment. "
                            "Enable when glasses produce inverted depth "
                            "(objects appear to recede instead of pop forward)."
                        ),
                    },
                ),
                "gamma": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.5,
                        "max": 3.0,
                        "step": 0.1,
                        "tooltip": (
                            "Gamma correction for display compensation. "
                            "1.0 = neutral (default); "
                            "2.2 = standard sRGB monitor; "
                            "1.8 = legacy Mac display. "
                            "Safe range: 0.8–2.2."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("anaglyph",)
    OUTPUT_NODE = True

    def execute(
        self,
        source_image,
        depth_map,
        mode="optimised",
        depth_factor=0.35,
        parallax_px=30,
        swap_eyes=False,
        gamma=1.0,
    ):
        from depthforge.core.anaglyph import AnaglyphMode, AnaglyphParams, make_anaglyph_from_depth

        mode_map = {
            "true": AnaglyphMode.TRUE_ANAGLYPH,
            "grey": AnaglyphMode.GREY_ANAGLYPH,
            "colour": AnaglyphMode.COLOUR_ANAGLYPH,
            "half_colour": AnaglyphMode.HALF_COLOUR,
            "optimised": AnaglyphMode.OPTIMISED,
        }
        src = _tensor_to_numpy(source_image)
        depth = _depth_to_numpy(depth_map)
        W = src.shape[1]
        px = int(W * 0.065 * depth_factor)

        params = AnaglyphParams(
            mode=mode_map[mode],
            parallax_px=px,
            swap_eyes=swap_eyes,
            gamma=gamma,
        )
        result = make_anaglyph_from_depth(src, depth, params)
        return (_numpy_to_tensor(result),)


# ===========================================================================
# Node 6 — DF_StereoPair
# ===========================================================================


class DF_StereoPair(_DFNodeBase):
    """Generate left/right stereo views with occlusion mask."""

    DESCRIPTION = (
        "Warp a source image into a left/right stereo pair using a depth map, "
        "outputting separate views, an occlusion mask, and an optional composed layout."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Source colour image to split into a left/right stereo pair. "
                            "Connect any RGB image output here."
                        )
                    },
                ),
                "depth_map": (
                    "DEPTH_MAP",
                    {
                        "tooltip": (
                            "Depth map controlling per-pixel left/right shift. "
                            "White = near (maximum disparity), black = far (no shift)."
                        )
                    },
                ),
            },
            "optional": {
                "max_parallax_fraction": (
                    "FLOAT",
                    {
                        "default": 0.033,
                        "min": 0.005,
                        "max": 0.1,
                        "step": 0.001,
                        "tooltip": (
                            "Maximum stereo shift as a fraction of image width. "
                            "0.033 = comfortable (1/30); 0.05 = strong; "
                            "above 0.06 causes eye strain. Safe range: 0.01–0.05."
                        ),
                    },
                ),
                "eye_balance": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "tooltip": (
                            "Distributes the total parallax shift between eyes. "
                            "0.5 = symmetric (recommended); "
                            "0.0 = shift right eye only; "
                            "1.0 = shift left eye only."
                        ),
                    },
                ),
                "background_fill": (
                    ["edge", "mirror", "black"],
                    {
                        "tooltip": (
                            "How to fill occlusion gaps at image edges after warping. "
                            "'edge' = repeat edge pixels (most natural); "
                            "'mirror' = mirror image content; "
                            "'black' = solid black fill."
                        )
                    },
                ),
                "feather_px": (
                    "INT",
                    {
                        "default": 3,
                        "min": 0,
                        "max": 20,
                        "tooltip": (
                            "Soft blend width at occlusion edges in pixels. "
                            "3–8 recommended for smooth transitions. "
                            "0 = hard edges (may show artifacts)."
                        ),
                    },
                ),
                "layout": (
                    ["separate", "side_by_side", "top_bottom"],
                    {
                        "tooltip": (
                            "Output arrangement for the composed output pin. "
                            "'separate' = just returns left view as composed; "
                            "'side_by_side' = SBS format for 3D TVs; "
                            "'top_bottom' = TB format for VR headsets."
                        )
                    },
                ),
                "gap_px": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 64,
                        "tooltip": (
                            "Gap in pixels between left and right views in "
                            "side_by_side or top_bottom layouts. "
                            "0–4 typical; 0 for most delivery formats."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("left_view", "right_view", "occlusion_mask", "composed")
    OUTPUT_NODE = True

    def execute(
        self,
        source_image,
        depth_map,
        max_parallax_fraction=0.033,
        eye_balance=0.5,
        background_fill="edge",
        feather_px=3,
        layout="separate",
        gap_px=0,
    ):
        from depthforge.core.stereo_pair import (
            StereoLayout,
            StereoPairParams,
            compose_side_by_side,
            make_stereo_pair,
        )

        src = _tensor_to_numpy(source_image)
        depth = _depth_to_numpy(depth_map)

        layout_map = {
            "separate": StereoLayout.SEPARATE,
            "side_by_side": StereoLayout.SIDE_BY_SIDE,
            "top_bottom": StereoLayout.TOP_BOTTOM,
        }

        params = StereoPairParams(
            max_parallax_fraction=max_parallax_fraction,
            eye_balance=eye_balance,
            background_fill=background_fill,
            feather_px=feather_px,
            layout=layout_map[layout],
        )
        left, right, occ = make_stereo_pair(src, depth, params)

        # Composed output
        if layout == "side_by_side":
            composed = compose_side_by_side(left, right, gap_px=gap_px)
        elif layout == "top_bottom":
            gap_arr = np.zeros((gap_px, left.shape[1], 4), dtype=np.uint8) if gap_px > 0 else None
            composed = np.concatenate(
                [left, gap_arr, right] if gap_arr is not None else [left, right], axis=0
            )
        else:
            composed = left  # "separate" → just return left as composed

        # occ is bool mask → convert to RGBA image
        occ_rgba = np.zeros((*occ.shape, 4), dtype=np.uint8)
        occ_rgba[occ > 0.5, :] = 255

        return (
            _numpy_to_tensor(left),
            _numpy_to_tensor(right),
            _numpy_to_tensor(occ_rgba),
            _numpy_to_tensor(composed),
        )


# ===========================================================================
# Node 7 — DF_HiddenImage
# ===========================================================================


class DF_HiddenImage(_DFNodeBase):
    """Encode a hidden shape or text inside a random dot stereogram."""

    DESCRIPTION = (
        "Encode a hidden 3D shape, text, or custom mask into a random dot stereogram "
        "— the hidden object is invisible until cross-fused by the viewer."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pattern": (
                    "PATTERN",
                    {
                        "tooltip": (
                            "Background dot pattern tile. "
                            "Connect DF_PatternGen or DF_PatternLibrary. "
                            "Random noise and fine-grain patterns work best."
                        )
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 1920,
                        "min": 256,
                        "max": 4096,
                        "tooltip": (
                            "Output stereogram width in pixels. "
                            "Must be at least 4× the pattern tile width."
                        ),
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1080,
                        "min": 128,
                        "max": 2160,
                        "tooltip": (
                            "Output stereogram height in pixels. "
                            "Must be at least 4× the pattern tile height."
                        ),
                    },
                ),
            },
            "optional": {
                "shape": (
                    ["none", "circle", "square", "triangle", "star", "diamond", "arrow"],
                    {
                        "tooltip": (
                            "Built-in geometric shape to encode as the hidden 3D object. "
                            "Set to 'none' to use text or mask_image input instead. "
                            "Shape is centred in the frame."
                        )
                    },
                ),
                "text": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Text string to encode as the hidden object. "
                            "Leave blank to use shape or mask_image. "
                            "Short words (3–8 chars) work best. Font is bold sans-serif."
                        ),
                    },
                ),
                "font_size": (
                    "INT",
                    {
                        "default": 180,
                        "min": 20,
                        "max": 600,
                        "tooltip": (
                            "Rendered text size in pixels. "
                            "120–240 recommended for clear legibility in the final stereogram. "
                            "Too small = hard to fuse; too large = depth artifacts at edges."
                        ),
                    },
                ),
                "foreground_depth": (
                    "FLOAT",
                    {
                        "default": 0.75,
                        "min": 0.1,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "Depth value for the hidden object — higher values pop further forward. "
                            "0.75 = strong comfortable pop. Safe range: 0.5–0.9. "
                            "Must be greater than background_depth."
                        ),
                    },
                ),
                "background_depth": (
                    "FLOAT",
                    {
                        "default": 0.10,
                        "min": 0.0,
                        "max": 0.9,
                        "step": 0.05,
                        "tooltip": (
                            "Depth value for the background plane — lower values recede further. "
                            "0.10 = naturally receding background. Safe range: 0.0–0.3. "
                            "Must be less than foreground_depth."
                        ),
                    },
                ),
                "edge_soften_px": (
                    "INT",
                    {
                        "default": 4,
                        "min": 0,
                        "max": 20,
                        "tooltip": (
                            "Gaussian blur radius at the shape boundary in pixels. "
                            "4–8 recommended for smooth depth transitions at edges. "
                            "0 = hard edge (may show depth artifacts)."
                        ),
                    },
                ),
                "mask_image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Optional custom greyscale mask — white areas become the hidden "
                            "foreground object, black areas become background. "
                            "Overrides shape and text inputs when connected."
                        )
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("hidden_stereogram", "depth_preview")
    OUTPUT_NODE = True

    def execute(
        self,
        pattern,
        width,
        height,
        shape="none",
        text="",
        font_size=180,
        foreground_depth=0.75,
        background_depth=0.10,
        edge_soften_px=4,
        mask_image=None,
    ):
        from depthforge.core.hidden_image import (
            HiddenImageParams,
            encode_hidden_image,
            shape_to_mask,
            text_to_mask,
        )

        if mask_image is not None:
            msk_np = _tensor_to_numpy(mask_image)
            msk = msk_np[:, :, 0].astype(np.float32) / 255.0
            H_m, W_m = msk.shape
        elif text.strip():
            msk = text_to_mask(text.strip(), width, height, font_size=font_size)
        elif shape != "none":
            msk = shape_to_mask(shape, width, height)
        else:
            # Default: star
            msk = shape_to_mask("star", width, height)

        params = HiddenImageParams(
            foreground_depth=foreground_depth,
            background_depth=background_depth,
            edge_soften_px=edge_soften_px,
        )
        result = encode_hidden_image(pattern, msk, params)

        # Depth preview
        if hasattr(msk, "shape") and msk.ndim == 2:
            depth_vis = (msk * 255).astype(np.uint8)
            depth_rgba = np.stack(
                [depth_vis, depth_vis, depth_vis, np.full_like(depth_vis, 255)], axis=-1
            )
        else:
            depth_rgba = np.full((height, width, 4), 128, dtype=np.uint8)

        return (_numpy_to_tensor(result), _numpy_to_tensor(depth_rgba))


# ===========================================================================
# Node 8 — DF_SafetyLimiter
# ===========================================================================


class DF_SafetyLimiter(_DFNodeBase):
    """Apply comfort safety limits to a depth map and stereo parameters."""

    DESCRIPTION = (
        "Clamp a depth map to comfort and PSE safety limits using one of four "
        "vergence profiles, outputting a corrected depth map and a violation report."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth_map": (
                    "DEPTH_MAP",
                    {
                        "tooltip": (
                            "Raw or conditioned depth map to be safety-checked. "
                            "Connect directly after DF_DepthPrep or before DF_Stereogram."
                        )
                    },
                ),
            },
            "optional": {
                "profile": (
                    ["conservative", "standard", "relaxed", "cinema"],
                    {
                        "tooltip": (
                            "Pre-configured safety profile. "
                            "'conservative' = minimal depth, safest for all audiences; "
                            "'standard' = comfortable for healthy adults; "
                            "'relaxed' = stronger depth, not for extended viewing; "
                            "'cinema' = maximum comfortable theatrical depth."
                        )
                    },
                ),
                "max_depth_factor": (
                    "FLOAT",
                    {
                        "default": 0.6,
                        "min": 0.1,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "Hard cap on depth_factor regardless of input. "
                            "0.6 = safe for most viewers; 0.4 = conservative broadcast limit. "
                            "Safe range: 0.2–0.7."
                        ),
                    },
                ),
                "max_gradient": (
                    "FLOAT",
                    {
                        "default": 0.15,
                        "min": 0.01,
                        "max": 0.5,
                        "step": 0.01,
                        "tooltip": (
                            "Maximum allowed depth rate-of-change per pixel. "
                            "0.15 prevents sudden depth jumps that cause eye discomfort. "
                            "Safe range: 0.05–0.25."
                        ),
                    },
                ),
                "near_clip": (
                    "FLOAT",
                    {
                        "default": 0.02,
                        "min": 0.0,
                        "max": 0.2,
                        "step": 0.01,
                        "tooltip": (
                            "Clip depth values below this threshold — removes extreme "
                            "near content that causes convergence stress. "
                            "Safe range: 0.02–0.05."
                        ),
                    },
                ),
                "far_clip": (
                    "FLOAT",
                    {
                        "default": 0.98,
                        "min": 0.8,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Clip depth values above this threshold — removes extreme "
                            "far content that causes divergence stress. "
                            "Safe range: 0.95–0.99."
                        ),
                    },
                ),
                "warn_on_violation": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Print a PSE safety warning to the ComfyUI console when "
                            "violations are detected and corrected. "
                            "Recommended to keep enabled for all production work."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("DEPTH_MAP", "IMAGE", "STRING")
    RETURN_NAMES = ("safe_depth", "depth_preview", "violation_report")

    def execute(
        self,
        depth_map,
        profile="standard",
        max_depth_factor=0.6,
        max_gradient=0.15,
        near_clip=0.02,
        far_clip=0.98,
        warn_on_violation=True,
    ):
        from depthforge.core.comfort import SafetyLimiter, SafetyLimiterParams, VergenceProfile

        profile_map = {
            "conservative": VergenceProfile.CONSERVATIVE,
            "standard": VergenceProfile.STANDARD,
            "relaxed": VergenceProfile.RELAXED,
            "cinema": VergenceProfile.CINEMA,
        }

        p = SafetyLimiterParams.from_profile(
            profile_map[profile],
            max_depth_factor=max_depth_factor,
            max_gradient=max_gradient,
            near_clip=near_clip,
            far_clip=far_clip,
            warn_on_violation=warn_on_violation,
        )
        limiter = SafetyLimiter(p)
        depth = _depth_to_numpy(depth_map)
        safe_depth, _, violations = limiter.apply(depth)

        from depthforge.core.flash_safety import EPILEPSY_WARNING_SHORT

        if violations:
            violation_lines = "\n".join(f"[{k}] {v}" for k, v in violations.items())
            report = (
                f"PSE SAFETY WARNING: {EPILEPSY_WARNING_SHORT}\n\n"
                f"Depth violations corrected:\n{violation_lines}"
            )
        else:
            report = f"✓ No depth violations\n\n" f"PSE REMINDER: {EPILEPSY_WARNING_SHORT}"
        preview = _numpy_depth_to_tensor(safe_depth)
        return (safe_depth, preview, report)


# ===========================================================================
# Node 9 — DF_QCOverlay
# ===========================================================================


class DF_QCOverlay(_DFNodeBase):
    """Generate QC visualisation overlays for depth maps."""

    DESCRIPTION = (
        "Generate quality-control visualisation overlays for a depth map — "
        "parallax heatmaps, depth band previews, violation highlights, and histograms."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth_map": (
                    "DEPTH_MAP",
                    {
                        "tooltip": (
                            "Depth map to analyse and visualise. "
                            "Connect DF_DepthPrep or DF_SafetyLimiter output here."
                        )
                    },
                ),
            },
            "optional": {
                "overlay_type": (
                    [
                        "parallax_heatmap",
                        "depth_bands",
                        "violation_overlay",
                        "safe_zone",
                        "histogram",
                    ],
                    {
                        "tooltip": (
                            "Visualisation mode. "
                            "'parallax_heatmap' = colour-coded pixel-shift map; "
                            "'depth_bands' = quantised depth zones; "
                            "'violation_overlay' = red highlights on problem areas; "
                            "'safe_zone' = green/red comfort zone indicator; "
                            "'histogram' = depth value distribution chart."
                        )
                    },
                ),
                "frame_width": (
                    "INT",
                    {
                        "default": 1920,
                        "min": 128,
                        "max": 7680,
                        "tooltip": (
                            "Output frame width used to calculate absolute pixel parallax "
                            "from depth fractions. Match to your actual output resolution."
                        ),
                    },
                ),
                "depth_factor": (
                    "FLOAT",
                    {
                        "default": 0.35,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Depth strength assumed for the parallax calculation. "
                            "Should match the depth_factor you will use in DF_Stereogram. "
                            "Safe range: 0.1–0.6."
                        ),
                    },
                ),
                "max_parallax": (
                    "FLOAT",
                    {
                        "default": 0.033,
                        "min": 0.005,
                        "max": 0.1,
                        "step": 0.001,
                        "tooltip": (
                            "Maximum parallax fraction assumed for violation detection. "
                            "Should match the max_parallax in DF_Stereogram. "
                            "Safe range: 0.01–0.05."
                        ),
                    },
                ),
                "n_bands": (
                    "INT",
                    {
                        "default": 8,
                        "min": 2,
                        "max": 16,
                        "tooltip": (
                            "Number of depth quantisation bands for the 'depth_bands' overlay. "
                            "8 bands recommended for clear zone identification. "
                            "More bands = finer depth resolution display."
                        ),
                    },
                ),
                "colormap": (
                    ["inferno", "jet", "viridis", "plasma", "turbo"],
                    {
                        "tooltip": (
                            "Colour palette for the heatmap overlay. "
                            "'inferno' = dark-to-yellow (perceptually uniform, recommended); "
                            "'viridis' = blue-to-yellow; "
                            "'jet' = classic rainbow; "
                            "'turbo' = improved rainbow with better contrast."
                        )
                    },
                ),
                "annotate": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Add text labels, pixel-count annotations, and scale bars "
                            "to the overlay image. Disable for clean compositing."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("qc_image", "comfort_summary")
    OUTPUT_NODE = True

    def execute(
        self,
        depth_map,
        overlay_type="parallax_heatmap",
        frame_width=1920,
        depth_factor=0.35,
        max_parallax=0.033,
        n_bands=8,
        colormap="inferno",
        annotate=True,
    ):
        from depthforge.core.comfort import ComfortAnalyzer, ComfortAnalyzerParams
        from depthforge.core.qc import (
            QCParams,
            depth_band_preview,
            depth_histogram,
            parallax_heatmap,
            safe_zone_indicator,
            window_violation_overlay,
        )

        depth = _depth_to_numpy(depth_map)
        H, W = depth.shape
        qp = QCParams(
            frame_width=frame_width,
            depth_factor=depth_factor,
            max_parallax_fraction=max_parallax,
            colormap=colormap,
        )

        if overlay_type == "parallax_heatmap":
            out = parallax_heatmap(depth, qp, annotate=annotate)
        elif overlay_type == "depth_bands":
            out = depth_band_preview(depth, n_bands=n_bands)
        elif overlay_type == "violation_overlay":
            out = window_violation_overlay(depth)
        elif overlay_type == "safe_zone":
            out = safe_zone_indicator(depth, qp)
        elif overlay_type == "histogram":
            out = depth_histogram(depth, width=W, height=max(H // 4, 100))
        else:
            out = parallax_heatmap(depth, qp, annotate=annotate)

        # Comfort summary
        analyzer = ComfortAnalyzer(
            ComfortAnalyzerParams(
                frame_width=frame_width,
                depth_factor=depth_factor,
                max_parallax_fraction=max_parallax,
            )
        )
        report = analyzer.analyze(depth)
        summary = report.summary()

        return (_numpy_to_tensor(out), summary)


# ===========================================================================
# Node 10 — DF_VideoSequence
# ===========================================================================


class DF_VideoSequence(_DFNodeBase):
    """Process a batch of depth frames for video stereogram output.

    Applies temporal smoothing between frames to prevent flickering.
    Input: a batch IMAGE tensor (B, H, W, C) representing sequential frames.
    Output: batch of stereogram frames (B, H, W, 4).
    """

    DESCRIPTION = (
        "Process a batch of depth frames into a temporally-coherent stereogram "
        "video sequence with flash safety checking and per-frame PSE reporting."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth_sequence": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Batch of depth frames as a ComfyUI IMAGE tensor (B, H, W, C). "
                            "Connect a video loader or batch depth estimator output here."
                        )
                    },
                ),
                "pattern": (
                    "PATTERN",
                    {
                        "tooltip": (
                            "Pattern tile to use for all frames in the sequence. "
                            "Keep the same pattern across the sequence for visual coherence."
                        )
                    },
                ),
            },
            "optional": {
                "depth_factor": (
                    "FLOAT",
                    {
                        "default": 0.35,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "Stereo depth strength applied to every frame. "
                            "0.35 = comfortable; 0.5 = strong. "
                            "Above 0.5 may cause fatigue during extended viewing. "
                            "Safe range: 0.1–0.5."
                        ),
                    },
                ),
                "max_parallax": (
                    "FLOAT",
                    {
                        "default": 0.033,
                        "min": 0.005,
                        "max": 0.1,
                        "step": 0.001,
                        "tooltip": (
                            "Maximum parallax as a fraction of image width per frame. "
                            "0.033 = comfortable for video; above 0.05 not recommended "
                            "for sustained video viewing. Safe range: 0.01–0.04."
                        ),
                    },
                ),
                "temporal_smooth": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": (
                            "Blend factor with the previous frame's depth to reduce flicker. "
                            "0.0 = no smoothing (sharp frame cuts); "
                            "0.3 = subtle (recommended); "
                            "0.8 = heavy temporal blur. "
                            "Higher values reduce depth pop during motion."
                        ),
                    },
                ),
                "safe_mode": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Auto-clamp depth per frame to prevent sudden depth spikes. "
                            "Highly recommended for video to avoid jarring depth changes "
                            "between frames. Leave enabled for all production work."
                        ),
                    },
                ),
                "preset_name": (
                    ["none", "shallow", "medium", "deep", "cinema", "broadcast"],
                    {
                        "tooltip": (
                            "Apply a named production preset to all frames. "
                            "When set (not 'none'), overrides depth_factor and max_parallax. "
                            "'broadcast' = ITU-R safe limits for television delivery."
                        )
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 42,
                        "min": 0,
                        "max": 99999,
                        "tooltip": (
                            "Base random seed — each frame uses seed + frame_index "
                            "for stable but varied dot patterns across the sequence. "
                            "Change to get a completely different sequence look."
                        ),
                    },
                ),
                "fps": (
                    "FLOAT",
                    {
                        "default": 24.0,
                        "min": 1.0,
                        "max": 120.0,
                        "step": 1.0,
                        "tooltip": (
                            "Playback frame rate — used for flash safety check (PSE limit: 3 Hz). "
                            "Match to your actual output frame rate: 24 = cinema; "
                            "25 = PAL; 29.97/30 = NTSC; 60 = high frame rate."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("stereogram_sequence", "flash_safety_report")
    OUTPUT_NODE = True

    def execute(
        self,
        depth_sequence,
        pattern,
        depth_factor=0.35,
        max_parallax=0.033,
        temporal_smooth=0.3,
        safe_mode=True,
        preset_name="none",
        seed=42,
        fps=24.0,
    ):
        from depthforge.core.depth_prep import DepthPrepParams, prep_depth
        from depthforge.core.synthesizer import StereoParams, synthesize

        if preset_name != "none":
            from depthforge.core.presets import get_preset

            p = get_preset(preset_name)
            sp = p.stereo_params
            dp = p.depth_params
        else:
            sp = StereoParams(
                depth_factor=depth_factor,
                max_parallax_fraction=max_parallax,
                safe_mode=safe_mode,
                seed=seed,
            )
            dp = DepthPrepParams()

        # Handle both batched tensor and single frame
        try:
            import torch

            if isinstance(depth_sequence, torch.Tensor):
                frames_np = depth_sequence.cpu().numpy()  # (B, H, W, C)
            else:
                frames_np = np.array(depth_sequence)
        except ImportError:
            frames_np = np.array(depth_sequence)

        if frames_np.ndim == 3:
            frames_np = frames_np[None]  # add batch dim

        n_frames = frames_np.shape[0]
        results = []
        prev_depth = None

        for i in range(n_frames):
            frame = frames_np[i]
            if frame.shape[-1] > 1:
                raw_depth = frame.mean(axis=-1).astype(np.float32)
            else:
                raw_depth = frame[:, :, 0].astype(np.float32)

            # Temporal smoothing — blend with previous frame's depth
            if prev_depth is not None and temporal_smooth > 0:
                raw_depth = raw_depth * (1 - temporal_smooth) + prev_depth * temporal_smooth
            prev_depth = raw_depth.copy()

            depth = prep_depth(raw_depth, dp)

            # Use per-frame seed offset for stable dot patterns
            sp_frame = StereoParams(**{k: v for k, v in sp.__dict__.items()})
            sp_frame.seed = sp.seed + i if sp.seed is not None else None

            stereo = synthesize(depth, pattern, sp_frame)
            results.append(stereo)

        # Stack into batch tensor
        batch = np.stack(results, axis=0).astype(np.float32) / 255.0  # (B,H,W,4)

        # --- Flash safety check ---
        from depthforge.core.flash_safety import EPILEPSY_WARNING_SHORT, check_frame_sequence

        flash_report = check_frame_sequence(results, fps=float(fps))
        flash_summary = (
            f"PSE Flash Safety Report\n" f"{flash_report.summary()}\n\n" f"{EPILEPSY_WARNING_SHORT}"
        )
        import warnings

        if not flash_report.passed:
            warnings.warn(
                f"DepthForge DF_VideoSequence: {flash_report.risk.value} flash risk "
                f"detected at {fps:.0f} fps.\n"
                + "\n".join(f"  · {v}" for v in flash_report.violations),
                UserWarning,
                stacklevel=2,
            )

        try:
            import torch

            return (torch.from_numpy(batch), flash_summary)
        except ImportError:
            return (batch, flash_summary)


# ===========================================================================
# QR Stereogram helpers
# ===========================================================================


def _make_qr_stereogram_image(
    pattern_rgba: np.ndarray,
    depth_2d: np.ndarray,
    max_shift_px: int,
    n_copies: int = 4,
) -> np.ndarray:
    """SIRDS-style QR stereogram.

    The first ``pattern_width`` columns of the output are the unmodified QR
    code (directly scannable).  Each subsequent column references a pixel
    ~one pattern-period to its left, offset forward by ``depth * max_shift_px``
    pixels, exactly as in a classic SIRDS algorithm.

    Parameters
    ----------
    pattern_rgba : (H, W, 4) uint8
        The QR code rendered as RGBA.
    depth_2d : (H_d, W_d) float32
        Depth map in [0, 1].  Values are bilinearly resampled to the output
        dimensions before the shift map is computed.
    max_shift_px : int
        Maximum horizontal shift in pixels at depth=1.0.
    n_copies : int
        How many QR-width columns to produce (first is always the clean QR).
    """
    H, P, _ = pattern_rgba.shape
    W_out = P * n_copies

    # Resize depth map to output dimensions
    try:
        from PIL import Image as _PILImg

        depth_uint8 = (np.clip(depth_2d, 0.0, 1.0) * 255).astype(np.uint8)
        depth_resized = (
            np.array(_PILImg.fromarray(depth_uint8).resize((W_out, H), _PILImg.BILINEAR)).astype(
                np.float32
            )
            / 255.0
        )
    except ImportError:
        # Fallback: tile-and-crop resize
        reps_h = max(1, H // depth_2d.shape[0] + 1)
        reps_w = max(1, W_out // depth_2d.shape[1] + 1)
        depth_resized = np.tile(depth_2d, (reps_h, reps_w))[:H, :W_out].astype(np.float32)

    shifts = np.clip(
        (depth_resized * max_shift_px).astype(np.int32), 0, max_shift_px
    )  # (H, W_out)

    output = np.zeros((H, W_out, 4), dtype=np.uint8)
    output[:, :P] = pattern_rgba  # seed: clean QR in first copy

    row_idx = np.arange(H)
    for x in range(P, W_out):
        # Each pixel looks back one period, shifted forward by depth offset
        src_x = np.clip(x - P + shifts[:, x], 0, x - 1)  # (H,) row-wise source
        output[:, x] = output[row_idx, src_x]

    return output


# ===========================================================================
# Node 11 — DF_QRStereogram
# ===========================================================================


class DF_QRStereogram(_DFNodeBase):
    """QR code that is simultaneously a viewable stereogram."""

    DESCRIPTION = (
        "Generate a QR code whose module grid becomes the dot pattern of a SIRDS stereogram — "
        "the output is scannable by a standard QR reader and reveals 3D depth when cross-fused."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url_or_text": (
                    "STRING",
                    {
                        "default": "https://github.com/BoxOfllc/SUPERIOR_StereoGram",
                        "tooltip": (
                            "Text or URL to encode. Shorter strings produce smaller, "
                            "lower-version QR codes — fewer modules means wider depth offsets "
                            "are safe, giving a stronger 3D effect."
                        ),
                    },
                ),
            },
            "optional": {
                "depth_map": (
                    "DEPTH_MAP",
                    {
                        "tooltip": (
                            "Optional depth map controlling the 3D shape. "
                            "White = near (pops forward), black = far. "
                            "If not connected, a centred radial gradient is used "
                            "(sphere appearing to float forward)."
                        )
                    },
                ),
                "error_correction": (
                    ["L", "M", "Q", "H"],
                    {
                        "tooltip": (
                            "QR error-correction level. Higher levels recover more data "
                            "if depth offsets distort the pattern. "
                            "'H' (30% recoverable) recommended when depth_factor > 0.10. "
                            "'M' (15%) is the minimum safe level for subtle depth. "
                            "Note: higher levels add more modules, increasing image size."
                        )
                    },
                ),
                "module_size": (
                    "INT",
                    {
                        "default": 10,
                        "min": 4,
                        "max": 32,
                        "tooltip": (
                            "Size of each QR module (square) in pixels. "
                            "Larger modules allow bigger depth shifts while staying scannable — "
                            "safe_mode caps shifts at module_size/4. "
                            "Safe range: 8–16 px. 10 px balances scan reliability and depth."
                        ),
                    },
                ),
                "border_modules": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 10,
                        "tooltip": (
                            "Quiet-zone width around the QR in modules. "
                            "The QR spec requires at least 4 for reliable scanning. "
                            "Reduce to 2 only when space is critical and you can test the scanner."
                        ),
                    },
                ),
                "depth_factor": (
                    "FLOAT",
                    {
                        "default": 0.15,
                        "min": 0.0,
                        "max": 0.40,
                        "step": 0.01,
                        "tooltip": (
                            "Stereogram depth strength as a fraction of the QR tile width. "
                            "0.05 = very subtle; 0.15 = visible safe depth; "
                            "0.25 = strong (test scannability); above 0.3 = artistic only. "
                            "With safe_mode on, this is automatically clamped to module_size/4 ÷ QR_width."
                        ),
                    },
                ),
                "n_copies": (
                    "INT",
                    {
                        "default": 4,
                        "min": 2,
                        "max": 8,
                        "tooltip": (
                            "Number of QR-width columns placed side by side. "
                            "The leftmost copy is always the clean, unshifted QR (point your scanner here). "
                            "Minimum 3 for comfortable stereogram cross-fusion. "
                            "4 is ideal — more copies widen the image."
                        ),
                    },
                ),
                "safe_mode": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Clamp depth_factor so pixel shifts stay ≤ module_size/4, "
                            "maximising QR scan reliability at the cost of shallower depth. "
                            "A console warning is always printed when shifts risk reducing scan rate. "
                            "Disable only after confirming the output scans reliably."
                        ),
                    },
                ),
                "invert_depth": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Flip the depth map (black = near, white = far). "
                            "Enable if the 3D shape appears to recede into the page "
                            "instead of floating forward when cross-fusing."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("qr_stereogram", "qr_only", "scan_info")
    OUTPUT_NODE = True

    def execute(
        self,
        url_or_text,
        depth_map=None,
        error_correction="H",
        module_size=10,
        border_modules=4,
        depth_factor=0.15,
        n_copies=4,
        safe_mode=True,
        invert_depth=False,
    ):
        try:
            import qrcode
            import qrcode.constants
        except ImportError:
            raise ImportError(
                "DF_QRStereogram requires the qrcode package.\n"
                "Install with:  pip install qrcode[pil]"
            )

        ec_map = {
            "L": qrcode.constants.ERROR_CORRECT_L,
            "M": qrcode.constants.ERROR_CORRECT_M,
            "Q": qrcode.constants.ERROR_CORRECT_Q,
            "H": qrcode.constants.ERROR_CORRECT_H,
        }
        ec_pct = {"L": "7%", "M": "15%", "Q": "25%", "H": "30%"}

        # --- 1. Generate QR code ---
        qr = qrcode.QRCode(
            error_correction=ec_map[error_correction],
            box_size=module_size,
            border=border_modules,
        )
        qr.add_data(url_or_text)
        qr.make(fit=True)

        # Render to PIL → numpy RGBA
        from PIL import Image as _PIL

        qr_raw = qr.make_image(fill_color="black", back_color="white")
        pil_img = qr_raw.get_image() if hasattr(qr_raw, "get_image") else qr_raw
        qr_arr = np.array(pil_img.convert("RGBA"), dtype=np.uint8)  # (H, W, 4)
        QH, QW = qr_arr.shape[:2]

        # --- 2. Prepare depth map ---
        if depth_map is not None:
            raw_depth = _depth_to_numpy(depth_map)
            # Resize depth to match QR tile dimensions
            depth_uint8 = (np.clip(raw_depth, 0, 1) * 255).astype(np.uint8)
            depth = (
                np.array(_PIL.fromarray(depth_uint8).resize((QW, QH), _PIL.BILINEAR)).astype(
                    np.float32
                )
                / 255.0
            )
        else:
            # Default: centred radial gradient — sphere floating forward
            ys = np.linspace(-1.0, 1.0, QH)
            xs = np.linspace(-1.0, 1.0, QW)
            xx, yy = np.meshgrid(xs, ys)
            depth = np.clip(1.0 - (xx**2 + yy**2), 0.0, 1.0).astype(np.float32)

        if invert_depth:
            depth = 1.0 - depth

        # --- 3. Compute shift budget and apply safe_mode ---
        max_shift_px = max(1, int(QW * depth_factor))
        effective_depth_factor = depth_factor

        if safe_mode:
            safe_max = max(1, module_size // 4)
            if max_shift_px > safe_max:
                import warnings

                effective_depth_factor = safe_max / QW
                warnings.warn(
                    f"DF_QRStereogram safe_mode: depth_factor {depth_factor:.3f} → "
                    f"{effective_depth_factor:.4f} "
                    f"(shift capped at {safe_max} px = module_size/4). "
                    f"Use error_correction='H' and disable safe_mode for stronger depth.",
                    UserWarning,
                    stacklevel=2,
                )
                max_shift_px = safe_max

        # --- 4. Build stereogram ---
        stereo = _make_qr_stereogram_image(qr_arr, depth, max_shift_px, n_copies)

        # --- 5. Build scan_info ---
        scan_info = (
            f"QR Version {qr.version}  ·  "
            f"{qr.modules_count}×{qr.modules_count} modules  ·  "
            f"Module size: {module_size} px  ·  "
            f"Error correction: {error_correction} ({ec_pct[error_correction]} recoverable)\n"
            f"QR tile: {QW}×{QH} px  ·  "
            f"Stereogram: {stereo.shape[1]}×{stereo.shape[0]} px  ·  "
            f"Max depth shift: {max_shift_px} px  ·  "
            f"depth_factor used: {effective_depth_factor:.4f}\n"
            f"Scan tip: Point your QR reader at the leftmost {QW}×{QH} px region.\n"
            f"View tip: Cross-fuse or diverge until the {n_copies} copies merge — "
            f"a 3D shape will emerge from the dot pattern."
        )

        return (_numpy_to_tensor(stereo), _numpy_to_tensor(qr_arr), scan_info)


# ===========================================================================
# ComfyUI registration
# ===========================================================================

NODE_CLASS_MAPPINGS = {
    "DF_DepthPrep": DF_DepthPrep,
    "DF_PatternGen": DF_PatternGen,
    "DF_PatternLibrary": DF_PatternLibrary,
    "DF_Stereogram": DF_Stereogram,
    "DF_AnaglyphOut": DF_AnaglyphOut,
    "DF_StereoPair": DF_StereoPair,
    "DF_HiddenImage": DF_HiddenImage,
    "DF_SafetyLimiter": DF_SafetyLimiter,
    "DF_QCOverlay": DF_QCOverlay,
    "DF_VideoSequence": DF_VideoSequence,
    "DF_QRStereogram": DF_QRStereogram,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DF_DepthPrep": "DepthForge Depth Prep",
    "DF_PatternGen": "DepthForge Pattern Gen",
    "DF_PatternLibrary": "DepthForge Pattern Library",
    "DF_Stereogram": "DepthForge Stereogram",
    "DF_AnaglyphOut": "DepthForge Anaglyph Out",
    "DF_StereoPair": "DepthForge Stereo Pair",
    "DF_HiddenImage": "DepthForge Hidden Image",
    "DF_SafetyLimiter": "DepthForge Safety Limiter",
    "DF_QCOverlay": "DepthForge QC Overlay",
    "DF_VideoSequence": "DepthForge Video Sequence",
    "DF_QRStereogram": "DepthForge QR Stereogram",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
