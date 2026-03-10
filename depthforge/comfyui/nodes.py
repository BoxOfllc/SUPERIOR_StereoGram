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
from typing import Any, Optional, Tuple


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
            "linear":      FalloffCurve.LINEAR,
            "gamma":       FalloffCurve.GAMMA,
            "s_curve":     FalloffCurve.S_CURVE,
            "logarithmic": FalloffCurve.LOGARITHMIC,
            "exponential": FalloffCurve.EXPONENTIAL,
        }


# ===========================================================================
# Node 1 — DF_DepthPrep
# ===========================================================================

class DF_DepthPrep(_DFNodeBase):
    """Condition a raw depth map through the 7-stage DepthForge pipeline."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth_image": ("IMAGE",),
            },
            "optional": {
                "invert":               ("BOOLEAN", {"default": False}),
                "bilateral_space":      ("FLOAT",   {"default": 5.0,  "min": 0.0,  "max": 30.0, "step": 0.5}),
                "bilateral_color":      ("FLOAT",   {"default": 0.1,  "min": 0.01, "max": 1.0,  "step": 0.01}),
                "dilation_px":          ("INT",     {"default": 3,    "min": 0,    "max": 20}),
                "falloff_curve":        (["linear", "gamma", "s_curve", "logarithmic", "exponential"],),
                "near_plane":           ("FLOAT",   {"default": 0.0,  "min": 0.0,  "max": 1.0,  "step": 0.01}),
                "far_plane":            ("FLOAT",   {"default": 1.0,  "min": 0.0,  "max": 1.0,  "step": 0.01}),
            },
        }

    RETURN_TYPES  = ("DEPTH_MAP", "IMAGE")
    RETURN_NAMES  = ("depth_map", "depth_preview")
    OUTPUT_NODE   = False

    def execute(self, depth_image, invert=False, bilateral_space=5.0,
                bilateral_color=0.1, dilation_px=3, falloff_curve="linear",
                near_plane=0.0, far_plane=1.0):
        from depthforge.core.depth_prep import prep_depth, DepthPrepParams

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

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pattern_type": (["random_noise", "perlin", "plasma", "voronoi",
                                   "geometric_grid", "mandelbrot", "dot_matrix"],),
                "color_mode":   (["greyscale", "monochrome", "psychedelic"],),
                "tile_width":   ("INT",   {"default": 128, "min": 32, "max": 512}),
                "tile_height":  ("INT",   {"default": 128, "min": 32, "max": 512}),
                "seed":         ("INT",   {"default": 42,  "min": 0, "max": 99999}),
                "scale":        ("FLOAT", {"default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("PATTERN", "IMAGE")
    RETURN_NAMES = ("pattern", "pattern_preview")

    def execute(self, pattern_type, color_mode, tile_width, tile_height, seed, scale):
        from depthforge.core.pattern_gen import generate_pattern, PatternParams, PatternType, ColorMode

        type_map = {
            "random_noise":    PatternType.RANDOM_NOISE,
            "perlin":          PatternType.PERLIN,
            "plasma":          PatternType.PLASMA,
            "voronoi":         PatternType.VORONOI,
            "geometric_grid":  PatternType.GEOMETRIC_GRID,
            "mandelbrot":      PatternType.MANDELBROT,
            "dot_matrix":      PatternType.DOT_MATRIX,
        }
        color_map = {
            "greyscale":   ColorMode.GREYSCALE,
            "monochrome":  ColorMode.MONOCHROME,
            "psychedelic": ColorMode.PSYCHEDELIC,
        }
        pattern = generate_pattern(PatternParams(
            pattern_type=type_map[pattern_type],
            tile_width=tile_width,
            tile_height=tile_height,
            color_mode=color_map[color_mode],
            seed=seed,
            scale=scale,
        ))
        preview = _numpy_to_tensor(pattern)
        return (pattern, preview)


# ===========================================================================
# Node 3 — DF_PatternLibrary
# ===========================================================================

class DF_PatternLibrary(_DFNodeBase):
    """Access the DepthForge named pattern library (28+ patterns)."""

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
                "pattern_name": (names,),
                "width":        ("INT",   {"default": 128, "min": 32, "max": 512}),
                "height":       ("INT",   {"default": 128, "min": 32, "max": 512}),
                "seed":         ("INT",   {"default": 42,  "min": 0,  "max": 99999}),
                "scale":        ("FLOAT", {"default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("PATTERN", "IMAGE", "STRING")
    RETURN_NAMES = ("pattern", "pattern_preview", "pattern_info")

    def execute(self, pattern_name, width, height, seed, scale):
        from depthforge.core.pattern_library import get_pattern, get_pattern_info
        pattern = get_pattern(pattern_name, width=width, height=height, seed=seed, scale=scale)
        info    = get_pattern_info(pattern_name)
        info_str = f"{pattern_name} [{info.category}]: {info.description}"
        preview = _numpy_to_tensor(pattern)
        return (pattern, preview, info_str)


# ===========================================================================
# Node 4 — DF_Stereogram (core synthesis)
# ===========================================================================

class DF_Stereogram(_DFNodeBase):
    """Synthesize a stereogram from a depth map and pattern tile."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth_map":    ("DEPTH_MAP",),
                "pattern":      ("PATTERN",),
            },
            "optional": {
                "depth_factor":       ("FLOAT",   {"default": 0.35, "min": -1.0, "max": 1.0,  "step": 0.01}),
                "max_parallax":       ("FLOAT",   {"default": 0.033,"min": 0.005,"max": 0.1,  "step": 0.001}),
                "oversample":         ("INT",     {"default": 1,    "min": 1,    "max": 4}),
                "safe_mode":          ("BOOLEAN", {"default": False}),
                "seed":               ("INT",     {"default": 42,   "min": 0,    "max": 99999}),
                "invert_depth":       ("BOOLEAN", {"default": False}),
                "preset_name":        (["none", "shallow", "medium", "deep", "cinema", "print", "broadcast"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("stereogram",)
    OUTPUT_NODE  = True

    def execute(self, depth_map, pattern, depth_factor=0.35, max_parallax=0.033,
                oversample=1, safe_mode=False, seed=42, invert_depth=False,
                preset_name="none"):
        from depthforge.core.synthesizer import synthesize, StereoParams

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

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "depth_map":    ("DEPTH_MAP",),
            },
            "optional": {
                "mode":           (["true", "grey", "colour", "half_colour", "optimised"],),
                "depth_factor":   ("FLOAT",   {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "parallax_px":    ("INT",     {"default": 30,   "min": 1,   "max": 200}),
                "swap_eyes":      ("BOOLEAN", {"default": False}),
                "gamma":          ("FLOAT",   {"default": 1.0,  "min": 0.5, "max": 3.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("anaglyph",)
    OUTPUT_NODE  = True

    def execute(self, source_image, depth_map, mode="optimised", depth_factor=0.35,
                parallax_px=30, swap_eyes=False, gamma=1.0):
        from depthforge.core.anaglyph import make_anaglyph_from_depth, AnaglyphParams, AnaglyphMode

        mode_map = {
            "true":        AnaglyphMode.TRUE_ANAGLYPH,
            "grey":        AnaglyphMode.GREY_ANAGLYPH,
            "colour":      AnaglyphMode.COLOUR_ANAGLYPH,
            "half_colour": AnaglyphMode.HALF_COLOUR,
            "optimised":   AnaglyphMode.OPTIMISED,
        }
        src   = _tensor_to_numpy(source_image)
        depth = _depth_to_numpy(depth_map)
        W     = src.shape[1]
        px    = int(W * 0.065 * depth_factor)

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

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "depth_map":    ("DEPTH_MAP",),
            },
            "optional": {
                "max_parallax_fraction": ("FLOAT", {"default": 0.033, "min": 0.005, "max": 0.1, "step": 0.001}),
                "eye_balance":           ("FLOAT", {"default": 0.5,   "min": 0.0,   "max": 1.0, "step": 0.1}),
                "background_fill":       (["edge", "mirror", "black"],),
                "feather_px":            ("INT",   {"default": 3, "min": 0, "max": 20}),
                "layout":                (["separate", "side_by_side", "top_bottom"],),
                "gap_px":                ("INT",   {"default": 0, "min": 0, "max": 64}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("left_view", "right_view", "occlusion_mask", "composed")
    OUTPUT_NODE  = True

    def execute(self, source_image, depth_map, max_parallax_fraction=0.033,
                eye_balance=0.5, background_fill="edge", feather_px=3,
                layout="separate", gap_px=0):
        from depthforge.core.stereo_pair import (
            make_stereo_pair, compose_side_by_side, StereoPairParams, StereoLayout
        )

        src   = _tensor_to_numpy(source_image)
        depth = _depth_to_numpy(depth_map)

        layout_map = {
            "separate":    StereoLayout.SEPARATE,
            "side_by_side": StereoLayout.SIDE_BY_SIDE,
            "top_bottom":  StereoLayout.TOP_BOTTOM,
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
            composed = np.concatenate([left, gap_arr, right] if gap_arr is not None else [left, right], axis=0)
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

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pattern":  ("PATTERN",),
                "width":    ("INT",   {"default": 1920, "min": 256,  "max": 4096}),
                "height":   ("INT",   {"default": 1080, "min": 128,  "max": 2160}),
            },
            "optional": {
                "shape":            (["none", "circle", "square", "triangle", "star", "diamond", "arrow"],),
                "text":             ("STRING",  {"default": ""}),
                "font_size":        ("INT",     {"default": 180, "min": 20,  "max": 600}),
                "foreground_depth": ("FLOAT",   {"default": 0.75,"min": 0.1, "max": 1.0, "step": 0.05}),
                "background_depth": ("FLOAT",   {"default": 0.10,"min": 0.0, "max": 0.9, "step": 0.05}),
                "edge_soften_px":   ("INT",     {"default": 4,   "min": 0,   "max": 20}),
                "mask_image":       ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("hidden_stereogram", "depth_preview")
    OUTPUT_NODE  = True

    def execute(self, pattern, width, height, shape="none", text="",
                font_size=180, foreground_depth=0.75, background_depth=0.10,
                edge_soften_px=4, mask_image=None):
        from depthforge.core.hidden_image import (
            encode_hidden_image, shape_to_mask, text_to_mask,
            HiddenImageParams,
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
        if hasattr(msk, 'shape') and msk.ndim == 2:
            depth_vis = (msk * 255).astype(np.uint8)
            depth_rgba = np.stack([depth_vis, depth_vis, depth_vis,
                                   np.full_like(depth_vis, 255)], axis=-1)
        else:
            depth_rgba = np.full((height, width, 4), 128, dtype=np.uint8)

        return (_numpy_to_tensor(result), _numpy_to_tensor(depth_rgba))


# ===========================================================================
# Node 8 — DF_SafetyLimiter
# ===========================================================================

class DF_SafetyLimiter(_DFNodeBase):
    """Apply comfort safety limits to a depth map and stereo parameters."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth_map": ("DEPTH_MAP",),
            },
            "optional": {
                "profile":             (["conservative", "standard", "relaxed", "cinema"],),
                "max_depth_factor":    ("FLOAT",   {"default": 0.6,  "min": 0.1, "max": 1.0, "step": 0.05}),
                "max_gradient":        ("FLOAT",   {"default": 0.15, "min": 0.01,"max": 0.5, "step": 0.01}),
                "near_clip":           ("FLOAT",   {"default": 0.02, "min": 0.0, "max": 0.2, "step": 0.01}),
                "far_clip":            ("FLOAT",   {"default": 0.98, "min": 0.8, "max": 1.0, "step": 0.01}),
                "warn_on_violation":   ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("DEPTH_MAP", "IMAGE", "STRING")
    RETURN_NAMES = ("safe_depth", "depth_preview", "violation_report")

    def execute(self, depth_map, profile="standard", max_depth_factor=0.6,
                max_gradient=0.15, near_clip=0.02, far_clip=0.98,
                warn_on_violation=True):
        from depthforge.core.comfort import SafetyLimiter, SafetyLimiterParams, VergenceProfile

        profile_map = {
            "conservative": VergenceProfile.CONSERVATIVE,
            "standard":     VergenceProfile.STANDARD,
            "relaxed":      VergenceProfile.RELAXED,
            "cinema":       VergenceProfile.CINEMA,
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

        report = "\n".join(f"[{k}] {v}" for k, v in violations.items()) if violations else "✓ No violations"
        preview = _numpy_depth_to_tensor(safe_depth)
        return (safe_depth, preview, report)


# ===========================================================================
# Node 9 — DF_QCOverlay
# ===========================================================================

class DF_QCOverlay(_DFNodeBase):
    """Generate QC visualisation overlays for depth maps."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth_map":   ("DEPTH_MAP",),
            },
            "optional": {
                "overlay_type":   (["parallax_heatmap", "depth_bands", "violation_overlay",
                                    "safe_zone", "histogram"],),
                "frame_width":    ("INT",   {"default": 1920, "min": 128, "max": 7680}),
                "depth_factor":   ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_parallax":   ("FLOAT", {"default": 0.033,"min": 0.005,"max": 0.1, "step": 0.001}),
                "n_bands":        ("INT",   {"default": 8,    "min": 2,   "max": 16}),
                "colormap":       (["inferno", "jet", "viridis", "plasma", "turbo"],),
                "annotate":       ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("qc_image", "comfort_summary")
    OUTPUT_NODE  = True

    def execute(self, depth_map, overlay_type="parallax_heatmap", frame_width=1920,
                depth_factor=0.35, max_parallax=0.033, n_bands=8,
                colormap="inferno", annotate=True):
        from depthforge.core.qc import (
            parallax_heatmap, depth_band_preview, window_violation_overlay,
            safe_zone_indicator, depth_histogram, QCParams
        )
        from depthforge.core.comfort import ComfortAnalyzer, ComfortAnalyzerParams

        depth = _depth_to_numpy(depth_map)
        H, W  = depth.shape
        qp    = QCParams(frame_width=frame_width, depth_factor=depth_factor,
                         max_parallax_fraction=max_parallax, colormap=colormap)

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
        analyzer = ComfortAnalyzer(ComfortAnalyzerParams(
            frame_width=frame_width,
            depth_factor=depth_factor,
            max_parallax_fraction=max_parallax,
        ))
        report  = analyzer.analyze(depth)
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

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth_sequence": ("IMAGE",),   # batch of depth frames
                "pattern":        ("PATTERN",),
            },
            "optional": {
                "depth_factor":        ("FLOAT",   {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_parallax":        ("FLOAT",   {"default": 0.033,"min": 0.005,"max": 0.1, "step": 0.001}),
                "temporal_smooth":     ("FLOAT",   {"default": 0.3,  "min": 0.0, "max": 1.0, "step": 0.05}),
                "safe_mode":           ("BOOLEAN", {"default": True}),
                "preset_name":         (["none", "shallow", "medium", "deep", "cinema", "broadcast"],),
                "seed":                ("INT",     {"default": 42, "min": 0, "max": 99999}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("stereogram_sequence",)
    OUTPUT_NODE  = True

    def execute(self, depth_sequence, pattern, depth_factor=0.35, max_parallax=0.033,
                temporal_smooth=0.3, safe_mode=True, preset_name="none", seed=42):
        from depthforge.core.synthesizer import synthesize, StereoParams
        from depthforge.core.depth_prep  import prep_depth, DepthPrepParams

        if preset_name != "none":
            from depthforge.core.presets import get_preset
            p = get_preset(preset_name)
            sp = p.stereo_params
            dp = p.depth_params
        else:
            sp = StereoParams(depth_factor=depth_factor, max_parallax_fraction=max_parallax,
                              safe_mode=safe_mode, seed=seed)
            dp = DepthPrepParams()

        # Handle both batched tensor and single frame
        try:
            import torch
            if isinstance(depth_sequence, torch.Tensor):
                frames_np = depth_sequence.cpu().numpy()   # (B, H, W, C)
            else:
                frames_np = np.array(depth_sequence)
        except ImportError:
            frames_np = np.array(depth_sequence)

        if frames_np.ndim == 3:
            frames_np = frames_np[None]   # add batch dim

        n_frames = frames_np.shape[0]
        results  = []
        prev_depth = None

        for i in range(n_frames):
            frame = frames_np[i]
            if frame.shape[-1] > 1:
                raw_depth = frame.mean(axis=-1).astype(np.float32)
            else:
                raw_depth = frame[:, :, 0].astype(np.float32)

            # Temporal smoothing — blend with previous frame's depth
            if prev_depth is not None and temporal_smooth > 0:
                raw_depth = (raw_depth * (1 - temporal_smooth) +
                             prev_depth * temporal_smooth)
            prev_depth = raw_depth.copy()

            depth = prep_depth(raw_depth, dp)

            # Use per-frame seed offset for stable dot patterns
            sp_frame     = StereoParams(**{k: v for k, v in sp.__dict__.items()})
            sp_frame.seed = sp.seed + i if sp.seed is not None else None

            stereo = synthesize(depth, pattern, sp_frame)
            results.append(stereo)

        # Stack into batch tensor
        batch = np.stack(results, axis=0).astype(np.float32) / 255.0   # (B,H,W,4)
        try:
            import torch
            return (torch.from_numpy(batch),)
        except ImportError:
            return (batch,)
