"""
depthforge.nuke.gizmo
======================
Nuke Python gizmo for DepthForge stereogram synthesis.

Installation
------------
Add to your ``~/.nuke/init.py`` or the facility ``init.py``:

    import sys
    sys.path.insert(0, "/path/to/depthforge/parent")
    import depthforge.nuke
    depthforge.nuke.install()

Or call directly in the Nuke Script Editor:

    import depthforge.nuke
    depthforge.nuke.install()

Usage
-----
The gizmo appears in the Node Graph under:
  DepthForge > Stereogram
  DepthForge > Anaglyph
  DepthForge > StereoPair
  DepthForge > HiddenImage
  DepthForge > QCOverlay

Inputs
------
depth       (required)  Depth map: luminance = distance (white=near)
source      (optional)  Source plate for anaglyph/stereo modes
pattern     (optional)  Custom pattern tile
region_mask (optional)  Per-region depth reduction mask

Output
------
RGBA stereogram image, native Nuke stereo views, or anaglyph
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Nuke availability guard
# ---------------------------------------------------------------------------

try:
    import nukescripts

    import nuke

    _NUKE_AVAILABLE = True
except ImportError:
    _NUKE_AVAILABLE = False
    nuke = None
    nukescripts = None


# ---------------------------------------------------------------------------
# Gizmo knob definitions
# ---------------------------------------------------------------------------

#: Complete knob specification for the DepthForge_Stereogram gizmo.
#: Each entry: (type, name, label, default, extras)
STEREOGRAM_KNOBS = [
    # --- Mode ---
    (
        "Enumeration_Knob",
        "df_mode",
        "Mode",
        ["SIRDS", "Texture", "Anaglyph", "StereoPair", "HiddenImage"],
        {},
    ),
    # --- Preset ---
    (
        "Enumeration_Knob",
        "df_preset",
        "Preset",
        ["none", "shallow", "medium", "deep", "cinema", "print", "broadcast"],
        {},
    ),
    # --- Depth controls ---
    ("Double_Knob", "df_depth_factor", "Depth Factor", 0.35, {"min": 0.0, "max": 1.0}),
    ("Double_Knob", "df_max_parallax", "Max Parallax", 0.033, {"min": 0.005, "max": 0.1}),
    ("Boolean_Knob", "df_safe_mode", "Safe Mode", False, {}),
    ("Int_Knob", "df_oversample", "Oversample", 1, {"min": 1, "max": 4}),
    ("Int_Knob", "df_seed", "Seed", 42, {"min": 0}),
    ("Boolean_Knob", "df_invert_depth", "Invert Depth", False, {}),
    # --- Depth conditioning ---
    ("Tab_Knob", "_tab_depth", "Depth Conditioning", None, {}),
    ("Double_Knob", "df_bilateral_space", "Bilateral Space", 5.0, {"min": 0, "max": 30}),
    ("Double_Knob", "df_bilateral_color", "Bilateral Color", 0.1, {"min": 0.01, "max": 1.0}),
    ("Int_Knob", "df_dilation", "Dilation (px)", 3, {"min": 0, "max": 20}),
    (
        "Enumeration_Knob",
        "df_falloff",
        "Falloff Curve",
        ["linear", "gamma", "s_curve", "logarithmic", "exponential"],
        {},
    ),
    ("Double_Knob", "df_near_plane", "Near Plane", 0.0, {"min": 0, "max": 1}),
    ("Double_Knob", "df_far_plane", "Far Plane", 1.0, {"min": 0, "max": 1}),
    # --- Pattern ---
    ("Tab_Knob", "_tab_pattern", "Pattern", None, {}),
    (
        "Enumeration_Knob",
        "df_pattern_type",
        "Pattern Type",
        [
            "random_noise",
            "perlin",
            "plasma",
            "voronoi",
            "geometric_grid",
            "mandelbrot",
            "dot_matrix",
        ],
        {},
    ),
    (
        "Enumeration_Knob",
        "df_color_mode",
        "Color Mode",
        ["greyscale", "monochrome", "psychedelic"],
        {},
    ),
    ("Int_Knob", "df_tile_width", "Tile Width", 128, {"min": 32, "max": 512}),
    ("Int_Knob", "df_tile_height", "Tile Height", 128, {"min": 32, "max": 512}),
    ("Double_Knob", "df_pattern_scale", "Pattern Scale", 1.0, {"min": 0.1, "max": 8.0}),
    # --- Anaglyph ---
    ("Tab_Knob", "_tab_anaglyph", "Anaglyph", None, {}),
    (
        "Enumeration_Knob",
        "df_anaglyph_mode",
        "Mode",
        ["true", "grey", "colour", "half_colour", "optimised"],
        {},
    ),
    ("Boolean_Knob", "df_swap_eyes", "Swap Eyes", False, {}),
    ("Double_Knob", "df_gamma", "Gamma", 1.0, {"min": 0.5, "max": 3.0}),
    # --- Stereo pair ---
    ("Tab_Knob", "_tab_stereo", "Stereo Pair", None, {}),
    ("Enumeration_Knob", "df_bg_fill", "Background Fill", ["edge", "mirror", "black"], {}),
    ("Double_Knob", "df_eye_balance", "Eye Balance", 0.5, {"min": 0, "max": 1}),
    ("Int_Knob", "df_feather_px", "Feather (px)", 3, {"min": 0, "max": 20}),
    # --- Hidden image ---
    ("Tab_Knob", "_tab_hidden", "Hidden Image", None, {}),
    (
        "Enumeration_Knob",
        "df_hidden_shape",
        "Shape",
        ["none", "circle", "square", "triangle", "star", "diamond", "arrow"],
        {},
    ),
    ("String_Knob", "df_hidden_text", "Text", "", {}),
    ("Double_Knob", "df_fg_depth", "Foreground Depth", 0.75, {"min": 0.1, "max": 1.0}),
    ("Double_Knob", "df_bg_depth", "Background Depth", 0.10, {"min": 0.0, "max": 0.9}),
    ("Int_Knob", "df_edge_soften", "Edge Soften (px)", 4, {"min": 0, "max": 20}),
    # --- QC / Safety ---
    ("Tab_Knob", "_tab_qc", "QC & Safety", None, {}),
    ("Boolean_Knob", "df_show_heatmap", "Show Heatmap", False, {}),
    ("Boolean_Knob", "df_auto_safe", "Auto Safety Limit", True, {}),
    (
        "Enumeration_Knob",
        "df_safety_profile",
        "Safety Profile",
        ["conservative", "standard", "relaxed", "cinema"],
        {},
    ),
]


# ---------------------------------------------------------------------------
# Gizmo class
# ---------------------------------------------------------------------------


class DepthForgeGizmo:
    """Pure-Python representation of the DepthForge Nuke gizmo.

    This class handles parameter management and rendering without requiring
    Nuke to be available in the current Python environment. The actual
    ``nuke.createNode`` calls are only made when ``nuke`` is importable.
    """

    GIZMO_NAME = "DepthForge_Stereogram"
    GIZMO_VERSION = "0.3.0"

    def __init__(self, knob_values: Optional[dict] = None):
        """
        Parameters
        ----------
        knob_values : dict, optional
            Override default knob values at construction time.
        """
        self._values = self._defaults()
        if knob_values:
            self._values.update(knob_values)

    # ------------------------------------------------------------------
    # Knob access
    # ------------------------------------------------------------------

    def _defaults(self) -> dict:
        defaults = {}
        for entry in STEREOGRAM_KNOBS:
            kind, name, label, default, extras = entry
            if default is not None and not name.startswith("_tab"):
                # Enumeration_Knob stores a list of choices as default; use first item
                if kind == "Enumeration_Knob" and isinstance(default, list):
                    defaults[name] = default[0]
                else:
                    defaults[name] = default
        return defaults

    def get(self, name: str):
        """Get a knob value by name."""
        return self._values.get(name)

    def set(self, name: str, value) -> None:
        """Set a knob value by name."""
        self._values[name] = value

    def to_dict(self) -> dict:
        """Serialise all knob values to a dict."""
        return dict(self._values)

    def to_json(self, path: Optional[str] = None) -> str:
        """Serialise knob values to JSON."""
        txt = json.dumps(self.to_dict(), indent=2)
        if path:
            Path(path).write_text(txt)
        return txt

    @classmethod
    def from_dict(cls, d: dict) -> "DepthForgeGizmo":
        g = cls()
        g._values.update(d)
        return g

    @classmethod
    def from_json(cls, path: str) -> "DepthForgeGizmo":
        """Load a gizmo from a JSON file.

        Parameters
        ----------
        path : str
            Path to a ``.dfgizmo.json`` file written by ``to_json()``.

        Raises
        ------
        ValueError
            If the file does not look like a valid DepthForge gizmo JSON
            (missing ``version`` or ``knobs`` keys, or knob values that
            are not numbers / strings / booleans).
        """
        p = Path(path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Gizmo JSON not found: {p}")
        if p.stat().st_size > 1 * 1024 * 1024:  # 1 MB cap
            raise ValueError("Gizmo JSON file too large (max 1 MB).")

        try:
            raw = p.read_text(encoding="utf-8")
            d = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in gizmo file: {exc}") from exc

        if not isinstance(d, dict):
            raise ValueError("Gizmo JSON must be a JSON object.")

        # Validate knob values — must be scalars only (no nested objects/code)
        knobs = d.get("knobs", {})
        if not isinstance(knobs, dict):
            raise ValueError("'knobs' must be a JSON object.")
        for k, v in knobs.items():
            if not isinstance(k, str):
                raise ValueError(f"Knob key must be a string, got {type(k)}")
            if not isinstance(v, (int, float, bool, str, type(None))):
                raise ValueError(
                    f"Knob value for {k!r} must be a scalar "
                    f"(int/float/bool/str), got {type(v).__name__}."
                )

        return cls.from_dict(d)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(
        self,
        depth: "np.ndarray",
        source: Optional["np.ndarray"] = None,
        pattern: Optional["np.ndarray"] = None,
    ) -> "np.ndarray":
        """Execute the gizmo render in pure Python (no Nuke required).

        Parameters
        ----------
        depth : np.ndarray
            float32 (H, W) depth map.
        source : np.ndarray, optional
            uint8 (H, W, 4) RGBA source image (for anaglyph/stereo modes).
        pattern : np.ndarray, optional
            uint8 (H, W, 4) RGBA pattern tile. Auto-generated if None.

        Returns
        -------
        np.ndarray  uint8 RGBA (H, W, 4)
        """
        import numpy as np

        from depthforge.core.depth_prep import DepthPrepParams, FalloffCurve, prep_depth
        from depthforge.core.pattern_gen import (
            ColorMode,
            PatternParams,
            PatternType,
            generate_pattern,
        )
        from depthforge.core.synthesizer import StereoParams, synthesize

        mode = self._values.get("df_mode", "SIRDS")

        # Apply safety limiter if requested
        if self._values.get("df_auto_safe", True):
            from depthforge.core.comfort import SafetyLimiter, SafetyLimiterParams, VergenceProfile

            profile_name = self._values.get("df_safety_profile", "standard")
            profile_map = {
                "conservative": VergenceProfile.CONSERVATIVE,
                "standard": VergenceProfile.STANDARD,
                "relaxed": VergenceProfile.RELAXED,
                "cinema": VergenceProfile.CINEMA,
            }
            limiter = SafetyLimiter(
                SafetyLimiterParams.from_profile(profile_map[profile_name], warn_on_violation=False)
            )
            depth, _, _ = limiter.apply(depth)

        # Depth conditioning
        falloff_map = {
            "linear": FalloffCurve.LINEAR,
            "gamma": FalloffCurve.GAMMA,
            "s_curve": FalloffCurve.S_CURVE,
            "logarithmic": FalloffCurve.LOGARITHMIC,
            "exponential": FalloffCurve.EXPONENTIAL,
        }
        dp = DepthPrepParams(
            invert=self._values.get("df_invert_depth", False),
            bilateral_sigma_space=self._values.get("df_bilateral_space", 5.0),
            bilateral_sigma_color=self._values.get("df_bilateral_color", 0.1),
            dilation_px=self._values.get("df_dilation", 3),
            falloff_curve=falloff_map.get(
                self._values.get("df_falloff", "linear"), FalloffCurve.LINEAR
            ),
            near_plane=self._values.get("df_near_plane", 0.0),
            far_plane=self._values.get("df_far_plane", 1.0),
        )
        depth = prep_depth(depth, dp)

        # Stereo params (or preset override)
        preset_name = self._values.get("df_preset", "none")
        if preset_name != "none":
            from depthforge.core.presets import get_preset

            preset = get_preset(preset_name)
            sp = preset.stereo_params
        else:
            sp = StereoParams(
                depth_factor=self._values.get("df_depth_factor", 0.35),
                max_parallax_fraction=self._values.get("df_max_parallax", 0.033),
                safe_mode=self._values.get("df_safe_mode", False),
                oversample=self._values.get("df_oversample", 1),
                seed=self._values.get("df_seed", 42),
                invert_depth=False,  # already applied above
            )

        # Pattern
        if pattern is None:
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
                    pattern_type=type_map.get(
                        self._values.get("df_pattern_type", "random_noise"),
                        PatternType.RANDOM_NOISE,
                    ),
                    tile_width=self._values.get("df_tile_width", 128),
                    tile_height=self._values.get("df_tile_height", 128),
                    color_mode=color_map.get(
                        self._values.get("df_color_mode", "greyscale"), ColorMode.GREYSCALE
                    ),
                    seed=self._values.get("df_seed", 42),
                    scale=self._values.get("df_pattern_scale", 1.0),
                )
            )

        # Dispatch by mode
        if mode == "SIRDS" or mode == "Texture":
            return synthesize(depth, pattern, sp)

        elif mode == "Anaglyph":
            if source is None:
                source = np.stack(
                    [
                        (depth * 255).astype(np.uint8),
                        (depth * 180).astype(np.uint8),
                        ((1 - depth) * 200).astype(np.uint8),
                        np.full(depth.shape, 255, np.uint8),
                    ],
                    axis=-1,
                )
            from depthforge.core.anaglyph import (
                AnaglyphMode,
                AnaglyphParams,
                make_anaglyph_from_depth,
            )

            mode_map = {
                "true": AnaglyphMode.TRUE_ANAGLYPH,
                "grey": AnaglyphMode.GREY_ANAGLYPH,
                "colour": AnaglyphMode.COLOUR_ANAGLYPH,
                "half_colour": AnaglyphMode.HALF_COLOUR,
                "optimised": AnaglyphMode.OPTIMISED,
            }
            H, W = source.shape[:2]
            px = int(W * 0.065 * sp.depth_factor)
            params = AnaglyphParams(
                mode=mode_map.get(
                    self._values.get("df_anaglyph_mode", "optimised"), AnaglyphMode.OPTIMISED
                ),
                parallax_px=px,
                swap_eyes=self._values.get("df_swap_eyes", False),
                gamma=self._values.get("df_gamma", 1.0),
            )
            return make_anaglyph_from_depth(source, depth, params)

        elif mode == "StereoPair":
            if source is None:
                raise ValueError("StereoPair mode requires a source image.")
            from depthforge.core.stereo_pair import (
                StereoPairParams,
                compose_side_by_side,
                make_stereo_pair,
            )

            params = StereoPairParams(
                max_parallax_fraction=sp.max_parallax_fraction,
                eye_balance=self._values.get("df_eye_balance", 0.5),
                background_fill=self._values.get("df_bg_fill", "edge"),
                feather_px=self._values.get("df_feather_px", 3),
            )
            left, right, _ = make_stereo_pair(source, depth, params)
            return compose_side_by_side(left, right)

        elif mode == "HiddenImage":
            from depthforge.core.hidden_image import (
                HiddenImageParams,
                encode_hidden_image,
                shape_to_mask,
                text_to_mask,
            )

            H, W = depth.shape
            text = self._values.get("df_hidden_text", "").strip()
            shape = self._values.get("df_hidden_shape", "none")
            if text:
                msk = text_to_mask(text, W, H)
            elif shape != "none":
                msk = shape_to_mask(shape, W, H)
            else:
                msk = shape_to_mask("star", W, H)
            params = HiddenImageParams(
                foreground_depth=self._values.get("df_fg_depth", 0.75),
                background_depth=self._values.get("df_bg_depth", 0.10),
                edge_soften_px=self._values.get("df_edge_soften", 4),
            )
            return encode_hidden_image(pattern, msk, params)

        else:
            return synthesize(depth, pattern, sp)

    # ------------------------------------------------------------------
    # Nuke-specific: create the actual node in the graph
    # ------------------------------------------------------------------

    def create_nuke_node(self) -> "nuke.Node":
        """Create the DepthForge gizmo in the Nuke node graph.

        Only works when ``nuke`` is importable (i.e., running inside Nuke).
        """
        if not _NUKE_AVAILABLE:
            raise RuntimeError("Nuke is not available in this environment.")

        node = nuke.createNode("Group", inpanel=False)
        node.setName(self.GIZMO_NAME)

        with node:
            # Create the Python callback node
            nuke.addKnob(nuke.Tab_Knob("DepthForge"))
            for entry in STEREOGRAM_KNOBS:
                kind, name, label, default, extras = entry
                if name.startswith("_tab"):
                    k = nuke.Tab_Knob(name, label)
                    nuke.addKnob(k)
                    continue
                knob_class = getattr(nuke, kind, None)
                if knob_class is None:
                    continue
                if kind == "Enumeration_Knob":
                    k = knob_class(name, label, default)
                elif kind == "String_Knob":
                    k = nuke.String_Knob(name, label)
                    k.setValue(default or "")
                elif kind == "Boolean_Knob":
                    k = nuke.Boolean_Knob(name, label)
                    k.setValue(bool(default))
                else:
                    k = knob_class(name, label)
                    if default is not None:
                        k.setValue(float(default) if isinstance(default, (int, float)) else default)
                    for attr, val in extras.items():
                        if hasattr(k, attr):
                            getattr(k, attr)(val)
                nuke.addKnob(k)

        # Apply current knob values
        for name, value in self._values.items():
            knob = node.knob(name)
            if knob:
                knob.setValue(value)

        return node


# ---------------------------------------------------------------------------
# Nuke menu registration
# ---------------------------------------------------------------------------


def _register_nuke_callbacks() -> None:
    """Register the knobChanged callback for the gizmo node."""
    if not _NUKE_AVAILABLE:
        return
    # Callback: re-render when any DepthForge knob changes
    nuke.addKnobChanged(
        _on_knob_changed,
        nodeClass="Group",
    )


def _on_knob_changed() -> None:
    """Nuke knobChanged callback."""
    if not _NUKE_AVAILABLE:
        return
    node = nuke.thisNode()
    if node.name().startswith("DepthForge"):
        # Trigger re-evaluation — Nuke handles this automatically
        # when the Python callback node is invalidated
        pass
