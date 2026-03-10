"""
depthforge.core.presets
=======================
Named preset profiles for stereogram synthesis and depth conditioning.

Usage
-----
    from depthforge.core.presets import get_preset, list_presets, Preset

    preset = get_preset("cinema")
    result = synthesize(depth, pattern, preset.stereo_params)
    depth  = prep_depth(raw_depth, preset.depth_params)

Presets
-------
    shallow     Subtle depth — comfortable for all viewers, print-safe
    medium      Standard depth — good balance of pop and comfort
    deep        Strong depth — high-impact, experienced viewers
    cinema      Broadcast/cinema delivery — DCI/Rec709 safe parallax limits
    print       High-resolution print output — 300dpi, CMYK-safe depth
    broadcast   TV/streaming delivery — conservative parallax, legal levels
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

from depthforge.core.synthesizer import StereoParams
from depthforge.core.depth_prep import DepthPrepParams, FalloffCurve


# ---------------------------------------------------------------------------
# Preset dataclass
# ---------------------------------------------------------------------------

@dataclass
class Preset:
    """A named configuration bundle for stereogram synthesis.

    Attributes
    ----------
    name : str
        Unique identifier (e.g. "cinema")
    display_name : str
        Human-readable label
    description : str
        Short description of intent and use case
    stereo_params : StereoParams
        Parameters for the synthesizer
    depth_params : DepthPrepParams
        Parameters for depth conditioning
    output_dpi : int
        Recommended output DPI (72 for web, 300 for print)
    color_space : str
        Recommended color space ("sRGB", "Rec.709", "P3", "CMYK-safe")
    max_parallax_px : Optional[int]
        Absolute maximum parallax in pixels at this preset's target resolution.
        None = derive from max_parallax_fraction + target_width.
    target_width : int
        Reference output width used to compute comfort limits
    notes : str
        Additional delivery or usage notes
    """
    name: str
    display_name: str
    description: str
    stereo_params: StereoParams
    depth_params: DepthPrepParams
    output_dpi: int = 72
    color_space: str = "sRGB"
    max_parallax_px: Optional[int] = None
    target_width: int = 1920
    notes: str = ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def comfort_limit_px(self, width: int) -> int:
        """Return absolute pixel comfort limit for a given output width."""
        fraction = self.stereo_params.max_parallax_fraction
        return int(width * fraction)

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dict."""
        d = asdict(self)
        # FalloffCurve → string
        dc = d["depth_params"]
        # FalloffCurve enum → int for JSON serialization
        if hasattr(dc["falloff_curve"], "value"):
            dc["falloff_curve"] = dc["falloff_curve"].value
        return d

    def to_json(self, path: Optional[str] = None) -> str:
        """Serialise to JSON string (and optionally write to file)."""
        txt = json.dumps(self.to_dict(), indent=2)
        if path:
            Path(path).write_text(txt)
        return txt

    @classmethod
    def from_dict(cls, d: dict) -> "Preset":
        sp = StereoParams(**d["stereo_params"])
        dp_raw = d["depth_params"].copy()
        dp_raw["falloff_curve"] = FalloffCurve(dp_raw["falloff_curve"])
        dp = DepthPrepParams(**dp_raw)
        return cls(
            name=d["name"],
            display_name=d["display_name"],
            description=d["description"],
            stereo_params=sp,
            depth_params=dp,
            output_dpi=d.get("output_dpi", 72),
            color_space=d.get("color_space", "sRGB"),
            max_parallax_px=d.get("max_parallax_px"),
            target_width=d.get("target_width", 1920),
            notes=d.get("notes", ""),
        )


# ---------------------------------------------------------------------------
# Built-in preset definitions
# ---------------------------------------------------------------------------

#: Subtle depth — safe for all audiences, first-time viewers
PRESET_SHALLOW = Preset(
    name="shallow",
    display_name="Shallow",
    description=(
        "Very subtle depth effect. Comfortable for all audiences including "
        "first-time stereogram viewers. Recommended for public-facing content."
    ),
    stereo_params=StereoParams(
        depth_factor=0.20,
        max_parallax_fraction=0.025,   # 2.5% — well within comfort zone
        eye_separation_fraction=0.065,
        convergence=0.0,
        safe_mode=True,
        oversample=1,
    ),
    depth_params=DepthPrepParams(
        bilateral_sigma_space=7.0,
        bilateral_sigma_color=0.12,
        dilation_px=4,
        falloff_curve=FalloffCurve.GAMMA,
        near_plane=0.05,
        far_plane=0.92,
    ),
    output_dpi=72,
    color_space="sRGB",
    notes="Use for social media, web galleries, and general audiences.",
)

#: Standard depth — good balance of impact and comfort
PRESET_MEDIUM = Preset(
    name="medium",
    display_name="Medium",
    description=(
        "Balanced depth effect. Strong enough for clear 3D perception "
        "while remaining comfortable for most viewers. General-purpose default."
    ),
    stereo_params=StereoParams(
        depth_factor=0.35,
        max_parallax_fraction=1/30,    # ~3.33% — classic comfort recommendation
        eye_separation_fraction=0.065,
        convergence=0.0,
        safe_mode=False,
        oversample=1,
    ),
    depth_params=DepthPrepParams(
        bilateral_sigma_space=5.0,
        bilateral_sigma_color=0.10,
        dilation_px=3,
        falloff_curve=FalloffCurve.LINEAR,
        near_plane=0.02,
        far_plane=0.98,
    ),
    output_dpi=72,
    color_space="sRGB",
    notes="Good starting point for most stereogram projects.",
)

#: Strong depth — high-impact for experienced viewers
PRESET_DEEP = Preset(
    name="deep",
    display_name="Deep",
    description=(
        "Strong depth effect for experienced stereogram viewers. "
        "May cause discomfort for some viewers — not recommended for public-facing content. "
        "Ideal for gallery prints where viewers can self-select viewing distance."
    ),
    stereo_params=StereoParams(
        depth_factor=0.55,
        max_parallax_fraction=0.05,   # 5% — pushing comfort limits
        eye_separation_fraction=0.065,
        convergence=0.0,
        safe_mode=False,
        oversample=2,                  # oversample for quality at large shifts
    ),
    depth_params=DepthPrepParams(
        bilateral_sigma_space=4.0,
        bilateral_sigma_color=0.08,
        dilation_px=5,
        falloff_curve=FalloffCurve.S_CURVE,
        near_plane=0.0,
        far_plane=1.0,
    ),
    output_dpi=72,
    color_space="sRGB",
    notes=(
        "Include a photosensitivity warning. Not suitable for general audiences. "
        "Oversample=2 increases render time but improves quality at high parallax."
    ),
)

#: Cinema / digital cinema delivery
PRESET_CINEMA = Preset(
    name="cinema",
    display_name="Cinema",
    description=(
        "Digital cinema delivery profile. Optimised for 2K/4K DCP projection "
        "(2048×1080 or 4096×2160). P3 color space. Parallax within SMPTE "
        "stereo comfort recommendations."
    ),
    stereo_params=StereoParams(
        depth_factor=0.30,
        max_parallax_fraction=0.022,   # SMPTE guideline: ~2% for theatre
        eye_separation_fraction=0.065,
        convergence=0.02,              # slight positive convergence for theatre
        safe_mode=False,
        oversample=2,
    ),
    depth_params=DepthPrepParams(
        bilateral_sigma_space=6.0,
        bilateral_sigma_color=0.10,
        dilation_px=4,
        falloff_curve=FalloffCurve.S_CURVE,
        near_plane=0.03,
        far_plane=0.97,
    ),
    output_dpi=72,
    color_space="P3",
    target_width=4096,
    notes=(
        "For DCP delivery: output at 4096×2160 or 2048×1080. "
        "Apply P3 → XYZ color transform in post. "
        "max_parallax_fraction=0.022 satisfies SMPTE ST 2098-1 guidelines."
    ),
)

#: High-resolution print output
PRESET_PRINT = Preset(
    name="print",
    display_name="Print",
    description=(
        "Optimised for high-resolution print at 300dpi. Tighter depth to account "
        "for the closer viewing distance typical when holding a printed page. "
        "CMYK-safe color guidance included."
    ),
    stereo_params=StereoParams(
        depth_factor=0.28,
        max_parallax_fraction=0.025,   # print viewed closer, keep parallax tight
        eye_separation_fraction=0.065,
        convergence=0.0,
        safe_mode=False,
        oversample=2,                  # oversample — upscale to print resolution
    ),
    depth_params=DepthPrepParams(
        bilateral_sigma_space=6.0,
        bilateral_sigma_color=0.10,
        dilation_px=4,
        falloff_curve=FalloffCurve.GAMMA,
        near_plane=0.03,
        far_plane=0.97,
    ),
    output_dpi=300,
    color_space="CMYK-safe",
    target_width=3000,     # 10" at 300dpi
    notes=(
        "Output at 300dpi minimum. For CMYK conversion: saturate pattern to "
        "<320% total ink coverage. Psychedelic color modes may shift significantly "
        "in CMYK — test proof before final print run. "
        "Pattern tile should be at least 400px wide at 300dpi output."
    ),
)

#: Broadcast / streaming delivery
PRESET_BROADCAST = Preset(
    name="broadcast",
    display_name="Broadcast",
    description=(
        "TV and streaming delivery profile (Rec.709). Conservative parallax "
        "within EBU R95 / SMPTE comfort recommendations. Legal level output "
        "(16–235). Suitable for linear broadcast and OTT platforms."
    ),
    stereo_params=StereoParams(
        depth_factor=0.22,
        max_parallax_fraction=0.020,   # EBU R95 guidance: ~2% for 16:9 displays
        eye_separation_fraction=0.065,
        convergence=0.01,
        safe_mode=True,
        oversample=1,
    ),
    depth_params=DepthPrepParams(
        bilateral_sigma_space=7.0,
        bilateral_sigma_color=0.12,
        dilation_px=5,
        falloff_curve=FalloffCurve.GAMMA,
        near_plane=0.05,
        far_plane=0.93,
    ),
    output_dpi=72,
    color_space="Rec.709",
    target_width=1920,
    notes=(
        "Clamp output to legal levels (16–235) before delivery. "
        "safe_mode=True enforces hard parallax limits. "
        "EBU R95 guideline: max positive parallax 2% of picture width. "
        "Include photosensitivity warning in programme information."
    ),
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_PRESETS: dict[str, Preset] = {
    p.name: p for p in [
        PRESET_SHALLOW,
        PRESET_MEDIUM,
        PRESET_DEEP,
        PRESET_CINEMA,
        PRESET_PRINT,
        PRESET_BROADCAST,
    ]
}


def list_presets() -> list[str]:
    """Return the names of all built-in presets."""
    return list(_PRESETS.keys())


def get_preset(name: str) -> Preset:
    """Retrieve a built-in preset by name.

    Parameters
    ----------
    name : str
        One of: "shallow", "medium", "deep", "cinema", "print", "broadcast"

    Raises
    ------
    KeyError
        If the name is not recognised.
    """
    name = name.lower().strip()
    if name not in _PRESETS:
        available = ", ".join(sorted(_PRESETS.keys()))
        raise KeyError(f"Unknown preset {name!r}. Available: {available}")
    return _PRESETS[name]


def register_preset(preset: Preset, overwrite: bool = False) -> None:
    """Register a custom preset in the global registry.

    Parameters
    ----------
    preset : Preset
    overwrite : bool
        If False (default), raises ValueError if name already registered.
    """
    if preset.name in _PRESETS and not overwrite:
        raise ValueError(
            f"Preset {preset.name!r} already registered. "
            "Use overwrite=True to replace it."
        )
    _PRESETS[preset.name] = preset


def load_preset_from_json(path: str) -> Preset:
    """Load a custom preset from a JSON file and register it.

    Returns the loaded Preset.
    """
    data = json.loads(Path(path).read_text())
    preset = Preset.from_dict(data)
    register_preset(preset, overwrite=True)
    return preset
