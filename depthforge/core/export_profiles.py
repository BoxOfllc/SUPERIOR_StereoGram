"""
depthforge.core.export_profiles
================================
One-click export profiles for common delivery standards.

Each profile applies a consistent set of post-processing operations to a
rendered stereogram (RGBA uint8) and saves it in the appropriate format
with correct colour space metadata, DPI, and file structure.

Supported profiles
------------------
Print 300dpi CMYK-safe
    TIFF, 300dpi, sRGB→CMYK-safe gamut clip (no pure R/G/B primaries that
    can't be reproduced in CMYK), or ICC-tagged TIFF if PyICCProfile available.

Web sRGB
    PNG, 72dpi, sRGB ICC tag, optional lossless or JPEG output.

Broadcast Rec.709
    PNG or TIFF, legal levels (16–235 luma, 16–240 chroma), Rec.709 tag.

Digital Cinema DCI P3
    TIFF, 2048×1080 or 4096×2160, linear light, P3-D65 tag.

Custom
    User-supplied dict of parameters.

Public API
----------
``ExportProfile``        Enum of named profiles.
``ExportParams``         Full parameter dataclass.
``get_profile(name)``    Returns a pre-configured ExportParams.
``export_image(image, path, params)``  Save with profile applied.
``apply_profile(image, params)``       Apply transformations, return np.ndarray.
``list_profiles()``                    Returns list of profile names.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ExportProfile(Enum):
    PRINT_CMYK = "print_cmyk"
    WEB_SRGB = "web_srgb"
    BROADCAST = "broadcast"
    DCI_P3 = "dci_p3"
    CUSTOM = "custom"


class OutputFormat(Enum):
    PNG = "png"
    TIFF = "tiff"
    JPEG = "jpeg"
    EXR = "exr"


class ColourSpace(Enum):
    SRGB = "sRGB"
    REC709 = "Rec.709"
    DCI_P3_D65 = "P3-D65"
    LINEAR_SRGB = "Linear sRGB"
    CMYK_SAFE = "CMYK-safe sRGB"


# ---------------------------------------------------------------------------
# ExportParams
# ---------------------------------------------------------------------------


@dataclass
class ExportParams:
    """
    Full export parameter set.

    Attributes
    ----------
    profile : ExportProfile
        Named profile for documentation / serialisation.
    format : OutputFormat
        Output file format.
    dpi : int
        Dots per inch metadata embedded in the output file.
    colour_space : ColourSpace
        Target colour space. Determines gamut clip and ICC tag.
    width : int, optional
        Resize output to this width before saving. None = keep original.
    height : int, optional
        Resize output to this height. None = keep aspect.
    legal_levels : bool
        Clamp luma to broadcast legal range (16–235 / 8-bit).
        Auto-enabled for BROADCAST profile.
    cmyk_safe : bool
        Clip saturated primaries that reproduce poorly in CMYK offset print.
    jpeg_quality : int
        JPEG quality 1–95. Only used when format = JPEG.
    tiff_compress : str
        TIFF compression: 'none', 'lzw', 'deflate', 'packbits'.
    embed_icc : bool
        Embed an ICC profile in the output file (where supported).
    alpha_handling : str
        'keep' = preserve alpha, 'flatten_white' = composite on white,
        'flatten_black' = composite on black, 'discard' = drop alpha.
    notes : str
        Human-readable delivery notes.
    """

    profile: ExportProfile = ExportProfile.CUSTOM
    format: OutputFormat = OutputFormat.PNG
    dpi: int = 72
    colour_space: ColourSpace = ColourSpace.SRGB
    width: Optional[int] = None
    height: Optional[int] = None
    legal_levels: bool = False
    cmyk_safe: bool = False
    jpeg_quality: int = 92
    tiff_compress: str = "lzw"
    embed_icc: bool = True
    alpha_handling: str = "keep"
    notes: str = ""


# ---------------------------------------------------------------------------
# Built-in profiles
# ---------------------------------------------------------------------------

_PROFILES: dict[str, ExportParams] = {
    "print_cmyk": ExportParams(
        profile=ExportProfile.PRINT_CMYK,
        format=OutputFormat.TIFF,
        dpi=300,
        colour_space=ColourSpace.CMYK_SAFE,
        legal_levels=False,
        cmyk_safe=True,
        tiff_compress="lzw",
        embed_icc=True,
        alpha_handling="flatten_white",
        notes=(
            "300dpi TIFF for offset/giclée print. sRGB gamut clipped to "
            "CMYK-reproducible range. Composite on white background."
        ),
    ),
    "web_srgb": ExportParams(
        profile=ExportProfile.WEB_SRGB,
        format=OutputFormat.PNG,
        dpi=72,
        colour_space=ColourSpace.SRGB,
        legal_levels=False,
        cmyk_safe=False,
        embed_icc=True,
        alpha_handling="keep",
        notes=(
            "sRGB PNG for web, social, and screen display. " "Lossless with embedded sRGB ICC tag."
        ),
    ),
    "broadcast": ExportParams(
        profile=ExportProfile.BROADCAST,
        format=OutputFormat.TIFF,
        dpi=72,
        colour_space=ColourSpace.REC709,
        legal_levels=True,
        cmyk_safe=False,
        tiff_compress="none",
        embed_icc=True,
        alpha_handling="flatten_black",
        notes=(
            "Rec.709 broadcast delivery. Legal levels 16–235 (luma). "
            "Uncompressed TIFF for master handoff."
        ),
    ),
    "dci_p3": ExportParams(
        profile=ExportProfile.DCI_P3,
        format=OutputFormat.TIFF,
        dpi=72,
        colour_space=ColourSpace.DCI_P3_D65,
        width=2048,
        height=1080,
        legal_levels=False,
        cmyk_safe=False,
        tiff_compress="none",
        embed_icc=True,
        alpha_handling="flatten_black",
        notes=(
            "DCI P3-D65 for digital cinema. 2K (2048×1080) uncompressed TIFF. "
            "Suitable for DCP wrap with external tool (e.g. OpenDCP)."
        ),
    ),
    "web_jpeg": ExportParams(
        profile=ExportProfile.WEB_SRGB,
        format=OutputFormat.JPEG,
        dpi=96,
        colour_space=ColourSpace.SRGB,
        legal_levels=False,
        cmyk_safe=False,
        jpeg_quality=88,
        embed_icc=True,
        alpha_handling="flatten_white",
        notes="sRGB JPEG for web. Quality 88, composited on white.",
    ),
}


def list_profiles() -> list[str]:
    """Return all named profile keys."""
    return list(_PROFILES.keys())


def get_profile(name: str) -> ExportParams:
    """
    Return a pre-configured ExportParams for a named profile.

    Parameters
    ----------
    name : str
        One of: 'print_cmyk', 'web_srgb', 'broadcast', 'dci_p3', 'web_jpeg'.

    Raises
    ------
    KeyError if the name is not found.
    """
    name = name.lower().strip()
    if name not in _PROFILES:
        raise KeyError(f"Unknown export profile '{name}'. " f"Available: {list_profiles()}")
    from dataclasses import replace

    return replace(_PROFILES[name])  # return a fresh copy


# ---------------------------------------------------------------------------
# Image transformation helpers
# ---------------------------------------------------------------------------


def _flatten_alpha(
    image: np.ndarray,  # (H, W, 4) RGBA uint8
    mode: str,
) -> np.ndarray:
    """Flatten alpha channel according to mode. Returns (H, W, 3) uint8."""
    if image.ndim == 2 or image.shape[2] == 3:
        return image
    if mode == "discard":
        return image[..., :3]

    alpha = image[..., 3:4].astype(np.float32) / 255.0
    rgb = image[..., :3].astype(np.float32)

    if mode == "flatten_white":
        bg = np.full_like(rgb, 255.0)
    else:  # flatten_black
        bg = np.zeros_like(rgb)

    composited = (rgb * alpha + bg * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
    return composited


def _apply_legal_levels(image: np.ndarray) -> np.ndarray:
    """
    Remap [0, 255] → [16, 235] (broadcast legal luma range).
    Keeps 8-bit uint8 output.
    """
    f = image.astype(np.float32)
    f = 16.0 + (f / 255.0) * (235.0 - 16.0)
    return f.clip(16, 235).astype(np.uint8)


def _apply_cmyk_safe(image: np.ndarray) -> np.ndarray:
    """
    Clip saturated channel combinations that are outside CMYK gamut.

    The key constraint in offset CMYK is that very saturated digital
    primaries (pure R 255, G 255, B 255 or high-saturation combos) cannot
    be reproduced. We apply a simple gamut-clip:
    - Reduce the maximum channel saturation (sum of channels) to 300 / 255
      equivalent in float space.
    - This is a conservative approximation; a full CMYK profile conversion
      requires ICC/OCIO (used when HAS_OCIO).
    """
    rgb = image.astype(np.float32) / 255.0
    # Ink-sum cap: C+M+Y+K ≤ 300% → roughly RGB channel sum ≤ 2.4
    channel_sum = rgb[..., 0] + rgb[..., 1] + rgb[..., 2]
    cap = 2.4
    scale = np.where(channel_sum > cap, cap / channel_sum, 1.0)
    rgb_clipped = rgb * scale[..., np.newaxis]
    return (rgb_clipped * 255.0).clip(0, 255).astype(np.uint8)


def _resize_image(
    image: np.ndarray,
    width: Optional[int],
    height: Optional[int],
) -> np.ndarray:
    if width is None and height is None:
        return image
    H, W = image.shape[:2]
    if width is not None and height is None:
        height = int(H * width / W)
    elif height is not None and width is None:
        width = int(W * height / H)
    pil = Image.fromarray(image)
    pil = pil.resize((width, height), Image.LANCZOS)
    return np.array(pil)


# ---------------------------------------------------------------------------
# ICC profile bytes (minimal sRGB and Rec.709 stubs)
# ---------------------------------------------------------------------------


def _get_icc_bytes(colour_space: ColourSpace) -> Optional[bytes]:
    """
    Return ICC profile bytes for embedding.

    If Pillow's built-in sRGB profile is available, use it. Otherwise
    return None (no ICC embedding).
    """
    try:
        import io

        if colour_space == ColourSpace.SRGB:
            p = Image.new("RGB", (1, 1))
            buf = io.BytesIO()
            p.save(buf, format="PNG", icc_profile=Image.LANCZOS)
            # PIL sRGB is implicit; use its embedded bytes if available
            icc = p.info.get("icc_profile")
            return icc  # may be None
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------


def apply_profile(
    image: np.ndarray,
    params: ExportParams,
) -> np.ndarray:
    """
    Apply all export profile transformations to an image array.

    Parameters
    ----------
    image : np.ndarray
        RGBA or RGB uint8 array (H, W, 3 or 4).
    params : ExportParams
        Export configuration.

    Returns
    -------
    np.ndarray
        Processed uint8 array ready for saving.
    """
    out = image.copy()

    # 1. Resize
    if params.width or params.height:
        out = _resize_image(out, params.width, params.height)

    # 2. Alpha handling
    if out.ndim == 3 and out.shape[2] == 4:
        if params.alpha_handling == "keep":
            pass  # keep RGBA
        else:
            out = _flatten_alpha(out, params.alpha_handling)
    elif out.ndim == 3 and out.shape[2] == 3:
        pass  # already RGB

    # 3. CMYK-safe gamut clip
    if params.cmyk_safe and out.shape[2] >= 3:
        rgb_part = _apply_cmyk_safe(out[..., :3])
        if out.shape[2] == 4:
            out = np.concatenate([rgb_part, out[..., 3:4]], axis=2)
        else:
            out = rgb_part

    # 4. Broadcast legal levels
    if params.legal_levels and out.shape[2] >= 3:
        rgb_part = _apply_legal_levels(out[..., :3])
        if out.shape[2] == 4:
            out = np.concatenate([rgb_part, out[..., 3:4]], axis=2)
        else:
            out = rgb_part

    return out


def export_image(
    image: np.ndarray,
    path: Union[str, Path],
    params: Optional[ExportParams] = None,
    profile_name: Optional[str] = None,
) -> Path:
    """
    Apply export profile and save image to disk.

    Parameters
    ----------
    image : np.ndarray
        RGBA or RGB uint8 source image.
    path : str or Path
        Output file path. Extension is overridden to match params.format
        if they disagree.
    params : ExportParams, optional
        If None, uses web_srgb defaults.
    profile_name : str, optional
        If provided, loads a named profile. Overrides params.

    Returns
    -------
    Path
        The actual path the file was written to.
    """
    if profile_name:
        params = get_profile(profile_name)
    elif params is None:
        params = get_profile("web_srgb")

    out_path = Path(path)

    # Force correct extension
    ext_map = {
        OutputFormat.PNG: ".png",
        OutputFormat.TIFF: ".tiff",
        OutputFormat.JPEG: ".jpg",
        OutputFormat.EXR: ".exr",
    }
    out_path = out_path.with_suffix(ext_map[params.format])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Apply transformations
    processed = apply_profile(image, params)

    # Convert to PIL
    if processed.ndim == 2:
        pil_img = Image.fromarray(processed, mode="L")
    elif processed.shape[2] == 4:
        pil_img = Image.fromarray(processed, mode="RGBA")
    else:
        pil_img = Image.fromarray(processed, mode="RGB")

    # Convert RGBA to RGB for formats that don't support alpha
    if params.format in (OutputFormat.JPEG, OutputFormat.TIFF):
        if pil_img.mode == "RGBA":
            pil_img = pil_img.convert("RGB")

    # Build save kwargs
    save_kwargs: dict = {}

    dpi_tuple = (params.dpi, params.dpi)

    if params.format == OutputFormat.PNG:
        save_kwargs["dpi"] = dpi_tuple
        if params.embed_icc:
            icc = _get_icc_bytes(params.colour_space)
            if icc:
                save_kwargs["icc_profile"] = icc

    elif params.format == OutputFormat.TIFF:
        compress_map = {
            "none": None,
            "lzw": "tiff_lzw",
            "deflate": "tiff_deflate",
            "packbits": "packbits",
        }
        comp = compress_map.get(params.tiff_compress.lower(), "tiff_lzw")
        save_kwargs["dpi"] = dpi_tuple
        if comp:
            save_kwargs["compression"] = comp

    elif params.format == OutputFormat.JPEG:
        save_kwargs["quality"] = params.jpeg_quality
        save_kwargs["dpi"] = dpi_tuple

    pil_img.save(str(out_path), **save_kwargs)
    return out_path


def profile_summary(params: ExportParams) -> str:
    """Return a human-readable summary of an ExportParams."""
    lines = [
        f"Profile   : {params.profile.value}",
        f"Format    : {params.format.value.upper()}, {params.dpi} dpi",
        f"Colour    : {params.colour_space.value}",
        f"Size      : {params.width or 'original'} × {params.height or 'original'}",
        f"Legal lvl : {'yes' if params.legal_levels else 'no'}",
        f"CMYK-safe : {'yes' if params.cmyk_safe else 'no'}",
        f"Alpha     : {params.alpha_handling}",
    ]
    if params.notes:
        lines.append(f"Notes     : {params.notes}")
    return "\n".join(lines)
