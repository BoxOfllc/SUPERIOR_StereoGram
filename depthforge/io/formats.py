"""
depthforge.io.formats
======================
Image I/O with support for EXR (float), multi-layer TIFF, and standard formats.

Functions
---------
    read_depth(path)          Load any format as float32 depth [0,1]
    write_depth(depth, path)  Save float32 depth to EXR/TIFF/PNG
    read_rgba(path)           Load any format as RGBA uint8
    write_rgba(rgba, path)    Save RGBA uint8 to any format
    write_multilayer(layers, path)  Write multi-channel EXR
    read_multilayer(path)     Read multi-channel EXR as dict of arrays

EXR support requires OpenEXR + Imath. Falls back to TIFF/PNG gracefully.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np


# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    import OpenEXR as _exr
    import Imath as _imath
    HAS_EXR = True
except ImportError:
    HAS_EXR = False

try:
    from PIL import Image as _PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import cv2 as _cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ---------------------------------------------------------------------------
# Read / write depth maps
# ---------------------------------------------------------------------------

def read_depth(path: str) -> np.ndarray:
    """Load a depth map as float32 (H, W) in [0, 1].

    Supports: EXR (float32 Z channel), TIFF (16/32-bit), PNG/JPG (8-bit).

    Depth convention: 1.0 = near, 0.0 = far.
    For EXR Z passes (near=small, far=large) use
    ``invert=True`` in ``DepthPrepParams``.

    Parameters
    ----------
    path : str
        Input file path.

    Returns
    -------
    np.ndarray  float32 (H, W)
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext == ".exr":
        return _read_exr_depth(path)
    elif ext in (".tiff", ".tif"):
        return _read_tiff_depth(path)
    else:
        return _read_pil_depth(path)


def write_depth(
    depth: np.ndarray,
    path: str,
    bit_depth: int = 16,
    channel: str = "Y",
) -> None:
    """Save a float32 depth map to file.

    Parameters
    ----------
    depth : np.ndarray
        float32 (H, W), values in [0, 1].
    path : str
        Output path. Extension determines format.
    bit_depth : int
        For TIFF: 16 (default) or 32 bits. For EXR: always 32-bit float.
    channel : str
        EXR channel name. Default "Y" (luminance). Use "Z" for Z-depth.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower()

    if ext == ".exr":
        _write_exr_depth(depth, path, channel=channel)
    elif ext in (".tiff", ".tif"):
        _write_tiff_depth(depth, path, bit_depth=bit_depth)
    else:
        # PNG fallback — 8-bit, some precision loss
        if HAS_PIL:
            _PILImage.fromarray((depth * 255).astype(np.uint8)).save(path)
        else:
            raise RuntimeError("Pillow required for PNG output.")


# ---------------------------------------------------------------------------
# Read / write RGBA
# ---------------------------------------------------------------------------

def read_rgba(path: str) -> np.ndarray:
    """Load any image as RGBA uint8 (H, W, 4).

    Supports EXR (converted to 8-bit), TIFF, PNG, JPG.
    """
    p = Path(path)
    if p.suffix.lower() == ".exr":
        return _read_exr_rgba(path)
    if HAS_PIL:
        return np.array(_PILImage.open(path).convert("RGBA"), dtype=np.uint8)
    if HAS_CV2:
        bgra = _cv2.imread(str(path), _cv2.IMREAD_UNCHANGED)
        if bgra is None:
            raise FileNotFoundError(f"Cannot read: {path}")
        if bgra.ndim == 2:
            bgra = np.stack([bgra, bgra, bgra, np.full_like(bgra, 255)], axis=-1)
        elif bgra.shape[2] == 3:
            alpha = np.full((*bgra.shape[:2], 1), 255, dtype=np.uint8)
            bgra  = np.concatenate([bgra, alpha], axis=-1)
        bgra[..., :3] = bgra[..., 2::-1]   # BGR → RGB
        return bgra
    raise RuntimeError("Pillow or OpenCV required to read images.")


def write_rgba(rgba: np.ndarray, path: str, dpi: int = 72) -> None:
    """Save RGBA uint8 array to file.

    Parameters
    ----------
    rgba : np.ndarray  uint8 (H, W, 4)
    path : str
    dpi  : int  Output resolution metadata.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower()

    if ext == ".exr":
        _write_exr_rgba(rgba, path)
        return

    if HAS_PIL:
        img = _PILImage.fromarray(rgba)
        kwargs = {"dpi": (dpi, dpi)}
        if ext in (".jpg", ".jpeg"):
            img = img.convert("RGB")
            img.save(path, quality=95, **kwargs)
        else:
            img.save(path, **kwargs)
    elif HAS_CV2:
        rgb  = rgba[..., :3][..., ::-1]   # RGB → BGR
        _cv2.imwrite(str(path), rgb)
    else:
        raise RuntimeError("Pillow or OpenCV required to write images.")


# ---------------------------------------------------------------------------
# Multi-layer EXR
# ---------------------------------------------------------------------------

def write_multilayer(
    layers: dict[str, np.ndarray],
    path: str,
    compress: bool = True,
) -> None:
    """Write multiple named float32 channels to an EXR file.

    Parameters
    ----------
    layers : dict[str, np.ndarray]
        Channel name → float32 array (H, W).
        Common names: "R", "G", "B", "A", "depth.Z", "left.R" etc.
    path : str
        Output EXR path.
    compress : bool
        Use ZIP compression. Default True.

    Raises
    ------
    RuntimeError
        If OpenEXR is not available.
    """
    if not HAS_EXR:
        raise RuntimeError(
            "Multi-layer EXR requires OpenEXR.\n"
            "Install: pip install openexr"
        )
    import struct
    H, W = next(iter(layers.values())).shape[:2]

    header = _exr.Header(W, H)
    header["channels"] = {
        name: _imath.Channel(_imath.PixelType(_imath.PixelType.FLOAT))
        for name in layers
    }
    if compress:
        header["compression"] = _imath.Compression(_imath.Compression.ZIP_COMPRESSION)

    out = _exr.OutputFile(str(path), header)
    channel_data = {
        name: arr.astype(np.float32).tobytes()
        for name, arr in layers.items()
    }
    out.writePixels(channel_data)
    out.close()


def read_multilayer(path: str) -> dict[str, np.ndarray]:
    """Read all channels from an EXR file.

    Returns
    -------
    dict[str, np.ndarray]
        Channel name → float32 array (H, W).

    Raises
    ------
    RuntimeError
        If OpenEXR is not available.
    """
    if not HAS_EXR:
        raise RuntimeError(
            "Reading EXR requires OpenEXR.\n"
            "Install: pip install openexr"
        )

    exr_file = _exr.InputFile(str(path))
    header = exr_file.header()

    dw = header["dataWindow"]
    W = dw.max.x - dw.min.x + 1
    H = dw.max.y - dw.min.y + 1

    float_type = _imath.PixelType(_imath.PixelType.FLOAT)
    channels = list(header["channels"].keys())

    result = {}
    for ch in channels:
        raw = exr_file.channel(ch, float_type)
        arr = np.frombuffer(raw, dtype=np.float32).reshape(H, W)
        result[ch] = arr

    exr_file.close()
    return result


# ---------------------------------------------------------------------------
# OCIO color transform (stub with graceful fallback)
# ---------------------------------------------------------------------------

def apply_color_transform(
    image: np.ndarray,
    src_colorspace: str,
    dst_colorspace: str,
    config_path: Optional[str] = None,
) -> np.ndarray:
    """Apply an OCIO color space transform to a float32 image.

    Parameters
    ----------
    image : np.ndarray
        float32 RGBA or RGB array (H, W, 3 or 4).
    src_colorspace : str
        Source color space name (must exist in OCIO config).
    dst_colorspace : str
        Destination color space name.
    config_path : str, optional
        Path to OCIO config file. If None, uses $OCIO env var or falls back
        to a simple sRGB↔Linear LUT.

    Returns
    -------
    np.ndarray  float32, same shape as input.
    """
    try:
        import PyOpenColorIO as ocio

        if config_path:
            cfg = ocio.Config.CreateFromFile(config_path)
        else:
            cfg = ocio.GetCurrentConfig()

        proc = cfg.getProcessor(src_colorspace, dst_colorspace)
        cpu  = proc.getDefaultCPUProcessor()

        img = image.astype(np.float32)
        if img.shape[2] == 4:
            rgb = img[..., :3].copy()
            cpu.applyRGB(rgb)
            img[..., :3] = rgb
        else:
            cpu.applyRGB(img)

        return img

    except ImportError:
        # Fallback: approximate sRGB ↔ Linear
        img = image.astype(np.float32)
        if src_colorspace.lower() in ("srgb", "srgb - texture") and "linear" in dst_colorspace.lower():
            # sRGB → Linear
            return np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)
        elif "linear" in src_colorspace.lower() and "srgb" in dst_colorspace.lower():
            # Linear → sRGB
            return np.where(img <= 0.0031308, img * 12.92, 1.055 * img ** (1/2.4) - 0.055)
        else:
            # Unknown transform — return unchanged with warning
            import warnings
            warnings.warn(
                f"PyOpenColorIO not available. Color transform {src_colorspace!r} → "
                f"{dst_colorspace!r} was not applied. Install PyOpenColorIO for full support.",
                UserWarning,
                stacklevel=2,
            )
            return img


# ---------------------------------------------------------------------------
# Internal EXR helpers
# ---------------------------------------------------------------------------

def _read_exr_depth(path: str) -> np.ndarray:
    if not HAS_EXR:
        raise RuntimeError(
            f"Cannot read EXR {path}: OpenEXR not installed.\n"
            "Install: pip install openexr"
        )
    exr  = _exr.InputFile(str(path))
    hdr  = exr.header()
    dw   = hdr["dataWindow"]
    W, H = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
    ft   = _imath.PixelType(_imath.PixelType.FLOAT)

    # Try Z channel first, then Y, then R
    for ch in ("Z", "Y", "R", "G", "B"):
        if ch in hdr.get("channels", {}):
            raw  = exr.channel(ch, ft)
            arr  = np.frombuffer(raw, dtype=np.float32).reshape(H, W)
            exr.close()
            lo, hi = arr.min(), arr.max()
            if hi > lo:
                arr = (arr - lo) / (hi - lo)
            return arr.astype(np.float32)

    exr.close()
    raise RuntimeError(f"No suitable depth channel found in {path}")


def _read_tiff_depth(path: str) -> np.ndarray:
    if HAS_PIL:
        img = _PILImage.open(path)
        if img.mode == "I;16":
            arr = np.array(img, dtype=np.float32) / 65535.0
        elif img.mode in ("I", "F"):
            arr = np.array(img, dtype=np.float32)
            lo, hi = arr.min(), arr.max()
            if hi > lo:
                arr = (arr - lo) / (hi - lo)
        else:
            arr = np.array(img.convert("L"), dtype=np.float32) / 255.0
        return arr
    raise RuntimeError("Pillow required to read TIFF.")


def _read_pil_depth(path: str) -> np.ndarray:
    if HAS_PIL:
        img = _PILImage.open(path)
        if img.mode == "I":
            arr = np.array(img, dtype=np.float32)
        elif img.mode == "F":
            arr = np.array(img, dtype=np.float32)
        else:
            arr = np.array(img.convert("L"), dtype=np.float32) / 255.0
        lo, hi = arr.min(), arr.max()
        if hi > lo:
            arr = (arr - lo) / (hi - lo)
        return arr.astype(np.float32)
    raise RuntimeError("Pillow required.")


def _write_exr_depth(depth: np.ndarray, path: str, channel: str = "Y") -> None:
    if not HAS_EXR:
        raise RuntimeError("OpenEXR required for EXR output.")
    H, W = depth.shape
    header = _exr.Header(W, H)
    header["channels"] = {
        channel: _imath.Channel(_imath.PixelType(_imath.PixelType.FLOAT))
    }
    out = _exr.OutputFile(str(path), header)
    out.writePixels({channel: depth.astype(np.float32).tobytes()})
    out.close()


def _write_tiff_depth(depth: np.ndarray, path: str, bit_depth: int = 16) -> None:
    if not HAS_PIL:
        raise RuntimeError("Pillow required for TIFF output.")
    if bit_depth == 32:
        img = _PILImage.fromarray(depth.astype(np.float32), mode="F")
    else:
        img = _PILImage.fromarray((depth * 65535).astype(np.uint16))
    img.save(path)


def _read_exr_rgba(path: str) -> np.ndarray:
    if not HAS_EXR:
        raise RuntimeError("OpenEXR required to read EXR as RGBA.")
    exr = _exr.InputFile(str(path))
    hdr = exr.header()
    dw  = hdr["dataWindow"]
    W, H = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
    ft  = _imath.PixelType(_imath.PixelType.FLOAT)
    channels = hdr.get("channels", {})

    def _ch(name):
        if name in channels:
            return np.frombuffer(exr.channel(name, ft), np.float32).reshape(H, W)
        return np.ones((H, W), np.float32)

    rgba = np.stack([_ch("R"), _ch("G"), _ch("B"), _ch("A")], axis=-1)
    exr.close()
    return np.clip(rgba * 255, 0, 255).astype(np.uint8)


def _write_exr_rgba(rgba: np.ndarray, path: str) -> None:
    if not HAS_EXR:
        raise RuntimeError("OpenEXR required for EXR output.")
    H, W = rgba.shape[:2]
    header = _exr.Header(W, H)
    header["channels"] = {
        ch: _imath.Channel(_imath.PixelType(_imath.PixelType.FLOAT))
        for ch in ("R", "G", "B", "A")
    }
    out = _exr.OutputFile(str(path), header)
    img_f = rgba.astype(np.float32) / 255.0
    out.writePixels({
        "R": img_f[..., 0].tobytes(),
        "G": img_f[..., 1].tobytes(),
        "B": img_f[..., 2].tobytes(),
        "A": img_f[..., 3].tobytes(),
    })
    out.close()
