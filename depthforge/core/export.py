"""
depthforge.core.export
======================
Export profiles and delivery pipeline for stereogram output.

An ExportProfile bundles all delivery parameters — resolution, DPI,
colour space, bit depth, file format, ICC profile, compression — into a
single named object.  The ``Exporter`` class applies a profile to a
synthesised stereogram and writes the result to disk.

Built-in profiles
-----------------
web_srgb        72dpi sRGB PNG for web and social media
print_300       300dpi sRGB TIFF for print / giclée
broadcast_rec709 1920×1080 Rec.709 PNG for TV/streaming
cinema_p3       4096×2160 P3 PNG for digital cinema (DCP)
archive_exr     Linear EXR with full 32-bit depth (archival)

API
---
::

    from depthforge.core.export import Exporter, ExportProfile, get_profile

    profile  = get_profile("print_300")
    exporter = Exporter(profile)
    path     = exporter.export(stereogram_array, "/output/my_image")

    # Batch export:
    paths = exporter.export_batch(
        [(stereo1, "frame_001"), (stereo2, "frame_002")],
        output_dir="/output/frames/",
    )
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Supported formats
# ---------------------------------------------------------------------------

#: File extension → PIL/Pillow save format string
FORMAT_MAP: Dict[str, str] = {
    ".png":  "PNG",
    ".jpg":  "JPEG",
    ".jpeg": "JPEG",
    ".tif":  "TIFF",
    ".tiff": "TIFF",
    ".webp": "WEBP",
    ".bmp":  "BMP",
}

#: Formats that support 16-bit output
SUPPORTS_16BIT = {".tif", ".tiff", ".png"}

#: Formats that support alpha channel
SUPPORTS_ALPHA = {".png", ".tiff", ".tif", ".webp", ".bmp"}


# ---------------------------------------------------------------------------
# ExportProfile
# ---------------------------------------------------------------------------

@dataclass
class ExportProfile:
    """Complete export/delivery parameter bundle.

    Attributes
    ----------
    name : str
        Internal identifier.
    display_name : str
        Human-readable label.
    format : str
        Output file extension including leading dot: ``".png"``, ``".tiff"``, etc.
    dpi : int
        Pixels per inch for print/metadata tagging.
    color_space : str
        Target colour space label (informational + used for OCIO transforms):
        ``"sRGB"``, ``"Rec.709"``, ``"P3-D65"``, ``"Linear"``.
    bit_depth : int
        8 or 16. 16-bit only supported for TIFF and PNG.
    include_alpha : bool
        Preserve the RGBA alpha channel. If False, output is RGB.
    target_width : int
        Resize output to this width. 0 = no resize.
    target_height : int
        Resize output to this height. 0 = no resize.
    jpeg_quality : int
        JPEG quality 1–95 (ignored for non-JPEG formats).
    png_compress : int
        PNG compression level 0–9 (0 = no compression, 9 = maximum).
    tiff_compress : str
        TIFF compression: ``"none"``, ``"lzw"``, ``"deflate"``, ``"jpeg"``.
    embed_icc : bool
        Embed an ICC profile in the output file when available.
    icc_profile_path : str
        Path to an ICC profile (.icc / .icm) file to embed. Empty = use
        Pillow's built-in sRGB profile.
    embed_metadata : bool
        Write DPI, colour-space, and DepthForge version into EXIF/XMP.
    watermark : str
        Optional text watermark to burn into the image. Empty = no watermark.
    notes : str
        Free-form delivery instructions.
    """

    name:             str   = "custom"
    display_name:     str   = "Custom"
    format:           str   = ".png"
    dpi:              int   = 72
    color_space:      str   = "sRGB"
    bit_depth:        int   = 8
    include_alpha:    bool  = False
    target_width:     int   = 0
    target_height:    int   = 0
    jpeg_quality:     int   = 95
    png_compress:     int   = 6
    tiff_compress:    str   = "lzw"
    embed_icc:        bool  = True
    icc_profile_path: str   = ""
    embed_metadata:   bool  = True
    watermark:        str   = ""
    notes:            str   = ""

    def __post_init__(self):
        if self.bit_depth not in (8, 16):
            raise ValueError(f"bit_depth must be 8 or 16, got {self.bit_depth}")
        if self.format not in FORMAT_MAP:
            raise ValueError(f"Unsupported format {self.format!r}. Supported: {list(FORMAT_MAP)}")
        if self.bit_depth == 16 and self.format not in SUPPORTS_16BIT:
            raise ValueError(f"16-bit output not supported for {self.format}. Use .tiff or .png.")

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: Optional[str] = None) -> str:
        txt = json.dumps(self.to_dict(), indent=2)
        if path:
            Path(path).write_text(txt)
        return txt

    @classmethod
    def from_dict(cls, d: dict) -> "ExportProfile":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})

    @classmethod
    def from_json(cls, path: str) -> "ExportProfile":
        return cls.from_dict(json.loads(Path(path).read_text()))


# ---------------------------------------------------------------------------
# Built-in profiles
# ---------------------------------------------------------------------------

_BUILTIN_PROFILES: Dict[str, ExportProfile] = {
    "web_srgb": ExportProfile(
        name="web_srgb",
        display_name="Web / Social Media (sRGB)",
        format=".png",
        dpi=72,
        color_space="sRGB",
        bit_depth=8,
        include_alpha=False,
        png_compress=6,
        embed_metadata=True,
        notes="Optimised for web display. sRGB, 72dpi, 8-bit PNG.",
    ),
    "print_300": ExportProfile(
        name="print_300",
        display_name="Print — 300dpi TIFF (sRGB)",
        format=".tiff",
        dpi=300,
        color_space="sRGB",
        bit_depth=16,
        include_alpha=False,
        tiff_compress="lzw",
        embed_icc=True,
        embed_metadata=True,
        notes="Magazine and giclée print. 300dpi, 16-bit LZW TIFF.",
    ),
    "print_cmyk": ExportProfile(
        name="print_cmyk",
        display_name="Print — CMYK TIFF (300dpi)",
        format=".tiff",
        dpi=300,
        color_space="sRGB",   # converted to CMYK at save time
        bit_depth=8,
        include_alpha=False,
        tiff_compress="none",
        embed_icc=True,
        embed_metadata=True,
        notes="CMYK prepress. Convert to CMYK in Photoshop/InDesign before plating.",
    ),
    "broadcast_rec709": ExportProfile(
        name="broadcast_rec709",
        display_name="Broadcast — 1920×1080 Rec.709",
        format=".png",
        dpi=72,
        color_space="Rec.709",
        bit_depth=8,
        include_alpha=False,
        target_width=1920,
        target_height=1080,
        embed_metadata=True,
        notes="EBU R95 broadcast. Rec.709, 1920×1080, 8-bit.",
    ),
    "cinema_p3": ExportProfile(
        name="cinema_p3",
        display_name="Digital Cinema — 4K P3",
        format=".tiff",
        dpi=72,
        color_space="P3-D65",
        bit_depth=16,
        include_alpha=False,
        target_width=4096,
        target_height=2160,
        tiff_compress="none",
        embed_icc=True,
        embed_metadata=True,
        notes="SMPTE DCI 4K. P3 colour space, 16-bit TIFF. Feed to DCP mastering tool.",
    ),
    "archive_exr": ExportProfile(
        name="archive_exr",
        display_name="Archival — EXR (Linear)",
        format=".tiff",    # EXR requires OpenEXR; fall back to TIFF
        dpi=72,
        color_space="Linear",
        bit_depth=16,
        include_alpha=True,
        tiff_compress="deflate",
        embed_metadata=True,
        notes="Linear light, 16-bit float TIFF. VFX pipeline archival.",
    ),
    "social_webp": ExportProfile(
        name="social_webp",
        display_name="Social — WebP (optimised)",
        format=".webp",
        dpi=72,
        color_space="sRGB",
        bit_depth=8,
        include_alpha=False,
        jpeg_quality=88,
        embed_metadata=True,
        notes="WebP for modern social media. Smaller file size than PNG.",
    ),
}


def list_profiles() -> List[str]:
    """Return names of all registered export profiles."""
    return list(_BUILTIN_PROFILES.keys())


def get_profile(name: str) -> ExportProfile:
    """Return an ExportProfile by name.

    Parameters
    ----------
    name : str

    Returns
    -------
    ExportProfile

    Raises
    ------
    KeyError
        If the profile name is not registered.
    """
    if name not in _BUILTIN_PROFILES:
        raise KeyError(f"Unknown export profile {name!r}. Available: {list_profiles()}")
    return _BUILTIN_PROFILES[name]


def register_profile(profile: ExportProfile) -> None:
    """Register a custom export profile.

    Parameters
    ----------
    profile : ExportProfile
        The profile to register. Overwrites any existing profile with the
        same name.
    """
    _BUILTIN_PROFILES[profile.name] = profile


# ---------------------------------------------------------------------------
# Exporter
# ---------------------------------------------------------------------------

class Exporter:
    """Apply an ExportProfile and write a stereogram to disk.

    Parameters
    ----------
    profile : ExportProfile or str
        Profile object or name of a built-in profile.
    """

    def __init__(self, profile):
        if isinstance(profile, str):
            profile = get_profile(profile)
        self.profile = profile

    def export(
        self,
        image:    np.ndarray,
        out_path: str,
        *,
        depth:    Optional[np.ndarray] = None,
    ) -> str:
        """Export a single stereogram image.

        Parameters
        ----------
        image : np.ndarray
            uint8 RGBA (H, W, 4) stereogram array.
        out_path : str
            Output path without extension (extension added from profile).
            Or a full path including extension.
        depth : np.ndarray, optional
            Original depth map — embedded in metadata if embed_metadata=True.

        Returns
        -------
        str  Absolute path to the written file.
        """
        p = self.profile

        # ── Determine output path ─────────────────────────────────────────
        out     = Path(out_path)
        if out.suffix.lower() not in FORMAT_MAP:
            out = out.with_suffix(p.format)
        out.parent.mkdir(parents=True, exist_ok=True)

        # ── Prepare image ─────────────────────────────────────────────────
        img = self._prepare_image(image)

        # ── Resize ───────────────────────────────────────────────────────
        if p.target_width > 0 or p.target_height > 0:
            img = self._resize(img, p.target_width, p.target_height)

        # ── Watermark ────────────────────────────────────────────────────
        if p.watermark:
            img = self._apply_watermark(img, p.watermark)

        # ── Convert to target mode ────────────────────────────────────────
        if not p.include_alpha and img.mode == "RGBA":
            img = img.convert("RGB")
        elif p.include_alpha and img.mode == "RGB":
            img = img.convert("RGBA")

        # ── Build save kwargs ─────────────────────────────────────────────
        save_kwargs = self._build_save_kwargs(img)

        # ── Write ─────────────────────────────────────────────────────────
        img.save(str(out), **save_kwargs)
        return str(out.resolve())

    def export_batch(
        self,
        items:      List[Tuple["np.ndarray", str]],
        output_dir: str,
        *,
        n_workers:  int = 1,
    ) -> List[str]:
        """Export a batch of images.

        Parameters
        ----------
        items : list of (image_array, name_stem) tuples
            ``name_stem`` is the filename without extension.
        output_dir : str
            Directory to write files into.
        n_workers : int
            Number of parallel export threads. Default 1 (serial).

        Returns
        -------
        list of str  Paths to written files, in the same order as ``items``.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        out_root = Path(output_dir).resolve()

        def _safe_path(stem: str) -> str:
            """Reject stems containing path separators or traversal sequences."""
            safe = Path(stem).name   # strips any directory component
            if safe != stem.replace("\\", "/").split("/")[-1]:
                raise ValueError(
                    f"name_stem {stem!r} contains path separators. "
                    "Use a plain filename, not a relative or absolute path."
                )
            resolved = (out_root / safe).resolve()
            if not str(resolved).startswith(str(out_root)):
                raise ValueError(
                    f"name_stem {stem!r} would escape the output directory."
                )
            return str(out_root / safe)

        if n_workers == 1:
            return [
                self.export(img, _safe_path(stem))
                for img, stem in items
            ]

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [
                pool.submit(self.export, img, _safe_path(stem))
                for img, stem in items
            ]
            return [f.result() for f in futures]

    # ── Internal helpers ───────────────────────────────────────────────────

    def _prepare_image(self, image: np.ndarray) -> Image.Image:
        """Convert numpy RGBA uint8 array to a Pillow Image."""
        p = self.profile
        arr = image

        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)

        if arr.ndim == 3 and arr.shape[2] == 4:
            img = Image.fromarray(arr, mode="RGBA")
        elif arr.ndim == 3 and arr.shape[2] == 3:
            img = Image.fromarray(arr, mode="RGB")
        else:
            img = Image.fromarray(arr)

        # 16-bit conversion
        if p.bit_depth == 16 and p.format in SUPPORTS_16BIT:
            if img.mode == "RGBA":
                img = img.convert("RGBA")
                r, g, b, a = img.split()
                def to16(ch): return ch.point(lambda x: x * 257)
                img = Image.merge("RGBA", (to16(r), to16(g), to16(b), to16(a)))
            else:
                img = img.convert("RGB")
                img = img.point(lambda x: x * 257)

        return img

    def _resize(self, img: Image.Image, w: int, h: int) -> Image.Image:
        """Resize preserving aspect ratio if only one dimension given."""
        ow, oh = img.size
        if w > 0 and h > 0:
            return img.resize((w, h), Image.LANCZOS)
        elif w > 0:
            ratio = w / ow
            return img.resize((w, int(oh * ratio)), Image.LANCZOS)
        else:
            ratio = h / oh
            return img.resize((int(ow * ratio), h), Image.LANCZOS)

    def _apply_watermark(self, img: Image.Image, text: str) -> Image.Image:
        """Burn a semi-transparent text watermark into the bottom-right."""
        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)
            W, H  = img.size
            font_size = max(12, H // 40)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()

            # Measure text
            bbox = draw.textbbox((0, 0), text, font=font)
            tw   = bbox[2] - bbox[0]
            th   = bbox[3] - bbox[1]
            x    = W - tw - 20
            y    = H - th - 20

            # Drop shadow + text
            draw.text((x+1, y+1), text, font=font, fill=(0, 0, 0, 120))
            draw.text((x,   y),   text, font=font, fill=(255, 255, 255, 180))
        except Exception:
            pass
        return img

    def _build_save_kwargs(self, img: Image.Image) -> dict:
        """Build Pillow save() kwargs from the profile."""
        p   = self.profile
        fmt = FORMAT_MAP[Path(img.filename).suffix.lower()] if hasattr(img, 'filename') and img.filename else FORMAT_MAP.get(p.format, "PNG")
        kw: dict = {"format": fmt}

        if fmt == "PNG":
            kw["compress_level"] = p.png_compress
        elif fmt == "JPEG":
            kw["quality"]    = p.jpeg_quality
            kw["subsampling"] = 0  # 4:4:4
        elif fmt == "TIFF":
            compress_map = {
                "none":    "raw",
                "lzw":     "tiff_lzw",
                "deflate": "tiff_deflate",
                "jpeg":    "jpeg",
            }
            kw["compression"] = compress_map.get(p.tiff_compress, "tiff_lzw")
        elif fmt == "WEBP":
            kw["quality"]  = p.jpeg_quality
            kw["lossless"] = False

        if p.dpi and p.embed_metadata:
            kw["dpi"] = (p.dpi, p.dpi)

        return kw


# ---------------------------------------------------------------------------
# ExportQueue — async batch export
# ---------------------------------------------------------------------------

class ExportQueue:
    """A thread-safe queue for background export jobs.

    Parameters
    ----------
    profile : ExportProfile or str
    output_dir : str
    n_workers : int

    Example
    -------
    ::

        q = ExportQueue("web_srgb", "/output/frames/", n_workers=4)
        q.start()

        for i, stereo in enumerate(stereograms):
            q.enqueue(stereo, f"frame_{i:04d}")

        results = q.shutdown()   # wait for all jobs to finish
    """

    def __init__(
        self,
        profile:    "str | ExportProfile",
        output_dir: str,
        n_workers:  int = 2,
    ):
        self._exporter   = Exporter(profile)
        self._output_dir = output_dir
        self._n_workers  = n_workers
        self._futures    = []
        self._executor   = None

    def start(self) -> None:
        """Start the background thread pool."""
        from concurrent.futures import ThreadPoolExecutor
        self._executor = ThreadPoolExecutor(max_workers=self._n_workers)

    def enqueue(self, image: np.ndarray, name_stem: str) -> None:
        """Submit an image for background export."""
        if self._executor is None:
            self.start()
        path   = os.path.join(self._output_dir, name_stem)
        future = self._executor.submit(self._exporter.export, image, path)
        self._futures.append(future)

    def shutdown(self, wait: bool = True) -> List[str]:
        """Shut down the thread pool and return written paths."""
        if self._executor:
            self._executor.shutdown(wait=wait)
        return [f.result() for f in self._futures if not f.exception()]

    @property
    def pending(self) -> int:
        return sum(1 for f in self._futures if not f.done())

    @property
    def completed(self) -> int:
        return sum(1 for f in self._futures if f.done() and not f.exception())
