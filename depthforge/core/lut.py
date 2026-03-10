"""
depthforge.core.lut
====================
LUT (Look-Up Table) support for colour grading the pattern layer.

Applying a colour grade to the pattern tile before stereogram synthesis
is a fast way to create cinematic or branded looks without touching the
underlying procedural generation. This module handles:

- Loading 1D and 3D LUTs in .cube format (the dominant interchange format)
- Applying 3D LUTs to pattern tiles or any RGBA image via trilinear
  interpolation
- A built-in library of 8 creative LUTs (generated algorithmically)
- LUT preview generation

Cube format support
-------------------
Implements the Adobe/Resolve .cube spec:
    LUT_1D_SIZE  / LUT_3D_SIZE
    DOMAIN_MIN / DOMAIN_MAX
    Comment lines (#)
    Data lines (R G B floats)

Public API
----------
``LUTType``              Enum: LUT_1D, LUT_3D.
``LUT``                  Dataclass holding table data + metadata.
``load_cube(path)``      Parse a .cube file. Returns LUT.
``apply_lut(image, lut)``  Apply LUT to RGBA uint8. Returns RGBA uint8.
``list_builtin_luts()``  Returns list of built-in LUT names.
``get_builtin_lut(name)``  Returns a built-in LUT.
``save_cube(lut, path)`` Save LUT to .cube file.
``LUTCache``             Thread-safe LUT cache (keyed by file path).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import numpy as np

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class LUTType(Enum):
    LUT_1D = "1D"
    LUT_3D = "3D"


@dataclass
class LUT:
    """
    A colour look-up table.

    Attributes
    ----------
    name : str
        Human-readable name.
    lut_type : LUTType
    size : int
        Number of entries per axis (1D) or per side (3D).
    table : np.ndarray
        For 1D: (size, 3) float32 in [0, 1].
        For 3D: (size, size, size, 3) float32 in [0, 1].
        Axes order: R, G, B (standard .cube layout).
    domain_min : np.ndarray (3,) float32
    domain_max : np.ndarray (3,) float32
    description : str
    """

    name: str
    lut_type: LUTType
    size: int
    table: np.ndarray
    domain_min: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    domain_max: np.ndarray = field(default_factory=lambda: np.ones(3, np.float32))
    description: str = ""


# ---------------------------------------------------------------------------
# .cube parser
# ---------------------------------------------------------------------------


def load_cube(path: Union[str, Path]) -> LUT:
    """
    Parse an Adobe/Resolve .cube LUT file.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    LUT

    Raises
    ------
    ValueError  for malformed files.
    FileNotFoundError  if path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LUT file not found: {path}")

    lut_type: Optional[LUTType] = None
    size: Optional[int] = None
    domain_min = np.zeros(3, np.float32)
    domain_max = np.ones(3, np.float32)
    name = path.stem
    description = ""
    data_rows: list[list[float]] = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                if line.startswith("# "):
                    description += line[2:] + "\n"
                continue

            upper = line.upper()

            if upper.startswith("LUT_1D_SIZE"):
                lut_type = LUTType.LUT_1D
                size = int(line.split()[1])
            elif upper.startswith("LUT_3D_SIZE"):
                lut_type = LUTType.LUT_3D
                size = int(line.split()[1])
            elif upper.startswith("TITLE"):
                name = line.split(None, 1)[1].strip('"')
            elif upper.startswith("DOMAIN_MIN"):
                parts = line.split()
                domain_min = np.array(
                    [float(parts[1]), float(parts[2]), float(parts[3])], np.float32
                )
            elif upper.startswith("DOMAIN_MAX"):
                parts = line.split()
                domain_max = np.array(
                    [float(parts[1]), float(parts[2]), float(parts[3])], np.float32
                )
            else:
                # Data line
                try:
                    row = [float(v) for v in line.split()]
                    if len(row) == 3:
                        data_rows.append(row)
                except ValueError:
                    pass  # ignore non-numeric lines

    if lut_type is None or size is None:
        raise ValueError(f"No LUT_1D_SIZE or LUT_3D_SIZE found in: {path}")

    data = np.array(data_rows, dtype=np.float32)

    if lut_type == LUTType.LUT_1D:
        expected = size
        if len(data) != expected:
            raise ValueError(f"1D LUT expected {expected} rows, got {len(data)}")
        table = data  # (size, 3)
    else:
        expected = size**3
        if len(data) != expected:
            raise ValueError(f"3D LUT expected {expected} rows, got {len(data)}")
        # Reshape to (R, G, B, 3) — .cube iterates B fastest
        table = data.reshape(size, size, size, 3)

    return LUT(
        name=name,
        lut_type=lut_type,
        size=size,
        table=table,
        domain_min=domain_min,
        domain_max=domain_max,
        description=description.strip(),
    )


# ---------------------------------------------------------------------------
# LUT application
# ---------------------------------------------------------------------------


def _apply_1d_lut(
    rgb: np.ndarray,  # (H, W, 3) float32 in [0, 1]
    lut: LUT,
) -> np.ndarray:
    """Apply 1D LUT via linear interpolation per channel."""
    size = lut.size
    indices = np.clip(rgb * (size - 1), 0, size - 1)
    idx_lo = indices.astype(np.int32)
    idx_hi = np.minimum(idx_lo + 1, size - 1)
    frac = indices - idx_lo

    out = np.empty_like(rgb)
    for c in range(3):
        lo = lut.table[idx_lo[..., c], c]
        hi = lut.table[idx_hi[..., c], c]
        out[..., c] = lo + frac[..., c] * (hi - lo)

    return out.astype(np.float32)


def _apply_3d_lut(
    rgb: np.ndarray,  # (H, W, 3) float32 in [0, 1]
    lut: LUT,
) -> np.ndarray:
    """Apply 3D LUT via trilinear interpolation."""
    size = lut.size
    N = size - 1

    # Scale input to [0, N]
    coords = np.clip(rgb * N, 0.0, float(N))
    r_idx = coords[..., 0]
    g_idx = coords[..., 1]
    b_idx = coords[..., 2]

    r0 = r_idx.astype(np.int32).clip(0, N)
    g0 = g_idx.astype(np.int32).clip(0, N)
    b0 = b_idx.astype(np.int32).clip(0, N)
    r1 = np.minimum(r0 + 1, N)
    g1 = np.minimum(g0 + 1, N)
    b1 = np.minimum(b0 + 1, N)

    fr = (r_idx - r0).astype(np.float32)
    fg = (g_idx - g0).astype(np.float32)
    fb = (b_idx - b0).astype(np.float32)

    t = lut.table
    # Trilinear: 8 corners
    c000 = t[r0, g0, b0]
    c001 = t[r0, g0, b1]
    c010 = t[r0, g1, b0]
    c011 = t[r0, g1, b1]
    c100 = t[r1, g0, b0]
    c101 = t[r1, g0, b1]
    c110 = t[r1, g1, b0]
    c111 = t[r1, g1, b1]

    fr_ = fr[..., np.newaxis]
    fg_ = fg[..., np.newaxis]
    fb_ = fb[..., np.newaxis]

    c00 = c000 * (1 - fb_) + c001 * fb_
    c01 = c010 * (1 - fb_) + c011 * fb_
    c10 = c100 * (1 - fb_) + c101 * fb_
    c11 = c110 * (1 - fb_) + c111 * fb_

    c0 = c00 * (1 - fg_) + c01 * fg_
    c1 = c10 * (1 - fg_) + c11 * fg_

    result = c0 * (1 - fr_) + c1 * fr_
    return result.astype(np.float32)


def apply_lut(
    image: np.ndarray,  # RGBA or RGB uint8
    lut: LUT,
    strength: float = 1.0,
) -> np.ndarray:
    """
    Apply a LUT to an image.

    Parameters
    ----------
    image : np.ndarray
        (H, W, 3 or 4) uint8 input.
    lut : LUT
    strength : float
        Blend factor 0 = original, 1 = full LUT. Default 1.0.

    Returns
    -------
    np.ndarray
        Same shape and dtype as input.
    """
    has_alpha = image.ndim == 3 and image.shape[2] == 4
    rgb = image[..., :3].astype(np.float32) / 255.0

    # Normalise to [0, 1] based on domain
    dmin = lut.domain_min
    dmax = lut.domain_max
    drange = dmax - dmin
    drange[drange == 0] = 1.0
    rgb_norm = (rgb - dmin) / drange
    rgb_norm = rgb_norm.clip(0.0, 1.0)

    if lut.lut_type == LUTType.LUT_1D:
        graded = _apply_1d_lut(rgb_norm, lut)
    else:
        graded = _apply_3d_lut(rgb_norm, lut)

    # Denormalise back
    graded_out = graded * drange + dmin

    # Blend with original
    if strength < 1.0:
        graded_out = rgb_norm * (1 - strength) + graded_out * strength

    rgb_out = (graded_out * 255.0).clip(0, 255).astype(np.uint8)

    if has_alpha:
        return np.concatenate([rgb_out, image[..., 3:4]], axis=2)
    return rgb_out


# ---------------------------------------------------------------------------
# .cube writer
# ---------------------------------------------------------------------------


def save_cube(lut: LUT, path: Union[str, Path]) -> Path:
    """
    Save a LUT to .cube format.

    Returns the path written.
    """
    path = Path(path).with_suffix(".cube")
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f'TITLE "{lut.name}"',
        f"# {lut.description}" if lut.description else "# Created by DepthForge",
        "",
        f"DOMAIN_MIN {lut.domain_min[0]:.6f} {lut.domain_min[1]:.6f} {lut.domain_min[2]:.6f}",
        f"DOMAIN_MAX {lut.domain_max[0]:.6f} {lut.domain_max[1]:.6f} {lut.domain_max[2]:.6f}",
        "",
    ]

    if lut.lut_type == LUTType.LUT_1D:
        lines.append(f"LUT_1D_SIZE {lut.size}")
        lines.append("")
        for row in lut.table:
            lines.append(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f}")
    else:
        lines.append(f"LUT_3D_SIZE {lut.size}")
        lines.append("")
        flat = lut.table.reshape(-1, 3)
        for row in flat:
            lines.append(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f}")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# Built-in LUT library (algorithmically generated)
# ---------------------------------------------------------------------------


def _make_identity_3d(size: int = 33) -> LUT:
    """Identity LUT — no change."""
    N = size
    r = np.linspace(0, 1, N)
    g = np.linspace(0, 1, N)
    b = np.linspace(0, 1, N)
    RR, GG, BB = np.meshgrid(r, g, b, indexing="ij")
    table = np.stack([RR, GG, BB], axis=-1).astype(np.float32)
    return LUT(
        name="identity",
        lut_type=LUTType.LUT_3D,
        size=N,
        table=table,
        description="Identity — no colour change",
    )


def _make_warm_shadows(size: int = 33) -> LUT:
    """Add warm orange to shadows, cool cyan to highlights."""
    lut = _make_identity_3d(size)
    t = lut.table.copy()
    # Luminance proxy per entry
    lum = 0.299 * t[..., 0] + 0.587 * t[..., 1] + 0.114 * t[..., 2]
    shadow = (1.0 - lum).clip(0, 1) * 0.12
    highlight = lum.clip(0, 1) * 0.05
    t[..., 0] += shadow - highlight * 0.5
    t[..., 1] += shadow * 0.3 - highlight * 0.2
    t[..., 2] -= shadow * 0.4 - highlight
    lut.table = t.clip(0, 1).astype(np.float32)
    lut.name = "warm_shadows"
    lut.description = "Warm orange shadows, cool cyan highlights"
    return lut


def _make_cold_steel(size: int = 33) -> LUT:
    """Desaturate + push cool blue-grey tone."""
    lut = _make_identity_3d(size)
    t = lut.table.copy()
    lum = (0.299 * t[..., 0] + 0.587 * t[..., 1] + 0.114 * t[..., 2])[..., np.newaxis]
    desat = t * 0.4 + lum * 0.6
    desat[..., 2] = (desat[..., 2] * 1.12).clip(0, 1)
    lut.table = desat.clip(0, 1).astype(np.float32)
    lut.name = "cold_steel"
    lut.description = "Desaturated cool steel look"
    return lut


def _make_neon_boost(size: int = 33) -> LUT:
    """Boost saturation and push secondary colours toward neon."""
    lut = _make_identity_3d(size)
    t = lut.table.copy()
    lum = (0.299 * t[..., 0] + 0.587 * t[..., 1] + 0.114 * t[..., 2])[..., np.newaxis]
    sat = t - lum
    t_out = lum + sat * 1.6
    lut.table = t_out.clip(0, 1).astype(np.float32)
    lut.name = "neon_boost"
    lut.description = "High-saturation neon look"
    return lut


def _make_sepia(size: int = 33) -> LUT:
    lut = _make_identity_3d(size)
    t = lut.table.copy()
    lum = 0.299 * t[..., 0] + 0.587 * t[..., 1] + 0.114 * t[..., 2]
    t[..., 0] = (lum * 1.08).clip(0, 1)
    t[..., 1] = (lum * 0.88).clip(0, 1)
    t[..., 2] = (lum * 0.68).clip(0, 1)
    lut.table = t.astype(np.float32)
    lut.name = "sepia"
    lut.description = "Classic sepia tone"
    return lut


def _make_inverse(size: int = 33) -> LUT:
    lut = _make_identity_3d(size)
    lut.table = (1.0 - lut.table).astype(np.float32)
    lut.name = "inverse"
    lut.description = "Colour inversion"
    return lut


def _make_psychedelic(size: int = 33) -> LUT:
    """Hue rotation + saturation boost → vivid multi-colour result."""
    lut = _make_identity_3d(size)
    t = lut.table.copy()
    # Rotate hues by shifting channels cyclically with a non-linear twist
    r, g, b = t[..., 0], t[..., 1], t[..., 2]
    t[..., 0] = (r * 0.3 + g * 0.7).clip(0, 1)
    t[..., 1] = (g * 0.2 + b * 0.8).clip(0, 1)
    t[..., 2] = (b * 0.4 + r * 0.6).clip(0, 1)
    lut.table = t.astype(np.float32)
    lut.name = "psychedelic"
    lut.description = "Vivid hue-shift psychedelic look"
    return lut


def _make_forest(size: int = 33) -> LUT:
    """Push greens and desaturate reds/blues for a natural forest feel."""
    lut = _make_identity_3d(size)
    t = lut.table.copy()
    t[..., 0] = (t[..., 0] * 0.85).clip(0, 1)
    t[..., 1] = (t[..., 1] * 1.12).clip(0, 1)
    t[..., 2] = (t[..., 2] * 0.90).clip(0, 1)
    lut.table = t.astype(np.float32)
    lut.name = "forest"
    lut.description = "Natural forest green tone"
    return lut


_BUILTIN_LUTS: dict[str, LUT] = {
    "identity": _make_identity_3d(),
    "warm_shadows": _make_warm_shadows(),
    "cold_steel": _make_cold_steel(),
    "neon_boost": _make_neon_boost(),
    "sepia": _make_sepia(),
    "inverse": _make_inverse(),
    "psychedelic": _make_psychedelic(),
    "forest": _make_forest(),
}


def list_builtin_luts() -> list[str]:
    """Return names of all built-in LUTs."""
    return list(_BUILTIN_LUTS.keys())


def get_builtin_lut(name: str) -> LUT:
    """
    Return a built-in LUT by name.

    Raises KeyError if not found.
    """
    name = name.lower().strip()
    if name not in _BUILTIN_LUTS:
        raise KeyError(f"Unknown built-in LUT '{name}'. " f"Available: {list_builtin_luts()}")
    return _BUILTIN_LUTS[name]


# ---------------------------------------------------------------------------
# Thread-safe LUT cache
# ---------------------------------------------------------------------------


class LUTCache:
    """Thread-safe LRU cache for file-loaded LUTs."""

    def __init__(self, max_luts: int = 16):
        from collections import OrderedDict

        self._store: OrderedDict[str, LUT] = OrderedDict()
        self._max = max_luts
        self._lock = threading.Lock()

    def get(self, path: Union[str, Path]) -> LUT:
        key = str(Path(path).resolve())
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                return self._store[key]
        lut = load_cube(path)
        with self._lock:
            self._store[key] = lut
            if len(self._store) > self._max:
                self._store.popitem(last=False)
        return lut

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        return len(self._store)
