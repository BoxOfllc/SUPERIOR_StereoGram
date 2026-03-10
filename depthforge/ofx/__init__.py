"""
depthforge.ofx — Python-side OFX bridge
========================================
This module provides the Python half of the DepthForge OFX plugin bridge.

The C++ plugin (plugin.cpp) invokes::

    python3 -m depthforge.cli.bridge --depth <path> --out <path> [flags]

This module is *also* importable for direct use from Python pipelines that
need to call the OFX plugin programmatically, or to test the bridge without
a host application.

Components
----------
``plugin.h / plugin.cpp``   C++ OFX plugin (compiled separately via CMake)
``CMakeLists.txt``          Build system for the .ofx.bundle
``__init__.py``             This file — Python bridge utilities

Public API
----------
::

    from depthforge.ofx import OFXBridge, OFXParams, build_bundle, check_build

    params  = OFXParams(depth_factor=0.35, preset="cinema")
    bridge  = OFXBridge()
    result  = bridge.synthesize(depth_array, params=params)

    # Build the C++ plugin
    success = build_bundle()
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# OFX bundle/plugin paths
_PLUGIN_DIR = Path(__file__).parent
_CMAKE_DIR  = _PLUGIN_DIR


# ---------------------------------------------------------------------------
# OFXParams — Python mirror of the OFX parameter set
# ---------------------------------------------------------------------------

@dataclass
class OFXParams:
    """Parameters exposed by the DepthForge OFX plugin.

    These match the parameter identifiers in plugin.h exactly.

    Attributes
    ----------
    mode : str
        Synthesis mode: ``"sirds"``, ``"anaglyph"``, ``"stereo_pair"``,
        ``"hidden_image"``, ``"adaptive_dots"``.
    preset : str
        Named preset: ``"shallow"``, ``"medium"``, ``"deep"``, ``"cinema"``,
        ``"print"``, ``"broadcast"``. Overrides per-parameter defaults.
    depth_factor : float
    max_parallax : float
    safe_mode : bool
    oversample : int
    seed : int
    invert_depth : bool
    bilateral_space : float
    bilateral_color : float
    dilation_px : int
    near_plane : float
    far_plane : float
    pattern_type : str
    pattern_name : str
    color_mode : str
    tile_width : int
    tile_height : int
    pattern_scale : float
    anaglyph_mode : str
    swap_eyes : bool
    gamma : float
    safety_profile : str
    auto_safe : bool
    export_profile : str
    """

    mode:            str   = "sirds"
    preset:          str   = ""
    depth_factor:    float = 0.35
    max_parallax:    float = 0.033
    safe_mode:       bool  = False
    oversample:      int   = 1
    seed:            int   = 42
    invert_depth:    bool  = False
    bilateral_space: float = 5.0
    bilateral_color: float = 0.1
    dilation_px:     int   = 3
    near_plane:      float = 0.0
    far_plane:       float = 1.0
    pattern_type:    str   = "random_noise"
    pattern_name:    str   = ""
    color_mode:      str   = "rgb"
    tile_width:      int   = 128
    tile_height:     int   = 128
    pattern_scale:   float = 1.0
    anaglyph_mode:   str   = "color"
    swap_eyes:       bool  = False
    gamma:           float = 1.0
    safety_profile:  str   = "default"
    auto_safe:       bool  = True
    export_profile:  str   = "web_srgb"

    def to_cli_args(self) -> list:
        """Convert to a list of CLI argument strings for depthforge.cli.bridge."""
        args = []

        def flag(name, val):
            if isinstance(val, bool):
                if val: args.append(f"--{name}")
            elif isinstance(val, float):
                args.extend([f"--{name}", f"{val:.6f}"])
            else:
                if str(val): args.extend([f"--{name}", str(val)])

        if self.preset:
            flag("preset", self.preset)
        else:
            flag("mode",            self.mode)
            flag("depth-factor",    self.depth_factor)
            flag("max-parallax",    self.max_parallax)
            flag("safe-mode",       self.safe_mode)
            flag("oversample",      self.oversample)
            flag("seed",            self.seed)
            flag("invert-depth",    self.invert_depth)
            flag("bilateral-space", self.bilateral_space)
            flag("bilateral-color", self.bilateral_color)
            flag("dilation-px",     self.dilation_px)
            flag("near-plane",      self.near_plane)
            flag("far-plane",       self.far_plane)
            flag("pattern-type",    self.pattern_type)
            if self.pattern_name:
                flag("pattern-name", self.pattern_name)
            flag("color-mode",      self.color_mode)
            flag("tile-width",      self.tile_width)
            flag("tile-height",     self.tile_height)
            flag("pattern-scale",   self.pattern_scale)
            flag("anaglyph-mode",   self.anaglyph_mode)
            flag("swap-eyes",       self.swap_eyes)
            flag("gamma",           self.gamma)
            flag("safety-profile",  self.safety_profile)
            flag("auto-safe",       self.auto_safe)

        flag("export-profile", self.export_profile)
        return args

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_preset(cls, name: str) -> "OFXParams":
        """Build params from a named preset, keeping all other defaults."""
        return cls(preset=name)


# ---------------------------------------------------------------------------
# OFXBridge — Python bridge to the DepthForge synthesis core
# ---------------------------------------------------------------------------

class OFXBridge:
    """Python bridge that mimics what the C++ OFX plugin does at render time.

    Writes depth (and optionally source) frames to temp files, calls the
    synthesis core, and returns the result as a numpy array.

    This class is useful for:
    - Testing the full OFX pipeline from Python
    - Batch rendering without a host application
    - Embedding OFX-compatible parameter handling in Python pipelines

    Parameters
    ----------
    python_exe : str
        Python interpreter to use. Defaults to the current interpreter.
    timeout : int
        Maximum seconds for a synthesis call.
    tmp_dir : str
        Directory for temporary frame files.
    """

    def __init__(
        self,
        python_exe: str = "",
        timeout:    int = 60,
        tmp_dir:    str = "",
    ):
        import sys, shutil
        exe = python_exe or sys.executable
        # Security: validate the executable is a real file, not a shell command
        resolved = shutil.which(exe)
        if resolved is None:
            raise ValueError(
                f"python_exe {exe!r} not found on PATH. "
                "Pass an absolute path to a Python interpreter."
            )
        self.python_exe = resolved
        # Clamp timeout to a sane range
        self.timeout    = max(5, min(int(timeout), 600))
        self.tmp_dir    = tmp_dir or tempfile.gettempdir()

    def synthesize(
        self,
        depth:   np.ndarray,
        source:  Optional[np.ndarray] = None,
        params:  Optional[OFXParams]  = None,
    ) -> np.ndarray:
        """Synthesize a stereogram from a depth map.

        Parameters
        ----------
        depth : np.ndarray
            float32 (H, W) depth map in [0, 1], or uint8 (H, W) / (H, W, 1).
        source : np.ndarray, optional
            RGB(A) source frame for anaglyph/stereo-pair modes.
        params : OFXParams, optional
            Synthesis parameters. Defaults to ``OFXParams()``.

        Returns
        -------
        np.ndarray  uint8 (H, W, 4) RGBA stereogram.
        """
        if params is None:
            params = OFXParams()

        # ── Write inputs to temp files ──────────────────────────────────
        td   = Path(self.tmp_dir)
        d_path = str(td / "df_bridge_depth.png")
        o_path = str(td / "df_bridge_out.png")
        s_path = str(td / "df_bridge_source.png") if source is not None else ""

        _save_depth_png(depth, d_path)
        if source is not None and s_path:
            _save_frame_png(source, s_path)

        # ── Build and run command ───────────────────────────────────────
        cmd = (
            [self.python_exe, "-m", "depthforge.cli.bridge",
             "--depth", d_path, "--out", o_path]
            + params.to_cli_args()
            + (["--source", s_path] if s_path else [])
        )

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"OFX bridge synthesis failed:\n"
                f"  stderr: {result.stderr.strip()}\n"
                f"  stdout: {result.stdout.strip()}"
            )

        # ── Load result ─────────────────────────────────────────────────
        return np.array(Image.open(o_path).convert("RGBA"))

    def synthesize_direct(
        self,
        depth:  np.ndarray,
        source: Optional[np.ndarray] = None,
        params: Optional[OFXParams]  = None,
    ) -> np.ndarray:
        """Synthesize in-process without subprocess overhead.

        Uses the Python core directly (no CLI round-trip). This is
        functionally equivalent to ``synthesize()`` but faster and requires
        no temp files.

        Parameters
        ----------
        depth, source, params : same as ``synthesize()``.

        Returns
        -------
        np.ndarray  uint8 (H, W, 4)
        """
        from depthforge.core.synthesizer import synthesize, StereoParams
        from depthforge.core.depth_prep  import prep_depth, DepthPrepParams
        from depthforge.core.pattern_gen import generate_pattern, PatternParams, PatternType, ColorMode
        from depthforge.core.presets     import get_preset

        if params is None:
            params = OFXParams()

        # Depth prep
        prep_p = DepthPrepParams(
            bilateral_sigma_space=params.bilateral_space,
            bilateral_sigma_color=params.bilateral_color,
            dilation_px=params.dilation_px,
            near_plane=params.near_plane,
            far_plane=params.far_plane,
            invert=params.invert_depth,
        )
        depth_f = prep_depth(depth.astype(np.float32), prep_p)

        # Pattern
        try:
            pt = PatternType[params.pattern_type.upper()]
        except KeyError:
            pt = PatternType.RANDOM_NOISE
        try:
            cm = ColorMode[params.color_mode.upper()]
        except KeyError:
            cm = ColorMode.MONOCHROME

        pat_p   = PatternParams(
            pattern_type=pt,
            color_mode=cm,
            tile_width=params.tile_width,
            tile_height=params.tile_height,
            seed=params.seed,
            scale=params.pattern_scale,
        )
        pattern = generate_pattern(pat_p)

        # Stereo params
        if params.preset:
            preset   = get_preset(params.preset)
            sp       = preset.stereo_params
            stereo_p = StereoParams(
                depth_factor=sp.depth_factor,
                max_parallax_fraction=sp.max_parallax_fraction,
                oversample=sp.oversample,
                safe_mode=sp.safe_mode,
                seed=params.seed,
            )
        else:
            stereo_p = StereoParams(
                depth_factor=params.depth_factor,
                max_parallax_fraction=params.max_parallax,
                oversample=params.oversample,
                safe_mode=params.safe_mode,
                seed=params.seed,
            )

        return synthesize(depth_f, pattern, stereo_p)


# ---------------------------------------------------------------------------
# Build utilities
# ---------------------------------------------------------------------------

def check_build() -> dict:
    """Check whether the compiled OFX plugin exists.

    Returns
    -------
    dict  ``{"built": bool, "path": str or None, "error": str}``
    """
    bundle = _PLUGIN_DIR.parent.parent / "build" / "DepthForge.ofx.bundle"
    if bundle.exists():
        arch = next(bundle.glob("Contents/*/DepthForge.ofx"), None)
        return {"built": arch is not None, "path": str(arch) if arch else None, "error": ""}
    return {"built": False, "path": None, "error": "Bundle not built. Run build_bundle()."}


def build_bundle(
    cmake_build_type: str = "Release",
    n_jobs:           int = 0,
    extra_cmake_args: list = None,
) -> bool:
    """Build the OFX plugin bundle using CMake.

    Requires cmake and a C++17 compiler.

    Parameters
    ----------
    cmake_build_type : str
        ``"Release"``, ``"Debug"``, ``"RelWithDebInfo"``.
    n_jobs : int
        Parallel build jobs. 0 = auto.
    extra_cmake_args : list of str
        Additional CMake -D flags.

    Returns
    -------
    bool  True on success.
    """
    build_dir = _PLUGIN_DIR.parent.parent / "build" / "ofx_cmake"
    build_dir.mkdir(parents=True, exist_ok=True)

    cmake_args = [
        "cmake",
        str(_CMAKE_DIR),
        f"-DCMAKE_BUILD_TYPE={cmake_build_type}",
    ] + (extra_cmake_args or [])

    r = subprocess.run(cmake_args, cwd=str(build_dir), capture_output=True)
    if r.returncode != 0:
        return False

    build_args = ["cmake", "--build", ".", "--target", "depthforge_ofx"]
    if n_jobs > 0:
        build_args += ["--", f"-j{n_jobs}"]

    r = subprocess.run(build_args, cwd=str(build_dir), capture_output=True)
    return r.returncode == 0


def plugin_count() -> int:
    """Return number of OFX plugins provided (always 1)."""
    return 1


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_depth_png(depth: np.ndarray, path: str) -> None:
    """Save a float32 depth map as 16-bit greyscale PNG."""
    arr = depth
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    arr = np.clip(arr, 0, 1)
    img = Image.fromarray((arr * 65535).astype(np.uint16), mode="I;16")
    img.save(path)


def _save_frame_png(frame: np.ndarray, path: str) -> None:
    """Save an RGB/RGBA uint8 frame as PNG."""
    if frame.dtype != np.uint8:
        frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
    if frame.ndim == 2:
        Image.fromarray(frame, mode="L").save(path)
    elif frame.shape[2] == 4:
        Image.fromarray(frame, mode="RGBA").save(path)
    else:
        Image.fromarray(frame, mode="RGB").save(path)
