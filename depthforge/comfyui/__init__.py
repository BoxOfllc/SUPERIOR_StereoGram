"""
depthforge.comfyui
==================
ComfyUI integration for DepthForge.

Installation
------------
1. Copy or symlink this depthforge directory into ``ComfyUI/custom_nodes/``:

   cp -r /path/to/depthforge  ComfyUI/custom_nodes/
   # OR
   ln -s /path/to/depthforge  ComfyUI/custom_nodes/depthforge

2. Restart ComfyUI. The DepthForge node group will appear in the node browser
   under the "DepthForge" category.

3. Alternatively, call ``depthforge.comfyui.install(comfyui_path)`` to install
   automatically.

Nodes
-----
DF_DepthPrep        Condition depth maps (smooth, dilate, curve, clamp)
DF_PatternGen       Procedural pattern tile generator (7 types)
DF_PatternLibrary   Named pattern library (28+ patterns)
DF_Stereogram       Core SIRDS/texture synthesis
DF_AnaglyphOut      Red/cyan anaglyph output
DF_StereoPair       L/R stereo pair with occlusion mask
DF_HiddenImage      Hidden shape/text stereogram
DF_SafetyLimiter    Comfort safety limiter
DF_QCOverlay        Parallax heatmap and violation overlays
DF_VideoSequence    Per-frame video stereogram with temporal smoothing
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Import all node classes
# ---------------------------------------------------------------------------

try:
    from depthforge.comfyui.nodes import (
        DF_DepthPrep,
        DF_PatternGen,
        DF_PatternLibrary,
        DF_Stereogram,
        DF_AnaglyphOut,
        DF_StereoPair,
        DF_HiddenImage,
        DF_SafetyLimiter,
        DF_QCOverlay,
        DF_VideoSequence,
    )
    _NODES_LOADED = True
except ImportError as e:
    _NODES_LOADED = False
    _NODES_ERROR  = str(e)


# ---------------------------------------------------------------------------
# ComfyUI registration mappings
# ---------------------------------------------------------------------------

if _NODES_LOADED:
    NODE_CLASS_MAPPINGS = {
        "DF_DepthPrep":      DF_DepthPrep,
        "DF_PatternGen":     DF_PatternGen,
        "DF_PatternLibrary": DF_PatternLibrary,
        "DF_Stereogram":     DF_Stereogram,
        "DF_AnaglyphOut":    DF_AnaglyphOut,
        "DF_StereoPair":     DF_StereoPair,
        "DF_HiddenImage":    DF_HiddenImage,
        "DF_SafetyLimiter":  DF_SafetyLimiter,
        "DF_QCOverlay":      DF_QCOverlay,
        "DF_VideoSequence":  DF_VideoSequence,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "DF_DepthPrep":      "DF Depth Prep",
        "DF_PatternGen":     "DF Pattern Gen",
        "DF_PatternLibrary": "DF Pattern Library",
        "DF_Stereogram":     "DF Stereogram",
        "DF_AnaglyphOut":    "DF Anaglyph Out",
        "DF_StereoPair":     "DF Stereo Pair",
        "DF_HiddenImage":    "DF Hidden Image",
        "DF_SafetyLimiter":  "DF Safety Limiter",
        "DF_QCOverlay":      "DF QC Overlay",
        "DF_VideoSequence":  "DF Video Sequence",
    }
else:
    NODE_CLASS_MAPPINGS      = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    import warnings
    warnings.warn(
        f"DepthForge ComfyUI nodes failed to load: {_NODES_ERROR}",
        ImportWarning,
        stacklevel=1,
    )


# ---------------------------------------------------------------------------
# Install helper
# ---------------------------------------------------------------------------

def install(comfyui_path: str) -> None:
    """Install the DepthForge ComfyUI extension into a ComfyUI installation.

    Creates a symlink or copies the depthforge package into the
    ``custom_nodes`` directory of the specified ComfyUI installation.

    Parameters
    ----------
    comfyui_path : str
        Root path of the ComfyUI installation.

    Example
    -------
    ::

        import depthforge.comfyui
        depthforge.comfyui.install("/home/user/ComfyUI")
    """
    comfyui = Path(comfyui_path)
    if not comfyui.exists():
        raise FileNotFoundError(f"ComfyUI path not found: {comfyui_path}")

    custom_nodes = comfyui / "custom_nodes"
    custom_nodes.mkdir(exist_ok=True)

    # Source: the depthforge package directory
    src = Path(__file__).parent.parent    # depthforge/
    dst = custom_nodes / "depthforge"

    if dst.exists() or dst.is_symlink():
        print(f"DepthForge already present at {dst}. Removing and reinstalling.")
        if dst.is_symlink():
            dst.unlink()
        else:
            shutil.rmtree(dst)

    # Prefer symlink (keeps package in sync); fall back to copy
    try:
        dst.symlink_to(src)
        print(f"✓ Symlinked: {dst} → {src}")
    except (OSError, NotImplementedError):
        shutil.copytree(src, dst)
        print(f"✓ Copied: {src} → {dst}")

    print(
        "Restart ComfyUI to load the DepthForge nodes.\n"
        "Nodes will appear in the node browser under 'DepthForge'."
    )


def uninstall(comfyui_path: str) -> None:
    """Remove the DepthForge ComfyUI extension.

    Parameters
    ----------
    comfyui_path : str
        Root path of the ComfyUI installation.
    """
    dst = Path(comfyui_path) / "custom_nodes" / "depthforge"
    if dst.is_symlink():
        dst.unlink()
        print(f"✓ Removed symlink: {dst}")
    elif dst.is_dir():
        shutil.rmtree(dst)
        print(f"✓ Removed: {dst}")
    else:
        print(f"DepthForge not found at {dst}")


def node_count() -> int:
    """Return the number of registered nodes."""
    return len(NODE_CLASS_MAPPINGS)
