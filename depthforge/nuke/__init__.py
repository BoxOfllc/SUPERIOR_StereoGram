"""
depthforge.nuke
================
Nuke integration for DepthForge.

Provides a Python gizmo, toolbar menu, and render callbacks that
expose the full DepthForge stereogram pipeline inside Nuke/NukeX/NukeStudio.

Installation
------------
Add to ``~/.nuke/init.py`` (or a facility-wide ``init.py``):

    import sys
    sys.path.insert(0, "/path/to/depthforge/parent_dir")
    import depthforge.nuke
    depthforge.nuke.install()

That single call registers the toolbar menu, creates the gizmo template,
and hooks in the knobChanged / beforeRender callbacks.

Alternatively, call from the Nuke Script Editor at any time:

    import depthforge.nuke
    depthforge.nuke.install()

Manual usage (no Nuke required — pure Python)
---------------------------------------------
    from depthforge.nuke import DepthForgeGizmo
    import numpy as np

    gizmo = DepthForgeGizmo({"df_mode": "Anaglyph", "df_depth_factor": 0.4})
    depth = np.random.rand(1080, 1920).astype(np.float32)
    result = gizmo.render(depth)   # → RGBA uint8 (1080, 1920, 4)

Public API
----------
install(menu_name)      Register toolbar menu and callbacks in Nuke
uninstall()             Remove DepthForge menu entries
create_node(mode)       Create a DepthForge gizmo node in the node graph
DepthForgeGizmo         Pure-Python gizmo class (no Nuke required)
STEREOGRAM_KNOBS        Full knob specification list
"""

from __future__ import annotations

from depthforge.nuke.gizmo import (
    DepthForgeGizmo,
    STEREOGRAM_KNOBS,
    _NUKE_AVAILABLE,
)

__all__ = [
    "install",
    "uninstall",
    "create_node",
    "DepthForgeGizmo",
    "STEREOGRAM_KNOBS",
]

_MENU_INSTALLED = False
_MENU_REF       = None   # nuke.Menu reference kept alive


def install(menu_name: str = "DepthForge") -> None:
    """Register DepthForge in the Nuke toolbar and set up render callbacks.

    Safe to call multiple times — subsequent calls are no-ops.

    Parameters
    ----------
    menu_name : str
        Name of the top-level toolbar menu. Default ``"DepthForge"``.
    """
    global _MENU_INSTALLED, _MENU_REF

    if _MENU_INSTALLED:
        return

    if not _NUKE_AVAILABLE:
        # Running outside Nuke — nothing to register, but don't error
        return

    from depthforge.nuke.toolbar import register_menu
    _MENU_REF = register_menu(menu_name)
    _MENU_INSTALLED = True


def uninstall() -> None:
    """Remove the DepthForge toolbar menu from Nuke."""
    global _MENU_INSTALLED, _MENU_REF

    if not _NUKE_AVAILABLE or not _MENU_INSTALLED:
        return

    try:
        import nuke
        menu_bar = nuke.menu("Nuke")
        menu_bar.removeItem("DepthForge")
    except Exception:
        pass

    _MENU_INSTALLED = False
    _MENU_REF       = None


def create_node(
    mode: str = "SIRDS",
    preset: str = "none",
    **knob_overrides,
):
    """Create a DepthForge gizmo node in the Nuke node graph.

    Parameters
    ----------
    mode : str
        One of: ``"SIRDS"``, ``"Texture"``, ``"Anaglyph"``,
        ``"StereoPair"``, ``"HiddenImage"``.
    preset : str
        Optional preset name. One of: ``"none"``, ``"shallow"``,
        ``"medium"``, ``"deep"``, ``"cinema"``, ``"print"``,
        ``"broadcast"``.
    **knob_overrides
        Any additional knob values to set on the created node.

    Returns
    -------
    nuke.Node  (when running inside Nuke)
    DepthForgeGizmo  (when running outside Nuke)
    """
    values = {"df_mode": mode, "df_preset": preset}
    values.update(knob_overrides)
    gizmo = DepthForgeGizmo(values)

    if _NUKE_AVAILABLE:
        return gizmo.create_nuke_node()
    return gizmo
