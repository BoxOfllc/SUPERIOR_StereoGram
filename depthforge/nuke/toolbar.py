"""
depthforge.nuke.toolbar
========================
Nuke toolbar / menu registration for all DepthForge gizmos.

Called automatically by ``depthforge.nuke.install()``.
Each entry creates a Python-driven Group node with the full
DepthForge knob set wired to a Python callback.

Menu structure
--------------
DepthForge/
  ├── Stereogram (SIRDS / Magic Eye)
  ├── Texture Stereogram
  ├── Anaglyph 3D
  ├── Stereo Pair
  ├── Hidden Image
  ├── ── (separator) ──
  ├── Depth Prep
  ├── Safety Limiter
  ├── QC Overlay
  ├── ── (separator) ──
  ├── Presets  ▶
  │     ├── Shallow
  │     ├── Medium (default)
  │     ├── Deep
  │     ├── Cinema (SMPTE)
  │     ├── Print (300 dpi)
  │     └── Broadcast (EBU)
  └── About DepthForge
"""

from __future__ import annotations

try:
    import nuke
    _NUKE_AVAILABLE = True
except ImportError:
    _NUKE_AVAILABLE = False
    nuke = None


# ---------------------------------------------------------------------------
# Python snippet executed by each menu item
# ---------------------------------------------------------------------------

_CREATE_NODE_SNIPPET = """
import depthforge.nuke as _df_nuke
_df_nuke.create_node(mode={mode!r}, preset={preset!r})
"""

_ABOUT_SNIPPET = """
import nuke
nuke.message(
    "<b>DepthForge v0.5.0</b><br><br>"
    "Professional stereogram synthesis engine.<br>"
    "Supports SIRDS, Texture, Anaglyph, Stereo Pair, Hidden Image.<br><br>"
    "Phases 1-3 complete: core engine, presets, comfort analysis,<br>"
    "CLI, AI depth, pattern library, ComfyUI, Nuke.<br><br>"
    "Documentation: https://github.com/your-org/depthforge"
)
"""


# ---------------------------------------------------------------------------
# Menu items definition
# ---------------------------------------------------------------------------

#: (label, mode, preset, shortcut_or_None)
_MAIN_ITEMS = [
    ("Stereogram (SIRDS)",      "SIRDS",       "medium",  "Alt+Shift+S"),
    ("Texture Stereogram",      "Texture",     "medium",  "Alt+Shift+T"),
    ("Anaglyph 3D",             "Anaglyph",    "medium",  "Alt+Shift+A"),
    ("Stereo Pair",             "StereoPair",  "medium",  "Alt+Shift+P"),
    ("Hidden Image",            "HiddenImage", "medium",  None),
    (None, None, None, None),   # separator
    ("Depth Prep",              "DepthPrep",   "none",    None),
    ("QC Overlay",              "QCOverlay",   "none",    None),
    ("Safety Limiter",          "Safety",      "none",    None),
]

_PRESET_ITEMS = [
    ("Shallow (Safe)",          "SIRDS", "shallow"),
    ("Medium (Default)",        "SIRDS", "medium"),
    ("Deep (High Impact)",      "SIRDS", "deep"),
    ("Cinema (SMPTE)",          "SIRDS", "cinema"),
    ("Print (300 dpi)",         "SIRDS", "print"),
    ("Broadcast (EBU R95)",     "SIRDS", "broadcast"),
]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register_menu(menu_name: str = "DepthForge"):
    """Register the DepthForge toolbar menu in Nuke.

    Parameters
    ----------
    menu_name : str
        Top-level menu name.

    Returns
    -------
    nuke.Menu
        The created menu object (keep a reference to prevent GC).
    """
    if not _NUKE_AVAILABLE:
        raise RuntimeError("Nuke is not available in this environment.")

    toolbar   = nuke.menu("Nuke")
    df_menu   = toolbar.addMenu(menu_name, icon="DepthForge.png")

    # Main synthesis nodes
    for label, mode, preset, shortcut in _MAIN_ITEMS:
        if label is None:
            df_menu.addSeparator()
            continue
        snippet = _CREATE_NODE_SNIPPET.format(mode=mode, preset=preset)
        kwargs  = {}
        if shortcut:
            kwargs["shortcutContext"] = 2   # Node graph context
        df_menu.addCommand(label, snippet, **kwargs)

    df_menu.addSeparator()

    # Presets submenu
    preset_menu = df_menu.addMenu("Presets")
    for label, mode, preset in _PRESET_ITEMS:
        snippet = _CREATE_NODE_SNIPPET.format(mode=mode, preset=preset)
        preset_menu.addCommand(label, snippet)

    df_menu.addSeparator()
    df_menu.addCommand("About DepthForge", _ABOUT_SNIPPET)

    return df_menu


def unregister_menu(menu_name: str = "DepthForge") -> None:
    """Remove the DepthForge toolbar menu."""
    if not _NUKE_AVAILABLE:
        return
    try:
        nuke.menu("Nuke").removeItem(menu_name)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Viewer LUT registration (cosmetic — adds "DepthForge Depth" LUT option)
# ---------------------------------------------------------------------------

def register_viewer_luts() -> None:
    """Register DepthForge-specific viewer LUT presets in Nuke."""
    if not _NUKE_AVAILABLE:
        return
    try:
        # Add a simple depth-to-colour LUT for the viewer
        nuke.ViewerProcess.register(
            "DepthForge Depth",
            nuke.createNode,
            ("Vectorfield",),
        )
    except Exception:
        pass   # Non-critical — viewer LUT is cosmetic


# ---------------------------------------------------------------------------
# knobChanged callback for DepthForge Group nodes
# ---------------------------------------------------------------------------

def _knob_changed_callback() -> None:
    """Update knob enabled-states when df_mode changes on a DepthForge Group node.

    Registered via nuke.addKnobChanged — no exec() required.
    """
    if not _NUKE_AVAILABLE:
        return
    node = nuke.thisNode()
    knob = nuke.thisKnob()
    if not (node and knob):
        return
    if not node.name().startswith("DepthForge"):
        return

    mode = node["df_mode"].value() if node.knob("df_mode") else "SIRDS"

    _anaglyph_knobs   = ["df_anaglyph_mode", "df_swap_eyes", "df_gamma"]
    _stereopair_knobs = ["df_bg_fill", "df_eye_balance", "df_feather_px"]
    _hidden_knobs     = ["df_hidden_shape", "df_hidden_text", "df_fg_depth",
                         "df_bg_depth", "df_edge_soften"]

    for k in _anaglyph_knobs:
        if node.knob(k):
            node[k].setEnabled(mode == "Anaglyph")
    for k in _stereopair_knobs:
        if node.knob(k):
            node[k].setEnabled(mode == "StereoPair")
    for k in _hidden_knobs:
        if node.knob(k):
            node[k].setEnabled(mode == "HiddenImage")


def register_callbacks() -> None:
    """Register knobChanged callbacks for DepthForge group nodes."""
    if not _NUKE_AVAILABLE:
        return
    try:
        nuke.addKnobChanged(_knob_changed_callback, nodeClass="Group")
    except Exception:
        pass
