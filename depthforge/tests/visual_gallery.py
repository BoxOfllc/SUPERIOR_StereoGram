"""
DepthForge Phase 1 — Visual Output Gallery
===========================================
Renders a comprehensive gallery of stereogram outputs for visual inspection.

Run:  python tests/visual_gallery.py
Output: tests/gallery/  (PNG files, one per mode)

Each output is saved as a labelled PNG that can be opened in any image viewer
or imported into Nuke/AE for visual QC.
"""

import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from depthforge.core.adaptive_dots import AdaptiveDotParams, generate_adaptive_dots
from depthforge.core.anaglyph import (
    AnaglyphMode,
    AnaglyphParams,
    make_anaglyph,
    make_anaglyph_from_depth,
)
from depthforge.core.depth_prep import DepthPrepParams, FalloffCurve, prep_depth
from depthforge.core.hidden_image import (
    HiddenImageParams,
    encode_hidden_image,
    shape_to_mask,
    text_to_mask,
)
from depthforge.core.pattern_gen import (
    ColorMode,
    GridStyle,
    PatternParams,
    PatternType,
    generate_pattern,
    tile_to_frame,
)
from depthforge.core.stereo_pair import (
    StereoPairParams,
    compose_side_by_side,
    make_stereo_pair,
)
from depthforge.core.synthesizer import StereoParams, synthesize

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
GALLERY_DIR = os.path.join(os.path.dirname(__file__), "gallery")
os.makedirs(GALLERY_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Test depth maps
# ---------------------------------------------------------------------------
SIZE = 256


def sphere_depth(H=SIZE, W=SIZE) -> np.ndarray:
    ys = np.linspace(-1, 1, H)[:, None]
    xs = np.linspace(-1, 1, W)[None, :]
    return np.sqrt(np.maximum(1.0 - xs**2 - ys**2, 0)).astype(np.float32)


def gradient_depth(H=SIZE, W=SIZE) -> np.ndarray:
    xs = np.linspace(0, 1, W)[None, :]
    ys = np.linspace(0, 1, H)[:, None]
    return ((xs + ys) / 2).astype(np.float32)


def multi_sphere_depth(H=SIZE, W=SIZE) -> np.ndarray:
    """Multiple spheres at different depths."""
    d = np.zeros((H, W), dtype=np.float32)
    centres = [(0.25, 0.25, 0.6), (0.75, 0.25, 0.9), (0.5, 0.75, 0.75)]
    for cx, cy, depth_val in centres:
        ys = np.linspace(0, 1, H)[:, None] - cy
        xs = np.linspace(0, 1, W)[None, :] - cx
        r = np.sqrt(xs**2 + ys**2)
        mask = r < 0.18
        d[mask] = np.maximum(d[mask], depth_val * (1 - r[mask] / 0.18))
    return d


def text_depth(text="3D", H=SIZE, W=SIZE) -> np.ndarray:
    """Depth map from text."""
    from depthforge.core.hidden_image import text_to_mask

    return text_to_mask(text, W, H, font_size=80)


DEPTH_SPHERE = sphere_depth()
DEPTH_GRADIENT = gradient_depth()
DEPTH_MULTI = multi_sphere_depth()
DEPTH_TEXT = text_depth()

# Prep all depth maps
prep = DepthPrepParams(bilateral_sigma_space=3, dilation_px=2)
DEPTH_SPHERE_P = prep_depth(DEPTH_SPHERE, prep)
DEPTH_GRADIENT_P = prep_depth(DEPTH_GRADIENT, prep)
DEPTH_MULTI_P = prep_depth(DEPTH_MULTI, prep)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def label_image(img_arr: np.ndarray, text: str) -> np.ndarray:
    """Add a text label strip at the top of an image."""
    img = Image.fromarray(img_arr[:, :, :3] if img_arr.shape[2] == 4 else img_arr)
    H, W = img_arr.shape[:2]
    bar = Image.new("RGB", (W, 28), (20, 20, 30))
    draw = ImageDraw.Draw(bar)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((6, 6), text, fill=(0, 220, 255), font=font)
    combined = Image.new("RGB", (W, H + 28))
    combined.paste(bar, (0, 0))
    combined.paste(img, (0, 28))
    return np.asarray(combined)


def save(name: str, arr: np.ndarray) -> str:
    """Save to gallery dir, return path."""
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    path = os.path.join(GALLERY_DIR, f"{name}.png")
    Image.fromarray(arr.astype(np.uint8)).save(path)
    return path


def make_mosaic(images: list, cols: int, padding: int = 4) -> np.ndarray:
    """Arrange images in a grid."""
    rows = math.ceil(len(images) / cols)
    H, W = images[0].shape[:2]
    canvas = np.full(
        ((H + padding) * rows + padding, (W + padding) * cols + padding, 3), 15, dtype=np.uint8
    )
    for i, img in enumerate(images):
        r = i // cols
        c = i % cols
        y = padding + r * (H + padding)
        x = padding + c * (W + padding)
        rgb = img[:, :, :3] if img.ndim == 3 and img.shape[2] == 4 else img
        canvas[y : y + H, x : x + W] = rgb[:H, :W]
    return canvas


# ---------------------------------------------------------------------------
# Section 1: Pattern library gallery
# ---------------------------------------------------------------------------
print("\n[1/6] Pattern library...")

pattern_configs = [
    (
        "SIRDS Random (Grey)",
        PatternParams(PatternType.RANDOM_NOISE, 128, 128, ColorMode.GREYSCALE, seed=0),
    ),
    (
        "SIRDS Random (Mono Blue)",
        PatternParams(PatternType.RANDOM_NOISE, 128, 128, ColorMode.MONOCHROME, hue=0.6, seed=0),
    ),
    (
        "SIRDS Random (Psychedelic)",
        PatternParams(PatternType.RANDOM_NOISE, 128, 128, ColorMode.PSYCHEDELIC, seed=0),
    ),
    (
        "Perlin Noise (Grey)",
        PatternParams(PatternType.PERLIN, 128, 128, ColorMode.GREYSCALE, seed=1),
    ),
    (
        "Perlin (Psychedelic)",
        PatternParams(PatternType.PERLIN, 128, 128, ColorMode.PSYCHEDELIC, seed=1),
    ),
    ("Plasma", PatternParams(PatternType.PLASMA, 128, 128, ColorMode.PSYCHEDELIC, seed=2)),
    ("Voronoi Cells", PatternParams(PatternType.VORONOI, 128, 128, ColorMode.GREYSCALE, seed=3)),
    (
        "Voronoi (Psychedelic)",
        PatternParams(PatternType.VORONOI, 128, 128, ColorMode.PSYCHEDELIC, seed=3),
    ),
    (
        "Dots Grid",
        PatternParams(
            PatternType.GEOMETRIC_GRID, 128, 128, grid_style=GridStyle.DOTS, grid_spacing=12
        ),
    ),
    (
        "Hex Grid",
        PatternParams(
            PatternType.GEOMETRIC_GRID, 128, 128, grid_style=GridStyle.HEXES, grid_spacing=16
        ),
    ),
    (
        "Checkerboard",
        PatternParams(
            PatternType.GEOMETRIC_GRID, 128, 128, grid_style=GridStyle.CHECKS, grid_spacing=16
        ),
    ),
    (
        "Stripes",
        PatternParams(
            PatternType.GEOMETRIC_GRID, 128, 128, grid_style=GridStyle.STRIPES, grid_spacing=12
        ),
    ),
    ("Mandelbrot", PatternParams(PatternType.MANDELBROT, 128, 128, ColorMode.GREYSCALE)),
    (
        "Mandelbrot (Psychedelic)",
        PatternParams(PatternType.MANDELBROT, 128, 128, ColorMode.PSYCHEDELIC),
    ),
    ("Dot Matrix", PatternParams(PatternType.DOT_MATRIX, 128, 128, grid_spacing=10)),
]

pat_images = []
for label, p in pattern_configs:
    tile = generate_pattern(p)
    frame = tile_to_frame(tile, SIZE, SIZE)
    frame = label_image(frame, label)
    pat_images.append(frame)

mosaic = make_mosaic(pat_images, cols=5)
save("01_pattern_library", mosaic)
print(f"   ✓ {len(pattern_configs)} patterns rendered")


# ---------------------------------------------------------------------------
# Section 2: SIRDS Stereograms
# ---------------------------------------------------------------------------
print("[2/6] SIRDS stereograms...")

sirds_outputs = []
for depth_name, depth in [
    ("Sphere", DEPTH_SPHERE_P),
    ("Multi-Sphere", DEPTH_MULTI_P),
    ("Gradient", DEPTH_GRADIENT_P),
]:
    for color_name, color_mode in [
        ("Monochrome", ColorMode.GREYSCALE),
        ("Psychedelic", ColorMode.PSYCHEDELIC),
    ]:
        pattern = generate_pattern(
            PatternParams(PatternType.RANDOM_NOISE, 64, 64, color_mode, seed=42)
        )
        stereo = synthesize(depth, pattern, StereoParams(depth_factor=0.4, seed=0))
        stereo = label_image(stereo, f"SIRDS {depth_name} ({color_name})")
        sirds_outputs.append(stereo)

# Texture pattern stereograms
for pat_type, pat_label in [
    (PatternType.PLASMA, "Plasma"),
    (PatternType.PERLIN, "Perlin"),
    (PatternType.VORONOI, "Voronoi"),
    (PatternType.MANDELBROT, "Mandelbrot"),
]:
    pattern = generate_pattern(PatternParams(pat_type, 128, 128, ColorMode.PSYCHEDELIC, seed=5))
    stereo = synthesize(DEPTH_MULTI_P, pattern, StereoParams(depth_factor=0.4, seed=0))
    stereo = label_image(stereo, f"Texture ({pat_label}) Multi-Sphere")
    sirds_outputs.append(stereo)

mosaic = make_mosaic(sirds_outputs, cols=4)
save("02_sirds_stereograms", mosaic)
print(f"   ✓ {len(sirds_outputs)} stereograms rendered")


# ---------------------------------------------------------------------------
# Section 3: Anaglyph modes
# ---------------------------------------------------------------------------
print("[3/6] Anaglyph modes...")

# Source image: coloured gradient
src = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)
src[:, :, 3] = 255
for y in range(SIZE):
    for x in range(SIZE):
        src[y, x, 0] = int(x / SIZE * 200) + 30
        src[y, x, 1] = int(y / SIZE * 150) + 50
        src[y, x, 2] = 80

ana_outputs = []
for mode in AnaglyphMode:
    out = make_anaglyph_from_depth(src, DEPTH_SPHERE_P, AnaglyphParams(mode=mode, parallax_px=12))
    out = label_image(out, f"Anaglyph: {mode.name}")
    ana_outputs.append(out)

# Also anaglyph from SIRDS stereo pair
for depth_name, depth in [("Sphere", DEPTH_SPHERE_P), ("Text", DEPTH_TEXT)]:
    L, R, _ = make_stereo_pair(src, depth, StereoPairParams(max_parallax_fraction=0.04))
    out = make_anaglyph(L, R, AnaglyphParams(mode=AnaglyphMode.OPTIMISED))
    out = label_image(out, f"Anaglyph (Optimised) from Pair: {depth_name}")
    ana_outputs.append(out)

mosaic = make_mosaic(ana_outputs, cols=3)
save("03_anaglyph_modes", mosaic)
print(f"   ✓ {len(ana_outputs)} anaglyph outputs rendered")


# ---------------------------------------------------------------------------
# Section 4: Stereo pairs + occlusion masks
# ---------------------------------------------------------------------------
print("[4/6] Stereo pairs & occlusion masks...")

pair_outputs = []
for depth_name, depth in [("Sphere", DEPTH_SPHERE_P), ("Multi-Sphere", DEPTH_MULTI_P)]:
    L, R, occ = make_stereo_pair(
        src, depth, StereoPairParams(max_parallax_fraction=0.06, feather_px=4)
    )
    # Occlusion mask as visible RGB
    occ_rgb = np.stack(
        [
            (occ * 255).astype(np.uint8),
            np.zeros_like(occ, dtype=np.uint8),
            np.zeros_like(occ, dtype=np.uint8),
            np.full_like(occ, 255, dtype=np.uint8),
        ],
        axis=-1,
    )

    pair_outputs.append(label_image(L, f"Left View ({depth_name})"))
    pair_outputs.append(label_image(R, f"Right View ({depth_name})"))
    pair_outputs.append(label_image(occ_rgb, f"Occlusion Mask ({depth_name})"))

    # Side-by-side
    sbs = compose_side_by_side(L, R, gap_px=4)
    pair_outputs.append(label_image(sbs[:, :SIZE], f"SBS Left Half ({depth_name})"))

mosaic = make_mosaic(pair_outputs, cols=4)
save("04_stereo_pairs", mosaic)
print(f"   ✓ {len(pair_outputs)//4} depth scenes × 4 outputs")


# ---------------------------------------------------------------------------
# Section 5: Hidden image mode
# ---------------------------------------------------------------------------
print("[5/6] Hidden image mode...")

hidden_outputs = []
pattern_grey = generate_pattern(
    PatternParams(PatternType.RANDOM_NOISE, 64, 64, ColorMode.GREYSCALE, seed=7)
)
pattern_psych = generate_pattern(
    PatternParams(PatternType.PLASMA, 128, 128, ColorMode.PSYCHEDELIC, seed=7)
)

for shape_name, mask in [
    ("Circle", shape_to_mask("circle", SIZE, SIZE)),
    ("Star", shape_to_mask("star", SIZE, SIZE)),
    ("Diamond", shape_to_mask("diamond", SIZE, SIZE)),
    ("Triangle", shape_to_mask("triangle", SIZE, SIZE)),
    ("Arrow", shape_to_mask("arrow", SIZE, SIZE)),
    ("Text: 3D", text_to_mask("3D", SIZE, SIZE)),
]:
    # Monochrome
    out_grey = encode_hidden_image(
        pattern_grey,
        mask,
        HiddenImageParams(foreground_depth=0.85, background_depth=0.0, edge_soften_px=4),
    )
    out_grey = label_image(out_grey, f"Hidden {shape_name} (SIRDS)")
    hidden_outputs.append(out_grey)

    # Psychedelic pattern
    out_psych = encode_hidden_image(
        pattern_psych,
        mask,
        HiddenImageParams(foreground_depth=0.85, background_depth=0.0, edge_soften_px=6),
    )
    out_psych = label_image(out_psych, f"Hidden {shape_name} (Plasma)")
    hidden_outputs.append(out_psych)

mosaic = make_mosaic(hidden_outputs, cols=4)
save("05_hidden_images", mosaic)
print(f"   ✓ {len(hidden_outputs)//2} shapes × 2 patterns")


# ---------------------------------------------------------------------------
# Section 6: Adaptive dots + depth-prep curves
# ---------------------------------------------------------------------------
print("[6/6] Adaptive dots + depth falloff curves...")

adot_outputs = []

# Adaptive dots for different depth complexities
for depth_name, depth in [
    ("Sphere", DEPTH_SPHERE_P),
    ("Multi-Sphere", DEPTH_MULTI_P),
    ("Text", DEPTH_TEXT),
]:
    params = AdaptiveDotParams(
        tile_width=SIZE,
        tile_height=SIZE,
        min_dot_radius=1,
        max_dot_radius=4,
        min_spacing=3,
        max_spacing=14,
        seed=0,
    )
    tile = generate_adaptive_dots(depth, params)
    stereo = synthesize(depth, tile, StereoParams(depth_factor=0.4, seed=0))
    adot_outputs.append(label_image(tile, f"Adaptive Dots Tile ({depth_name})"))
    adot_outputs.append(label_image(stereo, f"Adaptive SIRDS ({depth_name})"))

# Depth prep falloff curves comparison
curve_outputs = []
for curve_name, curve in [
    ("Linear", FalloffCurve.LINEAR),
    ("S-Curve", FalloffCurve.S_CURVE),
    ("Gamma 0.5", FalloffCurve.GAMMA),
    ("Logarithmic", FalloffCurve.LOGARITHMIC),
    ("Exponential", FalloffCurve.EXPONENTIAL),
]:
    gamma = 0.5 if curve_name == "Gamma 0.5" else 1.0
    p = DepthPrepParams(
        bilateral_sigma_space=3, dilation_px=2, falloff_curve=curve, falloff_gamma=gamma
    )
    depth_curved = prep_depth(DEPTH_SPHERE, p)
    # Visualise the depth map itself
    depth_vis = (depth_curved * 255).astype(np.uint8)
    depth_rgba = np.stack([depth_vis, depth_vis, depth_vis, np.full_like(depth_vis, 255)], axis=-1)
    pattern = generate_pattern(
        PatternParams(PatternType.RANDOM_NOISE, 64, 64, ColorMode.GREYSCALE, seed=0)
    )
    stereo = synthesize(depth_curved, pattern, StereoParams(depth_factor=0.4, seed=0))
    curve_outputs.append(label_image(depth_rgba, f"Depth Map ({curve_name})"))
    curve_outputs.append(label_image(stereo, f"Stereo ({curve_name})"))

all_6 = adot_outputs + curve_outputs
mosaic = make_mosaic(all_6, cols=4)
save("06_adaptive_dots_curves", mosaic)
print(f"   ✓ {len(adot_outputs)//2} adaptive dot scenes, {len(curve_outputs)//2} curve variants")


# ---------------------------------------------------------------------------
# Master summary mosaic — 1 representative from each section
# ---------------------------------------------------------------------------
print("\nBuilding master summary mosaic...")


def load_first_panel(arr: np.ndarray, W: int = SIZE) -> np.ndarray:
    return arr[: SIZE + 28, :W]  # first panel including label


summary = []

# One SIRDS per color mode
pattern_grey = generate_pattern(
    PatternParams(PatternType.RANDOM_NOISE, 64, 64, ColorMode.GREYSCALE, seed=0)
)
s = synthesize(DEPTH_SPHERE_P, pattern_grey, StereoParams(depth_factor=0.4, seed=0))
summary.append(label_image(s, "SIRDS Monochrome"))

pattern_psych = generate_pattern(
    PatternParams(PatternType.PLASMA, 128, 128, ColorMode.PSYCHEDELIC, seed=0)
)
s = synthesize(DEPTH_SPHERE_P, pattern_psych, StereoParams(depth_factor=0.4, seed=0))
summary.append(label_image(s, "Texture (Plasma) Psychedelic"))

# Anaglyph
s = make_anaglyph_from_depth(src, DEPTH_MULTI_P, AnaglyphParams(mode=AnaglyphMode.OPTIMISED))
summary.append(label_image(s, "Anaglyph (Optimised)"))

# Hidden image
mask = shape_to_mask("star", SIZE, SIZE)
s = encode_hidden_image(pattern_grey, mask, HiddenImageParams(foreground_depth=0.85))
summary.append(label_image(s, "Hidden Image (Star)"))

# Adaptive dots
adot = generate_adaptive_dots(
    DEPTH_MULTI_P, AdaptiveDotParams(tile_width=SIZE, tile_height=SIZE, seed=0)
)
s = synthesize(DEPTH_MULTI_P, adot, StereoParams(depth_factor=0.4, seed=0))
summary.append(label_image(s, "Adaptive Dots SIRDS"))

# Stereo pair SBS
L, R, _ = make_stereo_pair(src, DEPTH_SPHERE_P, StereoPairParams(max_parallax_fraction=0.05))
sbs = compose_side_by_side(L, R, gap_px=2)
summary.append(label_image(sbs[:, :SIZE], "Stereo Pair (L half)"))

master = make_mosaic(summary, cols=3)
path = save("00_MASTER_GALLERY", master)
print(f"   ✓ Master gallery saved: {path}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
files = sorted(os.listdir(GALLERY_DIR))
total_size = sum(os.path.getsize(os.path.join(GALLERY_DIR, f)) for f in files)

print(f"""
╔══════════════════════════════════════════════════════╗
║         DepthForge Phase 1 — Gallery Complete        ║
╠══════════════════════════════════════════════════════╣
║  Output dir : {GALLERY_DIR}
║  Files      : {len(files)} PNG images
║  Total size : {total_size / 1024:.0f} KB
╠══════════════════════════════════════════════════════╣
║  01_pattern_library    — 15 pattern types            ║
║  02_sirds_stereograms  — SIRDS + texture modes       ║
║  03_anaglyph_modes     — All 5 anaglyph variants     ║
║  04_stereo_pairs       — L/R views + occlusion mask  ║
║  05_hidden_images      — 6 shapes × 2 patterns       ║
║  06_adaptive_dots      — Complexity-driven SIRDS      ║
║  00_MASTER_GALLERY     — Summary of all modes         ║
╚══════════════════════════════════════════════════════╝
""")
