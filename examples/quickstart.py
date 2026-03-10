#!/usr/bin/env python3
"""
DepthForge — Quick Examples
Run this file to verify your installation and generate sample outputs.

Usage:
    python examples/quickstart.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from PIL import Image
from pathlib import Path

# Import DepthForge
from depthforge import synthesize, prep_depth, generate_pattern, capability_report
from depthforge import StereoParams, DepthPrepParams, PatternParams, PatternType, ColorMode, FalloffCurve
from depthforge.core.hidden_image import encode_hidden_image, shape_to_mask, text_to_mask, HiddenImageParams
from depthforge.core.anaglyph import make_anaglyph_from_depth, AnaglyphParams, AnaglyphMode

OUT = Path("examples/output")
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  DepthForge Quick Examples")
print("=" * 60)
print()
print(capability_report())
print()


def make_test_depth(W=800, H=600, mode="sphere"):
    """Generate a synthetic depth map."""
    y, x = np.mgrid[0:H, 0:W]
    cx, cy = W / 2, H / 2
    if mode == "sphere":
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        d = np.clip(1.0 - r / (min(W, H) * 0.4), 0, 1)
    elif mode == "gradient":
        d = x / W
    elif mode == "ramp":
        d = np.clip((1.0 - (y / H)) * 0.8 + 0.1, 0, 1)
    return d.astype(np.float32)


# ──────────────────────────────────────────────
# Example 1: Classic SIRDS (random dot)
# ──────────────────────────────────────────────
print("1. Classic SIRDS (random dot) ...")
depth = prep_depth(make_test_depth(), DepthPrepParams(dilation_px=3))
pattern = generate_pattern(PatternParams(
    pattern_type=PatternType.RANDOM_NOISE,
    tile_width=120, tile_height=120,
    color_mode=ColorMode.GREYSCALE, seed=1,
))
result = synthesize(depth, pattern, StereoParams(depth_factor=0.4, safe_mode=True, seed=1))
Image.fromarray(result).save(OUT / "01_sirds_greyscale.png")
print(f"   → {OUT / '01_sirds_greyscale.png'}")


# ──────────────────────────────────────────────
# Example 2: Psychedelic plasma pattern
# ──────────────────────────────────────────────
print("2. Psychedelic plasma pattern ...")
pattern = generate_pattern(PatternParams(
    pattern_type=PatternType.PLASMA,
    tile_width=128, tile_height=128,
    color_mode=ColorMode.PSYCHEDELIC, seed=7,
))
result = synthesize(depth, pattern, StereoParams(depth_factor=0.38, safe_mode=True))
Image.fromarray(result).save(OUT / "02_plasma_psychedelic.png")
print(f"   → {OUT / '02_plasma_psychedelic.png'}")


# ──────────────────────────────────────────────
# Example 3: Hidden star
# ──────────────────────────────────────────────
print("3. Hidden star ...")
W, H = 900, 600
mask = shape_to_mask("star", W, H)
pattern = generate_pattern(PatternParams(
    pattern_type=PatternType.RANDOM_NOISE,
    tile_width=128, tile_height=128,
    color_mode=ColorMode.GREYSCALE, seed=3,
))
result = encode_hidden_image(pattern, mask, HiddenImageParams(
    foreground_depth=0.75,
    background_depth=0.15,
    edge_soften_px=4,
))
Image.fromarray(result).save(OUT / "03_hidden_star.png")
print(f"   → {OUT / '03_hidden_star.png'}")


# ──────────────────────────────────────────────
# Example 4: Hidden text
# ──────────────────────────────────────────────
print("4. Hidden text: 'DEPTH' ...")
mask = text_to_mask("DEPTH", W, H, font_size=180)
result = encode_hidden_image(pattern, mask, HiddenImageParams(
    foreground_depth=0.8,
    background_depth=0.1,
    edge_soften_px=3,
))
Image.fromarray(result).save(OUT / "04_hidden_text.png")
print(f"   → {OUT / '04_hidden_text.png'}")


# ──────────────────────────────────────────────
# Example 5: Anaglyph (optimised)
# ──────────────────────────────────────────────
print("5. Anaglyph (optimised Dubois) ...")
depth_a = prep_depth(make_test_depth(W=800, H=600, mode="sphere"), DepthPrepParams())
# Use depth map itself as source (colourful result)
source_rgba = np.stack([
    (depth_a * 255).astype(np.uint8),
    (depth_a * 180).astype(np.uint8),
    ((1 - depth_a) * 255).astype(np.uint8),
    np.full(depth_a.shape, 255, dtype=np.uint8),
], axis=-1)
result = make_anaglyph_from_depth(source_rgba, depth_a, AnaglyphParams(
    mode=AnaglyphMode.OPTIMISED,
))
Image.fromarray(result).save(OUT / "05_anaglyph_optimised.png")
print(f"   → {OUT / '05_anaglyph_optimised.png'}")


# ──────────────────────────────────────────────
# Example 6: Voronoi cells
# ──────────────────────────────────────────────
print("6. Voronoi cells pattern ...")
depth = prep_depth(make_test_depth(mode="ramp"), DepthPrepParams(
    falloff_curve=FalloffCurve.S_CURVE,
))
pattern = generate_pattern(PatternParams(
    pattern_type=PatternType.VORONOI,
    tile_width=200, tile_height=200,
    color_mode=ColorMode.PSYCHEDELIC, seed=42,
))
result = synthesize(depth, pattern, StereoParams(depth_factor=0.35))
Image.fromarray(result).save(OUT / "06_voronoi.png")
print(f"   → {OUT / '06_voronoi.png'}")


print()
print(f"All examples saved to: {OUT.resolve()}")
print()
print("How to view stereograms:")
print("  1. Hold image at arm's length")
print("  2. Relax eyes as if looking past the screen")
print("  3. Wait 3–10 seconds — the 3D should 'snap' into view")
