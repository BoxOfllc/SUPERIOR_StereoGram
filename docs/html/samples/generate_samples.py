"""
Generate preset comparison samples for documentation.
Run from repo root with venv active:
    python docs/html/samples/generate_samples.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

OUT = Path(__file__).parent

# ── Build a rich synthetic depth map ────────────────────────────────────────
W, H = 800, 600

def make_depth_map(w, h):
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xx, yy = np.meshgrid(x, y)

    # Central sphere
    sphere = np.clip(1.0 - np.sqrt(xx**2 + yy**2) * 1.2, 0, 1)

    # Background rolling hills
    hills = (
        0.3 * np.sin(xx * 4) * np.cos(yy * 3) +
        0.15 * np.sin(xx * 7 + 0.5) +
        0.1 * np.cos(yy * 5 - 0.3)
    )
    hills = (hills - hills.min()) / (hills.max() - hills.min()) * 0.4

    # Smaller foreground bumps
    bump1 = np.clip(0.7 - np.sqrt((xx + 0.55)**2 + (yy + 0.35)**2) * 3.5, 0, 1) * 0.6
    bump2 = np.clip(0.6 - np.sqrt((xx - 0.58)**2 + (yy - 0.40)**2) * 3.2, 0, 1) * 0.5
    bump3 = np.clip(0.5 - np.sqrt((xx + 0.60)**2 + (yy - 0.42)**2) * 3.0, 0, 1) * 0.45

    depth = hills + sphere * 0.85 + bump1 + bump2 + bump3
    depth = np.clip(depth, 0, 1)
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth.astype(np.float32)

depth_raw = make_depth_map(W, H)

# Save depth map for reference
depth_img = Image.fromarray((depth_raw * 255).astype(np.uint8), mode='L')
depth_img.save(OUT / "depth_source.png")
print(f"Saved: depth_source.png  ({W}x{H})")

# ── Generate one stereogram per preset ──────────────────────────────────────
from depthforge.core.depth_prep import DepthPrepParams, prep_depth
from depthforge.core.synthesizer import StereoParams, synthesize
from depthforge.core.presets import get_preset
from depthforge.core.pattern_gen import PatternParams, PatternType, ColorMode, generate_pattern

PRESETS = ["shallow", "medium", "deep", "cinema", "print", "broadcast"]

# Use a nice greyscale perlin pattern so depth differences are clearly visible
pattern = generate_pattern(PatternParams(
    pattern_type=PatternType.PERLIN,
    tile_width=120,
    tile_height=120,
    color_mode=ColorMode.GREYSCALE,
    seed=77,
))

# Also prep depth once with standard settings
dp = DepthPrepParams(bilateral_sigma_space=5.0, bilateral_sigma_color=0.1, dilation_px=2)
depth_prepped = prep_depth(depth_raw, dp)

for name in PRESETS:
    p = get_preset(name)
    sp = p.stereo_params
    result = synthesize(depth_prepped, pattern, sp)
    img = Image.fromarray(result)

    # Add label bar at bottom
    label_h = 48
    canvas = Image.new("RGBA", (W, H + label_h), (15, 18, 26, 255))
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, H, W, H + label_h], fill=(22, 27, 36))
    draw.line([0, H, W, H], fill=(42, 51, 71), width=1)

    # Try to use a font, fall back gracefully
    try:
        font_large = ImageFont.truetype("arial.ttf", 18)
        font_small = ImageFont.truetype("arial.ttf", 13)
    except:
        font_large = ImageFont.load_default()
        font_small = font_large

    label = f"{name.upper()}   df={sp.depth_factor:.2f}  parallax≤{sp.max_parallax_fraction:.3f}  dpi={p.output_dpi}"
    draw.text((14, H + 8), label, fill=(226, 232, 240), font=font_large)

    out_path = OUT / f"preset_{name}.png"
    canvas.save(out_path)
    print(f"Saved: preset_{name}.png  (depth_factor={sp.depth_factor}, parallax<={sp.max_parallax_fraction:.3f})")

# ── Generate side-by-side comparison grid ───────────────────────────────────
# 2x3 grid of all presets
COLS, ROWS = 2, 3
THUMB_W, THUMB_H = 400, 300
GAP = 3
LABEL_H = 36
HEADER_H = 60

grid_w = COLS * THUMB_W + (COLS + 1) * GAP
grid_h = HEADER_H + ROWS * (THUMB_H + LABEL_H) + (ROWS + 1) * GAP
grid = Image.new("RGBA", (grid_w, grid_h), (13, 15, 20))
draw = ImageDraw.Draw(grid)

try:
    font_title = ImageFont.truetype("arial.ttf", 22)
    font_preset = ImageFont.truetype("arial.ttf", 15)
    font_params = ImageFont.truetype("arial.ttf", 11)
except:
    font_title = ImageFont.load_default()
    font_preset = font_title
    font_params = font_title

# Header
draw.text((grid_w // 2 - 180, 16), "SUPERIOR StereoGram — Preset Comparison", fill=(226, 232, 240), font=font_title)

for i, name in enumerate(PRESETS):
    col = i % COLS
    row = i // COLS
    x = GAP + col * (THUMB_W + GAP)
    y = HEADER_H + GAP + row * (THUMB_H + LABEL_H + GAP)

    # Load and resize the preset output
    src = Image.open(OUT / f"preset_{name}.png").convert("RGBA")
    # Crop off the label bar from the individual files
    src_cropped = src.crop((0, 0, W, H))
    thumb = src_cropped.resize((THUMB_W, THUMB_H), Image.LANCZOS)
    grid.paste(thumb, (x, y))

    # Border
    draw.rectangle([x-1, y-1, x+THUMB_W, y+THUMB_H], outline=(42, 51, 71))

    # Label area
    label_y = y + THUMB_H
    draw.rectangle([x, label_y, x + THUMB_W, label_y + LABEL_H], fill=(22, 27, 36))
    draw.line([x, label_y, x + THUMB_W, label_y], fill=(42, 51, 71))

    p = get_preset(name)
    sp = p.stereo_params

    # Preset name in accent colour
    accent_colors = {
        "shallow": (34, 211, 238),
        "medium":  (79, 142, 247),
        "deep":    (168, 85, 247),
        "cinema":  (245, 158, 11),
        "print":   (34, 197, 94),
        "broadcast": (239, 68, 68),
    }
    color = accent_colors.get(name, (200, 200, 200))
    draw.text((x + 8, label_y + 4), name.upper(), fill=color, font=font_preset)
    params_str = f"df={sp.depth_factor:.2f}  parallax≤{sp.max_parallax_fraction:.3f}  dpi={p.output_dpi}"
    draw.text((x + 8, label_y + 21), params_str, fill=(136, 153, 176), font=font_params)

grid.save(OUT / "preset_comparison_grid.png")
print(f"\nSaved: preset_comparison_grid.png  ({grid_w}x{grid_h})")
print("\nAll samples generated successfully.")
