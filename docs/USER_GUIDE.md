# DepthForge User Guide

**Version 0.1 — Phase 1 Release**

---

## Table of Contents

1. [What Is a Stereogram?](#1-what-is-a-stereogram)
2. [Safety](#2-safety)
3. [Installation](#3-installation)
4. [Core Concepts](#4-core-concepts)
5. [Getting Your First Stereogram](#5-getting-your-first-stereogram)
6. [Depth Maps — The Key Input](#6-depth-maps--the-key-input)
7. [Patterns and Tiles](#7-patterns-and-tiles)
8. [Synthesis Modes](#8-synthesis-modes)
   - [SIRDS (Random Dot)](#81-sirds-random-dot)
   - [Texture Pattern](#82-texture-pattern)
   - [Hidden Image](#83-hidden-image)
   - [Anaglyph](#84-anaglyph)
   - [Stereo Pair](#85-stereo-pair)
9. [Stereo Controls Reference](#9-stereo-controls-reference)
10. [Depth Conditioning](#10-depth-conditioning)
11. [Using the Python API](#11-using-the-python-api)
12. [CLI Usage (Phase 3)](#12-cli-usage-phase-3)
13. [OFX Plugin (Phase 5)](#13-ofx-plugin-phase-5)
14. [Nuke Gizmo (Phase 4)](#14-nuke-gizmo-phase-4)
15. [ComfyUI Nodes (Phase 4)](#15-comfyui-nodes-phase-4)
16. [Tips for Good Results](#16-tips-for-good-results)
17. [Troubleshooting](#17-troubleshooting)

---

## 1. What Is a Stereogram?

A stereogram is a flat 2D image that encodes 3D depth information in a way the brain perceives as a three-dimensional scene. To see the depth, you relax your eyes so that they look "through" the image (like looking at a distant point while the image is close) — or cross your eyes slightly. After a moment, the repeating pattern "snaps" into a 3D shape.

DepthForge creates stereograms from a **depth map** — a grayscale image where bright pixels are near and dark pixels are far — and a **pattern image** that tiles across the output.

**Types of stereograms DepthForge makes:**

- **SIRDS** (Single Image Random Dot Stereogram) — the classic "magic eye" look, with a field of colored dots
- **SIS** (Single Image Stereogram) — uses a repeating textured pattern tile instead of dots
- **Anaglyph** — red/cyan color separation for viewing with 3D glasses
- **Stereo pair** — separate left-eye and right-eye images, for side-by-side 3D displays
- **Hidden image** — a recognizable shape or text concealed inside dot noise

---

## 2. Safety

> **⚠️ Important: Read this section before generating any content.**

### Photosensitive Epilepsy

High-contrast flickering patterns and rapid color changes can trigger **photosensitive epileptic seizures** in susceptible individuals (~1 in 4,000 people). This risk applies especially to:

- Psychedelic/high-contrast color patterns
- Animated or video stereograms with fast depth changes
- Very high-frequency dot patterns

**What to do:**
- Use `safe_mode=True` in `StereoParams` (or `--safe-mode` on the CLI) when content may be viewed by the general public
- Test with a low `depth_factor` first (start at 0.2–0.3)
- For video, use temporal smoothing to avoid abrupt frame-to-frame changes
- Do not publish high-contrast psychedelic stereograms without a seizure warning

### Eye Strain and Discomfort

Stereograms require the brain to decouple focus from vergence (the angle of the eyes). This is harmless for most people but can cause discomfort with:

- Excessive `depth_factor` (parallax too wide)
- Very long viewing sessions
- Some types of binocular vision issues

The default `max_parallax_fraction=1/30` (about 3.3% of frame width) is within the recommended comfort zone for most viewers. Values above `0.05` (5%) should be used cautiously.

---

## 3. Installation

### Requirements

| Package | Required | Purpose |
|---|---|---|
| Python ≥ 3.9 | ✅ Always | Runtime |
| NumPy ≥ 1.22 | ✅ Always | Array processing |
| Pillow ≥ 9.0 | ✅ Always | Image I/O |
| OpenCV (`cv2`) | Optional | Bilateral filter, morphology, inpainting — highly recommended |
| SciPy | Optional | Advanced filtering fallback |
| PyTorch ≥ 2.0 | Optional | MiDaS/ZoeDepth AI depth estimation |

### Option 1: pip (recommended)

```bash
# Core only (NumPy + Pillow)
pip install depthforge

# Core + OpenCV (recommended for best quality)
pip install "depthforge[cv]"

# Core + AI depth estimation
pip install "depthforge[ai]"

# Everything
pip install "depthforge[full]"
```

### Option 2: From source

```bash
git clone https://github.com/your-org/depthforge.git
cd depthforge
pip install -e .
```

For development (includes testing tools):
```bash
pip install -e ".[dev]"
```

### Verify installation

```python
import depthforge
print(depthforge.__version__)
print(depthforge.capability_report())
```

This prints which optional packages are active (OpenCV, SciPy, PyTorch) and confirms the core is working.

### Platform Support

| Platform | Core | OFX Plugin | Nuke Gizmo | ComfyUI |
|---|---|---|---|---|
| macOS (Apple Silicon) | ✅ | Phase 5 | Phase 4 | Phase 4 |
| macOS (Intel) | ✅ | Phase 5 | Phase 4 | Phase 4 |
| Windows 10/11 | ✅ | Phase 5 | Phase 4 | Phase 4 |
| Linux (Ubuntu 20+) | ✅ | Phase 5 | Phase 4 | Phase 4 |

---

## 4. Core Concepts

Understanding these three things will make everything else click:

### Depth Map

A grayscale image where **white = near** and **black = far**. (This can be inverted with `invert_depth=True`.) The depth map drives all 3D depth in the output — it is the single most important input.

Good depth maps:
- Have smooth transitions (not hard edges) at depth boundaries
- Cover the full range from 0 to 255 (or 0.0 to 1.0 float)
- Are the same resolution as your target output, or larger

### Pattern Tile

A tileable image that repeats across the width of the output. For SIRDS mode, DepthForge generates this automatically as random dots. For texture mode, you supply one or DepthForge generates one procedurally (Perlin noise, plasma, Voronoi, etc.).

### Parallax

The horizontal pixel shift applied to the pattern based on depth. Deeper pixels shift less; nearer pixels shift more. The brain interprets this difference in shift as 3D depth. The `depth_factor` and `max_parallax_fraction` parameters control how much shift is applied.

---

## 5. Getting Your First Stereogram

### A) From a greyscale image (simplest path)

Create or find any greyscale image — white objects appear near, black recedes.

```python
from PIL import Image
import numpy as np
from depthforge import synthesize, prep_depth, generate_pattern
from depthforge import StereoParams, DepthPrepParams, PatternParams, PatternType, ColorMode

# Load your depth image
img = Image.open("my_depth.png").convert("L")
depth_raw = np.array(img, dtype=np.float32) / 255.0

# Condition the depth (smooth edges, prevent fringing)
depth = prep_depth(depth_raw, DepthPrepParams())

# Generate a simple random dot pattern
pattern = generate_pattern(PatternParams(
    pattern_type=PatternType.RANDOM_NOISE,
    tile_width=128,
    tile_height=128,
    color_mode=ColorMode.GREYSCALE,
    seed=1
))

# Synthesize
result = synthesize(depth, pattern, StereoParams(depth_factor=0.35))

# Save
Image.fromarray(result).save("my_stereogram.png")
print("Done! Open my_stereogram.png and let your eyes relax.")
```

### B) From a geometric shape (no depth image needed)

```python
from depthforge import synthesize, prep_depth, generate_pattern
from depthforge.core.hidden_image import shape_to_mask, mask_to_depth
from depthforge import StereoParams, DepthPrepParams, PatternParams, PatternType, ColorMode
from PIL import Image
import numpy as np

W, H = 1200, 800

# Create a star depth mask
mask = shape_to_mask("star", W, H)
depth_raw = mask_to_depth(mask)

# Condition
depth = prep_depth(depth_raw, DepthPrepParams(bilateral_sigma_space=3))

# Plasma pattern
pattern = generate_pattern(PatternParams(
    pattern_type=PatternType.PLASMA,
    tile_width=128,
    tile_height=128,
    color_mode=ColorMode.PSYCHEDELIC,
    seed=99,
))

result = synthesize(depth, pattern, StereoParams(depth_factor=0.3, safe_mode=True))
Image.fromarray(result).save("star_stereogram.png")
```

### C) How to view a stereogram

There are two techniques:

**Parallel (wall-eyed) method — most common:**
1. Hold the image at arm's length
2. Relax your eyes as if looking at something far beyond the screen/page
3. Wait — within a few seconds the image should "snap" into 3D
4. Once you see depth, you can bring the image slightly closer while maintaining the relaxed gaze

**Cross-eyed method:**
1. Hold the image at arm's length
2. Slowly cross your eyes by focusing on a point about halfway between you and the image
3. The depth will appear inverted (near becomes far) compared to parallel viewing
4. Use `invert_depth=True` to correct this if needed

---

## 6. Depth Maps — The Key Input

### What makes a good depth map

| Characteristic | Ideal | Avoid |
|---|---|---|
| Dynamic range | Full 0–255 (or 0.0–1.0) | Compressed to narrow range (e.g. 100–150) |
| Transitions | Smooth gradients at boundaries | Hard, aliased edges |
| Noise | Smooth or absent | High frequency noise |
| Resolution | Same as output or larger | Much smaller than output (will look pixelated) |

### Sources of depth maps

**1. Draw/paint your own**
Any paint application works. Use gradients for smooth depth transitions. White = closest to viewer.

**2. CG render — Z pass**
Most 3D renderers output a Z depth channel (EXR format). DepthForge accepts normalized float depth maps directly. Note that raw Z values from renderers are usually *inverted* (far = large values near = small) and not normalized — use `invert_depth=True` and DepthForge's normalizer handles the rest.

**3. AI depth estimation (Phase 3)**
Once Phase 3 is complete, `depthforge.depth_models` will provide `midas_depth(image)` and `zoedepth(image)` which return a float32 depth array from any color photograph.

**4. Grayscale photograph**
Any photo processed to grayscale can serve as a depth map. Results are artistic rather than geometrically accurate, but often look compelling.

### Depth conventions

By default DepthForge assumes **white = near, black = far**. This matches most artist-created depth maps.

For CG Z passes (where near = low value, far = high value), set `invert_depth=True` in `DepthPrepParams`:

```python
from depthforge import prep_depth, DepthPrepParams
depth = prep_depth(raw_z_pass, DepthPrepParams(invert=True))
```

### The depth conditioning pipeline

Raw depth maps often produce ghosting, fringing, or uncomfortable parallax. The `prep_depth()` function runs a 7-stage pipeline:

1. **Normalize** — maps the depth range to [0.0, 1.0]
2. **Invert** (optional) — flips near/far
3. **Bilateral smooth** — edge-aware blur that smooths noise without blurring depth edges
4. **Dilation** — expands near-object boundaries slightly to prevent background "peeking through"
5. **Falloff curve** — remaps depth distribution (linear, gamma, S-curve, etc.)
6. **Near/far clamp** — restricts depth to a comfortable range
7. **Region masks** — locally overrides depth in masked areas

```python
from depthforge import prep_depth, DepthPrepParams, FalloffCurve

depth = prep_depth(raw, DepthPrepParams(
    invert=False,
    bilateral_sigma_space=5.0,    # spatial radius of smooth
    bilateral_sigma_color=0.1,    # how much color difference stops the blur
    dilation_px=3,                # how many pixels to dilate near-object edges
    falloff_curve=FalloffCurve.S_CURVE,   # depth distribution remapping
    near_clip=0.05,               # discard depths closer than 5%
    far_clip=0.95,                # discard depths farther than 95%
))
```

---

## 7. Patterns and Tiles

### Built-in procedural patterns

```python
from depthforge import generate_pattern, PatternParams, PatternType, ColorMode

PatternType.RANDOM_NOISE      # pure random dots — classic SIRDS
PatternType.PERLIN            # smooth organic Perlin-like noise
PatternType.PLASMA            # smooth colorful plasma (demoscene style)
PatternType.VORONOI           # Worley/Voronoi cells
PatternType.GEOMETRIC_GRID    # dots, hexes, checkerboard, stripes
PatternType.MANDELBROT        # Mandelbrot fractal
PatternType.DOT_MATRIX        # structured dot grid
```

```python
# Example: Plasma in psychedelic colours, 128×128 tile
pattern = generate_pattern(PatternParams(
    pattern_type=PatternType.PLASMA,
    tile_width=128,
    tile_height=128,
    color_mode=ColorMode.PSYCHEDELIC,  # full HSV colour rotation
    seed=42,
    scale=1.5,    # pattern zoom level
))
```

### Color modes

| Mode | Description |
|---|---|
| `GREYSCALE` | Black-and-white output |
| `MONOCHROME` | Single hue — set with `hue` param (0.0–1.0) |
| `PSYCHEDELIC` | Full HSV colour rotation |
| `CUSTOM` | Provide your own colour palette |

### Using your own tile image

```python
from depthforge.core.pattern_gen import load_tile, tile_to_frame

# Load a PNG tile (will be tiled seamlessly across the output)
tile = load_tile("my_tile.png")

# Or stretch/tile it to match your output dimensions
pattern_frame = tile_to_frame(tile, height=1080, width=1920)
```

Any seamlessly tileable PNG works well. Non-seamless images will show visible repetition seams. Use an image editor's "offset and clone" technique to make tiles seamless before use.

---

## 8. Synthesis Modes

### 8.1 SIRDS (Random Dot)

The classic "magic eye" format. A field of dots where depth causes identical dot colors to appear in nearby positions, creating the 3D effect.

```python
from depthforge import synthesize, StereoParams
from depthforge import generate_pattern, PatternParams, PatternType, ColorMode

pattern = generate_pattern(PatternParams(
    pattern_type=PatternType.RANDOM_NOISE,
    tile_width=100,
    tile_height=100,
    color_mode=ColorMode.GREYSCALE,
    seed=1,
))

result = synthesize(depth, pattern, StereoParams(
    depth_factor=0.4,
    max_parallax_fraction=0.033,   # 1/30 of frame width
    seed=1,
    safe_mode=True,
))
```

**Tips for SIRDS:**
- Use `tile_width` of 80–150 pixels (this becomes the repeat period)
- Greyscale works best for the clearest depth perception
- Start with `depth_factor` around 0.3–0.4

### 8.2 Texture Pattern

Uses a colored pattern (procedural or custom) instead of random dots. Creates vibrant, eye-catching results.

```python
pattern = generate_pattern(PatternParams(
    pattern_type=PatternType.PLASMA,
    tile_width=128, tile_height=128,
    color_mode=ColorMode.PSYCHEDELIC,
    seed=7,
))

result = synthesize(depth, pattern, StereoParams(depth_factor=0.35))
```

### 8.3 Hidden Image

Encodes a recognizable shape inside the dot field. The shape is invisible at first glance but "pops out" as a 3D form.

```python
from depthforge.core.hidden_image import (
    encode_hidden_image, shape_to_mask, text_to_mask,
    HiddenImageParams, load_hidden_mask
)

# Built-in shapes: "circle", "square", "triangle", "star", "diamond", "arrow"
mask = shape_to_mask("star", width=1920, height=1080)

# Or hide text
mask = text_to_mask("HELLO", width=1920, height=1080, font_size=200)

# Or load a black-and-white image
mask = load_hidden_mask("my_logo.png", width=1920, height=1080)

# Generate the pattern tile
pattern = generate_pattern(PatternParams(
    pattern_type=PatternType.RANDOM_NOISE,
    tile_width=128, tile_height=128,
    color_mode=ColorMode.GREYSCALE, seed=3
))

# Encode the hidden image
result = encode_hidden_image(pattern, mask, HiddenImageParams(
    foreground_depth=0.7,   # depth of the hidden shape
    background_depth=0.1,   # depth of the background
    edge_soften_px=3,        # feather the edges
))
```

**Viewing tips for hidden images:**
- View from arm's length (about 50–60cm from a monitor)
- The image should be at least 1200 pixels wide for comfortable viewing
- Use `depth_scale=0.6` or higher to increase depth contrast

### 8.4 Anaglyph

Red/cyan color split for viewing with 3D anaglyph glasses.

```python
from depthforge.core.anaglyph import make_anaglyph_from_depth, AnaglyphParams, AnaglyphMode

# 5 modes available
AnaglyphMode.TRUE_ANAGLYPH      # pure R/C separation
AnaglyphMode.GREY_ANAGLYPH      # greyscale version
AnaglyphMode.COLOUR_ANAGLYPH    # colour-preserving
AnaglyphMode.HALF_COLOUR        # half colour (recommended for colour images)
AnaglyphMode.OPTIMISED          # Dubois least-squares (best quality)

result = make_anaglyph_from_depth(
    source=np.array(Image.open("photo.jpg").convert("RGBA")),
    depth=depth,
    params=AnaglyphParams(
        mode=AnaglyphMode.OPTIMISED,
        eye_separation=0.065,   # 65mm inter-ocular in normalized units
        convergence=0.0,         # convergence plane
        swap_eyes=False,
    )
)
```

**Anaglyph modes comparison:**

| Mode | Best for | Notes |
|---|---|---|
| `TRUE_ANAGLYPH` | Greyscale sources | Simplest, most ghosting |
| `GREY_ANAGLYPH` | Any source | Better ghosting, desaturated |
| `COLOUR_ANAGLYPH` | Colour sources | Colour preserved, more ghosting |
| `HALF_COLOUR` | Colour sources | Good balance |
| `OPTIMISED` | Any source | Best quality, Dubois matrices |

### 8.5 Stereo Pair

Generates separate left and right eye views for side-by-side 3D displays, VR content, or professional stereo workflows.

```python
from depthforge.core.stereo_pair import make_stereo_pair, compose_side_by_side, StereoLayout
from depthforge.core.stereo_pair import StereoParams as StereoPairParams

left, right, occlusion_mask = make_stereo_pair(
    source=source_image,
    depth=depth,
    params=StereoPairParams(
        eye_separation=0.065,
        convergence=0.0,
        bg_fill="edge",       # "edge", "mirror", or "black"
    )
)

# Side-by-side layout
sbs = compose_side_by_side(left, right, gap_px=0)
Image.fromarray(sbs).save("stereo_sbs.png")
```

The `occlusion_mask` (white = exposed background) can be fed into the inpainting module to fill newly visible background areas with plausible content rather than repeated edge pixels.

---

## 9. Stereo Controls Reference

### `StereoParams`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `depth_factor` | float | 0.3 | Parallax strength. 0 = flat, 1.0 = maximum. Negative inverts pop-out direction. |
| `max_parallax_fraction` | float | 1/30 | Maximum shift as fraction of frame width. Keep ≤ 0.05 for comfortable viewing. |
| `eye_separation_fraction` | float | 0.065 | Eye separation relative to frame width. |
| `convergence` | float | 0.0 | Where zero-parallax plane sits (0 = screen plane). |
| `invert_depth` | bool | False | Swap near/far in the depth map. |
| `oversample` | int | 1 | Render at N× then downsample. 2 gives noticeably smoother edges. |
| `seed` | int | None | Random seed for reproducible SIRDS patterns. |
| `safe_mode` | bool | False | Hard-clamps parallax to safe limits and reduces pattern contrast. |

### Recommended `depth_factor` values by use case

| Use case | `depth_factor` | Notes |
|---|---|---|
| First test | 0.2–0.3 | Subtle but easy to fuse |
| Standard | 0.3–0.45 | Good balance |
| Strong effect | 0.45–0.6 | Some viewers may struggle |
| Maximum (avoid public) | > 0.6 | Eye strain risk |

---

## 10. Depth Conditioning

### `DepthPrepParams` full reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `invert` | bool | False | Flip depth convention (near↔far) |
| `bilateral_sigma_space` | float | 5.0 | Spatial radius for bilateral smooth (pixels) |
| `bilateral_sigma_color` | float | 0.1 | Color range for bilateral smooth (0–1) |
| `dilation_px` | int | 3 | Morphological dilation of near-object boundaries |
| `falloff_curve` | FalloffCurve | LINEAR | Depth remapping curve |
| `near_clip` | float | 0.0 | Clip depths closer than this fraction |
| `far_clip` | float | 1.0 | Clip depths farther than this fraction |
| `region_masks` | list | [] | Per-region depth multipliers |

### Falloff curves

```
LINEAR:       depth unchanged
GAMMA:        subtle compression of near/far extremes
S_CURVE:      increased contrast in mid-range depths, compressed at extremes
LOGARITHMIC:  strong near detail, compressed far
EXPONENTIAL:  strong far detail, compressed near
```

### Region masks

Use region masks to locally reduce depth for areas with fine text, logos, or elements that shouldn't appear to float:

```python
from depthforge import prep_depth, DepthPrepParams, RegionMask
import numpy as np

# Create a mask (1.0 = full effect, 0.0 = no effect)
mask = np.zeros((H, W), dtype=np.float32)
mask[100:200, 100:400] = 1.0  # rectangle region

depth = prep_depth(raw, DepthPrepParams(
    region_masks=[RegionMask(mask=mask, multiplier=0.1)]  # 90% depth reduction in region
))
```

---

## 11. Using the Python API

### Module overview

```
depthforge/
├── core/
│   ├── synthesizer.py      # synthesize() — main entry point
│   ├── depth_prep.py       # prep_depth(), DepthPrepParams, FalloffCurve
│   ├── pattern_gen.py      # generate_pattern(), PatternParams, PatternType
│   ├── anaglyph.py         # make_anaglyph(), AnaglyphMode
│   ├── stereo_pair.py      # make_stereo_pair(), compose_side_by_side()
│   ├── hidden_image.py     # encode_hidden_image(), shape_to_mask(), text_to_mask()
│   ├── adaptive_dots.py    # generate_adaptive_dots()
│   └── inpainting.py       # inpaint_occlusion(), InpaintStrategy
```

### Top-level imports

```python
# Core synthesis
from depthforge import synthesize, StereoParams

# Depth conditioning
from depthforge import prep_depth, DepthPrepParams, FalloffCurve, RegionMask

# Pattern generation
from depthforge import generate_pattern, PatternParams, PatternType, ColorMode

# Capability check
from depthforge import capability_report
print(capability_report())
# → "OpenCV: ✓  SciPy: ✓  PyTorch: ✗  OpenEXR: ✗"
```

### Batch processing example

```python
import os
from pathlib import Path
from PIL import Image
import numpy as np
from depthforge import synthesize, prep_depth, generate_pattern
from depthforge import StereoParams, DepthPrepParams, PatternParams, PatternType, ColorMode

# Shared params
params = StereoParams(depth_factor=0.4, safe_mode=True)
prep = DepthPrepParams(bilateral_sigma_space=5, dilation_px=3)
pattern = generate_pattern(PatternParams(
    pattern_type=PatternType.PLASMA,
    tile_width=128, tile_height=128,
    color_mode=ColorMode.PSYCHEDELIC, seed=42
))

input_dir = Path("depth_maps/")
output_dir = Path("stereograms/")
output_dir.mkdir(exist_ok=True)

for depth_file in input_dir.glob("*.png"):
    raw = np.array(Image.open(depth_file).convert("L"), dtype=np.float32) / 255.0
    depth = prep_depth(raw, prep)
    result = synthesize(depth, pattern, params)
    out_path = output_dir / depth_file.name
    Image.fromarray(result).save(out_path)
    print(f"  → {out_path}")
```

---

## 12. CLI Usage (Phase 3)

> CLI will be available in Phase 3. This section documents the planned interface.

```bash
# Installation
pip install "depthforge[cli]"

# Basic SIRDS
depthforge sirds --depth depth.png --output out.png --depth-factor 0.4

# Texture pattern
depthforge texture \
  --depth depth.png \
  --pattern plasma \
  --color psychedelic \
  --tile-size 128 \
  --seed 42 \
  --output stereo.png

# Anaglyph
depthforge anaglyph \
  --source photo.jpg \
  --depth depth.png \
  --mode optimised \
  --output anaglyph.png

# Hidden image
depthforge hidden \
  --shape star \
  --width 1920 --height 1080 \
  --depth-factor 0.5 \
  --output hidden.png

# Hidden text
depthforge hidden \
  --text "HELLO WORLD" \
  --font-size 200 \
  --width 1920 --height 1080 \
  --output text_hidden.png

# Batch (folder of depth maps)
depthforge batch \
  --input-dir ./depth_maps/ \
  --output-dir ./stereograms/ \
  --mode texture \
  --pattern voronoi \
  --color psychedelic \
  --jobs 8       # parallel workers

# Safe mode (epilepsy-safe settings)
depthforge sirds --depth depth.png --safe-mode --output safe_stereo.png
```

### CLI flags reference

| Flag | Description |
|---|---|
| `--depth-factor FLOAT` | Parallax strength (0.0–1.0) |
| `--max-parallax FLOAT` | Max shift as fraction of width (default 0.033) |
| `--safe-mode` | Hard-clamp all values to safety limits |
| `--oversample INT` | Supersampling (1=none, 2=2x) |
| `--seed INT` | Random seed for reproducibility |
| `--invert-depth` | Flip near/far in depth map |
| `--format [png\|tiff\|jpg]` | Output format |
| `--jobs INT` | Parallel worker count (batch mode) |
| `--verbose` | Progress output |
| `--json-log PATH` | Machine-readable log (farm mode) |

---

## 13. OFX Plugin (Phase 5)

> The OFX plugin will be delivered in Phase 5. This section documents planned behaviour.

### Supported hosts

- Adobe After Effects CC 2022+
- Adobe Premiere Pro CC 2022+
- DaVinci Resolve 18+
- Blackmagic Fusion 18+
- HitFilm Pro 2022+
- VEGAS Pro 20+
- Natron 2.5+

### Installation paths

| Host | Windows | macOS | Linux |
|---|---|---|---|
| AE / Premiere | `C:\Program Files\Common Files\OFX\Plugins\` | `/Library/OFX/Plugins/` | `/usr/OFX/Plugins/` |
| DaVinci Resolve | `C:\Program Files\Common Files\OFX\Plugins\` | `/Library/OFX/Plugins/` | `/usr/OFX/Plugins/` |
| Natron | `C:\Program Files\Common Files\OFX\Plugins\` | `/Library/OFX/Plugins/` | `/usr/OFX/Plugins/` |

Copy the `DepthForge.ofx.bundle` folder to the appropriate path and restart the host application.

### Plugin inputs (in host)

| Input | Type | Description |
|---|---|---|
| Source | RGBA | Source image (optional — used for anaglyph mode) |
| Depth | Greyscale/Float | Depth map |
| Pattern | RGBA | Custom tile (leave empty for built-in procedural) |
| Mask | Greyscale | Region depth override mask |

---

## 14. Nuke Gizmo (Phase 4)

> Nuke integration is planned for Phase 4.

### Installation

```python
# In your Nuke init.py or menu.py:
import sys
sys.path.insert(0, "/path/to/depthforge")
import depthforge.nuke
depthforge.nuke.install()
```

### Gizmo inputs

| Knob | Type | Description |
|---|---|---|
| `depth` | Float | Normalized depth map (0–1 float) |
| `pattern` | RGBA | Custom tile (blank = SIRDS random dots) |
| `source` | RGBA | Source plate (anaglyph mode only) |
| `region_mask` | Float | Per-region depth reduction mask |

### Gizmo outputs

| Channel | Description |
|---|---|
| `rgba` | Stereogram output |
| `left` / `right` | Native Nuke stereo view channels |
| `occlusion` | Alpha mask of newly exposed background |

### Farm rendering with Nuke

Add DepthForge to your farm Nuke environment:

```bash
# In your render farm submit script or environment setup:
export NUKE_PATH=/path/to/depthforge:$NUKE_PATH
```

---

## 15. ComfyUI Nodes (Phase 4)

> ComfyUI integration is planned for Phase 4.

### Installation

```bash
cd ComfyUI/custom_nodes/
pip install depthforge
# OR
git clone https://github.com/your-org/depthforge.git
```

### Node catalog

| Node | Category | Description |
|---|---|---|
| `DF_DepthFromText` | Input | Text prompt → AI depth map |
| `DF_DepthFromImage` | Input | Color/grayscale image → depth map (MiDaS) |
| `DF_DepthPrep` | Processing | Smooth, dilate, curve, clamp depth |
| `DF_PatternGen` | Processing | Generate procedural pattern tile |
| `DF_Stereogram` | Synthesis | Core synthesis (SIRDS or texture) |
| `DF_AnaglyphOut` | Output | Red/cyan anaglyph output |
| `DF_StereoPair` | Output | L/R pair or side-by-side |
| `DF_HiddenImage` | Synthesis | Hidden shape/text stereogram |
| `DF_VideoSequence` | Video | Per-frame video processing |
| `DF_QCOverlay` | QC | Parallax heatmap, safe-zone indicator |

---

## 16. Tips for Good Results

### Depth map quality is everything

The single biggest factor in stereogram quality is the depth map. Spend time on it.
- Use bilateral smoothing to remove noise without blurring important depth edges
- Apply dilation to prevent background "peeking" at object edges
- Normalise to the full 0–1 range

### Start conservative, then push

Begin with `depth_factor=0.25` and `safe_mode=True`. Once the image fuses cleanly, gradually increase `depth_factor`. Most viewers are comfortable up to about 0.45. Beyond 0.6, you risk eye strain for the majority of viewers.

### Tile width affects viewing distance

A wider tile (`tile_width=200+`) feels more comfortable from a longer viewing distance (far from the screen). A narrower tile (`tile_width=80`) works better when held close. Match your tile width to the expected viewing context.

### Oversample for smoother results

```python
StereoParams(oversample=2)
```

This renders at 2× resolution then Lanczos-downsamples, producing noticeably smoother dot edges and less aliasing at depth transitions. The processing time doubles, but results are worth it for final output.

### Pattern contrast affects fusability

Very high-contrast patterns (especially high-frequency noise) are harder to fuse than smoother patterns like Perlin or plasma. If your image is hard to fuse, try a lower-contrast pattern or reduce `max_parallax_fraction`.

### Test with greyscale first

If you're troubleshooting, switch to `ColorMode.GREYSCALE` and `PatternType.RANDOM_NOISE` first. Colour and complex patterns add visual noise that makes depth perception harder to verify.

---

## 17. Troubleshooting

### "The stereogram won't fuse — I can't see the 3D"

Most common causes:

1. **Parallax too strong** — reduce `depth_factor` below 0.3 and try again
2. **Tile too narrow or too wide** — try `tile_width=120` as a starting point
3. **Depth map has no contrast** — check that your depth map uses the full range from black to white
4. **Viewing technique** — try both parallel and cross-eyed methods; some people find one much easier

### "I see ghosting or double edges"

The depth map has hard edges at depth transitions. Fix:

```python
DepthPrepParams(bilateral_sigma_space=7, dilation_px=5)
```

### "The pattern looks stretched or warped"

The pattern tile aspect ratio is mismatched with the output resolution, or the tile is too small relative to the output. Use a square tile (e.g. 128×128) and ensure it's at least 1/10 of the output width.

### "The anaglyph has too much colour fringing"

Use `AnaglyphMode.OPTIMISED` (Dubois matrices) for the lowest fringing. Also try desaturating the source image slightly before processing.

### "Video stereogram has flickering / shimmering"

Enable temporal smoothing (Phase 5). As a workaround, ensure your depth maps change slowly between frames — apply temporal blur to the depth sequence in your compositing package before passing to DepthForge.

### "The hidden image isn't visible"

- Increase `foreground_depth` and decrease `background_depth` for more contrast
- View from arm's length (not up close)
- Try `edge_soften_px=5` for cleaner shape edges
- Ensure the hidden shape is at least 1/4 of the image width

### "My depth map looks inverted (near is far)"

Set `invert=True` in `DepthPrepParams`. This is common with CG Z-depth passes which default to near=0, far=large_number.

### "ComfyUI node not found after installing"

1. Confirm `pip install depthforge` ran without errors
2. Restart ComfyUI completely (not just reload)
3. Check that the DepthForge folder appears in `ComfyUI/custom_nodes/`
4. Look for any import errors in the ComfyUI console at startup

### "Nuke gizmo throws Python errors on render farm"

The render nodes can't find the `depthforge` package. Add this to your farm's `init.py`:

```python
import sys
sys.path.insert(0, "/path/to/depthforge")
```

Or set `NUKE_PATH` in your farm environment to include the DepthForge directory.

### "OFX plugin doesn't appear in host"

1. Check the `.ofx.bundle` is in the correct OFX plugin directory for your OS (see [OFX Plugin section](#13-ofx-plugin-phase-5))
2. Confirm the plugin is 64-bit (all modern hosts require 64-bit OFX)
3. Restart the host application fully after installing
4. Check the host's OFX plugin manager/scan log for any load errors

---

*DepthForge User Guide v0.1 — Phase 1 Release*
*Full API reference: [API_REFERENCE.md](API_REFERENCE.md)*
*Troubleshooting FAQ: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)*
