# DepthForge User Guide

**Version 0.1.0 — Phase 1 (Core Engine)**

---

## Table of Contents

1. [Installation](#1-installation)
2. [Core Concepts](#2-core-concepts)
3. [Your First Stereogram](#3-your-first-stereogram)
4. [Depth Maps](#4-depth-maps)
5. [Patterns](#5-patterns)
6. [Stereogram Modes](#6-stereogram-modes)
7. [Anaglyph Output](#7-anaglyph-output)
8. [Stereo Pairs](#8-stereo-pairs)
9. [Hidden Images](#9-hidden-images)
10. [Adaptive Dot Density](#10-adaptive-dot-density)
11. [Inpainting Occlusion](#11-inpainting-occlusion)
12. [Parameters Reference](#12-parameters-reference)
13. [Presets](#13-presets)
14. [Working with Video](#14-working-with-video)
15. [Pipeline Examples](#15-pipeline-examples)

---

## 1. Installation

### System Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.9 | 3.11+ |
| RAM | 2 GB | 8 GB (16 GB for 4K+) |
| GPU | None | CUDA 11.8+ for AI depth |
| OS | Windows / macOS / Linux | Any |

### Step 1 — Install Python

Download Python 3.11 from [python.org](https://python.org) or use your system package manager.

```bash
python --version    # should show 3.9 or higher
```

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv depthforge-env

# Activate — macOS / Linux:
source depthforge-env/bin/activate

# Activate — Windows:
depthforge-env\Scripts\activate
```

### Step 3 — Install DepthForge

**Option A — Minimum (NumPy + Pillow only, works everywhere):**
```bash
pip install depthforge
```

**Option B — Recommended (adds OpenCV + SciPy for better quality):**
```bash
pip install "depthforge[cv2,scipy]"
```

**Option C — With AI depth estimation (requires ~4 GB download):**
```bash
pip install "depthforge[ai]"
```

**Option D — Everything:**
```bash
pip install "depthforge[full]"
```

**Option E — From source (for development):**
```bash
git clone https://github.com/depthforge/depthforge.git
cd depthforge
pip install -e ".[dev]"
```

### Step 4 — Verify installation

```python
import depthforge as df
print(df.__version__)

# Check which optional packages are available
print("OpenCV:  ", df.HAS_CV2)
print("SciPy:   ", df.HAS_SCIPY)
print("PyTorch: ", df.HAS_TORCH)
```

### Nuke Installation (Phase 4)

> Coming in Phase 4. The Nuke gizmo will be a single Python file
> dropped into your `~/.nuke/` folder.

### ComfyUI Installation (Phase 4)

> Coming in Phase 4. Install via ComfyUI Manager or:
> ```bash
> cd ComfyUI/custom_nodes
> git clone https://github.com/depthforge/depthforge-comfyui.git
> pip install depthforge
> ```

### OFX Installation (Phase 5)

> Coming in Phase 5. Standard OFX `.bundle` file installed to:
> - macOS: `/Library/OFX/Plugins/`
> - Windows: `C:\Program Files\Common Files\OFX\Plugins\`
> - Linux: `/usr/OFX/Plugins/`

---

## 2. Core Concepts

### How a stereogram works

A stereogram exploits the fact that your brain fuses two slightly different
images — one per eye — into a single 3D perception. In a SIRDS (Single Image
Random Dot Stereogram), the same image encodes *both* left and right eye views
by repeating colour patterns at a horizontal distance proportional to depth.

The further apart two identical dots are, the closer that part of the scene
appears to "pop out."

### The pipeline

Every DepthForge output follows the same three-stage pipeline:

```
Raw input
    ↓
[1] Depth acquisition    → float32 [0,1] depth map
    ↓
[2] Depth conditioning   → smooth, dilate, remap, clamp
    ↓
[3] Synthesis            → stereogram, anaglyph, or stereo pair
```

### Depth conventions

DepthForge uses **white = near, black = far** (0.0 = farthest, 1.0 = nearest).

If your depth source uses the opposite convention (common in CG Z-passes
where 0 = near/camera, large values = far), set `invert_depth=True`.

### Dependency tiers

The core runs on NumPy + Pillow alone. Optional packages improve quality:

```
Tier 0 (always):  NumPy + Pillow   — works everywhere, no extras needed
Tier 1 (better):  + OpenCV         — proper bilateral filter, fast dilation
Tier 2 (best):    + SciPy          — advanced morphological operations
Tier 3 (AI):      + PyTorch        — MiDaS / ZoeDepth depth estimation
```

Nothing breaks without them — DepthForge falls back gracefully.

---

## 3. Your First Stereogram

### From a B&W image (simplest case)

Any greyscale image works as a depth map — white regions pop forward,
black regions recede.

```python
import depthforge as df
import numpy as np
from PIL import Image

# Load your depth map (any B&W image)
depth_raw = np.asarray(Image.open("my_bw_image.png").convert("L"),
                        dtype=np.float32) / 255.0

# Condition the depth
depth = df.prep_depth(depth_raw)

# Generate a random-dot pattern
pattern = df.generate_pattern(df.PatternParams(
    pattern_type = df.PatternType.RANDOM_NOISE,
    tile_width   = 64,
    tile_height  = 64,
    color_mode   = df.ColorMode.GREYSCALE,
    seed         = 42,
))

# Synthesize the stereogram
stereo = df.synthesize(depth, pattern)

# Save
df.save_stereogram(stereo, "my_first_stereogram.png")
print("Saved!")
```

### From a text prompt (requires AI deps)

> Phase 3 feature — AI depth from text prompts via the CLI and ComfyUI.

### From scratch with a procedural depth

```python
import depthforge as df
import numpy as np

# Create a sphere depth map programmatically
H, W = 512, 512
ys = np.linspace(-1, 1, H)[:, None]
xs = np.linspace(-1, 1, W)[None, :]
depth = np.sqrt(np.maximum(1.0 - xs**2 - ys**2, 0)).astype(np.float32)

# Psychedelic plasma pattern
pattern = df.generate_pattern(df.PatternParams(
    pattern_type = df.PatternType.PLASMA,
    tile_width   = 256,
    tile_height  = 256,
    color_mode   = df.ColorMode.PSYCHEDELIC,
    seed         = 7,
))

# Synthesize
depth_c = df.prep_depth(depth, df.DepthPrepParams(dilation_px=4))
stereo  = df.synthesize(depth_c, pattern, df.StereoParams(depth_factor=0.4))
df.save_stereogram(stereo, "sphere_plasma.png")
```

---

## 4. Depth Maps

### Accepted input formats

| Format | Notes |
|---|---|
| PNG greyscale (8-bit) | Standard; load with Pillow |
| PNG greyscale (16-bit) | Higher precision; `depth_from_image()` handles automatically |
| TIFF (8 or 16-bit) | Common in VFX pipelines |
| EXR Z-pass | Requires OpenEXR (Phase 3) |
| NumPy float32 array | Direct API use |
| Any Pillow-readable | Auto-converted to luminance |

### Loading depth maps

```python
import depthforge as df

# From any image file
depth = df.depth_from_image("depth.png")           # runs prep pipeline
depth = df.load_depth_image("depth.png")           # raw load, no prep

# From a NumPy array
import numpy as np
raw_z = np.load("z_pass.npy")                      # your CG Z-pass
depth = df.prep_depth(raw_z, df.DepthPrepParams(invert=True))  # Z-passes often inverted
```

### Depth conditioning parameters

```python
params = df.DepthPrepParams(
    normalise             = True,     # re-normalise to [0,1] (recommended)
    invert                = False,    # flip near/far convention
    bilateral_sigma_space = 5.0,      # edge-aware smooth radius
    bilateral_sigma_color = 0.1,      # how much to smooth across depth edges
    dilation_px           = 3,        # expand near regions (prevents fringing)
    smooth_passes         = 1,        # number of smooth iterations
    falloff_curve         = df.FalloffCurve.S_CURVE,  # depth distribution remap
    near_plane            = 0.0,      # clip far values below this
    far_plane             = 1.0,      # clip near values above this
    edge_preserve         = True,     # use bilateral vs Gaussian
)
```

### Falloff curves

Control how depth values are distributed:

| Curve | Effect | When to use |
|---|---|---|
| `LINEAR` | No change | Default; predictable |
| `S_CURVE` | Ease in/out | Most natural-looking results |
| `GAMMA` | Power-law remap | Compress near or far |
| `LOGARITHMIC` | Emphasise near differences | Portrait work |
| `EXPONENTIAL` | Emphasise far differences | Landscape/wide shots |

```python
df.DepthPrepParams(
    falloff_curve  = df.FalloffCurve.GAMMA,
    falloff_gamma  = 0.5,   # < 1 brightens (stretches near), > 1 darkens
)
```

### Region masks — local depth override

Reduce depth in areas with fine detail, text, or logos that would
become illegible at full parallax:

```python
import numpy as np

# Create a mask for the bottom-right corner
mask = np.zeros((H, W), dtype=np.float32)
mask[H//2:, W//2:] = 1.0   # bottom-right = full mask

params = df.DepthPrepParams(
    region_masks = [
        df.RegionMask(mask=mask, multiplier=0.2)  # reduce to 20% depth
    ]
)
```

### Vergence comfort check

```python
# Check if your depth map will cause eye strain
vmap = df.compute_vergence_map(depth, eye_sep_fraction=0.06)
max_vergence = float(vmap.max())
print(f"Max vergence: {max_vergence:.2f}°")  # comfortable < ~3°
```

---

## 5. Patterns

### Built-in pattern types

```python
from depthforge import PatternType, ColorMode, GridStyle

# Random noise (classic SIRDS)
df.PatternParams(pattern_type=PatternType.RANDOM_NOISE, ...)

# Smooth organic noise
df.PatternParams(pattern_type=PatternType.PERLIN, octaves=4, scale=1.0, ...)

# Demoscene plasma
df.PatternParams(pattern_type=PatternType.PLASMA, ...)

# Cellular / Worley noise
df.PatternParams(pattern_type=PatternType.VORONOI, ...)

# Geometric grids
df.PatternParams(pattern_type=PatternType.GEOMETRIC_GRID,
                 grid_style=GridStyle.DOTS,     # DOTS | HEXES | CHECKS | STRIPES
                 grid_spacing=12, ...)

# Mandelbrot fractal
df.PatternParams(pattern_type=PatternType.MANDELBROT, scale=1.0, ...)

# Halftone dot matrix
df.PatternParams(pattern_type=PatternType.DOT_MATRIX, grid_spacing=10, ...)
```

### Colour modes

```python
# Classic SIRDS (greyscale dots)
PatternParams(..., color_mode=ColorMode.GREYSCALE)

# Single hue, varying brightness
PatternParams(..., color_mode=ColorMode.MONOCHROME, hue=0.6, saturation=0.8)

# Full HSV rainbow (psychedelic)
PatternParams(..., color_mode=ColorMode.PSYCHEDELIC)
```

### Loading a custom tile

```python
# Load any image as a repeating tile
tile = df.load_tile("my_logo.png", tile_width=256, tile_height=256)
stereo = df.synthesize(depth, tile)

# Tile a pattern to fill a full frame
frame = df.tile_to_frame(tile, height=1080, width=1920)
```

### Tile sizing guidelines

| Resolution | Recommended tile size | Notes |
|---|---|---|
| HD (1920×1080) | 128–256 px | |
| 4K (3840×2160) | 256–512 px | |
| Print (300 dpi) | 256–512 px | Larger tiles for print |
| SIRDS only | 32–128 px | Random noise tiles can be small |

### Safe mode — photosensitivity

```python
PatternParams(safe_mode=True)   # reduces pattern contrast to 50%
```

Always enable for psychedelic patterns in public-facing content.

---

## 6. Stereogram Modes

### SIRDS (Single Image Random Dot Stereogram)

The classic magic-eye style — random dots carry the 3D information.

```python
# Standard SIRDS
pattern = df.generate_pattern(df.PatternParams(
    pattern_type = df.PatternType.RANDOM_NOISE,
    tile_width   = 64,
    color_mode   = df.ColorMode.GREYSCALE,
    seed         = 42,
))
stereo = df.synthesize(depth, pattern, df.StereoParams(depth_factor=0.35))
```

### Texture pattern stereogram

Use any image as the repeating texture:

```python
pattern = df.generate_pattern(df.PatternParams(
    pattern_type = df.PatternType.PLASMA,
    tile_width   = 256,
    color_mode   = df.ColorMode.PSYCHEDELIC,
))
stereo = df.synthesize(depth, pattern, df.StereoParams(depth_factor=0.4))
```

### StereoParams — all controls

```python
df.StereoParams(
    depth_factor            = 0.4,    # parallax strength (-1 to 1)
    max_parallax_fraction   = 1/30,   # max shift as fraction of width
    eye_separation_fraction = 0.06,   # assumed inter-ocular / viewing distance
    convergence             = 0.5,    # which depth is at screen plane (0-1)
    invert_depth            = False,  # flip near/far
    oversample              = 1,      # 1 = native, 2 = 2x supersample
    seed                    = None,   # random seed for reproducibility
    safe_mode               = False,  # hard-clamp to comfort limits
)
```

### Choosing depth_factor

| Value | Effect | Use case |
|---|---|---|
| 0.2 | Subtle | Print, large format, long viewing distance |
| 0.35 | Moderate | Web, screen viewing — good default |
| 0.5 | Strong | Close viewing, impact shots |
| 0.7+ | Extreme | Creative/experimental only |
| Negative | Pops forward instead of receding | Pop-out effect |

### Convergence point

Controls which depth level appears *at* the screen plane:

```python
df.StereoParams(convergence=0.0)   # all depth recedes behind screen
df.StereoParams(convergence=0.5)   # mid-depth at screen (default)
df.StereoParams(convergence=1.0)   # all depth pops forward
```

---

## 7. Anaglyph Output

Produces red/cyan glasses-compatible 3D images.

### From a source image + depth

```python
import depthforge as df
import numpy as np
from PIL import Image

source = np.asarray(Image.open("photo.jpg").convert("RGBA"))
depth  = df.load_depth_image("depth.png")
depth  = df.prep_depth(depth)

anaglyph = df.make_anaglyph_from_depth(
    source = source,
    depth  = depth,
    params = df.AnaglyphParams(
        mode        = df.AnaglyphMode.OPTIMISED,  # best quality
        parallax_px = 15,
    )
)

from PIL import Image
Image.fromarray(anaglyph[:,:,:3]).save("anaglyph.png")
```

### Anaglyph modes compared

| Mode | Quality | Colour | Speed | Best for |
|---|---|---|---|---|
| `TRUE_ANAGLYPH` | Good | Colour loss | Fast | Quick previews |
| `GREY_ANAGLYPH` | Good | None (BW) | Fast | High contrast scenes |
| `HALF_COLOUR` | Better | Partial | Fast | General use |
| `COLOUR_ANAGLYPH` | Good | Full | Fast | Alias for TRUE |
| `OPTIMISED` | Best | Best | Moderate | Final output |

The `OPTIMISED` (Dubois) mode uses mathematically derived matrices to minimise
retinal rivalry and colour ghosting. Use it for all final deliverables.

### From a pre-computed stereo pair

```python
left, right, _ = df.make_stereo_pair(source, depth)
anaglyph = df.make_anaglyph(left, right, df.AnaglyphParams(mode=df.AnaglyphMode.OPTIMISED))
```

---

## 8. Stereo Pairs

### Basic stereo pair

```python
left, right, occlusion_mask = df.make_stereo_pair(
    source = source_rgba,    # RGBA uint8
    depth  = depth,          # float32
    params = df.StereoPairParams(
        max_parallax_fraction = 1/30,   # comfortable limit
        eye_balance           = 0.5,    # symmetric (L and R both shift)
        layout                = df.StereoLayout.SEPARATE,
        feather_px            = 3,      # soften occlusion mask edges
        background_fill       = "edge", # how to fill exposed regions
    )
)
```

### Layout options

```python
# Save separately
Image.fromarray(left[:,:,:3]).save("left.png")
Image.fromarray(right[:,:,:3]).save("right.png")

# Side-by-side in one image
combined = df.compose_side_by_side(left, right, gap_px=4)

# Or request directly:
combined, _, occ = df.make_stereo_pair(source, depth,
    df.StereoPairParams(layout=df.StereoLayout.SIDE_BY_SIDE))
```

### Occlusion mask

The occlusion mask (third return value) marks where background is exposed
because near objects have "moved" to reveal what was behind them.
Values are float32 [0=visible, 1=occluded].

```python
left, right, occ = df.make_stereo_pair(source, depth)

# Visualise
occ_visual = (occ * 255).astype(np.uint8)
Image.fromarray(occ_visual).save("occlusion_mask.png")

# Fill with inpainting (see section 11)
left_clean = df.inpaint_occlusion(left, occ)
```

### Eye balance

```python
StereoPairParams(eye_balance=0.5)   # both eyes shift equally (default)
StereoPairParams(eye_balance=0.0)   # only right eye shifts (left = source)
StereoPairParams(eye_balance=1.0)   # only left eye shifts (right = source)
```

---

## 9. Hidden Images

The "magic eye" mode — hide a recognisable shape inside the dot noise.
Viewers defocus their eyes and the shape emerges from the pattern.

### Built-in shapes

```python
# Primitive shapes
mask = df.shape_to_mask("circle",   width=512, height=512)
mask = df.shape_to_mask("star",     width=512, height=512)
mask = df.shape_to_mask("diamond",  width=512, height=512)
mask = df.shape_to_mask("triangle", width=512, height=512)
mask = df.shape_to_mask("arrow",    width=512, height=512)
mask = df.shape_to_mask("square",   width=512, height=512)
```

### Hidden text

```python
mask = df.text_to_mask(
    text       = "HELLO",
    width      = 512,
    height     = 256,
    font_size  = 0,           # 0 = auto-fit
    font_path  = None,        # None = default font; or path to .ttf
    padding    = 30,
)
```

### From a custom image

```python
# Any B&W image works — white = hidden shape
mask = df.load_hidden_mask("logo.png", target_size=(512, 512))
```

### Encoding

```python
pattern = df.generate_pattern(df.PatternParams(
    df.PatternType.RANDOM_NOISE, 64, 64, df.ColorMode.GREYSCALE, seed=0
))
stereo = df.encode_hidden_image(
    pattern = pattern,
    mask    = mask,
    params  = df.HiddenImageParams(
        foreground_depth = 0.85,   # how much the shape pops forward
        background_depth = 0.0,    # flat background
        edge_soften_px   = 6,      # smooth shape edges for easier fusion
        depth_scale      = 1.0,    # additional contrast on shape depth
        invert_mask      = False,  # True = dark areas become the shape
    )
)
```

### Tips for good hidden images

- Use `edge_soften_px=4–8` — sharp edges are harder to fuse
- `foreground_depth` between 0.7–0.9 gives good pop
- Simple shapes (circles, stars) fuse more reliably than fine text
- Avoid very small features — they get lost in the dot noise
- Random-dot patterns work best for hidden images; texture patterns can obscure the hidden shape

---

## 10. Adaptive Dot Density

Standard SIRDS uses uniform dots everywhere. Adaptive density varies the
dot size based on local depth complexity — smaller, denser dots near edges
and detail areas; larger, sparser dots in flat regions.

```python
depth  = df.prep_depth(raw_depth)

params = df.AdaptiveDotParams(
    tile_width      = 256,
    tile_height     = 256,
    min_dot_radius  = 1,     # smallest dots in complex areas
    max_dot_radius  = 5,     # largest dots in flat areas
    min_spacing     = 3,     # densest packing
    max_spacing     = 14,    # sparsest packing
    n_levels        = 5,     # number of density levels
    complexity_blur = 2.0,   # smooth the complexity map
    dot_color       = (255, 255, 255),
    bg_color        = (0, 0, 0),
    jitter          = 0.3,   # positional randomness (0=grid, 1=full random)
    seed            = 42,
)

# Generate the adaptive dot tile
tile   = df.generate_adaptive_dots(depth, params)

# Use it in synthesis
stereo = df.synthesize(depth, tile, df.StereoParams(depth_factor=0.4))
```

### Inspect the complexity map

```python
cmap = df.complexity_from_depth(depth)
# cmap is float32 [0,1] — 1.0 = high edge/detail complexity
Image.fromarray((cmap * 255).astype(np.uint8)).save("complexity.png")
```

---

## 11. Inpainting Occlusion

When stereo pairs are generated, depth discontinuities expose background
regions. Fill these with the inpainting module.

### Automatic (patch-based, no AI)

```python
left, right, occ = df.make_stereo_pair(source, depth)
left_clean = df.inpaint_occlusion(left, occ)
```

### With a clean plate

The best quality option when you have a background plate:

```python
background = np.asarray(Image.open("clean_plate.png").convert("RGBA"))

params = df.InpaintParams(
    method      = df.InpaintMethod.CLEAN_PLATE,
    clean_plate = background,
    blend_px    = 5,           # feather boundary
)
left_clean = df.inpaint_occlusion(left, occ, params)
```

### With an AI model (ComfyUI / Phase 4)

```python
# Register your SD inpainting model as a callback
def my_ai_inpaint(image_rgba, mask_float):
    # ... call your SD inpainting model ...
    return filled_rgba

df.register_ai_inpaint_callback(my_ai_inpaint)

# Now AUTO method will use it
params = df.InpaintParams(method=df.InpaintMethod.AUTO)
left_clean = df.inpaint_occlusion(left, occ, params)
```

### Inpainting methods compared

| Method | Quality | Speed | Requirements |
|---|---|---|---|
| `EDGE_EXTEND` | Fast/rough | Fastest | None |
| `PATCH_BASED` | Good | Moderate | OpenCV (fast) or NumPy (slow) |
| `CLEAN_PLATE` | Excellent | Fast | Clean background plate |
| `AI_CALLBACK` | Best | Slow | SD inpainting model |
| `AUTO` | Best available | — | Uses best of above |

---

## 12. Parameters Reference

### StereoParams

| Parameter | Type | Default | Description |
|---|---|---|---|
| `depth_factor` | float | 0.4 | Parallax multiplier (-1 to 1) |
| `max_parallax_fraction` | float | 1/30 | Max shift as fraction of width |
| `eye_separation_fraction` | float | 0.06 | Inter-ocular / viewing distance ratio |
| `convergence` | float | 0.5 | Screen plane depth (0–1) |
| `invert_depth` | bool | False | Flip near/far convention |
| `oversample` | int | 1 | Supersampling factor (1–4) |
| `seed` | int\|None | None | Random seed |
| `safe_mode` | bool | False | Hard-clamp comfort limits |

### DepthPrepParams

| Parameter | Type | Default | Description |
|---|---|---|---|
| `normalise` | bool | True | Normalise to [0,1] |
| `invert` | bool | False | Flip depth convention |
| `bilateral_sigma_space` | float | 5.0 | Spatial blur radius |
| `bilateral_sigma_color` | float | 0.1 | Edge sensitivity |
| `dilation_px` | int | 3 | Near-region expansion (px) |
| `smooth_passes` | int | 1 | Number of smooth iterations |
| `falloff_curve` | FalloffCurve | LINEAR | Depth distribution remap |
| `falloff_gamma` | float | 1.0 | Gamma for GAMMA curve |
| `near_plane` | float | 0.0 | Clip floor |
| `far_plane` | float | 1.0 | Clip ceiling |
| `region_masks` | list | [] | Local depth overrides |
| `edge_preserve` | bool | True | Bilateral vs Gaussian |

### PatternParams

| Parameter | Type | Default | Description |
|---|---|---|---|
| `pattern_type` | PatternType | RANDOM_NOISE | Generator type |
| `tile_width` | int | 128 | Tile width (px) |
| `tile_height` | int | 128 | Tile height (px) |
| `color_mode` | ColorMode | GREYSCALE | Colour scheme |
| `hue` | float | 0.0 | Base hue [0,1] (MONOCHROME) |
| `saturation` | float | 0.8 | Saturation [0,1] (MONOCHROME) |
| `scale` | float | 1.0 | Feature size multiplier |
| `octaves` | int | 4 | Noise octaves (Perlin/Plasma) |
| `grid_style` | GridStyle | DOTS | Sub-style for GEOMETRIC_GRID |
| `grid_spacing` | int | 8 | Grid cell size (px) |
| `seed` | int\|None | None | Random seed |
| `safe_mode` | bool | False | Limit contrast |

---

## 13. Presets

> Phase 3: JSON preset library with import/export (coming soon).
>
> Planned presets: `shallow`, `medium`, `deep`, `cinema`, `print_300dpi`,
> `web_srgb`, `broadcast_rec709`, `screen`.

For now, use these values as starting points:

```python
# Shallow — comfortable for any viewer
shallow = df.StereoParams(depth_factor=0.2, max_parallax_fraction=0.025)

# Medium — good default for web/screen
medium  = df.StereoParams(depth_factor=0.35, max_parallax_fraction=0.033)

# Deep — dramatic, close viewing
deep    = df.StereoParams(depth_factor=0.55, max_parallax_fraction=0.05)

# Print — large format, viewed from distance
print_  = df.StereoParams(depth_factor=0.25, max_parallax_fraction=0.02,
                           safe_mode=True)
```

---

## 14. Working with Video

> Phase 2/5: Full video pipeline with temporal smoothing (coming soon).
>
> For now, process video frame-by-frame:

```python
import depthforge as df
import numpy as np
from PIL import Image
import os

depth_dir  = "frames/depth/"
output_dir = "frames/stereo/"
os.makedirs(output_dir, exist_ok=True)

pattern = df.generate_pattern(df.PatternParams(
    df.PatternType.PLASMA, 256, 256, df.ColorMode.PSYCHEDELIC, seed=0
))
params  = df.StereoParams(depth_factor=0.35, seed=0)
prep    = df.DepthPrepParams(bilateral_sigma_space=5, dilation_px=3)

for i, fname in enumerate(sorted(os.listdir(depth_dir))):
    depth_raw = df.load_depth_image(os.path.join(depth_dir, fname))
    depth     = df.prep_depth(depth_raw, prep)
    stereo    = df.synthesize(depth, pattern, params)
    df.save_stereogram(stereo, os.path.join(output_dir, fname))
    print(f"\r  Frame {i+1}", end="")

print("\nDone.")
```

**Key notes for video:**
- Fix the `seed` in `StereoParams` to keep the random dot field locked across frames
- Use the same `pattern` tile for all frames to prevent flickering
- Coming in Phase 2: temporal depth smoothing to prevent depth "shimmer"

---

## 15. Pipeline Examples

### Example A: Photo → Anaglyph print

```python
import depthforge as df
import numpy as np
from PIL import Image

photo = np.asarray(Image.open("portrait.jpg").convert("RGBA"))
depth = df.depth_from_image("portrait_depth.png",
            df.DepthPrepParams(bilateral_sigma_space=7, dilation_px=4,
                               falloff_curve=df.FalloffCurve.S_CURVE))

anaglyph = df.make_anaglyph_from_depth(
    photo, depth,
    df.AnaglyphParams(mode=df.AnaglyphMode.OPTIMISED, parallax_px=18)
)
Image.fromarray(anaglyph[:,:,:3]).save("portrait_anaglyph.png")
```

### Example B: Logo → hidden image SIRDS

```python
import depthforge as df

mask    = df.load_hidden_mask("company_logo.png", target_size=(512, 512))
pattern = df.generate_pattern(df.PatternParams(
    df.PatternType.RANDOM_NOISE, 64, 64, df.ColorMode.GREYSCALE, seed=42))
stereo  = df.encode_hidden_image(pattern, mask,
            df.HiddenImageParams(foreground_depth=0.8, edge_soften_px=6))
df.save_stereogram(stereo, "logo_hidden.png")
```

### Example C: CG Z-pass → stereo pair for compositing

```python
import depthforge as df
import numpy as np
from PIL import Image

# CG render plate
plate = np.asarray(Image.open("beauty.exr").convert("RGBA"))   # use OpenEXR for proper EXR

# CG Z-pass (near = 0, far = large) — needs inversion and normalise
import numpy as np
z_raw = np.load("z_pass.npy")
depth = df.prep_depth(z_raw, df.DepthPrepParams(invert=True, dilation_px=5))

# Background plate for inpainting
bg = np.asarray(Image.open("background.png").convert("RGBA"))

# Generate stereo pair
L, R, occ = df.make_stereo_pair(plate, depth,
                df.StereoPairParams(max_parallax_fraction=0.04))

# Fill occlusion from background plate
L = df.inpaint_occlusion(L, occ, df.InpaintParams(
    method=df.InpaintMethod.CLEAN_PLATE, clean_plate=bg, blend_px=4))
R = df.inpaint_occlusion(R, occ, df.InpaintParams(
    method=df.InpaintMethod.CLEAN_PLATE, clean_plate=bg, blend_px=4))

Image.fromarray(L[:,:,:3]).save("left_view.png")
Image.fromarray(R[:,:,:3]).save("right_view.png")
```

### Example D: Adaptive SIRDS with text depth

```python
import depthforge as df

# Depth from text
depth = df.text_to_mask("DepthForge", width=800, height=400, font_size=120)
depth = df.prep_depth(depth, df.DepthPrepParams(bilateral_sigma_space=3, dilation_px=2))

# Adaptive dot tile
tile = df.generate_adaptive_dots(depth, df.AdaptiveDotParams(
    tile_width=256, tile_height=256,
    min_dot_radius=1, max_dot_radius=4,
    dot_color=(220, 220, 255), bg_color=(10, 10, 20),
    seed=0,
))

stereo = df.synthesize(depth, tile, df.StereoParams(depth_factor=0.4, seed=0))
df.save_stereogram(stereo, "depthforge_text_sirds.png")
```
