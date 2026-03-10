# Depth Map Guide — DepthForge

Everything a compositor needs to know about preparing depth maps for
DepthForge stereogram synthesis.

---

## The Core Convention

> **WHITE = near (close to camera)**
> **BLACK = far (background / infinity)**

This is the **DepthForge convention**. It matches the intuitive sense of
"brighter = closer to you" and is consistent with most CG Z-pass outputs
after normalisation.

Internally the synthesizer works in `[0.0, 1.0]` float range:
- `1.0` = nearest point in the scene
- `0.0` = farthest point (background)

---

## Why This Convention?

Standard CG Z-passes are often inverted (black = near, white = far) or stored
as linear world-space distance values. DepthForge normalises and optionally
inverts the depth in **DF_DepthPrep** so you can use any source format.

| Source format | White = ? | Action in DF_DepthPrep |
|---------------|-----------|------------------------|
| DepthForge default | Near | No inversion needed — `invert = false` |
| Inverted Z-pass | Far | Set `invert = true` |
| Linear world distance | Depends | Normalise externally first, then check orientation |

---

## Bit Depth and Format

| Format | Notes |
|--------|-------|
| **8-bit PNG greyscale** | Fastest; 256 depth levels — sufficient for most stereograms |
| **16-bit PNG greyscale** | Smoother gradients; recommended for print output |
| **32-bit EXR (half or full)** | Full linear range; best for CG renders with extreme near/far depth |
| **TIFF** | Supported via DepthForge I/O layer |
| **Colour PNG/EXR** | DepthForge averages the RGB channels to produce a single depth value — use greyscale for accuracy |

> The `_depth_to_numpy` function in `nodes.py` handles all of these by
> converting to float32 and averaging channels if needed.

---

## Preparing Depth Maps from CG Renders

### Blender (Cycles/EEVEE)

1. In the Compositor, add a **Render Layers** node.
2. Connect the **Depth** output to a **Normalize** node (maps to 0–1).
3. Connect **Normalize** → **File Output** node, format = EXR or PNG 16-bit.
4. **Important:** Blender depth is white=far by default. Set `invert = true`
   in DF_DepthPrep, or plug through a **Invert** node in Blender first.

### Nuke

```python
# Z-depth pass normalisation in Nuke
grade = nuke.createNode('Grade')
grade['whitepoint'].setValue(far_distance)   # e.g. 1000
grade['blackpoint'].setValue(near_distance)  # e.g. 0.1
```

The Nuke DepthForge gizmo (`depthforge/nuke/gizmo.py`) handles this
automatically — use it for Nuke-based pipelines.

### DaVinci Resolve / Fusion

Use a **Background** node set to the far clip distance, divide the Z-pass
by it, then clamp 0–1. Export as 16-bit TIFF.

### AI Depth Estimators (MiDaS, ZoeDepth)

DepthForge includes a `depth_models/` module with MiDaS and ZoeDepth
support:

```bash
depthforge depth estimate --model midas --input photo.jpg --output depth.png
```

MiDaS outputs are already in the white=near convention and normalised to
0–1, so no inversion is needed.

---

## What Makes a Good Depth Map for Stereograms

### Do use
- **Smooth gradients** between depth layers — avoids sharp parallax
  discontinuities that cause eye strain
- **Clear object separation** — foreground objects should be distinctly
  lighter than the background
- **Full dynamic range** — use the full 0–1 range; compressed depth
  (e.g. 0.3–0.7) produces flat-looking stereograms

### Avoid
- **Binary masks** (0 or 1 only) — creates harsh visible edges; use
  `edge_soften_px` in DF_HiddenImage, or increase `dilation_px` in DF_DepthPrep
- **Noisy depth** — high-frequency noise in the depth map produces
  shimmering stereogram dots; use DF_DepthPrep's bilateral smooth
- **Reversed values** — always confirm orientation before processing

---

## Verifying Depth Map Orientation

1. Load your depth map into ComfyUI with `LoadImage`
2. Wire it to **DF_DepthPrep** and connect `depth_preview` to `PreviewImage`
3. Look at the preview — the nearest object in your scene should appear
   **white** and the distant background should appear **dark grey or black**
4. If it's inverted, set `invert = true` in DF_DepthPrep

Alternatively, use **DF_QCOverlay** with `overlay_type = parallax_heatmap`:
warm colours (red/yellow) = high parallax = near objects. Verify that the
warm region corresponds to the intended foreground in your scene.

---

## Using the Included Test Depth Maps

The workflows reference `depth_test.png` — replace this with any of the
DepthForge test assets included in `depthforge/tests/`:

| File (relative to tests/) | Content | Notes |
|--------------------------|---------|-------|
| Test images in `test_phase1.py` | Synthetic gradient (near-to-far) | Generated in code; export via CLI if needed |

To generate a quick test depth map from the command line:

```bash
# Gradient depth map 512×512
python -c "
import numpy as np
from PIL import Image
arr = np.tile(np.linspace(1, 0, 512), (512, 1)).astype(np.float32)
Image.fromarray((arr * 255).astype(np.uint8)).save('depth_test.png')
"
```

This creates a simple horizontal gradient: white (near) on the left fading
to black (far) on the right — a good sanity-check for any stereogram workflow.

---

## Depth Map Specs at a Glance

| Property | Requirement |
|----------|-------------|
| Channels | Greyscale (1 channel) preferred; RGB averaged automatically |
| Bit depth | 8-bit minimum; 16-bit recommended; 32-bit EXR for CG |
| Value range | 0.0 (far) to 1.0 (near) after normalisation |
| White = ? | **NEAR** (DepthForge convention) |
| Resolution | Any; same as source image for anaglyph/stereo pair |
| File format | PNG, EXR, TIFF, JPEG (JPEG not recommended — lossy artefacts degrade depth) |

---

*DepthForge v0.5 — Superior Studios / BoxOf LLC*
