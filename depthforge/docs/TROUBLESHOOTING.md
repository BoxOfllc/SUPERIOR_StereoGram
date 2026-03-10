# DepthForge — Troubleshooting & FAQ

---

## Quick Diagnostic Checklist

Before diving into specific issues, run this:

```python
import depthforge as df
import numpy as np

print("DepthForge:", df.__version__)
print("NumPy:     ", np.__version__)

try:
    import PIL; print("Pillow:    ", PIL.__version__)
except: print("Pillow:     MISSING — pip install Pillow")

try:
    import cv2; print("OpenCV:    ", cv2.__version__)
except: print("OpenCV:     not installed (optional)")

try:
    import scipy; print("SciPy:     ", scipy.__version__)
except: print("SciPy:      not installed (optional)")

try:
    import torch; print("PyTorch:   ", torch.__version__)
except: print("PyTorch:    not installed (optional)")
```

---

## 1. Installation Issues

### `ModuleNotFoundError: No module named 'depthforge'`

**Cause:** DepthForge is not installed in the active Python environment.

**Fix:**
```bash
pip install depthforge

# If using a virtual environment, make sure it's activated:
source my-env/bin/activate        # macOS/Linux
my-env\Scripts\activate           # Windows
```

If running from source:
```bash
cd /path/to/depthforge
pip install -e .
# Then run with:
PYTHONPATH=/path/to/depthforge python your_script.py
```

---

### `pip install depthforge[full]` fails on Windows

**Cause:** OpenCV and PyTorch have different wheel names on Windows.

**Fix:**
```bash
pip install depthforge
pip install opencv-python
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

### `ImportError: DLL load failed` (Windows + OpenCV)

**Cause:** Missing Visual C++ redistributable.

**Fix:** Download and install the latest Microsoft Visual C++ Redistributable from the Microsoft website, then retry.

---

### Python 3.8 or lower

DepthForge requires Python 3.9+.

```bash
python --version     # check current version
```

Install Python 3.11 from [python.org](https://python.org) and create a fresh virtual environment.

---

## 2. Stereogram Won't Fuse (Eyes Can't "Lock In")

This is the most common issue. Several causes:

### Cause A: Parallax too large

The depth factor is too high — eyes cannot converge at such a wide separation.

**Fix:**
```python
# Reduce depth_factor
df.StereoParams(depth_factor=0.2)    # try 0.15–0.25 first

# Or enable safe mode (hard caps at comfortable limits)
df.StereoParams(safe_mode=True)
```

**Comfortable range:** `depth_factor` between 0.15 and 0.45 for most viewers.

---

### Cause B: Viewing at wrong distance

The image needs to be viewed at approximately arm's length (~60 cm).
On a monitor, try zooming out so the image is 10–20 cm wide on screen.

---

### Cause C: Image resolution too low

Very small stereograms (< 200 px wide) don't have enough pixels for
the parallax to register.

**Fix:** Render at minimum 400×400 px; 512+ recommended.

---

### Cause D: Tile size too large relative to image

If the pattern tile is close to or larger than the image, there's no
room for the parallax shifts.

**Fix:**
```python
# Tile should be at most 1/4 to 1/8 of the image width
# For a 512px wide image, use tile_width=64–128
PatternParams(tile_width=64, ...)
```

---

### Cause E: Eye strain or fatigue

Try again after resting your eyes. Stereogram fusion is a learned skill
that gets easier with practice.

**Viewing technique:**
1. Hold the image at arm's length
2. Look *through* the image as if staring at something behind it
3. Relax your eyes — don't focus on the surface
4. Wait 10–30 seconds; the depth should snap into place

---

## 3. Ghosting / Double Edges in the Stereogram

You can see faint double outlines or halos around depth transitions.

### Cause: Depth map not smoothed at boundaries

Sharp depth transitions create hard parallax jumps that the eye perceives as ghosting.

**Fix:**
```python
df.DepthPrepParams(
    bilateral_sigma_space = 7,    # increase smoothing
    bilateral_sigma_color = 0.15,
    dilation_px           = 5,    # expand near regions more
    smooth_passes         = 2,    # run smooth twice
)
```

---

### Cause: Depth map has incorrect values at object boundaries

CG depth passes often have slight inaccuracies at silhouette edges.

**Fix:**
```python
df.DepthPrepParams(
    dilation_px   = 6,
    falloff_curve = df.FalloffCurve.S_CURVE,  # smooths transitions
)
```

---

## 4. Anaglyph Has Bad Colour Fringing / Ghosting

### Cause A: Using non-Optimised mode

**Fix:** Always use `AnaglyphMode.OPTIMISED` (Dubois) for final output:
```python
df.AnaglyphParams(mode=df.AnaglyphMode.OPTIMISED)
```

---

### Cause B: Parallax too high for anaglyph

Anaglyph ghosting increases with parallax. Unlike SIRDS, anaglyph
is sensitive to very large depth values.

**Fix:**
```python
df.AnaglyphParams(parallax_px=8)   # try 6–15 px range
```

---

### Cause C: Source image is highly saturated

Highly saturated reds or cyans in the source create retinal rivalry.

**Fix:** Slightly desaturate the source image before passing to `make_anaglyph_from_depth`.

---

## 5. Video Stereogram Flickers / Shimmers

### Cause: Per-frame depth estimation is inconsistent

Monocular depth models estimate depth independently each frame,
causing values to jump frame-to-frame.

**Fix (Phase 2):** Use temporal depth smoothing (coming in Phase 2).

**Workaround now:**
```python
# Use a fixed seed and the same pattern tile for all frames
params  = df.StereoParams(depth_factor=0.35, seed=42)  # FIXED seed
pattern = df.generate_pattern(df.PatternParams(         # FIXED pattern
    df.PatternType.RANDOM_NOISE, 64, 64, seed=0))

for frame_depth in depth_sequence:
    stereo = df.synthesize(frame_depth, pattern, params)
```

---

### Cause: Different pattern tile per frame

If you regenerate the pattern for each frame without a fixed seed, the dot
positions change every frame — guaranteed flicker.

**Fix:** Generate the pattern *once* and reuse it.

---

## 6. Pattern Looks Stretched or Squished

### Cause A: Non-square tile with `tile_to_frame()`

If the tile is 64×128 but the frame is 512×512, the tiling repeats at
different rates horizontally vs vertically.

**Fix:** Use square tiles, or tiles with the same aspect ratio as your output.

---

### Cause B: Tile loaded from an image with incorrect resize

**Fix:**
```python
tile = df.load_tile("pattern.png", tile_width=128, tile_height=128)
# Both width and height specified forces a square tile
```

---

## 7. Hidden Image Not Visible

### Cause A: Depth contrast too low

If `foreground_depth` and `background_depth` are too close, the parallax
difference is imperceptible.

**Fix:**
```python
df.HiddenImageParams(
    foreground_depth = 0.85,   # push this higher
    background_depth = 0.0,    # and this lower
)
```

---

### Cause B: Viewing technique

Hidden images require the same defocus technique as SIRDS. The shape
pops out once the eyes relax past the surface plane.

**Technique:** Try crossing your eyes slightly (rather than defocusing).
Some people find cross-eye easier than wall-eye for hidden images.

---

### Cause C: Shape is too fine or complex

Very thin lines and small text (< 20 px) get lost in the dot noise.

**Fix:**
```python
# Use simple, bold shapes with generous padding
mask = df.shape_to_mask("circle", padding=60, width=512, height=512)

# For text: use large, bold font at > 80pt
mask = df.text_to_mask("HI", width=512, height=256, font_size=120)
```

---

### Cause D: Edge soften not applied

Without softening, the shape boundary is a hard step — harder to fuse.

**Fix:**
```python
df.HiddenImageParams(edge_soften_px=8)  # 4–12 works well
```

---

## 8. OFX Plugin Not Appearing in Host

> OFX is Phase 5 — this section will be updated when the plugin ships.

### After Effects / Premiere
- Check `.ofx.bundle` is in `C:\Program Files\Common Files\OFX\Plugins\` (Windows)
  or `/Library/OFX/Plugins/` (macOS)
- Restart After Effects after installation
- Look under `Effect → DepthForge` in the effects panel

### DaVinci Resolve / Fusion
- OFX plugins live in the same OS paths as above
- Resolve must be restarted after install
- Open Fusion page → Effects Library → OFX → DepthForge

### Check host OFX path
```bash
# macOS
ls /Library/OFX/Plugins/

# Windows (PowerShell)
ls "C:\Program Files\Common Files\OFX\Plugins\"
```

---

## 9. Nuke Gizmo Errors

> Nuke gizmo is Phase 4. This section will be updated when it ships.

### `ImportError` on farm render nodes

Farm render nodes need DepthForge in their Python path.

**Fix:** Add to your `init.py`:
```python
import sys
sys.path.insert(0, "/path/to/depthforge/install")
```

Or set the environment variable:
```bash
export PYTHONPATH=/path/to/depthforge:$PYTHONPATH
```

### Gizmo not found in Nuke menu

**Fix:** Ensure `depthforge/nuke/` is on `NUKE_PATH`.
Add to `~/.nuke/init.py`:
```python
import nuke
nuke.pluginAddPath("/path/to/depthforge/nuke")
```

---

## 10. ComfyUI Nodes Not Found

> ComfyUI nodes are Phase 4. This section will be updated when they ship.

### `DepthForge nodes not appearing in ComfyUI`

1. Make sure the folder is in `ComfyUI/custom_nodes/`:
   ```bash
   ls ComfyUI/custom_nodes/ | grep depthforge
   ```
2. Make sure `depthforge` is pip-installed in ComfyUI's Python environment:
   ```bash
   /path/to/ComfyUI/venv/bin/pip install depthforge
   ```
3. Restart ComfyUI (full restart, not reload)
4. Check ComfyUI console for import errors on startup

---

## 11. Depth Map Looks Wrong (Inverted)

### Cause: Z-pass convention mismatch

CG depth passes often use **small values = near camera**, large values = far.
DepthForge expects **large values (1.0) = near**.

**Fix:**
```python
df.DepthPrepParams(invert=True)
# or
df.StereoParams(invert_depth=True)
```

---

### Cause: EXR Z-pass with large float range

Raw Z-passes can have values like 0.1 to 10,000. The normalisation
step should handle this, but very large outliers can compress the
useful range.

**Fix:**
```python
import numpy as np
z = np.load("z_pass.npy")

# Clip outliers before passing to prep_depth
z = np.clip(z, 0, np.percentile(z, 99))   # clip top 1%
depth = df.prep_depth(z, df.DepthPrepParams(invert=True))
```

---

## 12. Performance Issues (Slow on Large Images)

### Python loop bottleneck

The synthesizer inner loop is currently pure Python. On 4K frames, this
can take 30–90 seconds per frame.

**Workarounds:**
- Use `oversample=1` (not 2+) unless you need sub-pixel quality
- Tile the synthesis (split into strips)
- Phase 5 will add GPU acceleration

---

### OpenCV missing (slower fallback)

Without OpenCV, depth prep uses pure NumPy — correct but slower.

**Fix:**
```bash
pip install opencv-python
```

---

### Large pattern tiles

Pattern tiles > 512×512 significantly increase memory usage and I/O.

**Fix:** Keep tiles at 128–256 px for most uses. The synthesizer tiles
them to fill the output regardless of tile size.

---

## 13. FAQ

### Q: What's the difference between SIRDS and texture stereograms?

**SIRDS** uses random dots — the 3D information is entirely carried by the
repeating pattern of dots. Classic magic-eye style.

**Texture stereograms** use a repeating image tile (photo, logo, noise, etc.)
as the pattern. The tile is recognisable on its own, but the 3D depth is
encoded in its horizontal repetition.

Both use the same core algorithm in DepthForge — just different input patterns.

---

### Q: Why does my stereogram look different every time I run?

Because `seed=None` in `StereoParams` (default). Set a fixed seed for
reproducible outputs:
```python
df.StereoParams(seed=42)
df.PatternParams(..., seed=42)
```

---

### Q: Can I use DepthForge commercially?

Yes — MIT licence. Attribution appreciated but not required.

---

### Q: Is there a GPU-accelerated version?

Phase 5 will add CUDA/Metal acceleration for the synthesizer inner loop.
The depth estimation models (Phase 3) can already use GPU via PyTorch.

---

### Q: What's the maximum resolution?

No hard limit. Tested up to 8K (7680×4320). Memory is the constraint:
- HD (1920×1080): ~50 MB working memory
- 4K (3840×2160): ~200 MB working memory
- 8K (7680×4320): ~800 MB working memory

---

### Q: Can I use DepthForge in a Docker container?

Yes. Minimum Dockerfile:
```dockerfile
FROM python:3.11-slim
RUN pip install depthforge
```

For OpenCV in Docker:
```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0
RUN pip install "depthforge[cv2,scipy]"
```

---

### Q: How do I process a whole folder of depth maps?

```python
import depthforge as df
import os
from pathlib import Path

input_dir  = Path("depth_frames/")
output_dir = Path("stereo_frames/")
output_dir.mkdir(exist_ok=True)

pattern = df.generate_pattern(df.PatternParams(seed=0))
params  = df.StereoParams(seed=0)

for depth_file in sorted(input_dir.glob("*.png")):
    depth  = df.depth_from_image(str(depth_file))
    stereo = df.synthesize(depth, pattern, params)
    df.save_stereogram(stereo, str(output_dir / depth_file.name))
    print(f"  {depth_file.name}")
```

---

### Q: Can I use a 16-bit depth map?

Yes. `depth_from_image()` handles 16-bit TIFF and PNG automatically.
```python
depth = df.depth_from_image("depth_16bit.tif")
```

---

### Q: Does DepthForge support EXR?

EXR I/O is a Phase 3 feature. For now, convert EXR to 16-bit TIFF
before loading:
```bash
# Using ImageMagick:
convert depth.exr -depth 16 depth.tif
```

---

### Q: Why is my anaglyph uncomfortable to view?

1. Parallax is too high — reduce `parallax_px` to 8–12
2. Source image has conflicting red/cyan colours — try `GREY_ANAGLYPH` mode
3. Your monitor's colour profile is off — calibrate monitor or use `GREY_ANAGLYPH`

---

### Q: Can I generate a stereogram from a video without a depth map?

Phase 3 will add AI monocular depth estimation (MiDaS/ZoeDepth) that
generates depth from any colour video frame. For now, you need to provide
a depth map or depth sequence.

---

### Q: Will there be a GUI?

Phase 4 includes a local web preview UI (FastAPI + browser). Full desktop
GUI is not currently planned — the Nuke and ComfyUI integrations serve
that purpose for professional users.

---

## Still Stuck?

1. Run the visual gallery to confirm the engine works: `python tests/visual_gallery.py`
2. Check the [API Reference](API_REFERENCE.md) for full parameter documentation
3. Open an issue on [GitHub](https://github.com/depthforge/depthforge/issues)
   with the output of the diagnostic script above

---

## Reporting Bugs

Please include:
1. Output of the diagnostic script (top of this document)
2. Minimal code to reproduce the issue
3. The error message or unexpected output
4. OS, Python version, and which optional packages are installed
