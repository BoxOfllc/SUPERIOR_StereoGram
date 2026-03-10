# DepthForge 🔮

**Professional stereogram engine for VFX, motion graphics, and creative pipelines.**

Generate SIRDS, texture pattern, anaglyph, and hidden-image stereograms from depth maps, images, or text prompts — as a Python library, CLI tool, Nuke gizmo, OFX plugin, or ComfyUI node cluster.

```
depth map + pattern → magic-eye stereogram
```

---

## ⚡ Quick Start

```bash
pip install depthforge
```

```python
import depthforge as df
import numpy as np

# Load or create a depth map
depth = df.load_depth_image("my_depth.png")        # or any float32 array

# Condition the depth
depth = df.prep_depth(depth, df.DepthPrepParams(dilation_px=3))

# Generate a pattern
pattern = df.generate_pattern(df.PatternParams(
    pattern_type = df.PatternType.PLASMA,
    color_mode   = df.ColorMode.PSYCHEDELIC,
    seed         = 42,
))

# Synthesize
stereo = df.synthesize(depth, pattern, df.StereoParams(depth_factor=0.4))

# Save
df.save_stereogram(stereo, "output.png")
```

**60-second result:** a full-colour psychedelic stereogram from any depth source.

---

## 🎯 What It Does

| Output type | Description |
|---|---|
| **SIRDS** | Classic random-dot magic-eye (mono or colour) |
| **Texture pattern** | Any image tile used as the stereogram pattern |
| **Anaglyph** | Red/cyan glasses — 5 modes including Dubois optimised |
| **Stereo pair** | Left/right views + occlusion mask for compositing |
| **Hidden image** | Shape or text hidden inside the dot field |
| **Video sequence** | Temporally-stable per-frame processing |

---

## 🖥️ Surfaces

| Surface | Status | Use case |
|---|---|---|
| **Python library** | ✅ Phase 1 | Import and use directly in scripts |
| **CLI / farm** | 🔜 Phase 3 | Batch processing, render farm |
| **ComfyUI nodes** | 🔜 Phase 4 | AI-driven depth + stereogram workflows |
| **Nuke gizmo** | 🔜 Phase 4 | VFX plate + CG depth integration |
| **OFX plugin** | 🔜 Phase 5 | After Effects, DaVinci, Premiere, Fusion |
| **Web preview** | 🔜 Phase 4 | Local browser UI for quick iteration |

---

## 📦 Installation

### Minimum install (core only)
```bash
pip install depthforge
# Requires: numpy, Pillow
```

### Recommended install (better quality)
```bash
pip install "depthforge[cv2,scipy]"
# Adds: OpenCV (better bilateral filter), SciPy (morphological ops)
```

### AI depth estimation
```bash
pip install "depthforge[ai]"
# Adds: PyTorch, timm (MiDaS / ZoeDepth support)
```

### Everything
```bash
pip install "depthforge[full]"
```

### From source
```bash
git clone https://github.com/depthforge/depthforge.git
cd depthforge
pip install -e ".[dev]"
python -m pytest tests/
```

---

## 🔑 Core Concepts

### Depth map conventions
DepthForge follows the convention **white = near, black = far** (0–1 float32).
Toggle `invert_depth=True` in `StereoParams` if your source uses the opposite.

### Dependency tiers
The engine degrades gracefully when optional packages are missing:

| Tier | Packages | Quality |
|---|---|---|
| 0 | NumPy + Pillow | Works everywhere |
| 1 | + OpenCV | Better bilateral filter, faster dilation |
| 2 | + SciPy | Superior morphological ops |
| 3 | + PyTorch | AI depth estimation |

Check availability:
```python
import depthforge as df
print(df.HAS_CV2, df.HAS_SCIPY, df.HAS_TORCH)
```

---

## 🔧 Core API

### `synthesize(depth, pattern, params)`
The main entry point. Converts a depth map + pattern tile into a stereogram.

```python
stereo = df.synthesize(
    depth   = depth,          # float32 (H, W)
    pattern = pattern,        # RGBA uint8 (tH, tW, 4)
    params  = df.StereoParams(
        depth_factor  = 0.4,  # parallax strength
        convergence   = 0.5,  # screen plane position
        safe_mode     = True, # limit eye strain
    )
)
```

### `prep_depth(raw, params)`
Full depth conditioning pipeline.

```python
depth = df.prep_depth(raw_depth, df.DepthPrepParams(
    bilateral_sigma_space = 5.0,
    dilation_px           = 3,
    falloff_curve         = df.FalloffCurve.S_CURVE,
    near_plane            = 0.05,
    far_plane             = 0.95,
))
```

### `generate_pattern(params)`
Procedural pattern generation.

```python
tile = df.generate_pattern(df.PatternParams(
    pattern_type = df.PatternType.VORONOI,
    tile_width   = 128,
    tile_height  = 128,
    color_mode   = df.ColorMode.PSYCHEDELIC,
    seed         = 7,
))
```

### `make_anaglyph_from_depth(source, depth, params)`
Anaglyph directly from a source image and depth map.

```python
anaglyph = df.make_anaglyph_from_depth(
    source = source_rgba,
    depth  = depth,
    params = df.AnaglyphParams(mode=df.AnaglyphMode.OPTIMISED),
)
```

### `encode_hidden_image(pattern, mask, params)`
Encode a shape or text as a hidden stereogram.

```python
mask   = df.shape_to_mask("star", width=512, height=512)
stereo = df.encode_hidden_image(pattern, mask, df.HiddenImageParams(
    foreground_depth = 0.85,
    edge_soften_px   = 6,
))
```

---

## 🗂️ Project Structure

```
depthforge/
├── core/               # Pure algorithm — no UI dependencies
│   ├── synthesizer.py  # Core stereogram engine
│   ├── depth_prep.py   # Depth conditioning pipeline
│   ├── pattern_gen.py  # Procedural pattern generators
│   ├── anaglyph.py     # Red/cyan anaglyph compositing
│   ├── stereo_pair.py  # L/R view synthesis + occlusion mask
│   ├── hidden_image.py # Hidden shape/text encoding
│   ├── adaptive_dots.py# Complexity-driven SIRDS dots
│   └── inpainting.py   # Occlusion fill (patch + AI hook)
├── depth_models/       # AI depth estimators (Phase 3)
├── video/              # Temporal processing (Phase 5)
├── io/                 # EXR, OCIO, format I/O (Phase 3)
├── cli/                # Command-line interface (Phase 3)
├── comfyui/            # ComfyUI node cluster (Phase 4)
├── nuke/               # Nuke Python gizmo (Phase 4)
├── ofx/                # OpenFX C++ plugin (Phase 5)
├── web/                # FastAPI preview server (Phase 4)
├── patterns/           # Built-in pattern library
├── presets/            # JSON depth/stereo presets
└── tests/
    ├── test_phase1.py  # 77 unit tests (all passing)
    └── visual_gallery.py  # Visual output gallery
```

---

## 🧪 Running Tests

```bash
# Unit tests (stdlib — no pytest needed)
python -m unittest discover tests/

# With pytest (if installed)
pytest tests/ -v

# Visual gallery — renders PNG outputs to tests/gallery/
python tests/visual_gallery.py
```

---

## ⚠️ Safety Notice

DepthForge can generate high-contrast patterns that **may trigger photosensitive epilepsy**. Always enable `safe_mode=True` when generating content for public display:

```python
StereoParams(safe_mode=True)     # limits parallax
PatternParams(safe_mode=True)    # limits contrast
```

See [SAFETY.md](docs/SAFETY.md) for full guidance.

---

## 📍 Roadmap

- [x] **Phase 1** — Core engine (8 modules, 77 tests)
- [ ] **Phase 2** — Depth tools & quality (temporal smoothing, video)
- [ ] **Phase 3** — CLI + AI depth (MiDaS, ZoeDepth, farm batch)
- [ ] **Phase 4** — ComfyUI + Nuke + Web Preview
- [ ] **Phase 5** — OFX plugin + GPU acceleration

---

## 📄 License

MIT — see [LICENSE](LICENSE).
Photosensitivity notice included.
