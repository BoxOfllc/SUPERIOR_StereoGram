# DepthForge

**A professional stereogram synthesis engine for VFX, motion graphics, and creative pipelines.**

[![Phase 1 Core](https://img.shields.io/badge/status-Phase%201%20Complete-brightgreen)](docs/CHANGELOG.md)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

DepthForge converts depth maps into stereograms across five surfaces — a CLI, an OFX plugin (After Effects, Premiere, DaVinci, Fusion), a Nuke gizmo, a ComfyUI custom node cluster, and a local web preview server — all powered by a single shared Python core.

```
┌─────────────────────────────────────────────────────────┐
│                    depthforge.core                       │
│  synthesizer · depth_prep · pattern_gen · anaglyph      │
│  stereo_pair · hidden_image · adaptive_dots · inpaint    │
└────┬──────────┬──────────┬──────────┬──────────┬────────┘
     │          │          │          │          │
    CLI       OFX        Nuke    ComfyUI      Web UI
```

---

## ⚠️ Photosensitivity Warning

Stereogram patterns — especially psychedelic/high-contrast modes — can trigger **photosensitive epilepsy**. Always use `--safe-mode` for public-facing content or when the audience is unknown. See [Safety](docs/USER_GUIDE.md#safety) for full guidance.

---

## Output Modes

| Mode | Description |
|---|---|
| **SIRDS** | Classic random-dot stereogram — the "magic eye" look |
| **Texture Pattern** | Any tileable image used as the pattern layer |
| **Hidden Image** | Shape or text encoded invisibly in dot noise |
| **Anaglyph** | Red/cyan (and variants) for glasses-based 3D viewing |
| **Stereo Pair** | Left/right views for side-by-side or top-bottom layouts |
| **Video Sequence** | Temporally stable stereogram sequences (Phase 5) |

---

## Quick Start

### Install (core only — no AI models)

```bash
pip install depthforge
```

### Install with AI depth estimation

```bash
pip install "depthforge[ai]"      # MiDaS + ZoeDepth
pip install "depthforge[full]"    # Everything including dev tools
```

### 60-Second Example

```python
import numpy as np
from depthforge import synthesize, StereoParams, prep_depth, DepthPrepParams
from depthforge import generate_pattern, PatternParams, PatternType, ColorMode
from PIL import Image

# 1. Load or create a depth map (white = near, black = far)
depth = np.array(Image.open("depth.png").convert("L"), dtype=np.float32) / 255.0

# 2. Condition the depth map
depth = prep_depth(depth, DepthPrepParams(bilateral_sigma_space=5, dilation_px=3))

# 3. Generate a pattern tile
pattern = generate_pattern(PatternParams(
    pattern_type=PatternType.PLASMA,
    tile_width=128, tile_height=128,
    color_mode=ColorMode.PSYCHEDELIC,
    seed=42,
))

# 4. Synthesize the stereogram
stereo = synthesize(depth, pattern, StereoParams(depth_factor=0.4))

# 5. Save
Image.fromarray(stereo).save("stereogram.png")
```

### CLI Quick Start

```bash
# SIRDS from a depth image
depthforge sirds --depth depth.png --output out.png

# Anaglyph with optimised mode
depthforge anaglyph --source photo.jpg --depth depth.png --mode optimised

# Hidden star shape
depthforge hidden --shape star --width 1920 --height 1080 --output hidden.png

# Psychedelic texture pattern
depthforge texture --depth depth.png --pattern plasma --color psychedelic --output stereo.png
```

---

## Documentation

| Document | Description |
|---|---|
| [User Guide](docs/USER_GUIDE.md) | Setup, installation, and step-by-step tutorials for every mode |
| [API Reference](docs/API_REFERENCE.md) | Full Python API — every class, function, and parameter |
| [Algorithm](docs/ALGORITHM.md) | How stereograms work and how DepthForge implements them |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Symptom-based FAQ and fixes |
| [Changelog](docs/CHANGELOG.md) | Version history |

---

## Feature Overview

### Inputs
- Grayscale/B&W image (luminance = depth)
- Color image + auto-depth (MiDaS/ZoeDepth) — Phase 3
- External depth map / EXR Z-pass
- Text prompt → AI depth → stereogram — Phase 4
- Video clip + depth sequence — Phase 5

### Pattern Library (Phase 3)
15+ built-in procedural generators: random noise, Perlin, plasma, Voronoi cells, Mandelbrot, dot matrix, hexagonal grid, checkerboard, stripes. Greyscale, monochrome, and psychedelic colour modes.

### Stereo Controls
- `depth_factor` — parallax strength (−1.0 to 1.0)
- `convergence` — where the stereo window sits in depth space
- `max_parallax_fraction` — comfort limit (default 1/30 of frame width)
- `safe_mode` — hard-clamp all parameters to comfortable limits
- Depth falloff curves: linear, gamma, S-curve, logarithmic, exponential
- Region mask overrides for local depth control

### Integrations
- **CLI** — batch processing, farm-ready (Phase 3)
- **OFX Plugin** — AE, Premiere, DaVinci Resolve, Fusion, HitFilm (Phase 5)
- **Nuke Gizmo** — Python gizmo with stereo view output (Phase 4)
- **ComfyUI Nodes** — 10-node cluster for AI-powered workflows (Phase 4)
- **Web Preview** — FastAPI local server (Phase 4)

---

## Project Status

| Phase | Scope | Status |
|---|---|---|
| Phase 1 | Core engine — all 8 synthesis modules | ✅ Complete |
| Phase 2 | Depth tools — presets, safety, QC overlays | 🔄 Next |
| Phase 3 | CLI + AI depth + full pattern library | ⏳ Planned |
| Phase 4 | ComfyUI + Nuke + Web preview | ⏳ Planned |
| Phase 5 | OFX plugin + GPU + video | ⏳ Planned |

---

## Requirements

**Minimum (core only):**
- Python 3.9+
- NumPy ≥ 1.22
- Pillow ≥ 9.0

**Recommended (better quality):**
- OpenCV (`cv2`) — bilateral filter, morphological ops, inpainting
- SciPy — Gaussian filter, advanced morphology

**Optional (AI depth):**
- PyTorch ≥ 2.0 — MiDaS, ZoeDepth
- timm — model architectures

DepthForge detects available packages at import time and automatically uses the best available implementation. Nothing breaks if optional packages are absent.

---

## Contributing

Pull requests welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on code style, testing, and the PR process.

```bash
git clone https://github.com/your-org/depthforge
cd depthforge
pip install -e ".[dev]"
python -m unittest discover tests/   # or pytest
```

---

## License

MIT — see [LICENSE](LICENSE).
