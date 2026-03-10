# Changelog

All notable changes to DepthForge are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
DepthForge uses [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Planned — Phase 2
- Temporal depth smoothing for video sequences (IIR filter + motion-aware blend)
- Bilateral depth smoothing with true edge-preserving video mode
- Depth map presets (shallow, medium, deep, cinema, print, broadcast)
- Stereo window violation detection improvements
- Vergence comfort analyzer with per-frame reporting

### Planned — Phase 3
- CLI entry point (`depthforge` command)
- Batch processing (`depthforge batch --input dir/ --output dir/`)
- MiDaS monocular depth estimation integration
- ZoeDepth metric depth estimation integration
- EXR I/O via OpenEXR
- OCIO colour management
- 30+ pattern library with manifest.json
- Bundled LUT pack (Kodak film, cinematic, neon, desaturated)
- Farm-ready output mode

### Planned — Phase 4
- ComfyUI node cluster (10 nodes: DF_DepthFromText, DF_DepthFromImage,
  DF_DepthPrep, DF_PatternGen, DF_Stereogram, DF_AnaglyphOut,
  DF_VideoSequence, DF_QCOverlay, DF_Inpaint, DF_PatternBrowser)
- Nuke Python gizmo (Source + Depth + Pattern → Stereogram/Anaglyph/LR)
- FastAPI web preview server

### Planned — Phase 5
- OFX C++ plugin (pybind11 bridge to Python core)
- GPU-accelerated synthesis (CUDA/Metal)
- Optical flow depth propagation (RAFT/DIS)
- Interlaced/lenticular print layout

---

## [0.1.0] — 2026-03-07

### Added

**Core engine — 8 modules, 77 tests, all passing.**

#### `core/synthesizer.py`
- `StereoParams` dataclass — full stereo synthesis parameter control
- `synthesize(depth, pattern, params)` — constraint-link SIRDS algorithm
- `load_depth_image(path)` — raw depth load utility
- `save_stereogram(arr, path)` — PNG save utility
- Oversample mode (2× Lanczos downsample for sub-pixel accuracy)
- `safe_mode` hard-clamps parallax to ergonomic limits

#### `core/depth_prep.py`
- `DepthPrepParams` dataclass — 7-stage pipeline configuration
- `FalloffCurve` enum — LINEAR, GAMMA, S_CURVE, LOGARITHMIC, EXPONENTIAL
- `RegionMask` dataclass — local depth override with blend mask
- `prep_depth(raw, params)` — full conditioning pipeline
- `normalise_depth(raw)` — single-step normalise
- `depth_from_image(path, params)` — load + prep from file
- `compute_vergence_map(depth)` — per-pixel vergence angle (degrees)
- `detect_window_violations(depth)` — frame-edge stereo violation detection
- Tier-0/1/2 fallback chain for all operations

#### `core/pattern_gen.py`
- `PatternParams` dataclass — full pattern configuration
- `PatternType` enum — RANDOM_NOISE, PERLIN, PLASMA, VORONOI,
  GEOMETRIC_GRID, MANDELBROT, DOT_MATRIX, CUSTOM_TILE
- `ColorMode` enum — MONOCHROME, PSYCHEDELIC, GREYSCALE, CUSTOM
- `GridStyle` enum — DOTS, HEXES, CHECKS, STRIPES
- `generate_pattern(params)` — all 7 procedural generators
- `load_tile(path)` — custom image tile loader
- `tile_to_frame(tile, H, W)` — tile pattern to fill frame
- `safe_mode` — 50% contrast limit

#### `core/anaglyph.py`
- `AnaglyphMode` enum — TRUE, GREY, COLOUR, HALF_COLOUR, OPTIMISED
- `AnaglyphParams` dataclass
- `make_anaglyph(left, right, params)` — composite L/R pair
- `make_anaglyph_from_depth(source, depth, params)` — direct from depth
- Dubois least-squares optimised matrices for `OPTIMISED` mode

#### `core/stereo_pair.py`
- `StereoPairParams` dataclass
- `StereoLayout` enum — SEPARATE, SIDE_BY_SIDE, TOP_BOTTOM, ANAGLYPH
- `make_stereo_pair(source, depth, params)` — returns (L, R, occlusion_mask)
- `compose_side_by_side(left, right, gap_px)` — layout helper
- Three background fill modes: edge, mirror, black
- Gaussian feathering on occlusion mask

#### `core/hidden_image.py`
- `HiddenImageParams` dataclass
- `encode_hidden_image(pattern, mask, params)` — full hidden image pipeline
- `mask_to_depth(mask, params)` — mask → depth map conversion
- `load_hidden_mask(path, target_size)` — load custom mask from file
- `text_to_mask(text, W, H, ...)` — Pillow text rendering → mask
- `shape_to_mask(shape, W, H)` — 6 built-in shapes: circle, square,
  triangle, star, diamond, arrow
- `edge_soften_px` Gaussian blur on mask boundary

#### `core/adaptive_dots.py`
- `AdaptiveDotParams` dataclass
- `generate_adaptive_dots(depth, params)` — complexity-driven SIRDS tile
- `complexity_from_depth(depth)` — Sobel + local variance complexity map
- 5-level density discretisation with jittered grid placement
- Tier-0/1/2 fallback for edge detection and map resize

#### `core/inpainting.py`
- `InpaintMethod` enum — PATCH_BASED, CLEAN_PLATE, EDGE_EXTEND,
  AI_CALLBACK, AUTO
- `InpaintParams` dataclass
- `inpaint_occlusion(image, mask, params)` — occlusion fill
- `register_ai_inpaint_callback(fn)` — global AI model hook
- OpenCV Telea fast-marching inpaint when available
- Pure NumPy patch-match fallback

#### Testing
- `tests/test_phase1.py` — 77 unit tests across all 8 modules
  (runs with both `pytest` and `unittest`)
- `tests/visual_gallery.py` — renders 7 labelled gallery PNG files

#### Project structure
- `pyproject.toml` — full build config with optional dependency groups
- `requirements.txt` / `requirements-dev.txt`
- `README.md`
- `LICENSE` (MIT)
- `.gitignore`
- `docs/USER_GUIDE.md`
- `docs/API_REFERENCE.md`
- `docs/ALGORITHM.md`
- `docs/TROUBLESHOOTING.md`
- `docs/SAFETY.md`
- `CHANGELOG.md`
