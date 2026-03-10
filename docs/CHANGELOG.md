# Changelog

All notable changes to DepthForge are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

_Changes staged for the next release._

---

## [0.1.0] — 2026-03-07

### Phase 1 — Core Engine

Initial release of the DepthForge core synthesis engine. All 8 core modules are implemented and passing 77/77 tests.

### Added

**`depthforge.core.synthesizer`**
- `synthesize(depth, pattern, params) → np.ndarray (H,W,4)` — main stereogram synthesis entry point
- `StereoParams` dataclass with full stereo control parameters
- Constraint-link SIRDS algorithm (per-scanline `same[]` array approach)
- Oversample mode: renders at N× then Lanczos-downsamples for sub-pixel quality
- `safe_mode` flag hard-clamps parallax to comfortable limits

**`depthforge.core.depth_prep`**
- `prep_depth(raw, params) → float32 (H,W)` — 7-stage depth conditioning pipeline
- `DepthPrepParams` dataclass
- `FalloffCurve`: LINEAR, GAMMA, S_CURVE, LOGARITHMIC, EXPONENTIAL
- `RegionMask(mask, multiplier)` for per-region depth overrides
- `compute_vergence_map()` — viewer comfort analysis
- `detect_window_violations()` — stereo window violation detection
- Dependency fallback: OpenCV bilateral → SciPy gaussian → NumPy box blur

**`depthforge.core.pattern_gen`**
- `generate_pattern(params) → RGBA uint8 tile`
- 7 generators: RANDOM_NOISE, PERLIN, PLASMA, VORONOI, GEOMETRIC_GRID, MANDELBROT, DOT_MATRIX
- `ColorMode`: GREYSCALE, MONOCHROME, PSYCHEDELIC, CUSTOM
- `load_tile(path)` — load custom tile image
- `tile_to_frame(tile, H, W)` — expand tile to frame dimensions

**`depthforge.core.anaglyph`**
- `make_anaglyph(left, right, params) → RGBA uint8`
- `make_anaglyph_from_depth(source, depth, params)` — direct from source + depth
- 5 modes: TRUE_ANAGLYPH, GREY_ANAGLYPH, COLOUR_ANAGLYPH, HALF_COLOUR, OPTIMISED (Dubois matrices)
- Gamma-correct pipeline; `swap_eyes` support

**`depthforge.core.stereo_pair`**
- `make_stereo_pair(source, depth, params) → (left, right, occlusion_mask)`
- Forward warp per row with 3 background fill modes: edge, mirror, black
- `compose_side_by_side(left, right, gap_px)`
- `StereoLayout`: SEPARATE, SIDE_BY_SIDE, TOP_BOTTOM, ANAGLYPH
- Feathered occlusion mask output

**`depthforge.core.hidden_image`**
- `encode_hidden_image(pattern, mask, params) → RGBA uint8`
- `mask_to_depth(mask, params)` — convert binary mask to depth map
- `load_hidden_mask(path)` — load black/white mask image
- `text_to_mask(text, width, height, ...)` — auto-fits font size
- `shape_to_mask(shape, width, height)` — circle, square, triangle, star, diamond, arrow
- `HiddenImageParams` dataclass

**`depthforge.core.adaptive_dots`**
- `generate_adaptive_dots(depth, params) → RGBA uint8 tile`
- Complexity map from Sobel edges + local variance
- Jittered grid placement; N discrete density levels
- `AdaptiveDotParams` dataclass

**`depthforge.core.inpainting`**
- `inpaint_occlusion(image, mask, params) → RGBA uint8`
- 4 strategies: PATCH_BASED, CLEAN_PLATE, EDGE_EXTEND, AI_CALLBACK
- AUTO mode picks best available
- `register_ai_inpaint_callback(fn)` — hook for ComfyUI SD inpaint (Phase 4)
- Pure NumPy patch-match fallback when OpenCV is unavailable

**Testing**
- 77-test suite covering all 8 modules
- Integration tests: hidden image, anaglyph, inpaint pipeline, adaptive SIRDS
- Visual gallery generator (7 PNG mosaics)

---

[Unreleased]: https://github.com/your-org/depthforge/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/your-org/depthforge/releases/tag/v0.1.0

## [0.4.0] — Phase 4

### Added
- **ComfyUI integration** (`depthforge/comfyui/`): 10 custom nodes
  - `DF_DepthPrep`, `DF_PatternGen`, `DF_PatternLibrary`, `DF_Stereogram`
  - `DF_AnaglyphOut`, `DF_StereoPair`, `DF_HiddenImage`
  - `DF_SafetyLimiter`, `DF_QCOverlay`, `DF_VideoSequence`
  - `install()` / `uninstall()` helpers for one-command deployment
- **Nuke integration** (`depthforge/nuke/`): Python gizmo + toolbar
  - `DepthForgeGizmo` class: 30+ knobs, all 5 synthesis modes, JSON serialization
  - `toolbar.py`: Full menu with presets submenu and keyboard shortcuts
  - `install()` / `create_node()` public API; safe no-op outside Nuke
- **Flask preview server** (`depthforge/web/`): local web UI
  - REST endpoints: `/api/synthesize`, `/api/analyze`, `/api/estimate_depth`, `/api/qc_overlay`
  - SSE progress streaming for async jobs (`/api/progress/<job_id>`)
  - Pattern library browser with live previews
  - Preset gallery with one-click apply
  - Single-page UI with dark theme, no external dependencies

## [0.5.0] — Phase 5

### Added
- **Multi-threaded synthesis** (`depthforge/core/parallel.py`)
  - `ParallelConfig`: n_workers, chunk_rows, min_rows_for_parallel, progress callback
  - `parallel_synthesize()`: row-parallel SIRDS using ThreadPoolExecutor
  - `benchmark()`: serial vs parallel timing utility
  - Transparent serial fallback for small images; tile-phase continuity across chunks

- **Optical flow depth** (`depthforge/core/optical_flow.py`)
  - `FlowDepthConfig`: Farneback parameters, motion_scale, blur_sigma, invert, min/max_motion
  - `compute_flow()`: dense optical flow via cv2.calcOpticalFlowFarneback
  - `flow_to_depth()`: motion magnitude → normalised depth proxy
  - `flow_warp()`: forward-warp a frame by a flow field
  - `FlowDepthEstimator`: stateful estimator — `feed()`, `estimate_from_pair()`, `estimate_sequence()`
  - `detect_scene_cut()`: mean absolute difference scene-cut detector

- **Temporal coherence** (`depthforge/core/temporal.py`)
  - `TemporalConfig`: strategy (ema/windowed/flow_guided/adaptive), alpha, window_size, scene-cut reset
  - `TemporalSmoother`: unified smoother with scene-cut auto-reset
  - `DepthHistory`: sliding window with Gaussian temporal smoothing for offline pipelines

- **Export profiles** (`depthforge/core/export.py`)
  - `ExportProfile`: full delivery parameter bundle (format, DPI, bit-depth, ICC, watermark, etc.)
  - 7 built-in profiles: web_srgb, print_300, print_cmyk, broadcast_rec709, cinema_p3, archive_exr, social_webp
  - `Exporter`: applies profile, resizes, watermarks, writes file
  - `ExportQueue`: thread-safe background batch export with pending/completed tracking
  - `register_profile()` / `list_profiles()` / `get_profile()`

- **OFX C++ plugin** (`depthforge/ofx/`)
  - `plugin.h`: `DepthForgePlugin` class, all 30+ parameter identifiers
  - `plugin.cpp`: full OFX lifecycle — `describeAction`, `describeInContext` (clips + params),
    `createInstance`, `renderAction`, `isIdentityAction`; Python subprocess bridge
  - `include/ofx_minimal.h`: self-contained OFX API stubs (no external SDK required for scaffold build)
  - `CMakeLists.txt`: cross-platform build (.ofx.bundle on macOS/Linux/Windows)
  - `DepthForge.ofx`: pre-compiled Linux-x86-64 shared library (g++ 13 / C++17)
  - Python bridge `ofx/__init__.py`: `OFXParams`, `OFXBridge.synthesize_direct()`,
    `OFXPluginInfo`, `build_command()`, `validate_plugin()`, `find_installed_bundles()`

### Testing
- 107-test Phase 5 suite (396 total across all phases)
