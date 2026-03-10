# DepthForge API Reference

**Version 0.1.0 — Phase 1 Core Engine**

All public symbols are importable directly from the top-level package:

```python
import depthforge as df
```

---

## Module Index

| Module | Public API |
|---|---|
| [`synthesizer`](#synthesizer) | `StereoParams`, `synthesize`, `load_depth_image`, `save_stereogram` |
| [`depth_prep`](#depth_prep) | `DepthPrepParams`, `FalloffCurve`, `RegionMask`, `prep_depth`, `normalise_depth`, `depth_from_image`, `compute_vergence_map`, `detect_window_violations` |
| [`pattern_gen`](#pattern_gen) | `PatternParams`, `PatternType`, `ColorMode`, `GridStyle`, `generate_pattern`, `load_tile`, `tile_to_frame` |
| [`anaglyph`](#anaglyph) | `AnaglyphMode`, `AnaglyphParams`, `make_anaglyph`, `make_anaglyph_from_depth` |
| [`stereo_pair`](#stereo_pair) | `StereoPairParams`, `StereoLayout`, `make_stereo_pair`, `compose_side_by_side` |
| [`hidden_image`](#hidden_image) | `HiddenImageParams`, `encode_hidden_image`, `mask_to_depth`, `load_hidden_mask`, `text_to_mask`, `shape_to_mask` |
| [`adaptive_dots`](#adaptive_dots) | `AdaptiveDotParams`, `generate_adaptive_dots`, `complexity_from_depth` |
| [`inpainting`](#inpainting) | `InpaintParams`, `InpaintMethod`, `inpaint_occlusion`, `register_ai_inpaint_callback` |

---

## Package-Level Flags

```python
df.HAS_CV2      # bool — OpenCV available
df.HAS_SCIPY    # bool — SciPy available
df.HAS_TORCH    # bool — PyTorch available
df.HAS_OCIO     # bool — PyOpenColorIO available
df.HAS_OPENEXR  # bool — OpenEXR available
```

---

## synthesizer

### `StereoParams`

Dataclass controlling the core stereogram synthesis algorithm.

```python
@dataclass
class StereoParams:
    depth_factor:            float         = 0.4
    max_parallax_fraction:   float         = 1/30
    eye_separation_fraction: float         = 0.06
    convergence:             float         = 0.5
    invert_depth:            bool          = False
    oversample:              int           = 1
    seed:                    Optional[int] = None
    safe_mode:               bool          = False
```

| Field | Range | Description |
|---|---|---|
| `depth_factor` | −1.0 … 1.0 | Parallax multiplier. Positive = scene recedes. Negative = pops forward. |
| `max_parallax_fraction` | 0.01 … 0.1 | Maximum shift as fraction of image width. 1/30 ≈ 0.033 is the ergonomic limit. |
| `eye_separation_fraction` | 0.04 … 0.08 | Assumed inter-ocular distance ÷ viewing distance. Default 0.06 ≈ 65 mm eyes at 1 m. |
| `convergence` | 0.0 … 1.0 | Which depth value sits at the screen plane (zero parallax). 0.5 = mid-scene. |
| `invert_depth` | bool | Flip near/far convention before synthesis. |
| `oversample` | 1 … 4 | Internal render scale. 2 = 2× resolution then Lanczos downsample. |
| `seed` | int or None | Random seed for SIRDS reproducibility. |
| `safe_mode` | bool | Hard-clamp `depth_factor` to ±0.5 and `max_parallax_fraction` to 1/30. |

**Raises:** `ValueError` if `oversample < 1` or `convergence` outside [0, 1].

**Methods:**

```python
params.max_shift_px(width: int) -> int
# Maximum pixel shift for a given image width.

params.eye_sep_px(width: int) -> int
# Inter-ocular separation in pixels for a given image width.
```

---

### `synthesize`

```python
def synthesize(
    depth:   np.ndarray,      # float32 (H, W), values [0, 1]
    pattern: np.ndarray,      # RGBA uint8 (tH, tW, 4)
    params:  StereoParams = StereoParams(),
) -> np.ndarray               # RGBA uint8 (H, W, 4)
```

Synthesise a single-image stereogram.

**Parameters:**

- `depth` — 2-D float32 array, shape `(H, W)`. Values in [0, 1]. 1.0 = nearest, 0.0 = farthest. Prepare with `prep_depth()` before calling.
- `pattern` — RGBA uint8 tile. Accepts `(H,W)`, `(H,W,3)`, or `(H,W,4)`. Will be tiled to fill the output.
- `params` — `StereoParams` instance.

**Returns:** RGBA uint8 stereogram, shape `(H, W, 4)`.

**Raises:** `ValueError` if `depth` is not 2-D.

---

### `load_depth_image`

```python
def load_depth_image(path: str) -> np.ndarray  # float32 (H, W), [0, 1]
```

Load any image file as a depth map (luminance only, no conditioning).

---

### `save_stereogram`

```python
def save_stereogram(arr: np.ndarray, path: str) -> None
```

Save an RGBA uint8 array as a PNG file.

---

## depth_prep

### `FalloffCurve`

```python
class FalloffCurve(Enum):
    LINEAR      # identity
    GAMMA       # power-law (use falloff_gamma to control)
    S_CURVE     # smooth ease-in/out (3t²−2t³)
    LOGARITHMIC # emphasises near depth differences
    EXPONENTIAL # emphasises far depth differences
```

---

### `RegionMask`

```python
@dataclass
class RegionMask:
    mask:       np.ndarray   # float32 (H, W), [0, 1]. 1 = full override.
    multiplier: float = 0.0  # depth multiplier where mask == 1.
```

Locally overrides depth values. `multiplier=0.0` flattens the region to zero parallax; `multiplier=2.0` doubles depth.

---

### `DepthPrepParams`

```python
@dataclass
class DepthPrepParams:
    normalise:             bool             = True
    invert:                bool             = False
    bilateral_sigma_space: float            = 5.0
    bilateral_sigma_color: float            = 0.1
    dilation_px:           int              = 3
    smooth_passes:         int              = 1
    falloff_curve:         FalloffCurve     = FalloffCurve.LINEAR
    falloff_gamma:         float            = 1.0
    near_plane:            float            = 0.0
    far_plane:             float            = 1.0
    region_masks:          List[RegionMask] = []
    edge_preserve:         bool             = True
```

---

### `prep_depth`

```python
def prep_depth(
    raw:    np.ndarray,
    params: DepthPrepParams = DepthPrepParams(),
) -> np.ndarray  # float32 (H, W), [0, 1]
```

Full depth conditioning pipeline. Accepts any numeric dtype, any shape `(H,W)`, `(H,W,1)`, or `(H,W,C)` (multichannel → luminance).

**Pipeline stages (in order):**
1. Collapse to 2-D float32
2. Normalise → [0, 1]
3. Invert (optional)
4. Bilateral or Gaussian smooth (`smooth_passes` iterations)
5. Morphological dilation (expand near regions)
6. Falloff curve remap
7. Near/far plane clamp
8. Region mask overrides

---

### `normalise_depth`

```python
def normalise_depth(raw: np.ndarray) -> np.ndarray  # float32 (H, W), [0, 1]
```

Single-step normalise only — no smoothing, dilation, or remapping.

---

### `depth_from_image`

```python
def depth_from_image(
    path:   str,
    params: Optional[DepthPrepParams] = None,
) -> np.ndarray  # float32 (H, W), [0, 1]
```

Load a depth image from disk and run the full prep pipeline. Handles 8-bit, 16-bit TIFF, and any Pillow-readable format.

---

### `compute_vergence_map`

```python
def compute_vergence_map(
    depth:            np.ndarray,
    eye_sep_fraction: float = 0.06,
    screen_distance:  float = 600.0,   # mm
) -> np.ndarray  # float32 (H, W), degrees
```

Estimate vergence angle per pixel. Values > ~3° indicate potential eye strain.

---

### `detect_window_violations`

```python
def detect_window_violations(
    depth:     np.ndarray,
    threshold: float = 0.05,
) -> np.ndarray  # bool (H, W)
```

Detect stereo window violations — near objects incorrectly breaking the frame edge. Returns a boolean mask; `True` = violation likely.

---

## pattern_gen

### `PatternType`

```python
class PatternType(Enum):
    RANDOM_NOISE    # random dots — classic SIRDS
    PERLIN          # smooth layered noise
    PLASMA          # sinusoidal wave composite
    VORONOI         # cellular / Worley noise
    GEOMETRIC_GRID  # dots, hexes, checks, stripes
    MANDELBROT      # Mandelbrot set fractal
    DOT_MATRIX      # halftone screen
    CUSTOM_TILE     # user-supplied image file
```

### `ColorMode`

```python
class ColorMode(Enum):
    MONOCHROME   # single hue, varying brightness
    PSYCHEDELIC  # full HSV rotation
    GREYSCALE    # classic monochrome
    CUSTOM       # reserved for future palette support
```

### `GridStyle`

```python
class GridStyle(Enum):
    DOTS     # circular dots on a grid
    HEXES    # hexagonal outlines
    CHECKS   # alternating filled squares
    STRIPES  # vertical stripes
```

---

### `PatternParams`

```python
@dataclass
class PatternParams:
    pattern_type:  PatternType  = PatternType.RANDOM_NOISE
    tile_width:    int          = 128
    tile_height:   int          = 128
    color_mode:    ColorMode    = ColorMode.GREYSCALE
    hue:           float        = 0.0
    saturation:    float        = 0.8
    scale:         float        = 1.0
    octaves:       int          = 4
    grid_style:    GridStyle    = GridStyle.DOTS
    grid_spacing:  int          = 8
    seed:          Optional[int]= None
    safe_mode:     bool         = False
    lut_path:      Optional[str]= None   # Phase 3: .cube LUT path
```

**Raises:** `ValueError` if `tile_width < 4` or `tile_height < 4`.

---

### `generate_pattern`

```python
def generate_pattern(params: PatternParams = PatternParams()) -> np.ndarray
# Returns: RGBA uint8 (tile_H, tile_W, 4)
```

Generate a tileable procedural pattern. All generators are pure Python/NumPy (no external deps required). OpenCV/SciPy improve speed when available.

---

### `load_tile`

```python
def load_tile(
    path:        str,
    tile_width:  int = 0,
    tile_height: int = 0,
) -> np.ndarray  # RGBA uint8 (H, W, 4)
```

Load an image as a pattern tile. If `tile_width`/`tile_height` are 0, uses native resolution.

---

### `tile_to_frame`

```python
def tile_to_frame(
    tile:   np.ndarray,  # (tH, tW, 4)
    height: int,
    width:  int,
) -> np.ndarray  # RGBA uint8 (height, width, 4)
```

Tile a pattern to fill a full frame using `np.tile`.

---

## anaglyph

### `AnaglyphMode`

```python
class AnaglyphMode(Enum):
    TRUE_ANAGLYPH    # R from left, GB from right
    GREY_ANAGLYPH    # both eyes → luminance before separation
    COLOUR_ANAGLYPH  # alias for TRUE_ANAGLYPH
    HALF_COLOUR      # left eye grey → R; right eye colour → Cyan
    OPTIMISED        # Dubois least-squares matrices (best quality)
```

---

### `AnaglyphParams`

```python
@dataclass
class AnaglyphParams:
    mode:        AnaglyphMode = AnaglyphMode.OPTIMISED
    parallax_px: int          = 20
    swap_eyes:   bool         = False
    gamma:       float        = 1.0
```

---

### `make_anaglyph`

```python
def make_anaglyph(
    left:   np.ndarray,       # RGB or RGBA uint8 (H, W, 3|4)
    right:  np.ndarray,       # RGB or RGBA uint8 (H, W, 3|4)
    params: AnaglyphParams = AnaglyphParams(),
) -> np.ndarray               # RGBA uint8 (H, W, 4)
```

Composite a pre-computed L/R stereo pair into an anaglyph.

---

### `make_anaglyph_from_depth`

```python
def make_anaglyph_from_depth(
    source: np.ndarray,       # RGB or RGBA uint8 (H, W, 3|4)
    depth:  np.ndarray,       # float32 (H, W), [0, 1]
    params: AnaglyphParams = AnaglyphParams(),
) -> np.ndarray               # RGBA uint8 (H, W, 4)
```

Generate an anaglyph directly from a source image and depth map. Internally synthesises L and R views via horizontal warp, then composites. For production use with occlusion fill, use `make_stereo_pair` + `inpaint_occlusion` + `make_anaglyph`.

---

## stereo_pair

### `StereoLayout`

```python
class StereoLayout(Enum):
    SEPARATE      # returns (left, right) as independent arrays
    SIDE_BY_SIDE  # concatenated horizontally [left | right]
    TOP_BOTTOM    # concatenated vertically [top=left / bottom=right]
    ANAGLYPH      # delegates to make_anaglyph()
```

---

### `StereoPairParams`

```python
@dataclass
class StereoPairParams:
    max_parallax_fraction: float       = 1/30
    eye_balance:           float       = 0.5
    layout:                StereoLayout= StereoLayout.SEPARATE
    feather_px:            int         = 3
    invert_depth:          bool        = False
    background_fill:       str         = "edge"  # "edge" | "mirror" | "black"
```

---

### `make_stereo_pair`

```python
def make_stereo_pair(
    source: np.ndarray,          # RGBA or RGB uint8 (H, W, 3|4)
    depth:  np.ndarray,          # float32 (H, W), [0, 1]
    params: StereoPairParams = StereoPairParams(),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
# Returns: (left, right, occlusion_mask)
#   left, right:      RGBA uint8 (H, W, 4)
#   occlusion_mask:   float32    (H, W), [0=visible, 1=occluded]
```

Synthesise a stereo pair via per-row horizontal warp. Occluded background regions are filled with `background_fill` strategy; the occlusion mask marks them for optional inpainting.

---

### `compose_side_by_side`

```python
def compose_side_by_side(
    left:      np.ndarray,
    right:     np.ndarray,
    gap_px:    int = 0,
    gap_color: Tuple[int,int,int,int] = (0,0,0,255),
) -> np.ndarray  # RGBA uint8 (H, left_W + gap + right_W, 4)
```

Compose L and R views horizontally with an optional gap strip.

---

## hidden_image

### `HiddenImageParams`

```python
@dataclass
class HiddenImageParams:
    foreground_depth: float       = 0.8
    background_depth: float       = 0.0
    edge_soften_px:   int         = 4
    depth_scale:      float       = 1.0
    invert_mask:      bool        = False
    stereo_params:    StereoParams= StereoParams(depth_factor=0.35)
```

**Raises:** `ValueError` if `foreground_depth` or `background_depth` outside [0, 1].

---

### `encode_hidden_image`

```python
def encode_hidden_image(
    pattern: np.ndarray,          # RGBA uint8 tile (tH, tW, 4)
    mask:    np.ndarray,          # (H, W), float [0,1] or uint8 [0,255]
    params:  HiddenImageParams = HiddenImageParams(),
) -> np.ndarray                   # RGBA uint8 (H, W, 4)
```

Encode a shape or text as a hidden stereogram. Converts mask → depth map via `mask_to_depth()`, then calls `synthesize()`.

---

### `mask_to_depth`

```python
def mask_to_depth(
    mask:   np.ndarray,           # (H, W) float or uint8
    params: HiddenImageParams,
) -> np.ndarray                   # float32 (H, W), [0, 1]
```

Convert a binary/greyscale mask to a depth map suitable for `synthesize()`.

---

### `load_hidden_mask`

```python
def load_hidden_mask(
    path:        str,
    target_size: Optional[Tuple[int,int]] = None,  # (W, H)
) -> np.ndarray  # float32 (H, W), [0, 1]
```

Load any image as a hidden-image mask. White = shape region (will appear near).

---

### `text_to_mask`

```python
def text_to_mask(
    text:      str,
    width:     int,
    height:    int,
    font_size: int           = 0,      # 0 = auto-fit
    font_path: Optional[str] = None,   # None = Pillow default
    padding:   int           = 20,
    center:    bool          = True,
) -> np.ndarray  # float32 (height, width), [0, 1]
```

Generate a hidden-image mask from a text string. White pixels = the text (will appear near).

---

### `shape_to_mask`

```python
def shape_to_mask(
    shape:   str,    # "circle" | "square" | "triangle" | "star" | "diamond" | "arrow"
    width:   int,
    height:  int,
    padding: int = 20,
) -> np.ndarray  # float32 (height, width), [0, 1]
```

Generate a mask from a named primitive shape.

**Raises:** `ValueError` for unknown shape names.

---

## adaptive_dots

### `AdaptiveDotParams`

```python
@dataclass
class AdaptiveDotParams:
    tile_width:      int                  = 256
    tile_height:     int                  = 256
    min_dot_radius:  int                  = 1
    max_dot_radius:  int                  = 4
    min_spacing:     int                  = 3
    max_spacing:     int                  = 12
    n_levels:        int                  = 5
    complexity_blur: float                = 2.0
    dot_color:       Tuple[int,int,int]   = (255, 255, 255)
    bg_color:        Tuple[int,int,int]   = (0, 0, 0)
    seed:            Optional[int]        = None
    jitter:          float                = 0.3
```

Auto-corrects: `min_dot_radius` floored to 1; `max_dot_radius` floored to `min`; `min_spacing` floored to `min_dot_radius * 2`.

---

### `generate_adaptive_dots`

```python
def generate_adaptive_dots(
    depth:  np.ndarray,              # float32 (H, W), [0, 1]
    params: AdaptiveDotParams = AdaptiveDotParams(),
) -> np.ndarray                      # RGBA uint8 (tile_H, tile_W, 4)
```

Generate a complexity-driven SIRDS dot tile. The complexity map is derived from the depth map's edge magnitude and local variance — high-complexity areas get smaller, denser dots.

The returned tile is passed directly to `synthesize()`.

---

### `complexity_from_depth`

```python
def complexity_from_depth(depth: np.ndarray) -> np.ndarray
# Returns: float32 (H, W), [0, 1]
```

Compute a normalised complexity map from a depth map. Uses Sobel edge magnitude + local variance. Useful for visualisation and debugging.

---

## inpainting

### `InpaintMethod`

```python
class InpaintMethod(Enum):
    PATCH_BASED   # exemplar / patch-match (OpenCV Telea or pure NumPy)
    CLEAN_PLATE   # composite from a background plate
    EDGE_EXTEND   # fast: propagate nearest edge pixel
    AI_CALLBACK   # delegate to registered external AI model
    AUTO          # choose best available method
```

**AUTO priority:** CLEAN_PLATE → AI_CALLBACK → PATCH_BASED.

---

### `InpaintParams`

```python
@dataclass
class InpaintParams:
    method:         InpaintMethod          = InpaintMethod.AUTO
    patch_size:     int                    = 8
    search_radius:  int                    = 64
    clean_plate:    Optional[np.ndarray]   = None
    ai_callback:    Optional[Callable]     = None
    dilate_mask_px: int                    = 2
    blend_px:       int                    = 3
```

---

### `inpaint_occlusion`

```python
def inpaint_occlusion(
    image:   np.ndarray,          # RGBA uint8 (H, W, 4)
    mask:    np.ndarray,          # float32 (H, W), [0, 1]. 1=occluded.
    params:  InpaintParams = InpaintParams(),
) -> np.ndarray                   # RGBA uint8 (H, W, 4)
```

Fill occluded (exposed background) regions in a stereo view.

If `mask.max() < 0.01`, returns `image` unchanged (fast path).

---

### `register_ai_inpaint_callback`

```python
def register_ai_inpaint_callback(
    callback: Callable[[np.ndarray, np.ndarray], np.ndarray]
) -> None
```

Register a global AI inpainting function. Signature:
```python
def my_callback(image_rgba: np.ndarray, mask_float: np.ndarray) -> np.ndarray:
    # image_rgba: RGBA uint8 (H, W, 4)
    # mask_float: float32 (H, W) — 1.0 = fill this region
    # returns:    RGBA uint8 (H, W, 4)
    ...
```

Called by `AUTO` and `AI_CALLBACK` methods. The ComfyUI `DF_Inpaint` node registers its SD model here during workflow execution.

---

## Type Aliases

```python
# Used throughout the codebase
DepthMap   = np.ndarray   # float32 (H, W), values [0, 1]
PatternTile= np.ndarray   # RGBA uint8 (tH, tW, 4)
Stereogram = np.ndarray   # RGBA uint8 (H, W, 4)
Mask       = np.ndarray   # float32 (H, W), values [0, 1]
```

---

## Exceptions

All public functions raise standard Python exceptions:

| Exception | When |
|---|---|
| `ValueError` | Invalid parameter values (bad oversample, convergence out of range, unknown shape name, depth not 2-D) |
| `FileNotFoundError` | File path does not exist (load functions) |
| `RuntimeError` | Internal processing failure (rare; always includes a descriptive message) |

---

## Compatibility

| Feature | Tier 0 (NumPy+Pillow) | Tier 1 (+OpenCV) | Tier 2 (+SciPy) |
|---|---|---|---|
| `synthesize` | ✅ Full | ✅ Full | ✅ Full |
| `prep_depth` bilateral | Box blur fallback | ✅ Proper bilateral | Gaussian fallback |
| `prep_depth` dilation | Sliding-window max | ✅ Morphological | ✅ Grey dilation |
| `generate_pattern` | ✅ Full | ✅ Full | ✅ Full |
| `make_anaglyph` | ✅ Full | ✅ Full | ✅ Full |
| `make_stereo_pair` | ✅ Full | ✅ Feathered mask | ✅ Full |
| `inpaint_occlusion` | NumPy patch-match | ✅ OpenCV Telea | ✅ Full |
| `generate_adaptive_dots` | NumPy gradient | ✅ Sobel + fast resize | ✅ Sobel + grey dilation |
