# DepthForge — Algorithm Reference

A technical deep-dive into the stereogram synthesis algorithms used in DepthForge.

---

## 1. The Core Stereogram Algorithm

### Background

A Single Image Random Dot Stereogram (SIRDS) works by exploiting **binocular disparity** — the horizontal difference between what the left and right eye see. The brain interprets this disparity as depth.

In a SIRDS, the same flat image encodes both views simultaneously. The trick is that identical-coloured dots appear at a horizontal spacing that varies with depth — wider separation = nearer, narrower = farther (or vice versa depending on convention).

### Constraint-Link Approach

DepthForge uses the constraint-link algorithm described by Thimbleby, Inglis & Witten (1994), which is both elegant and efficient.

**Key insight:** For a pixel at position `x` in a scanline, the depth value `d` implies that the left eye sees column `x − s/2` and the right eye sees column `x + s/2`, where `s` is the horizontal shift in pixels. These two positions *must be the same colour* for the brain to fuse them as a single 3D point.

**Shift computation:**

```
adj_depth = (depth[y,x] - convergence) × depth_factor
shift[y,x] = clamp(adj_depth × max_shift_px, -max_shift, max_shift)
```

Where `convergence` determines which depth level sits at the screen plane (zero shift).

**Building constraints (per scanline):**

```python
same = [-1] * W           # same[x] → x must match color of same[x]

for x in range(W):
    s = shift[x]
    if s == 0: continue
    x_left  = x - abs(s) // 2
    x_right = x + abs(s) - abs(s) // 2

    if 0 <= x_left < W and 0 <= x_right < W:
        # Find root of x_left's chain
        root = x_left
        while same[root] != -1:
            root = same[root]
        if root != x_right:
            same[x_right] = root    # link x_right → root
```

**Assigning colours (per scanline):**

```python
color_idx = [-1] * W

for x in range(W):
    linked = same[x]
    if linked != -1 and color_idx[linked] != -1:
        color_idx[x] = color_idx[linked]   # inherit from linked pixel
    else:
        color_idx[x] = x % tile_width      # sample from pattern tile

output[y] = pattern_row[color_idx]
```

The result: pixels that share a constraint get the same colour, creating the horizontal repetition the brain fuses into depth.

### Convergence Plane

The `convergence` parameter controls which depth level sits at zero parallax (the "screen plane"). Objects in front of the screen plane appear to pop forward; objects behind it recede.

```
adj_depth = (depth - convergence) × depth_factor

convergence = 0.0 → all depth recedes behind screen
convergence = 0.5 → mid-depth at screen (default)
convergence = 1.0 → all depth pops forward
```

### Oversampling

For sub-pixel accuracy, the synthesizer can render at N× resolution then Lanczos-downsample:

1. Upsample depth map by N× (bilinear)
2. Upsample pattern tile by N×
3. Run core algorithm at high resolution
4. Lanczos downsample to original size

This reduces aliasing at depth boundaries, at the cost of N² memory.

---

## 2. Depth Conditioning Pipeline

Raw depth maps from any source need conditioning before synthesis. The pipeline stages are:

```
Raw input  →  Normalise  →  Invert  →  Bilateral smooth  →  Dilation  →  Falloff curve  →  Clamp  →  Region masks
```

### Bilateral Filter

Edge-aware smoothing that blurs within regions but preserves object boundaries.

**Tier 1 (OpenCV):**
```
cv2.bilateralFilter(depth_uint8, d=sigma_space*2+1,
                    sigmaColor=sigma_color*255,
                    sigmaSpace=sigma_space)
```

**Tier 0 fallback:** Separable box blur — correct but lacks edge preservation.

### Morphological Dilation

Expands the "near" (bright) regions outward. This is critical to prevent the background from "peeking" through depth discontinuities — without dilation, you get fringing where the background colour bleeds into object edges.

**Tier 1 (OpenCV):** `cv2.dilate()` with elliptical structuring element.
**Tier 2 (SciPy):** `grey_dilation()`.
**Tier 0:** Sliding-window max via `numpy.lib.stride_tricks`.

### Falloff Curves

All curves map [0,1] → [0,1] and preserve endpoints (f(0)=0, f(1)=1).

| Curve | Formula |
|---|---|
| LINEAR | `t` |
| S_CURVE | `3t² − 2t³` (smooth step) |
| GAMMA | `t^γ` (γ<1 brightens near, γ>1 darkens) |
| LOGARITHMIC | `log(1 + 9t) / log(10)` |
| EXPONENTIAL | `(e^(2t) − 1) / (e² − 1)` |

---

## 3. Adaptive Dot Density

### Motivation

Uniform random dots give equal visual weight everywhere, which is wasteful in flat regions and insufficient near edges. Adaptive density:
- Places small, dense dots near depth discontinuities → preserves fine detail
- Places large, sparse dots in flat regions → cleaner, less noise

### Complexity Map

The complexity of each pixel is estimated from the depth map:

```
complexity = 0.7 × edge_magnitude + 0.3 × local_variance
```

**Edge magnitude** — Sobel gradient magnitude, normalised to [0,1].  
**Local variance** — 7×7 window variance, normalised to [0,1].  
The combined map is then Gaussian-blurred to remove isolated speckles.

### Density Levels

The complexity map is discretised into N levels. For each level `l` (0 = simple, N-1 = complex):

```
t       = l / (N - 1)
radius  = max_radius - t × (max_radius - min_radius)
spacing = max_spacing - t × (max_spacing - min_spacing)
```

Dots are placed on a jittered grid, only at positions whose complexity level matches. This creates spatially-varying density that respects the depth structure.

---

## 4. Anaglyph Compositing

### True / Colour Anaglyph

Trivially separates the channels:
```
R  ← left_eye.R
G  ← right_eye.G
B  ← right_eye.B
```
Fast but with colour shift (reds in left eye appear different).

### Grey Anaglyph

Both eyes converted to Rec.709 luminance first:
```
L_grey = 0.2126×R + 0.7152×G + 0.0722×B
```
No colour shift possible; good for high-contrast content.

### Half-Colour

Compromise: left eye grey → Red, right eye full colour → Cyan.
```
R  ← luminance(left_eye)
G  ← right_eye.G
B  ← right_eye.B
```

### Optimised (Dubois)

Uses 3×3 matrices derived from least-squares optimisation over a set of Munsell colour samples, minimising retinal rivalry.

```
output = left_float @ M_left.T + right_float @ M_right.T
```

Where `M_left` and `M_right` are the published Dubois matrices for red/cyan anaglyphs on typical sRGB monitors (Dubois 2001).

This is the recommended mode for all final output.

---

## 5. Stereo Pair Synthesis

### Forward Warp

For each scanline, each source pixel at `(y, x)` is warped to a new position based on depth:

```
shift = depth[y, x] × max_parallax_px
left_x  = clamp(x - shift × eye_balance,       0, W-1)
right_x = clamp(x + shift × (1 - eye_balance), 0, W-1)
```

Later pixels (higher x) overwrite earlier ones at the same destination, which correctly handles occlusion for objects moving in the shift direction.

### Occlusion Detection

After the forward warp, any destination pixel that received no source pixel is an occlusion — background that was hidden behind a near object in the source view.

These are detected by maintaining a `covered[H, W]` boolean array and taking `~covered` as the occlusion mask.

### Background Fill

Before inpainting, occluded pixels are filled with a fast placeholder:
- `"edge"` — propagate nearest non-occluded pixel left→right, then right→left
- `"mirror"` — reflect at image borders
- `"black"` — zero fill

These are low-quality; `inpaint_occlusion()` then replaces them with proper content.

---

## 6. Inpainting

### Patch-Based (Exemplar)

For each hole pixel, find the patch in the surrounding image that best matches the known (non-occluded) pixels adjacent to the hole, measured by sum of squared differences (SSD). Copy that patch's centre value to the hole pixel.

**With OpenCV:** Uses the Telea (2004) Fast Marching Method — much faster and higher quality than the NumPy fallback.

**Without OpenCV:** Pure NumPy pixel-by-pixel search. Correct but O(holes × search_area). Only practical for small occlusion regions (< 5% of image area).

### AI Callback (Phase 4)

The AI callback hook allows any inpainting model to be registered:

```python
register_ai_inpaint_callback(my_sd_inpaint_function)
```

When `inpaint_occlusion` is called with `AUTO` or `AI_CALLBACK` method, it passes the image and mask to the callback. The ComfyUI `DF_Inpaint` node wires a Stable Diffusion inpainting model (e.g. `sd-v1-5-inpainting`) here, enabling context-aware background reconstruction.

Falls back to patch-based if the callback raises an exception.

---

## 7. Pattern Generators

All generators produce tileable (seamlessly repeating) output.

### Perlin-style Noise

Pure NumPy implementation using summed directional sinusoids:

```
for each octave:
    contribution = amplitude × sin(freq × x·cos(θ) + y·sin(θ) + φ_x)
                             × cos(freq × x·sin(θ) − y·cos(θ) + φ_y)
    accumulated += contribution
    amplitude   *= 0.5
    frequency   *= 2.0
```

This approximates Perlin noise without requiring gradient tables. The random phase `φ` and angle `θ` per octave produce good organic variation.

### Plasma

Classic demoscene plasma: sum of five sinusoidal waves at different orientations:

```
val = sin(x + φ₀) + sin(y + φ₁)
    + sin((x + y)/2 + φ₂)
    + sin(√(x² + y²)/2 + φ₃)
    + cos(0.3x − 0.7y + φ₄)
```

### Voronoi (Worley Noise)

For each pixel, find the distance to the nearest of N random seed points. Normalise the resulting distance field. Produces cellular, organic structure.

### Mandelbrot

Standard escape-time colouring with 64 iterations. The tiling is achieved by choosing a small, period-1 region of the set boundary.

---

## References

- Thimbleby, H., Inglis, S., & Witten, I. (1994). *Displaying 3D Images: Algorithms for Single Image Random Dot Stereograms.* IEEE Computer, 27(10), 38–48.
- Tyler, C.W. & Clarke, M.B. (1990). *The autostereogram.* SPIE Stereoscopic Displays and Applications, 1258, 182–197.
- Dubois, E. (2001). *A Methodology for Developing Display and Image Processing Technology for Stereoscopic Video.* Journal of the Society for Information Display, 9(2), 109–119.
- Telea, A. (2004). *An Image Inpainting Technique Based on the Fast Marching Method.* Journal of Graphics Tools, 9(1), 23–34.
