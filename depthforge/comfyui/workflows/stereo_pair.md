# stereo_pair.json — Side-by-Side Stereo Pair Output

## What This Workflow Demonstrates

Generates the full **DF_StereoPair** output set: independent left view,
independent right view, an occlusion mask, and a composed side-by-side
image. The side-by-side composed output is the format expected by VR
headsets, 3D monitors, and stereo cinema delivery.

---

## DepthForge Nodes Used

| Node | Why it's here |
|------|---------------|
| **DF_DepthPrep** | Higher bilateral settings (space=8, dilation=5) chosen to minimise warping artefacts in the synthesized views |
| **DF_StereoPair** | Warps the source image into left and right perspectives using the depth map; also returns the occlusion mask and composed output |

---

## Required Inputs

| Input | Format | Notes |
|-------|--------|-------|
| Source image | RGB PNG/EXR, any resolution | The scene to be stereo-ified |
| Depth map | Greyscale PNG/EXR, **same resolution** | White = near, Black = far |

---

## The Four Outputs

| Output slot | Name | Description |
|-------------|------|-------------|
| 0 | `left_view` | Left-eye perspective shifted image |
| 1 | `right_view` | Right-eye perspective shifted image |
| 2 | `occlusion_mask` | White pixels = regions where the warp reveals background that was hidden; needs fill |
| 3 | `composed` | Left and right joined side-by-side (double width) |

> **Note on occlusion mask:** `make_stereo_pair()` returns the mask as
> `float32` internally. The node converts values `> 0.5` to white (255) in
> the RGBA preview. White regions are areas where inpainting or edge-fill
> is needed for clean delivery.

---

## Parameter Values and Rationale

### DF_DepthPrep
| Parameter | Value | Reason |
|-----------|-------|--------|
| `bilateral_space` | 8.0 | Stronger smoothing than default — view warping is sensitive to depth noise |
| `bilateral_color` | 0.08 | Lower colour sensitivity keeps edges cleaner |
| `dilation_px` | 5 | Fills depth gaps aggressively to reduce occluded regions |
| `falloff_curve` | `s_curve` | Cinematic S-curve depth roll-off |

### DF_StereoPair
| Parameter | Value | Reason |
|-----------|-------|--------|
| `max_parallax_fraction` | 0.033 | Safe 1/30 comfortable limit |
| `eye_balance` | 0.5 | Equal left/right shift — zero-parallax plane at mid-scene |
| `background_fill` | `edge` | Fills occluded pixels by smearing the edge colour inward — better than black for most content |
| `feather_px` | 3 | Blends the fill edge to avoid hard seams |
| `layout` | `side_by_side` | Standard SBS format for VR/3D monitors |
| `gap_px` | 0 | No gap between left and right — standard for delivery |

---

## Expected Output and Verification

1. **Left view** — looks almost identical to the source but slightly shifted right
2. **Right view** — slightly shifted left relative to source
3. **Occlusion mask** — white marks where background was exposed by the warp;
   should be minimal if `dilation_px` and `feather_px` are set correctly
4. **Composed** — double-width image; left half should match the left view,
   right half the right view

**Cross-eye viewing test:** to verify stereo without a headset, cross your eyes
gently while looking at the side-by-side composed image. The two panels should
merge into a single 3D image.

---

## Common Issues

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Large white areas in occlusion mask | Depth has extreme near values causing big pixel shifts | Reduce `max_parallax_fraction` or increase `dilation_px` |
| Hard seam visible in composed image | `feather_px` too low | Increase to 6–8 |
| Composed image is full-width not double | `layout` not set to `side_by_side` | Check layout widget |
| Warped edges look smeared | `background_fill` = `edge` with a non-flat edge | Switch to `mirror` which mirrors the edge inward |
| No visible 3D when cross-eye viewed | `max_parallax_fraction` too low | Increase to 0.05 |
