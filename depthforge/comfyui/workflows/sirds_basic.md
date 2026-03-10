# sirds_basic.json — Minimal SIRDS Test

## What This Workflow Demonstrates

The simplest possible DepthForge pipeline: one depth map in, one classic
random-dot stereogram out. Use this as a sanity-check that all nodes load
correctly and that your depth map is oriented correctly (white = near).

---

## DepthForge Nodes Used

| Node | Why it's here |
|------|---------------|
| **DF_DepthPrep** | Conditions the raw depth map — smooths noise, dilates thin edges, and applies a falloff curve so the synthesizer receives a clean 0–1 float array |
| **DF_PatternGen** | Generates the random-dot tile (`random_noise`, greyscale) that the SIRDS algorithm stamps and shifts to encode depth |
| **DF_Stereogram** | Core synthesis engine — reads the depth map and pattern tile and produces the finished stereogram image |

---

## Required Inputs

| Input | Format | Notes |
|-------|--------|-------|
| Depth map | Greyscale PNG/EXR, any resolution | **White = near (1.0), Black = far (0.0)** — DepthForge convention |

Load the image into the **Load Depth Map** (`LoadImage`) node. Any 8-bit
greyscale PNG works. For best results use a smooth gradient depth map —
sharp binary edges produce visible banding in the stereogram.

---

## Parameter Values and Rationale

### DF_DepthPrep
| Parameter | Value | Reason |
|-----------|-------|--------|
| `bilateral_space` | 5.0 | Light edge-preserving smooth — removes sensor noise without smearing depth boundaries |
| `bilateral_color` | 0.1 | Low colour sensitivity; works well for 8-bit depth |
| `dilation_px` | 3 | Fills tiny gaps at object edges where depth is undefined |
| `falloff_curve` | linear | Direct pass-through of depth values — best for comparing raw depth |
| `near_plane` | 0.0 | Use full depth range |
| `far_plane` | 1.0 | Use full depth range |

### DF_PatternGen
| Parameter | Value | Reason |
|-----------|-------|--------|
| `pattern_type` | `random_noise` | Classic SIRDS appearance — maximises camouflage of the hidden depth |
| `color_mode` | `greyscale` | Avoids colour aberration; easier on the eyes for first-time viewing |
| `tile_width/height` | 128×128 | Small enough to produce fine detail; large enough to avoid obvious tiling |
| `seed` | 42 | Deterministic — same seed = same output every run |

### DF_Stereogram
| Parameter | Value | Reason |
|-----------|-------|--------|
| `depth_factor` | 0.35 | Moderate depth pop — comfortable for most viewers at arm's length |
| `max_parallax` | 0.033 | 1/30 of image width — the widely cited comfortable-fusion limit |
| `oversample` | 1 | No oversampling — fast render for iteration |
| `safe_mode` | false | Manual control; enable if you want automatic clamping |
| `seed` | 42 | Matches pattern seed for reproducible output |

---

## Expected Output and Verification

The PreviewImage node shows a field of random grey dots with no obvious
structure visible at first glance. **To confirm correct output:**

1. **Relaxed-eye viewing** — hold the image at arm's length, relax your focus
   as if looking through it into the distance, and wait 5–10 seconds.
   A 3D scene should emerge.
2. **Depth verification** — bright (near) regions of the depth map should
   appear to float *in front* of dark (far) regions in the 3D view.
3. **Edge check** — object edges should not show obvious colour banding.
   If they do, increase `dilation_px` to 5–6.

---

## Common Issues

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Image looks completely flat | Depth map is all-grey (no contrast) | Check the depth map — use a high-contrast gradient |
| Scene appears inverted (far things pop out) | Depth map has white=far convention | Set `invert=true` in DF_DepthPrep, or tick `invert_depth` in DF_Stereogram |
| Too much eye strain / can't fuse | `depth_factor` too high | Reduce `depth_factor` to 0.2–0.25 |
| Faint repeating grid pattern | Pattern tile too small | Increase tile size to 256×256 |
| Output has sharp horizontal banding | Depth map has aliased hard edges | Increase `bilateral_space` to 8–12 |
