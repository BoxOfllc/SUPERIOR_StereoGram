# texture_pattern.json — Texture Stereogram with Pattern Library

## What This Workflow Demonstrates

Shows how to use the **DF_PatternLibrary** node to select from the 28+
named DepthForge patterns instead of generating one procedurally. The
library patterns are hand-tuned for stereogram use and cover categories
like noise, crystal, organic, and geometric. The workflow also previews
the actual pattern tile used alongside the final stereogram.

---

## DepthForge Nodes Used

| Node | Why it's here |
|------|---------------|
| **DF_DepthPrep** | Conditions the depth map with an S-curve falloff — softer depth transitions work better with visually rich texture patterns |
| **DF_PatternLibrary** | Fetches a named pattern from the built-in library and exposes its metadata as a STRING output |
| **DF_Stereogram** | Synthesizes the textured stereogram; depth_factor is slightly lower (0.3) than the SIRDS default because textured patterns carry more visual noise |

---

## Required Inputs

| Input | Format | Notes |
|-------|--------|-------|
| Depth map | Greyscale PNG/EXR, any resolution | White = near, Black = far |

---

## Parameter Values and Rationale

### DF_DepthPrep
| Parameter | Value | Reason |
|-----------|-------|--------|
| `falloff_curve` | `s_curve` | Softens transitions at near/far extremes — reduces edge shimmering when viewed with detailed texture patterns |
| `bilateral_space` | 5.0 | Standard smoothing |
| `dilation_px` | 3 | Standard gap fill |

### DF_PatternLibrary
| Parameter | Value | Reason |
|-----------|-------|--------|
| `pattern_name` | `perlin_noise` | Smooth organic noise — pleasant starting point; swap to explore the library |
| `width/height` | 128×128 | Standard tile size |
| `seed` | 42 | Reproducible output |

**Recommended patterns to try:**
- `crystalline` — geometric crystal facets, dramatic when fused
- `plasma_wave` — colourful plasma waves, psychedelic effect
- `fine_grain` — fine-grained noise, unobtrusive background
- `dot_matrix` — halftone-style dots, works well for print
- `voronoi_cells` — Voronoi diagram, strong but readable

### DF_Stereogram
| Parameter | Value | Reason |
|-----------|-------|--------|
| `depth_factor` | 0.3 | Slightly reduced (vs. 0.35 for SIRDS) — textured patterns are visually noisier and need less parallax to be comfortable |
| `max_parallax` | 0.033 | Safe 1/30 limit |

---

## Expected Output and Verification

- **Pattern Tile Preview** — shows the exact 128×128 tile used as the repeating base.
  Verify it matches the expected pattern type.
- **Texture Stereogram Output** — the full-resolution stereogram. Should show the
  texture visibly on the surface, with the 3D depth structure only becoming apparent
  when you fuse your eyes.

**To confirm depth is encoded:** fuse the stereogram with relaxed eyes. The depth
structure should be independent of the texture — the same pixel positions pop out
regardless of which pattern you choose.

---

## Common Issues

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Pattern is not visible / looks like noise | Pattern tile is too small | Increase tile size to 256×256 |
| Fusing is very difficult | `depth_factor` too high for this pattern type | Reduce to 0.2–0.25 |
| `pattern_name` dropdown is empty | Pattern library not installed | Confirm `depthforge/core/pattern_library.py` is present |
| Pattern repeating visibly | Tile is too small relative to output resolution | Increase tile size to 256 or 512 |
| Depth effect disappears with certain patterns | Pattern has too much high-frequency detail | Switch to `fine_grain` or `perlin_noise` |
