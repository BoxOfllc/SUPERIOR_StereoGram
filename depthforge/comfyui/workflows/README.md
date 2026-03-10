# DepthForge ComfyUI Workflows

Eight ready-to-use ComfyUI workflow JSON files covering all 10 DepthForge
custom nodes, plus full documentation for each.

---

## Quick Start

1. Open ComfyUI and drag any `.json` file from this folder onto the canvas.
2. Click **Load** if prompted.
3. Replace placeholder filenames (`depth_test.png`, `source_image.png`) in
   `LoadImage` nodes with your actual files.
4. Press **Queue Prompt**.

> **Depth map convention:** WHITE = near (close to camera), BLACK = far (background).
> All workflows assume this orientation. See [DEPTH_MAP_GUIDE.md](DEPTH_MAP_GUIDE.md)
> for how to prepare and verify your depth maps.

---

## Workflow Index

### Recommended Starting Points

| If you want to… | Start with |
|-----------------|-----------|
| Just confirm the nodes work | [`sirds_basic.json`](#1-sirds_basicjson) |
| Show clients a convincing demo | [`preset_workflow.json`](#6-preset_workflowjson) |
| Deliver for a 3D monitor or VR headset | [`stereo_pair.json`](#4-stereo_pairjson) |
| Deliver with red/cyan glasses | [`anaglyph_redcyan.json`](#3-anaglyph_redcyanjson) |
| Create a magic-eye poster | [`hidden_image.json`](#5-hidden_imagejson) |
| Prepare for broadcast delivery | Check safety limits with [`safety_check.json`](#8-safety_checkjson) first |
| Build your own pipeline from scratch | Use [`full_pipeline.json`](#7-full_pipelinejson) as the reference |

---

## 1. sirds_basic.json

**Nodes:** DF_DepthPrep, DF_PatternGen, DF_Stereogram
**Inputs:** one depth map
**Output:** classic random-dot SIRDS stereogram

The minimal working pipeline. Use this to verify your installation and test
a new depth map. Read left-to-right: load → prep → generate noise → synthesize → preview.

📄 [sirds_basic.md](sirds_basic.md)

---

## 2. texture_pattern.json

**Nodes:** DF_DepthPrep, DF_PatternLibrary, DF_Stereogram
**Inputs:** one depth map
**Output:** richly-textured stereogram + pattern tile preview

Demonstrates the 28+ named pattern library. Swap `pattern_name` to explore
the full catalogue (`crystalline`, `plasma_wave`, `voronoi_cells`, etc.).
The pattern tile preview is wired alongside the stereogram so you can see
exactly what texture is encoding your depth.

📄 [texture_pattern.md](texture_pattern.md)

---

## 3. anaglyph_redcyan.json

**Nodes:** DF_DepthPrep, DF_AnaglyphOut ×5
**Inputs:** RGB source image + depth map
**Output:** five anaglyph renders (one per mode)

All five DF_AnaglyphOut modes (`true`, `grey`, `colour`, `half_colour`,
`optimised`) rendered in parallel for side-by-side comparison. Requires
red/cyan anaglyph glasses to view. The `optimised` mode (Dubois algorithm)
is the recommended choice for production.

📄 [anaglyph_redcyan.md](anaglyph_redcyan.md)

---

## 4. stereo_pair.json

**Nodes:** DF_DepthPrep, DF_StereoPair
**Inputs:** RGB source image + depth map
**Output:** left view, right view, occlusion mask, side-by-side composed

Generates the four-output stereo pair set. The `composed` (SBS) output is
ready for VR headsets and 3D monitors. The occlusion mask shows where the
warp exposes background — useful for inpainting pipelines.

📄 [stereo_pair.md](stereo_pair.md)

---

## 5. hidden_image.json

**Nodes:** DF_PatternGen, DF_HiddenImage ×2
**Inputs:** none (depth generated internally from shape/text)
**Output:** hidden-shape and hidden-text stereograms

Classic magic-eye output where the viewer sees only noise until their eyes
fuse. Two branches: a star shape and the text "HELLO". No external depth map
needed — DF_HiddenImage creates its own binary depth mask from the shape/text
description.

📄 [hidden_image.md](hidden_image.md)

---

## 6. preset_workflow.json

**Nodes:** DF_DepthPrep, DF_PatternGen, DF_Stereogram ×3
**Inputs:** one depth map
**Output:** three stereograms using cinema / shallow / broadcast presets

The fastest way to produce broadcast-quality output. The `preset_name`
widget on DF_Stereogram selects a pre-tuned StereoParams bundle. Three
presets are compared in parallel so you can choose the right depth intensity
for your deliverable.

📄 [preset_workflow.md](preset_workflow.md)

---

## 7. full_pipeline.json

**Nodes:** ALL 10 DepthForge nodes
**Inputs:** RGB source image + depth map
**Output:** SIRDS, anaglyph, stereo pair, hidden image, video frame, QC heatmap

The complete production reference. All safety checks run before any synthesis
node. Use this as the starting template for custom pipelines — bypass or
disconnect branches you don't need.

📄 [full_pipeline.md](full_pipeline.md)

---

## 8. safety_check.json

**Nodes:** DF_DepthPrep, DF_SafetyLimiter ×3, DF_QCOverlay ×2
**Inputs:** one depth map
**Output:** parallax heatmap (before), three safe-depth previews, violation overlay (after)

The diagnostic workflow. Run this on any new depth map before production to
confirm it is within comfort limits. Three SafetyLimiter profiles are applied
in parallel so you can see how aggressively each one clips your depth range.

📄 [safety_check.md](safety_check.md)

---

## All 10 DepthForge Nodes — Quick Reference

| Node | Type | Purpose |
|------|------|---------|
| DF_DepthPrep | Processing | Smooth, dilate, apply falloff curve, clamp depth range |
| DF_PatternGen | Generator | Procedural tile: noise, perlin, plasma, voronoi, mandelbrot, dots, grid |
| DF_PatternLibrary | Generator | 28+ named production patterns from the built-in library |
| DF_Stereogram | Synthesis | Core SIRDS/texture stereogram from depth + pattern |
| DF_AnaglyphOut | Output | Red/cyan anaglyph — 5 modes including Dubois optimised |
| DF_StereoPair | Output | L/R views + occlusion mask + SBS composed |
| DF_HiddenImage | Output | Hidden shape or text in random-dot field |
| DF_SafetyLimiter | Safety | Clamp depth to comfort profiles; returns violation report |
| DF_QCOverlay | Quality | Parallax heatmap, depth bands, violation overlay, histogram |
| DF_VideoSequence | Video | Batch frame processing with temporal smoothing |

---

## Custom Type Reference

These custom types flow between DepthForge nodes and cannot be connected to
standard ComfyUI nodes unless converted:

| Type | Python representation | Direction |
|------|-----------------------|-----------|
| `DEPTH_MAP` | `numpy.ndarray` float32 (H, W) | Between DF processing nodes |
| `PATTERN` | `numpy.ndarray` uint8 (H, W, 4) RGBA | From PatternGen/Library to Stereogram/HiddenImage |
| `STRING` | Python `str` | Text reports — wire to a text display node |

`IMAGE` type is standard ComfyUI `torch.Tensor` (B, H, W, C) float32 — compatible with all standard nodes.

---

*DepthForge v0.5 — Superior Studios / BoxOf LLC*
