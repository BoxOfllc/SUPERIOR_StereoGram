# full_pipeline.json — Complete Production Pipeline (All 10 Nodes)

## What This Workflow Demonstrates

The complete DepthForge production pipeline with all ten custom nodes active
simultaneously. Every output type is generated from a single depth map and
source image pair, with safety limiting and QC overlays applied before any
synthesis node receives the depth data.

**All 10 DepthForge nodes:**
DF_DepthPrep → DF_SafetyLimiter → DF_QCOverlay → DF_PatternGen +
DF_PatternLibrary → DF_Stereogram + DF_AnaglyphOut + DF_StereoPair +
DF_HiddenImage + DF_VideoSequence

---

## Pipeline Stages

### Stage 1 — Load
Two `LoadImage` nodes: one for the RGB source image, one for the depth map.
Both must be the same pixel resolution.

### Stage 2 — Depth Prep
**DF_DepthPrep** conditions the raw depth map with S-curve falloff and
bilateral smoothing. Output is a `DEPTH_MAP` type (numpy float32).

### Stage 3 — Safety & QC
**DF_SafetyLimiter** (`standard` profile) clips extreme depth values before
any synthesis node sees them. Its `safe_depth` output feeds all five
synthesis nodes. Its `depth_preview` (IMAGE type) feeds DF_VideoSequence
since that node requires IMAGE input.

**DF_QCOverlay** (`parallax_heatmap`) shows the parallax distribution of
the safe depth map. `comfort_summary` (STRING) can be wired to a text
display node to read the analyzer report.

### Stage 4 — Pattern Sources
**DF_PatternGen** (`random_noise`) feeds DF_Stereogram and DF_VideoSequence.
**DF_PatternLibrary** (`crystalline`) feeds DF_HiddenImage — a more
structured pattern suits the hidden-image use case.

### Stage 5 — Output Synthesis
| Node | Output |
|------|--------|
| DF_Stereogram | Classic SIRDS stereogram |
| DF_AnaglyphOut | Optimised red/cyan anaglyph |
| DF_StereoPair | Side-by-side composed stereo pair |
| DF_HiddenImage | Hidden diamond shape |
| DF_VideoSequence | Single-frame stereogram (demonstrates the temporal API) |

---

## Required Inputs

| Input | Format | Notes |
|-------|--------|-------|
| Source image | RGB PNG/EXR | Scene for anaglyph and stereo pair |
| Depth map | Greyscale PNG/EXR, same size | White = near, Black = far |

---

## Parameter Values and Rationale

| Node | Key setting | Value | Reason |
|------|-------------|-------|--------|
| DF_DepthPrep | `falloff_curve` | `s_curve` | Cinematic depth roll-off |
| DF_SafetyLimiter | `profile` | `standard` | Balanced safety for general output |
| DF_QCOverlay | `overlay_type` | `parallax_heatmap` | Shows parallax distribution across the frame |
| DF_Stereogram | `depth_factor` | 0.35 | Standard comfortable depth |
| DF_AnaglyphOut | `mode` | `optimised` | Best anaglyph quality |
| DF_StereoPair | `layout` | `side_by_side` | VR/3D monitor delivery format |
| DF_HiddenImage | `shape` | `diamond` | Distinctive shape, easy to verify after fusing |
| DF_VideoSequence | `temporal_smooth` | 0.3 | Blends 30% of prior frame depth — stabilises flickering |

---

## Expected Output and Verification

Six PreviewImage nodes show:
1. **SIRDS Output** — random-dot stereogram with depth encoded
2. **QC Heatmap** — colour-coded parallax map (warm = near, cool = far)
3. **Anaglyph Output** — red/cyan 3D image (view with glasses)
4. **Stereo Pair (Composed)** — double-width SBS image
5. **Hidden Image Output** — random noise hiding a diamond
6. **Video Sequence Frame** — single stereogram frame processed via temporal API

---

## Common Issues

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Some branches produce black output | Source or depth image not loaded | Confirm both LoadImage nodes have valid files |
| QC heatmap is all one colour | Depth map has no dynamic range | Use a depth map with full 0–1 range |
| VideoSequence output is identical to SIRDS | Single frame input — expected | The temporal API is designed for multi-frame batch input |
| DF_HiddenImage uses wrong size | HiddenImage ignores depth map size; uses its own width/height widgets | Set `width`/`height` widgets to match your target resolution |
| STRING outputs invisible | ComfyUI has no built-in text display | Wire to a `ShowText` node from ComfyUI-Manager if available |
