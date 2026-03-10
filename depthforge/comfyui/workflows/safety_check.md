# safety_check.json — Comfort Analyzer & Safety Limiter

## What This Workflow Demonstrates

Tests the two safety-focused DepthForge nodes: **DF_SafetyLimiter** and
**DF_QCOverlay**. Three SafetyLimiter instances run in parallel with
different profiles (conservative, standard, cinema), and QC overlays show
the parallax distribution before and after limiting. A `violation_overlay`
QC mode is applied to the standard-profile output to highlight any pixels
that still exceed the safe zone.

---

## DepthForge Nodes Used

| Node | Why it's here |
|------|---------------|
| **DF_DepthPrep** | Shared upstream conditioning; one conditioned depth map fans out to all four downstream nodes |
| **DF_QCOverlay** ×2 | One shows the raw (pre-safety) parallax heatmap; one shows violation overlay after standard-profile limiting |
| **DF_SafetyLimiter** ×3 | Conservative, standard, and cinema profiles in parallel — depth preview output from each is previewed to compare clamping intensity |

---

## Required Inputs

| Input | Format | Notes |
|-------|--------|-------|
| Depth map | Greyscale PNG/EXR, any resolution | White = near, Black = far |

**Best test depth map:** use a high-contrast image with very bright white
regions (near 1.0) AND very dark black regions (near 0.0). Flat or low-
contrast depth maps will not trigger any violations and the test is less
informative.

---

## The Three Safety Profiles

| Profile | `max_depth_factor` | `max_gradient` | `near_clip` | `far_clip` | Best for |
|---------|--------------------|----------------|-------------|------------|---------|
| `conservative` | 0.4 | 0.08 | 0.05 | 0.95 | Migraine-sensitive viewers; accessibility |
| `standard` | 0.6 | 0.15 | 0.02 | 0.98 | General audiences; balanced quality/safety |
| `cinema` | 0.7 | 0.20 | 0.01 | 0.99 | Dark-room viewing; allows dramatic depth |

The parameters are passed to `SafetyLimiterParams.from_profile()` internally.
Manual overrides in the widget are applied on top of the profile defaults.

---

## The Two QC Overlay Modes Used

| Node | `overlay_type` | What it shows |
|------|---------------|---------------|
| QC Before Safety | `parallax_heatmap` | Colour-coded parallax in pixels across the frame; warm=high parallax, cool=low |
| QC After Safety | `violation_overlay` | Red highlights = pixels that exceed the safe parallax zone after limiting |

The `comfort_summary` STRING output from each QC node contains a plain-text
summary from `ComfortAnalyzer.analyze()` — wire to a text display node to
read it, or check the Python console for printed values.

---

## Parameter Values and Rationale

### DF_QCOverlay
| Parameter | Value | Reason |
|-----------|-------|--------|
| `frame_width` | 1920 | Parallax calculated assuming 1920px wide output |
| `depth_factor` | 0.35 | Matches the depth_factor you'd use in synthesis |
| `colormap` | `inferno` | High-contrast colourmap; easy to spot hot (dangerous) regions |
| `annotate` | true | Adds pixel-value labels to the heatmap |

---

## Expected Output and Verification

1. **Heatmap BEFORE Safety** — shows raw parallax. Bright yellow/white regions
   indicate dangerous near-field depth values.
2. **Conservative Safe Depth** — should look noticeably flatter (greyer) than
   the original depth map; extreme bright/dark values clipped hard.
3. **Standard Safe Depth** — moderate clipping; subtle difference from original.
4. **Cinema Safe Depth** — minimal clipping; closely resembles the original.
5. **Violation Overlay AFTER Safety** — ideally should show very few or no red
   pixels after standard limiting. Any remaining red indicates values that
   slipped through and may still cause discomfort.

**What "clean" looks like:** after standard limiting, the violation overlay
should be mostly or entirely blue/black (safe). If you still see large red
regions, the source depth map has extreme values that even standard limiting
can't bring into range — consider using `conservative` profile.

---

## Common Issues

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| All three safe-depth previews look identical to source | Depth map has no values outside safe range | Use a more extreme depth map for testing |
| `violation_report` output is empty / "✓ No violations" | Depth is already within limits | Expected — not a bug |
| Heatmap is all one colour | `frame_width` doesn't match actual depth map width | Adjust `frame_width` widget to match your depth map |
| Conservative preview is much darker than expected | Aggressive near-clip (0.05) blacks out near values | Expected behaviour for conservative profile |
| VergenceProfile import error | comfort.py missing VergenceProfile | Run full test suite; if Phase 2 fails, check comfort.py |
