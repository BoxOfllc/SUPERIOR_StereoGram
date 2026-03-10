# anaglyph_redcyan.json — Anaglyph Output, All Five Modes

## What This Workflow Demonstrates

Renders the same scene through all five **DF_AnaglyphOut** modes in a single
workflow so you can compare quality side-by-side. Requires red/cyan anaglyph
glasses to view the results. Requires both a source RGB image and a matching
depth map.

---

## DepthForge Nodes Used

| Node | Why it's here |
|------|---------------|
| **DF_DepthPrep** | Shared depth conditioning — one pass feeds all five anaglyph nodes |
| **DF_AnaglyphOut** ×5 | One instance per mode (`true`, `grey`, `colour`, `half_colour`, `optimised`) — all share the same source image and conditioned depth |

---

## Required Inputs

| Input | Format | Notes |
|-------|--------|-------|
| Source image | RGB PNG/EXR, any resolution | The scene you want to make 3D |
| Depth map | Greyscale PNG/EXR, **same resolution** | White = near, Black = far |

Both images must be loaded at the same pixel resolution. Mismatched sizes
will cause an array shape error in `make_anaglyph_from_depth`.

---

## The Five Anaglyph Modes

| Mode | `mode` value | Characteristic |
|------|-------------|----------------|
| True anaglyph | `true` | Minimal ghosting, low colour fidelity; R channel → left, Cyan → right |
| Grey anaglyph | `grey` | Converts to greyscale first — maximum depth perception, no colour |
| Colour anaglyph | `colour` | Preserves full colour, but ghosting is visible at high parallax |
| Half colour | `half_colour` | Left eye greyscale, right eye colour — good compromise for photos |
| Optimised | `optimised` | Dubois matrix optimisation — best overall quality for most scenes |

**Recommendation for production:** start with `optimised`. If ghosting is still
visible after reducing `parallax_px`, try `half_colour`.

---

## Parameter Values and Rationale

### DF_AnaglyphOut (all five nodes)
| Parameter | Value | Reason |
|-----------|-------|--------|
| `depth_factor` | 0.35 | Moderate depth used consistently so modes are directly comparable |
| `parallax_px` | 30 | Moderate pixel separation (~1.5% of 1920px width) — comfortable baseline |
| `swap_eyes` | false | Standard red=left, cyan=right orientation |
| `gamma` | 1.0 | Linear — no gamma compensation applied |

Only the `mode` widget differs between the five nodes.

---

## Expected Output and Verification

Each PreviewImage shows one anaglyph render. **With red/cyan glasses:**
- Near objects (white depth) should appear closer to you
- Far objects (black depth) should recede into the screen

**Without glasses:** you should see colour fringing — a red halo on one side
and cyan on the other side of every object edge. This is correct behaviour.

**Ghosting test:** look at a high-contrast edge in the `colour` mode output.
Visible ghost outlines mean `parallax_px` is too high for that mode — reduce
it, or switch to `optimised`.

---

## Common Issues

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Both images needed but output is black | Source and depth images have different sizes | Resize to match before loading |
| Strong ghosting in all modes | `parallax_px` too high (over 50 for 1920px) | Reduce to 20–30 |
| Depth appears inverted (far pops out) | Depth map white=far convention | Set `invert=true` in DF_DepthPrep |
| Grey anaglyph looks identical to others | Depth map has very low contrast | Use a high-contrast depth pass |
| Eyes hurt after a few seconds | `depth_factor` too high | Reduce to 0.2, re-test |
