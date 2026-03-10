# hidden_image.json — Hidden Shape / Text Stereogram

## What This Workflow Demonstrates

Creates **hidden-image** (autostereogram) output where an invisible shape or
text message is encoded into a random-dot field. The viewer sees nothing but
random dots until their eyes fuse — then a shape floats in front of the
background. No external depth map is required: DF_HiddenImage generates its
own depth mask from the shape/text description.

Two branches run in parallel:
- **Star shape** (left branch) — geometric primitive
- **Text "HELLO"** (right branch) — font-rendered mask

---

## DepthForge Nodes Used

| Node | Why it's here |
|------|---------------|
| **DF_PatternGen** | Generates a single `random_noise` tile that is shared by both hidden-image nodes — ensures both outputs have the same visual noise baseline |
| **DF_HiddenImage** ×2 | One for each demonstration; one uses `shape=star`, the other `text="HELLO"` |

---

## Required Inputs

No external depth map needed — DF_HiddenImage synthesizes its own depth
from the shape/text parameters.

| Parameter | What you control |
|-----------|-----------------|
| `width` / `height` | Output stereogram dimensions in pixels |
| `shape` | Geometric primitive: `circle`, `square`, `triangle`, `star`, `diamond`, `arrow` |
| `text` | Any short string — rendered with Pillow's default font at `font_size` |
| `foreground_depth` | Apparent depth of the encoded shape (0–1, where 1=closest) |
| `background_depth` | Apparent depth of the background plane (0–1, where 0=farthest) |

---

## Parameter Values and Rationale

### DF_PatternGen
| Parameter | Value | Reason |
|-----------|-------|--------|
| `pattern_type` | `random_noise` | Classic SIRDS dot noise — maximises camouflage of the hidden shape |
| `color_mode` | `greyscale` | Simpler dots are easier to fuse and less visually distracting |

### DF_HiddenImage — Star branch
| Parameter | Value | Reason |
|-----------|-------|--------|
| `shape` | `star` | Clear 5-point shape — easy to verify once fused |
| `foreground_depth` | 0.75 | Shape floats noticeably in front of background |
| `background_depth` | 0.10 | Background recedes well behind the screen plane |
| `edge_soften_px` | 4 | Blurs the shape mask edges to avoid ringing at depth discontinuities |

### DF_HiddenImage — Text branch
| Parameter | Value | Reason |
|-----------|-------|--------|
| `text` | `"HELLO"` | Short, wide text readable at stereogram scale |
| `font_size` | 180 | Large enough to be legible once fused at 1920px wide |
| `foreground_depth` | 0.75 | Same as star for consistent comparison |
| `background_depth` | 0.10 | Same as star |
| `edge_soften_px` | 4 | Smooths font stroke edges |

---

## Expected Output and Verification

- **Hidden Star Stereogram** — looks like random noise; after fusing with
  relaxed eyes a 5-point star should emerge floating in front of the background
- **Star Depth Map Preview** — shows the binary mask used to encode the shape;
  white = foreground, black = background
- **Hidden Text Stereogram (HELLO)** — as above but the word "HELLO" emerges

**Fusing tip:** this type of stereogram is easier to fuse with the
*wall-eyed* (parallel) technique — hold the image at about 50 cm and look
through it as if gazing at something far behind the screen.

---

## Common Issues

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Text is too small to read when fused | Font size too small | Increase `font_size` to 220–300 for 1920px wide output |
| Shape has visible ringing halo | `edge_soften_px` too low | Increase to 6–8 |
| Nothing visible at all after fusing | `foreground_depth` too close to `background_depth` | Increase the gap — try fg=0.8, bg=0.05 |
| Output is much smaller than expected | `width`/`height` widgets ignored | Confirm connection to pattern is PATTERN type, not IMAGE |
| Shape is inverted (background floats) | `foreground_depth` < `background_depth` | Swap the values |
