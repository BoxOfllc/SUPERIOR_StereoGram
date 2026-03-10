# preset_workflow.json — Named Production Presets

## What This Workflow Demonstrates

Compares three named production presets (`cinema`, `shallow`, `broadcast`)
by running three **DF_Stereogram** nodes in parallel on the same depth map
and pattern. Only the `preset_name` widget differs between the three nodes —
all other widget values are overridden by the preset at runtime.

---

## DepthForge Nodes Used

| Node | Why it's here |
|------|---------------|
| **DF_DepthPrep** | Shared depth conditioning — one conditioned depth map fans out to all three stereogram nodes |
| **DF_PatternGen** | One shared pattern tile fans out to all three nodes — isolates the preset as the only variable |
| **DF_Stereogram** ×3 | One per preset; `preset_name` drives the `StereoParams` via `get_preset()` internally |

---

## Required Inputs

| Input | Format | Notes |
|-------|--------|-------|
| Depth map | Greyscale PNG/EXR, any resolution | White = near, Black = far |

---

## The Three Presets

| Preset | Approx `depth_factor` | Use case |
|--------|----------------------|----------|
| `cinema` | ~0.5 | Dramatic depth for dark-room cinema viewing; not recommended for long desktop sessions |
| `shallow` | ~0.15 | Gentle, universally comfortable; ideal for accessibility use or first-time viewers |
| `broadcast` | ~0.25 | TV-safe parallax; controlled to avoid flicker on interlaced broadcast displays |

> **Other available presets** (not wired in this workflow but available in
> DF_Stereogram): `medium`, `deep`, `print`. Change `preset_name` to explore.

---

## Parameter Values and Rationale

### DF_DepthPrep
| Parameter | Value | Reason |
|-----------|-------|--------|
| `falloff_curve` | `linear` | Neutral — lets the preset's own depth_factor drive the apparent depth without additional curve shaping |

### DF_PatternGen
| Parameter | Value | Reason |
|-----------|-------|--------|
| `seed` | 42 | Same seed for all three — only depth changes, not dot pattern |

### DF_Stereogram nodes
The `depth_factor`, `max_parallax`, `oversample`, and `safe_mode` widget
values shown in the node are **ignored** when `preset_name` is not `"none"`.
The preset sets all StereoParams internally via `get_preset(preset_name)`.
Set `preset_name` back to `"none"` to use the manual sliders.

---

## Expected Output and Verification

Compare the three PreviewImage outputs side by side:

- **Cinema** — strongest 3D effect; dots shift the most horizontally
- **Shallow** — subtlest effect; dots shift barely; easiest to fuse
- **Broadcast** — intermediate, controlled pop

**Numerical check:** at the same image width, parallax increases with
`depth_factor`. You should be able to see the horizontal dot displacement
increasing from shallow → broadcast → cinema when zoomed in on a bright
(near) region.

---

## Common Issues

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| All three outputs look identical | `preset_name` not being applied | Confirm `preset_name` widget is set, not `"none"` |
| Cinema preset causes eye strain | Depth too aggressive for your viewing distance | Switch to `broadcast` or `shallow` for sustained viewing |
| Presets not found | `depthforge.core.presets` module missing | Run full test suite to verify Phase 2 modules are intact |
| Want to mix preset and manual overrides | Not possible — preset overrides all StereoParams | Use `"none"` and set manual values |
