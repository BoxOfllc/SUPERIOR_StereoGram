# DF_QRStereogram — QR Code + Stereogram Hybrid

> DepthForge v0.5 · Category: DepthForge

---

## What It Does

`DF_QRStereogram` generates a single image that is simultaneously:

- **Scannable** by any standard QR reader (phone camera, Zxing, etc.)
- **Viewable as a stereogram** — a 3D image emerges when you cross-fuse or
  parallel-fuse the repeating copies

The QR module grid serves as the dot pattern for the SIRDS algorithm.
Depth offsets shift subsequent copies of the QR tile by `depth × max_shift_px`
pixels, creating the stereogram parallax effect.

---

## How It Works

```
url/text → QR code (module grid)
                │
                ▼
         QR tile  ←──── depth map (optional)
                │              │
                ▼              ▼
         SIRDS algorithm  (shift = depth × max_shift_px)
                │
                ▼
    ┌──────┬──────┬──────┬──────┐
    │  QR  │ +d₁  │ +d₂  │ +d₃  │   n_copies = 4
    │ clean│shift │shift │shift │
    └──────┴──────┴──────┴──────┘
      ↑ scan here    ↑ fuse all to see 3D
```

The **leftmost copy is always the unmodified QR code** — point your scanner at it.
The remaining copies carry per-pixel horizontal shifts based on the depth map.
The brain fuses the slight inter-copy shifts into perceived depth.

---

## Parameters

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `url_or_text` | `https://...` | any string | Shorter = smaller QR = more depth headroom |
| `depth_map` | *(radial gradient)* | DEPTH_MAP | White = near, black = far |
| `error_correction` | `H` | L / M / Q / H | H = 30% recoverable — use for depth_factor > 0.10 |
| `module_size` | `10` | 4–32 px | Larger = more shift budget; safe_mode caps at size/4 |
| `border_modules` | `4` | 1–10 | QR spec requires ≥ 4 for reliable scanning |
| `depth_factor` | `0.15` | 0.0–0.40 | Fraction of QR width; clamped by safe_mode |
| `n_copies` | `4` | 2–8 | Minimum 3 for comfortable cross-fusion |
| `safe_mode` | `true` | bool | Caps shifts at module_size/4 to preserve scannability |
| `invert_depth` | `false` | bool | Flip near/far if depth appears inverted |

---

## Outputs

| Output | Type | Description |
|--------|------|-------------|
| `qr_stereogram` | IMAGE | Full stereogram — `n_copies` QR tiles side by side |
| `qr_only` | IMAGE | The clean QR tile on its own (for scan verification) |
| `scan_info` | STRING | QR version, module count, shift stats, scanning tips |

---

## The Fundamental Trade-off

QR scanning and stereogram depth are **in tension**:

| Goal | Prefers |
|------|---------|
| Reliable QR scan | Small depth shifts (< module_size / 4) |
| Strong 3D depth | Large depth shifts (> module_size / 2) |

`safe_mode=True` sits on the conservative side.
`error_correction=H` gives 30% damage recovery, widening the safe window.

### Practical shift budget

```
With module_size=10, QR v3 (290 px wide), error_correction=H:

  safe_mode ON:   max_shift = 2 px  → subtle but scannable
  safe_mode OFF:  max_shift = 43 px → strong depth, test scan rate

Rule of thumb: max_shift_px ≤ module_size / 4  →  very reliable
               max_shift_px ≤ module_size / 2  →  usually reliable with H
               max_shift_px > module_size       →  scan may fail
```

---

## Tips for Best Results

**For strongest scannability:**
- Keep `safe_mode=True`
- Use `error_correction=H`
- Use a short URL (< 30 chars) to get a lower QR version and fewer modules
- Use `module_size=12` or higher — more shift budget per module
- Test with multiple QR readers (Google Lens, iOS camera, ZXing)

**For strongest 3D depth:**
- Disable `safe_mode` and test scan rate manually
- Set `error_correction=H`, `module_size=14–16`
- Use a simple, high-contrast depth map (clear foreground object)
- Set `n_copies=5` or more for easier cross-fusion

**For the cleanest 3D shape:**
- Connect a conditioned depth map from `DF_DepthPrep`
- Use `bilateral_space=5`, `dilation_px=3` in DF_DepthPrep for smooth edges
- Avoid depth maps with sharp horizontal edges — these cause visible seams
- A centred sphere, face, or logo works particularly well

---

## Viewing the Stereogram

**Cross-fusion (cross-eyed):**
1. Hold the image at comfortable reading distance (~50 cm)
2. Focus on a point halfway between you and the screen
3. Slowly cross your eyes until the `n_copies` copies merge into `n_copies - 1`
4. Hold that focus — a 3D shape will pop out

**Parallel-fusion (wall-eyed / magic-eye style):**
1. Hold close (~15 cm) and let your eyes go unfocused (like looking through it)
2. Slowly pull the image away until patterns merge
3. The depth emerges as your focus locks

---

## Dependency

```bash
pip install qrcode[pil]
```

The node raises a clear `ImportError` with this install command if the
package is missing.

---

## Workflow File

Load `qr_stereogram.json` in ComfyUI for a ready-to-run setup with:
- `DF_QRStereogram` (standalone, no depth map connected by default)
- Two `PreviewImage` nodes (stereogram + clean QR reference)
- Optional `LoadImage → DF_DepthPrep` branch for custom depth shapes
- Annotated notes at every step

---

*DepthForge · Superior Studios / BoxOf LLC*
