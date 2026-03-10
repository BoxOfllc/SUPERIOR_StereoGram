# CLAUDE.md — SUPERIOR StereoGram / DepthForge

> Part of Superior Studios / BoxOf LLC — Skunkworks Division
> GitHub: https://github.com/BoxOfllc/SUPERIOR_StereoGram

---

## 1. What This Project Is

**DepthForge** is a production-ready stereogram synthesis engine branded as
SUPERIOR StereoGram. It converts depth maps into every major stereoscopic
image format across five integration surfaces sharing a single Python core.

**Current version:** v0.5.0 — Phases 1–5 complete. 396 tests. Security audited.

---

## 2. Architecture

```
depthforge/
  core/           Pure Python synthesis engine (14 modules — NumPy + Pillow only)
  comfyui/        10 ComfyUI custom nodes
  nuke/           Python gizmo + toolbar menu
  ofx/            C++ OFX plugin (cmake build — AE, DaVinci, Premiere)
  web/            Flask local preview server (localhost:5000)
  depth_models/   MiDaS + ZoeDepth AI depth estimators
  cli/            9-command Click CLI
  io/             EXR / TIFF / PNG / JPEG I/O
  tests/          396 tests across 5 phase files
```

---

## 3. Critical Parameter Names — DO NOT CHANGE

These are easy to get wrong. Always use exactly these names:

| Class | Correct Parameter | Wrong (do not use) |
|-------|------------------|--------------------|
| `DepthPrepParams` | `bilateral_sigma_space` | — |
| `DepthPrepParams` | `bilateral_sigma_color` | — |
| `StereoParams` | `max_parallax_fraction` | `max_parallax` |
| `AnaglyphParams` | `parallax_px` | `eye_separation` |
| `StereoPairParams` | `background_fill` | `bg_fill` |

Other critical implementation facts:
- `make_stereo_pair()` returns occlusion mask as **float32** — use `occ > 0.5`
- `FalloffCurve` serialises as `.value` (int) for JSON
- `_tensor_to_numpy` squeezes leading batch dim: `(1,H,W,C)` → `(H,W,C)`
- Flat depth at 0.5 produces ~3° vergence — this is physically correct

---

## 4. Dependency Tiers — Never Break the Fallbacks

All optional packages must remain optional. Always guard with flags:

```python
from depthforge import HAS_CV2, HAS_SCIPY, HAS_TORCH

if HAS_CV2:
    # Better OpenCV implementation
else:
    # Pure NumPy fallback — MUST always exist
```

Never add a feature that **requires** an optional package without a NumPy fallback.

---

## 5. Output Modes

| Mode | Description |
|------|-------------|
| SIRDS | Classic random-dot "Magic Eye" stereogram |
| Texture Pattern | Tileable image as pattern layer |
| Hidden Image | Shape/text encoded invisibly in dot noise |
| Anaglyph | Red/cyan (and variants) for glasses-based 3D |
| Stereo Pair | Left/right side-by-side or top/bottom |
| Video Sequence | Temporally stable stereogram sequences |

---

## 6. ComfyUI Integration

10 custom nodes load under the **DepthForge** category in the node browser.

**Install path:** `ComfyUI/custom_nodes/DepthForge/` (junction-linked to this repo)
**ComfyUI location:** `E:\ComfyUI\ComfyUI_windows_portable_nvidia\ComfyUI\`
**Launch:** `E:\ComfyUI\ComfyUI_windows_portable_nvidia\run_nvidia_gpu.bat`

Depth map convention: **white = near (close to camera), black = far (background)**

---

## 7. Running Tests

```bash
# Full suite (396 tests)
python -m unittest depthforge.tests.test_phase1 \
                   depthforge.tests.test_phase2 \
                   depthforge.tests.test_phase3 \
                   depthforge.tests.test_phase4 \
                   depthforge.tests.test_phase5

# Expected: Ran 396 tests — OK
# Known: 2 Windows-only temp file cleanup errors in phase3/phase5 — not real bugs
```

---

## 8. Development Rules

- **Do NOT modify** revenue splits, ticket types, or payout logic (BLOXoffice rule — does not apply here but kept for org consistency)
- **Do NOT push to git** without explicit instruction — owner reviews first
- **Do NOT break NumPy fallbacks** — all cv2/scipy/torch paths must have fallbacks
- **TypeScript rules do not apply** — this is Python only
- Run full test suite after any change to `depthforge/core/`
- All blockchain/NFT work is on hold (BLOXoffice parent org rule — N/A here)

---

## 9. Local Dev Environment

```
Repo:     E:\SupStudiosWeb3\ProductionTools\SUPERIOR_StereoGram_v0.5\
Venv:     .venv\ (activate with .venv\Scripts\Activate.ps1)
Python:   3.10 (system) / 3.11 (ComfyUI embedded)
GitHub:   https://github.com/BoxOfllc/SUPERIOR_StereoGram
```

**Every session — activate venv first:**
```powershell
cd E:\SupStudiosWeb3\ProductionTools\SUPERIOR_StereoGram_v0.5
.venv\Scripts\Activate.ps1
```

---

## 10. Known Issues

| Issue | Status |
|-------|--------|
| 2 Windows temp file errors in test_phase3 / test_phase5 | Known OS quirk, not a code bug |
| GitHub CI lint failures | Needs ruff/black formatting pass |
| `setuptools.backends.legacy` on Python 3.10 | Fixed in pyproject.toml — use `setuptools.build_meta` |

---

## 11. Phase History & Next Phase

| Phase | Scope | Status |
|-------|-------|--------|
| 1 | Core engine — 8 synthesis modules | ✅ Complete |
| 2 | Depth tools — presets, safety, QC overlays | ✅ Complete |
| 3 | CLI + AI depth + full pattern library (31 patterns) | ✅ Complete |
| 4 | ComfyUI + Nuke + Web preview | ✅ Complete |
| 5 | OFX plugin + GPU + video | ✅ Complete |
| 6 | CUDA kernels, WebGL renderer, OCIO, AE CEP | ⏳ Planned |

**Suggested Phase 6:**
- GPU-accelerated SIRDS via CUDA kernels (target 10× at 4K)
- Real-time WebGL stereogram renderer in web preview
- OpenColorIO colour pipeline integration
- After Effects CEP extension wrapping OFX bridge
- Depth-from-stereo pair (inverse stereogram estimation)

---

## 12. Key File Reference

| File/Folder | Purpose |
|-------------|---------|
| `depthforge/core/synthesizer.py` | Main synthesis engine — StereoParams lives here |
| `depthforge/core/depth_prep.py` | Depth map conditioning — DepthPrepParams |
| `depthforge/core/pattern_gen.py` | 31 procedural pattern generators |
| `depthforge/core/presets.py` | 6 named production presets |
| `depthforge/core/comfort.py` | ComfortAnalyzer + SafetyLimiter |
| `depthforge/core/qc.py` | QC overlays — heatmap, histogram, comparison grid |
| `depthforge/comfyui/nodes.py` | All 10 ComfyUI node definitions |
| `depthforge/nuke/gizmo.py` | Nuke Python gizmo |
| `depthforge/web/app.py` | Flask preview server |
| `depthforge/tests/test_phase*.py` | 396 tests across 5 files |
| `pyproject.toml` | Build config — use `setuptools.build_meta` |

---

*Maintained by: Superior Studios / BoxOf LLC*
*Last updated: March 2026*
