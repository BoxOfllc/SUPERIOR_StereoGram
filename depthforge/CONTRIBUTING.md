# Contributing to DepthForge

Thank you for your interest in contributing! This guide covers everything you need to get started.

---

## Development Setup

```bash
git clone https://github.com/depthforge/depthforge.git
cd depthforge

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
.venv\Scripts\activate          # Windows

# Install in editable mode with dev extras
pip install -e ".[cv2,scipy,dev]"

# Verify setup
python -m pytest tests/ -v
python tests/visual_gallery.py
```

---

## Project Structure

```
depthforge/core/     ← all algorithm code lives here
depthforge/tests/    ← tests and visual gallery
docs/                ← documentation
.github/             ← CI and templates
```

The `core/` package has zero UI dependencies — it only imports NumPy and Pillow (with graceful tier-1/2 fallbacks). Keep it that way.

---

## Code Style

- **Line length:** 100 characters
- **Formatter:** `black --line-length 100`
- **Linter:** `ruff check`
- **Docstrings:** NumPy-style for all public functions
- **Type hints:** on all public function signatures
- **Dataclasses:** all parameter sets must be dataclasses with full docstrings

Run before committing:
```bash
black depthforge/ --line-length 100
ruff check depthforge/
```

---

## The Dependency Tier Rule

**Every public function must work with NumPy + Pillow alone.**

Optional packages (OpenCV, SciPy, PyTorch) may improve quality or speed, but must never be required. Always provide a fallback:

```python
try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False

def my_operation(arr):
    if _CV2:
        return _fast_cv2_version(arr)
    return _numpy_fallback(arr)
```

Test both paths. CI runs once with and once without the optional packages.

---

## Writing Tests

All tests go in `tests/test_phase1.py` (or a new `test_phase2.py` etc. for new phases).

Tests must be compatible with both `pytest` and `unittest`:

```python
import unittest

class TestMyFeature(unittest.TestCase):
    def test_output_shape(self):
        result = my_function(...)
        self.assertEqual(result.shape, (64, 64, 4))
```

**Required tests for any new public function:**
1. Output shape and dtype
2. Valid value range
3. Parameter boundary conditions
4. Error handling (`assertRaises`)
5. Tier-0 fallback (works without optional deps)
6. Seed reproducibility (if random)

**Run tests:**
```bash
python -m pytest tests/ -v                      # with pytest
python -m unittest discover tests/             # without pytest
```

---

## Visual Gallery

If your change affects visual output, add a section to `tests/visual_gallery.py`:

```python
print("[N/N] My new feature...")
outputs = []
for variant_name, params in my_variants:
    result = my_function(depth, params)
    outputs.append(label_image(result, variant_name))
mosaic = make_mosaic(outputs, cols=3)
save(f"0N_my_feature", mosaic)
print(f"   ✓ {len(outputs)} variants")
```

---

## Adding a New Pattern Generator

1. Add the type to `PatternType` enum in `pattern_gen.py`
2. Add the generator function `_my_pattern(params, rng) -> np.ndarray` (float32, [0,1])
3. Add the case to the `generate_pattern()` dispatch
4. Add tests in `TestPatternGen`
5. Add a gallery entry in `visual_gallery.py` section 1

The generator must:
- Accept `PatternParams` and `np.random.Generator`
- Return float32 greyscale `(H, W)` — colouration is handled by `_colourise()`
- Be seamlessly tileable
- Work in pure NumPy

---

## Adding a New Depth Falloff Curve

1. Add the type to `FalloffCurve` enum
2. Add the case to `_apply_curve()` in `depth_prep.py`
3. Add a test verifying f(0)=0, f(1)=1, and non-linearity
4. Add to the `API_REFERENCE.md` table

---

## Documentation

- **Docstrings** — all public functions, classes, and fields
- **`docs/USER_GUIDE.md`** — update for user-visible changes
- **`docs/API_REFERENCE.md`** — update for any public API changes
- **`CHANGELOG.md`** — add entry under `[Unreleased]`

---

## Pull Request Process

1. Fork the repo and create a branch: `git checkout -b feature/my-thing`
2. Make your changes
3. Run `black`, `ruff`, and `pytest` — all must pass
4. Run `python tests/visual_gallery.py` — must complete without errors
5. Update `CHANGELOG.md`
6. Open a PR using the PR template

PRs are merged to `develop` and released to `main` in batches.

---

## Reporting Bugs

Use the [Bug Report](https://github.com/depthforge/depthforge/issues/new?template=bug_report.md) template. Always include the diagnostic output from the top of [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

---

## Questions?

Open a [Discussion](https://github.com/depthforge/depthforge/discussions) — not an Issue — for questions, ideas, and general chat.
