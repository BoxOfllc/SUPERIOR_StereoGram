# Contributing to DepthForge

Thank you for your interest in contributing! DepthForge is an active project and contributions of all kinds are welcome.

## Ways to Contribute

- **Bug reports** — open a GitHub issue with a minimal reproduction
- **Bug fixes** — fork, fix, open a PR
- **New features** — discuss in an issue first; check the roadmap phase
- **Documentation** — typos, clarity, missing examples
- **Pattern library** — new procedural pattern generators or sample tiles
- **Tests** — improving test coverage or edge cases

---

## Development Setup

```bash
git clone https://github.com/your-org/depthforge.git
cd depthforge
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Run the test suite:

```bash
python -m pytest tests/ -v
```

With optional deps:

```bash
pip install opencv-python scipy
python -m pytest tests/ -v
```

---

## Code Style

- **Formatter**: `black` — line length 100
- **Linter**: `ruff`
- **Type hints**: encouraged but not required for all internal functions

Run before committing:

```bash
black depthforge/
ruff check depthforge/
```

---

## Dependency Tiers

DepthForge uses a layered dependency model. All core functionality must work with **NumPy + Pillow only**. Better implementations using OpenCV, SciPy, or PyTorch should be behind `HAS_CV2`, `HAS_SCIPY`, `HAS_TORCH` flags:

```python
from depthforge import HAS_CV2

if HAS_CV2:
    # Better implementation using OpenCV
    result = cv2.bilateralFilter(...)
else:
    # Pure NumPy fallback
    result = numpy_box_blur(...)
```

Never make a new feature that **requires** an optional package without first implementing a NumPy fallback.

---

## Adding a New Pattern Generator

1. Add a new entry to `PatternType` enum in `pattern_gen.py`
2. Implement `_generate_<name>(params) -> np.ndarray` returning RGBA uint8 `(H, W, 4)`
3. Add the dispatch case in `generate_pattern()`
4. Write a test in `tests/test_phase1.py` class `TestPatternGen`
5. Add to the visual gallery in `tests/visual_gallery.py`

---

## PR Checklist

- [ ] Tests pass locally (`python -m pytest tests/ -v`)
- [ ] Black and ruff pass (`black --check . && ruff check .`)
- [ ] New functionality has at least one test
- [ ] Docstrings updated for public functions
- [ ] No new hard dependencies on optional packages without fallbacks
- [ ] CHANGELOG.md updated under `[Unreleased]`

---

## Roadmap Phases

See [README.md](README.md#project-status) for the current phase status. If you want to implement a Phase 3–5 feature early, open an issue to discuss API design first so it fits cleanly into the existing architecture.
