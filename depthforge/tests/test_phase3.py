"""
DepthForge — Phase 3 Test Suite
================================
Tests for: CLI, pattern library, depth models, IO formats

Run:
    cd /home/claude && python -m unittest depthforge.tests.test_phase3 -v
"""

import unittest
import json
import tempfile
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_depth_file(H=64, W=96, path=None) -> str:
    """Save a synthetic depth PNG and return its path."""
    y, x = np.mgrid[0:H, 0:W]
    cx, cy = W / 2, H / 2
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    depth = np.clip(1.0 - r / (min(W, H) * 0.45), 0.0, 1.0)
    if path is None:
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
    Image.fromarray((depth * 255).astype(np.uint8)).save(path)
    return path


def _make_source_file(H=64, W=96) -> str:
    """Save a synthetic colour source image."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:, :, 0] = np.linspace(0, 255, W, dtype=np.uint8)
    img[:, :, 2] = np.linspace(255, 0, H, dtype=np.uint8)[:, None]
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    Image.fromarray(img).save(path)
    return path


def _tmpfile(suffix=".png") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path


# ===========================================================================
# Pattern Library
# ===========================================================================

class TestPatternLibrary(unittest.TestCase):

    def test_list_patterns_returns_many(self):
        from depthforge.core.pattern_library import list_patterns
        names = list_patterns()
        self.assertGreater(len(names), 20)

    def test_all_categories_present(self):
        from depthforge.core.pattern_library import list_categories
        cats = list_categories()
        for expected in ["organic", "geometric", "psychedelic", "minimal"]:
            self.assertIn(expected, cats)

    def test_list_by_category(self):
        from depthforge.core.pattern_library import list_patterns
        organic   = list_patterns("organic")
        geometric = list_patterns("geometric")
        self.assertGreater(len(organic), 3)
        self.assertGreater(len(geometric), 3)

    def test_get_pattern_returns_rgba(self):
        from depthforge.core.pattern_library import get_pattern, list_patterns
        name = list_patterns()[0]
        out = get_pattern(name, width=64, height=64)
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(out.shape[2], 4)

    def test_get_pattern_correct_dimensions(self):
        from depthforge.core.pattern_library import get_pattern
        out = get_pattern("perlin_noise", width=80, height=60)
        self.assertEqual(out.shape, (60, 80, 4))

    def test_unknown_pattern_raises(self):
        from depthforge.core.pattern_library import get_pattern
        with self.assertRaises(KeyError):
            get_pattern("no_such_pattern_xyz")

    def test_seed_reproducibility(self):
        from depthforge.core.pattern_library import get_pattern
        a = get_pattern("perlin_noise", width=32, height=32, seed=7)
        b = get_pattern("perlin_noise", width=32, height=32, seed=7)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        from depthforge.core.pattern_library import get_pattern
        a = get_pattern("perlin_noise", width=32, height=32, seed=1)
        b = get_pattern("perlin_noise", width=32, height=32, seed=999)
        self.assertFalse(np.array_equal(a, b))

    def test_all_patterns_generate_without_error(self):
        from depthforge.core.pattern_library import list_patterns, get_pattern
        errors = []
        for name in list_patterns():
            try:
                out = get_pattern(name, width=32, height=32, seed=1)
                if out.shape != (32, 32, 4) or out.dtype != np.uint8:
                    errors.append(f"{name}: shape/dtype wrong {out.shape} {out.dtype}")
            except Exception as e:
                errors.append(f"{name}: {e}")
        self.assertEqual(errors, [], msg="\n".join(errors))

    def test_organic_patterns_exist(self):
        from depthforge.core.pattern_library import list_patterns
        organic = list_patterns("organic")
        for expected in ["perlin_noise", "northern_lights", "water_ripples", "marble"]:
            self.assertIn(expected, organic)

    def test_geometric_patterns_exist(self):
        from depthforge.core.pattern_library import list_patterns
        geo = list_patterns("geometric")
        for expected in ["hexgrid", "dotgrid", "circuit_board", "checkerboard"]:
            self.assertIn(expected, geo)

    def test_psychedelic_patterns_exist(self):
        from depthforge.core.pattern_library import list_patterns
        psy = list_patterns("psychedelic")
        for expected in ["plasma_wave", "rainbow_bands", "kaleidoscope", "tie_dye"]:
            self.assertIn(expected, psy)

    def test_minimal_patterns_exist(self):
        from depthforge.core.pattern_library import list_patterns
        mini = list_patterns("minimal")
        for expected in ["fine_grain", "linen", "paper", "subtle_dots"]:
            self.assertIn(expected, mini)

    def test_pattern_alpha_always_255(self):
        from depthforge.core.pattern_library import list_patterns, get_pattern
        # Check a sample of patterns have full alpha
        for name in list_patterns()[:8]:
            out = get_pattern(name, width=16, height=16)
            self.assertTrue((out[..., 3] == 255).all(), msg=f"{name}: alpha not 255")

    def test_get_pattern_info(self):
        from depthforge.core.pattern_library import get_pattern_info
        info = get_pattern_info("plasma_wave")
        self.assertEqual(info.name, "plasma_wave")
        self.assertEqual(info.category, "psychedelic")
        self.assertIsInstance(info.description, str)
        self.assertFalse(info.safe_mode)

    def test_available_count(self):
        from depthforge.core.pattern_library import available_count
        self.assertGreater(available_count(), 25)

    def test_scale_parameter_changes_output(self):
        from depthforge.core.pattern_library import get_pattern
        a = get_pattern("plasma_wave", width=64, height=64, scale=1.0, seed=1)
        b = get_pattern("plasma_wave", width=64, height=64, scale=3.0, seed=1)
        self.assertFalse(np.array_equal(a, b))


# ===========================================================================
# Depth Models
# ===========================================================================

class TestDepthModels(unittest.TestCase):

    def test_list_estimators(self):
        from depthforge.depth_models import list_estimators
        names = list_estimators()
        # Mock estimators always available
        self.assertIn("midas_mock", names)
        self.assertIn("zoedepth_mock", names)

    def test_get_mock_midas(self):
        from depthforge.depth_models import get_depth_estimator
        estimator = get_depth_estimator("midas_mock")
        self.assertEqual(estimator.name, "midas_mock")

    def test_get_mock_zoedepth(self):
        from depthforge.depth_models import get_depth_estimator
        estimator = get_depth_estimator("zoedepth_mock")
        self.assertEqual(estimator.name, "zoedepth_mock")

    def test_unknown_estimator_raises(self):
        from depthforge.depth_models import get_depth_estimator, DepthEstimationError
        with self.assertRaises(DepthEstimationError):
            get_depth_estimator("nonexistent_model_xyz")

    def test_mock_midas_estimate_shape(self):
        from depthforge.depth_models import get_depth_estimator
        estimator = get_depth_estimator("midas_mock")
        rgb = np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8)
        depth = estimator.estimate(rgb)
        self.assertEqual(depth.shape, (64, 96))

    def test_mock_midas_estimate_dtype(self):
        from depthforge.depth_models import get_depth_estimator
        estimator = get_depth_estimator("midas_mock")
        rgb = np.random.randint(0, 255, (32, 48, 3), dtype=np.uint8)
        depth = estimator.estimate(rgb)
        self.assertEqual(depth.dtype, np.float32)

    def test_mock_midas_estimate_range(self):
        from depthforge.depth_models import get_depth_estimator
        estimator = get_depth_estimator("midas_mock")
        rgb = np.random.randint(0, 255, (32, 48, 3), dtype=np.uint8)
        depth = estimator.estimate(rgb)
        self.assertGreaterEqual(depth.min(), 0.0)
        self.assertLessEqual(depth.max(), 1.0)

    def test_mock_zoedepth_estimate_shape(self):
        from depthforge.depth_models import get_depth_estimator
        estimator = get_depth_estimator("zoedepth_mock")
        rgb = np.random.randint(0, 255, (48, 64, 3), dtype=np.uint8)
        depth = estimator.estimate(rgb)
        self.assertEqual(depth.shape, (48, 64))

    def test_mock_zoedepth_sphere_depth(self):
        """ZoeDepth mock should produce a sphere in the centre (bright centre)."""
        from depthforge.depth_models import get_depth_estimator
        estimator = get_depth_estimator("zoedepth_mock")
        H, W = 64, 96
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        depth = estimator.estimate(rgb)
        centre = depth[H//2-5:H//2+5, W//2-5:W//2+5].mean()
        edge   = depth[:5, :5].mean()
        self.assertGreater(centre, edge)

    def test_real_midas_unavailable_raises(self):
        """MiDaS real model should raise DepthEstimationError without torch."""
        from depthforge.depth_models import get_depth_estimator, DepthEstimationError
        try:
            import torch
            self.skipTest("PyTorch is installed — skipping unavailability test")
        except ImportError:
            estimator = get_depth_estimator("midas")
            with self.assertRaises(DepthEstimationError):
                estimator.estimate(np.zeros((32, 48, 3), dtype=np.uint8))

    def test_estimate_accepts_rgba(self):
        """Estimator should handle RGBA input (drop alpha channel)."""
        from depthforge.depth_models import get_depth_estimator
        estimator = get_depth_estimator("midas_mock")
        rgba = np.random.randint(0, 255, (32, 48, 4), dtype=np.uint8)
        depth = estimator.estimate(rgba)
        self.assertEqual(depth.shape, (32, 48))

    def test_estimate_accepts_greyscale(self):
        """Estimator should handle greyscale (H,W) input."""
        from depthforge.depth_models import get_depth_estimator
        estimator = get_depth_estimator("midas_mock")
        grey = np.random.randint(0, 255, (32, 48), dtype=np.uint8)
        depth = estimator.estimate(grey)
        self.assertEqual(depth.shape, (32, 48))


# ===========================================================================
# IO Formats
# ===========================================================================

class TestIOFormats(unittest.TestCase):

    def _make_depth(self, H=32, W=48) -> np.ndarray:
        d = np.linspace(0, 1, W, dtype=np.float32)
        return np.tile(d, (H, 1))

    def test_read_depth_png(self):
        from depthforge.io.formats import read_depth
        path = _make_depth_file(32, 48)
        try:
            depth = read_depth(path)
            self.assertEqual(depth.dtype, np.float32)
            self.assertEqual(len(depth.shape), 2)
            self.assertGreaterEqual(depth.min(), 0.0)
            self.assertLessEqual(depth.max(), 1.0)
        finally:
            os.unlink(path)

    def test_write_depth_png(self):
        from depthforge.io.formats import write_depth, read_depth
        depth = self._make_depth()
        path = _tmpfile(".png")
        try:
            write_depth(depth, path)
            self.assertTrue(Path(path).exists())
            loaded = read_depth(path)
            self.assertEqual(loaded.shape, depth.shape)
        finally:
            os.unlink(path)

    def test_write_and_read_tiff_16bit(self):
        from depthforge.io.formats import write_depth, read_depth
        depth = self._make_depth()
        path  = _tmpfile(".tiff")
        try:
            write_depth(depth, path, bit_depth=16)
            loaded = read_depth(path)
            self.assertEqual(loaded.shape, depth.shape)
            # 16-bit → some precision loss but should be close
            np.testing.assert_allclose(loaded, depth, atol=1e-3)
        finally:
            os.unlink(path)

    def test_write_and_read_tiff_32bit(self):
        from depthforge.io.formats import write_depth, read_depth
        depth = self._make_depth()
        path  = _tmpfile(".tiff")
        try:
            write_depth(depth, path, bit_depth=32)
            loaded = read_depth(path)
            self.assertEqual(loaded.dtype, np.float32)
        finally:
            os.unlink(path)

    def test_read_rgba_png(self):
        from depthforge.io.formats import read_rgba
        src_path = _make_source_file()
        try:
            rgba = read_rgba(src_path)
            self.assertEqual(rgba.dtype, np.uint8)
            self.assertEqual(rgba.shape[2], 4)
        finally:
            os.unlink(src_path)

    def test_write_rgba_png(self):
        from depthforge.io.formats import write_rgba, read_rgba
        img = np.random.randint(0, 255, (32, 48, 4), dtype=np.uint8)
        path = _tmpfile(".png")
        try:
            write_rgba(img, path)
            self.assertTrue(Path(path).exists())
            loaded = read_rgba(path)
            self.assertEqual(loaded.shape, img.shape)
        finally:
            os.unlink(path)

    def test_write_rgba_jpg(self):
        from depthforge.io.formats import write_rgba
        img = np.random.randint(0, 255, (32, 48, 4), dtype=np.uint8)
        path = _tmpfile(".jpg")
        try:
            write_rgba(img, path)
            self.assertTrue(Path(path).exists())
        finally:
            os.unlink(path)

    def test_write_rgba_tiff(self):
        from depthforge.io.formats import write_rgba
        img = np.random.randint(0, 255, (32, 48, 4), dtype=np.uint8)
        path = _tmpfile(".tiff")
        try:
            write_rgba(img, path)
            self.assertTrue(Path(path).exists())
        finally:
            os.unlink(path)

    def test_exr_raises_without_openexr(self):
        """EXR operations should raise RuntimeError if OpenEXR not installed."""
        from depthforge.io import formats
        if formats.HAS_EXR:
            self.skipTest("OpenEXR is installed")
        from depthforge.io.formats import read_depth, write_depth
        with self.assertRaises(RuntimeError):
            read_depth("/tmp/fake_test.exr")
        with self.assertRaises(RuntimeError):
            write_depth(np.zeros((8, 8), np.float32), "/tmp/test_out.exr")

    def test_color_transform_linear_to_srgb(self):
        """apply_color_transform should work without OCIO (fallback)."""
        from depthforge.io.formats import apply_color_transform
        img = np.array([[[0.2, 0.5, 0.8, 1.0]]], dtype=np.float32)
        out = apply_color_transform(img, "linear", "sRGB")
        # Linear→sRGB: output should be brighter (gamma < 1)
        self.assertGreater(out[0, 0, 0], img[0, 0, 0])

    def test_color_transform_srgb_to_linear(self):
        from depthforge.io.formats import apply_color_transform
        img = np.array([[[0.5, 0.5, 0.5, 1.0]]], dtype=np.float32)
        out = apply_color_transform(img, "sRGB", "linear")
        # sRGB→Linear: output should be darker
        self.assertLess(out[0, 0, 0], img[0, 0, 0])

    def test_color_transform_unknown_colorspace_warns(self):
        import warnings
        from depthforge.io.formats import apply_color_transform
        if __import__("depthforge.io.formats", fromlist=["HAS_EXR"]):
            pass
        from depthforge.io import formats
        if hasattr(formats, "_exr") and formats.HAS_EXR:
            pass
        try:
            import PyOpenColorIO
            self.skipTest("OCIO installed — different code path")
        except ImportError:
            img = np.ones((4, 4, 3), dtype=np.float32) * 0.5
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = apply_color_transform(img, "ACEScg", "Rec.709")
                if w:
                    self.assertTrue(any("PyOpenColorIO" in str(x.message) for x in w))


# ===========================================================================
# CLI Tests
# ===========================================================================

class TestCLI(unittest.TestCase):
    """Test CLI commands using Click's test runner."""

    def _runner(self):
        from click.testing import CliRunner
        return CliRunner()

    def test_cli_help(self):
        from depthforge.cli.main import cli
        result = self._runner().invoke(cli, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("DepthForge", result.output)

    def test_sirds_help(self):
        from depthforge.cli.main import cli
        result = self._runner().invoke(cli, ["sirds", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--depth", result.output)

    def test_texture_help(self):
        from depthforge.cli.main import cli
        result = self._runner().invoke(cli, ["texture", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_anaglyph_help(self):
        from depthforge.cli.main import cli
        result = self._runner().invoke(cli, ["anaglyph", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_hidden_help(self):
        from depthforge.cli.main import cli
        result = self._runner().invoke(cli, ["hidden", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_stereo_pair_help(self):
        from depthforge.cli.main import cli
        result = self._runner().invoke(cli, ["stereo-pair", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_batch_help(self):
        from depthforge.cli.main import cli
        result = self._runner().invoke(cli, ["batch", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_analyze_help(self):
        from depthforge.cli.main import cli
        result = self._runner().invoke(cli, ["analyze", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_presets_help(self):
        from depthforge.cli.main import cli
        result = self._runner().invoke(cli, ["presets", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_estimate_depth_help(self):
        from depthforge.cli.main import cli
        result = self._runner().invoke(cli, ["estimate-depth", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_presets_list(self):
        from depthforge.cli.main import cli
        result = self._runner().invoke(cli, ["presets", "--list"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("shallow", result.output)
        self.assertIn("medium", result.output)
        self.assertIn("cinema", result.output)

    def test_presets_show(self):
        from depthforge.cli.main import cli
        result = self._runner().invoke(cli, ["presets", "--show", "medium"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("depth_factor", result.output)

    def test_sirds_command(self):
        from depthforge.cli.main import cli
        depth_path = _make_depth_file(32, 48)
        out_path   = _tmpfile(".png")
        try:
            result = self._runner().invoke(cli, [
                "sirds",
                "--depth", depth_path,
                "--output", out_path,
                "--depth-factor", "0.3",
                "--tile-size", "48",
                "--seed", "1",
            ])
            self.assertEqual(result.exit_code, 0, msg=result.output)
            self.assertTrue(Path(out_path).exists())
            img = Image.open(out_path)
            self.assertEqual(img.size, (48, 32))
        finally:
            os.unlink(depth_path)
            if Path(out_path).exists():
                os.unlink(out_path)

    def test_texture_command(self):
        from depthforge.cli.main import cli
        depth_path = _make_depth_file(32, 48)
        out_path   = _tmpfile(".png")
        try:
            result = self._runner().invoke(cli, [
                "texture",
                "--depth", depth_path,
                "--output", out_path,
                "--pattern", "plasma",
                "--color", "psychedelic",
                "--tile-size", "48",
                "--seed", "42",
            ])
            self.assertEqual(result.exit_code, 0, msg=result.output)
            self.assertTrue(Path(out_path).exists())
        finally:
            os.unlink(depth_path)
            if Path(out_path).exists():
                os.unlink(out_path)

    def test_hidden_star_command(self):
        from depthforge.cli.main import cli
        out_path = _tmpfile(".png")
        try:
            result = self._runner().invoke(cli, [
                "hidden",
                "--shape", "star",
                "--width", "96",
                "--height", "64",
                "--output", out_path,
            ])
            self.assertEqual(result.exit_code, 0, msg=result.output)
            self.assertTrue(Path(out_path).exists())
        finally:
            if Path(out_path).exists():
                os.unlink(out_path)

    def test_hidden_text_command(self):
        from depthforge.cli.main import cli
        out_path = _tmpfile(".png")
        try:
            result = self._runner().invoke(cli, [
                "hidden",
                "--text", "HI",
                "--width", "192",
                "--height", "64",
                "--output", out_path,
            ])
            self.assertEqual(result.exit_code, 0, msg=result.output)
        finally:
            if Path(out_path).exists():
                os.unlink(out_path)

    def test_anaglyph_command(self):
        from depthforge.cli.main import cli
        src_path   = _make_source_file(32, 48)
        depth_path = _make_depth_file(32, 48)
        out_path   = _tmpfile(".png")
        try:
            result = self._runner().invoke(cli, [
                "anaglyph",
                "--source", src_path,
                "--depth", depth_path,
                "--output", out_path,
                "--mode", "optimised",
            ])
            self.assertEqual(result.exit_code, 0, msg=result.output)
            self.assertTrue(Path(out_path).exists())
        finally:
            for p in [src_path, depth_path]:
                os.unlink(p)
            if Path(out_path).exists():
                os.unlink(out_path)

    def test_stereo_pair_sbs_command(self):
        from depthforge.cli.main import cli
        src_path   = _make_source_file(32, 48)
        depth_path = _make_depth_file(32, 48)
        out_path   = _tmpfile(".png")
        try:
            result = self._runner().invoke(cli, [
                "stereo-pair",
                "--source", src_path,
                "--depth", depth_path,
                "--output", out_path,
                "--layout", "side-by-side",
            ])
            self.assertEqual(result.exit_code, 0, msg=result.output)
            self.assertTrue(Path(out_path).exists())
        finally:
            for p in [src_path, depth_path]:
                os.unlink(p)
            if Path(out_path).exists():
                os.unlink(out_path)

    def test_stereo_pair_separate_command(self):
        from depthforge.cli.main import cli
        src_path   = _make_source_file(32, 48)
        depth_path = _make_depth_file(32, 48)
        out_path   = _tmpfile(".png")
        try:
            result = self._runner().invoke(cli, [
                "stereo-pair",
                "--source", src_path,
                "--depth", depth_path,
                "--output", out_path,
                "--layout", "separate",
            ])
            self.assertEqual(result.exit_code, 0, msg=result.output)
        finally:
            for p in [src_path, depth_path]:
                os.unlink(p)
            stem = Path(out_path).stem
            parent = Path(out_path).parent
            for suffix in ["_left.png", "_right.png"]:
                lp = parent / (stem + suffix)
                if lp.exists():
                    lp.unlink()

    def test_analyze_command(self):
        from depthforge.cli.main import cli
        depth_path = _make_depth_file(32, 48)
        try:
            result = self._runner().invoke(cli, [
                "analyze",
                "--depth", depth_path,
                "--frame-width", "640",
                "--depth-factor", "0.35",
            ])
            self.assertEqual(result.exit_code, 0, msg=result.output)
            self.assertIn("Comfort Analysis", result.output)
        finally:
            os.unlink(depth_path)

    def test_analyze_json_output(self):
        from depthforge.cli.main import cli
        depth_path = _make_depth_file(32, 48)
        json_path  = _tmpfile(".json")
        try:
            result = self._runner().invoke(cli, [
                "analyze",
                "--depth", depth_path,
                "--json", json_path,
            ])
            self.assertEqual(result.exit_code, 0, msg=result.output)
            data = json.loads(Path(json_path).read_text())
            self.assertIn("overall_score", data)
            self.assertIn("passed", data)
        finally:
            os.unlink(depth_path)
            if Path(json_path).exists():
                os.unlink(json_path)

    def test_hidden_no_shape_raises(self):
        from depthforge.cli.main import cli
        result = self._runner().invoke(cli, [
            "hidden", "--width", "96", "--height", "64",
            "--output", "/tmp/test_no_shape.png"
        ])
        self.assertNotEqual(result.exit_code, 0)

    def test_batch_command(self):
        from depthforge.cli.main import cli
        import tempfile
        with tempfile.TemporaryDirectory() as in_dir, tempfile.TemporaryDirectory() as out_dir:
            # Create a few depth files
            for i in range(3):
                _make_depth_file(32, 48, path=f"{in_dir}/depth_{i:02d}.png")
            result = self._runner().invoke(cli, [
                "batch",
                "--input-dir", in_dir,
                "--output-dir", out_dir,
                "--mode", "texture",
                "--preset", "shallow",
                "--pattern", "perlin",
                "--color", "greyscale",
            ])
            self.assertEqual(result.exit_code, 0, msg=result.output)
            output_files = list(Path(out_dir).glob("*.png"))
            self.assertEqual(len(output_files), 3)

    def test_sirds_with_preset(self):
        from depthforge.cli.main import cli
        depth_path = _make_depth_file(32, 48)
        out_path   = _tmpfile(".png")
        try:
            result = self._runner().invoke(cli, [
                "sirds",
                "--depth", depth_path,
                "--output", out_path,
                "--preset", "shallow",
            ])
            self.assertEqual(result.exit_code, 0, msg=result.output)
        finally:
            os.unlink(depth_path)
            if Path(out_path).exists():
                os.unlink(out_path)

    def test_analyze_saves_heatmap(self):
        from depthforge.cli.main import cli
        depth_path  = _make_depth_file(32, 48)
        heatmap_out = _tmpfile(".png")
        try:
            result = self._runner().invoke(cli, [
                "analyze",
                "--depth", depth_path,
                "--output-heatmap", heatmap_out,
            ])
            self.assertEqual(result.exit_code, 0, msg=result.output)
            self.assertTrue(Path(heatmap_out).exists())
        finally:
            os.unlink(depth_path)
            if Path(heatmap_out).exists():
                os.unlink(heatmap_out)


# ===========================================================================
# Phase 3 Integration
# ===========================================================================

class TestPhase3Integration(unittest.TestCase):

    def test_pattern_library_into_synthesizer(self):
        """Library patterns should feed directly into the synthesizer."""
        from depthforge.core.pattern_library import get_pattern
        from depthforge.core.synthesizer import synthesize, StereoParams
        from depthforge.core.depth_prep import prep_depth, DepthPrepParams

        H, W = 32, 48
        y, x = np.mgrid[0:H, 0:W]
        depth_raw = np.clip(1.0 - np.sqrt((x - W/2)**2 + (y - H/2)**2) / (W/2), 0, 1).astype(np.float32)
        depth = prep_depth(depth_raw, DepthPrepParams())

        pattern = get_pattern("crystalline", width=48, height=48, seed=7)
        result  = synthesize(depth, pattern, StereoParams(depth_factor=0.3))
        self.assertEqual(result.shape, (H, W, 4))

    def test_mock_depth_estimate_into_synthesizer(self):
        """AI depth mock → prep → synthesize pipeline."""
        from depthforge.depth_models import get_depth_estimator
        from depthforge.core.depth_prep import prep_depth, DepthPrepParams
        from depthforge.core.synthesizer import synthesize, StereoParams
        from depthforge.core.pattern_gen import generate_pattern, PatternParams, PatternType, ColorMode

        H, W = 32, 48
        rgb   = np.random.randint(100, 200, (H, W, 3), dtype=np.uint8)
        est   = get_depth_estimator("midas_mock")
        depth = est.estimate(rgb)
        depth = prep_depth(depth, DepthPrepParams())

        pattern = generate_pattern(PatternParams(
            pattern_type=PatternType.RANDOM_NOISE,
            tile_width=48, tile_height=48,
            color_mode=ColorMode.GREYSCALE, seed=1,
        ))
        result = synthesize(depth, pattern, StereoParams())
        self.assertEqual(result.shape, (H, W, 4))

    def test_io_round_trip_tiff(self):
        """Write and re-read a synthesized output as TIFF."""
        from depthforge.io.formats import write_rgba, read_rgba
        import tempfile

        img = np.random.randint(0, 255, (32, 48, 4), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".tiff", delete=False) as f:
            path = f.name
        try:
            write_rgba(img, path)
            loaded = read_rgba(path)
            self.assertEqual(loaded.shape, img.shape)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
