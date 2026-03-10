"""
DepthForge — Phase 2 Test Suite
================================
Tests for: presets, comfort analyzer, safety limiter, QC overlays

Run:
    cd /home/claude && python -m unittest depthforge.tests.test_phase2 -v
    # or
    pytest depthforge/tests/test_phase2.py -v
"""

import os
import tempfile
import unittest

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sphere_depth(H=128, W=192) -> np.ndarray:
    """Synthetic sphere depth map for testing."""
    y, x = np.mgrid[0:H, 0:W]
    cx, cy = W / 2, H / 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    d = np.clip(1.0 - r / (min(W, H) * 0.45), 0.0, 1.0)
    return d.astype(np.float32)


def _gradient_depth(H=128, W=192) -> np.ndarray:
    """Simple left→right gradient for testing."""
    d = np.tile(np.linspace(0, 1, W, dtype=np.float32), (H, 1))
    return d


def _flat_depth(H=128, W=192, value=0.5) -> np.ndarray:
    return np.full((H, W), value, dtype=np.float32)


def _edge_near_depth(H=128, W=192) -> np.ndarray:
    """Near objects touching the frame edge — should trigger window violations."""
    d = _flat_depth(H, W, 0.1)
    d[:, :5] = 0.98  # near object at left edge
    return d


# ===========================================================================
# Presets
# ===========================================================================


class TestPresets(unittest.TestCase):

    def test_list_presets_returns_all_six(self):
        from depthforge.core.presets import list_presets

        names = list_presets()
        self.assertEqual(len(names), 6)
        for n in ["shallow", "medium", "deep", "cinema", "print", "broadcast"]:
            self.assertIn(n, names)

    def test_get_preset_shallow(self):
        from depthforge.core.presets import get_preset

        p = get_preset("shallow")
        self.assertEqual(p.name, "shallow")
        self.assertIsNotNone(p.stereo_params)
        self.assertIsNotNone(p.depth_params)

    def test_get_preset_case_insensitive(self):
        from depthforge.core.presets import get_preset

        p1 = get_preset("MEDIUM")
        p2 = get_preset("medium")
        self.assertEqual(p1.name, p2.name)

    def test_get_preset_unknown_raises(self):
        from depthforge.core.presets import get_preset

        with self.assertRaises(KeyError):
            get_preset("nonexistent_preset")

    def test_preset_depth_factor_ordering(self):
        """Deeper presets should have higher depth_factor."""
        from depthforge.core.presets import get_preset

        shallow = get_preset("shallow").stereo_params.depth_factor
        medium = get_preset("medium").stereo_params.depth_factor
        deep = get_preset("deep").stereo_params.depth_factor
        self.assertLess(shallow, medium)
        self.assertLess(medium, deep)

    def test_preset_safe_mode_on_shallow_and_broadcast(self):
        from depthforge.core.presets import get_preset

        self.assertTrue(get_preset("shallow").stereo_params.safe_mode)
        self.assertTrue(get_preset("broadcast").stereo_params.safe_mode)

    def test_preset_safe_mode_off_deep(self):
        from depthforge.core.presets import get_preset

        self.assertFalse(get_preset("deep").stereo_params.safe_mode)

    def test_print_preset_dpi(self):
        from depthforge.core.presets import get_preset

        self.assertEqual(get_preset("print").output_dpi, 300)

    def test_cinema_color_space(self):
        from depthforge.core.presets import get_preset

        self.assertEqual(get_preset("cinema").color_space, "P3")

    def test_broadcast_color_space(self):
        from depthforge.core.presets import get_preset

        self.assertEqual(get_preset("broadcast").color_space, "Rec.709")

    def test_comfort_limit_px(self):
        from depthforge.core.presets import get_preset

        p = get_preset("medium")
        limit = p.comfort_limit_px(1920)
        self.assertAlmostEqual(limit, int(1920 * p.stereo_params.max_parallax_fraction))

    def test_to_dict_and_from_dict(self):
        from depthforge.core.presets import Preset, get_preset

        original = get_preset("medium")
        d = original.to_dict()
        restored = Preset.from_dict(d)
        self.assertEqual(restored.name, original.name)
        self.assertAlmostEqual(
            restored.stereo_params.depth_factor,
            original.stereo_params.depth_factor,
        )

    def test_to_json_roundtrip(self):
        from depthforge.core.presets import get_preset

        original = get_preset("cinema")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            fname = f.name
        try:
            original.to_json(fname)
            from depthforge.core.presets import load_preset_from_json

            restored = load_preset_from_json(fname)
            self.assertEqual(restored.name, original.name)
        finally:
            os.unlink(fname)

    def test_register_custom_preset(self):
        from depthforge.core.depth_prep import DepthPrepParams
        from depthforge.core.presets import Preset, get_preset, list_presets, register_preset
        from depthforge.core.synthesizer import StereoParams

        custom = Preset(
            name="test_custom_xyz",
            display_name="Test Custom",
            description="Unit test preset",
            stereo_params=StereoParams(depth_factor=0.1),
            depth_params=DepthPrepParams(),
        )
        register_preset(custom)
        self.assertIn("test_custom_xyz", list_presets())
        retrieved = get_preset("test_custom_xyz")
        self.assertAlmostEqual(retrieved.stereo_params.depth_factor, 0.1)

    def test_register_duplicate_raises_without_overwrite(self):
        from depthforge.core.presets import get_preset, register_preset

        p = get_preset("shallow")
        with self.assertRaises(ValueError):
            register_preset(p)

    def test_all_presets_have_notes(self):
        from depthforge.core.presets import get_preset, list_presets

        for name in list_presets():
            p = get_preset(name)
            self.assertIsInstance(p.notes, str)


# ===========================================================================
# Safety Limiter
# ===========================================================================


class TestSafetyLimiter(unittest.TestCase):

    def _get_limiter(self, **kwargs):
        from depthforge.core.comfort import SafetyLimiter, SafetyLimiterParams

        return SafetyLimiter(SafetyLimiterParams(warn_on_violation=False, **kwargs))

    def test_clean_depth_passes_without_changes(self):
        # Flat depth within default [0.02, 0.98] range — should not be clipped
        limiter = self._get_limiter()
        depth = _flat_depth(value=0.5)
        safe_depth, safe_params, violations = limiter.apply(depth)
        self.assertNotIn("depth_clipped", violations)

    def test_extreme_depth_gets_clamped(self):
        limiter = self._get_limiter(near_clip=0.1, far_clip=0.9)
        depth = _sphere_depth()  # has values near 0 and 1
        safe_depth, _, violations = limiter.apply(depth)
        self.assertIn("depth_clipped", violations)
        self.assertGreaterEqual(safe_depth.min(), 0.09)
        self.assertLessEqual(safe_depth.max(), 0.91)

    def test_high_gradient_triggers_smoothing(self):
        """Sharp depth step should trigger smoothing."""
        limiter = self._get_limiter(max_gradient=0.01)  # very tight limit
        depth = _flat_depth()
        depth[60:70, :] = 0.95  # sharp depth step
        _, _, violations = limiter.apply(depth)
        self.assertIn("high_gradient", violations)

    def test_stereo_params_depth_factor_clamped(self):
        from depthforge.core.synthesizer import StereoParams

        limiter = self._get_limiter(max_depth_factor=0.3)
        depth = _sphere_depth()
        params = StereoParams(depth_factor=0.8)
        _, safe_params, violations = limiter.apply(depth, params)
        self.assertIn("depth_factor", violations)
        self.assertAlmostEqual(safe_params.depth_factor, 0.3)

    def test_stereo_params_parallax_clamped(self):
        from depthforge.core.synthesizer import StereoParams

        limiter = self._get_limiter(max_parallax_fraction=0.025)
        depth = _sphere_depth()
        params = StereoParams(max_parallax_fraction=0.10)
        _, safe_params, violations = limiter.apply(depth, params)
        self.assertIn("parallax_fraction", violations)
        self.assertAlmostEqual(safe_params.max_parallax_fraction, 0.025)

    def test_no_params_returns_none(self):
        limiter = self._get_limiter()
        depth = _sphere_depth()
        _, safe_params, _ = limiter.apply(depth, None)
        self.assertIsNone(safe_params)

    def test_from_profile_conservative(self):
        from depthforge.core.comfort import SafetyLimiterParams, VergenceProfile

        p = SafetyLimiterParams.from_profile(VergenceProfile.CONSERVATIVE)
        self.assertAlmostEqual(p.max_parallax_fraction, 0.020)

    def test_from_profile_cinema(self):
        from depthforge.core.comfort import SafetyLimiterParams, VergenceProfile

        p = SafetyLimiterParams.from_profile(VergenceProfile.CINEMA)
        self.assertAlmostEqual(p.max_parallax_fraction, 0.022)

    def test_negative_depth_factor_clamped_correctly(self):
        """Negative depth_factor (pop-out) should be clamped in magnitude."""
        from depthforge.core.synthesizer import StereoParams

        limiter = self._get_limiter(max_depth_factor=0.3)
        depth = _sphere_depth()
        params = StereoParams(depth_factor=-0.7)
        _, safe_params, violations = limiter.apply(depth, params)
        self.assertAlmostEqual(safe_params.depth_factor, -0.3)


# ===========================================================================
# Comfort Analyzer
# ===========================================================================


class TestComfortAnalyzer(unittest.TestCase):

    def _get_analyzer(self, **kwargs):
        from depthforge.core.comfort import ComfortAnalyzer, ComfortAnalyzerParams

        return ComfortAnalyzer(ComfortAnalyzerParams(frame_width=640, frame_height=480, **kwargs))

    def test_analyze_returns_comfort_report(self):
        from depthforge.core.comfort import ComfortReport

        analyzer = self._get_analyzer()
        report = analyzer.analyze(_sphere_depth(64, 96))
        self.assertIsInstance(report, ComfortReport)

    def test_parallax_map_shape(self):
        analyzer = self._get_analyzer()
        depth = _sphere_depth(64, 96)
        report = analyzer.analyze(depth)
        self.assertEqual(report.parallax_map.shape, depth.shape)

    def test_vergence_map_shape(self):
        analyzer = self._get_analyzer()
        depth = _sphere_depth(64, 96)
        report = analyzer.analyze(depth)
        self.assertEqual(report.vergence_map.shape, depth.shape)

    def test_gradient_map_shape(self):
        analyzer = self._get_analyzer()
        depth = _sphere_depth(64, 96)
        report = analyzer.analyze(depth)
        self.assertEqual(report.gradient_map.shape, depth.shape)

    def test_flat_depth_no_parallax_violation(self):
        """Flat depth should have zero parallax violation (all pixels same depth)."""
        analyzer = self._get_analyzer(depth_factor=0.3)
        depth = _flat_depth(64, 96, 0.5)
        report = analyzer.analyze(depth)
        # Parallax is uniform (all same) — no pixels should exceed parallax limit
        limit_px = analyzer.params.frame_width * analyzer.params.max_parallax_fraction
        self.assertLessEqual(report.max_parallax_px, limit_px * analyzer.params.depth_factor + 0.1)

    def test_extreme_depth_fails(self):
        """Very high depth_factor + full depth range should fail."""
        analyzer = self._get_analyzer(
            max_parallax_fraction=0.01,
            depth_factor=0.8,
        )
        depth = _gradient_depth(64, 96)  # full 0→1 gradient
        report = analyzer.analyze(depth)
        self.assertFalse(report.passed)

    def test_violations_list_not_empty_on_failure(self):
        from depthforge.core.comfort import ComfortAnalyzer, ComfortAnalyzerParams

        analyzer = ComfortAnalyzer(
            ComfortAnalyzerParams(frame_width=640, max_parallax_fraction=0.001, depth_factor=0.5)
        )
        report = analyzer.analyze(_sphere_depth(64, 96))
        self.assertGreater(len(report.violations), 0)

    def test_advice_given_on_failure(self):
        from depthforge.core.comfort import ComfortAnalyzer, ComfortAnalyzerParams

        analyzer = ComfortAnalyzer(
            ComfortAnalyzerParams(frame_width=640, max_parallax_fraction=0.001, depth_factor=0.5)
        )
        report = analyzer.analyze(_sphere_depth(64, 96))
        self.assertGreater(len(report.advice), 0)

    def test_overall_score_range(self):
        analyzer = self._get_analyzer()
        report = analyzer.analyze(_sphere_depth(64, 96))
        self.assertGreaterEqual(report.overall_score, 0.0)
        self.assertLessEqual(report.overall_score, 1.0)

    def test_window_violation_detected(self):
        analyzer = self._get_analyzer(depth_factor=0.3)
        depth = _edge_near_depth(64, 96)
        report = analyzer.analyze(depth)
        self.assertTrue(report.window_violation_mask.any())

    def test_summary_string(self):
        analyzer = self._get_analyzer()
        report = analyzer.analyze(_sphere_depth(64, 96))
        s = report.summary()
        self.assertIn("Comfort Analysis", s)
        self.assertIn("Max parallax", s)

    def test_print_does_not_raise(self):
        analyzer = self._get_analyzer()
        report = analyzer.analyze(_sphere_depth(64, 96))
        try:
            report.print()
        except Exception as e:
            self.fail(f"report.print() raised {e}")

    def test_violation_overlay_shape(self):
        analyzer = self._get_analyzer()
        depth = _sphere_depth(64, 96)
        report = analyzer.analyze(depth)
        overlay = analyzer.violation_overlay(depth, report)
        self.assertEqual(overlay.shape[:2], depth.shape)
        self.assertEqual(overlay.shape[2], 4)

    def test_violation_overlay_without_precomputed_report(self):
        analyzer = self._get_analyzer()
        depth = _sphere_depth(64, 96)
        overlay = analyzer.violation_overlay(depth)  # no report passed
        self.assertEqual(overlay.shape[2], 4)


# ===========================================================================
# Vergence Strain Estimator
# ===========================================================================


class TestVergenceStrain(unittest.TestCase):

    def test_returns_dict_with_required_keys(self):
        from depthforge.core.comfort import estimate_vergence_strain

        result = estimate_vergence_strain(_sphere_depth(64, 96))
        for key in [
            "mean_vergence_deg",
            "max_vergence_deg",
            "vergence_range_deg",
            "rapid_change_fraction",
            "strain_rating",
        ]:
            self.assertIn(key, result)

    def test_flat_depth_zero_rapid_change(self):
        """Flat depth has no spatial vergence variation — zero rapid change fraction."""
        from depthforge.core.comfort import estimate_vergence_strain

        result = estimate_vergence_strain(_flat_depth(64, 96, 0.5))
        self.assertAlmostEqual(result["rapid_change_fraction"], 0.0, places=3)

    def test_strain_rating_is_string(self):
        from depthforge.core.comfort import estimate_vergence_strain

        result = estimate_vergence_strain(_sphere_depth(64, 96))
        self.assertIn(result["strain_rating"], ["Low", "Moderate", "High", "Severe"])

    def test_max_vergence_greater_than_mean(self):
        from depthforge.core.comfort import estimate_vergence_strain

        result = estimate_vergence_strain(_gradient_depth(64, 96))
        self.assertGreaterEqual(result["max_vergence_deg"], result["mean_vergence_deg"])

    def test_vergence_values_positive(self):
        from depthforge.core.comfort import estimate_vergence_strain

        result = estimate_vergence_strain(_sphere_depth(64, 96))
        self.assertGreater(result["mean_vergence_deg"], 0)
        self.assertGreater(result["max_vergence_deg"], 0)


# ===========================================================================
# QC Overlays
# ===========================================================================


class TestQCOverlays(unittest.TestCase):

    def test_parallax_heatmap_shape(self):
        from depthforge.core.qc import parallax_heatmap

        depth = _sphere_depth(64, 96)
        out = parallax_heatmap(depth, annotate=False)
        self.assertEqual(out.shape[:2], depth.shape)
        self.assertEqual(out.shape[2], 4)

    def test_parallax_heatmap_dtype(self):
        from depthforge.core.qc import parallax_heatmap

        out = parallax_heatmap(_sphere_depth(64, 96), annotate=False)
        self.assertEqual(out.dtype, np.uint8)

    def test_depth_band_preview_shape(self):
        from depthforge.core.qc import depth_band_preview

        depth = _gradient_depth(64, 96)
        out = depth_band_preview(depth, n_bands=8)
        self.assertEqual(out.shape[:2], depth.shape)
        self.assertEqual(out.shape[2], 4)

    def test_depth_band_preview_colours_differ_per_band(self):
        from depthforge.core.qc import depth_band_preview

        depth = _gradient_depth(64, 96)
        out = depth_band_preview(depth, n_bands=8)
        # Sample middle row — should have varying colours across the gradient
        row = out[32, :, :3]
        # Not all pixels the same colour
        self.assertGreater(np.std(row.astype(float)), 5.0)

    def test_depth_band_custom_palette(self):
        from depthforge.core.qc import depth_band_preview

        palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        depth = _gradient_depth(32, 64)
        out = depth_band_preview(depth, n_bands=4, palette=palette)
        self.assertEqual(out.shape[2], 4)

    def test_window_violation_overlay_shape(self):
        from depthforge.core.qc import window_violation_overlay

        depth = _edge_near_depth(64, 96)
        out = window_violation_overlay(depth)
        self.assertEqual(out.shape[:2], depth.shape)
        self.assertEqual(out.shape[2], 4)

    def test_window_violation_highlights_edge(self):
        from depthforge.core.qc import window_violation_overlay

        depth = _edge_near_depth(64, 96)
        out = window_violation_overlay(depth, highlight_colour=(255, 0, 0))
        # Left edge (col 0-4) should be highlighted red
        edge = out[:, 0, 0]  # R channel
        self.assertGreater(edge.mean(), 100)

    def test_safe_zone_indicator_wider_than_depth(self):
        from depthforge.core.qc import safe_zone_indicator

        depth = _sphere_depth(64, 96)
        out = safe_zone_indicator(depth, indicator_width=40)
        self.assertEqual(out.shape[1], 96 + 40)

    def test_depth_histogram_shape(self):
        from depthforge.core.qc import depth_histogram

        depth = _gradient_depth(64, 96)
        out = depth_histogram(depth, bins=32, width=200, height=100)
        self.assertEqual(out.shape, (100, 200, 4))

    def test_depth_histogram_dtype(self):
        from depthforge.core.qc import depth_histogram

        out = depth_histogram(_sphere_depth(64, 96))
        self.assertEqual(out.dtype, np.uint8)

    def test_comparison_grid_shape(self):
        from depthforge.core.qc import comparison_grid

        imgs = [np.random.randint(0, 255, (64, 96, 4), dtype=np.uint8) for _ in range(4)]
        grid = comparison_grid(imgs, cols=2)
        self.assertEqual(grid.shape[2], 4)
        # Should be wider than one image
        self.assertGreater(grid.shape[1], 96)

    def test_comparison_grid_with_labels(self):
        from depthforge.core.qc import comparison_grid

        imgs = [np.zeros((32, 48, 4), dtype=np.uint8) for _ in range(3)]
        labels = ["A", "B", "C"]
        grid = comparison_grid(imgs, labels=labels, cols=3)
        self.assertEqual(grid.shape[2], 4)

    def test_colormaps_dont_raise(self):
        from depthforge.core.qc import QCParams, parallax_heatmap

        depth = _sphere_depth(32, 48)
        for cm in ["jet", "inferno", "viridis", "plasma", "turbo"]:
            out = parallax_heatmap(depth, QCParams(colormap=cm), annotate=False)
            self.assertEqual(out.shape[2], 4, msg=f"colormap {cm} failed")

    def test_all_qc_outputs_rgba(self):
        """All QC functions should return RGBA uint8."""
        import depthforge.core.qc as qc

        depth = _sphere_depth(32, 48)
        results = [
            qc.parallax_heatmap(depth, annotate=False),
            qc.depth_band_preview(depth),
            qc.window_violation_overlay(depth),
            qc.depth_histogram(depth, width=100, height=50),
        ]
        for r in results:
            self.assertEqual(r.dtype, np.uint8)
            self.assertEqual(r.shape[2], 4)


# ===========================================================================
# Integration: Preset → Analyze → Report
# ===========================================================================


class TestPhase2Integration(unittest.TestCase):

    def test_preset_workflow(self):
        """Full workflow: get preset → prep depth → synthesize → analyze."""
        from depthforge.core.comfort import ComfortAnalyzer, ComfortAnalyzerParams
        from depthforge.core.depth_prep import prep_depth
        from depthforge.core.pattern_gen import (
            ColorMode,
            PatternParams,
            PatternType,
            generate_pattern,
        )
        from depthforge.core.presets import get_preset
        from depthforge.core.synthesizer import synthesize

        preset = get_preset("shallow")
        depth_raw = _sphere_depth(64, 96)
        depth = prep_depth(depth_raw, preset.depth_params)
        pattern = generate_pattern(
            PatternParams(
                pattern_type=PatternType.RANDOM_NOISE,
                tile_width=64,
                tile_height=64,
                color_mode=ColorMode.GREYSCALE,
                seed=1,
            )
        )
        result = synthesize(depth, pattern, preset.stereo_params)
        self.assertEqual(result.shape[2], 4)

        analyzer = ComfortAnalyzer(
            ComfortAnalyzerParams(
                frame_width=96,
                frame_height=64,
                depth_factor=preset.stereo_params.depth_factor,
            )
        )
        report = analyzer.analyze(depth)
        self.assertIsInstance(report.overall_score, float)

    def test_safety_limiter_then_synthesize(self):
        """Safety limiter output feeds cleanly into synthesizer."""
        from depthforge.core.comfort import SafetyLimiter, SafetyLimiterParams
        from depthforge.core.pattern_gen import (
            ColorMode,
            PatternParams,
            PatternType,
            generate_pattern,
        )
        from depthforge.core.synthesizer import StereoParams, synthesize

        limiter = SafetyLimiter(
            SafetyLimiterParams(
                max_depth_factor=0.5,
                max_parallax_fraction=0.033,
                warn_on_violation=False,
            )
        )
        depth = _sphere_depth(64, 96)
        params = StereoParams(depth_factor=0.9, max_parallax_fraction=0.08)
        safe_depth, safe_params, _ = limiter.apply(depth, params)

        pattern = generate_pattern(
            PatternParams(
                pattern_type=PatternType.RANDOM_NOISE,
                tile_width=64,
                tile_height=64,
                color_mode=ColorMode.GREYSCALE,
                seed=1,
            )
        )
        result = synthesize(safe_depth, pattern, safe_params)
        self.assertEqual(result.shape[2], 4)

    def test_qc_heatmap_from_preset(self):
        """QC heatmap runs after preset-driven synthesis."""
        from depthforge.core.presets import get_preset
        from depthforge.core.qc import QCParams, parallax_heatmap

        preset = get_preset("medium")
        depth = _sphere_depth(64, 96)
        p = QCParams(
            frame_width=640,
            depth_factor=preset.stereo_params.depth_factor,
            max_parallax_fraction=preset.stereo_params.max_parallax_fraction,
        )
        heatmap = parallax_heatmap(depth, p, annotate=False)
        self.assertEqual(heatmap.shape[2], 4)

    def test_all_presets_can_synthesize(self):
        """Every preset should produce valid stereogram output."""
        from depthforge.core.depth_prep import prep_depth
        from depthforge.core.pattern_gen import (
            ColorMode,
            PatternParams,
            PatternType,
            generate_pattern,
        )
        from depthforge.core.presets import get_preset, list_presets
        from depthforge.core.synthesizer import synthesize

        depth_raw = _sphere_depth(32, 48)
        pattern = generate_pattern(
            PatternParams(
                pattern_type=PatternType.RANDOM_NOISE,
                tile_width=48,
                tile_height=48,
                color_mode=ColorMode.GREYSCALE,
                seed=1,
            )
        )
        for name in list_presets():
            preset = get_preset(name)
            depth = prep_depth(depth_raw, preset.depth_params)
            result = synthesize(depth, pattern, preset.stereo_params)
            self.assertEqual(result.shape[2], 4, msg=f"Preset '{name}' failed")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
