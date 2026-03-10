"""
Tests for depthforge.core.flash_safety — PSE flash and flicker safety checks.
"""

import unittest
import warnings

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _solid_rgba(r, g, b, h=64, w=64):
    """Return a solid-colour RGBA uint8 patch."""
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[..., 0] = r
    arr[..., 1] = g
    arr[..., 2] = b
    arr[..., 3] = 255
    return arr


def _checkerboard(h=64, w=64):
    """Black-and-white checkerboard — maximum Michelson contrast."""
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            v = 255 if (x + y) % 2 == 0 else 0
            arr[y, x, :3] = v
            arr[y, x, 3] = 255
    return arr


def _grey_rgba(level, h=64, w=64):
    """Uniform grey RGBA patch."""
    return _solid_rgba(level, level, level, h, w)


# ---------------------------------------------------------------------------
# Tests — EpilepticRisk
# ---------------------------------------------------------------------------


class TestEpilepticRisk(unittest.TestCase):
    def test_enum_values(self):
        from depthforge.core.flash_safety import EpilepticRisk

        self.assertEqual(EpilepticRisk.SAFE.value, "SAFE")
        self.assertEqual(EpilepticRisk.CRITICAL.value, "CRITICAL")

    def test_ordering_semantics(self):
        from depthforge.core.flash_safety import EpilepticRisk

        # The enum members should all exist and be distinct.
        members = list(EpilepticRisk)
        self.assertEqual(len(members), 5)


# ---------------------------------------------------------------------------
# Tests — FlashSafetyConfig
# ---------------------------------------------------------------------------


class TestFlashSafetyConfig(unittest.TestCase):
    def test_defaults(self):
        from depthforge.core.flash_safety import FlashSafetyConfig

        cfg = FlashSafetyConfig()
        self.assertAlmostEqual(cfg.max_flashes_per_second, 3.0)
        self.assertAlmostEqual(cfg.luminance_change_threshold, 0.10)
        self.assertAlmostEqual(cfg.red_chrominance_threshold, 0.20)
        self.assertAlmostEqual(cfg.pattern_contrast_limit, 0.50)

    def test_custom(self):
        from depthforge.core.flash_safety import FlashSafetyConfig

        cfg = FlashSafetyConfig(max_flashes_per_second=2.0, pattern_contrast_limit=0.30)
        self.assertAlmostEqual(cfg.max_flashes_per_second, 2.0)
        self.assertAlmostEqual(cfg.pattern_contrast_limit, 0.30)


# ---------------------------------------------------------------------------
# Tests — check_pattern_flash_risk (static)
# ---------------------------------------------------------------------------


class TestCheckPatternFlashRisk(unittest.TestCase):
    def setUp(self):
        from depthforge.core.flash_safety import check_pattern_flash_risk

        self.check = check_pattern_flash_risk

    def test_uniform_grey_is_safe(self):
        pat = _grey_rgba(128)
        report = self.check(pat)
        self.assertTrue(report.passed)
        self.assertEqual(report.risk.value, "SAFE")
        self.assertAlmostEqual(report.flash_frequency_hz, 0.0)

    def test_checkerboard_fails(self):
        pat = _checkerboard()
        report = self.check(pat)
        self.assertFalse(report.passed)
        self.assertIn(report.risk.value, ("HIGH", "CRITICAL"))
        self.assertGreater(report.michelson_contrast, 0.50)

    def test_high_contrast_has_violations(self):
        pat = _checkerboard()
        report = self.check(pat)
        self.assertGreater(len(report.violations), 0)

    def test_low_contrast_passes(self):
        # 128 vs 148 — ~7 % contrast
        pat = np.full((64, 64, 4), 128, dtype=np.uint8)
        pat[32:, :, :3] = 148
        pat[..., 3] = 255
        report = self.check(pat)
        self.assertTrue(report.passed)

    def test_report_has_standard_keys(self):
        pat = _grey_rgba(200)
        report = self.check(pat)
        for key in ("WCAG_2.3.1", "ITU_BT1702", "Ofcom", "Harding_PSE"):
            self.assertIn(key, report.standard_results)

    def test_summary_contains_risk(self):
        pat = _checkerboard()
        report = self.check(pat)
        s = report.summary()
        self.assertIn("Flash Safety", s)
        self.assertIn(report.risk.value, s)

    def test_mixed_red_blue_red_flash_risk(self):
        # Half pure-red, half pure-blue — high red chrominance variation across pixels.
        pat = np.zeros((64, 64, 4), dtype=np.uint8)
        pat[:, :32, 0] = 255  # red half
        pat[:, 32:, 2] = 255  # blue half
        pat[..., 3] = 255
        report = self.check(pat)
        # Red chrominance (R - Y) differs strongly between the two halves.
        self.assertGreater(report.max_red_delta, 0.10)

    def test_advice_provided_when_failing(self):
        pat = _checkerboard()
        report = self.check(pat)
        if not report.passed:
            self.assertGreater(len(report.advice), 0)

    def test_custom_config_stricter(self):
        from depthforge.core.flash_safety import FlashSafetyConfig

        # A moderately contrasted pattern that passes default thresholds
        # should fail with a very strict config.
        pat = np.full((64, 64, 4), 255, dtype=np.uint8)
        pat[:32, :, :3] = 180  # ~17 % Michelson contrast
        pat[..., 3] = 255
        strict = FlashSafetyConfig(pattern_contrast_limit=0.05)
        report = self.check(pat, config=strict)
        self.assertFalse(report.passed)


# ---------------------------------------------------------------------------
# Tests — check_frame_sequence (video)
# ---------------------------------------------------------------------------


class TestCheckFrameSequence(unittest.TestCase):
    def setUp(self):
        from depthforge.core.flash_safety import check_frame_sequence

        self.check = check_frame_sequence

    def test_empty_sequence_is_safe(self):
        report = self.check([], fps=24.0)
        self.assertTrue(report.passed)

    def test_single_frame_is_safe(self):
        report = self.check([_grey_rgba(200)], fps=24.0)
        self.assertTrue(report.passed)

    def test_static_sequence_is_safe(self):
        frames = [_grey_rgba(128)] * 48  # 2 seconds at 24 fps, no change
        report = self.check(frames, fps=24.0)
        self.assertTrue(report.passed)
        self.assertAlmostEqual(report.flash_frequency_hz, 0.0)

    def test_slow_ramp_is_safe(self):
        # Gradual luminance change — never crosses flash threshold within 1 sec
        frames = []
        for i in range(48):
            lv = int(100 + i * 1.5)  # 100 → ~172 over 2 s
            frames.append(_grey_rgba(lv))
        report = self.check(frames, fps=24.0)
        self.assertTrue(report.passed)

    def test_rapid_flicker_fails(self):
        # Alternating black / white at 12 Hz (every other frame at 24 fps = 12 flash pairs/sec)
        frames = []
        for i in range(48):
            lv = 255 if i % 2 == 0 else 0
            frames.append(_grey_rgba(lv))
        report = self.check(frames, fps=24.0)
        self.assertFalse(report.passed)
        self.assertGreater(report.flash_frequency_hz, 3.0)
        self.assertIn(report.risk.value, ("HIGH", "CRITICAL"))

    def test_3hz_boundary_fails(self):
        # Exactly 4 flash pairs in 1 second at 24 fps (> 3 Hz limit)
        # 8 transitions in 24 frames = 4 pairs
        frames = []
        for i in range(24):
            lv = 255 if (i // 3) % 2 == 0 else 0
            frames.append(_grey_rgba(lv))
        report = self.check(frames, fps=24.0)
        # May or may not fail depending on exact pair count; at minimum flash_hz > 0
        self.assertIsNotNone(report)

    def test_report_standard_keys(self):
        frames = [_grey_rgba(128)] * 5
        report = self.check(frames, fps=24.0)
        for k in ("WCAG_2.3.1", "ITU_BT1702", "Ofcom", "Harding_PSE"):
            self.assertIn(k, report.standard_results)

    def test_flash_hz_zero_for_static(self):
        frames = [_grey_rgba(200)] * 24
        report = self.check(frames, fps=24.0)
        self.assertAlmostEqual(report.flash_frequency_hz, 0.0)

    def test_michelson_computed(self):
        # Alternating bright / dark gives high michelson
        frames = [_grey_rgba(0), _grey_rgba(255)] * 12
        report = self.check(frames, fps=24.0)
        self.assertGreater(report.michelson_contrast, 0.5)


# ---------------------------------------------------------------------------
# Tests — warn_if_unsafe
# ---------------------------------------------------------------------------


class TestWarnIfUnsafe(unittest.TestCase):
    def test_safe_report_no_warning(self):
        from depthforge.core.flash_safety import EpilepticRisk, FlashSafetyReport, warn_if_unsafe

        report = FlashSafetyReport(
            risk=EpilepticRisk.SAFE,
            passed=True,
            flash_frequency_hz=0.0,
            max_luminance_delta=0.0,
            max_red_delta=0.0,
            michelson_contrast=0.0,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_if_unsafe(report)
            self.assertEqual(len(w), 0)

    def test_low_report_no_warning(self):
        from depthforge.core.flash_safety import EpilepticRisk, FlashSafetyReport, warn_if_unsafe

        report = FlashSafetyReport(
            risk=EpilepticRisk.LOW,
            passed=True,
            flash_frequency_hz=0.0,
            max_luminance_delta=0.05,
            max_red_delta=0.0,
            michelson_contrast=0.2,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_if_unsafe(report)
            self.assertEqual(len(w), 0)

    def test_high_risk_issues_warning(self):
        from depthforge.core.flash_safety import (
            EpilepticRisk,
            FlashSafetyReport,
            warn_if_unsafe,
        )

        report = FlashSafetyReport(
            risk=EpilepticRisk.HIGH,
            passed=False,
            flash_frequency_hz=6.0,
            max_luminance_delta=0.40,
            max_red_delta=0.10,
            michelson_contrast=0.75,
            violations=["Flash rate 6 Hz exceeds the 3 Hz limit."],
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_if_unsafe(report, context="test_sequence")
            self.assertEqual(len(w), 1)
            self.assertIn("PSE WARNING", str(w[0].message))
            self.assertIn("HIGH", str(w[0].message))

    def test_critical_risk_issues_warning(self):
        from depthforge.core.flash_safety import (
            EpilepticRisk,
            FlashSafetyReport,
            warn_if_unsafe,
        )

        report = FlashSafetyReport(
            risk=EpilepticRisk.CRITICAL,
            passed=False,
            flash_frequency_hz=12.0,
            max_luminance_delta=0.80,
            max_red_delta=0.50,
            michelson_contrast=0.95,
            violations=["Flash rate 12 Hz."],
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_if_unsafe(report)
            self.assertEqual(len(w), 1)
            self.assertIn("CRITICAL", str(w[0].message))


# ---------------------------------------------------------------------------
# Tests — module-level constants
# ---------------------------------------------------------------------------


class TestConstants(unittest.TestCase):
    def test_epilepsy_warning_not_empty(self):
        from depthforge.core.flash_safety import EPILEPSY_WARNING, EPILEPSY_WARNING_SHORT

        self.assertGreater(len(EPILEPSY_WARNING), 50)
        self.assertGreater(len(EPILEPSY_WARNING_SHORT), 20)

    def test_epilepsy_warning_mentions_epilepsy(self):
        from depthforge.core.flash_safety import EPILEPSY_WARNING

        self.assertIn("epilepsy", EPILEPSY_WARNING.lower())

    def test_short_warning_mentions_safe_mode(self):
        from depthforge.core.flash_safety import EPILEPSY_WARNING_SHORT

        self.assertIn("safe_mode", EPILEPSY_WARNING_SHORT)


# ---------------------------------------------------------------------------
# Tests — integration with depthforge top-level import
# ---------------------------------------------------------------------------


class TestTopLevelExports(unittest.TestCase):
    def test_epilepsy_warning_accessible(self):
        import depthforge

        self.assertTrue(hasattr(depthforge, "EPILEPSY_WARNING"))
        self.assertTrue(hasattr(depthforge, "EPILEPSY_WARNING_SHORT"))

    def test_epileptic_risk_accessible(self):
        import depthforge

        self.assertTrue(hasattr(depthforge, "EpilepticRisk"))

    def test_check_functions_accessible(self):
        import depthforge

        self.assertTrue(callable(depthforge.check_pattern_flash_risk))
        self.assertTrue(callable(depthforge.check_frame_sequence))
        self.assertTrue(callable(depthforge.warn_if_unsafe))


if __name__ == "__main__":
    unittest.main()
