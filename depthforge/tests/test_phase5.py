"""
depthforge.tests.test_phase5
============================
Phase 5 test suite — Performance, Pipeline & Export.

Covers:
  - core.parallel   (ParallelConfig, parallel_synthesize, benchmark)
  - core.optical_flow (FlowDepthConfig, compute_flow, flow_to_depth,
                       flow_warp, FlowDepthEstimator, detect_scene_cut)
  - core.temporal   (TemporalConfig, TemporalSmoother, DepthHistory)
  - core.export     (ExportProfile, Exporter, ExportQueue,
                     list_profiles, get_profile, register_profile)
  - ofx.__init__    (OFXParams, OFXBridge.synthesize_direct, check_build,
                     plugin_count)
"""

import os
import tempfile
import unittest

import numpy as np

# ── Shared helpers ──────────────────────────────────────────────────────────


def _depth(H=64, W=96, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((H, W), dtype=np.float32)


def _frame(H=64, W=96, seed=1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((H, W, 3)) * 255).astype(np.uint8)


def _pattern():
    from depthforge.core.pattern_gen import ColorMode, PatternParams, PatternType, generate_pattern

    return generate_pattern(
        PatternParams(
            pattern_type=PatternType.RANDOM_NOISE,
            color_mode=ColorMode.GREYSCALE,
            tile_width=64,
            tile_height=64,
            seed=42,
        )
    )


def _stereo_params():
    from depthforge.core.synthesizer import StereoParams

    return StereoParams(depth_factor=0.3, seed=42)


# ════════════════════════════════════════════════════════════════════════════
# 1. core.parallel
# ════════════════════════════════════════════════════════════════════════════


class TestParallelConfig(unittest.TestCase):

    def test_default_workers_positive(self):
        from depthforge.core.parallel import ParallelConfig

        cfg = ParallelConfig()
        self.assertGreaterEqual(cfg.n_workers, 1)

    def test_negative_workers_resolved(self):
        from depthforge.core.parallel import ParallelConfig

        cfg = ParallelConfig(n_workers=-1)
        self.assertGreaterEqual(cfg.n_workers, 1)

    def test_zero_workers_resolved(self):
        from depthforge.core.parallel import ParallelConfig

        cfg = ParallelConfig(n_workers=0)
        self.assertGreaterEqual(cfg.n_workers, 1)

    def test_chunk_rows_minimum(self):
        from depthforge.core.parallel import ParallelConfig

        cfg = ParallelConfig(chunk_rows=0)
        self.assertEqual(cfg.chunk_rows, 1)

    def test_fields_preserved(self):
        from depthforge.core.parallel import ParallelConfig

        cfg = ParallelConfig(n_workers=3, chunk_rows=32, min_rows_for_parallel=128)
        self.assertEqual(cfg.n_workers, 3)
        self.assertEqual(cfg.chunk_rows, 32)
        self.assertEqual(cfg.min_rows_for_parallel, 128)


class TestParallelSynthesize(unittest.TestCase):

    def setUp(self):
        self.depth = _depth(128, 192)
        self.pattern = _pattern()
        self.params = _stereo_params()

    def test_returns_correct_shape(self):
        from depthforge.core.parallel import ParallelConfig, parallel_synthesize

        cfg = ParallelConfig(n_workers=2, chunk_rows=32)
        result = parallel_synthesize(self.depth, self.pattern, self.params, cfg)
        self.assertEqual(result.shape, (128, 192, 4))

    def test_returns_uint8(self):
        from depthforge.core.parallel import ParallelConfig, parallel_synthesize

        cfg = ParallelConfig(n_workers=2, chunk_rows=32)
        result = parallel_synthesize(self.depth, self.pattern, self.params, cfg)
        self.assertEqual(result.dtype, np.uint8)

    def test_matches_serial_output(self):
        """Parallel result should match serial synthesize exactly."""
        from depthforge.core.parallel import ParallelConfig, parallel_synthesize
        from depthforge.core.synthesizer import synthesize

        serial = synthesize(self.depth, self.pattern, self.params)
        cfg = ParallelConfig(n_workers=2, chunk_rows=32, min_rows_for_parallel=1)
        parallel = parallel_synthesize(self.depth, self.pattern, self.params, cfg)
        np.testing.assert_array_equal(serial, parallel)

    def test_single_worker_matches_serial(self):
        from depthforge.core.parallel import ParallelConfig, parallel_synthesize
        from depthforge.core.synthesizer import synthesize

        serial = synthesize(self.depth, self.pattern, self.params)
        cfg = ParallelConfig(n_workers=1, min_rows_for_parallel=1)
        result = parallel_synthesize(self.depth, self.pattern, self.params, cfg)
        np.testing.assert_array_equal(serial, result)

    def test_small_image_falls_back_to_serial(self):
        """Images smaller than min_rows_for_parallel use serial path."""
        from depthforge.core.parallel import ParallelConfig, parallel_synthesize
        from depthforge.core.synthesizer import synthesize

        small_depth = _depth(32, 64)
        cfg = ParallelConfig(n_workers=4, min_rows_for_parallel=256)
        result = parallel_synthesize(small_depth, self.pattern, self.params, cfg)
        serial = synthesize(small_depth, self.pattern, self.params)
        np.testing.assert_array_equal(serial, result)

    def test_progress_callback_called(self):
        from depthforge.core.parallel import ParallelConfig, parallel_synthesize

        calls = []

        def cb(done, total):
            calls.append((done, total))

        cfg = ParallelConfig(
            n_workers=2,
            chunk_rows=32,
            min_rows_for_parallel=1,
            show_progress=True,
            progress_callback=cb,
        )
        parallel_synthesize(self.depth, self.pattern, self.params, cfg)
        self.assertGreater(len(calls), 0)

    def test_progress_callback_total_correct(self):
        from depthforge.core.parallel import ParallelConfig, parallel_synthesize

        totals = []

        def cb(done, total):
            totals.append(total)

        cfg = ParallelConfig(
            n_workers=2,
            chunk_rows=32,
            min_rows_for_parallel=1,
            show_progress=True,
            progress_callback=cb,
        )
        parallel_synthesize(self.depth, self.pattern, self.params, cfg)
        self.assertTrue(all(t == 128 for t in totals))

    def test_none_config_uses_defaults(self):
        from depthforge.core.parallel import parallel_synthesize

        result = parallel_synthesize(self.depth, self.pattern, self.params, None)
        self.assertEqual(result.shape, (128, 192, 4))

    def test_oversample_delegates(self):
        """Oversampled params are handled without error."""
        from depthforge.core.parallel import ParallelConfig, parallel_synthesize
        from depthforge.core.synthesizer import StereoParams

        params = StereoParams(depth_factor=0.3, oversample=2, seed=42)
        cfg = ParallelConfig(n_workers=2)
        result = parallel_synthesize(self.depth, self.pattern, params, cfg)
        self.assertEqual(result.shape, (128, 192, 4))


class TestParallelBenchmark(unittest.TestCase):

    def test_returns_dict_with_timing(self):
        from depthforge.core.parallel import benchmark

        results = benchmark(H=64, W=96, n_workers_list=[1])
        self.assertIn(1, results)
        self.assertGreater(results[1], 0)

    def test_multiple_workers(self):
        from depthforge.core.parallel import benchmark

        results = benchmark(H=64, W=96, n_workers_list=[1, 2])
        self.assertEqual(set(results.keys()), {1, 2})


# ════════════════════════════════════════════════════════════════════════════
# 2. core.optical_flow
# ════════════════════════════════════════════════════════════════════════════


class TestFlowDepthConfig(unittest.TestCase):

    def test_defaults(self):
        from depthforge.core.optical_flow import FlowDepthConfig

        cfg = FlowDepthConfig()
        self.assertEqual(cfg.pyr_scale, 0.5)
        self.assertFalse(cfg.invert)

    def test_custom(self):
        from depthforge.core.optical_flow import FlowDepthConfig

        cfg = FlowDepthConfig(motion_scale=2.0, invert=True, max_motion=10.0)
        self.assertEqual(cfg.motion_scale, 2.0)
        self.assertTrue(cfg.invert)


class TestComputeFlow(unittest.TestCase):

    def setUp(self):
        self.a = _frame()
        self.b = _frame(seed=2)
        from depthforge.core.optical_flow import FlowDepthConfig

        self.cfg = FlowDepthConfig()

    def test_output_shape(self):
        from depthforge.core.optical_flow import compute_flow

        flow = compute_flow(self.a, self.b, self.cfg)
        H, W = self.a.shape[:2]
        self.assertEqual(flow.shape, (H, W, 2))

    def test_output_dtype(self):
        from depthforge.core.optical_flow import compute_flow

        flow = compute_flow(self.a, self.b, self.cfg)
        self.assertEqual(flow.dtype, np.float32)

    def test_identical_frames_near_zero(self):
        from depthforge.core.optical_flow import compute_flow

        flow = compute_flow(self.a, self.a, self.cfg)
        self.assertLess(float(np.abs(flow).max()), 1.0)

    def test_float_frames_accepted(self):
        from depthforge.core.optical_flow import compute_flow

        af = self.a.astype(np.float32) / 255.0
        bf = self.b.astype(np.float32) / 255.0
        flow = compute_flow(af, bf, self.cfg)
        self.assertEqual(flow.shape[:2], self.a.shape[:2])

    def test_rgba_frames_accepted(self):
        from depthforge.core.optical_flow import compute_flow

        rgba = np.concatenate(
            [self.a, np.full((*self.a.shape[:2], 1), 255, dtype=np.uint8)], axis=-1
        )
        flow = compute_flow(rgba, rgba, self.cfg)
        self.assertEqual(flow.shape[:2], self.a.shape[:2])

    def test_grey_frames_accepted(self):
        from depthforge.core.optical_flow import compute_flow

        grey = self.a[:, :, 0]
        flow = compute_flow(grey, grey, self.cfg)
        self.assertEqual(flow.shape[:2], grey.shape)


class TestFlowToDepth(unittest.TestCase):

    def _make_flow(self, H=64, W=96, mag=5.0, seed=0):
        rng = np.random.default_rng(seed)
        flow = rng.random((H, W, 2), dtype=np.float32) * mag
        return flow

    def test_output_shape(self):
        from depthforge.core.optical_flow import FlowDepthConfig, flow_to_depth

        flow = self._make_flow()
        depth = flow_to_depth(flow, FlowDepthConfig())
        self.assertEqual(depth.shape, (64, 96))

    def test_output_range(self):
        from depthforge.core.optical_flow import FlowDepthConfig, flow_to_depth

        flow = self._make_flow()
        depth = flow_to_depth(flow, FlowDepthConfig())
        self.assertGreaterEqual(float(depth.min()), 0.0)
        self.assertLessEqual(float(depth.max()), 1.0)

    def test_zero_flow_gives_zero_depth(self):
        from depthforge.core.optical_flow import FlowDepthConfig, flow_to_depth

        flow = np.zeros((64, 96, 2), dtype=np.float32)
        depth = flow_to_depth(flow, FlowDepthConfig())
        self.assertEqual(float(depth.max()), 0.0)

    def test_invert_flag(self):
        from depthforge.core.optical_flow import FlowDepthConfig, flow_to_depth

        flow = self._make_flow()
        normal = flow_to_depth(flow, FlowDepthConfig(invert=False))
        inverted = flow_to_depth(flow, FlowDepthConfig(invert=True))
        np.testing.assert_allclose(normal + inverted, np.ones_like(normal), atol=1e-5)

    def test_dtype_float32(self):
        from depthforge.core.optical_flow import FlowDepthConfig, flow_to_depth

        flow = self._make_flow()
        depth = flow_to_depth(flow, FlowDepthConfig())
        self.assertEqual(depth.dtype, np.float32)

    def test_max_motion_clamp(self):
        from depthforge.core.optical_flow import FlowDepthConfig, flow_to_depth

        flow = self._make_flow(mag=100.0)
        depth = flow_to_depth(flow, FlowDepthConfig(max_motion=10.0))
        self.assertLessEqual(float(depth.max()), 1.0)


class TestFlowWarp(unittest.TestCase):

    def test_output_shape(self):
        from depthforge.core.optical_flow import flow_warp

        frame = _frame()
        flow = np.zeros((*frame.shape[:2], 2), dtype=np.float32)
        warped = flow_warp(frame, flow)
        self.assertEqual(warped.shape, frame.shape)

    def test_zero_flow_identity(self):
        from depthforge.core.optical_flow import flow_warp

        frame = _frame()
        flow = np.zeros((*frame.shape[:2], 2), dtype=np.float32)
        warped = flow_warp(frame, flow)
        np.testing.assert_array_equal(frame, warped)


class TestFlowDepthEstimator(unittest.TestCase):

    def setUp(self):
        from depthforge.core.optical_flow import FlowDepthEstimator

        self.est = FlowDepthEstimator()
        self.fa = _frame()
        self.fb = _frame(seed=5)

    def test_first_frame_returns_none(self):
        self.assertIsNone(self.est.feed(self.fa))

    def test_second_frame_returns_depth(self):
        self.est.feed(self.fa)
        depth = self.est.feed(self.fb)
        self.assertIsNotNone(depth)
        self.assertEqual(depth.shape, self.fa.shape[:2])

    def test_reset_clears_state(self):
        self.est.feed(self.fa)
        self.est.reset()
        self.assertIsNone(self.est.feed(self.fb))

    def test_estimate_from_pair(self):
        depth = self.est.estimate_from_pair(self.fa, self.fb)
        self.assertEqual(depth.shape, self.fa.shape[:2])

    def test_estimate_sequence_length(self):
        frames = [_frame(seed=i) for i in range(5)]
        depths = self.est.estimate_sequence(frames, pad_first=True)
        self.assertEqual(len(depths), 5)

    def test_estimate_sequence_no_pad(self):
        frames = [_frame(seed=i) for i in range(5)]
        depths = self.est.estimate_sequence(frames, pad_first=False)
        self.assertEqual(len(depths), 4)

    def test_sequence_range(self):
        frames = [_frame(seed=i) for i in range(4)]
        depths = self.est.estimate_sequence(frames)
        for d in depths:
            self.assertGreaterEqual(float(d.min()), 0.0)
            self.assertLessEqual(float(d.max()), 1.0)


class TestDetectSceneCut(unittest.TestCase):

    def test_identical_not_cut(self):
        from depthforge.core.optical_flow import detect_scene_cut

        f = _frame()
        self.assertFalse(detect_scene_cut(f, f))

    def test_different_is_cut(self):
        from depthforge.core.optical_flow import detect_scene_cut

        a = np.zeros((64, 96, 3), dtype=np.uint8)
        b = np.full((64, 96, 3), 255, dtype=np.uint8)
        self.assertTrue(detect_scene_cut(a, b))

    def test_threshold_respected(self):
        from depthforge.core.optical_flow import detect_scene_cut

        a = np.zeros((64, 96, 3), dtype=np.uint8)
        b = np.full((64, 96, 3), 50, dtype=np.uint8)
        self.assertFalse(detect_scene_cut(a, b, threshold=0.5))
        self.assertTrue(detect_scene_cut(a, b, threshold=0.1))


# ════════════════════════════════════════════════════════════════════════════
# 3. core.temporal
# ════════════════════════════════════════════════════════════════════════════


class TestTemporalConfig(unittest.TestCase):

    def test_default_strategy(self):
        from depthforge.core.temporal import TemporalConfig

        cfg = TemporalConfig()
        self.assertEqual(cfg.strategy, "ema")

    def test_invalid_strategy_raises(self):
        from depthforge.core.temporal import TemporalConfig

        with self.assertRaises(ValueError):
            TemporalConfig(strategy="banana")

    def test_alpha_clamped(self):
        from depthforge.core.temporal import TemporalConfig

        cfg = TemporalConfig(alpha=5.0)
        self.assertLessEqual(cfg.alpha, 1.0)

    def test_valid_strategies(self):
        from depthforge.core.temporal import TemporalConfig

        for s in ("ema", "windowed", "flow_guided", "adaptive"):
            TemporalConfig(strategy=s)  # must not raise


class TestTemporalSmootherEMA(unittest.TestCase):

    def setUp(self):
        from depthforge.core.temporal import TemporalConfig, TemporalSmoother

        self.smoother = TemporalSmoother(
            TemporalConfig(strategy="ema", alpha=0.5, scene_cut_reset=False)
        )

    def test_first_frame_passes_through(self):
        d = _depth()
        r = self.smoother.update(d)
        np.testing.assert_array_almost_equal(r, d, decimal=5)

    def test_smoothing_reduces_delta(self):
        d1 = np.zeros((64, 96), dtype=np.float32)
        d2 = np.ones((64, 96), dtype=np.float32)
        self.smoother.update(d1)
        r = self.smoother.update(d2)
        # Alpha=0.5: result = 0.5*d1 + 0.5*d2 = 0.5
        np.testing.assert_allclose(r, 0.5, atol=1e-5)

    def test_frame_count(self):
        for _ in range(5):
            self.smoother.update(_depth())
        self.assertEqual(self.smoother.frame_count, 5)

    def test_reset_clears_count(self):
        for _ in range(3):
            self.smoother.update(_depth())
        self.smoother.reset()
        self.assertEqual(self.smoother.frame_count, 0)


class TestTemporalSmootherWindowed(unittest.TestCase):

    def setUp(self):
        from depthforge.core.temporal import TemporalConfig, TemporalSmoother

        self.smoother = TemporalSmoother(
            TemporalConfig(strategy="windowed", window_size=3, scene_cut_reset=False)
        )

    def test_output_shape(self):
        d = _depth()
        r = self.smoother.update(d)
        self.assertEqual(r.shape, d.shape)

    def test_window_mean(self):
        d1 = np.full((64, 96), 0.2, dtype=np.float32)
        d2 = np.full((64, 96), 0.6, dtype=np.float32)
        d3 = np.full((64, 96), 1.0, dtype=np.float32)
        self.smoother.update(d1)
        self.smoother.update(d2)
        r = self.smoother.update(d3)
        np.testing.assert_allclose(r, 0.6, atol=1e-5)


class TestTemporalSmootherFlowGuided(unittest.TestCase):

    def setUp(self):
        from depthforge.core.temporal import TemporalConfig, TemporalSmoother

        self.smoother = TemporalSmoother(
            TemporalConfig(strategy="flow_guided", scene_cut_reset=False)
        )

    def test_first_frame(self):
        d = _depth()
        r = self.smoother.update(d, frame=_frame())
        self.assertEqual(r.shape, d.shape)

    def test_second_frame_blends(self):
        d1 = _depth()
        d2 = _depth(seed=7)
        self.smoother.update(d1, frame=_frame())
        r = self.smoother.update(d2, frame=_frame(seed=2))
        self.assertEqual(r.shape, d1.shape)
        self.assertEqual(r.dtype, np.float32)


class TestTemporalSmootherAdaptive(unittest.TestCase):

    def test_adaptive_runs(self):
        from depthforge.core.temporal import TemporalConfig, TemporalSmoother

        smoother = TemporalSmoother(TemporalConfig(strategy="adaptive", scene_cut_reset=False))
        for i in range(5):
            r = smoother.update(_depth(seed=i), frame=_frame(seed=i))
            self.assertEqual(r.shape, (64, 96))


class TestSceneCutReset(unittest.TestCase):

    def test_scene_cut_resets_state(self):
        from depthforge.core.temporal import TemporalConfig, TemporalSmoother

        smoother = TemporalSmoother(
            TemporalConfig(strategy="ema", alpha=0.1, scene_cut_reset=True, scene_cut_threshold=0.1)
        )
        # Warm up with black frames
        black = np.zeros((64, 96, 3), dtype=np.uint8)
        d_black = np.zeros((64, 96), dtype=np.float32)
        for _ in range(5):
            smoother.update(d_black, frame=black)

        # Scene cut: pure white frame
        white = np.full((64, 96, 3), 255, dtype=np.uint8)
        d_white = np.ones((64, 96), dtype=np.float32)
        result = smoother.update(d_white, frame=white)

        # After reset, EMA should have re-initialised to d_white
        np.testing.assert_allclose(result, 1.0, atol=1e-4)


class TestDepthHistory(unittest.TestCase):

    def setUp(self):
        from depthforge.core.temporal import DepthHistory

        self.hist = DepthHistory(capacity=5)

    def test_push_and_len(self):
        for i in range(3):
            self.hist.push(_depth(seed=i))
        self.assertEqual(len(self.hist), 3)

    def test_full_flag(self):
        self.assertFalse(self.hist.full)
        for i in range(5):
            self.hist.push(_depth(seed=i))
        self.assertTrue(self.hist.full)

    def test_mean_shape(self):
        for i in range(5):
            self.hist.push(_depth(seed=i))
        m = self.hist.mean()
        self.assertEqual(m.shape, (64, 96))

    def test_mean_empty_returns_none(self):
        self.assertIsNone(self.hist.mean())

    def test_gaussian_smooth_shape(self):
        for i in range(5):
            self.hist.push(_depth(seed=i))
        g = self.hist.gaussian_smooth(sigma=1.0)
        self.assertEqual(g.shape, (64, 96))

    def test_gaussian_smooth_range(self):
        for i in range(5):
            self.hist.push(_depth(seed=i))
        g = self.hist.gaussian_smooth()
        self.assertGreaterEqual(float(g.min()), 0.0)
        self.assertLessEqual(float(g.max()), 1.0)

    def test_clear(self):
        for i in range(3):
            self.hist.push(_depth(seed=i))
        self.hist.clear()
        self.assertEqual(len(self.hist), 0)

    def test_capacity_overflow(self):
        for i in range(10):
            self.hist.push(_depth(seed=i))
        self.assertEqual(len(self.hist), 5)


# ════════════════════════════════════════════════════════════════════════════
# 4. core.export
# ════════════════════════════════════════════════════════════════════════════


class TestExportProfile(unittest.TestCase):

    def test_defaults(self):
        from depthforge.core.export import ExportProfile

        p = ExportProfile()
        self.assertEqual(p.format, ".png")
        self.assertEqual(p.dpi, 72)
        self.assertEqual(p.bit_depth, 8)

    def test_invalid_bit_depth_raises(self):
        from depthforge.core.export import ExportProfile

        with self.assertRaises(ValueError):
            ExportProfile(bit_depth=24)

    def test_invalid_format_raises(self):
        from depthforge.core.export import ExportProfile

        with self.assertRaises(ValueError):
            ExportProfile(format=".xyz")

    def test_16bit_jpeg_raises(self):
        from depthforge.core.export import ExportProfile

        with self.assertRaises(ValueError):
            ExportProfile(format=".jpg", bit_depth=16)

    def test_to_dict_round_trip(self):
        from depthforge.core.export import ExportProfile

        p = ExportProfile(name="test", dpi=300, bit_depth=16, format=".tiff")
        d = p.to_dict()
        p2 = ExportProfile.from_dict(d)
        self.assertEqual(p.dpi, p2.dpi)
        self.assertEqual(p.bit_depth, p2.bit_depth)

    def test_to_json_round_trip(self):
        from depthforge.core.export import ExportProfile

        p = ExportProfile(name="j", dpi=150)
        j = p.to_json()
        p2 = ExportProfile.from_dict(__import__("json").loads(j))
        self.assertEqual(p2.dpi, 150)


class TestBuiltinProfiles(unittest.TestCase):

    def test_list_profiles_nonempty(self):
        from depthforge.core.export import list_profiles

        self.assertGreater(len(list_profiles()), 0)

    def test_all_builtin_accessible(self):
        from depthforge.core.export import get_profile, list_profiles

        for name in list_profiles():
            p = get_profile(name)
            self.assertIsNotNone(p)

    def test_unknown_profile_raises(self):
        from depthforge.core.export import get_profile

        with self.assertRaises(KeyError):
            get_profile("does_not_exist")

    def test_register_custom(self):
        from depthforge.core.export import ExportProfile, get_profile, register_profile

        p = ExportProfile(name="test_custom", dpi=96)
        register_profile(p)
        self.assertEqual(get_profile("test_custom").dpi, 96)

    def test_web_srgb_profile(self):
        from depthforge.core.export import get_profile

        p = get_profile("web_srgb")
        self.assertEqual(p.dpi, 72)
        self.assertEqual(p.format, ".png")

    def test_print_300_profile(self):
        from depthforge.core.export import get_profile

        p = get_profile("print_300")
        self.assertEqual(p.dpi, 300)
        self.assertEqual(p.bit_depth, 16)

    def test_broadcast_profile_dimensions(self):
        from depthforge.core.export import get_profile

        p = get_profile("broadcast_rec709")
        self.assertEqual(p.target_width, 1920)
        self.assertEqual(p.target_height, 1080)


class TestExporter(unittest.TestCase):

    def _stereo(self, H=64, W=96):
        rng = np.random.default_rng(0)
        return (rng.random((H, W, 4)) * 255).astype(np.uint8)

    def test_export_png(self):
        from depthforge.core.export import Exporter

        e = Exporter("web_srgb")
        with tempfile.TemporaryDirectory() as td:
            path = e.export(self._stereo(), os.path.join(td, "test"))
            self.assertTrue(os.path.exists(path))
            self.assertTrue(path.endswith(".png"))

    def test_export_tiff(self):
        from depthforge.core.export import Exporter

        e = Exporter("print_300")
        with tempfile.TemporaryDirectory() as td:
            path = e.export(self._stereo(), os.path.join(td, "test"))
            self.assertTrue(os.path.exists(path))

    def test_export_jpeg(self):
        from depthforge.core.export import Exporter, ExportProfile

        p = ExportProfile(name="jpeg_test", format=".jpg", dpi=72)
        e = Exporter(p)
        with tempfile.TemporaryDirectory() as td:
            path = e.export(self._stereo(), os.path.join(td, "test"))
            self.assertTrue(os.path.exists(path))

    def test_export_with_explicit_extension(self):
        from depthforge.core.export import Exporter

        e = Exporter("web_srgb")
        with tempfile.TemporaryDirectory() as td:
            full = os.path.join(td, "test.png")
            path = e.export(self._stereo(), full)
            self.assertTrue(os.path.exists(path))

    def test_resize(self):
        from PIL import Image

        from depthforge.core.export import Exporter, ExportProfile

        p = ExportProfile(name="resized", format=".png", target_width=32, target_height=16)
        e = Exporter(p)
        with tempfile.TemporaryDirectory() as td:
            path = e.export(self._stereo(), os.path.join(td, "test"))
            img = Image.open(path)
            self.assertEqual(img.size, (32, 16))

    def test_watermark(self):
        from depthforge.core.export import Exporter, ExportProfile

        p = ExportProfile(name="wm", format=".png", watermark="DepthForge")
        e = Exporter(p)
        with tempfile.TemporaryDirectory() as td:
            path = e.export(self._stereo(), os.path.join(td, "test"))
            self.assertTrue(os.path.exists(path))

    def test_export_string_profile_name(self):
        from depthforge.core.export import Exporter

        e = Exporter("web_srgb")
        self.assertEqual(e.profile.name, "web_srgb")

    def test_float_input_converted(self):
        from depthforge.core.export import Exporter

        e = Exporter("web_srgb")
        flt = np.random.default_rng(0).random((64, 96, 4)).astype(np.float32)
        with tempfile.TemporaryDirectory() as td:
            path = e.export(flt, os.path.join(td, "float_test"))
            self.assertTrue(os.path.exists(path))

    def test_batch_export(self):
        from depthforge.core.export import Exporter

        e = Exporter("web_srgb")
        items = [(self._stereo(), f"frame_{i:03d}") for i in range(3)]
        with tempfile.TemporaryDirectory() as td:
            paths = e.export_batch(items, td)
            self.assertEqual(len(paths), 3)
            for p in paths:
                self.assertTrue(os.path.exists(p))

    def test_batch_parallel(self):
        from depthforge.core.export import Exporter

        e = Exporter("web_srgb")
        items = [(self._stereo(), f"par_{i:03d}") for i in range(4)]
        with tempfile.TemporaryDirectory() as td:
            paths = e.export_batch(items, td, n_workers=2)
            self.assertEqual(len(paths), 4)


class TestExportQueue(unittest.TestCase):

    def _stereo(self):
        return (np.random.default_rng(0).random((64, 96, 4)) * 255).astype(np.uint8)

    def test_enqueue_and_shutdown(self):
        from depthforge.core.export import ExportQueue

        with tempfile.TemporaryDirectory() as td:
            q = ExportQueue("web_srgb", td, n_workers=2)
            for i in range(3):
                q.enqueue(self._stereo(), f"q_frame_{i:03d}")
            paths = q.shutdown()
            self.assertEqual(len(paths), 3)

    def test_pending_count(self):
        from depthforge.core.export import ExportQueue

        with tempfile.TemporaryDirectory() as td:
            q = ExportQueue("web_srgb", td, n_workers=1)
            self.assertEqual(q.pending, 0)

    def test_completed_after_shutdown(self):
        from depthforge.core.export import ExportQueue

        with tempfile.TemporaryDirectory() as td:
            q = ExportQueue("web_srgb", td, n_workers=1)
            q.enqueue(self._stereo(), "qc_frame_001")
            q.shutdown()
            self.assertEqual(q.completed, 1)


# ════════════════════════════════════════════════════════════════════════════
# 5. ofx module
# ════════════════════════════════════════════════════════════════════════════


class TestOFXParams(unittest.TestCase):

    def test_defaults(self):
        from depthforge.ofx import OFXParams

        p = OFXParams()
        self.assertEqual(p.mode, "sirds")
        self.assertAlmostEqual(p.depth_factor, 0.35)

    def test_to_cli_args_basic(self):
        from depthforge.ofx import OFXParams

        p = OFXParams(depth_factor=0.4, seed=7)
        args = p.to_cli_args()
        self.assertIn("--depth-factor", args)
        df_idx = args.index("--depth-factor")
        self.assertAlmostEqual(float(args[df_idx + 1]), 0.4, places=4)
        self.assertIn("--seed", args)
        seed_idx = args.index("--seed")
        self.assertEqual(args[seed_idx + 1], "7")

    def test_bool_flags(self):
        from depthforge.ofx import OFXParams

        p = OFXParams(safe_mode=True, invert_depth=True, swap_eyes=True)
        args = p.to_cli_args()
        self.assertIn("--safe-mode", args)
        self.assertIn("--invert-depth", args)
        self.assertIn("--swap-eyes", args)

    def test_false_bools_omitted(self):
        from depthforge.ofx import OFXParams

        p = OFXParams(safe_mode=False)
        args = p.to_cli_args()
        self.assertNotIn("--safe-mode", args)

    def test_preset_mode(self):
        from depthforge.ofx import OFXParams

        p = OFXParams(preset="cinema")
        args = p.to_cli_args()
        self.assertIn("--preset", args)
        self.assertEqual(args[args.index("--preset") + 1], "cinema")
        # Should not emit individual params when preset given
        self.assertNotIn("--depth-factor", args)

    def test_from_preset(self):
        from depthforge.ofx import OFXParams

        p = OFXParams.from_preset("deep")
        self.assertEqual(p.preset, "deep")

    def test_to_dict(self):
        from depthforge.ofx import OFXParams

        p = OFXParams(seed=99)
        d = p.to_dict()
        self.assertEqual(d["seed"], 99)

    def test_to_json(self):
        import json

        from depthforge.ofx import OFXParams

        p = OFXParams(seed=77)
        j = p.to_json()
        d = json.loads(j)
        self.assertEqual(d["seed"], 77)


class TestOFXBridgeDirect(unittest.TestCase):
    """Test OFXBridge.synthesize_direct (in-process, no subprocess)."""

    def setUp(self):
        from depthforge.ofx import OFXBridge

        self.bridge = OFXBridge()
        self.depth = _depth()

    def test_returns_rgba(self):
        result = self.bridge.synthesize_direct(self.depth)
        self.assertEqual(result.ndim, 3)
        self.assertEqual(result.shape[2], 4)

    def test_output_shape(self):
        result = self.bridge.synthesize_direct(self.depth)
        self.assertEqual(result.shape[:2], self.depth.shape)

    def test_output_dtype_uint8(self):
        result = self.bridge.synthesize_direct(self.depth)
        self.assertEqual(result.dtype, np.uint8)

    def test_with_params(self):
        from depthforge.ofx import OFXParams

        params = OFXParams(depth_factor=0.2, seed=7)
        result = self.bridge.synthesize_direct(self.depth, params=params)
        self.assertEqual(result.shape[:2], self.depth.shape)

    def test_preset_params(self):
        from depthforge.ofx import OFXParams

        params = OFXParams.from_preset("shallow")
        result = self.bridge.synthesize_direct(self.depth, params=params)
        self.assertEqual(result.shape[:2], self.depth.shape)


class TestOFXModuleUtils(unittest.TestCase):

    def test_check_build_returns_dict(self):
        from depthforge.ofx import check_build

        r = check_build()
        self.assertIn("built", r)
        self.assertIn("path", r)
        self.assertIn("error", r)

    def test_plugin_count(self):
        from depthforge.ofx import plugin_count

        self.assertEqual(plugin_count(), 1)


# ════════════════════════════════════════════════════════════════════════════
# 6. Integration: parallel + temporal + export pipeline
# ════════════════════════════════════════════════════════════════════════════


class TestPhase5Pipeline(unittest.TestCase):
    """End-to-end Phase 5 pipeline: flow → temporal → parallel → export."""

    def test_flow_temporal_export(self):
        from depthforge.core.export import Exporter
        from depthforge.core.optical_flow import FlowDepthEstimator
        from depthforge.core.parallel import ParallelConfig, parallel_synthesize
        from depthforge.core.temporal import TemporalConfig, TemporalSmoother

        est = FlowDepthEstimator()
        smoother = TemporalSmoother(
            TemporalConfig(strategy="ema", alpha=0.4, scene_cut_reset=False)
        )
        exporter = Exporter("web_srgb")
        pattern = _pattern()
        params = _stereo_params()
        cfg = ParallelConfig(n_workers=2, chunk_rows=16, min_rows_for_parallel=1)

        frames = [_frame(seed=i) for i in range(4)]
        paths = []

        with tempfile.TemporaryDirectory() as td:
            for i, frame in enumerate(frames):
                raw_depth = est.feed(frame)
                if raw_depth is None:
                    continue
                smooth = smoother.update(raw_depth, frame=frame)
                stereo = parallel_synthesize(smooth, pattern, params, cfg)
                path = exporter.export(stereo, os.path.join(td, f"frame_{i:03d}"))
                paths.append(path)

            self.assertGreater(len(paths), 0)
            for p in paths:
                self.assertTrue(os.path.exists(p))

    def test_ofx_to_export_pipeline(self):
        from depthforge.core.export import Exporter
        from depthforge.ofx import OFXBridge, OFXParams

        bridge = OFXBridge()
        exporter = Exporter("web_srgb")
        depth = _depth(96, 128)
        params = OFXParams(depth_factor=0.3)

        stereo = bridge.synthesize_direct(depth, params=params)

        with tempfile.TemporaryDirectory() as td:
            path = exporter.export(stereo, os.path.join(td, "pipeline_out"))
            self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
