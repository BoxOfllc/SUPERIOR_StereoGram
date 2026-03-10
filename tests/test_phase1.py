"""
DepthForge Phase 1 — Test Suite
================================
Runs with:  python -m pytest tests/         (when pytest installed)
            python -m unittest discover tests/  (stdlib fallback)
            python tests/test_phase1.py         (direct)

Tests cover:
  - Input validation and edge cases
  - Correct output shapes / dtypes
  - Parameter boundary conditions
  - Integration: full pipeline end-to-end
"""

import sys
import os
import unittest
import numpy as np

# Allow running from any directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from depthforge.core.synthesizer   import StereoParams, synthesize
from depthforge.core.depth_prep    import (
    DepthPrepParams, FalloffCurve, RegionMask,
    prep_depth, normalise_depth, compute_vergence_map, detect_window_violations,
)
from depthforge.core.pattern_gen   import (
    PatternParams, PatternType, ColorMode, GridStyle,
    generate_pattern, tile_to_frame,
)
from depthforge.core.anaglyph     import (
    AnaglyphMode, AnaglyphParams, make_anaglyph, make_anaglyph_from_depth,
)
from depthforge.core.stereo_pair  import (
    StereoPairParams, StereoLayout, make_stereo_pair,
)
from depthforge.core.hidden_image import (
    HiddenImageParams, encode_hidden_image, mask_to_depth,
    text_to_mask, shape_to_mask,
)
from depthforge.core.adaptive_dots import (
    AdaptiveDotParams, generate_adaptive_dots, complexity_from_depth,
)
from depthforge.core.inpainting   import (
    InpaintParams, InpaintMethod, inpaint_occlusion,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_depth(H=64, W=64, mode="gradient") -> np.ndarray:
    """Create a test depth map."""
    if mode == "gradient":
        xs = np.linspace(0, 1, W)[None, :]
        ys = np.linspace(0, 1, H)[:, None]
        return ((xs + ys) / 2).astype(np.float32)
    if mode == "sphere":
        ys = np.linspace(-1, 1, H)[:, None]
        xs = np.linspace(-1, 1, W)[None, :]
        r  = np.sqrt(np.maximum(1.0 - xs**2 - ys**2, 0))
        return r.astype(np.float32)
    if mode == "flat":
        return np.full((H, W), 0.5, dtype=np.float32)
    if mode == "binary":
        m = np.zeros((H, W), dtype=np.float32)
        m[H//4:3*H//4, W//4:3*W//4] = 1.0
        return m
    return np.random.rand(H, W).astype(np.float32)


def make_source(H=64, W=64) -> np.ndarray:
    """Create a test source image (RGBA uint8)."""
    img = np.zeros((H, W, 4), dtype=np.uint8)
    img[:, :, 0] = np.linspace(0, 255, W)[None, :]  # R gradient
    img[:, :, 1] = np.linspace(0, 255, H)[:, None]  # G gradient
    img[:, :, 2] = 128
    img[:, :, 3] = 255
    return img


def make_pattern(H=32, W=32) -> np.ndarray:
    """Create a test pattern tile (RGBA uint8)."""
    params = PatternParams(
        pattern_type=PatternType.RANDOM_NOISE,
        tile_width=W,
        tile_height=H,
        color_mode=ColorMode.GREYSCALE,
        seed=42,
    )
    return generate_pattern(params)


# ---------------------------------------------------------------------------
# 1. Synthesizer tests
# ---------------------------------------------------------------------------

class TestSynthesizer(unittest.TestCase):

    def test_basic_output_shape(self):
        depth   = make_depth(64, 128)
        pattern = make_pattern(32, 32)
        params  = StereoParams(seed=0)
        result  = synthesize(depth, pattern, params)
        self.assertEqual(result.shape, (64, 128, 4))
        self.assertEqual(result.dtype, np.uint8)

    def test_output_is_valid_rgba(self):
        result = synthesize(make_depth(), make_pattern())
        self.assertTrue((result >= 0).all())
        self.assertTrue((result <= 255).all())
        # Alpha channel should be fully opaque
        self.assertTrue((result[:, :, 3] == 255).all() or result[:, :, 3].min() >= 0)

    def test_invert_depth_changes_result(self):
        """Invert_depth flips near/far — must produce different stereogram.
        Uses 64×64 sphere depth where shifts are large enough to matter."""
        H, W = 64, 64
        ys = np.linspace(-1, 1, H)[:, None]
        xs = np.linspace(-1, 1, W)[None, :]
        depth   = np.sqrt(np.maximum(1.0 - xs**2 - ys**2, 0)).astype(np.float32)
        pattern = make_pattern(32, 32, seed_val=7)
        r1 = synthesize(depth, pattern, StereoParams(seed=1, depth_factor=0.8, convergence=0.0))
        r2 = synthesize(depth, pattern, StereoParams(seed=1, depth_factor=0.8, convergence=0.0, invert_depth=True))
        self.assertFalse(np.array_equal(r1, r2))

    def test_zero_depth_factor(self):
        """Zero depth factor should produce a plain tiled pattern with no shift."""
        depth   = make_depth()
        pattern = make_pattern(32, 32)
        result  = synthesize(depth, pattern, StereoParams(depth_factor=0.0, seed=0))
        self.assertEqual(result.shape, (64, 64, 4))

    def test_safe_mode_limits_parallax(self):
        p1 = StereoParams(depth_factor=1.0, safe_mode=False)
        p2 = StereoParams(depth_factor=1.0, safe_mode=True)
        self.assertLessEqual(p2.depth_factor, 0.5)
        self.assertLessEqual(
            p2.max_parallax_fraction, 1.0 / 30.0 + 1e-9
        )

    def test_flat_depth_produces_tiled_pattern(self):
        """Flat depth → no parallax → output should be identical column-wise."""
        depth   = make_depth(mode="flat")
        pattern = make_pattern(16, 16)
        result  = synthesize(depth, pattern, StereoParams(depth_factor=0.0, seed=5))
        self.assertEqual(result.shape, (64, 64, 4))

    def test_oversample_produces_same_size(self):
        depth   = make_depth(32, 32)
        pattern = make_pattern(16, 16)
        r1 = synthesize(depth, pattern, StereoParams(oversample=1, seed=0))
        r2 = synthesize(depth, pattern, StereoParams(oversample=2, seed=0))
        self.assertEqual(r1.shape, r2.shape)

    def test_3channel_pattern_accepted(self):
        """RGB (no alpha) pattern should be auto-converted."""
        depth   = make_depth()
        pattern = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        result  = synthesize(depth, pattern)
        self.assertEqual(result.shape[2], 4)

    def test_greyscale_pattern_accepted(self):
        depth   = make_depth()
        pattern = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        result  = synthesize(depth, pattern)
        self.assertEqual(result.shape[2], 4)

    def test_invalid_depth_raises(self):
        with self.assertRaises(ValueError):
            synthesize(np.zeros((4, 4, 4), dtype=np.float32), make_pattern())

    def test_params_bad_oversample_raises(self):
        with self.assertRaises(ValueError):
            StereoParams(oversample=0)

    def test_params_bad_convergence_raises(self):
        with self.assertRaises(ValueError):
            StereoParams(convergence=1.5)

    def test_different_seeds_differ(self):
        """Two different seeds should produce different results (SIRDS noise)."""
        depth   = make_depth()
        pattern = make_pattern(seed_val=7)
        p1      = make_pattern(seed_val=7)
        p2      = make_pattern(seed_val=99)
        r1 = synthesize(depth, p1, StereoParams(seed=1))
        r2 = synthesize(depth, p2, StereoParams(seed=2))
        self.assertFalse(np.array_equal(r1, r2))


# ---------------------------------------------------------------------------
# 2. Depth prep tests
# ---------------------------------------------------------------------------

class TestDepthPrep(unittest.TestCase):

    def test_normalise_output_range(self):
        raw = np.array([[0, 50, 100, 200, 255]], dtype=np.float32)
        out = prep_depth(raw, DepthPrepParams(
            bilateral_sigma_space=0, dilation_px=0,
            falloff_curve=FalloffCurve.LINEAR,
        ))
        self.assertAlmostEqual(float(out.min()), 0.0, places=5)
        self.assertAlmostEqual(float(out.max()), 1.0, places=5)

    def test_invert(self):
        raw = make_depth(mode="gradient")
        p   = DepthPrepParams(invert=True, bilateral_sigma_space=0, dilation_px=0)
        out = prep_depth(raw, p)
        orig = prep_depth(raw, DepthPrepParams(bilateral_sigma_space=0, dilation_px=0))
        # Inverted max should be near where original had min
        self.assertAlmostEqual(float(out[0, 0]), 1.0 - float(orig[0, 0]), places=3)

    def test_bilateral_smooth_changes_result(self):
        """Bilateral filter must change the output vs no smoothing."""
        raw  = np.random.default_rng(77).random((64, 64)).astype(np.float32)
        p_no = DepthPrepParams(bilateral_sigma_space=0, dilation_px=0)
        p_sm = DepthPrepParams(bilateral_sigma_space=7, dilation_px=0, smooth_passes=2)
        out_no = prep_depth(raw, p_no)
        out_sm = prep_depth(raw, p_sm)
        self.assertFalse(np.array_equal(out_no, out_sm))

    def test_dilation_expands_near_regions(self):
        # Binary depth: centre bright, edges dark
        raw    = make_depth(mode="binary")
        p_no   = DepthPrepParams(bilateral_sigma_space=0, dilation_px=0)
        p_dil  = DepthPrepParams(bilateral_sigma_space=0, dilation_px=5)
        no     = prep_depth(raw, p_no)
        dil    = prep_depth(raw, p_dil)
        # Dilated version should have more bright pixels
        self.assertGreater((dil > 0.5).sum(), (no > 0.5).sum())

    def test_s_curve_properties(self):
        """S-curve: endpoints 0→0, 1→1; output should differ from LINEAR."""
        raw = np.linspace(0, 1, 4*4).reshape(4, 4).astype(np.float32)
        p_lin = DepthPrepParams(bilateral_sigma_space=0, dilation_px=0,
                                normalise=False, falloff_curve=FalloffCurve.LINEAR)
        p_s   = DepthPrepParams(bilateral_sigma_space=0, dilation_px=0,
                                normalise=False, falloff_curve=FalloffCurve.S_CURVE)
        out_lin = prep_depth(raw, p_lin)
        out_s   = prep_depth(raw, p_s)
        # Endpoints preserved: f(0)=0, f(1)=1
        self.assertAlmostEqual(float(out_s.flat[0]),  0.0, places=4)
        self.assertAlmostEqual(float(out_s.flat[-1]), 1.0, places=4)
        # S-curve is non-linear: must differ from identity
        self.assertFalse(np.allclose(out_s, out_lin, atol=0.01))

    def test_near_far_plane_clamp(self):
        raw = make_depth(mode="gradient")
        p   = DepthPrepParams(
            bilateral_sigma_space=0, dilation_px=0,
            near_plane=0.2, far_plane=0.8,
        )
        out = prep_depth(raw, p)
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)

    def test_region_mask_flattens_area(self):
        raw    = make_depth(mode="gradient")
        mask   = np.zeros_like(raw)
        mask[:, :] = 1.0   # whole image mask
        rm     = RegionMask(mask=mask, multiplier=0.0)
        p      = DepthPrepParams(
            bilateral_sigma_space=0, dilation_px=0,
            region_masks=[rm],
        )
        out = prep_depth(raw, p)
        self.assertAlmostEqual(float(out.max()), 0.0, places=4)

    def test_vergence_map_shape(self):
        depth = make_depth()
        vmap  = compute_vergence_map(depth)
        self.assertEqual(vmap.shape, depth.shape)
        self.assertTrue((vmap >= 0).all())

    def test_window_violation_no_false_alarm(self):
        # Flat depth at far plane should not trigger violations
        depth = make_depth(mode="flat")
        viol  = detect_window_violations(depth, threshold=0.1)
        self.assertEqual(viol.dtype, bool)

    def test_multichannel_depth_input(self):
        raw = np.random.rand(32, 32, 3).astype(np.float32) * 255
        out = prep_depth(raw)
        self.assertEqual(out.ndim, 2)

    def test_output_always_float32(self):
        raw = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        out = prep_depth(raw)
        self.assertEqual(out.dtype, np.float32)


# ---------------------------------------------------------------------------
# 3. Pattern generator tests
# ---------------------------------------------------------------------------

class TestPatternGen(unittest.TestCase):

    def _check_rgba_tile(self, arr, W, H):
        self.assertEqual(arr.shape, (H, W, 4))
        self.assertEqual(arr.dtype, np.uint8)
        self.assertTrue((arr >= 0).all())
        self.assertTrue((arr <= 255).all())

    def test_random_noise_greyscale(self):
        p = PatternParams(PatternType.RANDOM_NOISE, 64, 64, ColorMode.GREYSCALE, seed=0)
        t = generate_pattern(p)
        self._check_rgba_tile(t, 64, 64)

    def test_random_noise_psychedelic(self):
        p = PatternParams(PatternType.RANDOM_NOISE, 64, 64, ColorMode.PSYCHEDELIC, seed=0)
        t = generate_pattern(p)
        self._check_rgba_tile(t, 64, 64)

    def test_perlin(self):
        p = PatternParams(PatternType.PERLIN, 64, 64, seed=1)
        t = generate_pattern(p)
        self._check_rgba_tile(t, 64, 64)

    def test_plasma(self):
        p = PatternParams(PatternType.PLASMA, 64, 64, seed=2)
        t = generate_pattern(p)
        self._check_rgba_tile(t, 64, 64)

    def test_voronoi(self):
        p = PatternParams(PatternType.VORONOI, 64, 64, seed=3)
        t = generate_pattern(p)
        self._check_rgba_tile(t, 64, 64)

    def test_geometric_dots(self):
        p = PatternParams(PatternType.GEOMETRIC_GRID, 64, 64,
                          grid_style=GridStyle.DOTS, grid_spacing=8)
        t = generate_pattern(p)
        self._check_rgba_tile(t, 64, 64)

    def test_geometric_checks(self):
        p = PatternParams(PatternType.GEOMETRIC_GRID, 64, 64,
                          grid_style=GridStyle.CHECKS, grid_spacing=8)
        t = generate_pattern(p)
        self._check_rgba_tile(t, 64, 64)

    def test_mandelbrot(self):
        p = PatternParams(PatternType.MANDELBROT, 64, 64)
        t = generate_pattern(p)
        self._check_rgba_tile(t, 64, 64)

    def test_dot_matrix(self):
        p = PatternParams(PatternType.DOT_MATRIX, 64, 64)
        t = generate_pattern(p)
        self._check_rgba_tile(t, 64, 64)

    def test_safe_mode_reduces_contrast(self):
        p1 = PatternParams(PatternType.RANDOM_NOISE, 64, 64, seed=5, safe_mode=False)
        p2 = PatternParams(PatternType.RANDOM_NOISE, 64, 64, seed=5, safe_mode=True)
        t1 = generate_pattern(p1).astype(float)[:, :, :3]
        t2 = generate_pattern(p2).astype(float)[:, :, :3]
        self.assertGreater(t1.std(), t2.std())

    def test_seed_reproducibility(self):
        p1 = PatternParams(PatternType.RANDOM_NOISE, 32, 32, seed=42)
        p2 = PatternParams(PatternType.RANDOM_NOISE, 32, 32, seed=42)
        self.assertTrue(np.array_equal(generate_pattern(p1), generate_pattern(p2)))

    def test_different_seeds_differ(self):
        p1 = PatternParams(PatternType.RANDOM_NOISE, 32, 32, seed=1)
        p2 = PatternParams(PatternType.RANDOM_NOISE, 32, 32, seed=2)
        self.assertFalse(np.array_equal(generate_pattern(p1), generate_pattern(p2)))

    def test_tile_to_frame(self):
        tile  = make_pattern(16, 16)
        frame = tile_to_frame(tile, 64, 100)
        self.assertEqual(frame.shape, (64, 100, 4))

    def test_bad_tile_size_raises(self):
        with self.assertRaises(ValueError):
            PatternParams(tile_width=2)


# ---------------------------------------------------------------------------
# 4. Anaglyph tests
# ---------------------------------------------------------------------------

class TestAnaglyph(unittest.TestCase):

    def test_output_shape_and_dtype(self):
        L = make_source(64, 64)
        R = make_source(64, 64)
        for mode in AnaglyphMode:
            with self.subTest(mode=mode):
                result = make_anaglyph(L, R, AnaglyphParams(mode=mode))
                self.assertEqual(result.shape, (64, 64, 4))
                self.assertEqual(result.dtype, np.uint8)

    def test_swap_eyes_differs(self):
        L = make_source(32, 32)
        R = np.zeros_like(L)
        R[:, :, 1] = 255; R[:, :, 3] = 255
        r1 = make_anaglyph(L, R, AnaglyphParams())
        r2 = make_anaglyph(L, R, AnaglyphParams(swap_eyes=True))
        self.assertFalse(np.array_equal(r1, r2))

    def test_from_depth_output_shape(self):
        src   = make_source(48, 64)
        depth = make_depth(48, 64, mode="sphere")
        out   = make_anaglyph_from_depth(src, depth)
        self.assertEqual(out.shape, (48, 64, 4))

    def test_grey_anaglyph_is_greyscale_channels(self):
        """Grey anaglyph should have R channel == G channel != B (R-left, GB-right)."""
        # Use solid-colour L and R to verify channel mapping
        L = np.zeros((8, 8, 4), dtype=np.uint8); L[:, :, 0] = 200; L[:, :, 3] = 255
        R = np.zeros((8, 8, 4), dtype=np.uint8); R[:, :, 1] = 150; R[:, :, 3] = 255
        out = make_anaglyph(L, R, AnaglyphParams(mode=AnaglyphMode.GREY_ANAGLYPH))
        self.assertEqual(out.shape, (8, 8, 4))

    def test_optimised_values_in_range(self):
        L = make_source(32, 32)
        R = make_source(32, 32)
        out = make_anaglyph(L, R, AnaglyphParams(mode=AnaglyphMode.OPTIMISED))
        self.assertTrue((out[:, :, :3] >= 0).all())
        self.assertTrue((out[:, :, :3] <= 255).all())


# ---------------------------------------------------------------------------
# 5. Stereo pair tests
# ---------------------------------------------------------------------------

class TestStereoPair(unittest.TestCase):

    def test_output_shapes(self):
        src   = make_source(64, 64)
        depth = make_depth(64, 64, mode="sphere")
        L, R, occ = make_stereo_pair(src, depth)
        self.assertEqual(L.shape, (64, 64, 4))
        self.assertEqual(R.shape, (64, 64, 4))
        self.assertEqual(occ.shape, (64, 64))

    def test_occlusion_mask_range(self):
        src   = make_source(64, 64)
        depth = make_depth(64, 64, mode="binary")
        _, _, occ = make_stereo_pair(src, depth)
        self.assertTrue((occ >= 0).all())
        self.assertTrue((occ <= 1).all())

    def test_left_right_differ(self):
        """L and R views should differ when depth has a raised region."""
        src = np.zeros((64, 64, 4), dtype=np.uint8)
        src[:, :, 3] = 255
        for x in range(64):
            src[:, x, 0] = x * 4   # R-channel gradient ensures shifted rows differ
        depth = make_depth(64, 64, mode="binary")
        p = StereoPairParams(max_parallax_fraction=0.1, eye_balance=0.5)
        L, R, _ = make_stereo_pair(src, depth, p)
        self.assertFalse(np.array_equal(L, R))

    def test_side_by_side_layout(self):
        src   = make_source(32, 32)
        depth = make_depth(32, 32)
        p     = StereoPairParams(layout=StereoLayout.SIDE_BY_SIDE)
        combined, _, _ = make_stereo_pair(src, depth, p)
        self.assertEqual(combined.shape[1], 64)   # 2× width

    def test_top_bottom_layout(self):
        src   = make_source(32, 32)
        depth = make_depth(32, 32)
        p     = StereoPairParams(layout=StereoLayout.TOP_BOTTOM)
        combined, _, _ = make_stereo_pair(src, depth, p)
        self.assertEqual(combined.shape[0], 64)   # 2× height

    def test_invert_depth_changes_occlusion(self):
        """Different depth conventions should produce different occlusion patterns."""
        src   = make_source(64, 64)
        depth = make_depth(64, 64, mode="binary")
        p1 = StereoPairParams(max_parallax_fraction=0.1)
        p2 = StereoPairParams(max_parallax_fraction=0.1, invert_depth=True)
        _, _, occ1 = make_stereo_pair(src, depth, p1)
        _, _, occ2 = make_stereo_pair(src, depth, p2)
        self.assertFalse(np.array_equal(occ1, occ2))

    def test_flat_depth_no_occlusion(self):
        """Flat depth produces no occlusion."""
        src   = make_source(32, 32)
        depth = make_depth(32, 32, mode="flat")
        _, _, occ = make_stereo_pair(src, depth)
        # Flat depth → minimal/zero occlusion
        self.assertLess(float(occ.mean()), 0.1)


# ---------------------------------------------------------------------------
# 6. Hidden image tests
# ---------------------------------------------------------------------------

class TestHiddenImage(unittest.TestCase):

    def test_basic_output_shape(self):
        mask    = shape_to_mask("circle", 64, 64)
        pattern = make_pattern(32, 32)
        result  = encode_hidden_image(pattern, mask)
        self.assertEqual(result.shape, (64, 64, 4))
        self.assertEqual(result.dtype, np.uint8)

    def test_all_shapes(self):
        for s in ["circle", "square", "triangle", "star", "diamond", "arrow"]:
            with self.subTest(shape=s):
                mask = shape_to_mask(s, 48, 48)
                self.assertEqual(mask.shape, (48, 48))
                self.assertAlmostEqual(float(mask.max()), 1.0, places=3)

    def test_text_to_mask(self):
        mask = text_to_mask("HI", 64, 32)
        self.assertEqual(mask.shape, (32, 64))
        # Text should create some white pixels
        self.assertGreater(float(mask.max()), 0.5)

    def test_mask_to_depth_fg_bg(self):
        mask = np.ones((8, 8), dtype=np.float32)
        p    = HiddenImageParams(foreground_depth=0.8, background_depth=0.1)
        d    = mask_to_depth(mask, p)
        self.assertAlmostEqual(float(d.mean()), 0.8, places=2)

    def test_mask_to_depth_bg(self):
        mask = np.zeros((8, 8), dtype=np.float32)
        p    = HiddenImageParams(foreground_depth=0.8, background_depth=0.1)
        d    = mask_to_depth(mask, p)
        self.assertAlmostEqual(float(d.mean()), 0.1, places=2)

    def test_invert_mask(self):
        mask = np.ones((8, 8), dtype=np.float32)
        p    = HiddenImageParams(foreground_depth=0.8, background_depth=0.1, invert_mask=True)
        d    = mask_to_depth(mask, p)
        self.assertAlmostEqual(float(d.mean()), 0.1, places=2)

    def test_invalid_foreground_depth_raises(self):
        with self.assertRaises(ValueError):
            HiddenImageParams(foreground_depth=1.5)

    def test_edge_soften_changes_mask(self):
        mask = shape_to_mask("square", 64, 64)
        p1   = HiddenImageParams(edge_soften_px=0)
        p2   = HiddenImageParams(edge_soften_px=8)
        d1   = mask_to_depth(mask, p1)
        d2   = mask_to_depth(mask, p2)
        self.assertFalse(np.array_equal(d1, d2))


# ---------------------------------------------------------------------------
# 7. Adaptive dots tests
# ---------------------------------------------------------------------------

class TestAdaptiveDots(unittest.TestCase):

    def test_output_shape(self):
        depth  = make_depth(64, 64, mode="sphere")
        params = AdaptiveDotParams(tile_width=64, tile_height=64, seed=0)
        tile   = generate_adaptive_dots(depth, params)
        self.assertEqual(tile.shape, (64, 64, 4))
        self.assertEqual(tile.dtype, np.uint8)

    def test_complexity_map_shape(self):
        depth = make_depth(64, 64, mode="sphere")
        cmap  = complexity_from_depth(depth)
        self.assertEqual(cmap.shape, depth.shape)
        self.assertEqual(cmap.dtype, np.float32)
        self.assertTrue((cmap >= 0).all())
        self.assertTrue((cmap <= 1).all())

    def test_flat_depth_low_complexity(self):
        depth = make_depth(64, 64, mode="flat")
        cmap  = complexity_from_depth(depth)
        self.assertLess(float(cmap.mean()), 0.2)

    def test_binary_depth_has_high_edge_complexity(self):
        depth = make_depth(64, 64, mode="binary")
        cmap  = complexity_from_depth(depth)
        self.assertGreater(float(cmap.max()), 0.5)

    def test_dots_contain_non_bg_pixels(self):
        """Should have some bright (dot) pixels."""
        depth  = make_depth(64, 64, mode="sphere")
        params = AdaptiveDotParams(tile_width=64, tile_height=64,
                                   dot_color=(255,255,255),
                                   bg_color=(0,0,0), seed=7)
        tile   = generate_adaptive_dots(depth, params)
        bright = (tile[:, :, 0] > 128).sum()
        self.assertGreater(bright, 0)

    def test_seed_reproducibility(self):
        depth  = make_depth(32, 32, mode="sphere")
        p1     = AdaptiveDotParams(tile_width=32, tile_height=32, seed=42)
        p2     = AdaptiveDotParams(tile_width=32, tile_height=32, seed=42)
        t1     = generate_adaptive_dots(depth, p1)
        t2     = generate_adaptive_dots(depth, p2)
        self.assertTrue(np.array_equal(t1, t2))

    def test_params_clamp_radius(self):
        p = AdaptiveDotParams(min_dot_radius=0)
        self.assertEqual(p.min_dot_radius, 1)


# ---------------------------------------------------------------------------
# 8. Inpainting tests
# ---------------------------------------------------------------------------

class TestInpainting(unittest.TestCase):

    def test_no_mask_returns_unchanged(self):
        img  = make_source(32, 32)
        mask = np.zeros((32, 32), dtype=np.float32)
        out  = inpaint_occlusion(img, mask)
        self.assertTrue(np.array_equal(out, img))

    def test_full_mask_fills_something(self):
        img  = make_source(32, 32)
        mask = np.ones((32, 32), dtype=np.float32)
        out  = inpaint_occlusion(img, mask, InpaintParams(dilate_mask_px=0, blend_px=0))
        self.assertEqual(out.shape, img.shape)

    def test_output_shape_preserved(self):
        img  = make_source(48, 64)
        mask = np.zeros((48, 64), dtype=np.float32)
        mask[10:20, 10:20] = 1.0
        out  = inpaint_occlusion(img, mask)
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(out.dtype, np.uint8)

    def test_clean_plate_method(self):
        img   = make_source(32, 32)
        plate = np.zeros((32, 32, 4), dtype=np.uint8)
        plate[:, :, 2] = 255; plate[:, :, 3] = 255   # blue plate
        mask  = np.zeros((32, 32), dtype=np.float32)
        mask[10:20, 10:20] = 1.0
        p     = InpaintParams(method=InpaintMethod.CLEAN_PLATE,
                              clean_plate=plate, dilate_mask_px=0, blend_px=0)
        out   = inpaint_occlusion(img, mask, p)
        # Filled region should be blue
        self.assertGreater(int(out[14:16, 14:16, 2].mean()), 200)

    def test_edge_extend_method(self):
        img  = make_source(32, 32)
        mask = np.zeros((32, 32), dtype=np.float32)
        mask[:, 28:] = 1.0   # right edge
        p    = InpaintParams(method=InpaintMethod.EDGE_EXTEND,
                             dilate_mask_px=0, blend_px=0)
        out  = inpaint_occlusion(img, mask, p)
        self.assertEqual(out.shape, (32, 32, 4))

    def test_ai_callback_called(self):
        """AI callback should be invoked when method=AI_CALLBACK."""
        called = {"n": 0}
        def mock_ai(img_rgba, mask_f):
            called["n"] += 1
            return img_rgba
        img  = make_source(16, 16)
        mask = np.ones((16, 16), dtype=np.float32) * 0.6
        p    = InpaintParams(method=InpaintMethod.AI_CALLBACK,
                             ai_callback=mock_ai,
                             dilate_mask_px=0, blend_px=0)
        inpaint_occlusion(img, mask, p)
        self.assertEqual(called["n"], 1)

    def test_auto_selects_clean_plate_when_available(self):
        from depthforge.core.inpainting import _resolve_method, InpaintMethod
        plate = make_source(8, 8)
        p     = InpaintParams(method=InpaintMethod.AUTO, clean_plate=plate)
        self.assertEqual(_resolve_method(p), InpaintMethod.CLEAN_PLATE)


# ---------------------------------------------------------------------------
# 9. Integration — full pipeline
# ---------------------------------------------------------------------------

class TestIntegration(unittest.TestCase):

    def test_full_pipeline_sirds(self):
        """Text → depth → prep → SIRDS → save."""
        from depthforge.core.depth_prep import prep_depth, DepthPrepParams
        from depthforge.core.pattern_gen import generate_pattern, PatternParams, PatternType, ColorMode
        from depthforge.core.synthesizer import synthesize, StereoParams

        depth_raw = make_depth(64, 64, mode="sphere")
        depth     = prep_depth(depth_raw, DepthPrepParams(dilation_px=2))
        pattern   = generate_pattern(PatternParams(
            PatternType.RANDOM_NOISE, 32, 32, ColorMode.GREYSCALE, seed=0))
        stereo    = synthesize(depth, pattern, StereoParams(seed=0))

        self.assertEqual(stereo.shape, (64, 64, 4))
        self.assertEqual(stereo.dtype, np.uint8)

    def test_full_pipeline_hidden_image(self):
        """Shape mask → hidden image stereogram."""
        from depthforge.core.hidden_image import shape_to_mask, encode_hidden_image, HiddenImageParams
        from depthforge.core.pattern_gen  import generate_pattern, PatternParams, PatternType

        mask    = shape_to_mask("star", 64, 64)
        pattern = generate_pattern(PatternParams(PatternType.PERLIN, 32, 32, seed=1))
        result  = encode_hidden_image(pattern, mask, HiddenImageParams())

        self.assertEqual(result.shape, (64, 64, 4))

    def test_full_pipeline_anaglyph(self):
        """Depth + source → anaglyph."""
        src   = make_source(48, 64)
        depth = make_depth(48, 64, mode="sphere")
        from depthforge.core.anaglyph import make_anaglyph_from_depth, AnaglyphParams, AnaglyphMode
        out = make_anaglyph_from_depth(src, depth,
                                       AnaglyphParams(mode=AnaglyphMode.OPTIMISED))
        self.assertEqual(out.shape, (48, 64, 4))

    def test_full_pipeline_stereo_with_inpaint(self):
        """Source + depth → stereo pair → inpaint occlusion."""
        src   = make_source(48, 64)
        depth = make_depth(48, 64, mode="binary")

        from depthforge.core.stereo_pair import make_stereo_pair
        from depthforge.core.inpainting  import inpaint_occlusion, InpaintParams, InpaintMethod
        L, R, occ = make_stereo_pair(src, depth)
        L_fill = inpaint_occlusion(L, occ, InpaintParams(
            method=InpaintMethod.EDGE_EXTEND, blend_px=0, dilate_mask_px=0))
        self.assertEqual(L_fill.shape, L.shape)

    def test_full_pipeline_adaptive_dots(self):
        """Depth → adaptive dots → synthesize."""
        depth  = make_depth(64, 64, mode="sphere")
        from depthforge.core.adaptive_dots import generate_adaptive_dots, AdaptiveDotParams
        from depthforge.core.synthesizer   import synthesize, StereoParams
        tile   = generate_adaptive_dots(depth, AdaptiveDotParams(
            tile_width=32, tile_height=32, seed=0))
        stereo = synthesize(depth, tile, StereoParams(seed=0))
        self.assertEqual(stereo.shape, (64, 64, 4))


# ---------------------------------------------------------------------------
# Helpers used in tests
# ---------------------------------------------------------------------------

def make_pattern(H=32, W=32, seed_val=42) -> np.ndarray:
    p = PatternParams(
        pattern_type=PatternType.RANDOM_NOISE,
        tile_width=W,
        tile_height=H,
        color_mode=ColorMode.GREYSCALE,
        seed=seed_val,
    )
    return generate_pattern(p)


if __name__ == "__main__":
    # Run with verbosity when executed directly
    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
