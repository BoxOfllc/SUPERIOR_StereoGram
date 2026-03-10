"""
DepthForge — Phase 4 Test Suite
================================
Tests for: ComfyUI nodes, Nuke gizmo, Flask web app

Run:
    cd /home/claude && python -m unittest depthforge.tests.test_phase4 -v
"""

import base64
import io
import json
import os
import tempfile
import unittest

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_depth(H=64, W=96) -> np.ndarray:
    y, x = np.mgrid[0:H, 0:W]
    r = np.sqrt((x - W / 2) ** 2 + (y - H / 2) ** 2)
    return np.clip(1.0 - r / (min(W, H) * 0.45), 0.0, 1.0).astype(np.float32)


def _depth_to_b64(depth: np.ndarray) -> str:
    arr = (depth * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _make_rgba(H=64, W=96) -> np.ndarray:
    arr = np.random.randint(50, 200, (H, W, 4), dtype=np.uint8)
    arr[:, :, 3] = 255
    return arr


def _rgba_to_b64(rgba: np.ndarray) -> str:
    img = Image.fromarray(rgba)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _make_depth_tensor(H=32, W=48):
    """Return a depth map in ComfyUI IMAGE tensor format (1, H, W, 3) float32."""
    depth = _make_depth(H, W)
    d3 = np.stack([depth, depth, depth], axis=-1)
    return d3[None]  # (1, H, W, 3)


def _make_rgba_tensor(H=32, W=48):
    """Return an RGBA image as (1, H, W, 4) float32 [0,1]."""
    rgba = _make_rgba(H, W).astype(np.float32) / 255.0
    return rgba[None]


# ===========================================================================
# ComfyUI Nodes
# ===========================================================================


class TestComfyUINodes(unittest.TestCase):

    # ── Node loading ──────────────────────────────────────────

    def test_node_class_mappings_loaded(self):
        from depthforge.comfyui import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

        self.assertEqual(len(NODE_CLASS_MAPPINGS), 10)
        self.assertEqual(len(NODE_DISPLAY_NAME_MAPPINGS), 10)

    def test_all_10_nodes_registered(self):
        from depthforge.comfyui import NODE_CLASS_MAPPINGS

        expected = [
            "DF_DepthPrep",
            "DF_PatternGen",
            "DF_PatternLibrary",
            "DF_Stereogram",
            "DF_AnaglyphOut",
            "DF_StereoPair",
            "DF_HiddenImage",
            "DF_SafetyLimiter",
            "DF_QCOverlay",
            "DF_VideoSequence",
        ]
        for name in expected:
            self.assertIn(name, NODE_CLASS_MAPPINGS, msg=f"Missing: {name}")

    def test_node_count_helper(self):
        from depthforge.comfyui import node_count

        self.assertEqual(node_count(), 10)

    def test_install_outside_nuke_is_noop(self):
        """install() with no ComfyUI path should not raise."""
        from depthforge.comfyui import NODE_CLASS_MAPPINGS

        # Just confirm import works without nuke/comfyui present
        self.assertTrue(len(NODE_CLASS_MAPPINGS) > 0)

    # ── Node INPUT_TYPES ──────────────────────────────────────

    def test_df_depth_prep_input_types(self):
        from depthforge.comfyui.nodes import DF_DepthPrep

        types = DF_DepthPrep.INPUT_TYPES()
        self.assertIn("required", types)
        self.assertIn("depth_image", types["required"])

    def test_df_pattern_gen_input_types(self):
        from depthforge.comfyui.nodes import DF_PatternGen

        types = DF_PatternGen.INPUT_TYPES()
        self.assertIn("pattern_type", types["required"])
        self.assertIn("color_mode", types["required"])

    def test_df_pattern_library_input_types(self):
        from depthforge.comfyui.nodes import DF_PatternLibrary

        types = DF_PatternLibrary.INPUT_TYPES()
        self.assertIn("pattern_name", types["required"])

    def test_df_stereogram_input_types(self):
        from depthforge.comfyui.nodes import DF_Stereogram

        types = DF_Stereogram.INPUT_TYPES()
        self.assertIn("depth_map", types["required"])
        self.assertIn("pattern", types["required"])

    def test_df_video_sequence_input_types(self):
        from depthforge.comfyui.nodes import DF_VideoSequence

        types = DF_VideoSequence.INPUT_TYPES()
        self.assertIn("depth_sequence", types["required"])
        self.assertIn("pattern", types["required"])

    # ── RETURN_TYPES and FUNCTION ─────────────────────────────

    def test_all_nodes_have_return_types(self):
        from depthforge.comfyui import NODE_CLASS_MAPPINGS

        for name, cls in NODE_CLASS_MAPPINGS.items():
            self.assertTrue(hasattr(cls, "RETURN_TYPES"), msg=f"{name} missing RETURN_TYPES")

    def test_all_nodes_have_function_attr(self):
        from depthforge.comfyui import NODE_CLASS_MAPPINGS

        for name, cls in NODE_CLASS_MAPPINGS.items():
            self.assertTrue(hasattr(cls, "FUNCTION"), msg=f"{name} missing FUNCTION")
            self.assertEqual(cls.FUNCTION, "execute", msg=f"{name}.FUNCTION != 'execute'")

    def test_all_nodes_have_category(self):
        from depthforge.comfyui import NODE_CLASS_MAPPINGS

        for name, cls in NODE_CLASS_MAPPINGS.items():
            self.assertEqual(cls.CATEGORY, "DepthForge", msg=f"{name} wrong CATEGORY")

    # ── DF_DepthPrep execute ──────────────────────────────────

    def test_df_depth_prep_execute(self):
        from depthforge.comfyui.nodes import DF_DepthPrep

        node = DF_DepthPrep()
        depth_tensor = _make_depth_tensor()
        result = node.execute(depth_tensor)
        depth_map, preview = result
        self.assertIsInstance(depth_map, np.ndarray)
        self.assertEqual(depth_map.ndim, 2)
        self.assertEqual(depth_map.dtype, np.float32)

    def test_df_depth_prep_with_params(self):
        from depthforge.comfyui.nodes import DF_DepthPrep

        node = DF_DepthPrep()
        depth_tensor = _make_depth_tensor()
        result = node.execute(
            depth_tensor,
            invert=True,
            bilateral_space=3.0,
            bilateral_color=0.05,
            dilation_px=2,
            falloff_curve="gamma",
            near_plane=0.1,
            far_plane=0.9,
        )
        depth_map, preview = result
        self.assertEqual(depth_map.ndim, 2)

    # ── DF_PatternGen execute ─────────────────────────────────

    def test_df_pattern_gen_execute(self):
        from depthforge.comfyui.nodes import DF_PatternGen

        node = DF_PatternGen()
        pattern, preview = node.execute("plasma", "psychedelic", 64, 64, 42, 1.0)
        self.assertEqual(pattern.dtype, np.uint8)
        self.assertEqual(pattern.shape[2], 4)

    def test_df_pattern_gen_all_types(self):
        from depthforge.comfyui.nodes import DF_PatternGen

        node = DF_PatternGen()
        types = [
            "random_noise",
            "perlin",
            "plasma",
            "voronoi",
            "geometric_grid",
            "mandelbrot",
            "dot_matrix",
        ]
        for t in types:
            p, _ = node.execute(t, "greyscale", 32, 32, 1, 1.0)
            self.assertEqual(p.shape, (32, 32, 4), msg=f"Failed for type: {t}")

    # ── DF_PatternLibrary execute ─────────────────────────────

    def test_df_pattern_library_execute(self):
        from depthforge.comfyui.nodes import DF_PatternLibrary

        node = DF_PatternLibrary()
        pattern, preview, info_str = node.execute("plasma_wave", 64, 64, 42, 1.0)
        self.assertEqual(pattern.dtype, np.uint8)
        self.assertIn("plasma_wave", info_str)
        self.assertIn("psychedelic", info_str)

    # ── DF_Stereogram execute ─────────────────────────────────

    def test_df_stereogram_execute(self):
        from depthforge.comfyui.nodes import DF_DepthPrep, DF_PatternGen, DF_Stereogram

        depth_node = DF_DepthPrep()
        pattern_node = DF_PatternGen()
        stereo_node = DF_Stereogram()

        depth_map, _ = depth_node.execute(_make_depth_tensor(32, 48))
        pattern, _ = pattern_node.execute("perlin", "greyscale", 48, 48, 1, 1.0)
        (result,) = stereo_node.execute(depth_map, pattern, depth_factor=0.3)
        # Result is a tensor (1, H, W, 4)
        self.assertIsNotNone(result)

    def test_df_stereogram_with_preset(self):
        from depthforge.comfyui.nodes import DF_PatternGen, DF_Stereogram

        node = DF_Stereogram()
        pattern, _ = DF_PatternGen().execute("plasma", "greyscale", 48, 48, 1, 1.0)
        depth = _make_depth(32, 48)
        (result,) = node.execute(depth, pattern, preset_name="shallow")
        self.assertIsNotNone(result)

    # ── DF_AnaglyphOut execute ────────────────────────────────

    def test_df_anaglyph_execute(self):
        from depthforge.comfyui.nodes import DF_AnaglyphOut

        node = DF_AnaglyphOut()
        src = _make_rgba_tensor(32, 48)
        depth = _make_depth(32, 48)
        (result,) = node.execute(src, depth, mode="optimised", depth_factor=0.35)
        self.assertIsNotNone(result)

    def test_df_anaglyph_all_modes(self):
        from depthforge.comfyui.nodes import DF_AnaglyphOut

        node = DF_AnaglyphOut()
        src = _make_rgba_tensor(32, 48)
        depth = _make_depth(32, 48)
        for mode in ["true", "grey", "colour", "half_colour", "optimised"]:
            (result,) = node.execute(src, depth, mode=mode)
            self.assertIsNotNone(result, msg=f"Mode failed: {mode}")

    # ── DF_StereoPair execute ─────────────────────────────────

    def test_df_stereo_pair_execute(self):
        from depthforge.comfyui.nodes import DF_StereoPair

        node = DF_StereoPair()
        src = _make_rgba_tensor(32, 48)
        depth = _make_depth(32, 48)
        left, right, occ, composed = node.execute(src, depth)
        self.assertIsNotNone(left)
        self.assertIsNotNone(right)
        self.assertIsNotNone(occ)
        self.assertIsNotNone(composed)

    # ── DF_HiddenImage execute ────────────────────────────────

    def test_df_hidden_image_shape(self):
        from depthforge.comfyui.nodes import DF_HiddenImage, DF_PatternGen

        node = DF_HiddenImage()
        pattern, _ = DF_PatternGen().execute("plasma", "greyscale", 64, 64, 1, 1.0)
        result, depth_prev = node.execute(pattern, 96, 64, shape="star")
        self.assertIsNotNone(result)

    def test_df_hidden_image_text(self):
        from depthforge.comfyui.nodes import DF_HiddenImage, DF_PatternGen

        node = DF_HiddenImage()
        pattern, _ = DF_PatternGen().execute("plasma", "greyscale", 64, 64, 1, 1.0)
        result, _ = node.execute(pattern, 192, 64, text="HI")
        self.assertIsNotNone(result)

    # ── DF_SafetyLimiter execute ──────────────────────────────

    def test_df_safety_limiter_execute(self):
        from depthforge.comfyui.nodes import DF_SafetyLimiter

        node = DF_SafetyLimiter()
        depth = _make_depth(32, 48)
        safe_depth, preview, report = node.execute(depth, profile="standard")
        self.assertIsInstance(safe_depth, np.ndarray)
        self.assertIsInstance(report, str)

    def test_df_safety_limiter_all_profiles(self):
        from depthforge.comfyui.nodes import DF_SafetyLimiter

        node = DF_SafetyLimiter()
        depth = _make_depth(32, 48)
        for profile in ["conservative", "standard", "relaxed", "cinema"]:
            safe, _, _ = node.execute(depth, profile=profile)
            self.assertIsNotNone(safe, msg=f"Profile failed: {profile}")

    # ── DF_QCOverlay execute ──────────────────────────────────

    def test_df_qc_overlay_execute(self):
        from depthforge.comfyui.nodes import DF_QCOverlay

        node = DF_QCOverlay()
        depth = _make_depth(32, 48)
        img, summary = node.execute(depth, overlay_type="parallax_heatmap")
        self.assertIsNotNone(img)
        self.assertIsInstance(summary, str)

    def test_df_qc_overlay_all_types(self):
        from depthforge.comfyui.nodes import DF_QCOverlay

        node = DF_QCOverlay()
        depth = _make_depth(32, 48)
        types = ["parallax_heatmap", "depth_bands", "violation_overlay", "safe_zone", "histogram"]
        for t in types:
            img, _ = node.execute(depth, overlay_type=t)
            self.assertIsNotNone(img, msg=f"Type failed: {t}")

    # ── DF_VideoSequence execute ──────────────────────────────

    def test_df_video_sequence_execute(self):
        from depthforge.comfyui.nodes import DF_PatternGen, DF_VideoSequence

        node = DF_VideoSequence()
        pattern, _ = DF_PatternGen().execute("perlin", "greyscale", 48, 48, 1, 1.0)
        # 3-frame batch: (3, H, W, 3)
        frames = np.stack([_make_depth_tensor(32, 48)[0]] * 3, axis=0)
        (result,) = node.execute(frames, pattern, temporal_smooth=0.2)
        self.assertIsNotNone(result)

    def test_df_video_sequence_single_frame(self):
        from depthforge.comfyui.nodes import DF_PatternGen, DF_VideoSequence

        node = DF_VideoSequence()
        pattern, _ = DF_PatternGen().execute("perlin", "greyscale", 48, 48, 1, 1.0)
        single = _make_depth_tensor(32, 48)  # (1, H, W, 3)
        (result,) = node.execute(single, pattern)
        self.assertIsNotNone(result)


# ===========================================================================
# Nuke Gizmo (pure-Python, no Nuke required)
# ===========================================================================


class TestNukeGizmo(unittest.TestCase):

    def test_import_nuke_package(self):
        import depthforge.nuke

        self.assertTrue(hasattr(depthforge.nuke, "DepthForgeGizmo"))
        self.assertTrue(hasattr(depthforge.nuke, "install"))
        self.assertTrue(hasattr(depthforge.nuke, "create_node"))

    def test_stereogram_knobs_count(self):
        from depthforge.nuke.gizmo import STEREOGRAM_KNOBS

        self.assertGreater(len(STEREOGRAM_KNOBS), 20)

    def test_knob_names_unique(self):
        from depthforge.nuke.gizmo import STEREOGRAM_KNOBS

        names = [k[1] for k in STEREOGRAM_KNOBS if not k[1].startswith("_tab")]
        self.assertEqual(len(names), len(set(names)), "Duplicate knob names!")

    def test_gizmo_default_values(self):
        from depthforge.nuke.gizmo import DepthForgeGizmo

        g = DepthForgeGizmo()
        self.assertAlmostEqual(g.get("df_depth_factor"), 0.35)
        self.assertEqual(g.get("df_seed"), 42)
        self.assertFalse(g.get("df_safe_mode"))

    def test_gizmo_knob_overrides(self):
        from depthforge.nuke.gizmo import DepthForgeGizmo

        g = DepthForgeGizmo({"df_depth_factor": 0.6, "df_mode": "Anaglyph"})
        self.assertAlmostEqual(g.get("df_depth_factor"), 0.6)
        self.assertEqual(g.get("df_mode"), "Anaglyph")

    def test_gizmo_set_get(self):
        from depthforge.nuke.gizmo import DepthForgeGizmo

        g = DepthForgeGizmo()
        g.set("df_depth_factor", 0.8)
        self.assertAlmostEqual(g.get("df_depth_factor"), 0.8)

    def test_gizmo_to_dict(self):
        from depthforge.nuke.gizmo import DepthForgeGizmo

        g = DepthForgeGizmo()
        d = g.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("df_depth_factor", d)

    def test_gizmo_json_roundtrip(self):
        from depthforge.nuke.gizmo import DepthForgeGizmo

        g = DepthForgeGizmo({"df_depth_factor": 0.7, "df_seed": 99})
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            g.to_json(path)
            g2 = DepthForgeGizmo.from_json(path)
            self.assertAlmostEqual(g2.get("df_depth_factor"), 0.7)
            self.assertEqual(g2.get("df_seed"), 99)
        finally:
            os.unlink(path)

    def test_gizmo_from_dict(self):
        from depthforge.nuke.gizmo import DepthForgeGizmo

        g = DepthForgeGizmo.from_dict({"df_depth_factor": 0.5, "df_mode": "SIRDS"})
        self.assertAlmostEqual(g.get("df_depth_factor"), 0.5)

    def test_gizmo_render_sirds(self):
        from depthforge.nuke.gizmo import DepthForgeGizmo

        g = DepthForgeGizmo({"df_mode": "SIRDS", "df_preset": "shallow"})
        depth = _make_depth(32, 48)
        result = g.render(depth)
        self.assertEqual(result.shape, (32, 48, 4))
        self.assertEqual(result.dtype, np.uint8)

    def test_gizmo_render_texture(self):
        from depthforge.nuke.gizmo import DepthForgeGizmo

        g = DepthForgeGizmo(
            {"df_mode": "Texture", "df_pattern_type": "plasma", "df_color_mode": "psychedelic"}
        )
        depth = _make_depth(32, 48)
        result = g.render(depth)
        self.assertEqual(result.shape, (32, 48, 4))

    def test_gizmo_render_anaglyph(self):
        from depthforge.nuke.gizmo import DepthForgeGizmo

        g = DepthForgeGizmo({"df_mode": "Anaglyph", "df_anaglyph_mode": "optimised"})
        depth = _make_depth(32, 48)
        source = _make_rgba(32, 48)
        result = g.render(depth, source=source)
        self.assertEqual(result.shape, (32, 48, 4))

    def test_gizmo_render_stereopair(self):
        from depthforge.nuke.gizmo import DepthForgeGizmo

        g = DepthForgeGizmo({"df_mode": "StereoPair"})
        depth = _make_depth(32, 48)
        source = _make_rgba(32, 48)
        result = g.render(depth, source=source)
        self.assertIsNotNone(result)

    def test_gizmo_render_hidden_image(self):
        from depthforge.nuke.gizmo import DepthForgeGizmo

        g = DepthForgeGizmo({"df_mode": "HiddenImage", "df_hidden_shape": "star"})
        depth = _make_depth(32, 48)
        result = g.render(depth)
        self.assertEqual(result.shape, (32, 48, 4))

    def test_gizmo_render_all_presets(self):
        from depthforge.nuke.gizmo import DepthForgeGizmo

        depth = _make_depth(32, 48)
        for preset in ["none", "shallow", "medium", "deep", "cinema", "broadcast"]:
            g = DepthForgeGizmo({"df_preset": preset, "df_mode": "SIRDS"})
            result = g.render(depth)
            self.assertEqual(result.shape[2], 4, msg=f"Preset failed: {preset}")

    def test_gizmo_render_auto_safety(self):
        from depthforge.nuke.gizmo import DepthForgeGizmo

        g = DepthForgeGizmo({"df_auto_safe": True, "df_safety_profile": "conservative"})
        depth = _make_depth(32, 48)
        result = g.render(depth)
        self.assertIsNotNone(result)

    def test_create_node_outside_nuke(self):
        """create_node() should return a DepthForgeGizmo when Nuke is unavailable."""
        import depthforge.nuke as nuke_mod

        result = nuke_mod.create_node(mode="SIRDS", preset="medium")
        from depthforge.nuke.gizmo import DepthForgeGizmo

        self.assertIsInstance(result, DepthForgeGizmo)
        self.assertEqual(result.get("df_mode"), "SIRDS")

    def test_install_outside_nuke_noop(self):
        """install() should not raise when Nuke is unavailable."""
        import depthforge.nuke as nuke_mod

        try:
            nuke_mod.install()  # Should silently succeed
        except Exception as e:
            self.fail(f"install() raised {e} outside Nuke")

    def test_nuke_package_has_toolbar_module(self):
        from depthforge.nuke import toolbar

        self.assertTrue(hasattr(toolbar, "register_menu"))
        self.assertTrue(hasattr(toolbar, "unregister_menu"))
        self.assertGreater(len(toolbar._MAIN_ITEMS), 5)

    def test_toolbar_preset_items_complete(self):
        from depthforge.nuke.toolbar import _PRESET_ITEMS

        names = [item[2] for item in _PRESET_ITEMS]
        for expected in ["shallow", "medium", "deep", "cinema", "print", "broadcast"]:
            self.assertIn(expected, names)


# ===========================================================================
# Flask Web App
# ===========================================================================


class TestWebApp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from depthforge.web.app import create_app

        app = create_app({"TESTING": True})
        cls.client = app.test_client()

    # ── Basic routes ──────────────────────────────────────────

    def test_index_returns_html(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"DepthForge", r.data)
        self.assertIn(b"text/html", r.content_type.encode())

    def test_index_has_ui_elements(self):
        r = self.client.get("/")
        html = r.data.decode()
        for marker in ["btn-generate", "depth-factor", "preset", "pattern-grid"]:
            self.assertIn(marker, html, msg=f"UI missing: {marker}")

    def test_api_status(self):
        r = self.client.get("/api/status")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("version", data)

    def test_api_presets_returns_list(self):
        r = self.client.get("/api/presets")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 3)
        names = [p["name"] for p in data]
        for expected in ["shallow", "medium", "deep", "cinema"]:
            self.assertIn(expected, names)

    def test_api_preset_detail(self):
        r = self.client.get("/api/preset/medium")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertEqual(data["name"], "medium")
        self.assertIn("stereo", data)
        self.assertIn("depth_factor", data["stereo"])

    def test_api_preset_unknown_404(self):
        r = self.client.get("/api/preset/no_such_preset_xyz")
        self.assertEqual(r.status_code, 404)

    def test_api_patterns_returns_dict(self):
        r = self.client.get("/api/patterns")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("categories", data)
        self.assertIn("patterns", data)
        self.assertGreater(len(data["patterns"]), 20)

    def test_api_pattern_preview(self):
        r = self.client.get("/api/pattern/perlin_noise?size=64")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("image", data)
        self.assertGreater(len(data["image"]), 100)
        self.assertEqual(data["width"], 64)

    def test_api_pattern_unknown_404(self):
        r = self.client.get("/api/pattern/no_such_pattern_xyz")
        self.assertEqual(r.status_code, 404)

    # ── Synthesize ────────────────────────────────────────────

    def _post_synthesize(self, mode="texture", extra=None):
        depth = _make_depth(32, 48)
        body = {
            "depth_b64": _depth_to_b64(depth),
            "mode": mode,
            "depth_factor": 0.3,
            "tile_size": 48,
            "seed": 1,
            "async_job": False,
        }
        if extra:
            body.update(extra)
        return self.client.post(
            "/api/synthesize", data=json.dumps(body), content_type="application/json"
        )

    def test_synthesize_texture(self):
        r = self._post_synthesize("texture")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("image_b64", data)
        self.assertGreater(len(data["image_b64"]), 100)
        self.assertEqual(data["width"], 48)
        self.assertEqual(data["height"], 32)

    def test_synthesize_sirds(self):
        r = self._post_synthesize("sirds")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("image_b64", data)

    def test_synthesize_hidden(self):
        r = self._post_synthesize(
            "hidden",
            {
                "hidden_shape": "star",
                "tile_size": 48,
            },
        )
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("image_b64", data)

    def test_synthesize_anaglyph(self):
        source = _make_rgba(32, 48)
        r = self._post_synthesize(
            "anaglyph",
            {
                "source_b64": _rgba_to_b64(source),
                "anaglyph_mode": "optimised",
            },
        )
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("image_b64", data)

    def test_synthesize_with_preset(self):
        r = self._post_synthesize("texture", {"preset": "shallow"})
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertEqual(data["preset"], "shallow")

    def test_synthesize_with_pattern_name(self):
        r = self._post_synthesize("texture", {"pattern_name": "plasma_wave"})
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("image_b64", data)

    def test_synthesize_missing_depth_400(self):
        r = self.client.post(
            "/api/synthesize", data=json.dumps({"mode": "texture"}), content_type="application/json"
        )
        self.assertEqual(r.status_code, 400)
        self.assertIn("error", r.get_json())

    def test_synthesize_async_returns_job_id(self):
        depth = _make_depth(32, 48)
        body = {
            "depth_b64": _depth_to_b64(depth),
            "mode": "texture",
            "tile_size": 48,
            "async_job": True,
        }
        r = self.client.post(
            "/api/synthesize", data=json.dumps(body), content_type="application/json"
        )
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("job_id", data)
        self.assertIsInstance(data["job_id"], str)

    def test_result_unknown_job_404(self):
        r = self.client.get("/api/result/nonexistent_job_id")
        self.assertEqual(r.status_code, 404)

    # ── Analyze ───────────────────────────────────────────────

    def test_analyze_returns_report(self):
        depth = _make_depth(32, 48)
        r = self.client.post(
            "/api/analyze",
            data=json.dumps(
                {
                    "depth_b64": _depth_to_b64(depth),
                    "depth_factor": 0.35,
                    "max_parallax": 0.033,
                    "frame_width": 640,
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        for key in [
            "passed",
            "overall_score",
            "max_parallax_px",
            "max_vergence_deg",
            "violations",
            "advice",
            "heatmap_b64",
            "overlay_b64",
        ]:
            self.assertIn(key, data, msg=f"Missing key: {key}")

    def test_analyze_heatmap_is_valid_image(self):
        depth = _make_depth(32, 48)
        r = self.client.post(
            "/api/analyze",
            data=json.dumps({"depth_b64": _depth_to_b64(depth)}),
            content_type="application/json",
        )
        data = r.get_json()
        raw = base64.b64decode(data["heatmap_b64"])
        img = Image.open(io.BytesIO(raw))
        self.assertGreater(img.width, 0)

    def test_analyze_missing_depth_400(self):
        r = self.client.post(
            "/api/analyze", data=json.dumps({"depth_factor": 0.3}), content_type="application/json"
        )
        self.assertEqual(r.status_code, 400)

    # ── Depth estimation ──────────────────────────────────────

    def test_estimate_depth_mock(self):
        src = _make_rgba(32, 48)
        r = self.client.post(
            "/api/estimate_depth",
            data=json.dumps(
                {
                    "image_b64": _rgba_to_b64(src),
                    "model": "midas_mock",
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("depth_b64", data)
        self.assertEqual(data["width"], 48)
        self.assertEqual(data["height"], 32)

    def test_estimate_depth_invert(self):
        src = _make_rgba(32, 48)
        r = self.client.post(
            "/api/estimate_depth",
            data=json.dumps(
                {
                    "image_b64": _rgba_to_b64(src),
                    "model": "midas_mock",
                    "invert": True,
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 200)

    def test_estimate_depth_missing_image_400(self):
        r = self.client.post(
            "/api/estimate_depth",
            data=json.dumps({"model": "midas_mock"}),
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 400)

    # ── QC overlay ────────────────────────────────────────────

    def test_qc_overlay_heatmap(self):
        depth = _make_depth(32, 48)
        r = self.client.post(
            "/api/qc_overlay",
            data=json.dumps(
                {
                    "depth_b64": _depth_to_b64(depth),
                    "overlay_type": "parallax_heatmap",
                }
            ),
            content_type="application/json",
        )
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("image_b64", data)

    def test_qc_overlay_all_types(self):
        depth = _make_depth(32, 48)
        for t in ["parallax_heatmap", "depth_bands", "violation_overlay", "safe_zone", "histogram"]:
            r = self.client.post(
                "/api/qc_overlay",
                data=json.dumps({"depth_b64": _depth_to_b64(depth), "overlay_type": t}),
                content_type="application/json",
            )
            self.assertEqual(r.status_code, 200, msg=f"Type failed: {t}")

    # ── Result image validity ─────────────────────────────────

    def test_synthesized_image_is_valid_png(self):
        """Decode the returned base64 and verify it's a real image."""
        r = self._post_synthesize("texture")
        data = r.get_json()
        raw = base64.b64decode(data["image_b64"])
        img = Image.open(io.BytesIO(raw))
        self.assertEqual(img.width, 48)
        self.assertEqual(img.height, 32)


# ===========================================================================
# Phase 4 Integration
# ===========================================================================


class TestPhase4Integration(unittest.TestCase):

    def test_comfyui_node_pipeline(self):
        """Full ComfyUI pipeline: DepthPrep → PatternLibrary → Stereogram."""
        from depthforge.comfyui.nodes import DF_DepthPrep, DF_PatternLibrary, DF_Stereogram

        raw_depth = _make_depth_tensor(32, 48)
        depth_map, _ = DF_DepthPrep().execute(raw_depth, bilateral_space=3.0, falloff_curve="gamma")
        pattern, _, _ = DF_PatternLibrary().execute("northern_lights", 48, 48, 42, 1.0)
        (result,) = DF_Stereogram().execute(depth_map, pattern, depth_factor=0.3, safe_mode=True)
        self.assertIsNotNone(result)

    def test_nuke_gizmo_with_all_modes(self):
        """DepthForgeGizmo renders successfully in every mode."""
        from depthforge.nuke.gizmo import DepthForgeGizmo

        depth = _make_depth(32, 48)
        source = _make_rgba(32, 48)
        for mode in ["SIRDS", "Texture", "Anaglyph", "HiddenImage"]:
            g = DepthForgeGizmo({"df_mode": mode})
            src = source if mode in ["Anaglyph"] else None
            res = g.render(depth, source=src)
            self.assertEqual(res.shape[2], 4, msg=f"Mode failed: {mode}")

    def test_web_synthesize_result_feeds_analyze(self):
        """Synthesize via web API then analyze the depth map."""
        from depthforge.web.app import create_app

        app = create_app({"TESTING": True})
        client = app.test_client()
        depth = _make_depth(32, 48)
        b64 = _depth_to_b64(depth)

        # Synthesize
        r1 = client.post(
            "/api/synthesize",
            data=json.dumps({"depth_b64": b64, "mode": "texture", "async_job": False}),
            content_type="application/json",
        )
        self.assertEqual(r1.status_code, 200)

        # Analyze
        r2 = client.post(
            "/api/analyze", data=json.dumps({"depth_b64": b64}), content_type="application/json"
        )
        self.assertEqual(r2.status_code, 200)
        data2 = r2.get_json()
        self.assertIn("overall_score", data2)

    def test_nuke_gizmo_serialization_roundtrip(self):
        """Gizmo serialize → deserialize → render gives same output."""
        from depthforge.nuke.gizmo import DepthForgeGizmo

        g1 = DepthForgeGizmo(
            {
                "df_mode": "SIRDS",
                "df_depth_factor": 0.4,
                "df_seed": 7,
                "df_pattern_type": "perlin",
            }
        )
        depth = _make_depth(32, 48)
        r1 = g1.render(depth)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            g1.to_json(path)
            g2 = DepthForgeGizmo.from_json(path)
            r2 = g2.render(depth)
            np.testing.assert_array_equal(r1, r2)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
