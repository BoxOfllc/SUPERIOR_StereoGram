"""
depthforge.web.app
==================
Local Flask preview server for DepthForge.

Provides a single-page web UI for interactive stereogram generation,
with REST endpoints for all synthesis modes and SSE streaming for
progress on long renders.

Usage
-----
    # From the command line (via the CLI):
    depthforge serve --port 5000

    # Programmatically:
    from depthforge.web.app import create_app
    app = create_app()
    app.run(host="127.0.0.1", port=5000, debug=False)

REST Endpoints
--------------
GET  /                      Serve the single-page UI
GET  /api/status            Server status + capability report
GET  /api/presets           List all presets
GET  /api/patterns          List all patterns (with categories)
POST /api/synthesize        Generate stereogram (returns base64 PNG)
POST /api/analyze           Analyze a depth map (returns JSON report)
POST /api/estimate_depth    AI depth estimation (returns base64 depth PNG)
GET  /api/progress/<job_id> SSE stream for long-running job progress
GET  /api/pattern/<name>    Preview a named pattern tile (base64 PNG)
GET  /api/preset/<name>     Get preset details as JSON
"""

from __future__ import annotations

import base64
import io
import json
import queue
import threading
import time
import uuid
from typing import Any, Optional

import numpy as np
from flask import Flask, Response, jsonify, request, stream_with_context
from PIL import Image

# ---------------------------------------------------------------------------
# Job tracker for progress SSE
# ---------------------------------------------------------------------------


class _JobTracker:
    """Thread-safe job progress tracker."""

    def __init__(self):
        self._jobs: dict[str, dict] = {}
        self._queues: dict[str, queue.Queue] = {}
        self._lock = threading.Lock()

    def create(self, job_id: str) -> None:
        with self._lock:
            self._jobs[job_id] = {"status": "pending", "progress": 0, "result": None, "error": None}
            self._queues[job_id] = queue.Queue()

    def update(self, job_id: str, progress: int, message: str = "") -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["progress"] = progress
                self._jobs[job_id]["status"] = "running"
        if job_id in self._queues:
            self._queues[job_id].put({"progress": progress, "message": message})

    def complete(self, job_id: str, result: Any) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update({"status": "done", "progress": 100, "result": result})
        if job_id in self._queues:
            self._queues[job_id].put({"progress": 100, "message": "done", "done": True})

    def fail(self, job_id: str, error: str) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update({"status": "error", "error": error})
        if job_id in self._queues:
            self._queues[job_id].put({"progress": 0, "message": error, "error": True, "done": True})

    def get_result(self, job_id: str) -> Optional[Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            return job["result"] if job else None

    def stream(self, job_id: str):
        """Generator that yields SSE events for a job."""
        q = self._queues.get(job_id)
        if q is None:
            yield f"data: {json.dumps({'error': 'unknown job'})}\n\n"
            return
        while True:
            try:
                event = q.get(timeout=30)
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("done"):
                    break
            except queue.Empty:
                yield f"data: {json.dumps({'heartbeat': True})}\n\n"


_jobs = _JobTracker()


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def _array_to_b64(arr: np.ndarray, fmt: str = "PNG") -> str:
    """Convert RGBA uint8 numpy array to base64-encoded image string."""
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format=fmt, optimize=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _b64_to_array(b64_str: str) -> np.ndarray:
    """Decode a base64 image string to a numpy RGBA uint8 array."""
    data = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(data)).convert("RGBA")
    return np.array(img, dtype=np.uint8)


def _b64_to_depth(b64_str: str) -> np.ndarray:
    """Decode a base64 image string to a float32 depth map."""
    data = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(data)).convert("L")
    return np.array(img, dtype=np.float32) / 255.0


def _depth_to_b64(depth: np.ndarray) -> str:
    """Convert float32 depth map to base64 greyscale PNG."""
    arr = (np.clip(depth, 0, 1) * 255).astype(np.uint8)
    return _array_to_b64(np.stack([arr, arr, arr, np.full_like(arr, 255)], axis=-1))


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(config: Optional[dict] = None) -> Flask:
    """Create and configure the DepthForge Flask application.

    Parameters
    ----------
    config : dict, optional
        Optional Flask config overrides.

    Returns
    -------
    Flask
    """
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB
    app.config["JSON_SORT_KEYS"] = False
    if config:
        app.config.update(config)

    # ----------------------------------------------------------------
    # GET /
    # ----------------------------------------------------------------
    @app.route("/")
    def index():
        from depthforge.web.ui import HTML

        return Response(HTML, mimetype="text/html")

    # ----------------------------------------------------------------
    # GET /api/status
    # ----------------------------------------------------------------
    @app.route("/api/status")
    def api_status():
        try:
            import depthforge

            caps = depthforge.capability_report()
        except Exception:
            caps = {}
        return jsonify(
            {
                "status": "ok",
                "version": "0.3.0",
                "capabilities": caps,
            }
        )

    # ----------------------------------------------------------------
    # GET /api/presets
    # ----------------------------------------------------------------
    @app.route("/api/presets")
    def api_presets():
        from depthforge.core.presets import get_preset, list_presets

        result = []
        for name in list_presets():
            try:
                p = get_preset(name)
                result.append(
                    {
                        "name": p.name,
                        "display_name": p.display_name,
                        "description": p.description,
                        "depth_factor": p.stereo_params.depth_factor,
                        "max_parallax": p.stereo_params.max_parallax_fraction,
                        "safe_mode": p.stereo_params.safe_mode,
                        "output_dpi": p.output_dpi,
                        "color_space": p.color_space,
                        "target_width": p.target_width,
                        "notes": p.notes,
                    }
                )
            except Exception as e:
                result.append({"name": name, "error": str(e)})
        return jsonify(result)

    # ----------------------------------------------------------------
    # GET /api/preset/<name>
    # ----------------------------------------------------------------
    @app.route("/api/preset/<name>")
    def api_preset_detail(name):
        from depthforge.core.presets import get_preset

        try:
            p = get_preset(name)
            return jsonify(
                {
                    "name": p.name,
                    "display_name": p.display_name,
                    "description": p.description,
                    "stereo": {
                        "depth_factor": p.stereo_params.depth_factor,
                        "max_parallax_fraction": p.stereo_params.max_parallax_fraction,
                        "safe_mode": p.stereo_params.safe_mode,
                        "oversample": p.stereo_params.oversample,
                    },
                    "depth": {
                        "near_plane": p.depth_params.near_plane,
                        "far_plane": p.depth_params.far_plane,
                        "dilation_px": p.depth_params.dilation_px,
                        "falloff": p.depth_params.falloff_curve.name,
                    },
                    "output_dpi": p.output_dpi,
                    "target_width": p.target_width,
                    "color_space": p.color_space,
                    "notes": p.notes,
                }
            )
        except KeyError:
            return jsonify({"error": f"Unknown preset: {name}"}), 404

    # ----------------------------------------------------------------
    # GET /api/patterns
    # ----------------------------------------------------------------
    @app.route("/api/patterns")
    def api_patterns():
        from depthforge.core.pattern_library import get_pattern_info, list_categories, list_patterns

        cats = list_categories()
        result = {"categories": cats, "patterns": []}
        for name in list_patterns():
            try:
                info = get_pattern_info(name)
                result["patterns"].append(
                    {
                        "name": info.name,
                        "category": info.category,
                        "description": info.description,
                        "safe_mode": info.safe_mode,
                    }
                )
            except Exception:
                pass
        return jsonify(result)

    # ----------------------------------------------------------------
    # GET /api/pattern/<name>  — preview tile
    # ----------------------------------------------------------------
    @app.route("/api/pattern/<name>")
    def api_pattern_preview(name):
        from depthforge.core.pattern_library import get_pattern

        try:
            seed = int(request.args.get("seed", 42))
            size = min(int(request.args.get("size", 128)), 512)
            scale = float(request.args.get("scale", 1.0))
            pat = get_pattern(name, width=size, height=size, seed=seed, scale=scale)
            return jsonify(
                {"name": name, "image": _array_to_b64(pat), "width": size, "height": size}
            )
        except KeyError:
            return jsonify({"error": f"Unknown pattern: {name}"}), 404
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ----------------------------------------------------------------
    # POST /api/synthesize
    # ----------------------------------------------------------------
    @app.route("/api/synthesize", methods=["POST"])
    def api_synthesize():
        """
        Body (JSON):
            depth_b64       str      Base64-encoded depth image (required)
            mode            str      "sirds"|"texture"|"anaglyph"|"hidden" (default: "texture")
            preset          str      Preset name (overrides individual params)
            depth_factor    float    0.0–1.0
            max_parallax    float    0.005–0.1
            safe_mode       bool
            oversample      int      1–4
            seed            int
            pattern_type    str      pattern_gen type name
            pattern_name    str      pattern_library name (preferred over pattern_type)
            tile_size       int
            color_mode      str      greyscale|monochrome|psychedelic
            pattern_scale   float
            falloff         str      linear|gamma|s_curve|logarithmic|exponential
            near_plane      float
            far_plane       float
            dilation_px     int
            invert_depth    bool
            hidden_shape    str      For hidden mode
            hidden_text     str      For hidden mode
            source_b64      str      Source image for anaglyph/stereo pair
            async_job       bool     If true, returns job_id for SSE tracking
        """
        body = request.get_json(force=True, silent=True) or {}

        if not body.get("depth_b64"):
            return jsonify({"error": "depth_b64 is required"}), 400

        async_job = body.get("async_job", False)
        job_id = str(uuid.uuid4())[:8]

        if async_job:
            _jobs.create(job_id)
            t = threading.Thread(
                target=_run_synthesis,
                args=(job_id, body),
                daemon=True,
            )
            t.start()
            return jsonify({"job_id": job_id})
        else:
            try:
                result = _synthesize(body, progress_cb=None)
                return jsonify(result)
            except Exception as e:
                return jsonify({"error": str(e)}), 500

    # ----------------------------------------------------------------
    # GET /api/progress/<job_id>  — SSE
    # ----------------------------------------------------------------
    @app.route("/api/progress/<job_id>")
    def api_progress(job_id):
        def generate():
            yield from _jobs.stream(job_id)

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # ----------------------------------------------------------------
    # GET /api/result/<job_id>
    # ----------------------------------------------------------------
    @app.route("/api/result/<job_id>")
    def api_result(job_id):
        result = _jobs.get_result(job_id)
        if result is None:
            return jsonify({"error": "Result not ready or unknown job_id"}), 404
        return jsonify(result)

    # ----------------------------------------------------------------
    # POST /api/analyze
    # ----------------------------------------------------------------
    @app.route("/api/analyze", methods=["POST"])
    def api_analyze():
        """
        Body (JSON):
            depth_b64       str      Base64-encoded depth image (required)
            depth_factor    float    Planned depth factor
            max_parallax    float
            frame_width     int      Target frame width in pixels
        """
        body = request.get_json(force=True, silent=True) or {}
        if not body.get("depth_b64"):
            return jsonify({"error": "depth_b64 is required"}), 400
        try:
            from depthforge.core.comfort import (
                ComfortAnalyzer,
                ComfortAnalyzerParams,
                estimate_vergence_strain,
            )
            from depthforge.core.qc import QCParams, parallax_heatmap

            depth = _b64_to_depth(body["depth_b64"])
            depth_factor = float(body.get("depth_factor", 0.35))
            max_parallax = float(body.get("max_parallax", 0.033))
            frame_width = int(body.get("frame_width", 1920))

            params = ComfortAnalyzerParams(
                frame_width=frame_width,
                depth_factor=depth_factor,
                max_parallax_fraction=max_parallax,
            )
            analyzer = ComfortAnalyzer(params)
            report = analyzer.analyze(depth)
            strain = estimate_vergence_strain(
                depth, depth_factor=depth_factor, frame_width=frame_width
            )

            # Heatmap preview
            qp = QCParams(
                frame_width=frame_width,
                depth_factor=depth_factor,
                max_parallax_fraction=max_parallax,
            )
            heatmap = parallax_heatmap(depth, qp, annotate=True)
            overlay = analyzer.violation_overlay(depth, report)

            return jsonify(
                {
                    "passed": report.passed,
                    "overall_score": round(report.overall_score, 3),
                    "max_parallax_px": round(report.max_parallax_px, 1),
                    "max_vergence_deg": round(report.max_vergence_deg, 2),
                    "pct_uncomfortable": round(report.pct_uncomfortable, 3),
                    "violations": report.violations,
                    "advice": report.advice,
                    "summary": report.summary(),
                    "strain_rating": strain["strain_rating"],
                    "mean_vergence_deg": round(strain["mean_vergence_deg"], 2),
                    "heatmap_b64": _array_to_b64(heatmap),
                    "overlay_b64": _array_to_b64(overlay),
                }
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ----------------------------------------------------------------
    # POST /api/estimate_depth
    # ----------------------------------------------------------------
    @app.route("/api/estimate_depth", methods=["POST"])
    def api_estimate_depth():
        """
        Body (JSON):
            image_b64   str    Base64-encoded RGB source image (required)
            model       str    "midas_mock" | "zoedepth_mock" | "midas" | "zoedepth"
            invert      bool
        """
        body = request.get_json(force=True, silent=True) or {}
        if not body.get("image_b64"):
            return jsonify({"error": "image_b64 is required"}), 400
        try:
            from depthforge.depth_models import get_depth_estimator

            rgba = _b64_to_array(body["image_b64"])
            rgb = rgba[:, :, :3]
            model = body.get("model", "midas_mock")
            invert = bool(body.get("invert", False))

            est = get_depth_estimator(model)
            depth = est.estimate(rgb)
            if invert:
                depth = 1.0 - depth

            return jsonify(
                {
                    "model": model,
                    "depth_b64": _depth_to_b64(depth),
                    "width": depth.shape[1],
                    "height": depth.shape[0],
                    "depth_min": float(depth.min()),
                    "depth_max": float(depth.max()),
                    "depth_mean": float(depth.mean()),
                }
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # ----------------------------------------------------------------
    # POST /api/qc_overlay
    # ----------------------------------------------------------------
    @app.route("/api/qc_overlay", methods=["POST"])
    def api_qc_overlay():
        body = request.get_json(force=True, silent=True) or {}
        if not body.get("depth_b64"):
            return jsonify({"error": "depth_b64 is required"}), 400
        try:
            from depthforge.core.qc import (
                QCParams,
                depth_band_preview,
                depth_histogram,
                parallax_heatmap,
                safe_zone_indicator,
                window_violation_overlay,
            )

            depth = _b64_to_depth(body["depth_b64"])
            overlay_type = body.get("overlay_type", "parallax_heatmap")
            frame_width = int(body.get("frame_width", 1920))
            depth_factor = float(body.get("depth_factor", 0.35))
            max_parallax = float(body.get("max_parallax", 0.033))
            n_bands = int(body.get("n_bands", 8))
            colormap = body.get("colormap", "inferno")
            annotate = bool(body.get("annotate", True))

            qp = QCParams(
                frame_width=frame_width,
                depth_factor=depth_factor,
                max_parallax_fraction=max_parallax,
                colormap=colormap,
            )
            H, W = depth.shape

            if overlay_type == "parallax_heatmap":
                out = parallax_heatmap(depth, qp, annotate=annotate)
            elif overlay_type == "depth_bands":
                out = depth_band_preview(depth, n_bands=n_bands)
            elif overlay_type == "violation_overlay":
                out = window_violation_overlay(depth)
            elif overlay_type == "safe_zone":
                out = safe_zone_indicator(depth, qp)
            elif overlay_type == "histogram":
                out = depth_histogram(depth, width=W, height=max(H // 4, 100))
            else:
                out = parallax_heatmap(depth, qp, annotate=annotate)

            return jsonify({"image_b64": _array_to_b64(out), "type": overlay_type})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


# ---------------------------------------------------------------------------
# Synthesis worker (sync and async)
# ---------------------------------------------------------------------------


def _synthesize(body: dict, progress_cb=None) -> dict:
    """Core synthesis logic. Returns dict with image_b64 + metadata."""
    from depthforge.core.depth_prep import DepthPrepParams, FalloffCurve, prep_depth
    from depthforge.core.pattern_gen import ColorMode, PatternParams, PatternType, generate_pattern
    from depthforge.core.synthesizer import StereoParams, synthesize

    def _progress(pct, msg=""):
        if progress_cb:
            progress_cb(pct, msg)

    _progress(5, "Decoding depth map")
    depth_raw = _b64_to_depth(body["depth_b64"])
    H, W = depth_raw.shape

    # ── Image size guard ────────────────────────────────────────────────
    _MAX_DIM = 8192
    if H > _MAX_DIM or W > _MAX_DIM:
        raise ValueError(f"Input image too large ({W}×{H}). Maximum dimension is {_MAX_DIM}px.")

    # ── Safe float/int helpers with clamping ────────────────────────────
    def _f(key, default, lo, hi):
        try:
            return float(min(hi, max(lo, float(body.get(key, default)))))
        except (TypeError, ValueError):
            return float(default)

    def _i(key, default, lo, hi):
        try:
            return int(min(hi, max(lo, int(body.get(key, default)))))
        except (TypeError, ValueError):
            return int(default)

    def _s(key, default, allowed: set):
        val = str(body.get(key, default)).lower().strip()
        return val if val in allowed else default

    mode = _s("mode", "texture", {"sirds", "texture", "anaglyph", "stereo_pair", "hidden"})
    _progress(10, "Preparing depth map")

    # Preset overrides
    preset_name = body.get("preset", "none")
    if preset_name and preset_name != "none":
        from depthforge.core.presets import get_preset

        try:
            preset = get_preset(preset_name)
            sp = preset.stereo_params
            dp = preset.depth_params
        except KeyError:
            raise ValueError(f"Unknown preset: {preset_name!r}")
    else:
        falloff_map = {
            "linear": FalloffCurve.LINEAR,
            "gamma": FalloffCurve.GAMMA,
            "s_curve": FalloffCurve.S_CURVE,
            "logarithmic": FalloffCurve.LOGARITHMIC,
            "exponential": FalloffCurve.EXPONENTIAL,
        }
        falloff_key = _s("falloff", "linear", set(falloff_map.keys()))
        dp = DepthPrepParams(
            invert=bool(body.get("invert_depth", False)),
            bilateral_sigma_space=_f("bilateral_space", 5.0, 0.0, 30.0),
            bilateral_sigma_color=_f("bilateral_color", 0.1, 0.01, 1.0),
            dilation_px=_i("dilation_px", 3, 0, 20),
            falloff_curve=falloff_map[falloff_key],
            near_plane=_f("near_plane", 0.0, 0.0, 1.0),
            far_plane=_f("far_plane", 1.0, 0.0, 1.0),
        )
        sp = StereoParams(
            depth_factor=_f("depth_factor", 0.35, -1.0, 1.0),
            max_parallax_fraction=_f("max_parallax", 0.033, 0.005, 0.1),
            safe_mode=bool(body.get("safe_mode", False)),
            oversample=_i("oversample", 1, 1, 4),
            seed=_i("seed", 42, 0, 2**31 - 1),
            invert_depth=False,
        )

    depth = prep_depth(depth_raw, dp)
    _progress(25, "Generating pattern")

    # Pattern selection — library name takes priority over type name
    pattern_name = body.get("pattern_name")
    if pattern_name:
        from depthforge.core.pattern_library import get_pattern

        tile_size = int(body.get("tile_size", 128))
        seed = int(body.get("seed", 42))
        scale = float(body.get("pattern_scale", 1.0))
        pattern = get_pattern(
            pattern_name, width=tile_size, height=tile_size, seed=seed, scale=scale
        )
    else:
        type_map = {
            "random_noise": PatternType.RANDOM_NOISE,
            "perlin": PatternType.PERLIN,
            "plasma": PatternType.PLASMA,
            "voronoi": PatternType.VORONOI,
            "geometric_grid": PatternType.GEOMETRIC_GRID,
            "mandelbrot": PatternType.MANDELBROT,
            "dot_matrix": PatternType.DOT_MATRIX,
        }
        color_map = {
            "greyscale": ColorMode.GREYSCALE,
            "monochrome": ColorMode.MONOCHROME,
            "psychedelic": ColorMode.PSYCHEDELIC,
        }
        pattern = generate_pattern(
            PatternParams(
                pattern_type=type_map.get(body.get("pattern_type", "plasma"), PatternType.PLASMA),
                tile_width=int(body.get("tile_size", 128)),
                tile_height=int(body.get("tile_size", 128)),
                color_mode=color_map.get(
                    body.get("color_mode", "psychedelic"), ColorMode.PSYCHEDELIC
                ),
                seed=int(body.get("seed", 42)),
            )
        )

    _progress(40, "Synthesizing stereogram")

    if mode in ("sirds", "texture"):
        result = synthesize(depth, pattern, sp)
        _progress(90, "Encoding output")

    elif mode == "anaglyph":
        from depthforge.core.anaglyph import AnaglyphMode, AnaglyphParams, make_anaglyph_from_depth

        source_b64 = body.get("source_b64")
        if source_b64:
            source = _b64_to_array(source_b64)
        else:
            source = np.stack(
                [
                    (depth * 220).astype(np.uint8),
                    (depth * 160).astype(np.uint8),
                    ((1 - depth) * 200).astype(np.uint8),
                    np.full(depth.shape, 255, np.uint8),
                ],
                axis=-1,
            )
        anaglyph_modes = {
            "true": AnaglyphMode.TRUE_ANAGLYPH,
            "grey": AnaglyphMode.GREY_ANAGLYPH,
            "colour": AnaglyphMode.COLOUR_ANAGLYPH,
            "half_colour": AnaglyphMode.HALF_COLOUR,
            "optimised": AnaglyphMode.OPTIMISED,
        }
        px = int(W * 0.065 * sp.depth_factor)
        params = AnaglyphParams(
            mode=anaglyph_modes.get(body.get("anaglyph_mode", "optimised"), AnaglyphMode.OPTIMISED),
            parallax_px=px,
        )
        result = make_anaglyph_from_depth(source, depth, params)
        _progress(90, "Encoding output")

    elif mode in ("stereopair", "stereo_pair"):
        from depthforge.core.stereo_pair import (
            StereoPairParams,
            compose_side_by_side,
            make_stereo_pair,
        )

        source_b64 = body.get("source_b64")
        if not source_b64:
            return {"error": "source_b64 is required for stereo_pair mode"}
        source = _b64_to_array(source_b64)
        params = StereoPairParams(
            max_parallax_fraction=sp.max_parallax_fraction,
            background_fill=body.get("bg_fill", "edge"),
        )
        left, right, _ = make_stereo_pair(source, depth, params)
        _progress(80, "Composing side-by-side")
        result = compose_side_by_side(left, right)
        _progress(90, "Encoding output")

    elif mode == "hidden":
        from depthforge.core.hidden_image import (
            HiddenImageParams,
            encode_hidden_image,
            shape_to_mask,
            text_to_mask,
        )

        text = body.get("hidden_text", "").strip()
        shape = body.get("hidden_shape", "star")
        if text:
            msk = text_to_mask(text, W, H)
        else:
            msk = shape_to_mask(shape, W, H)
        params = HiddenImageParams(
            foreground_depth=float(body.get("fg_depth", 0.75)),
            background_depth=float(body.get("bg_depth", 0.10)),
            edge_soften_px=int(body.get("edge_soften", 4)),
        )
        result = encode_hidden_image(pattern, msk, params)
        _progress(90, "Encoding output")
    else:
        result = synthesize(depth, pattern, sp)
        _progress(90, "Encoding output")

    image_b64 = _array_to_b64(result)
    _progress(100, "Done")

    # --- PSE flash safety check ---
    from depthforge.core.flash_safety import (
        EPILEPSY_WARNING_SHORT,
        check_pattern_flash_risk,
    )

    flash_report = check_pattern_flash_risk(pattern)
    pse_safe = flash_report.passed
    pse_risk = flash_report.risk.value
    pse_warning = None if sp.safe_mode else EPILEPSY_WARNING_SHORT
    if not flash_report.passed and not sp.safe_mode:
        pse_warning = (
            f"PSE RISK [{pse_risk}]: {'; '.join(flash_report.violations[:2])}  "
            f"Enable safe_mode or use the 'broadcast' preset."
        )

    return {
        "image_b64": image_b64,
        "width": result.shape[1],
        "height": result.shape[0],
        "mode": mode,
        "depth_factor": sp.depth_factor,
        "max_parallax": sp.max_parallax_fraction,
        "preset": preset_name,
        "pse_safe": pse_safe,
        "pse_risk": pse_risk,
        "pse_warning": pse_warning,
    }


def _run_synthesis(job_id: str, body: dict) -> None:
    """Thread worker for async synthesis."""
    try:

        def cb(pct, msg):
            _jobs.update(job_id, pct, msg)

        result = _synthesize(body, progress_cb=cb)
        _jobs.complete(job_id, result)
    except Exception as e:
        _jobs.fail(job_id, str(e))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run_server(
    host: str = "127.0.0.1",
    port: int = 5000,
    debug: bool = False,
    open_browser: bool = True,
) -> None:
    """Start the DepthForge preview server.

    Parameters
    ----------
    host : str
    port : int
    debug : bool
    open_browser : bool
        Automatically open the UI in the default browser.
    """
    app = create_app()

    if open_browser:
        import threading
        import webbrowser

        def _open():
            time.sleep(1.0)
            webbrowser.open(f"http://{host}:{port}")

        threading.Thread(target=_open, daemon=True).start()

    print(f"DepthForge preview server running at http://{host}:{port}")
    app.run(host=host, port=port, debug=debug, use_reloader=False)
