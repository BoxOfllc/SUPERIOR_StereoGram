"""
depthforge.core.tiled_render
============================
Multi-threaded tiled renderer for 4K/8K frame processing.

Splits the output frame into horizontal tile strips and processes them
concurrently using a ThreadPoolExecutor. Each strip is synthesised
independently by the core synthesizer, then stitched back together.

The tile height is auto-tuned to balance thread overhead against
parallelism. For machines with N cores the optimal strip count is 2*N
(keeps threads busy even when one strip takes longer than average).

Thread safety
-------------
Each worker gets its own copy of ``StereoParams`` (frozen dataclass) and
operates on a read-only slice of the depth array, so no locking is needed.
The output array is pre-allocated and workers write to non-overlapping
row ranges.

Public API
----------
``TiledRenderer(params, n_threads=None)``
    ``render(depth, pattern) -> np.ndarray``   RGBA uint8 (H, W, 4)
    ``render_region(depth, pattern, y0, y1)``  render a sub-region

``tile_synthesize(depth, pattern, params, n_threads=None)``
    Convenience wrapper — drop-in replacement for ``synthesize()``.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from typing import Optional

import numpy as np

from depthforge.core.synthesizer import synthesize, StereoParams


# ---------------------------------------------------------------------------
# Auto-tune helpers
# ---------------------------------------------------------------------------

def _cpu_count() -> int:
    """Return usable CPU count, capped at 16 to avoid over-subscription."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    return min(n, 16)


def _optimal_strip_count(height: int, n_threads: int) -> int:
    """
    Compute strip count so each strip is at least 32 rows and we have
    2× more strips than threads for load-balancing.
    """
    ideal = n_threads * 2
    min_strip_h = 32
    max_strips = max(1, height // min_strip_h)
    return min(ideal, max_strips)


# ---------------------------------------------------------------------------
# Core worker
# ---------------------------------------------------------------------------

def _render_strip(
    depth_strip: np.ndarray,      # (strip_h, W) float32
    pattern: np.ndarray,          # (tile_h, tile_w, 4) uint8
    params: StereoParams,
    y_offset: int,                # for seeded RNG reproducibility
) -> np.ndarray:
    """Render one horizontal strip. Returns RGBA uint8 (strip_h, W, 4)."""
    # Adjust seed so each strip gets a deterministic but distinct RNG state
    strip_params = replace(params, seed=params.seed + y_offset)
    return synthesize(depth_strip, pattern, strip_params)


# ---------------------------------------------------------------------------
# TiledRenderer
# ---------------------------------------------------------------------------

class TiledRenderer:
    """
    Multi-threaded tiled renderer.

    Parameters
    ----------
    params : StereoParams
        Synthesis parameters shared across all tiles.
    n_threads : int, optional
        Number of worker threads. Defaults to CPU count.
    strip_count : int, optional
        Number of horizontal strips to divide the frame into.
        Auto-tuned if not set.
    """

    def __init__(
        self,
        params: StereoParams,
        n_threads: Optional[int] = None,
        strip_count: Optional[int] = None,
    ):
        self.params = params
        self.n_threads = n_threads or _cpu_count()
        self._strip_count = strip_count  # None = auto

    # ------------------------------------------------------------------
    def render(
        self,
        depth: np.ndarray,    # (H, W) float32
        pattern: np.ndarray,  # (tile_h, tile_w, 4) uint8
    ) -> np.ndarray:
        """
        Render the full frame using tiled multithreading.

        Returns
        -------
        np.ndarray
            RGBA uint8 array of shape (H, W, 4).
        """
        H, W = depth.shape[:2]

        # Single-threaded fast path for small frames
        if H * W < 640 * 480 or self.n_threads == 1:
            return synthesize(depth, pattern, self.params)

        n_strips = self._strip_count or _optimal_strip_count(H, self.n_threads)
        strip_height = math.ceil(H / n_strips)

        # Build strip slices
        slices: list[tuple[int, int]] = []
        y = 0
        while y < H:
            y1 = min(y + strip_height, H)
            slices.append((y, y1))
            y = y1

        # Pre-allocate output
        output = np.empty((H, W, 4), dtype=np.uint8)

        # Launch workers
        with ThreadPoolExecutor(max_workers=self.n_threads) as pool:
            futures = {
                pool.submit(
                    _render_strip,
                    depth[y0:y1],
                    pattern,
                    self.params,
                    y0,
                ): (y0, y1)
                for y0, y1 in slices
            }
            for future in as_completed(futures):
                y0, y1 = futures[future]
                output[y0:y1] = future.result()

        return output

    # ------------------------------------------------------------------
    def render_region(
        self,
        depth: np.ndarray,
        pattern: np.ndarray,
        y0: int,
        y1: int,
    ) -> np.ndarray:
        """Render a sub-region of the frame (used by OFX render-on-demand)."""
        strip_params = replace(self.params, seed=self.params.seed + y0)
        return synthesize(depth[y0:y1], pattern, strip_params)

    # ------------------------------------------------------------------
    def benchmark(
        self,
        width: int = 1920,
        height: int = 1080,
        tile_size: int = 128,
    ) -> dict:
        """
        Run a quick benchmark and return timing info.

        Returns
        -------
        dict with keys: width, height, n_threads, strip_count,
            single_ms, tiled_ms, speedup
        """
        import time
        from depthforge.core.pattern_gen import generate_pattern, PatternParams, PatternType

        rng = np.random.default_rng(42)
        depth = rng.random((height, width), dtype=np.float32)
        p_params = PatternParams(
            pattern_type=PatternType.RANDOM_NOISE,
            tile_width=tile_size,
            tile_height=tile_size,
        )
        pattern = generate_pattern(p_params)

        # Single-threaded
        t0 = time.perf_counter()
        synthesize(depth, pattern, self.params)
        single_ms = (time.perf_counter() - t0) * 1000

        # Tiled multi-threaded
        t0 = time.perf_counter()
        self.render(depth, pattern)
        tiled_ms = (time.perf_counter() - t0) * 1000

        n_strips = self._strip_count or _optimal_strip_count(height, self.n_threads)

        return {
            "width": width,
            "height": height,
            "n_threads": self.n_threads,
            "strip_count": n_strips,
            "single_ms": round(single_ms, 1),
            "tiled_ms": round(tiled_ms, 1),
            "speedup": round(single_ms / max(tiled_ms, 0.001), 2),
        }


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def tile_synthesize(
    depth: np.ndarray,
    pattern: np.ndarray,
    params: StereoParams,
    n_threads: Optional[int] = None,
) -> np.ndarray:
    """
    Drop-in replacement for ``synthesize()`` with automatic multithreading.

    For frames smaller than 640×480 or when n_threads=1, falls back to
    single-threaded ``synthesize()`` with no overhead.
    """
    renderer = TiledRenderer(params, n_threads=n_threads)
    return renderer.render(depth, pattern)
