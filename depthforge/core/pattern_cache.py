"""
depthforge.core.pattern_cache
==============================
LRU pattern tile cache for video sequence processing.

During video rendering the same pattern tile is synthesised thousands of
times — once per frame. The cache stores computed pattern tiles keyed by
their generation parameters so they are generated only once.

For GPU-accelerated paths (future) the cache also holds the device-side
copy of each tile and avoids re-uploading between frames.

Public API
----------
``PatternCache(max_tiles=32)``
    ``.get(params) -> np.ndarray``          cached or newly generated
    ``.get_or_load(path, size) -> np.ndarray``  file-backed tile
    ``.preload(params_list)``               warm the cache in background
    ``.stats() -> dict``                    hit/miss counters
    ``.clear()``                            evict all entries
    ``.invalidate(params)``                 evict one entry

``cached_pattern(params, cache=None) -> np.ndarray``
    Module-level convenience with a global default cache instance.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

from depthforge.core.pattern_gen import generate_pattern, PatternParams
from depthforge.core.pattern_library import get_pattern, PatternEntry


# ---------------------------------------------------------------------------
# Cache key helpers
# ---------------------------------------------------------------------------

def _params_key(params: PatternParams) -> str:
    """Deterministic hash of PatternParams for use as cache key."""
    d = asdict(params)
    # Normalise enum values to their string names so the key is stable
    # across Python sessions
    for k, v in d.items():
        if hasattr(v, "name"):
            d[k] = v.name
    canonical = json.dumps(d, sort_keys=True)
    return hashlib.sha1(canonical.encode()).hexdigest()[:16]


def _library_key(name: str, width: int, height: int, seed: int, scale: float) -> str:
    canonical = json.dumps(
        {"name": name, "w": width, "h": height, "seed": seed, "scale": scale},
        sort_keys=True,
    )
    return "lib_" + hashlib.sha1(canonical.encode()).hexdigest()[:14]


def _file_key(path: Union[str, Path], width: int, height: int) -> str:
    p = str(Path(path).resolve())
    canonical = json.dumps({"path": p, "w": width, "h": height})
    return "file_" + hashlib.sha1(canonical.encode()).hexdigest()[:14]


# ---------------------------------------------------------------------------
# PatternCache
# ---------------------------------------------------------------------------

class PatternCache:
    """
    Thread-safe LRU cache for generated pattern tiles.

    Parameters
    ----------
    max_tiles : int
        Maximum number of tiles to keep in memory. Each 256×256 RGBA tile
        is ~256 KB, so 32 tiles ≈ 8 MB.
    """

    def __init__(self, max_tiles: int = 32):
        self._max = max_tiles
        self._store: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Core get/put
    # ------------------------------------------------------------------

    def _get_raw(self, key: str) -> Optional[np.ndarray]:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)   # mark as recently used
                self._hits += 1
                return self._store[key]
            self._misses += 1
            return None

    def _put_raw(self, key: str, tile: np.ndarray) -> None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                return
            self._store[key] = tile
            if len(self._store) > self._max:
                self._store.popitem(last=False)  # evict LRU

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get(self, params: PatternParams) -> np.ndarray:
        """
        Return a pattern tile for the given PatternParams.

        The tile is generated once and cached. Subsequent calls with
        identical params return the cached array without copying.
        """
        key = _params_key(params)
        tile = self._get_raw(key)
        if tile is None:
            tile = generate_pattern(params)
            self._put_raw(key, tile)
        return tile

    def get_library(
        self,
        name: str,
        width: int = 128,
        height: int = 128,
        seed: int = 42,
        scale: float = 1.0,
    ) -> np.ndarray:
        """Return a named pattern-library tile, cached."""
        key = _library_key(name, width, height, seed, scale)
        tile = self._get_raw(key)
        if tile is None:
            tile = get_pattern(name, width, height, seed=seed, scale=scale)
            self._put_raw(key, tile)
        return tile

    def get_or_load(
        self,
        path: Union[str, Path],
        width: int,
        height: int,
    ) -> np.ndarray:
        """
        Load a pattern tile from disk, resize to (height, width), and cache.

        The on-disk image is loaded as RGBA uint8. If the image is smaller
        than the requested size it is tiled; if larger it is cropped to
        the top-left region.
        """
        key = _file_key(path, width, height)
        tile = self._get_raw(key)
        if tile is None:
            img = Image.open(path).convert("RGBA")
            iw, ih = img.size
            if iw < width or ih < height:
                # tile the source image
                reps_x = -(-width // iw)   # ceiling division
                reps_y = -(-height // ih)
                canvas = Image.new("RGBA", (iw * reps_x, ih * reps_y))
                for ty in range(reps_y):
                    for tx in range(reps_x):
                        canvas.paste(img, (tx * iw, ty * ih))
                img = canvas
            tile = np.array(img.crop((0, 0, width, height)), dtype=np.uint8)
            self._put_raw(key, tile)
        return tile

    def put(self, key: str, tile: np.ndarray) -> None:
        """Manually insert a pre-computed tile under a custom key."""
        self._put_raw(key, tile)

    def preload(self, params_list: list[PatternParams]) -> None:
        """
        Warm the cache for a list of PatternParams in background threads.

        Returns immediately; generation happens concurrently.
        """
        def _generate(p: PatternParams) -> None:
            self.get(p)

        with ThreadPoolExecutor(max_workers=min(len(params_list), 4)) as pool:
            list(pool.map(_generate, params_list))

    def invalidate(self, params: PatternParams) -> bool:
        """
        Evict one entry from the cache.

        Returns True if the entry existed.
        """
        key = _params_key(params)
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def clear(self) -> None:
        """Evict all cached tiles."""
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict:
        """Return cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._store),
                "max_tiles": self._max,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / max(total, 1), 3),
            }

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"PatternCache(size={s['size']}/{s['max_tiles']}, "
            f"hit_rate={s['hit_rate']:.1%})"
        )


# ---------------------------------------------------------------------------
# Module-level default cache
# ---------------------------------------------------------------------------

_default_cache: Optional[PatternCache] = None
_cache_lock = threading.Lock()


def _get_default_cache() -> PatternCache:
    global _default_cache
    with _cache_lock:
        if _default_cache is None:
            _default_cache = PatternCache(max_tiles=64)
        return _default_cache


def cached_pattern(
    params: PatternParams,
    cache: Optional[PatternCache] = None,
) -> np.ndarray:
    """
    Return a cached pattern tile for ``params``.

    Uses the module-level default cache unless ``cache`` is provided.
    """
    c = cache or _get_default_cache()
    return c.get(params)
