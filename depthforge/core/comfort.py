"""
depthforge.core.comfort
========================
Viewer comfort analysis for stereogram outputs.

This module provides:

- ``SafetyLimiter``     — clamps depth/parallax to configurable comfort bounds
- ``ComfortAnalyzer``   — full analysis: strain score, violation detection, advice
- ``ComfortReport``     — structured result with per-pixel maps and summary stats
- ``VergenceProfile``   — named vergence limit profiles (conservative, standard, relaxed)

References
----------
- SMPTE ST 2098-1: Mastering and distribution of stereoscopic 3D content
- EBU R95: Guidelines on the use of stereoscopic 3D television
- Lambooij et al. (2009): Visual Discomfort and Visual Fatigue of Stereoscopic Displays
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    import cv2 as _cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from scipy.ndimage import gaussian_filter as _scipy_gauss

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Vergence profiles
# ---------------------------------------------------------------------------


class VergenceProfile(Enum):
    """Named parallax limit profiles.

    Values represent ``max_parallax_fraction`` (fraction of frame width).
    """

    CONSERVATIVE = 0.020  # EBU R95 / broadcast — 2% of width
    STANDARD = 0.033  # 1/30 — classic comfort recommendation
    RELAXED = 0.050  # 5% — gallery/art contexts, experienced viewers
    CINEMA = 0.022  # SMPTE ST 2098-1
    CUSTOM = None  # user-defined


# ---------------------------------------------------------------------------
# Safety limiter
# ---------------------------------------------------------------------------


@dataclass
class SafetyLimiterParams:
    """Configuration for the depth/parallax safety limiter.

    Parameters
    ----------
    max_parallax_fraction : float
        Hard ceiling on parallax as a fraction of frame width.
        Default 1/30 (~3.33%).
    max_depth_factor : float
        Hard ceiling on depth_factor in StereoParams. Default 0.6.
    max_gradient : float
        Maximum allowed depth change between adjacent pixels (normalised).
        High local gradients cause rapid vergence changes — the leading
        cause of viewer discomfort. Default 0.15.
    gradient_blur_sigma : float
        Gaussian blur sigma applied to depth map to reduce spike gradients
        before the gradient check. Default 1.5.
    clamp_depth : bool
        If True, depth values are hard-clamped to [near_clip, far_clip].
        Default True.
    near_clip : float
        Minimum allowed depth value (0.0–1.0). Default 0.02.
    far_clip : float
        Maximum allowed depth value (0.0–1.0). Default 0.98.
    warn_on_violation : bool
        Issue a Python warning when limits are exceeded. Default True.
    """

    max_parallax_fraction: float = 1 / 30
    max_depth_factor: float = 0.60
    max_gradient: float = 0.15
    gradient_blur_sigma: float = 1.5
    clamp_depth: bool = True
    near_clip: float = 0.02
    far_clip: float = 0.98
    warn_on_violation: bool = True

    @classmethod
    def from_profile(cls, profile: VergenceProfile, **overrides) -> "SafetyLimiterParams":
        """Create params from a named vergence profile."""
        if profile == VergenceProfile.CUSTOM:
            return cls(**overrides)
        p = cls(max_parallax_fraction=profile.value, **overrides)
        return p


class SafetyLimiter:
    """Apply hard safety limits to a depth map and stereo parameters.

    Example
    -------
    ::

        from depthforge.core.comfort import SafetyLimiter, SafetyLimiterParams

        limiter = SafetyLimiter(SafetyLimiterParams(max_parallax_fraction=0.025))
        safe_depth, safe_params = limiter.apply(depth, stereo_params)
    """

    def __init__(self, params: Optional[SafetyLimiterParams] = None):
        self.params = params or SafetyLimiterParams()

    def apply(
        self,
        depth: np.ndarray,
        stereo_params=None,  # StereoParams — avoid circular import
    ):
        """Apply safety limits to depth and stereo params.

        Parameters
        ----------
        depth : np.ndarray
            Float32 depth map [0, 1].
        stereo_params : StereoParams, optional
            Will be clamped to safe limits if provided.

        Returns
        -------
        safe_depth : np.ndarray
            Depth map with gradients smoothed and values clamped.
        safe_stereo : StereoParams or None
            Clamped stereo params (copy), or None if not provided.
        violations : dict
            Dict of violation types found and what was corrected.
        """
        p = self.params
        violations = {}

        # 1. Clamp depth values
        safe_depth = depth.copy()
        if p.clamp_depth:
            clipped = np.clip(safe_depth, p.near_clip, p.far_clip)
            if not np.array_equal(clipped, safe_depth):
                violations["depth_clipped"] = (
                    f"Depth values outside [{p.near_clip}, {p.far_clip}] were clamped."
                )
            safe_depth = clipped

        # 2. Check and smooth high-gradient regions
        grad = _compute_gradient_magnitude(safe_depth)
        if grad.max() > p.max_gradient:
            violations["high_gradient"] = (
                f"Max depth gradient {grad.max():.3f} exceeded limit "
                f"{p.max_gradient:.3f}. Applied smoothing."
            )
            safe_depth = _smooth(safe_depth, p.gradient_blur_sigma)

        # 3. Clamp stereo params
        safe_stereo = None
        if stereo_params is not None:
            import copy

            safe_stereo = copy.copy(stereo_params)

            if safe_stereo.max_parallax_fraction > p.max_parallax_fraction:
                violations["parallax_fraction"] = (
                    f"max_parallax_fraction {safe_stereo.max_parallax_fraction:.4f} "
                    f"clamped to {p.max_parallax_fraction:.4f}."
                )
                safe_stereo.max_parallax_fraction = p.max_parallax_fraction

            if hasattr(safe_stereo, "depth_factor"):
                if abs(safe_stereo.depth_factor) > p.max_depth_factor:
                    new_df = p.max_depth_factor * np.sign(safe_stereo.depth_factor)
                    violations["depth_factor"] = (
                        f"depth_factor {safe_stereo.depth_factor:.3f} " f"clamped to {new_df:.3f}."
                    )
                    safe_stereo.depth_factor = new_df

        if violations and p.warn_on_violation:
            msg = "DepthForge SafetyLimiter applied corrections:\n" + "\n".join(
                f"  [{k}] {v}" for k, v in violations.items()
            )
            warnings.warn(msg, UserWarning, stacklevel=2)

        return safe_depth, safe_stereo, violations


# ---------------------------------------------------------------------------
# Comfort report
# ---------------------------------------------------------------------------


@dataclass
class ComfortReport:
    """Full comfort analysis result.

    Attributes
    ----------
    overall_score : float
        0.0 (severe discomfort risk) → 1.0 (fully comfortable).
    parallax_map : np.ndarray
        Per-pixel parallax magnitude in pixels (float32, H×W).
    vergence_map : np.ndarray
        Per-pixel vergence angle in degrees (float32, H×W).
    gradient_map : np.ndarray
        Per-pixel depth gradient magnitude (float32, H×W).
    violation_mask : np.ndarray
        Boolean mask of pixels exceeding comfort thresholds.
    window_violation_mask : np.ndarray
        Boolean mask of stereo window violations.
    max_parallax_px : float
        Maximum parallax found in the image (pixels).
    max_vergence_deg : float
        Maximum vergence angle found (degrees).
    pct_uncomfortable : float
        Percentage of pixels that exceed the comfort threshold.
    violations : list[str]
        Human-readable list of comfort issues found.
    advice : list[str]
        Actionable suggestions to improve comfort.
    passed : bool
        True if no comfort thresholds were exceeded.
    """

    overall_score: float
    parallax_map: np.ndarray
    vergence_map: np.ndarray
    gradient_map: np.ndarray
    violation_mask: np.ndarray
    window_violation_mask: np.ndarray
    max_parallax_px: float
    max_vergence_deg: float
    pct_uncomfortable: float
    violations: list
    advice: list
    passed: bool

    def summary(self) -> str:
        """Human-readable summary string."""
        status = "✓ PASS" if self.passed else "✗ FAIL"
        lines = [
            f"Comfort Analysis  {status}  (score: {self.overall_score:.2f}/1.0)",
            f"  Max parallax:   {self.max_parallax_px:.1f}px",
            f"  Max vergence:   {self.max_vergence_deg:.2f}°",
            f"  Uncomfortable:  {self.pct_uncomfortable:.1f}% of pixels",
        ]
        if self.violations:
            lines.append("  Issues:")
            for v in self.violations:
                lines.append(f"    · {v}")
        if self.advice:
            lines.append("  Advice:")
            for a in self.advice:
                lines.append(f"    → {a}")
        return "\n".join(lines)

    def print(self) -> None:
        """Print the summary to stdout."""
        import sys
        text = self.summary()
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        sys.stdout.write(text.encode(enc, errors="replace").decode(enc) + "\n")


# ---------------------------------------------------------------------------
# Comfort analyzer
# ---------------------------------------------------------------------------


@dataclass
class ComfortAnalyzerParams:
    """Configuration for the comfort analyzer.

    Parameters
    ----------
    frame_width : int
        Output frame width in pixels. Required to convert fractions to px.
    frame_height : int
        Output frame height in pixels.
    max_parallax_fraction : float
        Comfort ceiling as fraction of frame width. Default 1/30.
    max_vergence_deg : float
        Max vergence change considered comfortable (degrees). Default 1.5.
    max_gradient : float
        Max depth gradient per pixel (normalised). Default 0.12.
    screen_distance_mm : float
        Estimated viewer-to-screen distance in mm. Default 600 (arm's length).
    eye_separation_mm : float
        Inter-ocular distance in mm. Default 65.
    depth_factor : float
        depth_factor being used in synthesis. Needed to convert depth → px.
    """

    frame_width: int = 1920
    frame_height: int = 1080
    max_parallax_fraction: float = 1 / 30
    max_vergence_deg: float = 1.5
    max_gradient: float = 0.12
    screen_distance_mm: float = 600.0
    eye_separation_mm: float = 65.0
    depth_factor: float = 0.35


class ComfortAnalyzer:
    """Analyse a depth map for viewer comfort and produce a ComfortReport.

    Example
    -------
    ::

        from depthforge.core.comfort import ComfortAnalyzer, ComfortAnalyzerParams

        analyzer = ComfortAnalyzer(ComfortAnalyzerParams(
            frame_width=1920, frame_height=1080, depth_factor=0.4
        ))
        report = analyzer.analyze(depth)
        report.print()

        if not report.passed:
            # Show the violation overlay
            overlay = analyzer.violation_overlay(depth, report)
    """

    def __init__(self, params: Optional[ComfortAnalyzerParams] = None):
        self.params = params or ComfortAnalyzerParams()

    def analyze(self, depth: np.ndarray) -> ComfortReport:
        """Run full comfort analysis on a depth map.

        Parameters
        ----------
        depth : np.ndarray
            Float32 depth map [0, 1], shape (H, W).

        Returns
        -------
        ComfortReport
        """
        p = self.params
        H, W = depth.shape[:2]

        # --- Parallax map (px) ---
        max_shift_px = p.frame_width * p.max_parallax_fraction
        parallax_map = (depth * max_shift_px * p.depth_factor).astype(np.float32)

        # --- Vergence map ---
        vergence_map = _compute_vergence_map(
            depth,
            eye_sep_mm=p.eye_separation_mm,
            screen_dist_mm=p.screen_distance_mm,
        )

        # --- Gradient map ---
        gradient_map = _compute_gradient_magnitude(depth)

        # --- Violation masks ---
        parallax_limit_px = p.frame_width * p.max_parallax_fraction
        violation_mask = (
            (parallax_map > parallax_limit_px)
            | (vergence_map > p.max_vergence_deg)
            | (gradient_map > p.max_gradient)
        ).astype(bool)

        window_violation_mask = _detect_window_violations(depth, threshold=0.05)

        # --- Metrics ---
        max_parallax_px = float(parallax_map.max())
        max_vergence_deg = float(vergence_map.max())
        pct_uncomfortable = float(violation_mask.mean() * 100.0)

        # --- Score (0–1): penalise each violation type proportionally ---
        parallax_score = 1.0 - min(max_parallax_px / (parallax_limit_px * 1.5 + 1e-6), 1.0)
        vergence_score = 1.0 - min(max_vergence_deg / (p.max_vergence_deg * 1.5 + 1e-6), 1.0)
        gradient_score = 1.0 - min(gradient_map.max() / (p.max_gradient * 1.5 + 1e-6), 1.0)
        pct_score = 1.0 - min(pct_uncomfortable / 10.0, 1.0)  # 10% = 0

        overall_score = np.clip(
            0.35 * parallax_score
            + 0.25 * vergence_score
            + 0.20 * gradient_score
            + 0.20 * pct_score,
            0.0,
            1.0,
        )

        # --- Violations + advice ---
        violations = []
        advice = []

        if max_parallax_px > parallax_limit_px:
            violations.append(
                f"Max parallax {max_parallax_px:.1f}px exceeds "
                f"comfort limit {parallax_limit_px:.1f}px "
                f"({p.max_parallax_fraction:.3f} × {p.frame_width}px)."
            )
            advice.append(
                f"Reduce depth_factor (currently {p.depth_factor:.2f}) or "
                f"lower max_parallax_fraction below {p.max_parallax_fraction:.3f}."
            )

        if max_vergence_deg > p.max_vergence_deg:
            violations.append(
                f"Max vergence {max_vergence_deg:.2f}° exceeds limit {p.max_vergence_deg:.2f}°."
            )
            advice.append(
                "Apply more bilateral smoothing to the depth map to reduce rapid vergence changes."
            )

        if gradient_map.max() > p.max_gradient:
            violations.append(
                f"Depth gradient {gradient_map.max():.3f} exceeds limit {p.max_gradient:.3f}. "
                "Rapid depth changes will cause eye strain."
            )
            advice.append(
                "Increase bilateral_sigma_space in DepthPrepParams to smooth sharp depth transitions."
            )

        if window_violation_mask.any():
            violations.append(
                "Stereo window violations detected: near objects extend to the frame edge "
                "without sufficient depth falloff."
            )
            advice.append(
                "Apply depth reduction near frame edges, or use near_clip in DepthPrepParams "
                "to pull back the closest elements."
            )

        if pct_uncomfortable > 5.0:
            violations.append(f"{pct_uncomfortable:.1f}% of pixels exceed comfort thresholds.")
            advice.append(
                "Consider switching to the 'shallow' or 'broadcast' preset for this content."
            )

        passed = len(violations) == 0

        return ComfortReport(
            overall_score=float(overall_score),
            parallax_map=parallax_map,
            vergence_map=vergence_map,
            gradient_map=gradient_map,
            violation_mask=violation_mask,
            window_violation_mask=window_violation_mask,
            max_parallax_px=max_parallax_px,
            max_vergence_deg=max_vergence_deg,
            pct_uncomfortable=pct_uncomfortable,
            violations=violations,
            advice=advice,
            passed=passed,
        )

    def violation_overlay(
        self,
        depth: np.ndarray,
        report: Optional[ComfortReport] = None,
        alpha: float = 0.6,
    ) -> np.ndarray:
        """Render a colour overlay showing comfort violations.

        Parameters
        ----------
        depth : np.ndarray
            Float32 depth map (H, W).
        report : ComfortReport, optional
            Pre-computed report. If None, runs analyze() internally.
        alpha : float
            Overlay opacity (0 = invisible, 1 = opaque). Default 0.6.

        Returns
        -------
        np.ndarray  uint8 RGBA (H, W, 4)  — visualisation image.
        """
        if report is None:
            report = self.analyze(depth)

        H, W = depth.shape[:2]

        # Base: greyscale depth
        grey = (depth * 255).astype(np.uint8)
        out = np.stack([grey, grey, grey, np.full((H, W), 255, np.uint8)], axis=-1)

        def _blend(mask, r, g, b):
            m = mask.astype(np.float32)[:, :, None]
            colour = np.array([r, g, b, 255], dtype=np.float32)
            out[..., :3] = (
                out[..., :3].astype(np.float32) * (1 - m * alpha) + colour[:3] * m * alpha
            ).astype(np.uint8)

        # Parallax violation → red
        parallax_limit_px = self.params.frame_width * self.params.max_parallax_fraction
        _blend(report.parallax_map > parallax_limit_px, 220, 40, 40)

        # Gradient violation → orange
        _blend(report.gradient_map > self.params.max_gradient, 220, 130, 30)

        # Vergence violation → yellow
        _blend(report.vergence_map > self.params.max_vergence_deg, 220, 200, 30)

        # Window violation → magenta
        _blend(report.window_violation_mask, 180, 40, 200)

        return out


# ---------------------------------------------------------------------------
# Vergence strain estimator
# ---------------------------------------------------------------------------


def estimate_vergence_strain(
    depth: np.ndarray,
    depth_factor: float = 0.35,
    frame_width: int = 1920,
    screen_distance_mm: float = 600.0,
    eye_separation_mm: float = 65.0,
) -> dict:
    """Estimate vergence strain metrics for a depth map.

    Returns a dict with:
    - ``mean_vergence_deg``: mean vergence angle across the image
    - ``max_vergence_deg``: maximum vergence angle
    - ``vergence_range_deg``: total vergence range (max - min)
    - ``rapid_change_fraction``: fraction of pixels where vergence changes
      faster than 0.5°/px (leading cause of visual fatigue)
    - ``strain_rating``: qualitative rating ("Low", "Moderate", "High", "Severe")
    """
    vmap = _compute_vergence_map(depth, eye_separation_mm, screen_distance_mm)

    # Apply depth scaling
    vmap = vmap * depth_factor

    mean_v = float(vmap.mean())
    max_v = float(vmap.max())
    min_v = float(vmap.min())
    rng_v = max_v - min_v

    # Rapid spatial vergence change (≥0.5°/px)
    gy, gx = np.gradient(vmap)
    change_mag = np.sqrt(gx**2 + gy**2)
    rapid_fraction = float((change_mag > 0.5).mean())

    # Qualitative rating
    if max_v < 1.0 and rapid_fraction < 0.02:
        rating = "Low"
    elif max_v < 2.0 and rapid_fraction < 0.08:
        rating = "Moderate"
    elif max_v < 3.5 and rapid_fraction < 0.20:
        rating = "High"
    else:
        rating = "Severe"

    return {
        "mean_vergence_deg": mean_v,
        "max_vergence_deg": max_v,
        "vergence_range_deg": rng_v,
        "rapid_change_fraction": rapid_fraction,
        "strain_rating": rating,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_vergence_map(
    depth: np.ndarray,
    eye_sep_mm: float = 65.0,
    screen_dist_mm: float = 600.0,
) -> np.ndarray:
    """Per-pixel vergence angle in degrees."""
    # depth=1.0 → object at screen plane; depth=0.0 → far background
    viewing_dist = screen_dist_mm * (1.0 + (1.0 - depth) * 2.0)
    vergence_rad = np.arctan2(eye_sep_mm, viewing_dist)
    return np.degrees(vergence_rad).astype(np.float32)


def _compute_gradient_magnitude(depth: np.ndarray) -> np.ndarray:
    """Gradient magnitude of depth map using Sobel or np.gradient."""
    if HAS_CV2:
        d8 = (depth * 255).astype(np.uint8)
        gx = _cv2.Sobel(d8, _cv2.CV_32F, 1, 0, ksize=3)
        gy = _cv2.Sobel(d8, _cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2) / 255.0
    else:
        gy, gx = np.gradient(depth)
        mag = np.sqrt(gx**2 + gy**2)
    return mag.astype(np.float32)


def _smooth(depth: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth depth map — OpenCV > SciPy > NumPy fallbacks."""
    if sigma <= 0:
        return depth
    if HAS_CV2:
        ksize = max(3, int(sigma * 3) | 1)  # odd kernel
        return _cv2.GaussianBlur(depth, (ksize, ksize), sigma)
    if HAS_SCIPY:
        return _scipy_gauss(depth, sigma=sigma).astype(np.float32)
    # NumPy box-blur fallback
    k = max(1, int(sigma * 2))
    kernel = np.ones((k * 2 + 1, k * 2 + 1), np.float32) / ((k * 2 + 1) ** 2)
    from numpy.lib.stride_tricks import sliding_window_view

    pad = k
    padded = np.pad(depth, pad, mode="edge")
    windows = sliding_window_view(padded, (k * 2 + 1, k * 2 + 1))
    return windows.mean(axis=(-2, -1)).astype(np.float32)


def _detect_window_violations(depth: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """Detect stereo window violations at frame edges."""
    H, W = depth.shape
    violation = np.zeros((H, W), dtype=bool)
    border = max(5, int(min(H, W) * 0.02))
    near_t = 1.0 - threshold

    edges = [
        depth[:border, :],
        depth[-border:, :],
        depth[:, :border],
        depth[:, -border:],
    ]
    if any(e.max() > near_t for e in edges):
        # Mark frame border region
        mask = np.zeros((H, W), dtype=bool)
        mask[:border, :] = True
        mask[-border:, :] = True
        mask[:, :border] = True
        mask[:, -border:] = True
        violation = mask

    return violation
