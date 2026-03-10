"""
depthforge.core.flash_safety
============================
Photosensitive epilepsy (PSE) flash and flicker safety checks.

Standards implemented
---------------------
- W3C WCAG 2.3 Success Criterion 2.3.1 "Three Flashes or Below Threshold"
  https://www.w3.org/TR/WCAG21/#three-flashes-or-below-threshold
- ITU-R BT.1702-2 (2012): Guidance for the reduction of photosensitive
  epileptic seizures caused by television
- Ofcom Broadcasting Code Section Two: Harm and Offence — Flashing Images
  (applies the Harding PSE Test thresholds)
- Harding PSE Test: industry-standard broadcast flash analysis tool
- IEC 61966-2-1 / Rec. 709: Relative luminance for digital displays

Flash definition (WCAG / ITU-R BT.1702)
-----------------------------------------
A "general flash" is a pair of opposing luminance transitions where:
  1. Each luminance change is ≥ 10 % of peak white (ΔL ≥ 0.10, normalised 0–1).
  2. The lower luminance state is below 0.80 relative luminance.

A "red flash" is a pair of opposing transitions in red chrominance (R − Y)
where each change is ≥ 0.20 (Harding PSE Test threshold).

Threshold: no more than **3 paired flash events per second** for either type.

Pattern contrast risk (static patterns)
-----------------------------------------
For static patterns the risk is estimated from Michelson contrast
  (Lmax − Lmin) / (Lmax + Lmin)
and red chrominance range.  High-contrast patterns shown at video frame rates
are at risk of exceeding the 3 Hz general / red flash thresholds.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Mandatory warning text (keep these at module scope for easy import)
# ---------------------------------------------------------------------------

EPILEPSY_WARNING = (
    "PHOTOSENSITIVE EPILEPSY WARNING: This software generates high-contrast "
    "visual patterns that may trigger seizures in people with photosensitive "
    "epilepsy (PSE) or other photosensitive conditions.  Approximately 1 in "
    "4,000 people is affected.  Always enable safe_mode or use the 'broadcast' "
    "preset for publicly distributed content.  For guidance see: "
    "https://www.epilepsy.org.uk/info/photosensitive-epilepsy"
)

EPILEPSY_WARNING_SHORT = (
    "WARNING: High-contrast patterns may trigger photosensitive seizures. "
    "Enable safe_mode for public distribution."
)

# ---------------------------------------------------------------------------
# Risk classification
# ---------------------------------------------------------------------------


class EpilepticRisk(Enum):
    """Photosensitive epilepsy risk classification.

    SAFE     — passes all thresholds; suitable for public distribution.
    LOW      — minor concerns; acceptable for most audiences with a warning.
    MODERATE — caution advised; display epilepsy warning to end users.
    HIGH     — likely problematic; must enable safe_mode before distributing.
    CRITICAL — fails regulatory thresholds; do not distribute without fix.
    """

    SAFE = "SAFE"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class FlashSafetyConfig:
    """Thresholds for flash / flicker safety analysis.

    Parameters
    ----------
    max_flashes_per_second : float
        Maximum allowed flash pairs per second.
        W3C WCAG 2.3.1 / ITU-R BT.1702 limit: 3.0 Hz.
    luminance_change_threshold : float
        Minimum relative luminance change to count as a flash transition.
        Ofcom / WCAG limit: 0.10 (10 % of peak white, normalised 0–1).
    red_chrominance_threshold : float
        Minimum red chrominance (R − Y) change to count as a red flash.
        Harding PSE Test limit: 0.20.
    pattern_contrast_limit : float
        Michelson contrast above which a static pattern is flagged as risky.
        Conservative safe limit: 0.50 (50 % contrast).
    """

    max_flashes_per_second: float = 3.0
    luminance_change_threshold: float = 0.10
    red_chrominance_threshold: float = 0.20
    pattern_contrast_limit: float = 0.50


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class FlashSafetyReport:
    """Result from a flash / flicker safety check.

    Attributes
    ----------
    risk : EpilepticRisk
        Overall risk classification.
    passed : bool
        True if all checked thresholds are within safe limits.
    flash_frequency_hz : float
        Measured or estimated flash frequency in Hz.
        0.0 for static pattern checks (not applicable).
    max_luminance_delta : float
        Maximum relative luminance change detected (0–1).
    max_red_delta : float
        Maximum red chrominance change detected (0–1).
        0.0 for static checks.
    michelson_contrast : float
        Michelson contrast: (Lmax − Lmin) / (Lmax + Lmin).
    violations : list[str]
        Human-readable list of threshold violations found.
    advice : list[str]
        Actionable recommendations to bring the content within safe limits.
    standard_results : dict[str, bool]
        Pass / fail result per regulatory standard.
        Keys: "WCAG_2.3.1", "ITU_BT1702", "Ofcom", "Harding_PSE".
    """

    risk: EpilepticRisk
    passed: bool
    flash_frequency_hz: float
    max_luminance_delta: float
    max_red_delta: float
    michelson_contrast: float
    violations: List[str] = field(default_factory=list)
    advice: List[str] = field(default_factory=list)
    standard_results: dict = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary string."""
        status = "✓ PASS" if self.passed else "✗ FAIL"
        lines = [
            f"Flash Safety  {status}  [{self.risk.value}]",
            f"  Flash rate:         {self.flash_frequency_hz:.2f} Hz  (limit 3.0 Hz)",
            f"  Luminance delta:    {self.max_luminance_delta:.3f}  (limit 0.10)",
            f"  Michelson contrast: {self.michelson_contrast:.3f}  (limit 0.50)",
        ]
        for std, ok in self.standard_results.items():
            lines.append(f"  {'✓' if ok else '✗'} {std}")
        if self.violations:
            lines.append("  Issues:")
            for v in self.violations:
                lines.append(f"    · {v}")
        if self.advice:
            lines.append("  Advice:")
            for a in self.advice:
                lines.append(f"    → {a}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Luminance helpers  (Rec. 709 / IEC 61966-2-1)
# ---------------------------------------------------------------------------


def _relative_luminance(rgba: np.ndarray) -> np.ndarray:
    """Per-pixel relative luminance (Y) using Rec. 709 primaries.

    Parameters
    ----------
    rgba : np.ndarray
        uint8 or float32 image (H, W, 4) RGBA.

    Returns
    -------
    np.ndarray  float32 (H, W) in [0, 1].
    """
    if rgba.dtype == np.uint8:
        rgb = rgba[..., :3].astype(np.float32) / 255.0
    else:
        rgb = np.clip(rgba[..., :3].astype(np.float32), 0.0, 1.0)

    # Linearise sRGB  (IEC 61966-2-1 piecewise transfer function)
    linear = np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )

    # Rec. 709 luminance coefficients
    Y = 0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]
    return Y.astype(np.float32)


def _red_chrominance(rgba: np.ndarray) -> np.ndarray:
    """Per-pixel red chrominance (R − Y) for red-flash detection.

    Returns float32 (H, W) in approximately [−1, 1].
    """
    Y = _relative_luminance(rgba)
    if rgba.dtype == np.uint8:
        R = rgba[..., 0].astype(np.float32) / 255.0
    else:
        R = np.clip(rgba[..., 0].astype(np.float32), 0.0, 1.0)
    return (R - Y).astype(np.float32)


# ---------------------------------------------------------------------------
# Static pattern analysis
# ---------------------------------------------------------------------------


def check_pattern_flash_risk(
    pattern_rgba: np.ndarray,
    config: Optional[FlashSafetyConfig] = None,
) -> FlashSafetyReport:
    """Analyse a static pattern tile for photosensitive epilepsy risk.

    Evaluates Michelson contrast, luminance range, and red chrominance.
    High-contrast patterns shown at video frame rates risk exceeding the
    3 Hz general / red flash thresholds defined by WCAG 2.3.1 and
    ITU-R BT.1702.

    Parameters
    ----------
    pattern_rgba : np.ndarray
        uint8 RGBA pattern tile (H, W, 4).
    config : FlashSafetyConfig, optional
        Override thresholds.  Standard limits used if omitted.

    Returns
    -------
    FlashSafetyReport
    """
    cfg = config or FlashSafetyConfig()

    Y = _relative_luminance(pattern_rgba)
    L_max = float(Y.max())
    L_min = float(Y.min())

    denom = L_max + L_min
    michelson = float((L_max - L_min) / (denom + 1e-8))
    lum_delta = L_max - L_min

    red = _red_chrominance(pattern_rgba)
    max_red_delta = float(red.max() - red.min())

    violations: List[str] = []
    advice: List[str] = []

    # --- Contrast check ---
    if michelson > cfg.pattern_contrast_limit:
        violations.append(
            f"Michelson contrast {michelson:.3f} exceeds the safe limit "
            f"{cfg.pattern_contrast_limit:.2f}.  At video frame rates this "
            "pattern may trigger the WCAG 2.3.1 / ITU-R BT.1702 flash threshold."
        )
        advice.append(
            "Enable safe_mode=True to automatically reduce pattern contrast to 50 %.  "
            "Or choose a lower-contrast pattern (e.g. 'soft_bokeh', 'canvas')."
        )

    # --- Luminance range check ---
    if lum_delta > cfg.luminance_change_threshold:
        violations.append(
            f"Luminance range {lum_delta:.3f} exceeds the single-flash threshold "
            f"{cfg.luminance_change_threshold:.2f}.  If animated or scrolled at "
            "≥ 3 Hz this content will fail WCAG 2.3.1."
        )

    # --- Red flash check ---
    if max_red_delta > cfg.red_chrominance_threshold:
        violations.append(
            f"Red chrominance variation {max_red_delta:.3f} exceeds the Harding "
            f"PSE Test red-flash threshold {cfg.red_chrominance_threshold:.2f}.  "
            "Red-saturated patterns carry elevated seizure risk."
        )
        advice.append("Use GREYSCALE or a desaturated colour mode to eliminate red flash risk.")

    # --- Risk classification ---
    if not violations:
        risk = EpilepticRisk.SAFE
    elif michelson > 0.85 or (
        michelson > cfg.pattern_contrast_limit and max_red_delta > cfg.red_chrominance_threshold
    ):
        risk = EpilepticRisk.CRITICAL
    elif michelson > 0.70:
        risk = EpilepticRisk.HIGH
    elif michelson > cfg.pattern_contrast_limit:
        risk = EpilepticRisk.MODERATE
    else:
        risk = EpilepticRisk.LOW

    standard_results = {
        "WCAG_2.3.1": lum_delta <= cfg.luminance_change_threshold,
        "ITU_BT1702": michelson <= cfg.pattern_contrast_limit,
        "Ofcom": michelson <= 0.40,
        "Harding_PSE": max_red_delta <= cfg.red_chrominance_threshold,
    }

    if violations and not advice:
        advice.append("Use safe_mode=True or the 'broadcast' preset for public distribution.")

    return FlashSafetyReport(
        risk=risk,
        passed=len(violations) == 0,
        flash_frequency_hz=0.0,
        max_luminance_delta=lum_delta,
        max_red_delta=max_red_delta,
        michelson_contrast=michelson,
        violations=violations,
        advice=advice,
        standard_results=standard_results,
    )


# ---------------------------------------------------------------------------
# Video / sequence flash analysis
# ---------------------------------------------------------------------------


def check_frame_sequence(
    frames: Sequence[np.ndarray],
    fps: float,
    config: Optional[FlashSafetyConfig] = None,
) -> FlashSafetyReport:
    """Analyse a video frame sequence for flash / flicker events.

    Implements the ITU-R BT.1702 / WCAG 2.3.1 flash-counting algorithm:

    1. Compute mean relative luminance per frame.
    2. Find transitions where |ΔL| ≥ ``luminance_change_threshold``.
    3. Count paired opposing transitions (up + down) as flash events.
    4. Determine the worst-case flash count within any 1-second window.
    5. Apply the same logic to red chrominance for red-flash detection.

    Parameters
    ----------
    frames : sequence of np.ndarray
        RGBA uint8 (H, W, 4) stereogram frames in temporal order.
    fps : float
        Playback frame rate in frames per second.
    config : FlashSafetyConfig, optional
        Override thresholds.  Standard limits used if omitted.

    Returns
    -------
    FlashSafetyReport
    """
    cfg = config or FlashSafetyConfig()

    _all_pass = {k: True for k in ("WCAG_2.3.1", "ITU_BT1702", "Ofcom", "Harding_PSE")}
    _safe = FlashSafetyReport(
        risk=EpilepticRisk.SAFE,
        passed=True,
        flash_frequency_hz=0.0,
        max_luminance_delta=0.0,
        max_red_delta=0.0,
        michelson_contrast=0.0,
        standard_results=_all_pass,
    )

    if len(frames) < 2:
        return _safe

    # Mean luminance / red-chrominance time-series (one value per frame)
    lum = np.array([float(_relative_luminance(f).mean()) for f in frames], dtype=np.float32)
    red = np.array([float(_red_chrominance(f).mean()) for f in frames], dtype=np.float32)

    n = len(lum)
    lum_diff = np.diff(lum)
    red_diff = np.diff(red)

    max_lum_delta = float(np.abs(lum_diff).max())
    max_red_delta = float(np.abs(red_diff).max())

    # Flash pair counting in a 1-second sliding window
    win = max(1, int(fps))
    lum_evt = np.where(np.abs(lum_diff) >= cfg.luminance_change_threshold)[0]
    red_evt = np.where(np.abs(red_diff) >= cfg.red_chrominance_threshold)[0]

    def _max_flash_pairs(diff: np.ndarray, events: np.ndarray) -> int:
        best = 0
        for i in range(max(1, n - win)):
            ev = events[(events >= i) & (events < i + win)]
            if len(ev) < 2:
                continue
            pos = int(np.sum(diff[ev] > 0))
            neg = int(np.sum(diff[ev] < 0))
            best = max(best, min(pos, neg))
        return best

    flash_hz = float(_max_flash_pairs(lum_diff, lum_evt))
    red_hz = float(_max_flash_pairs(red_diff, red_evt))

    L_max = float(lum.max())
    L_min = float(lum.min())
    michelson = float((L_max - L_min) / (L_max + L_min + 1e-8))

    violations: List[str] = []
    advice: List[str] = []

    if flash_hz > cfg.max_flashes_per_second:
        violations.append(
            f"Flash rate {flash_hz:.0f} Hz exceeds the WCAG 2.3.1 / "
            f"ITU-R BT.1702 limit of {cfg.max_flashes_per_second:.0f} Hz.  "
            "This content will likely trigger photosensitive seizures."
        )
        advice.append(
            "Apply TemporalSmoother to depth maps before synthesis to dampen "
            "frame-to-frame luminance swings.  Or reduce the output frame rate "
            "below 3 Hz for strobing content."
        )

    if red_hz > cfg.max_flashes_per_second:
        violations.append(
            f"Red flash rate {red_hz:.0f} Hz exceeds the Harding PSE Test "
            f"limit of {cfg.max_flashes_per_second:.0f} Hz."
        )
        advice.append(
            "Switch to GREYSCALE or a reduced-saturation colour mode to "
            "eliminate red flash risk."
        )

    if max_lum_delta > cfg.luminance_change_threshold * 3:
        violations.append(
            f"Peak frame-to-frame luminance change {max_lum_delta:.3f} is "
            f"{max_lum_delta / cfg.luminance_change_threshold:.1f}× the flash "
            "threshold — severe contrast transitions detected."
        )

    if not violations:
        risk = EpilepticRisk.SAFE if flash_hz == 0.0 else EpilepticRisk.LOW
    elif flash_hz > cfg.max_flashes_per_second * 2 or red_hz > cfg.max_flashes_per_second:
        risk = EpilepticRisk.CRITICAL
    elif flash_hz > cfg.max_flashes_per_second:
        risk = EpilepticRisk.HIGH
    else:
        risk = EpilepticRisk.MODERATE

    standard_results = {
        "WCAG_2.3.1": flash_hz <= cfg.max_flashes_per_second,
        "ITU_BT1702": flash_hz <= cfg.max_flashes_per_second,
        "Ofcom": flash_hz <= 3.0 and max_lum_delta <= 0.20,
        "Harding_PSE": red_hz <= cfg.max_flashes_per_second,
    }

    if violations and not advice:
        advice.append("Use safe_mode=True or the 'broadcast' preset for public distribution.")

    return FlashSafetyReport(
        risk=risk,
        passed=len(violations) == 0,
        flash_frequency_hz=flash_hz,
        max_luminance_delta=max_lum_delta,
        max_red_delta=max_red_delta,
        michelson_contrast=michelson,
        violations=violations,
        advice=advice,
        standard_results=standard_results,
    )


# ---------------------------------------------------------------------------
# Convenience: issue Python warning if risk is elevated
# ---------------------------------------------------------------------------


def warn_if_unsafe(
    report: FlashSafetyReport,
    context: str = "pattern",
    stacklevel: int = 2,
) -> None:
    """Issue a Python UserWarning when the flash safety report flags a risk.

    SAFE and LOW results are silently ignored.

    Parameters
    ----------
    report : FlashSafetyReport
        Report from ``check_pattern_flash_risk()`` or ``check_frame_sequence()``.
    context : str
        Human-readable label included in the warning message.
    stacklevel : int
        Passed to ``warnings.warn``.
    """
    if report.risk in (EpilepticRisk.SAFE, EpilepticRisk.LOW):
        return

    severity = "CRITICAL" if report.risk == EpilepticRisk.CRITICAL else "WARNING"
    body = "\n".join(f"  · {v}" for v in report.violations)
    msg = (
        f"DepthForge PSE {severity} [{context}]: {report.risk.value} flash risk.\n"
        f"{body}\n"
        f"  Enable safe_mode=True to reduce risk.\n"
        f"  {EPILEPSY_WARNING_SHORT}"
    )
    warnings.warn(msg, UserWarning, stacklevel=stacklevel)
