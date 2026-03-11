"""
depthforge.comfyui.swirls_nodes
================================
10 optical illusion generator nodes — DepthForge/Swirls category.

All algorithms are implemented independently in pure NumPy + Pillow from
mathematical descriptions published in the academic literature cited below.
No copyrighted images or proprietary implementations were referenced.

Academic basis
--------------
Fraser, J. (1908). A new visual illusion of direction.
    British Journal of Psychology, 2, 307–320.

Addams, R. (1834). An account of a peculiar optical phenomenon.
    London and Edinburgh Philosophical Magazine, 5, 373–374.

Kitaoka, A., Pinna, B., & Brelstaff, G. (2001). New variations of
    spiral illusions. Perception, 30, 637–646.

Kitaoka, A., & Ashida, H. (2003). Phenomenal characteristics of the
    peripheral drift illusion. VISION, 15, 261–262.

Kitaoka, A. (2007). Tilt illusions after Oyama (1960): A review.
    Japanese Psychological Research, 49, 7–19.

Simanek, D. (1996). Ambiguous figures and visual paradoxes.
    Lock Haven University physics demonstrations.
"""

from __future__ import annotations

import colorsys
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ── PSE / epilepsy warning ────────────────────────────────────────────────────

_EPILEPSY_WARNING = (
    "PSE WARNING: This node produces strong apparent motion. "
    "Do NOT display to viewers with photosensitive epilepsy or visual migraine sensitivity. "
    "safe_mode=True reduces contrast to soften the effect."
)

# ── Shared helpers ────────────────────────────────────────────────────────────


def _hex_to_rgb(s: str) -> tuple[int, int, int]:
    """Parse '#RRGGBB' or 'RRGGBB' to (R, G, B). Falls back to mid-grey."""
    h = s.strip().lstrip("#")
    if len(h) == 3:
        h = h[0] * 2 + h[1] * 2 + h[2] * 2
    try:
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    except Exception:
        return (128, 128, 128)


def _to_tensor(img: Image.Image):
    """Convert a PIL RGBA image to a ComfyUI IMAGE tensor (1, H, W, 4)."""
    arr = np.array(img.convert("RGBA"), dtype=np.float32) / 255.0
    try:
        import torch

        return torch.from_numpy(arr).unsqueeze(0)
    except ImportError:
        return arr[None]


def _np_to_tensor(arr: np.ndarray):
    """Convert (H, W, 4) uint8 numpy array to ComfyUI IMAGE tensor."""
    f = arr.astype(np.float32) / 255.0
    try:
        import torch

        return torch.from_numpy(f).unsqueeze(0)
    except ImportError:
        return f[None]


def _arc_poly(
    cx: float,
    cy: float,
    r_in: float,
    r_out: float,
    a0: float,
    a1: float,
    n: int = 32,
) -> list[tuple[float, float]]:
    """Return polygon vertices for an annular arc sector.

    Outer arc sweeps a0 → a1, inner arc sweeps a1 → a0 (reversed),
    forming a closed ring segment suitable for PIL ``draw.polygon()``.
    """
    af = np.linspace(a0, a1, n)
    ar = np.linspace(a1, a0, n)
    return [(cx + r_out * math.cos(a), cy + r_out * math.sin(a)) for a in af] + [
        (cx + r_in * math.cos(a), cy + r_in * math.sin(a)) for a in ar
    ]


def _tensor_to_np(tensor) -> np.ndarray:
    """ComfyUI IMAGE tensor → (H, W, 4) uint8 numpy."""
    try:
        import torch

        if isinstance(tensor, torch.Tensor):
            arr = tensor.squeeze(0).cpu().numpy()
        else:
            arr = np.array(tensor)
    except ImportError:
        arr = np.array(tensor)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.max() <= 1.0:
        arr = (arr * 255).astype(np.uint8)
    if arr.shape[-1] == 3:
        alpha = np.full((*arr.shape[:2], 1), 255, dtype=np.uint8)
        arr = np.concatenate([arr, alpha], axis=-1)
    return arr.astype(np.uint8)


class _SwirlBase:
    """Shared utilities for all Swirls nodes."""

    CATEGORY = "DepthForge/Swirls"
    FUNCTION = "execute"


# ═══════════════════════════════════════════════════════════════════════════════
# Node 1 — DF_Swirl_FraserSpiral
# ═══════════════════════════════════════════════════════════════════════════════


class DF_Swirl_FraserSpiral(_SwirlBase):
    """Fraser (1908) twisted-cord spiral illusion."""

    DESCRIPTION = (
        "Fraser spiral illusion — twisted cord elements placed on true concentric circles "
        "create the overwhelming perception of a spiral. Based on Fraser (1908)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": (
                    "INT",
                    {
                        "default": 768,
                        "min": 128,
                        "max": 4096,
                        "tooltip": "Output image width in pixels.",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 768,
                        "min": 128,
                        "max": 4096,
                        "tooltip": "Output image height in pixels.",
                    },
                ),
            },
            "optional": {
                "num_rings": (
                    "INT",
                    {
                        "default": 9,
                        "min": 2,
                        "max": 24,
                        "tooltip": "Number of concentric circles. 8–12 gives the densest illusion. More rings fill the frame further.",
                    },
                ),
                "ring_spacing": (
                    "INT",
                    {
                        "default": 38,
                        "min": 8,
                        "max": 200,
                        "tooltip": "Radial gap between rings in pixels. Should be close to cord_length. Safe range: 20–60 px.",
                    },
                ),
                "cord_angle": (
                    "FLOAT",
                    {
                        "default": 20.0,
                        "min": 5.0,
                        "max": 60.0,
                        "step": 1.0,
                        "tooltip": "Tilt of each cord element from the local tangent, in degrees. 20° is the classic Fraser equiangle. Higher = stronger illusion but less realistic spiral.",
                    },
                ),
                "cord_length": (
                    "INT",
                    {
                        "default": 22,
                        "min": 6,
                        "max": 100,
                        "tooltip": "Length of each cord segment in pixels. Best close to ring_spacing.",
                    },
                ),
                "cord_width": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "max": 14,
                        "tooltip": "Stroke width of each cord in pixels. 2–4 px is optimal for readability.",
                    },
                ),
                "bg_color": (
                    "STRING",
                    {
                        "default": "#FFFFFF",
                        "tooltip": "Background colour as #RRGGBB hex. White is the classical setting.",
                    },
                ),
                "cord_color_a": (
                    "STRING",
                    {
                        "default": "#000000",
                        "tooltip": "First alternating cord colour. Black is classical. The contrast between A and B drives the illusion.",
                    },
                ),
                "cord_color_b": (
                    "STRING",
                    {
                        "default": "#808080",
                        "tooltip": "Second alternating cord colour. Grey on white background is classical. More contrast = stronger spiral misperception.",
                    },
                ),
                "supersampling": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 4,
                        "tooltip": "Anti-aliasing oversampling factor. 2 = 2× resolution then downsample (recommended). 4 = highest quality, 4× slower.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("illusion",)
    OUTPUT_NODE = True

    def execute(
        self,
        width,
        height,
        num_rings=9,
        ring_spacing=38,
        cord_angle=20.0,
        cord_length=22,
        cord_width=3,
        bg_color="#FFFFFF",
        cord_color_a="#000000",
        cord_color_b="#808080",
        supersampling=2,
    ):
        ss = supersampling
        W, H = width * ss, height * ss
        bg = (*_hex_to_rgb(bg_color), 255)
        ca = (*_hex_to_rgb(cord_color_a), 255)
        cb = (*_hex_to_rgb(cord_color_b), 255)

        img = Image.new("RGBA", (W, H), bg)
        draw = ImageDraw.Draw(img)

        cx, cy = W / 2.0, H / 2.0
        ang = math.radians(cord_angle)
        rs = ring_spacing * ss
        cl = cord_length * ss
        cw = max(1, cord_width * ss)

        for ri in range(num_rings):
            r = (ri + 1) * rs
            if r >= min(W, H) / 2.0:
                break
            n_cords = max(8, int(2 * math.pi * r / (cl * 0.72)))
            for i in range(n_cords):
                theta = 2 * math.pi * i / n_cords
                px = cx + r * math.cos(theta)
                py = cy + r * math.sin(theta)
                # local tangent ± alternating cord_angle per ring
                tilt = theta + math.pi / 2 + (ang if ri % 2 == 0 else -ang)
                half = cl / 2.0
                dtx = half * math.cos(tilt)
                dty = half * math.sin(tilt)
                color = ca if i % 2 == 0 else cb
                draw.line(
                    [(px - dtx, py - dty), (px + dtx, py + dty)],
                    fill=color,
                    width=cw,
                )

        return (_to_tensor(img.resize((width, height), Image.LANCZOS)),)


# ═══════════════════════════════════════════════════════════════════════════════
# Node 2 — DF_Swirl_ZollnerSpiral
# ═══════════════════════════════════════════════════════════════════════════════


class DF_Swirl_ZollnerSpiral(_SwirlBase):
    """Zöllner-based spiral — parallel rings appear tilted by crossing hatches."""

    DESCRIPTION = (
        "Zöllner spiral illusion — concentric ring outlines appear to be a spiral "
        "because crossing hatch marks on each ring create false angular continuation. "
        "Based on Kitaoka, Pinna & Brelstaff (2001)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": (
                    "INT",
                    {"default": 768, "min": 128, "max": 4096, "tooltip": "Output width in pixels."},
                ),
                "height": (
                    "INT",
                    {"default": 768, "min": 128, "max": 4096, "tooltip": "Output height in pixels."},
                ),
            },
            "optional": {
                "num_rings": (
                    "INT",
                    {
                        "default": 10,
                        "min": 2,
                        "max": 24,
                        "tooltip": "Number of concentric rings. 8–14 is optimal for the Zöllner effect.",
                    },
                ),
                "ring_thickness": (
                    "INT",
                    {
                        "default": 6,
                        "min": 2,
                        "max": 40,
                        "tooltip": "Stroke thickness of each ring in pixels. Thicker rings allow wider hatch marks.",
                    },
                ),
                "hatch_angle": (
                    "FLOAT",
                    {
                        "default": 45.0,
                        "min": 10.0,
                        "max": 80.0,
                        "step": 1.0,
                        "tooltip": "Angle of hatch marks in degrees from horizontal. 45° is classical. The contrast between adjacent ring hatch directions creates the tilt illusion.",
                    },
                ),
                "hatch_spacing": (
                    "INT",
                    {
                        "default": 16,
                        "min": 4,
                        "max": 80,
                        "tooltip": "Circumferential spacing between hatch marks in pixels. Denser hatches = stronger illusion.",
                    },
                ),
                "hatch_length": (
                    "INT",
                    {
                        "default": 14,
                        "min": 4,
                        "max": 60,
                        "tooltip": "Length of each hatch mark in pixels. Should cross the ring width by ~50%.",
                    },
                ),
                "ring_color": (
                    "STRING",
                    {
                        "default": "#000000",
                        "tooltip": "Ring outline colour as #RRGGBB hex.",
                    },
                ),
                "hatch_color": (
                    "STRING",
                    {
                        "default": "#000000",
                        "tooltip": "Hatch mark colour. Same as ring_color for a classic look.",
                    },
                ),
                "bg_color": (
                    "STRING",
                    {"default": "#FFFFFF", "tooltip": "Background colour as #RRGGBB hex."},
                ),
                "supersampling": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 4,
                        "tooltip": "Anti-aliasing oversampling factor. 2 = recommended.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("illusion",)
    OUTPUT_NODE = True

    def execute(
        self,
        width,
        height,
        num_rings=10,
        ring_thickness=6,
        hatch_angle=45.0,
        hatch_spacing=16,
        hatch_length=14,
        ring_color="#000000",
        hatch_color="#000000",
        bg_color="#FFFFFF",
        supersampling=2,
    ):
        ss = supersampling
        W, H = width * ss, height * ss
        bg = (*_hex_to_rgb(bg_color), 255)
        rc = (*_hex_to_rgb(ring_color), 255)
        hc = (*_hex_to_rgb(hatch_color), 255)

        img = Image.new("RGBA", (W, H), bg)
        draw = ImageDraw.Draw(img)

        cx, cy = W / 2.0, H / 2.0
        rt = ring_thickness * ss
        hs = hatch_spacing * ss
        hl = hatch_length * ss
        hw = max(1, ss)
        max_r = min(W, H) / 2.0
        gap = max_r / (num_rings + 1)

        for ri in range(num_rings):
            r = (ri + 1) * gap
            r_out = r + rt / 2.0
            r_in = max(1.0, r - rt / 2.0)

            # Draw ring as filled donut strip: large circle then punch inner hole
            bbox_out = [cx - r_out, cy - r_out, cx + r_out, cy + r_out]
            bbox_in = [cx - r_in, cy - r_in, cx + r_in, cy + r_in]
            draw.ellipse(bbox_out, fill=rc)
            draw.ellipse(bbox_in, fill=bg)

            # Alternating hatch angle per ring
            ha = math.radians(hatch_angle if ri % 2 == 0 else -hatch_angle)
            n_hatch = max(4, int(2 * math.pi * r / hs))
            half_hl = hl / 2.0
            cos_ha, sin_ha = math.cos(ha), math.sin(ha)

            for j in range(n_hatch):
                theta = 2 * math.pi * j / n_hatch
                hx = cx + r * math.cos(theta)
                hy = cy + r * math.sin(theta)
                draw.line(
                    [(hx - half_hl * cos_ha, hy - half_hl * sin_ha),
                     (hx + half_hl * cos_ha, hy + half_hl * sin_ha)],
                    fill=hc,
                    width=hw,
                )

        return (_to_tensor(img.resize((width, height), Image.LANCZOS)),)


# ═══════════════════════════════════════════════════════════════════════════════
# Node 3 — DF_Swirl_PeripheralDrift
# ═══════════════════════════════════════════════════════════════════════════════


class DF_Swirl_PeripheralDrift(_SwirlBase):
    """Peripheral drift / rotating snakes illusion. PSE WARNING — see DESCRIPTION."""

    DESCRIPTION = (
        "Peripheral drift illusion (rotating snakes) — a 4-step asymmetric luminance "
        "gradient around each ring creates vivid apparent rotation in peripheral vision. "
        "Alternating rings rotate in opposite directions. "
        "Based on Kitaoka & Ashida (2003). "
        + _EPILEPSY_WARNING
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": (
                    "INT",
                    {"default": 768, "min": 128, "max": 4096, "tooltip": "Output width in pixels."},
                ),
                "height": (
                    "INT",
                    {"default": 768, "min": 128, "max": 4096, "tooltip": "Output height in pixels."},
                ),
            },
            "optional": {
                "num_rings": (
                    "INT",
                    {
                        "default": 8,
                        "min": 2,
                        "max": 20,
                        "tooltip": "Number of concentric rings. Each alternating ring rotates in the opposite direction. 6–10 is recommended.",
                    },
                ),
                "segments_per_ring": (
                    "INT",
                    {
                        "default": 32,
                        "min": 8,
                        "max": 120,
                        "tooltip": "Number of arc segments per ring. Each segment holds one step of the luminance gradient. 24–48 creates smooth motion illusion.",
                    },
                ),
                "ring_width": (
                    "INT",
                    {
                        "default": 40,
                        "min": 10,
                        "max": 200,
                        "tooltip": "Radial width of each ring in pixels. Safe range: 20–60 px.",
                    },
                ),
                "rotation_direction": (
                    ["ALTERNATING", "CW", "CCW"],
                    {
                        "tooltip": (
                            "'ALTERNATING' = adjacent rings rotate opposite directions — strongest illusion. "
                            "'CW' = all rings appear to rotate clockwise. "
                            "'CCW' = all rings counter-clockwise."
                        )
                    },
                ),
                "color_mode": (
                    ["greyscale", "rainbow"],
                    {
                        "tooltip": (
                            "'greyscale' = classic black/grey/white gradient (strongest motion). "
                            "'rainbow' = hue shifted per ring (decorative, slightly weaker motion)."
                        )
                    },
                ),
                "luminance_steps": (
                    "INT",
                    {
                        "default": 4,
                        "min": 4,
                        "max": 8,
                        "tooltip": (
                            "Number of luminance levels in the gradient sequence. "
                            "4 = classic Kitaoka sequence: black, dark-grey, white, light-grey. "
                            "More steps = smoother gradient but weaker motion percept. "
                            "Kitaoka's original uses exactly 4."
                        ),
                    },
                ),
                "safe_mode": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Reduce luminance range by 50% to soften the apparent motion. "
                            "STRONGLY recommended for any public display. "
                            "safe_mode=False uses the full black-to-white range."
                        ),
                    },
                ),
                "supersampling": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 4,
                        "tooltip": "Anti-aliasing factor. 2 = recommended.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("illusion",)
    OUTPUT_NODE = True

    def execute(
        self,
        width,
        height,
        num_rings=8,
        segments_per_ring=32,
        ring_width=40,
        rotation_direction="ALTERNATING",
        color_mode="greyscale",
        luminance_steps=4,
        safe_mode=True,
        supersampling=2,
    ):
        ss = supersampling
        W, H = width * ss, height * ss

        # Kitaoka's critical 4-step asymmetric luminance sequence
        # black → dark-grey → white → light-grey → (repeat)
        base_seq = [0, 64, 255, 191]
        if luminance_steps == 4:
            lumin = base_seq
        else:
            step = 255.0 / (luminance_steps - 1)
            lumin = [int(i * step) for i in range(luminance_steps)]
            # Ensure the asymmetric waveform: reverse upper half
            mid = luminance_steps // 2
            lumin = lumin[:mid] + lumin[mid:][::-1]

        if safe_mode:
            # Compress range to 64–191 to reduce motion intensity
            lumin = [int(64 + l * 0.5) for l in lumin]

        # Mid-grey background
        img = Image.new("RGBA", (W, H), (128, 128, 128, 255))
        draw = ImageDraw.Draw(img)

        cx, cy = W / 2.0, H / 2.0
        rw = ring_width * ss

        for ri in range(num_rings):
            r_in = ri * rw
            r_out = r_in + rw * 0.9  # small gap between rings
            if r_out > min(W, H) / 2.0:
                break

            if rotation_direction == "ALTERNATING":
                direction = 1 if ri % 2 == 0 else -1
            elif rotation_direction == "CW":
                direction = 1
            else:
                direction = -1

            n_segs = segments_per_ring
            n_pts = max(8, int(40 * r_out / max(rw, 1)))

            for si in range(n_segs):
                a0 = 2 * math.pi * si / n_segs
                a1 = 2 * math.pi * (si + 1) / n_segs
                lum_idx = (si * direction) % len(lumin)
                lum = lumin[lum_idx]

                if color_mode == "greyscale":
                    color = (lum, lum, lum, 255)
                else:
                    hue = (ri / max(num_rings, 1) + si / n_segs * 0.3) % 1.0
                    r_c, g_c, b_c = colorsys.hsv_to_rgb(hue, 0.7, lum / 255.0)
                    color = (int(r_c * 255), int(g_c * 255), int(b_c * 255), 255)

                pts = _arc_poly(cx, cy, r_in, r_out, a0, a1, n=n_pts)
                draw.polygon(pts, fill=color)

        return (_to_tensor(img.resize((width, height), Image.LANCZOS)),)


# ═══════════════════════════════════════════════════════════════════════════════
# Node 4 — DF_Swirl_NeuralCircuit
# ═══════════════════════════════════════════════════════════════════════════════


class DF_Swirl_NeuralCircuit(_SwirlBase):
    """Kitaoka (2007) neural circuit — radial stripes appear to spiral inward."""

    DESCRIPTION = (
        "Neural circuit spiral illusion — radial stripes whose angle is a function of "
        "both angle and radius (theta + r × tightness) create the strong impression "
        "of inward-rotating concentric spirals. Based on Kitaoka (2007)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": (
                    "INT",
                    {"default": 768, "min": 128, "max": 4096, "tooltip": "Output width in pixels."},
                ),
                "height": (
                    "INT",
                    {"default": 768, "min": 128, "max": 4096, "tooltip": "Output height in pixels."},
                ),
            },
            "optional": {
                "num_stripes": (
                    "INT",
                    {
                        "default": 8,
                        "min": 2,
                        "max": 32,
                        "tooltip": "Number of radial stripe arms. More stripes = finer pattern. Safe range: 4–16.",
                    },
                ),
                "stripe_width": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.2,
                        "max": 4.0,
                        "step": 0.1,
                        "tooltip": "Width of each stripe as a multiplier (1.0 = default). Lower = thinner, sharper stripes. Higher = wider bands.",
                    },
                ),
                "spiral_tightness": (
                    "FLOAT",
                    {
                        "default": 0.04,
                        "min": 0.005,
                        "max": 0.2,
                        "step": 0.005,
                        "tooltip": "How strongly the stripe angle is offset by radius. 0.04 creates a gentle spiral; 0.12 = tight corkscrew. Safe range: 0.02–0.08.",
                    },
                ),
                "rotation_direction": (
                    ["inward", "outward"],
                    {
                        "tooltip": "'inward' = stripes appear to spiral toward centre. 'outward' = appear to expand. Controls the sign of the tightness offset.",
                    },
                ),
                "color_a": (
                    "STRING",
                    {
                        "default": "#000000",
                        "tooltip": "First stripe colour as #RRGGBB hex. Black on white is classical.",
                    },
                ),
                "color_b": (
                    "STRING",
                    {
                        "default": "#FFFFFF",
                        "tooltip": "Second stripe colour as #RRGGBB hex.",
                    },
                ),
                "center_x": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.1,
                        "max": 0.9,
                        "step": 0.05,
                        "tooltip": "Horizontal centre of the spiral as a fraction of width. 0.5 = centred.",
                    },
                ),
                "center_y": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.1,
                        "max": 0.9,
                        "step": 0.05,
                        "tooltip": "Vertical centre of the spiral as a fraction of height. 0.5 = centred.",
                    },
                ),
                "edge_feather": (
                    "FLOAT",
                    {
                        "default": 0.15,
                        "min": 0.0,
                        "max": 0.5,
                        "step": 0.05,
                        "tooltip": "Fade-out radius at the image edge as a fraction of the half-width. 0.15 creates a soft vignette.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("illusion",)
    OUTPUT_NODE = True

    def execute(
        self,
        width,
        height,
        num_stripes=8,
        stripe_width=1.0,
        spiral_tightness=0.04,
        rotation_direction="inward",
        color_a="#000000",
        color_b="#FFFFFF",
        center_x=0.5,
        center_y=0.5,
        edge_feather=0.15,
    ):
        cx = width * center_x
        cy = height * center_y

        yi, xi = np.mgrid[0:height, 0:width].astype(np.float32)
        dx = xi - cx
        dy = yi - cy
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)  # [-pi, pi]

        sign = 1.0 if rotation_direction == "inward" else -1.0
        # Stripe phase: angle shifts with radius to create false spiral
        phase = num_stripes * theta + sign * r * spiral_tightness * num_stripes
        # Stripe pattern via sine (smooth edges, good aliasing)
        pattern = (np.sin(phase / stripe_width) + 1.0) / 2.0  # [0, 1]

        ca = np.array(_hex_to_rgb(color_a), dtype=np.float32) / 255.0
        cb = np.array(_hex_to_rgb(color_b), dtype=np.float32) / 255.0
        rgb = ca[None, None, :] * pattern[:, :, None] + cb[None, None, :] * (1.0 - pattern[:, :, None])

        # Circular vignette
        max_r = min(cx, cy, width - cx, height - cy)
        fade_start = max_r * (1.0 - edge_feather)
        vignette = np.clip((max_r - r) / max(max_r - fade_start, 1.0), 0.0, 1.0)
        bg = (ca + cb) / 2.0
        rgb = rgb * vignette[:, :, None] + bg[None, None, :] * (1.0 - vignette[:, :, None])

        rgba = np.concatenate(
            [np.clip(rgb * 255, 0, 255).astype(np.uint8),
             np.full((height, width, 1), 255, dtype=np.uint8)],
            axis=-1,
        )
        return (_np_to_tensor(rgba),)


# ═══════════════════════════════════════════════════════════════════════════════
# Node 5 — DF_Swirl_MagneticStorm
# ═══════════════════════════════════════════════════════════════════════════════


class DF_Swirl_MagneticStorm(_SwirlBase):
    """Fraser spiral variant: coloured rings + twisted cord at equiangle 20°."""

    DESCRIPTION = (
        "Magnetic Storm — coloured concentric rings with Fraser twisted-cord elements "
        "appear to spiral inward with a vivid colour shimmer. "
        "A Fraser (1908) variant with filled coloured rings."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": (
                    "INT",
                    {"default": 768, "min": 128, "max": 4096, "tooltip": "Output width in pixels."},
                ),
                "height": (
                    "INT",
                    {"default": 768, "min": 128, "max": 4096, "tooltip": "Output height in pixels."},
                ),
            },
            "optional": {
                "num_rings": (
                    "INT",
                    {
                        "default": 10,
                        "min": 2,
                        "max": 24,
                        "tooltip": "Number of coloured concentric rings. 8–14 is optimal.",
                    },
                ),
                "equiangle": (
                    "FLOAT",
                    {
                        "default": 20.0,
                        "min": 5.0,
                        "max": 55.0,
                        "step": 1.0,
                        "tooltip": "Cord tilt from tangent in degrees. 20° is the canonical Fraser equiangle. Higher = stronger spiral illusion.",
                    },
                ),
                "ring_color": (
                    "STRING",
                    {
                        "default": "#CC2200",
                        "tooltip": "Primary ring fill colour as #RRGGBB. Deep red is the classic 'Magnetic Storm' palette.",
                    },
                ),
                "bg_color": (
                    "STRING",
                    {
                        "default": "#FFFFFF",
                        "tooltip": "Background colour between rings.",
                    },
                ),
                "ring_width": (
                    "INT",
                    {
                        "default": 18,
                        "min": 4,
                        "max": 80,
                        "tooltip": "Radial thickness of each coloured ring in pixels.",
                    },
                ),
                "cord_density": (
                    "FLOAT",
                    {
                        "default": 0.8,
                        "min": 0.3,
                        "max": 2.0,
                        "step": 0.1,
                        "tooltip": "Density of cord elements as a multiplier on circumference-based count. 1.0 = one cord per cord_length; 0.5 = sparser; 2.0 = overlapping.",
                    },
                ),
                "cord_color": (
                    "STRING",
                    {
                        "default": "#000000",
                        "tooltip": "Twisted cord colour drawn on top of the rings. Black gives the clearest illusory spiral.",
                    },
                ),
                "supersampling": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 4,
                        "tooltip": "Anti-aliasing factor. 2 = recommended.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("illusion",)
    OUTPUT_NODE = True

    def execute(
        self,
        width,
        height,
        num_rings=10,
        equiangle=20.0,
        ring_color="#CC2200",
        bg_color="#FFFFFF",
        ring_width=18,
        cord_density=0.8,
        cord_color="#000000",
        supersampling=2,
    ):
        ss = supersampling
        W, H = width * ss, height * ss
        bg = (*_hex_to_rgb(bg_color), 255)
        rc = (*_hex_to_rgb(ring_color), 255)
        cc = (*_hex_to_rgb(cord_color), 255)

        img = Image.new("RGBA", (W, H), bg)
        draw = ImageDraw.Draw(img)

        cx, cy = W / 2.0, H / 2.0
        ang = math.radians(equiangle)
        rw = ring_width * ss
        max_r = min(W, H) / 2.0
        gap = max_r / (num_rings + 0.5)
        cord_len = rw * 1.4

        for ri in range(num_rings):
            r = (ri + 0.5) * gap
            r_out = r + rw / 2.0
            r_in = max(1.0, r - rw / 2.0)
            if r_out > max_r:
                break

            # Filled ring
            bbox_out = [cx - r_out, cy - r_out, cx + r_out, cy + r_out]
            bbox_in = [cx - r_in, cy - r_in, cx + r_in, cy + r_in]
            draw.ellipse(bbox_out, fill=rc)
            draw.ellipse(bbox_in, fill=bg)

            # Twisted cord overlay
            n_cords = max(6, int(2 * math.pi * r / cord_len * cord_density))
            half = cord_len / 2.0
            cw = max(1, ss)
            for i in range(n_cords):
                theta = 2 * math.pi * i / n_cords
                px = cx + r * math.cos(theta)
                py = cy + r * math.sin(theta)
                tilt = theta + math.pi / 2 + (ang if ri % 2 == 0 else -ang)
                dtx = half * math.cos(tilt)
                dty = half * math.sin(tilt)
                draw.line(
                    [(px - dtx, py - dty), (px + dtx, py + dty)],
                    fill=cc,
                    width=cw,
                )

        return (_to_tensor(img.resize((width, height), Image.LANCZOS)),)


# ═══════════════════════════════════════════════════════════════════════════════
# Node 6 — DF_Swirl_Tornado
# ═══════════════════════════════════════════════════════════════════════════════


class DF_Swirl_Tornado(_SwirlBase):
    """Multi-colour rings + Zöllner hatches with funnel taper — rotational funnel."""

    DESCRIPTION = (
        "Tornado illusion — multi-colour concentric rings with Zöllner-type hatch marks "
        "and a logarithmic funnel taper create a vivid sense of rotating downward "
        "motion toward the centre."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": (
                    "INT",
                    {"default": 768, "min": 128, "max": 4096, "tooltip": "Output width in pixels."},
                ),
                "height": (
                    "INT",
                    {"default": 768, "min": 128, "max": 4096, "tooltip": "Output height in pixels."},
                ),
            },
            "optional": {
                "num_rings": (
                    "INT",
                    {
                        "default": 12,
                        "min": 3,
                        "max": 30,
                        "tooltip": "Number of rings. More rings increase the funnel depth illusion.",
                    },
                ),
                "color_cycle": (
                    "STRING",
                    {
                        "default": "#FF2200,#FF8800,#FFDD00,#22CC00,#0066FF,#8800CC",
                        "tooltip": "Comma-separated #RRGGBB hex colours cycled through the rings. Default = warm spectrum. Add/remove colours to taste.",
                    },
                ),
                "hatch_angle": (
                    "FLOAT",
                    {
                        "default": 50.0,
                        "min": 10.0,
                        "max": 80.0,
                        "step": 1.0,
                        "tooltip": "Hatch mark angle in degrees. Alternates sign per ring. 45–55° gives the strongest Zöllner tilt.",
                    },
                ),
                "funnel_taper": (
                    "FLOAT",
                    {
                        "default": 1.6,
                        "min": 1.0,
                        "max": 3.0,
                        "step": 0.1,
                        "tooltip": "Logarithmic taper exponent controlling ring compression toward the centre. 1.0 = uniform spacing; 2.0 = strong funnel (inner rings much thinner). Safe range: 1.3–2.0.",
                    },
                ),
                "ring_thickness_frac": (
                    "FLOAT",
                    {
                        "default": 0.55,
                        "min": 0.1,
                        "max": 0.9,
                        "step": 0.05,
                        "tooltip": "Ring thickness as a fraction of the local inter-ring gap. 0.55 = rings and gaps roughly equal. Higher = fatter rings.",
                    },
                ),
                "hatch_density": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.2,
                        "max": 3.0,
                        "step": 0.1,
                        "tooltip": "Hatch mark density multiplier. 1.0 = one hatch per ring-width arc. Higher = denser hatching.",
                    },
                ),
                "bg_color": (
                    "STRING",
                    {"default": "#111111", "tooltip": "Background colour between rings."},
                ),
                "supersampling": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 4,
                        "tooltip": "Anti-aliasing factor.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("illusion",)
    OUTPUT_NODE = True

    def execute(
        self,
        width,
        height,
        num_rings=12,
        color_cycle="#FF2200,#FF8800,#FFDD00,#22CC00,#0066FF,#8800CC",
        hatch_angle=50.0,
        funnel_taper=1.6,
        ring_thickness_frac=0.55,
        hatch_density=1.0,
        bg_color="#111111",
        supersampling=2,
    ):
        ss = supersampling
        W, H = width * ss, height * ss
        bg = (*_hex_to_rgb(bg_color), 255)

        colors = [_hex_to_rgb(c.strip()) for c in color_cycle.split(",") if c.strip()]
        if not colors:
            colors = [(200, 50, 0)]

        img = Image.new("RGBA", (W, H), bg)
        draw = ImageDraw.Draw(img)

        cx, cy = W / 2.0, H / 2.0
        max_r = min(W, H) / 2.0
        ha = math.radians(hatch_angle)
        hw = max(1, ss)

        # Logarithmic ring radii: compressed toward centre (funnel effect)
        radii = [max_r * ((i + 1) / num_rings) ** funnel_taper for i in range(num_rings)]
        radii_sorted = sorted(radii)

        for ri, r in enumerate(radii_sorted):
            gap = r - (radii_sorted[ri - 1] if ri > 0 else 0.0)
            rw = max(2.0, gap * ring_thickness_frac)
            r_out = r
            r_in = max(1.0, r - rw)

            color = (*colors[ri % len(colors)], 255)
            bbox_out = [cx - r_out, cy - r_out, cx + r_out, cy + r_out]
            bbox_in = [cx - r_in, cy - r_in, cx + r_in, cy + r_in]
            draw.ellipse(bbox_out, fill=color)
            draw.ellipse(bbox_in, fill=bg)

            # Hatch marks — alternating angle per ring
            hatch_dir = ha if ri % 2 == 0 else -ha
            n_hatch = max(4, int(2 * math.pi * r / max(rw, 1) * hatch_density))
            half_hl = rw * 0.9
            cos_h, sin_h = math.cos(hatch_dir), math.sin(hatch_dir)

            r_mid = r - rw / 2.0
            for j in range(n_hatch):
                theta = 2 * math.pi * j / n_hatch
                hx = cx + r_mid * math.cos(theta)
                hy = cy + r_mid * math.sin(theta)
                draw.line(
                    [(hx - half_hl * cos_h, hy - half_hl * sin_h),
                     (hx + half_hl * cos_h, hy + half_hl * sin_h)],
                    fill=(*_hex_to_rgb(bg_color), 255),
                    width=hw,
                )

        return (_to_tensor(img.resize((width, height), Image.LANCZOS)),)


# ═══════════════════════════════════════════════════════════════════════════════
# Node 7 — DF_Swirl_Waterfall
# ═══════════════════════════════════════════════════════════════════════════════


class DF_Swirl_Waterfall(_SwirlBase):
    """Archimedes spiral motion aftereffect primer. PSE WARNING — see DESCRIPTION."""

    DESCRIPTION = (
        "Waterfall illusion — an Archimedes spiral with alternating black/white bands "
        "appears to expand outward. After ~30 s of viewing, stationary images appear "
        "to contract (motion aftereffect). Animate rotation_phase for a video primer. "
        "Based on Addams (1834). "
        + _EPILEPSY_WARNING
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": (
                    "INT",
                    {"default": 768, "min": 128, "max": 4096, "tooltip": "Output width in pixels."},
                ),
                "height": (
                    "INT",
                    {"default": 768, "min": 128, "max": 4096, "tooltip": "Output height in pixels."},
                ),
            },
            "optional": {
                "spiral_turns": (
                    "INT",
                    {
                        "default": 6,
                        "min": 1,
                        "max": 20,
                        "tooltip": "Number of full 360° turns in the Archimedes spiral. More turns = more bands visible = stronger aftereffect primer. Safe range: 4–10.",
                    },
                ),
                "band_width": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.2,
                        "max": 3.0,
                        "step": 0.1,
                        "tooltip": "Width of each band as a multiplier on the base period. 1.0 = bands equal to gap; 0.5 = thin bands; 2.0 = wide bands.",
                    },
                ),
                "rotation_phase": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Phase offset of the spiral bands in [0, 1]. Animating this 0→1 produces one full rotation cycle. Connect to a frame counter ÷ animation_frames for smooth animation.",
                    },
                ),
                "animation_frames": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 120,
                        "tooltip": "When > 1, outputs a batch of frames with rotation_phase evenly stepped from 0 to 1 — ready for video output. 24 frames = one full rotation at 24 fps.",
                    },
                ),
                "direction": (
                    ["outward", "inward"],
                    {
                        "tooltip": "'outward' = bands appear to expand (standard waterfall). 'inward' = bands appear to contract.",
                    },
                ),
                "safe_mode": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Use smooth sinusoidal bands instead of hard black/white edges to reduce flash intensity. Always enable for public display.",
                    },
                ),
                "bg_color": (
                    "STRING",
                    {
                        "default": "#888888",
                        "tooltip": "Background colour outside the spiral circle.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("illusion",)
    OUTPUT_NODE = True

    def execute(
        self,
        width,
        height,
        spiral_turns=6,
        band_width=1.0,
        rotation_phase=0.0,
        animation_frames=1,
        direction="outward",
        safe_mode=True,
        bg_color="#888888",
    ):
        cx, cy = width / 2.0, height / 2.0
        max_r = min(cx, cy)
        bg_val = np.array(_hex_to_rgb(bg_color), dtype=np.float32) / 255.0

        yi, xi = np.mgrid[0:height, 0:width].astype(np.float32)
        dx = xi - cx
        dy = yi - cy
        r = np.sqrt(dx**2 + dy**2)
        theta_raw = np.arctan2(dy, dx)  # [-pi, pi]
        theta_norm = (theta_raw + math.pi) / (2.0 * math.pi)  # [0, 1]
        circle_mask = (r <= max_r)
        r_norm = r / max_r

        frames = []
        for fi in range(animation_frames):
            phase = (fi / max(animation_frames, 1)) if animation_frames > 1 else rotation_phase
            if direction == "inward":
                phase = -phase

            # Archimedes spiral band index
            band_idx = r_norm * spiral_turns - theta_norm + phase
            frac = band_idx % 1.0

            if safe_mode:
                # Smooth sinusoidal bands — no hard edges
                lum = (np.sin(frac * 2.0 * math.pi / band_width) + 1.0) / 2.0
            else:
                # Sharp alternating black/white
                lum = (frac < (0.5 * band_width)).astype(np.float32)

            # Apply bg outside circle
            lum_clamped = lum * circle_mask + bg_val[0] * (~circle_mask)
            lum_u8 = (lum_clamped * 255).clip(0, 255).astype(np.uint8)
            rgba = np.stack([lum_u8, lum_u8, lum_u8, np.full_like(lum_u8, 255)], axis=-1)
            frames.append(rgba.astype(np.float32) / 255.0)

        batch = np.stack(frames, axis=0)  # (B, H, W, 4)
        try:
            import torch

            return (torch.from_numpy(batch),)
        except ImportError:
            return (batch,)


# ═══════════════════════════════════════════════════════════════════════════════
# Node 8 — DF_Swirl_ImpossibleRing
# ═══════════════════════════════════════════════════════════════════════════════


class DF_Swirl_ImpossibleRing(_SwirlBase):
    """Ambiguous / impossible ring — inconsistent shading creates Escher-style paradox."""

    DESCRIPTION = (
        "Impossible ring illusion — the left half is shaded as if lit from above, "
        "the right half as if lit from below, creating an Escher-style paradox where "
        "the ring surface appears to be simultaneously the front and back face. "
        "Based on Simanek (1996)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": (
                    "INT",
                    {"default": 768, "min": 128, "max": 4096, "tooltip": "Output width in pixels."},
                ),
                "height": (
                    "INT",
                    {"default": 768, "min": 128, "max": 4096, "tooltip": "Output height in pixels."},
                ),
            },
            "optional": {
                "ring_radius": (
                    "FLOAT",
                    {
                        "default": 0.35,
                        "min": 0.1,
                        "max": 0.48,
                        "step": 0.01,
                        "tooltip": "Radius of the ring centreline as a fraction of the half image size. 0.35 fills most of the frame.",
                    },
                ),
                "ring_thickness": (
                    "FLOAT",
                    {
                        "default": 0.12,
                        "min": 0.02,
                        "max": 0.3,
                        "step": 0.01,
                        "tooltip": "Thickness of the ring tube as a fraction of image half-size. Should be well less than ring_radius.",
                    },
                ),
                "color_a": (
                    "STRING",
                    {
                        "default": "#888888",
                        "tooltip": "Base ring colour as #RRGGBB. Neutral grey is classical for the impossible shading to read clearly.",
                    },
                ),
                "color_b": (
                    "STRING",
                    {
                        "default": "#444444",
                        "tooltip": "Shadow / dark face colour as #RRGGBB.",
                    },
                ),
                "highlight_strength": (
                    "FLOAT",
                    {
                        "default": 0.45,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Brightness of the specular highlight on the tube edge. 0 = matte; 0.45 = soft highlight; 1.0 = bright specular.",
                    },
                ),
                "bg_color": (
                    "STRING",
                    {
                        "default": "#F0F0F0",
                        "tooltip": "Background colour outside the ring.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("illusion",)
    OUTPUT_NODE = True

    def execute(
        self,
        width,
        height,
        ring_radius=0.35,
        ring_thickness=0.12,
        color_a="#888888",
        color_b="#444444",
        highlight_strength=0.45,
        bg_color="#F0F0F0",
    ):
        cx, cy = width / 2.0, height / 2.0
        half = min(cx, cy)
        R = half * ring_radius        # ring centreline radius in pixels
        T = half * ring_thickness     # tube radius in pixels

        yi, xi = np.mgrid[0:height, 0:width].astype(np.float32)
        dx = xi - cx
        dy = yi - cy
        r = np.sqrt(dx**2 + dy**2)

        # Distance from ring centreline
        d = np.abs(r - R)
        on_ring = d < T

        # Shade = position within tube cross-section, 0=outer, 1=inner
        t_norm = np.where(on_ring, (T - d) / T, 0.0)  # 1 at centreline, 0 at edge

        # Outer vs inner half of tube
        is_outer = r > R

        # Left half uses consistent lighting: outer bright, inner dark
        # Right half uses inverted lighting: inner bright, outer dark
        left = dx < 0

        shade_consistent = np.where(is_outer, 0.85, 0.25)
        shade_inverted = 1.0 - shade_consistent

        shade = np.where(left, shade_consistent, shade_inverted)

        # Specular highlight at tube edge (proximity to d=0 boundary)
        highlight = np.exp(-d**2 / max((T * 0.3) ** 2, 1.0)) * highlight_strength

        ca = np.array(_hex_to_rgb(color_a), dtype=np.float32) / 255.0
        cb = np.array(_hex_to_rgb(color_b), dtype=np.float32) / 255.0
        bg = np.array(_hex_to_rgb(bg_color), dtype=np.float32) / 255.0

        s3 = shade[:, :, None]
        h3 = highlight[:, :, None]
        ring_rgb = (ca[None, None, :] * s3 + cb[None, None, :] * (1.0 - s3) + h3).clip(0.0, 1.0)

        on_ring_3 = on_ring[:, :, None]
        final = ring_rgb * on_ring_3 + bg[None, None, :] * (~on_ring_3)

        rgba = np.concatenate(
            [(final * 255).clip(0, 255).astype(np.uint8),
             np.full((height, width, 1), 255, dtype=np.uint8)],
            axis=-1,
        )
        return (_np_to_tensor(rgba),)


# ═══════════════════════════════════════════════════════════════════════════════
# Node 9 — DF_Swirl_GravityWave
# ═══════════════════════════════════════════════════════════════════════════════


class DF_Swirl_GravityWave(_SwirlBase):
    """Kitaoka gravity-wave — sinusoidal phase advance per ring creates rolling motion."""

    DESCRIPTION = (
        "Gravity Wave spiral — concentric rings with a sinusoidal luminance wave whose "
        "phase advances with radius. The result appears to roll inward like a standing "
        "gravitational wave. Inspired by Kitaoka's Gravity Wave Spirals series."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": (
                    "INT",
                    {"default": 768, "min": 128, "max": 4096, "tooltip": "Output width in pixels."},
                ),
                "height": (
                    "INT",
                    {"default": 768, "min": 128, "max": 4096, "tooltip": "Output height in pixels."},
                ),
            },
            "optional": {
                "num_rings": (
                    "INT",
                    {
                        "default": 10,
                        "min": 2,
                        "max": 30,
                        "tooltip": "Number of radial ring periods. More rings = more wave cycles = stronger illusion. Safe range: 6–16.",
                    },
                ),
                "wave_frequency": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 32,
                        "tooltip": "Number of sinusoidal wave crests around each ring circumference. 8 = octagonal wave; 1 = single lobe. Safe range: 4–16.",
                    },
                ),
                "wave_amplitude": (
                    "FLOAT",
                    {
                        "default": 0.48,
                        "min": 0.05,
                        "max": 0.5,
                        "step": 0.01,
                        "tooltip": "Peak-to-peak amplitude of the luminance wave. 0.5 = full black-to-white swing. Lower = softer wave, weaker illusion.",
                    },
                ),
                "phase_advance": (
                    "FLOAT",
                    {
                        "default": 1.8,
                        "min": 0.1,
                        "max": 6.28,
                        "step": 0.1,
                        "tooltip": "Phase advance per ring in radians. Pi/2 (1.57) = quarter-wave advance per ring; Pi (3.14) = half-wave (reverses direction). This value drives the apparent inward rolling motion.",
                    },
                ),
                "color_mode": (
                    ["greyscale", "warm", "cool", "rainbow"],
                    {
                        "tooltip": (
                            "'greyscale' = classic luminance-only wave. "
                            "'warm' = orange-red palette. "
                            "'cool' = blue-purple palette. "
                            "'rainbow' = full hue rotation per ring."
                        )
                    },
                ),
                "bg_color": (
                    "STRING",
                    {
                        "default": "#808080",
                        "tooltip": "Background colour outside the ring area. Mid-grey matches the wave neutral point.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("illusion",)
    OUTPUT_NODE = True

    def execute(
        self,
        width,
        height,
        num_rings=10,
        wave_frequency=8,
        wave_amplitude=0.48,
        phase_advance=1.8,
        color_mode="greyscale",
        bg_color="#808080",
    ):
        cx, cy = width / 2.0, height / 2.0
        max_r = min(cx, cy)
        bg = np.array(_hex_to_rgb(bg_color), dtype=np.float32) / 255.0

        yi, xi = np.mgrid[0:height, 0:width].astype(np.float32)
        dx = xi - cx
        dy = yi - cy
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)

        r_norm = r / max_r  # 0 at centre, 1 at edge
        ring_idx = r_norm * num_rings  # fractional ring index

        # Sinusoidal wave with phase advancing per ring
        phase = wave_frequency * theta + ring_idx * phase_advance
        wave = np.sin(phase) * wave_amplitude + 0.5  # [0.5-amp, 0.5+amp]
        wave = np.clip(wave, 0.0, 1.0)

        # Edge fade
        edge_fade = np.clip(1.0 - r_norm, 0.0, 1.0)
        wave = wave * edge_fade + 0.5 * (1.0 - edge_fade)

        if color_mode == "greyscale":
            w_u8 = (wave * 255).astype(np.uint8)
            rgba = np.stack([w_u8, w_u8, w_u8, np.full_like(w_u8, 255)], axis=-1)

        elif color_mode in ("warm", "cool"):
            hue_center = 0.05 if color_mode == "warm" else 0.65  # orange vs blue
            rgb_arr = np.zeros((height, width, 3), dtype=np.float32)
            for j in range(3):
                h_ch, s_ch, _ = colorsys.rgb_to_hsv(*([hue_center, 0.8, 1.0]))
                r_c, g_c, b_c = colorsys.hsv_to_rgb(hue_center, 0.7 + 0.3 * (j == 0), 1.0)
                base = [r_c, g_c, b_c][j]
                rgb_arr[:, :, j] = wave * base
            rgb_u8 = (np.clip(rgb_arr, 0, 1) * 255).astype(np.uint8)
            rgba = np.concatenate([rgb_u8, np.full((height, width, 1), 255, np.uint8)], axis=-1)

        else:  # rainbow
            hue_map = (ring_idx / num_rings + theta / (2.0 * math.pi) * 0.3) % 1.0
            rgb_arr = np.zeros((height, width, 3), dtype=np.float32)
            # vectorised HSV→RGB via numpy (H, S=0.8, V=wave)
            h6 = hue_map * 6.0
            hi = np.floor(h6).astype(int) % 6
            f = h6 - np.floor(h6)
            S = 0.8
            p = wave * (1.0 - S)
            q = wave * (1.0 - S * f)
            t_ = wave * (1.0 - S * (1.0 - f))
            cases = [
                (wave, t_, p),
                (q, wave, p),
                (p, wave, t_),
                (p, q, wave),
                (t_, p, wave),
                (wave, p, q),
            ]
            for idx, (rv, gv, bv) in enumerate(cases):
                mask = hi == idx
                rgb_arr[:, :, 0] = np.where(mask, rv, rgb_arr[:, :, 0])
                rgb_arr[:, :, 1] = np.where(mask, gv, rgb_arr[:, :, 1])
                rgb_arr[:, :, 2] = np.where(mask, bv, rgb_arr[:, :, 2])
            rgb_u8 = (np.clip(rgb_arr, 0, 1) * 255).astype(np.uint8)
            rgba = np.concatenate([rgb_u8, np.full((height, width, 1), 255, np.uint8)], axis=-1)

        return (_np_to_tensor(rgba),)


# ═══════════════════════════════════════════════════════════════════════════════
# Node 10 — DF_Swirl_Composer
# ═══════════════════════════════════════════════════════════════════════════════


class DF_Swirl_Composer(_SwirlBase):
    """Composite two Swirls illusions with blend modes and an optional mask."""

    DESCRIPTION = (
        "Composer — blend two Swirls (or any IMAGE) outputs together with standard "
        "photographic blend modes and an optional greyscale mask controlling where "
        "each illusion appears."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "illusion_a": (
                    "IMAGE",
                    {
                        "tooltip": "First illusion or image (base layer). Connect any DF_Swirl_ node output here."
                    },
                ),
                "illusion_b": (
                    "IMAGE",
                    {
                        "tooltip": "Second illusion or image (blend layer). Applied on top of illusion_a."
                    },
                ),
            },
            "optional": {
                "mask": (
                    "IMAGE",
                    {
                        "tooltip": "Optional greyscale mask (H, W). White = show illusion_b; black = show illusion_a. If not connected, opacity_b controls global blending.",
                    },
                ),
                "blend_mode": (
                    ["normal", "multiply", "screen", "overlay", "add", "difference"],
                    {
                        "tooltip": (
                            "'normal' = standard alpha compositing. "
                            "'multiply' = darkens (good for layering dark illusions). "
                            "'screen' = lightens (good for layering light illusions). "
                            "'overlay' = contrast boost where A is light, darken where dark. "
                            "'add' = additive glow. "
                            "'difference' = psychedelic inversion effect."
                        )
                    },
                ),
                "opacity_a": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Opacity of illusion_a (base layer). 1.0 = fully opaque.",
                    },
                ),
                "opacity_b": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Opacity of illusion_b (blend layer). 0.7 = 70% blend on top. Reduced by mask where mask is black.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("composed",)
    OUTPUT_NODE = True

    def execute(
        self,
        illusion_a,
        illusion_b,
        mask=None,
        blend_mode="normal",
        opacity_a=1.0,
        opacity_b=0.7,
    ):
        a_raw = _tensor_to_np(illusion_a).astype(np.float32) / 255.0  # (H, W, 4)
        b_raw = _tensor_to_np(illusion_b).astype(np.float32) / 255.0

        # Resize B to match A
        if b_raw.shape[:2] != a_raw.shape[:2]:
            H, W = a_raw.shape[:2]
            b_pil = Image.fromarray((b_raw * 255).astype(np.uint8), "RGBA")
            b_raw = np.array(b_pil.resize((W, H), Image.LANCZOS)).astype(np.float32) / 255.0

        a = a_raw[:, :, :3] * opacity_a
        b = b_raw[:, :, :3]

        if blend_mode == "normal":
            blended = b * opacity_b + a * (1.0 - opacity_b)
        elif blend_mode == "multiply":
            blended = a * b * opacity_b + a * (1.0 - opacity_b)
        elif blend_mode == "screen":
            blended_pure = 1.0 - (1.0 - a) * (1.0 - b)
            blended = blended_pure * opacity_b + a * (1.0 - opacity_b)
        elif blend_mode == "overlay":
            low = 2.0 * a * b
            high = 1.0 - 2.0 * (1.0 - a) * (1.0 - b)
            blended_pure = np.where(a < 0.5, low, high)
            blended = blended_pure * opacity_b + a * (1.0 - opacity_b)
        elif blend_mode == "add":
            blended = np.clip(a + b * opacity_b, 0.0, 1.0)
        elif blend_mode == "difference":
            blended_pure = np.abs(a - b)
            blended = blended_pure * opacity_b + a * (1.0 - opacity_b)
        else:
            blended = a

        # Apply mask
        if mask is not None:
            m_raw = _tensor_to_np(mask).astype(np.float32) / 255.0
            if m_raw.shape[:2] != a_raw.shape[:2]:
                H, W = a_raw.shape[:2]
                m_pil = Image.fromarray((m_raw * 255).astype(np.uint8), "RGBA")
                m_raw = np.array(m_pil.resize((W, H), Image.LANCZOS)).astype(np.float32) / 255.0
            m = m_raw[:, :, :1].mean(axis=-1, keepdims=True)  # greyscale
            blended = blended * m + a * (1.0 - m)

        blended = np.clip(blended, 0.0, 1.0)
        rgb_u8 = (blended * 255).astype(np.uint8)
        rgba = np.concatenate([rgb_u8, np.full((*rgb_u8.shape[:2], 1), 255, np.uint8)], axis=-1)
        return (_np_to_tensor(rgba),)


# ── Registration ──────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "DF_Swirl_FraserSpiral":    DF_Swirl_FraserSpiral,
    "DF_Swirl_ZollnerSpiral":   DF_Swirl_ZollnerSpiral,
    "DF_Swirl_PeripheralDrift": DF_Swirl_PeripheralDrift,
    "DF_Swirl_NeuralCircuit":   DF_Swirl_NeuralCircuit,
    "DF_Swirl_MagneticStorm":   DF_Swirl_MagneticStorm,
    "DF_Swirl_Tornado":         DF_Swirl_Tornado,
    "DF_Swirl_Waterfall":       DF_Swirl_Waterfall,
    "DF_Swirl_ImpossibleRing":  DF_Swirl_ImpossibleRing,
    "DF_Swirl_GravityWave":     DF_Swirl_GravityWave,
    "DF_Swirl_Composer":        DF_Swirl_Composer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DF_Swirl_FraserSpiral":    "DepthForge Swirl — Fraser Spiral",
    "DF_Swirl_ZollnerSpiral":   "DepthForge Swirl — Zöllner Spiral",
    "DF_Swirl_PeripheralDrift": "DepthForge Swirl — Peripheral Drift",
    "DF_Swirl_NeuralCircuit":   "DepthForge Swirl — Neural Circuit",
    "DF_Swirl_MagneticStorm":   "DepthForge Swirl — Magnetic Storm",
    "DF_Swirl_Tornado":         "DepthForge Swirl — Tornado",
    "DF_Swirl_Waterfall":       "DepthForge Swirl — Waterfall",
    "DF_Swirl_ImpossibleRing":  "DepthForge Swirl — Impossible Ring",
    "DF_Swirl_GravityWave":     "DepthForge Swirl — Gravity Wave",
    "DF_Swirl_Composer":        "DepthForge Swirl — Composer",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
