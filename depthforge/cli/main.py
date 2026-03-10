"""
depthforge.cli.main
===================
Command-line interface for DepthForge.

Usage
-----
    depthforge --help
    depthforge sirds --depth depth.png --output out.png
    depthforge texture --depth depth.png --pattern plasma --output out.png
    depthforge anaglyph --source photo.jpg --depth depth.png --output anaglyph.png
    depthforge hidden --shape star --width 1920 --height 1080 --output hidden.png
    depthforge stereo-pair --source photo.jpg --depth depth.png --output sbs.png
    depthforge batch --input-dir ./depths/ --output-dir ./stereos/ --mode texture
    depthforge analyze --depth depth.png
    depthforge presets [--list] [--show NAME]
    depthforge estimate-depth --source photo.jpg --output depth.png
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import click
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_depth(path: str) -> np.ndarray:
    """Load a depth image and return float32 [0,1] array."""
    img = Image.open(path)
    if img.mode == "I":
        arr = np.array(img, dtype=np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    elif img.mode == "F":
        arr = np.array(img, dtype=np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    else:
        arr = np.array(img.convert("L"), dtype=np.float32) / 255.0
    return arr


def _load_source(path: str) -> np.ndarray:
    """Load a colour source image as RGBA uint8."""
    return np.array(Image.open(path).convert("RGBA"), dtype=np.uint8)


def _save(arr: np.ndarray, path: str, dpi: int = 72) -> None:
    """Save RGBA uint8 array to file."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(arr)
    fmt = out_path.suffix.lower()
    if fmt in (".jpg", ".jpeg"):
        img = img.convert("RGB")
        img.save(out_path, quality=95, dpi=(dpi, dpi))
    elif fmt == ".tiff":
        img.save(out_path, dpi=(dpi, dpi))
    else:
        img.save(out_path, dpi=(dpi, dpi))


def _build_depth_params(
    bilateral_space: float,
    bilateral_color: float,
    dilation: int,
    falloff: str,
    near_plane: float,
    far_plane: float,
    invert: bool,
):
    from depthforge.core.depth_prep import DepthPrepParams, FalloffCurve

    curve_map = {
        "linear": FalloffCurve.LINEAR,
        "gamma": FalloffCurve.GAMMA,
        "s-curve": FalloffCurve.S_CURVE,
        "scurve": FalloffCurve.S_CURVE,
        "logarithmic": FalloffCurve.LOGARITHMIC,
        "exponential": FalloffCurve.EXPONENTIAL,
    }
    curve = curve_map.get(falloff.lower(), FalloffCurve.LINEAR)
    return DepthPrepParams(
        invert=invert,
        bilateral_sigma_space=bilateral_space,
        bilateral_sigma_color=bilateral_color,
        dilation_px=dilation,
        falloff_curve=curve,
        near_plane=near_plane,
        far_plane=far_plane,
    )


def _build_stereo_params(
    depth_factor: float,
    max_parallax: float,
    safe_mode: bool,
    oversample: int,
    seed: Optional[int],
    invert_depth: bool,
):
    from depthforge.core.synthesizer import StereoParams

    return StereoParams(
        depth_factor=depth_factor,
        max_parallax_fraction=max_parallax,
        safe_mode=safe_mode,
        oversample=oversample,
        seed=seed,
        invert_depth=invert_depth,
    )


def _build_pattern(pattern: str, tile_size: int, color: str, seed: int):
    from depthforge.core.pattern_gen import ColorMode, PatternParams, PatternType, generate_pattern

    type_map = {
        "noise": PatternType.RANDOM_NOISE,
        "random": PatternType.RANDOM_NOISE,
        "perlin": PatternType.PERLIN,
        "plasma": PatternType.PLASMA,
        "voronoi": PatternType.VORONOI,
        "grid": PatternType.GEOMETRIC_GRID,
        "mandelbrot": PatternType.MANDELBROT,
        "dots": PatternType.DOT_MATRIX,
    }
    color_map = {
        "greyscale": ColorMode.GREYSCALE,
        "gray": ColorMode.GREYSCALE,
        "mono": ColorMode.MONOCHROME,
        "monochrome": ColorMode.MONOCHROME,
        "psychedelic": ColorMode.PSYCHEDELIC,
        "color": ColorMode.PSYCHEDELIC,
    }
    ptype = type_map.get(pattern.lower(), PatternType.RANDOM_NOISE)
    cmode = color_map.get(color.lower(), ColorMode.GREYSCALE)
    return generate_pattern(
        PatternParams(
            pattern_type=ptype,
            tile_width=tile_size,
            tile_height=tile_size,
            color_mode=cmode,
            seed=seed,
        )
    )


# ---------------------------------------------------------------------------
# Common options (shared across commands)
# ---------------------------------------------------------------------------

_depth_opts = [
    click.option("--depth", "-d", required=True, help="Path to depth map image."),
    click.option("--output", "-o", required=True, help="Output file path."),
    click.option(
        "--preset",
        default=None,
        help="Use a named preset (shallow/medium/deep/cinema/print/broadcast).",
    ),
    click.option(
        "--depth-factor", default=0.35, show_default=True, help="Parallax strength (0.0–1.0)."
    ),
    click.option(
        "--max-parallax",
        default=1 / 30,
        show_default=True,
        help="Max parallax as fraction of frame width.",
    ),
    click.option(
        "--safe-mode", is_flag=True, default=False, help="Hard-clamp parallax to safe limits."
    ),
    click.option(
        "--oversample", default=1, show_default=True, help="Supersampling factor (1=off, 2=2x)."
    ),
    click.option("--seed", default=None, type=int, help="Random seed for reproducibility."),
    click.option(
        "--invert-depth", is_flag=True, default=False, help="Invert near/far in depth map."
    ),
    click.option(
        "--bilateral-space", default=5.0, show_default=True, help="Bilateral filter spatial sigma."
    ),
    click.option(
        "--bilateral-color", default=0.1, show_default=True, help="Bilateral filter color sigma."
    ),
    click.option(
        "--dilation", default=3, show_default=True, help="Morphological dilation radius (px)."
    ),
    click.option(
        "--falloff",
        default="linear",
        show_default=True,
        type=click.Choice(
            ["linear", "gamma", "s-curve", "logarithmic", "exponential"], case_sensitive=False
        ),
        help="Depth falloff curve.",
    ),
    click.option("--near-plane", default=0.0, show_default=True, help="Near depth clip (0–1)."),
    click.option("--far-plane", default=1.0, show_default=True, help="Far depth clip (0–1)."),
    click.option("--dpi", default=72, show_default=True, help="Output DPI (for print mode)."),
    click.option("--verbose", "-v", is_flag=True, default=False, help="Verbose output."),
]


def _add_depth_opts(f):
    for opt in reversed(_depth_opts):
        f = opt(f)
    return f


_pattern_opts = [
    click.option(
        "--pattern",
        "-p",
        default="noise",
        show_default=True,
        type=click.Choice(
            ["noise", "random", "perlin", "plasma", "voronoi", "grid", "mandelbrot", "dots"],
            case_sensitive=False,
        ),
        help="Pattern generator type.",
    ),
    click.option(
        "--color",
        "-c",
        default="greyscale",
        show_default=True,
        type=click.Choice(
            ["greyscale", "gray", "mono", "monochrome", "psychedelic", "color"],
            case_sensitive=False,
        ),
        help="Pattern color mode.",
    ),
    click.option(
        "--tile-size", default=128, show_default=True, help="Pattern tile size in pixels."
    ),
    click.option(
        "--tile-image", default=None, help="Path to custom tile image (overrides --pattern)."
    ),
]


def _add_pattern_opts(f):
    for opt in reversed(_pattern_opts):
        f = opt(f)
    return f


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(package_name="depthforge")
def cli():
    """DepthForge — Professional Stereogram Engine

    Generate SIRDS, texture stereograms, anaglyphs, stereo pairs,
    and hidden image stereograms from depth maps.

    \b
    Quick start:
        depthforge sirds --depth depth.png --output out.png
        depthforge texture --depth depth.png --pattern plasma --color psychedelic -o out.png
        depthforge analyze --depth depth.png
        depthforge presets --list
    """
    pass


# ---------------------------------------------------------------------------
# sirds command
# ---------------------------------------------------------------------------


@cli.command()
@_add_depth_opts
@_add_pattern_opts
def sirds(
    depth,
    output,
    preset,
    depth_factor,
    max_parallax,
    safe_mode,
    oversample,
    seed,
    invert_depth,
    bilateral_space,
    bilateral_color,
    dilation,
    falloff,
    near_plane,
    far_plane,
    dpi,
    verbose,
    pattern,
    color,
    tile_size,
    tile_image,
):
    """Generate a Single Image Random Dot Stereogram (SIRDS / Magic Eye)."""
    from depthforge.core.depth_prep import prep_depth
    from depthforge.core.synthesizer import synthesize

    t0 = time.time()
    _maybe_verbose(verbose, f"Loading depth: {depth}")

    # Preset overrides
    if preset:
        from depthforge.core.presets import get_preset

        p = get_preset(preset)
        depth_factor = p.stereo_params.depth_factor
        max_parallax = p.stereo_params.max_parallax_fraction
        safe_mode = p.stereo_params.safe_mode
        oversample = p.stereo_params.oversample
        dpi = p.output_dpi
        bilateral_space = p.depth_params.bilateral_sigma_space
        bilateral_color = p.depth_params.bilateral_sigma_color
        dilation = p.depth_params.dilation_px
        near_plane = p.depth_params.near_plane
        far_plane = p.depth_params.far_plane

    raw = _load_depth(depth)
    H, W = raw.shape
    _maybe_verbose(verbose, f"Depth size: {W}×{H}")

    dp = _build_depth_params(
        bilateral_space, bilateral_color, dilation, falloff, near_plane, far_plane, invert_depth
    )
    dep = prep_depth(raw, dp)

    if tile_image:
        from depthforge.core.pattern_gen import load_tile, tile_to_frame

        pat = tile_to_frame(load_tile(tile_image), H, W)
    else:
        pat = _build_pattern(
            pattern if pattern != "noise" else "random", tile_size, color, seed or 1
        )

    sp = _build_stereo_params(depth_factor, max_parallax, safe_mode, oversample, seed, False)
    _maybe_verbose(
        verbose, f"Synthesizing... (depth_factor={depth_factor}, oversample={oversample})"
    )

    result = synthesize(dep, pat, sp)
    _save(result, output, dpi=dpi)

    elapsed = time.time() - t0
    click.echo(f"✓ SIRDS saved: {output}  ({W}×{H}, {elapsed:.1f}s)")


# ---------------------------------------------------------------------------
# texture command
# ---------------------------------------------------------------------------


@cli.command()
@_add_depth_opts
@_add_pattern_opts
def texture(
    depth,
    output,
    preset,
    depth_factor,
    max_parallax,
    safe_mode,
    oversample,
    seed,
    invert_depth,
    bilateral_space,
    bilateral_color,
    dilation,
    falloff,
    near_plane,
    far_plane,
    dpi,
    verbose,
    pattern,
    color,
    tile_size,
    tile_image,
):
    """Generate a texture-pattern stereogram (SIS)."""
    # texture is basically sirds but defaults to plasma/psychedelic
    from depthforge.core.depth_prep import prep_depth
    from depthforge.core.synthesizer import synthesize

    t0 = time.time()

    if preset:
        from depthforge.core.presets import get_preset

        p = get_preset(preset)
        depth_factor = p.stereo_params.depth_factor
        max_parallax = p.stereo_params.max_parallax_fraction
        safe_mode = p.stereo_params.safe_mode
        oversample = p.stereo_params.oversample
        dpi = p.output_dpi

    raw = _load_depth(depth)
    H, W = raw.shape

    dp = _build_depth_params(
        bilateral_space, bilateral_color, dilation, falloff, near_plane, far_plane, invert_depth
    )
    dep = prep_depth(raw, dp)

    if tile_image:
        from depthforge.core.pattern_gen import load_tile, tile_to_frame

        pat = tile_to_frame(load_tile(tile_image), H, W)
    else:
        pat = _build_pattern(pattern, tile_size, color, seed or 42)

    sp = _build_stereo_params(depth_factor, max_parallax, safe_mode, oversample, seed, False)
    result = synthesize(dep, pat, sp)
    _save(result, output, dpi=dpi)

    elapsed = time.time() - t0
    click.echo(f"✓ Texture stereogram saved: {output}  ({W}×{H}, {elapsed:.1f}s)")


# ---------------------------------------------------------------------------
# anaglyph command
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--source", "-s", required=True, help="Source colour image.")
@click.option("--depth", "-d", required=True, help="Depth map image.")
@click.option("--output", "-o", required=True, help="Output file path.")
@click.option(
    "--mode",
    default="optimised",
    show_default=True,
    type=click.Choice(["true", "grey", "color", "half-color", "optimised"], case_sensitive=False),
    help="Anaglyph mode.",
)
@click.option(
    "--eye-separation", default=0.065, show_default=True, help="Eye separation (normalised)."
)
@click.option("--depth-factor", default=0.35, show_default=True, help="Parallax strength.")
@click.option("--swap-eyes", is_flag=True, default=False, help="Swap left/right eyes.")
@click.option("--invert-depth", is_flag=True, default=False, help="Invert near/far.")
@click.option("--dpi", default=72, show_default=True, help="Output DPI.")
@click.option("--verbose", "-v", is_flag=True, default=False)
def anaglyph(
    source, depth, output, mode, eye_separation, depth_factor, swap_eyes, invert_depth, dpi, verbose
):
    """Generate a red/cyan anaglyph image for 3D glasses."""
    from depthforge.core.anaglyph import AnaglyphMode, AnaglyphParams, make_anaglyph_from_depth
    from depthforge.core.depth_prep import DepthPrepParams, prep_depth

    mode_map = {
        "true": AnaglyphMode.TRUE_ANAGLYPH,
        "grey": AnaglyphMode.GREY_ANAGLYPH,
        "color": AnaglyphMode.COLOUR_ANAGLYPH,
        "half-color": AnaglyphMode.HALF_COLOUR,
        "optimised": AnaglyphMode.OPTIMISED,
    }

    t0 = time.time()
    src = _load_source(source)
    raw = _load_depth(depth)
    dep = prep_depth(raw, DepthPrepParams(invert=invert_depth))

    # Convert eye_separation + depth_factor to parallax_px at 1920-equivalent width
    W = src.shape[1]
    parallax_px = int(W * eye_separation * depth_factor)
    params = AnaglyphParams(
        mode=mode_map[mode],
        parallax_px=parallax_px,
        swap_eyes=swap_eyes,
    )
    result = make_anaglyph_from_depth(src, dep, params)
    _save(result, output, dpi=dpi)

    elapsed = time.time() - t0
    click.echo(f"✓ Anaglyph saved: {output}  (mode={mode}, {elapsed:.1f}s)")


# ---------------------------------------------------------------------------
# hidden command
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--output", "-o", required=True, help="Output file path.")
@click.option(
    "--shape", default=None, help="Built-in shape: circle, square, triangle, star, diamond, arrow."
)
@click.option("--text", default=None, help="Text to hide in the stereogram.")
@click.option("--mask", default=None, help="Path to B&W mask image.")
@click.option("--width", "-W", default=1920, show_default=True, help="Output width in pixels.")
@click.option("--height", "-H", default=1080, show_default=True, help="Output height in pixels.")
@click.option(
    "--foreground-depth", default=0.75, show_default=True, help="Depth of the hidden element."
)
@click.option(
    "--background-depth", default=0.10, show_default=True, help="Depth of the background."
)
@click.option("--edge-soften", default=3, show_default=True, help="Edge feather radius (px).")
@click.option(
    "--pattern",
    "-p",
    default="noise",
    show_default=True,
    type=click.Choice(
        ["noise", "perlin", "plasma", "voronoi", "grid", "mandelbrot", "dots"], case_sensitive=False
    ),
    help="Pattern type.",
)
@click.option(
    "--color",
    "-c",
    default="greyscale",
    show_default=True,
    type=click.Choice(["greyscale", "gray", "mono", "psychedelic", "color"], case_sensitive=False),
    help="Pattern color mode.",
)
@click.option("--tile-size", default=128, show_default=True, help="Pattern tile size.")
@click.option("--seed", default=1, show_default=True, help="Random seed.")
@click.option("--font-size", default=None, type=int, help="Font size for --text mode.")
@click.option("--dpi", default=72, show_default=True, help="Output DPI.")
@click.option("--verbose", "-v", is_flag=True, default=False)
def hidden(
    output,
    shape,
    text,
    mask,
    width,
    height,
    foreground_depth,
    background_depth,
    edge_soften,
    pattern,
    color,
    tile_size,
    seed,
    font_size,
    dpi,
    verbose,
):
    """Generate a hidden-image stereogram (shape, text, or mask)."""
    from depthforge.core.hidden_image import (
        HiddenImageParams,
        encode_hidden_image,
        load_hidden_mask,
        shape_to_mask,
        text_to_mask,
    )

    if not shape and not text and not mask:
        raise click.UsageError("Provide one of: --shape, --text, or --mask")

    t0 = time.time()

    if shape:
        msk = shape_to_mask(shape, width, height)
    elif text:
        kwargs = {}
        if font_size:
            kwargs["font_size"] = font_size
        msk = text_to_mask(text, width, height, **kwargs)
    else:
        msk = load_hidden_mask(mask, width, height)

    pat = _build_pattern(pattern, tile_size, color, seed)
    params = HiddenImageParams(
        foreground_depth=foreground_depth,
        background_depth=background_depth,
        edge_soften_px=edge_soften,
    )
    result = encode_hidden_image(pat, msk, params)
    _save(result, output, dpi=dpi)

    elapsed = time.time() - t0
    label = shape or text or Path(mask).name
    click.echo(f"✓ Hidden image saved: {output}  ({label}, {elapsed:.1f}s)")


# ---------------------------------------------------------------------------
# stereo-pair command
# ---------------------------------------------------------------------------


@cli.command("stereo-pair")
@click.option("--source", "-s", required=True, help="Source image.")
@click.option("--depth", "-d", required=True, help="Depth map.")
@click.option("--output", "-o", required=True, help="Output file.")
@click.option(
    "--layout",
    default="side-by-side",
    show_default=True,
    type=click.Choice(["side-by-side", "top-bottom", "separate", "anaglyph"], case_sensitive=False),
    help="Output layout.",
)
@click.option(
    "--eye-separation", default=0.065, show_default=True, help="Eye separation (normalised)."
)
@click.option("--depth-factor", default=0.35, show_default=True, help="Parallax strength.")
@click.option("--gap", default=0, show_default=True, help="Gap between views in pixels.")
@click.option(
    "--bg-fill",
    default="edge",
    show_default=True,
    type=click.Choice(["edge", "mirror", "black"], case_sensitive=False),
    help="Background fill mode for occluded regions.",
)
@click.option("--invert-depth", is_flag=True, default=False, help="Invert near/far.")
@click.option("--dpi", default=72, show_default=True, help="Output DPI.")
@click.option("--verbose", "-v", is_flag=True, default=False)
def stereo_pair(
    source,
    depth,
    output,
    layout,
    eye_separation,
    depth_factor,
    gap,
    bg_fill,
    invert_depth,
    dpi,
    verbose,
):
    """Generate a stereo pair (side-by-side, top-bottom, or separate files)."""
    from depthforge.core.depth_prep import DepthPrepParams, prep_depth
    from depthforge.core.stereo_pair import StereoPairParams as SPParams
    from depthforge.core.stereo_pair import compose_side_by_side, make_stereo_pair

    t0 = time.time()
    src = _load_source(source)
    raw = _load_depth(depth)
    dep = prep_depth(raw, DepthPrepParams(invert=invert_depth))

    sp = SPParams(background_fill=bg_fill)
    left, right, occ = make_stereo_pair(src, dep * depth_factor, sp)

    out_path = Path(output)
    if layout == "separate":
        stem = out_path.stem
        ext = out_path.suffix
        left_path = out_path.parent / f"{stem}_left{ext}"
        right_path = out_path.parent / f"{stem}_right{ext}"
        _save(left, str(left_path), dpi=dpi)
        _save(right, str(right_path), dpi=dpi)
        click.echo(f"✓ Stereo pair saved:\n  L: {left_path}\n  R: {right_path}")
    elif layout == "top-bottom":
        tb = np.concatenate([left, right], axis=0)
        _save(tb, output, dpi=dpi)
        click.echo(f"✓ Top-bottom stereo saved: {output}")
    elif layout == "anaglyph":
        from depthforge.core.anaglyph import AnaglyphMode, AnaglyphParams, make_anaglyph

        result = make_anaglyph(left, right, AnaglyphParams(mode=AnaglyphMode.OPTIMISED))
        _save(result, output, dpi=dpi)
        click.echo(f"✓ Anaglyph saved: {output}")
    else:
        sbs = compose_side_by_side(left, right, gap_px=gap)
        _save(sbs, output, dpi=dpi)
        H, W = sbs.shape[:2]
        click.echo(f"✓ Side-by-side stereo saved: {output}  ({W}×{H}, {time.time()-t0:.1f}s)")


# ---------------------------------------------------------------------------
# batch command
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--input-dir", "-i", required=True, help="Directory of depth maps.")
@click.option("--output-dir", "-o", required=True, help="Output directory.")
@click.option(
    "--mode",
    default="texture",
    show_default=True,
    type=click.Choice(["sirds", "texture", "anaglyph", "hidden"], case_sensitive=False),
    help="Synthesis mode.",
)
@click.option("--preset", default="medium", show_default=True, help="Preset name.")
@click.option("--pattern", default="plasma", show_default=True, help="Pattern type.")
@click.option("--color", default="psychedelic", show_default=True, help="Color mode.")
@click.option("--tile-size", default=128, show_default=True, help="Pattern tile size.")
@click.option("--seed", default=42, show_default=True, help="Random seed.")
@click.option("--jobs", "-j", default=1, show_default=True, help="Parallel workers.")
@click.option("--glob", default="*.png", show_default=True, help="File glob pattern.")
@click.option(
    "--format",
    default="png",
    show_default=True,
    type=click.Choice(["png", "tiff", "jpg"], case_sensitive=False),
    help="Output format.",
)
@click.option("--json-log", default=None, help="Write JSON log to this path.")
@click.option("--verbose", "-v", is_flag=True, default=False)
def batch(
    input_dir,
    output_dir,
    mode,
    preset,
    pattern,
    color,
    tile_size,
    seed,
    jobs,
    glob,
    format,
    json_log,
    verbose,
):
    """Batch-process a folder of depth maps into stereograms."""
    import json

    from depthforge.core.depth_prep import prep_depth
    from depthforge.core.presets import get_preset
    from depthforge.core.synthesizer import synthesize

    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob(glob))
    if not files:
        raise click.ClickException(f"No files matching '{glob}' in {input_dir}")

    click.echo(f"Batch: {len(files)} files  mode={mode}  preset={preset}  jobs={jobs}")

    p = get_preset(preset)
    pat = _build_pattern(pattern, tile_size, color, seed)
    log = []
    done = 0
    errors = 0

    t_start = time.time()

    for f in files:
        t0 = time.time()
        try:
            raw = _load_depth(str(f))
            dep = prep_depth(raw, p.depth_params)

            if mode in ("sirds", "texture"):
                result = synthesize(dep, pat, p.stereo_params)
            elif mode == "hidden":
                from depthforge.core.hidden_image import (
                    HiddenImageParams,
                    encode_hidden_image,
                    shape_to_mask,
                )

                H, W = dep.shape
                msk = shape_to_mask("star", W, H)
                result = encode_hidden_image(pat, msk, HiddenImageParams())
            else:
                result = synthesize(dep, pat, p.stereo_params)

            out_name = f.stem + f".{format}"
            out_path = out_dir / out_name
            _save(result, str(out_path), dpi=p.output_dpi)

            elapsed = time.time() - t0
            done += 1
            log.append(
                {"file": str(f), "output": str(out_path), "status": "ok", "elapsed": elapsed}
            )
            if verbose:
                click.echo(f"  ✓ {f.name} → {out_name}  ({elapsed:.1f}s)")
            else:
                click.echo(f"\r  {done}/{len(files)}", nl=False)

        except Exception as e:
            errors += 1
            log.append({"file": str(f), "status": "error", "error": str(e)})
            click.echo(f"\n  ✗ {f.name}: {e}", err=True)

    total = time.time() - t_start
    click.echo(f"\n✓ Batch complete: {done} ok, {errors} errors  ({total:.1f}s total)")

    if json_log:
        Path(json_log).write_text(json.dumps(log, indent=2))
        click.echo(f"  Log: {json_log}")


# ---------------------------------------------------------------------------
# analyze command
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--depth", "-d", required=True, help="Depth map image.")
@click.option("--frame-width", default=1920, show_default=True, help="Target output width.")
@click.option("--frame-height", default=1080, show_default=True, help="Target output height.")
@click.option("--depth-factor", default=0.35, show_default=True, help="Planned depth_factor.")
@click.option("--max-parallax", default=1 / 30, show_default=True, help="Comfort limit (fraction).")
@click.option("--output-overlay", default=None, help="Save violation overlay image.")
@click.option("--output-heatmap", default=None, help="Save parallax heatmap image.")
@click.option("--output-bands", default=None, help="Save depth band preview image.")
@click.option("--json", "json_out", default=None, help="Save analysis as JSON.")
def analyze(
    depth,
    frame_width,
    frame_height,
    depth_factor,
    max_parallax,
    output_overlay,
    output_heatmap,
    output_bands,
    json_out,
):
    """Analyze a depth map for viewer comfort and produce a report."""
    import json

    from depthforge.core.comfort import (
        ComfortAnalyzer,
        ComfortAnalyzerParams,
        estimate_vergence_strain,
    )
    from depthforge.core.qc import (
        QCParams,
        depth_band_preview,
        parallax_heatmap,
    )

    raw = _load_depth(depth)
    H, W = raw.shape

    params = ComfortAnalyzerParams(
        frame_width=frame_width,
        frame_height=frame_height,
        depth_factor=depth_factor,
        max_parallax_fraction=max_parallax,
    )
    analyzer = ComfortAnalyzer(params)
    report = analyzer.analyze(raw)
    strain = estimate_vergence_strain(raw, depth_factor=depth_factor, frame_width=frame_width)

    click.echo(f"\nDepth map: {depth}  ({W}×{H})")
    click.echo(report.summary())
    click.echo(
        f"\nVergence strain: {strain['strain_rating']}  "
        f"(max {strain['max_vergence_deg']:.2f}°, "
        f"rapid-change fraction: {strain['rapid_change_fraction']:.2%})"
    )

    qp = QCParams(
        frame_width=frame_width, depth_factor=depth_factor, max_parallax_fraction=max_parallax
    )

    if output_overlay:
        overlay = analyzer.violation_overlay(raw, report)
        _save(overlay, output_overlay)
        click.echo(f"  Overlay: {output_overlay}")

    if output_heatmap:
        hm = parallax_heatmap(raw, qp, annotate=False)
        _save(hm, output_heatmap)
        click.echo(f"  Heatmap: {output_heatmap}")

    if output_bands:
        bands = depth_band_preview(raw, n_bands=8)
        _save(bands, output_bands)
        click.echo(f"  Depth bands: {output_bands}")

    if json_out:
        data = {
            "depth_file": depth,
            "frame_width": frame_width,
            "frame_height": frame_height,
            "depth_factor": depth_factor,
            "overall_score": report.overall_score,
            "passed": report.passed,
            "max_parallax_px": report.max_parallax_px,
            "max_vergence_deg": report.max_vergence_deg,
            "pct_uncomfortable": report.pct_uncomfortable,
            "violations": report.violations,
            "advice": report.advice,
            "vergence_strain": strain,
        }
        Path(json_out).write_text(json.dumps(data, indent=2))
        click.echo(f"  JSON report: {json_out}")


# ---------------------------------------------------------------------------
# presets command
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--list", "do_list", is_flag=True, default=False, help="List all available presets.")
@click.option("--show", default=None, help="Show details for a named preset.")
@click.option("--export", default=None, help="Export preset to JSON file.")
def presets(do_list, show, export):
    """List and inspect available stereogram presets."""
    from depthforge.core.presets import get_preset, list_presets

    if do_list or (not show and not export):
        names = list_presets()
        click.echo("\nAvailable presets:")
        for name in names:
            p = get_preset(name)
            click.echo(
                f"  {name:<12}  df={p.stereo_params.depth_factor:.2f}  "
                f"parallax≤{p.stereo_params.max_parallax_fraction:.3f}  "
                f"dpi={p.output_dpi}  [{p.display_name}]"
            )

    if show:
        p = get_preset(show)
        click.echo(f"\nPreset: {p.name}")
        click.echo(f"  Display name : {p.display_name}")
        click.echo(f"  Description  : {p.description}")
        click.echo(f"  Color space  : {p.color_space}")
        click.echo(f"  Output DPI   : {p.output_dpi}")
        click.echo(f"  Target width : {p.target_width}px")
        click.echo("\n  Stereo params:")
        sp = p.stereo_params
        click.echo(f"    depth_factor          = {sp.depth_factor}")
        click.echo(f"    max_parallax_fraction = {sp.max_parallax_fraction:.4f}")
        click.echo(f"    safe_mode             = {sp.safe_mode}")
        click.echo(f"    oversample            = {sp.oversample}")
        click.echo("\n  Notes:")
        click.echo(f"    {p.notes}")

    if export:
        p = get_preset(show or "medium")
        p.to_json(export)
        click.echo(f"  Exported preset to: {export}")


# ---------------------------------------------------------------------------
# estimate-depth command (AI depth, Phase 3 — stub with graceful fallback)
# ---------------------------------------------------------------------------


@cli.command("estimate-depth")
@click.option("--source", "-s", required=True, help="Input colour image.")
@click.option("--output", "-o", required=True, help="Output depth map path.")
@click.option(
    "--model",
    default="midas",
    show_default=True,
    type=click.Choice(["midas", "zoedepth"], case_sensitive=False),
    help="Depth estimation model.",
)
@click.option("--invert", is_flag=True, default=False, help="Invert output depth (near/far swap).")
@click.option("--verbose", is_flag=True, default=False)
def estimate_depth(source, output, model, invert, verbose):
    """Estimate a depth map from a colour image using AI (requires PyTorch)."""
    from depthforge.depth_models import DepthEstimationError, get_depth_estimator

    _maybe_verbose(verbose, f"Loading: {source}")
    src_img = Image.open(source).convert("RGB")

    try:
        estimator = get_depth_estimator(model)
        _maybe_verbose(verbose, f"Running {model} depth estimation...")
        depth = estimator.estimate(np.array(src_img))

        if invert:
            depth = 1.0 - depth

        out_img = Image.fromarray((depth * 255).astype(np.uint8))
        out_img.save(output)
        click.echo(f"✓ Depth map saved: {output}")

    except DepthEstimationError as e:
        raise click.ClickException(str(e))


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _maybe_verbose(verbose: bool, msg: str) -> None:
    if verbose:
        click.echo(f"  {msg}")


if __name__ == "__main__":
    cli()
