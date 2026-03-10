"""depthforge.core — pure algorithm modules, zero UI dependencies."""

from depthforge.core.adaptive_dots import (
    AdaptiveDotParams,
    complexity_from_depth,
    generate_adaptive_dots,
)
from depthforge.core.anaglyph import (
    AnaglyphMode,
    AnaglyphParams,
    make_anaglyph,
    make_anaglyph_from_depth,
)
from depthforge.core.depth_prep import (
    DepthPrepParams,
    FalloffCurve,
    RegionMask,
    compute_vergence_map,
    detect_window_violations,
    normalise_depth,
    prep_depth,
)
from depthforge.core.hidden_image import (
    HiddenImageParams,
    encode_hidden_image,
    mask_to_depth,
    shape_to_mask,
    text_to_mask,
)
from depthforge.core.inpainting import (
    InpaintMethod,
    InpaintParams,
    inpaint_occlusion,
    register_ai_inpaint_callback,
)
from depthforge.core.pattern_gen import (
    ColorMode,
    GridStyle,
    PatternParams,
    PatternType,
    generate_pattern,
    load_tile,
    tile_to_frame,
)
from depthforge.core.stereo_pair import (
    StereoLayout,
    StereoPairParams,
    compose_side_by_side,
    make_stereo_pair,
)
from depthforge.core.synthesizer import StereoParams, load_depth_image, save_stereogram, synthesize
