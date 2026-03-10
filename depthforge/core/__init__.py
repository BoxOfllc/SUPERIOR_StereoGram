"""depthforge.core — pure algorithm modules, zero UI dependencies."""
from depthforge.core.synthesizer   import StereoParams, synthesize, load_depth_image, save_stereogram
from depthforge.core.depth_prep    import DepthPrepParams, FalloffCurve, RegionMask, prep_depth, normalise_depth, compute_vergence_map, detect_window_violations
from depthforge.core.pattern_gen   import PatternParams, PatternType, ColorMode, GridStyle, generate_pattern, load_tile, tile_to_frame
from depthforge.core.anaglyph      import AnaglyphMode, AnaglyphParams, make_anaglyph, make_anaglyph_from_depth
from depthforge.core.stereo_pair   import StereoPairParams, StereoLayout, make_stereo_pair, compose_side_by_side
from depthforge.core.hidden_image  import HiddenImageParams, encode_hidden_image, mask_to_depth, text_to_mask, shape_to_mask
from depthforge.core.adaptive_dots import AdaptiveDotParams, generate_adaptive_dots, complexity_from_depth
from depthforge.core.inpainting    import InpaintParams, InpaintMethod, inpaint_occlusion, register_ai_inpaint_callback
