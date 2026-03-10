/**
 * depthforge/ofx/plugin.h
 * =======================
 * DepthForge OFX plugin — class declarations and parameter identifiers.
 *
 * Architecture
 * ------------
 * The C++ plugin implements the full OFX image-effect lifecycle.
 * For the heavy Python synthesis work it spawns the DepthForge Python
 * CLI via subprocess and passes frame data through a named pipe or temp
 * file.  This keeps the plugin distributable without embedding a Python
 * interpreter, and lets the Python core be updated independently.
 *
 * Supported hosts
 * ---------------
 *   Adobe After Effects (OFX bridge required, e.g. Continuum)
 *   Adobe Premiere Pro
 *   DaVinci Resolve (Fusion)
 *   HitFilm
 *   Natron
 *   Nuke (also has native Python gizmo — see depthforge/nuke/)
 *   Vegas Pro
 *
 * Build
 * -----
 *   mkdir build && cd build
 *   cmake .. -DCMAKE_BUILD_TYPE=Release
 *   cmake --build . --target depthforge_ofx
 *
 * The output is a shared library (.ofx.bundle on macOS/Linux, .ofx on
 * Windows) placed in the standard OFX plugin directory.
 */

#pragma once

#include "ofx_minimal.h"
#include <string>
#include <memory>
#include <map>

// ── Plugin identity ───────────────────────────────────────────────────────

#define DEPTHFORGE_OFX_IDENTIFIER   "com.depthforge.Stereogram"
#define DEPTHFORGE_OFX_VERSION_MAJ  0
#define DEPTHFORGE_OFX_VERSION_MIN  4
#define DEPTHFORGE_OFX_LABEL        "DepthForge Stereogram"
#define DEPTHFORGE_OFX_SHORT_LABEL  "DF Stereo"
#define DEPTHFORGE_OFX_LONG_LABEL   "DepthForge Professional Stereogram Synthesis"
#define DEPTHFORGE_OFX_GROUP        "DepthForge"

// ── Parameter identifiers ─────────────────────────────────────────────────

// Synthesis mode
#define PARAM_MODE              "df_mode"
#define PARAM_PRESET            "df_preset"

// Depth
#define PARAM_DEPTH_FACTOR      "df_depth_factor"
#define PARAM_MAX_PARALLAX      "df_max_parallax"
#define PARAM_SAFE_MODE         "df_safe_mode"
#define PARAM_OVERSAMPLE        "df_oversample"
#define PARAM_SEED              "df_seed"
#define PARAM_INVERT_DEPTH      "df_invert_depth"

// Depth conditioning
#define PARAM_BILATERAL_SPACE   "df_bilateral_space"
#define PARAM_BILATERAL_COLOR   "df_bilateral_color"
#define PARAM_DILATION_PX       "df_dilation_px"
#define PARAM_FALLOFF_CURVE     "df_falloff_curve"
#define PARAM_NEAR_PLANE        "df_near_plane"
#define PARAM_FAR_PLANE         "df_far_plane"

// Pattern
#define PARAM_PATTERN_TYPE      "df_pattern_type"
#define PARAM_PATTERN_NAME      "df_pattern_name"
#define PARAM_COLOR_MODE        "df_color_mode"
#define PARAM_TILE_WIDTH        "df_tile_width"
#define PARAM_TILE_HEIGHT       "df_tile_height"
#define PARAM_PATTERN_SCALE     "df_pattern_scale"

// Anaglyph
#define PARAM_ANAGLYPH_MODE     "df_anaglyph_mode"
#define PARAM_SWAP_EYES         "df_swap_eyes"
#define PARAM_GAMMA             "df_gamma"

// Safety
#define PARAM_SAFETY_PROFILE    "df_safety_profile"
#define PARAM_AUTO_SAFE         "df_auto_safe"

// Export
#define PARAM_EXPORT_PROFILE    "df_export_profile"

// Python bridge
#define PARAM_PYTHON_EXE        "df_python_exe"
#define PARAM_PYTHON_TIMEOUT    "df_python_timeout"

// ── Clip names ────────────────────────────────────────────────────────────

#define CLIP_OUTPUT             "Output"
#define CLIP_DEPTH              "Depth"
#define CLIP_SOURCE             "Source"     // optional — for anaglyph/stereo

// ── Global host pointer (set on OfxSetHost) ───────────────────────────────

extern OfxHost*               gHost;
extern OfxPropertySuiteV1*    gPropSuite;
extern OfxImageEffectSuiteV1* gEffectSuite;
extern OfxParameterSuiteV1*   gParamSuite;

// ── Python bridge ─────────────────────────────────────────────────────────

/**
 * PythonBridge — serialise OFX frame data to disk, invoke the
 * DepthForge CLI, and read back the result.
 */
struct PythonBridgeConfig {
    std::string python_exe    = "python3";
    int         timeout_secs  = 60;
    std::string tmp_dir       = "/tmp";
    bool        verbose       = false;
};

class PythonBridge {
public:
    explicit PythonBridge(const PythonBridgeConfig& cfg);

    /**
     * Invoke DepthForge synthesis.
     *
     * @param depth_path    Path to the depth frame (PNG/EXR).
     * @param source_path   Path to the source frame, or empty string.
     * @param out_path      Path to write the stereogram result.
     * @param cli_args      Extra CLI arguments (pre-built argument string).
     * @return              0 on success, non-zero on failure.
     */
    int synthesize(
        const std::string& depth_path,
        const std::string& source_path,
        const std::string& out_path,
        const std::string& cli_args
    );

    /** Check that the Python executable and depthforge are available. */
    bool check_available();

    std::string last_error() const { return _last_error; }

private:
    PythonBridgeConfig _cfg;
    std::string        _last_error;

    int _run(const std::string& cmd);
};

// ── DepthForgePlugin ──────────────────────────────────────────────────────

class DepthForgePlugin {
public:
    DepthForgePlugin() = default;
    ~DepthForgePlugin() = default;

    // OFX lifecycle actions
    static OfxStatus describeAction       (OfxImageEffectHandle effect);
    static OfxStatus describeInContext    (OfxImageEffectHandle effect,
                                           OfxPropertySetHandle inArgs);
    static OfxStatus createInstance       (OfxImageEffectHandle effect);
    static OfxStatus destroyInstance      (OfxImageEffectHandle effect);
    static OfxStatus renderAction         (OfxImageEffectHandle effect,
                                           OfxPropertySetHandle inArgs,
                                           OfxPropertySetHandle outArgs);
    static OfxStatus isIdentityAction     (OfxImageEffectHandle effect,
                                           OfxPropertySetHandle inArgs,
                                           OfxPropertySetHandle outArgs);
    static OfxStatus getRODAction         (OfxImageEffectHandle effect,
                                           OfxPropertySetHandle inArgs,
                                           OfxPropertySetHandle outArgs);

    // Main dispatch
    static OfxStatus mainEntry(const char*          action,
                                const void*          handle,
                                OfxPropertySetHandle inArgs,
                                OfxPropertySetHandle outArgs);

    // Host setup
    static void setHost(OfxHost* host);

private:
    static OfxStatus _defineParams(OfxImageEffectHandle effect);
    static OfxStatus _defineClips (OfxImageEffectHandle effect);
    static std::string _buildCliArgs(OfxImageEffectHandle effect, OfxTime time);
    static PythonBridgeConfig _getBridgeConfig(OfxImageEffectHandle effect);
};
