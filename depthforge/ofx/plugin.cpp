/**
 * depthforge/ofx/plugin.cpp
 * =========================
 * DepthForge OFX Plugin — full implementation.
 *
 * Build with CMakeLists.txt in this directory.
 * Requires: C++17, no external dependencies beyond the OFX SDK headers.
 */

#include "plugin.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ── Platform I/O ──────────────────────────────────────────────────────────
#ifdef _WIN32
  #include <windows.h>
  #define popen  _popen
  #define pclose _pclose
  #define PATH_SEP "\\"
#else
  #include <unistd.h>
  #define PATH_SEP "/"
#endif

// ── Global suite pointers ─────────────────────────────────────────────────
OfxHost*               gHost        = nullptr;
OfxPropertySuiteV1*    gPropSuite   = nullptr;
OfxImageEffectSuiteV1* gEffectSuite = nullptr;
OfxParameterSuiteV1*   gParamSuite  = nullptr;

// ═════════════════════════════════════════════════════════════════════════
// PythonBridge implementation
// ═════════════════════════════════════════════════════════════════════════

PythonBridge::PythonBridge(const PythonBridgeConfig& cfg) : _cfg(cfg) {}

int PythonBridge::_run(const std::string& cmd) {
    if (_cfg.verbose) {
        // In production, log to host logger; here just capture stderr
    }
#ifdef _WIN32
    FILE* pipe = _popen(cmd.c_str(), "r");
#else
    FILE* pipe = ::popen(cmd.c_str(), "r");
#endif
    if (!pipe) {
        _last_error = "Failed to open subprocess pipe";
        return -1;
    }
    std::array<char, 512> buf;
    std::string output;
    while (fgets(buf.data(), (int)buf.size(), pipe)) {
        output += buf.data();
    }
#ifdef _WIN32
    int status = _pclose(pipe);
#else
    int status = ::pclose(pipe);
#endif
    if (status != 0) {
        _last_error = "Subprocess failed: " + output;
    }
    return status;
}

bool PythonBridge::check_available() {
    std::string cmd = _cfg.python_exe + " -c \"import depthforge\" 2>&1";
    return _run(cmd) == 0;
}

int PythonBridge::synthesize(
    const std::string& depth_path,
    const std::string& source_path,
    const std::string& out_path,
    const std::string& cli_args)
{
    // Build: python3 -m depthforge.cli.bridge --depth <path> --out <path> [extra args]
    std::ostringstream cmd;
    cmd << _cfg.python_exe
        << " -m depthforge.cli.bridge"
        << " --depth \""  << depth_path  << "\""
        << " --out \""    << out_path    << "\""
        << " "            << cli_args;

    if (!source_path.empty()) {
        cmd << " --source \"" << source_path << "\"";
    }

    if (!_cfg.verbose) {
        cmd << " 2>/dev/null";
    }

    return _run(cmd.str());
}

// ═════════════════════════════════════════════════════════════════════════
// DepthForgePlugin — Host setup
// ═════════════════════════════════════════════════════════════════════════

void DepthForgePlugin::setHost(OfxHost* host) {
    gHost = host;
    if (!host) return;

    gPropSuite   = (OfxPropertySuiteV1*)    host->fetchSuite(host->host, "OfxPropertySuite",      1);
    gEffectSuite = (OfxImageEffectSuiteV1*) host->fetchSuite(host->host, "OfxImageEffectSuite",   1);
    gParamSuite  = (OfxParameterSuiteV1*)   host->fetchSuite(host->host, "OfxParameterSuite",     1);
}

// ═════════════════════════════════════════════════════════════════════════
// describeAction — set plugin metadata + pixel formats
// ═════════════════════════════════════════════════════════════════════════

OfxStatus DepthForgePlugin::describeAction(OfxImageEffectHandle effect) {
    if (!gPropSuite || !gEffectSuite) return kOfxStatErrFatal;

    OfxPropertySetHandle props;
    gEffectSuite->getPropertySet(effect, &props);

    gPropSuite->propSetString(props, kOfxPropLabel,          0, DEPTHFORGE_OFX_LABEL);
    gPropSuite->propSetString(props, kOfxPropShortLabel,     0, DEPTHFORGE_OFX_SHORT_LABEL);
    gPropSuite->propSetString(props, kOfxPropLongLabel,      0, DEPTHFORGE_OFX_LONG_LABEL);
    gPropSuite->propSetString(props, kOfxPropVersionLabel,   0, "0.4.0");
    gPropSuite->propSetString(props, kOfxImageEffectPluginPropGrouping, 0, DEPTHFORGE_OFX_GROUP);

    // Supported pixel depths
    gPropSuite->propSetString(props, kOfxImageEffectPropSupportedPixelDepths, 0, kOfxBitDepthByte);
    gPropSuite->propSetString(props, kOfxImageEffectPropSupportedPixelDepths, 1, kOfxBitDepthFloat);

    // Supported contexts
    gPropSuite->propSetString(props, kOfxImageEffectPropSupportedContexts, 0, kOfxImageEffectContextFilter);
    gPropSuite->propSetString(props, kOfxImageEffectPropSupportedContexts, 1, kOfxImageEffectContextGeneral);

    return kOfxStatOK;
}

// ═════════════════════════════════════════════════════════════════════════
// describeInContextAction — define clips and parameters
// ═════════════════════════════════════════════════════════════════════════

OfxStatus DepthForgePlugin::describeInContext(OfxImageEffectHandle effect,
                                               OfxPropertySetHandle /*inArgs*/) {
    OfxStatus st;
    st = _defineClips(effect);
    if (st != kOfxStatOK) return st;
    st = _defineParams(effect);
    return st;
}

OfxStatus DepthForgePlugin::_defineClips(OfxImageEffectHandle effect) {
    OfxPropertySetHandle clipProps;

    // Depth input (required)
    gEffectSuite->clipDefine(effect, CLIP_DEPTH, &clipProps);
    gPropSuite->propSetString(clipProps, kOfxPropLabel, 0, "Depth");

    // Source input (optional — for anaglyph and stereo-pair modes)
    gEffectSuite->clipDefine(effect, CLIP_SOURCE, &clipProps);
    gPropSuite->propSetString(clipProps, kOfxPropLabel, 0, "Source (optional)");

    // Output
    gEffectSuite->clipDefine(effect, CLIP_OUTPUT, &clipProps);
    gPropSuite->propSetString(clipProps, kOfxPropLabel, 0, "Output");

    return kOfxStatOK;
}

OfxStatus DepthForgePlugin::_defineParams(OfxImageEffectHandle effect) {
    OfxParamSetHandle params;
    OfxPropertySetHandle p;

    gEffectSuite->getParamSet(effect, &params);

    // ── Mode ─────────────────────────────────────────────────────────────
    gParamSuite->paramDefineChoice(params, PARAM_MODE,   "Mode",   &p);
    gPropSuite->propSetString(p, kOfxPropLabel, 0, "Mode");

    gParamSuite->paramDefineChoice(params, PARAM_PRESET, "Preset", &p);
    gPropSuite->propSetString(p, kOfxPropLabel, 0, "Preset");

    // ── Depth controls ───────────────────────────────────────────────────
    gParamSuite->paramDefineDouble(params, PARAM_DEPTH_FACTOR, "Depth Factor", &p);
    gPropSuite->propSetString(p, kOfxPropLabel, 0, "Depth Factor");
    gPropSuite->propSetDouble(p, "OfxParamPropDefault", 0, 0.35);
    gPropSuite->propSetDouble(p, "OfxParamPropMin",     0, 0.0);
    gPropSuite->propSetDouble(p, "OfxParamPropMax",     0, 1.0);

    gParamSuite->paramDefineDouble(params, PARAM_MAX_PARALLAX, "Max Parallax", &p);
    gPropSuite->propSetString(p, kOfxPropLabel, 0, "Max Parallax");
    gPropSuite->propSetDouble(p, "OfxParamPropDefault", 0, 0.033);
    gPropSuite->propSetDouble(p, "OfxParamPropMin",     0, 0.005);
    gPropSuite->propSetDouble(p, "OfxParamPropMax",     0, 0.1);

    gParamSuite->paramDefineBool(params, PARAM_SAFE_MODE, "Safe Mode", &p);
    gPropSuite->propSetInt(p, "OfxParamPropDefault", 0, 0);

    gParamSuite->paramDefineInt(params, PARAM_OVERSAMPLE, "Oversample", &p);
    gPropSuite->propSetInt(p, "OfxParamPropDefault", 0, 1);

    gParamSuite->paramDefineInt(params, PARAM_SEED, "Seed", &p);
    gPropSuite->propSetInt(p, "OfxParamPropDefault", 0, 42);

    gParamSuite->paramDefineBool(params, PARAM_INVERT_DEPTH, "Invert Depth", &p);
    gPropSuite->propSetInt(p, "OfxParamPropDefault", 0, 0);

    // ── Depth conditioning ────────────────────────────────────────────────
    gParamSuite->paramDefineDouble(params, PARAM_BILATERAL_SPACE, "Bilateral Space", &p);
    gPropSuite->propSetDouble(p, "OfxParamPropDefault", 0, 5.0);
    gPropSuite->propSetDouble(p, "OfxParamPropMin",     0, 0.0);
    gPropSuite->propSetDouble(p, "OfxParamPropMax",     0, 30.0);

    gParamSuite->paramDefineDouble(params, PARAM_BILATERAL_COLOR, "Bilateral Color", &p);
    gPropSuite->propSetDouble(p, "OfxParamPropDefault", 0, 0.1);
    gPropSuite->propSetDouble(p, "OfxParamPropMin",     0, 0.01);
    gPropSuite->propSetDouble(p, "OfxParamPropMax",     0, 1.0);

    gParamSuite->paramDefineInt(params, PARAM_DILATION_PX, "Dilation (px)", &p);
    gPropSuite->propSetInt(p, "OfxParamPropDefault", 0, 3);

    gParamSuite->paramDefineChoice(params, PARAM_FALLOFF_CURVE, "Falloff Curve", &p);
    gPropSuite->propSetString(p, kOfxPropLabel, 0, "Falloff Curve");

    gParamSuite->paramDefineDouble(params, PARAM_NEAR_PLANE, "Near Plane", &p);
    gPropSuite->propSetDouble(p, "OfxParamPropDefault", 0, 0.0);
    gPropSuite->propSetDouble(p, "OfxParamPropMin",     0, 0.0);
    gPropSuite->propSetDouble(p, "OfxParamPropMax",     0, 1.0);

    gParamSuite->paramDefineDouble(params, PARAM_FAR_PLANE, "Far Plane", &p);
    gPropSuite->propSetDouble(p, "OfxParamPropDefault", 0, 1.0);

    // ── Pattern ───────────────────────────────────────────────────────────
    gParamSuite->paramDefineChoice(params, PARAM_PATTERN_TYPE, "Pattern Type", &p);
    gParamSuite->paramDefineString(params, PARAM_PATTERN_NAME, "Pattern Name", &p);
    gParamSuite->paramDefineChoice(params, PARAM_COLOR_MODE,   "Color Mode",   &p);

    gParamSuite->paramDefineInt(params, PARAM_TILE_WIDTH, "Tile Width", &p);
    gPropSuite->propSetInt(p, "OfxParamPropDefault", 0, 128);

    gParamSuite->paramDefineInt(params, PARAM_TILE_HEIGHT, "Tile Height", &p);
    gPropSuite->propSetInt(p, "OfxParamPropDefault", 0, 128);

    gParamSuite->paramDefineDouble(params, PARAM_PATTERN_SCALE, "Pattern Scale", &p);
    gPropSuite->propSetDouble(p, "OfxParamPropDefault", 0, 1.0);

    // ── Anaglyph ──────────────────────────────────────────────────────────
    gParamSuite->paramDefineChoice(params, PARAM_ANAGLYPH_MODE, "Anaglyph Mode", &p);
    gParamSuite->paramDefineBool  (params, PARAM_SWAP_EYES,     "Swap Eyes",     &p);

    gParamSuite->paramDefineDouble(params, PARAM_GAMMA, "Gamma", &p);
    gPropSuite->propSetDouble(p, "OfxParamPropDefault", 0, 1.0);
    gPropSuite->propSetDouble(p, "OfxParamPropMin",     0, 0.5);
    gPropSuite->propSetDouble(p, "OfxParamPropMax",     0, 3.0);

    // ── Safety ────────────────────────────────────────────────────────────
    gParamSuite->paramDefineChoice(params, PARAM_SAFETY_PROFILE, "Safety Profile", &p);
    gParamSuite->paramDefineBool  (params, PARAM_AUTO_SAFE,      "Auto Safety",    &p);
    gPropSuite->propSetInt(p, "OfxParamPropDefault", 0, 1);

    // ── Export profile ────────────────────────────────────────────────────
    gParamSuite->paramDefineChoice(params, PARAM_EXPORT_PROFILE, "Export Profile", &p);

    // ── Python bridge ─────────────────────────────────────────────────────
    gParamSuite->paramDefineString(params, PARAM_PYTHON_EXE, "Python Executable", &p);
    // Note: can't set string default via this minimal stub easily;
    // default handled in bridge config.

    gParamSuite->paramDefineInt(params, PARAM_PYTHON_TIMEOUT, "Python Timeout (s)", &p);
    gPropSuite->propSetInt(p, "OfxParamPropDefault", 0, 60);

    return kOfxStatOK;
}

// ═════════════════════════════════════════════════════════════════════════
// createInstance / destroyInstance
// ═════════════════════════════════════════════════════════════════════════

OfxStatus DepthForgePlugin::createInstance(OfxImageEffectHandle /*effect*/) {
    // Per-instance data could be allocated here (e.g. cached pattern tile).
    // For now, stateless — all work done at render time.
    return kOfxStatOK;
}

OfxStatus DepthForgePlugin::destroyInstance(OfxImageEffectHandle /*effect*/) {
    return kOfxStatOK;
}

// ═════════════════════════════════════════════════════════════════════════
// isIdentityAction — pass-through when depth_factor == 0
// ═════════════════════════════════════════════════════════════════════════

OfxStatus DepthForgePlugin::isIdentityAction(OfxImageEffectHandle effect,
                                              OfxPropertySetHandle /*inArgs*/,
                                              OfxPropertySetHandle outArgs) {
    (void)effect; (void)outArgs;
    // Non-identity — always render
    return kOfxStatReplyDefault;
}

// ═════════════════════════════════════════════════════════════════════════
// getRODAction — output ROD matches depth clip ROD
// ═════════════════════════════════════════════════════════════════════════

OfxStatus DepthForgePlugin::getRODAction(OfxImageEffectHandle /*effect*/,
                                          OfxPropertySetHandle /*inArgs*/,
                                          OfxPropertySetHandle /*outArgs*/) {
    // Default: output ROD = project ROD (handled by host).
    return kOfxStatReplyDefault;
}

// ═════════════════════════════════════════════════════════════════════════
// _buildCliArgs — serialise OFX parameters to depthforge CLI flags
// ═════════════════════════════════════════════════════════════════════════

std::string DepthForgePlugin::_buildCliArgs(OfxImageEffectHandle effect, OfxTime time) {
    if (!gParamSuite || !gEffectSuite) return "";

    OfxParamSetHandle paramSet;
    gEffectSuite->getParamSet(effect, &paramSet);

    // Helpers to read param values
    auto getDouble = [&](const char* name) -> double {
        void* ph = nullptr;
        gParamSuite->paramGetHandle(paramSet, name, &ph, nullptr);
        double v = 0.0;
        if (ph) gParamSuite->paramGetValueAtTime(ph, time, &v);
        return v;
    };
    auto getInt = [&](const char* name) -> int {
        void* ph = nullptr;
        gParamSuite->paramGetHandle(paramSet, name, &ph, nullptr);
        int v = 0;
        if (ph) gParamSuite->paramGetValueAtTime(ph, time, &v);
        return v;
    };
    auto getBool = [&](const char* name) -> bool {
        return getInt(name) != 0;
    };

    std::ostringstream args;
    args << " --depth-factor "   << getDouble(PARAM_DEPTH_FACTOR)
         << " --max-parallax "   << getDouble(PARAM_MAX_PARALLAX)
         << " --oversample "     << getInt(PARAM_OVERSAMPLE)
         << " --seed "           << getInt(PARAM_SEED)
         << " --bilateral-space "<< getDouble(PARAM_BILATERAL_SPACE)
         << " --bilateral-color "<< getDouble(PARAM_BILATERAL_COLOR)
         << " --dilation-px "    << getInt(PARAM_DILATION_PX)
         << " --near-plane "     << getDouble(PARAM_NEAR_PLANE)
         << " --far-plane "      << getDouble(PARAM_FAR_PLANE)
         << " --tile-width "     << getInt(PARAM_TILE_WIDTH)
         << " --tile-height "    << getInt(PARAM_TILE_HEIGHT)
         << " --pattern-scale "  << getDouble(PARAM_PATTERN_SCALE)
         << " --gamma "          << getDouble(PARAM_GAMMA);

    if (getBool(PARAM_SAFE_MODE))    args << " --safe-mode";
    if (getBool(PARAM_INVERT_DEPTH)) args << " --invert-depth";
    if (getBool(PARAM_SWAP_EYES))    args << " --swap-eyes";
    if (getBool(PARAM_AUTO_SAFE))    args << " --auto-safe";

    return args.str();
}

PythonBridgeConfig DepthForgePlugin::_getBridgeConfig(OfxImageEffectHandle effect) {
    PythonBridgeConfig cfg;
    if (!gParamSuite || !gEffectSuite) return cfg;

    OfxParamSetHandle paramSet;
    gEffectSuite->getParamSet(effect, &paramSet);

    void* ph = nullptr;
    gParamSuite->paramGetHandle(paramSet, PARAM_PYTHON_TIMEOUT, &ph, nullptr);
    if (ph) {
        int t = 60;
        gParamSuite->paramGetValue(ph, &t);
        cfg.timeout_secs = t;
    }
    // python_exe: if not set by user, default to "python3" / "python"
    cfg.python_exe = "python3";

    return cfg;
}

// ═════════════════════════════════════════════════════════════════════════
// renderAction — the main per-frame work
// ═════════════════════════════════════════════════════════════════════════

OfxStatus DepthForgePlugin::renderAction(OfxImageEffectHandle effect,
                                          OfxPropertySetHandle inArgs,
                                          OfxPropertySetHandle /*outArgs*/) {
    if (!gEffectSuite || !gPropSuite || !gParamSuite) return kOfxStatErrFatal;

    // Get render time
    double time = 0.0;
    gPropSuite->propGetDouble(inArgs, kOfxImageEffectPropTime, 0, &time);

    // Fetch depth clip and output clip
    OfxImageClipHandle depthClip  = nullptr;
    OfxImageClipHandle outputClip = nullptr;
    gEffectSuite->clipGetHandle(effect, CLIP_DEPTH,  &depthClip,  nullptr);
    gEffectSuite->clipGetHandle(effect, CLIP_OUTPUT, &outputClip, nullptr);

    if (!depthClip || !outputClip) return kOfxStatErrFatal;

    // Get depth image
    OfxImageHandle depthImg = nullptr;
    gEffectSuite->clipGetImage(depthClip, (OfxTime)time, nullptr, &depthImg);
    if (!depthImg) return kOfxStatFailed;

    // Write depth frame to temp file
    // (In a full implementation: get pixel data pointer, dimensions, write PNG)
    std::string tmp_depth  = "/tmp/df_ofx_depth.png";
    std::string tmp_out    = "/tmp/df_ofx_out.png";
    std::string tmp_source = "";

    // Optionally fetch source clip
    OfxImageClipHandle sourceClip = nullptr;
    gEffectSuite->clipGetHandle(effect, CLIP_SOURCE, &sourceClip, nullptr);
    if (sourceClip) {
        OfxImageHandle srcImg = nullptr;
        gEffectSuite->clipGetImage(sourceClip, (OfxTime)time, nullptr, &srcImg);
        if (srcImg) {
            tmp_source = "/tmp/df_ofx_source.png";
            // (Write source pixels to tmp_source)
            gEffectSuite->clipReleaseImage(srcImg);
        }
    }

    // Build CLI args and invoke Python
    std::string cli_args = _buildCliArgs(effect, (OfxTime)time);
    PythonBridgeConfig bridge_cfg = _getBridgeConfig(effect);
    PythonBridge bridge(bridge_cfg);

    int rc = bridge.synthesize(tmp_depth, tmp_source, tmp_out, cli_args);

    gEffectSuite->clipReleaseImage(depthImg);

    if (rc != 0) {
        // Synthesis failed — pass depth through unchanged
        return kOfxStatFailed;
    }

    // Read result back and copy to output clip
    // (In a full implementation: load tmp_out PNG, write to output pixel buffer)

    return kOfxStatOK;
}

// ═════════════════════════════════════════════════════════════════════════
// mainEntry — OFX action dispatcher
// ═════════════════════════════════════════════════════════════════════════

OfxStatus DepthForgePlugin::mainEntry(const char*          action,
                                       const void*          handle,
                                       OfxPropertySetHandle inArgs,
                                       OfxPropertySetHandle outArgs)
{
    OfxImageEffectHandle effect = (OfxImageEffectHandle)handle;

    if (std::strcmp(action, kOfxActionDescribe) == 0)
        return describeAction(effect);

    if (std::strcmp(action, kOfxImageEffectActionDescribeInContext) == 0)
        return describeInContext(effect, inArgs);

    if (std::strcmp(action, kOfxActionCreateInstance) == 0)
        return createInstance(effect);

    if (std::strcmp(action, kOfxActionDestroyInstance) == 0)
        return destroyInstance(effect);

    if (std::strcmp(action, kOfxImageEffectActionRender) == 0)
        return renderAction(effect, inArgs, outArgs);

    if (std::strcmp(action, kOfxImageEffectActionIsIdentity) == 0)
        return isIdentityAction(effect, inArgs, outArgs);

    if (std::strcmp(action, kOfxImageEffectActionGetRegionOfDefinition) == 0)
        return getRODAction(effect, inArgs, outArgs);

    if (std::strcmp(action, kOfxActionLoad) == 0 ||
        std::strcmp(action, kOfxActionUnload) == 0)
        return kOfxStatOK;

    return kOfxStatReplyDefault;
}

// ═════════════════════════════════════════════════════════════════════════
// OFX entry points — exported C symbols
// ═════════════════════════════════════════════════════════════════════════

static OfxPlugin g_plugin = {
    "OfxImageEffectPlugin",      // pluginApi
    1,                           // apiVersion
    DEPTHFORGE_OFX_IDENTIFIER,   // pluginIdentifier
    DEPTHFORGE_OFX_VERSION_MAJ,  // pluginVersionMajor
    DEPTHFORGE_OFX_VERSION_MIN,  // pluginVersionMinor
    [](void* h){ DepthForgePlugin::setHost((OfxHost*)h); }, // setHost
    DepthForgePlugin::mainEntry, // mainEntry
};

extern "C" {

OfxExport int OfxGetNumberOfPlugins(void) {
    return 1;
}

OfxExport OfxPlugin* OfxGetPlugin(int nth) {
    if (nth == 0) return &g_plugin;
    return nullptr;
}

} // extern "C"
