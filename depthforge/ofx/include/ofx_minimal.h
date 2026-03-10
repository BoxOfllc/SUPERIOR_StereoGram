/**
 * ofx_minimal.h — Minimal OFX API stubs for DepthForge plugin scaffold.
 *
 * In a real build, replace this with the official OpenFX SDK headers from:
 *   https://github.com/AcademySoftwareFoundation/openfx
 *
 * This stub defines only the symbols needed to compile the DepthForge
 * plugin entry points and type-check the bridge code.
 */

#pragma once
#include <stddef.h>

#ifdef _WIN32
  #define OfxExport __declspec(dllexport)
#else
  #define OfxExport __attribute__((visibility("default")))
#endif

// ── Core types ───────────────────────────────────────────────────────────
typedef void*       OfxPropertySetHandle;
typedef void*       OfxParamSetHandle;
typedef void*       OfxImageEffectHandle;
typedef void*       OfxImageClipHandle;
typedef void*       OfxImageHandle;
typedef long long   OfxTime;
typedef int         OfxStatus;

#define kOfxStatOK                    0
#define kOfxStatFailed                1
#define kOfxStatErrFatal              2
#define kOfxStatErrUnknown            3
#define kOfxStatErrMissingHostFeature 4
#define kOfxStatErrUnsupported        5
#define kOfxStatErrExists             6
#define kOfxStatErrFormat             7
#define kOfxStatErrMemory             8
#define kOfxStatErrValue              9
#define kOfxStatReplyYes             10
#define kOfxStatReplyNo              11
#define kOfxStatReplyDefault         12

// ── Action strings ───────────────────────────────────────────────────────
#define kOfxActionLoad              "OfxActionLoad"
#define kOfxActionUnload            "OfxActionUnload"
#define kOfxActionDescribe          "OfxActionDescribe"
#define kOfxActionCreateInstance    "OfxActionCreateInstance"
#define kOfxActionDestroyInstance   "OfxActionDestroyInstance"
#define kOfxImageEffectActionDescribeInContext  "OfxImageEffectActionDescribeInContext"
#define kOfxImageEffectActionRender            "OfxImageEffectActionRender"
#define kOfxImageEffectActionIsIdentity        "OfxImageEffectActionIsIdentity"
#define kOfxImageEffectActionGetRegionOfDefinition "OfxImageEffectActionGetRegionOfDefinition"
#define kOfxImageEffectActionGetRegionsOfInterest  "OfxImageEffectActionGetRegionsOfInterest"

// ── Context strings ──────────────────────────────────────────────────────
#define kOfxImageEffectContextFilter    "OfxImageEffectContextFilter"
#define kOfxImageEffectContextGeneral   "OfxImageEffectContextGeneral"

// ── Component and bit-depth strings ──────────────────────────────────────
#define kOfxBitDepthByte    "OfxBitDepthByte"
#define kOfxBitDepthShort   "OfxBitDepthShort"
#define kOfxBitDepthFloat   "OfxBitDepthFloat"
#define kOfxImageComponentRGBA      "OfxImageComponentRGBA"
#define kOfxImageComponentRGB       "OfxImageComponentRGB"
#define kOfxImageComponentAlpha     "OfxImageComponentAlpha"

// ── Property keys ────────────────────────────────────────────────────────
#define kOfxPluginPropFilePath          "OfxPluginPropFilePath"
#define kOfxImageEffectPropSupportedPixelDepths "OfxImageEffectPropSupportedPixelDepths"
#define kOfxImageEffectPropSupportedContexts    "OfxImageEffectPropSupportedContexts"
#define kOfxPropLabel                   "OfxPropLabel"
#define kOfxPropShortLabel              "OfxPropShortLabel"
#define kOfxPropLongLabel               "OfxPropLongLabel"
#define kOfxPropVersion                 "OfxPropVersion"
#define kOfxPropVersionLabel            "OfxPropVersionLabel"
#define kOfxImageEffectPluginPropGrouping  "OfxImageEffectPluginPropGrouping"
#define kOfxImageEffectPropRenderWindow    "OfxImageEffectPropRenderWindow"
#define kOfxImageEffectPropFrameRange      "OfxImageEffectPropFrameRange"
#define kOfxImageEffectPropTime            "OfxImageEffectPropTime"
#define kOfxImageEffectPropFieldToRender   "OfxImageEffectPropFieldToRender"
#define kOfxImagePropData                  "OfxImagePropData"
#define kOfxImagePropBounds                "OfxImagePropBounds"
#define kOfxImagePropRowBytes              "OfxImagePropRowBytes"
#define kOfxImagePropPixelAspectRatio      "OfxImagePropPixelAspectRatio"
#define kOfxImageEffectPropProjectSize     "OfxImageEffectPropProjectSize"

// ── Host suite structs (minimal — enough to compile, not to link) ─────────
typedef struct OfxPropertySuiteV1 {
    OfxStatus (*propSetString)(OfxPropertySetHandle, const char*, int, const char*);
    OfxStatus (*propSetDouble)(OfxPropertySetHandle, const char*, int, double);
    OfxStatus (*propSetInt)   (OfxPropertySetHandle, const char*, int, int);
    OfxStatus (*propGetString)(OfxPropertySetHandle, const char*, int, char**);
    OfxStatus (*propGetDouble)(OfxPropertySetHandle, const char*, int, double*);
    OfxStatus (*propGetInt)   (OfxPropertySetHandle, const char*, int, int*);
    OfxStatus (*propGetIntN)  (OfxPropertySetHandle, const char*, int, int*, int);
} OfxPropertySuiteV1;

typedef struct OfxImageEffectSuiteV1 {
    OfxStatus (*getPropertySet)    (OfxImageEffectHandle, OfxPropertySetHandle*);
    OfxStatus (*getParamSet)       (OfxImageEffectHandle, OfxParamSetHandle*);
    OfxStatus (*clipDefine)        (OfxImageEffectHandle, const char*, OfxPropertySetHandle*);
    OfxStatus (*clipGetHandle)     (OfxImageEffectHandle, const char*, OfxImageClipHandle*, OfxPropertySetHandle*);
    OfxStatus (*clipGetImage)      (OfxImageClipHandle, OfxTime, void*, OfxImageHandle*);
    OfxStatus (*clipReleaseImage)  (OfxImageHandle);
    OfxStatus (*clipGetPropertySet)(OfxImageClipHandle, OfxPropertySetHandle*);
    OfxStatus (*getImage)          (OfxImageEffectHandle, const char*, OfxTime, OfxImageHandle*);
    OfxStatus (*releaseImage)      (OfxImageHandle);
    OfxStatus (*abort)             (OfxImageEffectHandle);
} OfxImageEffectSuiteV1;

typedef struct OfxParameterSuiteV1 {
    OfxStatus (*paramDefineDouble)(OfxParamSetHandle, const char*, const char*, OfxPropertySetHandle*);
    OfxStatus (*paramDefineInt)   (OfxParamSetHandle, const char*, const char*, OfxPropertySetHandle*);
    OfxStatus (*paramDefineChoice)(OfxParamSetHandle, const char*, const char*, OfxPropertySetHandle*);
    OfxStatus (*paramDefineString)(OfxParamSetHandle, const char*, const char*, OfxPropertySetHandle*);
    OfxStatus (*paramDefineBool)  (OfxParamSetHandle, const char*, const char*, OfxPropertySetHandle*);
    OfxStatus (*paramGetHandle)   (OfxParamSetHandle, const char*, void**, OfxPropertySetHandle*);
    OfxStatus (*paramGetValueAtTime)(void*, OfxTime, ...);
    OfxStatus (*paramGetValue)    (void*, ...);
    OfxStatus (*paramSetValue)    (void*, ...);
} OfxParameterSuiteV1;

// ── Plugin struct ─────────────────────────────────────────────────────────
typedef struct OfxPlugin {
    const char*   pluginApi;
    int           apiVersion;
    const char*   pluginIdentifier;
    unsigned int  pluginVersionMajor;
    unsigned int  pluginVersionMinor;
    void          (*setHost)(void* host);
    OfxStatus     (*mainEntry)(const char* action, const void* handle,
                               OfxPropertySetHandle inArgs,
                               OfxPropertySetHandle outArgs);
} OfxPlugin;

// ── Host fetch type ───────────────────────────────────────────────────────
typedef struct OfxHost {
    OfxPropertySetHandle host;
    void* (*fetchSuite)(OfxPropertySetHandle host, const char* suiteName, int suiteVersion);
} OfxHost;

// ── Entry point type ──────────────────────────────────────────────────────
typedef int (*OfxGetNumberOfPluginsFunc)(void);
typedef OfxPlugin* (*OfxGetPluginFunc)(int nth);
