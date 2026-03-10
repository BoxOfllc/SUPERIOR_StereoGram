cmake_minimum_required(VERSION 3.16)
project(depthforge_ofx VERSION 0.4.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Options ──────────────────────────────────────────────────────────────

option(DF_USE_SYSTEM_OFX
    "Use system-installed OpenFX SDK headers instead of bundled stubs"
    OFF)

option(DF_INSTALL_TO_OFX_DIR
    "Install the plugin bundle to the standard OFX plugin directory"
    OFF)

# ── OFX SDK headers ──────────────────────────────────────────────────────

if(DF_USE_SYSTEM_OFX)
    find_path(OFX_INCLUDE_DIR NAMES ofxImageEffect.h
        HINTS
            $ENV{OFX_SDK}/include
            /usr/local/include/ofx
            /usr/include/ofx
        REQUIRED)
    message(STATUS "Using system OFX headers: ${OFX_INCLUDE_DIR}")
else()
    # Bundled minimal stubs (sufficient for compilation and unit-testing)
    set(OFX_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
    message(STATUS "Using bundled OFX stubs: ${OFX_INCLUDE_DIR}")
endif()

# ── Target ───────────────────────────────────────────────────────────────

add_library(depthforge_ofx SHARED
    plugin.cpp
)

target_include_directories(depthforge_ofx PRIVATE
    ${OFX_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Suppress unused-parameter warnings from OFX SDK headers
target_compile_options(depthforge_ofx PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
        -Wall -Wextra -Wno-unused-parameter>
    $<$<CXX_COMPILER_ID:MSVC>:
        /W3 /wd4100>
)

# ── Output name / suffix ─────────────────────────────────────────────────

# OFX bundles use .ofx extension on all platforms (inside .ofx.bundle on macOS)
set_target_properties(depthforge_ofx PROPERTIES
    OUTPUT_NAME  "DepthForge"
    PREFIX       ""
    SUFFIX       ".ofx"
)

# Position-independent code (required for shared libs on Linux)
set_target_properties(depthforge_ofx PROPERTIES
    POSITION_INDEPENDENT_CODE ON)

# macOS: don't add 'lib' prefix; set rpath
if(APPLE)
    set_target_properties(depthforge_ofx PROPERTIES
        MACOSX_RPATH ON)
endif()

# ── Bundle layout ────────────────────────────────────────────────────────
# Standard OFX bundle structure:
#   DepthForge.ofx.bundle/
#     Contents/
#       Info.plist          (macOS only)
#       Linux-x86-64/       (or MacOS, Win64)
#         DepthForge.ofx
#
# We set this up as a post-build step.

if(APPLE)
    set(OFX_ARCH_DIR "MacOS")
elseif(WIN32)
    set(OFX_ARCH_DIR "Win64")
else()
    set(OFX_ARCH_DIR "Linux-x86-64")
endif()

set(BUNDLE_DIR "${CMAKE_BINARY_DIR}/DepthForge.ofx.bundle/Contents/${OFX_ARCH_DIR}")

add_custom_command(TARGET depthforge_ofx POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${BUNDLE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy
        "$<TARGET_FILE:depthforge_ofx>"
        "${BUNDLE_DIR}/DepthForge.ofx"
    COMMENT "Assembling OFX bundle → ${BUNDLE_DIR}"
)

# macOS Info.plist
if(APPLE)
    set(PLIST_DIR "${CMAKE_BINARY_DIR}/DepthForge.ofx.bundle/Contents")
    add_custom_command(TARGET depthforge_ofx POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\
<plist version=\"1.0\"><dict>\
<key>CFBundleIdentifier</key><string>com.depthforge.Stereogram</string>\
<key>CFBundleName</key><string>DepthForge</string>\
<key>CFBundleVersion</key><string>0.4.0</string>\
</dict></plist>" > "${PLIST_DIR}/Info.plist"
        COMMENT "Writing Info.plist"
    )
endif()

# ── Install ───────────────────────────────────────────────────────────────

if(DF_INSTALL_TO_OFX_DIR)
    if(APPLE)
        set(OFX_PLUGIN_DIR "/Library/OFX/Plugins")
    elseif(WIN32)
        set(OFX_PLUGIN_DIR "C:/Program Files/Common Files/OFX/Plugins")
    else()
        set(OFX_PLUGIN_DIR "/usr/OFX/Plugins")
    endif()

    install(DIRECTORY "${CMAKE_BINARY_DIR}/DepthForge.ofx.bundle"
        DESTINATION "${OFX_PLUGIN_DIR}"
    )
    message(STATUS "Plugin will install to: ${OFX_PLUGIN_DIR}")
else()
    install(DIRECTORY "${CMAKE_BINARY_DIR}/DepthForge.ofx.bundle"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/share/depthforge/ofx"
    )
endif()

# ── Summary ───────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "DepthForge OFX Plugin configuration:")
message(STATUS "  Version       : 0.4.0")
message(STATUS "  OFX headers   : ${OFX_INCLUDE_DIR}")
message(STATUS "  Arch dir      : ${OFX_ARCH_DIR}")
message(STATUS "  Bundle dir    : ${BUNDLE_DIR}")
message(STATUS "")
message(STATUS "Build:")
message(STATUS "  mkdir build && cd build")
message(STATUS "  cmake .. -DCMAKE_BUILD_TYPE=Release")
message(STATUS "  cmake --build . --target depthforge_ofx")
message(STATUS "")
message(STATUS "To use system OFX SDK:")
message(STATUS "  cmake .. -DDF_USE_SYSTEM_OFX=ON -DOFX_SDK=/path/to/openfx")
message(STATUS "")
