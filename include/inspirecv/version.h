#ifndef INSPIRECV_VERSION_H
#define INSPIRECV_VERSION_H

#ifndef INSPIRECV_API
#define INSPIRECV_API
#endif

namespace inspirecv {

const char* INSPIRECV_API GetVersion();

const char* INSPIRECV_API GetCVBackend();

namespace task {

struct BuildInfo {
    const char* system_name;          // e.g. Darwin, Linux, Windows
    const char* system_processor;     // e.g. arm64, x86_64
    const char* compiler_id;          // e.g. AppleClang, GNU, MSVC
    const char* compiler_version;     // compiler version string
    const char* build_type;           // Debug/Release/RelWithDebInfo
    int         cxx_standard;         // e.g. 11/14/17
    bool        have_lto;             // link-time optimization enabled
    bool        have_sse;             // compiled with SSE paths available
    bool        have_avx2;            // compiled with AVX2 helper available
    bool        have_neon;            // compiled with NEON paths available
    bool        tiling_disabled;      // row tiling disabled at compile time
    int         tile_width;           // compile-time tile width
    const char* active_cxx_flags;     // CXX flags snapshot for the active config
};

// Returns a process-wide static snapshot of build info
const BuildInfo& GetBuildInfo();

// Convenience printer to stdout
void PrintBuildInfo();

} // namespace task

}  // namespace inspirecv

#endif  // INSPIRECV_VERSION_H
