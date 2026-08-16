#include "../common/common.h"

#include <inspirecv/version.h>

#include <string>

TEST_CASE("configured_backend_identity_matches_compiled_backend",
          "[build][contract]") {
    const auto& info = inspirecv::GetLibraryInfo();
    const std::string backend = inspirecv::GetCVBackend();
    const std::string version = inspirecv::GetVersion();
    REQUIRE(info.version_major == 1);
    REQUIRE(info.version_minor == 0);
    REQUIRE(info.version_patch == 2);
    REQUIRE(std::string(info.version_string) == version);
    REQUIRE(std::string(info.cv_backend) == backend);
    REQUIRE(info.build_date != nullptr);
    REQUIRE(info.system_name != nullptr);
    REQUIRE(info.system_processor != nullptr);
    REQUIRE(info.compiler_name != nullptr);
    REQUIRE(info.compiler_version != nullptr);
    REQUIRE(info.build_type != nullptr);
    REQUIRE(info.cxx_standard == 14);
    REQUIRE(info.task_tile_width > 0);
    REQUIRE(info.cxx_flags != nullptr);
    REQUIRE(info.neon_enabled == (INSPIRECV_TEST_HAVE_NEON != 0));
    REQUIRE(info.sse_enabled == (INSPIRECV_TEST_HAVE_SSE != 0));
    REQUIRE(info.avx2_enabled == (INSPIRECV_TEST_HAVE_AVX2 != 0));
#if INSPIRECV_TEST_BACKEND_OPENCV
    REQUIRE(backend == "OpenCV");
    REQUIRE(version.find("@OpenCV") != std::string::npos);
#else
    REQUIRE(backend == "OKCV");
    REQUIRE(version.find("@General") != std::string::npos);
#endif
}
