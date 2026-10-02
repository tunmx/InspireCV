#include "../common/common.h"
#include "inspirecv/core/runtime/cpu_features.h"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <thread>

namespace {
std::array<bool, 3> DetectedFeatures() {
    return {{inspirecv::cpu::HasSsse3(), inspirecv::cpu::HasSse41(),
             inspirecv::cpu::HasAvx2()}};
}
}  // namespace

TEST_CASE("cpu_feature_mask_restricts_detected_features_and_restores_scopes",
          "[cpu][simd-dispatch][contract]") {
    using namespace inspirecv::cpu;
    const uint32_t original = GetCpuFeatureMaskForTesting();
    const ScopedCpuFeatureMask restore_original(original);
    SetCpuFeatureMaskForTesting(kAllCpuFeatures);
    const auto detected = DetectedFeatures();
    {
        const ScopedCpuFeatureMask disabled(0);
        REQUIRE(DetectedFeatures() == (std::array<bool, 3>{{false, false, false}}));
        {
            const ScopedCpuFeatureMask sse(kSsse3 | kSse41);
            REQUIRE(DetectedFeatures() ==
                    (std::array<bool, 3>{{detected[0], detected[1], false}}));
        }
        REQUIRE(GetCpuFeatureMaskForTesting() == 0);
        REQUIRE_FALSE(HasAvx2());
    }
    REQUIRE(GetCpuFeatureMaskForTesting() == kAllCpuFeatures);
    REQUIRE(DetectedFeatures() == detected);
    // Unknown bits cannot authorize additional instructions. The baseline is
    // already bounded by the real CPU, OS save-state support and process cap.
    SetCpuFeatureMaskForTesting(UINT32_MAX);
    REQUIRE(GetCpuFeatureMaskForTesting() == kAllCpuFeatures);
    REQUIRE(DetectedFeatures() == detected);
    try {
        const ScopedCpuFeatureMask disabled(0);
        throw std::runtime_error("exercise scope unwinding");
    } catch (const std::runtime_error&) {
    }
    REQUIRE(GetCpuFeatureMaskForTesting() == kAllCpuFeatures);
    REQUIRE(DetectedFeatures() == detected);
}

TEST_CASE("cpu_feature_mask_is_thread_local", "[cpu][simd-dispatch][contract]") {
    using namespace inspirecv::cpu;
    const ScopedCpuFeatureMask all(kAllCpuFeatures);
    const auto detected = DetectedFeatures();
    const ScopedCpuFeatureMask parent_disabled(0);
    uint32_t initial_mask = 0;
    uint32_t restored_mask = 0;
    std::array<bool, 3> child_detected{}, child_disabled{};
    std::thread worker([&] {
        initial_mask = GetCpuFeatureMaskForTesting();
        child_detected = DetectedFeatures();
        {
            const ScopedCpuFeatureMask disabled(0);
            child_disabled = DetectedFeatures();
        }
        restored_mask = GetCpuFeatureMaskForTesting();
    });
    worker.join();
    REQUIRE(initial_mask == kAllCpuFeatures);
    REQUIRE(restored_mask == kAllCpuFeatures);
    REQUIRE(child_detected == detected);
    REQUIRE(child_disabled == (std::array<bool, 3>{{false, false, false}}));
    REQUIRE(GetCpuFeatureMaskForTesting() == 0);
    REQUIRE(DetectedFeatures() == (std::array<bool, 3>{{false, false, false}}));
}
