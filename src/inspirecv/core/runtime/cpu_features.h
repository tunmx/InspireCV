#ifndef INSPIRECV_CORE_RUNTIME_CPU_FEATURES_H_
#define INSPIRECV_CORE_RUNTIME_CPU_FEATURES_H_

#include <cstdint>

// Compiler capability is separate from runtime support. MSVC exposes the
// intrinsics without defining the GCC/Clang ISA macros.
#if defined(__SSE2__) || defined(_M_X64) || \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#define INSPIRECV_HAVE_SSE2
#endif

#if defined(__SSSE3__) || \
    (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86)))
#define INSPIRECV_HAVE_SSSE3_INTRINSICS
#endif

namespace inspirecv {
namespace cpu {

enum CpuFeature : uint32_t {
    kSsse3 = 1u << 0,
    kSse41 = 1u << 1,
    kAvx2 = 1u << 2,
};
constexpr uint32_t kAllCpuFeatures = kSsse3 | kSse41 | kAvx2;

bool HasSsse3() noexcept;
bool HasSse41() noexcept;
// Includes both hardware support and the OS XMM/YMM save-state checks.
bool HasAvx2() noexcept;

// Internal differential-test control. This can only disable detected optional
// features; it never authorizes instructions unsupported by this CPU or OS.
// Baseline SSE2 and ARM NEON are not controlled by this x86 dispatch mask.
uint32_t GetCpuFeatureMaskForTesting() noexcept;
void SetCpuFeatureMaskForTesting(uint32_t allowed) noexcept;

class ScopedCpuFeatureMask {
public:
    explicit ScopedCpuFeatureMask(uint32_t allowed) noexcept
        : previous_(GetCpuFeatureMaskForTesting()) {
        SetCpuFeatureMaskForTesting(allowed);
    }
    ~ScopedCpuFeatureMask() { SetCpuFeatureMaskForTesting(previous_); }
    ScopedCpuFeatureMask(const ScopedCpuFeatureMask&) = delete;
    ScopedCpuFeatureMask& operator=(const ScopedCpuFeatureMask&) = delete;

private:
    uint32_t previous_;
};

}  // namespace cpu
}  // namespace inspirecv

#endif  // INSPIRECV_CORE_RUNTIME_CPU_FEATURES_H_
