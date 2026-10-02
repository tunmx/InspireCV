#include <inspirecv/core/runtime/cpu_features.h>

#include <cstdlib>
#include <cstring>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace inspirecv {
namespace cpu {

namespace {

bool QueryCpuid(unsigned int leaf, unsigned int subleaf, unsigned int& eax, unsigned int& ebx,
                unsigned int& ecx, unsigned int& edx) {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#if defined(_MSC_VER)
    int registers[4] = {0, 0, 0, 0};
    __cpuid(registers, 0);
    if (static_cast<unsigned int>(registers[0]) < leaf) return false;
    __cpuidex(registers, static_cast<int>(leaf), static_cast<int>(subleaf));
    eax = static_cast<unsigned int>(registers[0]);
    ebx = static_cast<unsigned int>(registers[1]);
    ecx = static_cast<unsigned int>(registers[2]);
    edx = static_cast<unsigned int>(registers[3]);
    return true;
#else
    return __get_cpuid_count(leaf, subleaf, &eax, &ebx, &ecx, &edx) != 0;
#endif
#else
    (void)leaf;
    (void)subleaf;
    eax = ebx = ecx = edx = 0;
    return false;
#endif
}

unsigned long long ReadXcr0() {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#if defined(_MSC_VER)
    return _xgetbv(0);
#else
    unsigned int low = 0;
    unsigned int high = 0;
    __asm__ __volatile__("xgetbv" : "=a"(low), "=d"(high) : "c"(0));
    return (static_cast<unsigned long long>(high) << 32) | low;
#endif
#else
    return 0;
#endif
}

uint32_t ConfiguredFeatureMask() {
    // Diagnostic process-wide cap, sampled once. Per-thread test masks cannot
    // re-enable a feature disabled here. Production defaults to all detected
    // features. Unknown tokens are ignored, never treated as feature enables.
    static const uint32_t configured = []() {
        uint32_t mask = kAllCpuFeatures;
        const char* value = std::getenv("INSPIRECV_INTERNAL_CPU_DISABLE");
        if (!value) return mask;
        if (std::strcmp(value, "ALL") == 0) return uint32_t{0};
        const char* token = value;
        while (*token) {
            const char* end = std::strchr(token, ',');
            const size_t length = end ? static_cast<size_t>(end - token) : std::strlen(token);
            if (length == 5 && std::strncmp(token, "SSSE3", length) == 0) mask &= ~kSsse3;
            if (length == 5 && std::strncmp(token, "SSE41", length) == 0) mask &= ~kSse41;
            if (length == 4 && std::strncmp(token, "AVX2", length) == 0) mask &= ~kAvx2;
            if (!end) break;
            token = end + 1;
        }
        return mask;
    }();
    return configured;
}

thread_local uint32_t allowed_features = kAllCpuFeatures;

bool FeatureAllowed(uint32_t feature) {
    return (allowed_features & ConfiguredFeatureMask() & feature) != 0;
}

}  // namespace

static bool DetectSsse3() {
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    return QueryCpuid(1, 0, eax, ebx, ecx, edx) && (ecx & (1u << 9)) != 0u;
}

bool HasSsse3() noexcept {
    static const bool cached = DetectSsse3();
    return cached && FeatureAllowed(kSsse3);
}

static inline bool DetectSse41() {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (!QueryCpuid(1, 0, eax, ebx, ecx, edx)) return false;
    return (ecx & (1u << 19)) != 0u;
#else
    return false;
#endif
}

bool HasSse41() noexcept {
    static const bool cached = DetectSse41();
    return cached && FeatureAllowed(kSse41);
}

static inline bool DetectAvx2() {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (!QueryCpuid(1, 0, eax, ebx, ecx, edx)) return false;
    // XGETBV is only legal when both AVX and OSXSAVE are advertised.
    constexpr unsigned int kOsxsave = 1u << 27;
    constexpr unsigned int kAvx = 1u << 28;
    if ((ecx & (kOsxsave | kAvx)) != (kOsxsave | kAvx)) return false;
    if ((ReadXcr0() & 0x6u) != 0x6u) return false;
    if (!QueryCpuid(7, 0, eax, ebx, ecx, edx)) return false;
    return (ebx & (1u << 5)) != 0u;
#else
    return false;
#endif
}

bool HasAvx2() noexcept {
    static const bool cached = DetectAvx2();
    return cached && FeatureAllowed(kAvx2);
}

uint32_t GetCpuFeatureMaskForTesting() noexcept {
    return allowed_features;
}

void SetCpuFeatureMaskForTesting(uint32_t allowed) noexcept {
    allowed_features = allowed & kAllCpuFeatures;
}

}  // namespace cpu
}  // namespace inspirecv
