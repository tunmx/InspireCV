#include <inspirecv/task/platform/cpu_features.h>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace inspirecv {
namespace task {
namespace platform {

namespace {

bool QueryCpuid(unsigned int leaf, unsigned int subleaf, unsigned int& eax, unsigned int& ebx,
                unsigned int& ecx, unsigned int& edx) {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#if defined(_MSC_VER)
    int registers[4] = {0, 0, 0, 0};
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

}  // namespace

static inline bool DetectSse41() {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (!QueryCpuid(1, 0, eax, ebx, ecx, edx)) return false;
    return (ecx & (1u << 19)) != 0u;
#else
    return false;
#endif
}

bool HasSse41() {
    static const bool cached = DetectSse41();
    return cached;
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

bool HasAvx2() {
    static const bool cached = DetectAvx2();
    return cached;
}

}  // namespace platform
}  // namespace task
}  // namespace inspirecv
