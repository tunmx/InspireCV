#include <inspirecv/task/cpu_features.h>

namespace inspirecv {

static inline bool detect_sse41() {
#if defined(__x86_64__) || defined(__i386__)
#if defined(__GNUC__) || defined(__clang__)
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    // CPUID leaf 1: ECX bit 19 indicates SSE4.1
    __asm__ __volatile__(
        "cpuid\n\t"
        : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "a"(1)
    );
    return (ecx & (1u << 19)) != 0u;
#else
    return false;
#endif
#else
    return false;
#endif
}

bool cpu_has_sse41() {
    static int cached = -1;
    if (cached < 0) {
        cached = detect_sse41() ? 1 : 0;
    }
    return cached == 1;
}

static inline bool detect_avx2() {
#if defined(__x86_64__) || defined(__i386__)
#if defined(__GNUC__) || defined(__clang__)
    unsigned int eax=0, ebx=0, ecx=0, edx=0;
    // First check leaf 7 for AVX2 (EBX bit 5)
    __asm__ __volatile__("cpuid\n\t" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(7), "c"(0));
    bool avx2 = (ebx & (1u<<5)) != 0u;
    // Also ensure OS has enabled YMM via XGETBV (XCR0[2:1] = 11b)
    unsigned int xcr0_lo=0, xcr0_hi=0;
    __asm__ __volatile__("xgetbv" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
    bool ymm_ok = ((xcr0_lo & 0x6) == 0x6);
    return avx2 && ymm_ok;
#else
    return false;
#endif
#else
    return false;
#endif
}

bool cpu_has_avx2() {
    static int cached = -1;
    if (cached < 0) cached = detect_avx2() ? 1 : 0;
    return cached == 1;
}

}



