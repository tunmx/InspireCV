#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#include <stddef.h>
extern "C" void task_avx2_memcpy(void* dst, const void* src, size_t n) {
#if defined(__AVX2__)
    unsigned char* d = (unsigned char*)dst;
    const unsigned char* s = (const unsigned char*)src;
    // 64-byte blocks
    while (n >= 64) {
        __m256i a0 = _mm256_loadu_si256((const __m256i*)(s + 0));
        __m256i a1 = _mm256_loadu_si256((const __m256i*)(s + 32));
        _mm256_storeu_si256((__m256i*)(d + 0), a0);
        _mm256_storeu_si256((__m256i*)(d + 32), a1);
        s += 64; d += 64; n -= 64;
    }
    while (n >= 32) {
        __m256i a0 = _mm256_loadu_si256((const __m256i*)s);
        _mm256_storeu_si256((__m256i*)d, a0);
        s += 32; d += 32; n -= 32;
    }
    while (n >= 16) {
        __m128i a = _mm_loadu_si128((const __m128i*)s);
        _mm_storeu_si128((__m128i*)d, a);
        s += 16; d += 16; n -= 16;
    }
    if (n) {
        for (size_t i = 0; i < n; ++i) d[i] = s[i];
    }
#else
    // Compiled without AVX2 flags; fallback
    unsigned char* d = (unsigned char*)dst; const unsigned char* s = (const unsigned char*)src;
    for (size_t i=0;i<n;++i) d[i]=s[i];
#endif
}
#endif



