#ifndef INSPIRECV_TASK_KERNELS_X86_BLOCK_UTILS_H_
#define INSPIRECV_TASK_KERNELS_X86_BLOCK_UTILS_H_

#include <immintrin.h>
#include <cstdint>

namespace inspirecv {
namespace task {
namespace kernels {
namespace x86 {
namespace detail {

// Read exactly 24 bytes. A full-width load at source + 12 would over-read the
// final RGB block of tightly packed images or scratch tiles.
inline __m256i LoadTriple8(const uint8_t* source) {
    const __m128i first = _mm_loadu_si128(reinterpret_cast<const __m128i*>(source));
    const __m128i last = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(source + 16));
    const __m128i second = _mm_alignr_epi8(last, first, 12);
    const __m256i packed = _mm256_inserti128_si256(_mm256_castsi128_si256(first), second, 1);
    const __m128i expand = _mm_setr_epi8(0, 1, 2, -1, 3, 4, 5, -1,
                                        6, 7, 8, -1, 9, 10, 11, -1);
    return _mm256_shuffle_epi8(packed, _mm256_broadcastsi128_si256(expand));
}

// Write exactly 24 bytes; unlike overlapping 16-byte stores, this also works
// for the final block and for destinations immediately before a guard page.
inline void StoreTriple8(uint8_t* destination, __m256i pixels) {
    const __m128i compact = _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9,
                                         10, 12, 13, 14, -1, -1, -1, -1);
    const __m256i packed = _mm256_shuffle_epi8(pixels, _mm256_broadcastsi128_si256(compact));
    const __m128i first = _mm256_castsi256_si128(packed);
    const __m128i second = _mm256_extracti128_si256(packed, 1);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(destination),
                     _mm_or_si128(first, _mm_slli_si128(second, 12)));
    _mm_storel_epi64(reinterpret_cast<__m128i*>(destination + 16),
                     _mm_srli_si128(second, 4));
}

inline __m256 Normalize8(__m256 values, __m256 mean, __m256 scale) {
    // Keep the documented subtraction-then-multiplication order; no FMA.
    return _mm256_mul_ps(_mm256_sub_ps(values, mean), scale);
}

inline __m256 BytesToFloat8(__m128i bytes) {
    return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes));
}

}  // namespace detail
}  // namespace x86
}  // namespace kernels
}  // namespace task
}  // namespace inspirecv

#endif  // INSPIRECV_TASK_KERNELS_X86_BLOCK_UTILS_H_
