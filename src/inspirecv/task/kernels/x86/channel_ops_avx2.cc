#include "inspirecv/task/kernels/x86/avx2_kernels.h"
#include "inspirecv/task/kernels/x86/block_utils.h"

namespace inspirecv {
namespace task {
namespace kernels {
namespace x86 {
namespace {

template <int kSourceChannels, int kDestinationChannels, bool kReverse>
void Remap(const uint8_t* source, uint8_t* destination, size_t count) {
    const __m256i opaque = _mm256_set1_epi32(static_cast<int>(0xff000000u));
    const __m128i reverse4 = _mm_setr_epi8(2, 1, 0, 3, 6, 5, 4, 7,
                                           10, 9, 8, 11, 14, 13, 12, 15);
    const __m256i reverse8 = _mm256_broadcastsi128_si256(reverse4);
    while (count >= 8) {
        __m256i pixels;
        if (kSourceChannels == 1) {
            pixels = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(source)));
            pixels = _mm256_mullo_epi32(pixels, _mm256_set1_epi32(0x010101));
        } else if (kSourceChannels == 3) {
            pixels = detail::LoadTriple8(source);
        } else {
            pixels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(source));
        }
        if (kReverse) pixels = _mm256_shuffle_epi8(pixels, reverse8);
        if (kDestinationChannels == 4) {
            if (kSourceChannels != 4) pixels = _mm256_or_si256(pixels, opaque);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(destination), pixels);
        } else {
            detail::StoreTriple8(destination, pixels);
        }
        source += 8 * kSourceChannels;
        destination += 8 * kDestinationChannels;
        count -= 8;
    }
    while (count-- != 0) {
        // Read all lanes before writing so exact in-place reversal is safe.
        const uint8_t first = source[kReverse ? 2 : 0];
        const uint8_t middle = source[kSourceChannels == 1 ? 0 : 1];
        const uint8_t last = source[kSourceChannels == 1 ? 0 : (kReverse ? 0 : 2)];
        const uint8_t alpha = kSourceChannels == 4 ? source[3] : 255;
        destination[0] = first;
        destination[1] = middle;
        destination[2] = last;
        if (kDestinationChannels == 4) destination[3] = alpha;
        source += kSourceChannels;
        destination += kDestinationChannels;
    }
}

template <int kSourceChannels, bool kBlueFirst>
void Luma(const uint8_t* source, uint8_t* destination, size_t count) {
    const int packed_weights = kBlueFirst ? (7 | (38 << 8) | (19 << 16))
                                          : (19 | (38 << 8) | (7 << 16));
    const __m256i weights = _mm256_set1_epi32(packed_weights);
    while (count >= 8) {
        const __m256i pixels = kSourceChannels == 3
                                ? detail::LoadTriple8(source)
                                : _mm256_loadu_si256(reinterpret_cast<const __m256i*>(source));
        const __m256i pairs = _mm256_maddubs_epi16(pixels, weights);
        const __m256i weighted = _mm256_srli_epi32(
          _mm256_madd_epi16(pairs, _mm256_set1_epi16(1)), 6);
        const __m128i words = _mm_packus_epi32(_mm256_castsi256_si128(weighted),
                                               _mm256_extracti128_si256(weighted, 1));
        const __m128i bytes = _mm_packus_epi16(words, words);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(destination), bytes);
        source += 8 * kSourceChannels;
        destination += 8;
        count -= 8;
    }
    while (count-- != 0) {
        *destination++ = static_cast<uint8_t>(
          (19u * source[kBlueFirst ? 2 : 0] + 38u * source[1] +
           7u * source[kBlueFirst ? 0 : 2]) >> 6);
        source += kSourceChannels;
    }
}

}  // namespace

void ReplicateMonoToTripleAvx2(const uint8_t* source, uint8_t* destination, size_t count) {
    // Expand bytes directly instead of widening each group of eight to dwords,
    // multiplying by 0x010101, then compacting away the fourth byte again.
    // Each 32-byte input becomes exactly three contiguous 32-byte stores.
    const __m128i first = _mm_setr_epi8(0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5);
    const __m128i middle = _mm_setr_epi8(5,5,6,6,6,7,7,7,8,8,8,9,9,9,10,10);
    const __m128i last = _mm_setr_epi8(10,11,11,11,12,12,12,13,13,13,14,14,14,15,15,15);
    const __m256i first32 = _mm256_inserti128_si256(_mm256_castsi128_si256(first), middle, 1);
    const __m256i middle32 = _mm256_inserti128_si256(_mm256_castsi128_si256(last), first, 1);
    const __m256i last32 = _mm256_inserti128_si256(_mm256_castsi128_si256(middle), last, 1);
    while (count >= 32) {
        const __m256i gray = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(source));
        const __m256i low = _mm256_permute2x128_si256(gray, gray, 0x00);
        const __m256i high = _mm256_permute2x128_si256(gray, gray, 0x11);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(destination), _mm256_shuffle_epi8(low, first32));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(destination + 32), _mm256_shuffle_epi8(gray, middle32));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(destination + 64), _mm256_shuffle_epi8(high, last32));
        source += 32;
        destination += 96;
        count -= 32;
    }
    if (count >= 16) {
        const __m128i gray = _mm_loadu_si128(reinterpret_cast<const __m128i*>(source));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(destination), _mm_shuffle_epi8(gray, first));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(destination + 16), _mm_shuffle_epi8(gray, middle));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(destination + 32), _mm_shuffle_epi8(gray, last));
        source += 16;
        destination += 48;
        count -= 16;
    }
    if (count >= 8) {
        const __m128i gray = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(source));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(destination), _mm_shuffle_epi8(gray, first));
        _mm_storel_epi64(reinterpret_cast<__m128i*>(destination + 16), _mm_shuffle_epi8(gray, middle));
        source += 8;
        destination += 24;
        count -= 8;
    }
    while (count-- != 0) {
        const uint8_t gray = *source++;
        destination[0] = gray;
        destination[1] = gray;
        destination[2] = gray;
        destination += 3;
    }
}
void ReplicateMonoToQuadAvx2(const uint8_t* s, uint8_t* d, size_t n) { Remap<1, 4, false>(s, d, n); }
void AppendOpaqueAlphaAvx2(const uint8_t* s, uint8_t* d, size_t n) { Remap<3, 4, false>(s, d, n); }
void ReverseTripleAvx2(const uint8_t* s, uint8_t* d, size_t n) { Remap<3, 3, true>(s, d, n); }
void ReverseQuadColorAvx2(const uint8_t* s, uint8_t* d, size_t n) { Remap<4, 4, true>(s, d, n); }
void DropAlphaAvx2(const uint8_t* s, uint8_t* d, size_t n) { Remap<4, 3, false>(s, d, n); }
void ReverseAndDropAlphaAvx2(const uint8_t* s, uint8_t* d, size_t n) { Remap<4, 3, true>(s, d, n); }
void LumaFromRgbAvx2(const uint8_t* s, uint8_t* d, size_t n) { Luma<3, false>(s, d, n); }
void LumaFromBgrAvx2(const uint8_t* s, uint8_t* d, size_t n) { Luma<3, true>(s, d, n); }
void LumaFromRgbaAvx2(const uint8_t* s, uint8_t* d, size_t n) { Luma<4, false>(s, d, n); }
void LumaFromBgraAvx2(const uint8_t* s, uint8_t* d, size_t n) { Luma<4, true>(s, d, n); }

}  // namespace x86
}  // namespace kernels
}  // namespace task
}  // namespace inspirecv
