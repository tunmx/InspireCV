#include "inspirecv/task/kernels/x86/avx2_kernels.h"
#include "inspirecv/task/kernels/x86/block_utils.h"

namespace inspirecv {
namespace task {
namespace kernels {
namespace x86 {
namespace {

void LoadRgb8(const uint8_t* source, bool blue_first, __m256i channels[3]) {
    const __m256i rgb = detail::LoadTriple8(source);
    const __m256i mask = _mm256_set1_epi32(255);
    channels[blue_first ? 2 : 0] = _mm256_and_si256(rgb, mask);
    channels[1] = _mm256_and_si256(_mm256_srli_epi32(rgb, 8), mask);
    channels[blue_first ? 0 : 2] = _mm256_srli_epi32(rgb, 16);
}

__m256i ClampBytes(__m256i value) {
    return _mm256_min_epi32(_mm256_max_epi32(value, _mm256_setzero_si256()), _mm256_set1_epi32(255));
}

}  // namespace

size_t ColorMatrixAvx2(const uint8_t* source, uint8_t* destination, size_t count,
                       const int coefficients[3][3], int shift, const int* offset,
                       bool clamp, bool blue_first) {
    __m256i weights[3][3];
    __m256i offsets[3];
    for (int row = 0; row < 3; ++row) {
        for (int channel = 0; channel < 3; ++channel) weights[row][channel] = _mm256_set1_epi32(coefficients[row][channel]);
        offsets[row] = _mm256_set1_epi32(offset[row]);
    }
    const __m256i round = _mm256_set1_epi32(1 << (shift - 1));
    const __m256i shift_count = _mm256_set1_epi32(shift);
    const __m256i byte_mask = _mm256_set1_epi32(255);
    const size_t completed = count - count % 8;
    for (size_t pixel = 0; pixel < completed; pixel += 8) {
        __m256i rgb[3];
        LoadRgb8(source + 3 * pixel, blue_first, rgb);
        __m256i packed = _mm256_setzero_si256();
        for (int row = 0; row < 3; ++row) {
            const __m256i sum = _mm256_add_epi32(
              _mm256_add_epi32(_mm256_mullo_epi32(rgb[0], weights[row][0]),
                                 _mm256_mullo_epi32(rgb[1], weights[row][1])),
              _mm256_mullo_epi32(rgb[2], weights[row][2]));
            __m256i value = _mm256_add_epi32(_mm256_srav_epi32(_mm256_add_epi32(sum, round), shift_count), offsets[row]);
            value = clamp ? ClampBytes(value) : _mm256_and_si256(value, byte_mask);
            packed = _mm256_or_si256(packed, _mm256_sllv_epi32(value, _mm256_set1_epi32(8 * row)));
        }
        detail::StoreTriple8(destination + 3 * pixel, packed);
    }
    return completed;
}

size_t ColorHsvAvx2(const uint8_t* source, uint8_t* destination, size_t count,
                    int hue_range, bool blue_first) {
    const __m256i zero = _mm256_setzero_si256();
    const __m256i one = _mm256_set1_epi32(1);
    const __m256i rounding = _mm256_set1_epi32(1 << 11);
    const size_t completed = count - count % 8;
    for (size_t pixel = 0; pixel < completed; pixel += 8) {
        __m256i rgb[3];
        LoadRgb8(source + 3 * pixel, blue_first, rgb);
        const __m256i maximum = _mm256_max_epi32(rgb[0], _mm256_max_epi32(rgb[1], rgb[2]));
        const __m256i minimum = _mm256_min_epi32(rgb[0], _mm256_min_epi32(rgb[1], rgb[2]));
        const __m256i difference = _mm256_sub_epi32(maximum, minimum);
        const __m256i red_mask = _mm256_cmpeq_epi32(maximum, rgb[0]);
        const __m256i green_mask = _mm256_cmpeq_epi32(maximum, rgb[1]);
        // The MSVC Release caller's historical /fp:fast expression uses this
        // direct quotient. A separately rounded reciprocal changes saturation
        // by one at some half-value ties; dispatch is limited to that contract.
        const __m256i saturation = _mm256_srai_epi32(_mm256_add_epi32(
          _mm256_cvttps_epi32(_mm256_div_ps(
            _mm256_cvtepi32_ps(_mm256_mullo_epi32(difference, _mm256_set1_epi32(255 << 12))),
            _mm256_cvtepi32_ps(_mm256_max_epi32(maximum, one)))), rounding), 12);
        const __m256i green_hue = _mm256_add_epi32(_mm256_sub_epi32(rgb[2], rgb[0]), _mm256_slli_epi32(difference, 1));
        const __m256i blue_hue = _mm256_add_epi32(_mm256_sub_epi32(rgb[0], rgb[1]), _mm256_slli_epi32(difference, 2));
        __m256i hue = _mm256_blendv_epi8(
          _mm256_blendv_epi8(blue_hue, green_hue, green_mask),
          _mm256_sub_epi32(rgb[1], rgb[2]), red_mask);
        const __m256 divisor = _mm256_mul_ps(_mm256_set1_ps(6.0f),
          _mm256_cvtepi32_ps(_mm256_max_epi32(difference, one)));
        const __m256i hue_scale = _mm256_cvttps_epi32(_mm256_add_ps(
          _mm256_div_ps(_mm256_set1_ps(static_cast<float>(hue_range << 12)), divisor),
          _mm256_set1_ps(0.5f)));
        hue = _mm256_srai_epi32(_mm256_add_epi32(_mm256_mullo_epi32(hue, hue_scale), rounding), 12);
        hue = _mm256_add_epi32(hue, _mm256_and_si256(_mm256_cmpgt_epi32(zero, hue), _mm256_set1_epi32(hue_range)));
        const __m256i packed = _mm256_or_si256(ClampBytes(hue),
          _mm256_or_si256(_mm256_slli_epi32(saturation, 8), _mm256_slli_epi32(maximum, 16)));
        detail::StoreTriple8(destination + 3 * pixel, packed);
    }
    return completed;
}

size_t PackBgr16Avx2(const uint8_t* source, uint8_t* destination, size_t count,
                     bool green_six_bits, bool blue_first) {
    const size_t completed = count - count % 8;
    for (size_t pixel = 0; pixel < completed; pixel += 8) {
        __m256i rgb[3];
        LoadRgb8(source + 3 * pixel, blue_first, rgb);
        const __m256i blue = _mm256_srli_epi32(rgb[2], 3);
        const __m256i green = green_six_bits
          ? _mm256_slli_epi32(_mm256_and_si256(rgb[1], _mm256_set1_epi32(~3)), 3)
          : _mm256_slli_epi32(_mm256_and_si256(rgb[1], _mm256_set1_epi32(~7)), 2);
        const __m256i red = green_six_bits
          ? _mm256_slli_epi32(_mm256_and_si256(rgb[0], _mm256_set1_epi32(~7)), 8)
          : _mm256_slli_epi32(_mm256_and_si256(rgb[0], _mm256_set1_epi32(~7)), 7);
        const __m256i packed = _mm256_or_si256(blue, _mm256_or_si256(green, red));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(destination + 2 * pixel),
          _mm_packus_epi32(_mm256_castsi256_si128(packed), _mm256_extracti128_si256(packed, 1)));
    }
    return completed;
}

}  // namespace x86
}  // namespace kernels
}  // namespace task
}  // namespace inspirecv
