#include "inspirecv/task/kernels/x86/avx2_kernels.h"
#include "inspirecv/task/kernels/x86/block_utils.h"

namespace inspirecv {
namespace task {
namespace kernels {
namespace x86 {
namespace {

using detail::BytesToFloat8;
using detail::Normalize8;

float Normalize(uint8_t value, float mean, float scale) {
    return scale * (value - mean);
}

__m256 RepeatQuad(const float* values) {
    return _mm256_broadcast_ps(reinterpret_cast<const __m128*>(values));
}

void StoreQuad8(__m256i pixels, float* destination, __m256 mean, __m256 scale) {
    const __m128i first = _mm256_castsi256_si128(pixels);
    const __m128i last = _mm256_extracti128_si256(pixels, 1);
    _mm256_storeu_ps(destination, Normalize8(BytesToFloat8(first), mean, scale));
    _mm256_storeu_ps(destination + 8,
                     Normalize8(BytesToFloat8(_mm_srli_si128(first, 8)), mean, scale));
    _mm256_storeu_ps(destination + 16, Normalize8(BytesToFloat8(last), mean, scale));
    _mm256_storeu_ps(destination + 24,
                     Normalize8(BytesToFloat8(_mm_srli_si128(last, 8)), mean, scale));
}

}  // namespace

void InterleavedMonoAvx2(const uint8_t* source, float* destination,
                         const float* mean, const float* scale, size_t count) {
    const __m256 means = _mm256_set1_ps(mean[0]);
    const __m256 scales = _mm256_set1_ps(scale[0]);
    size_t pixel = 0;
    for (; count - pixel >= 16; pixel += 16) {
        const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(source + pixel));
        _mm256_storeu_ps(destination + pixel,
                         Normalize8(BytesToFloat8(bytes), means, scales));
        _mm256_storeu_ps(destination + pixel + 8,
                         Normalize8(BytesToFloat8(_mm_srli_si128(bytes, 8)), means, scales));
    }
    for (; pixel < count; ++pixel) destination[pixel] = Normalize(source[pixel], mean[0], scale[0]);
}

void InterleavedTripleAvx2(const uint8_t* source, float* destination,
                           const float* mean, const float* scale, size_t count) {
    const __m256 means[3] = {
      _mm256_setr_ps(mean[0], mean[1], mean[2], mean[0], mean[1], mean[2], mean[0], mean[1]),
      _mm256_setr_ps(mean[2], mean[0], mean[1], mean[2], mean[0], mean[1], mean[2], mean[0]),
      _mm256_setr_ps(mean[1], mean[2], mean[0], mean[1], mean[2], mean[0], mean[1], mean[2])};
    const __m256 scales[3] = {
      _mm256_setr_ps(scale[0], scale[1], scale[2], scale[0], scale[1], scale[2], scale[0], scale[1]),
      _mm256_setr_ps(scale[2], scale[0], scale[1], scale[2], scale[0], scale[1], scale[2], scale[0]),
      _mm256_setr_ps(scale[1], scale[2], scale[0], scale[1], scale[2], scale[0], scale[1], scale[2])};
    size_t pixel = 0;
    for (; count - pixel >= 8; pixel += 8) {
        const __m128i first = _mm_loadu_si128(reinterpret_cast<const __m128i*>(source + 3 * pixel));
        const __m128i last = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(source + 3 * pixel + 16));
        _mm256_storeu_ps(destination + 3 * pixel,
                         Normalize8(BytesToFloat8(first), means[0], scales[0]));
        _mm256_storeu_ps(destination + 3 * pixel + 8,
                         Normalize8(BytesToFloat8(_mm_srli_si128(first, 8)), means[1], scales[1]));
        _mm256_storeu_ps(destination + 3 * pixel + 16,
                         Normalize8(BytesToFloat8(last), means[2], scales[2]));
    }
    for (; pixel < count; ++pixel) {
        for (size_t channel = 0; channel < 3; ++channel) {
            destination[3 * pixel + channel] = Normalize(source[3 * pixel + channel], mean[channel], scale[channel]);
        }
    }
}

void InterleavedQuadAvx2(const uint8_t* source, float* destination,
                         const float* mean, const float* scale, size_t count) {
    const __m256 means = RepeatQuad(mean);
    const __m256 scales = RepeatQuad(scale);
    size_t pixel = 0;
    for (; count - pixel >= 8; pixel += 8) {
        StoreQuad8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(source + 4 * pixel)),
                    destination + 4 * pixel, means, scales);
    }
    for (; pixel < count; ++pixel) {
        for (size_t channel = 0; channel < 4; ++channel) {
            destination[4 * pixel + channel] = Normalize(source[4 * pixel + channel], mean[channel], scale[channel]);
        }
    }
}

void PlanarTripleAvx2(const uint8_t* source, float* destination,
                      size_t channel_stride, const float* mean,
                      const float* scale, size_t count) {
    const __m256 means[3] = {_mm256_set1_ps(mean[0]), _mm256_set1_ps(mean[1]), _mm256_set1_ps(mean[2])};
    const __m256 scales[3] = {_mm256_set1_ps(scale[0]), _mm256_set1_ps(scale[1]), _mm256_set1_ps(scale[2])};
    const __m256i mask = _mm256_set1_epi32(255);
    size_t pixel = 0;
    for (; count - pixel >= 8; pixel += 8) {
        const __m256i rgba = detail::LoadTriple8(source + 3 * pixel);
        const __m256 channels[3] = {
          _mm256_cvtepi32_ps(_mm256_and_si256(rgba, mask)),
          _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(rgba, 8), mask)),
          _mm256_cvtepi32_ps(_mm256_srli_epi32(rgba, 16))};
        for (size_t channel = 0; channel < 3; ++channel) {
            _mm256_storeu_ps(destination + channel_stride * channel + pixel,
                             Normalize8(channels[channel], means[channel], scales[channel]));
        }
    }
    for (; pixel < count; ++pixel) {
        for (size_t channel = 0; channel < 3; ++channel) {
            destination[channel_stride * channel + pixel] = Normalize(source[3 * pixel + channel], mean[channel], scale[channel]);
        }
    }
}

void QuadFromMonoAvx2(const uint8_t* source, float* destination,
                      const float* mean, const float* scale, size_t count) {
    const __m256 means = _mm256_set1_ps(mean[0]);
    const __m256 scales = _mm256_set1_ps(scale[0]);
    const __m256 zero = _mm256_setzero_ps();
    const __m256i orders[4] = {
      _mm256_setr_epi32(0, 0, 0, 0, 1, 0, 0, 0),
      _mm256_setr_epi32(2, 0, 0, 0, 3, 0, 0, 0),
      _mm256_setr_epi32(4, 0, 0, 0, 5, 0, 0, 0),
      _mm256_setr_epi32(6, 0, 0, 0, 7, 0, 0, 0)};
    size_t pixel = 0;
    for (; count - pixel >= 8; pixel += 8) {
        const __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(source + pixel));
        const __m256 values = Normalize8(BytesToFloat8(bytes), means, scales);
        for (size_t pair = 0; pair < 4; ++pair) {
            const __m256 expanded = _mm256_permutevar8x32_ps(values, orders[pair]);
            _mm256_storeu_ps(destination + 4 * pixel + 8 * pair,
                             _mm256_blend_ps(zero, expanded, 0x11));
        }
    }
    for (; pixel < count; ++pixel) {
        destination[4 * pixel] = Normalize(source[pixel], mean[0], scale[0]);
        destination[4 * pixel + 1] = 0.0f;
        destination[4 * pixel + 2] = 0.0f;
        destination[4 * pixel + 3] = 0.0f;
    }
}

void QuadFromTripleAvx2(const uint8_t* source, float* destination,
                        const float* mean, const float* scale, size_t count) {
    const __m256 means = _mm256_setr_ps(mean[0], mean[1], mean[2], 0.0f, mean[0], mean[1], mean[2], 0.0f);
    const __m256 scales = _mm256_setr_ps(scale[0], scale[1], scale[2], 0.0f, scale[0], scale[1], scale[2], 0.0f);
    size_t pixel = 0;
    for (; count - pixel >= 8; pixel += 8) {
        StoreQuad8(detail::LoadTriple8(source + 3 * pixel), destination + 4 * pixel, means, scales);
    }
    for (; pixel < count; ++pixel) {
        for (size_t channel = 0; channel < 3; ++channel) {
            destination[4 * pixel + channel] = Normalize(source[3 * pixel + channel], mean[channel], scale[channel]);
        }
        destination[4 * pixel + 3] = 0.0f;
    }
}

}  // namespace x86
}  // namespace kernels
}  // namespace task
}  // namespace inspirecv
