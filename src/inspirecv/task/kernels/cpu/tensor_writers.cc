#include "inspirecv/task/kernels/cpu/tensor_writers.h"

#include <cstring>

#if defined(INSPIRECV_TASK_USE_NEON)
#include <arm_neon.h>
#endif

#if defined(INSPIRECV_TASK_USE_SSE) && defined(__SSE4_1__)
#include <x86intrin.h>
#endif

namespace inspirecv {
namespace task {
namespace kernels {
namespace tensor {
namespace {

float Normalize(uint8_t value, float mean, float scale) {
    return scale * (value - mean);
}

void ScalarMono(const uint8_t* source, float* destination, const float* mean,
                const float* scale, size_t first, size_t count) {
    for (size_t pixel = first; pixel < count; ++pixel) {
        destination[pixel] = Normalize(source[pixel], mean[0], scale[0]);
    }
}

void ScalarTriple(const uint8_t* source, float* destination, const float* mean,
                  const float* scale, size_t first, size_t count) {
    for (size_t pixel = first; pixel < count; ++pixel) {
        for (size_t channel = 0; channel < 3; ++channel) {
            destination[3 * pixel + channel] =
              Normalize(source[3 * pixel + channel], mean[channel], scale[channel]);
        }
    }
}

#if defined(INSPIRECV_TASK_USE_NEON)
struct FourVectors {
    float32x4_t value[4];
};

FourVectors Normalize16(uint8x16_t bytes, float32x4_t offset,
                        float32x4_t multiplier) {
    const uint16x8_t low = vmovl_u8(vget_low_u8(bytes));
    const uint16x8_t high = vmovl_u8(vget_high_u8(bytes));
    FourVectors output = {{
      vcvtq_f32_u32(vmovl_u16(vget_low_u16(low))),
      vcvtq_f32_u32(vmovl_u16(vget_high_u16(low))),
      vcvtq_f32_u32(vmovl_u16(vget_low_u16(high))),
      vcvtq_f32_u32(vmovl_u16(vget_high_u16(high)))}};
    for (size_t group = 0; group < 4; ++group) {
        output.value[group] = vmulq_f32(
          vaddq_f32(output.value[group], offset), multiplier);
    }
    return output;
}

extern "C" {
void inspirecv_task_c1_to_float_c4_arm(const uint8_t*, float*, const float*,
                                        const float*, size_t);
void inspirecv_task_c3_to_float_c4_arm(const uint8_t*, float*, const float*,
                                        const float*, size_t);
}
#endif

#if defined(INSPIRECV_TASK_USE_SSE) && defined(__SSE4_1__)
__m128 Normalize4(__m128 value, __m128 mean, __m128 scale) {
    return _mm_mul_ps(_mm_sub_ps(value, mean), scale);
}

struct ByteGroups {
    __m128i value[4];
};

ByteGroups SplitBytes(__m128i bytes) {
    return {{bytes, _mm_srli_si128(bytes, 4), _mm_srli_si128(bytes, 8),
             _mm_srli_si128(bytes, 12)}};
}

void StoreMonoQuad(__m128 values, float* destination) {
    __m128 lane1 = _mm_setzero_ps();
    __m128 lane2 = _mm_setzero_ps();
    __m128 lane3 = _mm_setzero_ps();
    _MM_TRANSPOSE4_PS(values, lane1, lane2, lane3);
    _mm_storeu_ps(destination + 0, values);
    _mm_storeu_ps(destination + 4, lane1);
    _mm_storeu_ps(destination + 8, lane2);
    _mm_storeu_ps(destination + 12, lane3);
}

size_t WriteMonoSse(const uint8_t* source, float* destination, float mean,
                    float scale, size_t count) {
    const size_t blocks = count / 16;
    const __m128 mean4 = _mm_set1_ps(mean);
    const __m128 scale4 = _mm_set1_ps(scale);
    for (size_t block = 0; block < blocks; ++block) {
        const __m128i bytes = _mm_loadu_si128(
          reinterpret_cast<const __m128i*>(source + 16 * block));
        const ByteGroups groups = SplitBytes(bytes);
        for (size_t group = 0; group < 4; ++group) {
            const __m128 values =
              _mm_cvtepi32_ps(_mm_cvtepu8_epi32(groups.value[group]));
            _mm_storeu_ps(destination + 16 * block + 4 * group,
                          Normalize4(values, mean4, scale4));
        }
    }
    return blocks * 16;
}

size_t WriteTripleSse(const uint8_t* source, float* destination,
                      const float* mean, const float* scale, size_t count) {
    size_t blocks = 0;
    const size_t candidates = count / 4;
    if (candidates > 1) {
        blocks = candidates;
        if ((count % 4) * 3 < 4) --blocks;
    }
    const __m128 means[3] = {
      _mm_setr_ps(mean[0], mean[1], mean[2], mean[0]),
      _mm_setr_ps(mean[1], mean[2], mean[0], mean[1]),
      _mm_setr_ps(mean[2], mean[0], mean[1], mean[2])};
    const __m128 scales[3] = {
      _mm_setr_ps(scale[0], scale[1], scale[2], scale[0]),
      _mm_setr_ps(scale[1], scale[2], scale[0], scale[1]),
      _mm_setr_ps(scale[2], scale[0], scale[1], scale[2])};
    for (size_t block = 0; block < blocks; ++block) {
        const __m128i bytes = _mm_loadu_si128(
          reinterpret_cast<const __m128i*>(source + 12 * block));
        const ByteGroups groups = SplitBytes(bytes);
        for (size_t group = 0; group < 3; ++group) {
            const __m128 values = _mm_cvtepi32_ps(
              _mm_cvtepu8_epi32(groups.value[group]));
            _mm_storeu_ps(destination + 12 * block + 4 * group,
                          Normalize4(values, means[group], scales[group]));
        }
    }
    return blocks * 4;
}

size_t WriteMonoQuadSse(const uint8_t* source, float* destination, float mean,
                        float scale, size_t count) {
    std::memset(destination, 0, 4 * sizeof(float) * count);
    const size_t blocks = count / 16;
    const __m128 mean4 = _mm_set1_ps(mean);
    const __m128 scale4 = _mm_set1_ps(scale);
    for (size_t block = 0; block < blocks; ++block) {
        const __m128i bytes = _mm_loadu_si128(
          reinterpret_cast<const __m128i*>(source + 16 * block));
        const ByteGroups groups = SplitBytes(bytes);
        float* output = destination + 64 * block;
        for (size_t group = 0; group < 4; ++group) {
            const __m128 values = _mm_cvtepi32_ps(
              _mm_cvtepu8_epi32(groups.value[group]));
            StoreMonoQuad(Normalize4(values, mean4, scale4), output + 16 * group);
        }
    }
    return blocks * 16;
}

size_t WriteTripleQuadSse(const uint8_t* source, float* destination,
                          const float* mean, const float* scale, size_t count) {
    size_t blocks = 0;
    const size_t candidates = count / 4;
    if (candidates > 1) {
        blocks = candidates;
        if ((count % 4) * 3 < 4) --blocks;
    }
    const __m128 mean4 = _mm_setr_ps(mean[0], mean[1], mean[2], 0.0f);
    const __m128 scale4 = _mm_setr_ps(scale[0], scale[1], scale[2], 0.0f);
    const __m128i selectors[4] = {
      _mm_setr_epi8(0, 1, 2, 0, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15),
      _mm_setr_epi8(3, 4, 5, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15),
      _mm_setr_epi8(6, 7, 8, 6, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15),
      _mm_setr_epi8(9, 10, 11, 9, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15)};
    for (size_t block = 0; block < blocks; ++block) {
        const __m128i bytes = _mm_loadu_si128(
          reinterpret_cast<const __m128i*>(source + 12 * block));
        for (size_t pixel = 0; pixel < 4; ++pixel) {
            const __m128 values = _mm_cvtepi32_ps(
              _mm_cvtepu8_epi32(_mm_shuffle_epi8(bytes, selectors[pixel])));
            _mm_storeu_ps(destination + 16 * block + 4 * pixel,
                          Normalize4(values, mean4, scale4));
        }
    }
    return blocks * 4;
}
#endif

}  // namespace

void InterleavedMono(const uint8_t* source, float* destination,
                     const float* mean, const float* scale, size_t count) {
    size_t completed = 0;
#if defined(INSPIRECV_TASK_USE_SSE) && defined(__SSE4_1__)
    completed = WriteMonoSse(source, destination, mean[0], scale[0], count);
#elif defined(INSPIRECV_TASK_USE_NEON)
    const size_t blocks = count / 16;
    const float32x4_t offset = vdupq_n_f32(-mean[0]);
    const float32x4_t multiplier = vdupq_n_f32(scale[0]);
    for (size_t block = 0; block < blocks; ++block) {
        const FourVectors output =
          Normalize16(vld1q_u8(source + 16 * block), offset, multiplier);
        for (size_t group = 0; group < 4; ++group) {
            vst1q_f32(destination + 16 * block + 4 * group, output.value[group]);
        }
    }
    completed = blocks * 16;
#endif
    ScalarMono(source, destination, mean, scale, completed, count);
}

void InterleavedTriple(const uint8_t* source, float* destination,
                       const float* mean, const float* scale, size_t count) {
    size_t completed = 0;
#if defined(INSPIRECV_TASK_USE_SSE) && defined(__SSE4_1__)
    completed = WriteTripleSse(source, destination, mean, scale, count);
#elif defined(INSPIRECV_TASK_USE_NEON)
    const size_t blocks = count / 16;
    float32x4_t offsets[3];
    float32x4_t multipliers[3];
    for (size_t channel = 0; channel < 3; ++channel) {
        offsets[channel] = vdupq_n_f32(-mean[channel]);
        multipliers[channel] = vdupq_n_f32(scale[channel]);
    }
    for (size_t block = 0; block < blocks; ++block) {
        const uint8x16x3_t pixels = vld3q_u8(source + 48 * block);
        FourVectors channels[3];
        for (size_t channel = 0; channel < 3; ++channel) {
            channels[channel] =
              Normalize16(pixels.val[channel], offsets[channel],
                          multipliers[channel]);
        }
        for (size_t group = 0; group < 4; ++group) {
            const float32x4x3_t output = {{channels[0].value[group],
                                           channels[1].value[group],
                                           channels[2].value[group]}};
            vst3q_f32(destination + 48 * block + 12 * group, output);
        }
    }
    completed = blocks * 16;
#endif
    ScalarTriple(source, destination, mean, scale, completed, count);
}

void PlanarTriple(const uint8_t* source, float* destination, size_t channel_stride,
                  const float* mean, const float* scale, size_t count) {
    float* planes[3] = {destination, destination + channel_stride,
                        destination + 2 * channel_stride};
    size_t completed = 0;
#if defined(INSPIRECV_TASK_USE_NEON)
    const size_t blocks = count / 16;
    float32x4_t offsets[3];
    float32x4_t multipliers[3];
    for (size_t channel = 0; channel < 3; ++channel) {
        offsets[channel] = vdupq_n_f32(-mean[channel]);
        multipliers[channel] = vdupq_n_f32(scale[channel]);
    }
    for (size_t block = 0; block < blocks; ++block) {
        const uint8x16x3_t pixels = vld3q_u8(source + 48 * block);
        for (size_t channel = 0; channel < 3; ++channel) {
            const FourVectors output =
              Normalize16(pixels.val[channel], offsets[channel],
                          multipliers[channel]);
            for (size_t group = 0; group < 4; ++group) {
                vst1q_f32(planes[channel] + 16 * block + 4 * group,
                          output.value[group]);
            }
        }
    }
    completed = blocks * 16;
#endif
    for (size_t pixel = completed; pixel < count; ++pixel) {
        for (size_t channel = 0; channel < 3; ++channel) {
            planes[channel][pixel] =
              Normalize(source[3 * pixel + channel], mean[channel], scale[channel]);
        }
    }
}

void InterleavedQuad(const uint8_t* source, float* destination,
                     const float* mean, const float* scale, size_t count) {
    for (size_t pixel = 0; pixel < count; ++pixel) {
        for (size_t channel = 0; channel < 4; ++channel) {
            destination[4 * pixel + channel] =
              Normalize(source[4 * pixel + channel], mean[channel], scale[channel]);
        }
    }
}

void QuadFromMono(const uint8_t* source, float* destination, const float* mean,
                  const float* scale, size_t count) {
#if defined(INSPIRECV_TASK_USE_SSE) && defined(__SSE4_1__)
    const size_t completed =
      WriteMonoQuadSse(source, destination, mean[0], scale[0], count);
    for (size_t pixel = completed; pixel < count; ++pixel) {
        destination[4 * pixel] = Normalize(source[pixel], mean[0], scale[0]);
    }
#elif defined(INSPIRECV_TASK_USE_NEON)
    inspirecv_task_c1_to_float_c4_arm(source, destination, mean, scale, count);
#else
    std::memset(destination, 0, 4 * sizeof(float) * count);
    for (size_t pixel = 0; pixel < count; ++pixel) {
        destination[4 * pixel] = Normalize(source[pixel], mean[0], scale[0]);
    }
#endif
}

void QuadFromTriple(const uint8_t* source, float* destination, const float* mean,
                    const float* scale, size_t count) {
#if defined(INSPIRECV_TASK_USE_SSE) && defined(__SSE4_1__)
    const size_t completed =
      WriteTripleQuadSse(source, destination, mean, scale, count);
    for (size_t pixel = completed; pixel < count; ++pixel) {
        for (size_t channel = 0; channel < 3; ++channel) {
            destination[4 * pixel + channel] = Normalize(
              source[3 * pixel + channel], mean[channel], scale[channel]);
        }
        destination[4 * pixel + 3] = 0.0f;
    }
#elif defined(INSPIRECV_TASK_USE_NEON)
    inspirecv_task_c3_to_float_c4_arm(source, destination, mean, scale, count);
#else
    for (size_t pixel = 0; pixel < count; ++pixel) {
        for (size_t channel = 0; channel < 3; ++channel) {
            destination[4 * pixel + channel] = Normalize(
              source[3 * pixel + channel], mean[channel], scale[channel]);
        }
        destination[4 * pixel + 3] = 0.0f;
    }
#endif
}

}  // namespace tensor
}  // namespace kernels
}  // namespace task
}  // namespace inspirecv
