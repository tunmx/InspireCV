#include "inspirecv/task/kernels/x86/avx2_kernels.h"
#include "inspirecv/task/kernels/x86/block_utils.h"

#include <cmath>
#include <cstring>

namespace inspirecv {
namespace task {
namespace kernels {
namespace x86 {
namespace {

// std::min/max<float> COMDATs compiled with AVX2 could be reused by baseline
// callers. Keep the helpers local, including their original operand ordering.
template<typename T> T LocalMin(T a, T b) { return b < a ? b : a; }
template<typename T> T LocalMax(T a, T b) { return a < b ? b : a; }

struct Cursor {
    float x, y, dx, dy;
    void Advance() { x += dx; y += dy; }
};

void Coordinates8(Cursor* cursor, __m256* xs, __m256* ys) {
    // Preserve the scalar recurrence exactly. Multiplying the step by the lane
    // or block index can move a coordinate across a nearest-neighbour tie.
    // Advance x/y together in two packed lanes and assemble the coordinates in
    // registers, avoiding sixteen scalar stack stores followed by two reloads.
    const __m128 step = _mm_setr_ps(cursor->dx, cursor->dy, 0.0f, 0.0f);
    const __m128 xy0 = _mm_setr_ps(cursor->x, cursor->y, 0.0f, 0.0f);
    const __m128 xy1 = _mm_add_ps(xy0, step);
    const __m128 xy2 = _mm_add_ps(xy1, step);
    const __m128 xy3 = _mm_add_ps(xy2, step);
    const __m128 xy4 = _mm_add_ps(xy3, step);
    const __m128 xy5 = _mm_add_ps(xy4, step);
    const __m128 xy6 = _mm_add_ps(xy5, step);
    const __m128 xy7 = _mm_add_ps(xy6, step);
    const __m128 xy8 = _mm_add_ps(xy7, step);
    const __m128 pairs01 = _mm_unpacklo_ps(xy0, xy1);
    const __m128 pairs23 = _mm_unpacklo_ps(xy2, xy3);
    const __m128 pairs45 = _mm_unpacklo_ps(xy4, xy5);
    const __m128 pairs67 = _mm_unpacklo_ps(xy6, xy7);
    *xs = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_movelh_ps(pairs01, pairs23)),
                              _mm_movelh_ps(pairs45, pairs67), 1);
    *ys = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_movehl_ps(pairs23, pairs01)),
                              _mm_movehl_ps(pairs67, pairs45), 1);
    cursor->x = _mm_cvtss_f32(xy8);
    cursor->y = _mm_cvtss_f32(_mm_shuffle_ps(xy8, xy8, _MM_SHUFFLE(1, 1, 1, 1)));
}

__m256i RoundPositive8(__m256 value) {
    const __m256i integer = _mm256_cvttps_epi32(value);
    const __m256 fraction = _mm256_sub_ps(value, _mm256_cvtepi32_ps(integer));
    const __m256i increment = _mm256_and_si256(
      _mm256_castps_si256(_mm256_cmp_ps(fraction, _mm256_set1_ps(0.5f), _CMP_GE_OQ)),
      _mm256_set1_epi32(1));
    return _mm256_add_epi32(integer, increment);
}

__m256 HorizontalCoordinates8(float* cursor, float step) {
    const __m128 delta = _mm_set_ss(step);
    const __m128 x0 = _mm_set_ss(*cursor);
    const __m128 x1 = _mm_add_ss(x0, delta);
    const __m128 x2 = _mm_add_ss(x1, delta);
    const __m128 x3 = _mm_add_ss(x2, delta);
    const __m128 x4 = _mm_add_ss(x3, delta);
    const __m128 x5 = _mm_add_ss(x4, delta);
    const __m128 x6 = _mm_add_ss(x5, delta);
    const __m128 x7 = _mm_add_ss(x6, delta);
    *cursor = _mm_cvtss_f32(_mm_add_ss(x7, delta));
    const __m128 low = _mm_movelh_ps(_mm_unpacklo_ps(x0, x1), _mm_unpacklo_ps(x2, x3));
    const __m128 high = _mm_movelh_ps(_mm_unpacklo_ps(x4, x5), _mm_unpacklo_ps(x6, x7));
    return _mm256_insertf128_ps(_mm256_castps128_ps256(low), high, 1);
}

void NearestQuadHorizontal(const uint8_t* source, uint8_t* destination, const Point* line,
                            size_t count, size_t width, size_t height, size_t stride) {
    if (count == 0) return;
    // A horizontal scanline has one source row. Keep the same bounded nearest
    // index, but move its clamp/round and stride multiplication out of the loop.
    const float maximum_y = static_cast<float>(height - 1);
    const __m128 bounded_y = _mm_min_ss(
      _mm_max_ss(_mm_set_ss(line[0].fY), _mm_setzero_ps()), _mm_set_ss(maximum_y));
    const size_t row_index = static_cast<size_t>(::roundf(_mm_cvtss_f32(bounded_y)));
    const uint8_t* row = source + row_index * stride;
    const float maximum_x_scalar = static_cast<float>(width - 1);
    const __m256 maximum_x = _mm256_set1_ps(maximum_x_scalar);
    const __m256 zero = _mm256_setzero_ps();
    float x = line[0].fX;
    const float step = line[1].fX;
    while (count >= 8) {
        // Every x still advances through eight separate float additions. In
        // particular, neither lane*step nor block*step replaces the recurrence.
        const __m256 xs = _mm256_min_ps(
          _mm256_max_ps(HorizontalCoordinates8(&x, step), zero), maximum_x);
        const __m256i indices = RoundPositive8(xs);
        // Each C4 pixel is exactly one word. The existing dispatcher bounds
        // the image offsets; scale4 addresses the same words as the general
        // gather without building eight identical y*stride values.
        const __m256i pixels = _mm256_i32gather_epi32(reinterpret_cast<const int*>(row), indices, 4);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(destination), pixels);
        destination += 32;
        count -= 8;
    }
    while (count-- != 0) {
        const float bounded_x = LocalMin(LocalMax(x, 0.0f), maximum_x_scalar);
        const size_t column = static_cast<size_t>(::roundf(bounded_x));
        std::memcpy(destination, row + 4 * column, 4);
        destination += 4;
        x += step;
    }
}

template <int kChannels>
__m256i GatherPixels8(const uint8_t* source, __m256i x, __m256i y,
                      int width, int stride) {
    __m256i start = x;
    if (kChannels == 1) start = _mm256_min_epi32(x, _mm256_set1_epi32(width - 4));
    if (kChannels == 3) {
        // RGB has only three bytes. For a last-column pixel, load its preceding
        // byte and shift it away instead of reading past the final image row.
        const __m256i last = _mm256_cmpeq_epi32(x, _mm256_set1_epi32(width - 1));
        start = _mm256_sub_epi32(_mm256_mullo_epi32(x, _mm256_set1_epi32(3)),
                                  _mm256_and_si256(last, _mm256_set1_epi32(1)));
    } else if (kChannels == 4) {
        start = _mm256_slli_epi32(x, 2);
    }
    const __m256i offset = _mm256_add_epi32(start, _mm256_mullo_epi32(y, _mm256_set1_epi32(stride)));
    __m256i pixels = _mm256_i32gather_epi32(reinterpret_cast<const int*>(source), offset, 1);
    if (kChannels == 1) {
        pixels = _mm256_srlv_epi32(pixels, _mm256_slli_epi32(_mm256_sub_epi32(x, start), 3));
        return _mm256_and_si256(pixels, _mm256_set1_epi32(255));
    }
    if (kChannels == 3) {
        const __m256i last = _mm256_cmpeq_epi32(x, _mm256_set1_epi32(width - 1));
        pixels = _mm256_srlv_epi32(pixels, _mm256_and_si256(last, _mm256_set1_epi32(8)));
        return _mm256_and_si256(pixels, _mm256_set1_epi32(0x00ffffff));
    }
    return pixels;
}

template <int kChannels>
void StorePixels8(uint8_t* destination, __m256i pixels) {
    if (kChannels == 4) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(destination), pixels);
    } else if (kChannels == 3) {
        detail::StoreTriple8(destination, pixels);
    } else {
        const __m128i words = _mm_packus_epi32(_mm256_castsi256_si128(pixels),
                                               _mm256_extracti128_si256(pixels, 1));
        _mm_storel_epi64(reinterpret_cast<__m128i*>(destination), _mm_packus_epi16(words, words));
    }
}

__m256 Interpolate8(__m256 c00, __m256 c01, __m256 c10, __m256 c11,
                     __m256 fx, __m256 fy) {
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 inv_x = _mm256_sub_ps(one, fx);
    const __m256 inv_y = _mm256_sub_ps(one, fy);
    const __m256 top = _mm256_add_ps(
      _mm256_mul_ps(_mm256_mul_ps(inv_x, inv_y), c00),
      _mm256_mul_ps(_mm256_mul_ps(fx, inv_y), c01));
    const __m256 bottom_right = _mm256_mul_ps(_mm256_mul_ps(fx, fy), c11);
    // The established scalar formula contains a double literal in its third
    // term. Match that mixed-precision expression and addition order exactly.
    const __m256d one_d = _mm256_set1_pd(1.0);
    const __m256d left_low = _mm256_mul_pd(
      _mm256_mul_pd(_mm256_cvtps_pd(_mm256_castps256_ps128(fy)),
                      _mm256_sub_pd(one_d, _mm256_cvtps_pd(_mm256_castps256_ps128(fx)))),
      _mm256_cvtps_pd(_mm256_castps256_ps128(c10)));
    const __m256d left_high = _mm256_mul_pd(
      _mm256_mul_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(fy, 1)),
                      _mm256_sub_pd(one_d, _mm256_cvtps_pd(_mm256_extractf128_ps(fx, 1)))),
      _mm256_cvtps_pd(_mm256_extractf128_ps(c10, 1)));
    const __m128 low = _mm256_cvtpd_ps(_mm256_add_pd(
      _mm256_add_pd(_mm256_cvtps_pd(_mm256_castps256_ps128(top)), left_low),
      _mm256_cvtps_pd(_mm256_castps256_ps128(bottom_right))));
    const __m128 high = _mm256_cvtpd_ps(_mm256_add_pd(
      _mm256_add_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(top, 1)), left_high),
      _mm256_cvtps_pd(_mm256_extractf128_ps(bottom_right, 1))));
    return _mm256_insertf128_ps(_mm256_castps128_ps256(low), high, 1);
}

template <int kChannels, bool kBilinear>
void Sample(const uint8_t* source, uint8_t* destination, Point* line,
             size_t first, size_t count, size_t width, size_t height, size_t stride) {
    destination += kChannels * first;
    if (kChannels == 4 && !kBilinear && line[1].fY == 0.0f) {
        NearestQuadHorizontal(source, destination, line, count, width, height, stride);
        return;
    }
    Cursor cursor = {line[0].fX, line[0].fY, line[1].fX, line[1].fY};
    const __m256 zero = _mm256_setzero_ps();
    const __m256 maximum_x = _mm256_set1_ps(static_cast<float>(width - 1));
    const __m256 maximum_y = _mm256_set1_ps(static_cast<float>(height - 1));
    size_t remaining = count;
    while (remaining >= 8) {
        __m256 xs, ys;
        Coordinates8(&cursor, &xs, &ys);
        xs = _mm256_min_ps(_mm256_max_ps(xs, zero), maximum_x);
        ys = _mm256_min_ps(_mm256_max_ps(ys, zero), maximum_y);
        __m256i output;
        if (!kBilinear) {
            output = GatherPixels8<kChannels>(source, RoundPositive8(xs), RoundPositive8(ys),
                                               static_cast<int>(width), static_cast<int>(stride));
        } else {
            const __m256i x0 = _mm256_cvttps_epi32(xs);
            const __m256i y0 = _mm256_cvttps_epi32(ys);
            const __m256 fx = _mm256_sub_ps(xs, _mm256_cvtepi32_ps(x0));
            const __m256 fy = _mm256_sub_ps(ys, _mm256_cvtepi32_ps(y0));
            const __m256i x1 = _mm256_sub_epi32(x0, _mm256_castps_si256(_mm256_cmp_ps(fx, zero, _CMP_GT_OQ)));
            const __m256i y1 = _mm256_sub_epi32(y0, _mm256_castps_si256(_mm256_cmp_ps(fy, zero, _CMP_GT_OQ)));
            const __m256i p00 = GatherPixels8<kChannels>(source, x0, y0, static_cast<int>(width), static_cast<int>(stride));
            const __m256i p01 = GatherPixels8<kChannels>(source, x1, y0, static_cast<int>(width), static_cast<int>(stride));
            const __m256i p10 = GatherPixels8<kChannels>(source, x0, y1, static_cast<int>(width), static_cast<int>(stride));
            const __m256i p11 = GatherPixels8<kChannels>(source, x1, y1, static_cast<int>(width), static_cast<int>(stride));
            output = _mm256_setzero_si256();
            const __m256i byte_mask = _mm256_set1_epi32(255);
            for (int channel = 0; channel < kChannels; ++channel) {
                const __m256i shift = _mm256_set1_epi32(8 * channel);
                const __m256 value = Interpolate8(
                  _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srlv_epi32(p00, shift), byte_mask)),
                  _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srlv_epi32(p01, shift), byte_mask)),
                  _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srlv_epi32(p10, shift), byte_mask)),
                  _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srlv_epi32(p11, shift), byte_mask)), fx, fy);
                const __m256 bounded = _mm256_min_ps(_mm256_max_ps(value, zero), _mm256_set1_ps(255.0f));
                output = _mm256_or_si256(output, _mm256_sllv_epi32(RoundPositive8(bounded), shift));
            }
        }
        StorePixels8<kChannels>(destination, output);
        destination += 8 * kChannels;
        remaining -= 8;
    }
    while (remaining-- != 0) {
        const float x = LocalMin(LocalMax(cursor.x, 0.0f), static_cast<float>(width - 1));
        const float y = LocalMin(LocalMax(cursor.y, 0.0f), static_cast<float>(height - 1));
        if (!kBilinear) {
            const size_t offset = static_cast<size_t>(roundf(y)) * stride +
                                  kChannels * static_cast<size_t>(roundf(x));
            std::memcpy(destination, source + offset, kChannels);
        } else {
            const int x0 = static_cast<int>(x), y0 = static_cast<int>(y);
            const int x1 = static_cast<int>(ceilf(x)), y1 = static_cast<int>(ceilf(y));
            const float fx = x - static_cast<float>(x0), fy = y - static_cast<float>(y0);
            const size_t offsets[4] = {static_cast<size_t>(y0) * stride + kChannels * x0,
                                       static_cast<size_t>(y0) * stride + kChannels * x1,
                                       static_cast<size_t>(y1) * stride + kChannels * x0,
                                       static_cast<size_t>(y1) * stride + kChannels * x1};
            for (int channel = 0; channel < kChannels; ++channel) {
                float value = (1.0f - fx) * (1.0f - fy) * source[offsets[0] + channel] +
                              fx * (1.0f - fy) * source[offsets[1] + channel] +
                              fy * (1.0 - fx) * source[offsets[2] + channel] +
                              fx * fy * source[offsets[3] + channel];
                value = LocalMin(LocalMax(value, 0.0f), 255.0f);
                destination[channel] = static_cast<uint8_t>(roundf(value));
            }
        }
        destination += kChannels;
        cursor.Advance();
    }
}

}  // namespace

#define INSPIRECV_DEFINE_AVX2_SAMPLER(name, channels, bilinear) \
void name(const uint8_t* source, uint8_t* destination, Point* line, \
           size_t first, size_t count, size_t, size_t width, size_t height, size_t stride) { \
    Sample<channels, bilinear>(source, destination, line, first, count, width, height, stride); \
}
INSPIRECV_DEFINE_AVX2_SAMPLER(NearestMonoAvx2, 1, false)
INSPIRECV_DEFINE_AVX2_SAMPLER(NearestTripleAvx2, 3, false)
INSPIRECV_DEFINE_AVX2_SAMPLER(NearestQuadAvx2, 4, false)
INSPIRECV_DEFINE_AVX2_SAMPLER(BilinearMonoAvx2, 1, true)
INSPIRECV_DEFINE_AVX2_SAMPLER(BilinearTripleAvx2, 3, true)
INSPIRECV_DEFINE_AVX2_SAMPLER(BilinearQuadAvx2, 4, true)
#undef INSPIRECV_DEFINE_AVX2_SAMPLER

}  // namespace x86
}  // namespace kernels
}  // namespace task
}  // namespace inspirecv
