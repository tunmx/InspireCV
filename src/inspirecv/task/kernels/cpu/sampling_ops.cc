#include "inspirecv/task/kernels/cpu/sampling_ops.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "inspirecv/task/platform/cpu_features.h"

#if defined(INSPIRECV_TASK_USE_NEON)
#include <arm_neon.h>
#endif

#if defined(INSPIRECV_TASK_USE_SSE) && defined(__SSE4_1__)
#include <x86intrin.h>
#endif

namespace inspirecv {
namespace task {
namespace kernels {
namespace sampling {
namespace {

struct CoordinateCursor {
    float x;
    float y;
    float dx;
    float dy;

    void Advance() {
        x += dx;
        y += dy;
    }
};

template <typename T>
T Limit(T value, T low, T high) {
    return value < low ? low : (value > high ? high : value);
}

CoordinateCursor MakeCursor(const Point* line) {
    return {line[0].fX, line[0].fY, line[1].fX, line[1].fY};
}

template <size_t kChannels>
void DirectPacked(const uint8_t* source, uint8_t* destination, Point* line,
                  size_t first, size_t count, size_t /*capacity*/, size_t width,
                  size_t height, size_t stride) {
    const int x = static_cast<int>(roundf(Limit(line[0].fX, 0.0f,
                                                static_cast<float>(width - 1))));
    const int y = static_cast<int>(roundf(Limit(line[0].fY, 0.0f,
                                                static_cast<float>(height - 1))));
    const size_t source_offset = static_cast<size_t>(y) * stride + kChannels * x;
    std::memcpy(destination + kChannels * first, source + source_offset,
                kChannels * count);
}

template <size_t kChannels>
void NearestScalar(const uint8_t* source, uint8_t* destination, Point* line,
                   size_t first, size_t count, size_t width, size_t height,
                   size_t stride) {
    destination += kChannels * first;
    CoordinateCursor cursor = MakeCursor(line);
    const float maximum_x = static_cast<float>(width - 1);
    const float maximum_y = static_cast<float>(height - 1);
    while (count-- != 0) {
        const int x = static_cast<int>(roundf(Limit(cursor.x, 0.0f, maximum_x)));
        const int y = static_cast<int>(roundf(Limit(cursor.y, 0.0f, maximum_y)));
        const uint8_t* pixel = source + static_cast<size_t>(y) * stride + kChannels * x;
        for (size_t channel = 0; channel < kChannels; ++channel) {
            destination[channel] = pixel[channel];
        }
        destination += kChannels;
        cursor.Advance();
    }
}

template <size_t kChannels>
void BilinearScalar(const uint8_t* source, uint8_t* destination, Point* line,
                    size_t count, size_t width, size_t height, size_t stride) {
    CoordinateCursor cursor = MakeCursor(line);
    const float maximum_x = static_cast<float>(width - 1);
    const float maximum_y = static_cast<float>(height - 1);
    while (count-- != 0) {
        const float x = Limit(cursor.x, 0.0f, maximum_x);
        const float y = Limit(cursor.y, 0.0f, maximum_y);
        const int x0 = static_cast<int>(x);
        const int y0 = static_cast<int>(y);
        const int x1 = static_cast<int>(ceilf(x));
        const int y1 = static_cast<int>(ceilf(y));
        const float fraction_x = x - static_cast<float>(x0);
        const float fraction_y = y - static_cast<float>(y0);
        const size_t offsets[4] = {
          static_cast<size_t>(y0) * stride + kChannels * x0,
          static_cast<size_t>(y0) * stride + kChannels * x1,
          static_cast<size_t>(y1) * stride + kChannels * x0,
          static_cast<size_t>(y1) * stride + kChannels * x1};
        for (size_t channel = 0; channel < kChannels; ++channel) {
            const uint8_t c00 = source[offsets[0] + channel];
            const uint8_t c01 = source[offsets[1] + channel];
            const uint8_t c10 = source[offsets[2] + channel];
            const uint8_t c11 = source[offsets[3] + channel];
            float value = (1.0f - fraction_x) * (1.0f - fraction_y) * c00 +
                          fraction_x * (1.0f - fraction_y) * c01 +
                          fraction_y * (1.0 - fraction_x) * c10 +
                          fraction_x * fraction_y * c11;
            value = Limit(value, 0.0f, 255.0f);
            destination[channel] = static_cast<uint8_t>(roundf(value));
        }
        destination += kChannels;
        cursor.Advance();
    }
}

void BilinearTripleInterior(const uint8_t* source, uint8_t* destination,
                            Point* line, size_t count, size_t width,
                            size_t height, size_t stride) {
    CoordinateCursor cursor = MakeCursor(line);
    const int maximum_x = static_cast<int>(width - 1);
    const int maximum_y = static_cast<int>(height - 1);
    while (count-- != 0) {
        const int x0 = std::max(0, std::min(static_cast<int>(cursor.x), maximum_x));
        const int y0 = std::max(0, std::min(static_cast<int>(cursor.y), maximum_y));
        float fraction_x = cursor.x - static_cast<float>(x0);
        float fraction_y = cursor.y - static_cast<float>(y0);
        if ((x0 == 0 && cursor.x < 0.0f) ||
            (x0 == maximum_x && cursor.x > static_cast<float>(maximum_x))) {
            fraction_x = 0.0f;
        }
        if ((y0 == 0 && cursor.y < 0.0f) ||
            (y0 == maximum_y && cursor.y > static_cast<float>(maximum_y))) {
            fraction_y = 0.0f;
        }
        const int x1 = std::min(x0 + (fraction_x > 0.0f ? 1 : 0), maximum_x);
        const int y1 = std::min(y0 + (fraction_y > 0.0f ? 1 : 0), maximum_y);
        const size_t offsets[4] = {
          static_cast<size_t>(y0) * stride + 3 * x0,
          static_cast<size_t>(y0) * stride + 3 * x1,
          static_cast<size_t>(y1) * stride + 3 * x0,
          static_cast<size_t>(y1) * stride + 3 * x1};
        for (int channel = 0; channel < 3; ++channel) {
            float value =
              (1.0f - fraction_x) * (1.0f - fraction_y) *
                source[offsets[0] + channel] +
              fraction_x * (1.0f - fraction_y) * source[offsets[1] + channel] +
              fraction_y * (1.0 - fraction_x) * source[offsets[2] + channel] +
              fraction_x * fraction_y * source[offsets[3] + channel];
            value = Limit(value, 0.0f, 255.0f);
            destination[channel] = static_cast<uint8_t>(roundf(value));
        }
        destination += 3;
        cursor.Advance();
    }
}

#if defined(INSPIRECV_TASK_USE_SSE) && defined(__SSE4_1__)
void NearestQuadSse(const uint8_t* source, uint8_t* destination, Point* line,
                    size_t first, size_t count, size_t width, size_t height,
                    size_t stride) {
    destination += 4 * first;
    CoordinateCursor cursor = MakeCursor(line);
    const float maximum_x = static_cast<float>(width - 1);
    const float maximum_y = static_cast<float>(height - 1);
    const size_t blocks = count / 4;
    const __m128 zero = _mm_set1_ps(0.0f);
    const __m128 half = _mm_set1_ps(0.5f);
    const __m128 negative_half = _mm_set1_ps(-0.5f);
    const __m128 max_x = _mm_set1_ps(maximum_x);
    const __m128 max_y = _mm_set1_ps(maximum_y);
    const __m128i stride4 = _mm_set1_epi32(static_cast<int>(stride));
    const __m128i channels4 = _mm_set1_epi32(4);
    for (size_t block = 0; block < blocks; ++block) {
        __m128 xs = _mm_set_ps(cursor.x + 3 * cursor.dx, cursor.x + 2 * cursor.dx,
                               cursor.x + cursor.dx, cursor.x);
        __m128 ys = _mm_set_ps(cursor.y + 3 * cursor.dy, cursor.y + 2 * cursor.dy,
                               cursor.y + cursor.dy, cursor.y);
        xs = _mm_min_ps(_mm_max_ps(xs, zero), max_x);
        ys = _mm_min_ps(_mm_max_ps(ys, zero), max_y);
        const __m128 x_bias = _mm_blendv_ps(half, negative_half, _mm_cmplt_ps(xs, zero));
        const __m128 y_bias = _mm_blendv_ps(half, negative_half, _mm_cmplt_ps(ys, zero));
        const __m128i xi = _mm_cvtps_epi32(_mm_round_ps(_mm_add_ps(xs, x_bias), 3));
        const __m128i yi = _mm_cvtps_epi32(_mm_round_ps(_mm_add_ps(ys, y_bias), 3));
        const __m128i offsets =
          _mm_add_epi32(_mm_mullo_epi32(yi, stride4),
                        _mm_mullo_epi32(xi, channels4));
        alignas(16) int32_t position[4];
        _mm_store_si128(reinterpret_cast<__m128i*>(position), offsets);
        for (int pixel = 0; pixel < 4; ++pixel) {
            std::memcpy(destination + 4 * (4 * block + pixel),
                        source + position[pixel], 4);
        }
        cursor.x += 4 * cursor.dx;
        cursor.y += 4 * cursor.dy;
    }
    const size_t completed = blocks * 4;
    Point tail[2] = {{cursor.x, cursor.y}, {cursor.dx, cursor.dy}};
    NearestScalar<4>(source, destination + 4 * completed, tail, 0,
                     count - completed, width, height, stride);
}

void BilinearQuadSse(const uint8_t* source, uint8_t* destination, Point* line,
                     size_t count, size_t width, size_t height, size_t stride) {
    CoordinateCursor cursor = MakeCursor(line);
    const float maximum_x = static_cast<float>(width - 1);
    const float maximum_y = static_cast<float>(height - 1);
    const __m128i zero = _mm_setzero_si128();
    while (count-- != 0) {
        const float x = Limit(cursor.x, 0.0f, maximum_x);
        const float y = Limit(cursor.y, 0.0f, maximum_y);
        const int x0 = static_cast<int>(x);
        const int y0 = static_cast<int>(y);
        const int x1 = static_cast<int>(ceilf(x));
        const int y1 = static_cast<int>(ceilf(y));
        const float fx = x - static_cast<float>(x0);
        const float fy = y - static_cast<float>(y0);
        const int offsets[4] = {
          y0 * static_cast<int>(stride) + 4 * x0,
          y0 * static_cast<int>(stride) + 4 * x1,
          y1 * static_cast<int>(stride) + 4 * x0,
          y1 * static_cast<int>(stride) + 4 * x1};
        const __m128 weights[4] = {
          _mm_set1_ps((1.0f - fx) * (1.0f - fy)),
          _mm_set1_ps(fx * (1.0f - fy)),
          _mm_set1_ps(fy * (1.0f - fx)),
          _mm_set1_ps(fx * fy)};
        __m128 value = _mm_setzero_ps();
        for (int corner = 0; corner < 4; ++corner) {
            int32_t packed;
            std::memcpy(&packed, source + offsets[corner], sizeof(packed));
            const __m128i bytes = _mm_cvtsi32_si128(packed);
            const __m128i words = _mm_unpacklo_epi8(bytes, zero);
            const __m128i integers = _mm_unpacklo_epi16(words, zero);
            value = _mm_add_ps(value,
                               _mm_mul_ps(weights[corner], _mm_cvtepi32_ps(integers)));
        }
        __m128i rounded = _mm_cvtps_epi32(_mm_round_ps(value, 3));
        rounded = _mm_packs_epi32(rounded, rounded);
        rounded = _mm_packus_epi16(rounded, rounded);
        const int32_t packed = _mm_cvtsi128_si32(rounded);
        std::memcpy(destination, &packed, sizeof(packed));
        destination += 4;
        cursor.Advance();
    }
}
#endif

#if defined(INSPIRECV_TASK_USE_NEON)
extern "C" {
void inspirecv_task_sample_c4_bilinear_arm(const uint8_t*, uint8_t*, float*, size_t,
                                            size_t, size_t, size_t);
void inspirecv_task_sample_c1_bilinear_arm(const uint8_t*, uint8_t*, float*, size_t,
                                            size_t, size_t, size_t);
void inspirecv_task_sample_c4_nearest_arm(const uint8_t*, uint8_t*, float*, size_t,
                                           size_t, size_t, size_t);
void inspirecv_task_sample_c1_nearest_arm(const uint8_t*, uint8_t*, float*, size_t,
                                           size_t, size_t, size_t);
}
#endif

struct ChromaTarget {
    uint8_t* luma;
    uint8_t* vu;
};

ChromaTarget TargetPlanes(uint8_t* destination, size_t first, size_t capacity) {
    return {destination + first, destination + capacity + (first / 2) * 2};
}

template <bool kSwapUv>
void StoreSemiPlanarPairs(const uint8_t* source, uint8_t* destination, size_t pairs) {
    if (!kSwapUv) {
        std::memcpy(destination, source, pairs * 2);
        return;
    }
#if defined(INSPIRECV_TASK_USE_NEON)
    while (pairs >= 16) {
        const uint8x16x2_t uv = vld2q_u8(source);
        const uint8x16x2_t vu = {{uv.val[1], uv.val[0]}};
        vst2q_u8(destination, vu);
        source += 32;
        destination += 32;
        pairs -= 16;
    }
#endif
    while (pairs-- != 0) {
        destination[0] = source[1];
        destination[1] = source[0];
        source += 2;
        destination += 2;
    }
}

template <bool kSwapUv>
void DirectSemiPlanar(const uint8_t* source, uint8_t* destination, Point* line,
                      size_t first, size_t count, size_t capacity, size_t width,
                      size_t height, size_t source_stride) {
    const int x = static_cast<int>(roundf(Limit(line[0].fX, 0.0f,
                                                static_cast<float>(width - 1))));
    const int y = static_cast<int>(roundf(Limit(line[0].fY, 0.0f,
                                                static_cast<float>(height - 1))));
    const size_t stride = source_stride == 0 ? width : source_stride;
    const size_t chroma_height = std::max<size_t>(1, height / 2);
    const size_t chroma_y = std::min<size_t>(y / 2, chroma_height - 1);
    const size_t pairs = (count + 1) / 2;
    const size_t chroma_bytes = stride * chroma_height;
    const size_t chroma_offset = chroma_y * stride + static_cast<size_t>(x / 2) * 2;
    const uint8_t* chroma = source + stride * height;
    ChromaTarget target = TargetPlanes(destination, first, capacity);
    std::memcpy(target.luma, source + static_cast<size_t>(y) * stride + x, count);

    if (chroma_bytes >= 2 && chroma_offset <= chroma_bytes &&
        pairs * 2 <= chroma_bytes - chroma_offset) {
        StoreSemiPlanarPairs<kSwapUv>(chroma + chroma_offset, target.vu, pairs);
        return;
    }
    if (chroma_bytes < 2) return;
    const size_t last_pair = chroma_bytes - 2;
    for (size_t pair = 0; pair < pairs; ++pair) {
        const size_t offset = std::min(chroma_offset + 2 * pair, last_pair);
        target.vu[2 * pair] = chroma[offset + (kSwapUv ? 1 : 0)];
        target.vu[2 * pair + 1] = chroma[offset + (kSwapUv ? 0 : 1)];
    }
}

template <bool kSwapUv>
void NearestSemiPlanar(const uint8_t* source, uint8_t* destination, Point* line,
                       size_t first, size_t count, size_t capacity, size_t width,
                       size_t height, size_t source_stride) {
    const size_t stride = source_stride == 0 ? width : source_stride;
    ChromaTarget target = TargetPlanes(destination, first, capacity);
    NearestMono(source, target.luma, line, 0, count, capacity, width, height, stride);

    const size_t chroma_height = std::max<size_t>(1, height / 2);
    const size_t chroma_bytes = stride * chroma_height;
    const size_t pairs = (count + 1) / 2;
    const uint8_t* chroma = source + stride * height;
    CoordinateCursor cursor = {(line[0].fX - 0.01f) / 2.0f,
                               (line[0].fY - 0.01f) / 2.0f,
                               line[1].fX, line[1].fY};
    const float maximum_x = static_cast<float>((width + 1) / 2 - 1);
    const float maximum_y = static_cast<float>(chroma_height - 1);
    if (chroma_bytes < 2) return;
    const size_t last_pair = chroma_bytes - 2;
    for (size_t pair = 0; pair < pairs; ++pair) {
        const int x = static_cast<int>(roundf(Limit(cursor.x, 0.0f, maximum_x)));
        const int y = static_cast<int>(roundf(Limit(cursor.y, 0.0f, maximum_y)));
        const size_t offset = std::min(static_cast<size_t>(y) * stride + 2 * x,
                                       last_pair);
        target.vu[2 * pair] = chroma[offset + (kSwapUv ? 1 : 0)];
        target.vu[2 * pair + 1] = chroma[offset + (kSwapUv ? 0 : 1)];
        cursor.Advance();
    }
}

}  // namespace

void DirectMono(const uint8_t* source, uint8_t* destination, Point* line,
                size_t first, size_t count, size_t capacity, size_t width,
                size_t height, size_t stride) {
    DirectPacked<1>(source, destination, line, first, count, capacity, width, height, stride);
}

void DirectTriple(const uint8_t* source, uint8_t* destination, Point* line,
                  size_t first, size_t count, size_t capacity, size_t width,
                  size_t height, size_t stride) {
    DirectPacked<3>(source, destination, line, first, count, capacity, width, height, stride);
}

void DirectQuad(const uint8_t* source, uint8_t* destination, Point* line,
                size_t first, size_t count, size_t capacity, size_t width,
                size_t height, size_t stride) {
    DirectPacked<4>(source, destination, line, first, count, capacity, width, height, stride);
}

void NearestMono(const uint8_t* source, uint8_t* destination, Point* line,
                 size_t first, size_t count, size_t /*capacity*/, size_t width,
                 size_t height, size_t stride) {
#if defined(INSPIRECV_TASK_USE_NEON)
    inspirecv_task_sample_c1_nearest_arm(source, destination + first,
                                         reinterpret_cast<float*>(line), count,
                                         width - 1, height - 1, stride);
#else
    NearestScalar<1>(source, destination, line, first, count, width, height, stride);
#endif
}

void NearestTriple(const uint8_t* source, uint8_t* destination, Point* line,
                   size_t first, size_t count, size_t /*capacity*/, size_t width,
                   size_t height, size_t stride) {
    NearestScalar<3>(source, destination, line, first, count, width, height, stride);
}

void NearestQuad(const uint8_t* source, uint8_t* destination, Point* line,
                 size_t first, size_t count, size_t /*capacity*/, size_t width,
                 size_t height, size_t stride) {
#if defined(INSPIRECV_TASK_USE_SSE) && defined(__SSE4_1__)
    if (platform::HasSse41()) {
        NearestQuadSse(source, destination, line, first, count, width, height, stride);
        return;
    }
#endif
#if defined(INSPIRECV_TASK_USE_NEON)
    inspirecv_task_sample_c4_nearest_arm(source, destination + 4 * first,
                                         reinterpret_cast<float*>(line), count,
                                         width - 1, height - 1, stride);
#else
    NearestScalar<4>(source, destination, line, first, count, width, height, stride);
#endif
}

void BilinearMono(const uint8_t* source, uint8_t* destination, Point* line,
                  size_t first, size_t count, size_t /*capacity*/, size_t width,
                  size_t height, size_t stride) {
#if defined(INSPIRECV_TASK_USE_NEON)
    inspirecv_task_sample_c1_bilinear_arm(source, destination + first,
                                          reinterpret_cast<float*>(line), count,
                                          width - 1, height - 1, stride);
#else
    BilinearScalar<1>(source, destination + first, line, count, width, height, stride);
#endif
}

void BilinearTriple(const uint8_t* source, uint8_t* destination, Point* line,
                    size_t first, size_t count, size_t /*capacity*/, size_t width,
                    size_t height, size_t stride) {
    destination += 3 * first;
#if defined(INSPIRECV_TASK_USE_SSE)
    BilinearScalar<3>(source, destination, line, count, width, height, stride);
    return;
#endif
    if (count != 0) {
        const float last = static_cast<float>(count - 1);
        const float last_x = line[0].fX + last * line[1].fX;
        const float last_y = line[0].fY + last * line[1].fY;
        if (std::min(line[0].fX, last_x) >= 0.0f &&
            std::min(line[0].fY, last_y) >= 0.0f &&
            std::max(line[0].fX, last_x) <= static_cast<float>(width - 1) &&
            std::max(line[0].fY, last_y) <= static_cast<float>(height - 1)) {
            BilinearTripleInterior(source, destination, line, count, width, height, stride);
            return;
        }
    }
    BilinearScalar<3>(source, destination, line, count, width, height, stride);
}

void BilinearQuad(const uint8_t* source, uint8_t* destination, Point* line,
                  size_t first, size_t count, size_t /*capacity*/, size_t width,
                  size_t height, size_t stride) {
#if defined(INSPIRECV_TASK_USE_SSE) && defined(__SSE4_1__)
    if (platform::HasSse41()) {
        BilinearQuadSse(source, destination + 4 * first, line, count,
                        width, height, stride);
        return;
    }
#endif
#if defined(INSPIRECV_TASK_USE_NEON)
    inspirecv_task_sample_c4_bilinear_arm(source, destination + 4 * first,
                                          reinterpret_cast<float*>(line), count,
                                          width - 1, height - 1, stride);
#else
    BilinearScalar<4>(source, destination + 4 * first, line, count,
                      width, height, stride);
#endif
}

void DirectNv21(const uint8_t* source, uint8_t* destination, Point* line,
                size_t first, size_t count, size_t capacity, size_t width,
                size_t height, size_t stride) {
    DirectSemiPlanar<false>(source, destination, line, first, count, capacity,
                            width, height, stride);
}

void NearestNv21(const uint8_t* source, uint8_t* destination, Point* line,
                 size_t first, size_t count, size_t capacity, size_t width,
                 size_t height, size_t stride) {
    NearestSemiPlanar<false>(source, destination, line, first, count, capacity,
                             width, height, stride);
}

void DirectNv12(const uint8_t* source, uint8_t* destination, Point* line,
                size_t first, size_t count, size_t capacity, size_t width,
                size_t height, size_t stride) {
    DirectSemiPlanar<true>(source, destination, line, first, count, capacity,
                           width, height, stride);
}

void NearestNv12(const uint8_t* source, uint8_t* destination, Point* line,
                 size_t first, size_t count, size_t capacity, size_t width,
                 size_t height, size_t stride) {
    NearestSemiPlanar<true>(source, destination, line, first, count, capacity,
                            width, height, stride);
}

void DirectI420(const uint8_t* source, uint8_t* destination, Point* line,
                size_t first, size_t count, size_t capacity, size_t width,
                size_t height, size_t /*stride*/) {
    const int x = static_cast<int>(roundf(Limit(line[0].fX, 0.0f,
                                                static_cast<float>(width - 1))));
    const int y = static_cast<int>(roundf(Limit(line[0].fY, 0.0f,
                                                static_cast<float>(height - 1))));
    const size_t chroma_width = (width + 1) / 2;
    const size_t chroma_height = (height + 1) / 2;
    const size_t chroma_plane = chroma_width * chroma_height;
    const uint8_t* u = source + width * height + (y / 2) * chroma_width + x / 2;
    const uint8_t* v = u + chroma_plane;
    ChromaTarget target = TargetPlanes(destination, first, capacity);
    std::memcpy(target.luma, source + static_cast<size_t>(y) * width + x, count);
    const size_t pairs = (count + 1) / 2;
    for (size_t pair = 0; pair < pairs; ++pair) {
        target.vu[2 * pair] = v[pair];
        target.vu[2 * pair + 1] = u[pair];
    }
}

void NearestI420(const uint8_t* source, uint8_t* destination, Point* line,
                 size_t first, size_t count, size_t capacity, size_t width,
                 size_t height, size_t source_stride) {
    size_t stride = source_stride == 0 ? width : source_stride;
    const uint8_t* u = source + stride * height;
    ChromaTarget target = TargetPlanes(destination, first, capacity);
    NearestMono(source, target.luma, line, 0, count, capacity, width, height, stride);

    if (source_stride == 0) stride = (width + 1) / 2;
    const size_t chroma_width = (width + 1) / 2;
    const size_t chroma_height = (height + 1) / 2;
    const uint8_t* v = u + stride * chroma_height;
    CoordinateCursor cursor = {(line[0].fX - 0.01f) / 2.0f,
                               (line[0].fY - 0.01f) / 2.0f,
                               line[1].fX / 2.0f, line[1].fY / 2.0f};
    const float maximum_x = static_cast<float>(chroma_width - 1);
    const float maximum_y = static_cast<float>(chroma_height - 1);
    const size_t pairs = (count + 1) / 2;
    for (size_t pair = 0; pair < pairs; ++pair) {
        const int x = static_cast<int>(roundf(Limit(cursor.x, 0.0f, maximum_x)));
        const int y = static_cast<int>(roundf(Limit(cursor.y, 0.0f, maximum_y)));
        const size_t offset = static_cast<size_t>(y) * stride + x;
        target.vu[2 * pair] = v[offset];
        target.vu[2 * pair + 1] = u[offset];
        cursor.Advance();
    }
}

}  // namespace sampling
}  // namespace kernels
}  // namespace task
}  // namespace inspirecv
