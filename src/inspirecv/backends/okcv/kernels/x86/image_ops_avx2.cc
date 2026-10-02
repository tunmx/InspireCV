#include "image_ops_avx2.h"

#include <cfloat>
#include <cmath>
#include <cstring>
#include <immintrin.h>

namespace okcv {
namespace x86 {
namespace {

// Generic inline/template definitions emitted from an AVX2 translation unit
// can be selected by the linker for baseline callers. Keep scalar helpers local
// and allocation in the caller. Preserve std::min/max operand ordering for NaNs
// and signed zero; C math entry points do not emit C++ overload COMDATs here.
template<typename T> T LocalMin(T a, T b) { return b < a ? b : a; }
template<typename T> T LocalMax(T a, T b) { return a < b ? b : a; }
int Absolute(int value) { return value < 0 ? -value : value; }
double Absolute(double value) { return ::fabs(value); }

// No masked load is needed: every vector load stays inside the source image.
// Deinterleave sixteen 3-channel pixels from exactly 48 source bytes.
inline void LoadChannels16(const uint8_t* p, __m128i& b, __m128i& g, __m128i& r) {
    const __m128i a0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    const __m128i a1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 16));
    const __m128i a2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 32));
    const __m128i m0 = _mm_setr_epi8(0,3,6,9,12,15,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1);
    const __m128i m1 = _mm_setr_epi8(1,4,7,10,13,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1);
    const __m128i m2 = _mm_setr_epi8(2,5,8,11,14,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1);
    b = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(a0, m0),
                                 _mm_slli_si128(_mm_shuffle_epi8(a1, m2), 6)),
                     _mm_slli_si128(_mm_shuffle_epi8(a2, m1), 11));
    g = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(a0, m1),
                                 _mm_slli_si128(_mm_shuffle_epi8(a1, m0), 5)),
                     _mm_slli_si128(_mm_shuffle_epi8(a2, m2), 11));
    r = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(a0, m2),
                                 _mm_slli_si128(_mm_shuffle_epi8(a1, m1), 5)),
                     _mm_slli_si128(_mm_shuffle_epi8(a2, m0), 10));
}

inline __m256d Gray4(__m256d b, __m256d g, __m256d r) {
    // Preserve the scalar double expression and rounding, including half-way
    // cases. Fixed 8-bit coefficients differ by one gray level for some colors.
    return _mm256_add_pd(_mm256_add_pd(_mm256_mul_pd(b, _mm256_set1_pd(0.114)),
                                       _mm256_mul_pd(g, _mm256_set1_pd(0.587))),
                          _mm256_mul_pd(r, _mm256_set1_pd(0.299)));
}

inline void LoadChannels4(const float* p, __m256d& b, __m256d& g, __m256d& r) {
    // Four complete pixels occupy exactly twelve floats. Three contiguous
    // loads avoid the cost of three gathers while keeping every read in bounds.
    const __m128 a0 = _mm_loadu_ps(p);      // b0 g0 r0 b1
    const __m128 a1 = _mm_loadu_ps(p + 4);  // g1 r1 b2 g2
    const __m128 a2 = _mm_loadu_ps(p + 8);  // r2 b3 g3 r3
    const __m128 gr = _mm_shuffle_ps(a0, a1, _MM_SHUFFLE(1, 0, 2, 1)); // g0 r0 g1 r1
    const __m128 bg = _mm_shuffle_ps(a1, a2, _MM_SHUFFLE(2, 1, 3, 2)); // b2 g2 b3 g3
    const __m128 br = _mm_shuffle_ps(a0, a2, _MM_SHUFFLE(3, 0, 3, 0)); // b0 b1 r2 r3
    b = _mm256_cvtps_pd(_mm_shuffle_ps(br, bg, _MM_SHUFFLE(2, 0, 1, 0)));
    g = _mm256_cvtps_pd(_mm_shuffle_ps(gr, bg, _MM_SHUFFLE(3, 1, 2, 0)));
    r = _mm256_cvtps_pd(_mm_shuffle_ps(gr, br, _MM_SHUFFLE(3, 2, 3, 1)));
}

inline __m128i Narrow16(__m256i words) {
    return _mm_packus_epi16(_mm256_castsi256_si128(words),
                            _mm256_extracti128_si256(words, 1));
}

inline __m128i Blend16(__m128i a, __m128i b, __m128i mask) {
    const __m256i m = _mm256_cvtepu8_epi16(mask);
    const __m256i inv = _mm256_sub_epi16(_mm256_set1_epi16(255), m);
    __m256i t = _mm256_add_epi16(
      _mm256_add_epi16(_mm256_mullo_epi16(_mm256_cvtepu8_epi16(a), m),
                        _mm256_mullo_epi16(_mm256_cvtepu8_epi16(b), inv)),
      _mm256_set1_epi16(128));
    t = _mm256_add_epi16(t, _mm256_srli_epi16(t, 8));
    return Narrow16(_mm256_srli_epi16(t, 8));
}

inline __m128i MaskIndices(int channels, int component_offset) {
    alignas(16) uint8_t indices[16];
    for (int i = 0; i < 16; ++i) indices[i] = static_cast<uint8_t>((component_offset + i) / channels);
    return _mm_load_si128(reinterpret_cast<const __m128i*>(indices));
}

template<bool Erode>
inline __m256i Extremum(__m256i a, __m256i b) {
    return Erode ? _mm256_min_epu8(a, b) : _mm256_max_epu8(a, b);
}

template<bool Erode>
inline uint8_t Extremum(uint8_t a, uint8_t b) {
    return Erode ? LocalMin(a, b) : LocalMax(a, b);
}

template<bool Erode>
void Morphology(const uint8_t* source, uint8_t* destination, int width, int height,
                int left, int right, int top, int bottom, uint8_t* temporary) {
    if (width <= 0 || height <= 0) return;
    if (left == 1 && right == 1 && top == 1 && bottom == 1) {
        // Direct 3x3 kernel saves a temporary image and a second memory pass.
        for (int y = 0; y < height; ++y) {
            const uint8_t* r0 = source + static_cast<std::size_t>(LocalMax(0, y - 1)) * width;
            const uint8_t* r1 = source + static_cast<std::size_t>(y) * width;
            const uint8_t* r2 = source + static_cast<std::size_t>(LocalMin(height - 1, y + 1)) * width;
            uint8_t* out = destination + static_cast<std::size_t>(y) * width;
            int x = 0;
            auto scalar = [&](int column) {
                uint8_t v = Erode ? 255 : 0;
                for (int dx = -1; dx <= 1; ++dx) {
                    const int cx = LocalMax(0, LocalMin(width - 1, column + dx));
                    v = Extremum<Erode>(v, Extremum<Erode>(r0[cx], Extremum<Erode>(r1[cx], r2[cx])));
                }
                return v;
            };
            out[x++] = scalar(0);
            for (; x + 32 < width; x += 32) {
                __m256i v = _mm256_set1_epi8(Erode ? static_cast<char>(255) : 0);
                for (int dx = -1; dx <= 1; ++dx) {
                    const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r0 + x + dx));
                    const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r1 + x + dx));
                    const __m256i c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r2 + x + dx));
                    v = Extremum<Erode>(v, Extremum<Erode>(a, Extremum<Erode>(b, c)));
                }
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + x), v);
            }
            for (; x < width; ++x) out[x] = scalar(x);
        }
        return;
    }
    // Separable small rectangular kernels, with clipped (replicate-equivalent)
    // boundaries. Large windows retain the backend's linear-time deque path.
    for (int y = 0; y < height; ++y) {
        const uint8_t* row = source + static_cast<std::size_t>(y) * width;
        uint8_t* out = temporary + static_cast<std::size_t>(y) * width;
        int x = 0;
        auto scalar = [&](int column) {
            uint8_t v = Erode ? 255 : 0;
            for (int cx = LocalMax(0, column - left); cx <= LocalMin(width - 1, column + right); ++cx)
                v = Extremum<Erode>(v, row[cx]);
            return v;
        };
        for (; x < LocalMin(width, left); ++x) out[x] = scalar(x);
        for (; x + 32 + right <= width; x += 32) {
            __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + x - left));
            for (int dx = -left + 1; dx <= right; ++dx)
                v = Extremum<Erode>(v, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + x + dx)));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + x), v);
        }
        for (; x < width; ++x) out[x] = scalar(x);
    }
    for (int y = 0; y < height; ++y) {
        uint8_t* out = destination + static_cast<std::size_t>(y) * width;
        const int first = LocalMax(0, y - top), last = LocalMin(height - 1, y + bottom);
        const uint8_t* row = temporary + static_cast<std::size_t>(first) * width;
        int x = 0;
        for (; x + 32 <= width; x += 32) {
            __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + x));
            for (int cy = first + 1; cy <= last; ++cy)
                v = Extremum<Erode>(v, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                     temporary + static_cast<std::size_t>(cy) * width + x)));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + x), v);
        }
        for (; x < width; ++x) {
            uint8_t v = row[x];
            for (int cy = first + 1; cy <= last; ++cy)
                v = Extremum<Erode>(v, temporary[static_cast<std::size_t>(cy) * width + x]);
            out[x] = v;
        }
    }
}

}  // namespace

void GrayU8Avx2(const uint8_t* source, uint8_t* destination, std::size_t pixels) {
    std::size_t x = 0;
    for (; x + 16 <= pixels; x += 16) {
        __m128i b, g, r;
        LoadChannels16(source + x * 3, b, g, r);
        for (int group = 0; group < 4; ++group) {
            const __m256d y = Gray4(_mm256_cvtepi32_pd(_mm_cvtepu8_epi32(b)),
                                    _mm256_cvtepi32_pd(_mm_cvtepu8_epi32(g)),
                                    _mm256_cvtepi32_pd(_mm_cvtepu8_epi32(r)));
            const __m128i values = _mm256_cvttpd_epi32(_mm256_add_pd(y, _mm256_set1_pd(0.5)));
            const __m128i packed = _mm_packus_epi16(_mm_packs_epi32(values, values), _mm_setzero_si128());
            const uint32_t word = static_cast<uint32_t>(_mm_cvtsi128_si32(packed));
            std::memcpy(destination + x + group * 4, &word, sizeof(word));
            b = _mm_srli_si128(b, 4); g = _mm_srli_si128(g, 4); r = _mm_srli_si128(r, 4);
        }
    }
    for (; x < pixels; ++x) {
        const uint8_t* p = source + x * 3;
        destination[x] = static_cast<uint8_t>(::floor(0.114 * p[0] + 0.587 * p[1] + 0.299 * p[2] + 0.5));
    }
}

void GrayF32Avx2(const float* source, float* destination, std::size_t pixels) {
    std::size_t x = 0;
    for (; x + 4 <= pixels; x += 4) {
        __m256d b, g, r;
        LoadChannels4(source + x * 3, b, g, r);
        _mm_storeu_ps(destination + x, _mm256_cvtpd_ps(Gray4(b, g, r)));
    }
    for (; x < pixels; ++x) {
        const float* p = source + x * 3;
        destination[x] = static_cast<float>(0.114 * p[0] + 0.587 * p[1] + 0.299 * p[2]);
    }
}

void Mean3U8Avx2(const uint8_t* source, uint8_t* destination, std::size_t pixels) {
    std::size_t x = 0;
    for (; x + 16 <= pixels; x += 16) {
        __m128i b, g, r;
        LoadChannels16(source + x * 3, b, g, r);
        const __m256i sum = _mm256_add_epi16(_mm256_add_epi16(_mm256_cvtepu8_epi16(b),
                                     _mm256_cvtepu8_epi16(g)), _mm256_cvtepu8_epi16(r));
        // ceil(65536/3) is exact over the bounded sum [0, 765].
        _mm_storeu_si128(reinterpret_cast<__m128i*>(destination + x),
                         Narrow16(_mm256_mulhi_epu16(sum, _mm256_set1_epi16(21846))));
    }
    for (; x < pixels; ++x) destination[x] = static_cast<uint8_t>((source[3*x] + source[3*x+1] + source[3*x+2]) / 3);
}

void Mean3F32Avx2(const float* source, float* destination, std::size_t pixels) {
    std::size_t x = 0;
    for (; x + 4 <= pixels; x += 4) {
        __m256d b, g, r;
        LoadChannels4(source + x * 3, b, g, r);
        const __m256d sum = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(_mm256_setzero_pd(), b), g), r);
        _mm_storeu_ps(destination + x, _mm256_cvtpd_ps(_mm256_div_pd(sum, _mm256_set1_pd(3.0))));
    }
    for (; x < pixels; ++x) {
        double sum = 0.0;
        for (int c = 0; c < 3; ++c) sum += source[3*x+c];
        destination[x] = static_cast<float>(sum / 3.0);
    }
}

void ThresholdU8Avx2(const uint8_t* source, uint8_t* destination, std::size_t count,
                     uint8_t threshold, uint8_t maximum) {
    const __m256i offset = _mm256_set1_epi8(static_cast<char>(0x80));
    const __m256i t = _mm256_xor_si256(_mm256_set1_epi8(static_cast<char>(threshold)), offset);
    const __m256i m = _mm256_set1_epi8(static_cast<char>(maximum));
    std::size_t i = 0;
    for (; i + 32 <= count; i += 32) {
        const __m256i input = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(source + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(destination + i),
          _mm256_and_si256(_mm256_cmpgt_epi8(_mm256_xor_si256(input, offset), t), m));
    }
    for (; i < count; ++i) destination[i] = source[i] > threshold ? maximum : 0;
}

void ThresholdF32Avx2(const float* source, float* destination, std::size_t count,
                      float threshold, float maximum) {
    const __m256 t = _mm256_set1_ps(threshold), m = _mm256_set1_ps(maximum);
    std::size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        const __m256 input = _mm256_loadu_ps(source + i);
        _mm256_storeu_ps(destination + i, _mm256_and_ps(_mm256_cmp_ps(input, t, _CMP_GT_OQ), m));
    }
    for (; i < count; ++i) destination[i] = source[i] > threshold ? maximum : 0.0f;
}

void AbsDiffU8Avx2(const uint8_t* a, const uint8_t* b, uint8_t* destination, std::size_t count) {
    std::size_t i = 0;
    for (; i + 32 <= count; i += 32) {
        const __m256i av = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        const __m256i bv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(destination + i),
          _mm256_sub_epi8(_mm256_max_epu8(av, bv), _mm256_min_epu8(av, bv)));
    }
    for (; i < count; ++i) destination[i] = static_cast<uint8_t>(Absolute(int(a[i]) - int(b[i])));
}

void AbsDiffF32Avx2(const float* a, const float* b, float* destination, std::size_t count) {
    const __m256 sign = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    std::size_t i = 0;
    for (; i + 8 <= count; i += 8)
        _mm256_storeu_ps(destination + i, _mm256_and_ps(_mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)), sign));
    for (; i < count; ++i) destination[i] = static_cast<float>(Absolute(static_cast<double>(a[i]) - b[i]));
}

void BlendU8Avx2(const uint8_t* a, const uint8_t* b, const uint8_t* mask,
                 uint8_t* destination, std::size_t pixels, int channels) {
    __m128i indices[4];
    if (channels <= 4) for (int c = 0; c < channels; ++c) indices[c] = MaskIndices(channels, c * 16);
    std::size_t x = 0;
    if (channels <= 4) {
        for (; x + 16 <= pixels; x += 16) {
            const __m128i m = _mm_loadu_si128(reinterpret_cast<const __m128i*>(mask + x));
            for (int c = 0; c < channels; ++c) {
                const std::size_t base = x * channels + c * 16;
                _mm_storeu_si128(reinterpret_cast<__m128i*>(destination + base), Blend16(
                  _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + base)),
                  _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + base)), _mm_shuffle_epi8(m, indices[c])));
            }
        }
    }
    for (; x < pixels; ++x) for (int c = 0; c < channels; ++c) {
        const std::size_t i = x * channels + c;
        const int t = mask[x] * int(a[i]) + (255 - mask[x]) * int(b[i]) + 128;
        destination[i] = static_cast<uint8_t>((t + (t >> 8)) >> 8);
    }
}

void BlendF32Avx2(const float* a, const float* b, const uint8_t* mask,
                  float* destination, std::size_t pixels, int channels) {
    __m128i indices[4];
    if (channels <= 4) for (int c = 0; c < channels; ++c) indices[c] = MaskIndices(channels, c * 8);
    std::size_t x = 0;
    if (channels <= 4) {
        for (; x + 8 <= pixels; x += 8) {
            const __m128i m = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(mask + x));
            for (int c = 0; c < channels; ++c) {
                const std::size_t base = x * channels + c * 8;
                const __m256 weight = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(
                  _mm_shuffle_epi8(m, indices[c]))), _mm256_set1_ps(1.0f / 255.0f));
                _mm256_storeu_ps(destination + base, _mm256_add_ps(
                  _mm256_mul_ps(weight, _mm256_loadu_ps(a + base)),
                  _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(1.0f), weight), _mm256_loadu_ps(b + base))));
            }
        }
    }
    for (; x < pixels; ++x) for (int c = 0; c < channels; ++c) {
        const std::size_t i = x * channels + c;
        const float weight = static_cast<float>(mask[x]) * (1.0f / 255.0f);
        destination[i] = weight * a[i] + (1.0f - weight) * b[i];
    }
}

void MorphologyU8Avx2(const uint8_t* source, uint8_t* destination, int width, int height,
                      int left, int right, int top, int bottom, bool erode, uint8_t* scratch) {
    if (erode) Morphology<true>(source, destination, width, height, left, right, top, bottom, scratch);
    else Morphology<false>(source, destination, width, height, left, right, top, bottom, scratch);
}

void FlipU8Avx2(const uint8_t* source, uint8_t* destination, int width, int height,
                int channels, bool reverse_rows) {
    const std::size_t stride = static_cast<std::size_t>(width) * channels;
    const __m256i reverse_bytes = _mm256_setr_epi8(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,
                                                  15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
    const __m256i reverse_words = _mm256_setr_epi32(7,6,5,4,3,2,1,0);
    for (int y = 0; y < height; ++y) {
        const uint8_t* row = source + static_cast<std::size_t>(reverse_rows ? height - 1 - y : y) * stride;
        uint8_t* out = destination + static_cast<std::size_t>(y) * stride;
        int x = 0;
        if (channels == 1) for (; x + 32 <= width; x += 32) {
            const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + width - x - 32));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + x),
              _mm256_shuffle_epi8(_mm256_permute2x128_si256(v, v, 0x01), reverse_bytes));
        }
        if (channels == 4) for (; x + 8 <= width; x += 8) {
            const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + (width - x - 8) * 4));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + x * 4), _mm256_permutevar8x32_epi32(v, reverse_words));
        }
        for (; x < width; ++x)
            std::memcpy(out + static_cast<std::size_t>(x) * channels,
                         row + static_cast<std::size_t>(width - 1 - x) * channels, channels);
    }
}

void FlipF32Avx2(const float* source, float* destination, int width, int height,
                 int channels, bool reverse_rows) {
    const std::size_t stride = static_cast<std::size_t>(width) * channels;
    const __m256i reverse = _mm256_setr_epi32(7,6,5,4,3,2,1,0);
    for (int y = 0; y < height; ++y) {
        const float* row = source + static_cast<std::size_t>(reverse_rows ? height - 1 - y : y) * stride;
        float* out = destination + static_cast<std::size_t>(y) * stride;
        int x = 0;
        if (channels == 1) for (; x + 8 <= width; x += 8)
            _mm256_storeu_ps(out + x, _mm256_permutevar8x32_ps(_mm256_loadu_ps(row + width - x - 8), reverse));
        if (channels == 4) for (; x + 2 <= width; x += 2) {
            const __m256 v = _mm256_loadu_ps(row + (width - x - 2) * 4);
            _mm256_storeu_ps(out + x * 4, _mm256_permute2f128_ps(v, v, 0x01));
        }
        if (channels == 3) for (; x + 8 <= width; x += 8) {
            const float* input = row + (width - x - 8) * 3;
            const __m256 a = _mm256_loadu_ps(input);
            const __m256 b = _mm256_loadu_ps(input + 8);
            const __m256 c = _mm256_loadu_ps(input + 16);
            const __m256 first = _mm256_blend_ps(
              _mm256_permutevar8x32_ps(c, _mm256_setr_epi32(5,6,7,2,3,4,0,0)),
              _mm256_permutevar8x32_ps(b, _mm256_set1_epi32(7)), 0x40);
            const __m256 second = _mm256_blend_ps(_mm256_blend_ps(
              _mm256_permutevar8x32_ps(b, _mm256_setr_epi32(0,4,5,6,1,2,3,0)),
              _mm256_permutevar8x32_ps(c, _mm256_set1_epi32(1)), 0x01),
              _mm256_permutevar8x32_ps(a, _mm256_set1_epi32(6)), 0x80);
            const __m256 third = _mm256_blend_ps(
              _mm256_permutevar8x32_ps(a, _mm256_setr_epi32(7,0,3,4,5,0,1,2)),
              _mm256_permutevar8x32_ps(b, _mm256_setzero_si256()), 0x02);
            _mm256_storeu_ps(out + x * 3, first);
            _mm256_storeu_ps(out + x * 3 + 8, second);
            _mm256_storeu_ps(out + x * 3 + 16, third);
        }
        for (; x < width; ++x)
            std::memcpy(out + static_cast<std::size_t>(x) * channels,
                         row + static_cast<std::size_t>(width - 1 - x) * channels, channels * sizeof(float));
    }
}

void SwapRbU8C4Avx2(const uint8_t* source, uint8_t* destination, std::size_t pixels) {
    const __m256i indices = _mm256_setr_epi8(2,1,0,3,6,5,4,7,10,9,8,11,14,13,12,15,
                                            2,1,0,3,6,5,4,7,10,9,8,11,14,13,12,15);
    std::size_t x = 0;
    for (; x + 8 <= pixels; x += 8)
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(destination + 4*x),
          _mm256_shuffle_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(source + 4*x)), indices));
    for (; x < pixels; ++x) {
        destination[4*x] = source[4*x+2]; destination[4*x+1] = source[4*x+1];
        destination[4*x+2] = source[4*x]; destination[4*x+3] = source[4*x+3];
    }
}

void SwapRbF32Avx2(const float* source, float* destination, std::size_t pixels, int channels) {
    std::size_t x = 0;
    if (channels == 4) {
        for (; x + 2 <= pixels; x += 2)
            _mm256_storeu_ps(destination + x * 4,
              _mm256_permute_ps(_mm256_loadu_ps(source + x * 4), _MM_SHUFFLE(3,0,1,2)));
    } else {
        for (; x + 8 <= pixels; x += 8) {
            const __m256 a = _mm256_loadu_ps(source + x * 3);
            const __m256 b = _mm256_loadu_ps(source + x * 3 + 8);
            const __m256 c = _mm256_loadu_ps(source + x * 3 + 16);
            const __m256 first = _mm256_blend_ps(
              _mm256_permutevar8x32_ps(a, _mm256_setr_epi32(2,1,0,5,4,3,0,7)),
              _mm256_permutevar8x32_ps(b, _mm256_setzero_si256()), 0x40);
            const __m256 second = _mm256_blend_ps(_mm256_blend_ps(
              _mm256_permutevar8x32_ps(b, _mm256_setr_epi32(0,3,2,1,6,5,4,0)),
              _mm256_permutevar8x32_ps(a, _mm256_set1_epi32(6)), 0x01),
              _mm256_permutevar8x32_ps(c, _mm256_set1_epi32(1)), 0x80);
            const __m256 third = _mm256_blend_ps(
              _mm256_permutevar8x32_ps(c, _mm256_setr_epi32(0,0,4,3,2,7,6,5)),
              _mm256_permutevar8x32_ps(b, _mm256_set1_epi32(7)), 0x02);
            _mm256_storeu_ps(destination + x * 3, first);
            _mm256_storeu_ps(destination + x * 3 + 8, second);
            _mm256_storeu_ps(destination + x * 3 + 16, third);
        }
    }
    for (; x < pixels; ++x) {
        destination[x*channels] = source[x*channels+2];
        destination[x*channels+1] = source[x*channels+1];
        destination[x*channels+2] = source[x*channels];
        if (channels == 4) destination[x*channels+3] = source[x*channels+3];
    }
}

namespace {
inline __m256 LoadEightFloats(const uint8_t* source) {
    return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(source))));
}
inline __m256 LoadEightFloats(const float* source) { return _mm256_loadu_ps(source); }

template<typename Pixel, int Kernel>
void GaussianHorizontal(const Pixel* source, float* destination, int width, int channels, const float* kernel) {
    constexpr int radius = Kernel / 2;
    const int count = width * channels;
    const int edge = radius * channels;
    __m256 weights[Kernel];
    for (int k = 0; k < Kernel; ++k) weights[k] = _mm256_set1_ps(kernel[k]);
    auto scalar = [&](int component) {
        const int x = component / channels, c = component % channels;
        float sum = static_cast<float>(source[LocalMax(0, x - radius) * channels + c]) * kernel[0];
        for (int k = 1; k < Kernel; ++k) {
            const int sx = LocalMax(0, LocalMin(width - 1, x + k - radius));
            sum += static_cast<float>(source[sx * channels + c]) * kernel[k];
        }
        return sum;
    };
    int i = 0;
    for (; i < LocalMin(count, edge); ++i) destination[i] = scalar(i);
    for (; i + 8 + edge <= count; i += 8) {
        __m256 sum = _mm256_mul_ps(LoadEightFloats(source + i - edge), weights[0]);
        for (int k = 1; k < Kernel; ++k)
            sum = _mm256_add_ps(sum, _mm256_mul_ps(LoadEightFloats(source + i + (k - radius) * channels), weights[k]));
        _mm256_storeu_ps(destination + i, sum);
    }
    for (; i < count; ++i) destination[i] = scalar(i);
}

inline void StoreEightFloats(uint8_t* destination, __m256 value, bool) {
    value = _mm256_min_ps(_mm256_max_ps(value, _mm256_setzero_ps()), _mm256_set1_ps(255.0f));
    const __m256i words = _mm256_cvttps_epi32(_mm256_add_ps(value, _mm256_set1_ps(0.5f)));
    const __m128i shorts = _mm_packs_epi32(_mm256_castsi256_si128(words), _mm256_extracti128_si256(words, 1));
    _mm_storel_epi64(reinterpret_cast<__m128i*>(destination), _mm_packus_epi16(shorts, _mm_setzero_si128()));
}
inline void StoreEightFloats(float* destination, __m256 value, bool clamp) {
    if (clamp) {
        const __m256 lo = _mm256_set1_ps(-FLT_MAX);
        const __m256 hi = _mm256_set1_ps(FLT_MAX);
        value = _mm256_blendv_ps(value, lo, _mm256_cmp_ps(value, lo, _CMP_LT_OQ));
        value = _mm256_blendv_ps(value, hi, _mm256_cmp_ps(value, hi, _CMP_GT_OQ));
    }
    _mm256_storeu_ps(destination, value);
}
inline uint8_t GaussianResult(float value, uint8_t*, bool) {
    value = ::floorf(value + 0.5f);
    return static_cast<uint8_t>(LocalMax(0.0f, LocalMin(255.0f, value)));
}
inline float GaussianResult(float value, float*, bool clamp) {
    if (clamp) {
        const float high = FLT_MAX;
        value = value < -high ? -high : (value > high ? high : value);
    }
    return value;
}

template<typename Pixel, int Kernel>
void GaussianVertical(const float* source, Pixel* destination, int width, int height, int channels,
                       int y, const float* kernel) {
    constexpr int radius = Kernel / 2;
    const int count = width * channels;
    const float* rows[Kernel];
    __m256 weights[Kernel];
    for (int k = 0; k < Kernel; ++k) {
        rows[k] = source + static_cast<std::size_t>(LocalMax(0, LocalMin(height - 1, y + k - radius))) * count;
        weights[k] = _mm256_set1_ps(kernel[k]);
    }
    int i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256 sum = _mm256_setzero_ps();
        for (int k = 0; k < Kernel; ++k)
            sum = _mm256_add_ps(sum, _mm256_mul_ps(_mm256_loadu_ps(rows[k] + i), weights[k]));
        StoreEightFloats(destination + i, sum, channels == 1);
    }
    for (; i < count; ++i) {
        float sum = 0.0f;
        for (int k = 0; k < Kernel; ++k) sum += rows[k][i] * kernel[k];
        destination[i] = GaussianResult(sum, destination, channels == 1);
    }
}
}  // namespace

void GaussianHorizontalU8Avx2(const uint8_t* source, float* destination, int width, int channels,
                               const float* kernel, int size) {
    if (size == 3) GaussianHorizontal<uint8_t, 3>(source, destination, width, channels, kernel);
    else GaussianHorizontal<uint8_t, 5>(source, destination, width, channels, kernel);
}
void GaussianHorizontalF32Avx2(const float* source, float* destination, int width, int channels,
                                const float* kernel, int size) {
    if (size == 3) GaussianHorizontal<float, 3>(source, destination, width, channels, kernel);
    else GaussianHorizontal<float, 5>(source, destination, width, channels, kernel);
}
void GaussianVerticalU8Avx2(const float* source, uint8_t* destination, int width, int height, int channels,
                             int y, const float* kernel, int size) {
    if (size == 3) GaussianVertical<uint8_t, 3>(source, destination, width, height, channels, y, kernel);
    else GaussianVertical<uint8_t, 5>(source, destination, width, height, channels, y, kernel);
}
void GaussianVerticalF32Avx2(const float* source, float* destination, int width, int height, int channels,
                              int y, const float* kernel, int size) {
    if (size == 3) GaussianVertical<float, 3>(source, destination, width, height, channels, y, kernel);
    else GaussianVertical<float, 5>(source, destination, width, height, channels, y, kernel);
}

void ConvertU8ToF32Avx2(const uint8_t* source, float* destination, std::size_t count) {
    std::size_t i = 0;
    for (; i + 8 <= count; i += 8) _mm256_storeu_ps(destination + i, LoadEightFloats(source + i));
    for (; i < count; ++i) destination[i] = source[i];
}

namespace {
template<int Shift>
inline __m256i BilinearPackedChannel(__m256i tl, __m256i tr, __m256i bl, __m256i br,
                                     __m256 weight_x, __m256 weight_y) {
    const __m256i mask = _mm256_set1_epi32(255);
    const __m256 top_left = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(tl, Shift), mask));
    const __m256 top_right = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(tr, Shift), mask));
    const __m256 bottom_left = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(bl, Shift), mask));
    const __m256 bottom_right = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(br, Shift), mask));
    const __m256 top = _mm256_add_ps(top_left, _mm256_mul_ps(_mm256_sub_ps(top_right, top_left), weight_x));
    const __m256 bottom = _mm256_add_ps(bottom_left, _mm256_mul_ps(_mm256_sub_ps(bottom_right, bottom_left), weight_x));
    const __m256 value = _mm256_add_ps(top, _mm256_mul_ps(_mm256_sub_ps(bottom, top), weight_y));
    const __m256i integer = _mm256_cvttps_epi32(value);
    const __m256 fraction = _mm256_sub_ps(value, _mm256_cvtepi32_ps(integer));
    const __m256i increment = _mm256_and_si256(
      _mm256_castps_si256(_mm256_cmp_ps(fraction, _mm256_set1_ps(.5f), _CMP_GE_OQ)), _mm256_set1_epi32(1));
    return _mm256_slli_epi32(_mm256_add_epi32(integer, increment), Shift);
}

inline void Store12Bytes(uint8_t* destination, __m128i value) {
    _mm_storel_epi64(reinterpret_cast<__m128i*>(destination), value);
    const uint32_t last = static_cast<uint32_t>(_mm_cvtsi128_si32(_mm_srli_si128(value, 8)));
    std::memcpy(destination + 8, &last, sizeof(last));
}

template<int Channels>
void ResizeBilinearU8Row(const uint8_t* top, const uint8_t* bottom, uint8_t* destination,
                         int source_width, int destination_width,
                         const int* left, const int* right, const float* weight_x, float weight_y) {
    const int source_bytes = source_width * Channels;
    const __m256 vy = _mm256_set1_ps(weight_y);
    const __m256i stride = _mm256_set1_epi32(Channels);
    const __m256i last_word = _mm256_set1_epi32(source_bytes - 4);
    const __m128i rgb = _mm_setr_epi8(0,1,2,4,5,6,8,9,10,12,13,14,-1,-1,-1,-1);
    int x = 0;
    if (source_bytes >= 4) {
        for (; x <= destination_width - 8; x += 8) {
            const __m256i left_bytes = _mm256_mullo_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(left + x)), stride);
            const __m256i right_bytes = _mm256_mullo_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(right + x)), stride);
            // A gather reads four bytes. Move words at the right border inward
            // and shift the requested pixel back to lane byte0. This handles
            // even a one-byte or three-byte last pixel without crossing rows.
            const __m256i safe_left = _mm256_min_epi32(left_bytes, last_word);
            const __m256i safe_right = _mm256_min_epi32(right_bytes, last_word);
            const __m256i left_shift = _mm256_slli_epi32(_mm256_sub_epi32(left_bytes, safe_left), 3);
            const __m256i right_shift = _mm256_slli_epi32(_mm256_sub_epi32(right_bytes, safe_right), 3);
            const __m256i tl = _mm256_srlv_epi32(_mm256_i32gather_epi32(reinterpret_cast<const int*>(top), safe_left, 1), left_shift);
            const __m256i tr = _mm256_srlv_epi32(_mm256_i32gather_epi32(reinterpret_cast<const int*>(top), safe_right, 1), right_shift);
            const __m256i bl = _mm256_srlv_epi32(_mm256_i32gather_epi32(reinterpret_cast<const int*>(bottom), safe_left, 1), left_shift);
            const __m256i br = _mm256_srlv_epi32(_mm256_i32gather_epi32(reinterpret_cast<const int*>(bottom), safe_right, 1), right_shift);
            const __m256 vx = _mm256_loadu_ps(weight_x + x);
            __m256i pixels = BilinearPackedChannel<0>(tl, tr, bl, br, vx, vy);
            if (Channels >= 3) {
                pixels = _mm256_or_si256(pixels, BilinearPackedChannel<8>(tl, tr, bl, br, vx, vy));
                pixels = _mm256_or_si256(pixels, BilinearPackedChannel<16>(tl, tr, bl, br, vx, vy));
            }
            if (Channels == 4)
                pixels = _mm256_or_si256(pixels, BilinearPackedChannel<24>(tl, tr, bl, br, vx, vy));
            if (Channels == 1) {
                const __m128i words = _mm_packs_epi32(_mm256_castsi256_si128(pixels), _mm256_extracti128_si256(pixels, 1));
                _mm_storel_epi64(reinterpret_cast<__m128i*>(destination + x), _mm_packus_epi16(words, _mm_setzero_si128()));
            } else if (Channels == 3) {
                Store12Bytes(destination + x * 3, _mm_shuffle_epi8(_mm256_castsi256_si128(pixels), rgb));
                Store12Bytes(destination + x * 3 + 12, _mm_shuffle_epi8(_mm256_extracti128_si256(pixels, 1), rgb));
            } else {
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(destination + x * 4), pixels);
            }
        }
    }
    for (; x < destination_width; ++x) {
        for (int c = 0; c < Channels; ++c) {
            const float tl = top[left[x] * Channels + c], tr = top[right[x] * Channels + c];
            const float bl = bottom[left[x] * Channels + c], br = bottom[right[x] * Channels + c];
            const float upper = tl + (tr - tl) * weight_x[x];
            const float lower = bl + (br - bl) * weight_x[x];
            destination[x * Channels + c] = static_cast<uint8_t>(::roundf(upper + (lower - upper) * weight_y));
        }
    }
}
}  // namespace

void ResizeBilinearU8RowAvx2(const uint8_t* top, const uint8_t* bottom, uint8_t* destination,
                            int source_width, int destination_width, int channels,
                            const int* left, const int* right, const float* weight_x, float weight_y) {
    if (channels == 1) ResizeBilinearU8Row<1>(top, bottom, destination, source_width, destination_width, left, right, weight_x, weight_y);
    else if (channels == 3) ResizeBilinearU8Row<3>(top, bottom, destination, source_width, destination_width, left, right, weight_x, weight_y);
    else ResizeBilinearU8Row<4>(top, bottom, destination, source_width, destination_width, left, right, weight_x, weight_y);
}

// Compiled only in the existing isolated TU with contraction disabled.
void MulF32Avx2(const float* source, float* destination, std::size_t count, float a) {
    const __m256 factor = _mm256_set1_ps(a);
    const std::size_t full = count - count % 8;
    std::size_t i = 0;
    for (; i < full; i += 8) {
        const __m256 values = _mm256_loadu_ps(source + i);
        _mm256_storeu_ps(destination + i, _mm256_mul_ps(values, factor));
    }
    for (; i < count; ++i) destination[i] = source[i] * a;
}

void MulAddF32Avx2(const float* source, float* destination, std::size_t count, float a, float b) {
    const __m256 factor = _mm256_set1_ps(a);
    const __m256 offset = _mm256_set1_ps(b);
    const std::size_t full = count - count % 8;
    std::size_t i = 0;
    for (; i < full; i += 8) {
        const __m256 values = _mm256_loadu_ps(source + i);
        const __m256 product = _mm256_mul_ps(values, factor);
        _mm256_storeu_ps(destination + i, _mm256_add_ps(product, offset));
    }
    for (; i < count; ++i) destination[i] = source[i] * a + b;
}

}  // namespace x86
}  // namespace okcv
