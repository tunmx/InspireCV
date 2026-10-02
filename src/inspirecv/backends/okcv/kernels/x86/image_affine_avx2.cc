#include "inspirecv/backends/okcv/kernels/x86/image_affine_avx2.h"

#include <immintrin.h>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstring>

namespace okcv {
namespace x86 {
namespace {

// Never emit a generic scalar COMDAT with AVX instructions: a baseline caller
// could bind to it. These helpers have translation-unit-local linkage.
template<typename T> T LocalMin(T a, T b) { return b < a ? b : a; }
template<typename T> T LocalMax(T a, T b) { return a < b ? b : a; }
bool LocalFinite(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x7f800000u) != 0x7f800000u;
}

bool Supported(const uint8_t* source, int sw, int sh, uint8_t* destination,
                int dw, int dh, int channels, const float* matrix) {
    if (!source || !destination || !matrix || sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0 ||
        (channels != 1 && channels != 3 && channels != 4)) return false;
    // Keep coordinate-to-index conversion exact and safely below INT_MAX.
    // Larger extremely thin images retain the caller's generic implementation.
    constexpr int exact_integer_limit = 1 << 24;
    if (sw > exact_integer_limit || sh > exact_integer_limit ||
        dw > exact_integer_limit || dh > exact_integer_limit) return false;
    const size_t limit = size_t(INT_MAX);
    if (size_t(sw) > limit / size_t(sh) / size_t(channels) ||
        size_t(dw) > limit / size_t(dh) / size_t(channels)) return false;
    for (int i = 0; i < 6; ++i) if (!LocalFinite(matrix[i])) return false;
    // Finite coefficients can still overflow in x*a+y*b+tx. Reject before
    // writing anything; every intermediate of accepted transforms is finite.
    for (int row = 0; row < 2; ++row) {
        const float* m = matrix + row * 3;
        const double bound = ::fabs(double(m[0])) * double(dw - 1) +
                             ::fabs(double(m[1])) * double(dh - 1) + ::fabs(double(m[2]));
        if (bound > double(FLT_MAX)) return false;
    }
    return true;
}

__m256i RoundPositive(__m256 value) {
    const __m256i integral = _mm256_cvttps_epi32(value);
    const __m256 fraction = _mm256_sub_ps(value, _mm256_cvtepi32_ps(integral));
    const __m256i increment = _mm256_and_si256(
      _mm256_castps_si256(_mm256_cmp_ps(fraction, _mm256_set1_ps(.5f), _CMP_GE_OQ)),
      _mm256_set1_epi32(1));
    return _mm256_add_epi32(integral, increment);
}

// A gather always loads four bytes. Move reads near a row's right edge back
// within that SAME row, then shift to recover the requested C1/C3 pixel. C4
// already has a full word. No load crosses the final image byte or row padding.
template<int Channels>
__m256i GatherPixels(const uint8_t* source, __m256i x, __m256i y, int stride) {
    const __m256i byte_x = _mm256_mullo_epi32(x, _mm256_set1_epi32(Channels));
    const __m256i safe_x = Channels == 4 ? byte_x
      : _mm256_min_epi32(byte_x, _mm256_set1_epi32(stride - 4));
    const __m256i offset = _mm256_add_epi32(safe_x,
      _mm256_mullo_epi32(y, _mm256_set1_epi32(stride)));
    const __m256i words = _mm256_i32gather_epi32(reinterpret_cast<const int*>(source), offset, 1);
    return Channels == 4 ? words : _mm256_srlv_epi32(words,
      _mm256_slli_epi32(_mm256_sub_epi32(byte_x, safe_x), 3));
}

void StoreTriple8(uint8_t* destination, __m256i pixels) {
    const __m128i compact = _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9,
                                          10, 12, 13, 14, -1, -1, -1, -1);
    const __m256i packed = _mm256_shuffle_epi8(pixels, _mm256_broadcastsi128_si256(compact));
    const __m128i first = _mm256_castsi256_si128(packed);
    const __m128i second = _mm256_extracti128_si256(packed, 1);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(destination),
      _mm_or_si128(first, _mm_slli_si128(second, 12)));
    _mm_storel_epi64(reinterpret_cast<__m128i*>(destination + 16), _mm_srli_si128(second, 4));
}

template<int Channels>
void StorePixels(uint8_t* destination, __m256i pixels) {
    if (Channels == 4) _mm256_storeu_si256(reinterpret_cast<__m256i*>(destination), pixels);
    else if (Channels == 3) StoreTriple8(destination, pixels);
    else {
        const __m128i words = _mm_packus_epi32(_mm256_castsi256_si128(pixels),
                                               _mm256_extracti128_si256(pixels, 1));
        _mm_storel_epi64(reinterpret_cast<__m128i*>(destination),
          _mm_packus_epi16(words, _mm_setzero_si128()));
    }
}

template<int Channels>
void ScalarPixel(const uint8_t* source, int sw, int sh, uint8_t* output,
                  float sx, float sy, bool replicate, uint8_t border) {
    if (sx < 0.f || sy < 0.f || sx >= sw || sy >= sh) {
        if (!replicate) { std::memset(output, border, Channels); return; }
        sx = LocalMax(0.f, LocalMin(float(sw - 1), sx));
        sy = LocalMax(0.f, LocalMin(float(sh - 1), sy));
    }
    const int x0 = LocalMin(int(sx), sw - 1), x1 = LocalMin(x0 + 1, sw - 1);
    const int y0 = LocalMin(int(sy), sh - 1), y1 = LocalMin(y0 + 1, sh - 1);
    const float fx = sx - x0, fy = sy - y0;
    const size_t stride = size_t(sw) * Channels;
    for (int c = 0; c < Channels; ++c) {
        const float tl = source[size_t(y0) * stride + size_t(x0) * Channels + c];
        const float tr = source[size_t(y0) * stride + size_t(x1) * Channels + c];
        const float bl = source[size_t(y1) * stride + size_t(x0) * Channels + c];
        const float br = source[size_t(y1) * stride + size_t(x1) * Channels + c];
        const float top = tl + (tr - tl) * fx;
        const float bottom = bl + (br - bl) * fx;
        output[c] = static_cast<uint8_t>(::roundf(top + (bottom - top) * fy));
    }
}

template<int Channels>
void Affine(const uint8_t* source, int sw, int sh, uint8_t* destination,
             int width, int height, const float* matrix, bool replicate,
             uint8_t border) {
    const int stride = sw * Channels;
    const __m256 zero = _mm256_setzero_ps();
    const __m256 max_x = _mm256_set1_ps(float(sw - 1)), max_y = _mm256_set1_ps(float(sh - 1));
    const __m256 widthf = _mm256_set1_ps(float(sw)), heightf = _mm256_set1_ps(float(sh));
    const __m256 a = _mm256_set1_ps(matrix[0]), c = _mm256_set1_ps(matrix[3]);
    const __m256 tx = _mm256_set1_ps(matrix[2]), ty = _mm256_set1_ps(matrix[5]);
    const __m256i step = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i byte_mask = _mm256_set1_epi32(255), one = _mm256_set1_epi32(1);
    const __m256 border_float = _mm256_set1_ps(float(border));
    for (int y = 0; y < height; ++y) {
        uint8_t* output = destination + size_t(y) * width * Channels;
        // Do NOT fold translation into these row terms: Reference evaluates
        // ((float(x)*a)+(float(y)*b))+tx, with a float rounding at each operator.
        const float row_x = float(y) * matrix[1], row_y = float(y) * matrix[4];
        const __m256 bx = _mm256_set1_ps(row_x), dy = _mm256_set1_ps(row_y);
        int x = 0;
        if (stride >= 4) for (; x <= width - 8; x += 8) {
            const __m256 positions = _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_set1_epi32(x), step));
            __m256 sx = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(positions, a), bx), tx);
            __m256 sy = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(positions, c), dy), ty);
            const __m256 inside = _mm256_and_ps(
              _mm256_and_ps(_mm256_cmp_ps(sx, zero, _CMP_GE_OQ), _mm256_cmp_ps(sx, widthf, _CMP_LT_OQ)),
              _mm256_and_ps(_mm256_cmp_ps(sy, zero, _CMP_GE_OQ), _mm256_cmp_ps(sy, heightf, _CMP_LT_OQ)));
            if (!replicate && _mm256_movemask_ps(inside) == 0) {
                std::memset(output + size_t(x) * Channels, border, 8 * Channels);
                continue;
            }
            // Whole-pixel constant border lanes also clamp before any integer
            // conversion/gather, so rejected coordinates can never form pointers.
            sx = _mm256_max_ps(zero, _mm256_min_ps(max_x, sx));
            sy = _mm256_max_ps(zero, _mm256_min_ps(max_y, sy));
            const __m256i x0 = _mm256_cvttps_epi32(sx), y0 = _mm256_cvttps_epi32(sy);
            const __m256i x1 = _mm256_min_epi32(_mm256_add_epi32(x0, one), _mm256_set1_epi32(sw - 1));
            const __m256i y1 = _mm256_min_epi32(_mm256_add_epi32(y0, one), _mm256_set1_epi32(sh - 1));
            const __m256 fx = _mm256_sub_ps(sx, _mm256_cvtepi32_ps(x0));
            const __m256 fy = _mm256_sub_ps(sy, _mm256_cvtepi32_ps(y0));
            const __m256i tl = GatherPixels<Channels>(source, x0, y0, stride);
            const __m256i tr = GatherPixels<Channels>(source, x1, y0, stride);
            const __m256i bl = GatherPixels<Channels>(source, x0, y1, stride);
            const __m256i br = GatherPixels<Channels>(source, x1, y1, stride);
            __m256i packed = _mm256_setzero_si256();
            for (int channel = 0; channel < Channels; ++channel) {
                const __m256i shift = _mm256_set1_epi32(channel * 8);
                const __m256 p00 = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srlv_epi32(tl, shift), byte_mask));
                const __m256 p10 = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srlv_epi32(tr, shift), byte_mask));
                const __m256 p01 = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srlv_epi32(bl, shift), byte_mask));
                const __m256 p11 = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srlv_epi32(br, shift), byte_mask));
                const __m256 top = _mm256_add_ps(p00, _mm256_mul_ps(_mm256_sub_ps(p10, p00), fx));
                const __m256 bottom = _mm256_add_ps(p01, _mm256_mul_ps(_mm256_sub_ps(p11, p01), fx));
                __m256 value = _mm256_add_ps(top, _mm256_mul_ps(_mm256_sub_ps(bottom, top), fy));
                if (!replicate) value = _mm256_blendv_ps(border_float, value, inside);
                packed = _mm256_or_si256(packed, _mm256_sllv_epi32(RoundPositive(value), shift));
            }
            StorePixels<Channels>(output + size_t(x) * Channels, packed);
        }
        for (; x < width; ++x) {
            const float sx = (float(x) * matrix[0] + row_x) + matrix[2];
            const float sy = (float(x) * matrix[3] + row_y) + matrix[5];
            ScalarPixel<Channels>(source, sw, sh, output + size_t(x) * Channels, sx, sy, replicate, border);
        }
    }
}
}  // namespace

bool AffineBilinearReferenceU8Avx2(const uint8_t* source, int source_width,
                                  int source_height, uint8_t* destination,
                                  int width, int height, int channels,
                                  const float matrix[6], bool replicate_border,
                                  uint8_t border_value) {
    if (!Supported(source, source_width, source_height, destination, width, height, channels, matrix)) return false;
    switch (channels) {
      case 1: Affine<1>(source, source_width, source_height, destination, width, height, matrix, replicate_border, border_value); break;
      case 3: Affine<3>(source, source_width, source_height, destination, width, height, matrix, replicate_border, border_value); break;
      case 4: Affine<4>(source, source_width, source_height, destination, width, height, matrix, replicate_border, border_value); break;
    }
    return true;
}

}  // namespace x86
}  // namespace okcv
