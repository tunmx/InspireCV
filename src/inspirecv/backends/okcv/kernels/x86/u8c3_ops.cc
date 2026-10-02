#include "u8c3_ops.h"
#include "inspirecv/core/runtime/cpu_features.h"

#include <cstddef>
#include <cstring>

#if defined(INSPIRECV_HAVE_SSSE3_INTRINSICS)
#include <tmmintrin.h>
#endif

namespace okcv {
namespace x86 {

namespace {

// Keep generic scalar helpers inside the ISA translation unit.
template<typename T> T LocalMin(T a, T b) { return b < a ? b : a; }

inline void CopyPixel3(const uint8_t* source, uint8_t* destination) {
    destination[0] = source[0];
    destination[1] = source[1];
    destination[2] = source[2];
}

inline void CopyPixel4(const uint8_t* source, uint8_t* destination) {
    uint32_t value = 0;
    std::memcpy(&value, source, sizeof(value));
    std::memcpy(destination, &value, sizeof(value));
}

#if defined(INSPIRECV_HAVE_SSSE3_INTRINSICS)

inline __m128i Load12(const uint8_t* source) {
    const __m128i low =
      _mm_loadl_epi64(reinterpret_cast<const __m128i*>(source));
    uint32_t high = 0;
    std::memcpy(&high, source + 8, sizeof(high));
    return _mm_or_si128(
      low, _mm_slli_si128(_mm_cvtsi32_si128(static_cast<int>(high)), 8));
}

inline void Store12(uint8_t* destination, __m128i value) {
    _mm_storel_epi64(reinterpret_cast<__m128i*>(destination), value);
    const uint32_t high = static_cast<uint32_t>(
      _mm_cvtsi128_si32(_mm_srli_si128(value, 8)));
    std::memcpy(destination + 8, &high, sizeof(high));
}

inline void StoreAlternatingPixels3(uint8_t* destination, __m128i even, __m128i odd) {
    const __m128i even_low = _mm_setr_epi8(0,1,2,-1,-1,-1,3,4,5,-1,-1,-1,-1,-1,-1,-1);
    const __m128i odd_low = _mm_setr_epi8(-1,-1,-1,0,1,2,-1,-1,-1,3,4,5,-1,-1,-1,-1);
    const __m128i even_high = _mm_setr_epi8(6,7,8,-1,-1,-1,9,10,11,-1,-1,-1,-1,-1,-1,-1);
    const __m128i odd_high = _mm_setr_epi8(-1,-1,-1,6,7,8,-1,-1,-1,9,10,11,-1,-1,-1,-1);
    Store12(destination, _mm_or_si128(_mm_shuffle_epi8(even, even_low),
                                      _mm_shuffle_epi8(odd, odd_low)));
    Store12(destination + 12, _mm_or_si128(_mm_shuffle_epi8(even, even_high),
                                           _mm_shuffle_epi8(odd, odd_high)));
}

inline __m128i AverageFourBytes(__m128i a, __m128i b, __m128i c, __m128i d) {
    const __m128i zero = _mm_setzero_si128(), two = _mm_set1_epi16(2);
    const __m128i low = _mm_add_epi16(_mm_add_epi16(_mm_unpacklo_epi8(a, zero), _mm_unpacklo_epi8(b, zero)),
                                      _mm_add_epi16(_mm_unpacklo_epi8(c, zero), _mm_unpacklo_epi8(d, zero)));
    const __m128i high = _mm_add_epi16(_mm_add_epi16(_mm_unpackhi_epi8(a, zero), _mm_unpackhi_epi8(b, zero)),
                                       _mm_add_epi16(_mm_unpackhi_epi8(c, zero), _mm_unpackhi_epi8(d, zero)));
    return _mm_packus_epi16(_mm_srli_epi16(_mm_add_epi16(low, two), 2),
                             _mm_srli_epi16(_mm_add_epi16(high, two), 2));
}

#endif

}  // namespace

void ResizeBilinear2xU8C3(const uint8_t* source, uint8_t* destination,
                         int width, int height) {
    const std::size_t source_stride = static_cast<std::size_t>(width) * 3;
    const std::size_t destination_stride = source_stride * 2;
    for (int y = 0; y < height; ++y) {
        const uint8_t* top = source + static_cast<std::size_t>(y) * source_stride;
        const uint8_t* bottom = source + static_cast<std::size_t>(LocalMin(y + 1, height - 1)) * source_stride;
        uint8_t* even = destination + static_cast<std::size_t>(y) * 2 * destination_stride;
        uint8_t* odd = even + destination_stride;
        int x = 0;
#if defined(INSPIRECV_HAVE_SSSE3_INTRINSICS)
        // Four complete source pixels plus their right neighbors. Load12 keeps
        // the final vector inside a tight RGB row; no fourth channel is read.
        for (; x + 4 < width; x += 4) {
            const __m128i tl = Load12(top + x * 3), tr = Load12(top + (x + 1) * 3);
            const __m128i bl = Load12(bottom + x * 3), br = Load12(bottom + (x + 1) * 3);
            StoreAlternatingPixels3(even + x * 6, tl, _mm_avg_epu8(tl, tr));
            StoreAlternatingPixels3(odd + x * 6, _mm_avg_epu8(tl, bl), AverageFourBytes(tl, tr, bl, br));
        }
#endif
        for (; x < width; ++x) {
            const int right = LocalMin(x + 1, width - 1);
            for (int c = 0; c < 3; ++c) {
                const unsigned tl = top[x * 3 + c], tr = top[right * 3 + c];
                const unsigned bl = bottom[x * 3 + c], br = bottom[right * 3 + c];
                even[x * 6 + c] = static_cast<uint8_t>(tl);
                even[x * 6 + 3 + c] = static_cast<uint8_t>((tl + tr + 1) / 2);
                odd[x * 6 + c] = static_cast<uint8_t>((tl + bl + 1) / 2);
                odd[x * 6 + 3 + c] = static_cast<uint8_t>((tl + tr + bl + br + 2) / 4);
            }
        }
    }
}

void SwapRbU8C3(const uint8_t* source, uint8_t* destination,
                int width, int height) {
    const std::size_t row_bytes = static_cast<std::size_t>(width) * 3;
    for (int y = 0; y < height; ++y) {
        const uint8_t* source_row = source + static_cast<std::size_t>(y) * row_bytes;
        uint8_t* destination_row =
          destination + static_cast<std::size_t>(y) * row_bytes;
        int x = 0;
#if defined(INSPIRECV_HAVE_SSSE3_INTRINSICS)
        const __m128i swap_five = _mm_setr_epi8(
          2, 1, 0, 5, 4, 3, 8, 7, 6, 11, 10, 9, 14, 13, 12, -1);
        // Fifteen useful bytes per shuffle lowers instruction and redundant
        // store traffic. The extra byte is always overwritten by the next
        // vector block or by the scalar tail.
        for (; x + 10 < width; x += 10) {
            const std::size_t byte_offset = static_cast<std::size_t>(x) * 3;
            const __m128i first = _mm_loadu_si128(
              reinterpret_cast<const __m128i*>(source_row + byte_offset));
            const __m128i second = _mm_loadu_si128(
              reinterpret_cast<const __m128i*>(source_row + byte_offset + 15));
            _mm_storeu_si128(
              reinterpret_cast<__m128i*>(destination_row + byte_offset),
              _mm_shuffle_epi8(first, swap_five));
            _mm_storeu_si128(
              reinterpret_cast<__m128i*>(destination_row + byte_offset + 15),
              _mm_shuffle_epi8(second, swap_five));
        }
        for (; x + 5 < width; x += 5) {
            const std::size_t byte_offset = static_cast<std::size_t>(x) * 3;
            const __m128i pixels = _mm_loadu_si128(
              reinterpret_cast<const __m128i*>(source_row + byte_offset));
            _mm_storeu_si128(
              reinterpret_cast<__m128i*>(destination_row + byte_offset),
              _mm_shuffle_epi8(pixels, swap_five));
        }
#endif
        for (; x < width; ++x) {
            const uint8_t* input = source_row + static_cast<std::size_t>(x) * 3;
            uint8_t* output = destination_row + static_cast<std::size_t>(x) * 3;
            output[0] = input[2];
            output[1] = input[1];
            output[2] = input[0];
        }
    }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((aligned(32)))
#endif
void FlipHorizontalU8C3(const uint8_t* source, uint8_t* destination,
                        int width, int height, bool reverse_rows) {
    const std::size_t row_bytes = static_cast<std::size_t>(width) * 3;
    for (int y = 0; y < height; ++y) {
        const uint8_t* source_row = source + static_cast<std::size_t>(reverse_rows ? height - 1 - y : y) * row_bytes;
        uint8_t* destination_row =
          destination + static_cast<std::size_t>(y) * row_bytes;
        int x = 0;
#if defined(INSPIRECV_HAVE_SSSE3_INTRINSICS)
        const __m128i reverse_four = _mm_setr_epi8(
          9, 10, 11, 6, 7, 8, 3, 4, 5, 0, 1, 2, -1, -1, -1, -1);
        for (; x + 4 <= width; x += 4) {
            const int source_x = width - x - 4;
            const std::size_t source_offset =
              static_cast<std::size_t>(source_x) * 3;
            const __m128i pixels = source_offset + 16 <= row_bytes
                                      ? _mm_loadu_si128(
                                          reinterpret_cast<const __m128i*>(
                                            source_row + source_offset))
                                      : Load12(source_row + source_offset);
            const std::size_t destination_offset =
              static_cast<std::size_t>(x) * 3;
            const __m128i reversed =
              _mm_shuffle_epi8(pixels, reverse_four);
            if (destination_offset + 16 <= row_bytes) {
                _mm_storeu_si128(reinterpret_cast<__m128i*>(
                                   destination_row + destination_offset),
                                 reversed);
            } else {
                Store12(destination_row + destination_offset, reversed);
            }
        }
#endif
        for (; x < width; ++x) {
            CopyPixel3(source_row + static_cast<std::size_t>(width - 1 - x) * 3,
                       destination_row + static_cast<std::size_t>(x) * 3);
        }
    }
}

void Rotate90U8C3(const uint8_t* source, uint8_t* destination,
                  int width, int height) {
    const int tile = (height <= 256 || height >= 2048) ? 256 : 32;
    const std::size_t source_stride = static_cast<std::size_t>(width) * 3;
    const std::size_t destination_stride = static_cast<std::size_t>(height) * 3;
    for (int source_y_begin = 0; source_y_begin < height;
         source_y_begin += tile) {
        const int source_y_end = LocalMin(source_y_begin + tile, height);
        for (int source_x_begin = 0; source_x_begin < width;
             source_x_begin += tile) {
            const int source_x_end = LocalMin(source_x_begin + tile, width);
            for (int source_x = source_x_begin; source_x < source_x_end;
                 ++source_x) {
                uint8_t* destination_pixel =
                  destination +
                  static_cast<std::size_t>(source_x) * destination_stride +
                  static_cast<std::size_t>(height - source_y_end) * 3;
                const uint8_t* source_pixel =
                  source + static_cast<std::size_t>(source_y_end - 1) *
                             source_stride +
                  static_cast<std::size_t>(source_x) * 3;
                if (source_x + 1 < width) {
                    int remaining = source_y_end - source_y_begin - 1;
                    for (; remaining >= 8; remaining -= 8) {
                        CopyPixel4(source_pixel, destination_pixel);
                        CopyPixel4(source_pixel - source_stride,
                                   destination_pixel + 3);
                        CopyPixel4(source_pixel - source_stride * 2,
                                   destination_pixel + 6);
                        CopyPixel4(source_pixel - source_stride * 3,
                                   destination_pixel + 9);
                        CopyPixel4(source_pixel - source_stride * 4,
                                   destination_pixel + 12);
                        CopyPixel4(source_pixel - source_stride * 5,
                                   destination_pixel + 15);
                        CopyPixel4(source_pixel - source_stride * 6,
                                   destination_pixel + 18);
                        CopyPixel4(source_pixel - source_stride * 7,
                                   destination_pixel + 21);
                        source_pixel -= source_stride * 8;
                        destination_pixel += 24;
                    }
                    for (; remaining > 0; --remaining) {
                        CopyPixel4(source_pixel, destination_pixel);
                        source_pixel -= source_stride;
                        destination_pixel += 3;
                    }
                    CopyPixel3(source_pixel, destination_pixel);
                } else {
                    for (int source_y = source_y_end - 1;
                         source_y >= source_y_begin; --source_y) {
                        CopyPixel3(source_pixel, destination_pixel);
                        destination_pixel += 3;
                        if (source_y == source_y_begin) break;
                        source_pixel -= source_stride;
                    }
                }
            }
        }
    }
}

void Rotate270U8C3(const uint8_t* source, uint8_t* destination,
                  int width, int height) {
    const int tile = (height <= 256 || height >= 2048) ? 256 : 32;
    const std::size_t source_stride = static_cast<std::size_t>(width) * 3;
    const std::size_t destination_stride = static_cast<std::size_t>(height) * 3;
    for (int source_y_begin = 0; source_y_begin < height;
         source_y_begin += tile) {
        const int source_y_end = LocalMin(source_y_begin + tile, height);
        for (int source_x_begin = 0; source_x_begin < width;
             source_x_begin += tile) {
            const int source_x_end = LocalMin(source_x_begin + tile, width);
            for (int source_x = source_x_begin; source_x < source_x_end;
                 ++source_x) {
                uint8_t* destination_pixel =
                  destination +
                  static_cast<std::size_t>(width - 1 - source_x) * destination_stride +
                  static_cast<std::size_t>(source_y_begin) * 3;
                const uint8_t* source_pixel =
                  source + static_cast<std::size_t>(source_y_begin) *
                             source_stride +
                  static_cast<std::size_t>(source_x) * 3;
                if (source_x + 1 < width) {
                    int remaining = source_y_end - source_y_begin - 1;
                    for (; remaining >= 8; remaining -= 8) {
                        CopyPixel4(source_pixel, destination_pixel);
                        CopyPixel4(source_pixel + source_stride,
                                   destination_pixel + 3);
                        CopyPixel4(source_pixel + source_stride * 2,
                                   destination_pixel + 6);
                        CopyPixel4(source_pixel + source_stride * 3,
                                   destination_pixel + 9);
                        CopyPixel4(source_pixel + source_stride * 4,
                                   destination_pixel + 12);
                        CopyPixel4(source_pixel + source_stride * 5,
                                   destination_pixel + 15);
                        CopyPixel4(source_pixel + source_stride * 6,
                                   destination_pixel + 18);
                        CopyPixel4(source_pixel + source_stride * 7,
                                   destination_pixel + 21);
                        source_pixel += source_stride * 8;
                        destination_pixel += 24;
                    }
                    for (; remaining > 0; --remaining) {
                        CopyPixel4(source_pixel, destination_pixel);
                        source_pixel += source_stride;
                        destination_pixel += 3;
                    }
                    CopyPixel3(source_pixel, destination_pixel);
                } else {
                    for (int source_y = source_y_begin;
                         source_y < source_y_end; ++source_y) {
                        CopyPixel3(source_pixel, destination_pixel);
                        destination_pixel += 3;
                        if (source_y == source_y_end - 1) break;
                        source_pixel += source_stride;
                    }
                }
            }
        }
    }
}

}  // namespace x86
}  // namespace okcv
