#include "inspirecv/task/kernels/cpu/yuv_ops.h"

#include <algorithm>
#include <cstring>

#include "inspirecv/task/platform/cpu_features.h"

#if defined(INSPIRECV_TASK_USE_SSE) && defined(__SSE4_1__)
#include <x86intrin.h>
#endif

namespace inspirecv {
namespace task {
namespace kernels {
namespace yuv {
namespace {

enum class PixelOrder { kRgb, kBgr, kRgba, kBgra };

struct Color {
    int red;
    int green;
    int blue;
};

Color Decode(uint8_t luma, uint8_t v, uint8_t u) {
    const int scaled_luma = static_cast<int>(luma) << 6;
    const int centered_u = static_cast<int>(u) - 128;
    const int centered_v = static_cast<int>(v) - 128;
    return {
      std::min(std::max((scaled_luma + 73 * centered_v) >> 6, 0), 255),
      std::min(std::max((scaled_luma - 25 * centered_u - 37 * centered_v) >> 6,
                        0),
               255),
      std::min(std::max((scaled_luma + 130 * centered_u) >> 6, 0), 255)};
}

template <PixelOrder kOrder>
void StoreColor(uint8_t* destination, const Color& color) {
    constexpr bool kBlueFirst =
      kOrder == PixelOrder::kBgr || kOrder == PixelOrder::kBgra;
    destination[0] = static_cast<uint8_t>(kBlueFirst ? color.blue : color.red);
    destination[1] = static_cast<uint8_t>(color.green);
    destination[2] = static_cast<uint8_t>(kBlueFirst ? color.red : color.blue);
    if (kOrder == PixelOrder::kRgba || kOrder == PixelOrder::kBgra) {
        destination[3] = 255;
    }
}

template <PixelOrder kOrder>
void ConvertScalar(const uint8_t* luma, const uint8_t* vu, uint8_t* destination,
                   size_t first, size_t count) {
    constexpr size_t kChannels =
      kOrder == PixelOrder::kRgb || kOrder == PixelOrder::kBgr ? 3 : 4;
    for (size_t pixel = first; pixel < count; ++pixel) {
        const size_t chroma = (pixel / 2) * 2;
        StoreColor<kOrder>(destination + kChannels * pixel,
                           Decode(luma[pixel], vu[chroma], vu[chroma + 1]));
    }
}

#if defined(INSPIRECV_TASK_USE_NEON)
extern "C" {
void inspirecv_task_nv21_to_rgb_arm(const uint8_t*, uint8_t*, size_t,
                                     const uint8_t*);
void inspirecv_task_nv21_to_bgr_arm(const uint8_t*, uint8_t*, size_t,
                                     const uint8_t*);
void inspirecv_task_nv21_to_rgba_arm(const uint8_t*, uint8_t*, size_t,
                                      const uint8_t*);
void inspirecv_task_nv21_to_bgra_arm(const uint8_t*, uint8_t*, size_t,
                                      const uint8_t*);
}

template <PixelOrder kOrder>
void ConvertNeon(const uint8_t* source, uint8_t* destination, size_t blocks,
                 const uint8_t* vu) {
    if (kOrder == PixelOrder::kRgb) {
        inspirecv_task_nv21_to_rgb_arm(source, destination, blocks, vu);
    } else if (kOrder == PixelOrder::kBgr) {
        inspirecv_task_nv21_to_bgr_arm(source, destination, blocks, vu);
    } else if (kOrder == PixelOrder::kRgba) {
        inspirecv_task_nv21_to_rgba_arm(source, destination, blocks, vu);
    } else {
        inspirecv_task_nv21_to_bgra_arm(source, destination, blocks, vu);
    }
}
#endif

#if defined(INSPIRECV_TASK_USE_SSE) && defined(__SSE4_1__)
struct RgbaBlock {
    __m128i quarter[4];
};

RgbaBlock DecodeBlock16(const uint8_t* luma, const uint8_t* vu) {
    const __m128i even_then_odd = _mm_setr_epi8(
      0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
    const __m128i restore_order = _mm_setr_epi8(
      0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);
    const __m128i zero = _mm_setzero_si128();
    const __m128i opaque = _mm_set1_epi8(-1);
    const __m128i midpoint = _mm_set1_epi16(128);

    const __m128i reordered_y =
      _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(luma)),
                       even_then_odd);
    const __m128i reordered_vu =
      _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(vu)),
                       even_then_odd);
    const __m128i y_even =
      _mm_mullo_epi16(_mm_unpacklo_epi8(reordered_y, zero), _mm_set1_epi16(64));
    const __m128i y_odd =
      _mm_mullo_epi16(_mm_unpackhi_epi8(reordered_y, zero), _mm_set1_epi16(64));
    const __m128i u =
      _mm_sub_epi16(_mm_unpackhi_epi8(reordered_vu, zero), midpoint);
    const __m128i v =
      _mm_sub_epi16(_mm_unpacklo_epi8(reordered_vu, zero), midpoint);

    const __m128i r_bias = _mm_mullo_epi16(v, _mm_set1_epi16(73));
    const __m128i g_bias = _mm_add_epi16(
      _mm_mullo_epi16(u, _mm_set1_epi16(25)),
      _mm_mullo_epi16(v, _mm_set1_epi16(37)));
    const __m128i b_bias = _mm_mullo_epi16(u, _mm_set1_epi16(130));
    const __m128i shift = _mm_set1_epi16(1024);

    __m128i red = _mm_packus_epi16(
      _mm_mulhi_epi16(_mm_add_epi16(y_even, r_bias), shift),
      _mm_mulhi_epi16(_mm_add_epi16(y_odd, r_bias), shift));
    __m128i green = _mm_packus_epi16(
      _mm_mulhi_epi16(_mm_sub_epi16(y_even, g_bias), shift),
      _mm_mulhi_epi16(_mm_sub_epi16(y_odd, g_bias), shift));
    __m128i blue = _mm_packus_epi16(
      _mm_mulhi_epi16(_mm_add_epi16(y_even, b_bias), shift),
      _mm_mulhi_epi16(_mm_add_epi16(y_odd, b_bias), shift));
    red = _mm_shuffle_epi8(red, restore_order);
    green = _mm_shuffle_epi8(green, restore_order);
    blue = _mm_shuffle_epi8(blue, restore_order);

    const __m128i rg_low = _mm_unpacklo_epi8(red, green);
    const __m128i rg_high = _mm_unpackhi_epi8(red, green);
    const __m128i ba_low = _mm_unpacklo_epi8(blue, opaque);
    const __m128i ba_high = _mm_unpackhi_epi8(blue, opaque);
    return {{_mm_unpacklo_epi16(rg_low, ba_low),
             _mm_unpackhi_epi16(rg_low, ba_low),
             _mm_unpacklo_epi16(rg_high, ba_high),
             _mm_unpackhi_epi16(rg_high, ba_high)}};
}

template <PixelOrder kOrder>
void StoreBlock16(uint8_t* destination, const RgbaBlock& block) {
    constexpr bool kBlueFirst =
      kOrder == PixelOrder::kBgr || kOrder == PixelOrder::kBgra;
    constexpr bool kHasAlpha =
      kOrder == PixelOrder::kRgba || kOrder == PixelOrder::kBgra;
    const __m128i reverse = _mm_setr_epi8(
      2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15);
    const __m128i compact_rgb = kBlueFirst
      ? _mm_setr_epi8(2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12,
                      -1, -1, -1, -1)
      : _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14,
                      -1, -1, -1, -1);

    for (size_t quarter = 0; quarter < 4; ++quarter) {
        if (kHasAlpha) {
            const __m128i output =
              kBlueFirst ? _mm_shuffle_epi8(block.quarter[quarter], reverse)
                         : block.quarter[quarter];
            _mm_storeu_si128(reinterpret_cast<__m128i*>(destination + 16 * quarter),
                             output);
        } else {
            const __m128i output =
              _mm_shuffle_epi8(block.quarter[quarter], compact_rgb);
            alignas(16) uint8_t packed[16];
            _mm_store_si128(reinterpret_cast<__m128i*>(packed), output);
            std::memcpy(destination + 12 * quarter, packed, 12);
        }
    }
}

template <PixelOrder kOrder>
size_t ConvertSse(const uint8_t* luma, const uint8_t* vu, uint8_t* destination,
                  size_t count) {
    constexpr size_t kChannels =
      kOrder == PixelOrder::kRgb || kOrder == PixelOrder::kBgr ? 3 : 4;
    constexpr size_t kReservedBlocks = kChannels == 3 ? 2 : 1;
    const size_t available_blocks = count / 16;
    const size_t blocks =
      available_blocks > kReservedBlocks ? available_blocks - kReservedBlocks : 0;
    for (size_t block = 0; block < blocks; ++block) {
        StoreBlock16<kOrder>(destination + kChannels * 16 * block,
                             DecodeBlock16(luma + 16 * block, vu + 16 * block));
    }
    return blocks * 16;
}
#endif

template <PixelOrder kOrder>
void Convert(const uint8_t* source, uint8_t* destination, size_t count) {
    const uint8_t* luma = source;
    const uint8_t* vu = source + count;
    size_t completed = 0;
#if defined(INSPIRECV_TASK_USE_SSE) && defined(__SSE4_1__)
    if (platform::HasSse41()) {
        completed = ConvertSse<kOrder>(luma, vu, destination, count);
    }
#elif defined(INSPIRECV_TASK_USE_NEON)
    const size_t blocks = count / 16;
    if (blocks != 0) {
        ConvertNeon<kOrder>(source, destination, blocks, vu);
        completed = blocks * 16;
    }
#endif
    ConvertScalar<kOrder>(luma, vu, destination, completed, count);
}

}  // namespace

void ToRgb(const uint8_t* source, uint8_t* destination, size_t count) {
    Convert<PixelOrder::kRgb>(source, destination, count);
}

void ToBgr(const uint8_t* source, uint8_t* destination, size_t count) {
    Convert<PixelOrder::kBgr>(source, destination, count);
}

void ToRgba(const uint8_t* source, uint8_t* destination, size_t count) {
    Convert<PixelOrder::kRgba>(source, destination, count);
}

void ToBgra(const uint8_t* source, uint8_t* destination, size_t count) {
    Convert<PixelOrder::kBgra>(source, destination, count);
}

}  // namespace yuv
}  // namespace kernels
}  // namespace task
}  // namespace inspirecv
