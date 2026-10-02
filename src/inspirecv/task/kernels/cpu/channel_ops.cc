#include "inspirecv/task/kernels/cpu/channel_ops.h"

#include <cstring>

#include "inspirecv/task/platform/cpu_features.h"

#if defined(INSPIRECV_TASK_HAVE_AVX2_KERNELS)
#include "inspirecv/task/kernels/x86/avx2_kernels.h"
#endif

#if defined(INSPIRECV_TASK_USE_NEON)
#include <arm_neon.h>
#endif

#if defined(INSPIRECV_TASK_HAVE_SSE41_INTRINSICS)
#if defined(_MSC_VER)
#include <immintrin.h>
#else
#include <x86intrin.h>
#endif
#endif

namespace inspirecv {
namespace task {
namespace kernels {
namespace channel {
namespace {

template <size_t kPixelBytes>
void CopyPixels(const uint8_t* source, uint8_t* destination, size_t count) {
    std::memcpy(destination, source, kPixelBytes * count);
}

template <int kSourceWidth, int kDestinationWidth, int kLane0, int kLane1,
          int kLane2 = 0, int kLane3 = 0>
void RemapTail(const uint8_t* source, uint8_t* destination, size_t count) {
    constexpr int lanes[4] = {kLane0, kLane1, kLane2, kLane3};
    for (size_t pixel = 0; pixel < count; ++pixel) {
        // Exact in-place color reversal must not overwrite a source lane
        // before a later output lane reads it. SIMD blocks already load the
        // entire pixel group before storing; give their tails the same rule.
        uint8_t input[kSourceWidth];
        std::memcpy(input, source, kSourceWidth);
        for (int lane = 0; lane < kDestinationWidth; ++lane) {
            destination[lane] = lanes[lane] < 0 ? UINT8_C(255) : input[lanes[lane]];
        }
        source += kSourceWidth;
        destination += kDestinationWidth;
    }
}

template <int kSourceWidth, int kRedLane, int kGreenLane, int kBlueLane>
void LumaTail(const uint8_t* source, uint8_t* destination, size_t count) {
    for (size_t pixel = 0; pixel < count; ++pixel) {
        const unsigned weighted = 19u * source[kRedLane] +
                                  38u * source[kGreenLane] +
                                  7u * source[kBlueLane];
        *destination++ = static_cast<uint8_t>(weighted >> 6);
        source += kSourceWidth;
    }
}

template <size_t kPixelBytes>
void FillPixels(const uint8_t* color, uint8_t* destination, size_t count) {
    for (size_t pixel = 0; pixel < count; ++pixel) {
        std::memcpy(destination, color, kPixelBytes);
        destination += kPixelBytes;
    }
}

#if defined(INSPIRECV_TASK_HAVE_SSE41_INTRINSICS)
__m128i LoadTriple4(const uint8_t* source) {
    const __m128i first = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(source));
    int32_t last;
    std::memcpy(&last, source + 8, sizeof(last));
    return _mm_unpacklo_epi64(first, _mm_cvtsi32_si128(last));
}

void StoreTriple4(uint8_t* destination, __m128i rgba) {
    const __m128i compact = _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9,
                                         10, 12, 13, 14, -1, -1, -1, -1);
    const __m128i packed = _mm_shuffle_epi8(rgba, compact);
    _mm_storel_epi64(reinterpret_cast<__m128i*>(destination), packed);
    const int32_t last = _mm_cvtsi128_si32(_mm_srli_si128(packed, 8));
    std::memcpy(destination + 8, &last, sizeof(last));
}

template <int kSourceChannels, int kDestinationChannels, bool kReverse>
size_t RemapSse(const uint8_t*& source, uint8_t*& destination, size_t count) {
    if (!platform::HasSse41()) return count;
    const __m128i expand = _mm_setr_epi8(0, 1, 2, -1, 3, 4, 5, -1,
                                        6, 7, 8, -1, 9, 10, 11, -1);
    const __m128i reverse = _mm_setr_epi8(2, 1, 0, 3, 6, 5, 4, 7,
                                         10, 9, 8, 11, 14, 13, 12, 15);
    const __m128i opaque = _mm_set1_epi32(static_cast<int>(0xff000000u));
    while (count >= 4) {
        __m128i pixels;
        if (kSourceChannels == 1) {
            int32_t packed;
            std::memcpy(&packed, source, sizeof(packed));
            pixels = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(packed));
            pixels = _mm_mullo_epi32(pixels, _mm_set1_epi32(0x010101));
        } else if (kSourceChannels == 3) {
            pixels = _mm_shuffle_epi8(LoadTriple4(source), expand);
        } else {
            pixels = _mm_loadu_si128(reinterpret_cast<const __m128i*>(source));
        }
        if (kReverse) pixels = _mm_shuffle_epi8(pixels, reverse);
        if (kDestinationChannels == 4) {
            if (kSourceChannels != 4) pixels = _mm_or_si128(pixels, opaque);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(destination), pixels);
        } else {
            StoreTriple4(destination, pixels);
        }
        source += 4 * kSourceChannels;
        destination += 4 * kDestinationChannels;
        count -= 4;
    }
    return count;
}

template <int kSourceChannels, bool kBlueFirst>
size_t LumaSse(const uint8_t*& source, uint8_t*& destination, size_t count) {
    if (!platform::HasSse41()) return count;
    const __m128i expand = _mm_setr_epi8(0, 1, 2, -1, 3, 4, 5, -1,
                                        6, 7, 8, -1, 9, 10, 11, -1);
    const __m128i weights = _mm_set1_epi32(kBlueFirst ? (7 | (38 << 8) | (19 << 16))
                                                                   : (19 | (38 << 8) | (7 << 16)));
    while (count >= 4) {
        const __m128i pixels = kSourceChannels == 3
                                ? _mm_shuffle_epi8(LoadTriple4(source), expand)
                                : _mm_loadu_si128(reinterpret_cast<const __m128i*>(source));
        const __m128i pairs = _mm_maddubs_epi16(pixels, weights);
        const __m128i weighted = _mm_srli_epi32(_mm_madd_epi16(pairs, _mm_set1_epi16(1)), 6);
        const __m128i words = _mm_packus_epi32(weighted, weighted);
        const int32_t packed = _mm_cvtsi128_si32(_mm_packus_epi16(words, words));
        std::memcpy(destination, &packed, sizeof(packed));
        source += 4 * kSourceChannels;
        destination += 4;
        count -= 4;
    }
    return count;
}
#endif

#if defined(INSPIRECV_TASK_USE_NEON)
template <int kRedLane, int kBlueLane>
size_t LumaTripleNeon(const uint8_t*& source, uint8_t*& destination, size_t count) {
    static_assert(kRedLane == 0 || kRedLane == 2, "red must be an outer C3 lane");
    static_assert(kBlueLane == 0 || kBlueLane == 2, "blue must be an outer C3 lane");
    const uint8x8_t redWeight = vdup_n_u8(19);
    const uint8x8_t greenWeight = vdup_n_u8(38);
    const uint8x8_t blueWeight = vdup_n_u8(7);
    size_t remaining = count;
    while (remaining >= 8) {
        const uint8x8x3_t pixels = vld3_u8(source);
        uint16x8_t weighted = vmull_u8(pixels.val[kRedLane], redWeight);
        weighted = vmlal_u8(weighted, pixels.val[1], greenWeight);
        weighted = vmlal_u8(weighted, pixels.val[kBlueLane], blueWeight);
        vst1_u8(destination, vshrn_n_u16(weighted, 6));
        source += 24;
        destination += 8;
        remaining -= 8;
    }
    return remaining;
}
#endif

}  // namespace

void CopyMonoPixels(const uint8_t* source, uint8_t* destination, size_t count) {
    CopyPixels<1>(source, destination, count);
}

void CopyTriplePixels(const uint8_t* source, uint8_t* destination, size_t count) {
    CopyPixels<3>(source, destination, count);
}

void CopyQuadPixels(const uint8_t* source, uint8_t* destination, size_t count) {
    CopyPixels<4>(source, destination, count);
}

void ReplicateMonoToTriple(const uint8_t* source, uint8_t* destination, size_t count) {
#if defined(INSPIRECV_TASK_HAVE_AVX2_KERNELS)
    if (count >= 8 && platform::HasAvx2()) {
        x86::ReplicateMonoToTripleAvx2(source, destination, count);
        return;
    }
#endif
#if defined(INSPIRECV_TASK_HAVE_SSE41_INTRINSICS)
    count = RemapSse<1, 3, false>(source, destination, count);
#endif
#if defined(INSPIRECV_TASK_USE_NEON)
    while (count >= 8) {
        const uint8x8_t gray = vld1_u8(source);
        const uint8x8x3_t tripled = {{gray, gray, gray}};
        vst3_u8(destination, tripled);
        source += 8;
        destination += 24;
        count -= 8;
    }
#endif
    RemapTail<1, 3, 0, 0, 0>(source, destination, count);
}

void ReplicateMonoToQuad(const uint8_t* source, uint8_t* destination, size_t count) {
#if defined(INSPIRECV_TASK_HAVE_AVX2_KERNELS)
    if (count >= 8 && platform::HasAvx2()) {
        x86::ReplicateMonoToQuadAvx2(source, destination, count);
        return;
    }
#endif
#if defined(INSPIRECV_TASK_HAVE_SSE41_INTRINSICS)
    count = RemapSse<1, 4, false>(source, destination, count);
#endif
#if defined(INSPIRECV_TASK_USE_NEON)
    const uint8x8_t opaque = vdup_n_u8(UINT8_C(255));
    while (count >= 8) {
        const uint8x8_t gray = vld1_u8(source);
        const uint8x8x4_t expanded = {{gray, gray, gray, opaque}};
        vst4_u8(destination, expanded);
        source += 8;
        destination += 32;
        count -= 8;
    }
#endif
    RemapTail<1, 4, 0, 0, 0, -1>(source, destination, count);
}

void AppendOpaqueAlpha(const uint8_t* source, uint8_t* destination, size_t count) {
#if defined(INSPIRECV_TASK_HAVE_AVX2_KERNELS)
    if (count >= 8 && platform::HasAvx2()) {
        x86::AppendOpaqueAlphaAvx2(source, destination, count);
        return;
    }
#endif
#if defined(INSPIRECV_TASK_HAVE_SSE41_INTRINSICS)
    count = RemapSse<3, 4, false>(source, destination, count);
#endif
#if defined(INSPIRECV_TASK_USE_NEON)
    const uint8x8_t opaque = vdup_n_u8(UINT8_C(255));
    while (count >= 8) {
        const uint8x8x3_t triple = vld3_u8(source);
        const uint8x8x4_t expanded = {
          {triple.val[0], triple.val[1], triple.val[2], opaque}};
        vst4_u8(destination, expanded);
        source += 24;
        destination += 32;
        count -= 8;
    }
#endif
    RemapTail<3, 4, 0, 1, 2, -1>(source, destination, count);
}

void ReverseTriple(const uint8_t* source, uint8_t* destination, size_t count) {
#if defined(INSPIRECV_TASK_HAVE_AVX2_KERNELS)
    if (count >= 8 && platform::HasAvx2()) {
        x86::ReverseTripleAvx2(source, destination, count);
        return;
    }
#endif
#if defined(INSPIRECV_TASK_HAVE_SSE41_INTRINSICS)
    count = RemapSse<3, 3, true>(source, destination, count);
#endif
#if defined(INSPIRECV_TASK_USE_NEON)
    while (count >= 8) {
        const uint8x8x3_t input = vld3_u8(source);
        const uint8x8x3_t reversed = {{input.val[2], input.val[1], input.val[0]}};
        vst3_u8(destination, reversed);
        source += 24;
        destination += 24;
        count -= 8;
    }
#endif
    RemapTail<3, 3, 2, 1, 0>(source, destination, count);
}

void ReverseQuadColor(const uint8_t* source, uint8_t* destination, size_t count) {
#if defined(INSPIRECV_TASK_HAVE_AVX2_KERNELS)
    if (count >= 8 && platform::HasAvx2()) {
        x86::ReverseQuadColorAvx2(source, destination, count);
        return;
    }
#endif
#if defined(INSPIRECV_TASK_HAVE_SSE41_INTRINSICS)
    if (platform::HasSse41()) {
        const __m128i order =
          _mm_setr_epi8(2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15);
        while (count >= 4) {
            const __m128i input =
              _mm_loadu_si128(reinterpret_cast<const __m128i*>(source));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(destination),
                             _mm_shuffle_epi8(input, order));
            source += 16;
            destination += 16;
            count -= 4;
        }
    }
#endif
#if defined(INSPIRECV_TASK_USE_NEON)
    while (count >= 8) {
        const uint8x8x4_t input = vld4_u8(source);
        const uint8x8x4_t reversed = {
          {input.val[2], input.val[1], input.val[0], input.val[3]}};
        vst4_u8(destination, reversed);
        source += 32;
        destination += 32;
        count -= 8;
    }
#endif
    RemapTail<4, 4, 2, 1, 0, 3>(source, destination, count);
}

void DropAlpha(const uint8_t* source, uint8_t* destination, size_t count) {
#if defined(INSPIRECV_TASK_HAVE_AVX2_KERNELS)
    if (count >= 8 && platform::HasAvx2()) {
        x86::DropAlphaAvx2(source, destination, count);
        return;
    }
#endif
#if defined(INSPIRECV_TASK_HAVE_SSE41_INTRINSICS)
    count = RemapSse<4, 3, false>(source, destination, count);
#endif
#if defined(INSPIRECV_TASK_USE_NEON)
    while (count >= 8) {
        const uint8x8x4_t input = vld4_u8(source);
        const uint8x8x3_t triple = {{input.val[0], input.val[1], input.val[2]}};
        vst3_u8(destination, triple);
        source += 32;
        destination += 24;
        count -= 8;
    }
#endif
    RemapTail<4, 3, 0, 1, 2>(source, destination, count);
}

void ReverseAndDropAlpha(const uint8_t* source, uint8_t* destination, size_t count) {
#if defined(INSPIRECV_TASK_HAVE_AVX2_KERNELS)
    if (count >= 8 && platform::HasAvx2()) {
        x86::ReverseAndDropAlphaAvx2(source, destination, count);
        return;
    }
#endif
#if defined(INSPIRECV_TASK_HAVE_SSE41_INTRINSICS)
    count = RemapSse<4, 3, true>(source, destination, count);
#endif
#if defined(INSPIRECV_TASK_USE_NEON)
    while (count >= 8) {
        const uint8x8x4_t input = vld4_u8(source);
        const uint8x8x3_t triple = {{input.val[2], input.val[1], input.val[0]}};
        vst3_u8(destination, triple);
        source += 32;
        destination += 24;
        count -= 8;
    }
#endif
    RemapTail<4, 3, 2, 1, 0>(source, destination, count);
}

void LumaFromRgb(const uint8_t* source, uint8_t* destination, size_t count) {
#if defined(INSPIRECV_TASK_HAVE_AVX2_KERNELS)
    if (count >= 8 && platform::HasAvx2()) {
        x86::LumaFromRgbAvx2(source, destination, count);
        return;
    }
#endif
#if defined(INSPIRECV_TASK_USE_NEON)
    count = LumaTripleNeon<0, 2>(source, destination, count);
#endif
#if defined(INSPIRECV_TASK_HAVE_SSE41_INTRINSICS)
    count = LumaSse<3, false>(source, destination, count);
#endif
    LumaTail<3, 0, 1, 2>(source, destination, count);
}

void LumaFromBgr(const uint8_t* source, uint8_t* destination, size_t count) {
#if defined(INSPIRECV_TASK_HAVE_AVX2_KERNELS)
    if (count >= 8 && platform::HasAvx2()) {
        x86::LumaFromBgrAvx2(source, destination, count);
        return;
    }
#endif
#if defined(INSPIRECV_TASK_USE_NEON)
    count = LumaTripleNeon<2, 0>(source, destination, count);
#endif
#if defined(INSPIRECV_TASK_HAVE_SSE41_INTRINSICS)
    count = LumaSse<3, true>(source, destination, count);
#endif
    LumaTail<3, 2, 1, 0>(source, destination, count);
}

void LumaFromRgba(const uint8_t* source, uint8_t* destination, size_t count) {
#if defined(INSPIRECV_TASK_HAVE_AVX2_KERNELS)
    if (count >= 8 && platform::HasAvx2()) {
        x86::LumaFromRgbaAvx2(source, destination, count);
        return;
    }
#endif
#if defined(INSPIRECV_TASK_HAVE_SSE41_INTRINSICS)
    count = LumaSse<4, false>(source, destination, count);
#endif
    LumaTail<4, 0, 1, 2>(source, destination, count);
}

void LumaFromBgra(const uint8_t* source, uint8_t* destination, size_t count) {
#if defined(INSPIRECV_TASK_HAVE_AVX2_KERNELS)
    if (count >= 8 && platform::HasAvx2()) {
        x86::LumaFromBgraAvx2(source, destination, count);
        return;
    }
#endif
#if defined(INSPIRECV_TASK_HAVE_SSE41_INTRINSICS)
    count = LumaSse<4, true>(source, destination, count);
#endif
    LumaTail<4, 2, 1, 0>(source, destination, count);
}

void FillMonoPixels(const uint8_t* color, uint8_t* destination, size_t count) {
    FillPixels<1>(color, destination, count);
}

void FillTriplePixels(const uint8_t* color, uint8_t* destination, size_t count) {
    FillPixels<3>(color, destination, count);
}

void FillQuadPixels(const uint8_t* color, uint8_t* destination, size_t count) {
    FillPixels<4>(color, destination, count);
}

}  // namespace channel
}  // namespace kernels
}  // namespace task
}  // namespace inspirecv
