#include "inspirecv/task/kernels/cpu/channel_ops.h"

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
        for (int lane = 0; lane < kDestinationWidth; ++lane) {
            destination[lane] = lanes[lane] < 0 ? UINT8_C(255) : source[lanes[lane]];
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
#if defined(INSPIRECV_TASK_USE_SSE) && defined(__SSE4_1__)
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
#if defined(INSPIRECV_TASK_USE_NEON)
    count = LumaTripleNeon<0, 2>(source, destination, count);
#endif
    LumaTail<3, 0, 1, 2>(source, destination, count);
}

void LumaFromBgr(const uint8_t* source, uint8_t* destination, size_t count) {
#if defined(INSPIRECV_TASK_USE_NEON)
    count = LumaTripleNeon<2, 0>(source, destination, count);
#endif
    LumaTail<3, 2, 1, 0>(source, destination, count);
}

void LumaFromRgba(const uint8_t* source, uint8_t* destination, size_t count) {
    LumaTail<4, 0, 1, 2>(source, destination, count);
}

void LumaFromBgra(const uint8_t* source, uint8_t* destination, size_t count) {
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
