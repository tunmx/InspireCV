#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <arm_neon.h>

namespace {

struct CoordinateBlock {
    float32x4_t x;
    float32x4_t y;
    float32x4_t step_x;
    float32x4_t step_y;
};

CoordinateBlock MakeCoordinateBlock(const float* line) {
    float x[4] = {line[0], 0.0f, 0.0f, 0.0f};
    float y[4] = {line[1], 0.0f, 0.0f, 0.0f};
    for (int lane = 1; lane < 4; ++lane) {
        x[lane] = x[lane - 1] + line[2];
        y[lane] = y[lane - 1] + line[3];
    }
    return {vld1q_f32(x), vld1q_f32(y), vdupq_n_f32(4.0f * line[2]),
            vdupq_n_f32(4.0f * line[3])};
}

int ClampIndex(int value, size_t maximum) {
    return std::max(0, std::min(value, static_cast<int>(maximum)));
}

void ResolveNearest4(CoordinateBlock* coordinates, size_t maximum_x,
                     size_t maximum_y, int x[4], int y[4]) {
#if defined(__aarch64__)
    int32x4_t xi = vcvtnq_s32_f32(coordinates->x);
    int32x4_t yi = vcvtnq_s32_f32(coordinates->y);
    xi = vmaxq_s32(vdupq_n_s32(0),
                   vminq_s32(xi, vdupq_n_s32(static_cast<int>(maximum_x))));
    yi = vmaxq_s32(vdupq_n_s32(0),
                   vminq_s32(yi, vdupq_n_s32(static_cast<int>(maximum_y))));
    vst1q_s32(x, xi);
    vst1q_s32(y, yi);
#else
    float xf[4];
    float yf[4];
    vst1q_f32(xf, coordinates->x);
    vst1q_f32(yf, coordinates->y);
    for (int lane = 0; lane < 4; ++lane) {
        x[lane] = ClampIndex(static_cast<int>(nearbyintf(xf[lane])), maximum_x);
        y[lane] = ClampIndex(static_cast<int>(nearbyintf(yf[lane])), maximum_y);
    }
#endif
    coordinates->x = vaddq_f32(coordinates->x, coordinates->step_x);
    coordinates->y = vaddq_f32(coordinates->y, coordinates->step_y);
}

template <size_t kChannels>
void SampleNearest(const uint8_t* source, uint8_t* destination,
                   const float* line, size_t count, size_t maximum_x,
                   size_t maximum_y, size_t row_stride) {
    CoordinateBlock coordinates = MakeCoordinateBlock(line);
    while (count >= 4) {
        int x[4];
        int y[4];
        ResolveNearest4(&coordinates, maximum_x, maximum_y, x, y);
        for (int lane = 0; lane < 4; ++lane) {
            const uint8_t* pixel = source + static_cast<size_t>(y[lane]) * row_stride +
                                   kChannels * static_cast<size_t>(x[lane]);
            std::memcpy(destination + kChannels * lane, pixel, kChannels);
        }
        destination += 4 * kChannels;
        count -= 4;
    }

    float x = vgetq_lane_f32(coordinates.x, 0);
    float y = vgetq_lane_f32(coordinates.y, 0);
    while (count-- != 0) {
        const int column =
          ClampIndex(static_cast<int>(nearbyintf(x)), maximum_x);
        const int row = ClampIndex(static_cast<int>(nearbyintf(y)), maximum_y);
        const uint8_t* pixel = source + static_cast<size_t>(row) * row_stride +
                               kChannels * static_cast<size_t>(column);
        std::memcpy(destination, pixel, kChannels);
        destination += kChannels;
        x += line[2];
        y += line[3];
    }
}

float LimitCoordinate(float value, size_t maximum) {
    return std::max(0.0f, std::min(value, static_cast<float>(maximum)));
}

struct BilinearPosition {
    int x0;
    int x1;
    int y0;
    int y1;
    float fraction_x;
    float fraction_y;
};

BilinearPosition ResolveBilinear(float x, float y, size_t maximum_x,
                                 size_t maximum_y) {
    x = LimitCoordinate(x, maximum_x);
    y = LimitCoordinate(y, maximum_y);
    const int x0 = static_cast<int>(floorf(x));
    const int y0 = static_cast<int>(floorf(y));
    return {x0, static_cast<int>(ceilf(x)), y0, static_cast<int>(ceilf(y)),
            x - static_cast<float>(x0), y - static_cast<float>(y0)};
}

float32x4_t LoadPixel4(const uint8_t* source) {
    uint32_t packed;
    std::memcpy(&packed, source, sizeof(packed));
    const uint8x8_t bytes = vreinterpret_u8_u32(vdup_n_u32(packed));
    const uint16x8_t words = vmovl_u8(bytes);
    return vcvtq_f32_u32(vmovl_u16(vget_low_u16(words)));
}

uint32_t StorePixel4(float32x4_t value) {
    const int32x4_t integers = vcvtq_s32_f32(value);
    const uint16x4_t words = vqmovun_s32(integers);
    const uint8x8_t bytes = vqmovn_u16(vcombine_u16(words, words));
    return vget_lane_u32(vreinterpret_u32_u8(bytes), 0);
}

void SampleBilinearQuad(const uint8_t* source, uint8_t* destination,
                        const float* line, size_t count, size_t maximum_x,
                        size_t maximum_y, size_t row_stride) {
    float x = line[0];
    float y = line[1];
    while (count-- != 0) {
        const BilinearPosition position =
          ResolveBilinear(x, y, maximum_x, maximum_y);
        const size_t top = static_cast<size_t>(position.y0) * row_stride;
        const size_t bottom = static_cast<size_t>(position.y1) * row_stride;
        const float32x4_t top_left =
          LoadPixel4(source + top + 4 * position.x0);
        const float32x4_t top_right =
          LoadPixel4(source + top + 4 * position.x1);
        const float32x4_t bottom_left =
          LoadPixel4(source + bottom + 4 * position.x0);
        const float32x4_t bottom_right =
          LoadPixel4(source + bottom + 4 * position.x1);
        const float32x4_t upper = vmlaq_n_f32(
          top_left, vsubq_f32(top_right, top_left), position.fraction_x);
        const float32x4_t lower = vmlaq_n_f32(
          bottom_left, vsubq_f32(bottom_right, bottom_left),
          position.fraction_x);
        const float32x4_t value = vmlaq_n_f32(
          upper, vsubq_f32(lower, upper), position.fraction_y);
        const uint32_t packed = StorePixel4(value);
        std::memcpy(destination, &packed, sizeof(packed));
        destination += 4;
        x += line[2];
        y += line[3];
    }
}

uint8x8_t Interpolate8(uint8x8_t top_left, uint8x8_t top_right,
                       uint8x8_t bottom_left, uint8x8_t bottom_right,
                       const float* fraction_x, const float* fraction_y) {
    const uint16x8_t top_left_words = vmovl_u8(top_left);
    const uint16x8_t top_right_words = vmovl_u8(top_right);
    const uint16x8_t bottom_left_words = vmovl_u8(bottom_left);
    const uint16x8_t bottom_right_words = vmovl_u8(bottom_right);
    uint16x4_t packed_halves[2];
    for (int half = 0; half < 2; ++half) {
        const uint16x4_t tl = half == 0 ? vget_low_u16(top_left_words)
                                        : vget_high_u16(top_left_words);
        const uint16x4_t tr = half == 0 ? vget_low_u16(top_right_words)
                                        : vget_high_u16(top_right_words);
        const uint16x4_t bl = half == 0 ? vget_low_u16(bottom_left_words)
                                        : vget_high_u16(bottom_left_words);
        const uint16x4_t br = half == 0 ? vget_low_u16(bottom_right_words)
                                        : vget_high_u16(bottom_right_words);
        const float32x4_t a = vcvtq_f32_u32(vmovl_u16(tl));
        const float32x4_t b = vcvtq_f32_u32(vmovl_u16(tr));
        const float32x4_t c = vcvtq_f32_u32(vmovl_u16(bl));
        const float32x4_t d = vcvtq_f32_u32(vmovl_u16(br));
        const float32x4_t fx = vld1q_f32(fraction_x + 4 * half);
        const float32x4_t fy = vld1q_f32(fraction_y + 4 * half);
        const float32x4_t upper = vmlaq_f32(a, vsubq_f32(b, a), fx);
        const float32x4_t lower = vmlaq_f32(c, vsubq_f32(d, c), fx);
        const float32x4_t value =
          vmlaq_f32(upper, vsubq_f32(lower, upper), fy);
        packed_halves[half] = vqmovun_s32(vcvtq_s32_f32(value));
    }
    return vqmovn_u16(vcombine_u16(packed_halves[0], packed_halves[1]));
}

void SampleBilinearMono(const uint8_t* source, uint8_t* destination,
                        const float* line, size_t count, size_t maximum_x,
                        size_t maximum_y, size_t row_stride) {
    CoordinateBlock coordinates = MakeCoordinateBlock(line);
    while (count >= 8) {
        const float32x4_t second_x =
          vaddq_f32(coordinates.x, coordinates.step_x);
        const float32x4_t second_y =
          vaddq_f32(coordinates.y, coordinates.step_y);
        float x[8];
        float y[8];
        vst1q_f32(x, coordinates.x);
        vst1q_f32(x + 4, second_x);
        vst1q_f32(y, coordinates.y);
        vst1q_f32(y + 4, second_y);

        uint8_t top_left[8];
        uint8_t top_right[8];
        uint8_t bottom_left[8];
        uint8_t bottom_right[8];
        float fraction_x[8];
        float fraction_y[8];
        for (int lane = 0; lane < 8; ++lane) {
            const BilinearPosition position =
              ResolveBilinear(x[lane], y[lane], maximum_x, maximum_y);
            const size_t top = static_cast<size_t>(position.y0) * row_stride;
            const size_t bottom = static_cast<size_t>(position.y1) * row_stride;
            top_left[lane] = source[top + position.x0];
            top_right[lane] = source[top + position.x1];
            bottom_left[lane] = source[bottom + position.x0];
            bottom_right[lane] = source[bottom + position.x1];
            fraction_x[lane] = position.fraction_x;
            fraction_y[lane] = position.fraction_y;
        }
        vst1_u8(destination,
                Interpolate8(vld1_u8(top_left), vld1_u8(top_right),
                             vld1_u8(bottom_left), vld1_u8(bottom_right),
                             fraction_x, fraction_y));
        destination += 8;
        count -= 8;
        coordinates.x = vaddq_f32(second_x, coordinates.step_x);
        coordinates.y = vaddq_f32(second_y, coordinates.step_y);
    }

    float x = vgetq_lane_f32(coordinates.x, 0);
    float y = vgetq_lane_f32(coordinates.y, 0);
    while (count-- != 0) {
        const BilinearPosition position =
          ResolveBilinear(x, y, maximum_x, maximum_y);
        const size_t top = static_cast<size_t>(position.y0) * row_stride;
        const size_t bottom = static_cast<size_t>(position.y1) * row_stride;
        const float upper = source[top + position.x0] +
          (source[top + position.x1] - source[top + position.x0]) *
            position.fraction_x;
        const float lower = source[bottom + position.x0] +
          (source[bottom + position.x1] - source[bottom + position.x0]) *
            position.fraction_x;
        const float value =
          upper + (lower - upper) * position.fraction_y;
        *destination++ = static_cast<uint8_t>(std::max(
          0, std::min(255, static_cast<int>(value))));
        x += line[2];
        y += line[3];
    }
}

struct YuvChannels {
    uint8x8_t red;
    uint8x8_t green;
    uint8x8_t blue;
};

uint8x8_t ClampColor(int16x8_t value) {
    return vqshrun_n_s16(vmaxq_s16(value, vdupq_n_s16(0)), 6);
}

YuvChannels DecodeYuv8(uint8x8_t luma, int16x8_t u, int16x8_t v) {
    const int16x8_t y = vreinterpretq_s16_u16(vshll_n_u8(luma, 6));
    const int16x8_t red_bias = vmulq_n_s16(v, 73);
    const int16x8_t green_bias =
      vmlaq_n_s16(vmulq_n_s16(u, 25), v, 37);
    const int16x8_t blue_bias = vmulq_n_s16(u, 130);
    return {ClampColor(vaddq_s16(y, red_bias)),
            ClampColor(vsubq_s16(y, green_bias)),
            ClampColor(vaddq_s16(y, blue_bias))};
}

template <bool kBlueFirst, bool kAlpha>
void ConvertYuvBlocks(const uint8_t* luma, uint8_t* destination, size_t blocks,
                      const uint8_t* vu) {
    const uint8x8_t midpoint = vdup_n_u8(128);
    const uint8x8_t opaque = vdup_n_u8(255);
    while (blocks-- != 0) {
        const uint8x8x2_t chroma = vld2_u8(vu);
        const int16x8_t v =
          vreinterpretq_s16_u16(vsubl_u8(chroma.val[0], midpoint));
        const int16x8_t u =
          vreinterpretq_s16_u16(vsubl_u8(chroma.val[1], midpoint));
        const uint8x8x2_t y = vld2_u8(luma);
        const YuvChannels even = DecodeYuv8(y.val[0], u, v);
        const YuvChannels odd = DecodeYuv8(y.val[1], u, v);
        const uint8x8x2_t red = vzip_u8(even.red, odd.red);
        const uint8x8x2_t green = vzip_u8(even.green, odd.green);
        const uint8x8x2_t blue = vzip_u8(even.blue, odd.blue);
        for (int half = 0; half < 2; ++half) {
            const uint8x8_t first = kBlueFirst ? blue.val[half] : red.val[half];
            const uint8x8_t third = kBlueFirst ? red.val[half] : blue.val[half];
            if (kAlpha) {
                const uint8x8x4_t output =
                  {{first, green.val[half], third, opaque}};
                vst4_u8(destination, output);
                destination += 32;
            } else {
                const uint8x8x3_t output = {{first, green.val[half], third}};
                vst3_u8(destination, output);
                destination += 24;
            }
        }
        luma += 16;
        vu += 16;
    }
}

float32x4_t Normalize4(uint16x4_t values, float mean, float scale) {
    const float32x4_t converted = vcvtq_f32_u32(vmovl_u16(values));
    return vmulq_n_f32(vsubq_f32(converted, vdupq_n_f32(mean)), scale);
}

void StoreMonoQuad(float32x4_t values, float* destination) {
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4x4_t output = {{values, zero, zero, zero}};
    vst4q_f32(destination, output);
}

}  // namespace

extern "C" void inspirecv_task_sample_c4_bilinear_arm(
  const uint8_t* source, uint8_t* destination, float* line, size_t count,
  size_t maximum_x, size_t maximum_y, size_t row_stride) {
    SampleBilinearQuad(source, destination, line, count, maximum_x, maximum_y,
                       row_stride);
}

extern "C" void inspirecv_task_sample_c1_bilinear_arm(
  const uint8_t* source, uint8_t* destination, float* line, size_t count,
  size_t maximum_x, size_t maximum_y, size_t row_stride) {
    SampleBilinearMono(source, destination, line, count, maximum_x, maximum_y,
                       row_stride);
}

extern "C" void inspirecv_task_sample_c4_nearest_arm(
  const uint8_t* source, uint8_t* destination, float* line, size_t count,
  size_t maximum_x, size_t maximum_y, size_t row_stride) {
    SampleNearest<4>(source, destination, line, count, maximum_x, maximum_y,
                     row_stride);
}

extern "C" void inspirecv_task_sample_c1_nearest_arm(
  const uint8_t* source, uint8_t* destination, float* line, size_t count,
  size_t maximum_x, size_t maximum_y, size_t row_stride) {
    SampleNearest<1>(source, destination, line, count, maximum_x, maximum_y,
                     row_stride);
}

extern "C" void inspirecv_task_nv21_to_rgb_arm(
  const uint8_t* luma, uint8_t* destination, size_t blocks,
  const uint8_t* vu) {
    ConvertYuvBlocks<false, false>(luma, destination, blocks, vu);
}

extern "C" void inspirecv_task_nv21_to_bgr_arm(
  const uint8_t* luma, uint8_t* destination, size_t blocks,
  const uint8_t* vu) {
    ConvertYuvBlocks<true, false>(luma, destination, blocks, vu);
}

extern "C" void inspirecv_task_nv21_to_rgba_arm(
  const uint8_t* luma, uint8_t* destination, size_t blocks,
  const uint8_t* vu) {
    ConvertYuvBlocks<false, true>(luma, destination, blocks, vu);
}

extern "C" void inspirecv_task_nv21_to_bgra_arm(
  const uint8_t* luma, uint8_t* destination, size_t blocks,
  const uint8_t* vu) {
    ConvertYuvBlocks<true, true>(luma, destination, blocks, vu);
}

extern "C" void inspirecv_task_c3_to_float_c4_arm(
  const uint8_t* source, float* destination, const float* mean,
  const float* scale, size_t count) {
    while (count >= 8) {
        const uint8x8x3_t pixels = vld3_u8(source);
        const uint16x8_t channel[3] = {
          vmovl_u8(pixels.val[0]), vmovl_u8(pixels.val[1]),
          vmovl_u8(pixels.val[2])};
        for (int half = 0; half < 2; ++half) {
            const float32x4_t red = Normalize4(
              half == 0 ? vget_low_u16(channel[0])
                        : vget_high_u16(channel[0]),
              mean[0], scale[0]);
            const float32x4_t green = Normalize4(
              half == 0 ? vget_low_u16(channel[1])
                        : vget_high_u16(channel[1]),
              mean[1], scale[1]);
            const float32x4_t blue = Normalize4(
              half == 0 ? vget_low_u16(channel[2])
                        : vget_high_u16(channel[2]),
              mean[2], scale[2]);
            const float32x4x4_t output =
              {{red, green, blue, vdupq_n_f32(0.0f)}};
            vst4q_f32(destination, output);
            destination += 16;
        }
        source += 24;
        count -= 8;
    }
    while (count-- != 0) {
        for (int channel = 0; channel < 4; ++channel) {
            const float value = channel < 3 ? source[channel] : 0.0f;
            destination[channel] = (value - mean[channel]) * scale[channel];
        }
        source += 3;
        destination += 4;
    }
}

extern "C" void inspirecv_task_c1_to_float_c4_arm(
  const uint8_t* source, float* destination, const float* mean,
  const float* scale, size_t count) {
    while (count >= 8) {
        const uint16x8_t values = vmovl_u8(vld1_u8(source));
        StoreMonoQuad(Normalize4(vget_low_u16(values), mean[0], scale[0]),
                      destination);
        StoreMonoQuad(Normalize4(vget_high_u16(values), mean[0], scale[0]),
                      destination + 16);
        source += 8;
        destination += 32;
        count -= 8;
    }
    const float unused = (0.0f - mean[0]) * scale[0];
    while (count-- != 0) {
        destination[0] = (static_cast<float>(*source++) - mean[0]) * scale[0];
        destination[1] = unused;
        destination[2] = unused;
        destination[3] = unused;
        destination += 4;
    }
}
