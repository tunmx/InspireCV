#include "../../common/common.h"

#include <inspirecv/task/task.h>
#include "inspirecv/core/runtime/cpu_features.h"
#include "inspirecv/task/kernels/cpu/sampling_ops.h"
#include "inspirecv/task/kernels/cpu/channel_ops.h"
#include "inspirecv/task/planning/compiled_conversion.h"
#include "inspirecv/task/kernels/cpu/yuv_ops.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {

using inspirecv::cpu::ScopedCpuFeatureMask;
using namespace inspirecv::task;

std::vector<uint32_t> FeatureMasks() {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    return {0, inspirecv::cpu::kSsse3 | inspirecv::cpu::kSse41,
            inspirecv::cpu::kAllCpuFeatures};
#else
    // The x86 mask does not disable the ARM architecture's NEON baseline.
    return {inspirecv::cpu::kAllCpuFeatures};
#endif
}

uint8_t SaturatedYuv(int numerator) {
    // Independent arithmetic definition, avoiding signed right-shift behavior.
    const int value = static_cast<int>(std::floor(numerator / 64.0));
    return static_cast<uint8_t>(std::max(0, std::min(255, value)));
}

std::array<uint8_t, 3> DecodeYuv(uint8_t y, uint8_t u, uint8_t v) {
    const int luma = 64 * static_cast<int>(y);
    const int du = static_cast<int>(u) - 128;
    const int dv = static_cast<int>(v) - 128;
    return {{SaturatedYuv(luma + 73 * dv),
             SaturatedYuv(luma - 25 * du - 37 * dv),
             SaturatedYuv(luma + 130 * du)}};
}

uint32_t FloatBits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

void CheckBytes(const uint8_t* actual, const uint8_t* expected, size_t count) {
    size_t mismatch = 0;
    while (mismatch < count && actual[mismatch] == expected[mismatch]) ++mismatch;
    CAPTURE(count, mismatch);
    if (mismatch != count) INFO("actual=" << int(actual[mismatch]) << " expected=" << int(expected[mismatch]));
    REQUIRE(mismatch == count);
}

}  // namespace

TEST_CASE("task_yuv_simd_extreme_values_preserve_saturated_scalar_contract",
          "[task][simd-dispatch][yuv][regression]") {
    struct Output { kernels::yuv::Converter* convert; int channels; bool blue_first; };
    const Output outputs[] = {{kernels::yuv::ToRgb, 3, false},
                               {kernels::yuv::ToBgr, 3, true},
                               {kernels::yuv::ToRgba, 4, false},
                               {kernels::yuv::ToBgra, 4, true}};
    const size_t counts[] = {1, 2, 15, 16, 17, 31, 32, 33, 48, 49, 255, 256, 257};
    const uint8_t extrema[] = {0, 1, 16, 127, 128, 235, 254, 255};
    for (uint32_t mask : FeatureMasks()) {
        ScopedCpuFeatureMask features(mask);
        for (const Output& output : outputs) {
            for (size_t count : counts) {
                for (size_t offset : {size_t(0), size_t(1), size_t(7), size_t(15)}) {
                    const size_t source_bytes = count + 2 * ((count + 1) / 2);
                    std::vector<uint8_t> source(offset + source_bytes + 16, 0xA7);
                    std::vector<uint8_t> destination(offset + count * output.channels + 16, 0xCD);
                    std::vector<uint8_t> expected(count * output.channels);
                    for (uint8_t y : extrema) for (uint8_t u : extrema) for (uint8_t v : extrema) {
                        CAPTURE(mask, count, offset, output.channels, output.blue_first, int(y), int(u), int(v));
                        std::fill(source.begin() + offset, source.begin() + offset + count, y);
                        for (size_t uv = count; uv < source_bytes; uv += 2) {
                            source[offset + uv] = v;
                            source[offset + uv + 1] = u;
                        }
                        const auto rgb = DecodeYuv(y, u, v);
                        for (size_t pixel = 0; pixel < count; ++pixel) {
                            expected[pixel * output.channels] = rgb[output.blue_first ? 2 : 0];
                            expected[pixel * output.channels + 1] = rgb[1];
                            expected[pixel * output.channels + 2] = rgb[output.blue_first ? 0 : 2];
                            if (output.channels == 4) expected[pixel * 4 + 3] = 255;
                        }
                        output.convert(source.data() + offset, destination.data() + offset, count);
                        CheckBytes(destination.data() + offset, expected.data(), expected.size());
                    }
                    REQUIRE(std::all_of(destination.begin(), destination.begin() + offset,
                                        [](uint8_t v) { return v == 0xCD; }));
                    REQUIRE(std::all_of(destination.begin() + offset + expected.size(), destination.end(),
                                        [](uint8_t v) { return v == 0xCD; }));
                    REQUIRE(std::all_of(source.begin(), source.begin() + offset,
                                        [](uint8_t v) { return v == 0xA7; }));
                    REQUIRE(std::all_of(source.begin() + offset + source_bytes, source.end(),
                                        [](uint8_t v) { return v == 0xA7; }));
                }
            }
        }
    }
}

TEST_CASE("task_public_yuv_planes_honor_padded_stride_and_output_guards",
          "[task][simd-dispatch][yuv][stride]") {
    constexpr int height = 4;
    for (uint32_t mask : FeatureMasks()) {
        ScopedCpuFeatureMask features(mask);
        for (PixelFormat format : {PixelFormat::kNv12, PixelFormat::kNv21, PixelFormat::kI420}) {
            for (int width : {2, 16, 18, 32, 34, 64, 258}) {
                const int stride = width + 6;
                const size_t y_bytes = static_cast<size_t>(stride) * height;
                const size_t chroma_plane = static_cast<size_t>(stride / 2) * (height / 2);
                std::vector<uint8_t> input(1 + y_bytes + 2 * chroma_plane + 16, 0xA7);
                uint8_t* src = input.data() + 1;
                for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
                    src[y * stride + x] = static_cast<uint8_t>((x % 3 == 0) ? 255 : (x * 37 + y * 19) & 255);
                }
                for (int y = 0; y < height / 2; ++y) for (int x = 0; x < width / 2; ++x) {
                    const uint8_t u = static_cast<uint8_t>((x % 2 == 0) ? 255 : (37 * x + 83 * y) & 255);
                    const uint8_t v = static_cast<uint8_t>((x % 3 == 0) ? 0 : (97 * x + 11 * y) & 255);
                    if (format == PixelFormat::kI420) {
                        src[y_bytes + y * (stride / 2) + x] = u;
                        src[y_bytes + chroma_plane + y * (stride / 2) + x] = v;
                    } else {
                        src[y_bytes + y * stride + 2 * x] = format == PixelFormat::kNv12 ? u : v;
                        src[y_bytes + y * stride + 2 * x + 1] = format == PixelFormat::kNv12 ? v : u;
                    }
                }
                const auto original = input;
                const int output_stride = 3 * width + 5;
                std::vector<uint8_t> output(7 + output_stride * height + 16, 0xCD);
                auto expected = output;
                for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
                    uint8_t u, v;
                    if (format == PixelFormat::kI420) {
                        u = src[y_bytes + (y / 2) * (stride / 2) + x / 2];
                        v = src[y_bytes + chroma_plane + (y / 2) * (stride / 2) + x / 2];
                    } else {
                        const size_t uv = y_bytes + (y / 2) * stride + (x / 2) * 2;
                        u = src[uv + (format == PixelFormat::kNv12 ? 0 : 1)];
                        v = src[uv + (format == PixelFormat::kNv12 ? 1 : 0)];
                    }
                    const auto rgb = DecodeYuv(src[y * stride + x], u, v);
                    for (int c = 0; c < 3; ++c) expected[7 + y * output_stride + 3 * x + c] = rgb[2 - c];
                }
                PipelineOptions options;
                options.input_format = format;
                options.output_format = PixelFormat::kBgr;
                options.backend_preference = BackendPreference::kCpu;
                Pipeline pipeline(options);
                RawImageView source{src, static_cast<size_t>(stride), width, height};
                TensorBuffer destination;
                destination.data = output.data() + 7;
                destination.row_stride_bytes = output_stride;
                destination.width = width;
                destination.height = height;
                destination.channels = 3;
                destination.element_type = ElementType::kUInt8;
                CAPTURE(mask, format, width);
                REQUIRE(pipeline.Run(source, destination) == Status::kOk);
                CheckBytes(output.data(), expected.data(), output.size());
                REQUIRE(input == original);
            }
        }
    }
}

TEST_CASE("task_public_float_normalization_preserves_layout_stride_and_all_tails",
          "[task][simd-dispatch][float][stride]") {
    constexpr int height = 3;
    constexpr float sentinel = -9876.5f;
    struct Format { PixelFormat input; PixelFormat output; int source_channels; int channels; bool reverse; };
    const Format formats[] = {{PixelFormat::kGray, PixelFormat::kGray, 1, 1, false},
                              {PixelFormat::kBgr, PixelFormat::kRgb, 3, 3, true},
                              {PixelFormat::kRgba, PixelFormat::kBgr, 4, 3, true},
                              {PixelFormat::kBgra, PixelFormat::kBgra, 4, 4, false}};
    for (uint32_t mask : FeatureMasks()) {
        ScopedCpuFeatureMask features(mask);
        for (const auto& format : formats) for (int width : {1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 255, 256, 257}) {
            for (TensorOrder order : {TensorOrder::kHwc, TensorOrder::kChw, TensorOrder::kChannelPacked4}) {
                if (format.channels == 4 && order != TensorOrder::kHwc) continue;
                const size_t source_stride = width * format.source_channels + 7;
                std::vector<uint8_t> input(3 + source_stride * height + 16, 0xA7);
                for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) for (int c = 0; c < format.source_channels; ++c) {
                    input[3 + y * source_stride + x * format.source_channels + c] = static_cast<uint8_t>((37 * x + 19 * y + 53 * c) & 255);
                }
                const auto original = input;
                const int packed_channels = order == TensorOrder::kChannelPacked4 ? 4 : format.channels;
                const size_t row = (order == TensorOrder::kChw ? width : width * packed_channels) + 7;
                const size_t plane = row * height + 5;
                std::vector<float> output(3 + plane * (order == TensorOrder::kChw ? format.channels : 1) + 8, sentinel);
                auto expected = output;
                PipelineOptions options;
                options.input_format = format.input;
                options.output_format = format.output;
                options.mean = {{-7.25f, 127.5f, 0.125f, 255.75f}};
                options.scale = {{-0.25f, 1.f / 128.f, 0.5f, 2.f}};
                options.backend_preference = BackendPreference::kCpu;
                for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
                    for (int c = 0; c < packed_channels; ++c) {
                        if (order == TensorOrder::kChw && c >= format.channels) continue;
                        const size_t index = 3 + (order == TensorOrder::kChw ? c * plane + y * row + x : y * row + x * packed_channels + c);
                        if (c >= format.channels) {
                            expected[index] = 0.f;
#if defined(INSPIRECV_TASK_USE_NEON)
                            // Existing P0 ARM64 contract: the assembly vector
                            // block clears padding, but its final scalar pixels
                            // normalize an implicit zero. Do not change another
                            // platform's frozen behavior for x86 optimization.
                            if (x >= (width / 8) * 8) {
                                const int lane = format.channels == 1 ? 0 : 3;
                                expected[index] = (0.f - options.mean[lane]) * options.scale[lane];
                            }
#endif
                            continue;
                        }
                        const int source_channel = format.reverse ? 2 - c : c;
                        volatile float delta = static_cast<float>(input[3 + y * source_stride + x * format.source_channels + source_channel]) - options.mean[c];
                        expected[index] = delta * options.scale[c];
                    }
                }
                Pipeline pipeline(options);
                RawImageView source{input.data() + 3, source_stride, width, height};
                TensorBuffer destination;
                destination.data = output.data() + 3;
                destination.row_stride_bytes = row * sizeof(float);
                destination.channel_stride_bytes = order == TensorOrder::kChw ? plane * sizeof(float) : 0;
                destination.width = width;
                destination.height = height;
                destination.channels = format.channels;
                destination.element_type = ElementType::kFloat32;
                destination.order = order;
                CAPTURE(mask, width, format.input, format.output, order);
                REQUIRE(pipeline.Run(source, destination) == Status::kOk);
                size_t mismatch = 0;
                while (mismatch < output.size() && FloatBits(output[mismatch]) == FloatBits(expected[mismatch])) ++mismatch;
                CAPTURE(mismatch);
                if (mismatch < output.size()) INFO("actual=" << output[mismatch] << " expected=" << expected[mismatch]);
                REQUIRE(mismatch == output.size());
                REQUIRE(input == original);
            }
        }
    }
}

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
TEST_CASE("task_all_channels_sampling_matches_independent_coordinate_reference",
          "[task][simd-dispatch][sampling][regression]") {
    // ARM's historical NEON sampling contract is separately frozen by its
    // baseline tests. This oracle specifies the x86 scalar arithmetic contract.
    kernels::sampling::Sampler* const nearest[] = {
        kernels::sampling::NearestMono, kernels::sampling::NearestTriple,
        kernels::sampling::NearestQuad};
    kernels::sampling::Sampler* const bilinear[] = {
        kernels::sampling::BilinearMono, kernels::sampling::BilinearTriple,
        kernels::sampling::BilinearQuad};
    for (uint32_t mask : FeatureMasks()) {
        ScopedCpuFeatureMask features(mask);
        for (int format = 0; format < 3; ++format) {
            const int channels = format == 0 ? 1 : format + 2;
            for (int width : {1, 3, 4, 5, 17, 67}) {
                const int height = 9, stride = width * channels + 7;
                std::vector<uint8_t> source(1 + height * stride + 32, 0xA7);
                uint32_t random = 0x719453u;
                for (int y = 0; y < height; ++y)
                    for (int x = 0; x < width * channels; ++x) {
                        random = random * 1664525u + 1013904223u;
                        source[1 + y * stride + x] = uint8_t(random >> 24);
                    }
                const auto original = source;
                for (size_t count : {size_t(1), size_t(7), size_t(8), size_t(9), size_t(15), size_t(16),
                                     size_t(17), size_t(31), size_t(32), size_t(33), size_t(255), size_t(256), size_t(257)}) {
                    for (float dx : {0.f, 0.1f, 0.4999f, 1.0001f, -0.2f}) {
                        for (bool linear : {false, true}) {
                            constexpr size_t first = 3, offset = 7;
                            std::vector<uint8_t> output(offset + (first + count) * channels + 32, 0xCD);
                            auto expected = output;
                            Point line[2] = {{dx < 0 ? width - 0.4999f : -0.4999f, 4.3271f}, {dx, -0.0314159f}};
                            float cursor_x = line[0].fX, cursor_y = line[0].fY;
                            for (size_t i = 0; i < count; ++i) {
                                const float x = std::max(0.f, std::min(float(width-1), cursor_x));
                                const float y = std::max(0.f, std::min(float(height-1), cursor_y));
                                for (int c = 0; c < channels; ++c) {
                                    const auto at = [&](int ix, int iy) {
                                        return source[1 + iy*stride + ix*channels + c];
                                    };
                                    uint8_t value;
                                    if (!linear) value = at(int(std::round(x)), int(std::round(y)));
                                    else {
                                        const int x0 = int(x), y0 = int(y);
                                        const int x1 = int(std::ceil(x)), y1 = int(std::ceil(y));
                                        const float fx = x-x0, fy = y-y0;
                                        // Preserve the defined evaluation order: the third
                                        // term promotes the first two terms to double.
                                        volatile float a = (1.f-fx)*(1.f-fy)*at(x0,y0);
                                        volatile float b = fx*(1.f-fy)*at(x1,y0);
                                        volatile float ab = a+b;
                                        volatile float d = fx*fy*at(x1,y1);
                                        const double third = fy*(1.0-fx)*at(x0,y1);
                                        const float interpolated = float(double(ab)+third+double(d));
                                        value = uint8_t(std::round(std::max(0.f, std::min(255.f, interpolated))));
                                    }
                                    expected[offset+(first+i)*channels+c] = value;
                                }
                                cursor_x += dx;
                                cursor_y += line[1].fY;
                            }
                            CAPTURE(mask, channels, width, count, dx, linear);
                            (linear ? bilinear[format] : nearest[format])(
                                source.data()+1, output.data()+offset, line, first, count,
                                first+count, width, height, stride);
                            CheckBytes(output.data(), expected.data(), output.size());
                            REQUIRE(source == original);
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("task_c4_nearest_preserves_scalar_incremental_coordinates",
          "[task][simd-dispatch][sampling][regression]") {
    constexpr size_t width = 521, height = 3, stride = width * 4 + 7, first = 3;
    std::vector<uint8_t> source(1 + height * stride + 16, 0xA7);
    for (size_t y = 0; y < height; ++y) for (size_t x = 0; x < width; ++x) for (size_t c = 0; c < 4; ++c) {
        source[1 + y * stride + 4 * x + c] = static_cast<uint8_t>((x * 31 + y * 17 + c * 61) & 255);
    }
    for (uint32_t mask : FeatureMasks()) {
        ScopedCpuFeatureMask features(mask);
        for (size_t count : {size_t(3), size_t(4), size_t(5), size_t(31), size_t(32), size_t(33), size_t(256), size_t(257), size_t(1025)}) {
            for (float step : {0.1f, 0.2f, 0.4999f, 1.0001f, -0.1f}) {
                std::vector<uint8_t> output(7 + (first + count) * 4 + 16, 0xCD);
                auto expected = output;
                Point line[2] = {{step < 0 ? 255.49f : -0.49f, 1.25f}, {step, -0.0007f}};
                float x = line[0].fX, y = line[0].fY;
                for (size_t i = 0; i < count; ++i) {
                    const int ix = static_cast<int>(std::round(std::max(0.f, std::min(x, float(width - 1)))));
                    const int iy = static_cast<int>(std::round(std::max(0.f, std::min(y, float(height - 1)))));
                    std::copy_n(source.data() + 1 + iy * stride + 4 * ix, 4, expected.data() + 7 + (first + i) * 4);
                    x += line[1].fX;
                    y += line[1].fY;
                }
                CAPTURE(mask, count, step);
                kernels::sampling::NearestQuad(source.data() + 1, output.data() + 7, line,
                                                first, count, first + count, width, height, stride);
                CheckBytes(output.data(), expected.data(), output.size());
            }
        }
    }
}

TEST_CASE("task_c4_horizontal_nearest_preserves_coordinates_and_storage",
          "[task][simd-dispatch][sampling][regression]") {
    struct Shape { int width; int height; };
    struct Scan { float x; float y; float dx; };
    const Shape shapes[] = {{1, 1}, {2, 5}, {9, 3}, {521, 5}};
    const float below_half = std::nextafter(0.5f, 0.f);
    const float above_half = std::nextafter(0.5f, 1.f);
    // Include sequential additions that differ from x + i*dx, exact ties,
    // clamped rows/columns, reversed scans and repeated coordinates.
    const Scan scans[] = {{-100.f, -100.f, 0.1f}, {-0.49f, below_half, 0.1f},
                          {-0.49f, 0.5f, 0.2f}, {-0.49f, above_half, 0.4999f},
                          {below_half, 1.5f, 0.f}, {0.5f, 2.5f, 0.f},
                          {above_half, 100.f, 0.f}, {255.49f, 1.25f, -0.1f},
                          {521.5f, 4.5f, -1.0001f}, {0.f, -0.f, 1.0001f},
                          {10000.f, 10000.f, -0.2f}, {2.5f, 0.f, -0.f}};
    for (uint32_t mask : FeatureMasks()) {
        const ScopedCpuFeatureMask features(mask);
        for (const Shape shape : shapes) for (size_t padding : {size_t(0), size_t(7)}) {
            const size_t stride = 4 * shape.width + padding;
            for (size_t offset : {size_t(0), size_t(1), size_t(7), size_t(15)}) {
                std::vector<uint8_t> source(offset + shape.height * stride + 19, 0xA7);
                uint32_t random = 0x63791u;
                for (int y = 0; y < shape.height; ++y) for (int x = 0; x < 4 * shape.width; ++x) {
                    random = 1664525u * random + 1013904223u;
                    source[offset + y * stride + x] = uint8_t(random >> 24);
                }
                const auto original = source;
                const size_t first = offset % 5;
                for (size_t count : {size_t(0), size_t(1), size_t(7), size_t(8), size_t(9),
                                     size_t(15), size_t(16), size_t(17), size_t(31), size_t(32),
                                     size_t(33), size_t(255), size_t(256), size_t(257), size_t(1025)}) {
                    for (const Scan scan : scans) for (float dy : {0.f, -0.f}) {
                        std::vector<uint8_t> output(offset + 4 * (first + count) + 19, 0xCD);
                        auto expected = output;
                        // Volatile recurrence is an independent scalar oracle;
                        // no lane/block coordinate construction is shared with SIMD.
                        volatile float x = scan.x, y = scan.y;
                        for (size_t i = 0; i < count; ++i) {
                            const float bounded_x = std::max(0.f, std::min(float(x), float(shape.width - 1)));
                            const float bounded_y = std::max(0.f, std::min(float(y), float(shape.height - 1)));
                            const size_t ix = size_t(std::round(bounded_x));
                            const size_t iy = size_t(std::round(bounded_y));
                            for (size_t c = 0; c < 4; ++c)
                                expected[offset + 4 * (first + i) + c] = source[offset + iy * stride + 4 * ix + c];
                            x = x + scan.dx;
                            y = y + dy;
                        }
                        Point line[2] = {{scan.x, scan.y}, {scan.dx, dy}};
                        CAPTURE(mask, shape.width, shape.height, padding, offset, first,
                                count, scan.x, scan.y, scan.dx, FloatBits(dy));
                        kernels::sampling::NearestQuad(source.data() + offset, output.data() + offset,
                            line, first, count, first + count, shape.width, shape.height, stride);
                        CheckBytes(output.data(), expected.data(), output.size());
                        REQUIRE(source == original);
                    }
                }
            }
        }
    }
}

TEST_CASE("task_c4_bilinear_rounds_positive_half_values_away_from_zero",
          "[task][simd-dispatch][sampling][regression]") {
    const uint8_t source[] = {0, 2, 100, 254, 1, 3, 101, 255};
    for (uint32_t mask : FeatureMasks()) {
        ScopedCpuFeatureMask features(mask);
        for (float coordinate : {std::nextafter(0.5f, 0.f), 0.5f, std::nextafter(0.5f, 1.f)}) {
            constexpr size_t count = 33, first = 3;
            std::vector<uint8_t> output(1 + 4 * (first + count) + 16, 0xCD);
            auto expected = output;
            for (size_t i = 0; i < count; ++i) for (size_t c = 0; c < 4; ++c) {
                const float value = (1.f - coordinate) * source[c] + coordinate * source[4 + c];
                expected[1 + 4 * (first + i) + c] = static_cast<uint8_t>(std::round(value));
            }
            Point line[2] = {{coordinate, 0.f}, {0.f, 0.f}};
            CAPTURE(mask, coordinate);
            kernels::sampling::BilinearQuad(source, output.data() + 1, line, first, count,
                                             first + count, 2, 1, 8);
            CheckBytes(output.data(), expected.data(), output.size());
        }
    }
}
#endif

TEST_CASE("task_i420_planes_match_independent_identity_and_semiplanar_coordinates",
          "[task][simd-dispatch][yuv][regression]") {
    // Exact integer steps avoid imposing the x86 C1 floating-coordinate
    // accumulation contract on ARM's historical NEON implementation.
    for (uint32_t mask : FeatureMasks()) {
        const ScopedCpuFeatureMask features(mask);
        for (int width : {6, 18, 34}) for (int padding : {0, 6}) {
            constexpr int height = 6;
            const int stride = width+padding, uv_stride=stride/2;
            const size_t y_size=stride*height, uv_size=uv_stride*(height/2);
            std::vector<uint8_t> i420(1+y_size+uv_size*2+16, 0xA7);
            std::vector<uint8_t> nv12(i420.size(), 0xA7), nv21(i420.size(), 0xA7);
            for (int y=0; y<height; ++y) for (int x=0; x<width; ++x) {
                const uint8_t value=uint8_t(17*x+23*y);
                i420[1+y*stride+x]=nv12[1+y*stride+x]=nv21[1+y*stride+x]=value;
            }
            for (int y=0; y<height/2; ++y) for (int x=0; x<width/2; ++x) {
                const uint8_t u=uint8_t(11+11*x+31*y), v=uint8_t(44+11*x+17*y);
                i420[1+y_size+y*uv_stride+x]=u;
                i420[1+y_size+uv_size+y*uv_stride+x]=v;
                nv12[1+y_size+y*stride+2*x]=u; nv12[1+y_size+y*stride+2*x+1]=v;
                nv21[1+y_size+y*stride+2*x]=v; nv21[1+y_size+y*stride+2*x+1]=u;
            }
            const auto original=i420;
            for (bool direct : {false,true}) for (int row : {0,2,4}) {
                for (int count : {1,5,width}) {
                    const size_t first=2, capacity=first+count+3;
                    std::vector<uint8_t> expected(7+capacity+2*((capacity+1)/2)+16,0xCD);
                    for (int x=0; x<count; ++x) expected[7+first+x]=uint8_t(17*x+23*row);
                    // TargetPlanes packs VU after capacity luma bytes, with
                    // the first output pixel's chroma pair offset rounded down.
                    for (int x=0; x<(count+1)/2; ++x) {
                        expected[7+capacity+2*(first/2)+2*x]=uint8_t(44+11*x+17*(row/2));
                        expected[7+capacity+2*(first/2)+2*x+1]=uint8_t(11+11*x+31*(row/2));
                    }
                    kernels::sampling::Sampler* samplers[] = {
                        direct ? kernels::sampling::DirectI420 : kernels::sampling::NearestI420,
                        direct ? kernels::sampling::DirectNv12 : kernels::sampling::NearestNv12,
                        direct ? kernels::sampling::DirectNv21 : kernels::sampling::NearestNv21};
                    const uint8_t* sources[]={i420.data()+1,nv12.data()+1,nv21.data()+1};
                    for (int format=0; format<3; ++format) {
                        Point line[2]={{0.f,float(row)},{1.f,0.f}};
                        std::vector<uint8_t> output(expected.size(),0xCD);
                        CAPTURE(mask,width,padding,direct,row,count,format);
                        samplers[format](sources[format],output.data()+7,line,first,count,capacity,width,height,stride);
                        CheckBytes(output.data(),expected.data(),output.size());
                    }
                }
            }
            REQUIRE(i420==original);
        }
    }
}

TEST_CASE("task_channel_contraction_and_reorder_support_exact_in_place_tail",
          "[task][simd-dispatch][channel][alias]") {
    struct Operation { kernels::channel::PixelTransform* run; int from; int to; bool reverse; };
    const Operation operations[]={
        {kernels::channel::ReverseTriple,3,3,true},
        {kernels::channel::ReverseQuadColor,4,4,true},
        {kernels::channel::DropAlpha,4,3,false},
        {kernels::channel::ReverseAndDropAlpha,4,3,true}};
    for (uint32_t mask : FeatureMasks()) {
        const ScopedCpuFeatureMask features(mask);
        for (const auto& op : operations) for (size_t count=0; count<=33; ++count) {
            for (size_t offset : {size_t(0),size_t(1),size_t(7),size_t(15)}) {
                std::vector<uint8_t> storage(offset+count*op.from+19,0xA7);
                for (size_t i=0; i<count*op.from; ++i) storage[offset+i]=uint8_t(17*i+i/7);
                auto expected=storage;
                for (size_t i=0; i<count; ++i) for (int c=0;c<op.to;++c) {
                    const int source_channel=op.reverse && c<3 ? 2-c : c;
                    expected[offset+i*op.to+c]=storage[offset+i*op.from+source_channel];
                }
                CAPTURE(mask,op.from,op.to,op.reverse,count,offset);
                op.run(storage.data()+offset,storage.data()+offset,count);
                CheckBytes(storage.data(),expected.data(),storage.size());
            }
        }
    }
}

TEST_CASE("task_public_rejects_unaligned_float_storage_without_writes",
          "[task][simd-dispatch][validation][stride]") {
    const uint8_t input[18]={};
    PipelineOptions options;
    options.input_format=PixelFormat::kRgb;
    options.output_format=PixelFormat::kBgr;
    options.backend_preference=BackendPreference::kCpu;
    Pipeline pipeline(options);
    const RawImageView source{input,9,3,2};
    for (TensorOrder order : {TensorOrder::kHwc,TensorOrder::kChw,TensorOrder::kChannelPacked4}) {
        for (int violation : {0,1,2}) {
            if (violation==2 && order!=TensorOrder::kChw) continue;
            std::vector<float> aligned(128,-9876.5f);
            const auto original=aligned;
            TensorBuffer output;
            output.data=reinterpret_cast<uint8_t*>(aligned.data())+(violation==0?1:0);
            output.row_stride_bytes=64+(violation==1?1:0);
            output.channel_stride_bytes=order==TensorOrder::kChw?128+(violation==2?1:0):0;
            output.width=3;output.height=2;output.channels=3;
            output.element_type=ElementType::kFloat32;output.order=order;
            CAPTURE(order,violation);
            REQUIRE(pipeline.Run(source,output)==Status::kInvalidArgument);
            REQUIRE(aligned==original);
        }
    }
    for (int channels : {1,2,4}) {
        std::vector<uint8_t> bytes(128,0xCD),original=bytes;
        TensorBuffer output;
        output.data=bytes.data();output.width=3;output.height=2;output.channels=channels;
        output.element_type=ElementType::kUInt8;
        CAPTURE(channels);
        REQUIRE(pipeline.Run(source,output)==Status::kInvalidArgument);
        REQUIRE(bytes==original);
    }
}

TEST_CASE("task_request_compiler_counts_last_tile_without_signed_overflow",
          "[task][simd-dispatch][validation][size]") {
    using namespace inspirecv::task::internal;
    ConversionRequest request;
    request.source_channels=request.destination_channels=1;
    request.source_width=request.destination_width=std::numeric_limits<int>::max();
    request.source_height=request.destination_height=1;
    PipelineConfig config;
    config.source_format=config.destination_format=GRAY;
    Matrix identity;
    CompiledConversion compiled;
    REQUIRE(CompileConversion(config,identity,request,false,&compiled)==inspirecv::SUCCESS);
    REQUIRE(compiled.tile_count==std::numeric_limits<int>::max()/256+1);
}
