#include "../../common/common.h"

#include <inspirecv/core/transform_matrix.h>
#include <inspirecv/task/legacy.h>
#include <inspirecv/task/task.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

using inspirecv::task::Filter;
using inspirecv::task::StreamFormat;
using inspirecv::task::StreamTask;
using inspirecv::task::TensorLayout;
using inspirecv::task::TensorView;
using inspirecv::task::Wrap;

namespace {

int channelsOf(StreamFormat format) {
    switch (format) {
        case StreamFormat::GRAY:
            return 1;
        case StreamFormat::RGB:
        case StreamFormat::BGR:
            return 3;
        case StreamFormat::RGBA:
        case StreamFormat::BGRA:
            return 4;
        default:
            return 0;
    }
}

inspirecv::TaskStatus convert(const StreamTask::Config& config,
                              const inspirecv::TransformMatrix& matrix,
                              const uint8_t* source, int iw, int ih, int sourceStride,
                              void* dest, int ow, int oh, int outputBpp, int outputStride,
                              halide_type_t type, uint8_t padding = 0) {
    auto* task = StreamTask::Create(config);
    task->SetMatrix(matrix);
    task->SetPadding(padding);
    const auto status = task->Convert(source, iw, ih, sourceStride,
                                      dest, ow, oh, outputBpp, outputStride, type);
    StreamTask::Destroy(task);
    return status;
}

inspirecv::TaskStatus convertU8(const uint8_t* source, int iw, int ih, int sourceStride,
                                StreamFormat sourceFormat,
                                uint8_t* dest, int ow, int oh, int outputStride,
                                StreamFormat destFormat,
                                const inspirecv::TransformMatrix& matrix = inspirecv::TransformMatrix::Identity(),
                                Filter filter = Filter::NEAREST,
                                Wrap wrap = Wrap::CLAMP_TO_EDGE,
                                uint8_t padding = 0) {
    StreamTask::Config config;
    config.sourceFormat = sourceFormat;
    config.destFormat = destFormat;
    config.filterType = filter;
    config.wrap = wrap;
    return convert(config, matrix, source, iw, ih, sourceStride,
                   dest, ow, oh, channelsOf(destFormat), outputStride,
                   halide_type_of<uint8_t>(), padding);
}

float readFloat(const uint8_t* data) {
    float value = 0.0f;
    std::memcpy(&value, data, sizeof(value));
    return value;
}

}  // namespace

TEST_CASE("task_core_four_channel_conversions", "[task][core][formats]") {
    constexpr int kPixels = 19;  // Exercise both the ARM64 vector loop and scalar tail.
    std::vector<uint8_t> rgba(kPixels * 4);
    std::vector<uint8_t> bgra(kPixels * 4);
    for (int i = 0; i < kPixels; ++i) {
        rgba[4 * i + 0] = static_cast<uint8_t>(3 + 7 * i);
        rgba[4 * i + 1] = static_cast<uint8_t>(5 + 5 * i);
        rgba[4 * i + 2] = static_cast<uint8_t>(9 + 3 * i);
        rgba[4 * i + 3] = static_cast<uint8_t>(200 + i);
        bgra[4 * i + 0] = rgba[4 * i + 2];
        bgra[4 * i + 1] = rgba[4 * i + 1];
        bgra[4 * i + 2] = rgba[4 * i + 0];
        bgra[4 * i + 3] = rgba[4 * i + 3];
    }

    SECTION("RGBA and BGRA swap red and blue while preserving alpha") {
        std::vector<uint8_t> actual(kPixels * 4, 0);
        REQUIRE(convertU8(rgba.data(), kPixels, 1, 0, StreamFormat::RGBA,
                          actual.data(), kPixels, 1, 0, StreamFormat::BGRA) == inspirecv::SUCCESS);
        REQUIRE_EQ_C_ARRAY(actual.data(), bgra.data(), actual.size());

        std::fill(actual.begin(), actual.end(), 0);
        REQUIRE(convertU8(bgra.data(), kPixels, 1, 0, StreamFormat::BGRA,
                          actual.data(), kPixels, 1, 0, StreamFormat::RGBA) == inspirecv::SUCCESS);
        REQUIRE_EQ_C_ARRAY(actual.data(), rgba.data(), actual.size());
    }

    SECTION("RGBA and BGRA drop alpha using destination channel order") {
        std::vector<uint8_t> actual(kPixels * 3, 0);
        std::vector<uint8_t> expected(kPixels * 3, 0);

        for (int i = 0; i < kPixels; ++i) {
            expected[3 * i + 0] = rgba[4 * i + 0];
            expected[3 * i + 1] = rgba[4 * i + 1];
            expected[3 * i + 2] = rgba[4 * i + 2];
        }
        REQUIRE(convertU8(rgba.data(), kPixels, 1, 0, StreamFormat::RGBA,
                          actual.data(), kPixels, 1, 0, StreamFormat::RGB) == inspirecv::SUCCESS);
        REQUIRE_EQ_C_ARRAY(actual.data(), expected.data(), actual.size());

        std::fill(actual.begin(), actual.end(), 0);
        REQUIRE(convertU8(bgra.data(), kPixels, 1, 0, StreamFormat::BGRA,
                          actual.data(), kPixels, 1, 0, StreamFormat::RGB) == inspirecv::SUCCESS);
        REQUIRE_EQ_C_ARRAY(actual.data(), expected.data(), actual.size());

        for (int i = 0; i < kPixels; ++i) {
            expected[3 * i + 0] = rgba[4 * i + 2];
            expected[3 * i + 1] = rgba[4 * i + 1];
            expected[3 * i + 2] = rgba[4 * i + 0];
        }
        std::fill(actual.begin(), actual.end(), 0);
        REQUIRE(convertU8(rgba.data(), kPixels, 1, 0, StreamFormat::RGBA,
                          actual.data(), kPixels, 1, 0, StreamFormat::BGR) == inspirecv::SUCCESS);
        REQUIRE_EQ_C_ARRAY(actual.data(), expected.data(), actual.size());

        std::fill(actual.begin(), actual.end(), 0);
        REQUIRE(convertU8(bgra.data(), kPixels, 1, 0, StreamFormat::BGRA,
                          actual.data(), kPixels, 1, 0, StreamFormat::BGR) == inspirecv::SUCCESS);
        REQUIRE_EQ_C_ARRAY(actual.data(), expected.data(), actual.size());
    }
}

TEST_CASE("task_core_gray_and_alpha_conversions", "[task][core][formats]") {
    constexpr int kPixels = 19;

    SECTION("GRAY to RGBA and BGRA replicates gray and sets opaque alpha") {
        std::vector<uint8_t> gray(kPixels);
        std::vector<uint8_t> expected(kPixels * 4);
        for (int i = 0; i < kPixels; ++i) {
            gray[i] = static_cast<uint8_t>(11 * i);
            expected[4 * i + 0] = gray[i];
            expected[4 * i + 1] = gray[i];
            expected[4 * i + 2] = gray[i];
            expected[4 * i + 3] = 255;
        }

        for (const auto format : {StreamFormat::RGBA, StreamFormat::BGRA}) {
            std::vector<uint8_t> actual(kPixels * 4, 0);
            REQUIRE(convertU8(gray.data(), kPixels, 1, 0, StreamFormat::GRAY,
                              actual.data(), kPixels, 1, 0, format) == inspirecv::SUCCESS);
            REQUIRE_EQ_C_ARRAY(actual.data(), expected.data(), actual.size());
        }
    }

    SECTION("RGB and BGR to GRAY match the documented fixed-point luma") {
        std::vector<uint8_t> rgb(kPixels * 3);
        std::vector<uint8_t> bgr(kPixels * 3);
        std::vector<uint8_t> expected(kPixels);
        for (int i = 0; i < kPixels; ++i) {
            const uint8_t r = static_cast<uint8_t>(2 + 9 * i);
            const uint8_t g = static_cast<uint8_t>(7 + 6 * i);
            const uint8_t b = static_cast<uint8_t>(13 + 4 * i);
            rgb[3 * i + 0] = r;
            rgb[3 * i + 1] = g;
            rgb[3 * i + 2] = b;
            bgr[3 * i + 0] = b;
            bgr[3 * i + 1] = g;
            bgr[3 * i + 2] = r;
            expected[i] = static_cast<uint8_t>((19 * r + 38 * g + 7 * b) >> 6);
        }

        std::vector<uint8_t> actual(kPixels, 0);
        REQUIRE(convertU8(rgb.data(), kPixels, 1, 0, StreamFormat::RGB,
                          actual.data(), kPixels, 1, 0, StreamFormat::GRAY) == inspirecv::SUCCESS);
        REQUIRE_EQ_C_ARRAY(actual.data(), expected.data(), actual.size());

        std::fill(actual.begin(), actual.end(), 0);
        REQUIRE(convertU8(bgr.data(), kPixels, 1, 0, StreamFormat::BGR,
                          actual.data(), kPixels, 1, 0, StreamFormat::GRAY) == inspirecv::SUCCESS);
        REQUIRE_EQ_C_ARRAY(actual.data(), expected.data(), actual.size());
    }
}

TEST_CASE("task_core_sampling_uses_output_to_input_matrix", "[task][core][sampling]") {
    SECTION("nearest translation and non-tight source stride") {
        constexpr int kWidth = 3;
        constexpr int kHeight = 2;
        constexpr int kStride = 5;
        const uint8_t source[kHeight * kStride] = {
            10, 20, 30, 0xEE, 0xEE,
            40, 50, 60, 0xEE, 0xEE,
        };
        std::vector<uint8_t> actual(kWidth * kHeight, 0);
        const inspirecv::TransformMatrix outputToInput(1, 0, 1, 0, 1, 0);
        REQUIRE(convertU8(source, kWidth, kHeight, kStride, StreamFormat::GRAY,
                          actual.data(), kWidth, kHeight, 0, StreamFormat::GRAY,
                          outputToInput, Filter::NEAREST) == inspirecv::SUCCESS);
        const uint8_t expected[] = {20, 30, 30, 50, 60, 60};
        REQUIRE_EQ_C_ARRAY(actual.data(), expected, actual.size());
    }

    SECTION("bilinear samples the mathematical center of four pixels") {
        const uint8_t source[] = {0, 10, 20, 30};
        uint8_t actual = 0;
        const inspirecv::TransformMatrix outputToInput(1, 0, 0.5f, 0, 1, 0.5f);
        REQUIRE(convertU8(source, 2, 2, 0, StreamFormat::GRAY,
                          &actual, 1, 1, 0, StreamFormat::GRAY,
                          outputToInput, Filter::BILINEAR) == inspirecv::SUCCESS);
        REQUIRE(actual == 15);
    }

    SECTION("three-channel bilinear sampling uses round-to-nearest") {
        // C3 deliberately exercises the common scalar sampler used by
        // InspireFace FrameProcess. Values around 0.5 catch accidental
        // truncation, including the exact tie handled by roundf.
        const uint8_t source[] = {
            0, 0, 0,
            1, 1, 1,
        };
        const std::array<float, 3> sampleX = {0.49f, 0.50f, 0.51f};
        const std::array<uint8_t, 3> expected = {0, 1, 1};

        for (size_t i = 0; i < sampleX.size(); ++i) {
            uint8_t actual[3] = {0xFF, 0xFF, 0xFF};
            const inspirecv::TransformMatrix outputToInput(1, 0, sampleX[i], 0, 1, 0);
            REQUIRE(convertU8(source, 2, 1, 0, StreamFormat::RGB,
                              actual, 1, 1, 0, StreamFormat::RGB,
                              outputToInput, Filter::BILINEAR) == inspirecv::SUCCESS);
            INFO("sample x = " << sampleX[i]);
            REQUIRE(actual[0] == expected[i]);
            REQUIRE(actual[1] == expected[i]);
            REQUIRE(actual[2] == expected[i]);
        }
    }

    SECTION("three-channel bilinear sampling matches the four-term reference") {
        const uint8_t source[] = {
            3,  67, 251,   211, 19,  83,
            97, 173, 41,   29,  239, 137,
        };
        const std::array<float, 5> coordinates = {0.0f, 0.125f, 0.49f, 0.5f, 0.875f};

        for (const float y : coordinates) {
            for (const float x : coordinates) {
                uint8_t actual[3] = {0, 0, 0};
                const inspirecv::TransformMatrix outputToInput(1, 0, x, 0, 1, y);
                REQUIRE(convertU8(source, 2, 2, 0, StreamFormat::RGB,
                                  actual, 1, 1, 0, StreamFormat::RGB,
                                  outputToInput, Filter::BILINEAR) == inspirecv::SUCCESS);

                for (int channel = 0; channel < 3; ++channel) {
                    const float c00 = source[channel];
                    const float c01 = source[3 + channel];
                    const float c10 = source[6 + channel];
                    const float c11 = source[9 + channel];
                    float value = (1.0f - x) * (1.0f - y) * c00 +
                                  x * (1.0f - y) * c01 +
                                  y * (1.0 - x) * c10 +
                                  x * y * c11;
                    value = std::fmin(std::fmax(value, 0.0f), 255.0f);
                    const auto expected = static_cast<uint8_t>(std::round(value));
                    INFO("sample (" << x << ", " << y << "), channel " << channel);
                    REQUIRE(actual[channel] == expected);
                }
            }
        }
    }

    SECTION("three-channel bilinear clamps repeated-add endpoint drift") {
        constexpr int kInputWidth = 98;
        constexpr int kInputHeight = 2;
        constexpr int kOutputWidth = 4;
        std::vector<uint8_t> source(
            static_cast<size_t>(kInputWidth) * kInputHeight * 3);
        for (int y = 0; y < kInputHeight; ++y) {
            for (int x = 0; x < kInputWidth; ++x) {
                for (int channel = 0; channel < 3; ++channel) {
                    source[(static_cast<size_t>(y) * kInputWidth + x) * 3 + channel] =
                        static_cast<uint8_t>((37 * x + 71 * y + 53 * channel) & 255);
                }
            }
        }

        // start + 3 * step rounds to exactly 97.0f, so the C3 interior path is
        // selected. Repeated float additions round the fourth coordinate to
        // 97.0000076f; clamp it back to the last source column before
        // calculating interpolation weights.
        constexpr float kStart = 0.001003009034320712f;
        constexpr float kStep = 32.33300018310547f;
        const inspirecv::TransformMatrix outputToInput(
            kStep, 0, kStart, 0, 0, 0.5f);
        std::array<uint8_t, kOutputWidth * 3> actual{};
        REQUIRE(convertU8(source.data(), kInputWidth, kInputHeight, 0,
                          StreamFormat::RGB, actual.data(), kOutputWidth, 1, 0,
                          StreamFormat::RGB, outputToInput, Filter::BILINEAR) ==
                inspirecv::SUCCESS);

        const size_t top = static_cast<size_t>(kInputWidth - 1) * 3;
        const size_t bottom =
            (static_cast<size_t>(kInputWidth) + kInputWidth - 1) * 3;
        for (int channel = 0; channel < 3; ++channel) {
            const float expectedValue =
                0.5f * source[top + channel] + 0.5f * source[bottom + channel];
            const auto expected = static_cast<uint8_t>(std::round(expectedValue));
            REQUIRE(actual[(kOutputWidth - 1) * 3 + channel] == expected);
        }
    }
}

TEST_CASE("task_core_adaptive_threading_preserves_color_conversion", "[task][core][threading]") {
    struct Size {
        int width;
        int height;
    };
    // Face-sized inputs stay on the caller thread; an image above the 512K
    // threshold still exercises the row-worker path.
    for (const auto size : {Size{112, 112}, Size{1024, 513}}) {
        const int pixels = size.width * size.height;
        std::vector<uint8_t> source(static_cast<size_t>(pixels) * 3);
        std::vector<uint8_t> actual(source.size(), 0);
        std::vector<uint8_t> expected(source.size(), 0);
        for (int i = 0; i < pixels; ++i) {
            source[3 * i + 0] = static_cast<uint8_t>((3 * i + 17) & 255);
            source[3 * i + 1] = static_cast<uint8_t>((5 * i + 29) & 255);
            source[3 * i + 2] = static_cast<uint8_t>((7 * i + 43) & 255);
            expected[3 * i + 0] = source[3 * i + 2];
            expected[3 * i + 1] = source[3 * i + 1];
            expected[3 * i + 2] = source[3 * i + 0];
        }

        REQUIRE(convertU8(source.data(), size.width, size.height, 0, StreamFormat::BGR,
                          actual.data(), size.width, size.height, 0, StreamFormat::RGB,
                          inspirecv::TransformMatrix::Identity(), Filter::BILINEAR) == inspirecv::SUCCESS);
        REQUIRE_EQ_C_ARRAY(actual.data(), expected.data(), actual.size());
    }
}

TEST_CASE("task_core_wrap_and_padding", "[task][core][sampling][wrap]") {
    const uint8_t source[] = {10, 20, 30};
    const inspirecv::TransformMatrix onePixelLeft(1, 0, -1, 0, 1, 0);

    SECTION("CLAMP_TO_EDGE copies the nearest border pixel") {
        uint8_t actual[3] = {0, 0, 0};
        REQUIRE(convertU8(source, 3, 1, 0, StreamFormat::GRAY,
                          actual, 3, 1, 0, StreamFormat::GRAY,
                          onePixelLeft, Filter::NEAREST, Wrap::CLAMP_TO_EDGE) == inspirecv::SUCCESS);
        const uint8_t expected[] = {10, 10, 20};
        REQUIRE_EQ_C_ARRAY(actual, expected, sizeof(expected));
    }

    SECTION("ZERO uses SetPadding for a partially out-of-bounds row") {
        uint8_t actual[3] = {0, 0, 0};
        REQUIRE(convertU8(source, 3, 1, 0, StreamFormat::GRAY,
                          actual, 3, 1, 0, StreamFormat::GRAY,
                          onePixelLeft, Filter::NEAREST, Wrap::ZERO, 77) == inspirecv::SUCCESS);
        const uint8_t expected[] = {77, 10, 20};
        REQUIRE_EQ_C_ARRAY(actual, expected, sizeof(expected));
    }

    SECTION("ZERO fills a fully out-of-bounds row") {
        uint8_t actual[3] = {0, 0, 0};
        const inspirecv::TransformMatrix outside(1, 0, -10, 0, 1, 0);
        REQUIRE(convertU8(source, 3, 1, 0, StreamFormat::GRAY,
                          actual, 3, 1, 0, StreamFormat::GRAY,
                          outside, Filter::NEAREST, Wrap::ZERO, 123) == inspirecv::SUCCESS);
        const uint8_t expected[] = {123, 123, 123};
        REQUIRE_EQ_C_ARRAY(actual, expected, sizeof(expected));
    }
}

TEST_CASE("task_core_float_normalization_and_stride", "[task][core][float][stride]") {
    constexpr int kWidth = 17;  // Exercise NEON and scalar tail paths.
    constexpr int kHeight = 2;
    constexpr int kChannels = 3;
    constexpr int kRowBytes = kWidth * kChannels * static_cast<int>(sizeof(float));
    constexpr int kOutputStride = kRowBytes + 16;

    std::vector<uint8_t> source(kWidth * kHeight * kChannels);
    for (int i = 0; i < kWidth * kHeight; ++i) {
        source[3 * i + 0] = static_cast<uint8_t>(3 + i);      // B
        source[3 * i + 1] = static_cast<uint8_t>(40 + i);     // G
        source[3 * i + 2] = static_cast<uint8_t>(100 + i);    // R
    }
    std::vector<uint8_t> output(kHeight * kOutputStride, 0xA5);

    StreamTask::Config config;
    config.sourceFormat = StreamFormat::BGR;
    config.destFormat = StreamFormat::RGB;
    config.filterType = Filter::NEAREST;
    config.wrap = Wrap::CLAMP_TO_EDGE;
    config.mean[0] = 1.0f;
    config.mean[1] = 2.0f;
    config.mean[2] = 3.0f;
    config.normal[0] = 0.5f;
    config.normal[1] = 0.25f;
    config.normal[2] = 2.0f;

    REQUIRE(convert(config, inspirecv::TransformMatrix::Identity(),
                    source.data(), kWidth, kHeight, 0,
                    output.data(), kWidth, kHeight, kChannels, kOutputStride,
                    halide_type_of<float>()) == inspirecv::SUCCESS);

    for (int y = 0; y < kHeight; ++y) {
        const auto* row = output.data() + y * kOutputStride;
        for (int x = 0; x < kWidth; ++x) {
            const int pixel = y * kWidth + x;
            const float expected[] = {
                (source[3 * pixel + 2] - config.mean[0]) * config.normal[0],
                (source[3 * pixel + 1] - config.mean[1]) * config.normal[1],
                (source[3 * pixel + 0] - config.mean[2]) * config.normal[2],
            };
            for (int c = 0; c < kChannels; ++c) {
                const float actual = readFloat(row + (x * kChannels + c) * sizeof(float));
                REQUIRE(actual == Approx(expected[c]).margin(1e-6));
            }
        }
        for (int i = kRowBytes; i < kOutputStride; ++i) {
            REQUIRE(row[i] == 0xA5);
        }
    }
}

TEST_CASE("task_core_nchw_tensor_view_writes_caller_memory_directly",
          "[task][core][tensor_view][nchw]") {
    constexpr int kWidth = 19;  // Exercise one NEON block and the scalar tail.
    constexpr int kHeight = 3;
    constexpr int kChannels = 3;
    constexpr size_t kRowElements = kWidth + 3;
    constexpr size_t kPlaneElements = kRowElements * kHeight + 5;
    constexpr float kSentinel = -9876.5f;

    std::vector<uint8_t> source(kWidth * kHeight * kChannels);
    for (int i = 0; i < kWidth * kHeight; ++i) {
        source[3 * i + 0] = static_cast<uint8_t>((3 * i + 7) & 255);   // B
        source[3 * i + 1] = static_cast<uint8_t>((5 * i + 11) & 255);  // G
        source[3 * i + 2] = static_cast<uint8_t>((7 * i + 13) & 255);  // R
    }
    std::vector<float> output(kPlaneElements * kChannels, kSentinel);

    StreamTask::Config config;
    config.sourceFormat = StreamFormat::BGR;
    config.destFormat = StreamFormat::RGB;
    config.filterType = Filter::NEAREST;
    config.wrap = Wrap::CLAMP_TO_EDGE;
    config.mean[0] = 1.0f;
    config.mean[1] = 2.0f;
    config.mean[2] = 3.0f;
    config.normal[0] = 0.5f;
    config.normal[1] = 0.25f;
    config.normal[2] = 2.0f;

    TensorView view;
    view.data = output.data();
    view.width = kWidth;
    view.height = kHeight;
    view.channels = kChannels;
    view.type = halide_type_of<float>();
    view.layout = TensorLayout::NCHW;
    view.rowStride = kRowElements * sizeof(float);
    view.channelStride = kPlaneElements * sizeof(float);

    auto* task = StreamTask::Create(config);
    task->SetMatrix(inspirecv::TransformMatrix::Identity());
    const auto status = task->Convert(source.data(), kWidth, kHeight, 0, view);
    StreamTask::Destroy(task);
    REQUIRE(status == inspirecv::SUCCESS);

    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            const int pixel = y * kWidth + x;
            const size_t offset = static_cast<size_t>(y) * kRowElements + x;
            const float expected[] = {
                (source[3 * pixel + 2] - config.mean[0]) * config.normal[0],
                (source[3 * pixel + 1] - config.mean[1]) * config.normal[1],
                (source[3 * pixel + 0] - config.mean[2]) * config.normal[2],
            };
            for (int channel = 0; channel < kChannels; ++channel) {
                REQUIRE(output[channel * kPlaneElements + offset] == expected[channel]);
            }
        }
        for (size_t x = kWidth; x < kRowElements; ++x) {
            for (int channel = 0; channel < kChannels; ++channel) {
                REQUIRE(output[channel * kPlaneElements + y * kRowElements + x] == kSentinel);
            }
        }
    }
    for (int channel = 0; channel < kChannels; ++channel) {
        const size_t planeDataEnd = static_cast<size_t>(channel) * kPlaneElements +
                                    kRowElements * kHeight;
        const size_t planeEnd = static_cast<size_t>(channel + 1) * kPlaneElements;
        for (size_t i = planeDataEnd; i < planeEnd; ++i) {
            REQUIRE(output[i] == kSentinel);
        }
    }
}

TEST_CASE("task_core_nchw_tensor_view_vector_boundaries_match_scalar_reference",
          "[task][core][tensor_view][nchw][simd]") {
    constexpr int kHeight = 3;
    constexpr int kChannels = 3;
    constexpr float kSentinel = -4321.25f;
    const std::array<int, 13> widths = {
        1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 257,
    };

    StreamTask::Config config;
    config.sourceFormat = StreamFormat::BGR;
    config.destFormat = StreamFormat::BGR;
    config.filterType = Filter::NEAREST;
    config.wrap = Wrap::CLAMP_TO_EDGE;
    config.mean[0] = 1.0f;
    config.mean[1] = 2.0f;
    config.mean[2] = 3.0f;
    config.normal[0] = 0.5f;
    config.normal[1] = 0.25f;
    config.normal[2] = 2.0f;

    for (const int width : widths) {
        CAPTURE(width);
        const size_t rowElements = static_cast<size_t>(width) + 5;
        const size_t planeElements = rowElements * kHeight + 7;
        std::vector<uint8_t> source(static_cast<size_t>(width) * kHeight * kChannels);
        for (size_t i = 0; i < source.size(); ++i) {
            source[i] = static_cast<uint8_t>((37 * i + 11) & 255);
        }
        std::vector<float> output(planeElements * kChannels, kSentinel);

        TensorView view;
        view.data = output.data();
        view.width = width;
        view.height = kHeight;
        view.channels = kChannels;
        view.type = halide_type_of<float>();
        view.layout = TensorLayout::NCHW;
        view.rowStride = rowElements * sizeof(float);
        view.channelStride = planeElements * sizeof(float);

        auto* task = StreamTask::Create(config);
        task->SetMatrix(inspirecv::TransformMatrix::Identity());
        const auto status = task->Convert(source.data(), width, kHeight, 0, view);
        StreamTask::Destroy(task);
        REQUIRE(status == inspirecv::SUCCESS);

        for (int y = 0; y < kHeight; ++y) {
            for (int x = 0; x < width; ++x) {
                const size_t pixel = static_cast<size_t>(y) * width + x;
                const size_t offset = static_cast<size_t>(y) * rowElements + x;
                for (int channel = 0; channel < kChannels; ++channel) {
                    const float expected = config.normal[channel] *
                        (source[3 * pixel + channel] - config.mean[channel]);
                    REQUIRE(output[channel * planeElements + offset] == expected);
                }
            }
            for (size_t x = width; x < rowElements; ++x) {
                for (int channel = 0; channel < kChannels; ++channel) {
                    REQUIRE(output[channel * planeElements +
                                   static_cast<size_t>(y) * rowElements + x] ==
                            kSentinel);
                }
            }
        }
        for (int channel = 0; channel < kChannels; ++channel) {
            const size_t dataEnd = static_cast<size_t>(channel) * planeElements +
                                   rowElements * kHeight;
            const size_t planeEnd = static_cast<size_t>(channel + 1) * planeElements;
            for (size_t i = dataEnd; i < planeEnd; ++i) {
                REQUIRE(output[i] == kSentinel);
            }
        }
    }
}

TEST_CASE("benchmark_task_nchw_tensor_view_identity",
          "[benchmark][task_tensor_view][nchw]") {
    struct SizeCase {
        int size;
        int iterations;
    };
    constexpr std::array<SizeCase, 4> sizes = {{
        {112, 2000}, {160, 1000}, {320, 200}, {640, 50},
    }};

    inspirecv::task::PipelineOptions options;
    options.input_format = inspirecv::task::PixelFormat::kBgr;
    options.output_format = inspirecv::task::PixelFormat::kBgr;
    options.sampling = inspirecv::task::SamplingMode::kNearest;
    options.border = inspirecv::task::BorderMode::kReplicate;
    options.mean = {{127.5f, 127.5f, 127.5f, 0.0f}};
    options.scale = {{1.0f / 128.0f, 1.0f / 128.0f,
                      1.0f / 128.0f, 1.0f}};
    inspirecv::task::Pipeline pipeline(options);
    pipeline.SetTransform(inspirecv::TransformMatrix::Identity());

    for (const auto sizeCase : sizes) {
        std::vector<uint8_t> source(
            static_cast<size_t>(sizeCase.size) * sizeCase.size * 3);
        for (size_t i = 0; i < source.size(); ++i) {
            source[i] = static_cast<uint8_t>((29 * i + 17) & 255);
        }
        std::vector<float> output(
            static_cast<size_t>(sizeCase.size) * sizeCase.size * 3);
        inspirecv::task::RawImageView input;
        input.data = source.data();
        input.width = sizeCase.size;
        input.height = sizeCase.size;
        inspirecv::task::TensorBuffer tensor;
        tensor.data = output.data();
        tensor.width = sizeCase.size;
        tensor.height = sizeCase.size;
        tensor.channels = 3;
        tensor.element_type = inspirecv::task::ElementType::kFloat32;
        tensor.order = inspirecv::task::TensorOrder::kChw;

        REQUIRE(pipeline.Run(input, tensor) == inspirecv::task::Status::kOk);
        for (int i = 1; i < 20; ++i) {
            pipeline.Run(input, tensor);
        }
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < sizeCase.iterations; ++i) {
            pipeline.Run(input, tensor);
        }
        const auto end = std::chrono::steady_clock::now();
        const double averageUs =
            std::chrono::duration<double, std::micro>(end - start).count() /
            sizeCase.iterations;
        std::cout << "[TaskTensorViewBench] size=" << sizeCase.size
                  << " iterations=" << sizeCase.iterations
                  << " average_us=" << averageUs << std::endl;
    }
}

TEST_CASE("task_core_identity_float_source_order_is_opt_in",
          "[task][core][tensor_view][nchw][source_order_compat]") {
    constexpr int kWidth = 19;  // Exercise NEON plus the scalar tail.
    std::vector<uint8_t> source(static_cast<size_t>(kWidth) * 3);
    for (int x = 0; x < kWidth; ++x) {
        source[3 * x + 0] = static_cast<uint8_t>(3 * x + 1);  // B
        source[3 * x + 1] = static_cast<uint8_t>(3 * x + 2);  // G
        source[3 * x + 2] = static_cast<uint8_t>(3 * x + 3);  // R
    }

    StreamTask::Config config;
    config.sourceFormat = StreamFormat::BGR;
    config.destFormat = StreamFormat::RGB;
    config.filterType = Filter::BILINEAR;
    config.preserveIdentityFloatOrder = true;
    config.mean[0] = 1.0f;
    config.mean[1] = 2.0f;
    config.mean[2] = 3.0f;
    config.normal[0] = 0.5f;
    config.normal[1] = 0.25f;
    config.normal[2] = 2.0f;

    std::vector<float> output(static_cast<size_t>(kWidth) * 3);
    TensorView view;
    view.data = output.data();
    view.width = kWidth;
    view.height = 1;
    view.channels = 3;
    view.type = halide_type_of<float>();
    view.layout = TensorLayout::NCHW;

    auto* task = StreamTask::Create(config);
    task->SetMatrix(inspirecv::TransformMatrix::Identity());
    const auto status = task->Convert(source.data(), kWidth, 1, 0, view);
    StreamTask::Destroy(task);
    REQUIRE(status == inspirecv::SUCCESS);

    for (int x = 0; x < kWidth; ++x) {
        // This intentionally preserves BGR source order even though destFormat
        // is RGB. The default-disabled path is covered by
        // task_core_nchw_tensor_view_writes_caller_memory_directly above.
        REQUIRE(output[x] ==
                (source[3 * x + 0] - config.mean[0]) * config.normal[0]);
        REQUIRE(output[kWidth + x] ==
                (source[3 * x + 1] - config.mean[1]) * config.normal[1]);
        REQUIRE(output[2 * kWidth + x] ==
                (source[3 * x + 2] - config.mean[2]) * config.normal[2]);
    }
}

TEST_CASE("task_core_nc4hw4_tensor_view_writes_blocked_caller_memory_directly",
          "[task][core][tensor_view][nc4hw4]") {
    constexpr int kWidth = 19;  // Exercise ARM64 vector code and its tail.
    constexpr int kHeight = 3;
    constexpr int kLogicalChannels = 3;
    constexpr size_t kStoredChannels = 4;
    constexpr size_t kRowElements = kWidth * kStoredChannels + 4;
    constexpr float kSentinel = -7654.25f;

    std::vector<uint8_t> source(kWidth * kHeight * kLogicalChannels);
    for (int i = 0; i < kWidth * kHeight; ++i) {
        source[3 * i + 0] = static_cast<uint8_t>((3 * i + 7) & 255);   // B
        source[3 * i + 1] = static_cast<uint8_t>((5 * i + 11) & 255);  // G
        source[3 * i + 2] = static_cast<uint8_t>((7 * i + 13) & 255);  // R
    }
    std::vector<float> output(kRowElements * kHeight, kSentinel);

    StreamTask::Config config;
    config.sourceFormat = StreamFormat::BGR;
    config.destFormat = StreamFormat::RGB;
    config.filterType = Filter::NEAREST;
    config.mean[0] = 1.0f;
    config.mean[1] = 2.0f;
    config.mean[2] = 3.0f;
    config.normal[0] = 0.5f;
    config.normal[1] = 0.25f;
    config.normal[2] = 2.0f;

    TensorView view;
    view.data = output.data();
    view.width = kWidth;
    view.height = kHeight;
    view.channels = kLogicalChannels;
    view.type = halide_type_of<float>();
    view.layout = TensorLayout::NC4HW4;
    view.rowStride = kRowElements * sizeof(float);

    auto* task = StreamTask::Create(config);
    task->SetMatrix(inspirecv::TransformMatrix::Identity());
    const auto status = task->Convert(source.data(), kWidth, kHeight, 0, view);
    StreamTask::Destroy(task);
    REQUIRE(status == inspirecv::SUCCESS);

    for (int y = 0; y < kHeight; ++y) {
        const float* row = output.data() + static_cast<size_t>(y) * kRowElements;
        for (int x = 0; x < kWidth; ++x) {
            const int pixel = y * kWidth + x;
            REQUIRE(row[4 * x + 0] ==
                    (source[3 * pixel + 2] - config.mean[0]) * config.normal[0]);
            REQUIRE(row[4 * x + 1] ==
                    (source[3 * pixel + 1] - config.mean[1]) * config.normal[1]);
            REQUIRE(row[4 * x + 2] ==
                    (source[3 * pixel + 0] - config.mean[2]) * config.normal[2]);
            REQUIRE(row[4 * x + 3] == 0.0f);
        }
        for (size_t x = kWidth * kStoredChannels; x < kRowElements; ++x) {
            REQUIRE(row[x] == kSentinel);
        }
    }
}

TEST_CASE("task_core_tensor_view_rejects_unsafe_layouts",
          "[task][core][tensor_view][errors]") {
    StreamTask::Config config;
    config.sourceFormat = StreamFormat::BGR;
    config.destFormat = StreamFormat::RGB;
    const uint8_t source[12] = {0};
    float dest[12] = {0};

    TensorView view;
    view.data = dest;
    view.width = 2;
    view.height = 2;
    view.channels = 3;
    view.layout = TensorLayout::NCHW;
    view.type = halide_type_of<float>();

    auto* task = StreamTask::Create(config);
    REQUIRE(task->Convert(source, 2, 2, 0, view) == inspirecv::SUCCESS);

    view.type = halide_type_of<uint8_t>();
    REQUIRE(task->Convert(source, 2, 2, 0, view) == inspirecv::INPUT_DATA_ERROR);
    view.type = halide_type_of<float>();

    view.rowStride = sizeof(float);
    REQUIRE(task->Convert(source, 2, 2, 0, view) == inspirecv::INPUT_DATA_ERROR);
    view.rowStride = 2 * sizeof(float);
    view.channelStride = sizeof(float);
    REQUIRE(task->Convert(source, 2, 2, 0, view) == inspirecv::INPUT_DATA_ERROR);

    view.rowStride = 0;
    view.channelStride = 0;
    view.channels = 4;
    REQUIRE(task->Convert(source, 2, 2, 0, view) == inspirecv::INPUT_DATA_ERROR);

    view.channels = 3;
    view.layout = TensorLayout::NC4HW4;
    view.rowStride = 7 * sizeof(float);
    REQUIRE(task->Convert(source, 2, 2, 0, view) == inspirecv::INPUT_DATA_ERROR);
    view.rowStride = 8 * sizeof(float);
    view.channelStride = 8 * sizeof(float);
    REQUIRE(task->Convert(source, 2, 2, 0, view) == inspirecv::INPUT_DATA_ERROR);
    StreamTask::Destroy(task);
}

TEST_CASE("task_core_rejects_invalid_inputs", "[task][core][errors]") {
    StreamTask::Config config;
    config.sourceFormat = StreamFormat::BGR;
    config.destFormat = StreamFormat::BGR;
    const uint8_t source[12] = {0};
    uint8_t dest[64] = {0};
    const auto identity = inspirecv::TransformMatrix::Identity();

    SECTION("null pointers") {
        REQUIRE(convert(config, identity, nullptr, 2, 2, 0,
                        dest, 2, 2, 3, 0, halide_type_of<uint8_t>()) == inspirecv::INPUT_DATA_ERROR);
        REQUIRE(convert(config, identity, source, 2, 2, 0,
                        nullptr, 2, 2, 3, 0, halide_type_of<uint8_t>()) == inspirecv::INPUT_DATA_ERROR);
    }

    SECTION("non-positive dimensions") {
        REQUIRE(convert(config, identity, source, 0, 2, 0,
                        dest, 2, 2, 3, 0, halide_type_of<uint8_t>()) == inspirecv::INPUT_DATA_ERROR);
        REQUIRE(convert(config, identity, source, 2, 2, 0,
                        dest, 2, 0, 3, 0, halide_type_of<uint8_t>()) == inspirecv::INPUT_DATA_ERROR);
    }

    SECTION("undersized source and destination strides") {
        REQUIRE(convert(config, identity, source, 2, 2, 5,
                        dest, 2, 2, 3, 0, halide_type_of<uint8_t>()) == inspirecv::INPUT_DATA_ERROR);
        REQUIRE(convert(config, identity, source, 2, 2, 0,
                        dest, 2, 2, 3, 5, halide_type_of<uint8_t>()) == inspirecv::INPUT_DATA_ERROR);
    }

    SECTION("only uint8 and float32 scalar output types are accepted") {
        const halide_type_t uint16Type = {halide_type_uint, 16, 1};
        const halide_type_t int32Type = {halide_type_int, 32, 1};
        const halide_type_t floatVectorType = {halide_type_float, 32, 4};
        REQUIRE(convert(config, identity, source, 2, 2, 0,
                        dest, 2, 2, 3, 0, uint16Type) == inspirecv::INPUT_DATA_ERROR);
        REQUIRE(convert(config, identity, source, 2, 2, 0,
                        dest, 2, 2, 3, 0, int32Type) == inspirecv::INPUT_DATA_ERROR);
        REQUIRE(convert(config, identity, source, 2, 2, 0,
                        dest, 2, 2, 3, 0, floatVectorType) == inspirecv::INPUT_DATA_ERROR);
    }
}

TEST_CASE("task_core_draw_channels_and_validation", "[task][core][draw]") {
    StreamTask::Config config;

    SECTION("GRAY supports multiple regions") {
        std::vector<uint8_t> image(4 * 3, 0);
        const int regions[] = {0, 1, 2, 2, 0, 3};
        const uint8_t color[] = {91};
        auto* task = StreamTask::Create(config);
        task->SetDraw();
        task->Draw(image.data(), 4, 3, 1, regions, 2, color);
        StreamTask::Destroy(task);
        const uint8_t expected[] = {
            0, 91, 91, 0,
            0, 0, 0, 0,
            91, 91, 91, 91,
        };
        REQUIRE_EQ_C_ARRAY(image.data(), expected, image.size());
    }

    SECTION("RGBA uses all four exact color bytes") {
        std::vector<uint8_t> image(3 * 2 * 4, 0);
        const int regions[] = {1, 1, 2};
        const uint8_t color[] = {7, 11, 13, 17};
        auto* task = StreamTask::Create(config);
        task->SetDraw();
        task->Draw(image.data(), 3, 2, 4, regions, 1, color);
        StreamTask::Destroy(task);
        const uint8_t expected[] = {
            0,0,0,0, 0,0,0,0, 0,0,0,0,
            0,0,0,0, 7,11,13,17, 7,11,13,17,
        };
        REQUIRE_EQ_C_ARRAY(image.data(), expected, image.size());
    }

    SECTION("an invalid region leaves the entire image unchanged") {
        std::vector<uint8_t> image(3 * 2 * 3, 0xCC);
        const auto before = image;
        const int regions[] = {0, 0, 1, 2, 0, 3};  // Second xEnd is outside width.
        const uint8_t color[] = {1, 2, 3};
        auto* task = StreamTask::Create(config);
        task->SetDraw();
        task->Draw(image.data(), 3, 2, 3, regions, 2, color);
        StreamTask::Destroy(task);
        REQUIRE_EQ_C_ARRAY(image.data(), before.data(), image.size());
    }
}
