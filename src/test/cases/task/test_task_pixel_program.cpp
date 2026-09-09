#include "../../common/common.h"

#include "inspirecv/task/planning/pixel_program.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {

using inspirecv::SUCCESS;
using inspirecv::task::BGR;
using inspirecv::task::BGRA;
using inspirecv::task::BILINEAR;
using inspirecv::task::CLAMP_TO_EDGE;
using inspirecv::task::GRAY;
using inspirecv::task::NEAREST;
using inspirecv::task::RGB;
using inspirecv::task::RGBA;
using inspirecv::task::StreamFormat;
using inspirecv::task::TensorLayout;
using inspirecv::task::internal::CompilePixelProgram;
using inspirecv::task::internal::DestinationRow;
using inspirecv::task::internal::PixelProgram;
using inspirecv::task::internal::PixelProgramSpec;
using inspirecv::task::internal::RunPixelSpan;
using inspirecv::task::internal::RunSampleSpan;
using inspirecv::task::internal::SampleLine;
using inspirecv::task::internal::SourceRow;

constexpr std::array<size_t, 13> kPixelCounts = {
  1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 255, 256, 257};

struct ChannelCase {
    StreamFormat source_format;
    StreamFormat destination_format;
    int source_channels;
    int destination_channels;
};

PixelProgramSpec MakeSpec(const ChannelCase& operation, halide_type_t type,
                          TensorLayout layout = TensorLayout::NHWC) {
    PixelProgramSpec specification;
    specification.source_format = operation.source_format;
    specification.destination_format = operation.destination_format;
    specification.filter = NEAREST;
    specification.wrap = CLAMP_TO_EDGE;
    specification.destination_type = type;
    specification.destination_layout = layout;
    specification.source_channels = operation.source_channels;
    specification.destination_channels = operation.destination_channels;
    specification.direct_sampling = true;
    return specification;
}

uint8_t Pattern(size_t pixel, int channel) {
    return static_cast<uint8_t>((37 * pixel + 53 * static_cast<size_t>(channel) + 11) & 255);
}

void BuildExpected(const ChannelCase& operation, const uint8_t* source,
                   size_t count, uint8_t* destination) {
    for (size_t pixel = 0; pixel < count; ++pixel) {
        const uint8_t* input = source + pixel * operation.source_channels;
        uint8_t* output = destination + pixel * operation.destination_channels;
        if (operation.source_format == operation.destination_format) {
            std::copy(input, input + operation.source_channels, output);
        } else if (operation.source_format == BGR && operation.destination_format == RGB) {
            output[0] = input[2];
            output[1] = input[1];
            output[2] = input[0];
        } else if (operation.source_format == GRAY && operation.destination_format == BGRA) {
            output[0] = input[0];
            output[1] = input[0];
            output[2] = input[0];
            output[3] = 255;
        } else if (operation.source_format == RGB && operation.destination_format == RGBA) {
            output[0] = input[0];
            output[1] = input[1];
            output[2] = input[2];
            output[3] = 255;
        } else if (operation.source_format == RGBA && operation.destination_format == BGR) {
            output[0] = input[2];
            output[1] = input[1];
            output[2] = input[0];
        }
    }
}

uint32_t FloatBits(float value) {
    uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

uint8_t BilinearGrayReference(const uint8_t* source, int width, int height, int stride,
                              float sampleX, float sampleY) {
    const float x = std::max(0.0f, std::min(sampleX, static_cast<float>(width - 1)));
    const float y = std::max(0.0f, std::min(sampleY, static_cast<float>(height - 1)));
    const int x0 = static_cast<int>(x);
    const int y0 = static_cast<int>(y);
    const int x1 = static_cast<int>(std::ceil(x));
    const int y1 = static_cast<int>(std::ceil(y));
    const float fractionX = x - static_cast<float>(x0);
    const float fractionY = y - static_cast<float>(y0);
    const float value =
      (1.0f - fractionX) * (1.0f - fractionY) * source[y0 * stride + x0] +
      fractionX * (1.0f - fractionY) * source[y0 * stride + x1] +
      fractionY * (1.0f - fractionX) * source[y1 * stride + x0] +
      fractionX * fractionY * source[y1 * stride + x1];
    return static_cast<uint8_t>(std::round(std::max(0.0f, std::min(value, 255.0f))));
}

}  // namespace

TEST_CASE("task_pixel_program_channel_contract_handles_all_tail_and_alignment_classes",
          "[task][kernel-contract][channels]") {
    const std::array<ChannelCase, 5> operations = {{
      {BGR, BGR, 3, 3},
      {BGR, RGB, 3, 3},
      {GRAY, BGRA, 1, 4},
      {RGB, RGBA, 3, 4},
      {RGBA, BGR, 4, 3},
    }};

    for (const ChannelCase& operation : operations) {
        PixelProgram program;
        REQUIRE(CompilePixelProgram(MakeSpec(operation, halide_type_of<uint8_t>()), &program) ==
                SUCCESS);
        for (const size_t count : kPixelCounts) {
            for (size_t offset = 0; offset < 16; ++offset) {
                CAPTURE(operation.source_format, operation.destination_format, count, offset);
                std::vector<uint8_t> source(offset + count * operation.source_channels + 16,
                                            0xA7);
                std::vector<uint8_t> destination(
                  offset + count * operation.destination_channels + 16, 0xCD);
                std::vector<uint8_t> expected(count * operation.destination_channels);
                std::vector<uint8_t> temporary(offset + count * 4 + 16, 0xEF);
                for (size_t pixel = 0; pixel < count; ++pixel) {
                    for (int channel = 0; channel < operation.source_channels; ++channel) {
                        source[offset + pixel * operation.source_channels + channel] =
                          Pattern(pixel, channel);
                    }
                }
                BuildExpected(operation, source.data() + offset, count, expected.data());

                const SourceRow input = {
                  source.data() + offset, operation.source_format, count,
                  operation.source_channels, count * static_cast<size_t>(operation.source_channels)};
                const DestinationRow output = {
                  destination.data() + offset, TensorLayout::NHWC, count,
                  operation.destination_channels, 1,
                  count * static_cast<size_t>(operation.destination_channels), 0};
                REQUIRE(RunPixelSpan(program, input, output, nullptr, nullptr,
                                     temporary.data() + offset, count * 4) == SUCCESS);
                REQUIRE(std::equal(expected.begin(), expected.end(),
                                   destination.begin() + static_cast<std::ptrdiff_t>(offset)));
                REQUIRE(std::all_of(destination.begin(),
                                    destination.begin() + static_cast<std::ptrdiff_t>(offset),
                                    [](uint8_t value) { return value == 0xCD; }));
                REQUIRE(std::all_of(
                  destination.begin() + static_cast<std::ptrdiff_t>(
                                          offset + count * operation.destination_channels),
                  destination.end(), [](uint8_t value) { return value == 0xCD; }));
            }
        }
    }
}

TEST_CASE("task_pixel_program_float_writers_preserve_layout_and_bit_patterns",
          "[task][kernel-contract][float]") {
    constexpr float mean[4] = {1.0f, 2.0f, 3.0f, 0.0f};
    constexpr float normal[4] = {0.5f, 0.25f, 2.0f, 0.0f};
    const ChannelCase swap = {BGR, RGB, 3, 3};

    SECTION("interleaved conversion and normalization") {
        PixelProgram program;
        REQUIRE(CompilePixelProgram(MakeSpec(swap, halide_type_of<float>()), &program) ==
                SUCCESS);
        for (const size_t count : kPixelCounts) {
            std::vector<uint8_t> source(count * 3 + 16);
            std::vector<float> destination(count * 3 + 8, -999.0f);
            std::vector<uint8_t> temporary(count * 4 + 16, 0xEF);
            for (size_t pixel = 0; pixel < count; ++pixel) {
                for (int channel = 0; channel < 3; ++channel) {
                    source[3 * pixel + channel] = Pattern(pixel, channel);
                }
            }
            const SourceRow input = {source.data(), BGR, count, 3, count * 3};
            const DestinationRow output = {destination.data(), TensorLayout::NHWC,
                                           count, 3, 4, count * 3 * sizeof(float), 0};
            REQUIRE(RunPixelSpan(program, input, output, mean, normal,
                                 temporary.data(), temporary.size()) == SUCCESS);
            for (size_t pixel = 0; pixel < count; ++pixel) {
                const uint8_t reordered[3] = {
                  source[3 * pixel + 2], source[3 * pixel + 1], source[3 * pixel + 0]};
                for (int channel = 0; channel < 3; ++channel) {
                    const float expected =
                      normal[channel] * (reordered[channel] - mean[channel]);
                    CAPTURE(count, pixel, channel);
                    REQUIRE(FloatBits(destination[3 * pixel + channel]) == FloatBits(expected));
                }
            }
        }
    }

    SECTION("planar rows honor caller channel stride") {
        const ChannelCase identity = {RGB, RGB, 3, 3};
        PixelProgram program;
        REQUIRE(CompilePixelProgram(
                  MakeSpec(identity, halide_type_of<float>(), TensorLayout::NCHW), &program) ==
                SUCCESS);
        for (const size_t count : kPixelCounts) {
            const size_t channelStride = count + 5;
            std::vector<uint8_t> source(count * 3 + 16);
            std::vector<float> destination(3 * channelStride, -999.0f);
            for (size_t pixel = 0; pixel < count; ++pixel) {
                for (int channel = 0; channel < 3; ++channel) {
                    source[3 * pixel + channel] = Pattern(pixel, channel);
                }
            }
            const SourceRow input = {source.data(), RGB, count, 3, count * 3};
            const DestinationRow output = {destination.data(), TensorLayout::NCHW,
                                           count, 3, 4, count * sizeof(float), channelStride};
            REQUIRE(RunPixelSpan(program, input, output, mean, normal, nullptr, 0) == SUCCESS);
            for (int channel = 0; channel < 3; ++channel) {
                for (size_t pixel = 0; pixel < count; ++pixel) {
                    const float expected = normal[channel] *
                                           (source[3 * pixel + channel] - mean[channel]);
                    CAPTURE(count, channel, pixel);
                    REQUIRE(FloatBits(destination[channel * channelStride + pixel]) ==
                            FloatBits(expected));
                }
                REQUIRE(std::all_of(
                  destination.begin() + static_cast<std::ptrdiff_t>(channel * channelStride + count),
                  destination.begin() + static_cast<std::ptrdiff_t>((channel + 1) * channelStride),
                  [](float value) { return value == -999.0f; }));
            }
        }
    }
}

TEST_CASE("task_pixel_program_sampler_contract_covers_boundaries_and_long_tails",
          "[task][kernel-contract][sampling]") {
    constexpr int width = 17;
    constexpr int height = 5;
    constexpr int stride = width + 5;
    std::array<uint8_t, height * stride> source{};
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            source[y * stride + x] = static_cast<uint8_t>((29 * x + 61 * y + 7) & 255);
        }
    }

    const ChannelCase gray = {GRAY, GRAY, 1, 1};
    SECTION("nearest supports 257 pixels and a nonzero destination start") {
        PixelProgramSpec specification = MakeSpec(gray, halide_type_of<uint8_t>());
        specification.direct_sampling = false;
        PixelProgram program;
        REQUIRE(CompilePixelProgram(specification, &program) == SUCCESS);
        constexpr size_t count = 257;
        constexpr size_t first = 3;
        std::array<uint8_t, first + count + 5> destination{};
        destination.fill(0xCD);
        const SampleLine line = {{-0.49f, 1.2f}, {0.51f, -0.13f}, first, count,
                                 first + count};
        REQUIRE(RunSampleSpan(program, source.data(), width, height, stride,
                              line, destination.data()) == SUCCESS);
        float x = line.origin.fX;
        float y = line.origin.fY;
        for (size_t index = 0; index < count; ++index) {
            const int sourceX = static_cast<int>(std::round(
              std::max(0.0f, std::min(x, static_cast<float>(width - 1)))));
            const int sourceY = static_cast<int>(std::round(
              std::max(0.0f, std::min(y, static_cast<float>(height - 1)))));
            CAPTURE(index, x, y);
            REQUIRE(destination[first + index] == source[sourceY * stride + sourceX]);
            x += line.step.fX;
            y += line.step.fY;
        }
        REQUIRE(std::all_of(destination.begin(), destination.begin() + first,
                            [](uint8_t value) { return value == 0xCD; }));
        REQUIRE(std::all_of(destination.begin() + first + count, destination.end(),
                            [](uint8_t value) { return value == 0xCD; }));
    }

    SECTION("bilinear locks negative, integer, nextafter, and border coordinates") {
        PixelProgramSpec specification = MakeSpec(gray, halide_type_of<uint8_t>());
        specification.filter = BILINEAR;
        specification.direct_sampling = false;
        PixelProgram program;
        REQUIRE(CompilePixelProgram(specification, &program) == SUCCESS);
        const float beforeOne = std::nextafter(1.0f, -std::numeric_limits<float>::infinity());
        const float afterOne = std::nextafter(1.0f, std::numeric_limits<float>::infinity());
        const std::array<inspirecv::task::Point, 8> coordinates = {{
          {-0.25f, -0.75f},
          {0.0f, 0.0f},
          {beforeOne, 0.5f},
          {1.0f, 1.0f},
          {afterOne, 1.5f},
          {7.375f, 3.125f},
          {static_cast<float>(width - 1), static_cast<float>(height - 1)},
          {static_cast<float>(width) - 0.25f, static_cast<float>(height) + 0.5f},
        }};
        for (size_t index = 0; index < coordinates.size(); ++index) {
            const auto& coordinate = coordinates[index];
            uint8_t destination = 0;
            const SampleLine line = {coordinate, {0.0f, 0.0f}, 0, 1, 1};
            REQUIRE(RunSampleSpan(program, source.data(), width, height, stride,
                                  line, &destination) == SUCCESS);
            CAPTURE(coordinate.fX, coordinate.fY);
#if defined(INSPIRECV_TASK_USE_NEON)
            // Frozen P0 ARM kernels use truncating fixed-point accumulation
            // for C1 bilinear sampling. Three fractional cases intentionally
            // differ by one from the portable roundf reference.
            constexpr std::array<uint8_t, 8> expectedArm = {
              7, 7, 66, 97, 127, 155, 203, 203};
            const uint8_t expected = expectedArm[index];
#else
            const uint8_t expected = BilinearGrayReference(
              source.data(), width, height, stride, coordinate.fX, coordinate.fY);
#endif
            REQUIRE(destination == expected);
        }
    }
}

TEST_CASE("task_pixel_program_compilation_is_transactional",
          "[task][kernel-contract][validation]") {
    const ChannelCase operation = {BGR, RGB, 3, 3};
    PixelProgram program;
    REQUIRE(CompilePixelProgram(MakeSpec(operation, halide_type_of<uint8_t>()), &program) ==
            SUCCESS);
    const auto frozenConvert = program.convert;

    PixelProgramSpec invalid = MakeSpec(operation, halide_type_of<uint8_t>());
    invalid.filter = static_cast<inspirecv::task::Filter>(99);
    REQUIRE(CompilePixelProgram(invalid, &program) == inspirecv::UNSUPPORTED_SAMPLER);
    REQUIRE(program.convert == frozenConvert);
    REQUIRE(CompilePixelProgram(invalid, nullptr) == inspirecv::INPUT_DATA_ERROR);
}
