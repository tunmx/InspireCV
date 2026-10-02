#include "../../common/common.h"

#include <inspirecv/task/task.h>
#include "inspirecv/core/runtime/cpu_features.h"
#include "inspirecv/task/kernels/cpu/color_ops.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {
using namespace inspirecv::task;
using inspirecv::cpu::ScopedCpuFeatureMask;

std::vector<uint32_t> ColorMasks() {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    return {0, inspirecv::cpu::kSsse3 | inspirecv::cpu::kSse41,
            inspirecv::cpu::kAllCpuFeatures};
#else
    return {inspirecv::cpu::kAllCpuFeatures};
#endif
}

struct ColorCase {
    PixelFormat output;
    kernels::color::PixelTransform* rgb;
    kernels::color::PixelTransform* bgr;
    int channels;
};
const ColorCase kColors[] = {
    {PixelFormat::kYCrCb, kernels::color::RgbToYCrCb, kernels::color::BgrToYCrCb, 3},
    {PixelFormat::kYuv, kernels::color::RgbToYuv, kernels::color::BgrToYuv, 3},
    {PixelFormat::kXyz, kernels::color::RgbToXyz, kernels::color::BgrToXyz, 3},
    {PixelFormat::kHsv, kernels::color::RgbToHsv, kernels::color::BgrToHsv, 3},
    {PixelFormat::kHsvFull, kernels::color::RgbToHsvFull, kernels::color::BgrToHsvFull, 3},
    {PixelFormat::kBgr555, kernels::color::RgbToBgr555, kernels::color::BgrToBgr555, 2},
    {PixelFormat::kBgr565, kernels::color::RgbToBgr565, kernels::color::BgrToBgr565, 2},
};

std::vector<std::array<uint8_t, 3>> ColorCorpus() {
    std::vector<std::array<uint8_t, 3>> result;
    for (int gray = 0; gray != 256; ++gray)
        result.push_back({{uint8_t(gray), uint8_t(gray), uint8_t(gray)}});
    const int edges[] = {0, 1, 2, 7, 127, 128, 254, 255};
    for (int r : edges) for (int g : edges) for (int b : edges)
        result.push_back({{uint8_t(r), uint8_t(g), uint8_t(b)}});
    // Saturation half-values expose direct-division / reciprocal differences.
    const std::array<uint8_t, 3> saturation_ties[] = {
        {{194, 128, 97}}, {{17, 17, 34}}, {{1, 1, 2}},
        {{128, 97, 194}}, {{97, 194, 128}}};
    for (const auto& rgb : saturation_ties) {
        result.push_back(rgb);
    }
    uint32_t state = 0x7183892u;
    for (int i = 0; i < 2048; ++i) {
        std::array<uint8_t, 3> rgb;
        for (auto& channel : rgb) {
            state = 1664525u * state + 1013904223u;
            channel = uint8_t(state >> 24);
        }
        result.push_back(rgb);
    }
    return result;
}

std::array<uint8_t, 3> ColorReference(const std::array<uint8_t, 3>& rgb,
                                     PixelFormat format) {
    const int r = rgb[0], g = rgb[1], b = rgb[2];
    if (format == PixelFormat::kBgr555 || format == PixelFormat::kBgr565) {
        // The packed format contract is a little-endian 16-bit RGB word.
        const unsigned word = format == PixelFormat::kBgr565
          ? unsigned(r / 8) * 2048 + unsigned(g / 4) * 32 + unsigned(b / 8)
          : unsigned(r / 8) * 1024 + unsigned(g / 8) * 32 + unsigned(b / 8);
        return {{uint8_t(word % 256), uint8_t(word / 256), 0}};
    }
    if (format == PixelFormat::kHsv || format == PixelFormat::kHsvFull) {
        const int top = std::max(r, std::max(g, b));
        const int bottom = std::min(r, std::min(g, b));
        const int chroma = top - bottom;
        if (chroma == 0) return {{0, 0, uint8_t(top)}};
        const int range = format == PixelFormat::kHsvFull ? 256 : 180;
        // MSVC Release historically folds the reciprocal into a direct
        // quotient. Other scalar modes retain the separately rounded inverse;
        // the original auto-vectorized GNU/Clang mode can differ at S ties.
#if defined(INSPIRECV_TASK_MSVC_FAST_HSV) && INSPIRECV_TASK_MSVC_FAST_HSV
        volatile float scaled_saturation = float(chroma * 255 * 4096) / float(top);
#else
        volatile float reciprocal = 1.f / float(top);
        volatile float scaled_saturation = float(chroma * 255 * 4096) * reciprocal;
#endif
        const int saturation = int(std::floor((int(scaled_saturation) + 2048) / 4096.));
        const int sector = r == top ? g-b : g == top ? b-r+2*chroma : r-g+4*chroma;
        volatile float divisor = 6.f * float(chroma);
        volatile float hue_ratio = float(range * 4096) / divisor;
        const int quantized_ratio = int(hue_ratio + 0.5f);
        int hue = int(std::floor((double(sector) * quantized_ratio + 2048.) / 4096.));
        if (hue < 0) hue += range;
        return {{uint8_t(std::max(0, std::min(255, hue))), uint8_t(saturation), uint8_t(top)}};
    }
    // Versioned fixed-point coefficients, intentionally not OpenCV's generic
    // color definition. Arithmetic is independent double/floor; integers here
    // are exactly representable and negative values never use signed shifts.
    int numerators[3], divisor, offsets[3] = {0, 128, 128};
    if (format == PixelFormat::kYCrCb) {
        numerators[0] = 4899*r + 9617*g + 1868*b;
        numerators[1] = 8192*r - 6860*g - 1332*b;
        numerators[2] = -2765*r - 5427*g + 8192*b;
        divisor = 16384;
    } else if (format == PixelFormat::kYuv) {
        numerators[0] = 4899*r + 9617*g + 1868*b;
        numerators[1] = -2412*r - 4734*g + 7146*b;
        numerators[2] = 10076*r - 8438*g - 1638*b;
        divisor = 16384;
    } else {
        numerators[0] = 1689*r + 1465*g + 739*b;
        numerators[1] = 871*r + 2929*g + 296*b;
        numerators[2] = 79*r + 488*g + 3892*b;
        divisor = 4096;
        offsets[1] = offsets[2] = 0;
    }
    std::array<uint8_t, 3> output;
    for (int c = 0; c < 3; ++c) {
        int value = int(std::floor(double(numerators[c]) / divisor + 0.5)) + offsets[c];
        if (format == PixelFormat::kXyz) value = std::max(0, std::min(255, value));
        // YCrCb/YUV preserve historical byte wrapping rather than saturation.
        output[c] = uint8_t((value % 256 + 256) % 256);
    }
    return output;
}

uint8_t SaturationReferenceTolerance(const std::array<uint8_t, 3>& rgb,
                                     PixelFormat format) {
#if !defined(_MSC_VER)
    if (format == PixelFormat::kHsv || format == PixelFormat::kHsvFull) {
        const int high = std::max(rgb[0], std::max(rgb[1], rgb[2]));
        const int low = std::min(rgb[0], std::min(rgb[1], rgb[2]));
        // The preserved GNU/Clang automatic vectorization differs only at
        // mathematical S half-values. Gray, H, V and all other colors are exact.
        if (high != 0 && (2 * (high-low) * 255) % (2*high) == high) return 1;
    }
#else
    (void)rgb;
    (void)format;
#endif
    return 0;
}

void CheckColorBytes(const std::vector<uint8_t>& actual,
                     const std::vector<uint8_t>& expected,
                     const std::vector<uint8_t>& tolerance = {}) {
    REQUIRE(actual.size() == expected.size());
    size_t mismatch = 0;
    while (mismatch < actual.size()) {
        const int limit = tolerance.empty() ? 0 : tolerance[mismatch];
        if (std::abs(int(actual[mismatch]) - int(expected[mismatch])) > limit) break;
        ++mismatch;
    }
    CAPTURE(mismatch);
    const int got = mismatch < actual.size() ? actual[mismatch] : -1;
    const int wanted = mismatch < expected.size() ? expected[mismatch] : -1;
    CAPTURE(got, wanted);
    REQUIRE(mismatch == actual.size());
}
}  // namespace

TEST_CASE("task_color_simd_matches_fixed_point_and_hsv_reference",
          "[task][simd-dispatch][color][regression]") {
    const auto corpus = ColorCorpus();
    for (uint32_t mask : ColorMasks()) {
        const ScopedCpuFeatureMask features(mask);
        for (const auto& conversion : kColors) for (bool bgr : {false, true}) {
            for (size_t count : {size_t(0), size_t(1), size_t(7), size_t(8), size_t(9),
                                 size_t(15), size_t(16), size_t(17), size_t(31), size_t(32), size_t(33), size_t(257)}) {
                for (size_t offset : {size_t(0), size_t(1), size_t(7), size_t(15)}) {
                    for (size_t first = 0; first < corpus.size(); first += std::max(size_t(1), count)) {
                        const size_t n = std::min(count, corpus.size() - first);
                        std::vector<uint8_t> input(offset + n*3 + 19, 0xA7);
                        std::vector<uint8_t> output(offset + n*conversion.channels + 19, 0xCD);
                        auto expected = output;
                        std::vector<uint8_t> tolerance(output.size(), 0);
                        for (size_t i = 0; i < n; ++i) {
                            const auto& rgb = corpus[first+i];
                            for (int c = 0; c < 3; ++c) input[offset+i*3+c] = rgb[bgr ? 2-c : c];
                            const auto result = ColorReference(rgb, conversion.output);
                            for (int c = 0; c < conversion.channels; ++c)
                                expected[offset+i*conversion.channels+c] = result[c];
                            if (conversion.channels == 3)
                                tolerance[offset+i*3+1] = SaturationReferenceTolerance(rgb, conversion.output);
                        }
                        const auto original = input;
                        CAPTURE(mask, conversion.output, bgr, count, offset, first);

                        (bgr ? conversion.bgr : conversion.rgb)(input.data()+offset, output.data()+offset, n);
                        CheckColorBytes(output, expected, tolerance);
                        REQUIRE(input == original);
                        if (count == 0) break;
                    }
                }
            }
        }
    }
}

TEST_CASE("task_public_color_routes_preserve_stride_padding_and_hsvfull_shape",
          "[task][simd-dispatch][color][stride]") {
    const auto corpus = ColorCorpus();
    for (uint32_t mask : ColorMasks()) {
        const ScopedCpuFeatureMask features(mask);
        for (const auto& conversion : kColors) for (bool bgr : {false, true}) {
            for (int width : {1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 257}) {
                constexpr int height = 3;
                const int source_stride = width*3 + 7;
                const int output_stride = width*conversion.channels + 5;
                std::vector<uint8_t> input(1 + source_stride*height + 19, 0xA7);
                std::vector<uint8_t> output(7 + output_stride*height + 19, 0xCD);
                auto expected = output;
                std::vector<uint8_t> tolerance(output.size(), 0);
                for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
                    const auto& rgb = corpus[(x*13 + y*179) % corpus.size()];
                    for (int c = 0; c < 3; ++c) input[1+y*source_stride+x*3+c] = rgb[bgr ? 2-c : c];
                    const auto result = ColorReference(rgb, conversion.output);
                    for (int c = 0; c < conversion.channels; ++c)
                        expected[7+y*output_stride+x*conversion.channels+c] = result[c];
                    if (conversion.channels == 3)
                        tolerance[7+y*output_stride+x*3+1] = SaturationReferenceTolerance(rgb, conversion.output);
                }
                const auto original = input;
                PipelineOptions options;
                options.input_format = bgr ? PixelFormat::kBgr : PixelFormat::kRgb;
                options.output_format = conversion.output;
                options.backend_preference = BackendPreference::kCpu;
                Pipeline pipeline(options);
                RawImageView source{input.data()+1, size_t(source_stride), width, height};
                TensorBuffer destination;
                destination.data = output.data()+7;
                destination.row_stride_bytes = output_stride;
                destination.width = width;
                destination.height = height;
                destination.channels = conversion.channels;
                destination.element_type = ElementType::kUInt8;
                CAPTURE(mask, conversion.output, bgr, width);
                REQUIRE(pipeline.ConfigurationStatus() == Status::kOk);
                REQUIRE(pipeline.Run(source, destination) == Status::kOk);
                CheckColorBytes(output, expected, tolerance);
                REQUIRE(input == original);
            }
        }
    }
}

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
TEST_CASE("task_hsv_dispatch_is_bit_exact_with_same_platform_fallback",
          "[task][simd-dispatch][color][hsv]") {
    const auto corpus = ColorCorpus();
    std::vector<uint8_t> source(corpus.size()*3);
    for (size_t i=0; i<corpus.size(); ++i)
        std::copy(corpus[i].begin(), corpus[i].end(), source.begin()+i*3);
    for (const auto& conversion : kColors) {
        if (conversion.output != PixelFormat::kHsv && conversion.output != PixelFormat::kHsvFull) continue;
        for (bool bgr : {false, true}) {
            for (size_t chunk : {size_t(1), size_t(7), size_t(8), size_t(9), size_t(15), size_t(16),
                                  size_t(17), size_t(31), size_t(32), size_t(33), size_t(255),
                                  size_t(256), size_t(257), corpus.size()}) {
                std::vector<uint8_t> baseline(source.size(), 0xCD);
                {
                    const ScopedCpuFeatureMask disabled(0);
                    for (size_t first = 0; first < corpus.size(); first += chunk) {
                        const size_t count = std::min(chunk, corpus.size()-first);
                        (bgr ? conversion.bgr : conversion.rgb)(source.data()+3*first, baseline.data()+3*first, count);
                    }
                }
                for (uint32_t mask : ColorMasks()) {
                    const ScopedCpuFeatureMask enabled(mask);
                    std::vector<uint8_t> actual(source.size(), 0xCD);
                    for (size_t first = 0; first < corpus.size(); first += chunk) {
                        const size_t count = std::min(chunk, corpus.size()-first);
                        (bgr ? conversion.bgr : conversion.rgb)(source.data()+3*first, actual.data()+3*first, count);
                    }
                    CAPTURE(conversion.output, bgr, mask, chunk);
                    CheckColorBytes(actual, baseline);
                }
            }
        }
    }
}
#endif
