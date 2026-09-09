#include "../../common/common.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#if !INSPIRECV_TEST_BACKEND_OPENCV
#include <inspirecv/backends/okcv/bitmap/bitmap.h>

namespace {

template <typename Pixel>
void CheckAxisAffine(int source_width, int source_height, int output_width,
                     int output_height, int channels, float sx, float sy,
                     float tx, float ty, okcv::BorderMode border) {
    // Dyadic coordinates and integer pixels make the reference exact in
    // double precision. Do not reuse production interpolation/coordinate code.
    std::vector<Pixel> pixels(source_width * source_height * channels);
    for (size_t i = 0; i < pixels.size(); ++i) {
        pixels[i] = static_cast<Pixel>((i * 37 + i / 7 * 19 + 11) % 251);
    }
    okcv::Bitmap<Pixel> source;
    source.Reset(source_width, source_height, channels, pixels.data());
    const okcv::TransformMatrix matrix({sx, 0, tx, 0, sy, ty});
    const Pixel border_value = static_cast<Pixel>(17);
    const auto actual = source.AffineBilinearOptimized(
      output_width, output_height, matrix, border, border_value);
    CAPTURE(source_width, source_height, output_width, output_height,
            channels, sx, sy, tx, ty, border);
    for (int y = 0; y < output_height; ++y) {
        for (int x = 0; x < output_width; ++x) {
            double px = double(x) * sx + tx;
            double py = double(y) * sy + ty;
            if (border == okcv::BORDER_MODE_REPLICATE) {
                px = std::max(0.0, std::min(px, double(source_width - 1)));
                py = std::max(0.0, std::min(py, double(source_height - 1)));
            }
            const int x0 = static_cast<int>(std::floor(px));
            const int y0 = static_cast<int>(std::floor(py));
            const double fx = px - x0;
            const double fy = py - y0;
            for (int c = 0; c < channels; ++c) {
                auto sample = [&](int ix, int iy) -> double {
                    if (border == okcv::BORDER_MODE_REPLICATE) {
                        ix = std::max(0, std::min(ix, source_width - 1));
                        iy = std::max(0, std::min(iy, source_height - 1));
                    }
                    if (ix < 0 || ix >= source_width || iy < 0 || iy >= source_height) {
                        return border_value;
                    }
                    return pixels[(iy * source_width + ix) * channels + c];
                };
                const double expected =
                  (1 - fx) * (1 - fy) * sample(x0, y0) +
                  fx * (1 - fy) * sample(x0 + 1, y0) +
                  (1 - fx) * fy * sample(x0, y0 + 1) +
                  fx * fy * sample(x0 + 1, y0 + 1);
                const Pixel rounded = std::is_same<Pixel, uint8_t>::value
                  ? static_cast<Pixel>(std::round(expected))
                  : static_cast<Pixel>(expected);
                CAPTURE(x, y, c);
                REQUIRE(actual.Data()[(y * output_width + x) * channels + c] == rounded);
            }
        }
    }
}

template <typename Pixel>
void CheckAxisMatrix() {
    for (int channels : {1, 3}) {
        for (int width : {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 65}) {
            for (const auto border : {okcv::BORDER_MODE_CONSTANT, okcv::BORDER_MODE_REPLICATE}) {
                for (float offset : {-1.25f, -0.5f, 0.0f, 0.125f, 0.5f, 1.75f}) {
                    CheckAxisAffine<Pixel>(19, 7, width, 9, channels,
                                          0.75f, 0.5f, offset, offset, border);
                }
                CheckAxisAffine<Pixel>(19, 7, width, 9, channels,
                                      -0.5f, -1, 18, 6, border);
                CheckAxisAffine<Pixel>(1, 1, width, 3, channels,
                                      1, 1, -0.25f, -0.5f, border);
            }
        }
        for (int source_size : {31, 32}) {
            for (int output_size : {31, 32}) {
                CheckAxisAffine<Pixel>(source_size, source_size, output_size, output_size,
                                      channels, 0.75f, 0.5f, -1.25f, -0.5f,
                                      okcv::BORDER_MODE_CONSTANT);
            }
        }
    }
}

}  // namespace

TEST_CASE("image_axis_affine_u8_preserves_each_simd_lane_and_tail",
          "[image][affine][regression][simd]") {
    CheckAxisMatrix<uint8_t>();
}

TEST_CASE("image_axis_affine_f32_preserves_each_simd_lane_and_tail",
          "[image][affine][regression][simd]") {
    CheckAxisMatrix<float>();
}

TEST_CASE("image_axis_affine_rounds_half_values_without_double_rounding",
          "[image][affine][regression][simd]") {
    for (int channels : {1, 3}) {
        std::vector<uint8_t> pixels(2 * channels, 0);
        std::fill(pixels.begin() + channels, pixels.end(), 1);
        okcv::Bitmap<uint8_t> source;
        source.Reset(2, 1, channels, pixels.data());
        for (float offset : {std::nextafter(0.5f, 0.f), 0.5f,
                             std::nextafter(0.5f, 1.f)}) {
            const okcv::TransformMatrix matrix({0, 0, offset, 0, 1, 0});
            const auto actual = source.AffineBilinearOptimized(
              5, 1, matrix, okcv::BORDER_MODE_REPLICATE, 0);
            const uint8_t expected = offset < 0.5f ? 0 : 1;
            for (int i = 0; i < 5 * channels; ++i) {
                CAPTURE(channels, offset, i);
                REQUIRE(actual.Data()[i] == expected);
            }
        }
    }
}

TEST_CASE("image_axis_affine_byte_identity_returns_independent_storage",
          "[image][affine][regression]") {
    for (int channels : {1, 3, 4}) {
        std::vector<uint8_t> pixels(257 * 3 * channels);
        for (size_t i = 0; i < pixels.size(); ++i) pixels[i] = static_cast<uint8_t>(i);
        okcv::Bitmap<uint8_t> source;
        source.Reset(257, 3, channels, pixels.data());
        const okcv::TransformMatrix matrix({1, 0, 0, 0, 1, 0});
        for (auto border : {okcv::BORDER_MODE_CONSTANT, okcv::BORDER_MODE_REPLICATE}) {
            const auto actual = source.AffineBilinearOptimized(257, 3, matrix, border, 17);
            REQUIRE(actual.Data() != source.Data());
            REQUIRE(std::equal(pixels.begin(), pixels.end(), actual.Data()));
        }
    }
}

TEST_CASE("image_axis_affine_constant_padding_preserves_float_semantics",
          "[image][affine][regression]") {
    for (int channels : {1, 3}) {
        std::vector<float> pixels(17 * 3 * channels, 42.f);
        okcv::Bitmap<float> source;
        source.Reset(17, 3, channels, pixels.data());
        for (float border : {-0.f, 3.5f, std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::quiet_NaN()}) {
            for (const auto& matrix : {okcv::TransformMatrix({1, 0, 5, 0, 1, 0}),
                                       okcv::TransformMatrix({1, 0, 0, 0, 1, 5})}) {
                const auto actual = source.AffineBilinearOptimized(
                  17, 3, matrix, okcv::BORDER_MODE_CONSTANT, border);
                // These pixels have all four taps outside the image. A finite
                // border is unchanged except -0 becomes +0 in interpolation;
                // infinity/NaN still participate in the original arithmetic.
                for (int y = 0; y < 3; ++y) {
                    for (int x = 12; x < 17; ++x) {
                        for (int c = 0; c < channels; ++c) {
                            const float value = actual.Data()[(y * 17 + x) * channels + c];
                            if (std::isfinite(border)) {
                                REQUIRE(value == border);
                                if (border == 0) REQUIRE_FALSE(std::signbit(value));
                            } else {
                                REQUIRE(std::isnan(value));
                            }
                        }
                    }
                }
            }
        }
        std::fill(pixels.end() - channels, pixels.end(), std::numeric_limits<float>::quiet_NaN());
        source.Reset(17, 3, channels, pixels.data());
        for (auto border : {okcv::BORDER_MODE_CONSTANT, okcv::BORDER_MODE_REPLICATE}) {
            const auto actual = source.AffineBilinearOptimized(
              17, 3, okcv::TransformMatrix({1, 0, 0, 0, 1, 0}), border, 0.f);
            REQUIRE(actual.Data()[0] == 42.f);
            for (int y : {1, 2}) {
                for (int x : {15, 16}) {
                    for (int c = 0; c < channels; ++c) {
                        REQUIRE(std::isnan(actual.Data()[(y * 17 + x) * channels + c]));
                    }
                }
            }
        }
    }
}
#endif

TEST_CASE("image_public_affine_half_pixel_translation", "[image][affine][regression]") {
    for (int channels : {1, 3}) {
        std::vector<uint8_t> pixels(8 * 2 * channels);
        for (int y = 0; y < 2; ++y) {
            for (int x = 0; x < 8; ++x) {
                for (int c = 0; c < channels; ++c) pixels[(y * 8 + x) * channels + c] = x * 10;
            }
        }
        const auto source = inspirecv::Image::Create(8, 2, channels, pixels.data());
        const auto matrix = inspirecv::TransformMatrix::Create(1, 0, 0.5f, 0, 1, 0);
        const auto actual = source.WarpAffine(matrix, 5, 1);
        for (int x = 0; x < 5; ++x) {
            for (int c = 0; c < channels; ++c) {
                REQUIRE(actual.Data()[x * channels + c] == x * 10 + 5);
            }
        }
    }
}
