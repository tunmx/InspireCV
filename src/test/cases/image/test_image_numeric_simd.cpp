#include "../../common/common.h"
#include "image_test_utils.h"
#include "inspirecv/core/runtime/cpu_features.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <type_traits>
#include <vector>
#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

#if !defined(INSPIRECV_BACKEND_OPENCV)
#include "inspirecv/backends/okcv/bitmap/bitmap.h"
namespace {

std::vector<uint32_t> NumericMasks() {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    return {0, inspirecv::cpu::kSsse3 | inspirecv::cpu::kSse41,
            inspirecv::cpu::kAllCpuFeatures};
#else
    return {inspirecv::cpu::kAllCpuFeatures};
#endif
}

float F32(uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

bool SameFloat(float a, float b) {
    return std::memcmp(&a, &b, sizeof(float)) == 0 ||
           (std::isnan(a) && std::isnan(b));
}

// Force the scalar oracle to round each operation independently. This prevents
// the compiler from silently giving the reference the same FMA as a kernel.
float RoundedAdd(float a, float b) { volatile float v = a + b; return v; }
float RoundedMul(float a, float b) { volatile float v = a * b; return v; }
float RoundedSub(float a, float b) { volatile float v = a - b; return v; }
double RoundedAdd(double a, double b) { volatile double v = a + b; return v; }
double RoundedMul(double a, double b) { volatile double v = a * b; return v; }

bool LegacyImageAllowsFma() {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    return true;
#elif defined(INSPIRECV_TEST_WHOLE_PROJECT_AVX2) && INSPIRECV_TEST_WHOLE_PROJECT_AVX2
    // A runtime mask cannot remove compiler-generated FMA from an explicitly
    // whole-project AVX2 build. The isolated kernels still require strict math.
    return !inspirecv::cpu::HasAvx2();
#else
    return false;
#endif
}

template <typename T>
void RequireUnchanged(const std::vector<T>& before, const std::vector<T>& after) {
    REQUIRE(before.size() == after.size());
    REQUIRE(std::memcmp(before.data(), after.data(), before.size() * sizeof(T)) == 0);
}

template <typename T>
void RequireExact(const inspirecv::ImageT<T>& image, const std::vector<T>& expected,
                  int width, int height, int channels) {
    REQUIRE(image.Width() == width);
    REQUIRE(image.Height() == height);
    REQUIRE(image.Channels() == channels);
    REQUIRE(expected.size() == size_t(width) * height * channels);
    const bool equal = std::memcmp(image.Data(), expected.data(), expected.size() * sizeof(T)) == 0;
    if (!equal) {
        for (size_t i = 0; i < expected.size(); ++i) {
            if (std::memcmp(image.Data() + i, expected.data() + i, sizeof(T)) != 0) {
                INFO("first mismatch index=" << i << " actual=" << +image.Data()[i]
                     << " expected=" << +expected[i]);
                REQUIRE(equal);
            }
        }
    }
    REQUIRE(equal);
}

bool IntegerGrayContract() {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    return true;
#elif defined(INSPIRECV_TEST_WHOLE_PROJECT_AVX2) && INSPIRECV_TEST_WHOLE_PROJECT_AVX2
    // Explicit whole-project AVX2 builds retain the older integer fallback
    // after masking the new isolated, precise-double implementation.
    return !inspirecv::cpu::HasAvx2();
#else
    return false;
#endif
}

uint8_t GrayReference(uint8_t b, uint8_t g, uint8_t r) {
    if (IntegerGrayContract()) return uint8_t((29u * b + 150u * g + 77u * r + 128u) / 256u);
    const double sum = RoundedAdd(RoundedAdd(RoundedMul(.114, double(b)),
                                              RoundedMul(.587, double(g))),
                                  RoundedMul(.299, double(r)));
    return uint8_t(std::floor(sum + .5));
}

void RequireGrayF32(float actual, const float* p) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    // ARM's established float-coefficient route can contract either addition.
    // Check that finite values match one of those precise evaluation orders;
    // do not compare that contract against x86's double-coefficient formula.
    const float first = RoundedMul(.114f, p[0]);
    const float second[2] = {RoundedAdd(first, RoundedMul(.587f, p[1])),
                             std::fma(.587f, p[1], first)};
    bool equal = false;
    for (float value : second) {
        equal = equal || SameFloat(actual, RoundedAdd(value, RoundedMul(.299f, p[2]))) ||
                SameFloat(actual, std::fma(.299f, p[2], value));
    }
    REQUIRE(equal);
#else
    const double sum = RoundedAdd(RoundedAdd(RoundedMul(.114, double(p[0])),
                                              RoundedMul(.587, double(p[1]))),
                                  RoundedMul(.299, double(p[2])));
    REQUIRE(SameFloat(actual, float(sum)));
#endif
}

std::array<float, 20> FloatValues() {
    return {{0.f, -0.f, .5f, -.5f, 127.75f, -257.25f, .1f, -.1f,
             std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
             std::numeric_limits<float>::min(), -std::numeric_limits<float>::min(),
             std::numeric_limits<float>::denorm_min(), -std::numeric_limits<float>::denorm_min(),
             std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
             F32(0x7fc12345), F32(0xffc54321), std::nextafter(.5f, 0.f),
             std::nextafter(1.f, 2.f)}};
}

struct NumericPng {
    std::string path;
    NumericPng() {
        static unsigned sequence = 0;
        const auto directory = inspirecv_test_out_dir();
        REQUIRE(inspirecv_test_ensure_dir(directory));
#if defined(_WIN32)
        const int pid = _getpid();
#else
        const int pid = static_cast<int>(getpid());
#endif
        path = directory + "/numeric_simd_" + std::to_string(pid) + "_" +
               std::to_string(sequence++) + ".png";
    }
    ~NumericPng() { std::remove(path.c_str()); }
};

std::vector<float> GaussianWeights(int kernel_size, float sigma) {
    const int radius = kernel_size / 2;
    std::vector<float> weights(kernel_size);
    const float inv_sigma = 1.f / (2.f * sigma * sigma);
    float sum = 0.f;
    for (int x = -radius; x <= radius; ++x) {
        weights[x + radius] = std::exp(-(x * x) * inv_sigma);
        sum += weights[x + radius];
    }
    const float inverse_sum = 1.f / sum;
    for (float& value : weights) value *= inverse_sum;
    sum = 0.f;
    for (float value : weights) sum += value;
    weights[radius] += 1.f - sum;
    return weights;
}

template <typename T>
std::vector<T> GaussianReference(const T* input, int width, int height, int channels,
                                 int kernel_size, float sigma) {
    const int radius = kernel_size / 2;
    const auto weights = GaussianWeights(kernel_size, sigma);
    const size_t count = size_t(width) * height * channels;
    std::vector<float> horizontal(count);
    std::vector<T> output(count);
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
        for (int c = 0; c < channels; ++c) {
            float value = 0.f;
            for (int k = 0; k < kernel_size; ++k) {
                const int sx = std::max(0, std::min(width - 1, x + k - radius));
                const float product = RoundedMul(float(input[(size_t(y) * width + sx) * channels + c]), weights[k]);
                value = k == 0 ? product : RoundedAdd(value, product);
            }
            horizontal[(size_t(y) * width + x) * channels + c] = value;
        }
    }
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
        for (int c = 0; c < channels; ++c) {
            float value = 0.f;
            for (int k = 0; k < kernel_size; ++k) {
                const int sy = std::max(0, std::min(height - 1, y + k - radius));
                value = RoundedAdd(value, RoundedMul(horizontal[(size_t(sy) * width + x) * channels + c], weights[k]));
            }
            if (std::is_integral<T>::value) value = std::max(0.f, std::min(255.f, std::floor(value + .5f)));
            output[(size_t(y) * width + x) * channels + c] = T(value);
        }
    }
    return output;
}

// Enumerate exact rounded results of the existing left-to-right convolution,
// allowing a multiplication to contract with its adjacent addition. This checks
// legal FMA evaluation orders without a numerical tolerance or reassociation.
using GaussianValues = std::vector<float>;

void AddGaussianValue(GaussianValues& values, float value) {
    for (float existing : values) if (SameFloat(existing, value)) return;
    values.push_back(value);
}

GaussianValues GaussianConvolutionValues(const std::vector<GaussianValues>& samples,
                                         const std::vector<float>& weights) {
    GaussianValues values;
    for (float first : samples[0]) for (float second : samples[1]) {
        const float a = RoundedMul(first, weights[0]), b = RoundedMul(second, weights[1]);
        AddGaussianValue(values, RoundedAdd(a, b));
        AddGaussianValue(values, std::fma(first, weights[0], b));
        AddGaussianValue(values, std::fma(second, weights[1], a));
    }
    for (size_t k = 2; k < weights.size(); ++k) {
        GaussianValues next;
        for (float sum : values) for (float sample : samples[k]) {
            AddGaussianValue(next, RoundedAdd(sum, RoundedMul(sample, weights[k])));
            AddGaussianValue(next, std::fma(sample, weights[k], sum));
        }
        values = std::move(next);
    }
    return values;
}

template <typename T>
void RequireGaussianFmaReference(const inspirecv::ImageT<T>& actual, const T* input,
                                int width, int height, int channels, int kernel, float sigma) {
    const auto weights = GaussianWeights(kernel, sigma);
    const int radius = kernel / 2;
    const size_t count = size_t(width) * height * channels;
    std::vector<GaussianValues> horizontal(count);
    std::vector<GaussianValues> samples(kernel);
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x)
    for (int c = 0; c < channels; ++c) {
        for (int k = 0; k < kernel; ++k) {
            const int sx = std::max(0, std::min(width - 1, x + k - radius));
            samples[k] = {float(input[(size_t(y) * width + sx) * channels + c])};
        }
        horizontal[(size_t(y) * width + x) * channels + c] = GaussianConvolutionValues(samples, weights);
    }
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x)
    for (int c = 0; c < channels; ++c) {
        for (int k = 0; k < kernel; ++k) {
            const int sy = std::max(0, std::min(height - 1, y + k - radius));
            samples[k] = horizontal[(size_t(sy) * width + x) * channels + c];
        }
        const auto possible = GaussianConvolutionValues(samples, weights);
        const size_t i = (size_t(y) * width + x) * channels + c;
        bool equal = false;
        for (float value : possible) {
            if (std::is_integral<T>::value)
                value = std::max(0.f, std::min(255.f, std::floor(value + .5f)));
            equal = equal || SameFloat(float(actual.Data()[i]), float(T(value)));
        }
        if (!equal) {
            CAPTURE(x, y, c, i, actual.Data()[i], possible);
            REQUIRE(equal);
        }
    }
}

template <typename T>
void CheckGaussianPublic() {
    for (uint32_t mask : NumericMasks()) {
        const inspirecv::cpu::ScopedCpuFeatureMask scope(mask);
        for (int channels : {1, 3}) for (int width : {1, 2, 3, 7, 8, 9, 16, 17, 33})
        for (int height : {1, 2, 5}) for (int kernel : {3, 5}) for (float sigma : {.8f, 1.7f}) {
            CAPTURE(mask, channels, width, height, kernel, sigma);
            const size_t count = size_t(width) * height * channels;
            std::vector<T> source(count + 35, T(119));
            for (size_t i = 0; i < count; ++i) source[i + 3] = T((i * 71 + i / 11) % 256);
            const auto before = source;
            const auto expected = GaussianReference(source.data() + 3, width, height, channels, kernel, sigma);
            const auto image = inspirecv::ImageT<T>::Create(width, height, channels, source.data() + 3, false);
            const auto actual = image.GaussianBlur(kernel, sigma);
            if (LegacyImageAllowsFma()) {
                RequireGaussianFmaReference(actual, source.data() + 3,
                                            width, height, channels, kernel, sigma);
            } else {
                RequireExact(actual, expected, width, height, channels);
            }
            RequireUnchanged(before, source);
        }
    }
}

}  // namespace

TEST_CASE("image_numeric_gray_exhaustive_u8", "[image][simd-dispatch][numeric][gray]") {
    // Enumerate the complete 24-bit color domain without a giant image or one
    // Catch assertion per pixel. Each 65,536-pixel chunk covers a fixed blue.
    for (uint32_t mask : NumericMasks()) {
        const inspirecv::cpu::ScopedCpuFeatureMask scope(mask);
        std::vector<uint8_t> input(65536 * 3 + 35, 193), expected(65536);
        for (int blue = 0; blue < 256; ++blue) {
            CAPTURE(mask, blue);
            for (int green = 0; green < 256; ++green) for (int red = 0; red < 256; ++red) {
                const size_t i = size_t(green) * 256 + red;
                input[3 + i * 3] = uint8_t(blue);
                input[4 + i * 3] = uint8_t(green);
                input[5 + i * 3] = uint8_t(red);
                expected[i] = GrayReference(uint8_t(blue), uint8_t(green), uint8_t(red));
            }
            const auto before = input;
            const auto image = inspirecv::Image::Create(65536, 1, 3, input.data() + 3, false);
            RequireExact(image.ToGray(), expected, 65536, 1, 1);
            RequireUnchanged(before, input);
        }
    }
}

TEST_CASE("image_numeric_gray_mean_tails", "[image][simd-dispatch][numeric][gray][mean]") {
    const auto values = FloatValues();
    for (uint32_t mask : NumericMasks()) {
        const inspirecv::cpu::ScopedCpuFeatureMask scope(mask);
        for (int width : {1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 257}) {
            CAPTURE(mask, width);
            const int height = 3;
            std::vector<uint8_t> bytes(size_t(width) * height * 3 + 37, 173);
            std::vector<float> floats(bytes.size(), F32(0x7fc23456));
            std::vector<uint8_t> gray(size_t(width) * height);
            for (size_t i = 0; i < gray.size(); ++i) {
                for (int c = 0; c < 3; ++c) {
                    bytes[5 + i * 3 + c] = uint8_t(i * 67 + c * 103);
                    floats[5 + i * 3 + c] = values[(i * 3 + c) % values.size()];
                }
                gray[i] = GrayReference(bytes[5 + i * 3], bytes[6 + i * 3], bytes[7 + i * 3]);
            }
            const auto bytes_before = bytes;
            const auto floats_before = floats;
            RequireExact(inspirecv::Image::Create(width, height, 3, bytes.data() + 5, false).ToGray(), gray, width, height, 1);
            const auto image = inspirecv::ImageT<float>::Create(width, height, 3, floats.data() + 5, false);
            const auto fgray = image.ToGray();
            const auto mean = image.MeanChannels();
            for (size_t i = 0; i < gray.size(); ++i) {
                CAPTURE(i);
                const float* p = floats.data() + 5 + i * 3;
                RequireGrayF32(fgray.Data()[i], p);
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
                const float sum = RoundedAdd(RoundedAdd(p[0], p[1]), p[2]);
                const float expected = RoundedMul(sum, 1.f / 3.f);
#else
                const double sum = RoundedAdd(RoundedAdd(RoundedAdd(0., double(p[0])), double(p[1])), double(p[2]));
                const float expected = float(sum / 3.);
#endif
                REQUIRE(SameFloat(mean.Data()[i], expected));
            }
            RequireUnchanged(bytes_before, bytes);
            RequireUnchanged(floats_before, floats);
        }
    }
}

TEST_CASE("image_numeric_mean_all_u8_sums", "[image][simd-dispatch][numeric][mean]") {
    for (uint32_t mask : NumericMasks()) {
        const inspirecv::cpu::ScopedCpuFeatureMask scope(mask);
        for (int channels : {1, 3, 4}) {
            // Every attainable sum, repeated in all channel permutations and
            // with non-vector row widths, including sums divisible by three.
            const int width = channels * 255 + 1;
            std::vector<uint8_t> source(size_t(width) * channels * channels + 33, 211);
            std::vector<uint8_t> expected(size_t(width) * channels);
            for (int row = 0; row < channels; ++row) for (int sum = 0; sum < width; ++sum) {
                int remainder = sum;
                const size_t index = size_t(row) * width + sum;
                for (int c = 0; c < channels; ++c) {
                    const int value = std::min(remainder, 255);
                    source[1 + index * channels + (c + row) % channels] = uint8_t(value);
                    remainder -= value;
                }
                expected[index] = uint8_t(sum / channels);
            }
            CAPTURE(mask, channels);
            const auto before = source;
            const auto image = inspirecv::Image::Create(width, channels, channels, source.data() + 1, false);
            RequireExact(image.MeanChannels(), expected, width, channels, 1);
            RequireUnchanged(before, source);
        }
    }
}

TEST_CASE("image_numeric_float_threshold_difference_blend", "[image][simd-dispatch][numeric][float]") {
    const auto values = FloatValues();
    const uint8_t weights[] = {0, 1, 127, 128, 254, 255};
    for (uint32_t mask : NumericMasks()) {
        const inspirecv::cpu::ScopedCpuFeatureMask scope(mask);
        for (int channels : {1, 2, 3, 4, 5}) for (int width : {1, 3, 7, 8, 9, 15, 16, 17, 33, 65})
        for (int phase = 0; phase < 6; ++phase) {
            CAPTURE(mask, channels, width, phase);
            const int height = 3;
            const size_t pixels = size_t(width) * height, count = pixels * channels;
            std::vector<float> a(count + 35, F32(0x7fc34567)), b(count + 37, -991.f);
            std::vector<uint8_t> m(pixels + 35, 137);
            for (size_t i = 0; i < count; ++i) {
                a[3 + i] = values[(i + phase) % values.size()];
                b[5 + i] = values[(i * 7 + 3 + phase) % values.size()];
            }
            for (size_t i = 0; i < pixels; ++i) m[3 + i] = weights[(i + phase) % 6];
            const auto before_a = a, before_b = b;
            const auto before_m = m;
            const auto ia = inspirecv::ImageT<float>::Create(width, height, channels, a.data() + 3, false);
            const auto ib = inspirecv::ImageT<float>::Create(width, height, channels, b.data() + 5, false);
            const auto im = inspirecv::Image::Create(width, height, 1, m.data() + 3, false);
            const auto diff = ia.AbsDiff(ib), blend = ia.Blend(ib, im);
            for (size_t i = 0; i < count; ++i) {
                CAPTURE(i);
                const float av = a[3 + i], bv = b[5 + i];
                REQUIRE(SameFloat(diff.Data()[i], float(std::abs(double(av) - double(bv)))));
                const float weight = RoundedMul(float(m[3 + i / channels]), 1.f / 255.f);
                const float inverse = RoundedAdd(1.f, -weight);
                const float first = RoundedMul(weight, av), second = RoundedMul(inverse, bv);
                bool equal = SameFloat(blend.Data()[i], RoundedAdd(first, second));
                if (LegacyImageAllowsFma()) {
                    equal = equal || SameFloat(blend.Data()[i], std::fma(weight, av, second)) ||
                            SameFloat(blend.Data()[i], std::fma(inverse, bv, first));
                }
                REQUIRE(equal);
            }
            if (channels == 1) for (float threshold : {-1.f, 0.f, .5f, std::numeric_limits<float>::infinity(), F32(0x7fc45678)}) {
                for (float maximum : {-0.f, 17.25f, std::numeric_limits<float>::infinity(), F32(0x7fc56789)}) {
                    CAPTURE(threshold, maximum);
                    const auto out = ia.Threshold(threshold, maximum, 0);
                    for (size_t i = 0; i < count; ++i) {
                        const float expected = a[3 + i] > threshold ? maximum : 0.f;
                        REQUIRE(SameFloat(out.Data()[i], expected));
                    }
                }
            }
            RequireUnchanged(before_a, a);
            RequireUnchanged(before_b, b);
            RequireUnchanged(before_m, m);
        }
    }
}

TEST_CASE("image_numeric_swap_preserves_alpha_and_float_bits", "[image][simd-dispatch][numeric][swap]") {
    const uint32_t bits[] = {0, 0x80000000, 0x7fc12345, 0xffc54321, 0x7f800000,
                             0xff800000, 1, 0x80000001, 0x3f800001, 0x7f7fffff};
    for (uint32_t mask : NumericMasks()) {
        const inspirecv::cpu::ScopedCpuFeatureMask scope(mask);
        for (int channels : {3, 4}) for (int width : {1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65}) {
            CAPTURE(mask, channels, width);
            const size_t pixels = size_t(width) * 3, count = pixels * channels;
            std::vector<uint8_t> bytes(count + 35, 177), expected_bytes(count);
            std::vector<float> floats(count + 37, F32(0x7fc32145)), expected_floats(count);
            for (size_t i = 0; i < count; ++i) {
                bytes[i + 3] = uint8_t(i * 89 + i / 7);
                floats[i + 5] = F32(bits[i % 10]);
            }
            for (size_t pixel = 0; pixel < pixels; ++pixel) for (int c = 0; c < channels; ++c) {
                const int from = c == 0 ? 2 : c == 2 ? 0 : c;
                expected_bytes[pixel * channels + c] = bytes[3 + pixel * channels + from];
                std::memcpy(expected_floats.data() + pixel * channels + c,
                            floats.data() + 5 + pixel * channels + from, sizeof(float));
            }
            const auto before_bytes = bytes;
            const auto before_floats = floats;
            RequireExact(inspirecv::Image::Create(width, 3, channels, bytes.data() + 3, false).SwapRB(), expected_bytes, width, 3, channels);
            RequireExact(inspirecv::ImageT<float>::Create(width, 3, channels, floats.data() + 5, false).SwapRB(), expected_floats, width, 3, channels);
            RequireUnchanged(before_bytes, bytes);
            RequireUnchanged(before_floats, floats);
        }
    }
}

TEST_CASE("image_numeric_png_roundtrip_across_dispatch", "[image][simd-dispatch][numeric][io]") {
    for (uint32_t mask : NumericMasks()) {
        for (int channels : {1, 3}) for (int width : {1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 257}) {
            CAPTURE(mask, channels, width);
            const int height = 3;
            const size_t count = size_t(width) * height * channels;
            std::vector<uint8_t> bytes(count + 33, 153), expected(count);
            std::vector<float> floats(count + 35, -733.f), expected_float(count);
            for (size_t i = 0; i < count; ++i) {
                expected[i] = bytes[1 + i] = uint8_t(i * 53 + i / 7);
                expected_float[i] = floats[3 + i] = float(expected[i]);
            }
            const auto before_bytes = bytes;
            const auto before_floats = floats;
            NumericPng byte_file, float_file;
            {
                const inspirecv::cpu::ScopedCpuFeatureMask scope(mask);
                REQUIRE(inspirecv::Image::Create(width, height, channels, bytes.data() + 1, false).Write(byte_file.path));
                REQUIRE(inspirecv::ImageT<float>::Create(width, height, channels, floats.data() + 3, false).Write(float_file.path));
            }
            for (uint32_t read_mask : {uint32_t(0), inspirecv::cpu::kAllCpuFeatures}) {
                const inspirecv::cpu::ScopedCpuFeatureMask scope(read_mask);
                CAPTURE(read_mask);
                RequireExact(inspirecv::Image::Create(byte_file.path, channels), expected, width, height, channels);
                RequireExact(inspirecv::ImageT<float>::Create(byte_file.path, channels), expected_float, width, height, channels);
                RequireExact(inspirecv::Image::Create(float_file.path, channels), expected, width, height, channels);
                RequireExact(inspirecv::ImageT<float>::Create(float_file.path, channels), expected_float, width, height, channels);
            }
            RequireUnchanged(before_bytes, bytes);
            RequireUnchanged(before_floats, floats);
        }
    }
}

TEST_CASE("image_numeric_png_known_rgb_fixture", "[image][simd-dispatch][numeric][io]") {
    // Two known RGB pixels, encoded independently as a PNG scanline. A read/write
    // roundtrip alone would miss equal channel-order mistakes in both directions.
    const uint8_t png[] = {0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00, 0x00, 0x7b, 0x40, 0xe8, 0xdd, 0x00, 0x00, 0x00, 0x0f, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x63, 0xf8, 0xcf, 0x20, 0xc8, 0xe0, 0xf0, 0x1f, 0x00, 0x07, 0xc4, 0x02, 0x50, 0xf0, 0xa6, 0xaf, 0xc9, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82};
    NumericPng file;
    {
        std::ofstream stream(file.path, std::ios::binary);
        REQUIRE(stream.good());
        stream.write(reinterpret_cast<const char*>(png), sizeof(png));
        REQUIRE(stream.good());
    }
    for (uint32_t mask : NumericMasks()) {
        const inspirecv::cpu::ScopedCpuFeatureMask scope(mask);
        CAPTURE(mask);
        RequireExact(inspirecv::Image::Create(file.path, 3),
                     std::vector<uint8_t>{17, 0, 255, 255, 64, 0}, 2, 1, 3);
        RequireExact(inspirecv::ImageT<float>::Create(file.path, 3),
                     std::vector<float>{17, 0, 255, 255, 64, 0}, 2, 1, 3);
    }
}

TEST_CASE("image_numeric_png_float_saturation", "[image][simd-dispatch][numeric][io]") {
    const float values[] = {-1024.f, -1.f, -.5f, 0.f, .49f, .5f, .51f, 1.5f,
                             254.49f, 254.5f, 255.f, 255.5f, 1024.f};
    for (uint32_t mask : NumericMasks()) {
        const inspirecv::cpu::ScopedCpuFeatureMask scope(mask);
        for (int channels : {1, 3}) {
            CAPTURE(mask, channels);
            const int width = 35;
            std::vector<float> source(width * channels);
            std::vector<uint8_t> expected(source.size());
            for (size_t i = 0; i < source.size(); ++i) {
                source[i] = values[i % 13];
                expected[i] = uint8_t(std::floor(std::max(0., std::min(255., double(source[i]))) + .5));
            }
            NumericPng file;
            REQUIRE(inspirecv::ImageT<float>::Create(width, 1, channels, source.data()).Write(file.path));
            RequireExact(inspirecv::Image::Create(file.path, channels), expected, width, 1, channels);
        }
    }
}

TEST_CASE("image_numeric_gaussian_public_scalar_reference", "[image][simd-dispatch][numeric][gaussian]") {
    CheckGaussianPublic<uint8_t>();
    CheckGaussianPublic<float>();
}

TEST_CASE("image_numeric_resize_rounds_below_half_without_double_rounding",
          "[image][simd-dispatch][numeric][resize][regression]") {
    // 41 * float(2/164) is nextafter(.5f, 0), not .5f. Adding .5f in
    // single precision first changes it to exactly 1; roundf correctly gives 0.
    const float scale = 2.f / 164.f;
    const float halfway_neighbor = RoundedMul(41.f, scale);
    REQUIRE(halfway_neighbor == std::nextafter(.5f, 0.f));
    REQUIRE(std::round(halfway_neighbor) == 0.f);
    for (uint32_t mask : NumericMasks()) {
        const inspirecv::cpu::ScopedCpuFeatureMask scope(mask);
        for (int channels : {1, 3, 4}) {
            CAPTURE(mask, channels);
            std::vector<uint8_t> source(channels * 2 + 33, 173);
            std::fill_n(source.data() + 1, channels, uint8_t(0));
            std::fill_n(source.data() + 1 + channels, channels, uint8_t(1));
            std::vector<uint8_t> expected(164 * channels);
            for (int x = 0; x < 164; ++x) {
                const float position = RoundedMul(float(x), scale);
                const float value = std::min(position, 1.f);
                std::fill_n(expected.data() + x * channels, channels, uint8_t(std::round(value)));
            }
            const auto before = source;
            const auto image = inspirecv::Image::Create(2, 1, channels, source.data() + 1, false);
            RequireExact(image.Resize(164, 1, true), expected, 164, 1, channels);
            RequireUnchanged(before, source);
        }
    }
}

TEST_CASE("image_numeric_u8_resize_independent_coordinate_reference",
          "[image][simd-dispatch][numeric][resize]") {
    for (uint32_t mask : NumericMasks()) {
        const inspirecv::cpu::ScopedCpuFeatureMask scope(mask);
        for (int channels : {1, 3, 4}) for (int width : {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33})
        for (int height : {1, 2, 3, 7}) {
            const size_t count = size_t(width) * height * channels;
            std::vector<uint8_t> source(count + 35, 181);
            for (size_t i = 0; i < count; ++i) source[3 + i] = uint8_t(i * 53 + i / 17);
            const auto before = source;
            const auto image = inspirecv::Image::Create(width, height, channels, source.data() + 3, false);
            const std::pair<int, int> sizes[] = {
              {width * 2, height * 2}, {1, 1}, {width, 1}, {1, height},
              {width + 9, height + 1}, {17, 9}, {164, 3}, {std::max(1, width / 2), std::max(1, height / 2)}};
            for (const auto& size : sizes) {
                const int dw = size.first, dh = size.second;
                CAPTURE(mask, channels, width, height, dw, dh);
                const float x_scale = float(width) / dw, y_scale = float(height) / dh;
                const auto actual = image.Resize(dw, dh, true);
                REQUIRE(actual.Width() == dw);
                REQUIRE(actual.Height() == dh);
                REQUIRE(actual.Channels() == channels);
                for (int y = 0; y < dh; ++y) for (int x = 0; x < dw; ++x) {
                    const float sx = RoundedMul(float(x), x_scale), sy = RoundedMul(float(y), y_scale);
                    const int left = std::min(int(sx), width - 1), right = std::min(left + 1, width - 1);
                    const int top = std::min(int(sy), height - 1), bottom = std::min(top + 1, height - 1);
                    const float fx = sx - left, fy = sy - top;
                    for (int c = 0; c < channels; ++c) {
                        const float tl = source[3 + (top * width + left) * channels + c];
                        const float tr = source[3 + (top * width + right) * channels + c];
                        const float bl = source[3 + (bottom * width + left) * channels + c];
                        const float br = source[3 + (bottom * width + right) * channels + c];
                        const float upper = RoundedAdd(tl, RoundedMul(tr - tl, fx));
                        const float lower = RoundedAdd(bl, RoundedMul(br - bl, fx));
                        const float value = RoundedAdd(upper, RoundedMul(lower - upper, fy));
                        const int expected = int(std::round(value));
                        const int got = actual.Data()[(y * dw + x) * channels + c];
                        // Existing ARM and whole-project AVX2 fallback evaluators
                        // permit fused additions.
                        // Verify a precise possible FMA order instead of using
                        // an arbitrary pixel tolerance that could hide a bug.
                        bool equal = got == expected;
                        if (LegacyImageAllowsFma()) {
                            const float uppers[] = {upper, std::fma(tr - tl, fx, tl)};
                            const float lowers[] = {lower, std::fma(br - bl, fx, bl)};
                            for (float u : uppers) for (float l : lowers) {
                                equal = equal || got == int(std::round(RoundedAdd(u, RoundedMul(l - u, fy)))) ||
                                        got == int(std::round(std::fma(l - u, fy, u)));
                            }
                        }
                        if (!equal) {
                            CAPTURE(x, y, c, got, expected, fx, fy);
                            REQUIRE(equal);
                        }
                    }
                }
                RequireUnchanged(before, source);
            }
        }
    }
}

namespace {
std::vector<float> MorphologyFloatReference(const std::vector<float>& input,
                                           int width, int height, int kernel, bool erode) {
    std::vector<float> output(input.size());
    // The public square kernel is anchored at floor(kernel/2). Even kernels
    // therefore extend one pixel farther towards left/top. Clipping the window
    // is equivalent to replicating edge samples for finite min/max operations.
    const int before = kernel / 2, after = kernel - before - 1;
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
        const int y0 = std::max(0, y - before), y1 = std::min(height - 1, y + after);
        const int x0 = std::max(0, x - before), x1 = std::min(width - 1, x + after);
        float value = input[y0 * width + x0];
        for (int yy = y0; yy <= y1; ++yy) for (int xx = x0; xx <= x1; ++xx) {
            const float sample = input[yy * width + xx];
            if ((erode && sample < value) || (!erode && sample > value)) value = sample;
        }
        output[y * width + x] = value;
    }
    return output;
}

bool MatchesResizeFloat(float actual, float tl, float tr, float bl, float br, float fx, float fy) {
    const float upper_difference = RoundedSub(tr, tl), lower_difference = RoundedSub(br, bl);
    const float upper = RoundedAdd(tl, RoundedMul(upper_difference, fx));
    const float lower = RoundedAdd(bl, RoundedMul(lower_difference, fx));
    const float expected = RoundedAdd(upper, RoundedMul(RoundedSub(lower, upper), fy));
    if (SameFloat(actual, expected)) return true;
#if defined(__ARM_NEON) || defined(__ARM_NEON__) || \
    (defined(INSPIRECV_TEST_WHOLE_PROJECT_AVX2) && INSPIRECV_TEST_WHOLE_PROJECT_AVX2)
    // Existing ARM and explicitly enabled whole-project AVX2 paths permit
    // fused multiply-add. Enumerate their exact arithmetic, never a tolerance.
    const float uppers[] = {upper, std::fma(upper_difference, fx, tl)};
    const float lowers[] = {lower, std::fma(lower_difference, fx, bl)};
    for (float u : uppers) for (float l : lowers) {
        const float difference = RoundedSub(l, u);
        if (SameFloat(actual, RoundedAdd(u, RoundedMul(difference, fy))) ||
            SameFloat(actual, std::fma(difference, fy, u))) return true;
    }
#endif
    return false;
}

void CheckResizeFloatImage(const inspirecv::ImageT<float>& image, int width, int height) {
    const int sw = image.Width(), sh = image.Height(), channels = image.Channels();
    const auto linear = image.Resize(width, height, true);
    const auto nearest = image.Resize(width, height, false);
    REQUIRE(linear.Width() == width);
    REQUIRE(linear.Height() == height);
    REQUIRE(linear.Channels() == channels);
    REQUIRE(nearest.Width() == width);
    REQUIRE(nearest.Height() == height);
    REQUIRE(nearest.Channels() == channels);
    REQUIRE(linear.Data() != image.Data());
    REQUIRE(nearest.Data() != image.Data());
    const float scale_x = float(sw) / width, scale_y = float(sh) / height;
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
        const float sx = RoundedMul(float(x), scale_x), sy = RoundedMul(float(y), scale_y);
        const int x0 = std::min(int(sx), sw - 1), x1 = std::min(x0 + 1, sw - 1);
        const int y0 = std::min(int(sy), sh - 1), y1 = std::min(y0 + 1, sh - 1);
        const float fx = RoundedSub(sx, float(x0)), fy = RoundedSub(sy, float(y0));
        for (int c = 0; c < channels; ++c) {
            const auto tap = [&](int tx, int ty) { return image.Data()[(ty * sw + tx) * channels + c]; };
            const size_t index = (size_t(y) * width + x) * channels + c;
            const float expected_nearest = tap(x0, y0);
            const bool same_copy = std::memcmp(nearest.Data() + index, &expected_nearest, sizeof(float)) == 0;
            if (!same_copy) {
                CAPTURE(x, y, c, expected_nearest, nearest.Data()[index]);
                REQUIRE(same_copy);
            }
            // The same-size path explicitly clones instead of interpolating.
            const bool same_linear = width == sw && height == sh
              ? std::memcmp(linear.Data() + index, image.Data() + index, sizeof(float)) == 0
              : MatchesResizeFloat(linear.Data()[index], tap(x0, y0), tap(x1, y0),
                                   tap(x0, y1), tap(x1, y1), fx, fy);
            if (!same_linear) {
                CAPTURE(x, y, c, fx, fy, linear.Data()[index]);
                REQUIRE(same_linear);
            }
        }
    }
}
}  // namespace

TEST_CASE("image_numeric_f32_morphology_matches_independent_window_reference",
          "[image][simd-dispatch][numeric][morph][float]") {
    const float special[] = {
      -std::numeric_limits<float>::max(), -1e30f, -65536.5f, -257.25f, -1.f,
      -std::numeric_limits<float>::min(), -std::numeric_limits<float>::denorm_min(),
      0.f, std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::min(),
      .25f, 257.5f, 65536.f, 1e30f, std::numeric_limits<float>::max()};
    for (uint32_t mask : NumericMasks()) {
        const inspirecv::cpu::ScopedCpuFeatureMask scope(mask);
        for (int width : {1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65})
        for (int height : {1, 2, 3, 9, 17}) for (int pattern : {0, 1, 2}) {
            const size_t count = size_t(width) * height;
            std::vector<float> storage(count + 19, -197.25f);
            for (size_t i = 0; i < count; ++i) {
                if (pattern == 0) storage[i + 1] = special[(i * 7 + i / 13) % 15];
                else if (pattern == 1) storage[i + 1] = -1024.f - float((i * 37 + i / 11) % 997) * .125f;
                else storage[i + 1] = 1024.f + float((i * 53 + i / 7) % 991) * .25f;
            }
            const auto before = storage;
            const auto image = inspirecv::ImageT<float>::Create(width, height, 1, storage.data() + 1, false);
            for (int kernel : {1, 2, 3, 4, 7, 15, 16}) for (bool erode : {false, true}) {
                std::vector<float> expected(storage.begin() + 1, storage.begin() + 1 + count);
                for (int iterations : {1, 2}) {
                    CAPTURE(mask, width, height, pattern, kernel, erode, iterations);
                    expected = MorphologyFloatReference(expected, width, height, kernel, erode);
                    const auto actual = erode ? image.Erode(kernel, iterations) : image.Dilate(kernel, iterations);
                    RequireExact(actual, expected, width, height, 1);
                    REQUIRE(actual.Data() != image.Data());
                }
            }
            RequireUnchanged(before, storage);
        }
    }
}

TEST_CASE("image_numeric_f32_resize_matches_independent_coordinate_reference",
          "[image][simd-dispatch][numeric][resize][float]") {
    for (uint32_t mask : NumericMasks()) {
        const inspirecv::cpu::ScopedCpuFeatureMask scope(mask);
        for (int channels : {1, 3, 4})
        for (int width : {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33})
        for (int height : {1, 2, 3, 7}) for (int pattern : {0, 1}) {
            const size_t count = size_t(width) * height * channels;
            std::vector<float> storage(count + 19, -173.5f);
            for (size_t i = 0; i < count; ++i) {
                if (pattern == 0) storage[i + 1] = float(int((i * 53 + i / 7) % 2053) - 1026) * .125f;
                else {
                    // Diverse finite mantissas, signs and exponents, bounded
                    // so neighbor subtraction itself cannot overflow.
                    const uint32_t sign = (i & 1) ? 0x80000000u : 0u;
                    const uint32_t exponent = uint32_t(4 + (i * 37 + i / 11) % 244) << 23;
                    const uint32_t mantissa = uint32_t(i * 104729 + 3571) & 0x7fffffu;
                    storage[i + 1] = F32(sign | exponent | mantissa);
                }
            }
            const auto before = storage;
            const auto image = inspirecv::ImageT<float>::Create(width, height, channels, storage.data() + 1, false);
            const std::pair<int, int> sizes[] = {
              {width, height}, {width * 2, height * 2}, {1, 1}, {width, 1}, {1, height},
              {3, 5}, {4, 2}, {7, 3}, {8, 5}, {9, 2}, {15, 3}, {16, 5}, {17, 9},
              {width + 9, height + 1}, {33, 7}, {std::max(1, width / 2), std::max(1, height / 2)}};
            for (const auto& size : sizes) {
                CAPTURE(mask, channels, width, height, pattern, size);
                CheckResizeFloatImage(image, size.first, size.second);
            }
            RequireUnchanged(before, storage);
        }
    }
}

TEST_CASE("image_numeric_f32_resize_extremes_keep_interpolation_and_copy_contracts",
          "[image][simd-dispatch][numeric][resize][float][regression]") {
    // A zero interpolation weight does not permit replacing float arithmetic
    // with a copy: max - (-max) overflows before multiplication by zero.
    const float values[] = {
      std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
      0.f, -0.f, std::numeric_limits<float>::min(), -std::numeric_limits<float>::min(),
      std::numeric_limits<float>::denorm_min(), -std::numeric_limits<float>::denorm_min(),
      std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
      F32(0x7fc12345u), F32(0xffc54321u)};
    for (uint32_t mask : NumericMasks()) {
        const inspirecv::cpu::ScopedCpuFeatureMask scope(mask);
        for (int channels : {1, 3, 4}) {
            std::vector<float> storage(12 * channels + 19, -173.5f);
            for (int p = 0; p < 12; ++p) for (int c = 0; c < channels; ++c)
                storage[1 + p * channels + c] = values[p];
            const auto before = storage;
            const auto image = inspirecv::ImageT<float>::Create(6, 2, channels, storage.data() + 1, false);
            for (const auto& size : std::vector<std::pair<int, int>>{
                   {6, 2}, {3, 1}, {1, 1}, {12, 4}, {7, 3}, {8, 3}, {9, 3}, {17, 5}}) {
                CAPTURE(mask, channels, size);
                CheckResizeFloatImage(image, size.first, size.second);
            }
            RequireUnchanged(before, storage);
        }
    }
}

namespace {
template <typename T>
void CheckGeneralAffineReplicate() {
    // Binary fractions keep every operation exact, including on ARM with FMA.
    // The source begins at an unaligned address and retains guards on both sides.
    const std::array<float, 6> matrices[] = {
        {{1.f, .25f, -.25f, .25f, 1.f, 0.f}},
        {{1.f, .25f, 0.f, .25f, 1.f, -.25f}},
        {{.5f, -.25f, -3.75f, .25f, .5f, -2.75f}},
        {{-.5f, .25f, 4.75f, -.25f, -.5f, 3.75f}},
        {{.25f, .25f, 20.25f, .25f, .25f, 17.75f}}
    };
    for (uint32_t mask : NumericMasks()) {
        const inspirecv::cpu::ScopedCpuFeatureMask scope(mask);
        for (int channels : {1, 3, 4}) for (int sw : {1, 2, 7}) for (int sh : {1, 2, 5}) {
            const size_t count = size_t(sw) * sh * channels;
            std::vector<T> source(count + 19, T(197));
            for (size_t i = 0; i < count; ++i) source[i + 1] = T((i * 37 + i / 7) % 201);
            const auto before = source;
            okcv::Bitmap<T> image;
            image.Reset(sw, sh, channels, source.data() + 1, false);
            for (int width : {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17}) for (int height : {1, 3})
            for (const auto& m : matrices) {
                CAPTURE(mask, channels, sw, sh, width, height, m);
                const okcv::TransformMatrix matrix({m[0], m[1], m[2], m[3], m[4], m[5]});
                const auto actual = image.AffineBilinearOptimized(width, height, matrix, okcv::BORDER_MODE_REPLICATE);
                REQUIRE(actual.Width() == width);
                REQUIRE(actual.Height() == height);
                REQUIRE(actual.Channels() == channels);
                for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
                    const float sx = (y * m[1] + m[2]) + x * m[0];
                    const float sy = (y * m[4] + m[5]) + x * m[3];
                    const int x0 = int(std::floor(sx)), y0 = int(std::floor(sy));
                    const float fx = sx - x0, fy = sy - y0;
                    for (int c = 0; c < channels; ++c) {
                        const auto tap = [&](int tx, int ty) {
                            tx = std::max(0, std::min(sw - 1, tx));
                            ty = std::max(0, std::min(sh - 1, ty));
                            return float(source[1 + (ty * sw + tx) * channels + c]);
                        };
                        const float tl = tap(x0, y0), tr = tap(x0 + 1, y0);
                        const float bl = tap(x0, y0 + 1), br = tap(x0 + 1, y0 + 1);
                        const float upper = tl + (tr - tl) * fx, lower = bl + (br - bl) * fx;
                        const float value = upper + (lower - upper) * fy;
                        const T expected = std::is_same<T, uint8_t>::value ? T(std::round(value)) : T(value);
                        const T got = actual.Data()[(y * width + x) * channels + c];
                        if (got != expected) {
                            CAPTURE(x, y, c, got, expected, sx, sy);
                            REQUIRE(got == expected);
                        }
                    }
                }
            }
            RequireUnchanged(before, source);
        }
    }
}
}  // namespace

TEST_CASE("image_numeric_general_affine_clamps_each_replicate_tap_independently",
          "[image][simd-dispatch][numeric][affine-general][regression]") {
    CheckGeneralAffineReplicate<uint8_t>();
    CheckGeneralAffineReplicate<float>();
}

#endif  // !INSPIRECV_BACKEND_OPENCV
