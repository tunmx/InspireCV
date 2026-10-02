#include "../../common/common.h"

#include <inspirecv/inspirecv.h>
#include "inspirecv/core/runtime/cpu_features.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

namespace {
std::vector<uint32_t> ImageMasks() {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    return {0, inspirecv::cpu::kSsse3 | inspirecv::cpu::kSse41,
            inspirecv::cpu::kAllCpuFeatures};
#else
    return {inspirecv::cpu::kAllCpuFeatures};
#endif
}

template <typename T>
void CheckImage(const inspirecv::ImageT<T>& actual, const std::vector<T>& expected,
                int width, int height, int channels) {
    REQUIRE(actual.Width() == width);
    REQUIRE(actual.Height() == height);
    REQUIRE(actual.Channels() == channels);
    REQUIRE(expected.size() == static_cast<size_t>(width) * height * channels);
    // Bitwise comparison also checks copied negative zero and NaN payloads.
    const bool equal = std::memcmp(actual.Data(), expected.data(),
                                   expected.size() * sizeof(T)) == 0;
    if (!equal) {
        for (size_t i = 0; i < expected.size(); ++i) {
            if (std::memcmp(actual.Data() + i, expected.data() + i, sizeof(T))) {
                INFO("first mismatch index=" << i << " actual=" << +actual.Data()[i]
                     << " expected=" << +expected[i]);
                REQUIRE(equal);
            }
        }
    }
    REQUIRE(equal);
}

template <typename T>
void CheckGeometry() {
    for (uint32_t mask : ImageMasks()) {
        const inspirecv::cpu::ScopedCpuFeatureMask scope(mask);
        for (int channels : {1, 3, 4}) {
            for (int width : {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65}) {
                for (int height : {1, 3, 4, 5, 9, 17}) {
                    for (size_t offset : {size_t(0), size_t(1), size_t(7)}) {
                        CAPTURE(mask, channels, width, height, offset);
                        const size_t count = size_t(width) * height * channels;
                        std::vector<T> storage(count + offset + 32, T(113));
                        for (size_t i = 0; i < count; ++i)
                            storage[offset + i] = std::is_same<T, float>::value
                                ? T(int(i * 17 % 4093) - 2000) / T(8)
                                : T((i * 71 + i / 7) % 256);
                        const auto before = storage;
                        const T* src = storage.data() + offset;
                        auto image = inspirecv::ImageT<T>::Create(width, height, channels, src, false);
                        for (int operation = 0; operation < 5; ++operation) {
                            CAPTURE(operation);
                            const bool transpose = operation == 0 || operation == 2;
                            const int dw = transpose ? height : width;
                            const int dh = transpose ? width : height;
                            std::vector<T> expected(count);
                            for (int y = 0; y < dh; ++y) {
                                for (int x = 0; x < dw; ++x) {
                                    int sx = x, sy = y;
                                    if (operation == 0) { sx = y; sy = height - 1 - x; }
                                    if (operation == 1) { sx = width - 1 - x; sy = height - 1 - y; }
                                    if (operation == 2) { sx = width - 1 - y; sy = x; }
                                    if (operation == 3) sx = width - 1 - x;
                                    if (operation == 4) sy = height - 1 - y;
                                    std::copy_n(src + (sy * width + sx) * channels, channels,
                                                expected.data() + (y * dw + x) * channels);
                                }
                            }
                            auto out = operation == 0 ? image.Rotate90() :
                                       operation == 1 ? image.Rotate180() :
                                       operation == 2 ? image.Rotate270() :
                                       operation == 3 ? image.FlipHorizontal() : image.FlipVertical();
                            CheckImage(out, expected, dw, dh, channels);
                            REQUIRE(out.Data() != src);
                        }
                        REQUIRE(storage == before);
                    }
                }
            }
        }
    }
}
}  // namespace

TEST_CASE("image_u8_simd_geometry_independent_mapping", "[image][simd-dispatch][geometry]") {
    CheckGeometry<uint8_t>();
}

TEST_CASE("image_f32_simd_geometry_independent_mapping", "[image][simd-dispatch][geometry]") {
    CheckGeometry<float>();
}

#if !defined(INSPIRECV_BACKEND_OPENCV)
TEST_CASE("image_resize_u8_positive_halves_match_scalar_rounding",
          "[image][simd-dispatch][resize][regression]") {
    // OKCV maps x to x * source_width / destination_width. This 2x expansion
    // creates exact .5 samples, including SIMD blocks and the final tail.
    for (uint32_t mask : ImageMasks()) {
        const inspirecv::cpu::ScopedCpuFeatureMask scope(mask);
        for (int channels : {1, 3, 4}) {
            for (int width : {2, 3, 4, 5, 8, 9, 16, 17, 33}) {
                for (int height : {1, 3}) {
                    CAPTURE(mask, channels, width, height);
                    std::vector<uint8_t> src(size_t(width) * height * channels);
                    for (int y = 0; y < height; ++y)
                        for (int x = 0; x < width; ++x)
                            for (int c = 0; c < channels; ++c)
                                src[(y * width + x) * channels + c] = uint8_t(2*c + x);
                    std::vector<uint8_t> expected(src.size() * 4);
                    for (int y = 0; y < height * 2; ++y)
                        for (int x = 0; x < width * 2; ++x)
                            for (int c = 0; c < channels; ++c)
                                expected[(y * width * 2 + x) * channels + c] =
                                    uint8_t(2*c + std::min((x + 1) / 2, width - 1));
                    auto image = inspirecv::Image::Create(width, height, channels, src.data());
                    CheckImage(image.Resize(width * 2, height * 2, true), expected,
                               width * 2, height * 2, channels);
                }
            }
        }
    }
}

TEST_CASE("image_u8_simd_arithmetic_and_morphology_independent_reference",
          "[image][simd-dispatch][arithmetic][morphology]") {
    for (uint32_t mask : ImageMasks()) {
        const inspirecv::cpu::ScopedCpuFeatureMask scope(mask);
        for (int width : {1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 65}) {
            const int height = 5;
            for (int channels : {1, 3, 4}) {
                CAPTURE(mask, width, channels);
                const size_t count = size_t(width) * height * channels;
                std::vector<uint8_t> a(count + 33, 199), b(count + 35, 201);
                std::vector<uint8_t> m(size_t(width) * height + 33, 173);
                for (size_t i = 0; i < count; ++i) {
                    a[1 + i] = uint8_t(i * 37 + i / 5);
                    b[3 + i] = uint8_t(255 - i * 23);
                }
                for (size_t i = 0; i < size_t(width) * height; ++i)
                    m[1 + i] = uint8_t(i * 51);
                const auto before_a = a, before_b = b, before_m = m;
                auto ia = inspirecv::Image::Create(width, height, channels, a.data()+1, false);
                auto ib = inspirecv::Image::Create(width, height, channels, b.data()+3, false);
                auto im = inspirecv::Image::Create(width, height, 1, m.data()+1, false);
                std::vector<uint8_t> difference(count), blend(count);
                for (size_t i = 0; i < count; ++i) {
                    difference[i] = uint8_t(std::abs(int(a[1+i]) - int(b[3+i])));
                    // Mathematical nearest division, independent of the
                    // implementation's shift-based divide-by-255 identity.
                    const unsigned weight = m[1+i/channels];
                    blend[i] = uint8_t((weight*a[1+i]+(255-weight)*b[3+i]+127)/255);
                }
                CheckImage(ia.AbsDiff(ib), difference, width, height, channels);
                CheckImage(ia.Blend(ib, im), blend, width, height, channels);
                if (channels == 1) {
                    for (double threshold : {0., 127.75, 255.}) {
                        std::vector<uint8_t> expected(count);
                        for (size_t i = 0; i < count; ++i)
                            expected[i] = a[1+i] > uint8_t(threshold) ? 231 : 0;
                        CheckImage(ia.Threshold(threshold, 231.5, 0), expected, width, height, 1);
                    }
                    for (int kernel : {1, 2, 3, 4, 7, 15, 16}) {
                        for (int iterations : {1, 2}) {
                            CAPTURE(kernel, iterations);
                            for (bool maximum : {false, true}) {
                                std::vector<uint8_t> ref(a.begin()+1, a.begin()+1+count);
                                for (int pass = 0; pass < iterations; ++pass) {
                                    auto next = ref;
                                    for (int y = 0; y < height; ++y)
                                        for (int x = 0; x < width; ++x) {
                                            int v = maximum ? 0 : 255;
                                            for (int ky = 0; ky < kernel; ++ky)
                                                for (int kx = 0; kx < kernel; ++kx) {
                                                    const int sy = std::max(0, std::min(height-1, y+ky-kernel/2));
                                                    const int sx = std::max(0, std::min(width-1, x+kx-kernel/2));
                                                    v = maximum ? std::max(v, int(ref[sy*width+sx]))
                                                                : std::min(v, int(ref[sy*width+sx]));
                                                }
                                            next[y*width+x] = uint8_t(v);
                                        }
                                    ref.swap(next);
                                }
                                CheckImage(maximum ? ia.Dilate(kernel, iterations) : ia.Erode(kernel, iterations),
                                           ref, width, height, 1);
                            }
                        }
                    }
                }
                REQUIRE(a == before_a);
                REQUIRE(b == before_b);
                REQUIRE(m == before_m);
            }
        }
    }
}
#endif
