#include "../../common/common.h"
#include "image_test_utils.h"
#include "inspirecv/core/runtime/cpu_features.h"

#include <inspirecv/inspirecv.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

// These tests preserve the standalone OKCV contract. OpenCV has different
// scalar saturation, raster endpoints, native-handle and stream semantics.
#if !defined(INSPIRECV_BACKEND_OPENCV) && \
    (!defined(INSPIRECV_TEST_BACKEND_OPENCV) || !INSPIRECV_TEST_BACKEND_OPENCV)
namespace {

std::vector<uint32_t> ApiEdgeMasks() {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    return {0, inspirecv::cpu::kSsse3 | inspirecv::cpu::kSse41,
            inspirecv::cpu::kAllCpuFeatures};
#else
    return {inspirecv::cpu::kAllCpuFeatures};
#endif
}

template<class T>
void RequireEdgePixels(const inspirecv::ImageT<T>& image, const std::vector<T>& expected,
                       int width, int height, int channels) {
    REQUIRE(image.Width() == width);
    REQUIRE(image.Height() == height);
    REQUIRE(image.Channels() == channels);
    REQUIRE(expected.size() == size_t(width) * height * channels);
    REQUIRE(std::memcmp(image.Data(), expected.data(), expected.size() * sizeof(T)) == 0);
}

template<class T>
void RequireEdgeStorage(const std::vector<T>& actual, const std::vector<T>& expected) {
    REQUIRE(actual.size() == expected.size());
    REQUIRE(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(T)) == 0);
}

uint8_t WrappedByte(int integer) {
    // Explicit mathematical modulo, independent of the production signed cast.
    return uint8_t((integer % 256 + 256) % 256);
}

template<class T>
void CheckWholeFill(const std::vector<double>& values) {
    for (uint32_t mask : ApiEdgeMasks()) {
        const inspirecv::cpu::ScopedCpuFeatureMask scope(mask);
        for (int channels : {1, 3, 4})
        for (int width : {1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65, 257})
        for (int offset : {1, 3, 7}) {
            const int height = 3;
            const size_t count = size_t(width) * height * channels;
            for (double value : values) {
                CAPTURE(mask, channels, width, offset, value);
                std::vector<T> storage(count + offset + 33, T(91));
                auto expected_storage = storage;
                const T expected = static_cast<T>(value);
                std::fill_n(expected_storage.data() + offset, count, expected);
                auto image = inspirecv::ImageT<T>::Create(width, height, channels,
                                                         storage.data() + offset, false);
                image.Fill(value);
                REQUIRE(image.Data() == storage.data() + offset);
                RequireEdgeStorage(storage, expected_storage);
            }
        }
    }
}

struct RasterCase {
    const char* name;
    int x0, y0, x1, y1, thickness;
    std::array<const char*, 7> rows;
};

// Hand-reviewed pixel maps, not results from another production drawing path.
// Existing OKCV axis-aligned lines have half-open endpoints. Diagonal lines
// include both endpoints. The small dyadic slopes below avoid ambiguous ties
// other than the explicitly specified nearest-positive-half raster locations.
const RasterCase line_cases[] = {
    {"horizontal clipped", -2, 3, 9, 3, 1,
     {{".......", ".......", ".......", "#######", ".......", ".......", "......."}}},
    {"horizontal thickness two", 1, 3, 6, 3, 2,
     {{".......", ".......", ".#####.", ".#####.", ".......", ".......", "......."}}},
    {"horizontal thickness three clipped top", -2, 0, 9, 0, 3,
     {{"#######", "#######", ".......", ".......", ".......", ".......", "......."}}},
    {"vertical clipped", 3, -2, 3, 9, 1,
     {{"...#...", "...#...", "...#...", "...#...", "...#...", "...#...", "...#..."}}},
    {"vertical thickness two clipped left", 0, -2, 0, 9, 2,
     {{"#......", "#......", "#......", "#......", "#......", "#......", "#......"}}},
    {"diagonal clipped", -2, -2, 8, 8, 1,
     {{"#......", ".#.....", "..#....", "...#...", "....#..", ".....#.", "......#"}}},
    {"negative slope clipped", -2, 8, 8, -2, 1,
     {{"......#", ".....#.", "....#..", "...#...", "..#....", ".#.....", "#......"}}},
    {"diagonal thickness three", -2, -2, 8, 8, 3,
     {{"##.....", "###....", ".###...", "..###..", "...###.", "....###", ".....##"}}},
    {"shallow slope", 0, 1, 6, 4, 1,
     {{".......", "#......", ".##....", "...##..", ".....##", ".......", "......."}}},
    {"steep slope", 1, 0, 4, 6, 1,
     {{".#.....", "..#....", "..#....", "...#...", "...#...", "....#..", "....#.."}}},
    {"coincident endpoints are rejected without drawing", 3, 3, 3, 3, 3,
     {{".......", ".......", ".......", ".......", ".......", ".......", "......."}}},
    {"zero thickness", -2, 3, 9, 3, 0,
     {{".......", ".......", ".......", ".......", ".......", ".......", "......."}}},
    {"entirely above image", -2, -3, 9, -3, 1,
     {{".......", ".......", ".......", ".......", ".......", ".......", "......."}}},
};

// x0/y0 are the public rectangle origin; x1/y1 are WIDTH/HEIGHT here.
// The historical half-open four-edge implementation omits the thin rectangle's
// bottom-right corner. These are explicitly legacy compatibility fixtures, not
// a claim that its rasterization matches a closed geometric/OpenCV rectangle.
const RasterCase rect_cases[] = {
    {"thin rectangle legacy endpoints", 1, 1, 4, 4, 1,
     {{".......", ".#####.", ".#...#.", ".#...#.", ".#...#.", ".####..", "......."}}},
    {"left clipped rectangle", -2, 1, 5, 4, 1,
     {{".......", "####...", "...#...", "...#...", "...#...", "###....", "......."}}},
    {"rectangle thickness two", 1, 1, 4, 4, 2,
     {{".####..", "######.", "##..##.", "##..##.", "######.", ".####..", "......."}}},
    {"rectangle thickness three", 1, 1, 4, 4, 3,
     {{".####..", "#######", "#######", "###.###", "#######", ".####..", ".####.."}}},
    {"zero area", 2, 2, 0, 0, 1,
     {{".......", ".......", ".......", ".......", ".......", ".......", "......."}}},
    {"zero height", 1, 2, 4, 0, 2,
     {{".......", ".......", ".......", ".......", ".......", ".......", "......."}}},
    {"zero thickness", 1, 1, 4, 4, 0,
     {{".......", ".......", ".......", ".......", ".......", ".......", "......."}}},
};

template<class T>
void CheckRasterFixtures(bool rectangles) {
    const RasterCase* fixtures = rectangles ? rect_cases : line_cases;
    const size_t fixture_count = rectangles ? sizeof(rect_cases) / sizeof(*rect_cases)
                                            : sizeof(line_cases) / sizeof(*line_cases);
    for (uint32_t mask : ApiEdgeMasks()) {
        const inspirecv::cpu::ScopedCpuFeatureMask scope(mask);
        for (int channels : {1, 3, 4}) for (size_t index = 0; index < fixture_count; ++index) {
            const auto& item = fixtures[index];
            for (int reverse = 0; reverse < (rectangles ? 1 : 2); ++reverse) {
                CAPTURE(mask, channels, item.name, reverse);
                const std::vector<double> color(channels, 173.0);
                // Drawing's native channel order is already an existing Fill
                // contract. Use distinguishable channels to catch lane leakage.
                auto native_color = color;
                for (int c = 0; c < channels; ++c) native_color[c] -= 31 * c;
                std::vector<T> storage(7 * 7 * channels + 35, T(29));
                auto expected = storage;
                for (int y = 0; y < 7; ++y) for (int x = 0; x < 7; ++x) {
                    REQUIRE(std::strlen(item.rows[y]) == 7);
                    if (item.rows[y][x] == '#') for (int c = 0; c < channels; ++c)
                        expected[3 + (y * 7 + x) * channels + c] = T(native_color[c]);
                }
                auto image = inspirecv::ImageT<T>::Create(7, 7, channels, storage.data() + 3, false);
                if (rectangles) {
                    image.DrawRect(inspirecv::Rect<int>(item.x0, item.y0, item.x1, item.y1),
                                   native_color, item.thickness);
                } else {
                    const inspirecv::Point<int> a(item.x0, item.y0), b(item.x1, item.y1);
                    image.DrawLine(reverse ? b : a, reverse ? a : b, native_color, item.thickness);
                }
                REQUIRE(image.Data() == storage.data() + 3);
                RequireEdgeStorage(storage, expected);
            }
        }
    }
}

template<class T>
void CheckRoiFill() {
    const int rects[][4] = {{-2, -1, 5, 4}, {3, 2, 9, 8}, {0, 0, 7, 5},
                            {2, 1, 0, 3}, {1, 2, 3, 0}, {2, 2, 1, 1},
                            {0, -4, 7, 2}};
    for (uint32_t mask : ApiEdgeMasks()) {
        const inspirecv::cpu::ScopedCpuFeatureMask scope(mask);
        for (int channels : {1, 3, 4}) for (const auto& r : rects) {
            CAPTURE(mask, channels, r[0], r[1], r[2], r[3]);
            std::vector<double> color(channels);
            for (int c = 0; c < channels; ++c) color[c] = 37 + 41 * c;
            std::vector<T> source(7 * 5 * channels + 35, T(11));
            auto expected = source;
            // The independent ROI oracle is half-open set intersection, not a
            // sequence of clipping/fill calls into the implementation.
            for (int y = 0; y < 5; ++y) for (int x = 0; x < 7; ++x)
                if (x >= r[0] && x < r[0] + r[2] && y >= r[1] && y < r[1] + r[3])
                    for (int c = 0; c < channels; ++c)
                        expected[3 + (y * 7 + x) * channels + c] = T(color[c]);
            auto image = inspirecv::ImageT<T>::Create(7, 5, channels, source.data() + 3, false);
            image.Fill(inspirecv::Rect<int>(r[0], r[1], r[2], r[3]), color);
            RequireEdgeStorage(source, expected);
        }
    }
}

struct ApiEdgeFile {
    std::string path;
    ApiEdgeFile() {
        static unsigned sequence = 0;
#if defined(_WIN32)
        const int pid = _getpid();
#else
        const int pid = int(getpid());
#endif
        const auto directory = inspirecv_test_out_dir();
        REQUIRE(inspirecv_test_ensure_dir(directory));
        path = directory + "/api_edge_" + std::to_string(pid) + "_" +
               std::to_string(sequence++) + ".bmp";
    }
    ~ApiEdgeFile() { std::remove(path.c_str()); }
};

uint32_t ReadLe32(const std::vector<uint8_t>& bytes, size_t offset) {
    REQUIRE(offset + 4 <= bytes.size());
    return uint32_t(bytes[offset]) | (uint32_t(bytes[offset + 1]) << 8) |
           (uint32_t(bytes[offset + 2]) << 16) | (uint32_t(bytes[offset + 3]) << 24);
}

template<class T>
void CheckPropertiesAndStream() {
    auto empty = inspirecv::ImageT<T>::Create();
    REQUIRE(empty.Empty());
    REQUIRE(empty.Width() == 0);
    REQUIRE(empty.Height() == 0);
    REQUIRE(empty.Channels() == 0);
    REQUIRE(empty.GetInternalImage() == static_cast<const void*>(empty.Data()));
    std::ostringstream empty_stream;
    REQUIRE(&(empty_stream << empty) == &empty_stream);
    REQUIRE(empty_stream.str().find("Size(H x W x C): 0 x 0 x 0") != std::string::npos);
    for (int channels : {1, 3, 4}) for (int width : {2, 13}) {
        CAPTURE(channels, width);
        std::vector<T> storage(width * 2 * channels + 7, T(13));
        for (size_t i = 1; i + 6 < storage.size(); ++i) storage[i] = T(i % 113);
        const auto before = storage;
        auto image = inspirecv::ImageT<T>::Create(width, 2, channels, storage.data() + 1, false);
        REQUIRE_FALSE(image.Empty());
        REQUIRE(image.GetInternalImage() == static_cast<void*>(storage.data() + 1));
        std::ostringstream stream;
        REQUIRE(&(stream << image) == &stream);
        const auto text = stream.str();
        const auto dimensions = "Size(H x W x C): 2 x " + std::to_string(width) +
                                " x " + std::to_string(channels);
        REQUIRE(text.find(dimensions) != std::string::npos);
        REQUIRE((text.find("...") != std::string::npos) == (width > 10));
        REQUIRE(image.Data() == storage.data() + 1);
        RequireEdgeStorage(storage, before);
        auto clone = image.Clone();
        REQUIRE(clone.GetInternalImage() == static_cast<const void*>(clone.Data()));
        REQUIRE(clone.GetInternalImage() != image.GetInternalImage());
    }
}

} // namespace

TEST_CASE("image_api_scalar_fill_all_channels_tails_and_guards",
          "[image][simd-dispatch][api-edges][scalar][fill]") {
    // All conversions here are defined and representable: out-of-range double
    // -> uint8 and nonfinite integer conversions are not a saturation contract.
    CheckWholeFill<uint8_t>({0, .75, 1, 127.75, 254.75, 255});
    CheckWholeFill<float>({0, -0., .25, -.25, 255.5, -65536., 65536.,
                          double(std::numeric_limits<float>::max()),
                          -double(std::numeric_limits<float>::max())});
    CheckRoiFill<uint8_t>();
    CheckRoiFill<float>();
}

TEST_CASE("image_api_u8_add_wrap_and_representable_multiply_independent_rational_oracle",
          "[image][simd-dispatch][api-edges][scalar]") {
    for (uint32_t mask : ApiEdgeMasks()) {
        const inspirecv::cpu::ScopedCpuFeatureMask scope(mask);
        for (int channels : {1, 3, 4})
        for (int width : {1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65, 257})
        for (int offset : {1, 3, 7}) {
            const size_t count = size_t(width) * 3 * channels;
            std::vector<uint8_t> source(count + offset + 33, 181);
            for (size_t i = 0; i < count; ++i) source[offset + i] = uint8_t(i * 67 + i / 13);
            const auto before = source;
            const auto image = inspirecv::Image::Create(width, 3, channels, source.data() + offset, false);
            for (int numerator : {-262144, -1027, -1024, -1023, -3, 0, 3, 1023, 1024, 1027, 262144}) {
                CAPTURE(mask, channels, width, offset, numerator);
                std::vector<uint8_t> expected(count);
                for (size_t i = 0; i < count; ++i)
                    expected[i] = WrappedByte((4 * int(source[offset + i]) + numerator) / 4);
                const auto output = image.Add(double(numerator) / 4);
                RequireEdgePixels(output, expected, width, 3, channels);
                REQUIRE(output.Data() != image.Data());
                RequireEdgeStorage(source, before);
            }
            for (int numerator : {0, 1, 2, 3, 4}) {
                CAPTURE(mask, channels, width, offset, numerator);
                std::vector<uint8_t> expected(count);
                for (size_t i = 0; i < count; ++i)
                    expected[i] = uint8_t(int(source[offset + i]) * numerator / 4);
                RequireEdgePixels(image.Mul(double(numerator) / 4), expected, width, 3, channels);
                RequireEdgeStorage(source, before);
            }
            // Exercise amplification without invoking float-to-byte overflow.
            for (int scale : {2, 4}) {
                std::vector<uint8_t> limited = source, expected(count);
                for (size_t i = 0; i < count; ++i) {
                    limited[offset + i] = uint8_t(source[offset + i] / scale);
                    expected[i] = uint8_t(int(limited[offset + i]) * scale);
                }
                const auto limited_before = limited;
                const auto input = inspirecv::Image::Create(width, 3, channels, limited.data() + offset, false);
                RequireEdgePixels(input.Mul(scale), expected, width, 3, channels);
                RequireEdgeStorage(limited, limited_before);
            }
        }
    }
}

TEST_CASE("image_api_f32_add_multiply_finite_extremes_and_tails",
          "[image][simd-dispatch][api-edges][scalar][float]") {
    const float values[] = {0, -0.f, .25f, -.25f, 127.75f, -255.5f,
                            65536.f, -65536.f, std::numeric_limits<float>::min(),
                            -std::numeric_limits<float>::min(),
                            std::numeric_limits<float>::max() / 4,
                            -std::numeric_limits<float>::max() / 4};
    for (uint32_t mask : ApiEdgeMasks()) {
        const inspirecv::cpu::ScopedCpuFeatureMask scope(mask);
        for (int channels : {1, 3, 4})
        for (int width : {1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65, 257})
        for (int offset : {1, 3, 7}) {
            const size_t count = size_t(width) * 3 * channels;
            std::vector<float> source(count + offset + 33, -901.f);
            for (size_t i = 0; i < count; ++i) source[offset + i] = values[i % 12];
            const auto before = source;
            const auto image = inspirecv::ImageT<float>::Create(width, 3, channels, source.data() + offset, false);
            for (float scalar : {0.f, -.5f, .5f, 1.f, 2.f, -256.f, 256.f}) {
                CAPTURE(mask, channels, width, offset, scalar);
                std::vector<float> expected(count);
                // Dyadic scalars and binary32 inputs: one correctly rounded
                // real addition/multiplication, independently evaluated in
                // binary64. No float loop or production helper is reused.
                for (size_t i = 0; i < count; ++i)
                    expected[i] = float(double(source[offset + i]) + double(scalar));
                RequireEdgePixels(image.Add(scalar), expected, width, 3, channels);
                if (std::abs(scalar) <= 2.f) {
                    for (size_t i = 0; i < count; ++i)
                        expected[i] = float(double(source[offset + i]) * double(scalar));
                    RequireEdgePixels(image.Mul(scalar), expected, width, 3, channels);
                }
                RequireEdgeStorage(source, before);
            }
        }
    }
}

TEST_CASE("image_api_drawing_independent_pixel_maps_preserve_legacy_endpoints",
          "[image][simd-dispatch][api-edges][drawing][legacy-contract]") {
    CheckRasterFixtures<uint8_t>(false);
    CheckRasterFixtures<float>(false);
    CheckRasterFixtures<uint8_t>(true);
    CheckRasterFixtures<float>(true);
}

TEST_CASE("image_api_native_handle_and_stream_properties",
          "[image][api-edges][storage][stream]") {
    CheckPropertiesAndStream<uint8_t>();
    CheckPropertiesAndStream<float>();
}

TEST_CASE("image_api_nearest_c1_downsample_uses_exact_pixel_indices",
          "[image][simd-dispatch][api-edges][resize][nearest]") {
    // Dyadic scale ratios make the integer floor oracle exact in binary32.
    // The 7->3 case additionally exercises a non-dyadic scale whose two
    // nonzero coordinates stay at least 1/3 away from an integer boundary.
    // Same-width/vertically enlarged cases also exercise repeated source rows.
    const int shapes[][4] = {{17, 9, 8, 4}, {33, 7, 16, 8}, {65, 3, 32, 8},
                             {31, 7, 16, 4}, {7, 5, 3, 2}, {33, 9, 33, 16},
                             {1, 7, 1, 16}, {64, 8, 16, 2}, {17, 8, 1, 1}};
    for (uint32_t mask : ApiEdgeMasks()) {
        const inspirecv::cpu::ScopedCpuFeatureMask scope(mask);
        for (const auto& shape : shapes) for (int offset : {1, 3, 7, 31}) {
            const int sw = shape[0], sh = shape[1], dw = shape[2], dh = shape[3];
            CAPTURE(mask, offset, sw, sh, dw, dh);
            std::vector<uint8_t> source(size_t(sw) * sh + offset + 33, 179);
            for (int y = 0; y < sh; ++y) for (int x = 0; x < sw; ++x)
                source[offset + y * sw + x] = uint8_t(x * 37 + y * 67 + x * y);
            const auto before = source;
            std::vector<uint8_t> expected(size_t(dw) * dh);
            for (int y = 0; y < dh; ++y) for (int x = 0; x < dw; ++x)
                expected[y * dw + x] = source[offset + (y * sh / dh) * sw + x * sw / dw];
            const auto image = inspirecv::Image::Create(sw, sh, 1, source.data() + offset, false);
            const auto output = image.Resize(dw, dh, false);
            RequireEdgePixels(output, expected, dw, dh, 1);
            REQUIRE(output.Data() != image.Data());
            RequireEdgeStorage(source, before);
        }
    }
}

TEST_CASE("image_api_bmp_known_pixels_and_independent_writer_bytes",
          "[image][simd-dispatch][api-edges][io]") {
    // Independent 24-bit BMP, width3 x height2. Positive height means bottom-up;
    // three-byte BGR pixels use twelve-byte rows with three bytes of padding.
    const uint8_t bmp[] = {
        'B','M',78,0,0,0,0,0,0,0,54,0,0,0,40,0,0,0,3,0,0,0,2,0,0,0,1,0,24,0,
        0,0,0,0,24,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        17,0,255,255,64,0,1,2,3,0,0,0,
        11,22,33,44,55,66,77,88,99,0,0,0};
    const std::vector<uint8_t> expected = {11,22,33,44,55,66,77,88,99,
                                          17,0,255,255,64,0,1,2,3};
    const std::vector<float> expected_float(expected.begin(), expected.end());
    ApiEdgeFile fixture;
    {
        std::ofstream file(fixture.path, std::ios::binary);
        REQUIRE(file.good());
        file.write(reinterpret_cast<const char*>(bmp), sizeof(bmp));
        REQUIRE(file.good());
    }
    for (uint32_t mask : ApiEdgeMasks()) {
        const inspirecv::cpu::ScopedCpuFeatureMask scope(mask);
        CAPTURE(mask);
        RequireEdgePixels(inspirecv::Image::Create(fixture.path, 3), expected, 3, 2, 3);
        RequireEdgePixels(inspirecv::ImageT<float>::Create(fixture.path, 3), expected_float, 3, 2, 3);
        for (bool floats : {false, true}) {
            ApiEdgeFile output;
            if (floats)
                REQUIRE(inspirecv::ImageT<float>::Create(3, 2, 3, expected_float.data()).Write(output.path));
            else REQUIRE(inspirecv::Image::Create(3, 2, 3, expected.data()).Write(output.path));
            std::ifstream stream(output.path, std::ios::binary);
            REQUIRE(stream.good());
            const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(stream)),
                                              std::istreambuf_iterator<char>());
            REQUIRE(bytes.size() >= 78);
            REQUIRE(bytes[0] == 'B'); REQUIRE(bytes[1] == 'M');
            REQUIRE(ReadLe32(bytes, 18) == 3);
            REQUIRE(ReadLe32(bytes, 22) == 2);
            REQUIRE(bytes[28] == 24); REQUIRE(bytes[29] == 0);
            REQUIRE(ReadLe32(bytes, 30) == 0);
            const size_t start = ReadLe32(bytes, 10);
            REQUIRE(start + 24 <= bytes.size());
            for (int y = 0; y < 2; ++y) for (int x = 0; x < 3; ++x) for (int c = 0; c < 3; ++c)
                REQUIRE(bytes[start + (1 - y) * 12 + x * 3 + c] == expected[(y * 3 + x) * 3 + c]);
        }
    }
}
#endif // Standalone OKCV contract only.
