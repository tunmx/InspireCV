#include "../../common/common.h"

#include <inspirecv/task/core/matrix.h>
#include <inspirecv/version.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {

using inspirecv::task::Matrix;
using inspirecv::task::Point;
using inspirecv::task::Rect;

#if defined(__GNUC__) && !defined(__clang__)
constexpr uint32_t kPerspectiveInversePerspX = 0xb9939a86;
constexpr uint32_t kPerspectiveInversePerspY = 0x3b139a86;
constexpr uint32_t kComposedSkewX = 0xbe866667;
constexpr uint32_t kComposedSkewY = 0x3e400000;
constexpr uint32_t kComposedPerspX = 0x00000000;
constexpr uint32_t kComposedPoint0X = 0x41714821;
constexpr uint32_t kComposedPoint1X = 0x41a5403e;
constexpr uint32_t kComposedPoint2X = 0x424a81dc;
constexpr uint32_t kProjectiveScaleX = 0x3f76f912;
constexpr uint32_t kProjectiveSkewX = 0xbd4a22d0;
constexpr uint32_t kProjectiveTranslateX = 0x41274460;
constexpr uint32_t kProjectiveScaleY = 0x3fc823a7;
constexpr uint32_t kProjectiveTranslateY = 0x410131e8;
constexpr uint32_t kProjectivePerspX = 0xbd2a9d40;
constexpr uint32_t kProjectivePoint0X = 0x413ffffe;
constexpr uint32_t kProjectivePoint0Y = 0x407ffffb;
constexpr uint32_t kProjectivePoint1Y = 0x3ffffff3;
constexpr uint32_t kProjectivePoint2X = 0x41b00001;
constexpr uint32_t kProjectivePoint2Y = 0x41900000;
constexpr uint32_t kProjectivePoint3X = 0x41100000;
constexpr uint32_t kProjectivePoint3Y = 0x41700002;
#elif defined(__x86_64__) || defined(_M_X64)
constexpr uint32_t kPerspectiveInversePerspX = 0xb9939a86;
constexpr uint32_t kPerspectiveInversePerspY = 0x3b139a86;
constexpr uint32_t kComposedSkewX = 0xbe866667;
constexpr uint32_t kComposedSkewY = 0x3e400000;
constexpr uint32_t kComposedPerspX = 0x00000000;
constexpr uint32_t kComposedPoint0X = 0x41714820;
constexpr uint32_t kComposedPoint1X = 0x41a5403e;
constexpr uint32_t kComposedPoint2X = 0x424a81dd;
constexpr uint32_t kProjectiveScaleX = 0x3f76f910;
constexpr uint32_t kProjectiveSkewX = 0xbd4a22c8;
constexpr uint32_t kProjectiveTranslateX = 0x41274460;
constexpr uint32_t kProjectiveScaleY = 0x3fc823a8;
constexpr uint32_t kProjectiveTranslateY = 0x410131e9;
constexpr uint32_t kProjectivePerspX = 0xbd2a9d3f;
constexpr uint32_t kProjectivePoint0X = 0x413ffffe;
constexpr uint32_t kProjectivePoint0Y = 0x407fffff;
constexpr uint32_t kProjectivePoint1Y = 0x3ffffffe;
constexpr uint32_t kProjectivePoint2X = 0x41b00000;
constexpr uint32_t kProjectivePoint2Y = 0x41900001;
constexpr uint32_t kProjectivePoint3X = 0x41100000;
constexpr uint32_t kProjectivePoint3Y = 0x41700001;
#else
constexpr uint32_t kPerspectiveInversePerspX = 0xb9939a85;
constexpr uint32_t kPerspectiveInversePerspY = 0x3b139a87;
constexpr uint32_t kComposedSkewXWithoutLto = 0xbe866667;
constexpr uint32_t kComposedSkewXWithLto = 0xbe866666;
constexpr uint32_t kComposedSkewY = 0x3e3fffff;
constexpr uint32_t kComposedPerspX = 0x2e000000;
constexpr uint32_t kComposedPoint0X = 0x41714821;
constexpr uint32_t kComposedPoint1X = 0x41a5403f;
constexpr uint32_t kComposedPoint2X = 0x424a81dd;
constexpr uint32_t kProjectiveScaleX = 0x3f76f910;
constexpr uint32_t kProjectiveSkewXWithoutLto = 0xbd4a22cc;
constexpr uint32_t kProjectiveSkewXWithLto = 0xbd4a22d4;
constexpr uint32_t kProjectiveTranslateX = 0x41274461;
constexpr uint32_t kProjectiveScaleY = 0x3fc823a8;
constexpr uint32_t kProjectiveTranslateY = 0x410131e9;
constexpr uint32_t kProjectivePerspX = 0xbd2a9d3f;
constexpr uint32_t kProjectivePoint0X = 0x41400000;
constexpr uint32_t kProjectivePoint0Y = 0x40800000;
constexpr uint32_t kProjectivePoint1Y = 0x3ffffffe;
constexpr uint32_t kProjectivePoint2X = 0x41b00001;
constexpr uint32_t kProjectivePoint2Y = 0x41900001;
constexpr uint32_t kProjectivePoint3X = 0x41100001;
constexpr uint32_t kProjectivePoint3Y = 0x41700002;
#endif

#if (!defined(__GNUC__) || defined(__clang__)) && \
    !defined(__x86_64__) && !defined(_M_X64)
uint32_t ExpectedComposedSkewX() {
    return inspirecv::GetLibraryInfo().lto_enabled
             ? kComposedSkewXWithLto
             : kComposedSkewXWithoutLto;
}

uint32_t ExpectedProjectiveSkewX() {
    return inspirecv::GetLibraryInfo().lto_enabled
             ? kProjectiveSkewXWithLto
             : kProjectiveSkewXWithoutLto;
}
#endif

uint32_t FloatBits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float FloatFromBits(uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void RequireMatrixBits(const Matrix& matrix, const std::array<uint32_t, 9>& expected) {
    float values[9];
    matrix.get9(values);
    for (size_t i = 0; i < expected.size(); ++i) {
        INFO("matrix element " << i);
        REQUIRE(FloatBits(values[i]) == expected[i]);
    }
}

void RequirePointBits(const Point* points, size_t count, const uint32_t* expected) {
    for (size_t i = 0; i < count; ++i) {
        INFO("point " << i);
        REQUIRE(FloatBits(points[i].fX) == expected[2 * i]);
        REQUIRE(FloatBits(points[i].fY) == expected[2 * i + 1]);
    }
}

uint64_t HashPointBits(const Point* points, int count) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (int index = 0; index < count; ++index) {
        const uint32_t words[] = {FloatBits(points[index].fX), FloatBits(points[index].fY)};
        for (const uint32_t word : words) {
            hash ^= word;
            hash *= UINT64_C(1099511628211);
        }
    }
    return hash;
}

}  // namespace

TEST_CASE("task_affine_composition_allows_either_operand_to_alias",
          "[task][geometry][regression]") {
    Matrix first, second, expected;
    first.setAll(1.25f, -0.375f, 17.5f, 0.625f, 0.75f, -9.25f, 0, 0, 1);
    second.setAll(0.5f, 0.125f, 3, -0.25f, 1.5f, -2, 0, 0, 1);
    expected.setConcat(first, second);
    float values[9];
    expected.get9(values);
    std::array<uint32_t, 9> bits;
    for (size_t i = 0; i < bits.size(); ++i) bits[i] = FloatBits(values[i]);
    Matrix left = first, right = second;
    left.setConcat(left, second);
    right.setConcat(first, right);
    RequireMatrixBits(left, bits);
    RequireMatrixBits(right, bits);
    expected.setConcat(first, first);
    expected.get9(values);
    for (size_t i = 0; i < bits.size(); ++i) bits[i] = FloatBits(values[i]);
    first.setConcat(first, first);
    RequireMatrixBits(first, bits);
}

#if defined(__clang__) && defined(__aarch64__)
TEST_CASE("task_projective_inverse_preserves_unfused_last_minor",
          "[task][geometry][regression][numeric-bit-contract]") {
    // Runtime float bit patterns from the pre-change ARM implementation. The
    // last minor distinguishes two rounded products from a fused multiply-sub.
    const std::array<uint32_t, 9> input = {{
      0x414da124, 0xbfdacf66, 0x409843b0,
      0x401f251c, 0x40e038d1, 0x3fd887d8,
      0x3bc262ab, 0xbb0c3fd5, 0x3f807839}};
    float values[9];
    for (size_t i = 0; i < input.size(); ++i) values[i] = FloatFromBits(input[i]);
    Matrix matrix, inverse;
    matrix.set9(values);
    REQUIRE(matrix.invert(&inverse));
    RequireMatrixBits(inverse, {{
      0x3d988fe8, 0x3c93eb90, 0xbec466c1,
      0xbcd795bb, 0x3e0b88a0, 0xbdd6d7b0,
      0xba022116, 0x3940bf79,
      inspirecv::GetLibraryInfo().lto_enabled ? 0x3f7f965cU : 0x3f7f965eU}});
}
#endif

TEST_CASE("task_matrix_preserves_numeric_contract",
          "[task][geometry][contract][numeric-bit-contract]") {
    SECTION("affine and perspective inversion") {
        Matrix affine;
        affine.setAll(1.25f, -0.375f, 17.5f, 0.625f, 0.75f, -9.25f, 0, 0, 1);
        REQUIRE(affine.getType() == static_cast<Matrix::TypeMask>(7));

        Matrix affine_inverse;
        REQUIRE(affine.invert(&affine_inverse));
        RequireMatrixBits(affine_inverse,
                          {0x3f23d70a, 0x3ea3d70a, 0xc103d70a,
                           0xbf088889, 0x3f888889, 0x4199999a,
                           0x00000000, 0x00000000, 0x3f800000});

        Matrix perspective;
        perspective.setAll(1.1f, 0.2f, 4.0f, -0.3f, 0.9f, 7.0f,
                           0.001f, -0.002f, 1.0f);
        REQUIRE(perspective.getType() == static_cast<Matrix::TypeMask>(15));

        Matrix perspective_inverse;
        REQUIRE(perspective.invert(&perspective_inverse));
        RequireMatrixBits(perspective_inverse,
                          {0x3f5b945f, 0xbe47e140, 0xc00421d3,
                           0x3e9381ec, 0x3f83a6d3, 0xc105a236,
                           kPerspectiveInversePerspX, kPerspectiveInversePerspY, 0x3f7c4093});

        Matrix singular;
        singular.setScale(0.0f, 2.0f);
        REQUIRE_FALSE(singular.invert(nullptr));
    }

    SECTION("composition and mapped points") {
        Matrix affine;
        affine.setAll(1.25f, -0.375f, 17.5f, 0.625f, 0.75f, -9.25f, 0, 0, 1);
        Matrix perspective;
        perspective.setAll(1.1f, 0.2f, 4.0f, -0.3f, 0.9f, 7.0f,
                           0.001f, -0.002f, 1.0f);

        Matrix composed;
        composed.setConcat(perspective, affine);
#if (!defined(__GNUC__) || defined(__clang__)) && \
    !defined(__x86_64__) && !defined(_M_X64)
        const uint32_t composedSkewX = ExpectedComposedSkewX();
#else
        const uint32_t composedSkewX = kComposedSkewX;
#endif
        RequireMatrixBits(composed,
                          {0x3fc00000, composedSkewX, 0x41ab3333,
                           kComposedSkewY, 0x3f499999, 0xc0d26666,
                           kComposedPerspX, 0xbaf5c290, 0x3f849ba6});

        const Point source[] = {{-3.5f, 2.25f}, {0.0f, 0.0f}, {19.75f, -8.5f},
                                {200.0f, 100.0f}, {-0.125f, -0.25f}};
        Point destination[5];
        composed.mapPoints(destination, source, 5);
        const uint32_t expected[] = {
          kComposedPoint0X, 0xc0a9519e, kComposedPoint1X, 0xc0cb16bb, kComposedPoint2X,
          0xc1117e51, 0x43adecb0, 0x430141ec, 0x41a43c4b, 0xc0d1cc84};
        RequirePointBits(destination, 5, expected);
    }

    SECTION("rotation and pre/post composition") {
        Matrix matrix;
        matrix.setRotate(31.25f, 13.0f, -7.0f);
        RequireMatrixBits(matrix,
                          {0x3f5adb81, 0xbf04ce53, 0xbfdf64e8,
                           0x3f04ce53, 0x3f5adb81, 0xc0f84f36,
                           0x00000000, 0x00000000, 0x3f800000});
        matrix.preScale(0.75f, 1.5f);
        matrix.postTranslate(-4.5f, 2.25f);
        RequireMatrixBits(matrix,
                          {0x3f2424a1, 0xbf47357c, 0xc0c7d93a,
                           0x3ec7357c, 0x3fa424a1, 0xc0b04f36,
                           0x00000000, 0x00000000, 0x3f800000});
    }

#if defined(INSPIRECV_TASK_USE_NEON)
    SECTION("two-point affine mapping preserves ARM P0 rounding") {
        Matrix matrix;
        matrix.setAll(FloatFromBits(0x3f5db3d7), FloatFromBits(0xbf000000),
                      FloatFromBits(0x42c24c28), FloatFromBits(0x3f000000),
                      FloatFromBits(0x3f5db3d7), FloatFromBits(0xc22a419b),
                      0.0f, 0.0f, 1.0f);

        constexpr std::array<uint32_t, 3> kRows = {
          {0x41c00000, 0x42f40000, 0x43200000}};  // 24, 122, 160
        constexpr std::array<std::array<uint32_t, 4>, 3> kExpected = {{
          {{0x42aa4c28, 0xc1ae3c55, 0x43996cf6, 0x42d470eb}},
          {{0x42109850, 0x427c5d37, 0x4380ecf6, 0x433f174e}},
          {{0x418930a0, 0x42bfffff, 0x436ed9ec, 0x43600000}},
        }};

        for (size_t index = 0; index < kRows.size(); ++index) {
            const float row = FloatFromBits(kRows[index]);
            const Point source[] = {{0.0f, row}, {256.0f, row}};
            Point output[2];
            matrix.mapPoints(output, source, 2);
            RequirePointBits(output, 2, kExpected[index].data());

            Point in_place[] = {{0.0f, row}, {256.0f, row}};
            matrix.mapPoints(in_place, 2);
            RequirePointBits(in_place, 2, kExpected[index].data());
        }

        const Point threshold_source[] = {
          {0.0f, 24.0f}, {256.0f, 24.0f}, {0.0f, 122.0f},
          {256.0f, 122.0f}, {0.0f, 160.0f}, {256.0f, 160.0f}};
        const uint32_t expected_four[] = {
          0x42aa4c28, 0xc1ae3c55, 0x43996cf6, 0x42d470eb,
          0x42109850, 0x427c5d37, 0x4380ecf6, 0x433f174e};
        const uint32_t expected_five[] = {
          0x42aa4c28, 0xc1ae3c55, 0x43996cf6, 0x42d470ea,
          0x42109850, 0x427c5d37, 0x4380ecf6, 0x433f174d,
          0x418930a0, 0x42bfffff};
        const uint32_t expected_six[] = {
          0x42aa4c28, 0xc1ae3c55, 0x43996cf6, 0x42d470ea,
          0x42109850, 0x427c5d37, 0x4380ecf6, 0x433f174d,
          0x418930a0, 0x42bfffff, 0x436ed9ec, 0x43600000};

        Point threshold_output[6];
        matrix.mapPoints(threshold_output, threshold_source, 4);
        RequirePointBits(threshold_output, 4, expected_four);
        matrix.mapPoints(threshold_output, threshold_source, 5);
        RequirePointBits(threshold_output, 5, expected_five);
        matrix.mapPoints(threshold_output, threshold_source, 6);
        RequirePointBits(threshold_output, 6, expected_six);
    }
#endif

    SECTION("rectangle normalization and fitting") {
        Matrix map;
        map.setScaleTranslate(-2.0f, 0.5f, 3.0f, -4.0f);
        Rect output;
        REQUIRE(map.mapRect(&output, Rect::MakeLTRB(7.0f, -3.0f, -2.0f, 11.0f)));
        REQUIRE(FloatBits(output.fLeft) == 0xc1300000);
        REQUIRE(FloatBits(output.fTop) == 0xc0b00000);
        REQUIRE(FloatBits(output.fRight) == 0x40e00000);
        REQUIRE(FloatBits(output.fBottom) == 0x3fc00000);

        const Rect source = Rect::MakeLTRB(10.0f, 20.0f, 110.0f, 70.0f);
        const Rect destination = Rect::MakeLTRB(-40.0f, 5.0f, 260.0f, 405.0f);
        const std::array<std::array<uint32_t, 9>, 4> expected = {{
          {{0x40400000, 0x00000000, 0xc28c0000, 0x00000000, 0x41000000,
            0xc31b0000, 0x00000000, 0x00000000, 0x3f800000}},
          {{0x40400000, 0x00000000, 0xc28c0000, 0x00000000, 0x40400000,
            0xc25c0000, 0x00000000, 0x00000000, 0x3f800000}},
          {{0x40400000, 0x00000000, 0xc28c0000, 0x00000000, 0x40400000,
            0x428c0000, 0x00000000, 0x00000000, 0x3f800000}},
          {{0x40400000, 0x00000000, 0xc28c0000, 0x00000000, 0x40400000,
            0x43430000, 0x00000000, 0x00000000, 0x3f800000}},
        }};
        for (int mode = Matrix::kFill_ScaleToFit; mode <= Matrix::kEnd_ScaleToFit; ++mode) {
            Matrix fitted;
            REQUIRE(fitted.setRectToRect(source, destination,
                                         static_cast<Matrix::ScaleToFit>(mode)));
            RequireMatrixBits(fitted, expected[static_cast<size_t>(mode)]);
        }
    }

    SECTION("three and four point projective fitting") {
        const Point source4[] = {{1.5f, -2.0f}, {8.0f, -1.0f},
                                 {7.0f, 6.5f}, {-0.5f, 5.0f}};
        const Point destination4[] = {{12.0f, 4.0f}, {25.0f, 2.0f},
                                      {22.0f, 18.0f}, {9.0f, 15.0f}};
        Matrix projective;
        REQUIRE(projective.setPolyToPoly(source4, destination4, 4));
#if (!defined(__GNUC__) || defined(__clang__)) && \
    !defined(__x86_64__) && !defined(_M_X64)
        const uint32_t projectiveSkewX = ExpectedProjectiveSkewX();
#else
        const uint32_t projectiveSkewX = kProjectiveSkewX;
#endif
        RequireMatrixBits(projective,
                          {kProjectiveScaleX, projectiveSkewX, kProjectiveTranslateX,
                           0xbf21b54d, kProjectiveScaleY, kProjectiveTranslateY,
                           kProjectivePerspX, 0xb9cd7e80, 0x3f87e5af});

        Point mapped[4];
        projective.mapPoints(mapped, source4, 4);
        const uint32_t expected4[] = {kProjectivePoint0X, kProjectivePoint0Y,
                                      0x41c80000, kProjectivePoint1Y,
                                      kProjectivePoint2X, kProjectivePoint2Y,
                                      kProjectivePoint3X, kProjectivePoint3Y};
        RequirePointBits(mapped, 4, expected4);

        const Point source3[] = {{-2.0f, 1.0f}, {3.5f, -4.0f}, {7.0f, 6.0f}};
        const Point destination3[] = {{10.0f, 20.0f}, {13.0f, 5.0f}, {30.0f, 18.0f}};
        Matrix affine;
        REQUIRE(affine.setPolyToPoly(source3, destination3, 3));
        RequireMatrixBits(affine,
                          {0x3fcb08d4, 0x3f9289b6, 0x414070fe,
                           0xbf9611a8, 0x3fdaec94, 0x417f1e04,
                           0x00000000, 0x00000000, 0x3f800000});
    }

    SECTION("poly fitting boundaries and failures preserve state") {
        Matrix zero_points;
        zero_points.setTranslate(3.0f, -7.0f);
        REQUIRE(zero_points.setPolyToPoly(nullptr, nullptr, 0));
        REQUIRE(zero_points.isIdentity());

        const Point source1[] = {{-3.25f, 7.5f}};
        const Point destination1[] = {{20.5f, -11.25f}};
        Matrix one_point;
        REQUIRE(one_point.setPolyToPoly(source1, destination1, 1));
        RequireMatrixBits(one_point,
                          {0x3f800000, 0x00000000, 0x41be0000,
                           0x00000000, 0x3f800000, 0xc1960000,
                           0x00000000, 0x00000000, 0x3f800000});

        const Point source2[] = {{-3.25f, 7.5f}, {8.75f, -2.125f}};
        const Point destination2[] = {{20.5f, -11.25f}, {-4.0f, 18.75f}};
        Matrix two_points;
        REQUIRE(two_points.setPolyToPoly(source2, destination2, 2));
        RequireMatrixBits(two_points,
                          {0xc01d9b28, 0xbf0658e1, 0x418375cc,
                           0x3f0658e1, 0xc01d9b28, 0x410eccf8,
                           0x00000000, 0x00000000, 0x3f800000});

        Matrix unchanged;
        unchanged.setAll(1.25f, -0.375f, 17.5f,
                         0.625f, 0.75f, -9.25f, 0.0f, 0.0f, 1.0f);
        const Matrix frozen = unchanged;
        const Point repeated2[] = {{4.0f, 9.0f}, {4.0f, 9.0f}};
        REQUIRE_FALSE(unchanged.setPolyToPoly(repeated2, destination2, 2));
        REQUIRE(unchanged.cheapEqualTo(frozen));

        const Point collinear3[] = {{0.0f, 0.0f}, {0.0f, 0.0f}, {5.0f, 0.0f}};
        const Point target3[] = {{1.0f, 2.0f}, {7.0f, -3.0f}, {4.0f, 8.0f}};
        REQUIRE_FALSE(unchanged.setPolyToPoly(collinear3, target3, 3));
        REQUIRE(unchanged.cheapEqualTo(frozen));

        const Point repeated4[] = {
          {0.0f, 0.0f}, {1.0f, 1.0f}, {2.0f, 2.0f}, {1.0f, 1.0f}};
        const Point target4[] = {
          {0.0f, 0.0f}, {5.0f, 0.0f}, {5.0f, 5.0f}, {0.0f, 5.0f}};
        REQUIRE_FALSE(unchanged.setPolyToPoly(repeated4, target4, 4));
        REQUIRE(unchanged.cheapEqualTo(frozen));

        Point source5[5] = {};
        Point destination5[5] = {};
        REQUIRE_FALSE(unchanged.setPolyToPoly(source5, destination5, 5));
        REQUIRE(unchanged.cheapEqualTo(frozen));
    }

    SECTION("batch mapping boundaries preserve frozen bit patterns") {
        constexpr std::array<int, 9> counts = {{0, 1, 2, 3, 4, 5, 8, 17, 256}};
        const std::array<std::array<uint64_t, 9>, 3> expected = {{
          {{UINT64_C(0x14650fb0739d0383), UINT64_C(0xb029c808263d78fb),
            UINT64_C(0x2847be6ef9e0f133), UINT64_C(0x8aea72357ecbda2b),
            UINT64_C(0x87ab94baabfd91e3), UINT64_C(0x479a67731149a65b),
            UINT64_C(0x4e340aea4a816c43), UINT64_C(0x4373dd9d50fa367b),
            UINT64_C(0xa4257f5f92509b83)}},
          {{UINT64_C(0x14650fb0739d0383), UINT64_C(0x1bb9c678444688fb),
            UINT64_C(0x6633b1135b2c8133), UINT64_C(0x4d84d21edb5d6a2b),
            UINT64_C(0xdde0af6baff181e3), UINT64_C(0x1c090d1d4e87d65b),
            UINT64_C(0xa9095ab5a1691c43), UINT64_C(0xb2fd3566ff5b267b),
            UINT64_C(0x02ea7e6858299b83)}},
          {{UINT64_C(0x14650fb0739d0383), UINT64_C(0xcb49c8b3347ce8fb),
            UINT64_C(0xf5743366ccac6133), UINT64_C(0xf25097fe1a974a2b),
            UINT64_C(0x9ce91f5fbd4691e3), UINT64_C(0x10718cfddb47865b),
            UINT64_C(0x3203a045175cd043), UINT64_C(0xc848eac96e26ea7b),
            UINT64_C(0x81d2ff7e1f78d783)}},
        }};

        std::vector<Point> source(256);
        for (int index = 0; index < 256; ++index) {
            source[index].fX = static_cast<float>((index * 37) % 101 - 50) * 0.375f +
                               (index % 3 == 0 ? -0.0625f : 0.125f);
            source[index].fY = static_cast<float>((index * 53) % 89 - 44) * -0.3125f +
                               (index % 5 == 0 ? 0.03125f : -0.1875f);
        }

        std::array<Matrix, 3> transforms;
        transforms[0].setTranslate(-3.75f, 8.125f);
        transforms[1].setScaleTranslate(1.25f, -0.75f, 2.5f, -6.25f);
        transforms[2].setAll(1.25f, -0.375f, 17.5f,
                             0.625f, 0.75f, -9.25f, 0.0f, 0.0f, 1.0f);

        for (size_t transform_index = 0; transform_index < transforms.size();
             ++transform_index) {
            for (size_t case_index = 0; case_index < counts.size(); ++case_index) {
                const int count = counts[case_index];
                INFO("transform " << transform_index << ", point count " << count);
                std::vector<Point> output(static_cast<size_t>(count));
                transforms[transform_index].mapPoints(output.data(), source.data(), count);
                REQUIRE(HashPointBits(output.data(), count) ==
                        expected[transform_index][case_index]);

                std::vector<Point> in_place(source.begin(), source.begin() + count);
                transforms[transform_index].mapPoints(in_place.data(), count);
                REQUIRE(HashPointBits(in_place.data(), count) ==
                        expected[transform_index][case_index]);
            }
        }
    }
}

TEST_CASE("task_rect_preserves_numeric_and_edge_contract", "[task][geometry][contract]") {
    SECTION("construction and mutation retain scalar bits") {
        Rect rectangle = Rect::MakeXYWH(-3.5f, 2.25f, 11.75f, 9.5f);
        REQUIRE(FloatBits(rectangle.fLeft) == UINT32_C(0xc0600000));
        REQUIRE(FloatBits(rectangle.fTop) == UINT32_C(0x40100000));
        REQUIRE(FloatBits(rectangle.fRight) == UINT32_C(0x41040000));
        REQUIRE(FloatBits(rectangle.fBottom) == UINT32_C(0x413c0000));
        REQUIRE(FloatBits(rectangle.centerX()) == UINT32_C(0x40180000));
        REQUIRE(FloatBits(rectangle.centerY()) == UINT32_C(0x40e00000));

        rectangle.offsetTo(5.25f, -7.5f);
        rectangle.inset(0.5f, 1.25f);
        rectangle.outset(0.25f, 0.75f);
        REQUIRE(FloatBits(rectangle.fLeft) == UINT32_C(0x40b00000));
        REQUIRE(FloatBits(rectangle.fTop) == UINT32_C(0xc0e00000));
        REQUIRE(FloatBits(rectangle.fRight) == UINT32_C(0x41860000));
        REQUIRE(FloatBits(rectangle.fBottom) == UINT32_C(0x3fc00000));
    }

    SECTION("empty, containment, intersection, and union semantics") {
        const Rect bounds = Rect::MakeLTRB(-4.0f, 2.0f, 8.0f, 11.0f);
        REQUIRE_FALSE(bounds.isEmpty());
        REQUIRE(bounds.isSorted());
        REQUIRE(bounds.contains(-4.0f, 2.0f));
        REQUIRE_FALSE(bounds.contains(8.0f, 11.0f));
        REQUIRE(bounds.intersects(Rect::MakeLTRB(7.5f, 10.5f, 12.0f, 15.0f)));
        REQUIRE_FALSE(bounds.intersects(Rect::MakeLTRB(8.0f, 3.0f, 9.0f, 6.0f)));

        Rect joined = Rect::MakeEmpty();
        joined.joinNonEmptyArg(bounds);
        joined.joinPossiblyEmptyRect(Rect::MakeLTRB(-9.0f, 5.0f, 3.0f, 20.0f));
        REQUIRE(joined.fLeft == -9.0f);
        REQUIRE(joined.fTop == 2.0f);
        REQUIRE(joined.fRight == 8.0f);
        REQUIRE(joined.fBottom == 20.0f);
    }

    SECTION("signed zero and NaN preserve legacy edge behavior") {
        Rect zeros = Rect::MakeLTRB(-0.0f, 0.0f, 0.0f, -0.0f);
        const Rect sorted = zeros.makeSorted();
        REQUIRE(FloatBits(sorted.fLeft) == UINT32_C(0x80000000));
        REQUIRE(FloatBits(sorted.fTop) == UINT32_C(0x00000000));
        REQUIRE(FloatBits(sorted.fRight) == UINT32_C(0x80000000));
        REQUIRE(FloatBits(sorted.fBottom) == UINT32_C(0x00000000));
        zeros.sort();
        REQUIRE(FloatBits(zeros.fRight) == UINT32_C(0x00000000));
        REQUIRE(FloatBits(zeros.fBottom) == UINT32_C(0x80000000));

        const float nan = std::numeric_limits<float>::quiet_NaN();
        const Rect finite = Rect::MakeLTRB(0.0f, 0.0f, 4.0f, 4.0f);
        REQUIRE_FALSE(Rect::MakeLTRB(nan, 0.0f, 2.0f, 2.0f).intersects(finite));
        REQUIRE(finite.intersects(Rect::MakeLTRB(nan, 1.0f, 3.0f, 3.0f)));
    }
}
