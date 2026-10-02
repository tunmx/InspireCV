#include "inspirecv/task/kernels/cpu/color_ops.h"

#include <algorithm>
#include <cstring>

#include "inspirecv/task/platform/cpu_features.h"
#if defined(INSPIRECV_TASK_HAVE_AVX2_KERNELS)
#include "inspirecv/task/kernels/x86/avx2_kernels.h"
#endif

namespace inspirecv {
namespace task {
namespace kernels {
namespace color {
namespace {

enum class PackedOrder { kRgb, kBgr };

struct RgbPixel {
    int red;
    int green;
    int blue;
};

struct FixedMatrix {
    int coefficient[3][3];
    int shift;
    int offset[3];
    bool clamp;
};

constexpr FixedMatrix kYCrCbMatrix = {
  {{4899, 9617, 1868}, {8192, -6860, -1332}, {-2765, -5427, 8192}},
  14,
  {0, 128, 128},
  false};

constexpr FixedMatrix kYuvMatrix = {
  {{4899, 9617, 1868}, {-2412, -4734, 7146}, {10076, -8438, -1638}},
  14,
  {0, 128, 128},
  false};

constexpr FixedMatrix kXyzMatrix = {
  {{1689, 1465, 739}, {871, 2929, 296}, {79, 488, 3892}},
  12,
  {0, 0, 0},
  true};

template <PackedOrder kOrder>
RgbPixel LoadRgb(const uint8_t* source) {
    return kOrder == PackedOrder::kRgb
             ? RgbPixel{source[0], source[1], source[2]}
             : RgbPixel{source[2], source[1], source[0]};
}

uint8_t ClampByte(int value) {
    if (static_cast<unsigned>(value) <= 255u) return static_cast<uint8_t>(value);
    return value > 0 ? UINT8_C(255) : UINT8_C(0);
}

int RoundedShift(int value, int shift) {
    return (value + (1 << (shift - 1))) >> shift;
}

template <PackedOrder kOrder>
void ApplyMatrix(const uint8_t* source, uint8_t* destination, size_t count,
                 const FixedMatrix& matrix) {
#if defined(INSPIRECV_TASK_HAVE_AVX2_KERNELS)
    if (count >= 8 && platform::HasAvx2()) {
        const size_t completed = x86::ColorMatrixAvx2(source, destination, count, matrix.coefficient, matrix.shift, matrix.offset, matrix.clamp, kOrder == PackedOrder::kBgr);
        source += 3 * completed;
        destination += 3 * completed;
        count -= completed;
    }
#endif
    while (count-- != 0) {
        const RgbPixel pixel = LoadRgb<kOrder>(source);
        const int input[3] = {pixel.red, pixel.green, pixel.blue};
        for (int row = 0; row < 3; ++row) {
            const int sum = input[0] * matrix.coefficient[row][0] +
                            input[1] * matrix.coefficient[row][1] +
                            input[2] * matrix.coefficient[row][2];
            const int value = RoundedShift(sum, matrix.shift) + matrix.offset[row];
            destination[row] = matrix.clamp ? ClampByte(value)
                                            : static_cast<uint8_t>(value);
        }
        source += 3;
        destination += 3;
    }
}

template <PackedOrder kOrder, int kHueRange>
void ConvertHsv(const uint8_t* source, uint8_t* destination, size_t count) {
#if defined(INSPIRECV_TASK_HAVE_AVX2_KERNELS) && defined(INSPIRECV_TASK_MSVC_FAST_HSV) && INSPIRECV_TASK_MSVC_FAST_HSV
    // MSVC's optimized scalar expression uses a direct quotient, matched by
    // this AVX2 kernel. GCC/Clang retain their historical auto-vectorized
    // reciprocal arithmetic; precise MSVC Debug retains its scalar formula.
    if (count >= 8 && platform::HasAvx2()) {
        const size_t completed = x86::ColorHsvAvx2(source, destination, count, kHueRange, kOrder == PackedOrder::kBgr);
        source += 3 * completed;
        destination += 3 * completed;
        count -= completed;
    }
#endif
    while (count-- != 0) {
        const RgbPixel pixel = LoadRgb<kOrder>(source);
        const int minimum = std::min(pixel.red, std::min(pixel.green, pixel.blue));
        const int maximum = std::max(pixel.red, std::max(pixel.green, pixel.blue));
        // Byte-channel extrema guarantee 0 <= difference <= 255. Keep the
        // subtraction in int to avoid redundant narrowing in vectorized loops.
        const int difference = maximum - minimum;
        const int redMask = maximum == pixel.red ? -1 : 0;
        const int greenMask = maximum == pixel.green ? -1 : 0;

        // Safe denominators give H=S=0 for achromatic colors. Keeping this
        // branchless preserves the existing compiler vectorization and its
        // observable rounding for non-achromatic input.
        const int saturation =
          (static_cast<int>(difference * (255 << 12) *
                            (1.0f / static_cast<float>(std::max(maximum, 1)))) +
           (1 << 11)) >>
          12;
        int hue =
          (redMask & (pixel.green - pixel.blue)) +
          (~redMask &
           ((greenMask & (pixel.blue - pixel.red + 2 * difference)) +
            (~greenMask & (pixel.red - pixel.green + 4 * difference))));
        hue = (hue * static_cast<int>((kHueRange << 12) /
                                      (6.0f * std::max<int>(difference, 1)) + 0.5f) +
               (1 << 11)) >>
              12;
        if (hue < 0) hue += kHueRange;

        destination[0] = ClampByte(hue);
        destination[1] = static_cast<uint8_t>(saturation);
        destination[2] = static_cast<uint8_t>(maximum);
        source += 3;
        destination += 3;
    }
}

template <PackedOrder kOrder, bool kGreenHasSixBits>
void PackBgr16(const uint8_t* source, uint8_t* destination, size_t count) {
#if defined(INSPIRECV_TASK_HAVE_AVX2_KERNELS)
    if (count >= 8 && platform::HasAvx2()) {
        const size_t completed = x86::PackBgr16Avx2(source, destination, count, kGreenHasSixBits, kOrder == PackedOrder::kBgr);
        source += 3 * completed;
        destination += 2 * completed;
        count -= completed;
    }
#endif
    while (count-- != 0) {
        const RgbPixel pixel = LoadRgb<kOrder>(source);
        const uint16_t packed = kGreenHasSixBits
                                  ? static_cast<uint16_t>(
                                      (pixel.blue >> 3) |
                                      ((pixel.green & ~3) << 3) |
                                      ((pixel.red & ~7) << 8))
                                  : static_cast<uint16_t>(
                                      (pixel.blue >> 3) |
                                      ((pixel.green & ~7) << 2) |
                                      ((pixel.red & ~7) << 7));
        std::memcpy(destination, &packed, sizeof(packed));
        source += 3;
        destination += sizeof(packed);
    }
}

}  // namespace

void RgbToYCrCb(const uint8_t* source, uint8_t* destination, size_t count) {
    ApplyMatrix<PackedOrder::kRgb>(source, destination, count, kYCrCbMatrix);
}

void BgrToYCrCb(const uint8_t* source, uint8_t* destination, size_t count) {
    ApplyMatrix<PackedOrder::kBgr>(source, destination, count, kYCrCbMatrix);
}

void RgbToYuv(const uint8_t* source, uint8_t* destination, size_t count) {
    ApplyMatrix<PackedOrder::kRgb>(source, destination, count, kYuvMatrix);
}

void BgrToYuv(const uint8_t* source, uint8_t* destination, size_t count) {
    ApplyMatrix<PackedOrder::kBgr>(source, destination, count, kYuvMatrix);
}

void RgbToXyz(const uint8_t* source, uint8_t* destination, size_t count) {
    ApplyMatrix<PackedOrder::kRgb>(source, destination, count, kXyzMatrix);
}

void BgrToXyz(const uint8_t* source, uint8_t* destination, size_t count) {
    ApplyMatrix<PackedOrder::kBgr>(source, destination, count, kXyzMatrix);
}

void RgbToHsv(const uint8_t* source, uint8_t* destination, size_t count) {
    ConvertHsv<PackedOrder::kRgb, 180>(source, destination, count);
}

void BgrToHsv(const uint8_t* source, uint8_t* destination, size_t count) {
    ConvertHsv<PackedOrder::kBgr, 180>(source, destination, count);
}

void RgbToHsvFull(const uint8_t* source, uint8_t* destination, size_t count) {
    ConvertHsv<PackedOrder::kRgb, 256>(source, destination, count);
}

void BgrToHsvFull(const uint8_t* source, uint8_t* destination, size_t count) {
    ConvertHsv<PackedOrder::kBgr, 256>(source, destination, count);
}

void RgbToBgr555(const uint8_t* source, uint8_t* destination, size_t count) {
    PackBgr16<PackedOrder::kRgb, false>(source, destination, count);
}

void BgrToBgr555(const uint8_t* source, uint8_t* destination, size_t count) {
    PackBgr16<PackedOrder::kBgr, false>(source, destination, count);
}

void RgbToBgr565(const uint8_t* source, uint8_t* destination, size_t count) {
    PackBgr16<PackedOrder::kRgb, true>(source, destination, count);
}

void BgrToBgr565(const uint8_t* source, uint8_t* destination, size_t count) {
    PackBgr16<PackedOrder::kBgr, true>(source, destination, count);
}

}  // namespace color
}  // namespace kernels
}  // namespace task
}  // namespace inspirecv
