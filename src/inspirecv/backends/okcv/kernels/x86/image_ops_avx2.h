#ifndef INSPIRECV_BACKENDS_OKCV_KERNELS_X86_IMAGE_OPS_AVX2_H_
#define INSPIRECV_BACKENDS_OKCV_KERNELS_X86_IMAGE_OPS_AVX2_H_

#include <cstddef>
#include <cstdint>

namespace okcv {
namespace x86 {

// These entry points live in an isolated AVX2 translation unit. The caller
// must check both INSPIRECV_HAVE_IMAGE_AVX2_KERNELS and cpu::HasAvx2().
void GrayU8Avx2(const uint8_t* source, uint8_t* destination, std::size_t pixels);
void GrayF32Avx2(const float* source, float* destination, std::size_t pixels);
void Mean3U8Avx2(const uint8_t* source, uint8_t* destination, std::size_t pixels);
void Mean3F32Avx2(const float* source, float* destination, std::size_t pixels);
void ThresholdU8Avx2(const uint8_t* source, uint8_t* destination, std::size_t count,
                     uint8_t threshold, uint8_t maximum);
void ThresholdF32Avx2(const float* source, float* destination, std::size_t count,
                      float threshold, float maximum);
void AbsDiffU8Avx2(const uint8_t* a, const uint8_t* b, uint8_t* destination,
                   std::size_t count);
void AbsDiffF32Avx2(const float* a, const float* b, float* destination,
                    std::size_t count);
void BlendU8Avx2(const uint8_t* a, const uint8_t* b, const uint8_t* mask,
                 uint8_t* destination, std::size_t pixels, int channels);
void BlendF32Avx2(const float* a, const float* b, const uint8_t* mask,
                  float* destination, std::size_t pixels, int channels);
void MorphologyU8Avx2(const uint8_t* source, uint8_t* destination, int width,
                      int height, int left, int right, int top, int bottom,
                      bool erode, uint8_t* scratch);
// scratch is width * height writable bytes for non-3x3 windows, and may be null
// when left == right == top == bottom == 1. Allocation belongs to baseline code.
void FlipU8Avx2(const uint8_t* source, uint8_t* destination, int width, int height,
                int channels, bool reverse_rows);
void FlipF32Avx2(const float* source, float* destination, int width, int height,
                 int channels, bool reverse_rows);
void SwapRbU8C4Avx2(const uint8_t* source, uint8_t* destination, std::size_t pixels);
void SwapRbF32Avx2(const float* source, float* destination, std::size_t pixels, int channels);

void ConvertU8ToF32Avx2(const uint8_t* source, float* destination, std::size_t count);
void GaussianHorizontalU8Avx2(const uint8_t* source, float* destination, int width, int channels,
                               const float* kernel, int size);
void GaussianHorizontalF32Avx2(const float* source, float* destination, int width, int channels,
                                const float* kernel, int size);
void GaussianVerticalU8Avx2(const float* source, uint8_t* destination, int width, int height, int channels,
                             int y, const float* kernel, int size);
void GaussianVerticalF32Avx2(const float* source, float* destination, int width, int height, int channels,
                              int y, const float* kernel, int size);

void ResizeBilinearU8RowAvx2(const uint8_t* top, const uint8_t* bottom, uint8_t* destination,
                            int source_width, int destination_width, int channels,
                            const int* left, const int* right, const float* weight_x, float weight_y);

// Private arithmetic entry points; runtime AVX2 check required.
void MulF32Avx2(const float* source, float* destination, std::size_t count, float a);
void MulAddF32Avx2(const float* source, float* destination, std::size_t count, float a, float b);

}  // namespace x86
}  // namespace okcv

#endif  // INSPIRECV_BACKENDS_OKCV_KERNELS_X86_IMAGE_OPS_AVX2_H_
