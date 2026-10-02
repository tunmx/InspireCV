#ifndef INSPIRECV_BACKENDS_OKCV_KERNELS_X86_IMAGE_AFFINE_AVX2_H_
#define INSPIRECV_BACKENDS_OKCV_KERNELS_X86_IMAGE_AFFINE_AVX2_H_

#include <cstdint>

namespace okcv {
namespace x86 {

// Isolated AVX2 implementation of Bitmap<uint8_t>::AffineBilinearReference.
// The caller must first check cpu::HasAvx2(). Source/destination are tightly
// packed, disjoint C1/C3/C4 buffers. Matrix is destination-to-source, row-major
// [a,b,tx,c,d,ty]. Constant border applies to the whole pixel when its source
// coordinate is outside the image, matching Reference (not the distinct
// per-tap border contract of AffineBilinearOptimized's axis/C1/C4 paths).
// Returns false WITHOUT modifying destination for unsupported dimensions,
// nonfinite/overflowing coordinate transforms or oversized gather offsets.
// This file must compile without fast-math, reassociation or FMA contraction.
bool AffineBilinearReferenceU8Avx2(const uint8_t* source, int source_width,
                                  int source_height, uint8_t* destination,
                                  int width, int height, int channels,
                                  const float matrix[6], bool replicate_border,
                                  uint8_t border_value);

}  // namespace x86
}  // namespace okcv
#endif
