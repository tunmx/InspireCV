#ifndef INSPIRECV_TASK_KERNELS_X86_AVX2_KERNELS_H_
#define INSPIRECV_TASK_KERNELS_X86_AVX2_KERNELS_H_

#include <cstddef>
#include <cstdint>

#include <inspirecv/task/core/matrix.h>

namespace inspirecv {
namespace task {
namespace kernels {
namespace x86 {

// These entry points live in AVX2-only translation units. Call only after
// platform::HasAvx2() has checked both the CPU and operating system support.
void InterleavedMonoAvx2(const uint8_t*, float*, const float*, const float*, size_t);
void InterleavedTripleAvx2(const uint8_t*, float*, const float*, const float*, size_t);
void InterleavedQuadAvx2(const uint8_t*, float*, const float*, const float*, size_t);
void PlanarTripleAvx2(const uint8_t*, float*, size_t, const float*, const float*, size_t);
void QuadFromMonoAvx2(const uint8_t*, float*, const float*, const float*, size_t);
void QuadFromTripleAvx2(const uint8_t*, float*, const float*, const float*, size_t);

void ReplicateMonoToTripleAvx2(const uint8_t*, uint8_t*, size_t);
void ReplicateMonoToQuadAvx2(const uint8_t*, uint8_t*, size_t);
void AppendOpaqueAlphaAvx2(const uint8_t*, uint8_t*, size_t);
void ReverseTripleAvx2(const uint8_t*, uint8_t*, size_t);
void ReverseQuadColorAvx2(const uint8_t*, uint8_t*, size_t);
void DropAlphaAvx2(const uint8_t*, uint8_t*, size_t);
void ReverseAndDropAlphaAvx2(const uint8_t*, uint8_t*, size_t);
void LumaFromRgbAvx2(const uint8_t*, uint8_t*, size_t);
void LumaFromBgrAvx2(const uint8_t*, uint8_t*, size_t);
void LumaFromRgbaAvx2(const uint8_t*, uint8_t*, size_t);
void LumaFromBgraAvx2(const uint8_t*, uint8_t*, size_t);

size_t ColorMatrixAvx2(const uint8_t*, uint8_t*, size_t, const int (*)[3],
                       int, const int*, bool, bool);
size_t ColorHsvAvx2(const uint8_t*, uint8_t*, size_t, int, bool);
size_t PackBgr16Avx2(const uint8_t*, uint8_t*, size_t, bool, bool);

using Avx2Sampler = void(const uint8_t*, uint8_t*, Point*, size_t, size_t, size_t,
                          size_t, size_t, size_t);
Avx2Sampler NearestMonoAvx2;
Avx2Sampler NearestTripleAvx2;
Avx2Sampler NearestQuadAvx2;
Avx2Sampler BilinearMonoAvx2;
Avx2Sampler BilinearTripleAvx2;
Avx2Sampler BilinearQuadAvx2;

}  // namespace x86
}  // namespace kernels
}  // namespace task
}  // namespace inspirecv

#endif  // INSPIRECV_TASK_KERNELS_X86_AVX2_KERNELS_H_
