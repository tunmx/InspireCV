#ifndef INSPIRECV_TASK_KERNELS_CPU_TENSOR_WRITERS_H_
#define INSPIRECV_TASK_KERNELS_CPU_TENSOR_WRITERS_H_

#include <cstddef>
#include <cstdint>

namespace inspirecv {
namespace task {
namespace kernels {
namespace tensor {

using InterleavedWriter =
  void(const uint8_t*, float*, const float*, const float*, size_t);
using PlanarWriter =
  void(const uint8_t*, float*, size_t, const float*, const float*, size_t);

InterleavedWriter InterleavedMono;
InterleavedWriter InterleavedTriple;
PlanarWriter PlanarTriple;
InterleavedWriter InterleavedQuad;
InterleavedWriter QuadFromMono;
InterleavedWriter QuadFromTriple;

}  // namespace tensor
}  // namespace kernels
}  // namespace task
}  // namespace inspirecv

#endif  // INSPIRECV_TASK_KERNELS_CPU_TENSOR_WRITERS_H_
