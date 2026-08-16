#ifndef INSPIRECV_TASK_KERNELS_CPU_SAMPLING_OPS_H_
#define INSPIRECV_TASK_KERNELS_CPU_SAMPLING_OPS_H_

#include <cstddef>
#include <cstdint>

#include <inspirecv/task/core/matrix.h>

namespace inspirecv {
namespace task {
namespace kernels {
namespace sampling {

using Point = ::inspirecv::task::Point;
using Sampler = void(const uint8_t*, uint8_t*, Point*, size_t, size_t, size_t,
                     size_t, size_t, size_t);

Sampler DirectMono;
Sampler DirectTriple;
Sampler DirectQuad;
Sampler NearestMono;
Sampler NearestTriple;
Sampler NearestQuad;
Sampler BilinearMono;
Sampler BilinearTriple;
Sampler BilinearQuad;
Sampler DirectNv21;
Sampler NearestNv21;
Sampler DirectNv12;
Sampler NearestNv12;
Sampler DirectI420;
Sampler NearestI420;

}  // namespace sampling
}  // namespace kernels
}  // namespace task
}  // namespace inspirecv

#endif  // INSPIRECV_TASK_KERNELS_CPU_SAMPLING_OPS_H_
