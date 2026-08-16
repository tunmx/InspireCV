#ifndef INSPIRECV_TASK_KERNELS_CPU_YUV_OPS_H_
#define INSPIRECV_TASK_KERNELS_CPU_YUV_OPS_H_

#include <cstddef>
#include <cstdint>

namespace inspirecv {
namespace task {
namespace kernels {
namespace yuv {

using Converter = void(const uint8_t*, uint8_t*, size_t);

Converter ToRgb;
Converter ToBgr;
Converter ToRgba;
Converter ToBgra;

}  // namespace yuv
}  // namespace kernels
}  // namespace task
}  // namespace inspirecv

#endif  // INSPIRECV_TASK_KERNELS_CPU_YUV_OPS_H_
