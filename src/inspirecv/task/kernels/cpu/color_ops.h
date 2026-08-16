#ifndef INSPIRECV_TASK_KERNELS_CPU_COLOR_OPS_H_
#define INSPIRECV_TASK_KERNELS_CPU_COLOR_OPS_H_

#include <cstddef>
#include <cstdint>

namespace inspirecv {
namespace task {
namespace kernels {
namespace color {

using PixelTransform = void(const uint8_t*, uint8_t*, size_t);

PixelTransform RgbToYCrCb;
PixelTransform BgrToYCrCb;
PixelTransform RgbToYuv;
PixelTransform BgrToYuv;
PixelTransform RgbToXyz;
PixelTransform BgrToXyz;
PixelTransform RgbToHsv;
PixelTransform BgrToHsv;
PixelTransform RgbToHsvFull;
PixelTransform BgrToHsvFull;
PixelTransform RgbToBgr555;
PixelTransform BgrToBgr555;
PixelTransform RgbToBgr565;
PixelTransform BgrToBgr565;

}  // namespace color
}  // namespace kernels
}  // namespace task
}  // namespace inspirecv

#endif  // INSPIRECV_TASK_KERNELS_CPU_COLOR_OPS_H_
