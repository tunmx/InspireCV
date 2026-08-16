#ifndef INSPIRECV_CORE_RUNTIME_IMAGE_ACCELERATION_H_
#define INSPIRECV_CORE_RUNTIME_IMAGE_ACCELERATION_H_

#include "inspirecv/core/cuda/image_geometry_dispatch.h"

namespace inspirecv {
namespace internal {

// Returns true only when CUDA produced the destination. Unsupported requests,
// disabled acceleration and runtime failures all leave the caller free to use
// its original CPU implementation.
bool TryExecuteAcceleratedImageGeometry(
  const ImageGeometryRequest& request) noexcept;

bool ImageCudaAutoRecommends(
  const ImageGeometryRequest& request) noexcept;

}  // namespace internal
}  // namespace inspirecv

#endif  // INSPIRECV_CORE_RUNTIME_IMAGE_ACCELERATION_H_
