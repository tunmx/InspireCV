#include "inspirecv/core/runtime/image_acceleration.h"

#include <inspirecv/acceleration.h>

#include "inspirecv/core/runtime/acceleration_state.h"

namespace inspirecv {
namespace internal {
namespace {

size_t PixelCount(int width, int height) noexcept {
    return static_cast<size_t>(width) * static_cast<size_t>(height);
}

bool IsMeasuredThreeQuarterResize(
  const ImageGeometryRequest& request) noexcept {
    return static_cast<int64_t>(request.destination_width) * 4 ==
             static_cast<int64_t>(request.source_width) * 3 &&
           static_cast<int64_t>(request.destination_height) * 4 ==
             static_cast<int64_t>(request.source_height) * 3;
}

}  // namespace

bool ImageCudaAutoRecommends(
  const ImageGeometryRequest& request) noexcept {
    const size_t source_pixels =
      PixelCount(request.source_width, request.source_height);
    constexpr size_t kLargestMeasuredInput = 3840u * 2160u;
    if (source_pixels > kLargestMeasuredInput) return false;

    if (request.element_type == ImageElementType::kUInt8 &&
        request.channels == 3) {
        switch (request.operation) {
            case ImageGeometryOperation::kResizeNearest:
                // Repeated runs disagreed at the host-allocation crossover;
                // nearest remains CPU until that variance is explained.
                return false;
            case ImageGeometryOperation::kResizeBilinear:
                return IsMeasuredThreeQuarterResize(request) &&
                       source_pixels >= 128u * 128u;
            case ImageGeometryOperation::kWarpAffineBilinear:
                return request.destination_width == request.source_width &&
                       request.destination_height == request.source_height &&
                       source_pixels >= 64u * 64u;
            case ImageGeometryOperation::kRotate90:
            case ImageGeometryOperation::kRotate180:
            case ImageGeometryOperation::kRotate270:
                return source_pixels >= 96u * 96u;
            default:
                return false;
        }
    }

    if (request.element_type == ImageElementType::kFloat32 &&
        request.channels == 3) {
        switch (request.operation) {
            case ImageGeometryOperation::kResizeBilinear:
                return IsMeasuredThreeQuarterResize(request) &&
                       source_pixels >= 512u * 512u &&
                       source_pixels <= 2560u * 1440u;
            case ImageGeometryOperation::kRotate90:
            case ImageGeometryOperation::kRotate270:
                return source_pixels >= 512u * 512u;
            case ImageGeometryOperation::kRotate180:
                return source_pixels >= 512u * 512u;
            case ImageGeometryOperation::kResizeNearest:
            default:
                return false;
        }
    }
    return false;
}

bool TryExecuteAcceleratedImageGeometry(
  const ImageGeometryRequest& request) noexcept {
    if (!IsCudaAccelerationEnabled()) return false;

    const AccelerationPreference preference = GetAccelerationPreference();
    if (preference == AccelerationPreference::kCpu) return false;
    if (preference == AccelerationPreference::kAuto &&
        !ImageCudaAutoRecommends(request)) {
        return false;
    }
    if (!ImageCudaSupports(request)) return false;
    if (!ExecuteImageGeometryCudaHost(request)) return false;

    RecordImageExecutionBackend(AccelerationBackend::kCuda);
    return true;
}

}  // namespace internal
}  // namespace inspirecv
