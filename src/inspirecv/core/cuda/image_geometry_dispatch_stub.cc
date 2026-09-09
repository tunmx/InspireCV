#include "inspirecv/core/cuda/image_geometry_dispatch.h"

#include <utility>

#include <inspirecv/cuda/image.h>

namespace inspirecv {
namespace internal {

bool ImageCudaRuntimeAvailable() noexcept { return false; }
bool QueryImageCudaDeviceIdentity(CudaDeviceIdentity*) noexcept { return false; }
bool ImageCudaSupports(const ImageGeometryRequest&) noexcept { return false; }
bool ExecuteImageGeometryCudaHost(const ImageGeometryRequest&) noexcept {
    return false;
}

}  // namespace internal

namespace cuda {

struct DeviceImage::Impl {};

DeviceImage::DeviceImage() : impl_(new Impl) {}
DeviceImage::~DeviceImage() = default;
DeviceImage::DeviceImage(DeviceImage&& other) noexcept = default;
DeviceImage& DeviceImage::operator=(DeviceImage&& other) noexcept = default;

Status DeviceImage::UploadBytes(const void*, int, int, int, ElementType,
                                DeviceImage*) {
    return Status::kAccelerationDisabled;
}

Status DeviceImage::Upload(const Image&, DeviceImage*) {
    return Status::kAccelerationDisabled;
}

Status DeviceImage::Upload(const ImageT<float>&, DeviceImage*) {
    return Status::kAccelerationDisabled;
}

Status DeviceImage::Download(Image*, void*) const {
    return Status::kAccelerationDisabled;
}

Status DeviceImage::Download(ImageT<float>*, void*) const {
    return Status::kAccelerationDisabled;
}

Status DeviceImage::EnqueueGeometry(int, int, int, const float*, DeviceImage*,
                                    void*) const {
    return Status::kAccelerationDisabled;
}

Status DeviceImage::Resize(int, int, bool, DeviceImage*, void*) const {
    return Status::kAccelerationDisabled;
}

Status DeviceImage::WarpAffine(const TransformMatrix&, int, int, DeviceImage*,
                               void*) const {
    return Status::kAccelerationDisabled;
}

Status DeviceImage::Rotate90(DeviceImage*, void*) const {
    return Status::kAccelerationDisabled;
}

Status DeviceImage::Rotate180(DeviceImage*, void*) const {
    return Status::kAccelerationDisabled;
}

Status DeviceImage::Rotate270(DeviceImage*, void*) const {
    return Status::kAccelerationDisabled;
}

bool DeviceImage::Empty() const noexcept { return true; }
int DeviceImage::Width() const noexcept { return 0; }
int DeviceImage::Height() const noexcept { return 0; }
int DeviceImage::Channels() const noexcept { return 0; }
ElementType DeviceImage::Type() const noexcept { return ElementType::kUInt8; }
DeviceImageView DeviceImage::View() const noexcept { return DeviceImageView{}; }

Status Synchronize(void*) noexcept {
    return Status::kAccelerationDisabled;
}

const char* StatusMessage(Status status) noexcept {
    switch (status) {
        case Status::kOk: return "ok";
        case Status::kInvalidArgument: return "invalid argument";
        case Status::kAccelerationDisabled: return "acceleration disabled";
        case Status::kAccelerationUnavailable: return "acceleration unavailable";
        case Status::kUnsupportedOperation: return "unsupported operation";
        case Status::kAllocationFailure: return "device allocation failure";
        case Status::kExecutionFailure: return "CUDA execution failure";
    }
    return "unknown CUDA image status";
}

}  // namespace cuda
}  // namespace inspirecv
