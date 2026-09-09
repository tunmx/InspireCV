#ifndef INSPIRECV_CORE_CUDA_IMAGE_GEOMETRY_DISPATCH_H_
#define INSPIRECV_CORE_CUDA_IMAGE_GEOMETRY_DISPATCH_H_

#include <cstddef>
#include <cstdint>

namespace inspirecv {
namespace internal {

enum class ImageElementType : uint8_t {
    kUInt8 = 0,
    kFloat32 = 1,
};

enum class ImageGeometryOperation : uint8_t {
    kResizeNearest = 0,
    kResizeBilinear = 1,
    kWarpAffineBilinear = 2,
    kRotate90 = 3,
    kRotate180 = 4,
    kRotate270 = 5,
};

struct ImageGeometryRequest {
    const void* source = nullptr;
    void* destination = nullptr;
    int source_width = 0;
    int source_height = 0;
    int destination_width = 0;
    int destination_height = 0;
    int channels = 0;
    ImageElementType element_type = ImageElementType::kUInt8;
    ImageGeometryOperation operation = ImageGeometryOperation::kResizeNearest;
    float affine[6] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
};

struct CudaDeviceIdentity {
    char name[256] = {};
    int compute_capability_major = 0;
    int compute_capability_minor = 0;
    int runtime_version = 0;
    int driver_version = 0;
};

bool ImageCudaRuntimeAvailable() noexcept;
bool QueryImageCudaDeviceIdentity(CudaDeviceIdentity* identity) noexcept;
bool ImageCudaSupports(const ImageGeometryRequest& request) noexcept;
bool ExecuteImageGeometryCudaHost(const ImageGeometryRequest& request) noexcept;

}  // namespace internal
}  // namespace inspirecv

#endif  // INSPIRECV_CORE_CUDA_IMAGE_GEOMETRY_DISPATCH_H_
