#include "inspirecv/core/cuda/image_geometry_dispatch.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>

#include <inspirecv/acceleration.h>
#include <inspirecv/cuda/image.h>

namespace inspirecv {
namespace internal {
namespace {

struct DeviceRequest {
    int source_width;
    int source_height;
    int destination_width;
    int destination_height;
    int channels;
    int operation;
    float affine[6];
};

class ThreadWorkspace {
public:
    ~ThreadWorkspace() {
        if (source) cudaFree(source);
        if (destination) cudaFree(destination);
    }

    void* source = nullptr;
    void* destination = nullptr;
    size_t source_capacity = 0;
    size_t destination_capacity = 0;
};

thread_local ThreadWorkspace g_workspace;

bool EnsureAllocation(void** pointer, size_t* capacity, size_t required) {
    if (*capacity >= required) return true;
    if (*pointer) {
        if (cudaFree(*pointer) != cudaSuccess) return false;
        *pointer = nullptr;
        *capacity = 0;
    }
    if (cudaMalloc(pointer, required) != cudaSuccess) return false;
    *capacity = required;
    return true;
}

bool StorageBytes(int width, int height, int channels, size_t element_bytes,
                  size_t* result) {
    if (!result || width <= 0 || height <= 0 || channels <= 0) return false;
    const size_t pixels = static_cast<size_t>(width) *
                          static_cast<size_t>(height);
    if (pixels > std::numeric_limits<size_t>::max() /
                   static_cast<size_t>(channels)) {
        return false;
    }
    const size_t elements = pixels * static_cast<size_t>(channels);
    if (elements > std::numeric_limits<size_t>::max() / element_bytes) {
        return false;
    }
    *result = elements * element_bytes;
    return true;
}

template <typename Pixel>
__device__ Pixel BilinearValue(Pixel top_left, Pixel top_right,
                               Pixel bottom_left, Pixel bottom_right,
                               float fraction_x, float fraction_y);

template <>
__device__ float BilinearValue(float top_left, float top_right,
                               float bottom_left, float bottom_right,
                               float fraction_x, float fraction_y) {
    const float top = top_left + (top_right - top_left) * fraction_x;
    const float bottom =
      bottom_left + (bottom_right - bottom_left) * fraction_x;
    return top + (bottom - top) * fraction_y;
}

template <>
__device__ uint8_t BilinearValue(uint8_t top_left, uint8_t top_right,
                                 uint8_t bottom_left, uint8_t bottom_right,
                                 float fraction_x, float fraction_y) {
    const float top = top_left + (top_right - top_left) * fraction_x;
    const float bottom =
      bottom_left + (bottom_right - bottom_left) * fraction_x;
    return static_cast<uint8_t>(roundf(top + (bottom - top) * fraction_y));
}

template <typename Pixel>
__device__ Pixel ReadOrBorder(const Pixel* source, const DeviceRequest& request,
                              int x, int y, int channel) {
    if (x < 0 || x >= request.source_width || y < 0 ||
        y >= request.source_height) {
        return static_cast<Pixel>(0);
    }
    return source[(static_cast<size_t>(y) * request.source_width + x) *
                    request.channels +
                  channel];
}

template <typename Pixel>
__device__ Pixel SampleResize(const Pixel* source,
                              const DeviceRequest& request, int x, int y,
                              int channel, bool linear) {
    const float source_x =
      x * (static_cast<float>(request.source_width) /
           request.destination_width);
    const float source_y =
      y * (static_cast<float>(request.source_height) /
           request.destination_height);
    const int x0 = min(static_cast<int>(source_x), request.source_width - 1);
    const int y0 = min(static_cast<int>(source_y), request.source_height - 1);
    if (!linear) {
        return ReadOrBorder(source, request, x0, y0, channel);
    }
    const int x1 = min(x0 + 1, request.source_width - 1);
    const int y1 = min(y0 + 1, request.source_height - 1);
    const float fraction_x = source_x - x0;
    const float fraction_y = source_y - y0;
    return BilinearValue(
      ReadOrBorder(source, request, x0, y0, channel),
      ReadOrBorder(source, request, x1, y0, channel),
      ReadOrBorder(source, request, x0, y1, channel),
      ReadOrBorder(source, request, x1, y1, channel), fraction_x,
      fraction_y);
}

template <typename Pixel>
__device__ Pixel SampleAffine(const Pixel* source,
                              const DeviceRequest& request, int x, int y,
                              int channel) {
    const float source_x =
      x * request.affine[0] + y * request.affine[1] + request.affine[2];
    const float source_y =
      x * request.affine[3] + y * request.affine[4] + request.affine[5];
    const bool axis_aligned = fabsf(request.affine[1]) < 1.0e-6f &&
                              fabsf(request.affine[3]) < 1.0e-6f;

    if (!axis_aligned &&
        (source_x >= request.source_width ||
         source_y >= request.source_height || source_x < 0.0f ||
         source_y < 0.0f)) {
        return static_cast<Pixel>(0);
    }

    const int x0 = static_cast<int>(floorf(source_x));
    const int y0 = static_cast<int>(floorf(source_y));
    const int x1 = axis_aligned ? x0 + 1
                                : min(x0 + 1, request.source_width - 1);
    const int y1 = axis_aligned ? y0 + 1
                                : min(y0 + 1, request.source_height - 1);
    const float fraction_x = source_x - static_cast<float>(x0);
    const float fraction_y = source_y - static_cast<float>(y0);
    return BilinearValue(
      ReadOrBorder(source, request, x0, y0, channel),
      ReadOrBorder(source, request, x1, y0, channel),
      ReadOrBorder(source, request, x0, y1, channel),
      ReadOrBorder(source, request, x1, y1, channel), fraction_x,
      fraction_y);
}

template <typename Pixel>
__global__ void ImageGeometryKernel(const Pixel* source, Pixel* destination,
                                    DeviceRequest request) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= request.destination_width || y >= request.destination_height) {
        return;
    }

    int source_x = x;
    int source_y = y;
    const ImageGeometryOperation operation =
      static_cast<ImageGeometryOperation>(request.operation);
    if (operation == ImageGeometryOperation::kRotate90) {
        source_x = y;
        source_y = request.source_height - 1 - x;
    } else if (operation == ImageGeometryOperation::kRotate180) {
        source_x = request.source_width - 1 - x;
        source_y = request.source_height - 1 - y;
    } else if (operation == ImageGeometryOperation::kRotate270) {
        source_x = request.source_width - 1 - y;
        source_y = x;
    }

    const size_t destination_base =
      (static_cast<size_t>(y) * request.destination_width + x) *
      request.channels;
    for (int channel = 0; channel < request.channels; ++channel) {
        Pixel value;
        if (operation == ImageGeometryOperation::kResizeNearest) {
            value = SampleResize(source, request, x, y, channel, false);
        } else if (operation == ImageGeometryOperation::kResizeBilinear) {
            value = SampleResize(source, request, x, y, channel, true);
        } else if (operation ==
                   ImageGeometryOperation::kWarpAffineBilinear) {
            value = SampleAffine(source, request, x, y, channel);
        } else {
            value = ReadOrBorder(source, request, source_x, source_y, channel);
        }
        destination[destination_base + channel] = value;
    }
}

DeviceRequest MakeDeviceRequest(const ImageGeometryRequest& request) {
    DeviceRequest device{};
    device.source_width = request.source_width;
    device.source_height = request.source_height;
    device.destination_width = request.destination_width;
    device.destination_height = request.destination_height;
    device.channels = request.channels;
    device.operation = static_cast<int>(request.operation);
    for (int index = 0; index < 6; ++index) {
        device.affine[index] = request.affine[index];
    }
    return device;
}

template <typename Pixel>
bool LaunchGeometry(const ImageGeometryRequest& request, const void* source,
                    void* destination, cudaStream_t stream) {
    const DeviceRequest device_request = MakeDeviceRequest(request);
    const dim3 block(16, 16);
    const dim3 grid(
      static_cast<unsigned>((request.destination_width + block.x - 1) /
                            block.x),
      static_cast<unsigned>((request.destination_height + block.y - 1) /
                            block.y));
    ImageGeometryKernel<<<grid, block, 0, stream>>>(
      static_cast<const Pixel*>(source), static_cast<Pixel*>(destination),
      device_request);
    return cudaPeekAtLastError() == cudaSuccess;
}

}  // namespace

bool ImageCudaRuntimeAvailable() noexcept {
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

bool QueryImageCudaDeviceIdentity(CudaDeviceIdentity* identity) noexcept {
    if (!identity) return false;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return false;
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
        return false;
    }
    std::memset(identity, 0, sizeof(*identity));
    std::strncpy(identity->name, properties.name, sizeof(identity->name) - 1);
    identity->compute_capability_major = properties.major;
    identity->compute_capability_minor = properties.minor;
    if (cudaRuntimeGetVersion(&identity->runtime_version) != cudaSuccess) {
        return false;
    }
    if (cudaDriverGetVersion(&identity->driver_version) != cudaSuccess) {
        return false;
    }
    return true;
}

bool ImageCudaSupports(const ImageGeometryRequest& request) noexcept {
    if (!request.source || !request.destination || request.source_width <= 0 ||
        request.source_height <= 0 || request.destination_width <= 0 ||
        request.destination_height <= 0 || request.channels < 1 ||
        request.channels > 4) {
        return false;
    }
    if (request.element_type != ImageElementType::kUInt8 &&
        request.element_type != ImageElementType::kFloat32) {
        return false;
    }
    switch (request.operation) {
        case ImageGeometryOperation::kResizeNearest:
        case ImageGeometryOperation::kResizeBilinear:
            return true;
        case ImageGeometryOperation::kWarpAffineBilinear: {
            const bool axis_aligned = std::fabs(request.affine[1]) < 1.0e-6f &&
                                      std::fabs(request.affine[3]) < 1.0e-6f;
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
            // The frozen x86 CPU path has SIMD-specific rounding/weight
            // behavior for one- and three-channel axis-aligned affine work.
            // Keep those requests on CPU instead of silently changing pixels.
            if (axis_aligned &&
                (request.channels == 1 || request.channels == 3)) {
                return false;
            }
#endif
            return axis_aligned ||
                   (request.element_type == ImageElementType::kUInt8 &&
                    request.channels == 3);
        }
        case ImageGeometryOperation::kRotate90:
            return request.destination_width == request.source_height &&
                   request.destination_height == request.source_width;
        case ImageGeometryOperation::kRotate270:
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
            // Preserve the existing x86 float/1-channel tiled implementation,
            // including its historical lane order, by falling back to CPU.
            if (request.element_type == ImageElementType::kFloat32 &&
                request.channels == 1) {
                return false;
            }
#endif
            return request.destination_width == request.source_height &&
                   request.destination_height == request.source_width;
        case ImageGeometryOperation::kRotate180:
            return request.destination_width == request.source_width &&
                   request.destination_height == request.source_height;
        default:
            return false;
    }
}

bool ExecuteImageGeometryCudaHost(
  const ImageGeometryRequest& request) noexcept {
    if (!ImageCudaSupports(request)) return false;
    const size_t element_bytes =
      request.element_type == ImageElementType::kFloat32 ? sizeof(float)
                                                         : sizeof(uint8_t);
    size_t source_bytes = 0;
    size_t destination_bytes = 0;
    if (!StorageBytes(request.source_width, request.source_height,
                      request.channels, element_bytes, &source_bytes) ||
        !StorageBytes(request.destination_width, request.destination_height,
                      request.channels, element_bytes, &destination_bytes)) {
        return false;
    }
    if (!EnsureAllocation(&g_workspace.source, &g_workspace.source_capacity,
                          source_bytes) ||
        !EnsureAllocation(&g_workspace.destination,
                          &g_workspace.destination_capacity,
                          destination_bytes)) {
        return false;
    }
    if (cudaMemcpy(g_workspace.source, request.source, source_bytes,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
    }
    const bool launched =
      request.element_type == ImageElementType::kFloat32
        ? LaunchGeometry<float>(request, g_workspace.source,
                                g_workspace.destination, nullptr)
        : LaunchGeometry<uint8_t>(request, g_workspace.source,
                                  g_workspace.destination, nullptr);
    if (!launched) return false;
    return cudaMemcpy(request.destination, g_workspace.destination,
                      destination_bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
}

}  // namespace internal

namespace cuda {
namespace {

internal::ImageElementType ToInternalType(ElementType type) {
    return type == ElementType::kFloat32
             ? internal::ImageElementType::kFloat32
             : internal::ImageElementType::kUInt8;
}

size_t ElementBytes(ElementType type) {
    return type == ElementType::kFloat32 ? sizeof(float) : sizeof(uint8_t);
}

Status CheckRuntime() {
    if (!IsCudaAccelerationEnabled()) return Status::kAccelerationDisabled;
    return internal::ImageCudaRuntimeAvailable()
             ? Status::kOk
             : Status::kAccelerationUnavailable;
}

}  // namespace

struct DeviceImage::Impl {
    ~Impl() {
        if (data) cudaFree(data);
    }

    void* data = nullptr;
    size_t bytes = 0;
    size_t capacity = 0;
    int width = 0;
    int height = 0;
    int channels = 0;
    ElementType type = ElementType::kUInt8;
};

DeviceImage::DeviceImage() : impl_(new Impl) {}
DeviceImage::~DeviceImage() = default;
DeviceImage::DeviceImage(DeviceImage&& other) noexcept = default;
DeviceImage& DeviceImage::operator=(DeviceImage&& other) noexcept = default;

Status DeviceImage::UploadBytes(const void* source, int width, int height,
                                int channels, ElementType type,
                                DeviceImage* destination) {
    const Status runtime = CheckRuntime();
    if (runtime != Status::kOk) return runtime;
    if (!source || !destination || width <= 0 || height <= 0 || channels < 1 ||
        channels > 4 ||
        (type != ElementType::kUInt8 && type != ElementType::kFloat32)) {
        return Status::kInvalidArgument;
    }
    size_t bytes = 0;
    if (!internal::StorageBytes(width, height, channels, ElementBytes(type),
                                &bytes)) {
        return Status::kInvalidArgument;
    }

    if (!destination->impl_) destination->impl_.reset(new Impl);
    if (destination->impl_->capacity < bytes) {
        DeviceImage candidate;
        if (cudaMalloc(&candidate.impl_->data, bytes) != cudaSuccess) {
            return Status::kAllocationFailure;
        }
        candidate.impl_->capacity = bytes;
        *destination = std::move(candidate);
    }
    if (cudaMemcpy(destination->impl_->data, source, bytes,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        return Status::kExecutionFailure;
    }
    destination->impl_->bytes = bytes;
    destination->impl_->width = width;
    destination->impl_->height = height;
    destination->impl_->channels = channels;
    destination->impl_->type = type;
    return Status::kOk;
}

Status DeviceImage::Upload(const Image& source, DeviceImage* destination) {
    if (source.Empty()) return Status::kInvalidArgument;
    return UploadBytes(source.Data(), source.Width(), source.Height(),
                       source.Channels(), ElementType::kUInt8, destination);
}

Status DeviceImage::Upload(const ImageT<float>& source,
                           DeviceImage* destination) {
    if (source.Empty()) return Status::kInvalidArgument;
    return UploadBytes(source.Data(), source.Width(), source.Height(),
                       source.Channels(), ElementType::kFloat32, destination);
}

Status DeviceImage::Download(Image* destination, void* stream) const {
    const Status runtime = CheckRuntime();
    if (runtime != Status::kOk) return runtime;
    if (!destination || Empty() || impl_->type != ElementType::kUInt8) {
        return Status::kInvalidArgument;
    }
    destination->Reset(impl_->width, impl_->height, impl_->channels);
    if (destination->Empty()) return Status::kAllocationFailure;
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (cudaMemcpyAsync(destination->GetInternalImage(), impl_->data,
                        impl_->bytes,
                        cudaMemcpyDeviceToHost, cuda_stream) != cudaSuccess ||
        cudaStreamSynchronize(cuda_stream) != cudaSuccess) {
        return Status::kExecutionFailure;
    }
    return Status::kOk;
}

Status DeviceImage::Download(ImageT<float>* destination, void* stream) const {
    const Status runtime = CheckRuntime();
    if (runtime != Status::kOk) return runtime;
    if (!destination || Empty() || impl_->type != ElementType::kFloat32) {
        return Status::kInvalidArgument;
    }
    destination->Reset(impl_->width, impl_->height, impl_->channels);
    if (destination->Empty()) return Status::kAllocationFailure;
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (cudaMemcpyAsync(destination->GetInternalImage(), impl_->data,
                        impl_->bytes,
                        cudaMemcpyDeviceToHost, cuda_stream) != cudaSuccess ||
        cudaStreamSynchronize(cuda_stream) != cudaSuccess) {
        return Status::kExecutionFailure;
    }
    return Status::kOk;
}

Status DeviceImage::EnqueueGeometry(int operation, int width, int height,
                                    const float* affine,
                                    DeviceImage* destination,
                                    void* stream) const {
    const Status runtime = CheckRuntime();
    if (runtime != Status::kOk) return runtime;
    if (!destination || destination == this || Empty() || width <= 0 ||
        height <= 0) {
        return Status::kInvalidArgument;
    }

    internal::ImageGeometryRequest request;
    request.source = impl_->data;
    request.source_width = impl_->width;
    request.source_height = impl_->height;
    request.destination_width = width;
    request.destination_height = height;
    request.channels = impl_->channels;
    request.element_type = ToInternalType(impl_->type);
    request.operation = static_cast<internal::ImageGeometryOperation>(operation);
    if (affine) std::copy(affine, affine + 6, request.affine);

    size_t destination_bytes = 0;
    if (!internal::StorageBytes(width, height, impl_->channels,
                                ElementBytes(impl_->type),
                                &destination_bytes)) {
        return Status::kInvalidArgument;
    }
    request.destination = reinterpret_cast<void*>(1);
    if (!internal::ImageCudaSupports(request)) {
        return Status::kUnsupportedOperation;
    }
    if (!destination->impl_) destination->impl_.reset(new Impl);
    if (destination->impl_->capacity < destination_bytes) {
        DeviceImage candidate;
        if (cudaMalloc(&candidate.impl_->data, destination_bytes) !=
            cudaSuccess) {
            return Status::kAllocationFailure;
        }
        candidate.impl_->capacity = destination_bytes;
        *destination = std::move(candidate);
    }
    request.destination = destination->impl_->data;
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const bool launched =
      impl_->type == ElementType::kFloat32
        ? internal::LaunchGeometry<float>(request, impl_->data,
                                          destination->impl_->data, cuda_stream)
        : internal::LaunchGeometry<uint8_t>(request, impl_->data,
                                            destination->impl_->data, cuda_stream);
    if (!launched) return Status::kExecutionFailure;

    destination->impl_->bytes = destination_bytes;
    destination->impl_->width = width;
    destination->impl_->height = height;
    destination->impl_->channels = impl_->channels;
    destination->impl_->type = impl_->type;
    return Status::kOk;
}

Status DeviceImage::Resize(int width, int height, bool use_linear,
                           DeviceImage* destination, void* stream) const {
    const int operation = static_cast<int>(
      use_linear ? internal::ImageGeometryOperation::kResizeBilinear
                 : internal::ImageGeometryOperation::kResizeNearest);
    return EnqueueGeometry(operation, width, height, nullptr, destination,
                           stream);
}

Status DeviceImage::WarpAffine(const TransformMatrix& matrix, int width,
                               int height, DeviceImage* destination,
                               void* stream) const {
    float affine[6] = {matrix[0], matrix[1], matrix[2],
                       matrix[3], matrix[4], matrix[5]};
    return EnqueueGeometry(
      static_cast<int>(internal::ImageGeometryOperation::kWarpAffineBilinear),
      width, height, affine, destination, stream);
}

Status DeviceImage::Rotate90(DeviceImage* destination, void* stream) const {
    return EnqueueGeometry(
      static_cast<int>(internal::ImageGeometryOperation::kRotate90),
      Height(), Width(), nullptr, destination, stream);
}

Status DeviceImage::Rotate180(DeviceImage* destination, void* stream) const {
    return EnqueueGeometry(
      static_cast<int>(internal::ImageGeometryOperation::kRotate180),
      Width(), Height(), nullptr, destination, stream);
}

Status DeviceImage::Rotate270(DeviceImage* destination, void* stream) const {
    return EnqueueGeometry(
      static_cast<int>(internal::ImageGeometryOperation::kRotate270),
      Height(), Width(), nullptr, destination, stream);
}

bool DeviceImage::Empty() const noexcept {
    return !impl_ || !impl_->data || impl_->width <= 0 || impl_->height <= 0 ||
           impl_->channels <= 0;
}

int DeviceImage::Width() const noexcept { return impl_ ? impl_->width : 0; }
int DeviceImage::Height() const noexcept { return impl_ ? impl_->height : 0; }
int DeviceImage::Channels() const noexcept {
    return impl_ ? impl_->channels : 0;
}
ElementType DeviceImage::Type() const noexcept {
    return impl_ ? impl_->type : ElementType::kUInt8;
}

DeviceImageView DeviceImage::View() const noexcept {
    DeviceImageView view;
    if (Empty()) return view;
    view.data = reinterpret_cast<std::uintptr_t>(impl_->data);
    view.row_stride_bytes = static_cast<size_t>(impl_->width) * impl_->channels *
                            ElementBytes(impl_->type);
    view.width = impl_->width;
    view.height = impl_->height;
    view.channels = impl_->channels;
    view.element_type = impl_->type;
    return view;
}

Status Synchronize(void* stream) noexcept {
    const Status runtime = CheckRuntime();
    if (runtime != Status::kOk) return runtime;
    return cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)) ==
                   cudaSuccess
             ? Status::kOk
             : Status::kExecutionFailure;
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
