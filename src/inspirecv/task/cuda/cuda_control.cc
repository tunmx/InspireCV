#include <inspirecv/task/acceleration.h>
#include <inspirecv/task/cuda.h>
#include <inspirecv/acceleration.h>

#include <climits>
#include <utility>
#include <vector>

#include <inspirecv/task/core/preprocess_types.h>

#include "inspirecv/task/api/public_conversions.h"
#include "inspirecv/task/cuda/cuda_dispatch.h"
#include "inspirecv/task/planning/conversion_request.h"
#include "inspirecv/task/runtime/pipeline_config.h"

namespace inspirecv {
namespace task {
namespace {

bool ToInternalTensor(const cuda::DeviceTensorBuffer& source,
                      TensorView* destination) {
    if (!destination || source.data == 0) return false;
    destination->data = reinterpret_cast<void*>(source.data);
    destination->width = source.width;
    destination->height = source.height;
    destination->channels = source.channels;
    switch (source.element_type) {
        case ElementType::kUInt8:
            destination->type = halide_type_of<uint8_t>();
            break;
        case ElementType::kFloat32:
            destination->type = halide_type_of<float>();
            break;
        default:
            return false;
    }
    switch (source.order) {
        case TensorOrder::kHwc:
            destination->layout = TensorLayout::NHWC;
            break;
        case TensorOrder::kChw:
            destination->layout = TensorLayout::NCHW;
            break;
        case TensorOrder::kChannelPacked4:
            destination->layout = TensorLayout::NC4HW4;
            break;
        default:
            return false;
    }
    destination->rowStride = source.row_stride_bytes;
    destination->channelStride = source.channel_stride_bytes;
    return true;
}

}  // namespace

bool SetCudaEnabled(bool enabled) noexcept {
    return SetCudaAccelerationEnabled(enabled);
}

bool IsCudaEnabled() noexcept {
    return IsCudaAccelerationEnabled();
}

bool IsCudaAvailable() noexcept {
    return IsCudaAccelerationAvailable();
}

bool SetDefaultBackendPreference(BackendPreference preference) noexcept {
    if (preference != BackendPreference::kAuto &&
        preference != BackendPreference::kCpu &&
        preference != BackendPreference::kCuda) {
        return false;
    }
    AccelerationPreference common_preference = AccelerationPreference::kAuto;
    if (preference == BackendPreference::kCpu) {
        common_preference = AccelerationPreference::kCpu;
    } else if (preference == BackendPreference::kCuda) {
        common_preference = AccelerationPreference::kCuda;
    }
    SetAccelerationPreference(common_preference);
    return true;
}

BackendPreference GetDefaultBackendPreference() noexcept {
    switch (GetAccelerationPreference()) {
        case AccelerationPreference::kCpu:
            return BackendPreference::kCpu;
        case AccelerationPreference::kCuda:
            return BackendPreference::kCuda;
        case AccelerationPreference::kAuto:
        default:
            return BackendPreference::kAuto;
    }
}

namespace internal {

bool CudaAccelerationEnabled() noexcept {
    return IsCudaAccelerationEnabled();
}

}  // namespace internal

struct cuda::Pipeline::Impl {
    explicit Impl(const PipelineOptions& options) {
        configuration_status = api_internal::ConfigurePipeline(options, &config);
        transform.setIdentity();
        inverse.setIdentity();
    }

    ~Impl() { internal::DestroyCudaExecutor(executor); }

    internal::PipelineConfig config;
    Matrix transform;
    Matrix inverse;
    internal::CudaExecutor* executor = nullptr;
    Status configuration_status = Status::kInvalidArgument;
};

cuda::Pipeline::Pipeline(const PipelineOptions& options)
    : impl_(new Impl(options)) {}

cuda::Pipeline::~Pipeline() = default;

cuda::Pipeline::Pipeline(cuda::Pipeline&& other) noexcept
    : impl_(std::move(other.impl_)) {}

cuda::Pipeline& cuda::Pipeline::operator=(cuda::Pipeline&& other) noexcept {
    if (this != &other) impl_ = std::move(other.impl_);
    return *this;
}

Status cuda::Pipeline::SetTransform(const TransformMatrix& transform) {
    if (!impl_) return Status::kInvalidArgument;
    if (impl_->configuration_status != Status::kOk) {
        return impl_->configuration_status;
    }
    const Matrix candidate = api_internal::ToTaskMatrix(transform);
    Matrix candidate_inverse;
    if (!candidate.invert(&candidate_inverse)) {
        return Status::kNonInvertibleTransform;
    }
    impl_->transform = candidate;
    impl_->inverse = candidate_inverse;
    return Status::kOk;
}

Status cuda::Pipeline::ConfigurationStatus() const noexcept {
    return impl_ ? impl_->configuration_status : Status::kInvalidArgument;
}

Status cuda::Pipeline::Run(const cuda::DeviceImageView& source,
                           const cuda::DeviceTensorBuffer& destination,
                           void* stream) {
    if (!IsCudaEnabled()) return Status::kAccelerationDisabled;
    if (!impl_) return Status::kInvalidArgument;
    if (impl_->configuration_status != Status::kOk) {
        return impl_->configuration_status;
    }
    if (source.data == 0 || source.width <= 0 ||
        source.height <= 0 ||
        source.row_stride_bytes > static_cast<size_t>(INT32_MAX)) {
        return Status::kInvalidArgument;
    }
    if (!IsCudaAvailable()) return Status::kAccelerationUnavailable;

    TensorView tensor;
    if (!ToInternalTensor(destination, &tensor)) {
        return Status::kInvalidArgument;
    }
    internal::ConversionRequest request;
    const auto* source_pointer = reinterpret_cast<const uint8_t*>(source.data);
    const TaskStatus resolve = internal::ResolveTensorRequest(
      impl_->config, source_pointer, source.width, source.height,
      static_cast<int>(source.row_stride_bytes), tensor, &request);
    if (resolve != SUCCESS) return api_internal::ToPublicStatus(resolve);
    if (!internal::CudaSupportsConversion(impl_->config, impl_->transform,
                                          request)) {
        return Status::kUnsupportedConversion;
    }
    if (!impl_->executor) {
        impl_->executor = internal::CreateCudaExecutor();
        if (!impl_->executor) return Status::kAccelerationFailure;
    }

    bool handled = false;
    const TaskStatus status = internal::ExecuteCudaDevice(
      impl_->executor, impl_->config, impl_->transform, impl_->inverse,
      request, source.data, destination.data, stream, &handled);
    return handled ? api_internal::ToPublicStatus(status)
                   : Status::kUnsupportedConversion;
}

Status cuda::Pipeline::Run(
  const ::inspirecv::cuda::DeviceImage& source,
  const cuda::DeviceTensorBuffer& destination, void* stream) {
    const ::inspirecv::cuda::DeviceImageView view = source.View();
    if (view.data == 0 || view.channels != 3 ||
        view.element_type != ::inspirecv::cuda::ElementType::kUInt8) {
        return Status::kInvalidArgument;
    }
    cuda::DeviceImageView task_view;
    task_view.data = view.data;
    task_view.row_stride_bytes = view.row_stride_bytes;
    task_view.width = view.width;
    task_view.height = view.height;
    return Run(task_view, destination, stream);
}

Status cuda::Pipeline::RunBatch(
  const cuda::DeviceImageView* sources,
  const cuda::DeviceTensorBuffer* destinations, size_t count, void* stream) {
    if (!IsCudaEnabled()) return Status::kAccelerationDisabled;
    if (!impl_ || !sources || !destinations || count == 0) {
        return Status::kInvalidArgument;
    }
    if (impl_->configuration_status != Status::kOk) {
        return impl_->configuration_status;
    }
    if (!IsCudaAvailable()) return Status::kAccelerationUnavailable;

    std::vector<internal::ConversionRequest> requests(count);
    for (size_t index = 0; index < count; ++index) {
        const cuda::DeviceImageView& source = sources[index];
        if (source.data == 0 || source.width <= 0 || source.height <= 0 ||
            source.row_stride_bytes > static_cast<size_t>(INT32_MAX)) {
            return Status::kInvalidArgument;
        }
        TensorView tensor;
        if (!ToInternalTensor(destinations[index], &tensor)) {
            return Status::kInvalidArgument;
        }
        const auto* source_pointer =
          reinterpret_cast<const uint8_t*>(source.data);
        const TaskStatus resolve = internal::ResolveTensorRequest(
          impl_->config, source_pointer, source.width, source.height,
          static_cast<int>(source.row_stride_bytes), tensor,
          &requests[index]);
        if (resolve != SUCCESS) return api_internal::ToPublicStatus(resolve);
        if (!internal::CudaSupportsConversion(
              impl_->config, impl_->transform, requests[index])) {
            return Status::kUnsupportedConversion;
        }
    }

    if (!impl_->executor) {
        impl_->executor = internal::CreateCudaExecutor();
        if (!impl_->executor) return Status::kAccelerationFailure;
    }
    for (size_t index = 0; index < count; ++index) {
        bool handled = false;
        const TaskStatus status = internal::ExecuteCudaDevice(
          impl_->executor, impl_->config, impl_->transform, impl_->inverse,
          requests[index], sources[index].data, destinations[index].data,
          stream, &handled);
        if (!handled) return Status::kUnsupportedConversion;
        if (status != SUCCESS) return api_internal::ToPublicStatus(status);
    }
    return Status::kOk;
}

}  // namespace task
}  // namespace inspirecv
