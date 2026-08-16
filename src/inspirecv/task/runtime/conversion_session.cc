#include "inspirecv/task/runtime/conversion_session.h"

#include <vector>

#include "inspirecv/task/cuda/cuda_auto_policy.h"
#include "inspirecv/task/cuda/cuda_dispatch.h"
#include "inspirecv/task/planning/conversion_request.h"

namespace inspirecv {
namespace task {
namespace internal {

ConversionSession::ConversionSession(const PipelineConfig& config)
    : config_(config), engine_(config) {}

ConversionSession::~ConversionSession() {
    DestroyCudaExecutor(cuda_executor_);
}

void ConversionSession::SetMatrix(const Matrix& matrix) {
    transform_ = matrix;
    engine_.SetMatrix(matrix);
}

TaskStatus ConversionSession::Execute(const uint8_t* source, void* destination,
                                      const ConversionRequest& request) {
    if (IsExactIdentityCopy(config_, transform_, request)) {
        CopyExactImage(source, destination, request);
        last_backend_ = ExecutionBackend::kCpu;
        return SUCCESS;
    }
    BackendPreference preference = config_.backend_preference;
    if (preference == BackendPreference::kDefault) {
        preference = GetDefaultBackendPreference();
    }
    const bool cuda_supported =
      CudaAccelerationEnabled() && preference != BackendPreference::kCpu &&
      CudaSupportsConversion(config_, transform_, request);
    const bool use_cuda =
      cuda_supported &&
      (preference == BackendPreference::kCuda ||
       (preference == BackendPreference::kAuto &&
        CudaAutoRecommendsHost(config_, transform_, request)));
    if (use_cuda) {
        if (!cuda_executor_) cuda_executor_ = CreateCudaExecutor();
        if (!cuda_executor_) return ACCELERATION_FAILURE;
        Matrix inverse;
        transform_.invert(&inverse);
        bool handled = false;
        const TaskStatus cuda_status = ExecuteCudaHost(
          cuda_executor_, config_, transform_, inverse, request, source,
          destination, &handled);
        if (handled) {
            if (cuda_status == SUCCESS) {
                last_backend_ = ExecutionBackend::kCuda;
            }
            return cuda_status;
        }
    }
    engine_.SetPadding(padding_value_);
    const TaskStatus compile_status = engine_.Compile(request);
    if (compile_status != SUCCESS) return compile_status;
    const TaskStatus status = engine_.Execute(source, destination);
    if (status == SUCCESS) last_backend_ = ExecutionBackend::kCpu;
    return status;
}

TaskStatus ConversionSession::Convert(
  const uint8_t* source, int source_width, int source_height,
  int source_stride, void* destination, int destination_width,
  int destination_height, int destination_channels, int destination_stride,
  halide_type_t destination_type) {
    ConversionRequest request;
    const TaskStatus status = ResolveImageRequest(
      config_, source, source_width, source_height, source_stride, destination,
      destination_width, destination_height, destination_channels,
      destination_stride, destination_type, &request);
    if (status != SUCCESS) return status;
    return Execute(source, destination, request);
}

TaskStatus ConversionSession::Convert(const uint8_t* source, int source_width,
                                      int source_height, int source_stride,
                                      const TensorView& destination) {
    ConversionRequest request;
    const TaskStatus status = ResolveTensorRequest(
      config_, source, source_width, source_height, source_stride, destination,
      &request);
    if (status != SUCCESS) return status;
    return Execute(source, destination.data, request);
}

void ConversionSession::DrawRegions(uint8_t* image, int width, int height,
                                    int channels, const int* regions,
                                    int region_count, const uint8_t* color) {
    if (!image || !regions || !color || width <= 0 || height <= 0 ||
        channels <= 0 || channels > 4 || region_count <= 0) {
        return;
    }
    std::vector<int32_t> validated_regions(3 * region_count);
    for (int index = 0; index < region_count; ++index) {
        const int y = regions[3 * index];
        const int first_x = regions[3 * index + 1];
        const int last_x = regions[3 * index + 2];
        if (y < 0 || y >= height || first_x < 0 || first_x > last_x ||
            last_x >= width) {
            return;
        }
        validated_regions[3 * index] = y;
        validated_regions[3 * index + 1] = first_x;
        validated_regions[3 * index + 2] = last_x;
    }

    ConversionRequest request;
    request.source_channels = channels;
    request.source_width = width;
    request.source_height = height;
    request.destination_channels = channels;
    request.destination_width = width;
    request.destination_height = region_count;
    request.destination_type = halide_type_of<uint8_t>();
    if (engine_.Compile(request) != SUCCESS) return;
    engine_.DrawRegions(image, validated_regions.data(), region_count, color);
}

}  // namespace internal
}  // namespace task
}  // namespace inspirecv
