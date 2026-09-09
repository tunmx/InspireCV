#ifndef INSPIRECV_TASK_CUDA_DISPATCH_H_
#define INSPIRECV_TASK_CUDA_DISPATCH_H_

#include <cstdint>

#include <inspirecv/task/core/matrix.h>
#include <inspirecv/task/task_status.h>

#include "inspirecv/task/planning/conversion_request.h"
#include "inspirecv/task/runtime/pipeline_config.h"

namespace inspirecv {
namespace task {
namespace internal {

struct CudaExecutor;

struct CudaDeviceDescriptor {
    char name[128] = {};
    int compute_capability_major = 0;
    int compute_capability_minor = 0;
    int runtime_version = 0;
    int driver_version = 0;
};

bool CudaRuntimeAvailable() noexcept;
bool QueryCudaDeviceDescriptor(CudaDeviceDescriptor* descriptor) noexcept;
bool CudaSupportsConversion(const PipelineConfig& config,
                            const Matrix& destination_to_source,
                            const ConversionRequest& request) noexcept;
CudaExecutor* CreateCudaExecutor() noexcept;
void DestroyCudaExecutor(CudaExecutor* executor) noexcept;

TaskStatus ExecuteCudaHost(CudaExecutor* executor,
                           const PipelineConfig& config,
                           const Matrix& destination_to_source,
                           const Matrix& source_to_destination,
                           const ConversionRequest& request,
                           const uint8_t* source, void* destination,
                           bool* handled) noexcept;

TaskStatus ExecuteCudaDevice(CudaExecutor* executor,
                             const PipelineConfig& config,
                             const Matrix& destination_to_source,
                             const Matrix& source_to_destination,
                             const ConversionRequest& request,
                             std::uintptr_t source,
                             std::uintptr_t destination,
                             void* stream, bool* handled) noexcept;

bool CudaAccelerationEnabled() noexcept;

}  // namespace internal
}  // namespace task
}  // namespace inspirecv

#endif  // INSPIRECV_TASK_CUDA_DISPATCH_H_
