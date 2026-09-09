#include "inspirecv/task/cuda/cuda_dispatch.h"

namespace inspirecv {
namespace task {
namespace internal {

struct CudaExecutor {};

bool CudaRuntimeAvailable() noexcept { return false; }
bool QueryCudaDeviceDescriptor(CudaDeviceDescriptor*) noexcept { return false; }
bool CudaSupportsConversion(const PipelineConfig&, const Matrix&,
                            const ConversionRequest&) noexcept {
    return false;
}
CudaExecutor* CreateCudaExecutor() noexcept { return nullptr; }
void DestroyCudaExecutor(CudaExecutor*) noexcept {}

TaskStatus ExecuteCudaHost(CudaExecutor*, const PipelineConfig&, const Matrix&,
                           const Matrix&, const ConversionRequest&,
                           const uint8_t*, void*, bool* handled) noexcept {
    if (handled) *handled = false;
    return ACCELERATION_UNAVAILABLE;
}

TaskStatus ExecuteCudaDevice(CudaExecutor*, const PipelineConfig&, const Matrix&,
                             const Matrix&, const ConversionRequest&,
                             std::uintptr_t, std::uintptr_t, void*,
                             bool* handled) noexcept {
    if (handled) *handled = false;
    return ACCELERATION_UNAVAILABLE;
}

}  // namespace internal
}  // namespace task
}  // namespace inspirecv
