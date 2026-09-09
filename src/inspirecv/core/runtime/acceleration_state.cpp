#include <inspirecv/acceleration.h>

#include <atomic>

#include "inspirecv/core/cuda/image_geometry_dispatch.h"
#include "inspirecv/core/runtime/acceleration_state.h"

namespace inspirecv {
namespace {

std::atomic<bool> g_cuda_enabled{false};
std::atomic<uint8_t> g_acceleration_preference{
  static_cast<uint8_t>(AccelerationPreference::kAuto)};
thread_local AccelerationBackend g_last_image_backend =
  AccelerationBackend::kCpu;

}  // namespace

bool SetCudaAccelerationEnabled(bool enabled) noexcept {
    if (!enabled) {
        g_cuda_enabled.store(false, std::memory_order_release);
        return true;
    }
    if (!internal::ImageCudaRuntimeAvailable()) {
        g_cuda_enabled.store(false, std::memory_order_release);
        return false;
    }
    g_cuda_enabled.store(true, std::memory_order_release);
    return true;
}

bool IsCudaAccelerationEnabled() noexcept {
    return g_cuda_enabled.load(std::memory_order_acquire);
}

bool IsCudaAccelerationAvailable() noexcept {
    return internal::ImageCudaRuntimeAvailable();
}

void SetAccelerationPreference(AccelerationPreference preference) noexcept {
    g_acceleration_preference.store(static_cast<uint8_t>(preference),
                                    std::memory_order_release);
}

AccelerationPreference GetAccelerationPreference() noexcept {
    return static_cast<AccelerationPreference>(
      g_acceleration_preference.load(std::memory_order_acquire));
}

AccelerationBackend GetLastImageExecutionBackend() noexcept {
    return g_last_image_backend;
}

namespace internal {

void RecordImageExecutionBackend(AccelerationBackend backend) noexcept {
    g_last_image_backend = backend;
}

}  // namespace internal
}  // namespace inspirecv
