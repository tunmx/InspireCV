#ifndef INSPIRECV_CORE_RUNTIME_ACCELERATION_STATE_H_
#define INSPIRECV_CORE_RUNTIME_ACCELERATION_STATE_H_

#include <inspirecv/acceleration.h>

namespace inspirecv {
namespace internal {

void RecordImageExecutionBackend(AccelerationBackend backend) noexcept;

}  // namespace internal
}  // namespace inspirecv

#endif  // INSPIRECV_CORE_RUNTIME_ACCELERATION_STATE_H_
