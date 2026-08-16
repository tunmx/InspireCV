#ifndef INSPIRECV_TASK_CUDA_AUTO_POLICY_H_
#define INSPIRECV_TASK_CUDA_AUTO_POLICY_H_

#include <cstddef>

#include <inspirecv/task/core/matrix.h>

#include "inspirecv/task/planning/conversion_request.h"
#include "inspirecv/task/runtime/pipeline_config.h"

namespace inspirecv {
namespace task {
namespace internal {

// Maximum packed BGR/RGB source storage recommended for a host round-trip at
// the supplied output size. Zero means that the default 3060 baseline recommends
// CPU. Kept visible to unit tests so the shipped table cannot drift from its
// documented benchmark artifact.
size_t DefaultRecommendedMaxSourceBytes(size_t destination_pixels) noexcept;

bool CudaAutoRecommendsHost(const PipelineConfig& config,
                            const Matrix& destination_to_source,
                            const ConversionRequest& request) noexcept;

}  // namespace internal
}  // namespace task
}  // namespace inspirecv

#endif  // INSPIRECV_TASK_CUDA_AUTO_POLICY_H_
