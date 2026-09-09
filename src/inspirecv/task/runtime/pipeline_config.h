#ifndef INSPIRECV_TASK_RUNTIME_PIPELINE_CONFIG_H_
#define INSPIRECV_TASK_RUNTIME_PIPELINE_CONFIG_H_

#include <inspirecv/task/core/preprocess_types.h>
#include <inspirecv/task/acceleration.h>

namespace inspirecv {
namespace task {
namespace internal {

// Backend-neutral, value-owned configuration consumed by the execution
// pipeline. Public API facades translate into this type at their boundary.
struct PipelineConfig {
    Filter filter = NEAREST;
    StreamFormat source_format = RGBA;
    StreamFormat destination_format = RGBA;
    float mean[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float scale[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    Wrap wrap = CLAMP_TO_EDGE;
    bool preserve_identity_float_order = false;
    BackendPreference backend_preference = BackendPreference::kDefault;
};

}  // namespace internal
}  // namespace task
}  // namespace inspirecv

#endif  // INSPIRECV_TASK_RUNTIME_PIPELINE_CONFIG_H_
