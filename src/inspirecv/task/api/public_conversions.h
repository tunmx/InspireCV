#ifndef INSPIRECV_TASK_API_PUBLIC_CONVERSIONS_H_
#define INSPIRECV_TASK_API_PUBLIC_CONVERSIONS_H_

#include <inspirecv/core/transform_matrix.h>
#include <inspirecv/task/core/matrix.h>
#include <inspirecv/task/pipeline.h>
#include <inspirecv/task/task_status.h>

#include "inspirecv/task/runtime/pipeline_config.h"

namespace inspirecv {
namespace task {
namespace api_internal {

Status ConfigurePipeline(const PipelineOptions& options,
                         internal::PipelineConfig* destination);
Status ToPublicStatus(TaskStatus status) noexcept;
Matrix ToTaskMatrix(const TransformMatrix& transform);

}  // namespace api_internal
}  // namespace task
}  // namespace inspirecv

#endif  // INSPIRECV_TASK_API_PUBLIC_CONVERSIONS_H_
