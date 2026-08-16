#ifndef INSPIRECV_TASK_EXECUTION_TILE_PIPELINE_H_
#define INSPIRECV_TASK_EXECUTION_TILE_PIPELINE_H_

#include "inspirecv/task/planning/compiled_conversion.h"

namespace inspirecv {
namespace task {
namespace internal {

TaskStatus RunTilePipeline(const PipelineConfig& config,
                           const CompiledConversion& conversion,
                           const Matrix& destination_to_source,
                           const Matrix& source_to_destination,
                           uint8_t padding, bool drawing,
                           const uint8_t* source, uint8_t* destination,
                           const uint8_t* draw_color,
                           const int32_t* regions);

}  // namespace internal
}  // namespace task
}  // namespace inspirecv

#endif  // INSPIRECV_TASK_EXECUTION_TILE_PIPELINE_H_
