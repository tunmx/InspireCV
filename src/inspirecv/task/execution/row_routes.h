#ifndef INSPIRECV_TASK_EXECUTION_ROW_ROUTES_H_
#define INSPIRECV_TASK_EXECUTION_ROW_ROUTES_H_

#include "inspirecv/task/planning/compiled_conversion.h"

namespace inspirecv {
namespace task {
namespace internal {

TaskStatus CopyByteRows(const CompiledConversion& conversion,
                        const uint8_t* source, void* destination);

TaskStatus ConvertPackedRows(const PipelineConfig& config,
                             const CompiledConversion& conversion,
                             const uint8_t* source, void* destination);

TaskStatus ConvertBlockedRows(const PipelineConfig& config,
                              const CompiledConversion& conversion,
                              const Matrix& destination_to_source,
                              const Matrix& source_to_destination,
                              uint8_t padding, const uint8_t* source,
                              void* destination);

TaskStatus ConvertTripleRows(const PipelineConfig& config,
                             const CompiledConversion& conversion,
                             const Matrix& destination_to_source,
                             const Matrix& source_to_destination,
                             uint8_t padding, const uint8_t* source,
                             void* destination);

}  // namespace internal
}  // namespace task
}  // namespace inspirecv

#endif  // INSPIRECV_TASK_EXECUTION_ROW_ROUTES_H_
