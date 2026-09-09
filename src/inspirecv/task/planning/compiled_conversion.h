#ifndef INSPIRECV_TASK_PLANNING_COMPILED_CONVERSION_H_
#define INSPIRECV_TASK_PLANNING_COMPILED_CONVERSION_H_

#include "inspirecv/task/planning/conversion_request.h"
#include "inspirecv/task/planning/pixel_program.h"

namespace inspirecv {
namespace task {
namespace internal {

enum class ExecutionRoute {
    kByteRows,
    kPackedRows,
    kBlockedFloat,
    kPackedTriple,
    kGenericTiles,
    kDrawRegions,
};

struct CompiledConversion : ConversionRequest {
    PixelProgram pixels;
    ExecutionRoute route = ExecutionRoute::kGenericTiles;
    int element_bytes = 1;
    int effective_source_stride = 0;
    int effective_destination_stride = 0;
    int tile_count = 1;
};

TaskStatus CompileConversion(const PipelineConfig& config,
                             const Matrix& destination_to_source,
                             const ConversionRequest& request, bool drawing,
                             CompiledConversion* compiled);

}  // namespace internal
}  // namespace task
}  // namespace inspirecv

#endif  // INSPIRECV_TASK_PLANNING_COMPILED_CONVERSION_H_
