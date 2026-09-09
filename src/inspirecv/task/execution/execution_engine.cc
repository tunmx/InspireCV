#include "inspirecv/task/execution/execution_engine.h"

#include "inspirecv/task/execution/row_routes.h"
#include "inspirecv/task/execution/tile_pipeline.h"

namespace inspirecv {
namespace task {
namespace internal {
ExecutionEngine::ExecutionEngine(const PipelineConfig& config) : config_(config) {}

TaskStatus ExecutionEngine::Compile(const ConversionRequest& request) {
    CompiledConversion candidate;
    const TaskStatus status =
      CompileConversion(config_, transform_, request, drawing_enabled_, &candidate);
    if (status == SUCCESS) plan_ = candidate;
    return status;
}

void ExecutionEngine::SetMatrix(const Matrix& matrix) {
    transform_ = matrix;
    transform_.invert(&inverse_transform_);
}

TaskStatus ExecutionEngine::Execute(const uint8_t* source, void* destination) {
    switch (plan_.route) {
        case ExecutionRoute::kByteRows:
            return CopyByteRows(plan_, source, destination);
        case ExecutionRoute::kPackedRows:
            return ConvertPackedRows(config_, plan_, source, destination);
        case ExecutionRoute::kBlockedFloat:
            return ConvertBlockedRows(config_, plan_, transform_, inverse_transform_,
                                      padding_value_, source, destination);
        case ExecutionRoute::kPackedTriple:
            return ConvertTripleRows(config_, plan_, transform_, inverse_transform_,
                                     padding_value_, source, destination);
        case ExecutionRoute::kGenericTiles:
            return RunTilePipeline(config_, plan_, transform_, inverse_transform_,
                                   padding_value_, false, source,
                                   static_cast<uint8_t*>(destination), nullptr,
                                   nullptr);
        case ExecutionRoute::kDrawRegions:
            return INPUT_DATA_ERROR;
    }
    return INPUT_DATA_ERROR;
}

void ExecutionEngine::DrawRegions(uint8_t* image, const int* regions,
                                  int region_count, const uint8_t* color) {
    if (plan_.route != ExecutionRoute::kDrawRegions) return;
    RunTilePipeline(config_, plan_, transform_, inverse_transform_,
                    padding_value_, true, image, image, color, regions);
}

}  // namespace internal
}  // namespace task
}  // namespace inspirecv
