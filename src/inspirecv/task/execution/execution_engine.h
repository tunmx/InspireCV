#ifndef INSPIRECV_TASK_EXECUTION_EXECUTION_ENGINE_H_
#define INSPIRECV_TASK_EXECUTION_EXECUTION_ENGINE_H_

#include <cstddef>
#include <cstdint>

#include <inspirecv/task/core/matrix.h>
#include <inspirecv/task/task_status.h>

#include "inspirecv/task/planning/compiled_conversion.h"
#include "inspirecv/task/runtime/pipeline_config.h"

namespace inspirecv {
namespace task {
namespace internal {

// Private execution engine behind the public StreamTask facade. It owns no
// input or output image memory and performs no framework-specific allocation.
class ExecutionEngine {
   public:
    explicit ExecutionEngine(const PipelineConfig& config);

    ExecutionEngine(const ExecutionEngine&) = delete;
    ExecutionEngine& operator=(const ExecutionEngine&) = delete;

    void SetMatrix(const Matrix& matrix);
    void SetPadding(uint8_t value) { padding_value_ = value; }
    void EnableDrawing() { drawing_enabled_ = true; }

    TaskStatus Compile(const ConversionRequest& request);

    TaskStatus Execute(const uint8_t* source, void* destination);

    void DrawRegions(uint8_t* image, const int* regions, int region_count,
                     const uint8_t* color);

   private:
    const PipelineConfig config_;
    Matrix transform_;
    Matrix inverse_transform_;
    CompiledConversion plan_;
    uint8_t padding_value_ = 0;
    bool drawing_enabled_ = false;
};

}  // namespace internal
}  // namespace task
}  // namespace inspirecv

#endif  // INSPIRECV_TASK_EXECUTION_EXECUTION_ENGINE_H_
