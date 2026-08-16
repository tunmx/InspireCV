#ifndef INSPIRECV_TASK_RUNTIME_CONVERSION_SESSION_H_
#define INSPIRECV_TASK_RUNTIME_CONVERSION_SESSION_H_

#include <cstdint>

#include <inspirecv/task/core/matrix.h>
#include <inspirecv/task/core/preprocess_types.h>
#include <inspirecv/task/acceleration.h>
#include <inspirecv/task/task_status.h>

#include "inspirecv/task/execution/execution_engine.h"
#include "inspirecv/task/runtime/pipeline_config.h"

namespace inspirecv {
namespace task {
namespace internal {

struct CudaExecutor;

// Stateful raw-memory preprocessing session shared by public API facades. It
// owns compiled execution state but never owns source or destination memory.
class ConversionSession {
   public:
    explicit ConversionSession(const PipelineConfig& config);
    ~ConversionSession();

    ConversionSession(const ConversionSession&) = delete;
    ConversionSession& operator=(const ConversionSession&) = delete;

    void SetMatrix(const Matrix& matrix);
    void SetPadding(uint8_t value) { padding_value_ = value; }
    void EnableDrawing() { engine_.EnableDrawing(); }
    ExecutionBackend LastExecutionBackend() const { return last_backend_; }

    TaskStatus Convert(const uint8_t* source, int source_width,
                       int source_height, int source_stride, void* destination,
                       int destination_width, int destination_height,
                       int destination_channels, int destination_stride,
                       halide_type_t destination_type);

    TaskStatus Convert(const uint8_t* source, int source_width,
                       int source_height, int source_stride,
                       const TensorView& destination);

    void DrawRegions(uint8_t* image, int width, int height, int channels,
                     const int* regions, int region_count,
                     const uint8_t* color);

   private:
    TaskStatus Execute(const uint8_t* source, void* destination,
                       const ConversionRequest& request);

    PipelineConfig config_;
    Matrix transform_;
    ExecutionEngine engine_;
    CudaExecutor* cuda_executor_ = nullptr;
    ExecutionBackend last_backend_ = ExecutionBackend::kCpu;
    uint8_t padding_value_ = 0;
};

}  // namespace internal
}  // namespace task
}  // namespace inspirecv

#endif  // INSPIRECV_TASK_RUNTIME_CONVERSION_SESSION_H_
