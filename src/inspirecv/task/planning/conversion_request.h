#ifndef INSPIRECV_TASK_PLANNING_CONVERSION_REQUEST_H_
#define INSPIRECV_TASK_PLANNING_CONVERSION_REQUEST_H_

#include <cstddef>
#include <cstdint>

#include <inspirecv/task/core/matrix.h>
#include <inspirecv/task/core/preprocess_types.h>
#include <inspirecv/task/task_status.h>

#include "inspirecv/task/runtime/pipeline_config.h"

namespace inspirecv {
namespace task {
namespace internal {

// Validated, byte-level description of one public Convert invocation.
// Keeping this separate from kernel selection makes malformed dimensions and
// strides impossible to leak into the execution engine.
struct ConversionRequest {
    int source_channels = 0;
    int source_width = 0;
    int source_height = 0;
    int source_stride = 0;

    int destination_channels = 0;
    int destination_width = 0;
    int destination_height = 0;
    int destination_stride = 0;
    halide_type_t destination_type = halide_type_of<uint8_t>();
    TensorLayout destination_layout = TensorLayout::NHWC;
    size_t destination_channel_stride = 0;
};

int ChannelCount(StreamFormat format);

TaskStatus ResolveImageRequest(const PipelineConfig& config, const uint8_t* source,
                               int source_width, int source_height, int source_stride,
                               void* destination, int destination_width, int destination_height,
                               int requested_destination_channels, int destination_stride,
                               halide_type_t destination_type, ConversionRequest* request);

TaskStatus ResolveTensorRequest(const PipelineConfig& config, const uint8_t* source,
                                int source_width, int source_height, int source_stride,
                                const TensorView& output, ConversionRequest* request);

bool IsExactIdentityCopy(const PipelineConfig& config, const Matrix& transform,
                         const ConversionRequest& request);

void CopyExactImage(const uint8_t* source, void* destination, const ConversionRequest& request);

}  // namespace internal
}  // namespace task
}  // namespace inspirecv

#endif  // INSPIRECV_TASK_PLANNING_CONVERSION_REQUEST_H_
