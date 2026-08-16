#ifndef INSPIRECV_TASK_PLANNING_PIXEL_PROGRAM_H_
#define INSPIRECV_TASK_PLANNING_PIXEL_PROGRAM_H_

#include <cstddef>
#include <cstdint>

#include <inspirecv/task/core/matrix.h>
#include <inspirecv/task/core/preprocess_types.h>
#include <inspirecv/task/task_status.h>

namespace inspirecv {
namespace task {
namespace internal {

using ConvertSpan = void (*)(const uint8_t* source, uint8_t* destination, size_t count);
using WriteInterleavedSpan = void (*)(const uint8_t* source, float* destination,
                                      const float* mean, const float* normal, size_t count);
using WritePlanarSpan = void (*)(const uint8_t* source, float* destination,
                                 size_t channel_stride, const float* mean,
                                 const float* normal, size_t count);
using SampleSpan = void (*)(const uint8_t* source, uint8_t* destination, Point* points,
                            size_t first, size_t count, size_t capacity,
                            size_t source_width, size_t source_height,
                            size_t source_stride);

// One input span presented to a compiled pixel program. `stride_bytes` is
// descriptive metadata for row-based callers; span execution consumes only
// the contiguous pixels beginning at `data`.
struct SourceRow {
    const uint8_t* data = nullptr;
    StreamFormat format = BGR;
    size_t pixel_count = 0;
    int channel_count = 0;
    size_t stride_bytes = 0;
};

// Coordinates for a contiguous output span. The origin names the first
// visible sample, while step is the source-coordinate increment per pixel.
struct SampleLine {
    Point origin = {0.0f, 0.0f};
    Point step = {0.0f, 0.0f};
    size_t first_pixel = 0;
    size_t pixel_count = 0;
    size_t output_capacity = 0;
};

struct DestinationRow {
    void* data = nullptr;
    TensorLayout layout = TensorLayout::NHWC;
    size_t pixel_count = 0;
    int channel_count = 0;
    int element_bytes = 1;
    size_t stride_bytes = 0;
    size_t channel_stride_elements = 0;
};

struct PixelProgramSpec {
    StreamFormat source_format = BGR;
    StreamFormat destination_format = BGR;
    Filter filter = NEAREST;
    Wrap wrap = CLAMP_TO_EDGE;
    halide_type_t destination_type = halide_type_of<uint8_t>();
    TensorLayout destination_layout = TensorLayout::NHWC;
    int source_channels = 0;
    int destination_channels = 0;
    bool direct_sampling = false;
    bool drawing = false;
    bool preserve_identity_float = false;
};

// Immutable after compilation. It is a small composition of one operation
// from each family, rather than a global table mirroring backend symbols.
struct PixelProgram {
    StreamFormat source_format = BGR;
    StreamFormat destination_format = BGR;
    TensorLayout destination_layout = TensorLayout::NHWC;
    int source_channels = 0;
    int destination_channels = 0;

    ConvertSpan convert = nullptr;
    WriteInterleavedSpan write_interleaved = nullptr;
    WritePlanarSpan write_planar = nullptr;
    SampleSpan sample = nullptr;

    bool writes_float() const { return write_interleaved != nullptr || write_planar != nullptr; }
};

TaskStatus CompilePixelProgram(const PixelProgramSpec& specification,
                               PixelProgram* program);

TaskStatus RunSampleSpan(const PixelProgram& program, const uint8_t* source_image,
                         int source_width, int source_height, int source_stride,
                         const SampleLine& line, uint8_t* destination);

TaskStatus RunPixelSpan(const PixelProgram& program, const SourceRow& source,
                        const DestinationRow& destination, const float* mean,
                        const float* normal, uint8_t* temporary,
                        size_t temporary_bytes);

}  // namespace internal
}  // namespace task
}  // namespace inspirecv

#endif  // INSPIRECV_TASK_PLANNING_PIXEL_PROGRAM_H_
