#include "inspirecv/task/planning/compiled_conversion.h"

namespace inspirecv {
namespace task {
namespace internal {
namespace {

bool IsFloat32(halide_type_t type) {
    return type.code == halide_type_float && type.bits == 32 && type.lanes == 1;
}

bool IsUint8(halide_type_t type) {
    return type.code == halide_type_uint && type.bits == 8 && type.lanes == 1;
}

ExecutionRoute ChooseRoute(const PipelineConfig& config,
                           const Matrix& transform,
                           const CompiledConversion& conversion, bool drawing) {
    if (drawing) return ExecutionRoute::kDrawRegions;
    if (transform.isIdentity() && config.source_format == config.destination_format &&
        conversion.source_channels == conversion.destination_channels &&
        conversion.element_bytes == 1) {
        return ExecutionRoute::kByteRows;
    }
    if (transform.isIdentity() && conversion.pixels.sample == nullptr &&
        conversion.source_width >= conversion.destination_width &&
        conversion.source_height >= conversion.destination_height &&
        conversion.source_channels > 0) {
        return ExecutionRoute::kPackedRows;
    }
    const size_t destination_pixels =
      static_cast<size_t>(conversion.destination_width) *
      static_cast<size_t>(conversion.destination_height);
    if (IsFloat32(conversion.destination_type) &&
        conversion.destination_layout == TensorLayout::NC4HW4 &&
        conversion.source_channels == 3 && conversion.destination_channels == 4 &&
        conversion.pixels.sample != nullptr &&
        conversion.pixels.write_interleaved != nullptr &&
        destination_pixels <= 512 * 1024 &&
        conversion.effective_destination_stride ==
          conversion.destination_width * 4 * static_cast<int>(sizeof(float))) {
        return ExecutionRoute::kBlockedFloat;
    }
    if (IsUint8(conversion.destination_type) &&
        conversion.destination_layout == TensorLayout::NHWC &&
        conversion.destination_stride == 0 && conversion.source_channels == 3 &&
        conversion.destination_channels == 3 && conversion.pixels.sample != nullptr) {
        return ExecutionRoute::kPackedTriple;
    }
    return ExecutionRoute::kGenericTiles;
}

}  // namespace

TaskStatus CompileConversion(const PipelineConfig& config,
                             const Matrix& destination_to_source,
                             const ConversionRequest& request, bool drawing,
                             CompiledConversion* compiled) {
    if (compiled == nullptr) return INPUT_DATA_ERROR;

    CompiledConversion candidate;
    static_cast<ConversionRequest&>(candidate) = request;
    candidate.element_bytes = halide_type_bytes(request.destination_type);
    candidate.effective_source_stride = request.source_stride;
    if (candidate.effective_source_stride == 0 && request.source_channels > 0) {
        candidate.effective_source_stride = request.source_width * request.source_channels;
    }
    candidate.effective_destination_stride = request.destination_stride;
    if (candidate.effective_destination_stride == 0) {
        candidate.effective_destination_stride =
          candidate.element_bytes * request.destination_width *
          request.destination_channels;
    }
#if defined(INSPIRECV_TASK_DISABLE_TILING)
    candidate.tile_count = 1;
#else
    candidate.tile_count = (request.destination_width + 255) / 256;
#endif
    if (drawing) candidate.tile_count = 1;

    PixelProgramSpec specification;
    specification.source_format = config.source_format;
    specification.destination_format = config.destination_format;
    specification.filter = config.filter;
    specification.wrap = config.wrap;
    specification.destination_type = request.destination_type;
    specification.destination_layout = request.destination_layout;
    specification.source_channels = request.source_channels;
    specification.destination_channels = request.destination_channels;
    specification.direct_sampling =
      destination_to_source.isIdentity() &&
      request.source_width >= request.destination_width &&
      request.source_height >= request.destination_height;
    specification.drawing = drawing;
    specification.preserve_identity_float = config.preserve_identity_float_order;
    const TaskStatus status = CompilePixelProgram(specification, &candidate.pixels);
    if (status != SUCCESS) return status;

    candidate.route = ChooseRoute(config, destination_to_source, candidate, drawing);
    *compiled = candidate;
    return SUCCESS;
}

}  // namespace internal
}  // namespace task
}  // namespace inspirecv
