#include "inspirecv/task/planning/pixel_program.h"

#include <cstring>
#include <limits>
#include <type_traits>

#include "inspirecv/task/planning/kernel_registry.h"

namespace inspirecv {
namespace task {
namespace internal {

namespace {

// Invalid enum values can arrive at this validation boundary from legacy
// callers. Reading such an enum as an enum value is undefined behaviour under
// Clang's enum sanitizer, so inspect its stored representation first and only
// use the typed value after it has been accepted.
template <typename Enum>
typename std::underlying_type<Enum>::type EnumStorage(const Enum& value) {
    typename std::underlying_type<Enum>::type storage = 0;
    static_assert(sizeof(storage) == sizeof(value),
                  "enum and underlying storage must have the same size");
    std::memcpy(&storage, &value, sizeof(storage));
    return storage;
}

}  // namespace

TaskStatus CompilePixelProgram(const PixelProgramSpec& specification,
                               PixelProgram* program) {
    if (program == nullptr) {
        return INPUT_DATA_ERROR;
    }

    PixelProgram compiled;
    compiled.source_format = specification.source_format;
    compiled.destination_format = specification.destination_format;
    compiled.destination_layout = specification.destination_layout;
    compiled.source_channels = specification.source_channels;
    compiled.destination_channels = specification.destination_channels;

    if (specification.drawing) {
        compiled.convert = FindDrawWriter(
          specification.source_channels * halide_type_bytes(specification.destination_type));
        if (compiled.convert == nullptr) {
            return UNSUPPORTED_CONVERSION;
        }
        *program = compiled;
        return SUCCESS;
    }

    const auto filter = EnumStorage(specification.filter);
    if (filter != static_cast<decltype(filter)>(NEAREST) &&
        filter != static_cast<decltype(filter)>(BILINEAR)) {
        return UNSUPPORTED_SAMPLER;
    }
    const auto wrap = EnumStorage(specification.wrap);
    if (wrap != static_cast<decltype(wrap)>(CLAMP_TO_EDGE) &&
        wrap != static_cast<decltype(wrap)>(ZERO)) {
        return UNSUPPORTED_SAMPLER;
    }

    const bool yuv420 = specification.source_format == YUV_NV12 ||
                        specification.source_format == YUV_NV21 ||
                        specification.source_format == YUV_I420;
    if (!specification.direct_sampling || yuv420) {
        compiled.sample = FindSampler(specification.source_format,
                                      specification.filter,
                                      specification.direct_sampling);
        if (compiled.sample == nullptr) {
            return UNSUPPORTED_SAMPLER;
        }
    }

    const bool keepIdentityChannels =
      specification.preserve_identity_float && specification.direct_sampling &&
      specification.destination_type.code == halide_type_float &&
      specification.source_channels == 3 &&
      (specification.destination_channels == 3 ||
       (specification.destination_layout == TensorLayout::NC4HW4 &&
        specification.destination_channels == 4)) &&
      (specification.source_format == RGB || specification.source_format == BGR) &&
      (specification.destination_format == RGB || specification.destination_format == BGR);
    if (specification.source_format != specification.destination_format &&
        !keepIdentityChannels) {
        compiled.convert = FindColorConverter(specification.source_format,
                                              specification.destination_format);
        if (compiled.convert == nullptr) {
            return UNSUPPORTED_CONVERSION;
        }
    }

    if (specification.destination_type.code == halide_type_float) {
        if (specification.destination_layout == TensorLayout::NCHW &&
            specification.destination_channels > 1) {
            compiled.write_planar = FindPlanarFloatWriter(
              specification.destination_format, specification.destination_channels);
            if (compiled.write_planar == nullptr) {
                return UNSUPPORTED_FLOAT_CONVERSION;
            }
        } else {
            compiled.write_interleaved = FindInterleavedFloatWriter(
              specification.destination_format, specification.destination_channels);
            if (compiled.write_interleaved == nullptr) {
                return UNSUPPORTED_FLOAT_CONVERSION;
            }
        }
    }

    *program = compiled;
    return SUCCESS;
}

TaskStatus RunSampleSpan(const PixelProgram& program, const uint8_t* source_image,
                         int source_width, int source_height, int source_stride,
                         const SampleLine& line, uint8_t* destination) {
    if (program.sample == nullptr || source_image == nullptr || destination == nullptr ||
        source_width <= 0 || source_height <= 0 || source_stride < 0 ||
        line.first_pixel > line.output_capacity ||
        line.pixel_count > line.output_capacity - line.first_pixel) {
        return INPUT_DATA_ERROR;
    }

    Point legacyCoordinates[2] = {line.origin, line.step};
    program.sample(source_image, destination, legacyCoordinates,
                   line.first_pixel, line.pixel_count, line.output_capacity,
                   static_cast<size_t>(source_width), static_cast<size_t>(source_height),
                   static_cast<size_t>(source_stride));
    return SUCCESS;
}

TaskStatus RunPixelSpan(const PixelProgram& program, const SourceRow& source,
                        const DestinationRow& destination, const float* mean,
                        const float* normal, uint8_t* temporary,
                        size_t temporary_bytes) {
    if (source.data == nullptr || destination.data == nullptr ||
        source.pixel_count != destination.pixel_count || source.channel_count < 0 ||
        destination.channel_count <= 0 || destination.element_bytes <= 0) {
        return INPUT_DATA_ERROR;
    }

    const size_t count = source.pixel_count;
    const bool writesFloat = program.writes_float();
    if (writesFloat &&
        (destination.element_bytes != static_cast<int>(sizeof(float)) ||
         mean == nullptr || normal == nullptr ||
         (program.write_planar != nullptr &&
          destination.channel_stride_elements < count))) {
        return INPUT_DATA_ERROR;
    }

    const bool needsTemporary = program.convert != nullptr && writesFloat;
    if (needsTemporary &&
        (count > std::numeric_limits<size_t>::max() / 4 ||
         temporary == nullptr || temporary_bytes < count * 4)) {
        return INPUT_DATA_ERROR;
    }

    const uint8_t* writerSource = source.data;
    if (program.convert != nullptr) {
        uint8_t* converted = needsTemporary
                               ? temporary
                               : static_cast<uint8_t*>(destination.data);
        program.convert(source.data, converted, count);
        writerSource = converted;
    }

    if (program.write_interleaved != nullptr) {
        program.write_interleaved(writerSource, static_cast<float*>(destination.data),
                                  mean, normal, count);
    } else if (program.write_planar != nullptr) {
        program.write_planar(writerSource, static_cast<float*>(destination.data),
                             destination.channel_stride_elements, mean, normal, count);
    } else if (program.convert == nullptr && source.data != destination.data) {
        if (source.channel_count == 0 || source.channel_count != destination.channel_count ||
            destination.element_bytes != 1 ||
            count > std::numeric_limits<size_t>::max() /
                      static_cast<size_t>(source.channel_count)) {
            return INPUT_DATA_ERROR;
        }
        const size_t bytes = count * static_cast<size_t>(source.channel_count);
        std::memmove(destination.data, source.data, bytes);
    }
    return SUCCESS;
}

}  // namespace internal
}  // namespace task
}  // namespace inspirecv
