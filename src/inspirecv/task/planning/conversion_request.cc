#include "inspirecv/task/planning/conversion_request.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace inspirecv {
namespace task {
namespace internal {

namespace {

bool IsSupportedOutputType(halide_type_t type) {
    const bool is_uint8 = type.code == halide_type_uint && type.bits == 8 && type.lanes == 1;
    const bool is_float32 = type.code == halide_type_float && type.bits == 32 && type.lanes == 1;
    return is_uint8 || is_float32;
}

TaskStatus ValidateSamplingOptions(const PipelineConfig& config) {
    if (config.filter != NEAREST && config.filter != BILINEAR) {
        return UNSUPPORTED_SAMPLER;
    }
    if (config.wrap != CLAMP_TO_EDGE && config.wrap != ZERO) {
        return UNSUPPORTED_SAMPLER;
    }
    return SUCCESS;
}

}  // namespace

int ChannelCount(StreamFormat format) {
    switch (format) {
        case RGB:
        case BGR:
        case YCrCb:
        case YUV:
        case HSV:
        case XYZ:
            return 3;
        case RGBA:
        case BGRA:
            return 4;
        case GRAY:
            return 1;
        case BGR555:
        case BGR565:
            return 2;
        default:
            return 0;
    }
}

TaskStatus ResolveImageRequest(const PipelineConfig& config, const uint8_t* source,
                               int source_width, int source_height, int source_stride,
                               void* destination, int destination_width, int destination_height,
                               int requested_destination_channels, int destination_stride,
                               halide_type_t destination_type, ConversionRequest* request) {
    const int source_channels = ChannelCount(config.source_format);
    const int destination_channels = requested_destination_channels == 0
                                       ? ChannelCount(config.destination_format)
                                       : requested_destination_channels;
    const int destination_element_bytes = halide_type_bytes(destination_type);

    if (!request || !source || !destination || source_width <= 0 || source_height <= 0 ||
        destination_width <= 0 || destination_height <= 0 || source_stride < 0 ||
        destination_stride < 0 || destination_channels <= 0 ||
        !IsSupportedOutputType(destination_type) || destination_element_bytes <= 0) {
        return INPUT_DATA_ERROR;
    }

    const size_t input_row_bytes =
      static_cast<size_t>(source_width) * static_cast<size_t>(std::max(source_channels, 0));
    const size_t destination_width_bytes = static_cast<size_t>(destination_width);
    const size_t destination_channel_count = static_cast<size_t>(destination_channels);
    const size_t destination_element_size = static_cast<size_t>(destination_element_bytes);
    if (destination_channel_count > std::numeric_limits<size_t>::max() / destination_width_bytes ||
        destination_element_size > std::numeric_limits<size_t>::max() /
                                     (destination_width_bytes * destination_channel_count)) {
        return INPUT_DATA_ERROR;
    }
    const size_t output_row_bytes =
      destination_width_bytes * destination_channel_count * destination_element_size;
    if (input_row_bytes > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        output_row_bytes > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        (source_channels > 0 && source_stride > 0 &&
         static_cast<size_t>(source_stride) < input_row_bytes) ||
        (destination_stride > 0 && static_cast<size_t>(destination_stride) < output_row_bytes)) {
        return INPUT_DATA_ERROR;
    }

    const TaskStatus sampling_status = ValidateSamplingOptions(config);
    if (sampling_status != SUCCESS) return sampling_status;

    request->source_channels = source_channels;
    request->source_width = source_width;
    request->source_height = source_height;
    request->source_stride = source_stride;
    request->destination_channels = destination_channels;
    request->destination_width = destination_width;
    request->destination_height = destination_height;
    request->destination_stride = destination_stride;
    request->destination_type = destination_type;
    request->destination_layout = TensorLayout::NHWC;
    request->destination_channel_stride = 0;
    return SUCCESS;
}

TaskStatus ResolveTensorRequest(const PipelineConfig& config, const uint8_t* source,
                                int source_width, int source_height, int source_stride,
                                const TensorView& output, ConversionRequest* request) {
    if (output.layout == TensorLayout::NHWC) {
        if (output.rowStride > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            output.channelStride != 0) {
            return INPUT_DATA_ERROR;
        }
        return ResolveImageRequest(config, source, source_width, source_height, source_stride,
                                   output.data, output.width, output.height, output.channels,
                                   static_cast<int>(output.rowStride), output.type, request);
    }

    const bool is_float32 =
      output.type.code == halide_type_float && output.type.bits == 32 && output.type.lanes == 1;
    const int source_channels = ChannelCount(config.source_format);
    const int expected_channels = ChannelCount(config.destination_format);
    if (!request || !source || !output.data || source_width <= 0 || source_height <= 0 ||
        source_stride < 0 || output.width <= 0 || output.height <= 0 || !is_float32 ||
        (output.channels != 1 && output.channels != 3) || output.channels != expected_channels) {
        return INPUT_DATA_ERROR;
    }

    const bool blocked_c4 = output.layout == TensorLayout::NC4HW4;
    if ((!blocked_c4 && output.layout != TensorLayout::NCHW) ||
        (blocked_c4 && output.channelStride != 0)) {
        return INPUT_DATA_ERROR;
    }

    const size_t input_row_bytes =
      static_cast<size_t>(source_width) * static_cast<size_t>(std::max(source_channels, 0));
    const int stored_channels = blocked_c4 ? 4 : 1;
    const size_t packed_row_bytes =
      static_cast<size_t>(output.width) * static_cast<size_t>(stored_channels) * sizeof(float);
    const size_t row_stride = output.rowStride == 0 ? packed_row_bytes : output.rowStride;
    if (input_row_bytes > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        (source_channels > 0 && source_stride > 0 &&
         static_cast<size_t>(source_stride) < input_row_bytes) ||
        row_stride < packed_row_bytes ||
        row_stride > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        static_cast<size_t>(output.height) > std::numeric_limits<size_t>::max() / row_stride) {
        return INPUT_DATA_ERROR;
    }

    size_t channel_stride = 0;
    if (!blocked_c4) {
        const size_t packed_plane_bytes = row_stride * static_cast<size_t>(output.height);
        channel_stride = output.channelStride == 0 ? packed_plane_bytes : output.channelStride;
        if (channel_stride < packed_plane_bytes ||
            static_cast<size_t>(output.channels) >
              std::numeric_limits<size_t>::max() / channel_stride) {
            return INPUT_DATA_ERROR;
        }
    }

    const TaskStatus sampling_status = ValidateSamplingOptions(config);
    if (sampling_status != SUCCESS) return sampling_status;

    request->source_channels = source_channels;
    request->source_width = source_width;
    request->source_height = source_height;
    request->source_stride = source_stride;
    request->destination_channels = blocked_c4 ? 4 : output.channels;
    request->destination_width = output.width;
    request->destination_height = output.height;
    request->destination_stride = static_cast<int>(row_stride);
    request->destination_type = output.type;
    request->destination_layout = output.layout;
    request->destination_channel_stride = channel_stride;
    return SUCCESS;
}

bool IsExactIdentityCopy(const PipelineConfig& config, const Matrix& transform,
                         const ConversionRequest& request) {
    return config.source_format == config.destination_format &&
           request.destination_channels == request.source_channels &&
           halide_type_bytes(request.destination_type) == 1 &&
           request.destination_width == request.source_width &&
           request.destination_height == request.source_height && transform.isIdentity();
}

void CopyExactImage(const uint8_t* source, void* destination, const ConversionRequest& request) {
    const int row_bytes = request.destination_width * request.destination_channels;
    const int source_stride = request.source_stride == 0 ? row_bytes : request.source_stride;
    const int destination_stride =
      request.destination_stride == 0 ? row_bytes : request.destination_stride;
    auto* destination_bytes = static_cast<uint8_t*>(destination);
    if (source_stride == row_bytes && destination_stride == row_bytes) {
        ::memcpy(destination_bytes, source,
                 static_cast<size_t>(row_bytes) * request.destination_height);
        return;
    }
    for (int y = 0; y < request.destination_height; ++y) {
        ::memcpy(destination_bytes + static_cast<size_t>(y) * destination_stride,
                 source + static_cast<size_t>(y) * source_stride, row_bytes);
    }
}

}  // namespace internal
}  // namespace task
}  // namespace inspirecv
