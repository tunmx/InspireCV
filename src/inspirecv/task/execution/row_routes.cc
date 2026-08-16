#include "inspirecv/task/execution/row_routes.h"

#include <algorithm>
#include <cstring>

#include "inspirecv/task/platform/cpu_features.h"
#include "inspirecv/task/execution/row_schedule.h"

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

#if defined(__x86_64__) || defined(__i386__)
extern "C" void inspirecv_task_copy_avx2(void*, const void*, size_t);
#endif

namespace inspirecv {
namespace task {
namespace internal {
namespace {

void CopyContiguous(uint8_t* destination, const uint8_t* source, size_t bytes) {
#if defined(__x86_64__) || defined(__i386__)
    if (platform::HasAvx2()) {
        inspirecv_task_copy_avx2(destination, source, bytes);
        return;
    }
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    while (bytes >= 64) {
        const uint8x16_t block0 = vld1q_u8(source);
        const uint8x16_t block1 = vld1q_u8(source + 16);
        const uint8x16_t block2 = vld1q_u8(source + 32);
        const uint8x16_t block3 = vld1q_u8(source + 48);
        vst1q_u8(destination, block0);
        vst1q_u8(destination + 16, block1);
        vst1q_u8(destination + 32, block2);
        vst1q_u8(destination + 48, block3);
        source += 64;
        destination += 64;
        bytes -= 64;
    }
    while (bytes >= 16) {
        vst1q_u8(destination, vld1q_u8(source));
        source += 16;
        destination += 16;
        bytes -= 16;
    }
#endif
    if (bytes != 0) std::memcpy(destination, source, bytes);
}

}  // namespace

TaskStatus CopyByteRows(const CompiledConversion& conversion,
                        const uint8_t* source, void* destination) {
    const int row_bytes =
      conversion.destination_width * conversion.destination_channels;
    uint8_t* output = static_cast<uint8_t*>(destination);
    if (conversion.destination_width == conversion.source_width &&
        conversion.effective_source_stride == row_bytes &&
        conversion.effective_destination_stride == row_bytes) {
        CopyContiguous(output, source,
                       static_cast<size_t>(row_bytes) *
                         conversion.destination_height);
        return SUCCESS;
    }
    for (int row = 0; row < conversion.destination_height; ++row) {
#if defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(source + conversion.effective_source_stride, 0, 1);
#endif
        std::memcpy(output, source, row_bytes);
        source += conversion.effective_source_stride;
        output += conversion.effective_destination_stride;
    }
    return SUCCESS;
}

TaskStatus ConvertPackedRows(const PipelineConfig& config,
                             const CompiledConversion& conversion,
                             const uint8_t* source, void* destination) {
    constexpr int kTileWidth = 256;
    const bool planar = conversion.destination_layout == TensorLayout::NCHW;
    uint8_t converted[4 * kTileWidth];
    uint8_t* output = static_cast<uint8_t*>(destination);
    for (int row = 0; row < conversion.destination_height; ++row) {
        const uint8_t* input_row =
          source + static_cast<size_t>(row) * conversion.effective_source_stride;
        uint8_t* output_row =
          output + static_cast<size_t>(row) * conversion.effective_destination_stride;
        for (int column = 0; column < conversion.destination_width;
             column += kTileWidth) {
            const int count =
              std::min(kTileWidth, conversion.destination_width - column);
            const uint8_t* input =
              input_row + static_cast<size_t>(column) * conversion.source_channels;
            uint8_t* result = output_row +
              static_cast<size_t>(column) * conversion.element_bytes *
                (planar ? 1 : conversion.destination_channels);
            const uint8_t* writer_input = input;
            if (conversion.pixels.convert != nullptr) {
                if (!conversion.pixels.writes_float()) {
                    conversion.pixels.convert(input, result, count);
                    continue;
                }
                conversion.pixels.convert(input, converted, count);
                writer_input = converted;
            }
            if (conversion.pixels.write_interleaved != nullptr) {
                conversion.pixels.write_interleaved(
                  writer_input, reinterpret_cast<float*>(result), config.mean,
                  config.scale, count);
            } else if (conversion.pixels.write_planar != nullptr) {
                conversion.pixels.write_planar(
                  writer_input, reinterpret_cast<float*>(result),
                  conversion.destination_channel_stride / sizeof(float),
                  config.mean, config.scale, count);
            }
        }
    }
    return SUCCESS;
}

TaskStatus ConvertBlockedRows(const PipelineConfig& config,
                              const CompiledConversion& conversion,
                              const Matrix& destination_to_source,
                              const Matrix& source_to_destination,
                              uint8_t padding, const uint8_t* source,
                              void* destination) {
    constexpr int kTileWidth = 256;
    uint8_t sampled[4 * kTileWidth];
    uint8_t converted[4 * kTileWidth];
    uint8_t* output = static_cast<uint8_t*>(destination);
    for (int row = 0; row < conversion.destination_height; ++row) {
        float* output_row = reinterpret_cast<float*>(
          output + static_cast<size_t>(row) * conversion.effective_destination_stride);
        for (int column = 0; column < conversion.destination_width;
             column += kTileWidth) {
            const int count =
              std::min(kTileWidth, conversion.destination_width - column);
            uint8_t* sample = conversion.pixels.convert != nullptr ? sampled : converted;
            const RowWindow window = ScheduleRow(
              destination_to_source, source_to_destination, config.wrap,
              conversion.source_width, conversion.source_height, row, column, count);
            if (config.wrap == ZERO) {
                if (window.first > 0) {
                    std::memset(sample, padding, 3 * window.first);
                }
                if (window.last < count) {
                    std::memset(sample + 3 * window.last, padding,
                                3 * (count - window.last));
                }
            }
            Point coordinates[2] = {window.origin, window.step};
            conversion.pixels.sample(
              source, sample, coordinates, window.first,
              window.last - window.first, count, conversion.source_width,
              conversion.source_height, conversion.effective_source_stride);
            const uint8_t* writer_input = sample;
            if (conversion.pixels.convert != nullptr) {
                conversion.pixels.convert(sample, converted, count);
                writer_input = converted;
            }
            conversion.pixels.write_interleaved(
              writer_input, output_row + 4 * column, config.mean, config.scale,
              count);
        }
    }
    return SUCCESS;
}

TaskStatus ConvertTripleRows(const PipelineConfig& config,
                             const CompiledConversion& conversion,
                             const Matrix& destination_to_source,
                             const Matrix& source_to_destination,
                             uint8_t padding, const uint8_t* source,
                             void* destination) {
    constexpr int kTileWidth = 256;
    uint8_t sampled[3 * kTileWidth];
    uint8_t* output = static_cast<uint8_t*>(destination);
    for (int row = 0; row < conversion.destination_height; ++row) {
        uint8_t* output_row =
          output + static_cast<size_t>(row) * conversion.effective_destination_stride;
        for (int column = 0; column < conversion.destination_width;
             column += kTileWidth) {
            const int count =
              std::min(kTileWidth, conversion.destination_width - column);
            uint8_t* result = output_row + 3 * column;
            uint8_t* sample = conversion.pixels.convert != nullptr ? sampled : result;
            const RowWindow window = ScheduleRow(
              destination_to_source, source_to_destination, config.wrap,
              conversion.source_width, conversion.source_height, row, column, count);
            if (config.wrap == ZERO) {
                if (window.first > 0) {
                    std::memset(sample, padding, 3 * window.first);
                }
                if (window.last < count) {
                    std::memset(sample + 3 * window.last, padding,
                                3 * (count - window.last));
                }
            }
            Point coordinates[2] = {window.origin, window.step};
            conversion.pixels.sample(
              source, sample, coordinates, window.first,
              window.last - window.first, count, conversion.source_width,
              conversion.source_height, conversion.effective_source_stride);
            if (conversion.pixels.convert != nullptr) {
                conversion.pixels.convert(sample, result, count);
            }
        }
    }
    return SUCCESS;
}

}  // namespace internal
}  // namespace task
}  // namespace inspirecv
