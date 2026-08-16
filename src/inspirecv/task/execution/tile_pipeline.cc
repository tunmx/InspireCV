#include "inspirecv/task/execution/tile_pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>
#include <vector>

#include <inspirecv/task/core/st_defs.h>

#include "inspirecv/task/execution/row_schedule.h"

namespace inspirecv {
namespace task {
namespace internal {
namespace {

constexpr int kTileWidth = 256;
constexpr size_t kParallelThreshold = 512 * 1024;

struct TileScratch {
    uint8_t sampled[4 * kTileWidth];
    uint8_t converted[4 * kTileWidth];
};

class TileWorker {
   public:
    TileWorker(const PipelineConfig& config,
               const CompiledConversion& conversion,
               const Matrix& destination_to_source,
               const Matrix& source_to_destination, uint8_t padding,
               bool drawing, const uint8_t* source, uint8_t* destination,
               const uint8_t* draw_color, const int32_t* regions)
        : config_(config), conversion_(conversion),
          destination_to_source_(destination_to_source),
          source_to_destination_(source_to_destination), padding_(padding),
          drawing_(drawing), source_(source), destination_(destination),
          draw_color_(draw_color), regions_(regions) {}

    void RunRows(int first_row, int last_row) const {
        TileScratch scratch;
        for (int row = first_row; row < last_row; ++row) {
            const int output_y = drawing_ ? regions_[3 * row] : row;
            for (int tile = 0; tile < conversion_.tile_count; ++tile) {
                int output_x = tile * kTileWidth;
                int count =
                  std::min(kTileWidth, conversion_.destination_width - output_x);
                if (drawing_) {
                    output_x = regions_[3 * row + 1];
                    count = regions_[3 * row + 2] - output_x + 1;
                }
                RunTile(output_y, output_x, count, &scratch);
            }
        }
    }

   private:
    void PadOutside(const RowWindow& window, int count, uint8_t* sample) const {
        if (config_.wrap != ZERO ||
            (window.first == 0 && window.last == count)) {
            return;
        }
        if (conversion_.source_channels == 0) {
            std::memset(sample, padding_, count);
            std::memset(sample + count, 128, ((count + 1) / 2) * 2);
            return;
        }
        if (window.first > 0) {
            std::memset(sample, padding_,
                        conversion_.source_channels * window.first);
        }
        if (window.last < count) {
            std::memset(sample + window.last * conversion_.source_channels,
                        padding_,
                        (count - window.last) * conversion_.source_channels);
        }
    }

    const uint8_t* ResolveInput(int output_y, int output_x, int count,
                                uint8_t* sample) const {
        if (drawing_) return draw_color_;
        const RowWindow window = ScheduleRow(
          destination_to_source_, source_to_destination_, config_.wrap,
          conversion_.source_width, conversion_.source_height, output_y,
          output_x, count);
        PadOutside(window, count, sample);
        if (conversion_.pixels.sample != nullptr) {
            const SampleLine line = {
              window.origin, window.step, static_cast<size_t>(window.first),
              static_cast<size_t>(window.last - window.first),
              static_cast<size_t>(count)};
            const TaskStatus status = RunSampleSpan(
              conversion_.pixels, source_, conversion_.source_width,
              conversion_.source_height, conversion_.effective_source_stride,
              line, sample);
            INSPIRECV_TASK_ASSERT(status == SUCCESS);
            return sample;
        }
        const float x = std::max(
          0.0f, std::min(window.origin.fX,
                          static_cast<float>(conversion_.source_width - 1)));
        const float y = std::max(
          0.0f, std::min(window.origin.fY,
                          static_cast<float>(conversion_.source_height - 1)));
        return source_ +
          static_cast<size_t>(static_cast<int>(std::round(y))) *
            conversion_.effective_source_stride +
          static_cast<size_t>(static_cast<int>(std::round(x))) *
            conversion_.source_channels;
    }

    void RunTile(int output_y, int output_x, int count,
                 TileScratch* scratch) const {
        const bool planar =
          conversion_.destination_layout == TensorLayout::NCHW;
        uint8_t* output = destination_ +
          static_cast<size_t>(output_y) *
            conversion_.effective_destination_stride +
          static_cast<size_t>(output_x) * conversion_.element_bytes *
            (planar ? 1 : conversion_.destination_channels);
        uint8_t* temporary = conversion_.pixels.writes_float()
                               ? scratch->converted
                               : output;
        uint8_t* sample = conversion_.pixels.convert == nullptr
                            ? temporary
                            : (drawing_ ? const_cast<uint8_t*>(draw_color_)
                                        : scratch->sampled);
        const uint8_t* input = ResolveInput(output_y, output_x, count, sample);
        const SourceRow source_row = {
          input, config_.source_format, static_cast<size_t>(count),
          conversion_.source_channels, 0};
        const DestinationRow destination_row = {
          output, conversion_.destination_layout, static_cast<size_t>(count),
          conversion_.destination_channels, conversion_.element_bytes, 0,
          conversion_.destination_channel_stride / sizeof(float)};
        const TaskStatus status = RunPixelSpan(
          conversion_.pixels, source_row, destination_row, config_.mean,
          config_.scale, temporary, sizeof(scratch->converted));
        INSPIRECV_TASK_ASSERT(status == SUCCESS);
    }

    const PipelineConfig& config_;
    const CompiledConversion& conversion_;
    const Matrix& destination_to_source_;
    const Matrix& source_to_destination_;
    uint8_t padding_;
    bool drawing_;
    const uint8_t* source_;
    uint8_t* destination_;
    const uint8_t* draw_color_;
    const int32_t* regions_;
};

unsigned ChooseWorkerCount(const CompiledConversion& conversion, bool drawing) {
    const size_t pixels = static_cast<size_t>(conversion.destination_width) *
                          conversion.destination_height;
    if (drawing || conversion.destination_height <= 1 ||
        pixels <= kParallelThreshold) {
        return 1;
    }
    const unsigned detected = std::thread::hardware_concurrency();
    return std::min<unsigned>(detected == 0 ? 4 : detected,
                              conversion.destination_height);
}

}  // namespace

TaskStatus RunTilePipeline(const PipelineConfig& config,
                           const CompiledConversion& conversion,
                           const Matrix& destination_to_source,
                           const Matrix& source_to_destination,
                           uint8_t padding, bool drawing,
                           const uint8_t* source, uint8_t* destination,
                           const uint8_t* draw_color,
                           const int32_t* regions) {
    const TileWorker worker(config, conversion, destination_to_source,
                            source_to_destination, padding, drawing, source,
                            destination, draw_color, regions);
    const unsigned worker_count = ChooseWorkerCount(conversion, drawing);
    if (worker_count == 1) {
        worker.RunRows(0, conversion.destination_height);
        return SUCCESS;
    }

    std::vector<std::thread> threads;
    threads.reserve(worker_count);
    const int rows_per_worker =
      (conversion.destination_height + static_cast<int>(worker_count) - 1) /
      static_cast<int>(worker_count);
    for (unsigned index = 0; index < worker_count; ++index) {
        const int first = static_cast<int>(index) * rows_per_worker;
        const int last =
          std::min(conversion.destination_height, first + rows_per_worker);
        if (first >= last) break;
        threads.emplace_back([&worker, first, last]() {
            worker.RunRows(first, last);
        });
    }
    for (std::thread& thread : threads) thread.join();
    return SUCCESS;
}

}  // namespace internal
}  // namespace task
}  // namespace inspirecv
