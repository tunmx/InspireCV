#include "inspirecv/task/cuda/cuda_auto_policy.h"

#include <cstdint>

namespace inspirecv {
namespace task {
namespace internal {
namespace {

struct ThresholdAnchor {
    size_t destination_pixels;
    size_t maximum_source_bytes;
};

// Profile: NVIDIA GeForce RTX 3060 (SM 8.6, 12 GiB)
// Host: AMD Ryzen 5 5600, Linux 6.8.0-124-generic
// CUDA runtime: 12.2; driver: 550.144.03 (driver API 12.4)
// Compiler: GCC 11.4.0, Release
// Rule: CUDA p50 is at least 15% faster and CUDA p95 is no slower than CPU.
// Raw measurements live in benchmarks/cuda/task_host_rtx3060_cuda12_2.csv.
// These conservative values are the universal default. The RTX 3060 is their
// benchmark baseline, not a runtime hardware requirement.
constexpr ThresholdAnchor kDefaultThresholds[] = {
  {112u * 112u, 640u * 480u * 3u},
  {224u * 224u, 1920u * 1080u * 3u},
  {640u * 640u, 3840u * 2160u * 3u},
};

size_t SourceStorageBytes(const ConversionRequest& request) {
    const size_t row_bytes = request.source_stride == 0
                               ? static_cast<size_t>(request.source_width) * 3
                               : static_cast<size_t>(request.source_stride);
    return row_bytes * static_cast<size_t>(request.source_height);
}

}  // namespace

size_t DefaultRecommendedMaxSourceBytes(
  size_t destination_pixels) noexcept {
    if (destination_pixels < kDefaultThresholds[0].destination_pixels) {
        return 0;
    }
    for (size_t index = 1;
         index < sizeof(kDefaultThresholds) / sizeof(kDefaultThresholds[0]);
         ++index) {
        const ThresholdAnchor& lower = kDefaultThresholds[index - 1];
        const ThresholdAnchor& upper = kDefaultThresholds[index];
        if (destination_pixels > upper.destination_pixels) continue;
        const uint64_t output_offset = destination_pixels -
                                       lower.destination_pixels;
        const uint64_t output_span = upper.destination_pixels -
                                     lower.destination_pixels;
        const uint64_t source_span = upper.maximum_source_bytes -
                                     lower.maximum_source_bytes;
        return lower.maximum_source_bytes +
               static_cast<size_t>(source_span * output_offset / output_span);
    }
    // Do not extrapolate beyond the largest measured 4K-input case.
    return kDefaultThresholds[
      sizeof(kDefaultThresholds) / sizeof(kDefaultThresholds[0]) - 1]
      .maximum_source_bytes;
}

bool CudaAutoRecommendsHost(
  const PipelineConfig& config, const Matrix& destination_to_source,
  const ConversionRequest& request) noexcept {
    // This profile is intentionally narrow: only configurations represented
    // by the stored benchmark may use it. Nearest sampling and HWC writing
    // have different CPU crossover points and need separate profiles.
    if (config.filter != BILINEAR || config.source_format != BGR ||
        config.destination_format != RGB ||
        request.destination_layout != TensorLayout::NCHW ||
        request.destination_width != request.destination_height) {
        return false;
    }
    // Identity preprocessing has a much cheaper CPU route. It remained faster
    // at every measured size through 1024x1024.
    if (destination_to_source.isIdentity()) return false;

    const size_t destination_pixels =
      static_cast<size_t>(request.destination_width) *
      request.destination_height;
    const size_t maximum_source_bytes =
      DefaultRecommendedMaxSourceBytes(destination_pixels);
    return maximum_source_bytes != 0 &&
           SourceStorageBytes(request) <= maximum_source_bytes;
}

}  // namespace internal
}  // namespace task
}  // namespace inspirecv
