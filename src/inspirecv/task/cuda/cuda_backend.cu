#include "inspirecv/task/cuda/cuda_dispatch.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <new>
#include <vector>

#include "inspirecv/task/execution/row_schedule.h"

namespace inspirecv {
namespace task {
namespace internal {
namespace {

constexpr int kCoordinateTileWidth = 256;

struct Coordinate {
    float x;
    float y;
};

struct KernelParameters {
    int source_width;
    int source_height;
    int source_stride;
    int destination_width;
    int destination_height;
    size_t destination_row_stride;
    size_t destination_channel_stride;
    int destination_layout;
    int filter;
    int swap_red_blue;
    float mean[3];
    float scale[3];
};

struct YuvKernelParameters {
    int source_width;
    int source_height;
    int source_stride;
    int source_chroma_stride;
    int source_format;
    int destination_width;
    int destination_height;
    size_t destination_row_stride;
    size_t destination_channel_stride;
    int destination_layout;
    int destination_blue_first;
    float mean[3];
    float scale[3];
};

struct GeometryKey {
    int source_width = 0;
    int source_height = 0;
    int destination_width = 0;
    int destination_height = 0;
    float matrix[9] = {};
    bool valid = false;
};

__device__ float ClampCoordinate(float value, int limit) {
    return fminf(fmaxf(value, 0.0f), static_cast<float>(limit - 1));
}

__device__ uint8_t SampleNearest(const uint8_t* source,
                                 const KernelParameters& parameters,
                                 Coordinate coordinate, int channel) {
    const float x = ClampCoordinate(coordinate.x, parameters.source_width);
    const float y = ClampCoordinate(coordinate.y, parameters.source_height);
    const int source_x = static_cast<int>(roundf(x));
    const int source_y = static_cast<int>(roundf(y));
    return source[static_cast<size_t>(source_y) * parameters.source_stride +
                  3 * source_x + channel];
}

__device__ uint8_t SampleLinear(const uint8_t* source,
                                const KernelParameters& parameters,
                                Coordinate coordinate, int channel) {
    const float x = ClampCoordinate(coordinate.x, parameters.source_width);
    const float y = ClampCoordinate(coordinate.y, parameters.source_height);
    const int x0 = static_cast<int>(x);
    const int y0 = static_cast<int>(y);
    const int x1 = static_cast<int>(ceilf(x));
    const int y1 = static_cast<int>(ceilf(y));
    const float fraction_x = x - static_cast<float>(x0);
    const float fraction_y = y - static_cast<float>(y0);
    const size_t offset00 =
      static_cast<size_t>(y0) * parameters.source_stride + 3 * x0 + channel;
    const size_t offset01 =
      static_cast<size_t>(y0) * parameters.source_stride + 3 * x1 + channel;
    const size_t offset10 =
      static_cast<size_t>(y1) * parameters.source_stride + 3 * x0 + channel;
    const size_t offset11 =
      static_cast<size_t>(y1) * parameters.source_stride + 3 * x1 + channel;
    float value =
      (1.0f - fraction_x) * (1.0f - fraction_y) * source[offset00] +
      fraction_x * (1.0f - fraction_y) * source[offset01] +
      // Preserve the CPU scalar path's double promotion at rounding ties.
      fraction_y * (1.0 - fraction_x) * source[offset10] +
      fraction_x * fraction_y * source[offset11];
    value = fminf(fmaxf(value, 0.0f), 255.0f);
    return static_cast<uint8_t>(roundf(value));
}

__device__ int ClampInteger(int value, int low, int high) {
    return max(low, min(value, high));
}

struct YuvColor {
    int red;
    int green;
    int blue;
};

__device__ YuvColor DecodeYuv(uint8_t luma, uint8_t v, uint8_t u) {
    const int scaled_luma = static_cast<int>(luma) << 6;
    const int centered_u = static_cast<int>(u) - 128;
    const int centered_v = static_cast<int>(v) - 128;
    YuvColor color;
    color.red = ClampInteger((scaled_luma + 73 * centered_v) >> 6, 0, 255);
    color.green = ClampInteger(
      (scaled_luma - 25 * centered_u - 37 * centered_v) >> 6, 0, 255);
    color.blue = ClampInteger((scaled_luma + 130 * centered_u) >> 6, 0, 255);
    return color;
}

__device__ uint8_t SampleYuv(const uint8_t* source,
                             const Coordinate* coordinates,
                             const Coordinate* chroma_coordinates,
                             const YuvKernelParameters& parameters,
                             int destination_x, int destination_y,
                             int destination_channel) {
    const size_t pixel = static_cast<size_t>(destination_y) *
                           parameters.destination_width +
                         destination_x;
    const Coordinate luma_coordinate = coordinates[pixel];
    const int source_x = static_cast<int>(roundf(
      ClampCoordinate(luma_coordinate.x, parameters.source_width)));
    const int source_y = static_cast<int>(roundf(
      ClampCoordinate(luma_coordinate.y, parameters.source_height)));
    const uint8_t luma =
      source[static_cast<size_t>(source_y) * parameters.source_stride +
             source_x];

    const bool i420 = parameters.source_format == YUV_I420;
    const Coordinate chroma_coordinate = chroma_coordinates[pixel];
    const int chroma_width = (parameters.source_width + 1) / 2;
    const int chroma_height = i420
                                ? (parameters.source_height + 1) / 2
                                : max(1, parameters.source_height / 2);
    const int chroma_x = static_cast<int>(roundf(fminf(
      fmaxf(chroma_coordinate.x, 0.0f),
      static_cast<float>(chroma_width - 1))));
    const int chroma_y = static_cast<int>(roundf(fminf(
      fmaxf(chroma_coordinate.y, 0.0f),
      static_cast<float>(chroma_height - 1))));

    uint8_t u;
    uint8_t v;
    if (i420) {
        const uint8_t* u_plane =
          source + static_cast<size_t>(parameters.source_stride) *
                     parameters.source_height;
        const uint8_t* v_plane =
          u_plane + static_cast<size_t>(parameters.source_chroma_stride) *
                      chroma_height;
        const size_t chroma_offset =
          static_cast<size_t>(chroma_y) * parameters.source_chroma_stride +
          chroma_x;
        u = u_plane[chroma_offset];
        v = v_plane[chroma_offset];
    } else {
        const uint8_t* chroma =
          source + static_cast<size_t>(parameters.source_stride) *
                     parameters.source_height;
        const size_t chroma_offset =
          static_cast<size_t>(chroma_y) * parameters.source_stride +
          static_cast<size_t>(chroma_x) * 2;
        if (parameters.source_format == YUV_NV12) {
            u = chroma[chroma_offset];
            v = chroma[chroma_offset + 1];
        } else {
            v = chroma[chroma_offset];
            u = chroma[chroma_offset + 1];
        }
    }

    const YuvColor color = DecodeYuv(luma, v, u);
    if (destination_channel == 1) return static_cast<uint8_t>(color.green);
    const bool red_channel = parameters.destination_blue_first
                               ? destination_channel == 2
                               : destination_channel == 0;
    return static_cast<uint8_t>(red_channel ? color.red : color.blue);
}

__global__ void ResizeNormalizeKernel(const uint8_t* source,
                                      const Coordinate* coordinates,
                                      uint8_t* destination,
                                      KernelParameters parameters) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= parameters.destination_width ||
        y >= parameters.destination_height) {
        return;
    }
    const size_t pixel =
      static_cast<size_t>(y) * parameters.destination_width + x;
    const Coordinate coordinate = coordinates[pixel];
    for (int channel = 0; channel < 3; ++channel) {
        const int source_channel =
          parameters.swap_red_blue && channel != 1 ? 2 - channel : channel;
        const uint8_t sampled = parameters.filter == BILINEAR
                                  ? SampleLinear(source, parameters, coordinate,
                                                 source_channel)
                                  : SampleNearest(source, parameters, coordinate,
                                                  source_channel);
        const float normalized =
          parameters.scale[channel] *
          (static_cast<float>(sampled) - parameters.mean[channel]);
        size_t offset;
        if (parameters.destination_layout ==
            static_cast<int>(TensorLayout::NCHW)) {
            offset = static_cast<size_t>(channel) *
                       parameters.destination_channel_stride +
                     static_cast<size_t>(y) *
                       parameters.destination_row_stride +
                     static_cast<size_t>(x) * sizeof(float);
        } else {
            offset = static_cast<size_t>(y) *
                       parameters.destination_row_stride +
                     static_cast<size_t>(3 * x + channel) * sizeof(float);
        }
        *reinterpret_cast<float*>(destination + offset) = normalized;
    }
}

__global__ void YuvResizeNormalizeKernel(
  const uint8_t* source, const Coordinate* coordinates,
  const Coordinate* chroma_coordinates, uint8_t* destination,
  YuvKernelParameters parameters) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= parameters.destination_width ||
        y >= parameters.destination_height) {
        return;
    }
    for (int channel = 0; channel < 3; ++channel) {
        const uint8_t sampled = SampleYuv(
          source, coordinates, chroma_coordinates, parameters, x, y, channel);
        const float normalized =
          parameters.scale[channel] *
          (static_cast<float>(sampled) - parameters.mean[channel]);
        size_t offset;
        if (parameters.destination_layout ==
            static_cast<int>(TensorLayout::NCHW)) {
            offset = static_cast<size_t>(channel) *
                       parameters.destination_channel_stride +
                     static_cast<size_t>(y) *
                       parameters.destination_row_stride +
                     static_cast<size_t>(x) * sizeof(float);
        } else {
            offset = static_cast<size_t>(y) *
                       parameters.destination_row_stride +
                     static_cast<size_t>(3 * x + channel) * sizeof(float);
        }
        *reinterpret_cast<float*>(destination + offset) = normalized;
    }
}

bool IsFloat32(halide_type_t type) {
    return type.code == halide_type_float && type.bits == 32 && type.lanes == 1;
}

bool IsSupported(const PipelineConfig& config,
                 const Matrix& destination_to_source,
                 const ConversionRequest& request) {
    const bool packed_source = config.source_format == RGB ||
                               config.source_format == BGR;
    const bool yuv420_source = config.source_format == YUV_NV12 ||
                               config.source_format == YUV_NV21 ||
                               config.source_format == YUV_I420;
    const bool packed_destination = config.destination_format == RGB ||
                                    config.destination_format == BGR;
    const bool supported_layout =
      request.destination_layout == TensorLayout::NCHW ||
      request.destination_layout == TensorLayout::NHWC;
    // 4:2:0 semi-planar and planar images require complete 2x2 chroma
    // blocks. Keep non-standard odd-sized buffers on the CPU path, where
    // their legacy storage rules are already defined and tested.
    const bool yuv420_dimensions_supported =
      !yuv420_source ||
      ((request.source_width & 1) == 0 &&
       (request.source_height & 1) == 0);
    const bool source_shape_supported =
      (packed_source && request.source_channels == 3) ||
      (yuv420_source && request.source_channels == 0 &&
       (request.source_stride == 0 ||
        request.source_stride >= request.source_width));
    return source_shape_supported && yuv420_dimensions_supported &&
           packed_destination &&
           (config.filter == NEAREST || config.filter == BILINEAR) &&
           config.wrap == CLAMP_TO_EDGE &&
           !config.preserve_identity_float_order && IsFloat32(request.destination_type) &&
           request.destination_channels == 3 &&
           supported_layout &&
           destination_to_source[Matrix::kMPersp0] == 0.0f &&
           destination_to_source[Matrix::kMPersp1] == 0.0f &&
           destination_to_source[Matrix::kMPersp2] == 1.0f;
}

size_t SourceStorageBytes(const PipelineConfig& config,
                          const ConversionRequest& request) {
    const bool yuv420 = config.source_format == YUV_NV12 ||
                        config.source_format == YUV_NV21 ||
                        config.source_format == YUV_I420;
    if (yuv420) {
        const size_t stride = request.source_stride == 0
                                ? static_cast<size_t>(request.source_width)
                                : static_cast<size_t>(request.source_stride);
        if (config.source_format == YUV_I420) {
            const size_t chroma_stride = request.source_stride == 0
                                           ? static_cast<size_t>(
                                               (request.source_width + 1) / 2)
                                           : stride;
            const size_t chroma_height =
              static_cast<size_t>((request.source_height + 1) / 2);
            return stride * static_cast<size_t>(request.source_height) +
                   2 * chroma_stride * chroma_height;
        }
        const size_t chroma_height = static_cast<size_t>(
          std::max(1, request.source_height / 2));
        return stride * (static_cast<size_t>(request.source_height) +
                         chroma_height);
    }
    const size_t stride = request.source_stride == 0
                            ? static_cast<size_t>(request.source_width) * 3
                            : static_cast<size_t>(request.source_stride);
    return stride * static_cast<size_t>(request.source_height);
}

size_t DestinationStorageBytes(const ConversionRequest& request) {
    const size_t row_bytes =
      static_cast<size_t>(request.destination_width) * 3 * sizeof(float);
    const size_t row_stride = request.destination_stride == 0
                                ? row_bytes
                                : static_cast<size_t>(request.destination_stride);
    if (request.destination_layout == TensorLayout::NHWC) {
        return row_stride * static_cast<size_t>(request.destination_height);
    }
    const size_t channel_stride = request.destination_channel_stride;
    return channel_stride * 2 +
           row_stride * static_cast<size_t>(request.destination_height);
}

bool EnsureAllocation(void** allocation, size_t* capacity, size_t required) {
    if (*capacity >= required) return true;
    if (*allocation) {
        cudaFree(*allocation);
        *allocation = nullptr;
        *capacity = 0;
    }
    if (cudaMalloc(allocation, required) != cudaSuccess) return false;
    *capacity = required;
    return true;
}

bool GeometryMatches(const GeometryKey& key,
                     const Matrix& destination_to_source,
                     const ConversionRequest& request) {
    if (!key.valid || key.source_width != request.source_width ||
        key.source_height != request.source_height ||
        key.destination_width != request.destination_width ||
        key.destination_height != request.destination_height) {
        return false;
    }
    float matrix[9];
    destination_to_source.get9(matrix);
    return std::memcmp(key.matrix, matrix, sizeof(matrix)) == 0;
}

KernelParameters MakeParameters(const PipelineConfig& config,
                                const ConversionRequest& request) {
    KernelParameters parameters{};
    parameters.source_width = request.source_width;
    parameters.source_height = request.source_height;
    parameters.source_stride = request.source_stride == 0
                                 ? request.source_width * 3
                                 : request.source_stride;
    parameters.destination_width = request.destination_width;
    parameters.destination_height = request.destination_height;
    parameters.destination_layout =
      static_cast<int>(request.destination_layout);
    parameters.destination_row_stride =
      request.destination_stride == 0
        ? static_cast<size_t>(request.destination_width) * 3 * sizeof(float)
        : static_cast<size_t>(request.destination_stride);
    parameters.destination_channel_stride =
      request.destination_channel_stride;
    parameters.filter = static_cast<int>(config.filter);
    parameters.swap_red_blue = config.source_format != config.destination_format;
    for (int channel = 0; channel < 3; ++channel) {
        parameters.mean[channel] = config.mean[channel];
        parameters.scale[channel] = config.scale[channel];
    }
    return parameters;
}

YuvKernelParameters MakeYuvParameters(const PipelineConfig& config,
                                      const ConversionRequest& request) {
    YuvKernelParameters parameters{};
    parameters.source_width = request.source_width;
    parameters.source_height = request.source_height;
    parameters.source_stride = request.source_stride == 0
                                 ? request.source_width
                                 : request.source_stride;
    parameters.source_chroma_stride =
      config.source_format == YUV_I420 && request.source_stride == 0
        ? (request.source_width + 1) / 2
        : parameters.source_stride;
    parameters.source_format = static_cast<int>(config.source_format);
    parameters.destination_width = request.destination_width;
    parameters.destination_height = request.destination_height;
    parameters.destination_layout =
      static_cast<int>(request.destination_layout);
    parameters.destination_row_stride =
      request.destination_stride == 0
        ? static_cast<size_t>(request.destination_width) * 3 * sizeof(float)
        : static_cast<size_t>(request.destination_stride);
    parameters.destination_channel_stride =
      request.destination_channel_stride;
    parameters.destination_blue_first = config.destination_format == BGR;
    for (int channel = 0; channel < 3; ++channel) {
        parameters.mean[channel] = config.mean[channel];
        parameters.scale[channel] = config.scale[channel];
    }
    return parameters;
}

}  // namespace

struct CudaExecutor {
    uint8_t* device_source = nullptr;
    uint8_t* device_destination = nullptr;
    Coordinate* device_coordinates = nullptr;
    Coordinate* device_chroma_coordinates = nullptr;
    size_t source_capacity = 0;
    size_t destination_capacity = 0;
    size_t coordinate_capacity = 0;
    size_t chroma_coordinate_capacity = 0;
    cudaStream_t stream = nullptr;
    GeometryKey geometry;
    std::vector<Coordinate> host_coordinates;
    std::vector<Coordinate> host_chroma_coordinates;
};

namespace {

bool EnsureCoordinates(CudaExecutor* executor,
                       const PipelineConfig& config,
                       const Matrix& destination_to_source,
                       const Matrix& source_to_destination,
                       const ConversionRequest& request,
                       cudaStream_t stream) {
    if (GeometryMatches(executor->geometry, destination_to_source, request)) {
        return true;
    }
    const size_t count = static_cast<size_t>(request.destination_width) *
                         request.destination_height;
    const bool yuv420 = config.source_format == YUV_NV12 ||
                        config.source_format == YUV_NV21 ||
                        config.source_format == YUV_I420;
    try {
        executor->host_coordinates.resize(count);
        if (yuv420) executor->host_chroma_coordinates.resize(count);
    } catch (...) {
        return false;
    }
    for (int y = 0; y < request.destination_height; ++y) {
        for (int first_x = 0; first_x < request.destination_width;
             first_x += kCoordinateTileWidth) {
            const int tile_count = std::min(
              kCoordinateTileWidth, request.destination_width - first_x);
            const RowWindow window = ScheduleRow(
              destination_to_source, source_to_destination, CLAMP_TO_EDGE,
              request.source_width, request.source_height, y, first_x,
              tile_count);
            Point coordinate = window.origin;
            for (int local_x = 0; local_x < tile_count; ++local_x) {
                executor->host_coordinates[
                  static_cast<size_t>(y) * request.destination_width +
                  first_x + local_x] = {coordinate.fX, coordinate.fY};
                coordinate.fX += window.step.fX;
                coordinate.fY += window.step.fY;
            }
            if (yuv420) {
                Point chroma_coordinate = {
                  (window.origin.fX - 0.01f) / 2.0f,
                  (window.origin.fY - 0.01f) / 2.0f};
                const bool direct_sampling =
                  destination_to_source.isIdentity() &&
                  request.source_width >= request.destination_width &&
                  request.source_height >= request.destination_height;
                const float step_scale =
                  config.source_format == YUV_I420 && !direct_sampling
                    ? 0.5f
                    : 1.0f;
                for (int local_x = 0; local_x < tile_count; local_x += 2) {
                    const Coordinate stored = {chroma_coordinate.fX,
                                               chroma_coordinate.fY};
                    const size_t first =
                      static_cast<size_t>(y) * request.destination_width +
                      first_x + local_x;
                    executor->host_chroma_coordinates[first] = stored;
                    if (local_x + 1 < tile_count) {
                        executor->host_chroma_coordinates[first + 1] = stored;
                    }
                    chroma_coordinate.fX += window.step.fX * step_scale;
                    chroma_coordinate.fY += window.step.fY * step_scale;
                }
            }
        }
    }
    const size_t bytes = count * sizeof(Coordinate);
    if (!EnsureAllocation(reinterpret_cast<void**>(&executor->device_coordinates),
                          &executor->coordinate_capacity, bytes)) {
        return false;
    }
    if (cudaMemcpyAsync(executor->device_coordinates,
                        executor->host_coordinates.data(), bytes,
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        return false;
    }
    if (yuv420) {
        if (!EnsureAllocation(
              reinterpret_cast<void**>(&executor->device_chroma_coordinates),
              &executor->chroma_coordinate_capacity, bytes) ||
            cudaMemcpyAsync(executor->device_chroma_coordinates,
                            executor->host_chroma_coordinates.data(), bytes,
                            cudaMemcpyHostToDevice, stream) != cudaSuccess) {
            return false;
        }
    }
    executor->geometry.source_width = request.source_width;
    executor->geometry.source_height = request.source_height;
    executor->geometry.destination_width = request.destination_width;
    executor->geometry.destination_height = request.destination_height;
    destination_to_source.get9(executor->geometry.matrix);
    executor->geometry.valid = true;
    return true;
}

TaskStatus Launch(CudaExecutor* executor, const PipelineConfig& config,
                  const Matrix& destination_to_source,
                  const Matrix& source_to_destination,
                  const ConversionRequest& request,
                  const uint8_t* source, uint8_t* destination,
                  cudaStream_t stream, bool* handled) {
    *handled = IsSupported(config, destination_to_source, request);
    if (!*handled) return UNSUPPORTED_CONVERSION;
    if (!EnsureCoordinates(executor, config, destination_to_source,
                           source_to_destination, request, stream)) {
        return ACCELERATION_FAILURE;
    }
    const dim3 block(16, 16);
    const dim3 grid((request.destination_width + block.x - 1) / block.x,
                    (request.destination_height + block.y - 1) / block.y);
    const bool yuv420 = config.source_format == YUV_NV12 ||
                        config.source_format == YUV_NV21 ||
                        config.source_format == YUV_I420;
    if (yuv420) {
        const YuvKernelParameters parameters =
          MakeYuvParameters(config, request);
        YuvResizeNormalizeKernel<<<grid, block, 0, stream>>>(
          source, executor->device_coordinates,
          executor->device_chroma_coordinates, destination, parameters);
    } else {
        const KernelParameters parameters = MakeParameters(config, request);
        ResizeNormalizeKernel<<<grid, block, 0, stream>>>(
          source, executor->device_coordinates, destination, parameters);
    }
    return cudaPeekAtLastError() == cudaSuccess ? SUCCESS
                                                : ACCELERATION_FAILURE;
}

}  // namespace

bool CudaRuntimeAvailable() noexcept {
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

bool QueryCudaDeviceDescriptor(CudaDeviceDescriptor* descriptor) noexcept {
    if (!descriptor) return false;
    int device_index = 0;
    cudaDeviceProp properties{};
    int runtime_version = 0;
    int driver_version = 0;
    if (cudaGetDevice(&device_index) != cudaSuccess ||
        cudaGetDeviceProperties(&properties, device_index) != cudaSuccess ||
        cudaRuntimeGetVersion(&runtime_version) != cudaSuccess ||
        cudaDriverGetVersion(&driver_version) != cudaSuccess) {
        return false;
    }
    std::strncpy(descriptor->name, properties.name,
                 sizeof(descriptor->name) - 1);
    descriptor->name[sizeof(descriptor->name) - 1] = '\0';
    descriptor->compute_capability_major = properties.major;
    descriptor->compute_capability_minor = properties.minor;
    descriptor->runtime_version = runtime_version;
    descriptor->driver_version = driver_version;
    return true;
}

bool CudaSupportsConversion(const PipelineConfig& config,
                            const Matrix& destination_to_source,
                            const ConversionRequest& request) noexcept {
    return IsSupported(config, destination_to_source, request);
}

CudaExecutor* CreateCudaExecutor() noexcept {
    CudaExecutor* executor = new (std::nothrow) CudaExecutor;
    if (!executor) return nullptr;
    if (cudaStreamCreateWithFlags(&executor->stream,
                                  cudaStreamNonBlocking) != cudaSuccess) {
        delete executor;
        return nullptr;
    }
    return executor;
}

void DestroyCudaExecutor(CudaExecutor* executor) noexcept {
    if (!executor) return;
    cudaFree(executor->device_source);
    cudaFree(executor->device_destination);
    cudaFree(executor->device_coordinates);
    cudaFree(executor->device_chroma_coordinates);
    if (executor->stream) cudaStreamDestroy(executor->stream);
    delete executor;
}

TaskStatus ExecuteCudaHost(CudaExecutor* executor,
                           const PipelineConfig& config,
                           const Matrix& destination_to_source,
                           const Matrix& source_to_destination,
                           const ConversionRequest& request,
                           const uint8_t* source, void* destination,
                           bool* handled) noexcept {
    if (!handled) return ACCELERATION_FAILURE;
    *handled = IsSupported(config, destination_to_source, request);
    if (!*handled) return UNSUPPORTED_CONVERSION;
    if (!executor || !source || !destination) return ACCELERATION_FAILURE;

    const size_t source_bytes = SourceStorageBytes(config, request);
    const size_t destination_bytes = DestinationStorageBytes(request);
    if (!EnsureAllocation(reinterpret_cast<void**>(&executor->device_source),
                          &executor->source_capacity, source_bytes) ||
        !EnsureAllocation(
          reinterpret_cast<void**>(&executor->device_destination),
          &executor->destination_capacity, destination_bytes)) {
        return ACCELERATION_FAILURE;
    }
    const bool yuv420 = config.source_format == YUV_NV12 ||
                        config.source_format == YUV_NV21 ||
                        config.source_format == YUV_I420;
    if (yuv420) {
        if (cudaMemcpyAsync(executor->device_source, source, source_bytes,
                            cudaMemcpyHostToDevice,
                            executor->stream) != cudaSuccess) {
            return ACCELERATION_FAILURE;
        }
    } else {
        const size_t source_stride = request.source_stride == 0
          ? static_cast<size_t>(request.source_width) * 3
          : static_cast<size_t>(request.source_stride);
        if (cudaMemcpy2DAsync(executor->device_source, source_stride, source,
                              source_stride,
                              static_cast<size_t>(request.source_width) * 3,
                              request.source_height, cudaMemcpyHostToDevice,
                              executor->stream) != cudaSuccess) {
            return ACCELERATION_FAILURE;
        }
    }
    const TaskStatus launch = Launch(
      executor, config, destination_to_source, source_to_destination, request,
      executor->device_source, executor->device_destination, executor->stream,
      handled);
    if (launch != SUCCESS) return launch;

    const size_t destination_row_stride = request.destination_stride == 0
      ? static_cast<size_t>(request.destination_width) * 3 * sizeof(float)
      : static_cast<size_t>(request.destination_stride);
    if (request.destination_layout == TensorLayout::NHWC) {
        if (cudaMemcpy2DAsync(
              destination, destination_row_stride,
              executor->device_destination, destination_row_stride,
              static_cast<size_t>(request.destination_width) * 3 * sizeof(float),
              request.destination_height, cudaMemcpyDeviceToHost,
              executor->stream) != cudaSuccess) {
            return ACCELERATION_FAILURE;
        }
    } else {
        auto* destination_bytes_host = static_cast<uint8_t*>(destination);
        for (int channel = 0; channel < 3; ++channel) {
            const size_t channel_offset =
              static_cast<size_t>(channel) * request.destination_channel_stride;
            if (cudaMemcpy2DAsync(
                  destination_bytes_host + channel_offset,
                  destination_row_stride,
                  executor->device_destination + channel_offset,
                  destination_row_stride,
                  static_cast<size_t>(request.destination_width) * sizeof(float),
                  request.destination_height, cudaMemcpyDeviceToHost,
                  executor->stream) != cudaSuccess) {
                return ACCELERATION_FAILURE;
            }
        }
    }
    return cudaStreamSynchronize(executor->stream) == cudaSuccess
             ? SUCCESS
             : ACCELERATION_FAILURE;
}

TaskStatus ExecuteCudaDevice(CudaExecutor* executor,
                             const PipelineConfig& config,
                             const Matrix& destination_to_source,
                             const Matrix& source_to_destination,
                             const ConversionRequest& request,
                             std::uintptr_t source,
                             std::uintptr_t destination,
                             void* stream, bool* handled) noexcept {
    if (!handled) return ACCELERATION_FAILURE;
    if (!executor || source == 0 || destination == 0) {
        *handled = false;
        return ACCELERATION_FAILURE;
    }
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    return Launch(executor, config, destination_to_source,
                  source_to_destination, request,
                  reinterpret_cast<const uint8_t*>(source),
                  reinterpret_cast<uint8_t*>(destination), cuda_stream,
                  handled);
}

}  // namespace internal
}  // namespace task
}  // namespace inspirecv
