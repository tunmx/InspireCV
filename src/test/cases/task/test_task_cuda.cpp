#include "../../common/common.h"

#include <inspirecv/task/task.h>

#include "inspirecv/task/cuda/cuda_auto_policy.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#if defined(INSPIRECV_TASK_HAS_CUDA)
#include <cuda_runtime_api.h>
#if defined(__linux__) || defined(__APPLE__)
#include <sys/utsname.h>
#endif
#endif

namespace {

inspirecv::TransformMatrix ResizeTransform(int source_width, int source_height,
                                           int destination_width,
                                           int destination_height) {
    return inspirecv::TransformMatrix(
      static_cast<float>(source_width) / destination_width, 0.0f, 0.0f,
      0.0f, static_cast<float>(source_height) / destination_height, 0.0f);
}

std::vector<uint8_t> MakeCudaSource(int width, int height) {
    std::vector<uint8_t> source(static_cast<size_t>(width) * height * 3);
    for (size_t index = 0; index < source.size(); ++index) {
        source[index] = static_cast<uint8_t>((index * 37 + index / 11 + 19) & 255);
    }
    return source;
}

std::vector<uint8_t> MakeYuv420Source(
  int width, int height, inspirecv::task::PixelFormat format) {
    REQUIRE(width > 0);
    REQUIRE(height > 0);
    REQUIRE((width & 1) == 0);
    REQUIRE((height & 1) == 0);
    const size_t luma_bytes = static_cast<size_t>(width) * height;
    const size_t chroma_plane =
      static_cast<size_t>(width / 2) * (height / 2);
    std::vector<uint8_t> source(luma_bytes + 2 * chroma_plane);
    for (size_t index = 0; index < luma_bytes; ++index) {
        source[index] = static_cast<uint8_t>(
          (index * 17 + index / 13 + 31) & 255);
    }
    if (format == inspirecv::task::PixelFormat::kI420) {
        uint8_t* u = source.data() + luma_bytes;
        uint8_t* v = u + chroma_plane;
        for (size_t index = 0; index < chroma_plane; ++index) {
            u[index] = static_cast<uint8_t>((index * 19 + 73) & 255);
            v[index] = static_cast<uint8_t>((index * 29 + 151) & 255);
        }
        return source;
    }
    uint8_t* chroma = source.data() + luma_bytes;
    const bool nv12 = format == inspirecv::task::PixelFormat::kNv12;
    for (size_t index = 0; index < chroma_plane; ++index) {
        const uint8_t u = static_cast<uint8_t>((index * 19 + 73) & 255);
        const uint8_t v = static_cast<uint8_t>((index * 29 + 151) & 255);
        chroma[2 * index] = nv12 ? u : v;
        chroma[2 * index + 1] = nv12 ? v : u;
    }
    return source;
}

struct DisableCudaAtExit {
    DisableCudaAtExit()
        : previous_preference(
            inspirecv::task::GetDefaultBackendPreference()) {
        REQUIRE(inspirecv::task::SetDefaultBackendPreference(
          inspirecv::task::BackendPreference::kCuda));
    }

    ~DisableCudaAtExit() {
        inspirecv::task::SetCudaEnabled(false);
        inspirecv::task::SetDefaultBackendPreference(previous_preference);
    }

    inspirecv::task::BackendPreference previous_preference;
};

void RequireExactFloatOutput(const std::vector<float>& actual,
                             const std::vector<float>& expected) {
    REQUIRE(actual.size() == expected.size());
    size_t mismatch_count = 0;
    size_t first_mismatch = 0;
    float maximum_absolute_error = 0.0f;
    std::ostringstream mismatch_details;
    for (size_t index = 0; index < actual.size(); ++index) {
        if (std::memcmp(&actual[index], &expected[index], sizeof(float)) == 0) {
            continue;
        }
        if (mismatch_count == 0) first_mismatch = index;
        if (mismatch_count < 8) {
            mismatch_details << " [" << index << ": expected="
                             << expected[index] << ", actual="
                             << actual[index] << ']';
        }
        ++mismatch_count;
        maximum_absolute_error =
          std::max(maximum_absolute_error,
                   std::abs(actual[index] - expected[index]));
    }
    INFO("mismatches=" << mismatch_count << "/" << actual.size());
    INFO("first_index=" << first_mismatch);
    INFO("expected=" << expected[first_mismatch]);
    INFO("actual=" << actual[first_mismatch]);
    INFO("max_absolute_error=" << maximum_absolute_error);
    INFO("details=" << mismatch_details.str());
    REQUIRE(mismatch_count == 0);
}

#if defined(INSPIRECV_TASK_HAS_CUDA)

struct TimingDistribution {
    double p50_microseconds = 0.0;
    double p95_microseconds = 0.0;
};

double Percentile(std::vector<double> samples, double percentile) {
    REQUIRE_FALSE(samples.empty());
    std::sort(samples.begin(), samples.end());
    const size_t index = static_cast<size_t>(
      std::ceil(percentile * static_cast<double>(samples.size())) - 1.0);
    return samples[std::min(index, samples.size() - 1)];
}

TimingDistribution MeasureReadyHostPipeline(
  inspirecv::task::Pipeline* pipeline,
  const inspirecv::task::RawImageView& source,
  const inspirecv::task::TensorBuffer& destination, int sample_count) {
    REQUIRE(pipeline != nullptr);
    for (int warmup = 0; warmup < 10; ++warmup) {
        REQUIRE(pipeline->Run(source, destination) ==
                inspirecv::task::Status::kOk);
    }

    std::vector<double> samples;
    samples.reserve(sample_count);
    for (int sample = 0; sample < sample_count; ++sample) {
        const auto begin = std::chrono::steady_clock::now();
        const auto status = pipeline->Run(source, destination);
        const auto end = std::chrono::steady_clock::now();
        REQUIRE(status == inspirecv::task::Status::kOk);
        samples.push_back(
          std::chrono::duration<double, std::micro>(end - begin).count());
    }
    TimingDistribution result;
    result.p50_microseconds = Percentile(samples, 0.50);
    result.p95_microseconds = Percentile(samples, 0.95);
    return result;
}

TimingDistribution MeasureHostPipeline(
  inspirecv::task::Pipeline* pipeline,
  const inspirecv::task::RawImageView& source,
  const inspirecv::task::TensorBuffer& destination, bool cuda_enabled,
  int sample_count) {
    REQUIRE(inspirecv::task::SetCudaEnabled(cuda_enabled));
    const TimingDistribution result = MeasureReadyHostPipeline(
      pipeline, source, destination, sample_count);
    const auto expected_backend = cuda_enabled
                                    ? inspirecv::task::ExecutionBackend::kCuda
                                    : inspirecv::task::ExecutionBackend::kCpu;
    REQUIRE(pipeline->LastExecutionBackend() == expected_backend);
    return result;
}

std::string ReadLinuxCpuModel() {
#if defined(__linux__)
    std::ifstream input("/proc/cpuinfo");
    std::string line;
    while (std::getline(input, line)) {
        const std::string key = "model name";
        if (line.compare(0, key.size(), key) != 0) continue;
        const size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        const size_t value = line.find_first_not_of(" \t", colon + 1);
        return value == std::string::npos ? std::string() : line.substr(value);
    }
#endif
    return "unknown";
}

std::string ReadLinuxNvidiaDriverVersion() {
#if defined(__linux__)
    std::ifstream input("/proc/driver/nvidia/version");
    std::string line;
    if (std::getline(input, line)) {
        std::istringstream words(line);
        std::string word;
        while (words >> word) {
            if (word.find('.') != std::string::npos && !word.empty() &&
                word[0] >= '0' && word[0] <= '9') {
                return word;
            }
        }
    }
#endif
    return "unknown";
}

std::string CompilerDescription() {
#if defined(__clang__)
    return std::string("Clang ") + __clang_version__;
#elif defined(__GNUC__)
    return std::string("GCC ") + __VERSION__;
#elif defined(_MSC_VER)
    return std::string("MSVC ") + std::to_string(_MSC_VER);
#else
    return "unknown";
#endif
}

std::string OperatingSystemDescription() {
#if defined(__linux__) || defined(__APPLE__)
    struct utsname information {};
    if (uname(&information) == 0) {
        return std::string(information.sysname) + " " + information.release +
               " " + information.machine;
    }
#endif
    return "unknown";
}

#endif  // INSPIRECV_TASK_HAS_CUDA

uint8_t RestoreTensorByte(float value, float mean, float scale) {
    const long restored = std::lround(value / scale + mean);
    return static_cast<uint8_t>(std::max(0L, std::min(255L, restored)));
}

void WriteCudaVisualCase(
  const char* name, const inspirecv::TransformMatrix& transform,
  const inspirecv::task::RawImageView& source,
  const inspirecv::task::PipelineOptions& options,
  inspirecv::task::Pipeline* pipeline, const std::string& directory) {
    constexpr int kOutputWidth = 320;
    constexpr int kOutputHeight = 320;
    const size_t plane_elements =
      static_cast<size_t>(kOutputWidth) * kOutputHeight;
    std::vector<float> cpu_tensor(3 * plane_elements);
    std::vector<float> cuda_tensor(3 * plane_elements);
    inspirecv::task::TensorBuffer destination;
    destination.width = kOutputWidth;
    destination.height = kOutputHeight;
    destination.channels = 3;
    destination.element_type = inspirecv::task::ElementType::kFloat32;
    destination.order = inspirecv::task::TensorOrder::kChw;

    pipeline->SetTransform(transform);
    REQUIRE(inspirecv::task::SetCudaEnabled(false));
    destination.data = cpu_tensor.data();
    REQUIRE(pipeline->Run(source, destination) == inspirecv::task::Status::kOk);
    REQUIRE(pipeline->LastExecutionBackend() ==
            inspirecv::task::ExecutionBackend::kCpu);
    REQUIRE(inspirecv::task::SetCudaEnabled(true));
    destination.data = cuda_tensor.data();
    REQUIRE(pipeline->Run(source, destination) == inspirecv::task::Status::kOk);
    REQUIRE(pipeline->LastExecutionBackend() ==
            inspirecv::task::ExecutionBackend::kCuda);
    RequireExactFloatOutput(cuda_tensor, cpu_tensor);

    std::vector<uint8_t> cpu_bgr(3 * plane_elements);
    std::vector<uint8_t> cuda_bgr(3 * plane_elements);
    std::vector<uint8_t> difference(3 * plane_elements);
    size_t differing_bytes = 0;
    int maximum_error = 0;
    for (size_t pixel = 0; pixel < plane_elements; ++pixel) {
        for (int rgb_channel = 0; rgb_channel < 3; ++rgb_channel) {
            const size_t tensor_index =
              static_cast<size_t>(rgb_channel) * plane_elements + pixel;
            const size_t image_index =
              3 * pixel + static_cast<size_t>(2 - rgb_channel);
            cpu_bgr[image_index] = RestoreTensorByte(
              cpu_tensor[tensor_index], options.mean[rgb_channel],
              options.scale[rgb_channel]);
            cuda_bgr[image_index] = RestoreTensorByte(
              cuda_tensor[tensor_index], options.mean[rgb_channel],
              options.scale[rgb_channel]);
            const int error = std::abs(static_cast<int>(cpu_bgr[image_index]) -
                                       static_cast<int>(cuda_bgr[image_index]));
            difference[image_index] = static_cast<uint8_t>(error);
            if (error != 0) ++differing_bytes;
            maximum_error = std::max(maximum_error, error);
        }
    }
    REQUIRE(differing_bytes == 0);
    REQUIRE(maximum_error == 0);

    std::vector<uint8_t> comparison(3 * plane_elements * 3);
    for (int y = 0; y < kOutputHeight; ++y) {
        const size_t source_offset =
          static_cast<size_t>(y) * kOutputWidth * 3;
        const size_t comparison_offset =
          static_cast<size_t>(y) * kOutputWidth * 9;
        std::copy_n(cpu_bgr.data() + source_offset, kOutputWidth * 3,
                    comparison.data() + comparison_offset);
        std::copy_n(cuda_bgr.data() + source_offset, kOutputWidth * 3,
                    comparison.data() + comparison_offset + kOutputWidth * 3);
        std::copy_n(difference.data() + source_offset, kOutputWidth * 3,
                    comparison.data() + comparison_offset + kOutputWidth * 6);
    }

    const std::string stem = directory + "/" + name;
    REQUIRE(inspirecv::Image::Create(kOutputWidth, kOutputHeight, 3,
                                     cpu_bgr.data())
              .Write(stem + "_cpu.png"));
    REQUIRE(inspirecv::Image::Create(kOutputWidth, kOutputHeight, 3,
                                     cuda_bgr.data())
              .Write(stem + "_cuda.png"));
    REQUIRE(inspirecv::Image::Create(kOutputWidth, kOutputHeight, 3,
                                     difference.data())
              .Write(stem + "_absdiff.png"));
    REQUIRE(inspirecv::Image::Create(kOutputWidth * 3, kOutputHeight, 3,
                                     comparison.data())
              .Write(stem + "_comparison.png"));
    std::printf("[TaskCudaVisual] case=%s differing_bytes=%zu max_error=%d\n",
                name, differing_bytes, maximum_error);
}

inspirecv::TransformMatrix MakeCenteredAffine(
  int source_width, int source_height, int output_size, float angle_degrees,
  float crop_scale, float translate_x, float translate_y) {
    const float crop_size =
      static_cast<float>(std::min(source_width, source_height));
    const float source_step = crop_size * crop_scale / output_size;
    constexpr float kPi = 3.14159265358979323846f;
    const float angle = angle_degrees * kPi / 180.0f;
    const float cosine = std::cos(angle);
    const float sine = std::sin(angle);
    const float a00 = source_step * cosine;
    const float a01 = -source_step * sine;
    const float a10 = source_step * sine;
    const float a11 = source_step * cosine;
    const float destination_center = (output_size - 1) * 0.5f;
    const float source_center_x = (source_width - 1) * 0.5f + translate_x;
    const float source_center_y = (source_height - 1) * 0.5f + translate_y;
    const float offset_x =
      source_center_x - a00 * destination_center - a01 * destination_center;
    const float offset_y =
      source_center_y - a10 * destination_center - a11 * destination_center;
    return inspirecv::TransformMatrix(a00, a01, offset_x,
                                      a10, a11, offset_y);
}

void MaybeWriteCudaVisualReport() {
    const char* source_path = std::getenv("INSPIRECV_CUDA_VISUAL_SOURCE");
    const char* output_directory =
      std::getenv("INSPIRECV_CUDA_VISUAL_DIR");
    if (!source_path || !output_directory) return;

    REQUIRE(inspirecv::task::IsCudaAvailable());
    const auto input = inspirecv::Image::Create(source_path, 3);
    REQUIRE_FALSE(input.Empty());
    inspirecv::task::RawImageView source;
    source.data = input.Data();
    source.width = input.Width();
    source.height = input.Height();
    source.row_stride_bytes = static_cast<size_t>(input.Width()) * 3;

    inspirecv::task::PipelineOptions options;
    options.input_format = inspirecv::task::PixelFormat::kBgr;
    options.output_format = inspirecv::task::PixelFormat::kRgb;
    options.sampling = inspirecv::task::SamplingMode::kLinear;
    options.mean = {{123.675f, 116.28f, 103.53f, 0.0f}};
    options.scale = {{1.0f / 58.395f, 1.0f / 57.12f,
                      1.0f / 57.375f, 1.0f}};
    inspirecv::task::Pipeline pipeline(options);

    struct VisualCase {
        const char* name;
        inspirecv::TransformMatrix transform;
    };
    const VisualCase cases[] = {
      {"10_center_crop",
       MakeCenteredAffine(input.Width(), input.Height(), 320,
                          0.0f, 1.0f, 0.0f, 0.0f)},
      {"20_rotate_20deg",
       MakeCenteredAffine(input.Width(), input.Height(), 320,
                          20.0f, 1.0f, 0.0f, 0.0f)},
      {"30_translate",
       MakeCenteredAffine(input.Width(), input.Height(), 320,
                          0.0f, 1.0f, 80.0f, -60.0f)},
      {"40_scale_rotate_translate",
       MakeCenteredAffine(input.Width(), input.Height(), 320,
                          -15.0f, 0.78f, 45.0f, 35.0f)},
    };

    const std::string directory(output_directory);
    REQUIRE(input.Write(directory + "/00_input.png"));
    for (const VisualCase& current : cases) {
        WriteCudaVisualCase(current.name, current.transform, source, options,
                            &pipeline, directory);
    }
}

}  // namespace

TEST_CASE("task_cuda_backend_preference_contract",
          "[task][cuda][api][contract]") {
    const auto previous = inspirecv::task::GetDefaultBackendPreference();
    REQUIRE(inspirecv::task::SetDefaultBackendPreference(
      inspirecv::task::BackendPreference::kAuto));
    REQUIRE(inspirecv::task::GetDefaultBackendPreference() ==
            inspirecv::task::BackendPreference::kAuto);
    REQUIRE(inspirecv::task::SetDefaultBackendPreference(
      inspirecv::task::BackendPreference::kCpu));
    REQUIRE(inspirecv::task::GetDefaultBackendPreference() ==
            inspirecv::task::BackendPreference::kCpu);
    REQUIRE(inspirecv::task::SetDefaultBackendPreference(
      inspirecv::task::BackendPreference::kCuda));
    REQUIRE(inspirecv::task::GetDefaultBackendPreference() ==
            inspirecv::task::BackendPreference::kCuda);
    REQUIRE_FALSE(inspirecv::task::SetDefaultBackendPreference(
      inspirecv::task::BackendPreference::kDefault));
    REQUIRE(inspirecv::task::GetDefaultBackendPreference() ==
            inspirecv::task::BackendPreference::kCuda);
    REQUIRE(inspirecv::task::SetDefaultBackendPreference(previous));
}

TEST_CASE("task_cuda_default_3060_baseline_is_conservative",
          "[task][cuda][auto][threshold]") {
    using inspirecv::task::internal::DefaultRecommendedMaxSourceBytes;
    constexpr size_t kOutput112 = 112u * 112u;
    constexpr size_t kOutput224 = 224u * 224u;
    constexpr size_t kOutput640 = 640u * 640u;
    constexpr size_t kInput640x480 = 640u * 480u * 3u;
    constexpr size_t kInput1920x1080 = 1920u * 1080u * 3u;
    constexpr size_t kInput3840x2160 = 3840u * 2160u * 3u;

    REQUIRE(DefaultRecommendedMaxSourceBytes(kOutput112 - 1) == 0);
    REQUIRE(DefaultRecommendedMaxSourceBytes(kOutput112) ==
            kInput640x480);
    REQUIRE(DefaultRecommendedMaxSourceBytes(kOutput224) ==
            kInput1920x1080);
    REQUIRE(DefaultRecommendedMaxSourceBytes(kOutput640) ==
            kInput3840x2160);
    REQUIRE(DefaultRecommendedMaxSourceBytes(kOutput640 * 4) ==
            kInput3840x2160);

    const size_t midpoint = (kOutput112 + kOutput224) / 2;
    const size_t expected_midpoint =
      kInput640x480 + (kInput1920x1080 - kInput640x480) *
                         (midpoint - kOutput112) /
                         (kOutput224 - kOutput112);
    REQUIRE(DefaultRecommendedMaxSourceBytes(midpoint) ==
            expected_midpoint);
}

TEST_CASE("task_cuda_global_switch_is_explicit",
          "[task][cuda][api][contract]") {
    DisableCudaAtExit cleanup;
    REQUIRE(inspirecv::task::SetCudaEnabled(false));
    REQUIRE_FALSE(inspirecv::task::IsCudaEnabled());
    const bool available = inspirecv::task::IsCudaAvailable();
    const bool enabled = inspirecv::task::SetCudaEnabled(true);
    REQUIRE(enabled == available);
    REQUIRE(inspirecv::task::IsCudaEnabled() == available);
    REQUIRE(inspirecv::task::SetCudaEnabled(false));
    REQUIRE_FALSE(inspirecv::task::IsCudaEnabled());
}

TEST_CASE("task_cuda_device_pipeline_requires_global_opt_in",
          "[task][cuda][api][contract]") {
    DisableCudaAtExit cleanup;
    REQUIRE(inspirecv::task::SetCudaEnabled(false));

    inspirecv::task::PipelineOptions options;
    options.input_format = inspirecv::task::PixelFormat::kBgr;
    options.output_format = inspirecv::task::PixelFormat::kRgb;
    inspirecv::task::cuda::Pipeline pipeline(options);
    inspirecv::task::cuda::DeviceImageView source;
    source.data = 1;
    source.width = 1;
    source.height = 1;
    source.row_stride_bytes = 3;
    inspirecv::task::cuda::DeviceTensorBuffer destination;
    destination.data = 1;
    destination.width = 1;
    destination.height = 1;
    destination.channels = 3;

    REQUIRE(pipeline.Run(source, destination) ==
            inspirecv::task::Status::kAccelerationDisabled);
    REQUIRE(pipeline.RunBatch(&source, &destination, 1) ==
            inspirecv::task::Status::kAccelerationDisabled);
}

#if defined(INSPIRECV_TASK_HAS_CUDA)

TEST_CASE("task_cuda_auto_dispatch_uses_default_3060_baseline",
          "[task][cuda][auto][accuracy]") {
    DisableCudaAtExit cleanup;
    if (!inspirecv::task::IsCudaAvailable()) {
        SUCCEED("CUDA backend is not part of this build or no GPU is available");
        return;
    }
    REQUIRE(inspirecv::task::SetCudaEnabled(true));

    struct Case {
        int source_width;
        int source_height;
        int destination_width;
        int destination_height;
        inspirecv::task::ExecutionBackend expected_backend;
        inspirecv::task::SamplingMode sampling;
        inspirecv::task::TensorOrder order;
    };
    const Case cases[] = {
      {224, 224, 224, 224, inspirecv::task::ExecutionBackend::kCpu,
       inspirecv::task::SamplingMode::kLinear,
       inspirecv::task::TensorOrder::kChw},
      {640, 480, 112, 112, inspirecv::task::ExecutionBackend::kCuda,
       inspirecv::task::SamplingMode::kLinear,
       inspirecv::task::TensorOrder::kChw},
      {960, 540, 112, 112, inspirecv::task::ExecutionBackend::kCpu,
       inspirecv::task::SamplingMode::kLinear,
       inspirecv::task::TensorOrder::kChw},
      {1920, 1080, 224, 224, inspirecv::task::ExecutionBackend::kCuda,
       inspirecv::task::SamplingMode::kLinear,
       inspirecv::task::TensorOrder::kChw},
      {2560, 1440, 224, 224, inspirecv::task::ExecutionBackend::kCpu,
       inspirecv::task::SamplingMode::kLinear,
       inspirecv::task::TensorOrder::kChw},
      {3840, 2160, 640, 640, inspirecv::task::ExecutionBackend::kCuda,
       inspirecv::task::SamplingMode::kLinear,
       inspirecv::task::TensorOrder::kChw},
      {640, 480, 112, 112, inspirecv::task::ExecutionBackend::kCpu,
       inspirecv::task::SamplingMode::kNearest,
       inspirecv::task::TensorOrder::kChw},
      {640, 480, 224, 112, inspirecv::task::ExecutionBackend::kCpu,
       inspirecv::task::SamplingMode::kLinear,
       inspirecv::task::TensorOrder::kChw},
      {640, 480, 112, 112, inspirecv::task::ExecutionBackend::kCpu,
       inspirecv::task::SamplingMode::kLinear,
       inspirecv::task::TensorOrder::kHwc},
    };

    for (const Case& current : cases) {
        const auto source_bytes =
          MakeCudaSource(current.source_width, current.source_height);
        inspirecv::task::RawImageView source;
        source.data = source_bytes.data();
        source.width = current.source_width;
        source.height = current.source_height;
        source.row_stride_bytes =
          static_cast<size_t>(current.source_width) * 3;
        const size_t output_elements =
          static_cast<size_t>(current.destination_width) *
          current.destination_height * 3;
        std::vector<float> cpu_output(output_elements);
        std::vector<float> auto_output(output_elements);
        inspirecv::task::TensorBuffer destination;
        destination.width = current.destination_width;
        destination.height = current.destination_height;
        destination.channels = 3;
        destination.element_type = inspirecv::task::ElementType::kFloat32;
        destination.order = current.order;

        inspirecv::task::PipelineOptions cpu_options;
        cpu_options.input_format = inspirecv::task::PixelFormat::kBgr;
        cpu_options.output_format = inspirecv::task::PixelFormat::kRgb;
        cpu_options.sampling = current.sampling;
        cpu_options.mean = {{127.5f, 127.5f, 127.5f, 0.0f}};
        cpu_options.scale = {{1.0f / 128.0f, 1.0f / 128.0f,
                              1.0f / 128.0f, 1.0f}};
        cpu_options.backend_preference =
          inspirecv::task::BackendPreference::kCpu;
        const auto transform = ResizeTransform(
          current.source_width, current.source_height,
          current.destination_width, current.destination_height);
        inspirecv::task::Pipeline cpu_pipeline(cpu_options);
        cpu_pipeline.SetTransform(transform);
        destination.data = cpu_output.data();
        REQUIRE(cpu_pipeline.Run(source, destination) ==
                inspirecv::task::Status::kOk);
        REQUIRE(cpu_pipeline.LastExecutionBackend() ==
                inspirecv::task::ExecutionBackend::kCpu);

        inspirecv::task::PipelineOptions auto_options = cpu_options;
        auto_options.backend_preference =
          inspirecv::task::BackendPreference::kAuto;
        inspirecv::task::Pipeline auto_pipeline(auto_options);
        auto_pipeline.SetTransform(transform);
        destination.data = auto_output.data();
        REQUIRE(auto_pipeline.Run(source, destination) ==
                inspirecv::task::Status::kOk);
        REQUIRE(auto_pipeline.LastExecutionBackend() ==
                current.expected_backend);
        RequireExactFloatOutput(auto_output, cpu_output);
    }

    inspirecv::task::internal::PipelineConfig measured_config;
    measured_config.source_format = inspirecv::task::BGR;
    measured_config.destination_format = inspirecv::task::RGB;
    measured_config.filter = inspirecv::task::BILINEAR;
    inspirecv::task::internal::ConversionRequest measured_request;
    measured_request.source_width = 640;
    measured_request.source_height = 480;
    measured_request.source_stride = 640 * 3;
    measured_request.destination_width = 112;
    measured_request.destination_height = 112;
    measured_request.destination_layout = inspirecv::task::TensorLayout::NCHW;
    inspirecv::task::Matrix measured_transform;
    measured_transform.setScale(640.0f / 112.0f, 480.0f / 112.0f);
    REQUIRE(inspirecv::task::internal::CudaAutoRecommendsHost(
      measured_config, measured_transform, measured_request));
    measured_config.destination_format = inspirecv::task::BGR;
    REQUIRE_FALSE(inspirecv::task::internal::CudaAutoRecommendsHost(
      measured_config, measured_transform, measured_request));
    measured_config.destination_format = inspirecv::task::RGB;
    measured_config.filter = inspirecv::task::NEAREST;
    REQUIRE_FALSE(inspirecv::task::internal::CudaAutoRecommendsHost(
      measured_config, measured_transform, measured_request));
    measured_config.filter = inspirecv::task::BILINEAR;
    measured_request.destination_layout = inspirecv::task::TensorLayout::NHWC;
    REQUIRE_FALSE(inspirecv::task::internal::CudaAutoRecommendsHost(
      measured_config, measured_transform, measured_request));
    measured_request.destination_layout = inspirecv::task::TensorLayout::NCHW;
    measured_request.destination_height = 56;
    REQUIRE_FALSE(inspirecv::task::internal::CudaAutoRecommendsHost(
      measured_config, measured_transform, measured_request));

    // Per-pipeline overrides remain authoritative on either side of Auto's
    // boundary.
    const auto source_bytes = MakeCudaSource(112, 112);
    inspirecv::task::RawImageView source;
    source.data = source_bytes.data();
    source.width = 112;
    source.height = 112;
    source.row_stride_bytes = 112 * 3;
    std::vector<float> output(112 * 112 * 3);
    inspirecv::task::TensorBuffer destination;
    destination.data = output.data();
    destination.width = 112;
    destination.height = 112;
    destination.channels = 3;
    destination.element_type = inspirecv::task::ElementType::kFloat32;
    destination.order = inspirecv::task::TensorOrder::kChw;
    inspirecv::task::PipelineOptions force_options;
    force_options.input_format = inspirecv::task::PixelFormat::kBgr;
    force_options.output_format = inspirecv::task::PixelFormat::kRgb;
    force_options.sampling = inspirecv::task::SamplingMode::kLinear;
    force_options.backend_preference =
      inspirecv::task::BackendPreference::kCuda;
    inspirecv::task::Pipeline forced_cuda(force_options);
    forced_cuda.SetTransform(ResizeTransform(112, 112, 112, 112));
    REQUIRE(forced_cuda.Run(source, destination) ==
            inspirecv::task::Status::kOk);
    REQUIRE(forced_cuda.LastExecutionBackend() ==
            inspirecv::task::ExecutionBackend::kCuda);
}

#endif  // INSPIRECV_TASK_HAS_CUDA

TEST_CASE("task_cuda_host_pipeline_matches_cpu_fused_preprocessing",
          "[task][cuda][accuracy]") {
    DisableCudaAtExit cleanup;
    REQUIRE(inspirecv::task::SetCudaEnabled(false));
    if (!inspirecv::task::IsCudaAvailable()) {
        SUCCEED("CUDA backend is not part of this build or no GPU is available");
        return;
    }
    struct Case {
        int source_width;
        int source_height;
        int destination_width;
        int destination_height;
        inspirecv::task::SamplingMode sampling;
        inspirecv::task::PixelFormat source_format;
        inspirecv::task::PixelFormat destination_format;
        inspirecv::task::TensorOrder order;
    };
    const Case cases[] = {
      {640, 480, 112, 112, inspirecv::task::SamplingMode::kNearest,
       inspirecv::task::PixelFormat::kBgr,
       inspirecv::task::PixelFormat::kBgr,
       inspirecv::task::TensorOrder::kChw},
      {640, 480, 112, 112, inspirecv::task::SamplingMode::kLinear,
       inspirecv::task::PixelFormat::kBgr,
       inspirecv::task::PixelFormat::kRgb,
       inspirecv::task::TensorOrder::kChw},
      {1920, 1080, 224, 224, inspirecv::task::SamplingMode::kLinear,
       inspirecv::task::PixelFormat::kRgb,
       inspirecv::task::PixelFormat::kBgr,
       inspirecv::task::TensorOrder::kHwc},
      {3840, 2160, 640, 640, inspirecv::task::SamplingMode::kLinear,
       inspirecv::task::PixelFormat::kBgr,
       inspirecv::task::PixelFormat::kRgb,
       inspirecv::task::TensorOrder::kChw},
    };

    for (const Case& current : cases) {
        CAPTURE(current.source_width, current.source_height,
                current.destination_width, current.destination_height,
                static_cast<int>(current.sampling),
                static_cast<int>(current.order));
        const auto source_bytes =
          MakeCudaSource(current.source_width, current.source_height);
        inspirecv::task::RawImageView source;
        source.data = source_bytes.data();
        source.width = current.source_width;
        source.height = current.source_height;
        source.row_stride_bytes = current.source_width * 3;

        inspirecv::task::PipelineOptions options;
        options.input_format = current.source_format;
        options.output_format = current.destination_format;
        options.sampling = current.sampling;
        options.mean = {{127.5f, 103.25f, 91.75f, 0.0f}};
        options.scale = {{1.0f / 128.0f, 1.0f / 64.0f,
                          1.0f / 32.0f, 1.0f}};
        inspirecv::task::Pipeline pipeline(options);
        pipeline.SetTransform(ResizeTransform(
          current.source_width, current.source_height,
          current.destination_width, current.destination_height));

        const size_t elements = static_cast<size_t>(current.destination_width) *
                                current.destination_height * 3;
        std::vector<float> expected(elements);
        std::vector<float> actual(elements);
        inspirecv::task::TensorBuffer tensor;
        tensor.width = current.destination_width;
        tensor.height = current.destination_height;
        tensor.channels = 3;
        tensor.element_type = inspirecv::task::ElementType::kFloat32;
        tensor.order = current.order;
        tensor.data = expected.data();
        REQUIRE(pipeline.Run(source, tensor) == inspirecv::task::Status::kOk);
        REQUIRE(pipeline.LastExecutionBackend() ==
                inspirecv::task::ExecutionBackend::kCpu);

        REQUIRE(inspirecv::task::SetCudaEnabled(true));
        tensor.data = actual.data();
        REQUIRE(pipeline.Run(source, tensor) == inspirecv::task::Status::kOk);
        REQUIRE(pipeline.LastExecutionBackend() ==
                inspirecv::task::ExecutionBackend::kCuda);
        RequireExactFloatOutput(actual, expected);
        REQUIRE(inspirecv::task::SetCudaEnabled(false));
    }
    MaybeWriteCudaVisualReport();
}

TEST_CASE("task_cuda_host_pipeline_preserves_strides_and_affine_results",
          "[task][cuda][accuracy][stride][affine]") {
    DisableCudaAtExit cleanup;
    REQUIRE(inspirecv::task::SetCudaEnabled(false));
    if (!inspirecv::task::IsCudaAvailable()) {
        SUCCEED("CUDA backend is not part of this build or no GPU is available");
        return;
    }

    constexpr int kSourceWidth = 321;
    constexpr int kSourceHeight = 257;
    constexpr int kSourceStride = kSourceWidth * 3 + 17;
    constexpr int kDestinationWidth = 113;
    constexpr int kDestinationHeight = 127;
    std::vector<uint8_t> source_bytes(
      static_cast<size_t>(kSourceStride) * kSourceHeight, 0xa5);
    const auto packed_source = MakeCudaSource(kSourceWidth, kSourceHeight);
    for (int y = 0; y < kSourceHeight; ++y) {
        std::copy_n(packed_source.data() +
                      static_cast<size_t>(y) * kSourceWidth * 3,
                    kSourceWidth * 3,
                    source_bytes.data() + static_cast<size_t>(y) * kSourceStride);
    }

    inspirecv::task::RawImageView source;
    source.data = source_bytes.data();
    source.width = kSourceWidth;
    source.height = kSourceHeight;
    source.row_stride_bytes = kSourceStride;

    inspirecv::task::PipelineOptions options;
    options.input_format = inspirecv::task::PixelFormat::kBgr;
    options.output_format = inspirecv::task::PixelFormat::kRgb;
    options.sampling = inspirecv::task::SamplingMode::kLinear;
    options.mean = {{121.25f, 99.5f, 77.75f, 0.0f}};
    options.scale = {{1.0f / 127.0f, 1.0f / 63.0f,
                      1.0f / 31.0f, 1.0f}};
    inspirecv::task::Pipeline pipeline(options);
    pipeline.SetTransform(inspirecv::TransformMatrix(
      0.92f, 0.11f, -13.25f, -0.08f, 1.03f, 7.75f));

    SECTION("padded HWC rows") {
        constexpr size_t kRowFloats = kDestinationWidth * 3 + 7;
        const size_t storage_elements = kRowFloats * kDestinationHeight;
        std::vector<float> expected(storage_elements, -999.25f);
        std::vector<float> actual(storage_elements, -999.25f);
        inspirecv::task::TensorBuffer destination;
        destination.width = kDestinationWidth;
        destination.height = kDestinationHeight;
        destination.channels = 3;
        destination.element_type = inspirecv::task::ElementType::kFloat32;
        destination.order = inspirecv::task::TensorOrder::kHwc;
        destination.row_stride_bytes = kRowFloats * sizeof(float);

        destination.data = expected.data();
        REQUIRE(pipeline.Run(source, destination) ==
                inspirecv::task::Status::kOk);
        REQUIRE(pipeline.LastExecutionBackend() ==
                inspirecv::task::ExecutionBackend::kCpu);
        REQUIRE(inspirecv::task::SetCudaEnabled(true));
        destination.data = actual.data();
        REQUIRE(pipeline.Run(source, destination) ==
                inspirecv::task::Status::kOk);
        REQUIRE(pipeline.LastExecutionBackend() ==
                inspirecv::task::ExecutionBackend::kCuda);
        RequireExactFloatOutput(actual, expected);
    }

    SECTION("padded CHW rows and planes") {
        constexpr size_t kRowFloats = kDestinationWidth + 5;
        constexpr size_t kChannelFloats =
          kRowFloats * kDestinationHeight + 13;
        const size_t storage_elements =
          kChannelFloats * 2 + kRowFloats * kDestinationHeight;
        std::vector<float> expected(storage_elements, -777.5f);
        std::vector<float> actual(storage_elements, -777.5f);
        inspirecv::task::TensorBuffer destination;
        destination.width = kDestinationWidth;
        destination.height = kDestinationHeight;
        destination.channels = 3;
        destination.element_type = inspirecv::task::ElementType::kFloat32;
        destination.order = inspirecv::task::TensorOrder::kChw;
        destination.row_stride_bytes = kRowFloats * sizeof(float);
        destination.channel_stride_bytes = kChannelFloats * sizeof(float);

        destination.data = expected.data();
        REQUIRE(pipeline.Run(source, destination) ==
                inspirecv::task::Status::kOk);
        REQUIRE(pipeline.LastExecutionBackend() ==
                inspirecv::task::ExecutionBackend::kCpu);
        REQUIRE(inspirecv::task::SetCudaEnabled(true));
        destination.data = actual.data();
        REQUIRE(pipeline.Run(source, destination) ==
                inspirecv::task::Status::kOk);
        REQUIRE(pipeline.LastExecutionBackend() ==
                inspirecv::task::ExecutionBackend::kCuda);
        RequireExactFloatOutput(actual, expected);
    }
}

TEST_CASE("task_cuda_yuv420_fused_preprocessing_matches_cpu",
          "[task][cuda][accuracy][yuv420]") {
    DisableCudaAtExit cleanup;
    REQUIRE(inspirecv::task::SetCudaEnabled(false));
    if (!inspirecv::task::IsCudaAvailable()) {
        SUCCEED("CUDA backend is not part of this build or no GPU is available");
        return;
    }

    struct Case {
        inspirecv::task::PixelFormat source_format;
        inspirecv::task::PixelFormat destination_format;
        inspirecv::task::SamplingMode sampling;
        inspirecv::task::TensorOrder order;
    };
    const Case cases[] = {
      {inspirecv::task::PixelFormat::kNv12,
       inspirecv::task::PixelFormat::kRgb,
       inspirecv::task::SamplingMode::kNearest,
       inspirecv::task::TensorOrder::kChw},
      {inspirecv::task::PixelFormat::kNv21,
       inspirecv::task::PixelFormat::kBgr,
       inspirecv::task::SamplingMode::kLinear,
       inspirecv::task::TensorOrder::kHwc},
      {inspirecv::task::PixelFormat::kI420,
       inspirecv::task::PixelFormat::kRgb,
       inspirecv::task::SamplingMode::kNearest,
       inspirecv::task::TensorOrder::kHwc},
      {inspirecv::task::PixelFormat::kI420,
       inspirecv::task::PixelFormat::kBgr,
       inspirecv::task::SamplingMode::kLinear,
       inspirecv::task::TensorOrder::kChw},
    };

    constexpr int kSourceWidth = 128;
    constexpr int kSourceHeight = 96;
    constexpr int kDestinationWidth = 61;
    constexpr int kDestinationHeight = 47;
    const inspirecv::TransformMatrix affine(
      1.71f, 0.13f, -7.25f, -0.09f, 1.83f, 5.5f);
    for (const Case& current : cases) {
        CAPTURE(static_cast<int>(current.source_format),
                static_cast<int>(current.destination_format),
                static_cast<int>(current.sampling),
                static_cast<int>(current.order));
        const auto source_bytes = MakeYuv420Source(
          kSourceWidth, kSourceHeight, current.source_format);
        inspirecv::task::RawImageView source;
        source.data = source_bytes.data();
        source.width = kSourceWidth;
        source.height = kSourceHeight;
        source.row_stride_bytes = 0;

        inspirecv::task::PipelineOptions options;
        options.input_format = current.source_format;
        options.output_format = current.destination_format;
        options.sampling = current.sampling;
        options.mean = {{123.25f, 101.5f, 79.75f, 0.0f}};
        options.scale = {{1.0f / 127.0f, 1.0f / 63.0f,
                          1.0f / 31.0f, 1.0f}};
        inspirecv::task::Pipeline pipeline(options);
        REQUIRE(pipeline.SetTransform(affine) ==
                inspirecv::task::Status::kOk);

        const size_t elements =
          static_cast<size_t>(kDestinationWidth) * kDestinationHeight * 3;
        std::vector<float> expected(elements);
        std::vector<float> actual(elements);
        inspirecv::task::TensorBuffer destination;
        destination.width = kDestinationWidth;
        destination.height = kDestinationHeight;
        destination.channels = 3;
        destination.element_type = inspirecv::task::ElementType::kFloat32;
        destination.order = current.order;

        destination.data = expected.data();
        REQUIRE(pipeline.Run(source, destination) ==
                inspirecv::task::Status::kOk);
        REQUIRE(pipeline.LastExecutionBackend() ==
                inspirecv::task::ExecutionBackend::kCpu);

        REQUIRE(inspirecv::task::SetCudaEnabled(true));
        destination.data = actual.data();
        REQUIRE(pipeline.Run(source, destination) ==
                inspirecv::task::Status::kOk);
        REQUIRE(pipeline.LastExecutionBackend() ==
                inspirecv::task::ExecutionBackend::kCuda);
        RequireExactFloatOutput(actual, expected);
        REQUIRE(inspirecv::task::SetCudaEnabled(false));
    }
}

TEST_CASE("task_cuda_yuv420_device_rejects_odd_chroma_dimensions",
          "[task][cuda][device][contract][yuv420]") {
    DisableCudaAtExit cleanup;
    if (!inspirecv::task::IsCudaAvailable()) {
        SUCCEED("CUDA backend is not part of this build or no GPU is available");
        return;
    }
    REQUIRE(inspirecv::task::SetCudaEnabled(true));

    inspirecv::task::PipelineOptions options;
    options.input_format = inspirecv::task::PixelFormat::kNv12;
    options.output_format = inspirecv::task::PixelFormat::kRgb;
    inspirecv::task::cuda::Pipeline pipeline(options);

    inspirecv::task::cuda::DeviceImageView source;
    source.data = 1;
    source.width = 127;
    source.height = 96;
    source.row_stride_bytes = 0;
    inspirecv::task::cuda::DeviceTensorBuffer destination;
    destination.data = 1;
    destination.width = 61;
    destination.height = 47;
    destination.channels = 3;
    destination.element_type = inspirecv::task::ElementType::kFloat32;
    destination.order = inspirecv::task::TensorOrder::kChw;

    REQUIRE(pipeline.Run(source, destination) ==
            inspirecv::task::Status::kUnsupportedConversion);
    source.width = 128;
    source.height = 95;
    REQUIRE(pipeline.Run(source, destination) ==
            inspirecv::task::Status::kUnsupportedConversion);
}

#if defined(INSPIRECV_TASK_HAS_CUDA)

TEST_CASE("task_cuda_yuv420_performance",
          "[benchmark][task][cuda][yuv420]") {
    DisableCudaAtExit cleanup;
    REQUIRE(inspirecv::task::SetCudaEnabled(true));
    struct Case {
        int source_width;
        int source_height;
        int destination_width;
        int destination_height;
    };
    const Case cases[] = {
      {640, 480, 224, 224},
      {1920, 1080, 224, 224},
      {3840, 2160, 640, 640},
    };
    for (const Case& current : cases) {
        const auto source_bytes = MakeYuv420Source(
          current.source_width, current.source_height,
          inspirecv::task::PixelFormat::kNv12);
        inspirecv::task::RawImageView source;
        source.data = source_bytes.data();
        source.width = current.source_width;
        source.height = current.source_height;
        source.row_stride_bytes = 0;

        inspirecv::task::PipelineOptions options;
        options.input_format = inspirecv::task::PixelFormat::kNv12;
        options.output_format = inspirecv::task::PixelFormat::kRgb;
        options.sampling = inspirecv::task::SamplingMode::kNearest;
        options.mean = {{127.5f, 103.25f, 91.75f, 0.0f}};
        options.scale = {{1.0f / 128.0f, 1.0f / 64.0f,
                          1.0f / 32.0f, 1.0f}};
        inspirecv::task::Pipeline host(options);
        const auto transform = ResizeTransform(
          current.source_width, current.source_height,
          current.destination_width, current.destination_height);
        host.SetTransform(transform);
        const size_t output_elements =
          static_cast<size_t>(current.destination_width) *
          current.destination_height * 3;
        std::vector<float> host_output(output_elements);
        inspirecv::task::TensorBuffer destination;
        destination.data = host_output.data();
        destination.width = current.destination_width;
        destination.height = current.destination_height;
        destination.channels = 3;
        destination.element_type = inspirecv::task::ElementType::kFloat32;
        destination.order = inspirecv::task::TensorOrder::kChw;
        const TimingDistribution cpu = MeasureHostPipeline(
          &host, source, destination, false, 51);
        const TimingDistribution cuda_host = MeasureHostPipeline(
          &host, source, destination, true, 51);

        uint8_t* device_source = nullptr;
        float* device_destination = nullptr;
        REQUIRE(cudaMalloc(reinterpret_cast<void**>(&device_source),
                           source_bytes.size()) == cudaSuccess);
        REQUIRE(cudaMalloc(reinterpret_cast<void**>(&device_destination),
                           output_elements * sizeof(float)) == cudaSuccess);
        REQUIRE(cudaMemcpy(device_source, source_bytes.data(),
                           source_bytes.size(), cudaMemcpyHostToDevice) ==
                cudaSuccess);
        inspirecv::task::cuda::DeviceImageView device_view;
        device_view.data = reinterpret_cast<std::uintptr_t>(device_source);
        device_view.width = current.source_width;
        device_view.height = current.source_height;
        device_view.row_stride_bytes = 0;
        inspirecv::task::cuda::DeviceTensorBuffer device_tensor;
        device_tensor.data =
          reinterpret_cast<std::uintptr_t>(device_destination);
        device_tensor.width = current.destination_width;
        device_tensor.height = current.destination_height;
        device_tensor.channels = 3;
        device_tensor.element_type = inspirecv::task::ElementType::kFloat32;
        device_tensor.order = inspirecv::task::TensorOrder::kChw;
        inspirecv::task::cuda::Pipeline device(options);
        device.SetTransform(transform);
        cudaStream_t stream = nullptr;
        REQUIRE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) ==
                cudaSuccess);
        std::vector<double> samples;
        samples.reserve(51);
        for (int sample = -10; sample < 51; ++sample) {
            const auto begin = std::chrono::steady_clock::now();
            REQUIRE(device.Run(device_view, device_tensor,
                               reinterpret_cast<void*>(stream)) ==
                    inspirecv::task::Status::kOk);
            REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
            const auto end = std::chrono::steady_clock::now();
            if (sample >= 0) {
                samples.push_back(
                  std::chrono::duration<double, std::micro>(end - begin)
                    .count());
            }
        }
        const double device_p50 = Percentile(samples, 0.50);
        std::cout << "[TaskCudaYuvBenchmark] "
                  << current.source_width << 'x' << current.source_height
                  << "->" << current.destination_width << 'x'
                  << current.destination_height
                  << " cpu_p50_us=" << cpu.p50_microseconds
                  << " cuda_host_p50_us=" << cuda_host.p50_microseconds
                  << " cuda_device_p50_us=" << device_p50 << '\n';
        REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
        REQUIRE(cudaFree(device_destination) == cudaSuccess);
        REQUIRE(cudaFree(device_source) == cudaSuccess);
    }
}

#endif  // INSPIRECV_TASK_HAS_CUDA

TEST_CASE("task_cuda_enabled_pipeline_falls_back_for_unsupported_work",
          "[task][cuda][fallback][accuracy]") {
    DisableCudaAtExit cleanup;
    if (!inspirecv::task::IsCudaAvailable()) {
        SUCCEED("CUDA backend is not part of this build or no GPU is available");
        return;
    }

    constexpr int kSourceWidth = 96;
    constexpr int kSourceHeight = 64;
    constexpr int kDestinationWidth = 48;
    constexpr int kDestinationHeight = 32;
    std::vector<uint8_t> source_bytes(
      static_cast<size_t>(kSourceWidth) * kSourceHeight * 4);
    for (size_t index = 0; index < source_bytes.size(); ++index) {
        source_bytes[index] = static_cast<uint8_t>((index * 29 + 7) & 255);
    }
    inspirecv::task::RawImageView source;
    source.data = source_bytes.data();
    source.width = kSourceWidth;
    source.height = kSourceHeight;
    source.row_stride_bytes = kSourceWidth * 4;

    inspirecv::task::PipelineOptions options;
    options.input_format = inspirecv::task::PixelFormat::kBgra;
    options.output_format = inspirecv::task::PixelFormat::kBgr;
    options.sampling = inspirecv::task::SamplingMode::kLinear;
    inspirecv::task::Pipeline pipeline(options);
    pipeline.SetTransform(ResizeTransform(
      kSourceWidth, kSourceHeight, kDestinationWidth, kDestinationHeight));
    std::vector<float> expected(
      static_cast<size_t>(kDestinationWidth) * kDestinationHeight * 3);
    std::vector<float> actual(expected.size());
    inspirecv::task::TensorBuffer destination;
    destination.width = kDestinationWidth;
    destination.height = kDestinationHeight;
    destination.channels = 3;
    destination.element_type = inspirecv::task::ElementType::kFloat32;
    destination.order = inspirecv::task::TensorOrder::kChw;

    REQUIRE(inspirecv::task::SetCudaEnabled(false));
    destination.data = expected.data();
    REQUIRE(pipeline.Run(source, destination) == inspirecv::task::Status::kOk);
    REQUIRE(inspirecv::task::SetCudaEnabled(true));
    destination.data = actual.data();
    REQUIRE(pipeline.Run(source, destination) == inspirecv::task::Status::kOk);
    REQUIRE(pipeline.LastExecutionBackend() ==
            inspirecv::task::ExecutionBackend::kCpu);
    RequireExactFloatOutput(actual, expected);
}

#if defined(INSPIRECV_TASK_HAS_CUDA)

TEST_CASE("task_cuda_device_pipeline_matches_cpu_output",
          "[task][cuda][device][accuracy]") {
    DisableCudaAtExit cleanup;
    REQUIRE(inspirecv::task::SetCudaEnabled(true));
    constexpr int kSourceWidth = 640;
    constexpr int kSourceHeight = 480;
    constexpr int kDestinationWidth = 112;
    constexpr int kDestinationHeight = 112;
    const auto source_bytes = MakeCudaSource(kSourceWidth, kSourceHeight);
    const size_t output_elements =
      static_cast<size_t>(kDestinationWidth) * kDestinationHeight * 3;

    inspirecv::task::PipelineOptions options;
    options.input_format = inspirecv::task::PixelFormat::kBgr;
    options.output_format = inspirecv::task::PixelFormat::kRgb;
    options.sampling = inspirecv::task::SamplingMode::kLinear;
    options.mean = {{127.5f, 103.25f, 91.75f, 0.0f}};
    options.scale = {{1.0f / 128.0f, 1.0f / 64.0f,
                      1.0f / 32.0f, 1.0f}};
    const auto transform = ResizeTransform(
      kSourceWidth, kSourceHeight, kDestinationWidth, kDestinationHeight);

    std::vector<float> expected(output_elements);
    inspirecv::task::RawImageView host_source;
    host_source.data = source_bytes.data();
    host_source.width = kSourceWidth;
    host_source.height = kSourceHeight;
    host_source.row_stride_bytes = kSourceWidth * 3;
    inspirecv::task::TensorBuffer host_destination;
    host_destination.data = expected.data();
    host_destination.width = kDestinationWidth;
    host_destination.height = kDestinationHeight;
    host_destination.channels = 3;
    host_destination.element_type = inspirecv::task::ElementType::kFloat32;
    host_destination.order = inspirecv::task::TensorOrder::kChw;
    inspirecv::task::SetCudaEnabled(false);
    inspirecv::task::Pipeline cpu(options);
    cpu.SetTransform(transform);
    REQUIRE(cpu.Run(host_source, host_destination) ==
            inspirecv::task::Status::kOk);
    REQUIRE(inspirecv::task::SetCudaEnabled(true));

    uint8_t* device_source = nullptr;
    float* device_destination = nullptr;
    REQUIRE(cudaMalloc(reinterpret_cast<void**>(&device_source),
                       source_bytes.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(reinterpret_cast<void**>(&device_destination),
                       output_elements * sizeof(float)) == cudaSuccess);
    REQUIRE(cudaMemcpy(device_source, source_bytes.data(), source_bytes.size(),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    inspirecv::task::cuda::DeviceImageView source;
    source.data = reinterpret_cast<std::uintptr_t>(device_source);
    source.width = kSourceWidth;
    source.height = kSourceHeight;
    source.row_stride_bytes = kSourceWidth * 3;
    inspirecv::task::cuda::DeviceTensorBuffer destination;
    destination.data = reinterpret_cast<std::uintptr_t>(device_destination);
    destination.width = kDestinationWidth;
    destination.height = kDestinationHeight;
    destination.channels = 3;
    destination.element_type = inspirecv::task::ElementType::kFloat32;
    destination.order = inspirecv::task::TensorOrder::kChw;
    inspirecv::task::cuda::Pipeline gpu(options);
    gpu.SetTransform(transform);
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) ==
            cudaSuccess);
    REQUIRE(gpu.Run(source, destination, reinterpret_cast<void*>(stream)) ==
            inspirecv::task::Status::kOk);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    std::vector<float> actual(output_elements);
    REQUIRE(cudaMemcpy(actual.data(), device_destination,
                       output_elements * sizeof(float),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    RequireExactFloatOutput(actual, expected);
    REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
    REQUIRE(cudaFree(device_destination) == cudaSuccess);
    REQUIRE(cudaFree(device_source) == cudaSuccess);
}

TEST_CASE("task_cuda_accepts_device_resident_image_without_host_bridge",
          "[task][cuda][device][image_bridge][accuracy]") {
    DisableCudaAtExit cleanup;
    constexpr int kSourceWidth = 640;
    constexpr int kSourceHeight = 480;
    constexpr int kIntermediateWidth = 320;
    constexpr int kIntermediateHeight = 240;
    constexpr int kDestinationWidth = 112;
    constexpr int kDestinationHeight = 112;
    const auto source_bytes = MakeCudaSource(kSourceWidth, kSourceHeight);
    const auto host_source = inspirecv::Image::Create(
      kSourceWidth, kSourceHeight, 3, source_bytes.data());

    inspirecv::task::PipelineOptions options;
    options.input_format = inspirecv::task::PixelFormat::kBgr;
    options.output_format = inspirecv::task::PixelFormat::kRgb;
    options.sampling = inspirecv::task::SamplingMode::kLinear;
    options.mean = {{127.5f, 103.25f, 91.75f, 0.0f}};
    options.scale = {{1.0f / 128.0f, 1.0f / 64.0f,
                      1.0f / 32.0f, 1.0f}};
    const auto task_transform = ResizeTransform(
      kIntermediateWidth, kIntermediateHeight,
      kDestinationWidth, kDestinationHeight);
    const size_t output_elements =
      static_cast<size_t>(kDestinationWidth) * kDestinationHeight * 3;

    REQUIRE(inspirecv::task::SetCudaEnabled(false));
    const auto host_intermediate = host_source.Resize(
      kIntermediateWidth, kIntermediateHeight, true);
    std::vector<float> expected(output_elements);
    inspirecv::task::TensorBuffer host_destination;
    host_destination.data = expected.data();
    host_destination.width = kDestinationWidth;
    host_destination.height = kDestinationHeight;
    host_destination.channels = 3;
    host_destination.element_type = inspirecv::task::ElementType::kFloat32;
    host_destination.order = inspirecv::task::TensorOrder::kChw;
    inspirecv::task::Pipeline cpu(options);
    cpu.SetTransform(task_transform);
    REQUIRE(cpu.Run(host_intermediate, host_destination) ==
            inspirecv::task::Status::kOk);

    REQUIRE(inspirecv::task::SetCudaEnabled(true));
    inspirecv::cuda::DeviceImage uploaded;
    inspirecv::cuda::DeviceImage resized;
    REQUIRE(inspirecv::cuda::DeviceImage::Upload(host_source, &uploaded) ==
            inspirecv::cuda::Status::kOk);
    REQUIRE(uploaded.Resize(kIntermediateWidth, kIntermediateHeight, true,
                            &resized) == inspirecv::cuda::Status::kOk);

    float* device_destination = nullptr;
    REQUIRE(cudaMalloc(reinterpret_cast<void**>(&device_destination),
                       output_elements * sizeof(float)) == cudaSuccess);
    inspirecv::task::cuda::DeviceTensorBuffer destination;
    destination.data =
      reinterpret_cast<std::uintptr_t>(device_destination);
    destination.width = kDestinationWidth;
    destination.height = kDestinationHeight;
    destination.channels = 3;
    destination.element_type = inspirecv::task::ElementType::kFloat32;
    destination.order = inspirecv::task::TensorOrder::kChw;
    inspirecv::task::cuda::Pipeline gpu(options);
    gpu.SetTransform(task_transform);
    REQUIRE(gpu.Run(resized, destination) == inspirecv::task::Status::kOk);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    std::vector<float> actual(output_elements);
    REQUIRE(cudaMemcpy(actual.data(), device_destination,
                       output_elements * sizeof(float),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    RequireExactFloatOutput(actual, expected);
    REQUIRE(cudaFree(device_destination) == cudaSuccess);
}

TEST_CASE("task_cuda_device_batch_matches_individual_cpu_outputs",
          "[task][cuda][device][batch][accuracy]") {
    DisableCudaAtExit cleanup;
    REQUIRE(inspirecv::task::SetCudaEnabled(true));
    constexpr size_t kBatch = 3;
    constexpr int kSourceWidth = 320;
    constexpr int kSourceHeight = 240;
    constexpr int kDestinationWidth = 112;
    constexpr int kDestinationHeight = 112;
    const size_t source_bytes =
      static_cast<size_t>(kSourceWidth) * kSourceHeight * 3;
    const size_t output_elements =
      static_cast<size_t>(kDestinationWidth) * kDestinationHeight * 3;

    inspirecv::task::PipelineOptions options;
    options.input_format = inspirecv::task::PixelFormat::kBgr;
    options.output_format = inspirecv::task::PixelFormat::kRgb;
    options.sampling = inspirecv::task::SamplingMode::kLinear;
    options.mean = {{127.5f, 103.25f, 91.75f, 0.0f}};
    options.scale = {{1.0f / 128.0f, 1.0f / 64.0f,
                      1.0f / 32.0f, 1.0f}};
    const auto transform = ResizeTransform(
      kSourceWidth, kSourceHeight, kDestinationWidth, kDestinationHeight);

    std::vector<std::vector<uint8_t>> host_sources(kBatch);
    std::vector<std::vector<float>> expected(kBatch);
    inspirecv::task::Pipeline cpu(options);
    cpu.SetTransform(transform);
    REQUIRE(inspirecv::task::SetCudaEnabled(false));
    for (size_t batch = 0; batch < kBatch; ++batch) {
        host_sources[batch] = MakeCudaSource(kSourceWidth, kSourceHeight);
        for (size_t index = 0; index < host_sources[batch].size(); ++index) {
            host_sources[batch][index] = static_cast<uint8_t>(
              host_sources[batch][index] + 23 * batch);
        }
        expected[batch].resize(output_elements);
        inspirecv::task::RawImageView source;
        source.data = host_sources[batch].data();
        source.width = kSourceWidth;
        source.height = kSourceHeight;
        source.row_stride_bytes = kSourceWidth * 3;
        inspirecv::task::TensorBuffer destination;
        destination.data = expected[batch].data();
        destination.width = kDestinationWidth;
        destination.height = kDestinationHeight;
        destination.channels = 3;
        destination.element_type = inspirecv::task::ElementType::kFloat32;
        destination.order = inspirecv::task::TensorOrder::kChw;
        REQUIRE(cpu.Run(source, destination) == inspirecv::task::Status::kOk);
    }

    REQUIRE(inspirecv::task::SetCudaEnabled(true));
    std::vector<uint8_t*> device_sources(kBatch, nullptr);
    std::vector<float*> device_destinations(kBatch, nullptr);
    std::vector<inspirecv::task::cuda::DeviceImageView> sources(kBatch);
    std::vector<inspirecv::task::cuda::DeviceTensorBuffer> destinations(kBatch);
    for (size_t batch = 0; batch < kBatch; ++batch) {
        REQUIRE(cudaMalloc(reinterpret_cast<void**>(&device_sources[batch]),
                           source_bytes) == cudaSuccess);
        REQUIRE(cudaMalloc(reinterpret_cast<void**>(&device_destinations[batch]),
                           output_elements * sizeof(float)) == cudaSuccess);
        REQUIRE(cudaMemcpy(device_sources[batch], host_sources[batch].data(),
                           source_bytes, cudaMemcpyHostToDevice) == cudaSuccess);
        sources[batch].data =
          reinterpret_cast<std::uintptr_t>(device_sources[batch]);
        sources[batch].width = kSourceWidth;
        sources[batch].height = kSourceHeight;
        sources[batch].row_stride_bytes = kSourceWidth * 3;
        destinations[batch].data =
          reinterpret_cast<std::uintptr_t>(device_destinations[batch]);
        destinations[batch].width = kDestinationWidth;
        destinations[batch].height = kDestinationHeight;
        destinations[batch].channels = 3;
        destinations[batch].element_type =
          inspirecv::task::ElementType::kFloat32;
        destinations[batch].order = inspirecv::task::TensorOrder::kChw;
    }

    inspirecv::task::cuda::Pipeline gpu(options);
    gpu.SetTransform(transform);
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) ==
            cudaSuccess);
    REQUIRE(gpu.RunBatch(sources.data(), destinations.data(), kBatch,
                         reinterpret_cast<void*>(stream)) ==
            inspirecv::task::Status::kOk);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);

    for (size_t batch = 0; batch < kBatch; ++batch) {
        std::vector<float> actual(output_elements);
        REQUIRE(cudaMemcpy(actual.data(), device_destinations[batch],
                           output_elements * sizeof(float),
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        RequireExactFloatOutput(actual, expected[batch]);
        REQUIRE(cudaFree(device_destinations[batch]) == cudaSuccess);
        REQUIRE(cudaFree(device_sources[batch]) == cudaSuccess);
    }
    REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
}

TEST_CASE("task_cuda_device_batch_performance",
          "[benchmark][task][cuda][device][batch]") {
    DisableCudaAtExit cleanup;
    REQUIRE(inspirecv::task::SetCudaEnabled(true));
    constexpr size_t kMaximumBatch = 16;
    constexpr int kSourceWidth = 320;
    constexpr int kSourceHeight = 240;
    constexpr int kDestinationWidth = 112;
    constexpr int kDestinationHeight = 112;
    const auto host_source = MakeCudaSource(kSourceWidth, kSourceHeight);
    const size_t output_elements =
      static_cast<size_t>(kDestinationWidth) * kDestinationHeight * 3;

    inspirecv::task::PipelineOptions options;
    options.input_format = inspirecv::task::PixelFormat::kBgr;
    options.output_format = inspirecv::task::PixelFormat::kRgb;
    options.sampling = inspirecv::task::SamplingMode::kLinear;
    options.mean = {{127.5f, 103.25f, 91.75f, 0.0f}};
    options.scale = {{1.0f / 128.0f, 1.0f / 64.0f,
                      1.0f / 32.0f, 1.0f}};
    inspirecv::task::cuda::Pipeline pipeline(options);
    pipeline.SetTransform(ResizeTransform(
      kSourceWidth, kSourceHeight, kDestinationWidth, kDestinationHeight));

    std::vector<uint8_t*> device_sources(kMaximumBatch, nullptr);
    std::vector<float*> device_destinations(kMaximumBatch, nullptr);
    std::vector<inspirecv::task::cuda::DeviceImageView> sources(kMaximumBatch);
    std::vector<inspirecv::task::cuda::DeviceTensorBuffer> destinations(
      kMaximumBatch);
    for (size_t index = 0; index < kMaximumBatch; ++index) {
        REQUIRE(cudaMalloc(reinterpret_cast<void**>(&device_sources[index]),
                           host_source.size()) == cudaSuccess);
        REQUIRE(cudaMalloc(reinterpret_cast<void**>(&device_destinations[index]),
                           output_elements * sizeof(float)) == cudaSuccess);
        REQUIRE(cudaMemcpy(device_sources[index], host_source.data(),
                           host_source.size(), cudaMemcpyHostToDevice) ==
                cudaSuccess);
        sources[index].data =
          reinterpret_cast<std::uintptr_t>(device_sources[index]);
        sources[index].width = kSourceWidth;
        sources[index].height = kSourceHeight;
        sources[index].row_stride_bytes = kSourceWidth * 3;
        destinations[index].data =
          reinterpret_cast<std::uintptr_t>(device_destinations[index]);
        destinations[index].width = kDestinationWidth;
        destinations[index].height = kDestinationHeight;
        destinations[index].channels = 3;
        destinations[index].element_type =
          inspirecv::task::ElementType::kFloat32;
        destinations[index].order = inspirecv::task::TensorOrder::kChw;
    }
    cudaStream_t stream = nullptr;
    REQUIRE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) ==
            cudaSuccess);

    const size_t batch_sizes[] = {1, 4, 8, 16};
    for (size_t batch_size : batch_sizes) {
        auto measure = [&](bool batched) {
            std::vector<double> timings;
            timings.reserve(51);
            for (int warmup = -5; warmup < 51; ++warmup) {
                const auto begin = std::chrono::steady_clock::now();
                if (batched) {
                    REQUIRE(pipeline.RunBatch(
                              sources.data(), destinations.data(), batch_size,
                              reinterpret_cast<void*>(stream)) ==
                            inspirecv::task::Status::kOk);
                    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
                } else {
                    for (size_t index = 0; index < batch_size; ++index) {
                        REQUIRE(pipeline.Run(
                                  sources[index], destinations[index],
                                  reinterpret_cast<void*>(stream)) ==
                                inspirecv::task::Status::kOk);
                        REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
                    }
                }
                const auto end = std::chrono::steady_clock::now();
                if (warmup >= 0) {
                    timings.push_back(
                      std::chrono::duration<double, std::micro>(end - begin)
                        .count());
                }
            }
            std::sort(timings.begin(), timings.end());
            return timings[timings.size() / 2];
        };
        const double serialized_us = measure(false);
        const double batch_us = measure(true);
        std::cout << "[TaskCudaBatchBenchmark] batch=" << batch_size
                  << " serialized_us=" << serialized_us
                  << " batch_us=" << batch_us
                  << " batch_us_per_image=" << batch_us / batch_size
                  << " speedup=" << serialized_us / batch_us << 'x' << '\n';
    }

    REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
    for (size_t index = 0; index < kMaximumBatch; ++index) {
        REQUIRE(cudaFree(device_destinations[index]) == cudaSuccess);
        REQUIRE(cudaFree(device_sources[index]) == cudaSuccess);
    }
}

TEST_CASE("task_cuda_multiresolution_performance",
          "[benchmark][task][cuda]") {
    DisableCudaAtExit cleanup;
    REQUIRE(inspirecv::task::SetCudaEnabled(true));
    struct Case {
        int source_width;
        int source_height;
        int destination_width;
        int destination_height;
        int iterations;
    };
    const Case cases[] = {
      {112, 112, 112, 112, 800},
      {320, 240, 112, 112, 600},
      {640, 480, 112, 112, 400},
      {1280, 720, 224, 224, 200},
      {1920, 1080, 224, 224, 120},
      {3840, 2160, 640, 640, 30},
    };

    for (const Case& current : cases) {
        const auto source_bytes =
          MakeCudaSource(current.source_width, current.source_height);
        const size_t output_elements =
          static_cast<size_t>(current.destination_width) *
          current.destination_height * 3;
        std::vector<float> host_output(output_elements);
        inspirecv::task::RawImageView source;
        source.data = source_bytes.data();
        source.width = current.source_width;
        source.height = current.source_height;
        source.row_stride_bytes = current.source_width * 3;
        inspirecv::task::TensorBuffer destination;
        destination.data = host_output.data();
        destination.width = current.destination_width;
        destination.height = current.destination_height;
        destination.channels = 3;
        destination.element_type = inspirecv::task::ElementType::kFloat32;
        destination.order = inspirecv::task::TensorOrder::kChw;

        inspirecv::task::PipelineOptions options;
        options.input_format = inspirecv::task::PixelFormat::kBgr;
        options.output_format = inspirecv::task::PixelFormat::kRgb;
        options.sampling = inspirecv::task::SamplingMode::kLinear;
        options.mean = {{127.5f, 127.5f, 127.5f, 0.0f}};
        options.scale = {{1.0f / 128.0f, 1.0f / 128.0f,
                          1.0f / 128.0f, 1.0f}};
        const auto transform = ResizeTransform(
          current.source_width, current.source_height,
          current.destination_width, current.destination_height);
        inspirecv::task::Pipeline host_pipeline(options);
        host_pipeline.SetTransform(transform);

        auto measure_host = [&](bool cuda_enabled) {
            REQUIRE(inspirecv::task::SetCudaEnabled(cuda_enabled));
            for (int warmup = 0; warmup < 5; ++warmup) {
                REQUIRE(host_pipeline.Run(source, destination) ==
                        inspirecv::task::Status::kOk);
            }
            const auto start = std::chrono::steady_clock::now();
            for (int iteration = 0; iteration < current.iterations;
                 ++iteration) {
                REQUIRE(host_pipeline.Run(source, destination) ==
                        inspirecv::task::Status::kOk);
            }
            const auto stop = std::chrono::steady_clock::now();
            return std::chrono::duration<double, std::micro>(stop - start).count() /
                   current.iterations;
        };
        const double cpu_microseconds = measure_host(false);
        const std::vector<float> expected_output = host_output;
        const double cuda_roundtrip_microseconds = measure_host(true);
        REQUIRE(host_pipeline.LastExecutionBackend() ==
                inspirecv::task::ExecutionBackend::kCuda);
        RequireExactFloatOutput(host_output, expected_output);

        uint8_t* device_source = nullptr;
        float* device_destination = nullptr;
        REQUIRE(cudaMalloc(reinterpret_cast<void**>(&device_source),
                           source_bytes.size()) == cudaSuccess);
        REQUIRE(cudaMalloc(reinterpret_cast<void**>(&device_destination),
                           output_elements * sizeof(float)) == cudaSuccess);
        REQUIRE(cudaMemcpy(device_source, source_bytes.data(), source_bytes.size(),
                           cudaMemcpyHostToDevice) == cudaSuccess);
        inspirecv::task::cuda::DeviceImageView device_input;
        device_input.data = reinterpret_cast<std::uintptr_t>(device_source);
        device_input.width = current.source_width;
        device_input.height = current.source_height;
        device_input.row_stride_bytes = current.source_width * 3;
        inspirecv::task::cuda::DeviceTensorBuffer device_output;
        device_output.data =
          reinterpret_cast<std::uintptr_t>(device_destination);
        device_output.width = current.destination_width;
        device_output.height = current.destination_height;
        device_output.channels = 3;
        device_output.element_type = inspirecv::task::ElementType::kFloat32;
        device_output.order = inspirecv::task::TensorOrder::kChw;
        inspirecv::task::cuda::Pipeline device_pipeline(options);
        device_pipeline.SetTransform(transform);
        for (int warmup = 0; warmup < 5; ++warmup) {
            REQUIRE(device_pipeline.Run(device_input, device_output) ==
                    inspirecv::task::Status::kOk);
        }
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        cudaEvent_t begin = nullptr;
        cudaEvent_t end = nullptr;
        REQUIRE(cudaEventCreate(&begin) == cudaSuccess);
        REQUIRE(cudaEventCreate(&end) == cudaSuccess);
        REQUIRE(cudaEventRecord(begin) == cudaSuccess);
        for (int iteration = 0; iteration < current.iterations; ++iteration) {
            REQUIRE(device_pipeline.Run(device_input, device_output) ==
                    inspirecv::task::Status::kOk);
        }
        REQUIRE(cudaEventRecord(end) == cudaSuccess);
        REQUIRE(cudaEventSynchronize(end) == cudaSuccess);
        float elapsed_milliseconds = 0.0f;
        REQUIRE(cudaEventElapsedTime(&elapsed_milliseconds, begin, end) ==
                cudaSuccess);
        const double device_microseconds =
          elapsed_milliseconds * 1000.0 / current.iterations;
        std::vector<float> device_result(output_elements);
        REQUIRE(cudaMemcpy(device_result.data(), device_destination,
                           output_elements * sizeof(float),
                           cudaMemcpyDeviceToHost) == cudaSuccess);
        RequireExactFloatOutput(device_result, expected_output);
        REQUIRE(cudaEventDestroy(end) == cudaSuccess);
        REQUIRE(cudaEventDestroy(begin) == cudaSuccess);
        REQUIRE(cudaFree(device_destination) == cudaSuccess);
        REQUIRE(cudaFree(device_source) == cudaSuccess);

        std::printf(
          "[TaskCudaBench] source=%dx%d output=%dx%d cpu_us=%.3f "
          "cuda_roundtrip_us=%.3f cuda_device_us=%.3f "
          "roundtrip_speedup=%.3fx device_speedup=%.3fx\n",
          current.source_width, current.source_height,
          current.destination_width, current.destination_height,
          cpu_microseconds, cuda_roundtrip_microseconds, device_microseconds,
          cpu_microseconds / cuda_roundtrip_microseconds,
          cpu_microseconds / device_microseconds);
    }
}

TEST_CASE("task_cuda_host_auto_threshold_benchmark",
          "[benchmark][task][cuda][threshold]") {
    DisableCudaAtExit cleanup;
    if (!inspirecv::task::IsCudaAvailable()) {
        SUCCEED("CUDA backend is not part of this build or no GPU is available");
        return;
    }

    struct Case {
        const char* family;
        int source_width;
        int source_height;
        int destination_width;
        int destination_height;
    };
    const Case cases[] = {
      {"identity", 64, 64, 64, 64},
      {"identity", 96, 96, 96, 96},
      {"identity", 112, 112, 112, 112},
      {"identity", 128, 128, 128, 128},
      {"identity", 160, 160, 160, 160},
      {"identity", 192, 192, 192, 192},
      {"identity", 224, 224, 224, 224},
      {"identity", 256, 256, 256, 256},
      {"identity", 320, 320, 320, 320},
      {"identity", 384, 384, 384, 384},
      {"identity", 512, 512, 512, 512},
      {"identity", 640, 640, 640, 640},
      {"identity", 768, 768, 768, 768},
      {"identity", 1024, 1024, 1024, 1024},

      {"resize_112", 160, 120, 112, 112},
      {"resize_112", 224, 224, 112, 112},
      {"resize_112", 320, 240, 112, 112},
      {"resize_112", 480, 360, 112, 112},
      {"resize_112", 640, 480, 112, 112},
      {"resize_112", 960, 540, 112, 112},
      {"resize_112", 1280, 720, 112, 112},
      {"resize_112", 1920, 1080, 112, 112},
      {"resize_112", 2560, 1440, 112, 112},
      {"resize_112", 3840, 2160, 112, 112},

      {"resize_224", 320, 240, 224, 224},
      {"resize_224", 480, 360, 224, 224},
      {"resize_224", 640, 480, 224, 224},
      {"resize_224", 960, 540, 224, 224},
      {"resize_224", 1280, 720, 224, 224},
      {"resize_224", 1920, 1080, 224, 224},
      {"resize_224", 2560, 1440, 224, 224},
      {"resize_224", 3840, 2160, 224, 224},

      {"resize_640", 960, 540, 640, 640},
      {"resize_640", 1280, 720, 640, 640},
      {"resize_640", 1920, 1080, 640, 640},
      {"resize_640", 2560, 1440, 640, 640},
      {"resize_640", 3840, 2160, 640, 640},
    };

    int sample_count = 101;
    if (const char* configured =
          std::getenv("INSPIRECV_CUDA_THRESHOLD_SAMPLES")) {
        sample_count = std::max(21, std::atoi(configured));
    }

    cudaDeviceProp device{};
    int runtime_version = 0;
    int driver_version = 0;
    REQUIRE(cudaGetDeviceProperties(&device, 0) == cudaSuccess);
    REQUIRE(cudaRuntimeGetVersion(&runtime_version) == cudaSuccess);
    REQUIRE(cudaDriverGetVersion(&driver_version) == cudaSuccess);

    std::ofstream report;
    if (const char* report_path =
          std::getenv("INSPIRECV_CUDA_THRESHOLD_REPORT")) {
        report.open(report_path, std::ios::out | std::ios::trunc);
        REQUIRE(report.good());
        report << "# benchmark=task_cuda_host_auto_threshold\n"
               << "# inspirecv=" << inspirecv::GetVersion() << "\n"
               << "# gpu=" << device.name << "\n"
               << "# compute_capability=" << device.major << '.'
               << device.minor << "\n"
               << "# gpu_memory_bytes=" << device.totalGlobalMem << "\n"
               << "# multiprocessors=" << device.multiProcessorCount << "\n"
               << "# cuda_runtime=" << runtime_version / 1000 << '.'
               << (runtime_version % 1000) / 10 << "\n"
               << "# cuda_driver_api=" << driver_version / 1000 << '.'
               << (driver_version % 1000) / 10 << "\n"
               << "# nvidia_driver=" << ReadLinuxNvidiaDriverVersion()
               << "\n"
               << "# cpu=" << ReadLinuxCpuModel() << "\n"
               << "# os=" << OperatingSystemDescription() << "\n"
               << "# compiler=" << CompilerDescription() << "\n"
#if defined(NDEBUG)
               << "# build_type=Release\n"
#else
               << "# build_type=Debug\n"
#endif
               << "# samples=" << sample_count << "\n"
               << "# recommendation=cuda_p50_at_least_15_percent_faster_and_cuda_p95_not_slower\n"
               << "family,source_width,source_height,destination_width,"
                  "destination_height,source_bytes,destination_pixels,"
                  "source_to_destination_ratio,cpu_p50_us,cpu_p95_us,"
                  "cuda_roundtrip_p50_us,cuda_roundtrip_p95_us,"
                  "p50_speedup,recommended_backend\n";
    }

    std::printf(
      "[TaskCudaThreshold] gpu=%s cc=%d.%d runtime=%d.%d driver_api=%d.%d "
      "samples=%d\n",
      device.name, device.major, device.minor, runtime_version / 1000,
      (runtime_version % 1000) / 10, driver_version / 1000,
      (driver_version % 1000) / 10, sample_count);

    for (const Case& current : cases) {
        const auto source_bytes =
          MakeCudaSource(current.source_width, current.source_height);
        const size_t destination_pixels =
          static_cast<size_t>(current.destination_width) *
          current.destination_height;
        std::vector<float> output(destination_pixels * 3);

        inspirecv::task::RawImageView source;
        source.data = source_bytes.data();
        source.width = current.source_width;
        source.height = current.source_height;
        source.row_stride_bytes =
          static_cast<size_t>(current.source_width) * 3;
        inspirecv::task::TensorBuffer destination;
        destination.data = output.data();
        destination.width = current.destination_width;
        destination.height = current.destination_height;
        destination.channels = 3;
        destination.element_type = inspirecv::task::ElementType::kFloat32;
        destination.order = inspirecv::task::TensorOrder::kChw;

        inspirecv::task::PipelineOptions options;
        options.input_format = inspirecv::task::PixelFormat::kBgr;
        options.output_format = inspirecv::task::PixelFormat::kRgb;
        options.sampling = inspirecv::task::SamplingMode::kLinear;
        options.mean = {{127.5f, 127.5f, 127.5f, 0.0f}};
        options.scale = {{1.0f / 128.0f, 1.0f / 128.0f,
                          1.0f / 128.0f, 1.0f}};
        inspirecv::task::Pipeline pipeline(options);
        pipeline.SetTransform(ResizeTransform(
          current.source_width, current.source_height,
          current.destination_width, current.destination_height));

        const TimingDistribution cpu = MeasureHostPipeline(
          &pipeline, source, destination, false, sample_count);
        const std::vector<float> expected = output;
        const TimingDistribution cuda = MeasureHostPipeline(
          &pipeline, source, destination, true, sample_count);
        RequireExactFloatOutput(output, expected);

        const double speedup =
          cpu.p50_microseconds / cuda.p50_microseconds;
        const bool recommend_cuda =
          cuda.p50_microseconds <= cpu.p50_microseconds * 0.85 &&
          cuda.p95_microseconds <= cpu.p95_microseconds;
        const double source_to_destination_ratio =
          static_cast<double>(current.source_width) * current.source_height /
          static_cast<double>(destination_pixels);
        const char* recommendation = recommend_cuda ? "cuda" : "cpu";

        std::printf(
          "[TaskCudaThreshold] family=%s source=%dx%d output=%dx%d "
          "cpu_p50_us=%.3f cpu_p95_us=%.3f cuda_p50_us=%.3f "
          "cuda_p95_us=%.3f speedup=%.3fx recommended=%s\n",
          current.family, current.source_width, current.source_height,
          current.destination_width, current.destination_height,
          cpu.p50_microseconds, cpu.p95_microseconds,
          cuda.p50_microseconds, cuda.p95_microseconds, speedup,
          recommendation);
        if (report.good()) {
            report << current.family << ',' << current.source_width << ','
                   << current.source_height << ','
                   << current.destination_width << ','
                   << current.destination_height << ',' << source_bytes.size()
                   << ',' << destination_pixels << ',' << std::fixed
                   << std::setprecision(6) << source_to_destination_ratio << ','
                   << std::setprecision(3) << cpu.p50_microseconds << ','
                   << cpu.p95_microseconds << ',' << cuda.p50_microseconds << ','
                   << cuda.p95_microseconds << ',' << speedup << ','
                   << recommendation << '\n';
        }
    }
}

TEST_CASE("task_cuda_auto_dispatch_performance",
          "[benchmark][task][cuda][auto]") {
    DisableCudaAtExit cleanup;
    if (!inspirecv::task::IsCudaAvailable()) {
        SUCCEED("CUDA backend is not part of this build or no GPU is available");
        return;
    }
    REQUIRE(inspirecv::task::SetCudaEnabled(true));

    struct Case {
        int source_width;
        int source_height;
        int destination_width;
        int destination_height;
        inspirecv::task::ExecutionBackend expected_backend;
    };
    const Case cases[] = {
      {112, 112, 112, 112, inspirecv::task::ExecutionBackend::kCpu},
      {640, 480, 112, 112, inspirecv::task::ExecutionBackend::kCuda},
      {960, 540, 112, 112, inspirecv::task::ExecutionBackend::kCpu},
      {1920, 1080, 224, 224, inspirecv::task::ExecutionBackend::kCuda},
      {2560, 1440, 224, 224, inspirecv::task::ExecutionBackend::kCpu},
      {3840, 2160, 640, 640, inspirecv::task::ExecutionBackend::kCuda},
    };

    for (const Case& current : cases) {
        const auto source_bytes =
          MakeCudaSource(current.source_width, current.source_height);
        inspirecv::task::RawImageView source;
        source.data = source_bytes.data();
        source.width = current.source_width;
        source.height = current.source_height;
        source.row_stride_bytes =
          static_cast<size_t>(current.source_width) * 3;
        const size_t output_elements =
          static_cast<size_t>(current.destination_width) *
          current.destination_height * 3;
        std::vector<float> cpu_output(output_elements);
        std::vector<float> cuda_output(output_elements);
        std::vector<float> auto_output(output_elements);

        inspirecv::task::PipelineOptions base_options;
        base_options.input_format = inspirecv::task::PixelFormat::kBgr;
        base_options.output_format = inspirecv::task::PixelFormat::kRgb;
        base_options.sampling = inspirecv::task::SamplingMode::kLinear;
        base_options.mean = {{127.5f, 127.5f, 127.5f, 0.0f}};
        base_options.scale = {{1.0f / 128.0f, 1.0f / 128.0f,
                               1.0f / 128.0f, 1.0f}};
        const auto transform = ResizeTransform(
          current.source_width, current.source_height,
          current.destination_width, current.destination_height);

        inspirecv::task::PipelineOptions cpu_options = base_options;
        cpu_options.backend_preference =
          inspirecv::task::BackendPreference::kCpu;
        inspirecv::task::Pipeline cpu_pipeline(cpu_options);
        cpu_pipeline.SetTransform(transform);
        inspirecv::task::PipelineOptions cuda_options = base_options;
        cuda_options.backend_preference =
          inspirecv::task::BackendPreference::kCuda;
        inspirecv::task::Pipeline cuda_pipeline(cuda_options);
        cuda_pipeline.SetTransform(transform);
        inspirecv::task::PipelineOptions auto_options = base_options;
        auto_options.backend_preference =
          inspirecv::task::BackendPreference::kAuto;
        inspirecv::task::Pipeline auto_pipeline(auto_options);
        auto_pipeline.SetTransform(transform);

        inspirecv::task::TensorBuffer destination;
        destination.width = current.destination_width;
        destination.height = current.destination_height;
        destination.channels = 3;
        destination.element_type = inspirecv::task::ElementType::kFloat32;
        destination.order = inspirecv::task::TensorOrder::kChw;
        destination.data = cpu_output.data();
        const TimingDistribution cpu = MeasureReadyHostPipeline(
          &cpu_pipeline, source, destination, 101);
        destination.data = cuda_output.data();
        const TimingDistribution cuda = MeasureReadyHostPipeline(
          &cuda_pipeline, source, destination, 101);
        destination.data = auto_output.data();
        const TimingDistribution automatic = MeasureReadyHostPipeline(
          &auto_pipeline, source, destination, 101);

        REQUIRE(cpu_pipeline.LastExecutionBackend() ==
                inspirecv::task::ExecutionBackend::kCpu);
        REQUIRE(cuda_pipeline.LastExecutionBackend() ==
                inspirecv::task::ExecutionBackend::kCuda);
        REQUIRE(auto_pipeline.LastExecutionBackend() ==
                current.expected_backend);
        RequireExactFloatOutput(cuda_output, cpu_output);
        RequireExactFloatOutput(auto_output, cpu_output);

        const double selected_microseconds =
          current.expected_backend == inspirecv::task::ExecutionBackend::kCuda
            ? cuda.p50_microseconds
            : cpu.p50_microseconds;
        INFO("source=" << current.source_width << 'x'
                       << current.source_height << " output="
                       << current.destination_width << 'x'
                       << current.destination_height);
        REQUIRE(automatic.p50_microseconds <=
                selected_microseconds * 1.10 + 2.0);
        std::printf(
          "[TaskCudaAutoBench] source=%dx%d output=%dx%d cpu_p50_us=%.3f "
          "cuda_p50_us=%.3f auto_p50_us=%.3f selected=%s\n",
          current.source_width, current.source_height,
          current.destination_width, current.destination_height,
          cpu.p50_microseconds, cuda.p50_microseconds,
          automatic.p50_microseconds,
          current.expected_backend == inspirecv::task::ExecutionBackend::kCuda
            ? "cuda"
            : "cpu");
    }
}

#endif  // INSPIRECV_TASK_HAS_CUDA
