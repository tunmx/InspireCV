#include "../../common/common.h"

#include <inspirecv/acceleration.h>
#include <inspirecv/inspirecv.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if defined(INSPIRECV_HAS_CUDA)
#include "inspirecv/core/cuda/image_geometry_dispatch.h"

namespace {

enum class BenchOperation {
    kResizeNearest,
    kResizeBilinear,
    kWarpAffine,
    kRotate90,
    kRotate180,
    kRotate270,
};

struct BenchSpec {
    BenchOperation operation;
    const char* name;
    bool use_float;
    int channels;
    int source_width;
    int source_height;
    int destination_width;
    int destination_height;
};

struct TimingDistribution {
    double p50_microseconds = 0.0;
    double p95_microseconds = 0.0;
};

volatile double g_image_benchmark_sink = 0.0;

int BenchmarkSamples() {
    const char* value = std::getenv("INSPIRECV_IMAGE_CUDA_BENCHMARK_SAMPLES");
    if (!value) return 31;
    const int parsed = std::atoi(value);
    return parsed > 0 ? parsed : 31;
}

double Percentile(std::vector<double> samples, double percentile) {
    REQUIRE_FALSE(samples.empty());
    std::sort(samples.begin(), samples.end());
    const size_t index = static_cast<size_t>(
      std::ceil(percentile * static_cast<double>(samples.size())) - 1.0);
    return samples[std::min(index, samples.size() - 1)];
}

std::string LinuxCpuModel() {
#if defined(__linux__)
    std::ifstream input("/proc/cpuinfo");
    std::string line;
    while (std::getline(input, line)) {
        if (line.compare(0, 10, "model name") != 0) continue;
        const size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        const size_t value = line.find_first_not_of(" \t", colon + 1);
        if (value != std::string::npos) return line.substr(value);
    }
#endif
    return "unknown";
}

std::string CsvEscape(const std::string& value) {
    std::string escaped = "\"";
    for (char character : value) {
        if (character == '\"') escaped += '\"';
        escaped += character;
    }
    escaped += '\"';
    return escaped;
}

template <typename Pixel>
std::vector<Pixel> BenchmarkPixels(int width, int height, int channels);

template <>
std::vector<uint8_t> BenchmarkPixels(int width, int height, int channels) {
    std::vector<uint8_t> pixels(
      static_cast<size_t>(width) * height * channels);
    for (size_t index = 0; index < pixels.size(); ++index) {
        pixels[index] = static_cast<uint8_t>(
          (index * 37 + index / 11 + (index % 17) * 13 + 19) & 255);
    }
    return pixels;
}

template <>
std::vector<float> BenchmarkPixels(int width, int height, int channels) {
    std::vector<float> pixels(
      static_cast<size_t>(width) * height * channels);
    for (size_t index = 0; index < pixels.size(); ++index) {
        pixels[index] = static_cast<float>(
          std::sin(static_cast<double>(index) * 0.013) * 91.75 +
          static_cast<double>(index % 43) * 0.625 - 11.0);
    }
    return pixels;
}

inspirecv::TransformMatrix BenchmarkAffine(int width, int height) {
    const float radians = 0.27f;
    const float cosine = std::cos(radians);
    const float sine = std::sin(radians);
    const float center_x = (width - 1) * 0.5f;
    const float center_y = (height - 1) * 0.5f;
    const float translate_x = center_x - cosine * center_x + sine * center_y;
    const float translate_y = center_y - sine * center_x - cosine * center_y;
    return inspirecv::TransformMatrix(
      cosine, -sine, translate_x, sine, cosine, translate_y);
}

template <typename Pixel>
inspirecv::ImageT<Pixel> RunOperation(
  const inspirecv::ImageT<Pixel>& source, const BenchSpec& spec,
  const inspirecv::TransformMatrix& affine) {
    switch (spec.operation) {
        case BenchOperation::kResizeNearest:
            return source.Resize(spec.destination_width,
                                 spec.destination_height, false);
        case BenchOperation::kResizeBilinear:
            return source.Resize(spec.destination_width,
                                 spec.destination_height, true);
        case BenchOperation::kWarpAffine:
            return source.WarpAffine(affine, spec.destination_width,
                                     spec.destination_height);
        case BenchOperation::kRotate90:
            return source.Rotate90();
        case BenchOperation::kRotate180:
            return source.Rotate180();
        case BenchOperation::kRotate270:
            return source.Rotate270();
        default:
            return source.Clone();
    }
}

template <typename Pixel>
void RequireExact(const inspirecv::ImageT<Pixel>& actual,
                  const inspirecv::ImageT<Pixel>& expected) {
    REQUIRE(actual.Width() == expected.Width());
    REQUIRE(actual.Height() == expected.Height());
    REQUIRE(actual.Channels() == expected.Channels());
    const size_t count = static_cast<size_t>(actual.Width()) *
                         actual.Height() * actual.Channels();
    REQUIRE(std::memcmp(actual.Data(), expected.Data(),
                        count * sizeof(Pixel)) == 0);
}

template <typename Pixel>
TimingDistribution Measure(
  const inspirecv::ImageT<Pixel>& source, const BenchSpec& spec,
  const inspirecv::TransformMatrix& affine,
  inspirecv::AccelerationPreference preference,
  inspirecv::AccelerationBackend expected_backend, int samples) {
    REQUIRE(inspirecv::SetCudaAccelerationEnabled(
      preference != inspirecv::AccelerationPreference::kCpu));
    inspirecv::SetAccelerationPreference(preference);
    for (int warmup = 0; warmup < 10; ++warmup) {
        auto output = RunOperation(source, spec, affine);
        REQUIRE(inspirecv::GetLastImageExecutionBackend() == expected_backend);
        g_image_benchmark_sink += output.Data()[
          static_cast<size_t>(warmup) %
          (static_cast<size_t>(output.Width()) * output.Height() *
           output.Channels())];
    }

    std::vector<double> elapsed;
    elapsed.reserve(samples);
    for (int sample = 0; sample < samples; ++sample) {
        const auto begin = std::chrono::steady_clock::now();
        auto output = RunOperation(source, spec, affine);
        const auto end = std::chrono::steady_clock::now();
        REQUIRE(inspirecv::GetLastImageExecutionBackend() == expected_backend);
        g_image_benchmark_sink += output.Data()[
          static_cast<size_t>(sample) %
          (static_cast<size_t>(output.Width()) * output.Height() *
           output.Channels())];
        elapsed.push_back(
          std::chrono::duration<double, std::micro>(end - begin).count());
    }
    TimingDistribution distribution;
    distribution.p50_microseconds = Percentile(elapsed, 0.50);
    distribution.p95_microseconds = Percentile(elapsed, 0.95);
    return distribution;
}

std::vector<BenchSpec> MakeSpecs() {
    const int source_sizes[][2] = {
      {64, 64},       {96, 96},       {128, 128},
      {192, 192},     {256, 256},     {384, 384},
      {512, 512},     {768, 768},     {1024, 1024},
      {1280, 720},    {1920, 1080},   {2560, 1440},
      {3840, 2160},
    };
    const int resize_sizes[][2] = {
      {48, 48},       {72, 72},       {96, 96},
      {144, 144},     {192, 192},     {288, 288},
      {384, 384},     {576, 576},     {768, 768},
      {960, 540},     {1440, 810},    {1920, 1080},
      {2880, 1620},
    };
    const struct OperationName {
        BenchOperation operation;
        const char* name;
    } u8_operations[] = {
      {BenchOperation::kResizeNearest, "resize_nearest"},
      {BenchOperation::kResizeBilinear, "resize_bilinear"},
      {BenchOperation::kWarpAffine, "warp_affine"},
      {BenchOperation::kRotate90, "rotate90"},
      {BenchOperation::kRotate180, "rotate180"},
      {BenchOperation::kRotate270, "rotate270"},
    };
    const OperationName float_operations[] = {
      {BenchOperation::kResizeNearest, "resize_nearest"},
      {BenchOperation::kResizeBilinear, "resize_bilinear"},
      {BenchOperation::kRotate90, "rotate90"},
      {BenchOperation::kRotate180, "rotate180"},
      {BenchOperation::kRotate270, "rotate270"},
    };

    std::vector<BenchSpec> specs;
    for (const auto& operation : u8_operations) {
        for (size_t index = 0; index < 13; ++index) {
            const bool resize =
              operation.operation == BenchOperation::kResizeNearest ||
              operation.operation == BenchOperation::kResizeBilinear;
            const bool quarter_turn =
              operation.operation == BenchOperation::kRotate90 ||
              operation.operation == BenchOperation::kRotate270;
            specs.push_back(BenchSpec{
              operation.operation, operation.name, false, 3,
              source_sizes[index][0], source_sizes[index][1],
              resize ? resize_sizes[index][0]
                     : (quarter_turn ? source_sizes[index][1]
                                     : source_sizes[index][0]),
              resize ? resize_sizes[index][1]
                     : (quarter_turn ? source_sizes[index][0]
                                     : source_sizes[index][1]),
            });
        }
    }
    for (const auto& operation : float_operations) {
        for (size_t index = 0; index < 13; ++index) {
            const bool resize =
              operation.operation == BenchOperation::kResizeNearest ||
              operation.operation == BenchOperation::kResizeBilinear;
            const bool quarter_turn =
              operation.operation == BenchOperation::kRotate90 ||
              operation.operation == BenchOperation::kRotate270;
            specs.push_back(BenchSpec{
              operation.operation, operation.name, true, 3,
              source_sizes[index][0], source_sizes[index][1],
              resize ? resize_sizes[index][0]
                     : (quarter_turn ? source_sizes[index][1]
                                     : source_sizes[index][0]),
              resize ? resize_sizes[index][1]
                     : (quarter_turn ? source_sizes[index][0]
                                     : source_sizes[index][1]),
            });
        }
    }
    return specs;
}

template <typename Pixel>
void RunSpec(const BenchSpec& spec, int samples, std::ostream* report,
             const std::string& cpu_model,
             const inspirecv::internal::CudaDeviceIdentity& gpu) {
    const auto pixels = BenchmarkPixels<Pixel>(
      spec.source_width, spec.source_height, spec.channels);
    const auto source = inspirecv::ImageT<Pixel>::Create(
      spec.source_width, spec.source_height, spec.channels, pixels.data());
    const auto affine = BenchmarkAffine(spec.source_width, spec.source_height);

    REQUIRE(inspirecv::SetCudaAccelerationEnabled(false));
    auto cpu_output = RunOperation(source, spec, affine);
    REQUIRE(inspirecv::SetCudaAccelerationEnabled(true));
    inspirecv::SetAccelerationPreference(
      inspirecv::AccelerationPreference::kCuda);
    auto cuda_output = RunOperation(source, spec, affine);
    REQUIRE(inspirecv::GetLastImageExecutionBackend() ==
            inspirecv::AccelerationBackend::kCuda);
    RequireExact(cuda_output, cpu_output);

    const TimingDistribution cpu =
      Measure(source, spec, affine, inspirecv::AccelerationPreference::kCpu,
              inspirecv::AccelerationBackend::kCpu, samples);
    const TimingDistribution cuda =
      Measure(source, spec, affine, inspirecv::AccelerationPreference::kCuda,
              inspirecv::AccelerationBackend::kCuda, samples);
    const double p50_speedup = cpu.p50_microseconds / cuda.p50_microseconds;
    const bool recommended = cuda.p50_microseconds <=
                               cpu.p50_microseconds * 0.85 &&
                             cuda.p95_microseconds <= cpu.p95_microseconds;

    std::cout << std::left << std::setw(16) << spec.name << ' '
              << (spec.use_float ? "f32" : "u8 ") << " c" << spec.channels
              << ' ' << spec.source_width << 'x' << spec.source_height << " -> "
              << spec.destination_width << 'x' << spec.destination_height
              << " cpu=" << std::fixed << std::setprecision(3)
              << cpu.p50_microseconds << "/" << cpu.p95_microseconds
              << " us cuda=" << cuda.p50_microseconds << '/'
              << cuda.p95_microseconds << " us speedup=" << p50_speedup
              << " recommend=" << (recommended ? "cuda" : "cpu") << '\n';

    if (report && *report) {
        *report << CsvEscape(gpu.name) << ',' << gpu.compute_capability_major
                << '.' << gpu.compute_capability_minor << ','
                << gpu.runtime_version << ',' << gpu.driver_version << ','
                << CsvEscape(cpu_model) << ',' << spec.name << ','
                << (spec.use_float ? "f32" : "u8") << ',' << spec.channels
                << ',' << spec.source_width << ',' << spec.source_height << ','
                << spec.destination_width << ',' << spec.destination_height
                << ',' << samples << ',' << std::fixed << std::setprecision(3)
                << cpu.p50_microseconds << ',' << cpu.p95_microseconds << ','
                << cuda.p50_microseconds << ',' << cuda.p95_microseconds << ','
                << p50_speedup << ',' << (recommended ? "cuda" : "cpu")
                << ",bit_exact\n";
    }
}

template <typename Pixel>
void RequireAutoPerformance(const BenchSpec& spec,
                            inspirecv::AccelerationBackend expected_backend,
                            int samples) {
    const auto pixels = BenchmarkPixels<Pixel>(
      spec.source_width, spec.source_height, spec.channels);
    const auto source = inspirecv::ImageT<Pixel>::Create(
      spec.source_width, spec.source_height, spec.channels, pixels.data());
    const auto affine = BenchmarkAffine(spec.source_width, spec.source_height);
    const inspirecv::AccelerationPreference selected_preference =
      expected_backend == inspirecv::AccelerationBackend::kCuda
        ? inspirecv::AccelerationPreference::kCuda
        : inspirecv::AccelerationPreference::kCpu;
    const TimingDistribution selected = Measure(
      source, spec, affine, selected_preference, expected_backend, samples);
    const TimingDistribution automatic = Measure(
      source, spec, affine, inspirecv::AccelerationPreference::kAuto,
      expected_backend, samples);
    INFO("operation=" << spec.name);
    INFO("type=" << (spec.use_float ? "f32" : "u8"));
    INFO("source=" << spec.source_width << 'x' << spec.source_height);
    INFO("selected_p50_us=" << selected.p50_microseconds);
    INFO("auto_p50_us=" << automatic.p50_microseconds);
    REQUIRE(automatic.p50_microseconds <=
            selected.p50_microseconds * 1.10 + 2.0);
}

}  // namespace

TEST_CASE("image_cuda_full_geometry_benchmark",
          "[benchmark][image][cuda][threshold]") {
    if (!inspirecv::IsCudaAccelerationAvailable()) {
        SUCCEED("CUDA device unavailable");
        return;
    }
    const auto previous_preference = inspirecv::GetAccelerationPreference();
    const bool previous_enabled = inspirecv::IsCudaAccelerationEnabled();

    inspirecv::internal::CudaDeviceIdentity gpu;
    REQUIRE(inspirecv::internal::QueryImageCudaDeviceIdentity(&gpu));
    const std::string cpu_model = LinuxCpuModel();
    const int samples = BenchmarkSamples();
    std::ofstream report_file;
    std::ostream* report = nullptr;
    if (const char* report_path =
          std::getenv("INSPIRECV_IMAGE_CUDA_BENCHMARK_REPORT")) {
        report_file.open(report_path, std::ios::trunc);
        REQUIRE(report_file.good());
        report = &report_file;
        report_file
          << "gpu,compute_capability,cuda_runtime,cuda_driver,cpu,operation,"
             "element_type,channels,source_width,source_height,destination_width,"
             "destination_height,samples,cpu_p50_us,cpu_p95_us,cuda_p50_us,"
             "cuda_p95_us,p50_speedup,recommendation,accuracy\n";
    }

    std::cout << "Image CUDA host round-trip benchmark: samples=" << samples
              << ", gpu=" << gpu.name << ", cpu=" << cpu_model << '\n';
    const char* operation_filter =
      std::getenv("INSPIRECV_IMAGE_CUDA_BENCHMARK_OPERATION");
    const char* type_filter =
      std::getenv("INSPIRECV_IMAGE_CUDA_BENCHMARK_TYPE");
    for (const BenchSpec& spec : MakeSpecs()) {
        if (operation_filter &&
            std::string(operation_filter) != spec.name) {
            continue;
        }
        if (type_filter &&
            std::string(type_filter) != (spec.use_float ? "f32" : "u8")) {
            continue;
        }
        if (spec.use_float) {
            RunSpec<float>(spec, samples, report, cpu_model, gpu);
        } else {
            RunSpec<uint8_t>(spec, samples, report, cpu_model, gpu);
        }
    }

    inspirecv::SetAccelerationPreference(previous_preference);
    REQUIRE(inspirecv::SetCudaAccelerationEnabled(previous_enabled));
}

TEST_CASE("image_cuda_auto_dispatch_performance",
          "[benchmark][image][cuda][auto]") {
    if (!inspirecv::IsCudaAccelerationAvailable()) {
        SUCCEED("CUDA device unavailable");
        return;
    }

    REQUIRE(inspirecv::SetCudaAccelerationEnabled(true));
    inspirecv::SetAccelerationPreference(
      inspirecv::AccelerationPreference::kAuto);

    const int samples = 51;
    RequireAutoPerformance<uint8_t>(
      BenchSpec{BenchOperation::kResizeBilinear, "resize_bilinear", false,
                3, 128, 128, 96, 96},
      inspirecv::AccelerationBackend::kCuda, samples);
    RequireAutoPerformance<uint8_t>(
      BenchSpec{BenchOperation::kResizeNearest, "resize_nearest", false,
                3, 1920, 1080, 1440, 810},
      inspirecv::AccelerationBackend::kCpu, samples);
    RequireAutoPerformance<float>(
      BenchSpec{BenchOperation::kResizeBilinear, "resize_bilinear", true,
                3, 512, 512, 384, 384},
      inspirecv::AccelerationBackend::kCuda, samples);
    RequireAutoPerformance<float>(
      BenchSpec{BenchOperation::kResizeBilinear, "resize_bilinear", true,
                3, 3840, 2160, 2880, 1620},
      inspirecv::AccelerationBackend::kCpu, samples);
    RequireAutoPerformance<float>(
      BenchSpec{BenchOperation::kRotate90, "rotate90", true,
                3, 512, 512, 512, 512},
      inspirecv::AccelerationBackend::kCuda, samples);

    REQUIRE(inspirecv::SetCudaAccelerationEnabled(false));
    inspirecv::SetAccelerationPreference(
      inspirecv::AccelerationPreference::kAuto);
}

#endif  // INSPIRECV_HAS_CUDA
