#include "../../common/common.h"

#include <inspirecv/acceleration.h>
#include <inspirecv/cuda/image.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

class RestoreCudaState {
public:
    RestoreCudaState()
        : enabled_(inspirecv::IsCudaAccelerationEnabled()),
          preference_(inspirecv::GetAccelerationPreference()) {}
    ~RestoreCudaState() {
        inspirecv::SetAccelerationPreference(preference_);
        inspirecv::SetCudaAccelerationEnabled(enabled_);
    }

private:
    bool enabled_;
    inspirecv::AccelerationPreference preference_;
};

template <typename Pixel>
std::vector<Pixel> MakeDevicePixels(int width, int height, int channels);

template <>
std::vector<uint8_t> MakeDevicePixels(int width, int height, int channels) {
    std::vector<uint8_t> pixels(
      static_cast<size_t>(width) * height * channels);
    for (size_t index = 0; index < pixels.size(); ++index) {
        pixels[index] = static_cast<uint8_t>(
          (index * 41 + index / 9 + (index % 13) * 17) & 255);
    }
    return pixels;
}

template <>
std::vector<float> MakeDevicePixels(int width, int height, int channels) {
    std::vector<float> pixels(
      static_cast<size_t>(width) * height * channels);
    for (size_t index = 0; index < pixels.size(); ++index) {
        pixels[index] = static_cast<float>(
          std::sin(static_cast<double>(index) * 0.043) * 47.25 +
          static_cast<double>(index % 19) * 0.125);
    }
    return pixels;
}

template <typename Pixel>
void RequireDeviceImageExact(const inspirecv::ImageT<Pixel>& actual,
                             const inspirecv::ImageT<Pixel>& expected) {
    REQUIRE(actual.Width() == expected.Width());
    REQUIRE(actual.Height() == expected.Height());
    REQUIRE(actual.Channels() == expected.Channels());
    const size_t bytes = static_cast<size_t>(actual.Width()) * actual.Height() *
                         actual.Channels() * sizeof(Pixel);
    REQUIRE(std::memcmp(actual.Data(), expected.Data(), bytes) == 0);
}

#if defined(INSPIRECV_HAS_CUDA)
template <typename Operation>
double MeasureDeviceMedian(Operation operation, int samples) {
    for (int warmup = 0; warmup < 5; ++warmup) operation();
    std::vector<double> timings;
    timings.reserve(samples);
    for (int sample = 0; sample < samples; ++sample) {
        const auto begin = std::chrono::steady_clock::now();
        operation();
        const auto end = std::chrono::steady_clock::now();
        timings.push_back(
          std::chrono::duration<double, std::micro>(end - begin).count());
    }
    std::sort(timings.begin(), timings.end());
    return timings[timings.size() / 2];
}
#endif

}  // namespace

TEST_CASE("cuda_device_image_cpu_build_contract",
          "[image][cuda][device][contract]") {
    inspirecv::cuda::DeviceImage image;
    REQUIRE(image.Empty());
    REQUIRE(image.Width() == 0);
    REQUIRE(image.Height() == 0);
    REQUIRE(image.Channels() == 0);
    REQUIRE(image.View().data == 0);

    REQUIRE(std::strcmp(
              inspirecv::cuda::StatusMessage(
                inspirecv::cuda::Status::kAccelerationDisabled),
              "acceleration disabled") == 0);

#if !defined(INSPIRECV_HAS_CUDA)
    const auto source = inspirecv::Image::Create(2, 2, 3);
    REQUIRE(inspirecv::cuda::DeviceImage::Upload(source, &image) ==
            inspirecv::cuda::Status::kAccelerationDisabled);
#endif
}

#if defined(INSPIRECV_HAS_CUDA)

TEST_CASE("cuda_device_image_chains_geometry_without_host_round_trips",
          "[image][cuda][device][accuracy]") {
    RestoreCudaState restore;
    if (!inspirecv::IsCudaAccelerationAvailable()) {
        SUCCEED("CUDA device unavailable");
        return;
    }
    REQUIRE(inspirecv::SetCudaAccelerationEnabled(true));

    const auto pixels = MakeDevicePixels<uint8_t>(73, 51, 3);
    const auto source = inspirecv::Image::Create(73, 51, 3, pixels.data());
    const inspirecv::TransformMatrix affine(
      0.93f, 0.17f, -3.25f, -0.11f, 1.04f, 2.5f);
    const auto expected = source.Resize(61, 47, true)
                            .WarpAffine(affine, 59, 43)
                            .Rotate90();

    inspirecv::cuda::DeviceImage uploaded;
    inspirecv::cuda::DeviceImage resized;
    inspirecv::cuda::DeviceImage warped;
    inspirecv::cuda::DeviceImage rotated;
    REQUIRE(inspirecv::cuda::DeviceImage::Upload(source, &uploaded) ==
            inspirecv::cuda::Status::kOk);
    REQUIRE(uploaded.View().data != 0);
    REQUIRE(uploaded.View().row_stride_bytes == 73u * 3u);
    REQUIRE(uploaded.Resize(61, 47, true, &resized) ==
            inspirecv::cuda::Status::kOk);
    REQUIRE(resized.WarpAffine(affine, 59, 43, &warped) ==
            inspirecv::cuda::Status::kOk);
    REQUIRE(warped.Rotate90(&rotated) == inspirecv::cuda::Status::kOk);

    inspirecv::Image actual;
    REQUIRE(rotated.Download(&actual) == inspirecv::cuda::Status::kOk);
    RequireDeviceImageExact(actual, expected);
}

TEST_CASE("cuda_device_image_float_chain_matches_cpu",
          "[image][cuda][device][accuracy][float]") {
    RestoreCudaState restore;
    if (!inspirecv::IsCudaAccelerationAvailable()) {
        SUCCEED("CUDA device unavailable");
        return;
    }
    REQUIRE(inspirecv::SetCudaAccelerationEnabled(true));

    const auto pixels = MakeDevicePixels<float>(48, 36, 4);
    const auto source = inspirecv::ImageT<float>::Create(
      48, 36, 4, pixels.data());
    const auto expected = source.Resize(39, 31, false).Rotate180();

    inspirecv::cuda::DeviceImage uploaded;
    inspirecv::cuda::DeviceImage resized;
    inspirecv::cuda::DeviceImage rotated;
    REQUIRE(inspirecv::cuda::DeviceImage::Upload(source, &uploaded) ==
            inspirecv::cuda::Status::kOk);
    REQUIRE(uploaded.Type() == inspirecv::cuda::ElementType::kFloat32);
    REQUIRE(uploaded.Resize(39, 31, false, &resized) ==
            inspirecv::cuda::Status::kOk);
    REQUIRE(resized.Rotate180(&rotated) == inspirecv::cuda::Status::kOk);

    inspirecv::ImageT<float> actual;
    REQUIRE(rotated.Download(&actual) == inspirecv::cuda::Status::kOk);
    RequireDeviceImageExact(actual, expected);
}

TEST_CASE("cuda_device_image_rejects_incompatible_geometry_transactionally",
          "[image][cuda][device][fallback]") {
    RestoreCudaState restore;
    if (!inspirecv::IsCudaAccelerationAvailable()) {
        SUCCEED("CUDA device unavailable");
        return;
    }
    REQUIRE(inspirecv::SetCudaAccelerationEnabled(true));

    const auto pixels = MakeDevicePixels<float>(31, 27, 3);
    const auto source = inspirecv::ImageT<float>::Create(
      31, 27, 3, pixels.data());
    inspirecv::cuda::DeviceImage uploaded;
    inspirecv::cuda::DeviceImage destination;
    REQUIRE(inspirecv::cuda::DeviceImage::Upload(source, &uploaded) ==
            inspirecv::cuda::Status::kOk);
    const inspirecv::TransformMatrix general(
      0.91f, 0.15f, -2.0f, -0.12f, 1.03f, 1.5f);
    REQUIRE(uploaded.WarpAffine(general, 29, 25, &destination) ==
            inspirecv::cuda::Status::kUnsupportedOperation);
    REQUIRE(destination.Empty());
}

TEST_CASE("cuda_device_image_chain_benchmark",
          "[benchmark][image][cuda][device]") {
    RestoreCudaState restore;
    if (!inspirecv::IsCudaAccelerationAvailable()) {
        SUCCEED("CUDA device unavailable");
        return;
    }
    int samples = 21;
    if (const char* value =
          std::getenv("INSPIRECV_DEVICE_IMAGE_BENCHMARK_SAMPLES")) {
        samples = std::max(3, std::atoi(value));
    }
    std::ofstream report;
    if (const char* path =
          std::getenv("INSPIRECV_DEVICE_IMAGE_BENCHMARK_REPORT")) {
        report.open(path, std::ios::trunc);
        REQUIRE(report.good());
        report << "source_width,source_height,cpu_us,host_cuda_us,"
                  "resident_roundtrip_us,resident_device_us\n";
    }

    const int sizes[][2] = {{640, 480}, {1280, 720}, {1920, 1080}};
    for (const auto& size : sizes) {
        const int width = size[0];
        const int height = size[1];
        const int resized_width = width * 3 / 4;
        const int resized_height = height * 3 / 4;
        const auto pixels = MakeDevicePixels<uint8_t>(width, height, 3);
        const auto source = inspirecv::Image::Create(
          width, height, 3, pixels.data());
        const inspirecv::TransformMatrix affine(
          0.97f, 0.08f, -9.0f, -0.06f, 1.02f, 7.0f);

        REQUIRE(inspirecv::SetCudaAccelerationEnabled(false));
        auto cpu_operation = [&]() {
            auto output = source.Resize(resized_width, resized_height, true)
                            .WarpAffine(affine, resized_width, resized_height)
                            .Rotate90();
            REQUIRE_FALSE(output.Empty());
        };
        const double cpu_us = MeasureDeviceMedian(cpu_operation, samples);
        const auto expected = source.Resize(resized_width, resized_height, true)
                                .WarpAffine(affine, resized_width,
                                            resized_height)
                                .Rotate90();

        REQUIRE(inspirecv::SetCudaAccelerationEnabled(true));
        inspirecv::SetAccelerationPreference(
          inspirecv::AccelerationPreference::kCuda);
        auto host_cuda_operation = [&]() {
            auto output = source.Resize(resized_width, resized_height, true)
                            .WarpAffine(affine, resized_width, resized_height)
                            .Rotate90();
            REQUIRE_FALSE(output.Empty());
        };
        const double host_cuda_us =
          MeasureDeviceMedian(host_cuda_operation, samples);
        const auto host_cuda_actual =
          source.Resize(resized_width, resized_height, true)
            .WarpAffine(affine, resized_width, resized_height)
            .Rotate90();
        RequireDeviceImageExact(host_cuda_actual, expected);

        inspirecv::cuda::DeviceImage uploaded;
        inspirecv::cuda::DeviceImage resized;
        inspirecv::cuda::DeviceImage warped;
        inspirecv::cuda::DeviceImage rotated;
        inspirecv::Image downloaded;
        auto resident_roundtrip_operation = [&]() {
            REQUIRE(inspirecv::cuda::DeviceImage::Upload(source, &uploaded) ==
                    inspirecv::cuda::Status::kOk);
            REQUIRE(uploaded.Resize(resized_width, resized_height, true,
                                    &resized) ==
                    inspirecv::cuda::Status::kOk);
            REQUIRE(resized.WarpAffine(affine, resized_width, resized_height,
                                      &warped) ==
                    inspirecv::cuda::Status::kOk);
            REQUIRE(warped.Rotate90(&rotated) ==
                    inspirecv::cuda::Status::kOk);
            REQUIRE(rotated.Download(&downloaded) ==
                    inspirecv::cuda::Status::kOk);
        };
        const double resident_roundtrip_us =
          MeasureDeviceMedian(resident_roundtrip_operation, samples);
        RequireDeviceImageExact(downloaded, expected);

        REQUIRE(inspirecv::cuda::DeviceImage::Upload(source, &uploaded) ==
                inspirecv::cuda::Status::kOk);
        auto resident_device_operation = [&]() {
            REQUIRE(uploaded.Resize(resized_width, resized_height, true,
                                    &resized) ==
                    inspirecv::cuda::Status::kOk);
            REQUIRE(resized.WarpAffine(affine, resized_width, resized_height,
                                      &warped) ==
                    inspirecv::cuda::Status::kOk);
            REQUIRE(warped.Rotate90(&rotated) ==
                    inspirecv::cuda::Status::kOk);
            REQUIRE(inspirecv::cuda::Synchronize() ==
                    inspirecv::cuda::Status::kOk);
        };
        const double resident_device_us =
          MeasureDeviceMedian(resident_device_operation, samples);
        REQUIRE(rotated.Download(&downloaded) ==
                inspirecv::cuda::Status::kOk);
        RequireDeviceImageExact(downloaded, expected);

        std::cout << "[CudaDeviceImageBenchmark] " << width << 'x' << height
                  << " cpu_us=" << cpu_us
                  << " host_cuda_us=" << host_cuda_us
                  << " resident_roundtrip_us=" << resident_roundtrip_us
                  << " resident_device_us=" << resident_device_us << '\n';
        if (report.good()) {
            report << width << ',' << height << ',' << cpu_us << ','
                   << host_cuda_us << ',' << resident_roundtrip_us << ','
                   << resident_device_us << '\n';
        }
    }
}

#endif
