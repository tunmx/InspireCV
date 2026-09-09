#include "../../common/common.h"

#include <inspirecv/acceleration.h>
#include <inspirecv/inspirecv.h>
#include <inspirecv/task/acceleration.h>

#include "inspirecv/core/runtime/image_acceleration.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <vector>

namespace {

class RestoreAccelerationState {
public:
    RestoreAccelerationState()
        : preference_(inspirecv::GetAccelerationPreference()),
          enabled_(inspirecv::IsCudaAccelerationEnabled()) {}

    ~RestoreAccelerationState() {
        inspirecv::SetAccelerationPreference(preference_);
        inspirecv::SetCudaAccelerationEnabled(enabled_);
    }

private:
    inspirecv::AccelerationPreference preference_;
    bool enabled_;
};

template <typename Pixel>
std::vector<Pixel> MakePixels(int width, int height, int channels);

template <>
std::vector<uint8_t> MakePixels(int width, int height, int channels) {
    std::vector<uint8_t> pixels(
      static_cast<size_t>(width) * height * channels);
    for (size_t index = 0; index < pixels.size(); ++index) {
        pixels[index] = static_cast<uint8_t>(
          (index * 53 + index / 7 + (index % 5) * 29 + 17) & 255);
    }
    return pixels;
}

template <>
std::vector<float> MakePixels(int width, int height, int channels) {
    std::vector<float> pixels(
      static_cast<size_t>(width) * height * channels);
    for (size_t index = 0; index < pixels.size(); ++index) {
        pixels[index] = static_cast<float>(
          std::sin(static_cast<double>(index) * 0.071) * 103.25 +
          static_cast<double>(index % 31) * 0.375 - 17.0);
    }
    return pixels;
}

template <typename Pixel>
void RequireExactImage(const inspirecv::ImageT<Pixel>& actual,
                       const inspirecv::ImageT<Pixel>& expected) {
    REQUIRE(actual.Width() == expected.Width());
    REQUIRE(actual.Height() == expected.Height());
    REQUIRE(actual.Channels() == expected.Channels());
    const size_t count = static_cast<size_t>(actual.Width()) *
                         actual.Height() * actual.Channels();
    size_t mismatch_count = 0;
    size_t first_mismatch = 0;
    double maximum_error = 0.0;
    for (size_t index = 0; index < count; ++index) {
        if (std::memcmp(actual.Data() + index, expected.Data() + index,
                        sizeof(Pixel)) == 0) {
            continue;
        }
        if (mismatch_count == 0) first_mismatch = index;
        ++mismatch_count;
        maximum_error = std::max(
          maximum_error,
          std::fabs(static_cast<double>(actual.Data()[index]) -
                    static_cast<double>(expected.Data()[index])));
    }
    INFO("mismatches=" << mismatch_count << "/" << count);
    INFO("first_index=" << first_mismatch);
    INFO("maximum_error=" << maximum_error);
    REQUIRE(mismatch_count == 0);
}

template <typename Pixel, typename Operation>
void RequireCudaMatchesCpu(const inspirecv::ImageT<Pixel>& source,
                           Operation operation) {
    REQUIRE(inspirecv::SetCudaAccelerationEnabled(false));
    auto expected = operation(source);
    REQUIRE(inspirecv::GetLastImageExecutionBackend() ==
            inspirecv::AccelerationBackend::kCpu);

    REQUIRE(inspirecv::SetCudaAccelerationEnabled(true));
    inspirecv::SetAccelerationPreference(
      inspirecv::AccelerationPreference::kCuda);
    auto actual = operation(source);
    REQUIRE(inspirecv::GetLastImageExecutionBackend() ==
            inspirecv::AccelerationBackend::kCuda);
    RequireExactImage(actual, expected);
}

template <typename Pixel, typename Operation>
void RequireForcedCudaFallsBack(const inspirecv::ImageT<Pixel>& source,
                                Operation operation) {
    REQUIRE(inspirecv::SetCudaAccelerationEnabled(false));
    auto cpu_output = operation(source);
    REQUIRE_FALSE(cpu_output.Empty());
    REQUIRE(inspirecv::SetCudaAccelerationEnabled(true));
    inspirecv::SetAccelerationPreference(
      inspirecv::AccelerationPreference::kCuda);
    auto actual = operation(source);
    REQUIRE(inspirecv::GetLastImageExecutionBackend() ==
            inspirecv::AccelerationBackend::kCpu);
    REQUIRE_FALSE(actual.Empty());
}

void SaveVisualComparison(const inspirecv::Image& cpu,
                          const inspirecv::Image& cuda,
                          const std::string& directory,
                          const std::string& name) {
    RequireExactImage(cuda, cpu);
    REQUIRE(cpu.Write(directory + "/" + name + "_cpu.png"));
    REQUIRE(cuda.Write(directory + "/" + name + "_cuda.png"));
    auto difference = cpu.AbsDiff(cuda);
    REQUIRE(difference.Write(directory + "/" + name + "_diff.png"));

    const int width = cpu.Width();
    const int height = cpu.Height();
    const int channels = cpu.Channels();
    std::vector<uint8_t> comparison(
      static_cast<size_t>(width) * 2 * height * channels);
    const size_t row_bytes = static_cast<size_t>(width) * channels;
    for (int y = 0; y < height; ++y) {
        uint8_t* destination = comparison.data() +
                               static_cast<size_t>(y) * row_bytes * 2;
        std::memcpy(destination, cpu.Data() + static_cast<size_t>(y) * row_bytes,
                    row_bytes);
        std::memcpy(destination + row_bytes,
                    cuda.Data() + static_cast<size_t>(y) * row_bytes,
                    row_bytes);
    }
    auto side_by_side = inspirecv::Image::Create(
      width * 2, height, channels, comparison.data());
    REQUIRE(side_by_side.Write(directory + "/" + name + "_side_by_side.png"));
}

}  // namespace

TEST_CASE("image_cuda_global_control_is_shared_with_task",
          "[image][cuda][contract]") {
    RestoreAccelerationState restore;
    REQUIRE(inspirecv::SetCudaAccelerationEnabled(false));
    REQUIRE_FALSE(inspirecv::IsCudaAccelerationEnabled());
    REQUIRE_FALSE(inspirecv::task::IsCudaEnabled());

    inspirecv::SetAccelerationPreference(
      inspirecv::AccelerationPreference::kCpu);
    REQUIRE(inspirecv::task::GetDefaultBackendPreference() ==
            inspirecv::task::BackendPreference::kCpu);
    REQUIRE(inspirecv::task::SetDefaultBackendPreference(
      inspirecv::task::BackendPreference::kAuto));
    REQUIRE(inspirecv::GetAccelerationPreference() ==
            inspirecv::AccelerationPreference::kAuto);

    if (!inspirecv::IsCudaAccelerationAvailable()) {
        REQUIRE_FALSE(inspirecv::SetCudaAccelerationEnabled(true));
        REQUIRE_FALSE(inspirecv::IsCudaAccelerationEnabled());
    }
}

TEST_CASE("image_cuda_auto_policy_uses_default_3060_baseline",
          "[image][cuda][auto][contract]") {
    inspirecv::internal::ImageGeometryRequest request;
    request.operation =
      inspirecv::internal::ImageGeometryOperation::kResizeBilinear;
    request.element_type = inspirecv::internal::ImageElementType::kUInt8;
    request.source_width = 128;
    request.source_height = 128;
    request.destination_width = 96;
    request.destination_height = 96;
    request.channels = 3;
    REQUIRE(inspirecv::internal::ImageCudaAutoRecommends(request));

    request.source_width = 64;
    request.source_height = 64;
    request.destination_width = 48;
    request.destination_height = 48;
    REQUIRE_FALSE(inspirecv::internal::ImageCudaAutoRecommends(request));

    request.operation = inspirecv::internal::ImageGeometryOperation::kRotate90;
    request.source_width = 96;
    request.source_height = 96;
    request.destination_width = 96;
    request.destination_height = 96;
    REQUIRE(inspirecv::internal::ImageCudaAutoRecommends(request));
}

#if defined(INSPIRECV_HAS_CUDA)

TEST_CASE("image_cuda_resize_matches_cpu_bit_exact",
          "[image][cuda][accuracy][resize]") {
    RestoreAccelerationState restore;
    if (!inspirecv::IsCudaAccelerationAvailable()) {
        SUCCEED("CUDA device unavailable");
        return;
    }

    const int channel_counts[] = {1, 3, 4};
    for (int channels : channel_counts) {
        DYNAMIC_SECTION("u8 channels=" << channels) {
            const auto pixels = MakePixels<uint8_t>(37, 23, channels);
            const auto source = inspirecv::Image::Create(
              37, 23, channels, pixels.data());
            RequireCudaMatchesCpu(source, [](const inspirecv::Image& image) {
                return image.Resize(19, 31, true);
            });
            RequireCudaMatchesCpu(source, [](const inspirecv::Image& image) {
                return image.Resize(61, 17, false);
            });
        }
        DYNAMIC_SECTION("f32 channels=" << channels) {
            const auto pixels = MakePixels<float>(37, 23, channels);
            const auto source = inspirecv::ImageT<float>::Create(
              37, 23, channels, pixels.data());
            RequireCudaMatchesCpu(
              source, [](const inspirecv::ImageT<float>& image) {
                  return image.Resize(19, 31, true);
              });
            RequireCudaMatchesCpu(
              source, [](const inspirecv::ImageT<float>& image) {
                  return image.Resize(61, 17, false);
              });
        }
    }
}

TEST_CASE("image_cuda_rotations_match_cpu_bit_exact",
          "[image][cuda][accuracy][rotate]") {
    RestoreAccelerationState restore;
    if (!inspirecv::IsCudaAccelerationAvailable()) {
        SUCCEED("CUDA device unavailable");
        return;
    }

    const int channel_counts[] = {1, 3, 4};
    for (int channels : channel_counts) {
        const auto u8_pixels = MakePixels<uint8_t>(37, 23, channels);
        const auto u8 = inspirecv::Image::Create(
          37, 23, channels, u8_pixels.data());
        {
            INFO("operation=u8_rotate90 channels=" << channels);
            RequireCudaMatchesCpu(u8, [](const inspirecv::Image& image) {
                return image.Rotate90();
            });
        }
        {
            INFO("operation=u8_rotate180 channels=" << channels);
            RequireCudaMatchesCpu(u8, [](const inspirecv::Image& image) {
                return image.Rotate180();
            });
        }
        {
            INFO("operation=u8_rotate270 channels=" << channels);
            RequireCudaMatchesCpu(u8, [](const inspirecv::Image& image) {
                return image.Rotate270();
            });
        }

        const auto f32_pixels = MakePixels<float>(37, 23, channels);
        const auto f32 = inspirecv::ImageT<float>::Create(
          37, 23, channels, f32_pixels.data());
        {
            INFO("operation=f32_rotate90 channels=" << channels);
            RequireCudaMatchesCpu(
              f32, [](const inspirecv::ImageT<float>& image) {
                  return image.Rotate90();
              });
        }
        {
            INFO("operation=f32_rotate180 channels=" << channels);
            RequireCudaMatchesCpu(
              f32, [](const inspirecv::ImageT<float>& image) {
                  return image.Rotate180();
              });
        }
        {
            INFO("operation=f32_rotate270 channels=" << channels);
            if (channels == 1) {
                RequireForcedCudaFallsBack(
                  f32, [](const inspirecv::ImageT<float>& image) {
                      return image.Rotate270();
                  });
            } else {
                RequireCudaMatchesCpu(
                  f32, [](const inspirecv::ImageT<float>& image) {
                      return image.Rotate270();
                  });
            }
        }
    }
}

TEST_CASE("image_cuda_warp_affine_matches_cpu_contract",
          "[image][cuda][accuracy][warp_affine]") {
    RestoreAccelerationState restore;
    if (!inspirecv::IsCudaAccelerationAvailable()) {
        SUCCEED("CUDA device unavailable");
        return;
    }

    const auto u8_pixels = MakePixels<uint8_t>(53, 41, 3);
    const auto u8 = inspirecv::Image::Create(53, 41, 3, u8_pixels.data());
    const inspirecv::TransformMatrix axis_aligned(
      0.83f, 0.0f, -0.35f, 0.0f, 1.17f, 0.6f);
    {
        INFO("operation=u8_axis_aligned");
        RequireForcedCudaFallsBack(u8, [&](const inspirecv::Image& image) {
            return image.WarpAffine(axis_aligned, 47, 39);
        });
    }

    const inspirecv::TransformMatrix general(
      0.91f, 0.19f, -4.25f, -0.13f, 1.04f, 3.5f);
    {
        INFO("operation=u8_general");
        RequireCudaMatchesCpu(u8, [&](const inspirecv::Image& image) {
            return image.WarpAffine(general, 47, 39);
        });
    }

    const auto f32_pixels = MakePixels<float>(53, 41, 4);
    const auto f32 =
      inspirecv::ImageT<float>::Create(53, 41, 4, f32_pixels.data());
    {
        INFO("operation=f32_axis_aligned");
        RequireCudaMatchesCpu(
          f32, [&](const inspirecv::ImageT<float>& image) {
              return image.WarpAffine(axis_aligned, 47, 39);
      });
    }

    const auto u8_four_pixels = MakePixels<uint8_t>(53, 41, 4);
    const auto u8_four = inspirecv::Image::Create(
      53, 41, 4, u8_four_pixels.data());
    {
        INFO("operation=u8_axis_aligned_channels4");
        RequireCudaMatchesCpu(u8_four, [&](const inspirecv::Image& image) {
            return image.WarpAffine(axis_aligned, 47, 39);
        });
    }

    REQUIRE(inspirecv::SetCudaAccelerationEnabled(true));
    inspirecv::SetAccelerationPreference(
      inspirecv::AccelerationPreference::kCuda);
    auto fallback = f32.WarpAffine(general, 47, 39);
    REQUIRE_FALSE(fallback.Empty());
    REQUIRE(inspirecv::GetLastImageExecutionBackend() ==
            inspirecv::AccelerationBackend::kCpu);
}

TEST_CASE("image_cuda_auto_dispatch_follows_default_3060_baseline",
          "[image][cuda][auto][accuracy]") {
    RestoreAccelerationState restore;
    if (!inspirecv::IsCudaAccelerationAvailable()) {
        SUCCEED("CUDA device unavailable");
        return;
    }
    REQUIRE(inspirecv::SetCudaAccelerationEnabled(true));
    inspirecv::SetAccelerationPreference(
      inspirecv::AccelerationPreference::kAuto);

    const auto require_backend = [](inspirecv::AccelerationBackend expected) {
        REQUIRE(inspirecv::GetLastImageExecutionBackend() == expected);
    };

    const auto small_u8_pixels = MakePixels<uint8_t>(64, 64, 3);
    const auto small_u8 = inspirecv::Image::Create(
      64, 64, 3, small_u8_pixels.data());
    auto small_bilinear = small_u8.Resize(48, 48, true);
    require_backend(inspirecv::AccelerationBackend::kCpu);
    auto small_rotation = small_u8.Rotate90();
    require_backend(inspirecv::AccelerationBackend::kCpu);

    const auto rotation_pixels = MakePixels<uint8_t>(96, 96, 3);
    const auto rotation_source = inspirecv::Image::Create(
      96, 96, 3, rotation_pixels.data());
    auto profiled_rotation = rotation_source.Rotate90();
    require_backend(inspirecv::AccelerationBackend::kCuda);

    const auto near_pixels = MakePixels<uint8_t>(1280, 720, 3);
    const auto near_source = inspirecv::Image::Create(
      1280, 720, 3, near_pixels.data());
    auto near_output = near_source.Resize(960, 540, false);
    require_backend(inspirecv::AccelerationBackend::kCpu);

    const auto far_pixels = MakePixels<uint8_t>(1920, 1080, 3);
    const auto far_source = inspirecv::Image::Create(
      1920, 1080, 3, far_pixels.data());
    auto far_output = far_source.Resize(1440, 810, false);
    require_backend(inspirecv::AccelerationBackend::kCpu);

    const auto float_small_pixels = MakePixels<float>(256, 256, 3);
    const auto float_small = inspirecv::ImageT<float>::Create(
      256, 256, 3, float_small_pixels.data());
    auto float_small_output = float_small.Resize(192, 192, true);
    require_backend(inspirecv::AccelerationBackend::kCpu);
    auto float_small_rotate = float_small.Rotate90();
    require_backend(inspirecv::AccelerationBackend::kCpu);

    const auto float_profile_pixels = MakePixels<float>(384, 384, 3);
    const auto float_profile = inspirecv::ImageT<float>::Create(
      384, 384, 3, float_profile_pixels.data());
    auto float_profile_output = float_profile.Resize(288, 288, true);
    require_backend(inspirecv::AccelerationBackend::kCpu);

    const auto float_large_pixels = MakePixels<float>(512, 512, 3);
    const auto float_large = inspirecv::ImageT<float>::Create(
      512, 512, 3, float_large_pixels.data());
    auto float_large_output = float_large.Resize(384, 384, true);
    require_backend(inspirecv::AccelerationBackend::kCuda);
    auto float_large_rotation = float_large.Rotate90();
    require_backend(inspirecv::AccelerationBackend::kCuda);

    const auto unprofiled_pixels = MakePixels<uint8_t>(512, 512, 4);
    const auto unprofiled = inspirecv::Image::Create(
      512, 512, 4, unprofiled_pixels.data());
    auto unprofiled_output = unprofiled.Rotate180();
    require_backend(inspirecv::AccelerationBackend::kCpu);

    REQUIRE_FALSE(small_bilinear.Empty());
    REQUIRE_FALSE(small_rotation.Empty());
    REQUIRE_FALSE(profiled_rotation.Empty());
    REQUIRE_FALSE(near_output.Empty());
    REQUIRE_FALSE(far_output.Empty());
    REQUIRE_FALSE(float_small_output.Empty());
    REQUIRE_FALSE(float_small_rotate.Empty());
    REQUIRE_FALSE(float_profile_output.Empty());
    REQUIRE_FALSE(float_large_output.Empty());
    REQUIRE_FALSE(float_large_rotation.Empty());
    REQUIRE_FALSE(unprofiled_output.Empty());
}

TEST_CASE("image_cuda_visual_report", "[image][cuda][visual]") {
    const char* output_directory =
      std::getenv("INSPIRECV_IMAGE_CUDA_VISUAL_DIR");
    if (!output_directory || !*output_directory ||
        !inspirecv::IsCudaAccelerationAvailable()) {
        SUCCEED("visual output not requested or CUDA unavailable");
        return;
    }
    RestoreAccelerationState restore;
    const std::string directory(output_directory);
    const auto pixels = MakePixels<uint8_t>(640, 480, 3);
    const auto source = inspirecv::Image::Create(
      640, 480, 3, pixels.data());
    REQUIRE(source.Write(directory + "/source.png"));

    const inspirecv::TransformMatrix affine(
      0.94f, 0.23f, -31.0f, -0.17f, 1.03f, 27.0f);
    const auto run_pair = [&](const std::string& name, const auto& operation) {
        REQUIRE(inspirecv::SetCudaAccelerationEnabled(false));
        auto cpu = operation(source);
        REQUIRE(inspirecv::SetCudaAccelerationEnabled(true));
        inspirecv::SetAccelerationPreference(
          inspirecv::AccelerationPreference::kCuda);
        auto cuda = operation(source);
        REQUIRE(inspirecv::GetLastImageExecutionBackend() ==
                inspirecv::AccelerationBackend::kCuda);
        SaveVisualComparison(cpu, cuda, directory, name);
    };

    run_pair("resize_bilinear", [](const inspirecv::Image& image) {
        return image.Resize(384, 288, true);
    });
    run_pair("warp_affine", [&](const inspirecv::Image& image) {
        return image.WarpAffine(affine, 640, 480);
    });
    run_pair("rotate90", [](const inspirecv::Image& image) {
        return image.Rotate90();
    });
    run_pair("rotate180", [](const inspirecv::Image& image) {
        return image.Rotate180();
    });
    run_pair("rotate270", [](const inspirecv::Image& image) {
        return image.Rotate270();
    });
}

#endif  // INSPIRECV_HAS_CUDA
