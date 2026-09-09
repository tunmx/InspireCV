#include "../../common/common.h"

#include <inspirecv/task/legacy.h>
#include <inspirecv/task/task.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

inspirecv::TransformMatrix ResizeTransform(int source_width, int source_height,
                                           int destination_width,
                                           int destination_height) {
    return inspirecv::TransformMatrix(
      static_cast<float>(source_width) / destination_width, 0.0f, 0.0f,
      0.0f, static_cast<float>(source_height) / destination_height, 0.0f);
}

inspirecv::task::StreamTask::Config LegacyConfig(
  const inspirecv::task::PipelineOptions& options) {
    inspirecv::task::StreamTask::Config legacy;
    legacy.filterType =
      static_cast<inspirecv::task::Filter>(options.sampling);
    legacy.sourceFormat =
      static_cast<inspirecv::task::StreamFormat>(options.input_format);
    legacy.destFormat =
      static_cast<inspirecv::task::StreamFormat>(options.output_format);
    for (int channel = 0; channel < 4; ++channel) {
        legacy.mean[channel] = options.mean[channel];
        legacy.normal[channel] = options.scale[channel];
    }
    legacy.wrap = static_cast<inspirecv::task::Wrap>(options.border);
    legacy.preserveIdentityFloatOrder =
      options.preserve_identity_float_order;
    return legacy;
}

std::vector<uint8_t> MakeSource(int width, int height) {
    std::vector<uint8_t> source(static_cast<size_t>(width) * height * 3);
    for (size_t index = 0; index < source.size(); ++index) {
        source[index] = static_cast<uint8_t>((index * 53 + 17) & 255);
    }
    return source;
}

double Median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

}  // namespace

static_assert(!std::is_copy_constructible<inspirecv::task::Pipeline>::value,
              "Pipeline must not duplicate execution state");
static_assert(std::is_nothrow_move_constructible<
                inspirecv::task::Pipeline>::value,
              "Pipeline should be safely movable");
static_assert(std::is_standard_layout<inspirecv::task::RawImageView>::value,
              "RawImageView must remain easy to bind across API boundaries");
static_assert(std::is_standard_layout<inspirecv::task::TensorBuffer>::value,
              "TensorBuffer must remain easy to bind across API boundaries");

TEST_CASE("task_pipeline_matches_frozen_raw_engine",
          "[task][api][pipeline][contract]") {
    constexpr int kSourceWidth = 7;
    constexpr int kSourceHeight = 5;
    const auto source_bytes = MakeSource(kSourceWidth, kSourceHeight);
    const auto source = inspirecv::Image::Create(
      kSourceWidth, kSourceHeight, 3, source_bytes.data(), false);

    inspirecv::task::PipelineOptions options;
    options.sampling = inspirecv::task::SamplingMode::kLinear;
    options.input_format = inspirecv::task::PixelFormat::kBgr;
    options.output_format = inspirecv::task::PixelFormat::kRgb;
    options.border = inspirecv::task::BorderMode::kReplicate;
    options.mean = {{11.0f, 23.0f, 47.0f, 0.0f}};
    options.scale = {{0.25f, 0.125f, 0.0625f, 1.0f}};

    inspirecv::task::Pipeline pipeline(options);
    const auto legacy_config = LegacyConfig(options);
    REQUIRE(pipeline.ConfigurationStatus() == inspirecv::task::Status::kOk);
    REQUIRE(pipeline.OutputFormat() == inspirecv::task::PixelFormat::kRgb);

    SECTION("non-invertible transform is rejected transactionally") {
        REQUIRE(pipeline.SetTransform(inspirecv::TransformMatrix::Identity()) ==
                inspirecv::task::Status::kOk);
        inspirecv::Image expected;
        REQUIRE(pipeline.Run(source, kSourceWidth, kSourceHeight, &expected) ==
                inspirecv::task::Status::kOk);

        const inspirecv::TransformMatrix singular(
          0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
        REQUIRE(pipeline.SetTransform(singular) ==
                inspirecv::task::Status::kNonInvertibleTransform);
        REQUIRE(std::string(inspirecv::task::StatusMessage(
                             inspirecv::task::Status::kNonInvertibleTransform)) ==
                "transform is not invertible");

        inspirecv::Image actual;
        REQUIRE(pipeline.Run(source, kSourceWidth, kSourceHeight, &actual) ==
                inspirecv::task::Status::kOk);
        REQUIRE_EQ_C_ARRAY(actual.Data(), expected.Data(), source_bytes.size());

        inspirecv::task::cuda::Pipeline device_pipeline(options);
        REQUIRE(device_pipeline.ConfigurationStatus() ==
                inspirecv::task::Status::kOk);
        REQUIRE(device_pipeline.SetTransform(singular) ==
                inspirecv::task::Status::kNonInvertibleTransform);
    }

    SECTION("owned Image output is byte-exact and reusable") {
        const int output_sizes[][2] = {{7, 5}, {5, 4}, {3, 2}};
        for (const auto& size : output_sizes) {
            const int output_width = size[0];
            const int output_height = size[1];
            const auto transform = ResizeTransform(
              kSourceWidth, kSourceHeight, output_width, output_height);
            pipeline.SetTransform(transform);

            inspirecv::Image actual;
            REQUIRE(pipeline.Run(source, output_width, output_height,
                                 &actual) == inspirecv::task::Status::kOk);

            std::vector<uint8_t> expected(
              static_cast<size_t>(output_width) * output_height * 3);
            auto* legacy = inspirecv::task::StreamTask::Create(legacy_config);
            legacy->SetMatrix(transform);
            REQUIRE(legacy->Convert(
                      source_bytes.data(), kSourceWidth, kSourceHeight,
                      kSourceWidth * 3, expected.data(), output_width,
                      output_height, 3, 0,
                      halide_type_of<uint8_t>()) == inspirecv::SUCCESS);
            inspirecv::task::StreamTask::Destroy(legacy);

            REQUIRE(actual.Width() == output_width);
            REQUIRE(actual.Height() == output_height);
            REQUIRE(actual.Channels() == 3);
            REQUIRE_EQ_C_ARRAY(actual.Data(), expected.data(), expected.size());
        }
        REQUIRE_EQ_C_ARRAY(source.Data(), source_bytes.data(),
                           source_bytes.size());
    }

    SECTION("caller-owned CHW float output is bit-exact") {
        constexpr int kOutputWidth = 5;
        constexpr int kOutputHeight = 4;
        constexpr size_t kRowElements = kOutputWidth + 2;
        constexpr size_t kPlaneElements = kRowElements * kOutputHeight + 3;
        constexpr float kSentinel = -9876.5f;
        std::vector<float> actual(kPlaneElements * 3, kSentinel);
        std::vector<float> expected(actual.size(), kSentinel);
        inspirecv::task::TensorBuffer actual_buffer;
        actual_buffer.data = actual.data();
        actual_buffer.width = kOutputWidth;
        actual_buffer.height = kOutputHeight;
        actual_buffer.channels = 3;
        actual_buffer.element_type = inspirecv::task::ElementType::kFloat32;
        actual_buffer.order = inspirecv::task::TensorOrder::kChw;
        actual_buffer.row_stride_bytes = kRowElements * sizeof(float);
        actual_buffer.channel_stride_bytes =
          kPlaneElements * sizeof(float);

        inspirecv::task::TensorView expected_view;
        expected_view.data = expected.data();
        expected_view.width = kOutputWidth;
        expected_view.height = kOutputHeight;
        expected_view.channels = 3;
        expected_view.type = halide_type_of<float>();
        expected_view.layout = inspirecv::task::TensorLayout::NCHW;
        expected_view.rowStride = actual_buffer.row_stride_bytes;
        expected_view.channelStride = actual_buffer.channel_stride_bytes;
        const auto transform = ResizeTransform(
          kSourceWidth, kSourceHeight, kOutputWidth, kOutputHeight);

        pipeline.SetTransform(transform);
        REQUIRE(pipeline.Run(source, actual_buffer) ==
                inspirecv::task::Status::kOk);

        auto* legacy = inspirecv::task::StreamTask::Create(legacy_config);
        legacy->SetMatrix(transform);
        REQUIRE(legacy->Convert(source_bytes.data(), kSourceWidth,
                                kSourceHeight, kSourceWidth * 3,
                                expected_view) == inspirecv::SUCCESS);
        inspirecv::task::StreamTask::Destroy(legacy);
        REQUIRE(actual == expected);
    }

    SECTION("preallocated Image output preserves storage and is byte-exact") {
        constexpr int kOutputWidth = 5;
        constexpr int kOutputHeight = 4;
        pipeline.SetTransform(ResizeTransform(
          kSourceWidth, kSourceHeight, kOutputWidth, kOutputHeight));

        inspirecv::Image expected;
        REQUIRE(pipeline.Run(source, kOutputWidth, kOutputHeight, &expected) ==
                inspirecv::task::Status::kOk);

        auto reusable = inspirecv::Image::Create(
          kOutputWidth, kOutputHeight, 3);
        reusable.Fill(211.0);
        const auto* allocation = reusable.Data();
        REQUIRE(pipeline.RunInto(source, &reusable) ==
                inspirecv::task::Status::kOk);
        REQUIRE(reusable.Data() == allocation);
        REQUIRE_EQ_C_ARRAY(reusable.Data(), expected.Data(),
                           static_cast<size_t>(kOutputWidth) *
                             kOutputHeight * 3);
    }

    SECTION("in-place Image conversion is rejected before writing") {
        auto in_place = inspirecv::Image::Create(
          kSourceWidth, kSourceHeight, 3, source_bytes.data());
        const std::vector<uint8_t> before(
          in_place.Data(), in_place.Data() + source_bytes.size());
        pipeline.SetTransform(inspirecv::TransformMatrix::Identity());
        REQUIRE(pipeline.RunInto(in_place, &in_place) ==
                inspirecv::task::Status::kInvalidArgument);
        REQUIRE_EQ_C_ARRAY(in_place.Data(), before.data(), before.size());
    }

    SECTION("raw view and Image input share one execution path") {
        std::vector<uint8_t> from_image(source_bytes.size());
        std::vector<uint8_t> from_raw(source_bytes.size());
        inspirecv::task::TensorBuffer output;
        output.width = kSourceWidth;
        output.height = kSourceHeight;
        output.channels = 3;
        output.element_type = inspirecv::task::ElementType::kUInt8;
        output.order = inspirecv::task::TensorOrder::kHwc;
        pipeline.SetTransform(inspirecv::TransformMatrix::Identity());

        output.data = from_image.data();
        REQUIRE(pipeline.Run(source, output) == inspirecv::task::Status::kOk);
        output.data = from_raw.data();
        inspirecv::task::RawImageView raw;
        raw.data = source_bytes.data();
        raw.width = kSourceWidth;
        raw.height = kSourceHeight;
        raw.row_stride_bytes = kSourceWidth * 3;
        REQUIRE(pipeline.Run(raw, output) == inspirecv::task::Status::kOk);
        REQUIRE(from_raw == from_image);
    }

    SECTION("move transfers reusable execution state") {
        pipeline.SetTransform(inspirecv::TransformMatrix::Identity());
        inspirecv::task::Pipeline moved(std::move(pipeline));
        inspirecv::Image output;
        REQUIRE(moved.Run(source, kSourceWidth, kSourceHeight, &output) ==
                inspirecv::task::Status::kOk);
        REQUIRE(output.Width() == kSourceWidth);
        REQUIRE(output.Height() == kSourceHeight);
    }

    SECTION("invalid packed source leaves destination unchanged") {
        const uint8_t gray_bytes[] = {1, 2, 3, 4};
        const auto gray = inspirecv::Image::Create(2, 2, 1, gray_bytes);
        auto destination = inspirecv::Image::Create(1, 1, 1);
        destination.Fill(77.0);
        const auto* original_pointer = destination.Data();
        REQUIRE(pipeline.Run(gray, 2, 2, &destination) ==
                inspirecv::task::Status::kInvalidArgument);
        REQUIRE(destination.Data() == original_pointer);
        REQUIRE(destination.Data()[0] == 77);
        REQUIRE(std::string(inspirecv::task::StatusMessage(
                             inspirecv::task::Status::kInvalidArgument)) ==
                "invalid argument");
    }
}

TEST_CASE("task_pipeline_image_output_matches_tightly_packed_hwc",
          "[task][api][pipeline][image_bridge][contract]") {
    constexpr int kWidth = 3;
    constexpr int kHeight = 2;
    const std::vector<uint8_t> source_bytes = {
      3,  17, 251, 29, 43, 227, 61, 79, 199,
      97, 113, 173, 131, 149, 151, 181, 197, 127,
    };

    struct FormatCase {
        inspirecv::task::PixelFormat input;
        inspirecv::task::PixelFormat output;
        int channels;
    };
    const FormatCase cases[] = {
      {inspirecv::task::PixelFormat::kBgr,
       inspirecv::task::PixelFormat::kGray, 1},
      {inspirecv::task::PixelFormat::kBgr,
       inspirecv::task::PixelFormat::kBgr, 3},
      {inspirecv::task::PixelFormat::kBgr,
       inspirecv::task::PixelFormat::kRgb, 3},
      {inspirecv::task::PixelFormat::kBgr,
       inspirecv::task::PixelFormat::kBgra, 4},
      {inspirecv::task::PixelFormat::kRgb,
       inspirecv::task::PixelFormat::kRgba, 4},
    };

    inspirecv::task::RawImageView source;
    source.data = source_bytes.data();
    source.width = kWidth;
    source.height = kHeight;
    source.row_stride_bytes = 0;

    for (const auto& format_case : cases) {
        CAPTURE(format_case.channels);
        CAPTURE(static_cast<int>(format_case.input));
        CAPTURE(static_cast<int>(format_case.output));

        inspirecv::task::PipelineOptions options;
        options.input_format = format_case.input;
        options.output_format = format_case.output;
        inspirecv::task::Pipeline pipeline(options);
        pipeline.SetTransform(inspirecv::TransformMatrix::Identity());
        REQUIRE(pipeline.OutputFormat() == format_case.output);

        std::vector<uint8_t> expected(
          static_cast<size_t>(kWidth) * kHeight * format_case.channels);
        inspirecv::task::TensorBuffer tensor;
        tensor.data = expected.data();
        tensor.width = kWidth;
        tensor.height = kHeight;
        tensor.channels = format_case.channels;
        tensor.element_type = inspirecv::task::ElementType::kUInt8;
        tensor.order = inspirecv::task::TensorOrder::kHwc;
        REQUIRE(pipeline.Run(source, tensor) ==
                inspirecv::task::Status::kOk);

        inspirecv::Image owned;
        REQUIRE(pipeline.Run(source, kWidth, kHeight, &owned) ==
                inspirecv::task::Status::kOk);
        REQUIRE(owned.Width() == kWidth);
        REQUIRE(owned.Height() == kHeight);
        REQUIRE(owned.Channels() == format_case.channels);
        REQUIRE_EQ_C_ARRAY(owned.Data(), expected.data(), expected.size());

        auto reusable = inspirecv::Image::Create(
          kWidth, kHeight, format_case.channels);
        reusable.Fill(207.0);
        const auto* allocation = reusable.Data();
        REQUIRE(pipeline.RunInto(source, &reusable) ==
                inspirecv::task::Status::kOk);
        REQUIRE(reusable.Data() == allocation);
        REQUIRE_EQ_C_ARRAY(reusable.Data(), expected.data(), expected.size());
    }
}

TEST_CASE("task_pipeline_raw_yuv_and_status_contract",
          "[task][api][pipeline][raw][contract]") {
    constexpr int kWidth = 4;
    constexpr int kHeight = 4;
    const std::vector<uint8_t> nv21 = {
      16, 32, 64, 96, 24, 48, 80, 112,
      40, 72, 104, 136, 56, 88, 120, 152,
      140, 90, 132, 100, 124, 110, 116, 120,
    };

    inspirecv::task::PipelineOptions options;
    options.input_format = inspirecv::task::PixelFormat::kNv21;
    options.output_format = inspirecv::task::PixelFormat::kBgr;
    options.sampling = inspirecv::task::SamplingMode::kNearest;
    inspirecv::task::Pipeline pipeline(options);
    REQUIRE(pipeline.OutputFormat() == inspirecv::task::PixelFormat::kBgr);

    inspirecv::task::RawImageView source;
    source.data = nv21.data();
    source.width = kWidth;
    source.height = kHeight;
    std::vector<uint8_t> actual(kWidth * kHeight * 3);
    inspirecv::task::TensorBuffer destination;
    destination.data = actual.data();
    destination.width = kWidth;
    destination.height = kHeight;
    destination.channels = 3;
    destination.element_type = inspirecv::task::ElementType::kUInt8;
    destination.order = inspirecv::task::TensorOrder::kHwc;
    REQUIRE(pipeline.Run(source, destination) ==
            inspirecv::task::Status::kOk);

    const auto legacy_options = LegacyConfig(options);
    std::vector<uint8_t> expected(actual.size());
    auto* legacy = inspirecv::task::StreamTask::Create(legacy_options);
    REQUIRE(legacy->Convert(nv21.data(), kWidth, kHeight, 0,
                            expected.data(), kWidth, kHeight, 3, 0,
                            halide_type_of<uint8_t>()) == inspirecv::SUCCESS);
    inspirecv::task::StreamTask::Destroy(legacy);
    REQUIRE(actual == expected);

    SECTION("raw YUV converts directly into an owned Image") {
        inspirecv::Image image;
        REQUIRE(pipeline.Run(source, kWidth, kHeight, &image) ==
                inspirecv::task::Status::kOk);
        REQUIRE(image.Width() == kWidth);
        REQUIRE(image.Height() == kHeight);
        REQUIRE(image.Channels() == 3);
        REQUIRE_EQ_C_ARRAY(image.Data(), expected.data(), expected.size());
    }

    SECTION("raw YUV reuses preallocated Image storage") {
        auto image = inspirecv::Image::Create(kWidth, kHeight, 3);
        image.Fill(199.0);
        const auto* allocation = image.Data();
        REQUIRE(pipeline.RunInto(source, &image) ==
                inspirecv::task::Status::kOk);
        REQUIRE(image.Data() == allocation);
        REQUIRE_EQ_C_ARRAY(image.Data(), expected.data(), expected.size());

        image.Fill(17.0);
        REQUIRE(pipeline.RunInto(source, &image) ==
                inspirecv::task::Status::kOk);
        REQUIRE(image.Data() == allocation);
        REQUIRE_EQ_C_ARRAY(image.Data(), expected.data(), expected.size());
    }

    SECTION("safe raw Image output is transactional on invalid input") {
        auto unchanged = inspirecv::Image::Create(1, 1, 3);
        unchanged.Fill(73.0);
        const auto* allocation = unchanged.Data();
        auto invalid_source = source;
        invalid_source.data = nullptr;
        REQUIRE(pipeline.Run(invalid_source, kWidth, kHeight, &unchanged) ==
                inspirecv::task::Status::kInvalidArgument);
        REQUIRE(unchanged.Data() == allocation);
        REQUIRE(unchanged.Width() == 1);
        REQUIRE(unchanged.Height() == 1);
        REQUIRE(unchanged.Data()[0] == 73);
    }

    SECTION("safe raw Image output is transactional on execution failure") {
        auto cubic_options = options;
        cubic_options.sampling = inspirecv::task::SamplingMode::kCubic;
        inspirecv::task::Pipeline cubic(cubic_options);
        auto unchanged = inspirecv::Image::Create(1, 1, 3);
        unchanged.Fill(91.0);
        const auto* allocation = unchanged.Data();
        REQUIRE(cubic.Run(source, kWidth, kHeight, &unchanged) ==
                inspirecv::task::Status::kUnsupportedSampling);
        REQUIRE(unchanged.Data() == allocation);
        REQUIRE(unchanged.Width() == 1);
        REQUIRE(unchanged.Height() == 1);
        REQUIRE(unchanged.Data()[0] == 91);
    }

    SECTION("RunInto validates destination storage and aliasing") {
        auto wrong_channels = inspirecv::Image::Create(kWidth, kHeight, 1);
        wrong_channels.Fill(31.0);
        REQUIRE(pipeline.RunInto(source, &wrong_channels) ==
                inspirecv::task::Status::kInvalidArgument);
        REQUIRE(wrong_channels.Data()[0] == 31);

        auto overlapping = inspirecv::Image::Create(kWidth, kHeight, 3);
        inspirecv::task::RawImageView aliased = source;
        aliased.data = overlapping.Data();
        REQUIRE(pipeline.RunInto(aliased, &overlapping) ==
                inspirecv::task::Status::kInvalidArgument);
    }

    SECTION("invalid enum values cannot construct a runnable pipeline") {
        const auto require_invalid = [&](const inspirecv::task::PipelineOptions&
                                           invalid_options) {
            inspirecv::task::Pipeline invalid(invalid_options);
            REQUIRE(invalid.ConfigurationStatus() ==
                    inspirecv::task::Status::kInvalidArgument);
            REQUIRE(invalid.Run(source, destination) ==
                    inspirecv::task::Status::kInvalidArgument);
        };

        auto invalid_options = options;
        invalid_options.input_format =
          static_cast<inspirecv::task::PixelFormat>(255);
        require_invalid(invalid_options);
        invalid_options = options;
        invalid_options.output_format =
          static_cast<inspirecv::task::PixelFormat>(255);
        require_invalid(invalid_options);
        invalid_options = options;
        invalid_options.sampling =
          static_cast<inspirecv::task::SamplingMode>(255);
        require_invalid(invalid_options);
        invalid_options = options;
        invalid_options.border =
          static_cast<inspirecv::task::BorderMode>(255);
        require_invalid(invalid_options);
        invalid_options = options;
        invalid_options.backend_preference =
          static_cast<inspirecv::task::BackendPreference>(255);
        require_invalid(invalid_options);
    }

    SECTION("unsupported cubic sampling has a typed status") {
        auto cubic_options = options;
        cubic_options.sampling = inspirecv::task::SamplingMode::kCubic;
        inspirecv::task::Pipeline cubic(cubic_options);
        REQUIRE(cubic.Run(source, destination) ==
                inspirecv::task::Status::kUnsupportedSampling);
        REQUIRE(std::string(inspirecv::task::StatusMessage(
                             inspirecv::task::Status::kUnsupportedSampling)) ==
                "unsupported sampling mode");
    }
}

TEST_CASE("task_pipeline_preallocated_image_performance",
          "[benchmark][task][api][image_bridge]") {
    constexpr int kSourceWidth = 640;
    constexpr int kSourceHeight = 480;
    constexpr int kOutputWidth = 112;
    constexpr int kOutputHeight = 112;
    constexpr int kRounds = 9;
    constexpr int kLoopsPerRound = 400;
    const auto source_bytes = MakeSource(kSourceWidth, kSourceHeight);

    inspirecv::task::RawImageView source;
    source.data = source_bytes.data();
    source.width = kSourceWidth;
    source.height = kSourceHeight;
    source.row_stride_bytes = kSourceWidth * 3;

    inspirecv::task::PipelineOptions options;
    options.input_format = inspirecv::task::PixelFormat::kBgr;
    options.output_format = inspirecv::task::PixelFormat::kBgr;
    options.sampling = inspirecv::task::SamplingMode::kLinear;
    inspirecv::task::Pipeline pipeline(options);
    pipeline.SetTransform(ResizeTransform(
      kSourceWidth, kSourceHeight, kOutputWidth, kOutputHeight));

    std::vector<uint8_t> tensor_bytes(
      static_cast<size_t>(kOutputWidth) * kOutputHeight * 3);
    inspirecv::task::TensorBuffer tensor;
    tensor.data = tensor_bytes.data();
    tensor.width = kOutputWidth;
    tensor.height = kOutputHeight;
    tensor.channels = 3;
    tensor.element_type = inspirecv::task::ElementType::kUInt8;
    tensor.order = inspirecv::task::TensorOrder::kHwc;
    auto image = inspirecv::Image::Create(kOutputWidth, kOutputHeight, 3);

    REQUIRE(pipeline.Run(source, tensor) == inspirecv::task::Status::kOk);
    REQUIRE(pipeline.RunInto(source, &image) ==
            inspirecv::task::Status::kOk);
    REQUIRE_EQ_C_ARRAY(image.Data(), tensor_bytes.data(), tensor_bytes.size());

    auto measure = [&](bool image_output) {
        inspirecv::task::Status status = inspirecv::task::Status::kOk;
        const auto start = std::chrono::steady_clock::now();
        for (int loop = 0; loop < kLoopsPerRound; ++loop) {
            status = image_output ? pipeline.RunInto(source, &image)
                                  : pipeline.Run(source, tensor);
        }
        const auto stop = std::chrono::steady_clock::now();
        REQUIRE(status == inspirecv::task::Status::kOk);
        return std::chrono::duration<double, std::micro>(stop - start).count() /
               kLoopsPerRound;
    };

    std::vector<double> tensor_samples;
    std::vector<double> image_samples;
    for (int round = 0; round < kRounds; ++round) {
        if ((round & 1) == 0) {
            tensor_samples.push_back(measure(false));
            image_samples.push_back(measure(true));
        } else {
            image_samples.push_back(measure(true));
            tensor_samples.push_back(measure(false));
        }
    }

    const double tensor_median = Median(tensor_samples);
    const double image_median = Median(image_samples);
    std::printf(
      "[TaskImageBridgeBench] tensor_hwc_us=%.5f run_into_image_us=%.5f "
      "delta=%+.2f%%\n",
      tensor_median, image_median,
      (image_median / tensor_median - 1.0) * 100.0);
    REQUIRE(image_median <= tensor_median * 1.10);
}
