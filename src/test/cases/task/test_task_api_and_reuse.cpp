#include "../../common/common.h"

#include <inspirecv/task/legacy.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using inspirecv::TaskStatus;
using inspirecv::TransformMatrix;
using inspirecv::task::Matrix;
using inspirecv::task::StreamTask;
using inspirecv::task::TensorLayout;
using inspirecv::task::TensorView;

using RawConvertSignature = TaskStatus (StreamTask::*)(const uint8_t*, int, int, int, void*, int,
                                                       int, int, int, halide_type_t);
using TensorConvertSignature = TaskStatus (StreamTask::*)(const uint8_t*, int, int, int,
                                                          const TensorView&);

static_assert(static_cast<int>(TensorLayout::NHWC) == 0, "TensorLayout ABI changed");
static_assert(static_cast<int>(TensorLayout::NCHW) == 1, "TensorLayout ABI changed");
static_assert(static_cast<int>(TensorLayout::NC4HW4) == 2, "TensorLayout ABI changed");
static_assert(
  std::is_same<decltype(std::declval<const StreamTask&>().GetMatrix()), const Matrix&>::value,
  "GetMatrix signature changed");

constexpr RawConvertSignature kMixedCaseRawConvert =
  static_cast<RawConvertSignature>(&StreamTask::Convert);
constexpr TensorConvertSignature kMixedCaseTensorConvert =
  static_cast<TensorConvertSignature>(&StreamTask::Convert);
constexpr RawConvertSignature kLegacyRawConvert =
  static_cast<RawConvertSignature>(&StreamTask::convert);
constexpr TensorConvertSignature kLegacyTensorConvert =
  static_cast<TensorConvertSignature>(&StreamTask::convert);

TransformMatrix ResizeTransform(int source_width, int source_height, int destination_width,
                                int destination_height) {
    return TransformMatrix(static_cast<float>(source_width) / destination_width, 0.0f, 0.0f, 0.0f,
                           static_cast<float>(source_height) / destination_height, 0.0f);
}

TaskStatus ConvertTensorWithFreshTask(const StreamTask::Config& config,
                                      const TransformMatrix& transform,
                                      const std::vector<uint8_t>& source, int source_width,
                                      int source_height, const TensorView& output) {
    auto* task = StreamTask::Create(config);
    task->SetMatrix(transform);
    const TaskStatus status = task->Convert(source.data(), source_width, source_height, 0, output);
    StreamTask::Destroy(task);
    return status;
}

TaskStatus ConvertImageWithFreshTask(const StreamTask::Config& config,
                                     const TransformMatrix& transform,
                                     const std::vector<uint8_t>& source, int source_width,
                                     int source_height, std::vector<uint8_t>* output,
                                     int destination_width, int destination_height) {
    auto* task = StreamTask::Create(config);
    task->SetMatrix(transform);
    const TaskStatus status =
      task->Convert(source.data(), source_width, source_height, 0, output->data(),
                    destination_width, destination_height, 3, 0, halide_type_of<uint8_t>());
    StreamTask::Destroy(task);
    return status;
}

}  // namespace

TEST_CASE("task_public_mixed_case_and_legacy_entry_points_match", "[task][api][contract]") {
    REQUIRE(kMixedCaseRawConvert != nullptr);
    REQUIRE(kMixedCaseTensorConvert != nullptr);
    REQUIRE(kLegacyRawConvert != nullptr);
    REQUIRE(kLegacyTensorConvert != nullptr);

    StreamTask::Config config;
    config.sourceFormat = inspirecv::task::BGR;
    config.destFormat = inspirecv::task::RGB;
    const uint8_t source[] = {
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    };
    uint8_t mixed_case_output[12] = {};
    uint8_t legacy_output[12] = {};

    auto* mixed_case = StreamTask::Create(config);
    auto* legacy = StreamTask::create(config);
    mixed_case->SetMatrix(TransformMatrix::Identity());
    legacy->setMatrix(TransformMatrix::Identity());
    REQUIRE((mixed_case->*kMixedCaseRawConvert)(source, 2, 2, 0, mixed_case_output, 2, 2, 3, 0,
                                                halide_type_of<uint8_t>()) == inspirecv::SUCCESS);
    REQUIRE((legacy->*kLegacyRawConvert)(source, 2, 2, 0, legacy_output, 2, 2, 3, 0,
                                         halide_type_of<uint8_t>()) == inspirecv::SUCCESS);
    REQUIRE(std::equal(std::begin(mixed_case_output), std::end(mixed_case_output),
                       std::begin(legacy_output)));
    StreamTask::Destroy(mixed_case);
    StreamTask::destroy(legacy);
}

TEST_CASE("task_reused_instance_rebuilds_every_execution_plan",
          "[task][core][state][tensor_view]") {
    constexpr int kSourceWidth = 7;
    constexpr int kSourceHeight = 5;
    constexpr int kDestinationWidth = 5;
    constexpr int kDestinationHeight = 4;
    constexpr float kSentinel = -12345.0f;

    std::vector<uint8_t> source(static_cast<size_t>(kSourceWidth) * kSourceHeight * 3);
    for (size_t i = 0; i < source.size(); ++i) {
        source[i] = static_cast<uint8_t>((i * 37 + 19) & 255);
    }

    StreamTask::Config config;
    config.sourceFormat = inspirecv::task::BGR;
    config.destFormat = inspirecv::task::RGB;
    config.filterType = inspirecv::task::BILINEAR;
    config.wrap = inspirecv::task::CLAMP_TO_EDGE;
    config.mean[0] = 17.0f;
    config.mean[1] = 31.0f;
    config.mean[2] = 47.0f;
    config.normal[0] = 0.25f;
    config.normal[1] = 0.125f;
    config.normal[2] = 0.0625f;

    auto* reused = StreamTask::Create(config);

    SECTION("NHWC to NCHW to NC4HW4 to uint8") {
        std::vector<float> actual_nhwc(kDestinationWidth * kDestinationHeight * 3, kSentinel);
        std::vector<float> expected_nhwc(actual_nhwc.size(), kSentinel);
        TensorView actual_nhwc_view{actual_nhwc.data(),
                                    kDestinationWidth,
                                    kDestinationHeight,
                                    3,
                                    halide_type_of<float>(),
                                    TensorLayout::NHWC,
                                    0,
                                    0};
        TensorView expected_nhwc_view = actual_nhwc_view;
        expected_nhwc_view.data = expected_nhwc.data();

        reused->SetMatrix(TransformMatrix::Identity());
        REQUIRE(reused->Convert(source.data(), kSourceWidth, kSourceHeight, 0, actual_nhwc_view) ==
                inspirecv::SUCCESS);
        REQUIRE(ConvertTensorWithFreshTask(config, TransformMatrix::Identity(), source,
                                           kSourceWidth, kSourceHeight,
                                           expected_nhwc_view) == inspirecv::SUCCESS);
        REQUIRE(actual_nhwc == expected_nhwc);

        constexpr size_t kNchwRowElements = kDestinationWidth + 3;
        constexpr size_t kNchwPlaneElements = kNchwRowElements * kDestinationHeight + 5;
        std::vector<float> actual_nchw(kNchwPlaneElements * 3, kSentinel);
        std::vector<float> expected_nchw(actual_nchw.size(), kSentinel);
        TensorView actual_nchw_view{actual_nchw.data(),
                                    kDestinationWidth,
                                    kDestinationHeight,
                                    3,
                                    halide_type_of<float>(),
                                    TensorLayout::NCHW,
                                    kNchwRowElements * sizeof(float),
                                    kNchwPlaneElements * sizeof(float)};
        TensorView expected_nchw_view = actual_nchw_view;
        expected_nchw_view.data = expected_nchw.data();
        const auto resize =
          ResizeTransform(kSourceWidth, kSourceHeight, kDestinationWidth, kDestinationHeight);

        reused->SetMatrix(resize);
        REQUIRE(reused->Convert(source.data(), kSourceWidth, kSourceHeight, 0, actual_nchw_view) ==
                inspirecv::SUCCESS);
        REQUIRE(ConvertTensorWithFreshTask(config, resize, source, kSourceWidth, kSourceHeight,
                                           expected_nchw_view) == inspirecv::SUCCESS);
        REQUIRE(actual_nchw == expected_nchw);

        constexpr size_t kC4RowElements = kDestinationWidth * 4 + 4;
        std::vector<float> actual_c4(kC4RowElements * kDestinationHeight, kSentinel);
        std::vector<float> expected_c4(actual_c4.size(), kSentinel);
        TensorView actual_c4_view{actual_c4.data(),
                                  kDestinationWidth,
                                  kDestinationHeight,
                                  3,
                                  halide_type_of<float>(),
                                  TensorLayout::NC4HW4,
                                  kC4RowElements * sizeof(float),
                                  0};
        TensorView expected_c4_view = actual_c4_view;
        expected_c4_view.data = expected_c4.data();

        reused->SetMatrix(TransformMatrix::Identity());
        REQUIRE(reused->Convert(source.data(), kSourceWidth, kSourceHeight, 0, actual_c4_view) ==
                inspirecv::SUCCESS);
        REQUIRE(ConvertTensorWithFreshTask(config, TransformMatrix::Identity(), source,
                                           kSourceWidth, kSourceHeight,
                                           expected_c4_view) == inspirecv::SUCCESS);
        REQUIRE(actual_c4 == expected_c4);

        std::vector<uint8_t> actual_u8(kDestinationWidth * kDestinationHeight * 3, 0xCD);
        std::vector<uint8_t> expected_u8(actual_u8.size(), 0xCD);
        reused->SetMatrix(resize);
        REQUIRE(reused->Convert(source.data(), kSourceWidth, kSourceHeight, 0, actual_u8.data(),
                                kDestinationWidth, kDestinationHeight, 3, 0,
                                halide_type_of<uint8_t>()) == inspirecv::SUCCESS);
        REQUIRE(ConvertImageWithFreshTask(config, resize, source, kSourceWidth, kSourceHeight,
                                          &expected_u8, kDestinationWidth,
                                          kDestinationHeight) == inspirecv::SUCCESS);
        REQUIRE(actual_u8 == expected_u8);
    }

    SECTION("a rejected request cannot poison the next valid request") {
        float invalid_output[12] = {};
        TensorView invalid_view;
        invalid_view.data = invalid_output;
        invalid_view.width = 2;
        invalid_view.height = 2;
        invalid_view.channels = 3;
        invalid_view.type = halide_type_of<float>();
        invalid_view.layout = TensorLayout::NCHW;
        invalid_view.rowStride = sizeof(float);
        REQUIRE(reused->Convert(source.data(), kSourceWidth, kSourceHeight, 0, invalid_view) ==
                inspirecv::INPUT_DATA_ERROR);

        std::vector<uint8_t> actual(kDestinationWidth * kDestinationHeight * 3, 0xCD);
        std::vector<uint8_t> expected(actual.size(), 0xCD);
        const auto resize =
          ResizeTransform(kSourceWidth, kSourceHeight, kDestinationWidth, kDestinationHeight);
        reused->SetMatrix(resize);
        REQUIRE(reused->Convert(source.data(), kSourceWidth, kSourceHeight, 0, actual.data(),
                                kDestinationWidth, kDestinationHeight, 3, 0,
                                halide_type_of<uint8_t>()) == inspirecv::SUCCESS);
        REQUIRE(ConvertImageWithFreshTask(config, resize, source, kSourceWidth, kSourceHeight,
                                          &expected, kDestinationWidth,
                                          kDestinationHeight) == inspirecv::SUCCESS);
        REQUIRE(actual == expected);
    }

    StreamTask::Destroy(reused);
}
