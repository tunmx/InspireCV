#include "../../common/common.h"

// The umbrella must keep legacy source compatibility for integrations that
// historically included only task.h (for example InspireFace's optional Task
// preprocessing backend).
#include <inspirecv/task/task.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace {

using inspirecv::TaskStatus;
using inspirecv::task::Matrix;
using inspirecv::task::Point;
using inspirecv::task::Rect;
using inspirecv::task::StreamTask;
using inspirecv::task::TensorLayout;
using inspirecv::task::TensorView;

static_assert(inspirecv::SUCCESS == 0, "TaskStatus ABI changed");
static_assert(inspirecv::INPUT_DATA_ERROR == 1, "TaskStatus ABI changed");
static_assert(inspirecv::UNSUPPORTED_SAMPLER == 2, "TaskStatus ABI changed");
static_assert(inspirecv::UNSUPPORTED_CONVERSION == 3, "TaskStatus ABI changed");
static_assert(inspirecv::UNSUPPORTED_FLOAT_CONVERSION == 4, "TaskStatus ABI changed");

static_assert(inspirecv::task::RGBA == 0, "StreamFormat ABI changed");
static_assert(inspirecv::task::RGB == 1, "StreamFormat ABI changed");
static_assert(inspirecv::task::BGR == 2, "StreamFormat ABI changed");
static_assert(inspirecv::task::GRAY == 3, "StreamFormat ABI changed");
static_assert(inspirecv::task::BGRA == 4, "StreamFormat ABI changed");
static_assert(inspirecv::task::YUV_NV21 == 11, "StreamFormat ABI changed");
static_assert(inspirecv::task::YUV_NV12 == 12, "StreamFormat ABI changed");
static_assert(inspirecv::task::YUV_I420 == 13, "StreamFormat ABI changed");
static_assert(inspirecv::task::HSV_FULL == 14, "StreamFormat ABI changed");
static_assert(inspirecv::task::NEAREST == 0, "Filter ABI changed");
static_assert(inspirecv::task::BILINEAR == 1, "Filter ABI changed");
static_assert(inspirecv::task::BICUBIC == 2, "Filter ABI changed");
static_assert(inspirecv::task::CLAMP_TO_EDGE == 0, "Wrap ABI changed");
static_assert(inspirecv::task::ZERO == 1, "Wrap ABI changed");
static_assert(inspirecv::task::REPEAT == 2, "Wrap ABI changed");
static_assert(static_cast<int>(TensorLayout::NHWC) == 0, "TensorLayout ABI changed");
static_assert(static_cast<int>(TensorLayout::NCHW) == 1, "TensorLayout ABI changed");
static_assert(static_cast<int>(TensorLayout::NC4HW4) == 2, "TensorLayout ABI changed");

static_assert(sizeof(halide_type_t) == 4, "halide_type_t ABI changed");
static_assert(sizeof(Point) == 8 && alignof(Point) == 4, "Point ABI changed");
static_assert(sizeof(Rect) == 16 && alignof(Rect) == 4, "Rect ABI changed");
static_assert(sizeof(Matrix) == 40 && alignof(Matrix) == 4, "Matrix ABI changed");
static_assert(std::is_standard_layout<Point>::value, "Point must remain standard-layout");
static_assert(std::is_standard_layout<Rect>::value, "Rect must remain standard-layout");
static_assert(std::is_standard_layout<TensorView>::value,
              "TensorView must remain standard-layout");
static_assert(std::is_standard_layout<StreamTask::Config>::value,
              "StreamTask::Config must remain standard-layout");

static_assert(offsetof(Rect, fLeft) == 0, "Rect field layout changed");
static_assert(offsetof(Rect, fTop) == 4, "Rect field layout changed");
static_assert(offsetof(Rect, fRight) == 8, "Rect field layout changed");
static_assert(offsetof(Rect, fBottom) == 12, "Rect field layout changed");

static_assert(offsetof(StreamTask::Config, filterType) == 0, "Config layout changed");
static_assert(offsetof(StreamTask::Config, sourceFormat) == 4, "Config layout changed");
static_assert(offsetof(StreamTask::Config, destFormat) == 8, "Config layout changed");
static_assert(offsetof(StreamTask::Config, mean) == 12, "Config layout changed");
static_assert(offsetof(StreamTask::Config, normal) == 28, "Config layout changed");
static_assert(offsetof(StreamTask::Config, wrap) == 44, "Config layout changed");
static_assert(offsetof(StreamTask::Config, preserveIdentityFloatOrder) == 48,
              "Config layout changed");
static_assert(sizeof(StreamTask::Config) == 52, "Config size changed");

constexpr std::size_t kTensorViewRowStrideOffset = sizeof(void*) == 8 ? 32 : 24;
constexpr std::size_t kTensorViewSize = sizeof(void*) == 8 ? 48 : 32;
static_assert(offsetof(TensorView, data) == 0, "TensorView layout changed");
static_assert(offsetof(TensorView, width) == sizeof(void*), "TensorView layout changed");
static_assert(offsetof(TensorView, rowStride) == kTensorViewRowStrideOffset,
              "TensorView layout changed");
static_assert(offsetof(TensorView, channelStride) ==
                kTensorViewRowStrideOffset + sizeof(std::size_t),
              "TensorView layout changed");
static_assert(sizeof(TensorView) == kTensorViewSize, "TensorView size changed");

using CreateSignature = StreamTask* (*)(const StreamTask::Config&);
using DestroySignature = void (*)(StreamTask*);
using MatrixSetterSignature = void (StreamTask::*)(const Matrix&);
using RawConvertSignature = TaskStatus (StreamTask::*)(
    const uint8_t*, int, int, int, void*, int, int, int, int, halide_type_t);
using TensorConvertSignature = TaskStatus (StreamTask::*)(
    const uint8_t*, int, int, int, const TensorView&);

static_assert(std::is_same<decltype(&StreamTask::Create), CreateSignature>::value,
              "StreamTask::Create signature changed");
static_assert(std::is_same<decltype(&StreamTask::Destroy), DestroySignature>::value,
              "StreamTask::Destroy signature changed");
constexpr MatrixSetterSignature kSetMatrix =
    static_cast<MatrixSetterSignature>(&StreamTask::SetMatrix);
constexpr RawConvertSignature kRawConvert =
    static_cast<RawConvertSignature>(&StreamTask::Convert);
constexpr TensorConvertSignature kTensorConvert =
    static_cast<TensorConvertSignature>(&StreamTask::Convert);

}  // namespace

TEST_CASE("task_legacy_raw_engine_contract_is_frozen",
          "[task][legacy][contract]") {
    REQUIRE(kSetMatrix != nullptr);
    REQUIRE(kRawConvert != nullptr);
    REQUIRE(kTensorConvert != nullptr);
}
