#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
constexpr int kWindowsErrorMacro = ERROR;
#endif

#include <inspirecv/inspirecv.h>
#include <inspirecv/check.h>
#include <inspirecv/task/pipeline.h>

#include <cstdint>
#include <sstream>
#include <vector>

#if defined(INSPIRECV_TEST_EMBEDDED_HALIDE_HOST)
extern "C" __declspec(dllimport) uint32_t InspireCvHalideType();
extern "C" __declspec(dllimport) uint32_t HostHalideType();
#endif

#if defined(_WIN32)
#ifndef ERROR
#error "Including InspireCV must preserve the Windows ERROR macro"
#endif
static_assert(ERROR == kWindowsErrorMacro, "InspireCV must retain Windows ERROR");
#pragma push_macro("ERROR")
#undef ERROR
#endif

using LogSeverity = inspirecv::logging::ISFLogging::LogSeverity;
static_assert(static_cast<int>(LogSeverity::INFO) == 0, "INFO value must remain stable");
static_assert(static_cast<int>(LogSeverity::WARN) == 1, "WARN value must remain stable");
static_assert(static_cast<int>(LogSeverity::ERROR) == 2, "ERROR value must remain stable");
static_assert(static_cast<int>(LogSeverity::FATAL) == 3, "FATAL value must remain stable");

#if defined(_WIN32)
#pragma pop_macro("ERROR")
#endif

// Compile direct and forwarded severity arguments, including caller aliases.
// The branches are never taken, so consumer verification emits no log.
#define INSPIRECV_CONSUMER_INFO_ALIAS INFO
#define INSPIRECV_CONSUMER_ERROR_ALIAS ERROR
static void CheckLoggingCompatibility(bool emit) {
    if (emit) INSPIRECV_LOG(INFO) << "direct info";
    if (emit) INSPIRECV_LOG(ERROR) << "direct error";
    if (emit) INSPIRECV_LOG(INSPIRECV_CONSUMER_INFO_ALIAS) << "aliased info";
    if (emit) INSPIRECV_LOG(INSPIRECV_CONSUMER_ERROR_ALIAS) << "aliased error";
    INSPIRECV_LOG_IF(INFO, emit) << "conditional info";
    INSPIRECV_LOG_IF(WARN, emit) << "conditional warning";
    INSPIRECV_LOG_IF(ERROR, emit) << "conditional error";
    INSPIRECV_LOG_IF(FATAL, emit) << "conditional fatal";
    INSPIRECV_LOG_IF(INSPIRECV_CONSUMER_ERROR_ALIAS, emit) << "conditional alias";
    INSPIRECV_CHECK(!emit);
    INSPIRECV_CHECK_EQ(1, 1);
    INSPIRECV_CHECK_NE(1, 2);
    INSPIRECV_CHECK_LE(1, 2);
    INSPIRECV_CHECK_LT(1, 2);
    INSPIRECV_CHECK_GE(2, 1);
    INSPIRECV_CHECK_GT(2, 1);
}
#undef INSPIRECV_CONSUMER_INFO_ALIAS
#undef INSPIRECV_CONSUMER_ERROR_ALIAS

// Exercise each supported public function-template instantiation through
// installed libraries and, on Windows, an OBJECT payload embedded in a DLL.
template <typename T>
static bool CheckGeometryTemplates() {
    const std::vector<inspirecv::Point<T>> points = {{T(0), T(0)}, {T(4), T(0)}, {T(0), T(4)}};
    const auto transformed = inspirecv::ApplyTransformToPoints(
      points, inspirecv::TransformMatrix::Identity());
    if (transformed.size() != points.size() || transformed[1].GetX() != T(4)) return false;
    const auto bounds = inspirecv::MinBoundingRect(points);
    if (bounds.GetWidth() != T(4) || bounds.GetHeight() != T(4)) return false;
    if (inspirecv::SimilarityTransformEstimate(points, points).Squeeze().size() != 6) return false;
    if (inspirecv::SimilarityTransformEstimateUmeyama(points, points).Squeeze().size() != 6) return false;
    const auto transformed_bounds = inspirecv::ApplyTransformToRect(
      bounds, inspirecv::TransformMatrix::Identity());
    if (transformed_bounds.GetWidth() != T(4) || transformed_bounds.GetHeight() != T(4)) return false;
    if (bounds.template As<int>().GetWidth() != 4 ||
        bounds.template As<float>().GetWidth() != 4.0f ||
        bounds.template As<double>().GetWidth() != 4.0) return false;
    std::ostringstream description;
    description << points.front() << points << bounds << inspirecv::Size<T>(T(4), T(4));
    return !description.str().empty();
}

int main() {
    CheckLoggingCompatibility(false);
#if defined(INSPIRECV_TEST_EMBEDDED_HALIDE_HOST)
    if (InspireCvHalideType() != 0x12002u || HostHalideType() != 0x12002u) return 9;
#endif
    const auto& library = inspirecv::GetLibraryInfo();
    if (library.version_major != 1 || library.version_minor != 0) return 1;

    const uint8_t pixels[] = {
      1, 2, 3, 4, 5, 6,
      7, 8, 9, 10, 11, 12,
    };
    const auto source = inspirecv::Image::Create(2, 2, 3, pixels);

    inspirecv::task::PipelineOptions options;
    options.input_format = inspirecv::task::PixelFormat::kBgr;
    options.output_format = inspirecv::task::PixelFormat::kBgr;
    inspirecv::task::Pipeline pipeline(options);
    if (pipeline.ConfigurationStatus() != inspirecv::task::Status::kOk ||
        pipeline.SetTransform(inspirecv::TransformMatrix::Identity()) !=
          inspirecv::task::Status::kOk) {
        return 2;
    }

    inspirecv::Image destination;
    if (pipeline.Run(source, 2, 2, &destination) !=
        inspirecv::task::Status::kOk) {
        return 3;
    }
    if (destination.Width() != 2 || destination.Height() != 2 ||
        destination.Channels() != 3) {
        return 4;
    }
    for (size_t index = 0; index < sizeof(pixels); ++index) {
        if (destination.Data()[index] != pixels[index]) return 5;
    }

    if (!CheckGeometryTemplates<int>() || !CheckGeometryTemplates<float>() ||
        !CheckGeometryTemplates<double>()) return 6;
    const inspirecv::Point<int> point_i(1, 2);
    const inspirecv::Point<float> point_f(1.0f, 2.0f);
    const inspirecv::Point<double> point_d(1.0, 2.0);
    if (point_i.As<float>().GetX() != 1.0f || point_i.As<double>().GetX() != 1.0 ||
        point_f.As<int>().GetX() != 1 || point_f.As<double>().GetX() != 1.0 ||
        point_d.As<int>().GetX() != 1 || point_d.As<float>().GetX() != 1.0f) return 7;
    const float float_pixel = 1.0f;
    const auto float_image = inspirecv::ImageT<float>::Create(1, 1, 1, &float_pixel);
    std::ostringstream images;
    images << source << float_image;
    if (images.str().empty()) return 8;
    return 0;
}
