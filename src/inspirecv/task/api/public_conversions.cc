#include "inspirecv/task/api/public_conversions.h"

#include <algorithm>

namespace inspirecv {
namespace task {
namespace api_internal {
namespace {

bool ToInternalFormat(PixelFormat format, StreamFormat* destination) {
    if (!destination) return false;
    switch (format) {
        case PixelFormat::kRgba: *destination = RGBA; return true;
        case PixelFormat::kRgb: *destination = RGB; return true;
        case PixelFormat::kBgr: *destination = BGR; return true;
        case PixelFormat::kGray: *destination = GRAY; return true;
        case PixelFormat::kBgra: *destination = BGRA; return true;
        case PixelFormat::kYCrCb: *destination = YCrCb; return true;
        case PixelFormat::kYuv: *destination = YUV; return true;
        case PixelFormat::kHsv: *destination = HSV; return true;
        case PixelFormat::kXyz: *destination = XYZ; return true;
        case PixelFormat::kBgr555: *destination = BGR555; return true;
        case PixelFormat::kBgr565: *destination = BGR565; return true;
        case PixelFormat::kNv21: *destination = YUV_NV21; return true;
        case PixelFormat::kNv12: *destination = YUV_NV12; return true;
        case PixelFormat::kI420: *destination = YUV_I420; return true;
        case PixelFormat::kHsvFull: *destination = HSV_FULL; return true;
    }
    return false;
}

bool ToInternalSampling(SamplingMode sampling, Filter* destination) {
    if (!destination) return false;
    switch (sampling) {
        case SamplingMode::kNearest: *destination = NEAREST; return true;
        case SamplingMode::kLinear: *destination = BILINEAR; return true;
        case SamplingMode::kCubic: *destination = BICUBIC; return true;
    }
    return false;
}

bool ToInternalBorder(BorderMode border, Wrap* destination) {
    if (!destination) return false;
    switch (border) {
        case BorderMode::kReplicate: *destination = CLAMP_TO_EDGE; return true;
        case BorderMode::kConstant: *destination = ZERO; return true;
        case BorderMode::kRepeat: *destination = REPEAT; return true;
    }
    return false;
}

bool IsValidPreference(BackendPreference preference) {
    switch (preference) {
        case BackendPreference::kDefault:
        case BackendPreference::kAuto:
        case BackendPreference::kCpu:
        case BackendPreference::kCuda:
            return true;
    }
    return false;
}

}  // namespace

Status ConfigurePipeline(const PipelineOptions& options,
                         internal::PipelineConfig* destination) {
    if (!destination || !IsValidPreference(options.backend_preference) ||
        !ToInternalFormat(options.input_format, &destination->source_format) ||
        !ToInternalFormat(options.output_format,
                          &destination->destination_format) ||
        !ToInternalSampling(options.sampling, &destination->filter) ||
        !ToInternalBorder(options.border, &destination->wrap)) {
        return Status::kInvalidArgument;
    }
    std::copy(options.mean.begin(), options.mean.end(), destination->mean);
    std::copy(options.scale.begin(), options.scale.end(), destination->scale);
    destination->preserve_identity_float_order =
      options.preserve_identity_float_order;
    destination->backend_preference = options.backend_preference;
    return Status::kOk;
}

Status ToPublicStatus(TaskStatus status) noexcept {
    switch (status) {
        case SUCCESS: return Status::kOk;
        case INPUT_DATA_ERROR: return Status::kInvalidArgument;
        case UNSUPPORTED_SAMPLER: return Status::kUnsupportedSampling;
        case UNSUPPORTED_CONVERSION: return Status::kUnsupportedConversion;
        case UNSUPPORTED_FLOAT_CONVERSION:
            return Status::kUnsupportedElementType;
        case ACCELERATION_DISABLED: return Status::kAccelerationDisabled;
        case ACCELERATION_UNAVAILABLE: return Status::kAccelerationUnavailable;
        case ACCELERATION_FAILURE: return Status::kAccelerationFailure;
    }
    return Status::kInvalidArgument;
}

Matrix ToTaskMatrix(const TransformMatrix& transform) {
    float affine[6];
    affine[Matrix::kAScaleX] = transform[0];
    affine[Matrix::kASkewY] = transform[3];
    affine[Matrix::kASkewX] = transform[1];
    affine[Matrix::kAScaleY] = transform[4];
    affine[Matrix::kATransX] = transform[2];
    affine[Matrix::kATransY] = transform[5];
    Matrix result;
    result.setAffine(affine);
    return result;
}

}  // namespace api_internal
}  // namespace task
}  // namespace inspirecv
