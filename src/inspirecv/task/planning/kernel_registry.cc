#include "inspirecv/task/planning/kernel_registry.h"

#include "inspirecv/task/kernels/cpu/channel_ops.h"
#include "inspirecv/task/kernels/cpu/color_ops.h"
#include "inspirecv/task/kernels/cpu/sampling_ops.h"
#include "inspirecv/task/kernels/cpu/tensor_writers.h"
#include "inspirecv/task/kernels/cpu/yuv_ops.h"

namespace inspirecv {
namespace task {
namespace internal {

namespace channel = ::inspirecv::task::kernels::channel;
namespace color = ::inspirecv::task::kernels::color;
namespace sampling = ::inspirecv::task::kernels::sampling;
namespace tensor = ::inspirecv::task::kernels::tensor;
namespace yuv = ::inspirecv::task::kernels::yuv;

ConvertSpan FindColorConverter(StreamFormat source, StreamFormat destination) {
    // All YUV420 samplers produce the same temporary NV21-style pair layout.
    if (source == YUV_NV12 || source == YUV_I420) source = YUV_NV21;

    if (source == RGBA && destination == RGBA) return channel::CopyQuadPixels;
    if (source == RGBA && destination == BGRA) return channel::ReverseQuadColor;
    if (source == RGBA && destination == BGR) return channel::ReverseAndDropAlpha;
    if (source == RGBA && destination == RGB) return channel::DropAlpha;
    if (source == RGBA && destination == GRAY) return channel::LumaFromRgba;

    if (source == BGRA && destination == RGBA) return channel::ReverseQuadColor;
    if (source == BGRA && destination == BGRA) return channel::CopyQuadPixels;
    if (source == BGRA && destination == BGR) return channel::DropAlpha;
    if (source == BGRA && destination == RGB) return channel::ReverseAndDropAlpha;
    if (source == BGRA && destination == GRAY) return channel::LumaFromBgra;

    if (source == RGB && destination == RGB) return channel::CopyTriplePixels;
    if (source == RGB && destination == BGR) return channel::ReverseTriple;
    if (source == RGB && destination == GRAY) return channel::LumaFromRgb;
    if (source == RGB && destination == RGBA) return channel::AppendOpaqueAlpha;
    if (source == RGB && destination == YCrCb) return color::RgbToYCrCb;
    if (source == RGB && destination == YUV) return color::RgbToYuv;
    if (source == RGB && destination == XYZ) return color::RgbToXyz;
    if (source == RGB && destination == HSV) return color::RgbToHsv;
    if (source == RGB && destination == BGR555) return color::RgbToBgr555;
    if (source == RGB && destination == BGR565) return color::RgbToBgr565;
    if (source == RGB && destination == HSV_FULL) return color::RgbToHsvFull;

    if (source == BGR && destination == BGR) return channel::CopyTriplePixels;
    if (source == BGR && destination == RGB) return channel::ReverseTriple;
    if (source == BGR && destination == GRAY) return channel::LumaFromBgr;
    if (source == BGR && destination == BGRA) return channel::AppendOpaqueAlpha;
    if (source == BGR && destination == YCrCb) return color::BgrToYCrCb;
    if (source == BGR && destination == YUV) return color::BgrToYuv;
    if (source == BGR && destination == XYZ) return color::BgrToXyz;
    if (source == BGR && destination == HSV) return color::BgrToHsv;
    if (source == BGR && destination == BGR555) return color::BgrToBgr555;
    if (source == BGR && destination == BGR565) return color::BgrToBgr565;
    if (source == BGR && destination == HSV_FULL) return color::BgrToHsvFull;

    if (source == GRAY && destination == RGBA) return channel::ReplicateMonoToQuad;
    if (source == GRAY && destination == BGRA) return channel::ReplicateMonoToQuad;
    if (source == GRAY && destination == BGR) return channel::ReplicateMonoToTriple;
    if (source == GRAY && destination == RGB) return channel::ReplicateMonoToTriple;
    if (source == GRAY && destination == GRAY) return channel::CopyMonoPixels;

    if (source == YUV_NV21 && destination == GRAY) return channel::CopyMonoPixels;
    if (source == YUV_NV21 && destination == RGB) return yuv::ToRgb;
    if (source == YUV_NV21 && destination == BGR) return yuv::ToBgr;
    if (source == YUV_NV21 && destination == RGBA) return yuv::ToRgba;
    if (source == YUV_NV21 && destination == BGRA) return yuv::ToBgra;
    return nullptr;
}

ConvertSpan FindDrawWriter(int pixel_bytes) {
    switch (pixel_bytes) {
        case 1:
            return channel::FillMonoPixels;
        case 3:
            return channel::FillTriplePixels;
        case 4:
            return channel::FillQuadPixels;
        default:
            return nullptr;
    }
}

SampleSpan FindSampler(StreamFormat format, Filter filter, bool direct_sampling) {
    if (direct_sampling) {
        switch (format) {
            case RGBA:
            case BGRA:
                return sampling::DirectQuad;
            case GRAY:
                return sampling::DirectMono;
            case RGB:
            case BGR:
                return sampling::DirectTriple;
            case YUV_NV21:
                return sampling::DirectNv21;
            case YUV_NV12:
                return sampling::DirectNv12;
            case YUV_I420:
                return sampling::DirectI420;
            default:
                break;
        }
    }

    if (filter == BILINEAR) {
        switch (format) {
            case RGBA:
            case BGRA:
                return sampling::BilinearQuad;
            case GRAY:
                return sampling::BilinearMono;
            case RGB:
            case BGR:
                return sampling::BilinearTriple;
            default:
                break;
        }
    }

    switch (format) {
        case RGBA:
        case BGRA:
            return sampling::NearestQuad;
        case GRAY:
            return sampling::NearestMono;
        case RGB:
        case BGR:
            return sampling::NearestTriple;
        case YUV_NV12:
            return sampling::NearestNv12;
        case YUV_NV21:
            return sampling::NearestNv21;
        case YUV_I420:
            return sampling::NearestI420;
        default:
            return nullptr;
    }
}

WriteInterleavedSpan FindInterleavedFloatWriter(StreamFormat format, int destination_channels) {
    if (destination_channels == 4) {
        switch (format) {
            case GRAY:
                return tensor::QuadFromMono;
            case RGBA:
            case BGRA:
                return tensor::InterleavedQuad;
            case RGB:
            case BGR:
                return tensor::QuadFromTriple;
            default:
                break;
        }
    }

    switch (format) {
        case GRAY:
            return tensor::InterleavedMono;
        case RGBA:
        case BGRA:
            return tensor::InterleavedQuad;
        case RGB:
        case BGR:
            return tensor::InterleavedTriple;
        default:
            return nullptr;
    }
}

WritePlanarSpan FindPlanarFloatWriter(StreamFormat format, int destination_channels) {
    if (destination_channels != 3) return nullptr;
    if (format == RGB || format == BGR) return tensor::PlanarTriple;
    return nullptr;
}

}  // namespace internal
}  // namespace task
}  // namespace inspirecv
