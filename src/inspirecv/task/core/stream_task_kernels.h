#ifndef INSPIRECV_STREAMTASK_CORE_STREAM_TASK_KERNELS_H_
#define INSPIRECV_STREAMTASK_CORE_STREAM_TASK_KERNELS_H_

#include <inspirecv/task/core/stream_task.h>
#include <stdio.h>
#include <memory>

// blitter functions
void _TaskGRAYToC4(const unsigned char* source, unsigned char* dest, size_t count);
void _TaskGRAYToC3(const unsigned char* source, unsigned char* dest, size_t count);
void _TaskC3ToC4(const unsigned char* source, unsigned char* dest, size_t count);
void _TaskRGBAToBGRA(const unsigned char* source, unsigned char* dest, size_t count);
void _TaskRGBAToBGR(const unsigned char* source, unsigned char* dest, size_t count);
void _TaskRGBToBGR(const unsigned char* source, unsigned char* dest, size_t count);
void _TaskBGRAToBGR(const unsigned char* source, unsigned char* dest, size_t count);
void _TaskBGRAToGRAY(const unsigned char* source, unsigned char* dest, size_t count);
void _TaskRGBAToGRAY(const unsigned char* source, unsigned char* dest, size_t count);
void _TaskC3ToYUV(const unsigned char* source, unsigned char* dest, size_t count, bool bgr, bool yuv);
void _TaskC3ToXYZ(const unsigned char* source, unsigned char* dest, size_t count, bool bgr);
void _TaskC3ToHSV(const unsigned char* source, unsigned char* dest, size_t count, bool bgr, bool full);
void _TaskC3ToBGR555(const unsigned char* source, unsigned char* dest, size_t count, bool bgr);
void _TaskC3ToBGR565(const unsigned char* source, unsigned char* dest, size_t count, bool bgr);
void _TaskRGBToGRAY(const unsigned char* source, unsigned char* dest, size_t count);
void _TaskBRGToGRAY(const unsigned char* source, unsigned char* dest, size_t count);
void _TaskNV21ToRGBA(const unsigned char* source, unsigned char* dest, size_t count);
void _TaskNV21ToRGB(const unsigned char* source, unsigned char* dest, size_t count);
void _TaskNV21ToBGRA(const unsigned char* source, unsigned char* dest, size_t count);
void _TaskNV21ToBGR(const unsigned char* source, unsigned char* dest, size_t count);

// float blitter functions
void _TaskC1ToFloatC1(const unsigned char* source, float* dest, const float* mean, const float* normal, size_t count);
void _TaskC3ToFloatC3(const unsigned char* source, float* dest, const float* mean, const float* normal, size_t count);
void _TaskC4ToFloatC4(const unsigned char* source, float* dest, const float* mean, const float* normal, size_t count);
void _TaskC1ToFloatRGBA(const unsigned char* source, float* dest, const float* mean, const float* normal, size_t count);
void _TaskC3ToFloatRGBA(const unsigned char* source, float* dest, const float* mean, const float* normal, size_t count);

// simple blitter functions
inline void _TaskCopyC1(const unsigned char* source, unsigned char* dest, size_t count) {
    ::memcpy(dest, source, count * sizeof(unsigned char));
}
inline void _TaskCopyC4(const unsigned char* source, unsigned char* dest, size_t count) {
    ::memcpy(dest, source, 4 * count * sizeof(unsigned char));
}
inline void _TaskCopyC3(const unsigned char* source, unsigned char* dest, size_t count) {
    ::memcpy(dest, source, 3 * count * sizeof(unsigned char));
}
inline void _TaskRGBToCrCb(const unsigned char* source, unsigned char* dest, size_t count) {
    _TaskC3ToYUV(source, dest, count, false, false);
}
inline void _TaskBGRToCrCb(const unsigned char* source, unsigned char* dest, size_t count) {
    _TaskC3ToYUV(source, dest, count, true, false);
}
inline void _TaskRGBToYUV(const unsigned char* source, unsigned char* dest, size_t count) {
    _TaskC3ToYUV(source, dest, count, false, true);
}
inline void _TaskBGRToYUV(const unsigned char* source, unsigned char* dest, size_t count) {
    _TaskC3ToYUV(source, dest, count, true, true);
}
inline void _TaskRGBToXYZ(const unsigned char* source, unsigned char* dest, size_t count) {
    _TaskC3ToXYZ(source, dest, count, false);
}
inline void _TaskBGRToXYZ(const unsigned char* source, unsigned char* dest, size_t count) {
    _TaskC3ToXYZ(source, dest, count, true);
}
static void _TaskRGBToHSV(const unsigned char* source, unsigned char* dest, size_t count) {
    _TaskC3ToHSV(source, dest, count, false, false);
}
inline void _TaskBGRToHSV(const unsigned char* source, unsigned char* dest, size_t count) {
    _TaskC3ToHSV(source, dest, count, true, false);
}
static void _TaskRGBToHSV_FULL(const unsigned char* source, unsigned char* dest, size_t count) {
    _TaskC3ToHSV(source, dest, count, false, true);
}
inline void _TaskBGRToHSV_FULL(const unsigned char* source, unsigned char* dest, size_t count) {
    _TaskC3ToHSV(source, dest, count, true, true);
}
inline void _TaskRGBToBGR555(const unsigned char* source, unsigned char* dest, size_t count) {
    _TaskC3ToBGR555(source, dest, count, false);
}
inline void _TaskBGRToBGR555(const unsigned char* source, unsigned char* dest, size_t count) {
    _TaskC3ToBGR555(source, dest, count, true);
}
inline void _TaskRGBToBGR565(const unsigned char* source, unsigned char* dest, size_t count) {
    _TaskC3ToBGR565(source, dest, count, false);
}
inline void _TaskBGRToBGR565(const unsigned char* source, unsigned char* dest, size_t count) {
    _TaskC3ToBGR565(source, dest, count, true);
}

// sampler
void _TaskSamplerC4Bilinear(const unsigned char* source, unsigned char* dest, inspirecv::task::Point* points, size_t sta,
                          size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride);
void _TaskSamplerC3Bilinear(const unsigned char* source, unsigned char* dest, inspirecv::task::Point* points, size_t sta,
                          size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride);
void _TaskSamplerC1Bilinear(const unsigned char* source, unsigned char* dest, inspirecv::task::Point* points, size_t sta,
                          size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride);
void _TaskSamplerNearest(const unsigned char* source, unsigned char* dest, inspirecv::task::Point* points, size_t sta, size_t count,
                       size_t iw, size_t ih, size_t yStride, int bpp);
void _TaskSamplerC4Nearest(const unsigned char* source, unsigned char* dest, inspirecv::task::Point* points, size_t sta,
                         size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride);
void _TaskSamplerC1Nearest(const unsigned char* source, unsigned char* dest, inspirecv::task::Point* points, size_t sta,
                         size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride);
void _TaskSamplerC3Nearest(const unsigned char* source, unsigned char* dest, inspirecv::task::Point* points, size_t sta,
                         size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride);
void _TaskSamplerCopyCommon(const unsigned char* source, unsigned char* dest, inspirecv::task::Point* points, size_t sta,
                          size_t count, size_t iw, size_t ih, size_t yStride, int bpp);
inline void _TaskSamplerC1Copy(const unsigned char* source, unsigned char* dest, inspirecv::task::Point* points, size_t sta, size_t count,
                      size_t capacity, size_t iw, size_t ih, size_t yStride) {
    _TaskSamplerCopyCommon(source, dest, points, sta, count, iw, ih, yStride, 1);
}
inline void _TaskSamplerC3Copy(const unsigned char* source, unsigned char* dest, inspirecv::task::Point* points, size_t sta, size_t count,
                      size_t capacity, size_t iw, size_t ih, size_t yStride) {
    _TaskSamplerCopyCommon(source, dest, points, sta, count, iw, ih, yStride, 3);
}
inline void _TaskSamplerC4Copy(const unsigned char* source, unsigned char* dest, inspirecv::task::Point* points, size_t sta, size_t count,
                      size_t capacity, size_t iw, size_t ih, size_t yStride) {
    _TaskSamplerCopyCommon(source, dest, points, sta, count, iw, ih, yStride, 4);
}
void _TaskSamplerI420Copy(const unsigned char* source, unsigned char* dest, inspirecv::task::Point* points, size_t sta,
                        size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride);
void _TaskSamplerI420Nearest(const unsigned char* source, unsigned char* dest, inspirecv::task::Point* points, size_t sta,
                           size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride);
void _TaskSamplerNV21Copy(const unsigned char* source, unsigned char* dest, inspirecv::task::Point* points, size_t sta,
                        size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride);
void _TaskSamplerNV21Nearest(const unsigned char* source, unsigned char* dest, inspirecv::task::Point* points, size_t sta,
                           size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride);
void _TaskSamplerNV12Copy(const unsigned char* source, unsigned char* dest, inspirecv::task::Point* points, size_t sta,
                        size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride);
void _TaskSamplerNV12Nearest(const unsigned char* source, unsigned char* dest, inspirecv::task::Point* points, size_t sta,
                           size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride);

// draw blit
void _TaskC1blitH(const unsigned char* source, unsigned char* dest, size_t count);
void _TaskC3blitH(const unsigned char* source, unsigned char* dest, size_t count);
void _TaskC4blitH(const unsigned char* source, unsigned char* dest, size_t count);

#endif /* INSPIRECV_STREAMTASK_CORE_STREAM_TASK_KERNELS_H_ */


