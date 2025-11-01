#include <algorithm>
#include <map>
#include <cmath>
#include <cstring>
#include <thread>
#include <vector>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif
#if defined(__x86_64__) || defined(__i386__)
extern "C" void task_avx2_memcpy(void* dst, const void* src, size_t n);
#endif
#include <inspirecv/task/core/stream_task_utils.h>
#include <inspirecv/task/core/st_defs.h>
#include <inspirecv/task/core/stream_task_kernels.h>
#include <inspirecv/task/cpu_features.h>

namespace inspirecv {
using namespace task; // allowed in source file for readability

#define CHECKFORMAT(src, dst, func) if (source == src && dest == dst) return func

struct StreamTaskUtils::InsideProperty {
    task::StreamTask::Config config;
    bool draw_ = false;
    halide_type_t dtype_;
    int stride_ = 0;
    int oc_, oh_, ow_, ic_, ih_, iw_;
    BLIT_FLOAT blit_float_ = nullptr;
    BLITTER blitter_ = nullptr;
    SAMPLER sampler_ = nullptr;
};

void StreamTaskUtils::Destroy(StreamTaskUtils* pro) {
    if (nullptr != pro) {
        delete pro;
    }
}

StreamTaskUtils::~StreamTaskUtils() {
    delete inside_;
}

StreamTaskUtils::StreamTaskUtils(const task::StreamTask::Config& config) {
    inside_         = new InsideProperty;
    inside_->config = config;
    for (int i = 0; i < 4; ++i) {
        inside_->config.mean[i]   = config.mean[i];
        inside_->config.normal[i] = config.normal[i];
    }
}

BLITTER StreamTaskUtils::Choose(StreamFormat source, StreamFormat dest) {
    // YUV only different in sampler
    if (source == YUV_NV12) {
        source = YUV_NV21;
    }
    if (source == YUV_I420) {
        source = YUV_NV21;
    }
    CHECKFORMAT(RGBA, RGBA, _TaskCopyC4);
    CHECKFORMAT(RGBA, BGRA, _TaskRGBAToBGRA);
    CHECKFORMAT(RGBA, BGR, _TaskRGBAToBGR);
    CHECKFORMAT(RGBA, RGB, _TaskBGRAToBGR);
    CHECKFORMAT(RGBA, GRAY, _TaskRGBAToGRAY);

    CHECKFORMAT(BGRA, RGBA, _TaskRGBAToBGRA);
    CHECKFORMAT(BGRA, BGRA, _TaskCopyC4);
    CHECKFORMAT(BGRA, BGR, _TaskBGRAToBGR);
    CHECKFORMAT(BGRA, RGB, _TaskRGBToBGR);
    CHECKFORMAT(BGRA, GRAY, _TaskBGRAToGRAY);

    CHECKFORMAT(RGB, RGB, _TaskCopyC3);
    CHECKFORMAT(RGB, BGR, _TaskRGBToBGR);
    CHECKFORMAT(RGB, GRAY, _TaskRGBToGRAY);
    CHECKFORMAT(RGB, RGBA, _TaskC3ToC4);
    CHECKFORMAT(RGB, YCrCb, _TaskRGBToCrCb);
    CHECKFORMAT(RGB, YUV, _TaskRGBToYUV);
    CHECKFORMAT(RGB, XYZ, _TaskRGBToXYZ);
    CHECKFORMAT(RGB, HSV, _TaskRGBToHSV);
    CHECKFORMAT(RGB, BGR555, _TaskRGBToBGR555);
    CHECKFORMAT(RGB, BGR565, _TaskRGBToBGR565);
    CHECKFORMAT(RGB, HSV_FULL, _TaskRGBToHSV_FULL);

    CHECKFORMAT(BGR, BGR, _TaskCopyC3);
    CHECKFORMAT(BGR, RGB, _TaskRGBToBGR);
    CHECKFORMAT(BGR, GRAY, _TaskBRGToGRAY);
    CHECKFORMAT(BGR, BGRA, _TaskC3ToC4);
    CHECKFORMAT(BGR, YCrCb, _TaskBGRToCrCb);
    CHECKFORMAT(BGR, YUV, _TaskBGRToYUV);
    CHECKFORMAT(BGR, XYZ, _TaskBGRToXYZ);
    CHECKFORMAT(BGR, HSV, _TaskBGRToHSV);
    CHECKFORMAT(BGR, BGR555, _TaskBGRToBGR555);
    CHECKFORMAT(BGR, BGR565, _TaskBGRToBGR565);
    CHECKFORMAT(BGR, HSV_FULL, _TaskBGRToHSV_FULL);

    CHECKFORMAT(GRAY, RGBA, _TaskGRAYToC4);
    CHECKFORMAT(GRAY, BGRA, _TaskGRAYToC4);
    CHECKFORMAT(GRAY, BGR, _TaskGRAYToC3);
    CHECKFORMAT(GRAY, RGB, _TaskGRAYToC3);
    CHECKFORMAT(GRAY, GRAY, _TaskCopyC1);

    CHECKFORMAT(YUV_NV21, GRAY, _TaskCopyC1);
    CHECKFORMAT(YUV_NV21, RGB, _TaskNV21ToRGB);
    CHECKFORMAT(YUV_NV21, BGR, _TaskNV21ToBGR);
    CHECKFORMAT(YUV_NV21, RGBA, _TaskNV21ToRGBA);
    CHECKFORMAT(YUV_NV21, BGRA, _TaskNV21ToBGRA);
    return nullptr;
}

BLITTER StreamTaskUtils::Choose(int channelByteSize) {
    switch (channelByteSize) {
        case 4:
            return _TaskC4blitH;
        case 3:
            return _TaskC3blitH;
        case 1:
            return _TaskC1blitH;
        default:
            return nullptr;
    }
}

SAMPLER StreamTaskUtils::Choose(StreamFormat format, Filter type, bool identity) {
    if (identity) {
        switch (format) {
            case RGBA:
            case BGRA:
                return _TaskSamplerC4Copy;
            case GRAY:
                return _TaskSamplerC1Copy;
            case RGB:
            case BGR:
                return _TaskSamplerC3Copy;
            case YUV_NV21:
                return _TaskSamplerNV21Copy;
            case YUV_NV12:
                return _TaskSamplerNV12Copy;
            case YUV_I420:
                return _TaskSamplerI420Copy;
            default:
                break;
        }
    }
    if (type == BILINEAR) {
        switch (format) {
            case RGBA:
            case BGRA:
                return _TaskSamplerC4Bilinear;
            case GRAY:
                return _TaskSamplerC1Bilinear;
            case RGB:
            case BGR:
                return _TaskSamplerC3Bilinear;
            default:
                break;
        }
    }

    // Nearest
    switch (format) {
        case RGBA:
        case BGRA:
            return _TaskSamplerC4Nearest;
        case GRAY:
            return _TaskSamplerC1Nearest;
        case RGB:
        case BGR:
            return _TaskSamplerC3Nearest;
        case YUV_NV12:
            return _TaskSamplerNV12Nearest;
        case YUV_NV21:
            return _TaskSamplerNV21Nearest;
        case YUV_I420:
            return _TaskSamplerI420Nearest;
        default:
            break;
    }
    INSPIRECV_TASK_PRINT("Don't support sampler for format:%d, type:%d", format, type);
    return nullptr;
}

BLIT_FLOAT StreamTaskUtils::Choose(StreamFormat format, int dstBpp) {
    if (4 == dstBpp) {
        switch (format) {
            case GRAY:
                return _TaskC1ToFloatRGBA;
            case RGBA:
            case BGRA:
                return _TaskC4ToFloatC4;
            case RGB:
            case BGR:
                return _TaskC3ToFloatRGBA;
            default:
                break;
        }
    }
    switch (format) {
        case GRAY:
            return _TaskC1ToFloatC1;
        case RGBA:
        case BGRA:
            return _TaskC4ToFloatC4;
        case RGB:
        case BGR:
            return _TaskC3ToFloatC3;
        default:
            break;
    }
    return nullptr;
}

TaskStatus StreamTaskUtils::SelectImageProcer(bool identity, bool /*hasBackend*/, bool isdraw) {
    if (isdraw) {
        inside_->blitter_ = Choose(inside_->ic_ * halide_type_bytes(inside_->dtype_));
        return SUCCESS;
    }
    // Choose sampler
    inside_->sampler_ = Choose(inside_->config.sourceFormat, inside_->config.filterType, identity);
    if (nullptr == inside_->sampler_) {
        return UNSUPPORTED_SAMPLER;
    }
    // Choose blitter for format convert
    if (inside_->config.sourceFormat != inside_->config.destFormat) {
        inside_->blitter_ = Choose(inside_->config.sourceFormat, inside_->config.destFormat);
        if (nullptr == inside_->blitter_) {
            return UNSUPPORTED_CONVERSION;
        }
    }
    // Choose float blitter
    if (inside_->dtype_.code == halide_type_float) {
        inside_->blit_float_ = StreamTaskUtils::Choose(inside_->config.destFormat, inside_->oc_);
        if (nullptr == inside_->blit_float_) {
            return UNSUPPORTED_FLOAT_CONVERSION;
        }
    }
    return SUCCESS;
}

TaskStatus StreamTaskUtils::ResizeFunc(int ic, int iw, int ih, int oc, int ow, int oh, halide_type_t type, int stride) {
    bool identity = transform_.isIdentity() && iw >= ow && ih >= oh;
    bool hasBackend = false;
    inside_->dtype_ = type;
    inside_->ow_ = ow;
    inside_->oh_ = oh;
    inside_->oc_ = oc;
    inside_->iw_ = iw;
    inside_->ih_ = ih;
    inside_->ic_ = ic;
    inside_->stride_ = stride;
    return SelectImageProcer(identity, hasBackend, inside_->draw_);
}

static constexpr int kLeft   = 1 << 0;
static constexpr int kRight  = 1 << 1;
static constexpr int kTop    = 1 << 2;
static constexpr int kBottom = 1 << 3;

inline static uint8_t _encode(const task::Point& p, int iw, int ih) {
    uint8_t mask = 0;
    if (p.fX < 0) mask |= kLeft;
    if (p.fX > iw - 1) mask |= kRight;
    if (p.fY < 0) mask |= kTop;
    if (p.fY > ih - 1) mask |= kBottom;
    return mask;
}

static std::pair<int, int> _computeClip(task::Point* points, int iw, int ih, const task::Matrix& invert, int xStart, int count) {
    auto code1 = _encode(points[0], iw, ih);
    auto code2 = _encode(points[1], iw, ih);
    int sta    = 0;
    int end    = count;

    float x1     = points[0].fX;
    float x2     = points[1].fX;
    float y1     = points[0].fY;
    float y2     = points[1].fY;
    int code     = 0;
    int pIndex   = 0;
    float deltaY = y2 - y1;
    float deltaX = x2 - x1;
    if (deltaX > 0.01f || deltaX < -0.01f) {
        deltaY = (y2 - y1) / (x2 - x1);
    } else {
        deltaY = 0;
    }
    if (deltaY > 0.01f || deltaY < -0.01f) {
        deltaX = (x2 - x1) / (y2 - y1);
    } else {
        deltaX = 0;
    }
    while (code1 != 0 || code2 != 0) {
        if ((code1 & code2) != 0) {
            sta = end;
            break;
        }
        if (code1 != 0) {
            code   = code1;
            pIndex = 0;
        } else if (code2 != 0) {
            code   = code2;
            pIndex = 1;
        }
        if ((kLeft & code) != 0) {
            points[pIndex].fY = points[pIndex].fY + deltaY * (0 - points[pIndex].fX);
            points[pIndex].fX = 0;
        } else if ((kRight & code) != 0) {
            points[pIndex].fY = points[pIndex].fY + deltaY * (iw - 1 - points[pIndex].fX);
            points[pIndex].fX = iw - 1;
        } else if ((kBottom & code) != 0) {
            points[pIndex].fX = points[pIndex].fX + deltaX * (ih - 1 - points[pIndex].fY);
            points[pIndex].fY = ih - 1;
        } else if ((kTop & code) != 0) {
            points[pIndex].fX = points[pIndex].fX + deltaX * (0 - points[pIndex].fY);
            points[pIndex].fY = 0;
        }
        auto tmp = invert.mapXY(points[pIndex].fX, points[pIndex].fY);
        if (0 == pIndex) {
            code1 = _encode(points[pIndex], iw, ih);
            sta = (int)::round(tmp.fX) - xStart;
        } else {
            code2 = _encode(points[pIndex], iw, ih);
            end = (int)::floor(tmp.fX) - xStart + 1;
        }
    }
    if (end > count) end = count;
    if (sta > end) sta = end;
    return std::make_pair(sta, end);
}

TaskStatus StreamTaskUtils::TransformImage(const uint8_t* source, uint8_t* dst, uint8_t* /*samplerDest*/, uint8_t* /*blitDest*/, int tileCount, int destBytes, const int32_t* regions) {
    static constexpr int kTileWidth = 256;
    if (inside_->stride_ == 0) {
        // Keep 0 for YUV so samplers can derive proper strides (iw / (iw, iw/2) per plane)
        if (!(inside_->config.sourceFormat == task::YUV_NV21 || inside_->config.sourceFormat == task::YUV_NV12 || inside_->config.sourceFormat == task::YUV_I420)) {
            inside_->stride_ = inside_->iw_ * inside_->ic_;
        }
    }
    auto workerRow = [&](int i) {
        task::Point points[2];
        uint8_t samplerLocal[4 * 256];
        uint8_t blitLocal[4 * 256];
        uint8_t* samplerPtr = samplerLocal;
        uint8_t* blitPtr    = blitLocal;
        int dy = inside_->draw_ ? regions[3 * i] : i;
        auto dstY = dst + dy * destBytes * inside_->ow_ * inside_->oc_;
        for (int tIndex = 0; tIndex < tileCount; ++tIndex) {
            int xStart    = tIndex * kTileWidth;
            int count     = std::min(kTileWidth, inside_->ow_ - xStart);
            if (inside_->draw_) {
                xStart = regions[3 * i + 1];
                count = regions[3 * i + 2] - xStart + 1;
            }
            auto dstStart = dstY + destBytes * inside_->oc_ * xStart;

            if (!inside_->blit_float_) {
                blitPtr = dstStart;
            }
            if (!inside_->blitter_) {
                samplerPtr = blitPtr;
            }

            if (!inside_->draw_) {
                points[0].fX = xStart;
                points[0].fY = dy;
                points[1].fX = xStart + count;
                points[1].fY = dy;
                transform_.mapPoints(points, 2);
                float deltaY = points[1].fY - points[0].fY;
                float deltaX = points[1].fX - points[0].fX;

                int sta = 0;
                int end = count;
                if (inside_->config.wrap == task::ZERO) {
                    auto clip    = _computeClip(points, inside_->iw_, inside_->ih_, transform_invert_, xStart, count);
                    sta          = clip.first;
                    end          = clip.second;
                    points[0].fX = sta + xStart;
                    points[0].fY = dy;
                    transform_.mapPoints(points, 1);
                    if (sta != 0 || end < count) {
                        if (inside_->ic_ > 0) {
                            if (sta > 0) {
                                ::memset(samplerPtr, padding_value_, inside_->ic_ * sta);
                            }
                            if (end < count) {
                                ::memset(samplerPtr + end * inside_->ic_, padding_value_, (count - end) * inside_->ic_);
                            }
                        } else {
                            ::memset(samplerPtr, padding_value_, count);
                            ::memset(samplerPtr + count, 128, ((count + 1) / 2) * 2);
                        }
                    }
                }
                points[1].fX = (deltaX) / (float)(count);
                points[1].fY = (deltaY) / (float)(count);
                inside_->sampler_(source, samplerPtr, points, sta, end - sta, count, inside_->iw_, inside_->ih_, inside_->stride_);
            }
            if (inside_->blitter_) {
                inside_->blitter_(samplerPtr, blitPtr, count);
            }
            if (inside_->blit_float_) {
                inside_->blit_float_(blitPtr, reinterpret_cast<float*>(dstStart), inside_->config.mean, inside_->config.normal, count);
            }
        }
    };

    // Parallelize rows if not in draw mode
    if (!inside_->draw_) {
        unsigned hw = std::thread::hardware_concurrency();
        if (hw == 0) hw = 4;
        unsigned threads = std::min<unsigned>(hw, (unsigned)std::max(1, inside_->oh_));
        if (threads > 1 && inside_->oh_ >= (int)threads) {
            std::vector<std::thread> pool;
            pool.reserve(threads);
            int rowsPer = (inside_->oh_ + (int)threads - 1) / (int)threads;
            for (unsigned ti = 0; ti < threads; ++ti) {
                int start = (int)ti * rowsPer;
                int end   = std::min(inside_->oh_, start + rowsPer);
                if (start >= end) break;
                pool.emplace_back([&, start, end](){
                    for (int i = start; i < end; ++i) workerRow(i);
                });
            }
            for (auto& th : pool) th.join();
            return SUCCESS;
        }
    }
    for (int i = 0; i < inside_->oh_; ++i) {
        workerRow(i);
    }
    return SUCCESS;
}

void StreamTaskUtils::SetMatrix(const task::Matrix& matrix) {
    transform_ = matrix;
    transform_.invert(&transform_invert_);
}

TaskStatus StreamTaskUtils::ExecFunc(const uint8_t *source, int stride, void *dest) {
    uint8_t sampleDest[4 * 256];
    uint8_t blitDest[4 * 256];
    int destBytes = halide_type_bytes(inside_->dtype_);
    // Fast path: identity + same format + uint8 -> direct memcpy
    if (transform_.isIdentity() &&
        inside_->config.sourceFormat == inside_->config.destFormat &&
        inside_->ic_ == inside_->oc_ &&
        destBytes == 1 && !inside_->draw_) {
        int srcStride = stride;
        if (srcStride == 0) srcStride = inside_->iw_ * inside_->ic_;
        const int rowBytes = inside_->ow_ * inside_->oc_;
        uint8_t* dstPtr = reinterpret_cast<uint8_t*>(dest);
        // If contiguous, copy whole plane at once
        if (inside_->ow_ == inside_->iw_ && srcStride == rowBytes) {
#if defined(__GNUC__) || defined(__clang__)
            const uint8_t* __restrict srcAl = reinterpret_cast<const uint8_t*>(__builtin_assume_aligned(source, 16));
            uint8_t* __restrict dstAl = reinterpret_cast<uint8_t*>(__builtin_assume_aligned(dstPtr, 16));
            size_t total = (size_t)rowBytes * (size_t)inside_->oh_;
#if defined(__x86_64__) || defined(__i386__)
            if (inspirecv::cpu_has_avx2()) { task_avx2_memcpy(dstAl, srcAl, total); return SUCCESS; }
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
            // Manual 64-byte unrolled copy to help some platforms reach peak bandwidth
            const uint8_t* srcp = srcAl;
            uint8_t* dstp = dstAl;
            // Align to 16B boundary for safety (assume_aligned hints already given)
            // Main 64B loop
            while (total >= 64) {
                uint8x16_t a0 = vld1q_u8(srcp + 0);
                uint8x16_t a1 = vld1q_u8(srcp + 16);
                uint8x16_t a2 = vld1q_u8(srcp + 32);
                uint8x16_t a3 = vld1q_u8(srcp + 48);
                vst1q_u8(dstp + 0, a0);
                vst1q_u8(dstp + 16, a1);
                vst1q_u8(dstp + 32, a2);
                vst1q_u8(dstp + 48, a3);
                srcp += 64; dstp += 64; total -= 64;
            }
            // 16B loop
            while (total >= 16) {
                uint8x16_t a0 = vld1q_u8(srcp);
                vst1q_u8(dstp, a0);
                srcp += 16; dstp += 16; total -= 16;
            }
            if (total) { ::memcpy(dstp, srcp, total); }
#else
            ::memcpy(dstAl, srcAl, total);
#endif
#else
            ::memcpy(dstPtr, source, (size_t)rowBytes * (size_t)inside_->oh_);
#endif
        } else {
            for (int y = 0; y < inside_->oh_; ++y) {
                const uint8_t* srcRow = source + y * srcStride;
                uint8_t* dstRow = dstPtr + y * rowBytes;
#if defined(__GNUC__) || defined(__clang__)
                __builtin_prefetch(srcRow + srcStride, 0, 1);
#endif
                ::memcpy(dstRow, srcRow, rowBytes);
            }
        }
        return SUCCESS;
    }
#ifdef INSPIRECV_TASK_DISABLE_TILING
    int tileCount = 1;
#else
    int tileCount = (inside_->ow_ + 256 - 1) / 256;
#endif
    if (inside_->draw_) {
        tileCount = 1;
    }
    return TransformImage(source, reinterpret_cast<uint8_t*>(dest), sampleDest, blitDest, tileCount, destBytes, nullptr);
}

void StreamTaskUtils::SetDraw() {
    if (inside_) {
        inside_->draw_ = true;
    }
}

void StreamTaskUtils::Draw(uint8_t* img, int w, int h, int c, const int* regions, int num, uint8_t* color) {
    uint8_t blitDest[4 * 256];
    int destBytes = halide_type_bytes(inside_->dtype_);
    inside_->oh_ = num;
    TransformImage(img, img, color, blitDest, 1, destBytes, regions);
}

} // namespace inspirecv



