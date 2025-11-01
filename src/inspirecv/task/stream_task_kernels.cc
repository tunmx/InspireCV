#include <inspirecv/task/core/stream_task_kernels.h>
#include <inspirecv/task/core/st_defs.h>
#include <inspirecv/task/cpu_features.h>
#include <algorithm>
#include <string.h>
#ifndef INSPIRECV_TASK_CLAMP_DECL
static inline float __clamp(float v, float minV, float maxV) {
    return v < minV ? minV : (v > maxV ? maxV : v);
}
#define INSPIRECV_TASK_CLAMP_DECL 1
#endif
#ifdef INSPIRECV_TASK_USE_NEON
#include <arm_neon.h>
#endif

// x86 SSE4.1 path (optional)
#ifdef INSPIRECV_TASK_USE_SSE
#if defined(__SSE4_1__)
#include <x86intrin.h>
// Macros from upstream SSE path
#define INSPIRECV_TASK_SSE_YUV_INIT \
countUnit -= 1;\
const auto c_6 = _mm_set1_epi16((1 << 6));\
const auto c_10 = _mm_set1_epi16((1 << 10));\
const auto c_73 = _mm_set1_epi16(73);\
const auto c_25 = _mm_set1_epi16(25);\
const auto c_37 = _mm_set1_epi16(37);\
const auto c_130 = _mm_set1_epi16(130);\
const auto c_128 = _mm_set1_epi16(128);\
const auto zero = _mm_set1_epi8(0);\
const auto alpha = _mm_set1_epi8(-1);\
const auto crossMask = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);\
const auto revertCrossMask = _mm_setr_epi8(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);

#define INSPIRECV_TASK_SSE_YUV_CONVERT \
auto Y_ = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(y + z * 16)), crossMask);\
auto UV = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(uv + z * 16)), crossMask);\
auto y0 = _mm_mullo_epi16(_mm_unpacklo_epi8(Y_, zero), c_6);\
auto y1 = _mm_mullo_epi16(_mm_unpackhi_epi8(Y_, zero), c_6);\
auto U_ = _mm_sub_epi16(_mm_unpackhi_epi8(UV, zero), c_128);\
auto V_ = _mm_sub_epi16(_mm_unpacklo_epi8(UV, zero), c_128);\
auto r0 = _mm_add_epi16(y0, _mm_mullo_epi16(V_, c_73));\
auto r1 = _mm_add_epi16(y1, _mm_mullo_epi16(V_, c_73));\
auto g0 = _mm_sub_epi16(_mm_sub_epi16(y0, _mm_mullo_epi16(U_, c_25)), _mm_mullo_epi16(V_, c_37));\
auto g1 = _mm_sub_epi16(_mm_sub_epi16(y1, _mm_mullo_epi16(U_, c_25)), _mm_mullo_epi16(V_, c_37));\
auto b0 = _mm_add_epi16(y0, _mm_mullo_epi16(U_, c_130));\
auto b1 = _mm_add_epi16(y1, _mm_mullo_epi16(U_, c_130));\
r0 = _mm_mulhi_epi16(r0, c_10);\
r1 = _mm_mulhi_epi16(r1, c_10);\
g0 = _mm_mulhi_epi16(g0, c_10);\
g1 = _mm_mulhi_epi16(g1, c_10);\
b0 = _mm_mulhi_epi16(b0, c_10);\
b1 = _mm_mulhi_epi16(b1, c_10);\
auto dR = _mm_packus_epi16(r0, r1);\
auto dG = _mm_packus_epi16(g0, g1);\
auto dB = _mm_packus_epi16(b0, b1);\
dR = _mm_shuffle_epi8(dR, revertCrossMask);\
dG = _mm_shuffle_epi8(dG, revertCrossMask);\
dB = _mm_shuffle_epi8(dB, revertCrossMask);\
auto RG0 = _mm_unpacklo_epi8(dR, dG);\
auto RG1 = _mm_unpackhi_epi8(dR, dG);\
auto BA0 = _mm_unpacklo_epi8(dB, alpha);\
auto BA1 = _mm_unpackhi_epi8(dB, alpha);\
auto RGBA0 = _mm_unpacklo_epi16(RG0, BA0);\
auto RGBA1 = _mm_unpackhi_epi16(RG0, BA0);\
auto RGBA2 = _mm_unpacklo_epi16(RG1, BA1);\
auto RGBA3 = _mm_unpackhi_epi16(RG1, BA1);
#endif
#endif
// SSE4.1 sampling helpers (defined outside macro block for visibility)
#ifdef INSPIRECV_TASK_USE_SSE
#if defined(__SSE4_1__)
static void _SSE_TaskSamplerNearest(const unsigned char* source, unsigned char* dest, inspirecv::task::Point* points, size_t sta, size_t count,
                       size_t iw, size_t ih, size_t yStride, int bpp) {
    dest = dest + bpp * sta;
    inspirecv::task::Point curPoints;
    curPoints.fX = points[0].fX;
    curPoints.fY = points[0].fY;
    float dy     = points[1].fY;
    float dx     = points[1].fX;
    float xMax   = iw - 1;
    float yMax   = ih - 1;
    int start = 0;
    int sizedQuad = (int)count / 4;
    if (sizedQuad > 0 && bpp == 4) {
        auto yStride4 = _mm_set1_epi32((int)yStride);
        auto varBpp   = _mm_set1_epi32(bpp);
        auto varZero  = _mm_set1_ps(0.f);
        auto plus  = _mm_set1_ps(0.5f);
        auto minus = _mm_set1_ps(-0.5f);
        auto xmax4 = _mm_set1_ps(xMax);
        auto ymax4 = _mm_set1_ps(yMax);
        for (int i = 0; i < sizedQuad; ++i) {
            auto cury4 = _mm_set_ps(curPoints.fY + 3 * dy, curPoints.fY + 2 * dy, curPoints.fY + dy, curPoints.fY);
            auto curx4 = _mm_set_ps(curPoints.fX + 3 * dx, curPoints.fX + 2 * dx, curPoints.fX + dx, curPoints.fX);
            cury4 = _mm_max_ps(cury4, varZero);
            curx4 = _mm_max_ps(curx4, varZero);
            cury4 = _mm_min_ps(cury4, ymax4);
            curx4 = _mm_min_ps(curx4, xmax4);
            auto x0 = _mm_cmplt_ps(curx4, varZero);
            auto y0 = _mm_cmplt_ps(cury4, varZero);
            x0 = _mm_blendv_ps(plus, minus, x0);
            y0 = _mm_blendv_ps(plus, minus, y0);
            curx4 = _mm_add_ps(curx4, x0);
            cury4 = _mm_add_ps(cury4, y0);
            auto ix0 = _mm_cvtps_epi32(_mm_round_ps(curx4, 3));
            auto iy0 = _mm_cvtps_epi32(_mm_round_ps(cury4, 3));
            curPoints.fY += 4 * dy;
            curPoints.fX += 4 * dx;
            auto sourcePos = _mm_add_epi32(_mm_mullo_epi32(iy0, yStride4), _mm_mullo_epi32(varBpp, ix0));
            int32_t pos4[4];
            _mm_store_si128((__m128i*)pos4, sourcePos);
            int iStart = 16 * i;
            *reinterpret_cast<int*>(dest + iStart)       = *reinterpret_cast<const int*>(source + pos4[0]);
            *reinterpret_cast<int*>(dest + iStart + 4)   = *reinterpret_cast<const int*>(source + pos4[1]);
            *reinterpret_cast<int*>(dest + iStart + 8)   = *reinterpret_cast<const int*>(source + pos4[2]);
            *reinterpret_cast<int*>(dest + iStart + 12)  = *reinterpret_cast<const int*>(source + pos4[3]);
        }
        start = sizedQuad * 4;
    }
    for (int i = start; i < (int)count; ++i) {
        int y = (int)roundf(__clamp(curPoints.fY, 0, yMax));
        int x = (int)roundf(__clamp(curPoints.fX, 0, xMax));
        curPoints.fY += dy;
        curPoints.fX += dx;
        auto sourcePos = (int)y * (int)yStride + bpp * x;
        for (int j = 0; j < bpp; ++j) {
            dest[bpp * i + j] = source[sourcePos + j];
        }
    }
}

static void _SSE_TaskSampleBilinear(const unsigned char* source, unsigned char* dest, inspirecv::task::Point* points, size_t count,
                            size_t iw, size_t ih, size_t yStride, size_t bpp) {
    float dy   = points[1].fY;
    float dx   = points[1].fX;
    float xMax = iw - 1;
    float yMax = ih - 1;
    inspirecv::task::Point curPoints;
    curPoints.fX = points[0].fX;
    curPoints.fY = points[0].fY;
    for (int i = 0; i < (int)count; ++i) {
        float y  = __clamp(curPoints.fY, 0, yMax);
        float x  = __clamp(curPoints.fX, 0, xMax);
        int y0   = (int)y;
        int x0   = (int)x;
        int y1   = (int)ceilf(y);
        int x1   = (int)ceilf(x);
        float xF = x - (float)x0;
        float yF = y - (float)y0;
        int index0 = y0 * (int)yStride + (int)bpp * x0;
        int index1 = y0 * (int)yStride + (int)bpp * x1;
        int index2 = y1 * (int)yStride + (int)bpp * x0;
        int index3 = y1 * (int)yStride + (int)bpp * x1;
        auto f0 = _mm_set1_ps((1.0f - xF) * (1.0f - yF));
        auto f1 = _mm_set1_ps(xF * (1.0f - yF));
        auto f2 = _mm_set1_ps(yF * (1.0f - xF));
        auto f3 = _mm_set1_ps(xF * yF);
        __m128i zero = _mm_set1_epi32(0);
        auto c00 = _mm_set_epi32(0, 0, 0, *(int32_t*)(source + index0));
        auto c01 = _mm_set_epi32(0, 0, 0, *(int32_t*)(source + index1));
        auto c10 = _mm_set_epi32(0, 0, 0, *(int32_t*)(source + index2));
        auto c11 = _mm_set_epi32(0, 0, 0, *(int32_t*)(source + index3));
        auto c00_16 = _mm_unpacklo_epi8(c00, zero);
        auto c01_16 = _mm_unpacklo_epi8(c01, zero);
        auto c10_16 = _mm_unpacklo_epi8(c10, zero);
        auto c11_16 = _mm_unpacklo_epi8(c11, zero);
        auto c00_32 = _mm_unpacklo_epi16(c00_16, zero);
        auto c01_32 = _mm_unpacklo_epi16(c01_16, zero);
        auto c10_32 = _mm_unpacklo_epi16(c10_16, zero);
        auto c11_32 = _mm_unpacklo_epi16(c11_16, zero);
        auto c00_f = _mm_cvtepi32_ps(c00_32);
        auto c01_f = _mm_cvtepi32_ps(c01_32);
        auto c10_f = _mm_cvtepi32_ps(c10_32);
        auto c11_f = _mm_cvtepi32_ps(c11_32);
        auto v0 = _mm_mul_ps(f0, c00_f);
        v0 = _mm_add_ps(v0, _mm_mul_ps(f1, c01_f));
        v0 = _mm_add_ps(v0, _mm_mul_ps(f2, c10_f));
        v0 = _mm_add_ps(v0, _mm_mul_ps(f3, c11_f));
        auto v0i = _mm_cvtps_epi32(_mm_round_ps(v0, 3));
        v0i = _mm_packs_epi32(v0i, v0i);
        v0i = _mm_packus_epi16(v0i, v0i);
        *(reinterpret_cast<int*>(dest) + i) = _mm_cvtsi128_si32(v0i);
        curPoints.fY += dy;
        curPoints.fX += dx;
    }
}
#endif
#endif
// Only declare NEON externs when NEON enabled
#ifdef INSPIRECV_TASK_USE_NEON
extern "C" {
void _TaskNV21ToRGBUnit(const unsigned char* source, unsigned char* dest, size_t countDiv8, const unsigned char* uv);
void _TaskNV21ToBGRUnit(const unsigned char* source, unsigned char* dest, size_t countDiv8, const unsigned char* uv);
void _TaskNV21ToRGBAUnit(const unsigned char* source, unsigned char* dest, size_t countDiv8, const unsigned char* uv);
void _TaskNV21ToBGRAUnit(const unsigned char* source, unsigned char* dest, size_t countDiv8, const unsigned char* uv);
void _TaskSamplerC4BilinearOpt(const unsigned char* source, unsigned char* dest, float* points, size_t count, size_t xMax, size_t yMax, size_t yStride);
void _TaskSamplerC1BilinearOpt(const unsigned char* source, unsigned char* dest, float* points, size_t count, size_t xMax, size_t yMax, size_t yStride);
void _TaskSamplerC4NearestOpt(const unsigned char* source, unsigned char* dest, float* points, size_t count, size_t iw, size_t ih, size_t yStride);
void _TaskSamplerC1NearestOpt(const unsigned char* source, unsigned char* dest, float* points, size_t count, size_t iw, size_t ih, size_t yStride);
void _TaskBlitC1ToFloatRGBA(const unsigned char* source, float* dest, const float* mean, const float* normal, size_t count);
void _TaskBlitC3ToFloatRGBA(const unsigned char* source, float* dest, const float* mean, const float* normal, size_t count);
}
#endif

// Below are the function bodies copied and adapted from the upstream file,
// with includes redirected to this folder.

void _TaskGRAYToC4(const unsigned char* source, unsigned char* dest, size_t count) {
    int sta = 0;
#ifdef INSPIRECV_TASK_USE_NEON
    int countD8 = (int)count / 8;
    if (countD8 > 0) {
        for (int i = 0; i < countD8; ++i) {
            auto gray = vld1_u8(source + 8 * i);

            uint8x8x4_t rgba;
            rgba.val[0] = gray;
            rgba.val[1] = gray;
            rgba.val[2] = gray;
            rgba.val[3] = vdup_n_u8(255);
            vst4_u8(dest + 32 * i, rgba);
        }
        sta = countD8 * 8;
    }
#endif
    for (int i = sta; i < count; ++i) {
        dest[4 * i + 0] = source[i];
        dest[4 * i + 1] = source[i];
        dest[4 * i + 2] = source[i];
        dest[4 * i + 3] = 255;
    }
}

void _TaskGRAYToC3(const unsigned char* source, unsigned char* dest, size_t count) {
    int sta = 0;
#ifdef INSPIRECV_TASK_USE_NEON
    int countD8 = (int)count / 8;
    if (countD8 > 0) {
        for (int i = 0; i < countD8; ++i) {
            auto gray = vld1_u8(source + 8 * i);

            uint8x8x3_t rgba;
            rgba.val[0] = gray;
            rgba.val[1] = gray;
            rgba.val[2] = gray;
            vst3_u8(dest + 24 * i, rgba);
        }
        sta = countD8 * 8;
    }
#endif
    for (int i = sta; i < count; ++i) {
        dest[3 * i + 0] = source[i];
        dest[3 * i + 1] = source[i];
        dest[3 * i + 2] = source[i];
    }
}

void _TaskC3ToC4(const unsigned char* source, unsigned char* dest, size_t count) {
    int sta = 0;
#ifdef INSPIRECV_TASK_USE_NEON
    int countD8 = (int)count / 8;
    if (countD8 > 0) {
        for (int i = 0; i < countD8; ++i) {
            uint8x8x3_t c3 = vld3_u8(source + 24 * i);

            uint8x8x4_t c4;
            c4.val[0] = c3.val[0];
            c4.val[1] = c3.val[1];
            c4.val[2] = c3.val[2];
            c4.val[3] = vdup_n_u8(255);
            vst4_u8(dest + 32 * i, c4);
        }
        sta = countD8 * 8;
    }
#endif
    for (int i = sta; i < count; i++) {
        dest[i * 4 + 0] = source[i * 3 + 0];
        dest[i * 4 + 1] = source[i * 3 + 1];
        dest[i * 4 + 2] = source[i * 3 + 2];
        dest[i * 4 + 3] = 255;
    }
}

void _TaskRGBAToBGRA(const unsigned char* source, unsigned char* dest, size_t count) {
    int sta = 0;
#ifdef INSPIRECV_TASK_USE_SSE
#if defined(__SSE4_1__)
    if (inspirecv::cpu_has_sse41()) {
        int countD4 = (int)count / 4;
        if (countD4 > 0) {
            const auto swapRB = _mm_setr_epi8(2,1,0,3, 6,5,4,7, 10,9,8,11, 14,13,12,15);
            for (int i = 0; i < countD4; ++i) {
                auto rgba = _mm_loadu_si128((const __m128i*)(source + 16 * i));
                auto bgra = _mm_shuffle_epi8(rgba, swapRB);
                _mm_storeu_si128((__m128i*)(dest + 16 * i), bgra);
            }
            sta = countD4 * 4;
        }
    }
#endif
#endif
#ifdef INSPIRECV_TASK_USE_NEON
    int countD8 = (int)count / 8;
    if (countD8 > 0) {
        for (int i = 0; i < countD8; ++i) {
            uint8x8x4_t rgba = vld4_u8(source + 32 * i);
            auto t           = rgba.val[0];
            rgba.val[0]      = rgba.val[2];
            rgba.val[2]      = t;
            vst4_u8(dest + 32 * i, rgba);
        }
        sta = countD8 * 8;
    }
#endif
    for (int i = sta; i < count; ++i) {
        dest[4 * i + 0] = source[4 * i + 2];
        dest[4 * i + 1] = source[4 * i + 1];
        dest[4 * i + 2] = source[4 * i + 0];
        dest[4 * i + 3] = source[4 * i + 3];
    }
}

void _TaskRGBAToBGR(const unsigned char* source, unsigned char* dest, size_t count) {
    int sta = 0;
#ifdef INSPIRECV_TASK_USE_NEON
    int countD8 = (int)count / 8;
    if (countD8 > 0) {
        for (int i = 0; i < countD8; ++i) {
            uint8x8x4_t rgba = vld4_u8(source + 32 * i);

            uint8x8x3_t bgr;
            bgr.val[0] = rgba.val[2];
            bgr.val[1] = rgba.val[1];
            bgr.val[2] = rgba.val[0];
            vst3_u8(dest + 24 * i, bgr);
        }
        sta = countD8 * 8;
    }
#endif
    for (int i = sta; i < count; ++i) {
        dest[3 * i + 0] = source[4 * i + 2];
        dest[3 * i + 1] = source[4 * i + 1];
        dest[3 * i + 2] = source[4 * i + 0];
    }
}

void _TaskRGBToBGR(const unsigned char* source, unsigned char* dest, size_t count) {
    int sta = 0;
#ifdef INSPIRECV_TASK_USE_NEON
    int countD8 = (int)count / 8;
    if (countD8 > 0) {
        for (int i = 0; i < countD8; ++i) {
            uint8x8x3_t rgba = vld3_u8(source + 24 * i);
            uint8x8x3_t bgr;
            bgr.val[0] = rgba.val[2];
            bgr.val[1] = rgba.val[1];
            bgr.val[2] = rgba.val[0];
            vst3_u8(dest + 24 * i, bgr);
        }
        sta = countD8 * 8;
    }
#endif
    for (int i = sta; i < count; ++i) {
        dest[3 * i + 0] = source[3 * i + 2];
        dest[3 * i + 1] = source[3 * i + 1];
        dest[3 * i + 2] = source[3 * i + 0];
    }
}

void _TaskBGRAToBGR(const unsigned char* source, unsigned char* dest, size_t count) {
    int sta = 0;
#ifdef INSPIRECV_TASK_USE_NEON
    int countD8 = (int)count / 8;
    if (countD8 > 0) {
        for (int i = 0; i < countD8; ++i) {
            uint8x8x4_t bgra = vld4_u8(source + 32 * i);

            uint8x8x3_t bgr;
            bgr.val[0] = bgra.val[0];
            bgr.val[1] = bgra.val[1];
            bgr.val[2] = bgra.val[2];
            vst3_u8(dest + 24 * i, bgr);
        }
        sta = countD8 * 8;
    }
#endif
    for (int i = sta; i < count; ++i) {
        dest[3 * i + 0] = source[4 * i + 0];
        dest[3 * i + 1] = source[4 * i + 1];
        dest[3 * i + 2] = source[4 * i + 2];
    }
}

void _TaskBGRAToGRAY(const unsigned char* source, unsigned char* dest, size_t count) {
    int sta = 0;
    for (int i = sta; i < count; ++i) {
        int r = source[4 * i + 2];
        int g = source[4 * i + 1];
        int b = source[4 * i + 0];
        int y = (19 * r + 38 * g + 7 * b) >> 6;
        dest[i] = y;
    }
}

void _TaskRGBAToGRAY(const unsigned char* source, unsigned char* dest, size_t count) {
    int sta = 0;
    for (int i = sta; i < count; ++i) {
        int r = source[4 * i + 0];
        int g = source[4 * i + 1];
        int b = source[4 * i + 2];
        int y = (19 * r + 38 * g + 7 * b) >> 6;
        dest[i] = y;
    }
}

uint8_t saturate_cast(int v) { return (uint8_t)((unsigned)v <= 255 ? v : v > 0 ? 255 : 0); }
#define CV_DESCALE(x,n)     (((x) + (1 << ((n)-1))) >> (n))

void _TaskC3ToYUV(const unsigned char* source, unsigned char* dest, size_t count, bool bgr, bool yuv) {
    static const int coeffs[] = { 4899, 9617, 1868, 8192, -6860, -1332, -2765, -5427, 8192, -2412, -4734, 7146, 10076, -8438, -1638 };
    int r0 = 0, r1 = 3, r2 = 6, g0 = 1, g1 = 4, g2 = 7, b0 = 2, b1 = 5, b2 = 8;
    if (yuv) { r1 = 9; r2 = 12; g1 = 10; g2 = 13; b1 = 11; b2 = 14; }
    if (bgr) { std::swap(r0, b0); std::swap(r1, b1); std::swap(r2, b2); }
    int C0 = coeffs[r0], C1 = coeffs[g0], C2 = coeffs[b0], C3 = coeffs[r1], C4 = coeffs[g1], C5 = coeffs[b1], C6 = coeffs[r2], C7 = coeffs[g2], C8 = coeffs[b2];
    int sta = 0;
    for (int i = sta; i < count; ++i) {
        int r = source[3 * i + 0];
        int g = source[3 * i + 1];
        int b = source[3 * i + 2];
        int y = CV_DESCALE(r*C0 + g*C1 + b*C2, 14);
        int u = CV_DESCALE(r*C3 + g*C4 + b*C5, 14) + 128;
        int v = CV_DESCALE(r*C6 + g*C7 + b*C8, 14) + 128;
        dest[3 * i + 0] = y;
        dest[3 * i + 1] = u;
        dest[3 * i + 2] = v;
    }
}

void _TaskC3ToXYZ(const unsigned char* source, unsigned char* dest, size_t count, bool bgr) {
    static const int coeffs[] = { 1689, 1465, 739, 871, 2929, 296, 79, 488, 3892 };
    int r0 = 0, r1 = 3, r2 = 6, b0 = 2, b1 = 5, b2 = 8;
    if (bgr) { std::swap(r0, b0); std::swap(r1, b1); std::swap(r2, b2); }
    int C0 = coeffs[r0], C1 = coeffs[1], C2 = coeffs[b0], C3 = coeffs[r1], C4 = coeffs[4], C5 = coeffs[b1], C6 = coeffs[r2], C7 = coeffs[7], C8 = coeffs[b2];
    int sta = 0;
    for (int i = sta; i < count; ++i) {
        int r = source[3 * i + 0];
        int g = source[3 * i + 1];
        int b = source[3 * i + 2];
        int x = CV_DESCALE(r*C0 + g*C1 + b*C2, 12);
        int y = CV_DESCALE(r*C3 + g*C4 + b*C5, 12);
        int z = CV_DESCALE(r*C6 + g*C7 + b*C8, 12);
        dest[3 * i + 0] = saturate_cast(x);
        dest[3 * i + 1] = saturate_cast(y);
        dest[3 * i + 2] = saturate_cast(z);
    }
}

void _TaskC3ToHSV(const unsigned char* source, unsigned char* dest, size_t count, bool bgr, bool full) {
    int hrange = full ? 256 : 180;
    for (int i = 0; i < count; ++i) {
        int r = source[3 * i + 0];
        int g = source[3 * i + 1];
        int b = source[3 * i + 2];
        if (bgr) std::swap(r, b);
        int h, s, v = b;
        int vmin = std::min({r, g, b});
        v = std::max({r, g, b});
        uint8_t diff = saturate_cast(v - vmin);
        int vr = v == r ? -1 : 0;
        int vg = v == g ? -1 : 0;
        s = (int(diff * (255 << 12) * (1.0f/(float)v)) + (1 << (11))) >> 12;
        h = (vr & (g - b)) + (~vr & ((vg & (b - r + 2 * diff)) + ((~vg) & (r - g + 4 * diff))));
        h = ((h * int((hrange << 12)/(6.f*diff) + 0.5)) + (1 << (11))) >> 12;
        h += h < 0 ? hrange : 0;
        dest[3 * i + 0] = saturate_cast(h);
        dest[3 * i + 1] = s;
        dest[3 * i + 2] = v;
    }
}

void _TaskC3ToBGR555(const unsigned char* source, unsigned char* dest, size_t count, bool bgr) {
    for (int i = 0; i < count; ++i) {
        int r = source[3 * i + 0];
        int g = source[3 * i + 1];
        int b = source[3 * i + 2];
        if (bgr) std::swap(r, b);
        reinterpret_cast<unsigned short*>(dest)[i] = (b >> 3)|((g & ~7) << 2)|((r & ~7) << 7);
    }
}

void _TaskC3ToBGR565(const unsigned char* source, unsigned char* dest, size_t count, bool bgr) {
    for (int i = 0; i < count; ++i) {
        int r = source[3 * i + 0];
        int g = source[3 * i + 1];
        int b = source[3 * i + 2];
        if (bgr) std::swap(r, b);
        reinterpret_cast<unsigned short*>(dest)[i] = (b >> 3)|((g&~3) << 3)|((r&~7) << 8);
    }
}

void _TaskRGBToGRAY(const unsigned char* source, unsigned char* dest, size_t count) {
    int sta = 0;
#ifdef INSPIRECV_TASK_USE_NEON
    int countD8 = (int)count / 8;
    if (countD8 > 0) {
        auto rC = vdup_n_u8(19);
        auto gC = vdup_n_u8(38);
        auto bC = vdup_n_u8(7);
        for (int i = 0; i < countD8; ++i) {
            auto rgb   = vld3_u8(source + 24 * i);
            auto res   = vmull_u8(rC, rgb.val[0]) + vmull_u8(gC, rgb.val[1]) + vmull_u8(bC, rgb.val[2]);
            auto resU8 = vshrn_n_u16(res, 6);
            vst1_u8(dest + 8 * i, resU8);
        }
        sta = countD8 * 8;
    }
#endif
    for (int i = sta; i < count; ++i) {
        int r = source[3 * i + 0];
        int g = source[3 * i + 1];
        int b = source[3 * i + 2];
        int y = (19 * r + 38 * g + 7 * b) >> 6;
        dest[i] = y;
    }
}

void _TaskBRGToGRAY(const unsigned char* source, unsigned char* dest, size_t count) {
    int sta = 0;
#ifdef INSPIRECV_TASK_USE_NEON
    int countD8 = (int)count / 8;
    if (countD8 > 0) {
        auto rC = vdup_n_u8(19);
        auto gC = vdup_n_u8(38);
        auto bC = vdup_n_u8(7);
        for (int i = 0; i < countD8; ++i) {
            auto rgb   = vld3_u8(source + 24 * i);
            auto res   = vmull_u8(rC, rgb.val[2]) + vmull_u8(gC, rgb.val[1]) + vmull_u8(bC, rgb.val[0]);
            auto resU8 = vshrn_n_u16(res, 6);
            vst1_u8(dest + 8 * i, resU8);
        }
        sta = countD8 * 8;
    }
#endif
    for (int i = sta; i < count; ++i) {
        int r = source[3 * i + 2];
        int g = source[3 * i + 1];
        int b = source[3 * i + 0];
        int y = (19 * r + 38 * g + 7 * b) >> 6;
        dest[i] = y;
    }
}

void _TaskNV21ToRGBA(const unsigned char* source, unsigned char* dest, size_t count) {
    auto y   = source;
    auto uv  = source + count;
    auto dst = dest;
    int sta  = 0;
#ifdef INSPIRECV_TASK_USE_SSE
#if defined(__SSE4_1__)
    if (inspirecv::cpu_has_sse41()) {
        const int unit = 16;
        size_t countUnit = count / unit;
        if (countUnit > 0) {
            INSPIRECV_TASK_SSE_YUV_INIT;
            for (int z=0; z<(int)countUnit; ++z) {
                INSPIRECV_TASK_SSE_YUV_CONVERT;
                _mm_storeu_si128((__m128i*)(dst + 64 * z + 16 * 0), RGBA0);
                _mm_storeu_si128((__m128i*)(dst + 64 * z + 16 * 1), RGBA1);
                _mm_storeu_si128((__m128i*)(dst + 64 * z + 16 * 2), RGBA2);
                _mm_storeu_si128((__m128i*)(dst + 64 * z + 16 * 3), RGBA3);
            }
            sta = (int)countUnit * unit;
        }
    }
#endif
#endif
#ifdef INSPIRECV_TASK_USE_NEON
    const int unit   = 16;
    size_t countDiv8 = count / unit;
    if (countDiv8 > 0) {
        _TaskNV21ToRGBAUnit(source, dest, countDiv8, uv);
        sta = (int)countDiv8 * unit;
    }
#endif
    for (int i = sta; i < count; ++i) {
        int Y = y[i];
        int U = (int)uv[(i / 2) * 2 + 1] - 128;
        int V = (int)uv[(i / 2) * 2 + 0] - 128;
        Y     = Y << 6;
        int R = (Y + 73 * V) >> 6;
        int G = (Y - 25 * U - 37 * V) >> 6;
        int B = (Y + 130 * U) >> 6;
        R = std::min(std::max(R, 0), 255);
        G = std::min(std::max(G, 0), 255);
        B = std::min(std::max(B, 0), 255);
        dst[4 * i + 0] = (uint8_t)R;
        dst[4 * i + 1] = (uint8_t)G;
        dst[4 * i + 2] = (uint8_t)B;
        dst[4 * i + 3] = 255;
    }
}

void _TaskNV21ToRGB(const unsigned char* source, unsigned char* dest, size_t count) {
    auto y   = source;
    auto uv  = source + count;
    auto dst = dest;
    int sta  = 0;
#ifdef INSPIRECV_TASK_USE_SSE
#if defined(__SSE4_1__)
    if (inspirecv::cpu_has_sse41()) {
        const int unit = 16;
        size_t countUnit = count / unit;
        if (countUnit > 1) {
            countUnit -= 1;
            INSPIRECV_TASK_SSE_YUV_INIT;
            const auto rgbSelect = _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1);
            for (int z=0; z<(int)countUnit; ++z) {
                INSPIRECV_TASK_SSE_YUV_CONVERT;
                _mm_storeu_si128((__m128i*)(dst + 48 * z + 12 * 0), _mm_shuffle_epi8(RGBA0, rgbSelect));
                _mm_storeu_si128((__m128i*)(dst + 48 * z + 12 * 1), _mm_shuffle_epi8(RGBA1, rgbSelect));
                _mm_storeu_si128((__m128i*)(dst + 48 * z + 12 * 2), _mm_shuffle_epi8(RGBA2, rgbSelect));
                _mm_storeu_si128((__m128i*)(dst + 48 * z + 12 * 3), _mm_shuffle_epi8(RGBA3, rgbSelect));
            }
            sta = (int)countUnit * unit;
        }
    }
#endif
#endif
#ifdef INSPIRECV_TASK_USE_NEON
    const int unit   = 16;
    size_t countDiv8 = count / unit;
    if (countDiv8 > 0) {
        _TaskNV21ToRGBUnit(source, dest, countDiv8, uv);
        sta = (int)countDiv8 * unit;
    }
#endif
    for (int i = sta; i < count; ++i) {
        int Y = y[i];
        int U = (int)uv[(i / 2) * 2 + 1] - 128;
        int V = (int)uv[(i / 2) * 2 + 0] - 128;
        Y     = Y << 6;
        int R = (Y + 73 * V) >> 6;
        int G = (Y - 25 * U - 37 * V) >> 6;
        int B = (Y + 130 * U) >> 6;
        R = std::min(std::max(R, 0), 255);
        G = std::min(std::max(G, 0), 255);
        B = std::min(std::max(B, 0), 255);
        dst[3 * i + 0] = (uint8_t)R;
        dst[3 * i + 1] = (uint8_t)G;
        dst[3 * i + 2] = (uint8_t)B;
    }
}

void _TaskNV21ToBGRA(const unsigned char* source, unsigned char* dest, size_t count) {
    auto y   = source;
    auto uv  = source + count;
    auto dst = dest;
    int sta  = 0;
#ifdef INSPIRECV_TASK_USE_SSE
#if defined(__SSE4_1__)
    if (inspirecv::cpu_has_sse41()) {
        const int unit = 16;
        size_t countUnit = count / unit;
        if (countUnit > 0) {
            INSPIRECV_TASK_SSE_YUV_INIT;
            const auto rgbaSelect = _mm_setr_epi8(2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15);
            for (int z=0; z<(int)countUnit; ++z) {
                INSPIRECV_TASK_SSE_YUV_CONVERT;
                _mm_storeu_si128((__m128i*)(dst + 64 * z + 16 * 0), _mm_shuffle_epi8(RGBA0, rgbaSelect));
                _mm_storeu_si128((__m128i*)(dst + 64 * z + 16 * 1), _mm_shuffle_epi8(RGBA1, rgbaSelect));
                _mm_storeu_si128((__m128i*)(dst + 64 * z + 16 * 2), _mm_shuffle_epi8(RGBA2, rgbaSelect));
                _mm_storeu_si128((__m128i*)(dst + 64 * z + 16 * 3), _mm_shuffle_epi8(RGBA3, rgbaSelect));
            }
            sta = (int)countUnit * unit;
        }
    }
#endif
#endif
#ifdef INSPIRECV_TASK_USE_NEON
    const int unit   = 16;
    size_t countDiv8 = count / unit;
    if (countDiv8 > 0) {
        _TaskNV21ToBGRAUnit(source, dest, countDiv8, uv);
        sta = (int)countDiv8 * unit;
    }
#endif
    for (int i = sta; i < count; ++i) {
        int Y = y[i];
        int U = (int)uv[(i / 2) * 2 + 1] - 128;
        int V = (int)uv[(i / 2) * 2 + 0] - 128;
        Y     = Y << 6;
        int R = (Y + 73 * V) >> 6;
        int G = (Y - 25 * U - 37 * V) >> 6;
        int B = (Y + 130 * U) >> 6;
        R = std::min(std::max(R, 0), 255);
        G = std::min(std::max(G, 0), 255);
        B = std::min(std::max(B, 0), 255);
        dst[4 * i + 0] = (uint8_t)B;
        dst[4 * i + 1] = (uint8_t)G;
        dst[4 * i + 2] = (uint8_t)R;
        dst[4 * i + 3] = 255;
    }
}

void _TaskNV21ToBGR(const unsigned char* source, unsigned char* dest, size_t count) {
    auto y   = source;
    auto uv  = source + count;
    auto dst = dest;
    int sta  = 0;
#ifdef INSPIRECV_TASK_USE_SSE
#if defined(__SSE4_1__)
    if (inspirecv::cpu_has_sse41()) {
        const int unit = 16;
        size_t countUnit = count / unit;
        if (countUnit > 1) {
            countUnit -= 1;
            INSPIRECV_TASK_SSE_YUV_INIT;
            const auto rgbSelect = _mm_setr_epi8(2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, -1, -1, -1, -1);
            for (int z=0; z<(int)countUnit; ++z) {
                INSPIRECV_TASK_SSE_YUV_CONVERT;
                _mm_storeu_si128((__m128i*)(dst + 48 * z + 12 * 0), _mm_shuffle_epi8(RGBA0, rgbSelect));
                _mm_storeu_si128((__m128i*)(dst + 48 * z + 12 * 1), _mm_shuffle_epi8(RGBA1, rgbSelect));
                _mm_storeu_si128((__m128i*)(dst + 48 * z + 12 * 2), _mm_shuffle_epi8(RGBA2, rgbSelect));
                _mm_storeu_si128((__m128i*)(dst + 48 * z + 12 * 3), _mm_shuffle_epi8(RGBA3, rgbSelect));
            }
            sta = (int)countUnit * unit;
        }
    }
#endif
#endif
#ifdef INSPIRECV_TASK_USE_NEON
    const int unit   = 16;
    size_t countDiv8 = count / unit;
    if (countDiv8 > 0) {
        _TaskNV21ToBGRUnit(source, dest, countDiv8, uv);
        sta = (int)countDiv8 * unit;
    }
#endif
    for (int i = sta; i < count; ++i) {
        int Y = y[i];
        int U = (int)uv[(i / 2) * 2 + 1] - 128;
        int V = (int)uv[(i / 2) * 2 + 0] - 128;
        Y     = Y << 6;
        int R = (Y + 73 * V) >> 6;
        int G = (Y - 25 * U - 37 * V) >> 6;
        int B = (Y + 130 * U) >> 6;
        R = std::min(std::max(R, 0), 255);
        G = std::min(std::max(G, 0), 255);
        B = std::min(std::max(B, 0), 255);
        dst[3 * i + 0] = (uint8_t)B;
        dst[3 * i + 1] = (uint8_t)G;
        dst[3 * i + 2] = (uint8_t)R;
    }
}

void _TaskC1ToFloatC1(const unsigned char* source, float* dest, const float* mean, const float* normal, size_t count) {
#ifdef INSPIRECV_TASK_USE_SSE
#if defined(__SSE4_1__)
    {
        int remain = 0;
        int countC16 = (int)count / 16;
        remain = countC16 * 16;
        const auto meanC4 = _mm_set1_ps(mean[0]);
        const auto normalC4 = _mm_set1_ps(normal[0]);
        const __m128i l1 = _mm_setr_epi8(4,5,6,7, 6,5,4,7, 10,9,8,11, 14,13,12,15);
        const __m128i l2 = _mm_setr_epi8(8,9,10,11, 6,5,4,7, 10,9,8,11, 14,13,12,15);
        const __m128i l3 = _mm_setr_epi8(12,13,14,15, 6,5,4,7, 10,9,8,11, 14,13,12,15);
        for (int i=0; i<countC16; ++i) {
            auto srcInt8 = _mm_loadu_si128((const __m128i*)(source + i * 16));
            auto int3200 = _mm_cvtepu8_epi32(srcInt8);
            auto int3201 = _mm_cvtepu8_epi32(_mm_shuffle_epi8(srcInt8, l1));
            auto int3210 = _mm_cvtepu8_epi32(_mm_shuffle_epi8(srcInt8, l2));
            auto int3211 = _mm_cvtepu8_epi32(_mm_shuffle_epi8(srcInt8, l3));
            auto float00 = _mm_cvtepi32_ps(int3200);
            auto float01 = _mm_cvtepi32_ps(int3201);
            auto float10 = _mm_cvtepi32_ps(int3210);
            auto float11 = _mm_cvtepi32_ps(int3211);
            _mm_storeu_ps(dest + 16 * i + 4 * 0, _mm_mul_ps(_mm_sub_ps(float00, meanC4), normalC4));
            _mm_storeu_ps(dest + 16 * i + 4 * 1, _mm_mul_ps(_mm_sub_ps(float01, meanC4), normalC4));
            _mm_storeu_ps(dest + 16 * i + 4 * 2, _mm_mul_ps(_mm_sub_ps(float10, meanC4), normalC4));
            _mm_storeu_ps(dest + 16 * i + 4 * 3, _mm_mul_ps(_mm_sub_ps(float11, meanC4), normalC4));
        }
        for (int i = remain; i < (int)count; ++i) {
            dest[i + 0] = normal[0] * (source[i + 0] - mean[0]);
        }
        return;
    }
#endif
#endif
#ifdef INSPIRECV_TASK_USE_NEON
    unsigned long size  = count >> 4;
    float32x4_t cache   = vdupq_n_f32(0);
    float32x4_t _mean   = vdupq_n_f32(-mean[0]);
    float32x4_t _normal = vdupq_n_f32(normal[0]);
    for (int i = 0; i < size; i++, source += 16) {
        uint8x16_t v = vld1q_u8(source);
        int16x8_t vl = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v)));  // 0..7
        int16x8_t vh = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v))); // 8..15
        float32x4_t vll = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vl))); // 0..3
        cache           = vaddq_f32(_mean, vll);
        cache           = vmulq_f32(cache, _normal);
        vst1q_f32(dest, cache);
        dest += 4;
        float32x4_t vlh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vl))); // 4..7
        cache           = vaddq_f32(_mean, vlh);
        cache           = vmulq_f32(cache, _normal);
        vst1q_f32(dest, cache);
        dest += 4;
        float32x4_t vhl = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vh))); // 8..11
        cache           = vaddq_f32(_mean, vhl);
        cache           = vmulq_f32(cache, _normal);
        vst1q_f32(dest, cache);
        dest += 4;
        float32x4_t vhh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vh))); // 12..15
        cache           = vaddq_f32(_mean, vhh);
        cache           = vmulq_f32(cache, _normal);
        vst1q_f32(dest, cache);
        dest += 4;
    }
    int left = count & 15;
    if (left == 0) return;
    for (int i = 0; i < left; ++i, ++dest, ++source) {
        *dest = normal[0] * (*source - mean[0]);
    }
#else
    for (int i = 0; i < count; ++i) {
        dest[i + 0] = normal[0] * (source[i + 0] - mean[0]);
    }
#endif
}

void _TaskC3ToFloatC3(const unsigned char* source, float* dest, const float* mean, const float* normal, size_t count) {
#ifdef INSPIRECV_TASK_USE_SSE
#if defined(__SSE4_1__)
    {
        int remain = 0;
        int countC4 = (int)count / 4;
        if (countC4 > 1) {
            if (((int)count % 4) * 3 < 4) {
                countC4 -= 1;
            }
            auto alpha0 = _mm_setr_ps(normal[0], normal[1], normal[2], normal[0]);
            auto alpha1 = _mm_setr_ps(normal[1], normal[2], normal[0], normal[1]);
            auto alpha2 = _mm_setr_ps(normal[2], normal[0], normal[1], normal[2]);
            auto beta0 = _mm_setr_ps(mean[0], mean[1], mean[2], mean[0]);
            auto beta1 = _mm_setr_ps(mean[1], mean[2], mean[0], mean[1]);
            auto beta2 = _mm_setr_ps(mean[2], mean[0], mean[1], mean[2]);
            remain = countC4 * 4;
            const __m128i gM = _mm_setr_epi8(4, 5, 6, 7, 6,5,4,7, 10,9,8,11, 14,13,12,15);
            const __m128i bM = _mm_setr_epi8(8, 9,10,11, 6,5,4,7, 10,9,8,11, 14,13,12,15);
            for (int i = 0; i < countC4; ++i) {
                auto sInt8 = _mm_loadu_si128((const __m128i*)(source + 12 * i));
                auto s0 = _mm_cvtepu8_epi32(sInt8);
                auto s1 = _mm_cvtepu8_epi32(_mm_shuffle_epi8(sInt8, gM));
                auto s2 = _mm_cvtepu8_epi32(_mm_shuffle_epi8(sInt8, bM));
                auto f0 = _mm_cvtepi32_ps(s0);
                auto f1 = _mm_cvtepi32_ps(s1);
                auto f2 = _mm_cvtepi32_ps(s2);
                _mm_storeu_ps(dest + 12 * i + 4 * 0, _mm_mul_ps(_mm_sub_ps(f0, beta0), alpha0));
                _mm_storeu_ps(dest + 12 * i + 4 * 1, _mm_mul_ps(_mm_sub_ps(f1, beta1), alpha1));
                _mm_storeu_ps(dest + 12 * i + 4 * 2, _mm_mul_ps(_mm_sub_ps(f2, beta2), alpha2));
            }
        }
        for (int i = remain; i < (int)count; ++i) {
            dest[3 * i + 0] = normal[0] * (source[3 * i + 0] - mean[0]);
            dest[3 * i + 1] = normal[1] * (source[3 * i + 1] - mean[1]);
            dest[3 * i + 2] = normal[2] * (source[3 * i + 2] - mean[2]);
        }
        return;
    }
#endif
#endif
#ifdef INSPIRECV_TASK_USE_NEON
    int size = (int)count / 16;
    float32x4x3_t cachell = {vmovq_n_f32(0), vmovq_n_f32(0), vmovq_n_f32(0)};
    float32x4x3_t cachelh = {vmovq_n_f32(0), vmovq_n_f32(0), vmovq_n_f32(0)};
    float32x4x3_t cachehl = {vmovq_n_f32(0), vmovq_n_f32(0), vmovq_n_f32(0)};
    float32x4x3_t cachehh = {vmovq_n_f32(0), vmovq_n_f32(0), vmovq_n_f32(0)};
    float32x4x3_t _mean; float32x4x3_t _normal;
    for (int c = 0; c < 3; c++) { _mean.val[c] = vmovq_n_f32(-mean[c]); _normal.val[c] = vmovq_n_f32(normal[c]); }
    for (int i = 0; i < size; i++) {
        uint8x16x3_t v = vld3q_u8(source + 16 * 3 * i);
        int c = 0; {
            int16x8_t vl = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v.val[c])));
            float32x4_t vll = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vl)));
            cachell.val[c]  = vaddq_f32(_mean.val[c], vll);
            cachell.val[c]  = vmulq_f32(cachell.val[c], _normal.val[c]);
            float32x4_t vlh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vl)));
            cachelh.val[c]  = vaddq_f32(_mean.val[c], vlh);
            cachelh.val[c]  = vmulq_f32(cachelh.val[c], _normal.val[c]);
            int16x8_t vh = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v.val[c])));
            float32x4_t vhl = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vh)));
            cachehl.val[c]  = vaddq_f32(_mean.val[c], vhl);
            cachehl.val[c]  = vmulq_f32(cachehl.val[c], _normal.val[c]);
            float32x4_t vhh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vh)));
            cachehh.val[c]  = vaddq_f32(_mean.val[c], vhh);
            cachehh.val[c]  = vmulq_f32(cachehh.val[c], _normal.val[c]);
        }
        c = 1; {
            int16x8_t vl = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v.val[c])));
            float32x4_t vll = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vl)));
            cachell.val[c]  = vaddq_f32(_mean.val[c], vll);
            cachell.val[c]  = vmulq_f32(cachell.val[c], _normal.val[c]);
            float32x4_t vlh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vl)));
            cachelh.val[c]  = vaddq_f32(_mean.val[c], vlh);
            cachelh.val[c]  = vmulq_f32(cachelh.val[c], _normal.val[c]);
            int16x8_t vh = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v.val[c])));
            float32x4_t vhl = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vh)));
            cachehl.val[c]  = vaddq_f32(_mean.val[c], vhl);
            cachehl.val[c]  = vmulq_f32(cachehl.val[c], _normal.val[c]);
            float32x4_t vhh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vh)));
            cachehh.val[c]  = vaddq_f32(_mean.val[c], vhh);
            cachehh.val[c]  = vmulq_f32(cachehh.val[c], _normal.val[c]);
        }
        c = 2; {
            int16x8_t vl = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v.val[c])));
            float32x4_t vll = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vl)));
            cachell.val[c]  = vaddq_f32(_mean.val[c], vll);
            cachell.val[c]  = vmulq_f32(cachell.val[c], _normal.val[c]);
            float32x4_t vlh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vl)));
            cachelh.val[c]  = vaddq_f32(_mean.val[c], vlh);
            cachelh.val[c]  = vmulq_f32(cachelh.val[c], _normal.val[c]);
            int16x8_t vh = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v.val[c])));
            float32x4_t vhl = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vh)));
            cachehl.val[c]  = vaddq_f32(_mean.val[c], vhl);
            cachehl.val[c]  = vmulq_f32(cachehl.val[c], _normal.val[c]);
            float32x4_t vhh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vh)));
            cachehh.val[c]  = vaddq_f32(_mean.val[c], vhh);
            cachehh.val[c]  = vmulq_f32(cachehh.val[c], _normal.val[c]);
        }
        vst3q_f32(dest + 48 * i + 0 * 3, cachell);
        vst3q_f32(dest + 48 * i + 4 * 3, cachelh);
        vst3q_f32(dest + 48 * i + 8 * 3, cachehl);
        vst3q_f32(dest + 48 * i + 12 * 3, cachehh);
    }
    int remain = size * 16;
    for (int i = remain; i < count; i++) {
        dest[3 * i + 0] = normal[0] * (source[3 * i + 0] - mean[0]);
        dest[3 * i + 1] = normal[1] * (source[3 * i + 1] - mean[1]);
        dest[3 * i + 2] = normal[2] * (source[3 * i + 2] - mean[2]);
    }
#else
    for (int i = 0; i < count; ++i) {
        dest[3 * i + 0] = normal[0] * (source[3 * i + 0] - mean[0]);
        dest[3 * i + 1] = normal[1] * (source[3 * i + 1] - mean[1]);
        dest[3 * i + 2] = normal[2] * (source[3 * i + 2] - mean[2]);
    }
#endif
}

void _TaskC4ToFloatC4(const unsigned char* source, float* dest, const float* mean, const float* normal, size_t count) {
    for (int i = 0; i < count; ++i) {
        dest[4 * i + 0] = normal[0] * (source[4 * i + 0] - mean[0]);
        dest[4 * i + 1] = normal[1] * (source[4 * i + 1] - mean[1]);
        dest[4 * i + 2] = normal[2] * (source[4 * i + 2] - mean[2]);
        dest[4 * i + 3] = normal[3] * (source[4 * i + 3] - mean[3]);
    }
}

void _TaskC1ToFloatRGBA(const unsigned char* source, float* dest, const float* mean, const float* normal, size_t count) {
#ifdef INSPIRECV_TASK_USE_SSE
#if defined(__SSE4_1__)
    {
        ::memset(dest, 0, 4 * sizeof(float) * count);
        int remain = 0;
        int countC16 = (int)count / 16;
        remain = countC16 * 16;
        if (countC16 > 0) {
            const __m128i gM = _mm_setr_epi8(4, 5, 6, 7, 6,5,4,7, 10,9,8,11, 14,13,12,15);
            const __m128i bM = _mm_setr_epi8(8, 9,10,11, 6,5,4,7, 10,9,8,11, 14,13,12,15);
            const __m128i aM = _mm_setr_epi8(12,13,14,15, 6,5,4,7, 10,9,8,11, 14,13,12,15);
            auto normalC4 = _mm_set1_ps(normal[0]);
            auto meanC4 = _mm_set1_ps(mean[0]);
            for (int i=0; i<countC16; ++i) {
                auto sInt8 = _mm_loadu_si128((const __m128i*)(source + 16 * i));
                auto s0 = _mm_cvtepu8_epi32(sInt8);
                auto s1 = _mm_cvtepu8_epi32(_mm_shuffle_epi8(sInt8, gM));
                auto s2 = _mm_cvtepu8_epi32(_mm_shuffle_epi8(sInt8, bM));
                auto s3 = _mm_cvtepu8_epi32(_mm_shuffle_epi8(sInt8, aM));
                auto float00 = _mm_cvtepi32_ps(s0);
                auto float01 = _mm_cvtepi32_ps(s1);
                auto float10 = _mm_cvtepi32_ps(s2);
                auto float11 = _mm_cvtepi32_ps(s3);
                auto f0 = _mm_mul_ps(_mm_sub_ps(float00, meanC4), normalC4);
                auto f1 = _mm_mul_ps(_mm_sub_ps(float01, meanC4), normalC4);
                auto f2 = _mm_mul_ps(_mm_sub_ps(float10, meanC4), normalC4);
                auto f3 = _mm_mul_ps(_mm_sub_ps(float11, meanC4), normalC4);
                auto r1 = _mm_set1_ps(0.0f);
                auto r2 = _mm_set1_ps(0.0f);
                auto r3 = _mm_set1_ps(0.0f);
                auto curDst = dest +  4 * i * 16;
                _MM_TRANSPOSE4_PS(f0, r1, r2, r3);
                _mm_storeu_ps(curDst + 4 * 0, f0);
                _mm_storeu_ps(curDst + 4 * 1, r1);
                _mm_storeu_ps(curDst + 4 * 2, r2);
                _mm_storeu_ps(curDst + 4 * 3, r3);
                curDst += 16;
                _MM_TRANSPOSE4_PS(f1, r1, r2, r3);
                _mm_storeu_ps(curDst + 4 * 0, f1);
                _mm_storeu_ps(curDst + 4 * 1, r1);
                _mm_storeu_ps(curDst + 4 * 2, r2);
                _mm_storeu_ps(curDst + 4 * 3, r3);
                curDst += 16;
                _MM_TRANSPOSE4_PS(f2, r1, r2, r3);
                _mm_storeu_ps(curDst + 4 * 0, f2);
                _mm_storeu_ps(curDst + 4 * 1, r1);
                _mm_storeu_ps(curDst + 4 * 2, r2);
                _mm_storeu_ps(curDst + 4 * 3, r3);
                curDst += 16;
                _MM_TRANSPOSE4_PS(f3, r1, r2, r3);
                _mm_storeu_ps(curDst + 4 * 0, f3);
                _mm_storeu_ps(curDst + 4 * 1, r1);
                _mm_storeu_ps(curDst + 4 * 2, r2);
                _mm_storeu_ps(curDst + 4 * 3, r3);
            }
        }
        for (int i = remain; i < (int)count; ++i) {
            dest[4 * i + 0] = normal[0] * (source[i + 0] - mean[0]);
        }
        return;
    }
#endif
#endif
#ifdef INSPIRECV_TASK_USE_NEON
    _TaskBlitC1ToFloatRGBA(source, dest, mean, normal, count);
#else
    ::memset(dest, 0, 4 * sizeof(float) * count);
    for (int i = 0; i < count; ++i) {
        dest[4 * i + 0] = normal[0] * (source[i + 0] - mean[0]);
    }
#endif
}

void _TaskC3ToFloatRGBA(const unsigned char* source, float* dest, const float* mean, const float* normal, size_t count) {
#ifdef INSPIRECV_TASK_USE_SSE
#if defined(__SSE4_1__)
    {
        int remain = 0;
        int countC4 = (int)count / 4;
        if (countC4 > 1) {
            if (((int)count % 4) * 3 < 4) {
                countC4 -=1;
            }
            auto alpha0 = _mm_setr_ps(normal[0], normal[1], normal[2], 0.0f);
            auto beta0 = _mm_setr_ps(mean[0], mean[1], mean[2], 0.0f);
            remain = countC4 * 4;
            const __m128i rM = _mm_setr_epi8(0, 1, 2, 0, 6,5,4,7, 10,9,8,11, 14,13,12,15);
            const __m128i gM = _mm_setr_epi8(3, 4, 5, 3, 6,5,4,7, 10,9,8,11, 14,13,12,15);
            const __m128i bM = _mm_setr_epi8(6, 7, 8, 6, 6,5,4,7, 10,9,8,11, 14,13,12,15);
            const __m128i aM = _mm_setr_epi8(9, 10, 11, 9, 6,5,4,7, 10,9,8,11, 14,13,12,15);
            for (int i = 0; i < countC4; ++i) {
                auto sInt8 = _mm_loadu_si128((const __m128i*)(source + 12 * i));
                auto s0 = _mm_cvtepu8_epi32(_mm_shuffle_epi8(sInt8, rM));
                auto s1 = _mm_cvtepu8_epi32(_mm_shuffle_epi8(sInt8, gM));
                auto s2 = _mm_cvtepu8_epi32(_mm_shuffle_epi8(sInt8, bM));
                auto s3 = _mm_cvtepu8_epi32(_mm_shuffle_epi8(sInt8, aM));
                auto f0 = _mm_cvtepi32_ps(s0);
                auto f1 = _mm_cvtepi32_ps(s1);
                auto f2 = _mm_cvtepi32_ps(s2);
                auto f3 = _mm_cvtepi32_ps(s3);
                _mm_storeu_ps(dest + 16 * i + 4 * 0, _mm_mul_ps(_mm_sub_ps(f0, beta0), alpha0));
                _mm_storeu_ps(dest + 16 * i + 4 * 1, _mm_mul_ps(_mm_sub_ps(f1, beta0), alpha0));
                _mm_storeu_ps(dest + 16 * i + 4 * 2, _mm_mul_ps(_mm_sub_ps(f2, beta0), alpha0));
                _mm_storeu_ps(dest + 16 * i + 4 * 3, _mm_mul_ps(_mm_sub_ps(f3, beta0), alpha0));
            }
        }
        for (int i = remain; i < (int)count; ++i) {
            dest[4 * i + 0] = normal[0] * (source[3 * i + 0] - mean[0]);
            dest[4 * i + 1] = normal[1] * (source[3 * i + 1] - mean[1]);
            dest[4 * i + 2] = normal[2] * (source[3 * i + 2] - mean[2]);
            dest[4 * i + 3] = 0.0f;
        }
        return;
    }
#endif
#endif
#ifdef INSPIRECV_TASK_USE_NEON
    _TaskBlitC3ToFloatRGBA(source, dest, mean, normal, count);
#else
    for (int i = 0; i < count; ++i) {
        dest[4 * i + 0] = normal[0] * (source[3 * i + 0] - mean[0]);
        dest[4 * i + 1] = normal[1] * (source[3 * i + 1] - mean[1]);
        dest[4 * i + 2] = normal[2] * (source[3 * i + 2] - mean[2]);
        dest[4 * i + 3] = 0.0f;
    }
#endif
}

#if !defined(INSPIRECV_TASK_CLAMP_DECL)
static inline float __clamp(float v, float minV, float maxV) {
    return std::max(std::min(v, maxV), minV);
}
#endif

static void _sampleBilinearCommon(const unsigned char* source, unsigned char* dest, inspirecv::task::Point* points, size_t count,
                                  size_t iw, size_t ih, size_t yStride, size_t bpp) {
    float dy   = points[1].fY;
    float dx   = points[1].fX;
    float xMax = iw - 1;
    float yMax = ih - 1;
    inspirecv::task::Point curPoints;
    curPoints.fX = points[0].fX;
    curPoints.fY = points[0].fY;
    for (int i = 0; i < count; ++i) {
        float y  = __clamp(curPoints.fY, 0, yMax);
        float x  = __clamp(curPoints.fX, 0, xMax);
        int y0   = (int)y;
        int x0   = (int)x;
        int y1   = (int)ceilf(y);
        int x1   = (int)ceilf(x);
        float xF = x - (float)x0;
        float yF = y - (float)y0;
        for (int b = 0; b < bpp; ++b) {
            unsigned char c00 = source[y0 * yStride + bpp * x0 + b];
            unsigned char c01 = source[y0 * yStride + bpp * x1 + b];
            unsigned char c10 = source[y1 * yStride + bpp * x0 + b];
            unsigned char c11 = source[y1 * yStride + bpp * x1 + b];
            float v = (1.0f - xF) * (1.0f - yF) * c00 + xF * (1.0f - yF) * c01 + yF * (1.0 - xF) * c10 + xF * yF * c11;
            v                 = std::min(std::max(v, 0.0f), 255.0f);
            dest[bpp * i + b] = (unsigned char)v;
        }
        curPoints.fY += dy;
        curPoints.fX += dx;
    }
}

void _TaskSamplerC4Bilinear(const unsigned char* source, unsigned char* dest, inspirecv::task::Point* points, size_t sta,
                          size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride) {
#ifdef INSPIRECV_TASK_USE_SSE
#if defined(__SSE4_1__)
    if (inspirecv::cpu_has_sse41()) {
        _SSE_TaskSampleBilinear(source, dest + 4 * sta, points, count, iw, ih, yStride, 4);
        return;
    }
#endif
#endif
#ifdef INSPIRECV_TASK_USE_NEON
    _TaskSamplerC4BilinearOpt(source, dest + 4 * sta, reinterpret_cast<float*>(points), count, iw - 1, ih - 1, yStride);
#else
    _sampleBilinearCommon(source, dest + 4 * sta, points, count, iw, ih, yStride, 4);
#endif
}

void _TaskSamplerC3Bilinear(const unsigned char* source, unsigned char* dest, inspirecv::task::Point* points, size_t sta,
                          size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride) {
    _sampleBilinearCommon(source, dest + 3 * sta, points, count, iw, ih, yStride, 3);
}

void _TaskSamplerC1Bilinear(const unsigned char* source, unsigned char* dest, inspirecv::task::Point* points, size_t sta,
                          size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride) {
#ifdef INSPIRECV_TASK_USE_NEON
    _TaskSamplerC1BilinearOpt(source, dest + sta, reinterpret_cast<float*>(points), count, iw - 1, ih - 1, yStride);
#else
    _sampleBilinearCommon(source, dest + sta, points, count, iw, ih, yStride, 1);
#endif
}

void _TaskSamplerNearest(const unsigned char* source, unsigned char* dest, inspirecv::task::Point* points, size_t sta, size_t count,
                       size_t iw, size_t ih, size_t yStride, int bpp) {
    dest = dest + bpp * sta;
    inspirecv::task::Point curPoints;
    curPoints.fX = points[0].fX;
    curPoints.fY = points[0].fY;
    float dy     = points[1].fY;
    float dx     = points[1].fX;
    float xMax   = iw - 1;
    float yMax   = ih - 1;
    for (int i = 0; i < count; ++i) {
        int y = (int)roundf(__clamp(curPoints.fY, 0, yMax));
        int x = (int)roundf(__clamp(curPoints.fX, 0, xMax));
        curPoints.fY += dy;
        curPoints.fX += dx;
        auto sourcePos = y * yStride + bpp * x;
        for (int j = 0; j < bpp; ++j) {
            dest[bpp * i + j] = source[sourcePos + j];
        }
    }
}

void _TaskSamplerC4Nearest(const unsigned char* source, unsigned char* dest, inspirecv::task::Point* points, size_t sta,
                         size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride) {
#ifdef INSPIRECV_TASK_USE_SSE
#if defined(__SSE4_1__)
    if (inspirecv::cpu_has_sse41()) {
        _SSE_TaskSamplerNearest(source, dest, points, sta, count, iw, ih, yStride, 4);
        return;
    }
#endif
#endif
#ifdef INSPIRECV_TASK_USE_NEON
    _TaskSamplerC4NearestOpt(source, dest + 4 * sta, reinterpret_cast<float*>(points), count, iw - 1, ih - 1, yStride);
#else
    _TaskSamplerNearest(source, dest, points, sta, count, iw, ih, yStride, 4);
#endif
}

void _TaskSamplerC1Nearest(const unsigned char* source, unsigned char* dest, inspirecv::task::Point* points, size_t sta,
                         size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride) {
#ifdef INSPIRECV_TASK_USE_NEON
    _TaskSamplerC1NearestOpt(source, dest + sta, reinterpret_cast<float*>(points), count, iw - 1, ih - 1, yStride);
#else
    _TaskSamplerNearest(source, dest, points, sta, count, iw, ih, yStride, 1);
#endif
}

void _TaskSamplerC3Nearest(const unsigned char* source, unsigned char* dest, inspirecv::task::Point* points, size_t sta,
                         size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride) {
    _TaskSamplerNearest(source, dest, points, sta, count, iw, ih, yStride, 3);
}

void _TaskSamplerCopyCommon(const unsigned char* source, unsigned char* dest, inspirecv::task::Point* points, size_t sta,
                          size_t count, size_t iw, size_t ih, size_t yStride, int bpp) {
    dest = dest + bpp * sta;
    inspirecv::task::Point curPoints;
    curPoints.fX   = points[0].fX;
    curPoints.fY   = points[0].fY;
    float xMax     = iw - 1;
    float yMax     = ih - 1;
    int y          = (int)roundf(__clamp(curPoints.fY, 0, yMax));
    int x          = (int)roundf(__clamp(curPoints.fX, 0, xMax));
    auto sourcePos = y * yStride + bpp * x;
    ::memcpy(dest, source + sourcePos, bpp * count);
}

void _TaskSamplerI420Copy(const unsigned char* source, unsigned char* dest, inspirecv::task::Point* points, size_t sta,
                        size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride) {
    inspirecv::task::Point curPoints;
    curPoints.fX    = points[0].fX;
    curPoints.fY    = points[0].fY;
    float xMax      = iw - 1;
    float yMax      = ih - 1;
    int y           = (int)roundf(__clamp(curPoints.fY, 0, yMax));
    int x           = (int)roundf(__clamp(curPoints.fX, 0, xMax));
    auto uvPlane = (((int)iw + 1) / 2) * ((int(ih) + 1) / 2);
    int sourcePosY  = y * (int)iw + x;
    auto sourcePosU = source + (int)iw * (int)ih + (y / 2) * (((int)iw + 1) / 2) + (x / 2);
    auto sourcePosV = source + (int)iw * (int)ih + (y / 2) * (((int)iw + 1) / 2) + (x / 2) + uvPlane;
    auto uvCount = (count + 1) / 2;
    ::memcpy(dest + sta, source + sourcePosY, count);
    auto uDest = dest + (capacity) + (sta / 2) * 2;
    for (int i=0; i<uvCount; ++i) {
        uDest[2 * i + 0] = sourcePosV[i];
        uDest[2 * i + 1] = sourcePosU[i];
    }
}

void _TaskSamplerI420Nearest(const unsigned char* source, unsigned char* dest, inspirecv::task::Point* points, size_t sta,
                           size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride) {
    auto srcY  = source;
    auto dstY  = dest + sta;
    auto dstUV = dest + (capacity) + (sta / 2) * 2;
    auto stride = yStride;
    if (yStride == 0) stride = iw;
    auto srcU = source + stride * ih;
    _TaskSamplerC1Nearest(srcY, dstY, points, 0, count, capacity, iw, ih, stride);
    inspirecv::task::Point uvPoints[2];
    uvPoints[0].fX = (points[0].fX - 0.01f) / 2.0f;
    uvPoints[0].fY = (points[0].fY - 0.01f) / 2.0f;
    uvPoints[1].fX = points[1].fX / 2.0f;
    uvPoints[1].fY = points[1].fY / 2.0f;
    if (yStride == 0) stride =  ((iw + 1) / 2);
    auto srcV = srcU + stride * ((ih + 1) / 2);
    auto uvCount = (count + 1) / 2;
    {
        inspirecv::task::Point curPoints;
        curPoints.fX = uvPoints[0].fX;
        curPoints.fY = uvPoints[0].fY;
        float dy     = uvPoints[1].fY;
        float dx     = uvPoints[1].fX;
        float xMax   = ((iw + 1) / 2) - 1;
        float yMax   = ((ih + 1) / 2) - 1;
        for (int i = 0; i < uvCount; ++i) {
            int y = (int)roundf(__clamp(curPoints.fY, 0, yMax));
            int x = (int)roundf(__clamp(curPoints.fX, 0, xMax));
            curPoints.fY += dy;
            curPoints.fX += dx;
            auto offset = y * stride + x;
            dstUV[2 * i + 0] = srcV[offset];
            dstUV[2 * i + 1] = srcU[offset];
        }
    }
}

void _TaskSamplerNV21Copy(const unsigned char* source, unsigned char* dest, inspirecv::task::Point* points, size_t sta,
                        size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride) {
    inspirecv::task::Point curPoints;
    curPoints.fX    = points[0].fX;
    curPoints.fY    = points[0].fY;
    float xMax      = iw - 1;
    float yMax      = ih - 1;
    int y           = (int)roundf(__clamp(curPoints.fY, 0, yMax));
    int x           = (int)roundf(__clamp(curPoints.fX, 0, xMax));
    int stride = (int)yStride;
    int hstride = (int)yStride;
    if (yStride == 0) { stride = (int)iw; hstride = (((int)iw + 1) / 2) * 2; }
    int sourcePosY  = y * stride + x;
    int sourcePosUV = (int)stride * (int)ih + (y / 2) * hstride + (x / 2) * 2;
    ::memcpy(dest + sta, source + sourcePosY, count);
    ::memcpy(dest + (capacity) + (sta / 2) * 2, source + sourcePosUV, ((count + 1) / 2) * 2);
}

void _TaskSamplerNV21Nearest(const unsigned char* source, unsigned char* dest, inspirecv::task::Point* points, size_t sta,
                           size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride) {
    auto srcY  = source;
    auto dstY  = dest + sta;
    auto dstUV = dest + (capacity) + (sta / 2) * 2;
    auto stride = yStride;
    if (yStride == 0) stride = iw;
    auto srcUV = source + stride * ih;
    _TaskSamplerC1Nearest(srcY, dstY, points, 0, count, capacity, iw, ih, stride);
    inspirecv::task::Point uvPoints[2];
    uvPoints[0].fX = (points[0].fX - 0.01f) / 2.0f;
    uvPoints[0].fY = (points[0].fY - 0.01f) / 2.0f;
    uvPoints[1].fX = points[1].fX;
    uvPoints[1].fY = points[1].fY;
    if (yStride == 0) stride =  ((iw + 1) / 2) * 2;
    _TaskSamplerNearest(srcUV, dstUV, uvPoints, 0, (count + 1) / 2, (iw + 1) / 2, (ih + 1) / 2, stride, 2);
}

static void _swapUV(const unsigned char* source, unsigned char* dest, size_t countC2) {
    int sta = 0;
#ifdef INSPIRECV_TASK_USE_NEON
    int countC2C16 = (int)countC2 / 16;
    sta = countC2C16 * 16;
    for (int i=0; i<countC2C16; ++i) {
        auto src = vld2q_u8(source + i * 32);
        auto temp = src.val[0];
        src.val[0] = src.val[1];
        src.val[1] = temp;
        vst2q_u8(dest + i * 32, src);
    }
#endif
    for (int i=sta; i < countC2; ++i) {
        auto temp = source[2*i];
        dest[2*i] = source[2*i+1];
        dest[2*i+1] = temp;
    }
}

void _TaskSamplerNV12Copy(const unsigned char* source, unsigned char* dest, inspirecv::task::Point* points, size_t sta,
                        size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride) {
    inspirecv::task::Point curPoints;
    curPoints.fX    = points[0].fX;
    curPoints.fY    = points[0].fY;
    float xMax      = iw - 1;
    float yMax      = ih - 1;
    int y           = (int)roundf(__clamp(curPoints.fY, 0, yMax));
    int x           = (int)roundf(__clamp(curPoints.fX, 0, xMax));
    int stride = (int)yStride;
    int hstride = (int)yStride;
    if (yStride == 0) { stride = (int)iw; hstride = (((int)iw + 1) / 2) * 2; }
    int sourcePosY  = y * stride + x;
    int sourcePosUV = (int)stride * (int)ih + (y / 2) * hstride + (x / 2) * 2;
    // Copy Y directly
    ::memcpy(dest + sta, source + sourcePosY, count);
    // Copy UV with U/V swapped to VU (NV21 layout expected by blitters)
    auto destUV = dest + (capacity) + (sta / 2) * 2;
    int uvPairs = (int)((count + 1) / 2);
#ifdef INSPIRECV_TASK_USE_NEON
    int blocks = uvPairs / 16; // 16 pairs -> 32 bytes
    for (int i = 0; i < blocks; ++i) {
        auto uv = vld2q_u8(source + sourcePosUV + i * 32); // uv.val[0]=U, uv.val[1]=V
        // write as VU interleaved
        uint8x16x2_t vu;
        vu.val[0] = uv.val[1];
        vu.val[1] = uv.val[0];
        vst2q_u8(destUV + i * 32, vu);
    }
    int consumed = blocks * 16;
    for (int i = consumed; i < uvPairs; ++i) {
        uint8_t u = source[sourcePosUV + 2 * i + 0];
        uint8_t v = source[sourcePosUV + 2 * i + 1];
        destUV[2 * i + 0] = v; // V first
        destUV[2 * i + 1] = u; // then U
    }
#else
    for (int i = 0; i < uvPairs; ++i) {
        uint8_t u = source[sourcePosUV + 2 * i + 0];
        uint8_t v = source[sourcePosUV + 2 * i + 1];
        destUV[2 * i + 0] = v;
        destUV[2 * i + 1] = u;
    }
#endif
}

void _TaskSamplerNV12Nearest(const unsigned char* source, unsigned char* dest, inspirecv::task::Point* points, size_t sta,
                           size_t count, size_t capacity, size_t iw, size_t ih, size_t yStride) {
    // Y plane
    auto srcY  = source;
    auto dstY  = dest + sta;
    auto stride = yStride;
    if (yStride == 0) stride = iw;
    _TaskSamplerC1Nearest(srcY, dstY, points, 0, count, capacity, iw, ih, stride);
    // UV plane: sample from NV12 (UV interleaved), write NV21 (VU interleaved)
    inspirecv::task::Point uvPoints[2];
    uvPoints[0].fX = (points[0].fX - 0.01f) / 2.0f;
    uvPoints[0].fY = (points[0].fY - 0.01f) / 2.0f;
    uvPoints[1].fX = points[1].fX;
    uvPoints[1].fY = points[1].fY;
    auto dstUV = dest + (capacity) + (sta / 2) * 2;
    auto uvCount = (count + 1) / 2;
    size_t uvStride = yStride == 0 ? (((iw + 1) / 2) * 2) : yStride; // bytes per UV row
    auto srcUV = source + stride * ih; // NV12 UV starts here
    float dy   = uvPoints[1].fY;
    float dx   = uvPoints[1].fX;
    float xMax = ((iw + 1) / 2) - 1;
    float yMax = ((ih + 1) / 2) - 1;
    inspirecv::task::Point curPoints;
    curPoints.fX = uvPoints[0].fX;
    curPoints.fY = uvPoints[0].fY;
    for (int i = 0; i < (int)uvCount; ++i) {
        int y = (int)roundf(__clamp(curPoints.fY, 0, yMax));
        int x = (int)roundf(__clamp(curPoints.fX, 0, xMax));
        curPoints.fY += dy;
        curPoints.fX += dx;
        auto offset = y * uvStride + 2 * x;
        uint8_t u = srcUV[offset + 0];
        uint8_t v = srcUV[offset + 1];
        dstUV[2 * i + 0] = v; // V
        dstUV[2 * i + 1] = u; // U
    }
}

void _TaskC3blitH(const unsigned char* source, unsigned char* dest, size_t count) {
    for (int i = 0; i < count; i++) { memcpy(dest + 3 * i, source, 3); }
}

void _TaskC4blitH(const unsigned char* source, unsigned char* dest, size_t count) {
    for (int i = 0; i < count; i++) { memcpy(dest + 4 * i, source, 4); }
}

void _TaskC1blitH(const unsigned char* source, unsigned char* dest, size_t count) {
    for (int i = 0; i < count; i++) { memcpy(dest + i, source, 1); }
}


