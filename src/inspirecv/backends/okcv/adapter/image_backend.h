#ifndef INSPIRECV_BACKENDS_OKCV_ADAPTER_IMAGE_BACKEND_H_
#define INSPIRECV_BACKENDS_OKCV_ADAPTER_IMAGE_BACKEND_H_

#include "inspirecv/backends/okcv/backend.h"
#include "inspirecv/core/image.h"
#include "inspirecv/core/rect.h"
#include "inspirecv/core/point.h"
#include "inspirecv/core/transform_matrix.h"
#include "inspirecv/core/runtime/acceleration_state.h"
#include "inspirecv/core/runtime/image_acceleration.h"
#include "logging.h"

#ifndef BACKUP_AFFINE
#define BACKUP_AFFINE 0
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#  if defined(__has_include)
#    if __has_include(<arm_neon.h>)
#      include <arm_neon.h>
#    endif
#  else
#    include <arm_neon.h>
#  endif
#endif

#if defined(__AVX2__) || defined(__SSE2__)
#  if defined(__has_include)
#    if __has_include(<immintrin.h>)
#      include <immintrin.h>
#    endif
#  else
#    include <immintrin.h>
#  endif
#endif

namespace inspirecv {

// OKCV adapter for the public ImageT API. Pixel storage and backend operations
// live in okcv::Bitmap so debugger stacks do not contain two unrelated Image
// classes.
template<typename Pixel>
class ImageT<Pixel>::Impl {
public:
    // Creation and conversion
    Impl(const okcv::Bitmap<Pixel>& bitmap) {
        bitmap_ = bitmap.Clone();
    }

    Impl() : bitmap_() {}

    Impl(int width, int height, int channels, const Pixel* data = nullptr,
         bool copy_data = true) {
        bitmap_.Reset(width, height, channels, data, copy_data);
    }

    void Reset(int width, int height, int channels, const Pixel* data = nullptr,
               bool copy_data = true) {
        bitmap_.Reset(width, height, channels, data, copy_data);
    }

    ImageT<Pixel> Clone() const {
        ImageT<Pixel> result;
        result.impl_->bitmap_ = bitmap_.Clone();
        return result;
    }

    ~Impl() = default;

    // Basic properties
    int Width() const {
        return bitmap_.Width();
    }
    int Height() const {
        return bitmap_.Height();
    }
    int Channels() const {
        return bitmap_.Channels();
    }
    bool Empty() const {
        return bitmap_.Empty();
    }
    const Pixel* Data() const {
        return bitmap_.Data();
    }

    // Get internal image implementation
    void* GetInternalImage() const {
        return static_cast<void*>(const_cast<Pixel*>(bitmap_.Data()));
    }

    // I/O operations
    bool Read(const std::string& filename, int channels) {
        bitmap_.Read(filename, channels);
        return !Empty();
    }

    bool Write(const std::string& filename) const {
        bitmap_.Write(filename);
        return true;
    }

    void Show(const std::string& window_name, int delay) const {
        bitmap_.Show(window_name, delay);
    }

    // Access to core image for interop if needed
    const okcv::Bitmap<Pixel>& CoreBitmap() const { return bitmap_; }

    // Basic operations
    void Fill(double value) {
        bitmap_.Fill(static_cast<Pixel>(value));
    }

    ImageT<Pixel> Mul(double scale) const {
        ImageT<Pixel> result;
        result.impl_->bitmap_ = bitmap_.Mul(static_cast<float>(scale));
        return result;
    }

    ImageT<Pixel> Add(double value) const {
        ImageT<Pixel> result;
        result.impl_->bitmap_ = bitmap_.MulAdd(1.0f, static_cast<float>(value));
        return result;
    }

    // Geometric transformations
    ImageT<Pixel> Resize(int width, int height, bool use_linear) const {
        ImageT<Pixel> result;
        const internal::ImageGeometryOperation operation =
          use_linear ? internal::ImageGeometryOperation::kResizeBilinear
                     : internal::ImageGeometryOperation::kResizeNearest;
        if (TryCudaGeometry(result, operation, width, height, nullptr)) {
            return result;
        }
        internal::RecordImageExecutionBackend(AccelerationBackend::kCpu);
        if (use_linear) {
            result.impl_->bitmap_ = bitmap_.ResizeBilinear(width, height);
        } else {
            result.impl_->bitmap_ = bitmap_.ResizeNearest(width, height);
        }
        return result;
    }

    ImageT<Pixel> Crop(const Rect<int>& rect) const {
        ImageT<Pixel> result;
        int x2 = rect.GetX() + std::max(1, rect.GetWidth());
        int y2 = rect.GetY() + std::max(1, rect.GetHeight());

        okcv::Rect2i okcv_rect(rect.GetX(), rect.GetY(), x2, y2);

        result.impl_->bitmap_ = bitmap_.Crop(okcv_rect);
        return result;
    }

    ImageT<Pixel> WarpAffine(const TransformMatrix& matrix, int width, int height) const {
#if BACKUP_AFFINE
        // Original implementation (backup)
        ImageT<Pixel> result_backup;
        okcv::TransformMatrix okcv_matrix_backup =
          *static_cast<okcv::TransformMatrix*>(matrix.GetInternalMatrix());
        result_backup.impl_->bitmap_ = bitmap_.AffineBilinear(width, height, okcv_matrix_backup);
        return result_backup;
#endif
        // Current implementation (modifiable copy)
        ImageT<Pixel> result;
        okcv::TransformMatrix okcv_matrix =
          *static_cast<okcv::TransformMatrix*>(matrix.GetInternalMatrix());
        float affine[6];
        for (int index = 0; index < 6; ++index) {
            affine[index] = okcv_matrix[index];
        }
        if (TryCudaGeometry(
              result, internal::ImageGeometryOperation::kWarpAffineBilinear,
              width, height, affine)) {
            return result;
        }
        internal::RecordImageExecutionBackend(AccelerationBackend::kCpu);
        // Route to optimized affine (falls back internally if unsupported)
        result.impl_->bitmap_ = bitmap_.AffineBilinearOptimized(width, height, okcv_matrix);
        return result;
    }

    ImageT<Pixel> Rotate90() const {
        ImageT<Pixel> result;
        if (TryCudaGeometry(result, internal::ImageGeometryOperation::kRotate90,
                            bitmap_.Height(), bitmap_.Width(), nullptr)) {
            return result;
        }
        internal::RecordImageExecutionBackend(AccelerationBackend::kCpu);
        result.impl_->bitmap_ = bitmap_.Rotate90();
        return result;
    }

    ImageT<Pixel> Rotate180() const {
        ImageT<Pixel> result;
        if (TryCudaGeometry(result, internal::ImageGeometryOperation::kRotate180,
                            bitmap_.Width(), bitmap_.Height(), nullptr)) {
            return result;
        }
        internal::RecordImageExecutionBackend(AccelerationBackend::kCpu);
        result.impl_->bitmap_ = bitmap_.Rotate180();
        return result;
    }

    ImageT<Pixel> Rotate270() const {
        ImageT<Pixel> result;
        if (TryCudaGeometry(result, internal::ImageGeometryOperation::kRotate270,
                            bitmap_.Height(), bitmap_.Width(), nullptr)) {
            return result;
        }
        internal::RecordImageExecutionBackend(AccelerationBackend::kCpu);
        result.impl_->bitmap_ = bitmap_.Rotate270();
        return result;
    }

    ImageT<Pixel> SwapRB() const {
        ImageT<Pixel> result;
        result.impl_->bitmap_ = bitmap_.SwapRB();
        return result;
    }

    ImageT<Pixel> FlipHorizontal() const {
        ImageT<Pixel> result;
        result.impl_->bitmap_ = bitmap_.FlipLeftRight();
        return result;
    }

    ImageT<Pixel> FlipVertical() const {
        ImageT<Pixel> result;
        result.impl_->bitmap_ = bitmap_.FlipUpDown();
        return result;
    }

    ImageT<Pixel> Pad(int top, int bottom, int left, int right, const std::vector<double>& color) const {
        ImageT<Pixel> result;
        if (color.empty()) {
            result.impl_->bitmap_ =
              bitmap_.Pad(top, bottom, left, right, static_cast<Pixel>(0));
            return result;
        }
        const Pixel first = static_cast<Pixel>(color[0]);
        const bool uniform =
          std::all_of(color.begin() + 1, color.end(),
                      [&](double value) { return static_cast<Pixel>(value) == first; });
        if (uniform) {
            result.impl_->bitmap_ = bitmap_.Pad(top, bottom, left, right, first);
            return result;
        }
        result.impl_->bitmap_ =
          bitmap_.Pad(top, bottom, left, right, color.data(), color.size());
        return result;
    }

    // Image processing
    ImageT<Pixel> GaussianBlur(int kernel_size, double sigma) const {
        ImageT<Pixel> result;
        result.impl_->bitmap_ = bitmap_.GaussianBlur(kernel_size, static_cast<float>(sigma));
        return result;
    }

    ImageT<Pixel> Erode(int kernel_size, int iterations) const {
        INSPIRECV_CHECK(bitmap_.Channels() == 1) << "Erode only supports single-channel";
        int left = kernel_size / 2;
        int right = kernel_size - left - 1;
        int top = kernel_size / 2;
        int bottom = kernel_size - top - 1;
        ImageT<Pixel> result;
        okcv::Bitmap<Pixel> cur = bitmap_.Clone();
        for (int it = 0; it < std::max(1, iterations); ++it) {
            okcv::Bitmap<Pixel> tmp = cur.MinFilter(left, right, top, bottom);
            cur = std::move(tmp);
        }
        result.impl_->bitmap_ = std::move(cur);
        return result;
    }

    ImageT<Pixel> Dilate(int kernel_size, int iterations) const {
        INSPIRECV_CHECK(bitmap_.Channels() == 1) << "Dilate only supports single-channel";
        int left = kernel_size / 2;
        int right = kernel_size - left - 1;
        int top = kernel_size / 2;
        int bottom = kernel_size - top - 1;
        ImageT<Pixel> result;
        okcv::Bitmap<Pixel> cur = bitmap_.Clone();
        for (int it = 0; it < std::max(1, iterations); ++it) {
            okcv::Bitmap<Pixel> tmp = cur.MaxFilter(left, right, top, bottom);
            cur = std::move(tmp);
        }
        result.impl_->bitmap_ = std::move(cur);
        return result;
    }

    ImageT<Pixel> Threshold(double thresh, double maxval, int type) const {
        // Support binary only: type==0 -> THRESH_BINARY
        INSPIRECV_CHECK(bitmap_.Channels() == 1);
        ImageT<Pixel> result;
        result.impl_->bitmap_.Reset(bitmap_.Width(), bitmap_.Height(), 1);
        const Pixel* src = bitmap_.Data();
        Pixel* dst = result.impl_->bitmap_.Data();
        const int total = bitmap_.DataSize();
        const Pixel t = static_cast<Pixel>(thresh);
        const Pixel mv = static_cast<Pixel>(maxval);
        // SIMD fast paths
#if defined(__AVX2__)
        if (std::is_same<Pixel, uint8_t>::value) {
            const int step = 32;
            __m256i vth = _mm256_set1_epi8((char)t);
            __m256i vmaxv = _mm256_set1_epi8((char)mv);
            __m256i vzero = _mm256_setzero_si256();
            __m256i off = _mm256_set1_epi8((char)0x80);
            int i = 0;
            for (; i + step <= total; i += step) {
                __m256i v = _mm256_loadu_si256((const __m256i*)(src + i));
                __m256i cmp = _mm256_cmpgt_epi8(_mm256_xor_si256(v, off), _mm256_xor_si256(vth, off));
                __m256i out = _mm256_or_si256(_mm256_and_si256(cmp, vmaxv), _mm256_andnot_si256(cmp, vzero));
                _mm256_storeu_si256((__m256i*)(dst + i), out);
            }
            for (; i < total; ++i) dst[i] = src[i] > t ? mv : static_cast<Pixel>(0);
            return result;
        }
        if (std::is_same<Pixel, float>::value) {
            const int step = 8;
            __m256 vth = _mm256_set1_ps(static_cast<float>(thresh));
            __m256 vmaxv = _mm256_set1_ps(static_cast<float>(maxval));
            __m256 vzero = _mm256_set1_ps(0.0f);
            int i = 0;
            for (; i + step <= total; i += step) {
                __m256 v = _mm256_loadu_ps(reinterpret_cast<const float*>(src) + i);
                __m256 m = _mm256_cmp_ps(v, vth, _CMP_GT_OQ);
                __m256 out = _mm256_blendv_ps(vzero, vmaxv, m);
                _mm256_storeu_ps(reinterpret_cast<float*>(dst) + i, out);
            }
            for (; i < total; ++i) dst[i] = src[i] > t ? mv : static_cast<Pixel>(0);
            return result;
        }
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        if (std::is_same<Pixel, uint8_t>::value) {
            const int step = 16;
            uint8x16_t vth = vdupq_n_u8(static_cast<uint8_t>(t));
            uint8x16_t vmaxv = vdupq_n_u8(static_cast<uint8_t>(mv));
            uint8x16_t vzero = vdupq_n_u8(0);
            int i = 0;
            for (; i + step <= total; i += step) {
                uint8x16_t v = vld1q_u8(reinterpret_cast<const uint8_t*>(src) + i);
                uint8x16_t m = vcgtq_u8(v, vth);
                uint8x16_t out = vbslq_u8(m, vmaxv, vzero);
                vst1q_u8(reinterpret_cast<uint8_t*>(dst) + i, out);
            }
            for (; i < total; ++i) dst[i] = src[i] > t ? mv : static_cast<Pixel>(0);
            return result;
        }
        if (std::is_same<Pixel, float>::value) {
            const int step = 4;
            float32x4_t vth = vdupq_n_f32(static_cast<float>(thresh));
            float32x4_t vmaxv = vdupq_n_f32(static_cast<float>(maxval));
            float32x4_t vzero = vdupq_n_f32(0.0f);
            int i = 0;
            for (; i + step <= total; i += step) {
                float32x4_t v = vld1q_f32(reinterpret_cast<const float*>(src) + i);
                uint32x4_t m = vcgtq_f32(v, vth);
                float32x4_t out = vbslq_f32(m, vmaxv, vzero);
                vst1q_f32(reinterpret_cast<float*>(dst) + i, out);
            }
            for (; i < total; ++i) dst[i] = src[i] > t ? mv : static_cast<Pixel>(0);
            return result;
        }
#endif
        // Scalar fallback
        for (int i = 0; i < total; ++i) dst[i] = src[i] > t ? mv : static_cast<Pixel>(0);
        return result;
    }

    ImageT<Pixel> AbsDiff(const typename ImageT<Pixel>::Impl& other) const {
        INSPIRECV_CHECK(bitmap_.Width() == other.Width() && bitmap_.Height() == other.Height() &&
                        bitmap_.Channels() == other.Channels());
        ImageT<Pixel> result;
        result.impl_->bitmap_.Reset(bitmap_.Width(), bitmap_.Height(), bitmap_.Channels());
        const Pixel* a = bitmap_.Data();
        const Pixel* b = other.bitmap_.Data();
        Pixel* d = result.impl_->bitmap_.Data();
        const int n = bitmap_.DataSize();
#if defined(__AVX2__)
        if (std::is_same<Pixel, uint8_t>::value) {
            const int step = 32;
            int i = 0;
            for (; i + step <= n; i += step) {
                __m256i va = _mm256_loadu_si256((const __m256i*)(a + i));
                __m256i vb = _mm256_loadu_si256((const __m256i*)(b + i));
                __m256i vmaxv = _mm256_max_epu8(va, vb);
                __m256i vminv = _mm256_min_epu8(va, vb);
                __m256i vout = _mm256_sub_epi8(vmaxv, vminv);
                _mm256_storeu_si256((__m256i*)(d + i), vout);
            }
            for (; i < n; ++i) d[i] = static_cast<Pixel>(std::abs(int(a[i]) - int(b[i])));
            return result;
        }
        if (std::is_same<Pixel, float>::value) {
            const int step = 8;
            __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
            int i = 0;
            for (; i + step <= n; i += step) {
                __m256 va = _mm256_loadu_ps(reinterpret_cast<const float*>(a) + i);
                __m256 vb = _mm256_loadu_ps(reinterpret_cast<const float*>(b) + i);
                __m256 diff = _mm256_sub_ps(va, vb);
                __m256 vout = _mm256_and_ps(diff, mask);
                _mm256_storeu_ps(reinterpret_cast<float*>(d) + i, vout);
            }
            for (; i < n; ++i) d[i] = static_cast<Pixel>(std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
            return result;
        }
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        if (std::is_same<Pixel, uint8_t>::value) {
            const int step = 16;
            int i = 0;
            for (; i + step <= n; i += step) {
                uint8x16_t va = vld1q_u8(reinterpret_cast<const uint8_t*>(a) + i);
                uint8x16_t vb = vld1q_u8(reinterpret_cast<const uint8_t*>(b) + i);
                uint8x16_t vout = vabdq_u8(va, vb);
                vst1q_u8(reinterpret_cast<uint8_t*>(d) + i, vout);
            }
            for (; i < n; ++i) d[i] = static_cast<Pixel>(std::abs(int(a[i]) - int(b[i])));
            return result;
        }
        if (std::is_same<Pixel, float>::value) {
            const int step = 4;
            int i = 0;
            for (; i + step <= n; i += step) {
                float32x4_t va = vld1q_f32(reinterpret_cast<const float*>(a) + i);
                float32x4_t vb = vld1q_f32(reinterpret_cast<const float*>(b) + i);
                float32x4_t vout = vabsq_f32(vsubq_f32(va, vb));
                vst1q_f32(reinterpret_cast<float*>(d) + i, vout);
            }
            for (; i < n; ++i) d[i] = static_cast<Pixel>(std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
            return result;
        }
#endif
        // Scalar fallback
        for (int i = 0; i < n; ++i) {
            auto diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
            double ad = std::abs(diff);
            d[i] = static_cast<Pixel>(ad);
        }
        return result;
    }

    ImageT<Pixel> MeanChannels() const {
        INSPIRECV_CHECK(bitmap_.Channels() >= 1);
        if (bitmap_.Channels() == 1) return Clone();
        ImageT<Pixel> result;
        result.impl_->bitmap_.Reset(bitmap_.Width(), bitmap_.Height(), 1);
        // Common fast path: 3-channel interleaved images
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        if (bitmap_.Channels() == 3) {
            if (std::is_same<Pixel, uint8_t>::value) {
                const int step = 16;
                for (int y = 0; y < bitmap_.Height(); ++y) {
                    const uint8_t* row = reinterpret_cast<const uint8_t*>(bitmap_.Row(y));
                    uint8_t* out = reinterpret_cast<uint8_t*>(result.impl_->bitmap_.Row(y));
                    int x = 0;
                    for (; x + step <= bitmap_.Width(); x += step) {
                        uint8x16x3_t bgr = vld3q_u8(row + x * 3);
                        // Sum to 16-bit: [0..765]
                        uint16x8_t s0 = vaddl_u8(vget_low_u8(bgr.val[0]), vget_low_u8(bgr.val[1]));
                        s0 = vaddw_u8(s0, vget_low_u8(bgr.val[2]));
                        uint16x8_t s1 = vaddl_u8(vget_high_u8(bgr.val[0]), vget_high_u8(bgr.val[1]));
                        s1 = vaddw_u8(s1, vget_high_u8(bgr.val[2]));
                        // Integer divide by 3: q = floor(n * 21845 / 65536)
                        const uint16x4_t k = vdup_n_u16(21845);
                        uint32x4_t m00 = vmull_u16(vget_low_u16(s0), k);
                        uint32x4_t m01 = vmull_u16(vget_high_u16(s0), k);
                        uint32x4_t m10 = vmull_u16(vget_low_u16(s1), k);
                        uint32x4_t m11 = vmull_u16(vget_high_u16(s1), k);
                        uint16x8_t q0 = vcombine_u16(vshrn_n_u32(m00, 16), vshrn_n_u32(m01, 16));
                        uint16x8_t q1 = vcombine_u16(vshrn_n_u32(m10, 16), vshrn_n_u32(m11, 16));
                        uint8x16_t y8 = vcombine_u8(vqmovn_u16(q0), vqmovn_u16(q1));
                        vst1q_u8(out + x, y8);
                    }
                    for (; x < bitmap_.Width(); ++x) {
                        const uint8_t* p = row + x * 3;
                        out[x] = static_cast<uint8_t>((static_cast<unsigned>(p[0]) +
                                                       static_cast<unsigned>(p[1]) +
                                                       static_cast<unsigned>(p[2])) / 3u);
                    }
                }
                return result;
            }
            if (std::is_same<Pixel, float>::value) {
                const float32x4_t k = vdupq_n_f32(1.0f / 3.0f);
                for (int y = 0; y < bitmap_.Height(); ++y) {
                    const float* row = reinterpret_cast<const float*>(bitmap_.Row(y));
                    float* out = reinterpret_cast<float*>(result.impl_->bitmap_.Row(y));
                    int x = 0;
                    for (; x + 4 <= bitmap_.Width(); x += 4) {
                        float32x4x3_t bgr = vld3q_f32(row + x * 3);
                        float32x4_t s = vaddq_f32(vaddq_f32(bgr.val[0], bgr.val[1]), bgr.val[2]);
                        vst1q_f32(out + x, vmulq_f32(s, k));
                    }
                    for (; x < bitmap_.Width(); ++x) {
                        const float* p = row + x * 3;
                        out[x] = (p[0] + p[1] + p[2]) * (1.0f / 3.0f);
                    }
                }
                return result;
            }
        }
#endif
        // Fallback: generic scalar for other channel counts / types
        for (int y = 0; y < bitmap_.Height(); ++y) {
            const Pixel* row = bitmap_.Row(y);
            Pixel* out = result.impl_->bitmap_.Row(y);
            for (int x = 0; x < bitmap_.Width(); ++x) {
                double s = 0;
                for (int c = 0; c < bitmap_.Channels(); ++c) s += row[x * bitmap_.Channels() + c];
                out[x] = static_cast<Pixel>(s / bitmap_.Channels());
            }
        }
        return result;
    }

    ImageT<Pixel> Blend(const typename ImageT<Pixel>::Impl& other, const typename ImageT<uint8_t>::Impl& mask) const {
        INSPIRECV_CHECK(bitmap_.Width() == other.Width() && bitmap_.Height() == other.Height() &&
                        bitmap_.Channels() == other.Channels());
        INSPIRECV_CHECK(mask.Channels() == 1 && mask.Width() == bitmap_.Width() &&
                        mask.Height() == bitmap_.Height());

        const int width = bitmap_.Width();
        const int height = bitmap_.Height();
        const int cn = bitmap_.Channels();

        ImageT<Pixel> result;
        result.impl_->bitmap_.Reset(width, height, cn);

        const Pixel* a = bitmap_.Data();
        const Pixel* b = other.Data();
        const uint8_t* m = mask.Data();
        Pixel* d = result.impl_->bitmap_.Data();

        // Scalar per-channel blender with exact round(x/255)
        auto blend_u8 = [](uint8_t va, uint8_t vb, uint8_t mm) -> uint8_t {
            int x = static_cast<int>(mm) * (static_cast<int>(va) - static_cast<int>(vb))
                    + ((static_cast<int>(vb) << 8) - static_cast<int>(vb));
            int t = x + 128;
            return static_cast<uint8_t>((t + (t >> 8)) >> 8);
        };
        auto blend_f32 = [](float va, float vb, uint8_t mm) -> float {
            float m = static_cast<float>(mm) * (1.0f / 255.0f);
            return m * va + (1.0f - m) * vb;
        };

        // SIMD fast paths where feasible
#if defined(__AVX2__)
        if (std::is_same<Pixel, float>::value) {
            const int step = 8;
            for (int y = 0; y < height; ++y) {
                const float* pa = reinterpret_cast<const float*>(a) + static_cast<size_t>(y) * width * cn;
                const float* pb = reinterpret_cast<const float*>(b) + static_cast<size_t>(y) * width * cn;
                const uint8_t* pm = m + static_cast<size_t>(y) * width;
                float* pd = reinterpret_cast<float*>(d) + static_cast<size_t>(y) * width * cn;
                if (cn == 1) {
                    int x = 0;
                    for (; x + step <= width; x += step) {
                        __m256 va = _mm256_loadu_ps(pa + x);
                        __m256 vb = _mm256_loadu_ps(pb + x);
                        __m256 m8 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(pm + x))));
                        __m256 m01 = _mm256_mul_ps(m8, _mm256_set1_ps(1.0f / 255.0f));
                        __m256 inv = _mm256_sub_ps(_mm256_set1_ps(1.0f), m01);
                        __m256 out = _mm256_add_ps(_mm256_mul_ps(m01, va), _mm256_mul_ps(inv, vb));
                        _mm256_storeu_ps(pd + x, out);
                    }
                    for (; x < width; ++x) pd[x] = blend_f32(pa[x], pb[x], pm[x]);
                } else {
                    for (int x = 0; x < width; ++x) {
                        uint8_t mm = pm[x];
                        int base = x * cn;
                        for (int c = 0; c < cn; ++c) {
                            pd[base + c] = blend_f32(pa[base + c], pb[base + c], mm);
                        }
                    }
                }
            }
            return result;
        }
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        if (std::is_same<Pixel, uint8_t>::value) {
            const uint16x8_t v255 = vdupq_n_u16(255);
            for (int y = 0; y < height; ++y) {
                const uint8_t* pa = reinterpret_cast<const uint8_t*>(a) + static_cast<size_t>(y) * width * cn;
                const uint8_t* pb = reinterpret_cast<const uint8_t*>(b) + static_cast<size_t>(y) * width * cn;
                const uint8_t* pm = m + static_cast<size_t>(y) * width;
                uint8_t* pd = reinterpret_cast<uint8_t*>(d) + static_cast<size_t>(y) * width * cn;

                if (cn == 1) {
                    int x = 0;
                    for (; x + 16 <= width; x += 16) {
                        uint8x16_t va8 = vld1q_u8(pa + x);
                        uint8x16_t vb8 = vld1q_u8(pb + x);
                        uint8x16_t mm8 = vld1q_u8(pm + x);

                        uint8x8_t va8_lo = vget_low_u8(va8), va8_hi = vget_high_u8(va8);
                        uint8x8_t vb8_lo = vget_low_u8(vb8), vb8_hi = vget_high_u8(vb8);
                        uint8x8_t mm8_lo = vget_low_u8(mm8), mm8_hi = vget_high_u8(mm8);

                        uint16x8_t va16_lo = vmovl_u8(va8_lo);
                        uint16x8_t va16_hi = vmovl_u8(va8_hi);
                        uint16x8_t vb16_lo = vmovl_u8(vb8_lo);
                        uint16x8_t vb16_hi = vmovl_u8(vb8_hi);
                        uint16x8_t mm16_lo = vmovl_u8(mm8_lo);
                        uint16x8_t mm16_hi = vmovl_u8(mm8_hi);
                        uint16x8_t mrev_lo = vsubq_u16(v255, mm16_lo);
                        uint16x8_t mrev_hi = vsubq_u16(v255, mm16_hi);

                        uint32x4_t s0 = vaddq_u32(vmull_u16(vget_low_u16(va16_lo), vget_low_u16(mm16_lo)),
                                                   vmull_u16(vget_low_u16(vb16_lo), vget_low_u16(mrev_lo)));
                        uint32x4_t s1 = vaddq_u32(vmull_u16(vget_high_u16(va16_lo), vget_high_u16(mm16_lo)),
                                                   vmull_u16(vget_high_u16(vb16_lo), vget_high_u16(mrev_lo)));
                        uint32x4_t s2 = vaddq_u32(vmull_u16(vget_low_u16(va16_hi), vget_low_u16(mm16_hi)),
                                                   vmull_u16(vget_low_u16(vb16_hi), vget_low_u16(mrev_hi)));
                        uint32x4_t s3 = vaddq_u32(vmull_u16(vget_high_u16(va16_hi), vget_high_u16(mm16_hi)),
                                                   vmull_u16(vget_high_u16(vb16_hi), vget_high_u16(mrev_hi)));

                        uint32x4_t t0 = vaddq_u32(s0, vdupq_n_u32(128));
                        uint32x4_t t1 = vaddq_u32(s1, vdupq_n_u32(128));
                        uint32x4_t t2 = vaddq_u32(s2, vdupq_n_u32(128));
                        uint32x4_t t3 = vaddq_u32(s3, vdupq_n_u32(128));
                        t0 = vaddq_u32(t0, vshrq_n_u32(t0, 8));
                        t1 = vaddq_u32(t1, vshrq_n_u32(t1, 8));
                        t2 = vaddq_u32(t2, vshrq_n_u32(t2, 8));
                        t3 = vaddq_u32(t3, vshrq_n_u32(t3, 8));

                        uint16x8_t r_lo = vcombine_u16(vshrn_n_u32(t0, 8), vshrn_n_u32(t1, 8));
                        uint16x8_t r_hi = vcombine_u16(vshrn_n_u32(t2, 8), vshrn_n_u32(t3, 8));
                        uint8x16_t r8 = vcombine_u8(vqmovn_u16(r_lo), vqmovn_u16(r_hi));
                        vst1q_u8(pd + x, r8);
                    }
                    for (; x < width; ++x) pd[x] = blend_u8(pa[x], pb[x], pm[x]);
                } else if (cn == 3) {
                    int x = 0;
                    for (; x + 16 <= width; x += 16) {
                        uint8x16x3_t a3 = vld3q_u8(pa + x * 3);
                        uint8x16x3_t b3 = vld3q_u8(pb + x * 3);
                        uint8x16_t mm8 = vld1q_u8(pm + x);

                        uint16x8_t mm16_lo = vmovl_u8(vget_low_u8(mm8));
                        uint16x8_t mm16_hi = vmovl_u8(vget_high_u8(mm8));
                        uint16x8_t mrev_lo = vsubq_u16(v255, mm16_lo);
                        uint16x8_t mrev_hi = vsubq_u16(v255, mm16_hi);

                        auto blend_ch = [&](uint8x16_t va, uint8x16_t vb) -> uint8x16_t {
                            // Exact integer formula:
                            // out = (mm*(va - vb) + (vb*255) + 128 + ((... )>>8)) >> 8
                            uint16x8_t va_lo_u16 = vmovl_u8(vget_low_u8(va));
                            uint16x8_t va_hi_u16 = vmovl_u8(vget_high_u8(va));
                            uint16x8_t vb_lo_u16 = vmovl_u8(vget_low_u8(vb));
                            uint16x8_t vb_hi_u16 = vmovl_u8(vget_high_u8(vb));
                            // diff (signed 16)
                            int16x8_t diff_lo = vsubq_s16(vreinterpretq_s16_u16(va_lo_u16),
                                                          vreinterpretq_s16_u16(vb_lo_u16));
                            int16x8_t diff_hi = vsubq_s16(vreinterpretq_s16_u16(va_hi_u16),
                                                          vreinterpretq_s16_u16(vb_hi_u16));
                            // mm as signed 16
                            int16x8_t mm_lo_s = vreinterpretq_s16_u16(mm16_lo);
                            int16x8_t mm_hi_s = vreinterpretq_s16_u16(mm16_hi);
                            // mm*(va-vb) -> s32
                            int32x4_t p0 = vmull_s16(vget_low_s16(diff_lo), vget_low_s16(mm_lo_s));
                            int32x4_t p1 = vmull_s16(vget_high_s16(diff_lo), vget_high_s16(mm_lo_s));
                            int32x4_t p2 = vmull_s16(vget_low_s16(diff_hi), vget_low_s16(mm_hi_s));
                            int32x4_t p3 = vmull_s16(vget_high_s16(diff_hi), vget_high_s16(mm_hi_s));
                            // vb*255 -> u32 then reinterpret to s32
                            uint16x8_t c255 = vdupq_n_u16(255);
                            uint32x4_t vb0 = vmull_u16(vget_low_u16(vb_lo_u16), vget_low_u16(c255));
                            uint32x4_t vb1 = vmull_u16(vget_high_u16(vb_lo_u16), vget_high_u16(c255));
                            uint32x4_t vb2 = vmull_u16(vget_low_u16(vb_hi_u16), vget_low_u16(c255));
                            uint32x4_t vb3 = vmull_u16(vget_high_u16(vb_hi_u16), vget_high_u16(c255));
                            // sum = p + vb*255
                            int32x4_t s0 = vaddq_s32(p0, vreinterpretq_s32_u32(vb0));
                            int32x4_t s1 = vaddq_s32(p1, vreinterpretq_s32_u32(vb1));
                            int32x4_t s2 = vaddq_s32(p2, vreinterpretq_s32_u32(vb2));
                            int32x4_t s3 = vaddq_s32(p3, vreinterpretq_s32_u32(vb3));
                            // rounding divide by 255
                            int32x4_t t0 = vaddq_s32(s0, vdupq_n_s32(128));
                            int32x4_t t1 = vaddq_s32(s1, vdupq_n_s32(128));
                            int32x4_t t2 = vaddq_s32(s2, vdupq_n_s32(128));
                            int32x4_t t3 = vaddq_s32(s3, vdupq_n_s32(128));
                            t0 = vaddq_s32(t0, vshrq_n_s32(t0, 8));
                            t1 = vaddq_s32(t1, vshrq_n_s32(t1, 8));
                            t2 = vaddq_s32(t2, vshrq_n_s32(t2, 8));
                            t3 = vaddq_s32(t3, vshrq_n_s32(t3, 8));
                            // Logical shift to 16-bit (values are non-negative)
                            uint16x8_t r_lo = vcombine_u16(vshrn_n_u32(vreinterpretq_u32_s32(t0), 8),
                                                           vshrn_n_u32(vreinterpretq_u32_s32(t1), 8));
                            uint16x8_t r_hi = vcombine_u16(vshrn_n_u32(vreinterpretq_u32_s32(t2), 8),
                                                           vshrn_n_u32(vreinterpretq_u32_s32(t3), 8));
                            return vcombine_u8(vqmovn_u16(r_lo), vqmovn_u16(r_hi));
                        };

                        uint8x16x3_t out3;
                        out3.val[0] = blend_ch(a3.val[0], b3.val[0]);
                        out3.val[1] = blend_ch(a3.val[1], b3.val[1]);
                        out3.val[2] = blend_ch(a3.val[2], b3.val[2]);
                        vst3q_u8(pd + x * 3, out3);
                    }
                    for (; x < width; ++x) {
                        uint8_t mm = pm[x];
                        int base = x * 3;
                        pd[base + 0] = blend_u8(pa[base + 0], pb[base + 0], mm);
                        pd[base + 1] = blend_u8(pa[base + 1], pb[base + 1], mm);
                        pd[base + 2] = blend_u8(pa[base + 2], pb[base + 2], mm);
                    }
                } else {
                    for (int x = 0; x < width; ++x) {
                        uint8_t mm = pm[x];
                        int base = x * cn;
                        for (int c = 0; c < cn; ++c) {
                            pd[base + c] = blend_u8(pa[base + c], pb[base + c], mm);
                        }
                    }
                }
            }
            return result;
        }
        if (std::is_same<Pixel, float>::value) {
            const float32x4_t inv255 = vdupq_n_f32(1.0f / 255.0f);
            for (int y = 0; y < height; ++y) {
                const float* pa = reinterpret_cast<const float*>(a) + static_cast<size_t>(y) * width * cn;
                const float* pb = reinterpret_cast<const float*>(b) + static_cast<size_t>(y) * width * cn;
                const uint8_t* pm = m + static_cast<size_t>(y) * width;
                float* pd = reinterpret_cast<float*>(d) + static_cast<size_t>(y) * width * cn;
                if (cn == 1) {
                    int x = 0;
                    for (; x + 8 <= width; x += 8) {
                        const uint16x8_t mask16 = vmovl_u8(vld1_u8(pm + x));
                        // first 4
                        {
                            float32x4_t va = vld1q_f32(pa + x);
                            float32x4_t vb = vld1q_f32(pb + x);
                            uint16x4_t mm16l = vget_low_u16(mask16);
                            uint32x4_t mm32 = vmovl_u16(mm16l);
                            float32x4_t m = vcvtq_f32_u32(mm32);
                            float32x4_t m01 = vmulq_f32(m, inv255);
                            float32x4_t inv = vsubq_f32(vdupq_n_f32(1.0f), m01);
                            float32x4_t out = vaddq_f32(vmulq_f32(m01, va), vmulq_f32(inv, vb));
                            vst1q_f32(pd + x, out);
                        }
                        // next 4
                        {
                            float32x4_t va = vld1q_f32(pa + x + 4);
                            float32x4_t vb = vld1q_f32(pb + x + 4);
                            uint16x4_t mm16h = vget_high_u16(mask16);
                            uint32x4_t mm32 = vmovl_u16(mm16h);
                            float32x4_t m = vcvtq_f32_u32(mm32);
                            float32x4_t m01 = vmulq_f32(m, inv255);
                            float32x4_t inv = vsubq_f32(vdupq_n_f32(1.0f), m01);
                            float32x4_t out = vaddq_f32(vmulq_f32(m01, va), vmulq_f32(inv, vb));
                            vst1q_f32(pd + x + 4, out);
                        }
                    }
                    for (; x < width; ++x) pd[x] = blend_f32(pa[x], pb[x], pm[x]);
                } else if (cn == 3) {
                    int x = 0;
                    for (; x + 8 <= width; x += 8) {
                        const uint16x8_t mask16 = vmovl_u8(vld1_u8(pm + x));
                        // first 4 pixels
                        {
                            float32x4x3_t a3 = vld3q_f32(pa + x * 3);
                            float32x4x3_t b3 = vld3q_f32(pb + x * 3);
                            uint16x4_t mm16l = vget_low_u16(mask16);
                            uint32x4_t mm32 = vmovl_u16(mm16l);
                            float32x4_t m = vcvtq_f32_u32(mm32);
                            float32x4_t m01 = vmulq_f32(m, inv255);
                            float32x4_t inv = vsubq_f32(vdupq_n_f32(1.0f), m01);
                            float32x4x3_t o3;
                            o3.val[0] = vaddq_f32(vmulq_f32(m01, a3.val[0]), vmulq_f32(inv, b3.val[0]));
                            o3.val[1] = vaddq_f32(vmulq_f32(m01, a3.val[1]), vmulq_f32(inv, b3.val[1]));
                            o3.val[2] = vaddq_f32(vmulq_f32(m01, a3.val[2]), vmulq_f32(inv, b3.val[2]));
                            vst3q_f32(pd + x * 3, o3);
                        }
                        // next 4 pixels
                        {
                            float32x4x3_t a3 = vld3q_f32(pa + (x + 4) * 3);
                            float32x4x3_t b3 = vld3q_f32(pb + (x + 4) * 3);
                            uint16x4_t mm16h = vget_high_u16(mask16);
                            uint32x4_t mm32 = vmovl_u16(mm16h);
                            float32x4_t m = vcvtq_f32_u32(mm32);
                            float32x4_t m01 = vmulq_f32(m, inv255);
                            float32x4_t inv = vsubq_f32(vdupq_n_f32(1.0f), m01);
                            float32x4x3_t o3;
                            o3.val[0] = vaddq_f32(vmulq_f32(m01, a3.val[0]), vmulq_f32(inv, b3.val[0]));
                            o3.val[1] = vaddq_f32(vmulq_f32(m01, a3.val[1]), vmulq_f32(inv, b3.val[1]));
                            o3.val[2] = vaddq_f32(vmulq_f32(m01, a3.val[2]), vmulq_f32(inv, b3.val[2]));
                            vst3q_f32(pd + (x + 4) * 3, o3);
                        }
                    }
                    // tail handled by scalar below
                    for (; x < width; ++x) {
                        uint8_t mm = pm[x];
                        int base = x * 3;
                        pd[base + 0] = blend_f32(pa[base + 0], pb[base + 0], mm);
                        pd[base + 1] = blend_f32(pa[base + 1], pb[base + 1], mm);
                        pd[base + 2] = blend_f32(pa[base + 2], pb[base + 2], mm);
                    }
                } else {
                    for (int x = 0; x < width; ++x) {
                        uint8_t mm = pm[x];
                        int base = x * cn;
                        for (int c = 0; c < cn; ++c) {
                            pd[base + c] = blend_f32(pa[base + c], pb[base + c], mm);
                        }
                    }
                }
            }
            return result;
        }
#endif

        // Parallelize rows optionally
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int y = 0; y < height; ++y) {
            const Pixel* pa = a + static_cast<size_t>(y) * width * cn;
            const Pixel* pb = b + static_cast<size_t>(y) * width * cn;
            const uint8_t* pm = m + static_cast<size_t>(y) * width;
            Pixel* pd = d + static_cast<size_t>(y) * width * cn;

            if (cn == 1) {
                for (int x = 0; x < width; ++x) {
                    if (std::is_same<Pixel, uint8_t>::value) {
                        pd[x] = static_cast<Pixel>(blend_u8(static_cast<uint8_t>(pa[x]), static_cast<uint8_t>(pb[x]), pm[x]));
                    } else {
                        pd[x] = static_cast<Pixel>(blend_f32(static_cast<float>(pa[x]), static_cast<float>(pb[x]), pm[x]));
                    }
                }
            } else if (cn == 3) {
                for (int x = 0; x < width; ++x) {
                    uint8_t mm = pm[x];
                    int base = x * 3;
                    if (std::is_same<Pixel, uint8_t>::value) {
                        pd[base + 0] = static_cast<Pixel>(blend_u8(static_cast<uint8_t>(pa[base + 0]), static_cast<uint8_t>(pb[base + 0]), mm));
                        pd[base + 1] = static_cast<Pixel>(blend_u8(static_cast<uint8_t>(pa[base + 1]), static_cast<uint8_t>(pb[base + 1]), mm));
                        pd[base + 2] = static_cast<Pixel>(blend_u8(static_cast<uint8_t>(pa[base + 2]), static_cast<uint8_t>(pb[base + 2]), mm));
                    } else {
                        pd[base + 0] = static_cast<Pixel>(blend_f32(static_cast<float>(pa[base + 0]), static_cast<float>(pb[base + 0]), mm));
                        pd[base + 1] = static_cast<Pixel>(blend_f32(static_cast<float>(pa[base + 1]), static_cast<float>(pb[base + 1]), mm));
                        pd[base + 2] = static_cast<Pixel>(blend_f32(static_cast<float>(pa[base + 2]), static_cast<float>(pb[base + 2]), mm));
                    }
                }
            } else if (cn == 4) {
                for (int x = 0; x < width; ++x) {
                    uint8_t mm = pm[x];
                    int base = x * 4;
                    if (std::is_same<Pixel, uint8_t>::value) {
                        pd[base + 0] = static_cast<Pixel>(blend_u8(static_cast<uint8_t>(pa[base + 0]), static_cast<uint8_t>(pb[base + 0]), mm));
                        pd[base + 1] = static_cast<Pixel>(blend_u8(static_cast<uint8_t>(pa[base + 1]), static_cast<uint8_t>(pb[base + 1]), mm));
                        pd[base + 2] = static_cast<Pixel>(blend_u8(static_cast<uint8_t>(pa[base + 2]), static_cast<uint8_t>(pb[base + 2]), mm));
                        pd[base + 3] = static_cast<Pixel>(blend_u8(static_cast<uint8_t>(pa[base + 3]), static_cast<uint8_t>(pb[base + 3]), mm));
                    } else {
                        pd[base + 0] = static_cast<Pixel>(blend_f32(static_cast<float>(pa[base + 0]), static_cast<float>(pb[base + 0]), mm));
                        pd[base + 1] = static_cast<Pixel>(blend_f32(static_cast<float>(pa[base + 1]), static_cast<float>(pb[base + 1]), mm));
                        pd[base + 2] = static_cast<Pixel>(blend_f32(static_cast<float>(pa[base + 2]), static_cast<float>(pb[base + 2]), mm));
                        pd[base + 3] = static_cast<Pixel>(blend_f32(static_cast<float>(pa[base + 3]), static_cast<float>(pb[base + 3]), mm));
                    }
                }
            } else {
                for (int x = 0; x < width; ++x) {
                    uint8_t mm = pm[x];
                    int base = x * cn;
                    for (int c = 0; c < cn; ++c) {
                        if (std::is_same<Pixel, uint8_t>::value) {
                            pd[base + c] = static_cast<Pixel>(blend_u8(static_cast<uint8_t>(pa[base + c]), static_cast<uint8_t>(pb[base + c]), mm));
                        } else {
                            pd[base + c] = static_cast<Pixel>(blend_f32(static_cast<float>(pa[base + c]), static_cast<float>(pb[base + c]), mm));
                        }
                    }
                }
            }
        }

        return result;
    }

    // Drawing operations
    void DrawLine(const Point<int>& p1, const Point<int>& p2, const std::vector<double>& color,
                  int thickness = 1) {
        okcv::Point2i start = *static_cast<okcv::Point2i*>(p1.GetInternalPoint());
        okcv::Point2i end = *static_cast<okcv::Point2i*>(p2.GetInternalPoint());
        std::vector<Pixel> okcv_color;
        for (const auto& c : color) {
            okcv_color.push_back(static_cast<Pixel>(c));
        }
        bitmap_.DrawLine(start, end, okcv_color, thickness);
    }

    void DrawRect(const Rect<int>& rect, const std::vector<double>& color, int thickness = 1) {
        okcv::Rect2i okcv_rect = *static_cast<okcv::Rect2i*>(rect.GetInternalRect());
        std::vector<Pixel> okcv_color;
        for (const auto& c : color) {
            okcv_color.push_back(static_cast<Pixel>(c));
        }
        bitmap_.DrawRect(okcv_rect, okcv_color, thickness);
    }

    void DrawCircle(const Point<int>& center, int radius, const std::vector<double>& color,
                    int thickness = 1) {
        okcv::Point2f center_point =
          okcv::Point2f(static_cast<okcv::Point2i*>(center.GetInternalPoint())->x,
                        static_cast<okcv::Point2i*>(center.GetInternalPoint())->y);
        std::vector<Pixel> okcv_color;
        for (const auto& c : color) {
            okcv_color.push_back(static_cast<Pixel>(c));
        }
        bitmap_.DrawCircle(center_point, static_cast<float>(radius), okcv_color, thickness);
    }

    void Fill(const Rect<int>& rect, const std::vector<double>& color) {
        okcv::Rect2i okcv_rect = *static_cast<okcv::Rect2i*>(rect.GetInternalRect());
        std::vector<Pixel> okcv_color;
        for (const auto& c : color) {
            okcv_color.push_back(static_cast<Pixel>(c));
        }
        bitmap_.FillRect(okcv_rect, okcv_color);
    }

    // Image format conversion
    ImageT<Pixel> ToGray() const {
        ImageT<Pixel> result;
        result.impl_->bitmap_ = bitmap_.RgbToGray();
        return result;
    }

    void Print(std::ostream& os) const {
        const int N = 10;  // Threshold for truncated display
        if (bitmap_.Height() > N || bitmap_.Width() > N) {
            // For large matrices, show truncated view
            os << "[";
            for (int i = 0; i < std::min(3, bitmap_.Height()); i++) {
                if (i > 0)
                    os << " ";
                os << "[";
                // Show first 3 elements
                for (int j = 0; j < std::min(3, bitmap_.Width()); j++) {
                    if (j > 0)
                        os << " ";
                    if (bitmap_.Channels() == 1) {
                        os << (int)*(bitmap_.at(i, j));
                    } else {
                        os << "[";
                        for (int c = 0; c < bitmap_.Channels(); c++) {
                            if (c > 0)
                                os << " ";
                            os << (int)*(bitmap_.at(i, j) + c);
                        }
                        os << "]";
                    }
                }
                if (bitmap_.Width() > 3)
                    os << " ... ";
                // Show last 3 elements if there are more columns
                if (bitmap_.Width() > 6) {
                    for (int j = bitmap_.Width() - 3; j < bitmap_.Width(); j++) {
                        if (bitmap_.Channels() == 1) {
                            os << (int)*(bitmap_.at(i, j)) << " ";
                        } else {
                            os << "[";
                            for (int c = 0; c < bitmap_.Channels(); c++) {
                                if (c > 0)
                                    os << " ";
                                os << (int)*(bitmap_.at(i, j) + c);
                            }
                            os << "] ";
                        }
                    }
                }
                os << "]\n";
            }
            if (bitmap_.Height() > 6) {
                os << "...\n";
                // Show last 3 rows
                for (int i = bitmap_.Height() - 3; i < bitmap_.Height(); i++) {
                    os << "[";
                    for (int j = 0; j < std::min(3, bitmap_.Width()); j++) {
                        if (j > 0)
                            os << " ";
                        if (bitmap_.Channels() == 1) {
                            os << (int)*(bitmap_.at(i, j));
                        } else {
                            os << "[";
                            for (int c = 0; c < bitmap_.Channels(); c++) {
                                if (c > 0)
                                    os << " ";
                                os << (int)*(bitmap_.at(i, j) + c);
                            }
                            os << "]";
                        }
                    }
                    if (bitmap_.Width() > 3)
                        os << " ... ";
                    if (bitmap_.Width() > 6) {
                        for (int j = bitmap_.Width() - 3; j < bitmap_.Width(); j++) {
                            if (bitmap_.Channels() == 1) {
                                os << (int)*(bitmap_.at(i, j)) << " ";
                            } else {
                                os << "[";
                                for (int c = 0; c < bitmap_.Channels(); c++) {
                                    if (c > 0)
                                        os << " ";
                                    os << (int)*(bitmap_.at(i, j) + c);
                                }
                                os << "] ";
                            }
                        }
                    }
                    os << "]\n";
                }
            }
            os << "]\n";
        } else {
            // For small matrices, show full content
            os << "[";
            for (int i = 0; i < bitmap_.Height(); i++) {
                if (i > 0)
                    os << " ";
                os << "[";
                for (int j = 0; j < bitmap_.Width(); j++) {
                    if (j > 0)
                        os << " ";
                    if (bitmap_.Channels() == 1) {
                        os << (int)*(bitmap_.at(i, j));
                    } else {
                        os << "[";
                        for (int c = 0; c < bitmap_.Channels(); c++) {
                            if (c > 0)
                                os << " ";
                            os << (int)*(bitmap_.at(i, j) + c);
                        }
                        os << "]";
                    }
                }
                os << "]\n";
            }
            os << "]\n";
        }
        os << "Size(H x W x C): " << bitmap_.Height() << " x " << bitmap_.Width() << " x "
           << bitmap_.Channels() << "\n";
    }

private:
    bool TryCudaGeometry(ImageT<Pixel>& result,
                         internal::ImageGeometryOperation operation,
                         int destination_width, int destination_height,
                         const float* affine) const {
        if (bitmap_.Empty() || destination_width <= 0 ||
            destination_height <= 0 || bitmap_.Channels() <= 0) {
            return false;
        }
        if ((operation == internal::ImageGeometryOperation::kResizeNearest ||
             operation == internal::ImageGeometryOperation::kResizeBilinear) &&
            destination_width == bitmap_.Width() &&
            destination_height == bitmap_.Height()) {
            return false;
        }

        result.impl_->bitmap_.Reset(destination_width, destination_height,
                                    bitmap_.Channels());
        internal::ImageGeometryRequest request;
        request.source = bitmap_.Data();
        request.destination = result.impl_->bitmap_.Data();
        request.source_width = bitmap_.Width();
        request.source_height = bitmap_.Height();
        request.destination_width = destination_width;
        request.destination_height = destination_height;
        request.channels = bitmap_.Channels();
        request.element_type = std::is_same<Pixel, float>::value
                                 ? internal::ImageElementType::kFloat32
                                 : internal::ImageElementType::kUInt8;
        request.operation = operation;
        if (affine) {
            std::copy(affine, affine + 6, request.affine);
        }
        return internal::TryExecuteAcceleratedImageGeometry(request);
    }

    okcv::Bitmap<Pixel> bitmap_;
};

}  // namespace inspirecv

#endif  // INSPIRECV_BACKENDS_OKCV_ADAPTER_IMAGE_BACKEND_H_
