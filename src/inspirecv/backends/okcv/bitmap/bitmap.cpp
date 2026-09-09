
#include <cmath>
#include <cstdlib>
#include <string>
#include <utility>
#include <cstring>
#include <sstream>
#include <algorithm>
#include <vector>
#include <limits>
#include <type_traits>
#include <thread>
#include <deque>

#if defined(__has_include)
#  if (defined(__ARM_NEON) || defined(__ARM_NEON__)) && __has_include(<arm_neon.h>)
#    include <arm_neon.h>
#  endif
#  if (defined(__AVX2__) || defined(__SSE2__)) && __has_include(<immintrin.h>)
#    include <immintrin.h>
#  endif
#else
#  if defined(__ARM_NEON) || defined(__ARM_NEON__)
#    include <arm_neon.h>
#  endif
#  if defined(__AVX2__) || defined(__SSE2__)
#    include <immintrin.h>
#  endif
#endif

#include "bitmap.h"
#include "check.h"
#include "inspirecv/backends/okcv/kernels/x86/u8c3_ops.h"

#ifndef INSPIRECV_BACKEND_OKCV_USE_OPENCV_IO
#include "inspirecv/backends/okcv/io/stb_wrapper.h"
#endif

namespace okcv {

#if defined(__SSSE3__)
namespace {

const bool kX86U8C3KernelsEnabled = []() {
    const char* disabled =
      std::getenv("INSPIRECV_INTERNAL_DISABLE_X86_U8C3_KERNELS");
    return disabled == nullptr || std::strcmp(disabled, "1") != 0;
}();

}  // namespace
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
namespace {

inline uint8x16_t ZipLowerBytes(uint8x16_t first, uint8x16_t second) {
#if defined(__aarch64__)
    return vzip1q_u8(first, second);
#else
    return vzipq_u8(first, second).val[0];
#endif
}

inline uint8x16_t ZipUpperBytes(uint8x16_t first, uint8x16_t second) {
#if defined(__aarch64__)
    return vzip2q_u8(first, second);
#else
    return vzipq_u8(first, second).val[1];
#endif
}

inline float32x4_t ZipLowerFloats(float32x4_t first, float32x4_t second) {
#if defined(__aarch64__)
    return vzip1q_f32(first, second);
#else
    return vzipq_f32(first, second).val[0];
#endif
}

inline float32x4_t ZipUpperFloats(float32x4_t first, float32x4_t second) {
#if defined(__aarch64__)
    return vzip2q_f32(first, second);
#else
    return vzipq_f32(first, second).val[1];
#endif
}

}  // namespace
#endif

template <typename D>
Bitmap<D>::Bitmap(Bitmap &&that) {
    data_ = std::move(that.data_);
    external_data_ = that.external_data_;
    is_external_ = that.is_external_;
    height_ = that.height_;
    width_ = that.width_;
    channels_ = that.channels_;

    that.height_ = 0;
    that.width_ = 0;
    that.channels_ = 0;
    that.external_data_ = nullptr;
    that.is_external_ = false;
}

template <typename D>
Bitmap<D> &Bitmap<D>::operator=(Bitmap &&that) {
    if (this == &that) return *this;
    data_ = std::move(that.data_);
    external_data_ = that.external_data_;
    is_external_ = that.is_external_;
    height_ = that.height_;
    width_ = that.width_;
    channels_ = that.channels_;

    that.height_ = 0;
    that.width_ = 0;
    that.channels_ = 0;
    that.external_data_ = nullptr;
    that.is_external_ = false;
    return *this;
}

template <typename D>
void Bitmap<D>::Reset() {
    height_ = 0;
    width_ = 0;
    channels_ = 0;
    data_.reset();
    external_data_ = nullptr;
    is_external_ = false;
}

template <typename D>
void Bitmap<D>::Reset(int width, int height, int channels, const D *data, bool copy_data) {
    if (width <= 0 || height <= 0 || channels <= 0) {
        Reset();
        return;
    }

    const size_t element_count = static_cast<size_t>(width) *
                                 static_cast<size_t>(height) *
                                 static_cast<size_t>(channels);
    INSPIRECV_CHECK(element_count <= static_cast<size_t>(std::numeric_limits<int>::max()))
      << "Bitmap dimensions exceed the supported element count: "
      << width << "x" << height << "x" << channels;

    if (data == nullptr || copy_data) {
        const int new_size = static_cast<int>(element_count);
        if (is_external_ || DataSize() != new_size || !data_) {
            data_.reset(new D[element_count]);
        }
        height_ = height;
        width_ = width;
        channels_ = channels;
        is_external_ = false;
        external_data_ = nullptr;

        if (data != nullptr) {
            std::memcpy(data_.get(), data, sizeof(D) * element_count);
        }
    } else {
        // External data path
        data_.reset();  // Release any owned data
        external_data_ = data;
        is_external_ = true;
        height_ = height;
        width_ = width;
        channels_ = channels;
    }
}

template <typename D>
Bitmap<D> Bitmap<D>::Clone() const {
    Bitmap<D> dst;
    dst.Reset(width_, height_, channels_, Data());
    return dst;
}

template <typename D>
void Bitmap<D>::CopyTo(Bitmap<D> &dst) const {
    if (this != &dst) {
        INSPIRECV_CHECK(!Empty());
        dst.Reset(width_, height_, channels_, Data());
    }
}

template <typename D>
void Bitmap<D>::Read(const char *filename, int channels) {
#ifdef INSPIRECV_BACKEND_OKCV_USE_OPENCV_IO
    INSPIRECV_CHECK(channels == 1 || channels == 3);
    int flag = channels == 3 ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE;
    cv::Mat image = cv::imread(filename, flag);
    INSPIRECV_CHECK(image.data) << "Could not open file " << filename;
    this->FromCVMat(image, false);
#else
    ImageReader::ImageData data;
    bool succ = ImageReader::Read(filename, data, channels, ImageReader::ColorOrder::BGR);
    INSPIRECV_CHECK(succ) << "Could not open file " << filename;
    // The default BGR is used as with OpenCV
    if (std::is_same<D, uint8_t>::value) {
        // direct copy for U8
        this->Reset(data.width, data.height, data.channels,
                    reinterpret_cast<const D*>(data.data.data()));
    } else {
        // convert bytes -> float (0..255 range)
        const size_t count = static_cast<size_t>(data.width) * data.height * data.channels;
        std::vector<D> buffer(count);
        const unsigned char* src = data.data.data();

        if (std::is_same<D, float>::value) {
            // SIMD-accelerated u8 -> f32 conversion
            float* dst = reinterpret_cast<float*>(buffer.data());
            size_t i = 0;
#if defined(__AVX2__)
            for (; i + 32 <= count; i += 32) {
                __m128i m0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
                __m128i m1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i + 16));

                __m256i u16_0 = _mm256_cvtepu8_epi16(m0); // 16 x u16
                __m256i u16_1 = _mm256_cvtepu8_epi16(m1); // 16 x u16

                __m128i u16_0_lo = _mm256_castsi256_si128(u16_0);
                __m128i u16_0_hi = _mm256_extracti128_si256(u16_0, 1);
                __m128i u16_1_lo = _mm256_castsi256_si128(u16_1);
                __m128i u16_1_hi = _mm256_extracti128_si256(u16_1, 1);

                __m256i u32_00 = _mm256_cvtepu16_epi32(u16_0_lo);
                __m256i u32_01 = _mm256_cvtepu16_epi32(u16_0_hi);
                __m256i u32_10 = _mm256_cvtepu16_epi32(u16_1_lo);
                __m256i u32_11 = _mm256_cvtepu16_epi32(u16_1_hi);

                __m256 f00 = _mm256_cvtepi32_ps(u32_00);
                __m256 f01 = _mm256_cvtepi32_ps(u32_01);
                __m256 f10 = _mm256_cvtepi32_ps(u32_10);
                __m256 f11 = _mm256_cvtepi32_ps(u32_11);

                _mm256_storeu_ps(dst + i + 0, f00);
                _mm256_storeu_ps(dst + i + 8, f01);
                _mm256_storeu_ps(dst + i + 16, f10);
                _mm256_storeu_ps(dst + i + 24, f11);
            }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
            for (; i + 16 <= count; i += 16) {
                uint8x16_t v = vld1q_u8(src + i);
                uint8x8_t vlo = vget_low_u8(v);
                uint8x8_t vhi = vget_high_u8(v);

                uint16x8_t u16lo = vmovl_u8(vlo);
                uint16x8_t u16hi = vmovl_u8(vhi);

                uint32x4_t u32_0 = vmovl_u16(vget_low_u16(u16lo));
                uint32x4_t u32_1 = vmovl_u16(vget_high_u16(u16lo));
                uint32x4_t u32_2 = vmovl_u16(vget_low_u16(u16hi));
                uint32x4_t u32_3 = vmovl_u16(vget_high_u16(u16hi));

                float32x4_t f0 = vcvtq_f32_u32(u32_0);
                float32x4_t f1 = vcvtq_f32_u32(u32_1);
                float32x4_t f2 = vcvtq_f32_u32(u32_2);
                float32x4_t f3 = vcvtq_f32_u32(u32_3);

                vst1q_f32(dst + i + 0, f0);
                vst1q_f32(dst + i + 4, f1);
                vst1q_f32(dst + i + 8, f2);
                vst1q_f32(dst + i + 12, f3);
            }
#endif
            for (; i < count; ++i) dst[i] = static_cast<float>(src[i]);
        } else {
            for (size_t i = 0; i < count; ++i) buffer[i] = static_cast<D>(src[i]);
        }
        this->Reset(data.width, data.height, data.channels, buffer.data(), true);
    }
#endif  // INSPIRECV_BACKEND_OKCV_USE_OPENCV_IO
}

template <typename D>
void Bitmap<D>::Read(const std::string &filename, int channels) {
    Read(filename.c_str(), channels);
}

template <typename D>
void Bitmap<D>::Write(const char *filename) const {
#ifdef INSPIRECV_BACKEND_OKCV_USE_OPENCV_IO
    cv::Mat mat;
    this->ToCVMat(mat, false);
    cv::imwrite(filename, mat);
#else
    ImageReader::WriteConfig config;
    config.jpg_quality = 100;
    config.png_compression = 9;
    config.color_order = ImageReader::ColorOrder::BGR;

    if (std::is_same<D, uint8_t>::value) {
        bool succ = ImageReader::Write(filename, reinterpret_cast<const unsigned char*>(Data()),
                                       Width(), Height(), Channels(), config);
        INSPIRECV_CHECK(succ) << "Could not write file " << filename;
    } else {
        // convert float -> U8 with saturation
        const size_t count = static_cast<size_t>(Width()) * Height() * Channels();
        std::vector<unsigned char> tmp(count);
        const D* src = Data();
        for (size_t i = 0; i < count; ++i) {
            double v = static_cast<double>(src[i]);
            if (v < 0.0) v = 0.0; else if (v > 255.0) v = 255.0;
            tmp[i] = static_cast<unsigned char>(std::floor(v + 0.5));
        }
        bool succ = ImageReader::Write(filename, tmp.data(), Width(), Height(), Channels(), config);
        INSPIRECV_CHECK(succ) << "Could not write file " << filename;
    }
#endif  // INSPIRECV_BACKEND_OKCV_USE_OPENCV_IO
}

template <typename D>
void Bitmap<D>::Write(const std::string &filename) const {
    Write(filename.c_str());
}

template <typename D>
void Bitmap<D>::FromImageBuffer(const std::vector<char> &buffer, int channels) {
#ifdef INSPIRECV_BACKEND_OKCV_USE_OPENCV_IO
    INSPIRECV_CHECK(channels == 1 || channels == 3);
    int flag = channels == 3 ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE;
    cv::Mat image = cv::imdecode(buffer, flag);
    INSPIRECV_CHECK(image.data) << "Decode image error!";
    this->FromCVMat(image, false);
#else
    INSPIRECV_LOG(FATAL) << "Not implemented okcv::Bitmap::FromImageBuffer!";
#endif  // INSPIRECV_BACKEND_OKCV_USE_OPENCV_IO
}

template <typename D>
void Bitmap<D>::Show(const std::string &name, int time) const {
#ifdef INSPIRECV_BACKEND_OKCV_USE_OPENCV_GUI
    cv::Mat mat;
    this->ToCVMat(mat, false);
    mat.convertTo(mat, CV_8U);
    cv::imshow(name, mat);
    cv::waitKey(time);
#else
    INSPIRECV_LOG(FATAL) << "Not implemented okcv::Bitmap::Show!";
#endif
}

template <typename D>
void Bitmap<D>::Fill(D v) {
    std::fill_n(Data(), DataSize(), v);
}

template <typename D>
Bitmap<D> Bitmap<D>::Mul(float a) const {
    Bitmap<D> res;
    res.Reset(width_, height_, channels_);
    auto res_iter = res.Data();
    for (int i = 0; i < DataSize(); ++i) {
        *res_iter++ = Data()[i] * a;
    }
    return res;
}

template <typename D>
Bitmap<D> Bitmap<D>::MulAdd(float a, float b) const {
    Bitmap<D> res;
    res.Reset(width_, height_, channels_);
    auto res_iter = res.Data();
    for (int i = 0; i < DataSize(); ++i) {
        const float value = Data()[i] * a + b;
        // Convert through a signed integer so uint8 wraparound is defined and
        // consistent across architectures and with the OpenCV backend.
        if (std::is_same<D, uint8_t>::value) {
            *res_iter++ = static_cast<D>(static_cast<int>(value));
        } else {
            *res_iter++ = static_cast<D>(value);
        }
    }
    return res;
}

template <typename D>
Bitmap<D> Bitmap<D>::ElementWiseOperate(const Bitmap<D> &image,
                                      const std::function<D(D, D)> &op) const {
    Bitmap<D> res;
    INSPIRECV_CHECK(Width() == image.Width())
      << "width=" << Width() << ", image.width=" << image.Width();
    INSPIRECV_CHECK(Height() == image.Height())
      << "height=" << Height() << ", image.height=" << image.Height();
    INSPIRECV_CHECK(Channels() == image.Channels())
      << "channels=" << Channels() << ", image.channels=" << image.Channels();
    res.Reset(width_, height_, channels_);
    auto res_iter = res.Data();
    auto iter1 = Data();
    auto iter2 = image.Data();
    for (int i = 0; i < DataSize(); ++i) {
        *res_iter = op(*iter1, *iter2);
        ++res_iter;
        ++iter1;
        ++iter2;
    }
    return res;
}

template <typename D>
void Bitmap<D>::ApplyPixelwiseOperation(const std::function<D(D)> &func) {
    auto iter = Data();
    for (int i = 0; i < DataSize(); ++i) {
        *iter = func(*iter);
        ++iter;
    }
}

template <>
inline float Bitmap<float>::InterpolateBilinear(const float top_left, const float top_right,
                                               const float bottom_left, const float bottom_right,
                                               const float x_lerp, const float y_lerp) const {
    const float top = top_left + (top_right - top_left) * x_lerp;
    const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
    return top + (bottom - top) * y_lerp;
}

template <>
inline uint8_t Bitmap<uint8_t>::InterpolateBilinear(const uint8_t top_left, const uint8_t top_right,
                                                   const uint8_t bottom_left,
                                                   const uint8_t bottom_right, const float x_lerp,
                                                   const float y_lerp) const {
    const float top = top_left + (top_right - top_left) * x_lerp;
    const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
    return static_cast<uint8_t>(roundf(top + (bottom - top) * y_lerp));
}

template <>
inline void Bitmap<uint8_t>::OutInterpolateBilinearFloatx4x1Neon(
  const uint8_t *top_left_0, const uint8_t *top_left_1, const uint8_t *top_left_2,
  const uint8_t *top_left_3, const uint8_t *top_right_0, const uint8_t *top_right_1,
  const uint8_t *top_right_2, const uint8_t *top_right_3, const uint8_t *bottom_left_0,
  const uint8_t *bottom_left_1, const uint8_t *bottom_left_2, const uint8_t *bottom_left_3,
  const uint8_t *bottom_right_0, const uint8_t *bottom_right_1, const uint8_t *bottom_right_2,
  const uint8_t *bottom_right_3, const float *x_lerp, const float *y_lerp, uint8_t *out) const {
    INSPIRECV_LOG(ERROR) << "OutComputeLerpFloatx4x1Neon no support";
}

template <>
inline void Bitmap<float>::OutInterpolateBilinearFloatx4x1Neon(
  const float *top_left_0, const float *top_left_1, const float *top_left_2,
  const float *top_left_3, const float *top_right_0, const float *top_right_1,
  const float *top_right_2, const float *top_right_3, const float *bottom_left_0,
  const float *bottom_left_1, const float *bottom_left_2, const float *bottom_left_3,
  const float *bottom_right_0, const float *bottom_right_1, const float *bottom_right_2,
  const float *bottom_right_3, const float *x_lerp, const float *y_lerp, float *out) const {
#if (defined(__ARM_NEON__) || defined(__ARM_NEON))
    static const float32x4_t v_zero = vmovq_n_f32(0.0);
    float32x4_t v_top_left, v_top_right, v_bottom_left, v_bottom_right;
    float32x4_t v_x_lerp = vld1q_f32(x_lerp);
    float32x4_t v_y_lerp = vld1q_f32(y_lerp);
    v_top_left = vld1q_lane_f32(top_left_0, v_zero, 0);
    v_top_left = vld1q_lane_f32(top_left_1, v_top_left, 1);
    v_top_left = vld1q_lane_f32(top_left_2, v_top_left, 2);
    v_top_left = vld1q_lane_f32(top_left_3, v_top_left, 3);

    v_top_right = vld1q_lane_f32(top_right_0, v_zero, 0);
    v_top_right = vld1q_lane_f32(top_right_1, v_top_right, 1);
    v_top_right = vld1q_lane_f32(top_right_2, v_top_right, 2);
    v_top_right = vld1q_lane_f32(top_right_3, v_top_right, 3);

    v_bottom_left = vld1q_lane_f32(bottom_left_0, v_zero, 0);
    v_bottom_left = vld1q_lane_f32(bottom_left_1, v_bottom_left, 1);
    v_bottom_left = vld1q_lane_f32(bottom_left_2, v_bottom_left, 2);
    v_bottom_left = vld1q_lane_f32(bottom_left_3, v_bottom_left, 3);

    v_bottom_right = vld1q_lane_f32(bottom_right_0, v_zero, 0);
    v_bottom_right = vld1q_lane_f32(bottom_right_1, v_bottom_right, 1);
    v_bottom_right = vld1q_lane_f32(bottom_right_2, v_bottom_right, 2);
    v_bottom_right = vld1q_lane_f32(bottom_right_3, v_bottom_right, 3);

    float32x4_t v_sub = vsubq_f32(v_top_right, v_top_left);
    float32x4_t v_top = vaddq_f32(v_top_left, vmulq_f32(v_sub, v_x_lerp));
    float32x4_t v_sub_b = vsubq_f32(v_bottom_right, v_bottom_left);
    float32x4_t v_bottom = vaddq_f32(v_bottom_left, vmulq_f32(v_sub_b, v_x_lerp));
    float32x4_t v_ret_sub = vsubq_f32(v_bottom, v_top);
    float32x4_t v_ret = vaddq_f32(v_top, vmulq_f32(v_ret_sub, v_y_lerp));

    vst1q_f32(out, v_ret);
#endif
    return;
}

template <typename D>
Bitmap<D> Bitmap<D>::ResizeBilinear(int width, int height) const {
    INSPIRECV_CHECK(height > 0 && width > 0) << "height=" << height << ", width=" << width;
    if (height_ == height && width_ == width) {
        return Clone();
    }

    Bitmap<D> dst;
    dst.Reset(width, height, channels_);
    const float height_scale = static_cast<float>(height_) / height;
    const float width_scale = static_cast<float>(width_) / width;
    std::vector<int> in_x_low(width);
    std::vector<int> in_x_up(width);
    std::vector<float> lerp_x(width);
    for (int x = 0; x < width; ++x) {
        float in_x = x * width_scale;
        in_x_low[x] = std::min(static_cast<int>(in_x), width_ - 1);
        in_x_up[x] = std::min(in_x_low[x] + 1, width_ - 1);
        lerp_x[x] = in_x - in_x_low[x];
    }

    auto dst_iter = dst.Data();
#if defined(__AVX2__)
    // AVX2 fast path: float single-channel
    if (channels_ == 1 && std::is_same<D, float>::value) {
        const __m256 ones = _mm256_set1_ps(1.0f);
        for (int y = 0; y < height; ++y) {
            float in_y = y * height_scale;
            int in_y_low = std::min(static_cast<int>(in_y), height_ - 1);
            int in_y_up = std::min(in_y_low + 1, height_ - 1);
            float lerp_y = in_y - in_y_low;
            __m256 ay = _mm256_set1_ps(lerp_y);
            const float* row_low = reinterpret_cast<const float*>(Row(in_y_low));
            const float* row_up  = reinterpret_cast<const float*>(Row(in_y_up));
            float* out_row = reinterpret_cast<float*>(dst.Row(y));

            int x = 0;
            for (; x + 8 <= width; x += 8) {
                __m256 ax = _mm256_loadu_ps(&lerp_x[x]);
                __m256i vidxL = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&in_x_low[x]));
                __m256i vidxU = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&in_x_up[x]));
                __m256 tl = _mm256_i32gather_ps(row_low, vidxL, 4);
                __m256 tr = _mm256_i32gather_ps(row_low, vidxU, 4);
                __m256 bl = _mm256_i32gather_ps(row_up,  vidxL, 4);
                __m256 br = _mm256_i32gather_ps(row_up,  vidxU, 4);
                __m256 top = _mm256_fmadd_ps(_mm256_sub_ps(tr, tl), ax, tl);
                __m256 bottom = _mm256_fmadd_ps(_mm256_sub_ps(br, bl), ax, bl);
                __m256 outv = _mm256_fmadd_ps(_mm256_sub_ps(bottom, top), ay, top);
                _mm256_storeu_ps(out_row + x, outv);
            }
            for (; x < width; ++x) {
                const float tl = *(Row(in_y_low) + in_x_low[x]);
                const float tr = *(Row(in_y_low) + in_x_up[x]);
                const float bl = *(Row(in_y_up)  + in_x_low[x]);
                const float br = *(Row(in_y_up)  + in_x_up[x]);
                float top = tl + (tr - tl) * lerp_x[x];
                float bottom = bl + (br - bl) * lerp_x[x];
                out_row[x] = top + (bottom - top) * lerp_y;
            }
        }
        return dst;
    }

    // AVX2 fast path: float 3-channel (interleaved), compute per-channel
    if (channels_ == 3 && std::is_same<D, float>::value) {
        for (int y = 0; y < height; ++y) {
            float in_y = y * height_scale;
            int in_y_low = std::min(static_cast<int>(in_y), height_ - 1);
            int in_y_up = std::min(in_y_low + 1, height_ - 1);
            float lerp_y = in_y - in_y_low;
            __m256 ay = _mm256_set1_ps(lerp_y);
            const float* row_low = reinterpret_cast<const float*>(Row(in_y_low));
            const float* row_up  = reinterpret_cast<const float*>(Row(in_y_up));
            float* out_row = reinterpret_cast<float*>(dst.Row(y));

            int x = 0;
            alignas(32) int idxL[8];
            alignas(32) int idxU[8];
            for (; x + 8 <= width; x += 8) {
                __m256 ax = _mm256_loadu_ps(&lerp_x[x]);
                // lanes' base indices for interleaved 3ch
                for (int i = 0; i < 8; ++i) {
                    idxL[i] = in_x_low[x + i] * 3;
                    idxU[i] = in_x_up[x + i] * 3;
                }
                // do per channel c=0,1,2
                for (int c = 0; c < 3; ++c) {
                    // gather neighbors
                    __m256i vL = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(idxL));
                    __m256i vU = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(idxU));
                    __m256 tl = _mm256_i32gather_ps(row_low, vL, 4);
                    __m256 tr = _mm256_i32gather_ps(row_low, vU, 4);
                    __m256 bl = _mm256_i32gather_ps(row_up,  vL, 4);
                    __m256 br = _mm256_i32gather_ps(row_up,  vU, 4);
                    // top/bottom
                    __m256 top = _mm256_fmadd_ps(_mm256_sub_ps(tr, tl), ax, tl);
                    __m256 bottom = _mm256_fmadd_ps(_mm256_sub_ps(br, bl), ax, bl);
                    __m256 outv = _mm256_fmadd_ps(_mm256_sub_ps(bottom, top), ay, top);
                    // scatter (no AVX2 scatter, do per-lane)
                    float tmp[8];
                    _mm256_store_ps(tmp, outv);
                    for (int i = 0; i < 8; ++i) {
                        out_row[(x + i) * 3 + c] = tmp[i];
                        // advance indices for next channel
                        idxL[i] += 1;
                        idxU[i] += 1;
                    }
                }
            }
            // tail
            for (; x < width; ++x) {
                int l = in_x_low[x] * 3;
                int u = in_x_up[x] * 3;
                for (int c = 0; c < 3; ++c) {
                    float tl = row_low[l + c];
                    float tr = row_low[u + c];
                    float bl = row_up[l + c];
                    float br = row_up[u + c];
                    float top = tl + (tr - tl) * lerp_x[x];
                    float bottom = bl + (br - bl) * lerp_x[x];
                    out_row[x * 3 + c] = top + (bottom - top) * lerp_y;
                }
            }
        }
        return dst;
    }
#endif
#if defined(__SSE2__)
    // SSE2 fast path: uint8 single-channel, 4 pixels per step (compute in float, round to nearest)
    if (channels_ == 1 && std::is_same<D, uint8_t>::value) {
        for (int y = 0; y < height; ++y) {
            float in_y = y * height_scale;
            int in_y_low = std::min(static_cast<int>(in_y), height_ - 1);
            int in_y_up = std::min(in_y_low + 1, height_ - 1);
            float lerp_y = in_y - in_y_low;
            const uint8_t* row_low = reinterpret_cast<const uint8_t*>(Row(in_y_low));
            const uint8_t* row_up  = reinterpret_cast<const uint8_t*>(Row(in_y_up));
            uint8_t* out_row = reinterpret_cast<uint8_t*>(dst.Row(y));

            __m128 vy = _mm_set1_ps(lerp_y);
            int x = 0;
            for (; x + 4 <= width; x += 4) {
                float tl0 = static_cast<float>(row_low[in_x_low[x + 0]]);
                float tr0 = static_cast<float>(row_low[in_x_up [x + 0]]);
                float bl0 = static_cast<float>(row_up [in_x_low[x + 0]]);
                float br0 = static_cast<float>(row_up [in_x_up [x + 0]]);
                float tl1 = static_cast<float>(row_low[in_x_low[x + 1]]);
                float tr1 = static_cast<float>(row_low[in_x_up [x + 1]]);
                float bl1 = static_cast<float>(row_up [in_x_low[x + 1]]);
                float br1 = static_cast<float>(row_up [in_x_up [x + 1]]);
                float tl2 = static_cast<float>(row_low[in_x_low[x + 2]]);
                float tr2 = static_cast<float>(row_low[in_x_up [x + 2]]);
                float bl2 = static_cast<float>(row_up [in_x_low[x + 2]]);
                float br2 = static_cast<float>(row_up [in_x_up [x + 2]]);
                float tl3 = static_cast<float>(row_low[in_x_low[x + 3]]);
                float tr3 = static_cast<float>(row_low[in_x_up [x + 3]]);
                float bl3 = static_cast<float>(row_up [in_x_low[x + 3]]);
                float br3 = static_cast<float>(row_up [in_x_up [x + 3]]);

                __m128 vtl = _mm_set_ps(tl3, tl2, tl1, tl0);
                __m128 vtr = _mm_set_ps(tr3, tr2, tr1, tr0);
                __m128 vbl = _mm_set_ps(bl3, bl2, bl1, bl0);
                __m128 vbr = _mm_set_ps(br3, br2, br1, br0);
                __m128 vx  = _mm_loadu_ps(&lerp_x[x]);

                __m128 vtop = _mm_add_ps(vtl, _mm_mul_ps(_mm_sub_ps(vtr, vtl), vx));
                __m128 vbot = _mm_add_ps(vbl, _mm_mul_ps(_mm_sub_ps(vbr, vbl), vx));
                __m128 vout = _mm_add_ps(vtop, _mm_mul_ps(_mm_sub_ps(vbot, vtop), vy));
                __m128i vi  = _mm_cvtps_epi32(vout); // round to nearest
                __m128i vi16 = _mm_packs_epi32(vi, _mm_setzero_si128());
                __m128i vi8  = _mm_packus_epi16(vi16, _mm_setzero_si128());
                alignas(16) uint8_t tmp[16];
                _mm_storeu_si128(reinterpret_cast<__m128i*>(tmp), vi8);
                out_row[x + 0] = tmp[0];
                out_row[x + 1] = tmp[1];
                out_row[x + 2] = tmp[2];
                out_row[x + 3] = tmp[3];
            }
            for (; x < width; ++x) {
                float tl = static_cast<float>(row_low[in_x_low[x]]);
                float tr = static_cast<float>(row_low[in_x_up[x]]);
                float bl = static_cast<float>(row_up [in_x_low[x]]);
                float br = static_cast<float>(row_up [in_x_up[x]]);
                float top = tl + (tr - tl) * lerp_x[x];
                float bottom = bl + (br - bl) * lerp_x[x];
                out_row[x] = static_cast<uint8_t>(roundf(top + (bottom - top) * lerp_y));
            }
        }
        return dst;
    }

    // SSE2 fast path: uint8 3-channel (interleaved), 4 pixels per step
    if (channels_ == 3 && std::is_same<D, uint8_t>::value) {
        for (int y = 0; y < height; ++y) {
            float in_y = y * height_scale;
            int in_y_low = std::min(static_cast<int>(in_y), height_ - 1);
            int in_y_up = std::min(in_y_low + 1, height_ - 1);
            float lerp_y = in_y - in_y_low;
            const uint8_t* row_low = reinterpret_cast<const uint8_t*>(Row(in_y_low));
            const uint8_t* row_up  = reinterpret_cast<const uint8_t*>(Row(in_y_up));
            uint8_t* out_row = reinterpret_cast<uint8_t*>(dst.Row(y));

            __m128 vy = _mm_set1_ps(lerp_y);
            int x = 0;
            for (; x + 4 <= width; x += 4) {
                __m128 vx = _mm_loadu_ps(&lerp_x[x]);
                for (int c = 0; c < 3; ++c) {
                    int l0 = in_x_low[x + 0] * 3 + c;
                    int u0 = in_x_up [x + 0] * 3 + c;
                    int l1 = in_x_low[x + 1] * 3 + c;
                    int u1 = in_x_up [x + 1] * 3 + c;
                    int l2 = in_x_low[x + 2] * 3 + c;
                    int u2 = in_x_up [x + 2] * 3 + c;
                    int l3 = in_x_low[x + 3] * 3 + c;
                    int u3 = in_x_up [x + 3] * 3 + c;

                    __m128 vtl = _mm_set_ps(static_cast<float>(row_low[l3]),
                                            static_cast<float>(row_low[l2]),
                                            static_cast<float>(row_low[l1]),
                                            static_cast<float>(row_low[l0]));
                    __m128 vtr = _mm_set_ps(static_cast<float>(row_low[u3]),
                                            static_cast<float>(row_low[u2]),
                                            static_cast<float>(row_low[u1]),
                                            static_cast<float>(row_low[u0]));
                    __m128 vbl = _mm_set_ps(static_cast<float>(row_up[l3]),
                                            static_cast<float>(row_up[l2]),
                                            static_cast<float>(row_up[l1]),
                                            static_cast<float>(row_up[l0]));
                    __m128 vbr = _mm_set_ps(static_cast<float>(row_up[u3]),
                                            static_cast<float>(row_up[u2]),
                                            static_cast<float>(row_up[u1]),
                                            static_cast<float>(row_up[u0]));
                    __m128 vtop = _mm_add_ps(vtl, _mm_mul_ps(_mm_sub_ps(vtr, vtl), vx));
                    __m128 vbot = _mm_add_ps(vbl, _mm_mul_ps(_mm_sub_ps(vbr, vbl), vx));
                    __m128 vout = _mm_add_ps(vtop, _mm_mul_ps(_mm_sub_ps(vbot, vtop), vy));
                    __m128i vi  = _mm_cvtps_epi32(vout);
                    __m128i vi16 = _mm_packs_epi32(vi, _mm_setzero_si128());
                    __m128i vi8  = _mm_packus_epi16(vi16, _mm_setzero_si128());
                    alignas(16) uint8_t tmp[16];
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(tmp), vi8);
                    out_row[(x + 0) * 3 + c] = tmp[0];
                    out_row[(x + 1) * 3 + c] = tmp[1];
                    out_row[(x + 2) * 3 + c] = tmp[2];
                    out_row[(x + 3) * 3 + c] = tmp[3];
                }
            }
            for (; x < width; ++x) {
                int l = in_x_low[x] * 3;
                int u = in_x_up[x] * 3;
                for (int c = 0; c < 3; ++c) {
                    float tl = static_cast<float>(row_low[l + c]);
                    float tr = static_cast<float>(row_low[u + c]);
                    float bl = static_cast<float>(row_up [l + c]);
                    float br = static_cast<float>(row_up [u + c]);
                    float top = tl + (tr - tl) * lerp_x[x];
                    float bottom = bl + (br - bl) * lerp_x[x];
                    out_row[x * 3 + c] = static_cast<uint8_t>(roundf(top + (bottom - top) * lerp_y));
                }
            }
        }
        return dst;
    }
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    // NEON fast path: uint8 single-channel, 4 pixels per step (compute in float, round to nearest)
    if (channels_ == 1 && std::is_same<D, uint8_t>::value) {
        for (int y = 0; y < height; ++y) {
            float in_y = y * height_scale;
            int in_y_low = std::min(static_cast<int>(in_y), height_ - 1);
            int in_y_up = std::min(in_y_low + 1, height_ - 1);
            float lerp_y = in_y - in_y_low;
            const uint8_t* row_low = reinterpret_cast<const uint8_t*>(Row(in_y_low));
            const uint8_t* row_up  = reinterpret_cast<const uint8_t*>(Row(in_y_up));
            uint8_t* out_row = reinterpret_cast<uint8_t*>(dst.Row(y));

            int x = 0;
            for (; x + 4 <= width; x += 4) {
                uint8_t tl0 = row_low[in_x_low[x + 0]];
                uint8_t tr0 = row_low[in_x_up[x + 0]];
                uint8_t bl0 = row_up [in_x_low[x + 0]];
                uint8_t br0 = row_up [in_x_up[x + 0]];

                uint8_t tl1 = row_low[in_x_low[x + 1]];
                uint8_t tr1 = row_low[in_x_up[x + 1]];
                uint8_t bl1 = row_up [in_x_low[x + 1]];
                uint8_t br1 = row_up [in_x_up[x + 1]];

                uint8_t tl2 = row_low[in_x_low[x + 2]];
                uint8_t tr2 = row_low[in_x_up[x + 2]];
                uint8_t bl2 = row_up [in_x_low[x + 2]];
                uint8_t br2 = row_up [in_x_up[x + 2]];

                uint8_t tl3 = row_low[in_x_low[x + 3]];
                uint8_t tr3 = row_low[in_x_up[x + 3]];
                uint8_t bl3 = row_up [in_x_low[x + 3]];
                uint8_t br3 = row_up [in_x_up[x + 3]];

                float32x4_t v_x = vld1q_f32(&lerp_x[x]);
                float32x4_t v_y = vdupq_n_f32(lerp_y);

                float32x4_t v_tl = vdupq_n_f32(0.f);
                v_tl = vsetq_lane_f32(static_cast<float>(tl0), v_tl, 0);
                v_tl = vsetq_lane_f32(static_cast<float>(tl1), v_tl, 1);
                v_tl = vsetq_lane_f32(static_cast<float>(tl2), v_tl, 2);
                v_tl = vsetq_lane_f32(static_cast<float>(tl3), v_tl, 3);

                float32x4_t v_tr = vdupq_n_f32(0.f);
                v_tr = vsetq_lane_f32(static_cast<float>(tr0), v_tr, 0);
                v_tr = vsetq_lane_f32(static_cast<float>(tr1), v_tr, 1);
                v_tr = vsetq_lane_f32(static_cast<float>(tr2), v_tr, 2);
                v_tr = vsetq_lane_f32(static_cast<float>(tr3), v_tr, 3);

                float32x4_t v_bl = vdupq_n_f32(0.f);
                v_bl = vsetq_lane_f32(static_cast<float>(bl0), v_bl, 0);
                v_bl = vsetq_lane_f32(static_cast<float>(bl1), v_bl, 1);
                v_bl = vsetq_lane_f32(static_cast<float>(bl2), v_bl, 2);
                v_bl = vsetq_lane_f32(static_cast<float>(bl3), v_bl, 3);

                float32x4_t v_br = vdupq_n_f32(0.f);
                v_br = vsetq_lane_f32(static_cast<float>(br0), v_br, 0);
                v_br = vsetq_lane_f32(static_cast<float>(br1), v_br, 1);
                v_br = vsetq_lane_f32(static_cast<float>(br2), v_br, 2);
                v_br = vsetq_lane_f32(static_cast<float>(br3), v_br, 3);

                float32x4_t v_top = vaddq_f32(v_tl, vmulq_f32(vsubq_f32(v_tr, v_tl), v_x));
                float32x4_t v_bot = vaddq_f32(v_bl, vmulq_f32(vsubq_f32(v_br, v_bl), v_x));
                float32x4_t v_out = vaddq_f32(v_top, vmulq_f32(vsubq_f32(v_bot, v_top), v_y));
                float outv[4];
                vst1q_f32(outv, v_out);
                out_row[x + 0] = static_cast<uint8_t>(roundf(outv[0]));
                out_row[x + 1] = static_cast<uint8_t>(roundf(outv[1]));
                out_row[x + 2] = static_cast<uint8_t>(roundf(outv[2]));
                out_row[x + 3] = static_cast<uint8_t>(roundf(outv[3]));
            }
            for (; x < width; ++x) {
                float tl = static_cast<float>(row_low[in_x_low[x]]);
                float tr = static_cast<float>(row_low[in_x_up[x]]);
                float bl = static_cast<float>(row_up [in_x_low[x]]);
                float br = static_cast<float>(row_up [in_x_up[x]]);
                float top = tl + (tr - tl) * lerp_x[x];
                float bottom = bl + (br - bl) * lerp_x[x];
                out_row[x] = static_cast<uint8_t>(roundf(top + (bottom - top) * lerp_y));
            }
        }
        return dst;
    }

    // NEON fast path: uint8 3-channel (interleaved), 4 pixels per step
    if (channels_ == 3 && std::is_same<D, uint8_t>::value) {
        for (int y = 0; y < height; ++y) {
            float in_y = y * height_scale;
            int in_y_low = std::min(static_cast<int>(in_y), height_ - 1);
            int in_y_up = std::min(in_y_low + 1, height_ - 1);
            float lerp_y = in_y - in_y_low;
            const uint8_t* row_low = reinterpret_cast<const uint8_t*>(Row(in_y_low));
            const uint8_t* row_up  = reinterpret_cast<const uint8_t*>(Row(in_y_up));
            uint8_t* out_row = reinterpret_cast<uint8_t*>(dst.Row(y));

            int x = 0;
            for (; x + 4 <= width; x += 4) {
                float32x4_t v_x = vld1q_f32(&lerp_x[x]);
                float32x4_t v_y = vdupq_n_f32(lerp_y);
                for (int c = 0; c < 3; ++c) {
                    int l0 = in_x_low[x + 0] * 3 + c;
                    int u0 = in_x_up [x + 0] * 3 + c;
                    int l1 = in_x_low[x + 1] * 3 + c;
                    int u1 = in_x_up [x + 1] * 3 + c;
                    int l2 = in_x_low[x + 2] * 3 + c;
                    int u2 = in_x_up [x + 2] * 3 + c;
                    int l3 = in_x_low[x + 3] * 3 + c;
                    int u3 = in_x_up [x + 3] * 3 + c;

                    float32x4_t v_tl = vdupq_n_f32(0.f);
                    v_tl = vsetq_lane_f32(static_cast<float>(row_low[l0]), v_tl, 0);
                    v_tl = vsetq_lane_f32(static_cast<float>(row_low[l1]), v_tl, 1);
                    v_tl = vsetq_lane_f32(static_cast<float>(row_low[l2]), v_tl, 2);
                    v_tl = vsetq_lane_f32(static_cast<float>(row_low[l3]), v_tl, 3);

                    float32x4_t v_tr = vdupq_n_f32(0.f);
                    v_tr = vsetq_lane_f32(static_cast<float>(row_low[u0]), v_tr, 0);
                    v_tr = vsetq_lane_f32(static_cast<float>(row_low[u1]), v_tr, 1);
                    v_tr = vsetq_lane_f32(static_cast<float>(row_low[u2]), v_tr, 2);
                    v_tr = vsetq_lane_f32(static_cast<float>(row_low[u3]), v_tr, 3);

                    float32x4_t v_bl = vdupq_n_f32(0.f);
                    v_bl = vsetq_lane_f32(static_cast<float>(row_up[l0]), v_bl, 0);
                    v_bl = vsetq_lane_f32(static_cast<float>(row_up[l1]), v_bl, 1);
                    v_bl = vsetq_lane_f32(static_cast<float>(row_up[l2]), v_bl, 2);
                    v_bl = vsetq_lane_f32(static_cast<float>(row_up[l3]), v_bl, 3);

                    float32x4_t v_br = vdupq_n_f32(0.f);
                    v_br = vsetq_lane_f32(static_cast<float>(row_up[u0]), v_br, 0);
                    v_br = vsetq_lane_f32(static_cast<float>(row_up[u1]), v_br, 1);
                    v_br = vsetq_lane_f32(static_cast<float>(row_up[u2]), v_br, 2);
                    v_br = vsetq_lane_f32(static_cast<float>(row_up[u3]), v_br, 3);

                    float32x4_t v_top = vaddq_f32(v_tl, vmulq_f32(vsubq_f32(v_tr, v_tl), v_x));
                    float32x4_t v_bot = vaddq_f32(v_bl, vmulq_f32(vsubq_f32(v_br, v_bl), v_x));
                    float32x4_t v_out = vaddq_f32(v_top, vmulq_f32(vsubq_f32(v_bot, v_top), v_y));
                    float outv[4];
                    vst1q_f32(outv, v_out);
                    out_row[(x + 0) * 3 + c] = static_cast<uint8_t>(roundf(outv[0]));
                    out_row[(x + 1) * 3 + c] = static_cast<uint8_t>(roundf(outv[1]));
                    out_row[(x + 2) * 3 + c] = static_cast<uint8_t>(roundf(outv[2]));
                    out_row[(x + 3) * 3 + c] = static_cast<uint8_t>(roundf(outv[3]));
                }
            }
            for (; x < width; ++x) {
                int l = in_x_low[x] * 3;
                int u = in_x_up[x] * 3;
                for (int c = 0; c < 3; ++c) {
                    float tl = static_cast<float>(row_low[l + c]);
                    float tr = static_cast<float>(row_low[u + c]);
                    float bl = static_cast<float>(row_up [l + c]);
                    float br = static_cast<float>(row_up [u + c]);
                    float top = tl + (tr - tl) * lerp_x[x];
                    float bottom = bl + (br - bl) * lerp_x[x];
                    out_row[x * 3 + c] = static_cast<uint8_t>(roundf(top + (bottom - top) * lerp_y));
                }
            }
        }
        return dst;
    }

    // NEON fast path: float single-channel, 4 pixels per step
    if (channels_ == 1 && std::is_same<D, float>::value) {
        for (int y = 0; y < height; ++y) {
            float in_y = y * height_scale;
            int in_y_low = std::min(static_cast<int>(in_y), height_ - 1);
            int in_y_up = std::min(in_y_low + 1, height_ - 1);
            float lerp_y = in_y - in_y_low;
            const float* row_low = reinterpret_cast<const float*>(Row(in_y_low));
            const float* row_up  = reinterpret_cast<const float*>(Row(in_y_up));
            float* out_row = reinterpret_cast<float*>(dst.Row(y));

            int x = 0;
            for (; x + 4 <= width; x += 4) {
                const float* tl0 = row_low + in_x_low[x + 0];
                const float* tr0 = row_low + in_x_up[x + 0];
                const float* bl0 = row_up  + in_x_low[x + 0];
                const float* br0 = row_up  + in_x_up[x + 0];

                const float* tl1 = row_low + in_x_low[x + 1];
                const float* tr1 = row_low + in_x_up[x + 1];
                const float* bl1 = row_up  + in_x_low[x + 1];
                const float* br1 = row_up  + in_x_up[x + 1];

                const float* tl2 = row_low + in_x_low[x + 2];
                const float* tr2 = row_low + in_x_up[x + 2];
                const float* bl2 = row_up  + in_x_low[x + 2];
                const float* br2 = row_up  + in_x_up[x + 2];

                const float* tl3 = row_low + in_x_low[x + 3];
                const float* tr3 = row_low + in_x_up[x + 3];
                const float* bl3 = row_up  + in_x_low[x + 3];
                const float* br3 = row_up  + in_x_up[x + 3];

                // Build vectors for x/y lerp
                float32x4_t v_x = vld1q_f32(&lerp_x[x]);
                float32x4_t v_y = vdupq_n_f32(lerp_y);
                // Load lanes for neighbors
                const float32x4_t v_zero = vdupq_n_f32(0.0f);
                float32x4_t v_tl = vld1q_lane_f32(tl0, v_zero, 0);
                v_tl = vld1q_lane_f32(tl1, v_tl, 1);
                v_tl = vld1q_lane_f32(tl2, v_tl, 2);
                v_tl = vld1q_lane_f32(tl3, v_tl, 3);

                float32x4_t v_tr = vld1q_lane_f32(tr0, v_zero, 0);
                v_tr = vld1q_lane_f32(tr1, v_tr, 1);
                v_tr = vld1q_lane_f32(tr2, v_tr, 2);
                v_tr = vld1q_lane_f32(tr3, v_tr, 3);

                float32x4_t v_bl = vld1q_lane_f32(bl0, v_zero, 0);
                v_bl = vld1q_lane_f32(bl1, v_bl, 1);
                v_bl = vld1q_lane_f32(bl2, v_bl, 2);
                v_bl = vld1q_lane_f32(bl3, v_bl, 3);

                float32x4_t v_br = vld1q_lane_f32(br0, v_zero, 0);
                v_br = vld1q_lane_f32(br1, v_br, 1);
                v_br = vld1q_lane_f32(br2, v_br, 2);
                v_br = vld1q_lane_f32(br3, v_br, 3);

                // top/bottom interpolation and final mix
                float32x4_t v_top = vaddq_f32(v_tl, vmulq_f32(vsubq_f32(v_tr, v_tl), v_x));
                float32x4_t v_bot = vaddq_f32(v_bl, vmulq_f32(vsubq_f32(v_br, v_bl), v_x));
                float32x4_t v_out = vaddq_f32(v_top, vmulq_f32(vsubq_f32(v_bot, v_top), v_y));
                float outv[4];
                vst1q_f32(outv, v_out);
                out_row[x + 0] = outv[0];
                out_row[x + 1] = outv[1];
                out_row[x + 2] = outv[2];
                out_row[x + 3] = outv[3];
            }
            for (; x < width; ++x) {
                const float tl = *(row_low + in_x_low[x]);
                const float tr = *(row_low + in_x_up[x]);
                const float bl = *(row_up  + in_x_low[x]);
                const float br = *(row_up  + in_x_up[x]);
                float top = tl + (tr - tl) * lerp_x[x];
                float bottom = bl + (br - bl) * lerp_x[x];
                out_row[x] = top + (bottom - top) * lerp_y;
            }
        }
        return dst;
    }

    // NEON fast path: float 3-channel (interleaved), 4 pixels per step
    if (channels_ == 3 && std::is_same<D, float>::value) {
        for (int y = 0; y < height; ++y) {
            float in_y = y * height_scale;
            int in_y_low = std::min(static_cast<int>(in_y), height_ - 1);
            int in_y_up = std::min(in_y_low + 1, height_ - 1);
            float lerp_y = in_y - in_y_low;
            const float* row_low = reinterpret_cast<const float*>(Row(in_y_low));
            const float* row_up  = reinterpret_cast<const float*>(Row(in_y_up));
            float* out_row = reinterpret_cast<float*>(dst.Row(y));

            int x = 0;
            for (; x + 4 <= width; x += 4) {
                float x_lerp[4] = { lerp_x[x + 0], lerp_x[x + 1], lerp_x[x + 2], lerp_x[x + 3] };
                float y_lerp[4] = { lerp_y, lerp_y, lerp_y, lerp_y };
                // per channel c=0,1,2
                for (int c = 0; c < 3; ++c) {
                    const float* tl0 = row_low + in_x_low[x + 0] * 3 + c;
                    const float* tr0 = row_low + in_x_up[x + 0] * 3 + c;
                    const float* bl0 = row_up  + in_x_low[x + 0] * 3 + c;
                    const float* br0 = row_up  + in_x_up[x + 0] * 3 + c;

                    const float* tl1 = row_low + in_x_low[x + 1] * 3 + c;
                    const float* tr1 = row_low + in_x_up[x + 1] * 3 + c;
                    const float* bl1 = row_up  + in_x_low[x + 1] * 3 + c;
                    const float* br1 = row_up  + in_x_up[x + 1] * 3 + c;

                    const float* tl2 = row_low + in_x_low[x + 2] * 3 + c;
                    const float* tr2 = row_low + in_x_up[x + 2] * 3 + c;
                    const float* bl2 = row_up  + in_x_low[x + 2] * 3 + c;
                    const float* br2 = row_up  + in_x_up[x + 2] * 3 + c;

                    const float* tl3 = row_low + in_x_low[x + 3] * 3 + c;
                    const float* tr3 = row_low + in_x_up[x + 3] * 3 + c;
                    const float* bl3 = row_up  + in_x_low[x + 3] * 3 + c;
                    const float* br3 = row_up  + in_x_up[x + 3] * 3 + c;

                    // Build vectors for x/y lerp
                    float32x4_t v_x = vld1q_f32(&lerp_x[x]);
                    float32x4_t v_y = vdupq_n_f32(lerp_y);
                    // Load lanes for neighbors
                    const float32x4_t v_zero = vdupq_n_f32(0.0f);
                    float32x4_t v_tl = vld1q_lane_f32(tl0, v_zero, 0);
                    v_tl = vld1q_lane_f32(tl1, v_tl, 1);
                    v_tl = vld1q_lane_f32(tl2, v_tl, 2);
                    v_tl = vld1q_lane_f32(tl3, v_tl, 3);

                    float32x4_t v_tr = vld1q_lane_f32(tr0, v_zero, 0);
                    v_tr = vld1q_lane_f32(tr1, v_tr, 1);
                    v_tr = vld1q_lane_f32(tr2, v_tr, 2);
                    v_tr = vld1q_lane_f32(tr3, v_tr, 3);

                    float32x4_t v_bl = vld1q_lane_f32(bl0, v_zero, 0);
                    v_bl = vld1q_lane_f32(bl1, v_bl, 1);
                    v_bl = vld1q_lane_f32(bl2, v_bl, 2);
                    v_bl = vld1q_lane_f32(bl3, v_bl, 3);

                    float32x4_t v_br = vld1q_lane_f32(br0, v_zero, 0);
                    v_br = vld1q_lane_f32(br1, v_br, 1);
                    v_br = vld1q_lane_f32(br2, v_br, 2);
                    v_br = vld1q_lane_f32(br3, v_br, 3);

                    float32x4_t v_top = vaddq_f32(v_tl, vmulq_f32(vsubq_f32(v_tr, v_tl), v_x));
                    float32x4_t v_bot = vaddq_f32(v_bl, vmulq_f32(vsubq_f32(v_br, v_bl), v_x));
                    float32x4_t v_out = vaddq_f32(v_top, vmulq_f32(vsubq_f32(v_bot, v_top), v_y));
                    float outv[4];
                    vst1q_f32(outv, v_out);
                    out_row[(x + 0) * 3 + c] = outv[0];
                    out_row[(x + 1) * 3 + c] = outv[1];
                    out_row[(x + 2) * 3 + c] = outv[2];
                    out_row[(x + 3) * 3 + c] = outv[3];
                }
            }
            for (; x < width; ++x) {
                int l = in_x_low[x] * 3;
                int u = in_x_up[x] * 3;
                for (int c = 0; c < 3; ++c) {
                    float tl = row_low[l + c];
                    float tr = row_low[u + c];
                    float bl = row_up[l + c];
                    float br = row_up[u + c];
                    float top = tl + (tr - tl) * lerp_x[x];
                    float bottom = bl + (br - bl) * lerp_x[x];
                    out_row[x * 3 + c] = top + (bottom - top) * lerp_y;
                }
            }
        }
        return dst;
    }
#endif
    for (int y = 0; y < height; ++y) {
        float in_y = y * height_scale;
        int in_y_low = std::min(static_cast<int>(in_y), height_ - 1);
        int in_y_up = std::min(in_y_low + 1, height_ - 1);
        float lerp_y = in_y - in_y_low;
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels_; ++c) {
                *dst_iter++ = InterpolateBilinear(
                  at(in_y_low, in_x_low[x])[c], at(in_y_low, in_x_up[x])[c],
                  at(in_y_up, in_x_low[x])[c], at(in_y_up, in_x_up[x])[c], lerp_x[x], lerp_y);
            }
        }
    }

    return dst;
}

template <typename D>
Bitmap<D> Bitmap<D>::ResizeNearest(int width, int height) const {
    INSPIRECV_CHECK(height > 0 && width > 0) << "height=" << height << ", width=" << width;
    if (height_ == height && width_ == width) {
        return Clone();
    }

    Bitmap<D> dst;
    dst.Reset(width, height, channels_);
    const float height_scale = static_cast<float>(height_) / height;
    const float width_scale = static_cast<float>(width_) / width;
    // SIMD fast path: nearest 2x upsample (both dims doubled)
    if (std::abs(width_scale - 0.5f) < 1e-6f && std::abs(height_scale - 0.5f) < 1e-6f) {
        // uint8 single-channel
        if (channels_ == 1 && std::is_same<D, uint8_t>::value) {
            for (int y = 0; y < height; ++y) {
                int in_y = std::min(y >> 1, height_ - 1);
                const uint8_t* src_row = reinterpret_cast<const uint8_t*>(Row(in_y));
                uint8_t* dst_row = reinterpret_cast<uint8_t*>(dst.Row(y));
                int x = 0;
#if defined(__AVX2__) || defined(__SSE2__)
                for (; x + 32 <= width; x += 32) {
                    int sx = (x >> 1);
                    __m128i s = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_row + sx));
                    __m128i lo = _mm_unpacklo_epi8(s, s);
                    __m128i hi = _mm_unpackhi_epi8(s, s);
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst_row + x + 0), lo);
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst_row + x + 16), hi);
                }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
                for (; x + 32 <= width; x += 32) {
                    int sx = (x >> 1);
                    uint8x16_t s = vld1q_u8(src_row + sx);
                    uint8x16_t lo = ZipLowerBytes(s, s);
                    uint8x16_t hi = ZipUpperBytes(s, s);
                    vst1q_u8(dst_row + x + 0, lo);
                    vst1q_u8(dst_row + x + 16, hi);
                }
#endif
                for (; x < width; ++x) dst_row[x] = src_row[x >> 1];
            }
            return dst;
        }
        // uint8 three-channel (BGR interleaved)
        if (channels_ == 3 && std::is_same<D, uint8_t>::value) {
            for (int y = 0; y < height; ++y) {
                int in_y = std::min(y >> 1, height_ - 1);
                const uint8_t* src_row = reinterpret_cast<const uint8_t*>(Row(in_y));
                uint8_t* dst_row = reinterpret_cast<uint8_t*>(dst.Row(y));
                int x = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
                for (; x + 32 <= width; x += 32) {
                    int sx = (x >> 1);
                    uint8x16x3_t s3 = vld3q_u8(src_row + sx * 3); // 16 src pixels
                    // duplicate each channel horizontally
                    uint8x16_t b_lo = ZipLowerBytes(s3.val[0], s3.val[0]);
                    uint8x16_t b_hi = ZipUpperBytes(s3.val[0], s3.val[0]);
                    uint8x16_t g_lo = ZipLowerBytes(s3.val[1], s3.val[1]);
                    uint8x16_t g_hi = ZipUpperBytes(s3.val[1], s3.val[1]);
                    uint8x16_t r_lo = ZipLowerBytes(s3.val[2], s3.val[2]);
                    uint8x16_t r_hi = ZipUpperBytes(s3.val[2], s3.val[2]);
                    uint8x16x3_t o_lo{b_lo, g_lo, r_lo};
                    uint8x16x3_t o_hi{b_hi, g_hi, r_hi};
                    vst3q_u8(dst_row + x * 3 + 0, o_lo);
                    vst3q_u8(dst_row + (x + 16) * 3, o_hi);
                }
#endif
                for (; x < width; ++x) {
                    int sx = (x >> 1);
                    const uint8_t* p = src_row + sx * 3;
                    uint8_t* q = dst_row + x * 3;
                    q[0] = p[0]; q[1] = p[1]; q[2] = p[2];
                }
            }
            return dst;
        }
        // float single-channel
        if (channels_ == 1 && std::is_same<D, float>::value) {
            for (int y = 0; y < height; ++y) {
                int in_y = std::min(y >> 1, height_ - 1);
                const float* src_row = reinterpret_cast<const float*>(Row(in_y));
                float* dst_row = reinterpret_cast<float*>(dst.Row(y));
                int x = 0;
#if defined(__SSE2__)
                for (; x + 8 <= width; x += 8) {
                    int sx = (x >> 1);
                    __m128 s = _mm_loadu_ps(src_row + sx);          // a0 a1 a2 a3
                    __m128 lo = _mm_unpacklo_ps(s, s);              // a0 a0 a1 a1
                    __m128 hi = _mm_unpackhi_ps(s, s);              // a2 a2 a3 a3
                    _mm_storeu_ps(dst_row + x + 0, lo);
                    _mm_storeu_ps(dst_row + x + 4, hi);
                }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
                for (; x + 8 <= width; x += 8) {
                    int sx = (x >> 1);
                    float32x4_t s = vld1q_f32(src_row + sx);
                    float32x4_t lo = ZipLowerFloats(s, s);
                    float32x4_t hi = ZipUpperFloats(s, s);
                    vst1q_f32(dst_row + x + 0, lo);
                    vst1q_f32(dst_row + x + 4, hi);
                }
#endif
                for (; x < width; ++x) dst_row[x] = src_row[x >> 1];
            }
            return dst;
        }
    }
    // Generic nearest: precompute x/y index maps for better cache reuse and row replication
    std::vector<int> in_x_idx(static_cast<size_t>(width));
    for (int x = 0; x < width; ++x) {
        in_x_idx[x] = std::min(static_cast<int>(x * width_scale), width_ - 1);
    }
    std::vector<int> in_y_idx(static_cast<size_t>(height));
    for (int y = 0; y < height; ++y) {
        in_y_idx[y] = std::min(static_cast<int>(y * height_scale), height_ - 1);
    }
    const size_t row_bytes = static_cast<size_t>(width) * static_cast<size_t>(channels_) * sizeof(D);
    for (int y = 0; y < height; ++y) {
        int sy = in_y_idx[y];
        D* dst_row = dst.Row(y);
        // Row replication for upsampling (same source row as previous dest row)
        if (y > 0 && sy == in_y_idx[y - 1]) {
            std::memcpy(dst_row, dst.Row(y - 1), row_bytes);
            continue;
        }
        const D* src_row = Row(sy);
        if (channels_ == 1 && std::is_same<D, uint8_t>::value) {
            int x = 0;
            while (x < width) {
                const int sx = in_x_idx[x];
                int run = 1;
                while (x + run < width && in_x_idx[x + run] == sx) ++run;
#if defined(__SSE2__)
                while (run >= 16) {
                    __m128i vv = _mm_set1_epi8(static_cast<char>(src_row[sx]));
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst_row + x), vv);
                    x += 16;
                    run -= 16;
                }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
                while (run >= 16) {
                    uint8x16_t vv = vdupq_n_u8(static_cast<uint8_t>(src_row[sx]));
                    vst1q_u8(reinterpret_cast<uint8_t*>(dst_row + x), vv);
                    x += 16;
                    run -= 16;
                }
#endif
                for (int i = 0; i < run; ++i) dst_row[x + i] = src_row[sx];
                x += run;
            }
            continue;
        }
        if (channels_ == 1 && std::is_same<D, float>::value) {
            int x = 0;
            while (x < width) {
                const int sx = in_x_idx[x];
                int run = 1;
                while (x + run < width && in_x_idx[x + run] == sx) ++run;
#if defined(__SSE2__)
                while (run >= 4) {
                    __m128 vv = _mm_set1_ps(reinterpret_cast<const float*>(src_row)[sx]);
                    _mm_storeu_ps(reinterpret_cast<float*>(dst_row + x), vv);
                    x += 4;
                    run -= 4;
                }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
                while (run >= 4) {
                    float32x4_t vv = vdupq_n_f32(reinterpret_cast<const float*>(src_row)[sx]);
                    vst1q_f32(reinterpret_cast<float*>(dst_row + x), vv);
                    x += 4;
                    run -= 4;
                }
#endif
                for (int i = 0; i < run; ++i) dst_row[x + i] = src_row[sx];
                x += run;
            }
            continue;
        }
        if (channels_ == 3 && std::is_same<D, uint8_t>::value) {
            int x = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
            while (x < width) {
                const int sx = in_x_idx[x];
                int run = 1;
                while (x + run < width && in_x_idx[x + run] == sx) ++run;
                const uint8_t* p = reinterpret_cast<const uint8_t*>(src_row) + sx * 3;
                uint8_t b = p[0], g = p[1], r = p[2];
                while (run >= 16) {
                    uint8x16_t vb = vdupq_n_u8(b);
                    uint8x16_t vg = vdupq_n_u8(g);
                    uint8x16_t vr = vdupq_n_u8(r);
                    uint8x16x3_t o{vb, vg, vr};
                    vst3q_u8(reinterpret_cast<uint8_t*>(dst_row) + x * 3, o);
                    x += 16;
                    run -= 16;
                }
                for (int i = 0; i < run; ++i) {
                    uint8_t* q = reinterpret_cast<uint8_t*>(dst_row) + (x + i) * 3;
                    q[0] = b; q[1] = g; q[2] = r;
                }
                x += run;
            }
#else
            for (; x < width; ++x) {
                int sx = in_x_idx[x];
                std::memcpy(reinterpret_cast<uint8_t*>(dst_row) + x * 3,
                            reinterpret_cast<const uint8_t*>(src_row) + sx * 3, 3);
            }
#endif
            continue;
        }
        // Fallback: generic memcpy per pixel (other channel/types)
        for (int x = 0; x < width; ++x) {
            int sx = in_x_idx[x];
            std::memcpy(reinterpret_cast<uint8_t*>(dst_row) + static_cast<size_t>(x) * channels_ * sizeof(D),
                        reinterpret_cast<const uint8_t*>(src_row) + static_cast<size_t>(sx) * channels_ * sizeof(D),
                        channels_ * sizeof(D));
        }
    }

    return dst;
}

template <typename D>
void Bitmap<D>::CropAndResizeNearest(Bitmap<D> &dst, const Rect<int> &rect, int resize_width,
                                    int resize_height) const {
    INSPIRECV_CHECK(this != &dst);
    INSPIRECV_CHECK(resize_height > 0 && resize_width > 0)
      << ", resize_height=" << resize_height << ", resize_width=" << resize_width;
    INSPIRECV_CHECK(Rect<int>(0, 0, width_, height_).Contains(rect)) << rect;

    dst.Reset(resize_width, resize_height, channels_);
    const float height_scale = static_cast<float>(rect.GetHeight()) / resize_height;
    const float width_scale = static_cast<float>(rect.GetWidth()) / resize_width;
    // Precompute index maps within ROI
    std::vector<int> in_x_idx(static_cast<size_t>(resize_width));
    for (int x = 0; x < resize_width; ++x) {
        int rx = std::min(static_cast<int>(x * width_scale), rect.GetWidth() - 1);
        in_x_idx[x] = rect.xmin() + rx;
    }
    std::vector<int> in_y_idx(static_cast<size_t>(resize_height));
    for (int y = 0; y < resize_height; ++y) {
        int ry = std::min(static_cast<int>(y * height_scale), rect.GetHeight() - 1);
        in_y_idx[y] = rect.ymin() + ry;
    }
    const size_t row_bytes = static_cast<size_t>(resize_width) * static_cast<size_t>(channels_) * sizeof(D);
    for (int y = 0; y < resize_height; ++y) {
        int sy = in_y_idx[y];
        D* dst_row = dst.Row(y);
        // Row replication for upsampling
        if (y > 0 && sy == in_y_idx[y - 1]) {
            std::memcpy(dst_row, dst.Row(y - 1), row_bytes);
            continue;
        }
        const D* src_row = Row(sy);
        if (channels_ == 1 && std::is_same<D, uint8_t>::value) {
            int x = 0;
            while (x < resize_width) {
                const int sx = in_x_idx[x];
                int run = 1;
                while (x + run < resize_width && in_x_idx[x + run] == sx) ++run;
#if defined(__SSE2__)
                while (run >= 16) {
                    __m128i vv = _mm_set1_epi8(static_cast<char>(src_row[sx]));
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst_row + x), vv);
                    x += 16;
                    run -= 16;
                }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
                while (run >= 16) {
                    uint8x16_t vv = vdupq_n_u8(static_cast<uint8_t>(src_row[sx]));
                    vst1q_u8(reinterpret_cast<uint8_t*>(dst_row + x), vv);
                    x += 16;
                    run -= 16;
                }
#endif
                for (int i = 0; i < run; ++i) dst_row[x + i] = src_row[sx];
                x += run;
            }
            continue;
        }
        if (channels_ == 1 && std::is_same<D, float>::value) {
            int x = 0;
            while (x < resize_width) {
                const int sx = in_x_idx[x];
                int run = 1;
                while (x + run < resize_width && in_x_idx[x + run] == sx) ++run;
#if defined(__SSE2__)
                while (run >= 4) {
                    __m128 vv = _mm_set1_ps(reinterpret_cast<const float*>(src_row)[sx]);
                    _mm_storeu_ps(reinterpret_cast<float*>(dst_row + x), vv);
                    x += 4;
                    run -= 4;
                }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
                while (run >= 4) {
                    float32x4_t vv = vdupq_n_f32(reinterpret_cast<const float*>(src_row)[sx]);
                    vst1q_f32(reinterpret_cast<float*>(dst_row + x), vv);
                    x += 4;
                    run -= 4;
                }
#endif
                for (int i = 0; i < run; ++i) dst_row[x + i] = src_row[sx];
                x += run;
            }
            continue;
        }
        if (channels_ == 3 && std::is_same<D, uint8_t>::value) {
            int x = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
            while (x < resize_width) {
                const int sx = in_x_idx[x];
                int run = 1;
                while (x + run < resize_width && in_x_idx[x + run] == sx) ++run;
                const uint8_t* p = reinterpret_cast<const uint8_t*>(src_row) + sx * 3;
                uint8_t b = p[0], g = p[1], r = p[2];
                while (run >= 16) {
                    uint8x16_t vb = vdupq_n_u8(b);
                    uint8x16_t vg = vdupq_n_u8(g);
                    uint8x16_t vr = vdupq_n_u8(r);
                    uint8x16x3_t o{vb, vg, vr};
                    vst3q_u8(reinterpret_cast<uint8_t*>(dst_row) + x * 3, o);
                    x += 16;
                    run -= 16;
                }
                for (int i = 0; i < run; ++i) {
                    uint8_t* q = reinterpret_cast<uint8_t*>(dst_row) + (x + i) * 3;
                    q[0] = b; q[1] = g; q[2] = r;
                }
                x += run;
            }
#else
            for (; x < resize_width; ++x) {
                int sx = in_x_idx[x];
                std::memcpy(reinterpret_cast<uint8_t*>(dst_row) + x * 3,
                            reinterpret_cast<const uint8_t*>(src_row) + sx * 3, 3);
            }
#endif
            continue;
        }
        // Fallback
        for (int x = 0; x < resize_width; ++x) {
            int sx = in_x_idx[x];
            std::memcpy(reinterpret_cast<uint8_t*>(dst_row) + static_cast<size_t>(x) * channels_ * sizeof(D),
                        reinterpret_cast<const uint8_t*>(src_row) + static_cast<size_t>(sx) * channels_ * sizeof(D),
                        channels_ * sizeof(D));
        }
    }
}

template <typename D>
void Bitmap<D>::CropAndResizeBilinear(Bitmap<D> &dst, const Rect<int> &rect, int resize_width,
                                     int resize_height) const {
    INSPIRECV_CHECK(this != &dst);
    INSPIRECV_CHECK(resize_height > 0 && resize_width > 0)
      << ", resize_height=" << resize_height << ", resize_width=" << resize_width;
    INSPIRECV_CHECK(Rect<int>(0, 0, width_, height_).Contains(rect)) << rect;

    dst.Reset(resize_width, resize_height, channels_);
    const float height_scale = static_cast<float>(rect.GetHeight()) / resize_height;
    const float width_scale = static_cast<float>(rect.GetWidth()) / resize_width;

    std::vector<int> in_x_low(resize_width);
    std::vector<int> in_x_up(resize_width);
    std::vector<float> lerp_x(resize_width);
    for (int x = 0; x < resize_width; ++x) {
        float in_x = x * width_scale + rect.xmin();
        in_x_low[x] = std::min(static_cast<int>(in_x), width_ - 1);
        in_x_up[x] = std::min(in_x_low[x] + 1, width_ - 1);
        lerp_x[x] = in_x - in_x_low[x];
    }

    // SIMD fast paths for u8 using 4-pixel steps (compute in float, round to nearest)
#if defined(__SSE2__)
    if (channels_ == 1 && std::is_same<D, uint8_t>::value) {
        for (int y = 0; y < resize_height; ++y) {
            float in_y = y * height_scale + rect.ymin();
            int in_y_low = std::min(static_cast<int>(in_y), height_ - 1);
            int in_y_up = std::min(in_y_low + 1, height_ - 1);
            float lerp_y = in_y - in_y_low;
            const uint8_t* row_low = reinterpret_cast<const uint8_t*>(Row(in_y_low));
            const uint8_t* row_up  = reinterpret_cast<const uint8_t*>(Row(in_y_up));
            uint8_t* out_row = reinterpret_cast<uint8_t*>(dst.Row(y));
            __m128 vy = _mm_set1_ps(lerp_y);
            int x = 0;
            for (; x + 4 <= resize_width; x += 4) {
                float tl0 = static_cast<float>(row_low[in_x_low[x + 0]]);
                float tr0 = static_cast<float>(row_low[in_x_up [x + 0]]);
                float bl0 = static_cast<float>(row_up [in_x_low[x + 0]]);
                float br0 = static_cast<float>(row_up [in_x_up [x + 0]]);
                float tl1 = static_cast<float>(row_low[in_x_low[x + 1]]);
                float tr1 = static_cast<float>(row_low[in_x_up [x + 1]]);
                float bl1 = static_cast<float>(row_up [in_x_low[x + 1]]);
                float br1 = static_cast<float>(row_up [in_x_up [x + 1]]);
                float tl2 = static_cast<float>(row_low[in_x_low[x + 2]]);
                float tr2 = static_cast<float>(row_low[in_x_up [x + 2]]);
                float bl2 = static_cast<float>(row_up [in_x_low[x + 2]]);
                float br2 = static_cast<float>(row_up [in_x_up [x + 2]]);
                float tl3 = static_cast<float>(row_low[in_x_low[x + 3]]);
                float tr3 = static_cast<float>(row_low[in_x_up [x + 3]]);
                float bl3 = static_cast<float>(row_up [in_x_low[x + 3]]);
                float br3 = static_cast<float>(row_up [in_x_up [x + 3]]);
                __m128 vtl = _mm_set_ps(tl3, tl2, tl1, tl0);
                __m128 vtr = _mm_set_ps(tr3, tr2, tr1, tr0);
                __m128 vbl = _mm_set_ps(bl3, bl2, bl1, bl0);
                __m128 vbr = _mm_set_ps(br3, br2, br1, br0);
                __m128 vx  = _mm_loadu_ps(&lerp_x[x]);
                __m128 vtop = _mm_add_ps(vtl, _mm_mul_ps(_mm_sub_ps(vtr, vtl), vx));
                __m128 vbot = _mm_add_ps(vbl, _mm_mul_ps(_mm_sub_ps(vbr, vbl), vx));
                __m128 vout = _mm_add_ps(vtop, _mm_mul_ps(_mm_sub_ps(vbot, vtop), vy));
                __m128i vi  = _mm_cvtps_epi32(vout);
                __m128i vi16 = _mm_packs_epi32(vi, _mm_setzero_si128());
                __m128i vi8  = _mm_packus_epi16(vi16, _mm_setzero_si128());
                alignas(16) uint8_t tmp[16];
                _mm_storeu_si128(reinterpret_cast<__m128i*>(tmp), vi8);
                out_row[x + 0] = tmp[0];
                out_row[x + 1] = tmp[1];
                out_row[x + 2] = tmp[2];
                out_row[x + 3] = tmp[3];
            }
            for (; x < resize_width; ++x) {
                float tl = static_cast<float>(row_low[in_x_low[x]]);
                float tr = static_cast<float>(row_low[in_x_up[x]]);
                float bl = static_cast<float>(row_up [in_x_low[x]]);
                float br = static_cast<float>(row_up [in_x_up[x]]);
                float top = tl + (tr - tl) * lerp_x[x];
                float bottom = bl + (br - bl) * lerp_x[x];
                out_row[x] = static_cast<uint8_t>(roundf(top + (bottom - top) * lerp_y));
            }
        }
        return;
    }
    if (channels_ == 3 && std::is_same<D, uint8_t>::value) {
        for (int y = 0; y < resize_height; ++y) {
            float in_y = y * height_scale + rect.ymin();
            int in_y_low = std::min(static_cast<int>(in_y), height_ - 1);
            int in_y_up = std::min(in_y_low + 1, height_ - 1);
            float lerp_y = in_y - in_y_low;
            const uint8_t* row_low = reinterpret_cast<const uint8_t*>(Row(in_y_low));
            const uint8_t* row_up  = reinterpret_cast<const uint8_t*>(Row(in_y_up));
            uint8_t* out_row = reinterpret_cast<uint8_t*>(dst.Row(y));
            __m128 vy = _mm_set1_ps(lerp_y);
            int x = 0;
            for (; x + 4 <= resize_width; x += 4) {
                __m128 vx = _mm_loadu_ps(&lerp_x[x]);
                for (int c = 0; c < 3; ++c) {
                    int l0 = in_x_low[x + 0] * 3 + c;
                    int u0 = in_x_up [x + 0] * 3 + c;
                    int l1 = in_x_low[x + 1] * 3 + c;
                    int u1 = in_x_up [x + 1] * 3 + c;
                    int l2 = in_x_low[x + 2] * 3 + c;
                    int u2 = in_x_up [x + 2] * 3 + c;
                    int l3 = in_x_low[x + 3] * 3 + c;
                    int u3 = in_x_up [x + 3] * 3 + c;
                    __m128 vtl = _mm_set_ps(static_cast<float>(row_low[l3]),
                                            static_cast<float>(row_low[l2]),
                                            static_cast<float>(row_low[l1]),
                                            static_cast<float>(row_low[l0]));
                    __m128 vtr = _mm_set_ps(static_cast<float>(row_low[u3]),
                                            static_cast<float>(row_low[u2]),
                                            static_cast<float>(row_low[u1]),
                                            static_cast<float>(row_low[u0]));
                    __m128 vbl = _mm_set_ps(static_cast<float>(row_up[l3]),
                                            static_cast<float>(row_up[l2]),
                                            static_cast<float>(row_up[l1]),
                                            static_cast<float>(row_up[l0]));
                    __m128 vbr = _mm_set_ps(static_cast<float>(row_up[u3]),
                                            static_cast<float>(row_up[u2]),
                                            static_cast<float>(row_up[u1]),
                                            static_cast<float>(row_up[u0]));
                    __m128 vtop = _mm_add_ps(vtl, _mm_mul_ps(_mm_sub_ps(vtr, vtl), vx));
                    __m128 vbot = _mm_add_ps(vbl, _mm_mul_ps(_mm_sub_ps(vbr, vbl), vx));
                    __m128 vout = _mm_add_ps(vtop, _mm_mul_ps(_mm_sub_ps(vbot, vtop), vy));
                    __m128i vi  = _mm_cvtps_epi32(vout);
                    __m128i vi16 = _mm_packs_epi32(vi, _mm_setzero_si128());
                    __m128i vi8  = _mm_packus_epi16(vi16, _mm_setzero_si128());
                    alignas(16) uint8_t tmp[16];
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(tmp), vi8);
                    out_row[(x + 0) * 3 + c] = tmp[0];
                    out_row[(x + 1) * 3 + c] = tmp[1];
                    out_row[(x + 2) * 3 + c] = tmp[2];
                    out_row[(x + 3) * 3 + c] = tmp[3];
                }
            }
            for (; x < resize_width; ++x) {
                int l = in_x_low[x] * 3;
                int u = in_x_up[x] * 3;
                for (int c = 0; c < 3; ++c) {
                    float tl = static_cast<float>(row_low[l + c]);
                    float tr = static_cast<float>(row_low[u + c]);
                    float bl = static_cast<float>(row_up [l + c]);
                    float br = static_cast<float>(row_up [u + c]);
                    float top = tl + (tr - tl) * lerp_x[x];
                    float bottom = bl + (br - bl) * lerp_x[x];
                    out_row[x * 3 + c] = static_cast<uint8_t>(roundf(top + (bottom - top) * lerp_y));
                }
            }
        }
        return;
    }
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    if (channels_ == 1 && std::is_same<D, uint8_t>::value) {
        for (int y = 0; y < resize_height; ++y) {
            float in_y = y * height_scale + rect.ymin();
            int in_y_low = std::min(static_cast<int>(in_y), height_ - 1);
            int in_y_up = std::min(in_y_low + 1, height_ - 1);
            float lerp_y = in_y - in_y_low;
            const uint8_t* row_low = reinterpret_cast<const uint8_t*>(Row(in_y_low));
            const uint8_t* row_up  = reinterpret_cast<const uint8_t*>(Row(in_y_up));
            uint8_t* out_row = reinterpret_cast<uint8_t*>(dst.Row(y));
            int x = 0;
            for (; x + 4 <= resize_width; x += 4) {
                uint8_t tl0 = row_low[in_x_low[x + 0]];
                uint8_t tr0 = row_low[in_x_up [x + 0]];
                uint8_t bl0 = row_up [in_x_low[x + 0]];
                uint8_t br0 = row_up [in_x_up [x + 0]];
                uint8_t tl1 = row_low[in_x_low[x + 1]];
                uint8_t tr1 = row_low[in_x_up [x + 1]];
                uint8_t bl1 = row_up [in_x_low[x + 1]];
                uint8_t br1 = row_up [in_x_up [x + 1]];
                uint8_t tl2 = row_low[in_x_low[x + 2]];
                uint8_t tr2 = row_low[in_x_up [x + 2]];
                uint8_t bl2 = row_up [in_x_low[x + 2]];
                uint8_t br2 = row_up [in_x_up [x + 2]];
                uint8_t tl3 = row_low[in_x_low[x + 3]];
                uint8_t tr3 = row_low[in_x_up [x + 3]];
                uint8_t bl3 = row_up [in_x_low[x + 3]];
                uint8_t br3 = row_up [in_x_up [x + 3]];
                float32x4_t v_x = vld1q_f32(&lerp_x[x]);
                float32x4_t v_y = vdupq_n_f32(lerp_y);
                float32x4_t v_tl = vdupq_n_f32(0.f);
                v_tl = vsetq_lane_f32(static_cast<float>(tl0), v_tl, 0);
                v_tl = vsetq_lane_f32(static_cast<float>(tl1), v_tl, 1);
                v_tl = vsetq_lane_f32(static_cast<float>(tl2), v_tl, 2);
                v_tl = vsetq_lane_f32(static_cast<float>(tl3), v_tl, 3);
                float32x4_t v_tr = vdupq_n_f32(0.f);
                v_tr = vsetq_lane_f32(static_cast<float>(tr0), v_tr, 0);
                v_tr = vsetq_lane_f32(static_cast<float>(tr1), v_tr, 1);
                v_tr = vsetq_lane_f32(static_cast<float>(tr2), v_tr, 2);
                v_tr = vsetq_lane_f32(static_cast<float>(tr3), v_tr, 3);
                float32x4_t v_bl = vdupq_n_f32(0.f);
                v_bl = vsetq_lane_f32(static_cast<float>(bl0), v_bl, 0);
                v_bl = vsetq_lane_f32(static_cast<float>(bl1), v_bl, 1);
                v_bl = vsetq_lane_f32(static_cast<float>(bl2), v_bl, 2);
                v_bl = vsetq_lane_f32(static_cast<float>(bl3), v_bl, 3);
                float32x4_t v_br = vdupq_n_f32(0.f);
                v_br = vsetq_lane_f32(static_cast<float>(br0), v_br, 0);
                v_br = vsetq_lane_f32(static_cast<float>(br1), v_br, 1);
                v_br = vsetq_lane_f32(static_cast<float>(br2), v_br, 2);
                v_br = vsetq_lane_f32(static_cast<float>(br3), v_br, 3);
                float32x4_t v_top = vaddq_f32(v_tl, vmulq_f32(vsubq_f32(v_tr, v_tl), v_x));
                float32x4_t v_bot = vaddq_f32(v_bl, vmulq_f32(vsubq_f32(v_br, v_bl), v_x));
                float32x4_t v_out = vaddq_f32(v_top, vmulq_f32(vsubq_f32(v_bot, v_top), v_y));
                float outv[4];
                vst1q_f32(outv, v_out);
                out_row[x + 0] = static_cast<uint8_t>(roundf(outv[0]));
                out_row[x + 1] = static_cast<uint8_t>(roundf(outv[1]));
                out_row[x + 2] = static_cast<uint8_t>(roundf(outv[2]));
                out_row[x + 3] = static_cast<uint8_t>(roundf(outv[3]));
            }
            for (; x < resize_width; ++x) {
                float tl = static_cast<float>(row_low[in_x_low[x]]);
                float tr = static_cast<float>(row_low[in_x_up[x]]);
                float bl = static_cast<float>(row_up [in_x_low[x]]);
                float br = static_cast<float>(row_up [in_x_up[x]]);
                float top = tl + (tr - tl) * lerp_x[x];
                float bottom = bl + (br - bl) * lerp_x[x];
                out_row[x] = static_cast<uint8_t>(roundf(top + (bottom - top) * lerp_y));
            }
        }
        return;
    }
    if (channels_ == 3 && std::is_same<D, uint8_t>::value) {
        for (int y = 0; y < resize_height; ++y) {
            float in_y = y * height_scale + rect.ymin();
            int in_y_low = std::min(static_cast<int>(in_y), height_ - 1);
            int in_y_up = std::min(in_y_low + 1, height_ - 1);
            float lerp_y = in_y - in_y_low;
            const uint8_t* row_low = reinterpret_cast<const uint8_t*>(Row(in_y_low));
            const uint8_t* row_up  = reinterpret_cast<const uint8_t*>(Row(in_y_up));
            uint8_t* out_row = reinterpret_cast<uint8_t*>(dst.Row(y));
            int x = 0;
            for (; x + 4 <= resize_width; x += 4) {
                float32x4_t v_x = vld1q_f32(&lerp_x[x]);
                float32x4_t v_y = vdupq_n_f32(lerp_y);
                for (int c = 0; c < 3; ++c) {
                    int l0 = in_x_low[x + 0] * 3 + c;
                    int u0 = in_x_up [x + 0] * 3 + c;
                    int l1 = in_x_low[x + 1] * 3 + c;
                    int u1 = in_x_up [x + 1] * 3 + c;
                    int l2 = in_x_low[x + 2] * 3 + c;
                    int u2 = in_x_up [x + 2] * 3 + c;
                    int l3 = in_x_low[x + 3] * 3 + c;
                    int u3 = in_x_up [x + 3] * 3 + c;
                    float32x4_t v_tl = vdupq_n_f32(0.f);
                    v_tl = vsetq_lane_f32(static_cast<float>(row_low[l0]), v_tl, 0);
                    v_tl = vsetq_lane_f32(static_cast<float>(row_low[l1]), v_tl, 1);
                    v_tl = vsetq_lane_f32(static_cast<float>(row_low[l2]), v_tl, 2);
                    v_tl = vsetq_lane_f32(static_cast<float>(row_low[l3]), v_tl, 3);
                    float32x4_t v_tr = vdupq_n_f32(0.f);
                    v_tr = vsetq_lane_f32(static_cast<float>(row_low[u0]), v_tr, 0);
                    v_tr = vsetq_lane_f32(static_cast<float>(row_low[u1]), v_tr, 1);
                    v_tr = vsetq_lane_f32(static_cast<float>(row_low[u2]), v_tr, 2);
                    v_tr = vsetq_lane_f32(static_cast<float>(row_low[u3]), v_tr, 3);
                    float32x4_t v_bl = vdupq_n_f32(0.f);
                    v_bl = vsetq_lane_f32(static_cast<float>(row_up[l0]), v_bl, 0);
                    v_bl = vsetq_lane_f32(static_cast<float>(row_up[l1]), v_bl, 1);
                    v_bl = vsetq_lane_f32(static_cast<float>(row_up[l2]), v_bl, 2);
                    v_bl = vsetq_lane_f32(static_cast<float>(row_up[l3]), v_bl, 3);
                    float32x4_t v_br = vdupq_n_f32(0.f);
                    v_br = vsetq_lane_f32(static_cast<float>(row_up[u0]), v_br, 0);
                    v_br = vsetq_lane_f32(static_cast<float>(row_up[u1]), v_br, 1);
                    v_br = vsetq_lane_f32(static_cast<float>(row_up[u2]), v_br, 2);
                    v_br = vsetq_lane_f32(static_cast<float>(row_up[u3]), v_br, 3);
                    float32x4_t v_top = vaddq_f32(v_tl, vmulq_f32(vsubq_f32(v_tr, v_tl), v_x));
                    float32x4_t v_bot = vaddq_f32(v_bl, vmulq_f32(vsubq_f32(v_br, v_bl), v_x));
                    float32x4_t v_out = vaddq_f32(v_top, vmulq_f32(vsubq_f32(v_bot, v_top), v_y));
                    float outv[4];
                    vst1q_f32(outv, v_out);
                    out_row[(x + 0) * 3 + c] = static_cast<uint8_t>(roundf(outv[0]));
                    out_row[(x + 1) * 3 + c] = static_cast<uint8_t>(roundf(outv[1]));
                    out_row[(x + 2) * 3 + c] = static_cast<uint8_t>(roundf(outv[2]));
                    out_row[(x + 3) * 3 + c] = static_cast<uint8_t>(roundf(outv[3]));
                }
            }
            for (; x < resize_width; ++x) {
                int l = in_x_low[x] * 3;
                int u = in_x_up[x] * 3;
                for (int c = 0; c < 3; ++c) {
                    float tl = static_cast<float>(row_low[l + c]);
                    float tr = static_cast<float>(row_low[u + c]);
                    float bl = static_cast<float>(row_up [l + c]);
                    float br = static_cast<float>(row_up [u + c]);
                    float top = tl + (tr - tl) * lerp_x[x];
                    float bottom = bl + (br - bl) * lerp_x[x];
                    out_row[x * 3 + c] = static_cast<uint8_t>(roundf(top + (bottom - top) * lerp_y));
                }
            }
        }
        return;
    }
#endif

    auto dst_iter = dst.Data();
    for (int y = 0; y < resize_height; ++y) {
        float in_y = y * height_scale + rect.ymin();
        int in_y_low = std::min(static_cast<int>(in_y), height_ - 1);
        int in_y_up = std::min(in_y_low + 1, height_ - 1);
        float lerp_y = in_y - in_y_low;
        for (int x = 0; x < resize_width; ++x) {
            for (int c = 0; c < channels_; ++c) {
                *(dst_iter++) = InterpolateBilinear(
                  at(in_y_low, in_x_low[x])[c], at(in_y_low, in_x_up[x])[c],
                  at(in_y_up, in_x_low[x])[c], at(in_y_up, in_x_up[x])[c], lerp_x[x], lerp_y);
            }
        }
    }
}

template <typename D>
void Bitmap<D>::GetTransformMatrix(int width, int height, const Rect<int> &rect,
                                  TransformMatrix &matrix) const {
    matrix[0] = static_cast<float>(rect.GetWidth()) / width;
    matrix[1] = 0;
    matrix[2] = rect.xmin();
    matrix[3] = 0;
    matrix[4] = static_cast<float>(rect.GetHeight()) / height;
    matrix[5] = rect.ymin();
}

template <typename D>
Bitmap<D> Bitmap<D>::AffineBilinearReference(int width, int height, const TransformMatrix &matrix,
                                           BorderMode border_mode, D border_value) const {
    Bitmap<D> dst;
    dst.Reset(width, height, channels_);
    dst.Fill(0);
    auto dst_iter = dst.Data();
    for (int dst_y = 0; dst_y < height; ++dst_y) {
        for (int dst_x = 0; dst_x < width; ++dst_x) {
            float src_x = dst_x * matrix[0] + dst_y * matrix[1] + matrix[2];
            float src_y = dst_x * matrix[3] + dst_y * matrix[4] + matrix[5];

            if (src_x >= width_ || src_y >= height_ || src_x < 0 || src_y < 0) {
                if (border_mode == BORDER_MODE_CONSTANT) {
                    for (int c = 0; c < channels_; ++c)
                        *(dst_iter++) = border_value;
                    continue;
                } else if (border_mode == BORDER_MODE_REPLICATE) {
                    if (src_x >= width_) {
                        src_x = width_ - 1;
                    }
                    if (src_y >= height_) {
                        src_y = height_ - 1;
                    }
                    if (src_x < 0) {
                        src_x = 0;
                    }
                    if (src_y < 0) {
                        src_y = 0;
                    }
                } else {
                    INSPIRECV_LOG(ERROR) << "unsupport border mode:" << border_mode;
                }
            }

            int src_x_low = std::min(static_cast<int>(src_x), width_ - 1);
            int src_x_up = std::min(src_x_low + 1, width_ - 1);
            float lerp_x = src_x - src_x_low;
            int src_y_low = std::min(static_cast<int>(src_y), height_ - 1);
            int src_y_up = std::min(src_y_low + 1, height_ - 1);
            float lerp_y = src_y - src_y_low;

            for (int c = 0; c < channels_; ++c) {
                *(dst_iter++) = InterpolateBilinear(
                  at(src_y_low, src_x_low)[c], at(src_y_low, src_x_up)[c],
                  at(src_y_up, src_x_low)[c], at(src_y_up, src_x_up)[c], lerp_x, lerp_y);
            }
        }
    }

    return dst;
}

template <>
Bitmap<float> Bitmap<float>::AffineBilinear(int width, int height, const TransformMatrix &matrix,
                                          BorderMode border_mode, float border_value) const {
    Bitmap<float> dst;
    dst.Reset(width, height, channels_);
    float *dst_iter = dst.Data();

    if (channels_ != 1) {
        return AffineBilinearReference(width, height, matrix);
    }
    if ((!(fabs(matrix[0]) < 1e-3 && fabs(matrix[4]) < 1e-3)) &&
        (!(fabs(matrix[1]) < 1e-3 && fabs(matrix[3]) < 1e-3))) {
        return AffineBilinearReference(width, height, matrix);
    }

    struct AffineIndex {
        int src_low;
        int src_up;
        float lerp;
        bool overstep_lower;
    };

    if (fabs(matrix[0]) < 1e-3 && fabs(matrix[4]) < 1e-3) {
        std::vector<AffineIndex> affine_index(height + width);
        AffineIndex *affine_index_x = &affine_index[0];
        AffineIndex *affine_index_y = &affine_index[height];

        for (int dst_y = 0; dst_y < height; ++dst_y) {
            float src_x = dst_y * matrix[1] + matrix[2];
            if (src_x >= width_ || src_x < 0) {
                if (border_mode == BORDER_MODE_CONSTANT) {
                    affine_index_x[dst_y].src_low = -1;
                } else if (border_mode == BORDER_MODE_REPLICATE) {
                    if (src_x >= width_) {
                        src_x = width_ - 1;
                    }
                    if (src_x < 0) {
                        src_x = 0;
                    }
                } else {
                    INSPIRECV_LOG(ERROR) << "unsupport border mode:" << border_mode;
                }
            } else {
                int src_x_low = std::min(static_cast<int>(src_x), width_ - 1);
                int src_x_up = std::min(src_x_low + 1, width_ - 1);
                float lerp_x = src_x - src_x_low;
                affine_index_x[dst_y].src_low = src_x_low;
                affine_index_x[dst_y].src_up = src_x_up;
                affine_index_x[dst_y].lerp = lerp_x;
            }
        }

        for (int dst_x = 0; dst_x < width; ++dst_x) {
            float src_y = dst_x * matrix[3] + matrix[5];
            if (src_y >= height_ || src_y < 0) {
                if (border_mode == BORDER_MODE_CONSTANT) {
                    affine_index_y[dst_x].src_low = -1;
                } else if (border_mode == BORDER_MODE_REPLICATE) {
                    if (src_y >= height_) {
                        src_y = height_ - 1;
                    }
                    if (src_y < 0) {
                        src_y = 0;
                    }
                    int src_y_low = std::min(static_cast<int>(src_y), height_ - 1);
                    int src_y_up = std::min(src_y_low + 1, height_ - 1);
                    float lerp_y = src_y - src_y_low;
                    affine_index_y[dst_x].src_low = src_y_low;
                    affine_index_y[dst_x].src_up = src_y_up;
                    affine_index_y[dst_x].lerp = lerp_y;
                } else {
                    INSPIRECV_LOG(ERROR) << "unsupport border mode:" << border_mode;
                }
            } else {
                int src_y_low = std::min(static_cast<int>(src_y), height_ - 1);
                int src_y_up = std::min(src_y_low + 1, height_ - 1);
                float lerp_y = src_y - src_y_low;
                affine_index_y[dst_x].src_low = src_y_low;
                affine_index_y[dst_x].src_up = src_y_up;
                affine_index_y[dst_x].lerp = lerp_y;
            }
        }
        int dst_y = 0;
        if (border_mode == BORDER_MODE_CONSTANT) {
            for (; dst_y <= height - 4; dst_y += 4) {
                if (affine_index_x[dst_y].src_low == -1 ||
                    affine_index_x[dst_y + 1].src_low == -1 ||
                    affine_index_x[dst_y + 2].src_low == -1 ||
                    affine_index_x[dst_y + 3].src_low == -1) {
                    affine_index_x[dst_y / 4].overstep_lower = true;
                } else {
                    affine_index_x[dst_y / 4].overstep_lower = false;
                }
            }
            affine_index_x[dst_y].overstep_lower = false;
            for (; dst_y < height; dst_y++) {
                if (affine_index_x[dst_y].src_low == -1) {
                    affine_index_x[dst_y / 4].overstep_lower = true;
                    break;
                }
            }
        }

        dst_y = 0;
#if (defined(__ARM_NEON__) || defined(__ARM_NEON))
        for (; dst_y <= height - 4; dst_y += 4) {
            for (int dst_x = 0; dst_x < width; dst_x++) {
                const float *top_left_0, *top_left_1, *top_left_2, *top_left_3;
                const float *top_right_0, *top_right_1, *top_right_2, *top_right_3;
                const float *bottom_left_0, *bottom_left_1, *bottom_left_2, *bottom_left_3;
                const float *bottom_right_0, *bottom_right_1, *bottom_right_2, *bottom_right_3;
                float x_lerp[4];
                float y_lerp[4];
                if (border_mode == BORDER_MODE_CONSTANT &&
                    (affine_index_y[dst_x].src_low == -1 ||
                     affine_index_x[dst_y / 4].overstep_lower == true)) {
                    for (int dst_y_tmp = dst_y; dst_y_tmp < dst_y + 4; ++dst_y_tmp) {
                        if (affine_index_x[dst_y_tmp].src_low == -1 ||
                            affine_index_y[dst_x].src_low == -1) {
                            dst_iter[dst_y_tmp * width + dst_x] = border_value;
                        } else {
                            dst_iter[dst_y_tmp * width + dst_x] = InterpolateBilinear(
                              at(affine_index_y[dst_x].src_low,
                                 affine_index_x[dst_y_tmp].src_low)[0],
                              at(affine_index_y[dst_x].src_low,
                                 affine_index_x[dst_y_tmp].src_up)[0],
                              at(affine_index_y[dst_x].src_up,
                                 affine_index_x[dst_y_tmp].src_low)[0],
                              at(affine_index_y[dst_x].src_up, affine_index_x[dst_y_tmp].src_up)[0],
                              affine_index_x[dst_y_tmp].lerp, affine_index_y[dst_x].lerp);
                        }
                    }
                } else {
                    struct AffineIndex *affine_index_local_x = affine_index_x + dst_y;
                    struct AffineIndex *affine_index_local_y = affine_index_y + dst_x;

                    top_left_0 = Data() + affine_index_local_y->src_low * width_ +
                                 affine_index_local_x->src_low;
                    top_left_1 = Data() + affine_index_local_y->src_low * width_ +
                                 (affine_index_local_x + 1)->src_low;
                    top_left_2 = Data() + affine_index_local_y->src_low * width_ +
                                 (affine_index_local_x + 2)->src_low;
                    top_left_3 = Data() + affine_index_local_y->src_low * width_ +
                                 (affine_index_local_x + 3)->src_low;

                    top_right_0 = Data() + affine_index_local_y->src_low * width_ +
                                  affine_index_local_x->src_up;
                    top_right_1 = Data() + affine_index_local_y->src_low * width_ +
                                  (affine_index_local_x + 1)->src_up;
                    top_right_2 = Data() + affine_index_local_y->src_low * width_ +
                                  (affine_index_local_x + 2)->src_up;
                    top_right_3 = Data() + affine_index_local_y->src_low * width_ +
                                  (affine_index_local_x + 3)->src_up;

                    bottom_left_0 = Data() + affine_index_local_y->src_up * width_ +
                                    affine_index_local_x->src_low;
                    bottom_left_1 = Data() + affine_index_local_y->src_up * width_ +
                                    (affine_index_local_x + 1)->src_low;
                    bottom_left_2 = Data() + affine_index_local_y->src_up * width_ +
                                    (affine_index_local_x + 2)->src_low;
                    bottom_left_3 = Data() + affine_index_local_y->src_up * width_ +
                                    (affine_index_local_x + 3)->src_low;

                    bottom_right_0 =
                      Data() + affine_index_local_y->src_up * width_ + affine_index_local_x->src_up;
                    bottom_right_1 = Data() + affine_index_local_y->src_up * width_ +
                                     (affine_index_local_x + 1)->src_up;
                    bottom_right_2 = Data() + affine_index_local_y->src_up * width_ +
                                     (affine_index_local_x + 2)->src_up;
                    bottom_right_3 = Data() + affine_index_local_y->src_up * width_ +
                                     (affine_index_local_x + 3)->src_up;
                    x_lerp[0] = affine_index_local_x->lerp;
                    x_lerp[1] = (affine_index_local_x + 1)->lerp;
                    x_lerp[2] = (affine_index_local_x + 2)->lerp;
                    x_lerp[3] = (affine_index_local_x + 3)->lerp;
                    y_lerp[0] = affine_index_local_y->lerp;
                    y_lerp[1] = affine_index_local_y->lerp;
                    y_lerp[2] = affine_index_local_y->lerp;
                    y_lerp[3] = affine_index_local_y->lerp;

                    float out[4];

                    OutInterpolateBilinearFloatx4x1Neon(
                      top_left_0, top_left_1, top_left_2, top_left_3, top_right_0, top_right_1,
                      top_right_2, top_right_3, bottom_left_0, bottom_left_1, bottom_left_2,
                      bottom_left_3, bottom_right_0, bottom_right_1, bottom_right_2, bottom_right_3,
                      x_lerp, y_lerp, out);
                    dst_iter[dst_y * width + dst_x] = out[0];
                    dst_iter[(dst_y + 1) * width + dst_x] = out[1];
                    dst_iter[(dst_y + 2) * width + dst_x] = out[2];
                    dst_iter[(dst_y + 3) * width + dst_x] = out[3];
                }
            }
        }
#endif
        for (; dst_y < height; ++dst_y) {
            for (int dst_x = 0; dst_x < width; dst_x++) {
                if (affine_index_x[dst_y].src_low == -1 || affine_index_y[dst_x].src_low == -1) {
                    dst_iter[dst_y * width + dst_x] = border_value;
                } else {
                    dst_iter[dst_y * width + dst_x] = InterpolateBilinear(
                      at(affine_index_y[dst_x].src_low, affine_index_x[dst_y].src_low)[0],
                      at(affine_index_y[dst_x].src_low, affine_index_x[dst_y].src_up)[0],
                      at(affine_index_y[dst_x].src_up, affine_index_x[dst_y].src_low)[0],
                      at(affine_index_y[dst_x].src_up, affine_index_x[dst_y].src_up)[0],
                      affine_index_x[dst_y].lerp, affine_index_y[dst_x].lerp);
                }
            }
        }
    } else if (fabs(matrix[1]) < 1e-3 && fabs(matrix[3]) < 1e-3) {
        std::vector<AffineIndex> affine_index(width + height);
        AffineIndex *affine_index_x = &affine_index[0];
        AffineIndex *affine_index_y = &affine_index[width];

        for (int dst_y = 0; dst_y < height; ++dst_y) {
            float src_y = dst_y * matrix[4] + matrix[5];
            if (src_y >= height_ || src_y < 0) {
                if (border_mode == BORDER_MODE_CONSTANT) {
                    affine_index_y[dst_y].src_low = -1;
                } else if (border_mode == BORDER_MODE_REPLICATE) {
                    if (src_y >= height_) {
                        src_y = height_ - 1;
                    }
                    if (src_y < 0) {
                        src_y = 0;
                    }
                } else {
                    INSPIRECV_LOG(ERROR) << "unsupport border mode:" << border_mode;
                }
            } else {
                int src_y_low = std::min(static_cast<int>(src_y), height_ - 1);
                int src_y_up = std::min(src_y_low + 1, height_ - 1);
                float lerp_y = src_y - src_y_low;
                affine_index_y[dst_y].src_low = src_y_low;
                affine_index_y[dst_y].src_up = src_y_up;
                affine_index_y[dst_y].lerp = lerp_y;
            }
        }
        for (int dst_x = 0; dst_x < width; ++dst_x) {
            float src_x = dst_x * matrix[0] + matrix[2];
            if (src_x >= width_ || src_x < 0) {
                if (border_mode == BORDER_MODE_CONSTANT) {
                    affine_index_x[dst_x].src_low = -1;
                } else if (border_mode == BORDER_MODE_REPLICATE) {
                    if (src_x >= width_) {
                        src_x = width_ - 1;
                    }
                    if (src_x < 0) {
                        src_x = 0;
                    }
                    int src_x_low = std::min(static_cast<int>(src_x), width_ - 1);
                    int src_x_up = std::min(src_x_low + 1, width_ - 1);
                    float lerp_x = src_x - src_x_low;
                    affine_index_x[dst_x].src_low = src_x_low;
                    affine_index_x[dst_x].src_up = src_x_up;
                    affine_index_x[dst_x].lerp = lerp_x;
                } else {
                    INSPIRECV_LOG(ERROR) << "unsupport border mode:" << border_mode;
                }
            } else {
                int src_x_low = std::min(static_cast<int>(src_x), width_ - 1);
                int src_x_up = std::min(src_x_low + 1, width_ - 1);
                float lerp_x = src_x - src_x_low;
                affine_index_x[dst_x].src_low = src_x_low;
                affine_index_x[dst_x].src_up = src_x_up;
                affine_index_x[dst_x].lerp = lerp_x;
            }
        }

        int dst_x = 0;
        if (border_mode == BORDER_MODE_CONSTANT) {
            for (; dst_x <= width - 4; dst_x += 4) {
                if (affine_index_x[dst_x].src_low == -1 ||
                    affine_index_x[dst_x + 1].src_low == -1 ||
                    affine_index_x[dst_x + 2].src_low == -1 ||
                    affine_index_x[dst_x + 3].src_low == -1) {
                    affine_index_x[dst_x / 4].overstep_lower = true;
                } else {
                    affine_index_x[dst_x / 4].overstep_lower = false;
                }
            }
            affine_index_x[dst_x / 4].overstep_lower = false;
            for (; dst_x < width; dst_x++) {
                if (affine_index_x[dst_x].src_low == -1) {
                    affine_index_x[dst_x / 4].overstep_lower = true;
                    break;
                }
            }
        }

        for (int dst_y = 0; dst_y < height; ++dst_y) {
            int dst_x = 0;
#if (defined(__ARM_NEON__) || defined(__ARM_NEON))
            for (; dst_x <= width - 4; dst_x += 4) {
                const float *top_left_0, *top_left_1, *top_left_2, *top_left_3;
                const float *top_right_0, *top_right_1, *top_right_2, *top_right_3;
                const float *bottom_left_0, *bottom_left_1, *bottom_left_2, *bottom_left_3;
                const float *bottom_right_0, *bottom_right_1, *bottom_right_2, *bottom_right_3;
                float x_lerp[4];
                float y_lerp[4];
                if (border_mode == BORDER_MODE_CONSTANT &&
                    (affine_index_y[dst_y].src_low == -1 ||
                     affine_index_x[dst_x / 4].overstep_lower == true)) {
                    for (int dst_x_tmp = dst_x; dst_x_tmp < dst_x + 4; ++dst_x_tmp) {
                        if (affine_index_y[dst_y].src_low == -1 ||
                            affine_index_x[dst_x_tmp].src_low == -1) {
                            *dst_iter++ = border_value;
                        } else {
                            *(dst_iter++) = InterpolateBilinear(
                              at(affine_index_y[dst_y].src_low,
                                 affine_index_x[dst_x_tmp].src_low)[0],
                              at(affine_index_y[dst_y].src_low,
                                 affine_index_x[dst_x_tmp].src_up)[0],
                              at(affine_index_y[dst_y].src_up,
                                 affine_index_x[dst_x_tmp].src_low)[0],
                              at(affine_index_y[dst_y].src_up, affine_index_x[dst_x_tmp].src_up)[0],
                              affine_index_x[dst_x_tmp].lerp, affine_index_y[dst_y].lerp);
                        }
                    }
                } else {
                    top_left_0 = Data() + affine_index_y[dst_y].src_low * width_ +
                                 affine_index_x[dst_x].src_low;
                    top_left_1 = Data() + affine_index_y[dst_y].src_low * width_ +
                                 affine_index_x[dst_x + 1].src_low;
                    top_left_2 = Data() + affine_index_y[dst_y].src_low * width_ +
                                 affine_index_x[dst_x + 2].src_low;
                    top_left_3 = Data() + affine_index_y[dst_y].src_low * width_ +
                                 affine_index_x[dst_x + 3].src_low;

                    top_right_0 = Data() + affine_index_y[dst_y].src_low * width_ +
                                  affine_index_x[dst_x].src_up;
                    top_right_1 = Data() + affine_index_y[dst_y].src_low * width_ +
                                  affine_index_x[dst_x + 1].src_up;
                    top_right_2 = Data() + affine_index_y[dst_y].src_low * width_ +
                                  affine_index_x[dst_x + 2].src_up;
                    top_right_3 = Data() + affine_index_y[dst_y].src_low * width_ +
                                  affine_index_x[dst_x + 3].src_up;

                    bottom_left_0 = Data() + affine_index_y[dst_y].src_up * width_ +
                                    affine_index_x[dst_x].src_low;
                    bottom_left_1 = Data() + affine_index_y[dst_y].src_up * width_ +
                                    affine_index_x[dst_x + 1].src_low;
                    bottom_left_2 = Data() + affine_index_y[dst_y].src_up * width_ +
                                    affine_index_x[dst_x + 2].src_low;
                    bottom_left_3 = Data() + affine_index_y[dst_y].src_up * width_ +
                                    affine_index_x[dst_x + 3].src_low;

                    bottom_right_0 =
                      Data() + affine_index_y[dst_y].src_up * width_ + affine_index_x[dst_x].src_up;
                    bottom_right_1 = Data() + affine_index_y[dst_y].src_up * width_ +
                                     affine_index_x[dst_x + 1].src_up;
                    bottom_right_2 = Data() + affine_index_y[dst_y].src_up * width_ +
                                     affine_index_x[dst_x + 2].src_up;
                    bottom_right_3 = Data() + affine_index_y[dst_y].src_up * width_ +
                                     affine_index_x[dst_x + 3].src_up;
                    x_lerp[0] = affine_index_x[dst_x].lerp;
                    x_lerp[1] = affine_index_x[dst_x + 1].lerp;
                    x_lerp[2] = affine_index_x[dst_x + 2].lerp;
                    x_lerp[3] = affine_index_x[dst_x + 3].lerp;
                    y_lerp[0] = y_lerp[1] = y_lerp[2] = y_lerp[3] = affine_index_y[dst_y].lerp;

                    OutInterpolateBilinearFloatx4x1Neon(
                      top_left_0, top_left_1, top_left_2, top_left_3, top_right_0, top_right_1,
                      top_right_2, top_right_3, bottom_left_0, bottom_left_1, bottom_left_2,
                      bottom_left_3, bottom_right_0, bottom_right_1, bottom_right_2, bottom_right_3,
                      x_lerp, y_lerp, dst_iter);
                    dst_iter += 4;
                }
            }
#endif
            for (; dst_x < width; dst_x++) {
                if (affine_index_x[dst_x].src_low == -1 || affine_index_y[dst_y].src_low == -1) {
                    *(dst_iter++) = border_value;
                } else {
                    *(dst_iter++) = InterpolateBilinear(
                      at(affine_index_y[dst_y].src_low, affine_index_x[dst_x].src_low)[0],
                      at(affine_index_y[dst_y].src_low, affine_index_x[dst_x].src_up)[0],
                      at(affine_index_y[dst_y].src_up, affine_index_x[dst_x].src_low)[0],
                      at(affine_index_y[dst_y].src_up, affine_index_x[dst_x].src_up)[0],
                      affine_index_x[dst_x].lerp, affine_index_y[dst_y].lerp);
                }
            }
        }
    }

    return dst;
}

template <typename D>
Bitmap<D> Bitmap<D>::AffineBilinear(int width, int height, const TransformMatrix &matrix,
                                  BorderMode border_mode, D border_value) const {
    return AffineBilinearReference(width, height, matrix, border_mode, border_value);
}

namespace {

// One allocation, with each four-pixel block holding a genuinely contiguous
// weight vector. Keep indices/flags separately typed: loading across an AoS
// member also reads padding and is neither a gather nor a valid float array.
class AffineAxisTable {
public:
    struct Tap {
        int xl, xu;
        uint8_t in_l, in_u;
    };

    explicit AffineAxisTable(int width) : blocks_((static_cast<size_t>(width) + 3) / 4) {}

    Tap& operator[](size_t x) { return blocks_[x / 4].taps[x % 4]; }
    const Tap& operator[](size_t x) const { return blocks_[x / 4].taps[x % 4]; }
    float& Fraction(size_t x) { return blocks_[x / 4].weights[x % 4]; }
    const float* Weights(size_t x) const { return blocks_[x / 4].weights + x % 4; }

private:
    struct Block {
        float weights[4];
        Tap taps[4];
    };
    std::vector<Block> blocks_;
};

template <typename Pixel, int Channels>
void AffineReplicateRow(const Pixel* low, const Pixel* up, Pixel* output, int width, float fy,
                        const AffineAxisTable& table, int begin = 0) {
    int x = begin;
#if defined(__SSE2__) || defined(__ARM_NEON) || defined(__ARM_NEON__)
    for (; x + 4 <= width; x += 4) {
        const auto& p0 = table[x];
        const auto& p1 = table[x + 1];
        const auto& p2 = table[x + 2];
        const auto& p3 = table[x + 3];
        // Replicate/interior indices are already valid; no per-tap border
        // flags or padding branches are needed in this row.
        for (int channel = 0; channel < Channels; ++channel) {
#if defined(__SSE2__)
            const __m128 tl =
                _mm_setr_ps(low[p0.xl * Channels + channel], low[p1.xl * Channels + channel],
                            low[p2.xl * Channels + channel], low[p3.xl * Channels + channel]);
            const __m128 tr =
                _mm_setr_ps(low[p0.xu * Channels + channel], low[p1.xu * Channels + channel],
                            low[p2.xu * Channels + channel], low[p3.xu * Channels + channel]);
            const __m128 bl =
                _mm_setr_ps(up[p0.xl * Channels + channel], up[p1.xl * Channels + channel],
                            up[p2.xl * Channels + channel], up[p3.xl * Channels + channel]);
            const __m128 br =
                _mm_setr_ps(up[p0.xu * Channels + channel], up[p1.xu * Channels + channel],
                            up[p2.xu * Channels + channel], up[p3.xu * Channels + channel]);
            const __m128 fx = _mm_loadu_ps(table.Weights(x));
            const __m128 top = _mm_add_ps(tl, _mm_mul_ps(_mm_sub_ps(tr, tl), fx));
            const __m128 bottom = _mm_add_ps(bl, _mm_mul_ps(_mm_sub_ps(br, bl), fx));
            const __m128 result =
                _mm_add_ps(top, _mm_mul_ps(_mm_sub_ps(bottom, top), _mm_set1_ps(fy)));
            if (std::is_same<Pixel, float>::value) {
                if (Channels == 1) {
                    _mm_storeu_ps(reinterpret_cast<float*>(output + x), result);
                } else {
                    float values[4];
                    _mm_storeu_ps(values, result);
                    for (int lane = 0; lane < 4; ++lane)
                        output[(x + lane) * Channels + channel] = values[lane];
                }
            } else {
                __m128i rounded = _mm_cvttps_epi32(result);
                const __m128 fraction = _mm_sub_ps(result, _mm_cvtepi32_ps(rounded));
                rounded = _mm_add_epi32(
                    rounded,
                    _mm_and_si128(_mm_castps_si128(_mm_cmpge_ps(fraction, _mm_set1_ps(0.5f))),
                                  _mm_set1_epi32(1)));
                const __m128i halves = _mm_packs_epi32(rounded, _mm_setzero_si128());
                const __m128i bytes = _mm_packus_epi16(halves, _mm_setzero_si128());
                const int packed = _mm_cvtsi128_si32(bytes);
                if (Channels == 1) {
                    std::memcpy(output + x, &packed, sizeof(packed));
                } else {
                    uint8_t values[4];
                    std::memcpy(values, &packed, sizeof(packed));
                    for (int lane = 0; lane < 4; ++lane)
                        output[(x + lane) * Channels + channel] = values[lane];
                }
            }
#else
            const float32x4_t tl = {
                float(low[p0.xl * Channels + channel]), float(low[p1.xl * Channels + channel]),
                float(low[p2.xl * Channels + channel]), float(low[p3.xl * Channels + channel])};
            const float32x4_t tr = {
                float(low[p0.xu * Channels + channel]), float(low[p1.xu * Channels + channel]),
                float(low[p2.xu * Channels + channel]), float(low[p3.xu * Channels + channel])};
            const float32x4_t bl = {
                float(up[p0.xl * Channels + channel]), float(up[p1.xl * Channels + channel]),
                float(up[p2.xl * Channels + channel]), float(up[p3.xl * Channels + channel])};
            const float32x4_t br = {
                float(up[p0.xu * Channels + channel]), float(up[p1.xu * Channels + channel]),
                float(up[p2.xu * Channels + channel]), float(up[p3.xu * Channels + channel])};
            const float32x4_t fx = vld1q_f32(table.Weights(x));
            const float32x4_t top = vaddq_f32(tl, vmulq_f32(vsubq_f32(tr, tl), fx));
            const float32x4_t bottom = vaddq_f32(bl, vmulq_f32(vsubq_f32(br, bl), fx));
            const float32x4_t result =
                vaddq_f32(top, vmulq_f32(vsubq_f32(bottom, top), vdupq_n_f32(fy)));
            if (std::is_same<Pixel, float>::value) {
                if (Channels == 1) {
                    vst1q_f32(reinterpret_cast<float*>(output + x), result);
                } else {
                    float values[4];
                    vst1q_f32(values, result);
                    for (int lane = 0; lane < 4; ++lane)
                        output[(x + lane) * Channels + channel] = values[lane];
                }
            } else {
#if defined(__aarch64__)
                const uint16x4_t rounded = vmovn_u32(vcvtaq_u32_f32(result));
                const uint8x8_t bytes = vmovn_u16(vcombine_u16(rounded, rounded));
                const uint32_t packed = vget_lane_u32(vreinterpret_u32_u8(bytes), 0);
                if (Channels == 1) {
                    std::memcpy(output + x, &packed, sizeof(packed));
                } else {
                    uint8_t values[4];
                    std::memcpy(values, &packed, sizeof(packed));
                    for (int lane = 0; lane < 4; ++lane)
                        output[(x + lane) * Channels + channel] = values[lane];
                }
#else
                float values[4];
                vst1q_f32(values, result);
                for (int lane = 0; lane < 4; ++lane)
                    output[(x + lane) * Channels + channel] =
                        static_cast<Pixel>(std::round(values[lane]));
#endif
            }
#endif
        }
    }
#endif
    for (; x < width; ++x) {
        const auto& tap = table[x];
        const float fx = *table.Weights(x);
        for (int channel = 0; channel < Channels; ++channel) {
            const int left = tap.xl * Channels + channel, right = tap.xu * Channels + channel;
            const float top = low[left] + (float(low[right]) - low[left]) * fx;
            const float bottom = up[left] + (float(up[right]) - up[left]) * fx;
            const float value = top + (bottom - top) * fy;
            output[x * Channels + channel] = std::is_same<Pixel, uint8_t>::value
                                                 ? static_cast<Pixel>(std::round(value))
                                                 : static_cast<Pixel>(value);
        }
    }
}

template <typename Pixel>
// Only small single-channel sources use this bounded stack buffer. Outlining
// prevents its stack cost from affecting the general affine entry point.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline, flatten))
#endif
void AffineConstantSmallImage(const Pixel* source, int source_width, int source_height,
                              Pixel* output, int width, int height, float sy, float ty,
                              AffineAxisTable& table, Pixel border) {
    Pixel padded[32 * 32];
    const int pitch = source_width + 1;
    for (int y = 0; y < source_height; ++y) {
        std::copy_n(source + y * source_width, source_width, padded + y * pitch);
        padded[y * pitch + source_width] = border;
    }
    std::fill_n(padded + source_height * pitch, pitch, border);
    // Every missing x tap now addresses the padding column, and every missing
    // y tap the padding row. Keep all four-tap arithmetic, including NaN/Inf.
    for (int x = 0; x < width; ++x) {
        if (!table[x].in_l) table[x].xl = source_width;
        if (!table[x].in_u) table[x].xu = source_width;
    }
    for (int y = 0; y < height; ++y) {
        const float position = y * sy + ty;
        const int yl = static_cast<int>(std::floor(position)), yu = yl + 1;
        const float fraction = position - yl;
        if ((yu < 0 || yl >= source_height) && std::isfinite(float(border))) {
            std::fill_n(output + y * width, width, static_cast<Pixel>(float(border) + 0.f));
            continue;
        }
        const Pixel* low = padded + ((yl >= 0 && yl < source_height) ? yl : source_height) * pitch;
        const Pixel* up = padded + ((yu >= 0 && yu < source_height) ? yu : source_height) * pitch;
        AffineReplicateRow<Pixel, 1>(low, up, output + y * width, width, fraction, table);
    }
}

template <typename Pixel, int Channels>
inline void AffineConstantRowEdges(const Pixel* low, const Pixel* up, Pixel* output, int begin, int end,
                                  float fy, Pixel border, const AffineAxisTable& table) {
    for (int x = begin; x < end; ++x) {
        const auto& tap = table[x];
        const float fx = *table.Weights(x);
        if (!tap.in_l && !tap.in_u && std::isfinite(float(border))) {
            // Match interpolation's positive zero for a -0 border, while
            // avoiding four identical samples for fully padded pixels.
            std::fill_n(output + x * Channels, Channels, static_cast<Pixel>(float(border) + 0.f));
            continue;
        }
        for (int channel = 0; channel < Channels; ++channel) {
            const float tl = tap.in_l ? low[tap.xl * Channels + channel] : border;
            const float tr = tap.in_u ? low[tap.xu * Channels + channel] : border;
            const float bl = tap.in_l ? up[tap.xl * Channels + channel] : border;
            const float br = tap.in_u ? up[tap.xu * Channels + channel] : border;
            const float top = tl + (tr - tl) * fx;
            const float bottom = bl + (br - bl) * fx;
            const float value = top + (bottom - top) * fy;
            output[x * Channels + channel] = std::is_same<Pixel, uint8_t>::value
                                                 ? static_cast<Pixel>(std::round(value))
                                                 : static_cast<Pixel>(value);
        }
    }
}

}  // namespace

template <typename D>
Bitmap<D> Bitmap<D>::AffineBilinearOptimized(int width, int height, const TransformMatrix& matrix,
                                             BorderMode border_mode, D border_value) const {
#ifndef OKCV_ENABLE_AFFINE_GENERAL_SIMD
#define OKCV_ENABLE_AFFINE_GENERAL_SIMD 1
#endif
    // Fast path: pure scale + translate (no shear/rotation)
    // src_x = x * a + tx; src_y = y * d + ty
    const float a = matrix[0];
    const float b = matrix[1];
    const float c = matrix[3];
    const float d = matrix[4];
    const float tx = matrix[2];
    const float ty = matrix[5];

    // Byte-valued identity transforms need only an independent copy. Keep the
    // float path unchanged: its NaN/infinity arithmetic has different semantics.
    if (std::is_same<D, uint8_t>::value && width == width_ && height == height_ &&
        a == 1.f && d == 1.f && b == 0.f && c == 0.f && tx == 0.f && ty == 0.f) {
        return Clone();
    }

    if (std::fabs(b) < 1e-6f && std::fabs(c) < 1e-6f) {
        Bitmap<D> dst;
        dst.Reset(width, height, channels_);

        // Precompute x indices and weights (+ in-bounds flags for constant border)
        using XInfo = AffineAxisTable::Tap;
        AffineAxisTable xinfo(width);
        int interior_begin = width, interior_end = 0;
        if (border_mode == BORDER_MODE_REPLICATE) {
            for (int x = 0; x < width; ++x) {
                float srcx = x * a + tx;
                int xl = static_cast<int>(srcx);
                float fx = srcx - xl;
                if (srcx < 0.f) { xl = 0; fx = 0.f; }
                if (xl >= width_ - 1) { xl = std::max(0, width_ - 1); fx = 0.f; }
                xinfo[x].xl = xl;
                xinfo[x].xu = std::min(xl + 1, width_ - 1);
                xinfo.Fraction(x) = fx;
                xinfo[x].in_l = 1;
                xinfo[x].in_u = 1;
            }
        } else { // BORDER_MODE_CONSTANT
            for (int x = 0; x < width; ++x) {
                float srcx = x * a + tx;
                int xl = static_cast<int>(std::floor(srcx));
                float fx = srcx - static_cast<float>(xl);
                int xu = xl + 1;
                xinfo[x].xl = xl;
                xinfo[x].xu = xu;
                xinfo.Fraction(x) = fx;
                xinfo[x].in_l = (xl >= 0 && xl < width_) ? 1 : 0;
                xinfo[x].in_u = (xu >= 0 && xu < width_) ? 1 : 0;
                if (xinfo[x].in_l && xinfo[x].in_u) {
                    interior_begin = std::min(interior_begin, x);
                    interior_end = x + 1;
                }
            }
        }
        // A linear x mapping has one contiguous interior, including negative
        // scales. Align it to the weight-table blocks for the branch-free row.
        interior_begin = static_cast<int>(std::min<size_t>(
          (static_cast<size_t>(interior_begin) + 3) / 4 * 4, static_cast<size_t>(width)));
        interior_end = interior_end / 4 * 4;

        if (border_mode == BORDER_MODE_CONSTANT && channels_ == 1 &&
            width < 32 && height < 32 && width_ < 32 && height_ < 32) {
            AffineConstantSmallImage(Data(), width_, height_, dst.Data(), width, height,
                                     d, ty, xinfo, border_value);
            return dst;
        }

        for (int y = 0; y < height; ++y) {
            float srcy = y * d + ty;
            int yl, yu;
            float fy;
            uint8_t in_yl = 1, in_yu = 1;
            if (border_mode == BORDER_MODE_REPLICATE) {
                yl = static_cast<int>(srcy);
                fy = srcy - yl;
                if (srcy < 0.f) { yl = 0; fy = 0.f; }
                if (yl >= height_ - 1) { yl = std::max(0, height_ - 1); fy = 0.f; }
                yu = std::min(yl + 1, height_ - 1);
            } else {
                yl = static_cast<int>(std::floor(srcy));
                fy = srcy - static_cast<float>(yl);
                yu = yl + 1;
                in_yl = (yl >= 0 && yl < height_) ? 1 : 0;
                in_yu = (yu >= 0 && yu < height_) ? 1 : 0;
            }

            if (!in_yl && !in_yu && std::isfinite(float(border_value))) {
                std::fill_n(dst.Row(y), static_cast<size_t>(width) * channels_,
                            static_cast<D>(float(border_value) + 0.f));
                continue;
            }

            if (channels_ == 1 && border_mode == BORDER_MODE_REPLICATE) {
                AffineReplicateRow<D, 1>(Row(yl), Row(yu), dst.Row(y), width, fy, xinfo);
                continue;
            }
            if (channels_ == 3 && border_mode == BORDER_MODE_REPLICATE) {
                AffineReplicateRow<D, 3>(Row(yl), Row(yu), dst.Row(y), width, fy, xinfo);
                continue;
            }
            if (border_mode == BORDER_MODE_CONSTANT && in_yl && in_yu &&
                (channels_ == 1 || channels_ == 3) && interior_end - interior_begin >= 8) {
                const D* low = Row(yl);
                const D* up = Row(yu);
                D* output = dst.Row(y);
                if (channels_ == 1) {
                    AffineConstantRowEdges<D, 1>(low, up, output, 0, interior_begin, fy, border_value, xinfo);
                    AffineReplicateRow<D, 1>(low, up, output, interior_end, fy, xinfo, interior_begin);
                    AffineConstantRowEdges<D, 1>(low, up, output, interior_end, width, fy, border_value, xinfo);
                } else {
                    AffineConstantRowEdges<D, 3>(low, up, output, 0, interior_begin, fy, border_value, xinfo);
                    AffineReplicateRow<D, 3>(low, up, output, interior_end, fy, xinfo, interior_begin);
                    AffineConstantRowEdges<D, 3>(low, up, output, interior_end, width, fy, border_value, xinfo);
                }
                continue;
            }

            if (channels_ == 1 && std::is_same<D, float>::value) {
                const float* row_low = in_yl ? reinterpret_cast<const float*>(Row(yl)) : nullptr;
                const float* row_up  = in_yu ? reinterpret_cast<const float*>(Row(yu)) : nullptr;
                float* out_row = reinterpret_cast<float*>(dst.Row(y));
                int x = 0;
#if defined(__SSE2__)
                for (; x + 4 <= width; x += 4) {
                    __m128 vx = _mm_loadu_ps(xinfo.Weights(x));
                    __m128 vy = _mm_set1_ps(fy);
                    float tl0, tr0, bl0, br0, tl1, tr1, bl1, br1, tl2, tr2, bl2, br2, tl3, tr3, bl3, br3;
                    const XInfo& xi0 = xinfo[x + 0];
                    const XInfo& xi1 = xinfo[x + 1];
                    const XInfo& xi2 = xinfo[x + 2];
                    const XInfo& xi3 = xinfo[x + 3];
                    // lane 0
                    tl0 = (in_yl && xi0.in_l) ? row_low[xi0.xl] : static_cast<float>(border_value);
                    tr0 = (in_yl && xi0.in_u) ? row_low[xi0.xu] : static_cast<float>(border_value);
                    bl0 = (in_yu && xi0.in_l) ? row_up [xi0.xl] : static_cast<float>(border_value);
                    br0 = (in_yu && xi0.in_u) ? row_up [xi0.xu] : static_cast<float>(border_value);
                    // lane 1
                    tl1 = (in_yl && xi1.in_l) ? row_low[xi1.xl] : static_cast<float>(border_value);
                    tr1 = (in_yl && xi1.in_u) ? row_low[xi1.xu] : static_cast<float>(border_value);
                    bl1 = (in_yu && xi1.in_l) ? row_up [xi1.xl] : static_cast<float>(border_value);
                    br1 = (in_yu && xi1.in_u) ? row_up [xi1.xu] : static_cast<float>(border_value);
                    // lane 2
                    tl2 = (in_yl && xi2.in_l) ? row_low[xi2.xl] : static_cast<float>(border_value);
                    tr2 = (in_yl && xi2.in_u) ? row_low[xi2.xu] : static_cast<float>(border_value);
                    bl2 = (in_yu && xi2.in_l) ? row_up [xi2.xl] : static_cast<float>(border_value);
                    br2 = (in_yu && xi2.in_u) ? row_up [xi2.xu] : static_cast<float>(border_value);
                    // lane 3
                    tl3 = (in_yl && xi3.in_l) ? row_low[xi3.xl] : static_cast<float>(border_value);
                    tr3 = (in_yl && xi3.in_u) ? row_low[xi3.xu] : static_cast<float>(border_value);
                    bl3 = (in_yu && xi3.in_l) ? row_up [xi3.xl] : static_cast<float>(border_value);
                    br3 = (in_yu && xi3.in_u) ? row_up [xi3.xu] : static_cast<float>(border_value);
                    __m128 vtl = _mm_set_ps(tl3, tl2, tl1, tl0);
                    __m128 vtr = _mm_set_ps(tr3, tr2, tr1, tr0);
                    __m128 vbl = _mm_set_ps(bl3, bl2, bl1, bl0);
                    __m128 vbr = _mm_set_ps(br3, br2, br1, br0);
                    __m128 vtop = _mm_add_ps(vtl, _mm_mul_ps(_mm_sub_ps(vtr, vtl), vx));
                    __m128 vbot = _mm_add_ps(vbl, _mm_mul_ps(_mm_sub_ps(vbr, vbl), vx));
                    __m128 vout = _mm_add_ps(vtop, _mm_mul_ps(_mm_sub_ps(vbot, vtop), vy));
                    _mm_storeu_ps(out_row + x, vout);
                }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
                for (; x + 4 <= width; x += 4) {
                    float32x4_t vx = vld1q_f32(xinfo.Weights(x));
                    float32x4_t vy = vdupq_n_f32(fy);
                    float tl0, tr0, bl0, br0, tl1, tr1, bl1, br1, tl2, tr2, bl2, br2, tl3, tr3, bl3, br3;
                    const XInfo& xi0 = xinfo[x + 0];
                    const XInfo& xi1 = xinfo[x + 1];
                    const XInfo& xi2 = xinfo[x + 2];
                    const XInfo& xi3 = xinfo[x + 3];
                    tl0 = (in_yl && xi0.in_l) ? row_low[xi0.xl] : static_cast<float>(border_value);
                    tr0 = (in_yl && xi0.in_u) ? row_low[xi0.xu] : static_cast<float>(border_value);
                    bl0 = (in_yu && xi0.in_l) ? row_up [xi0.xl] : static_cast<float>(border_value);
                    br0 = (in_yu && xi0.in_u) ? row_up [xi0.xu] : static_cast<float>(border_value);
                    tl1 = (in_yl && xi1.in_l) ? row_low[xi1.xl] : static_cast<float>(border_value);
                    tr1 = (in_yl && xi1.in_u) ? row_low[xi1.xu] : static_cast<float>(border_value);
                    bl1 = (in_yu && xi1.in_l) ? row_up [xi1.xl] : static_cast<float>(border_value);
                    br1 = (in_yu && xi1.in_u) ? row_up [xi1.xu] : static_cast<float>(border_value);
                    tl2 = (in_yl && xi2.in_l) ? row_low[xi2.xl] : static_cast<float>(border_value);
                    tr2 = (in_yl && xi2.in_u) ? row_low[xi2.xu] : static_cast<float>(border_value);
                    bl2 = (in_yu && xi2.in_l) ? row_up [xi2.xl] : static_cast<float>(border_value);
                    br2 = (in_yu && xi2.in_u) ? row_up [xi2.xu] : static_cast<float>(border_value);
                    tl3 = (in_yl && xi3.in_l) ? row_low[xi3.xl] : static_cast<float>(border_value);
                    tr3 = (in_yl && xi3.in_u) ? row_low[xi3.xu] : static_cast<float>(border_value);
                    bl3 = (in_yu && xi3.in_l) ? row_up [xi3.xl] : static_cast<float>(border_value);
                    br3 = (in_yu && xi3.in_u) ? row_up [xi3.xu] : static_cast<float>(border_value);
                    float32x4_t vtl = vdupq_n_f32(0.f);
                    vtl = vsetq_lane_f32(tl0, vtl, 0);
                    vtl = vsetq_lane_f32(tl1, vtl, 1);
                    vtl = vsetq_lane_f32(tl2, vtl, 2);
                    vtl = vsetq_lane_f32(tl3, vtl, 3);
                    float32x4_t vtr = vdupq_n_f32(0.f);
                    vtr = vsetq_lane_f32(tr0, vtr, 0);
                    vtr = vsetq_lane_f32(tr1, vtr, 1);
                    vtr = vsetq_lane_f32(tr2, vtr, 2);
                    vtr = vsetq_lane_f32(tr3, vtr, 3);
                    float32x4_t vbl = vdupq_n_f32(0.f);
                    vbl = vsetq_lane_f32(bl0, vbl, 0);
                    vbl = vsetq_lane_f32(bl1, vbl, 1);
                    vbl = vsetq_lane_f32(bl2, vbl, 2);
                    vbl = vsetq_lane_f32(bl3, vbl, 3);
                    float32x4_t vbr = vdupq_n_f32(0.f);
                    vbr = vsetq_lane_f32(br0, vbr, 0);
                    vbr = vsetq_lane_f32(br1, vbr, 1);
                    vbr = vsetq_lane_f32(br2, vbr, 2);
                    vbr = vsetq_lane_f32(br3, vbr, 3);
                    float32x4_t vtop = vaddq_f32(vtl, vmulq_f32(vsubq_f32(vtr, vtl), vx));
                    float32x4_t vbot = vaddq_f32(vbl, vmulq_f32(vsubq_f32(vbr, vbl), vx));
                    float32x4_t vout = vaddq_f32(vtop, vmulq_f32(vsubq_f32(vbot, vtop), vy));
                    vst1q_f32(out_row + x, vout);
                }
#endif
                for (; x < width; ++x) {
                    const XInfo& xi = xinfo[x];
                    float tl = (in_yl && xi.in_l) ? row_low[xi.xl] : static_cast<float>(border_value);
                    float tr = (in_yl && xi.in_u) ? row_low[xi.xu] : static_cast<float>(border_value);
                    float bl = (in_yu && xi.in_l) ? row_up [xi.xl] : static_cast<float>(border_value);
                    float br = (in_yu && xi.in_u) ? row_up [xi.xu] : static_cast<float>(border_value);
                    float top = tl + (tr - tl) * xinfo.Fraction(x);
                    float bottom = bl + (br - bl) * xinfo.Fraction(x);
                    out_row[x] = top + (bottom - top) * fy;
                }
                continue;
            }

            if (channels_ == 1 && std::is_same<D, uint8_t>::value) {
                const uint8_t* row_low = in_yl ? reinterpret_cast<const uint8_t*>(Row(yl)) : nullptr;
                const uint8_t* row_up  = in_yu ? reinterpret_cast<const uint8_t*>(Row(yu)) : nullptr;
                uint8_t* out_row = reinterpret_cast<uint8_t*>(dst.Row(y));
                int x = 0;
#if defined(__SSE2__)
                for (; x + 4 <= width; x += 4) {
                    __m128 vx = _mm_loadu_ps(xinfo.Weights(x));
                    __m128 vy = _mm_set1_ps(fy);
                    float tl0, tr0, bl0, br0, tl1, tr1, bl1, br1, tl2, tr2, bl2, br2, tl3, tr3, bl3, br3;
                    const XInfo& xi0 = xinfo[x + 0];
                    const XInfo& xi1 = xinfo[x + 1];
                    const XInfo& xi2 = xinfo[x + 2];
                    const XInfo& xi3 = xinfo[x + 3];
                    tl0 = (in_yl && xi0.in_l) ? row_low[xi0.xl] : static_cast<uint8_t>(border_value);
                    tr0 = (in_yl && xi0.in_u) ? row_low[xi0.xu] : static_cast<uint8_t>(border_value);
                    bl0 = (in_yu && xi0.in_l) ? row_up [xi0.xl] : static_cast<uint8_t>(border_value);
                    br0 = (in_yu && xi0.in_u) ? row_up [xi0.xu] : static_cast<uint8_t>(border_value);
                    tl1 = (in_yl && xi1.in_l) ? row_low[xi1.xl] : static_cast<uint8_t>(border_value);
                    tr1 = (in_yl && xi1.in_u) ? row_low[xi1.xu] : static_cast<uint8_t>(border_value);
                    bl1 = (in_yu && xi1.in_l) ? row_up [xi1.xl] : static_cast<uint8_t>(border_value);
                    br1 = (in_yu && xi1.in_u) ? row_up [xi1.xu] : static_cast<uint8_t>(border_value);
                    tl2 = (in_yl && xi2.in_l) ? row_low[xi2.xl] : static_cast<uint8_t>(border_value);
                    tr2 = (in_yl && xi2.in_u) ? row_low[xi2.xu] : static_cast<uint8_t>(border_value);
                    bl2 = (in_yu && xi2.in_l) ? row_up [xi2.xl] : static_cast<uint8_t>(border_value);
                    br2 = (in_yu && xi2.in_u) ? row_up [xi2.xu] : static_cast<uint8_t>(border_value);
                    tl3 = (in_yl && xi3.in_l) ? row_low[xi3.xl] : static_cast<uint8_t>(border_value);
                    tr3 = (in_yl && xi3.in_u) ? row_low[xi3.xu] : static_cast<uint8_t>(border_value);
                    bl3 = (in_yu && xi3.in_l) ? row_up [xi3.xl] : static_cast<uint8_t>(border_value);
                    br3 = (in_yu && xi3.in_u) ? row_up [xi3.xu] : static_cast<uint8_t>(border_value);
                    __m128 vtl = _mm_set_ps(tl3, tl2, tl1, tl0);
                    __m128 vtr = _mm_set_ps(tr3, tr2, tr1, tr0);
                    __m128 vbl = _mm_set_ps(bl3, bl2, bl1, bl0);
                    __m128 vbr = _mm_set_ps(br3, br2, br1, br0);
                    __m128 vtop = _mm_add_ps(vtl, _mm_mul_ps(_mm_sub_ps(vtr, vtl), vx));
                    __m128 vbot = _mm_add_ps(vbl, _mm_mul_ps(_mm_sub_ps(vbr, vbl), vx));
                    __m128 vout = _mm_add_ps(vtop, _mm_mul_ps(_mm_sub_ps(vbot, vtop), vy));
                    // Match the scalar tail's round-half-up for nonnegative u8.
                    // Adding 0.5f first would double-round nextafter(0.5f, 0).
                    __m128i vi = _mm_cvttps_epi32(vout);
                    const __m128 fraction = _mm_sub_ps(vout, _mm_cvtepi32_ps(vi));
                    const __m128i increment = _mm_and_si128(
                      _mm_castps_si128(_mm_cmpge_ps(fraction, _mm_set1_ps(0.5f))),
                      _mm_set1_epi32(1));
                    vi = _mm_add_epi32(vi, increment);
                    __m128i vi16 = _mm_packs_epi32(vi, _mm_setzero_si128());
                    __m128i vi8  = _mm_packus_epi16(vi16, _mm_setzero_si128());
                    alignas(16) uint8_t tmp[16];
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(tmp), vi8);
                    out_row[x + 0] = tmp[0];
                    out_row[x + 1] = tmp[1];
                    out_row[x + 2] = tmp[2];
                    out_row[x + 3] = tmp[3];
                }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
                for (; x + 4 <= width; x += 4) {
                    float32x4_t vx = vld1q_f32(xinfo.Weights(x));
                    float32x4_t vy = vdupq_n_f32(fy);
                    float tl0, tr0, bl0, br0, tl1, tr1, bl1, br1, tl2, tr2, bl2, br2, tl3, tr3, bl3, br3;
                    const XInfo& xi0 = xinfo[x + 0];
                    const XInfo& xi1 = xinfo[x + 1];
                    const XInfo& xi2 = xinfo[x + 2];
                    const XInfo& xi3 = xinfo[x + 3];
                    tl0 = (in_yl && xi0.in_l) ? row_low[xi0.xl] : static_cast<uint8_t>(border_value);
                    tr0 = (in_yl && xi0.in_u) ? row_low[xi0.xu] : static_cast<uint8_t>(border_value);
                    bl0 = (in_yu && xi0.in_l) ? row_up [xi0.xl] : static_cast<uint8_t>(border_value);
                    br0 = (in_yu && xi0.in_u) ? row_up [xi0.xu] : static_cast<uint8_t>(border_value);
                    tl1 = (in_yl && xi1.in_l) ? row_low[xi1.xl] : static_cast<uint8_t>(border_value);
                    tr1 = (in_yl && xi1.in_u) ? row_low[xi1.xu] : static_cast<uint8_t>(border_value);
                    bl1 = (in_yu && xi1.in_l) ? row_up [xi1.xl] : static_cast<uint8_t>(border_value);
                    br1 = (in_yu && xi1.in_u) ? row_up [xi1.xu] : static_cast<uint8_t>(border_value);
                    tl2 = (in_yl && xi2.in_l) ? row_low[xi2.xl] : static_cast<uint8_t>(border_value);
                    tr2 = (in_yl && xi2.in_u) ? row_low[xi2.xu] : static_cast<uint8_t>(border_value);
                    bl2 = (in_yu && xi2.in_l) ? row_up [xi2.xl] : static_cast<uint8_t>(border_value);
                    br2 = (in_yu && xi2.in_u) ? row_up [xi2.xu] : static_cast<uint8_t>(border_value);
                    tl3 = (in_yl && xi3.in_l) ? row_low[xi3.xl] : static_cast<uint8_t>(border_value);
                    tr3 = (in_yl && xi3.in_u) ? row_low[xi3.xu] : static_cast<uint8_t>(border_value);
                    bl3 = (in_yu && xi3.in_l) ? row_up [xi3.xl] : static_cast<uint8_t>(border_value);
                    br3 = (in_yu && xi3.in_u) ? row_up [xi3.xu] : static_cast<uint8_t>(border_value);
                    float32x4_t vtl = vdupq_n_f32(0.f);
                    vtl = vsetq_lane_f32(tl0, vtl, 0);
                    vtl = vsetq_lane_f32(tl1, vtl, 1);
                    vtl = vsetq_lane_f32(tl2, vtl, 2);
                    vtl = vsetq_lane_f32(tl3, vtl, 3);
                    float32x4_t vtr = vdupq_n_f32(0.f);
                    vtr = vsetq_lane_f32(tr0, vtr, 0);
                    vtr = vsetq_lane_f32(tr1, vtr, 1);
                    vtr = vsetq_lane_f32(tr2, vtr, 2);
                    vtr = vsetq_lane_f32(tr3, vtr, 3);
                    float32x4_t vbl = vdupq_n_f32(0.f);
                    vbl = vsetq_lane_f32(bl0, vbl, 0);
                    vbl = vsetq_lane_f32(bl1, vbl, 1);
                    vbl = vsetq_lane_f32(bl2, vbl, 2);
                    vbl = vsetq_lane_f32(bl3, vbl, 3);
                    float32x4_t vbr = vdupq_n_f32(0.f);
                    vbr = vsetq_lane_f32(br0, vbr, 0);
                    vbr = vsetq_lane_f32(br1, vbr, 1);
                    vbr = vsetq_lane_f32(br2, vbr, 2);
                    vbr = vsetq_lane_f32(br3, vbr, 3);
                    float32x4_t vtop = vaddq_f32(vtl, vmulq_f32(vsubq_f32(vtr, vtl), vx));
                    float32x4_t vbot = vaddq_f32(vbl, vmulq_f32(vsubq_f32(vbr, vbl), vx));
                    float32x4_t vout = vaddq_f32(vtop, vmulq_f32(vsubq_f32(vbot, vtop), vy));
#if defined(__aarch64__)
                    // Round all four lanes exactly as std::round, then narrow
                    // and store one packed word (no scalar lane round-trips).
                    const uint16x4_t rounded = vmovn_u32(vcvtaq_u32_f32(vout));
                    const uint8x8_t bytes = vmovn_u16(vcombine_u16(rounded, rounded));
                    const uint32_t packed = vget_lane_u32(vreinterpret_u32_u8(bytes), 0);
                    std::memcpy(out_row + x, &packed, sizeof(packed));
#else
                    float outv[4];
                    vst1q_f32(outv, vout);
                    out_row[x + 0] = static_cast<uint8_t>(std::round(outv[0]));
                    out_row[x + 1] = static_cast<uint8_t>(std::round(outv[1]));
                    out_row[x + 2] = static_cast<uint8_t>(std::round(outv[2]));
                    out_row[x + 3] = static_cast<uint8_t>(std::round(outv[3]));
#endif
                }
#endif
                for (; x < width; ++x) {
                    const XInfo& xi = xinfo[x];
                    float tl = (in_yl && xi.in_l) ? row_low[xi.xl] : static_cast<uint8_t>(border_value);
                    float tr = (in_yl && xi.in_u) ? row_low[xi.xu] : static_cast<uint8_t>(border_value);
                    float bl = (in_yu && xi.in_l) ? row_up [xi.xl] : static_cast<uint8_t>(border_value);
                    float br = (in_yu && xi.in_u) ? row_up [xi.xu] : static_cast<uint8_t>(border_value);
                    float top = tl + (tr - tl) * xinfo.Fraction(x);
                    float bottom = bl + (br - bl) * xinfo.Fraction(x);
                    out_row[x] = static_cast<uint8_t>(std::round(top + (bottom - top) * fy));
                }
                continue;
            }

            // 3-channel float (NEON/SSE, replicate and constant)
            if (channels_ == 3 && std::is_same<D, float>::value) {
                const float* row_low = in_yl ? reinterpret_cast<const float*>(Row(yl)) : nullptr;
                const float* row_up  = in_yu ? reinterpret_cast<const float*>(Row(yu)) : nullptr;
                float* out_row = reinterpret_cast<float*>(dst.Row(y));
                int x = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
                for (; x + 4 <= width; x += 4) {
                    float32x4_t vx = vld1q_f32(xinfo.Weights(x));
                    float32x4_t vy = vdupq_n_f32(fy);
                    for (int cc = 0; cc < 3; ++cc) {
                        const XInfo& xi0 = xinfo[x + 0];
                        const XInfo& xi1 = xinfo[x + 1];
                        const XInfo& xi2 = xinfo[x + 2];
                        const XInfo& xi3 = xinfo[x + 3];
                        // gather neighbors with border handling
                        float tl0 = (in_yl && xi0.in_l) ? row_low[xi0.xl * 3 + cc] : static_cast<float>(border_value);
                        float tr0 = (in_yl && xi0.in_u) ? row_low[xi0.xu * 3 + cc] : static_cast<float>(border_value);
                        float bl0 = (in_yu && xi0.in_l) ? row_up [xi0.xl * 3 + cc] : static_cast<float>(border_value);
                        float br0 = (in_yu && xi0.in_u) ? row_up [xi0.xu * 3 + cc] : static_cast<float>(border_value);
                        float tl1 = (in_yl && xi1.in_l) ? row_low[xi1.xl * 3 + cc] : static_cast<float>(border_value);
                        float tr1 = (in_yl && xi1.in_u) ? row_low[xi1.xu * 3 + cc] : static_cast<float>(border_value);
                        float bl1 = (in_yu && xi1.in_l) ? row_up [xi1.xl * 3 + cc] : static_cast<float>(border_value);
                        float br1 = (in_yu && xi1.in_u) ? row_up [xi1.xu * 3 + cc] : static_cast<float>(border_value);
                        float tl2 = (in_yl && xi2.in_l) ? row_low[xi2.xl * 3 + cc] : static_cast<float>(border_value);
                        float tr2 = (in_yl && xi2.in_u) ? row_low[xi2.xu * 3 + cc] : static_cast<float>(border_value);
                        float bl2 = (in_yu && xi2.in_l) ? row_up [xi2.xl * 3 + cc] : static_cast<float>(border_value);
                        float br2 = (in_yu && xi2.in_u) ? row_up [xi2.xu * 3 + cc] : static_cast<float>(border_value);
                        float tl3 = (in_yl && xi3.in_l) ? row_low[xi3.xl * 3 + cc] : static_cast<float>(border_value);
                        float tr3 = (in_yl && xi3.in_u) ? row_low[xi3.xu * 3 + cc] : static_cast<float>(border_value);
                        float bl3 = (in_yu && xi3.in_l) ? row_up [xi3.xl * 3 + cc] : static_cast<float>(border_value);
                        float br3 = (in_yu && xi3.in_u) ? row_up [xi3.xu * 3 + cc] : static_cast<float>(border_value);
                        float32x4_t vtl = vdupq_n_f32(0.f);
                        vtl = vsetq_lane_f32(tl0, vtl, 0);
                        vtl = vsetq_lane_f32(tl1, vtl, 1);
                        vtl = vsetq_lane_f32(tl2, vtl, 2);
                        vtl = vsetq_lane_f32(tl3, vtl, 3);
                        float32x4_t vtr = vdupq_n_f32(0.f);
                        vtr = vsetq_lane_f32(tr0, vtr, 0);
                        vtr = vsetq_lane_f32(tr1, vtr, 1);
                        vtr = vsetq_lane_f32(tr2, vtr, 2);
                        vtr = vsetq_lane_f32(tr3, vtr, 3);
                        float32x4_t vbl = vdupq_n_f32(0.f);
                        vbl = vsetq_lane_f32(bl0, vbl, 0);
                        vbl = vsetq_lane_f32(bl1, vbl, 1);
                        vbl = vsetq_lane_f32(bl2, vbl, 2);
                        vbl = vsetq_lane_f32(bl3, vbl, 3);
                        float32x4_t vbr = vdupq_n_f32(0.f);
                        vbr = vsetq_lane_f32(br0, vbr, 0);
                        vbr = vsetq_lane_f32(br1, vbr, 1);
                        vbr = vsetq_lane_f32(br2, vbr, 2);
                        vbr = vsetq_lane_f32(br3, vbr, 3);
                        float32x4_t vtop = vaddq_f32(vtl, vmulq_f32(vsubq_f32(vtr, vtl), vx));
                        float32x4_t vbot = vaddq_f32(vbl, vmulq_f32(vsubq_f32(vbr, vbl), vx));
                        float32x4_t vout = vaddq_f32(vtop, vmulq_f32(vsubq_f32(vbot, vtop), vy));
                        float outv[4];
                        vst1q_f32(outv, vout);
                        out_row[(x + 0) * 3 + cc] = outv[0];
                        out_row[(x + 1) * 3 + cc] = outv[1];
                        out_row[(x + 2) * 3 + cc] = outv[2];
                        out_row[(x + 3) * 3 + cc] = outv[3];
                    }
                }
#elif defined(__SSE2__)
                for (; x + 4 <= width; x += 4) {
                    __m128 vx = _mm_loadu_ps(xinfo.Weights(x));
                    __m128 vy = _mm_set1_ps(fy);
                    for (int cc = 0; cc < 3; ++cc) {
                        const XInfo& xi0 = xinfo[x + 0];
                        const XInfo& xi1 = xinfo[x + 1];
                        const XInfo& xi2 = xinfo[x + 2];
                        const XInfo& xi3 = xinfo[x + 3];
                        float tl0 = (in_yl && xi0.in_l) ? row_low[xi0.xl * 3 + cc] : static_cast<float>(border_value);
                        float tr0 = (in_yl && xi0.in_u) ? row_low[xi0.xu * 3 + cc] : static_cast<float>(border_value);
                        float bl0 = (in_yu && xi0.in_l) ? row_up [xi0.xl * 3 + cc] : static_cast<float>(border_value);
                        float br0 = (in_yu && xi0.in_u) ? row_up [xi0.xu * 3 + cc] : static_cast<float>(border_value);
                        float tl1 = (in_yl && xi1.in_l) ? row_low[xi1.xl * 3 + cc] : static_cast<float>(border_value);
                        float tr1 = (in_yl && xi1.in_u) ? row_low[xi1.xu * 3 + cc] : static_cast<float>(border_value);
                        float bl1 = (in_yu && xi1.in_l) ? row_up [xi1.xl * 3 + cc] : static_cast<float>(border_value);
                        float br1 = (in_yu && xi1.in_u) ? row_up [xi1.xu * 3 + cc] : static_cast<float>(border_value);
                        float tl2 = (in_yl && xi2.in_l) ? row_low[xi2.xl * 3 + cc] : static_cast<float>(border_value);
                        float tr2 = (in_yl && xi2.in_u) ? row_low[xi2.xu * 3 + cc] : static_cast<float>(border_value);
                        float bl2 = (in_yu && xi2.in_l) ? row_up [xi2.xl * 3 + cc] : static_cast<float>(border_value);
                        float br2 = (in_yu && xi2.in_u) ? row_up [xi2.xu * 3 + cc] : static_cast<float>(border_value);
                        float tl3 = (in_yl && xi3.in_l) ? row_low[xi3.xl * 3 + cc] : static_cast<float>(border_value);
                        float tr3 = (in_yl && xi3.in_u) ? row_low[xi3.xu * 3 + cc] : static_cast<float>(border_value);
                        float bl3 = (in_yu && xi3.in_l) ? row_up [xi3.xl * 3 + cc] : static_cast<float>(border_value);
                        float br3 = (in_yu && xi3.in_u) ? row_up [xi3.xu * 3 + cc] : static_cast<float>(border_value);
                        __m128 vtl = _mm_set_ps(tl3, tl2, tl1, tl0);
                        __m128 vtr = _mm_set_ps(tr3, tr2, tr1, tr0);
                        __m128 vbl = _mm_set_ps(bl3, bl2, bl1, bl0);
                        __m128 vbr = _mm_set_ps(br3, br2, br1, br0);
                        __m128 vtop = _mm_add_ps(vtl, _mm_mul_ps(_mm_sub_ps(vtr, vtl), vx));
                        __m128 vbot = _mm_add_ps(vbl, _mm_mul_ps(_mm_sub_ps(vbr, vbl), vx));
                        __m128 vout = _mm_add_ps(vtop, _mm_mul_ps(_mm_sub_ps(vbot, vtop), vy));
                        float outv[4];
                        _mm_storeu_ps(outv, vout);
                        out_row[(x + 0) * 3 + cc] = outv[0];
                        out_row[(x + 1) * 3 + cc] = outv[1];
                        out_row[(x + 2) * 3 + cc] = outv[2];
                        out_row[(x + 3) * 3 + cc] = outv[3];
                    }
                }
#endif
                for (; x < width; ++x) {
                    const XInfo& xi = xinfo[x];
                    for (int cc = 0; cc < 3; ++cc) {
                        float tl = (in_yl && xi.in_l) ? row_low[xi.xl * 3 + cc] : static_cast<float>(border_value);
                        float tr = (in_yl && xi.in_u) ? row_low[xi.xu * 3 + cc] : static_cast<float>(border_value);
                        float bl = (in_yu && xi.in_l) ? row_up [xi.xl * 3 + cc] : static_cast<float>(border_value);
                        float br = (in_yu && xi.in_u) ? row_up [xi.xu * 3 + cc] : static_cast<float>(border_value);
                        float top = tl + (tr - tl) * xinfo.Fraction(x);
                        float bottom = bl + (br - bl) * xinfo.Fraction(x);
                        out_row[x * 3 + cc] = top + (bottom - top) * fy;
                    }
                }
                continue;
            }

            // 3-channel uint8 (NEON/SSE, replicate and constant)
            if (channels_ == 3 && std::is_same<D, uint8_t>::value) {
                const uint8_t* row_low = in_yl ? reinterpret_cast<const uint8_t*>(Row(yl)) : nullptr;
                const uint8_t* row_up  = in_yu ? reinterpret_cast<const uint8_t*>(Row(yu)) : nullptr;
                uint8_t* out_row = reinterpret_cast<uint8_t*>(dst.Row(y));
                int x = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
                for (; x + 4 <= width; x += 4) {
                    float32x4_t vx = vld1q_f32(xinfo.Weights(x));
                    float32x4_t vy = vdupq_n_f32(fy);
                    for (int cc = 0; cc < 3; ++cc) {
                        const XInfo& xi0 = xinfo[x + 0];
                        const XInfo& xi1 = xinfo[x + 1];
                        const XInfo& xi2 = xinfo[x + 2];
                        const XInfo& xi3 = xinfo[x + 3];
                        float tl0 = (in_yl && xi0.in_l) ? static_cast<float>(row_low[xi0.xl * 3 + cc]) : static_cast<float>(border_value);
                        float tr0 = (in_yl && xi0.in_u) ? static_cast<float>(row_low[xi0.xu * 3 + cc]) : static_cast<float>(border_value);
                        float bl0 = (in_yu && xi0.in_l) ? static_cast<float>(row_up [xi0.xl * 3 + cc]) : static_cast<float>(border_value);
                        float br0 = (in_yu && xi0.in_u) ? static_cast<float>(row_up [xi0.xu * 3 + cc]) : static_cast<float>(border_value);
                        float tl1 = (in_yl && xi1.in_l) ? static_cast<float>(row_low[xi1.xl * 3 + cc]) : static_cast<float>(border_value);
                        float tr1 = (in_yl && xi1.in_u) ? static_cast<float>(row_low[xi1.xu * 3 + cc]) : static_cast<float>(border_value);
                        float bl1 = (in_yu && xi1.in_l) ? static_cast<float>(row_up [xi1.xl * 3 + cc]) : static_cast<float>(border_value);
                        float br1 = (in_yu && xi1.in_u) ? static_cast<float>(row_up [xi1.xu * 3 + cc]) : static_cast<float>(border_value);
                        float tl2 = (in_yl && xi2.in_l) ? static_cast<float>(row_low[xi2.xl * 3 + cc]) : static_cast<float>(border_value);
                        float tr2 = (in_yl && xi2.in_u) ? static_cast<float>(row_low[xi2.xu * 3 + cc]) : static_cast<float>(border_value);
                        float bl2 = (in_yu && xi2.in_l) ? static_cast<float>(row_up [xi2.xl * 3 + cc]) : static_cast<float>(border_value);
                        float br2 = (in_yu && xi2.in_u) ? static_cast<float>(row_up [xi2.xu * 3 + cc]) : static_cast<float>(border_value);
                        float tl3 = (in_yl && xi3.in_l) ? static_cast<float>(row_low[xi3.xl * 3 + cc]) : static_cast<float>(border_value);
                        float tr3 = (in_yl && xi3.in_u) ? static_cast<float>(row_low[xi3.xu * 3 + cc]) : static_cast<float>(border_value);
                        float bl3 = (in_yu && xi3.in_l) ? static_cast<float>(row_up [xi3.xl * 3 + cc]) : static_cast<float>(border_value);
                        float br3 = (in_yu && xi3.in_u) ? static_cast<float>(row_up [xi3.xu * 3 + cc]) : static_cast<float>(border_value);
                        float32x4_t vtl = vdupq_n_f32(0.f);
                        vtl = vsetq_lane_f32(tl0, vtl, 0);
                        vtl = vsetq_lane_f32(tl1, vtl, 1);
                        vtl = vsetq_lane_f32(tl2, vtl, 2);
                        vtl = vsetq_lane_f32(tl3, vtl, 3);
                        float32x4_t vtr = vdupq_n_f32(0.f);
                        vtr = vsetq_lane_f32(tr0, vtr, 0);
                        vtr = vsetq_lane_f32(tr1, vtr, 1);
                        vtr = vsetq_lane_f32(tr2, vtr, 2);
                        vtr = vsetq_lane_f32(tr3, vtr, 3);
                        float32x4_t vbl = vdupq_n_f32(0.f);
                        vbl = vsetq_lane_f32(bl0, vbl, 0);
                        vbl = vsetq_lane_f32(bl1, vbl, 1);
                        vbl = vsetq_lane_f32(bl2, vbl, 2);
                        vbl = vsetq_lane_f32(bl3, vbl, 3);
                        float32x4_t vbr = vdupq_n_f32(0.f);
                        vbr = vsetq_lane_f32(br0, vbr, 0);
                        vbr = vsetq_lane_f32(br1, vbr, 1);
                        vbr = vsetq_lane_f32(br2, vbr, 2);
                        vbr = vsetq_lane_f32(br3, vbr, 3);
                        float32x4_t vtop = vaddq_f32(vtl, vmulq_f32(vsubq_f32(vtr, vtl), vx));
                        float32x4_t vbot = vaddq_f32(vbl, vmulq_f32(vsubq_f32(vbr, vbl), vx));
                        float32x4_t vout = vaddq_f32(vtop, vmulq_f32(vsubq_f32(vbot, vtop), vy));
                        float outv[4];
                        vst1q_f32(outv, vout);
                        for (int k = 0; k < 4; ++k) {
                            float vv = outv[k];
                            if (vv < 0.f) vv = 0.f;
                            if (vv > 255.f) vv = 255.f;
                            out_row[(x + k) * 3 + cc] = static_cast<uint8_t>(std::round(vv));
                        }
                    }
                }
#elif defined(__SSE2__)
                for (; x + 4 <= width; x += 4) {
                    __m128 vx = _mm_loadu_ps(xinfo.Weights(x));
                    __m128 vy = _mm_set1_ps(fy);
                    for (int cc = 0; cc < 3; ++cc) {
                        const XInfo& xi0 = xinfo[x + 0];
                        const XInfo& xi1 = xinfo[x + 1];
                        const XInfo& xi2 = xinfo[x + 2];
                        const XInfo& xi3 = xinfo[x + 3];
                        float tl0 = (in_yl && xi0.in_l) ? static_cast<float>(row_low[xi0.xl * 3 + cc]) : static_cast<float>(border_value);
                        float tr0 = (in_yl && xi0.in_u) ? static_cast<float>(row_low[xi0.xu * 3 + cc]) : static_cast<float>(border_value);
                        float bl0 = (in_yu && xi0.in_l) ? static_cast<float>(row_up [xi0.xl * 3 + cc]) : static_cast<float>(border_value);
                        float br0 = (in_yu && xi0.in_u) ? static_cast<float>(row_up [xi0.xu * 3 + cc]) : static_cast<float>(border_value);
                        float tl1 = (in_yl && xi1.in_l) ? static_cast<float>(row_low[xi1.xl * 3 + cc]) : static_cast<float>(border_value);
                        float tr1 = (in_yl && xi1.in_u) ? static_cast<float>(row_low[xi1.xu * 3 + cc]) : static_cast<float>(border_value);
                        float bl1 = (in_yu && xi1.in_l) ? static_cast<float>(row_up [xi1.xl * 3 + cc]) : static_cast<float>(border_value);
                        float br1 = (in_yu && xi1.in_u) ? static_cast<float>(row_up [xi1.xu * 3 + cc]) : static_cast<float>(border_value);
                        float tl2 = (in_yl && xi2.in_l) ? static_cast<float>(row_low[xi2.xl * 3 + cc]) : static_cast<float>(border_value);
                        float tr2 = (in_yl && xi2.in_u) ? static_cast<float>(row_low[xi2.xu * 3 + cc]) : static_cast<float>(border_value);
                        float bl2 = (in_yu && xi2.in_l) ? static_cast<float>(row_up [xi2.xl * 3 + cc]) : static_cast<float>(border_value);
                        float br2 = (in_yu && xi2.in_u) ? static_cast<float>(row_up [xi2.xu * 3 + cc]) : static_cast<float>(border_value);
                        float tl3 = (in_yl && xi3.in_l) ? static_cast<float>(row_low[xi3.xl * 3 + cc]) : static_cast<float>(border_value);
                        float tr3 = (in_yl && xi3.in_u) ? static_cast<float>(row_low[xi3.xu * 3 + cc]) : static_cast<float>(border_value);
                        float bl3 = (in_yu && xi3.in_l) ? static_cast<float>(row_up [xi3.xl * 3 + cc]) : static_cast<float>(border_value);
                        float br3 = (in_yu && xi3.in_u) ? static_cast<float>(row_up [xi3.xu * 3 + cc]) : static_cast<float>(border_value);
                        __m128 vtl = _mm_set_ps(tl3, tl2, tl1, tl0);
                        __m128 vtr = _mm_set_ps(tr3, tr2, tr1, tr0);
                        __m128 vbl = _mm_set_ps(bl3, bl2, bl1, bl0);
                        __m128 vbr = _mm_set_ps(br3, br2, br1, br0);
                        __m128 vtop = _mm_add_ps(vtl, _mm_mul_ps(_mm_sub_ps(vtr, vtl), vx));
                        __m128 vbot = _mm_add_ps(vbl, _mm_mul_ps(_mm_sub_ps(vbr, vbl), vx));
                        __m128 vout = _mm_add_ps(vtop, _mm_mul_ps(_mm_sub_ps(vbot, vtop), vy));
                        float outv[4];
                        _mm_storeu_ps(outv, vout);
                        for (int k = 0; k < 4; ++k) {
                            float vv = outv[k];
                            if (vv < 0.f) vv = 0.f;
                            if (vv > 255.f) vv = 255.f;
                            out_row[(x + k) * 3 + cc] = static_cast<uint8_t>(std::round(vv));
                        }
                    }
                }
#endif
                for (; x < width; ++x) {
                    const XInfo& xi = xinfo[x];
                    for (int cc = 0; cc < 3; ++cc) {
                        float tl = (in_yl && xi.in_l) ? static_cast<float>(row_low[xi.xl * 3 + cc]) : static_cast<float>(border_value);
                        float tr = (in_yl && xi.in_u) ? static_cast<float>(row_low[xi.xu * 3 + cc]) : static_cast<float>(border_value);
                        float bl = (in_yu && xi.in_l) ? static_cast<float>(row_up [xi.xl * 3 + cc]) : static_cast<float>(border_value);
                        float br = (in_yu && xi.in_u) ? static_cast<float>(row_up [xi.xu * 3 + cc]) : static_cast<float>(border_value);
                        float top = tl + (tr - tl) * xinfo.Fraction(x);
                        float bottom = bl + (br - bl) * xinfo.Fraction(x);
                        float vv = top + (bottom - top) * fy;
                        if (vv < 0.f) vv = 0.f;
                        if (vv > 255.f) vv = 255.f;
                        out_row[x * 3 + cc] = static_cast<uint8_t>(std::round(vv));
                    }
                }
                continue;
            }

            // Fallback for other channel/types under this fast shape
            auto dst_iter = dst.Row(y);
            for (int x = 0; x < width; ++x) {
                for (int cch = 0; cch < channels_; ++cch) {
                    *(dst_iter++) = InterpolateBilinear(
                      (in_yl && xinfo[x].in_l && yl >= 0 && yl < height_ && xinfo[x].xl >= 0 && xinfo[x].xl < width_) ? at(yl, xinfo[x].xl)[cch] : border_value,
                      (in_yl && xinfo[x].in_u && yl >= 0 && yl < height_ && xinfo[x].xu >= 0 && xinfo[x].xu < width_) ? at(yl, xinfo[x].xu)[cch] : border_value,
                      (in_yu && xinfo[x].in_l && yu >= 0 && yu < height_ && xinfo[x].xl >= 0 && xinfo[x].xl < width_) ? at(yu, xinfo[x].xl)[cch] : border_value,
                      (in_yu && xinfo[x].in_u && yu >= 0 && yu < height_ && xinfo[x].xu >= 0 && xinfo[x].xu < width_) ? at(yu, xinfo[x].xu)[cch] : border_value,
                      xinfo.Fraction(x), fy);
                }
            }
        }
        return dst;
    }

    // General affine (with shear/rotation) SIMD micro-kernel
#if OKCV_ENABLE_AFFINE_GENERAL_SIMD
    {
        Bitmap<D> dst;
        dst.Reset(width, height, channels_);
        const float a = matrix[0], b = matrix[1], c = matrix[3], d = matrix[4], tx = matrix[2], ty = matrix[5];
        // Gate 3ch u8 general-affine SIMD with a macro for safe rollback
#ifndef OKCV_ENABLE_AFFINE_U8_3CH_SIMD
#define OKCV_ENABLE_AFFINE_U8_3CH_SIMD 0    // Tested that opening it will be slower
#endif
#if !OKCV_ENABLE_AFFINE_U8_3CH_SIMD
        if (channels_ == 3 && std::is_same<D, uint8_t>::value) {
            return AffineBilinearReference(width, height, matrix, border_mode, border_value);
        }
#endif
        for (int y = 0; y < height; ++y) {
            float yb = y * b + tx;
            float yd = y * d + ty;
            D* out_row_any = dst.Row(y);

            if (channels_ == 1 && std::is_same<D, float>::value) {
                float* out_row = reinterpret_cast<float*>(out_row_any);
                int x = 0;
#if defined(__SSE2__)
                for (; x + 4 <= width; x += 4) {
                    float sx0 = yb + (x + 0) * a;
                    float sy0 = yd + (x + 0) * c;
                    float sx1 = yb + (x + 1) * a;
                    float sy1 = yd + (x + 1) * c;
                    float sx2 = yb + (x + 2) * a;
                    float sy2 = yd + (x + 2) * c;
                    float sx3 = yb + (x + 3) * a;
                    float sy3 = yd + (x + 3) * c;
                    int xl0 = static_cast<int>(std::floor(sx0)), yl0 = static_cast<int>(std::floor(sy0));
                    int xl1 = static_cast<int>(std::floor(sx1)), yl1 = static_cast<int>(std::floor(sy1));
                    int xl2 = static_cast<int>(std::floor(sx2)), yl2 = static_cast<int>(std::floor(sy2));
                    int xl3 = static_cast<int>(std::floor(sx3)), yl3 = static_cast<int>(std::floor(sy3));
                    float fx0 = sx0 - xl0, fy0 = sy0 - yl0;
                    float fx1 = sx1 - xl1, fy1 = sy1 - yl1;
                    float fx2 = sx2 - xl2, fy2 = sy2 - yl2;
                    float fx3 = sx3 - xl3, fy3 = sy3 - yl3;
                    auto clamp = [&](int& u, int lo, int hi) { if (u < lo) u = lo; if (u > hi) u = hi; };
                    if (border_mode == BORDER_MODE_REPLICATE) {
                        clamp(xl0, 0, width_ - 1); clamp(xl1, 0, width_ - 1);
                        clamp(xl2, 0, width_ - 1); clamp(xl3, 0, width_ - 1);
                        clamp(yl0, 0, height_ - 1); clamp(yl1, 0, height_ - 1);
                        clamp(yl2, 0, height_ - 1); clamp(yl3, 0, height_ - 1);
                    }
                    int xu0 = xl0 + 1, yu0 = yl0 + 1;
                    int xu1 = xl1 + 1, yu1 = yl1 + 1;
                    int xu2 = xl2 + 1, yu2 = yl2 + 1;
                    int xu3 = xl3 + 1, yu3 = yl3 + 1;
                    if (border_mode == BORDER_MODE_REPLICATE) {
                        if (xu0 >= width_) xu0 = width_ - 1; if (yu0 >= height_) yu0 = height_ - 1;
                        if (xu1 >= width_) xu1 = width_ - 1; if (yu1 >= height_) yu1 = height_ - 1;
                        if (xu2 >= width_) xu2 = width_ - 1; if (yu2 >= height_) yu2 = height_ - 1;
                        if (xu3 >= width_) xu3 = width_ - 1; if (yu3 >= height_) yu3 = height_ - 1;
                    }
                    float tl0, tr0, bl0, br0, tl1, tr1, bl1, br1, tl2, tr2, bl2, br2, tl3, tr3, bl3, br3;
                    if (border_mode == BORDER_MODE_CONSTANT) {
                        auto inx = [&](int u){ return u >= 0 && u < width_; };
                        auto iny = [&](int v){ return v >= 0 && v < height_; };
                        tl0 = (inx(xl0) && iny(yl0)) ? at(yl0, xl0)[0] : static_cast<float>(border_value);
                        tr0 = (inx(xu0) && iny(yl0)) ? at(yl0, xu0)[0] : static_cast<float>(border_value);
                        bl0 = (inx(xl0) && iny(yu0)) ? at(yu0, xl0)[0] : static_cast<float>(border_value);
                        br0 = (inx(xu0) && iny(yu0)) ? at(yu0, xu0)[0] : static_cast<float>(border_value);
                        tl1 = (inx(xl1) && iny(yl1)) ? at(yl1, xl1)[0] : static_cast<float>(border_value);
                        tr1 = (inx(xu1) && iny(yl1)) ? at(yl1, xu1)[0] : static_cast<float>(border_value);
                        bl1 = (inx(xl1) && iny(yu1)) ? at(yu1, xl1)[0] : static_cast<float>(border_value);
                        br1 = (inx(xu1) && iny(yu1)) ? at(yu1, xu1)[0] : static_cast<float>(border_value);
                        tl2 = (inx(xl2) && iny(yl2)) ? at(yl2, xl2)[0] : static_cast<float>(border_value);
                        tr2 = (inx(xu2) && iny(yl2)) ? at(yl2, xu2)[0] : static_cast<float>(border_value);
                        bl2 = (inx(xl2) && iny(yu2)) ? at(yu2, xl2)[0] : static_cast<float>(border_value);
                        br2 = (inx(xu2) && iny(yu2)) ? at(yu2, xu2)[0] : static_cast<float>(border_value);
                        tl3 = (inx(xl3) && iny(yl3)) ? at(yl3, xl3)[0] : static_cast<float>(border_value);
                        tr3 = (inx(xu3) && iny(yl3)) ? at(yl3, xu3)[0] : static_cast<float>(border_value);
                        bl3 = (inx(xl3) && iny(yu3)) ? at(yu3, xl3)[0] : static_cast<float>(border_value);
                        br3 = (inx(xu3) && iny(yu3)) ? at(yu3, xu3)[0] : static_cast<float>(border_value);
                    } else {
                        tl0 = at(yl0, xl0)[0]; tr0 = at(yl0, xu0)[0]; bl0 = at(yu0, xl0)[0]; br0 = at(yu0, xu0)[0];
                        tl1 = at(yl1, xl1)[0]; tr1 = at(yl1, xu1)[0]; bl1 = at(yu1, xl1)[0]; br1 = at(yu1, xu1)[0];
                        tl2 = at(yl2, xl2)[0]; tr2 = at(yl2, xu2)[0]; bl2 = at(yu2, xl2)[0]; br2 = at(yu2, xu2)[0];
                        tl3 = at(yl3, xl3)[0]; tr3 = at(yl3, xu3)[0]; bl3 = at(yu3, xl3)[0]; br3 = at(yu3, xu3)[0];
                    }
                    __m128 vfx = _mm_set_ps(fx3, fx2, fx1, fx0);
                    __m128 vfy = _mm_set_ps(fy3, fy2, fy1, fy0);
                    __m128 vtl = _mm_set_ps(tl3, tl2, tl1, tl0);
                    __m128 vtr = _mm_set_ps(tr3, tr2, tr1, tr0);
                    __m128 vbl = _mm_set_ps(bl3, bl2, bl1, bl0);
                    __m128 vbr = _mm_set_ps(br3, br2, br1, br0);
                    __m128 vtop = _mm_add_ps(vtl, _mm_mul_ps(_mm_sub_ps(vtr, vtl), vfx));
                    __m128 vbot = _mm_add_ps(vbl, _mm_mul_ps(_mm_sub_ps(vbr, vbl), vfx));
                    __m128 vout = _mm_add_ps(vtop, _mm_mul_ps(_mm_sub_ps(vbot, vtop), vfy));
                    _mm_storeu_ps(out_row + x, vout);
                }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
                for (; x + 4 <= width; x += 4) {
                    float sx0 = yb + (x + 0) * a;
                    float sy0 = yd + (x + 0) * c;
                    float sx1 = yb + (x + 1) * a;
                    float sy1 = yd + (x + 1) * c;
                    float sx2 = yb + (x + 2) * a;
                    float sy2 = yd + (x + 2) * c;
                    float sx3 = yb + (x + 3) * a;
                    float sy3 = yd + (x + 3) * c;
                    int xl0 = static_cast<int>(std::floor(sx0)), yl0 = static_cast<int>(std::floor(sy0));
                    int xl1 = static_cast<int>(std::floor(sx1)), yl1 = static_cast<int>(std::floor(sy1));
                    int xl2 = static_cast<int>(std::floor(sx2)), yl2 = static_cast<int>(std::floor(sy2));
                    int xl3 = static_cast<int>(std::floor(sx3)), yl3 = static_cast<int>(std::floor(sy3));
                    float fx0 = sx0 - xl0, fy0 = sy0 - yl0;
                    float fx1 = sx1 - xl1, fy1 = sy1 - yl1;
                    float fx2 = sx2 - xl2, fy2 = sy2 - yl2;
                    float fx3 = sx3 - xl3, fy3 = sy3 - yl3;
                    auto clamp = [&](int& u, int lo, int hi) { if (u < lo) u = lo; if (u > hi) u = hi; };
                    if (border_mode == BORDER_MODE_REPLICATE) {
                        clamp(xl0, 0, width_ - 1); clamp(xl1, 0, width_ - 1);
                        clamp(xl2, 0, width_ - 1); clamp(xl3, 0, width_ - 1);
                        clamp(yl0, 0, height_ - 1); clamp(yl1, 0, height_ - 1);
                        clamp(yl2, 0, height_ - 1); clamp(yl3, 0, height_ - 1);
                    }
                    int xu0 = xl0 + 1, yu0 = yl0 + 1;
                    int xu1 = xl1 + 1, yu1 = yl1 + 1;
                    int xu2 = xl2 + 1, yu2 = yl2 + 1;
                    int xu3 = xl3 + 1, yu3 = yl3 + 1;
                    if (border_mode == BORDER_MODE_REPLICATE) {
                        if (xu0 >= width_) xu0 = width_ - 1; if (yu0 >= height_) yu0 = height_ - 1;
                        if (xu1 >= width_) xu1 = width_ - 1; if (yu1 >= height_) yu1 = height_ - 1;
                        if (xu2 >= width_) xu2 = width_ - 1; if (yu2 >= height_) yu2 = height_ - 1;
                        if (xu3 >= width_) xu3 = width_ - 1; if (yu3 >= height_) yu3 = height_ - 1;
                    }
                    float tl0, tr0, bl0, br0, tl1, tr1, bl1, br1, tl2, tr2, bl2, br2, tl3, tr3, bl3, br3;
                    if (border_mode == BORDER_MODE_CONSTANT) {
                        auto inx = [&](int u){ return u >= 0 && u < width_; };
                        auto iny = [&](int v){ return v >= 0 && v < height_; };
                        tl0 = (inx(xl0) && iny(yl0)) ? at(yl0, xl0)[0] : static_cast<float>(border_value);
                        tr0 = (inx(xu0) && iny(yl0)) ? at(yl0, xu0)[0] : static_cast<float>(border_value);
                        bl0 = (inx(xl0) && iny(yu0)) ? at(yu0, xl0)[0] : static_cast<float>(border_value);
                        br0 = (inx(xu0) && iny(yu0)) ? at(yu0, xu0)[0] : static_cast<float>(border_value);
                        tl1 = (inx(xl1) && iny(yl1)) ? at(yl1, xl1)[0] : static_cast<float>(border_value);
                        tr1 = (inx(xu1) && iny(yl1)) ? at(yl1, xu1)[0] : static_cast<float>(border_value);
                        bl1 = (inx(xl1) && iny(yu1)) ? at(yu1, xl1)[0] : static_cast<float>(border_value);
                        br1 = (inx(xu1) && iny(yu1)) ? at(yu1, xu1)[0] : static_cast<float>(border_value);
                        tl2 = (inx(xl2) && iny(yl2)) ? at(yl2, xl2)[0] : static_cast<float>(border_value);
                        tr2 = (inx(xu2) && iny(yl2)) ? at(yl2, xu2)[0] : static_cast<float>(border_value);
                        bl2 = (inx(xl2) && iny(yu2)) ? at(yu2, xl2)[0] : static_cast<float>(border_value);
                        br2 = (inx(xu2) && iny(yu2)) ? at(yu2, xu2)[0] : static_cast<float>(border_value);
                        tl3 = (inx(xl3) && iny(yl3)) ? at(yl3, xl3)[0] : static_cast<float>(border_value);
                        tr3 = (inx(xu3) && iny(yl3)) ? at(yl3, xu3)[0] : static_cast<float>(border_value);
                        bl3 = (inx(xl3) && iny(yu3)) ? at(yu3, xl3)[0] : static_cast<float>(border_value);
                        br3 = (inx(xu3) && iny(yu3)) ? at(yu3, xu3)[0] : static_cast<float>(border_value);
                    } else {
                        tl0 = at(yl0, xl0)[0]; tr0 = at(yl0, xu0)[0]; bl0 = at(yu0, xl0)[0]; br0 = at(yu0, xu0)[0];
                        tl1 = at(yl1, xl1)[0]; tr1 = at(yl1, xu1)[0]; bl1 = at(yu1, xl1)[0]; br1 = at(yu1, xu1)[0];
                        tl2 = at(yl2, xl2)[0]; tr2 = at(yl2, xu2)[0]; bl2 = at(yu2, xl2)[0]; br2 = at(yu2, xu2)[0];
                        tl3 = at(yl3, xl3)[0]; tr3 = at(yl3, xu3)[0]; bl3 = at(yu3, xl3)[0]; br3 = at(yu3, xu3)[0];
                    }
                    float32x4_t vfx = vdupq_n_f32(0.f);
                    vfx = vsetq_lane_f32(fx0, vfx, 0);
                    vfx = vsetq_lane_f32(fx1, vfx, 1);
                    vfx = vsetq_lane_f32(fx2, vfx, 2);
                    vfx = vsetq_lane_f32(fx3, vfx, 3);
                    float32x4_t vfy = vdupq_n_f32(0.f);
                    vfy = vsetq_lane_f32(fy0, vfy, 0);
                    vfy = vsetq_lane_f32(fy1, vfy, 1);
                    vfy = vsetq_lane_f32(fy2, vfy, 2);
                    vfy = vsetq_lane_f32(fy3, vfy, 3);
                    float32x4_t vtl = vdupq_n_f32(0.f);
                    vtl = vsetq_lane_f32(tl0, vtl, 0);
                    vtl = vsetq_lane_f32(tl1, vtl, 1);
                    vtl = vsetq_lane_f32(tl2, vtl, 2);
                    vtl = vsetq_lane_f32(tl3, vtl, 3);
                    float32x4_t vtr = vdupq_n_f32(0.f);
                    vtr = vsetq_lane_f32(tr0, vtr, 0);
                    vtr = vsetq_lane_f32(tr1, vtr, 1);
                    vtr = vsetq_lane_f32(tr2, vtr, 2);
                    vtr = vsetq_lane_f32(tr3, vtr, 3);
                    float32x4_t vbl = vdupq_n_f32(0.f);
                    vbl = vsetq_lane_f32(bl0, vbl, 0);
                    vbl = vsetq_lane_f32(bl1, vbl, 1);
                    vbl = vsetq_lane_f32(bl2, vbl, 2);
                    vbl = vsetq_lane_f32(bl3, vbl, 3);
                    float32x4_t vbr = vdupq_n_f32(0.f);
                    vbr = vsetq_lane_f32(br0, vbr, 0);
                    vbr = vsetq_lane_f32(br1, vbr, 1);
                    vbr = vsetq_lane_f32(br2, vbr, 2);
                    vbr = vsetq_lane_f32(br3, vbr, 3);
                    float32x4_t vtop = vaddq_f32(vtl, vmulq_f32(vsubq_f32(vtr, vtl), vfx));
                    float32x4_t vbot = vaddq_f32(vbl, vmulq_f32(vsubq_f32(vbr, vbl), vfx));
                    float32x4_t vout = vaddq_f32(vtop, vmulq_f32(vsubq_f32(vbot, vtop), vfy));
                    float outv[4];
                    vst1q_f32(outv, vout);
                    out_row[x + 0] = outv[0];
                    out_row[x + 1] = outv[1];
                    out_row[x + 2] = outv[2];
                    out_row[x + 3] = outv[3];
                }
#endif
                for (; x < width; ++x) {
                    float sx = yb + x * a;
                    float sy = yd + x * c;
                    int xl = static_cast<int>(std::floor(sx));
                    int yl = static_cast<int>(std::floor(sy));
                    float fx = sx - xl, fy = sy - yl;
                    int xu = xl + 1, yu = yl + 1;
                    auto clamp = [&](int& u, int lo, int hi) { if (u < lo) u = lo; if (u > hi) u = hi; };
                    if (border_mode == BORDER_MODE_REPLICATE) {
                        clamp(xl, 0, width_ - 1); clamp(xu, 0, width_ - 1);
                        clamp(yl, 0, height_ - 1); clamp(yu, 0, height_ - 1);
                    }
                    float tl, tr, bl, br;
                    if (border_mode == BORDER_MODE_CONSTANT) {
                        auto inx = [&](int u){ return u >= 0 && u < width_; };
                        auto iny = [&](int v){ return v >= 0 && v < height_; };
                        tl = (inx(xl) && iny(yl)) ? at(yl, xl)[0] : static_cast<float>(border_value);
                        tr = (inx(xu) && iny(yl)) ? at(yl, xu)[0] : static_cast<float>(border_value);
                        bl = (inx(xl) && iny(yu)) ? at(yu, xl)[0] : static_cast<float>(border_value);
                        br = (inx(xu) && iny(yu)) ? at(yu, xu)[0] : static_cast<float>(border_value);
                    } else {
                        tl = at(yl, xl)[0]; tr = at(yl, xu)[0]; bl = at(yu, xl)[0]; br = at(yu, xu)[0];
                    }
                    float top = tl + (tr - tl) * fx;
                    float bottom = bl + (br - bl) * fx;
                    out_row[x] = top + (bottom - top) * fy;
                }
                continue;
            }

            if (channels_ == 3 && std::is_same<D, float>::value) {
                float* out_row = reinterpret_cast<float*>(out_row_any);
                int x = 0;
#if defined(__SSE2__)
                for (; x + 4 <= width; x += 4) {
                    // lane-wise src coords
                    float sx0 = yb + (x + 0) * a, sy0 = yd + (x + 0) * c;
                    float sx1 = yb + (x + 1) * a, sy1 = yd + (x + 1) * c;
                    float sx2 = yb + (x + 2) * a, sy2 = yd + (x + 2) * c;
                    float sx3 = yb + (x + 3) * a, sy3 = yd + (x + 3) * c;
                    int xl0 = static_cast<int>(std::floor(sx0)), yl0 = static_cast<int>(std::floor(sy0));
                    int xl1 = static_cast<int>(std::floor(sx1)), yl1 = static_cast<int>(std::floor(sy1));
                    int xl2 = static_cast<int>(std::floor(sx2)), yl2 = static_cast<int>(std::floor(sy2));
                    int xl3 = static_cast<int>(std::floor(sx3)), yl3 = static_cast<int>(std::floor(sy3));
                    float fx0 = sx0 - xl0, fy0 = sy0 - yl0;
                    float fx1 = sx1 - xl1, fy1 = sy1 - yl1;
                    float fx2 = sx2 - xl2, fy2 = sy2 - yl2;
                    float fx3 = sx3 - xl3, fy3 = sy3 - yl3;
                    auto clamp = [&](int& u, int lo, int hi) { if (u < lo) u = lo; if (u > hi) u = hi; };
                    if (border_mode == BORDER_MODE_REPLICATE) {
                        clamp(xl0, 0, width_ - 1); clamp(xl1, 0, width_ - 1);
                        clamp(xl2, 0, width_ - 1); clamp(xl3, 0, width_ - 1);
                        clamp(yl0, 0, height_ - 1); clamp(yl1, 0, height_ - 1);
                        clamp(yl2, 0, height_ - 1); clamp(yl3, 0, height_ - 1);
                    }
                    int xu0 = xl0 + 1, yu0 = yl0 + 1;
                    int xu1 = xl1 + 1, yu1 = yl1 + 1;
                    int xu2 = xl2 + 1, yu2 = yl2 + 1;
                    int xu3 = xl3 + 1, yu3 = yl3 + 1;
                    if (border_mode == BORDER_MODE_REPLICATE) {
                        if (xu0 >= width_) xu0 = width_ - 1; if (yu0 >= height_) yu0 = height_ - 1;
                        if (xu1 >= width_) xu1 = width_ - 1; if (yu1 >= height_) yu1 = height_ - 1;
                        if (xu2 >= width_) xu2 = width_ - 1; if (yu2 >= height_) yu2 = height_ - 1;
                        if (xu3 >= width_) xu3 = width_ - 1; if (yu3 >= height_) yu3 = height_ - 1;
                    }
                    auto inx = [&](int u){ return u >= 0 && u < width_; };
                    auto iny = [&](int v){ return v >= 0 && v < height_; };
                    __m128 vfx = _mm_set_ps(fx3, fx2, fx1, fx0);
                    __m128 vfy = _mm_set_ps(fy3, fy2, fy1, fy0);
                    for (int cc = 0; cc < 3; ++cc) {
                        float tl0, tr0, bl0, br0, tl1, tr1, bl1, br1, tl2, tr2, bl2, br2, tl3, tr3, bl3, br3;
                        if (border_mode == BORDER_MODE_CONSTANT) {
                            tl0 = (inx(xl0) && iny(yl0)) ? at(yl0, xl0)[cc] : static_cast<float>(border_value);
                            tr0 = (inx(xu0) && iny(yl0)) ? at(yl0, xu0)[cc] : static_cast<float>(border_value);
                            bl0 = (inx(xl0) && iny(yu0)) ? at(yu0, xl0)[cc] : static_cast<float>(border_value);
                            br0 = (inx(xu0) && iny(yu0)) ? at(yu0, xu0)[cc] : static_cast<float>(border_value);
                            tl1 = (inx(xl1) && iny(yl1)) ? at(yl1, xl1)[cc] : static_cast<float>(border_value);
                            tr1 = (inx(xu1) && iny(yl1)) ? at(yl1, xu1)[cc] : static_cast<float>(border_value);
                            bl1 = (inx(xl1) && iny(yu1)) ? at(yu1, xl1)[cc] : static_cast<float>(border_value);
                            br1 = (inx(xu1) && iny(yu1)) ? at(yu1, xu1)[cc] : static_cast<float>(border_value);
                            tl2 = (inx(xl2) && iny(yl2)) ? at(yl2, xl2)[cc] : static_cast<float>(border_value);
                            tr2 = (inx(xu2) && iny(yl2)) ? at(yl2, xu2)[cc] : static_cast<float>(border_value);
                            bl2 = (inx(xl2) && iny(yu2)) ? at(yu2, xl2)[cc] : static_cast<float>(border_value);
                            br2 = (inx(xu2) && iny(yu2)) ? at(yu2, xu2)[cc] : static_cast<float>(border_value);
                            tl3 = (inx(xl3) && iny(yl3)) ? at(yl3, xl3)[cc] : static_cast<float>(border_value);
                            tr3 = (inx(xu3) && iny(yl3)) ? at(yl3, xu3)[cc] : static_cast<float>(border_value);
                            bl3 = (inx(xl3) && iny(yu3)) ? at(yu3, xl3)[cc] : static_cast<float>(border_value);
                            br3 = (inx(xu3) && iny(yu3)) ? at(yu3, xu3)[cc] : static_cast<float>(border_value);
                        } else {
                            tl0 = at(yl0, xl0)[cc]; tr0 = at(yl0, xu0)[cc]; bl0 = at(yu0, xl0)[cc]; br0 = at(yu0, xu0)[cc];
                            tl1 = at(yl1, xl1)[cc]; tr1 = at(yl1, xu1)[cc]; bl1 = at(yu1, xl1)[cc]; br1 = at(yu1, xu1)[cc];
                            tl2 = at(yl2, xl2)[cc]; tr2 = at(yl2, xu2)[cc]; bl2 = at(yu2, xl2)[cc]; br2 = at(yu2, xu2)[cc];
                            tl3 = at(yl3, xl3)[cc]; tr3 = at(yl3, xu3)[cc]; bl3 = at(yu3, xl3)[cc]; br3 = at(yu3, xu3)[cc];
                        }
                        __m128 vtl = _mm_set_ps(tl3, tl2, tl1, tl0);
                        __m128 vtr = _mm_set_ps(tr3, tr2, tr1, tr0);
                        __m128 vbl = _mm_set_ps(bl3, bl2, bl1, bl0);
                        __m128 vbr = _mm_set_ps(br3, br2, br1, br0);
                        __m128 vtop = _mm_add_ps(vtl, _mm_mul_ps(_mm_sub_ps(vtr, vtl), vfx));
                        __m128 vbot = _mm_add_ps(vbl, _mm_mul_ps(_mm_sub_ps(vbr, vbl), vfx));
                        __m128 vout = _mm_add_ps(vtop, _mm_mul_ps(_mm_sub_ps(vbot, vtop), vfy));
                        float outv[4];
                        _mm_storeu_ps(outv, vout);
                        out_row[(x + 0) * 3 + cc] = outv[0];
                        out_row[(x + 1) * 3 + cc] = outv[1];
                        out_row[(x + 2) * 3 + cc] = outv[2];
                        out_row[(x + 3) * 3 + cc] = outv[3];
                    }
                }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
                for (; x + 4 <= width; x += 4) {
                    float sx0 = yb + (x + 0) * a, sy0 = yd + (x + 0) * c;
                    float sx1 = yb + (x + 1) * a, sy1 = yd + (x + 1) * c;
                    float sx2 = yb + (x + 2) * a, sy2 = yd + (x + 2) * c;
                    float sx3 = yb + (x + 3) * a, sy3 = yd + (x + 3) * c;
                    int xl0 = static_cast<int>(std::floor(sx0)), yl0 = static_cast<int>(std::floor(sy0));
                    int xl1 = static_cast<int>(std::floor(sx1)), yl1 = static_cast<int>(std::floor(sy1));
                    int xl2 = static_cast<int>(std::floor(sx2)), yl2 = static_cast<int>(std::floor(sy2));
                    int xl3 = static_cast<int>(std::floor(sx3)), yl3 = static_cast<int>(std::floor(sy3));
                    float fx0 = sx0 - xl0, fy0 = sy0 - yl0;
                    float fx1 = sx1 - xl1, fy1 = sy1 - yl1;
                    float fx2 = sx2 - xl2, fy2 = sy2 - yl2;
                    float fx3 = sx3 - xl3, fy3 = sy3 - yl3;
                    auto clamp = [&](int& u, int lo, int hi) { if (u < lo) u = lo; if (u > hi) u = hi; };
                    if (border_mode == BORDER_MODE_REPLICATE) {
                        clamp(xl0, 0, width_ - 1); clamp(xl1, 0, width_ - 1);
                        clamp(xl2, 0, width_ - 1); clamp(xl3, 0, width_ - 1);
                        clamp(yl0, 0, height_ - 1); clamp(yl1, 0, height_ - 1);
                        clamp(yl2, 0, height_ - 1); clamp(yl3, 0, height_ - 1);
                    }
                    int xu0 = xl0 + 1, yu0 = yl0 + 1;
                    int xu1 = xl1 + 1, yu1 = yl1 + 1;
                    int xu2 = xl2 + 1, yu2 = yl2 + 1;
                    int xu3 = xl3 + 1, yu3 = yl3 + 1;
                    if (border_mode == BORDER_MODE_REPLICATE) {
                        if (xu0 >= width_) xu0 = width_ - 1; if (yu0 >= height_) yu0 = height_ - 1;
                        if (xu1 >= width_) xu1 = width_ - 1; if (yu1 >= height_) yu1 = height_ - 1;
                        if (xu2 >= width_) xu2 = width_ - 1; if (yu2 >= height_) yu2 = height_ - 1;
                        if (xu3 >= width_) xu3 = width_ - 1; if (yu3 >= height_) yu3 = height_ - 1;
                    }
                    auto inx = [&](int u){ return u >= 0 && u < width_; };
                    auto iny = [&](int v){ return v >= 0 && v < height_; };
                    float32x4_t vfx = vdupq_n_f32(0.f);
                    vfx = vsetq_lane_f32(fx0, vfx, 0);
                    vfx = vsetq_lane_f32(fx1, vfx, 1);
                    vfx = vsetq_lane_f32(fx2, vfx, 2);
                    vfx = vsetq_lane_f32(fx3, vfx, 3);
                    float32x4_t vfy = vdupq_n_f32(0.f);
                    vfy = vsetq_lane_f32(fy0, vfy, 0);
                    vfy = vsetq_lane_f32(fy1, vfy, 1);
                    vfy = vsetq_lane_f32(fy2, vfy, 2);
                    vfy = vsetq_lane_f32(fy3, vfy, 3);
                    for (int cc = 0; cc < 3; ++cc) {
                        float tl0, tr0, bl0, br0, tl1, tr1, bl1, br1, tl2, tr2, bl2, br2, tl3, tr3, bl3, br3;
                        if (border_mode == BORDER_MODE_CONSTANT) {
                            tl0 = (inx(xl0) && iny(yl0)) ? at(yl0, xl0)[cc] : static_cast<float>(border_value);
                            tr0 = (inx(xu0) && iny(yl0)) ? at(yl0, xu0)[cc] : static_cast<float>(border_value);
                            bl0 = (inx(xl0) && iny(yu0)) ? at(yu0, xl0)[cc] : static_cast<float>(border_value);
                            br0 = (inx(xu0) && iny(yu0)) ? at(yu0, xu0)[cc] : static_cast<float>(border_value);
                            tl1 = (inx(xl1) && iny(yl1)) ? at(yl1, xl1)[cc] : static_cast<float>(border_value);
                            tr1 = (inx(xu1) && iny(yl1)) ? at(yl1, xu1)[cc] : static_cast<float>(border_value);
                            bl1 = (inx(xl1) && iny(yu1)) ? at(yu1, xl1)[cc] : static_cast<float>(border_value);
                            br1 = (inx(xu1) && iny(yu1)) ? at(yu1, xu1)[cc] : static_cast<float>(border_value);
                            tl2 = (inx(xl2) && iny(yl2)) ? at(yl2, xl2)[cc] : static_cast<float>(border_value);
                            tr2 = (inx(xu2) && iny(yl2)) ? at(yl2, xu2)[cc] : static_cast<float>(border_value);
                            bl2 = (inx(xl2) && iny(yu2)) ? at(yu2, xl2)[cc] : static_cast<float>(border_value);
                            br2 = (inx(xu2) && iny(yu2)) ? at(yu2, xu2)[cc] : static_cast<float>(border_value);
                            tl3 = (inx(xl3) && iny(yl3)) ? at(yl3, xl3)[cc] : static_cast<float>(border_value);
                            tr3 = (inx(xu3) && iny(yl3)) ? at(yl3, xu3)[cc] : static_cast<float>(border_value);
                            bl3 = (inx(xl3) && iny(yu3)) ? at(yu3, xl3)[cc] : static_cast<float>(border_value);
                            br3 = (inx(xu3) && iny(yu3)) ? at(yu3, xu3)[cc] : static_cast<float>(border_value);
                        } else {
                            tl0 = at(yl0, xl0)[cc]; tr0 = at(yl0, xu0)[cc]; bl0 = at(yu0, xl0)[cc]; br0 = at(yu0, xu0)[cc];
                            tl1 = at(yl1, xl1)[cc]; tr1 = at(yl1, xu1)[cc]; bl1 = at(yu1, xl1)[cc]; br1 = at(yu1, xu1)[cc];
                            tl2 = at(yl2, xl2)[cc]; tr2 = at(yl2, xu2)[cc]; bl2 = at(yu2, xl2)[cc]; br2 = at(yu2, xu2)[cc];
                            tl3 = at(yl3, xl3)[cc]; tr3 = at(yl3, xu3)[cc]; bl3 = at(yu3, xl3)[cc]; br3 = at(yu3, xu3)[cc];
                        }
                        float32x4_t vtl = vdupq_n_f32(0.f);
                        vtl = vsetq_lane_f32(tl0, vtl, 0);
                        vtl = vsetq_lane_f32(tl1, vtl, 1);
                        vtl = vsetq_lane_f32(tl2, vtl, 2);
                        vtl = vsetq_lane_f32(tl3, vtl, 3);
                        float32x4_t vtr = vdupq_n_f32(0.f);
                        vtr = vsetq_lane_f32(tr0, vtr, 0);
                        vtr = vsetq_lane_f32(tr1, vtr, 1);
                        vtr = vsetq_lane_f32(tr2, vtr, 2);
                        vtr = vsetq_lane_f32(tr3, vtr, 3);
                        float32x4_t vbl = vdupq_n_f32(0.f);
                        vbl = vsetq_lane_f32(bl0, vbl, 0);
                        vbl = vsetq_lane_f32(bl1, vbl, 1);
                        vbl = vsetq_lane_f32(bl2, vbl, 2);
                        vbl = vsetq_lane_f32(bl3, vbl, 3);
                        float32x4_t vbr = vdupq_n_f32(0.f);
                        vbr = vsetq_lane_f32(br0, vbr, 0);
                        vbr = vsetq_lane_f32(br1, vbr, 1);
                        vbr = vsetq_lane_f32(br2, vbr, 2);
                        vbr = vsetq_lane_f32(br3, vbr, 3);
                        float32x4_t vtop = vaddq_f32(vtl, vmulq_f32(vsubq_f32(vtr, vtl), vfx));
                        float32x4_t vbot = vaddq_f32(vbl, vmulq_f32(vsubq_f32(vbr, vbl), vfx));
                        float32x4_t vout = vaddq_f32(vtop, vmulq_f32(vsubq_f32(vbot, vtop), vfy));
                        float outv[4];
                        vst1q_f32(outv, vout);
                        out_row[(x + 0) * 3 + cc] = outv[0];
                        out_row[(x + 1) * 3 + cc] = outv[1];
                        out_row[(x + 2) * 3 + cc] = outv[2];
                        out_row[(x + 3) * 3 + cc] = outv[3];
                    }
                }
#endif
                for (; x < width; ++x) {
                    float sx = yb + x * a;
                    float sy = yd + x * c;
                    int xl = static_cast<int>(std::floor(sx));
                    int yl = static_cast<int>(std::floor(sy));
                    float fx = sx - xl, fy = sy - yl;
                    int xu = xl + 1, yu = yl + 1;
                    auto clamp = [&](int& u, int lo, int hi) { if (u < lo) u = lo; if (u > hi) u = hi; };
                    if (border_mode == BORDER_MODE_REPLICATE) {
                        clamp(xl, 0, width_ - 1); clamp(xu, 0, width_ - 1);
                        clamp(yl, 0, height_ - 1); clamp(yu, 0, height_ - 1);
                    }
                    for (int cc = 0; cc < 3; ++cc) {
                        float tl, tr, bl, br;
                        if (border_mode == BORDER_MODE_CONSTANT) {
                            auto inx2 = [&](int u){ return u >= 0 && u < width_; };
                            auto iny2 = [&](int v){ return v >= 0 && v < height_; };
                            tl = (inx2(xl) && iny2(yl)) ? at(yl, xl)[cc] : static_cast<float>(border_value);
                            tr = (inx2(xu) && iny2(yl)) ? at(yl, xu)[cc] : static_cast<float>(border_value);
                            bl = (inx2(xl) && iny2(yu)) ? at(yu, xl)[cc] : static_cast<float>(border_value);
                            br = (inx2(xu) && iny2(yu)) ? at(yu, xu)[cc] : static_cast<float>(border_value);
                        } else {
                            tl = at(yl, xl)[cc]; tr = at(yl, xu)[cc]; bl = at(yu, xl)[cc]; br = at(yu, xu)[cc];
                        }
                        float top = tl + (tr - tl) * fx;
                        float bottom = bl + (br - bl) * fx;
                        out_row[x * 3 + cc] = top + (bottom - top) * fy;
                    }
                }
                continue;
            }

            if (channels_ == 1 && std::is_same<D, uint8_t>::value) {
                uint8_t* out_row = reinterpret_cast<uint8_t*>(out_row_any);
                int x = 0;
// SSE2 4-lane u8 (compute in float then pack)
#if defined(__SSE2__)
                for (; x + 4 <= width; x += 4) {
                    float sx0 = yb + (x + 0) * a;
                    float sy0 = yd + (x + 0) * c;
                    float sx1 = yb + (x + 1) * a;
                    float sy1 = yd + (x + 1) * c;
                    float sx2 = yb + (x + 2) * a;
                    float sy2 = yd + (x + 2) * c;
                    float sx3 = yb + (x + 3) * a;
                    float sy3 = yd + (x + 3) * c;
                    int xl0 = static_cast<int>(std::floor(sx0)), yl0 = static_cast<int>(std::floor(sy0));
                    int xl1 = static_cast<int>(std::floor(sx1)), yl1 = static_cast<int>(std::floor(sy1));
                    int xl2 = static_cast<int>(std::floor(sx2)), yl2 = static_cast<int>(std::floor(sy2));
                    int xl3 = static_cast<int>(std::floor(sx3)), yl3 = static_cast<int>(std::floor(sy3));
                    float fx0 = sx0 - xl0, fy0 = sy0 - yl0;
                    float fx1 = sx1 - xl1, fy1 = sy1 - yl1;
                    float fx2 = sx2 - xl2, fy2 = sy2 - yl2;
                    float fx3 = sx3 - xl3, fy3 = sy3 - yl3;
                    auto clamp = [&](int& u, int lo, int hi) { if (u < lo) u = lo; if (u > hi) u = hi; };
                    if (border_mode == BORDER_MODE_REPLICATE) {
                        clamp(xl0, 0, width_ - 1); clamp(xl1, 0, width_ - 1);
                        clamp(xl2, 0, width_ - 1); clamp(xl3, 0, width_ - 1);
                        clamp(yl0, 0, height_ - 1); clamp(yl1, 0, height_ - 1);
                        clamp(yl2, 0, height_ - 1); clamp(yl3, 0, height_ - 1);
                    }
                    int xu0 = xl0 + 1, yu0 = yl0 + 1;
                    int xu1 = xl1 + 1, yu1 = yl1 + 1;
                    int xu2 = xl2 + 1, yu2 = yl2 + 1;
                    int xu3 = xl3 + 1, yu3 = yl3 + 1;
                    if (border_mode == BORDER_MODE_REPLICATE) {
                        if (xu0 >= width_) xu0 = width_ - 1; if (yu0 >= height_) yu0 = height_ - 1;
                        if (xu1 >= width_) xu1 = width_ - 1; if (yu1 >= height_) yu1 = height_ - 1;
                        if (xu2 >= width_) xu2 = width_ - 1; if (yu2 >= height_) yu2 = height_ - 1;
                        if (xu3 >= width_) xu3 = width_ - 1; if (yu3 >= height_) yu3 = height_ - 1;
                    }
                    auto inx = [&](int u){ return u >= 0 && u < width_; };
                    auto iny = [&](int v){ return v >= 0 && v < height_; };
                    float tl0, tr0, bl0, br0, tl1, tr1, bl1, br1, tl2, tr2, bl2, br2, tl3, tr3, bl3, br3;
                    if (border_mode == BORDER_MODE_CONSTANT) {
                        tl0 = (inx(xl0) && iny(yl0)) ? at(yl0, xl0)[0] : static_cast<uint8_t>(border_value);
                        tr0 = (inx(xu0) && iny(yl0)) ? at(yl0, xu0)[0] : static_cast<uint8_t>(border_value);
                        bl0 = (inx(xl0) && iny(yu0)) ? at(yu0, xl0)[0] : static_cast<uint8_t>(border_value);
                        br0 = (inx(xu0) && iny(yu0)) ? at(yu0, xu0)[0] : static_cast<uint8_t>(border_value);
                        tl1 = (inx(xl1) && iny(yl1)) ? at(yl1, xl1)[0] : static_cast<uint8_t>(border_value);
                        tr1 = (inx(xu1) && iny(yl1)) ? at(yl1, xu1)[0] : static_cast<uint8_t>(border_value);
                        bl1 = (inx(xl1) && iny(yu1)) ? at(yu1, xl1)[0] : static_cast<uint8_t>(border_value);
                        br1 = (inx(xu1) && iny(yu1)) ? at(yu1, xu1)[0] : static_cast<uint8_t>(border_value);
                        tl2 = (inx(xl2) && iny(yl2)) ? at(yl2, xl2)[0] : static_cast<uint8_t>(border_value);
                        tr2 = (inx(xu2) && iny(yl2)) ? at(yl2, xu2)[0] : static_cast<uint8_t>(border_value);
                        bl2 = (inx(xl2) && iny(yu2)) ? at(yu2, xl2)[0] : static_cast<uint8_t>(border_value);
                        br2 = (inx(xu2) && iny(yu2)) ? at(yu2, xu2)[0] : static_cast<uint8_t>(border_value);
                        tl3 = (inx(xl3) && iny(yl3)) ? at(yl3, xl3)[0] : static_cast<uint8_t>(border_value);
                        tr3 = (inx(xu3) && iny(yl3)) ? at(yl3, xu3)[0] : static_cast<uint8_t>(border_value);
                        bl3 = (inx(xl3) && iny(yu3)) ? at(yu3, xl3)[0] : static_cast<uint8_t>(border_value);
                        br3 = (inx(xu3) && iny(yu3)) ? at(yu3, xu3)[0] : static_cast<uint8_t>(border_value);
                    } else {
                        tl0 = at(yl0, xl0)[0]; tr0 = at(yl0, xu0)[0]; bl0 = at(yu0, xl0)[0]; br0 = at(yu0, xu0)[0];
                        tl1 = at(yl1, xl1)[0]; tr1 = at(yl1, xu1)[0]; bl1 = at(yu1, xl1)[0]; br1 = at(yu1, xu1)[0];
                        tl2 = at(yl2, xl2)[0]; tr2 = at(yl2, xu2)[0]; bl2 = at(yu2, xl2)[0]; br2 = at(yu2, xu2)[0];
                        tl3 = at(yl3, xl3)[0]; tr3 = at(yl3, xu3)[0]; bl3 = at(yu3, xl3)[0]; br3 = at(yu3, xu3)[0];
                    }
                    __m128 vfx = _mm_set_ps(fx3, fx2, fx1, fx0);
                    __m128 vfy = _mm_set_ps(fy3, fy2, fy1, fy0);
                    __m128 vtl = _mm_set_ps(tl3, tl2, tl1, tl0);
                    __m128 vtr = _mm_set_ps(tr3, tr2, tr1, tr0);
                    __m128 vbl = _mm_set_ps(bl3, bl2, bl1, bl0);
                    __m128 vbr = _mm_set_ps(br3, br2, br1, br0);
                    __m128 vtop = _mm_add_ps(vtl, _mm_mul_ps(_mm_sub_ps(vtr, vtl), vfx));
                    __m128 vbot = _mm_add_ps(vbl, _mm_mul_ps(_mm_sub_ps(vbr, vbl), vfx));
                    __m128 vout = _mm_add_ps(vtop, _mm_mul_ps(_mm_sub_ps(vbot, vtop), vfy));
                    __m128i vi  = _mm_cvtps_epi32(vout);
                    __m128i vi16 = _mm_packs_epi32(vi, _mm_setzero_si128());
                    __m128i vi8  = _mm_packus_epi16(vi16, _mm_setzero_si128());
                    alignas(16) uint8_t tmpb[16];
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(tmpb), vi8);
                    out_row[x + 0] = tmpb[0];
                    out_row[x + 1] = tmpb[1];
                    out_row[x + 2] = tmpb[2];
                    out_row[x + 3] = tmpb[3];
                }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
                for (; x + 4 <= width; x += 4) {
                    float sx0 = yb + (x + 0) * a;
                    float sy0 = yd + (x + 0) * c;
                    float sx1 = yb + (x + 1) * a;
                    float sy1 = yd + (x + 1) * c;
                    float sx2 = yb + (x + 2) * a;
                    float sy2 = yd + (x + 2) * c;
                    float sx3 = yb + (x + 3) * a;
                    float sy3 = yd + (x + 3) * c;
                    int xl0 = static_cast<int>(std::floor(sx0)), yl0 = static_cast<int>(std::floor(sy0));
                    int xl1 = static_cast<int>(std::floor(sx1)), yl1 = static_cast<int>(std::floor(sy1));
                    int xl2 = static_cast<int>(std::floor(sx2)), yl2 = static_cast<int>(std::floor(sy2));
                    int xl3 = static_cast<int>(std::floor(sx3)), yl3 = static_cast<int>(std::floor(sy3));
                    float fx0 = sx0 - xl0, fy0 = sy0 - yl0;
                    float fx1 = sx1 - xl1, fy1 = sy1 - yl1;
                    float fx2 = sx2 - xl2, fy2 = sy2 - yl2;
                    float fx3 = sx3 - xl3, fy3 = sy3 - yl3;
                    auto clamp = [&](int& u, int lo, int hi) { if (u < lo) u = lo; if (u > hi) u = hi; };
                    if (border_mode == BORDER_MODE_REPLICATE) {
                        clamp(xl0, 0, width_ - 1); clamp(xl1, 0, width_ - 1);
                        clamp(xl2, 0, width_ - 1); clamp(xl3, 0, width_ - 1);
                        clamp(yl0, 0, height_ - 1); clamp(yl1, 0, height_ - 1);
                        clamp(yl2, 0, height_ - 1); clamp(yl3, 0, height_ - 1);
                    }
                    int xu0 = xl0 + 1, yu0 = yl0 + 1;
                    int xu1 = xl1 + 1, yu1 = yl1 + 1;
                    int xu2 = xl2 + 1, yu2 = yl2 + 1;
                    int xu3 = xl3 + 1, yu3 = yl3 + 1;
                    if (border_mode == BORDER_MODE_REPLICATE) {
                        if (xu0 >= width_) xu0 = width_ - 1; if (yu0 >= height_) yu0 = height_ - 1;
                        if (xu1 >= width_) xu1 = width_ - 1; if (yu1 >= height_) yu1 = height_ - 1;
                        if (xu2 >= width_) xu2 = width_ - 1; if (yu2 >= height_) yu2 = height_ - 1;
                        if (xu3 >= width_) xu3 = width_ - 1; if (yu3 >= height_) yu3 = height_ - 1;
                    }
                    auto inx = [&](int u){ return u >= 0 && u < width_; };
                    auto iny = [&](int v){ return v >= 0 && v < height_; };
                    float tl0, tr0, bl0, br0, tl1, tr1, bl1, br1, tl2, tr2, bl2, br2, tl3, tr3, bl3, br3;
                    if (border_mode == BORDER_MODE_CONSTANT) {
                        tl0 = (inx(xl0) && iny(yl0)) ? at(yl0, xl0)[0] : static_cast<uint8_t>(border_value);
                        tr0 = (inx(xu0) && iny(yl0)) ? at(yl0, xu0)[0] : static_cast<uint8_t>(border_value);
                        bl0 = (inx(xl0) && iny(yu0)) ? at(yu0, xl0)[0] : static_cast<uint8_t>(border_value);
                        br0 = (inx(xu0) && iny(yu0)) ? at(yu0, xu0)[0] : static_cast<uint8_t>(border_value);
                        tl1 = (inx(xl1) && iny(yl1)) ? at(yl1, xl1)[0] : static_cast<uint8_t>(border_value);
                        tr1 = (inx(xu1) && iny(yl1)) ? at(yl1, xu1)[0] : static_cast<uint8_t>(border_value);
                        bl1 = (inx(xl1) && iny(yu1)) ? at(yu1, xl1)[0] : static_cast<uint8_t>(border_value);
                        br1 = (inx(xu1) && iny(yu1)) ? at(yu1, xu1)[0] : static_cast<uint8_t>(border_value);
                        tl2 = (inx(xl2) && iny(yl2)) ? at(yl2, xl2)[0] : static_cast<uint8_t>(border_value);
                        tr2 = (inx(xu2) && iny(yl2)) ? at(yl2, xu2)[0] : static_cast<uint8_t>(border_value);
                        bl2 = (inx(xl2) && iny(yu2)) ? at(yu2, xl2)[0] : static_cast<uint8_t>(border_value);
                        br2 = (inx(xu2) && iny(yu2)) ? at(yu2, xu2)[0] : static_cast<uint8_t>(border_value);
                        tl3 = (inx(xl3) && iny(yl3)) ? at(yl3, xl3)[0] : static_cast<uint8_t>(border_value);
                        tr3 = (inx(xu3) && iny(yl3)) ? at(yl3, xu3)[0] : static_cast<uint8_t>(border_value);
                        bl3 = (inx(xl3) && iny(yu3)) ? at(yu3, xl3)[0] : static_cast<uint8_t>(border_value);
                        br3 = (inx(xu3) && iny(yu3)) ? at(yu3, xu3)[0] : static_cast<uint8_t>(border_value);
                    } else {
                        tl0 = at(yl0, xl0)[0]; tr0 = at(yl0, xu0)[0]; bl0 = at(yu0, xl0)[0]; br0 = at(yu0, xu0)[0];
                        tl1 = at(yl1, xl1)[0]; tr1 = at(yl1, xu1)[0]; bl1 = at(yu1, xl1)[0]; br1 = at(yu1, xu1)[0];
                        tl2 = at(yl2, xl2)[0]; tr2 = at(yl2, xu2)[0]; bl2 = at(yu2, xl2)[0]; br2 = at(yu2, xu2)[0];
                        tl3 = at(yl3, xl3)[0]; tr3 = at(yl3, xu3)[0]; bl3 = at(yu3, xl3)[0]; br3 = at(yu3, xu3)[0];
                    }
                    float32x4_t vfx = vdupq_n_f32(0.f);
                    vfx = vsetq_lane_f32(fx0, vfx, 0);
                    vfx = vsetq_lane_f32(fx1, vfx, 1);
                    vfx = vsetq_lane_f32(fx2, vfx, 2);
                    vfx = vsetq_lane_f32(fx3, vfx, 3);
                    float32x4_t vfy = vdupq_n_f32(0.f);
                    vfy = vsetq_lane_f32(fy0, vfy, 0);
                    vfy = vsetq_lane_f32(fy1, vfy, 1);
                    vfy = vsetq_lane_f32(fy2, vfy, 2);
                    vfy = vsetq_lane_f32(fy3, vfy, 3);
                    float32x4_t vtl = vdupq_n_f32(0.f);
                    vtl = vsetq_lane_f32(tl0, vtl, 0);
                    vtl = vsetq_lane_f32(tl1, vtl, 1);
                    vtl = vsetq_lane_f32(tl2, vtl, 2);
                    vtl = vsetq_lane_f32(tl3, vtl, 3);
                    float32x4_t vtr = vdupq_n_f32(0.f);
                    vtr = vsetq_lane_f32(tr0, vtr, 0);
                    vtr = vsetq_lane_f32(tr1, vtr, 1);
                    vtr = vsetq_lane_f32(tr2, vtr, 2);
                    vtr = vsetq_lane_f32(tr3, vtr, 3);
                    float32x4_t vbl = vdupq_n_f32(0.f);
                    vbl = vsetq_lane_f32(bl0, vbl, 0);
                    vbl = vsetq_lane_f32(bl1, vbl, 1);
                    vbl = vsetq_lane_f32(bl2, vbl, 2);
                    vbl = vsetq_lane_f32(bl3, vbl, 3);
                    float32x4_t vbr = vdupq_n_f32(0.f);
                    vbr = vsetq_lane_f32(br0, vbr, 0);
                    vbr = vsetq_lane_f32(br1, vbr, 1);
                    vbr = vsetq_lane_f32(br2, vbr, 2);
                    vbr = vsetq_lane_f32(br3, vbr, 3);
                    float32x4_t vtop = vaddq_f32(vtl, vmulq_f32(vsubq_f32(vtr, vtl), vfx));
                    float32x4_t vbot = vaddq_f32(vbl, vmulq_f32(vsubq_f32(vbr, vbl), vfx));
                    float32x4_t vout = vaddq_f32(vtop, vmulq_f32(vsubq_f32(vbot, vtop), vfy));
                    float outv[4];
                    vst1q_f32(outv, vout);
                    for (int k = 0; k < 4; ++k) {
                        float v = outv[k];
                        if (v < 0.f) v = 0.f; if (v > 255.f) v = 255.f;
                        out_row[x + k] = static_cast<uint8_t>(std::round(v));
                    }
                }
#endif
                for (; x < width; ++x) {
                    float sx = yb + x * a;
                    float sy = yd + x * c;
                    int xl = static_cast<int>(std::floor(sx));
                    int yl = static_cast<int>(std::floor(sy));
                    float fx = sx - xl, fy = sy - yl;
                    int xu = xl + 1, yu = yl + 1;
                    auto clamp = [&](int& u, int lo, int hi) { if (u < lo) u = lo; if (u > hi) u = hi; };
                    if (border_mode == BORDER_MODE_REPLICATE) {
                        clamp(xl, 0, width_ - 1); clamp(xu, 0, width_ - 1);
                        clamp(yl, 0, height_ - 1); clamp(yu, 0, height_ - 1);
                    }
                    float tl, tr, bl, br;
                    if (border_mode == BORDER_MODE_CONSTANT) {
                        auto inx = [&](int u){ return u >= 0 && u < width_; };
                        auto iny = [&](int v){ return v >= 0 && v < height_; };
                        tl = (inx(xl) && iny(yl)) ? at(yl, xl)[0] : static_cast<float>(border_value);
                        tr = (inx(xu) && iny(yl)) ? at(yl, xu)[0] : static_cast<float>(border_value);
                        bl = (inx(xl) && iny(yu)) ? at(yu, xl)[0] : static_cast<float>(border_value);
                        br = (inx(xu) && iny(yu)) ? at(yu, xu)[0] : static_cast<float>(border_value);
                    } else {
                        tl = at(yl, xl)[0]; tr = at(yl, xu)[0]; bl = at(yu, xl)[0]; br = at(yu, xu)[0];
                    }
                    float top = tl + (tr - tl) * fx;
                    float bottom = bl + (br - bl) * fx;
                    float v = top + (bottom - top) * fy;
                    if (v < 0.f) v = 0.f; if (v > 255.f) v = 255.f;
                    out_row[x] = static_cast<uint8_t>(std::round(v));
                }
                continue;
            }

            if (channels_ == 3 && std::is_same<D, uint8_t>::value) {
                uint8_t* out_row = reinterpret_cast<uint8_t*>(out_row_any);
                int x = 0;
#if defined(__SSE2__)
                for (; x + 4 <= width; x += 4) {
                    float sx0 = yb + (x + 0) * a, sy0 = yd + (x + 0) * c;
                    float sx1 = yb + (x + 1) * a, sy1 = yd + (x + 1) * c;
                    float sx2 = yb + (x + 2) * a, sy2 = yd + (x + 2) * c;
                    float sx3 = yb + (x + 3) * a, sy3 = yd + (x + 3) * c;
                    int xl0 = static_cast<int>(std::floor(sx0)), yl0 = static_cast<int>(std::floor(sy0));
                    int xl1 = static_cast<int>(std::floor(sx1)), yl1 = static_cast<int>(std::floor(sy1));
                    int xl2 = static_cast<int>(std::floor(sx2)), yl2 = static_cast<int>(std::floor(sy2));
                    int xl3 = static_cast<int>(std::floor(sx3)), yl3 = static_cast<int>(std::floor(sy3));
                    float fx0 = sx0 - xl0, fy0 = sy0 - yl0;
                    float fx1 = sx1 - xl1, fy1 = sy1 - yl1;
                    float fx2 = sx2 - xl2, fy2 = sy2 - yl2;
                    float fx3 = sx3 - xl3, fy3 = sy3 - yl3;
                    auto clamp = [&](int& u, int lo, int hi) { if (u < lo) u = lo; if (u > hi) u = hi; };
                    if (border_mode == BORDER_MODE_REPLICATE) {
                        clamp(xl0, 0, width_ - 1); clamp(xl1, 0, width_ - 1);
                        clamp(xl2, 0, width_ - 1); clamp(xl3, 0, width_ - 1);
                        clamp(yl0, 0, height_ - 1); clamp(yl1, 0, height_ - 1);
                        clamp(yl2, 0, height_ - 1); clamp(yl3, 0, height_ - 1);
                    }
                    int xu0 = xl0 + 1, yu0 = yl0 + 1;
                    int xu1 = xl1 + 1, yu1 = yl1 + 1;
                    int xu2 = xl2 + 1, yu2 = yl2 + 1;
                    int xu3 = xl3 + 1, yu3 = yl3 + 1;
                    if (border_mode == BORDER_MODE_REPLICATE) {
                        if (xu0 >= width_) xu0 = width_ - 1; if (yu0 >= height_) yu0 = height_ - 1;
                        if (xu1 >= width_) xu1 = width_ - 1; if (yu1 >= height_) yu1 = height_ - 1;
                        if (xu2 >= width_) xu2 = width_ - 1; if (yu2 >= height_) yu2 = height_ - 1;
                        if (xu3 >= width_) xu3 = width_ - 1; if (yu3 >= height_) yu3 = height_ - 1;
                    }
                    auto inx = [&](int u){ return u >= 0 && u < width_; };
                    auto iny = [&](int v){ return v >= 0 && v < height_; };
                    __m128 vfx = _mm_set_ps(fx3, fx2, fx1, fx0);
                    __m128 vfy = _mm_set_ps(fy3, fy2, fy1, fy0);
                    for (int cc = 0; cc < 3; ++cc) {
                        float tl0, tr0, bl0, br0, tl1, tr1, bl1, br1, tl2, tr2, bl2, br2, tl3, tr3, bl3, br3;
                        if (border_mode == BORDER_MODE_CONSTANT) {
                            tl0 = (inx(xl0) && iny(yl0)) ? at(yl0, xl0)[cc] : static_cast<uint8_t>(border_value);
                            tr0 = (inx(xu0) && iny(yl0)) ? at(yl0, xu0)[cc] : static_cast<uint8_t>(border_value);
                            bl0 = (inx(xl0) && iny(yu0)) ? at(yu0, xl0)[cc] : static_cast<uint8_t>(border_value);
                            br0 = (inx(xu0) && iny(yu0)) ? at(yu0, xu0)[cc] : static_cast<uint8_t>(border_value);
                            tl1 = (inx(xl1) && iny(yl1)) ? at(yl1, xl1)[cc] : static_cast<uint8_t>(border_value);
                            tr1 = (inx(xu1) && iny(yl1)) ? at(yl1, xu1)[cc] : static_cast<uint8_t>(border_value);
                            bl1 = (inx(xl1) && iny(yu1)) ? at(yu1, xl1)[cc] : static_cast<uint8_t>(border_value);
                            br1 = (inx(xu1) && iny(yu1)) ? at(yu1, xu1)[cc] : static_cast<uint8_t>(border_value);
                            tl2 = (inx(xl2) && iny(yl2)) ? at(yl2, xl2)[cc] : static_cast<uint8_t>(border_value);
                            tr2 = (inx(xu2) && iny(yl2)) ? at(yl2, xu2)[cc] : static_cast<uint8_t>(border_value);
                            bl2 = (inx(xl2) && iny(yu2)) ? at(yu2, xl2)[cc] : static_cast<uint8_t>(border_value);
                            br2 = (inx(xu2) && iny(yu2)) ? at(yu2, xu2)[cc] : static_cast<uint8_t>(border_value);
                            tl3 = (inx(xl3) && iny(yl3)) ? at(yl3, xl3)[cc] : static_cast<uint8_t>(border_value);
                            tr3 = (inx(xu3) && iny(yl3)) ? at(yl3, xu3)[cc] : static_cast<uint8_t>(border_value);
                            bl3 = (inx(xl3) && iny(yu3)) ? at(yu3, xl3)[cc] : static_cast<uint8_t>(border_value);
                            br3 = (inx(xu3) && iny(yu3)) ? at(yu3, xu3)[cc] : static_cast<uint8_t>(border_value);
                        } else {
                            tl0 = at(yl0, xl0)[cc]; tr0 = at(yl0, xu0)[cc]; bl0 = at(yu0, xl0)[cc]; br0 = at(yu0, xu0)[cc];
                            tl1 = at(yl1, xl1)[cc]; tr1 = at(yl1, xu1)[cc]; bl1 = at(yu1, xl1)[cc]; br1 = at(yu1, xu1)[cc];
                            tl2 = at(yl2, xl2)[cc]; tr2 = at(yl2, xu2)[cc]; bl2 = at(yu2, xl2)[cc]; br2 = at(yu2, xu2)[cc];
                            tl3 = at(yl3, xl3)[cc]; tr3 = at(yl3, xu3)[cc]; bl3 = at(yu3, xl3)[cc]; br3 = at(yu3, xu3)[cc];
                        }
                        __m128 vtl = _mm_set_ps(tl3, tl2, tl1, tl0);
                        __m128 vtr = _mm_set_ps(tr3, tr2, tr1, tr0);
                        __m128 vbl = _mm_set_ps(bl3, bl2, bl1, bl0);
                        __m128 vbr = _mm_set_ps(br3, br2, br1, br0);
                        __m128 vtop = _mm_add_ps(vtl, _mm_mul_ps(_mm_sub_ps(vtr, vtl), vfx));
                        __m128 vbot = _mm_add_ps(vbl, _mm_mul_ps(_mm_sub_ps(vbr, vbl), vfx));
                        __m128 vout = _mm_add_ps(vtop, _mm_mul_ps(_mm_sub_ps(vbot, vtop), vfy));
                        alignas(16) float outv[4];
                        _mm_storeu_ps(outv, vout);
                        for (int k = 0; k < 4; ++k) {
                            float v = outv[k];
                            if (v < 0.f) v = 0.f; if (v > 255.f) v = 255.f;
                            out_row[(x + k) * 3 + cc] = static_cast<uint8_t>(std::round(v));
                        }
                    }
                }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
                for (; x + 4 <= width; x += 4) {
                    float sx0 = yb + (x + 0) * a, sy0 = yd + (x + 0) * c;
                    float sx1 = yb + (x + 1) * a, sy1 = yd + (x + 1) * c;
                    float sx2 = yb + (x + 2) * a, sy2 = yd + (x + 2) * c;
                    float sx3 = yb + (x + 3) * a, sy3 = yd + (x + 3) * c;
                    int xl0 = static_cast<int>(std::floor(sx0)), yl0 = static_cast<int>(std::floor(sy0));
                    int xl1 = static_cast<int>(std::floor(sx1)), yl1 = static_cast<int>(std::floor(sy1));
                    int xl2 = static_cast<int>(std::floor(sx2)), yl2 = static_cast<int>(std::floor(sy2));
                    int xl3 = static_cast<int>(std::floor(sx3)), yl3 = static_cast<int>(std::floor(sy3));
                    float fx0 = sx0 - xl0, fy0 = sy0 - yl0;
                    float fx1 = sx1 - xl1, fy1 = sy1 - yl1;
                    float fx2 = sx2 - xl2, fy2 = sy2 - yl2;
                    float fx3 = sx3 - xl3, fy3 = sy3 - yl3;
                    auto clamp = [&](int& u, int lo, int hi) { if (u < lo) u = lo; if (u > hi) u = hi; };
                    if (border_mode == BORDER_MODE_REPLICATE) {
                        clamp(xl0, 0, width_ - 1); clamp(xl1, 0, width_ - 1);
                        clamp(xl2, 0, width_ - 1); clamp(xl3, 0, width_ - 1);
                        clamp(yl0, 0, height_ - 1); clamp(yl1, 0, height_ - 1);
                        clamp(yl2, 0, height_ - 1); clamp(yl3, 0, height_ - 1);
                    }
                    int xu0 = xl0 + 1, yu0 = yl0 + 1;
                    int xu1 = xl1 + 1, yu1 = yl1 + 1;
                    int xu2 = xl2 + 1, yu2 = yl2 + 1;
                    int xu3 = xl3 + 1, yu3 = yl3 + 1;
                    if (border_mode == BORDER_MODE_REPLICATE) {
                        if (xu0 >= width_) xu0 = width_ - 1; if (yu0 >= height_) yu0 = height_ - 1;
                        if (xu1 >= width_) xu1 = width_ - 1; if (yu1 >= height_) yu1 = height_ - 1;
                        if (xu2 >= width_) xu2 = width_ - 1; if (yu2 >= height_) yu2 = height_ - 1;
                        if (xu3 >= width_) xu3 = width_ - 1; if (yu3 >= height_) yu3 = height_ - 1;
                    }
                    auto inx = [&](int u){ return u >= 0 && u < width_; };
                    auto iny = [&](int v){ return v >= 0 && v < height_; };
                    float32x4_t vfx = vdupq_n_f32(0.f);
                    vfx = vsetq_lane_f32(fx0, vfx, 0);
                    vfx = vsetq_lane_f32(fx1, vfx, 1);
                    vfx = vsetq_lane_f32(fx2, vfx, 2);
                    vfx = vsetq_lane_f32(fx3, vfx, 3);
                    float32x4_t vfy = vdupq_n_f32(0.f);
                    vfy = vsetq_lane_f32(fy0, vfy, 0);
                    vfy = vsetq_lane_f32(fy1, vfy, 1);
                    vfy = vsetq_lane_f32(fy2, vfy, 2);
                    vfy = vsetq_lane_f32(fy3, vfy, 3);
                    for (int cc = 0; cc < 3; ++cc) {
                        float tl0, tr0, bl0, br0, tl1, tr1, bl1, br1, tl2, tr2, bl2, br2, tl3, tr3, bl3, br3;
                        if (border_mode == BORDER_MODE_CONSTANT) {
                            tl0 = (inx(xl0) && iny(yl0)) ? at(yl0, xl0)[cc] : static_cast<uint8_t>(border_value);
                            tr0 = (inx(xu0) && iny(yl0)) ? at(yl0, xu0)[cc] : static_cast<uint8_t>(border_value);
                            bl0 = (inx(xl0) && iny(yu0)) ? at(yu0, xl0)[cc] : static_cast<uint8_t>(border_value);
                            br0 = (inx(xu0) && iny(yu0)) ? at(yu0, xu0)[cc] : static_cast<uint8_t>(border_value);
                            tl1 = (inx(xl1) && iny(yl1)) ? at(yl1, xl1)[cc] : static_cast<uint8_t>(border_value);
                            tr1 = (inx(xu1) && iny(yl1)) ? at(yl1, xu1)[cc] : static_cast<uint8_t>(border_value);
                            bl1 = (inx(xl1) && iny(yu1)) ? at(yu1, xl1)[cc] : static_cast<uint8_t>(border_value);
                            br1 = (inx(xu1) && iny(yu1)) ? at(yu1, xu1)[cc] : static_cast<uint8_t>(border_value);
                            tl2 = (inx(xl2) && iny(yl2)) ? at(yl2, xl2)[cc] : static_cast<uint8_t>(border_value);
                            tr2 = (inx(xu2) && iny(yl2)) ? at(yl2, xu2)[cc] : static_cast<uint8_t>(border_value);
                            bl2 = (inx(xl2) && iny(yu2)) ? at(yu2, xl2)[cc] : static_cast<uint8_t>(border_value);
                            br2 = (inx(xu2) && iny(yu2)) ? at(yu2, xu2)[cc] : static_cast<uint8_t>(border_value);
                            tl3 = (inx(xl3) && iny(yl3)) ? at(yl3, xl3)[cc] : static_cast<uint8_t>(border_value);
                            tr3 = (inx(xu3) && iny(yl3)) ? at(yl3, xu3)[cc] : static_cast<uint8_t>(border_value);
                            bl3 = (inx(xl3) && iny(yu3)) ? at(yu3, xl3)[cc] : static_cast<uint8_t>(border_value);
                            br3 = (inx(xu3) && iny(yu3)) ? at(yu3, xu3)[cc] : static_cast<uint8_t>(border_value);
                        } else {
                            tl0 = at(yl0, xl0)[cc]; tr0 = at(yl0, xu0)[cc]; bl0 = at(yu0, xl0)[cc]; br0 = at(yu0, xu0)[cc];
                            tl1 = at(yl1, xl1)[cc]; tr1 = at(yl1, xu1)[cc]; bl1 = at(yu1, xl1)[cc]; br1 = at(yu1, xu1)[cc];
                            tl2 = at(yl2, xl2)[cc]; tr2 = at(yl2, xu2)[cc]; bl2 = at(yu2, xl2)[cc]; br2 = at(yu2, xu2)[cc];
                            tl3 = at(yl3, xl3)[cc]; tr3 = at(yl3, xu3)[cc]; bl3 = at(yu3, xl3)[cc]; br3 = at(yu3, xu3)[cc];
                        }
                        float32x4_t vtl = vdupq_n_f32(0.f);
                        vtl = vsetq_lane_f32(tl0, vtl, 0);
                        vtl = vsetq_lane_f32(tl1, vtl, 1);
                        vtl = vsetq_lane_f32(tl2, vtl, 2);
                        vtl = vsetq_lane_f32(tl3, vtl, 3);
                        float32x4_t vtr = vdupq_n_f32(0.f);
                        vtr = vsetq_lane_f32(tr0, vtr, 0);
                        vtr = vsetq_lane_f32(tr1, vtr, 1);
                        vtr = vsetq_lane_f32(tr2, vtr, 2);
                        vtr = vsetq_lane_f32(tr3, vtr, 3);
                        float32x4_t vbl = vdupq_n_f32(0.f);
                        vbl = vsetq_lane_f32(bl0, vbl, 0);
                        vbl = vsetq_lane_f32(bl1, vbl, 1);
                        vbl = vsetq_lane_f32(bl2, vbl, 2);
                        vbl = vsetq_lane_f32(bl3, vbl, 3);
                        float32x4_t vbr = vdupq_n_f32(0.f);
                        vbr = vsetq_lane_f32(br0, vbr, 0);
                        vbr = vsetq_lane_f32(br1, vbr, 1);
                        vbr = vsetq_lane_f32(br2, vbr, 2);
                        vbr = vsetq_lane_f32(br3, vbr, 3);
                        float32x4_t vtop = vaddq_f32(vtl, vmulq_f32(vsubq_f32(vtr, vtl), vfx));
                        float32x4_t vbot = vaddq_f32(vbl, vmulq_f32(vsubq_f32(vbr, vbl), vfx));
                        float32x4_t vout = vaddq_f32(vtop, vmulq_f32(vsubq_f32(vbot, vtop), vfy));
                        float outv[4];
                        vst1q_f32(outv, vout);
                        for (int k = 0; k < 4; ++k) {
                            float v = outv[k];
                            if (v < 0.f) v = 0.f; if (v > 255.f) v = 255.f;
                            out_row[(x + k) * 3 + cc] = static_cast<uint8_t>(std::round(v));
                        }
                    }
                }
#endif
                for (; x < width; ++x) {
                    float sx = yb + x * a;
                    float sy = yd + x * c;
                    int xl = static_cast<int>(std::floor(sx));
                    int yl = static_cast<int>(std::floor(sy));
                    float fx = sx - xl, fy = sy - yl;
                    int xu = xl + 1, yu = yl + 1;
                    auto clamp = [&](int& u, int lo, int hi) { if (u < lo) u = lo; if (u > hi) u = hi; };
                    if (border_mode == BORDER_MODE_REPLICATE) {
                        clamp(xl, 0, width_ - 1); clamp(xu, 0, width_ - 1);
                        clamp(yl, 0, height_ - 1); clamp(yu, 0, height_ - 1);
                    }
                    for (int cc = 0; cc < 3; ++cc) {
                        float tl, tr, bl, br;
                        if (border_mode == BORDER_MODE_CONSTANT) {
                            auto inx2 = [&](int u){ return u >= 0 && u < width_; };
                            auto iny2 = [&](int v){ return v >= 0 && v < height_; };
                            tl = (inx2(xl) && iny2(yl)) ? at(yl, xl)[cc] : static_cast<uint8_t>(border_value);
                            tr = (inx2(xu) && iny2(yl)) ? at(yl, xu)[cc] : static_cast<uint8_t>(border_value);
                            bl = (inx2(xl) && iny2(yu)) ? at(yu, xl)[cc] : static_cast<uint8_t>(border_value);
                            br = (inx2(xu) && iny2(yu)) ? at(yu, xu)[cc] : static_cast<uint8_t>(border_value);
                        } else {
                            tl = at(yl, xl)[cc]; tr = at(yl, xu)[cc]; bl = at(yu, xl)[cc]; br = at(yu, xu)[cc];
                        }
                        float top = tl + (tr - tl) * fx;
                        float bottom = bl + (br - bl) * fx;
                        float v = top + (bottom - top) * fy;
                        if (v < 0.f) v = 0.f; if (v > 255.f) v = 255.f;
                        out_row[x * 3 + cc] = static_cast<uint8_t>(std::round(v));
                    }
                }
                continue;
            }

            // Other channels/types: scalar fallback using incremental coords
            auto dst_iter = dst.Row(y);
            for (int x = 0; x < width; ++x) {
                float sx = yb + x * a;
                float sy = yd + x * c;
                int xl = static_cast<int>(std::floor(sx));
                int yl = static_cast<int>(std::floor(sy));
                float fx = sx - xl, fy = sy - yl;
                int xu = xl + 1, yu = yl + 1;
                if (border_mode == BORDER_MODE_REPLICATE) {
                    auto clamp = [&](int& u, int lo, int hi) { if (u < lo) u = lo; if (u > hi) u = hi; };
                    clamp(xl, 0, width_ - 1); clamp(xu, 0, width_ - 1);
                    clamp(yl, 0, height_ - 1); clamp(yu, 0, height_ - 1);
                    for (int cch = 0; cch < channels_; ++cch) {
                        *(dst_iter++) = InterpolateBilinear(
                          at(yl, xl)[cch], at(yl, xu)[cch],
                          at(yu, xl)[cch], at(yu, xu)[cch], fx, fy);
                    }
                } else {
                    auto inx = [&](int u){ return u >= 0 && u < width_; };
                    auto iny = [&](int v){ return v >= 0 && v < height_; };
                    for (int cch = 0; cch < channels_; ++cch) {
                        D tl = (inx(xl) && iny(yl)) ? at(yl, xl)[cch] : border_value;
                        D tr = (inx(xu) && iny(yl)) ? at(yl, xu)[cch] : border_value;
                        D bl = (inx(xl) && iny(yu)) ? at(yu, xl)[cch] : border_value;
                        D br = (inx(xu) && iny(yu)) ? at(yu, xu)[cch] : border_value;
                        *(dst_iter++) = InterpolateBilinear(tl, tr, bl, br, fx, fy);
                    }
                }
            }
        }
        return dst;
    }
#endif
    // Fallback to the proven reference path for general affine
    return AffineBilinearReference(width, height, matrix, border_mode, border_value);
}
template <typename D>
Bitmap<D> Bitmap<D>::Pad(int top, int down, int left, int right, D value) const {
    Bitmap<D> dst;
    dst.width_ = width_ + left + right;
    dst.height_ = height_ + top + down;
    dst.channels_ = channels_;
    dst.data_.reset(new D[static_cast<size_t>(dst.width_) * dst.height_ * channels_]);
    const int destination_width = dst.width_;
    const int channel_count = channels_;
    for (int y = 0; y < top; ++y) {
        std::fill_n(dst.Row(y), destination_width * channel_count, value);
    }
    for (int y = dst.height_ - down; y < dst.height_; ++y) {
        std::fill_n(dst.Row(y), destination_width * channel_count, value);
    }
    for (int y = 0; y < height_; ++y) {
        D* destination = dst.Row(y + top);
        std::fill_n(destination, left * channel_count, value);
        std::memcpy(destination + left * channel_count, Row(y),
                    sizeof(D) * width_ * channel_count);
        std::fill_n(destination + (left + width_) * channel_count,
                    right * channel_count, value);
    }
    return dst;
}

template <typename D>
template <typename Color>
Bitmap<D> Bitmap<D>::PadChannels(int top, int down, int left, int right,
                              const Color *values, size_t value_count) const {
    INSPIRECV_CHECK(top >= 0 && down >= 0 && left >= 0 && right >= 0);
    INSPIRECV_CHECK(value_count == 1 || value_count == static_cast<size_t>(channels_))
      << "padding color size=" << value_count << ", channels=" << channels_;

    const D first = static_cast<D>(values[0]);
    bool uniform = true;
    for (size_t channel = 1; channel < value_count; ++channel) {
        uniform = uniform && static_cast<D>(values[channel]) == first;
    }
    if (uniform) return Pad(top, down, left, right, first);

    Bitmap<D> dst;
    dst.width_ = width_ + left + right;
    dst.height_ = height_ + top + down;
    dst.channels_ = channels_;
    dst.data_.reset(new D[static_cast<size_t>(dst.width_) * dst.height_ * channels_]);
    const int dst_w = dst.width_;
    const int dst_h = dst.height_;
    const int C = channels_;

    auto fill_pixels = [&](D* destination, int pixel_count) {
        if (pixel_count <= 0) return;
        for (int channel = 0; channel < C; ++channel) {
            destination[channel] = static_cast<D>(values[channel]);
        }
        size_t initialized = static_cast<size_t>(C);
        const size_t element_count = static_cast<size_t>(pixel_count) * C;
        while (initialized < element_count) {
            const size_t copy_count = std::min(initialized, element_count - initialized);
            std::memcpy(destination + initialized, destination,
                        copy_count * sizeof(D));
            initialized += copy_count;
        }
    };
    auto copy_border = [](D* destination, const D* source,
                          size_t element_count) {
        // Ten RGB pixels are the overwhelmingly common preprocessing border.
        // Keeping the byte count compile-time constant lets Clang/GCC emit two
        // inline vector moves instead of a tiny out-of-line memcpy call.
        if (element_count == 30) {
            std::memcpy(destination, source, 30 * sizeof(D));
        } else {
            std::memcpy(destination, source, element_count * sizeof(D));
        }
    };
    D* border_template = top > 0
                           ? dst.Row(0)
                           : (down > 0 ? dst.Row(top + height_) : dst.Row(0));
    fill_pixels(border_template, dst_w);
    if (top > 0) {
        for (int y = 1; y < top; ++y) {
            std::memcpy(dst.Row(y), border_template, sizeof(D) * dst_w * C);
        }
    }
    if (down > 0) {
        for (int y = dst_h - down; y < dst_h; ++y) {
            if (dst.Row(y) != border_template) {
                std::memcpy(dst.Row(y), border_template, sizeof(D) * dst_w * C);
            }
        }
    }
    for (int y = 0; y < height_; ++y) {
        D* destination = dst.Row(y + top);
        if (left > 0 && destination != border_template) {
            copy_border(destination, border_template,
                        static_cast<size_t>(left) * C);
        }
        std::memcpy(destination + left * C, Row(y), sizeof(D) * width_ * C);
        if (right > 0 && destination != border_template) {
            copy_border(destination + (left + width_) * C,
                        border_template + (left + width_) * C,
                        static_cast<size_t>(right) * C);
        }
    }
    return dst;
}

template <typename D>
Bitmap<D> Bitmap<D>::Pad(int top, int down, int left, int right,
                       const std::vector<D> &values) const {
    return PadChannels(top, down, left, right, values.data(), values.size());
}

template <typename D>
Bitmap<D> Bitmap<D>::Pad(int top, int down, int left, int right,
                       const double *values, size_t value_count) const {
    return PadChannels(top, down, left, right, values, value_count);
}

template <typename D>
void Bitmap<D>::AddAlphaChannel(Bitmap<D> &dst, int index, D alpha) const {
    INSPIRECV_CHECK(this != &dst);
    INSPIRECV_CHECK(channels_ == 3) << "channels_=" << channels_;
    INSPIRECV_CHECK(index == 0 || index == channels_);
    dst.Reset(width_, height_, channels_ + 1);
    auto src_iter = Data();
    auto dst_iter = dst.Data();
    for (int i = 0; i < height_ * width_; ++i) {
        if (index == 0)
            *(dst_iter++) = alpha;
        std::memcpy(dst_iter, src_iter, sizeof(D) * channels_);
        dst_iter += channels_;
        src_iter += channels_;
        if (index == channels_)
            *(dst_iter++) = alpha;
    }
}

template <typename D>
Bitmap<D> Bitmap<D>::Crop(const Rect<int> &rect, bool allow_padding) const {
    Bitmap<D> dst;
    if (allow_padding) {
        if (rect.ymin() >= height_ || rect.xmin() >= width_ || rect.ymax() <= 0 ||
            rect.xmax() <= 0) {
            dst.Reset(rect.GetWidth(), rect.GetHeight(), channels_);
            dst.Fill(0);
        } else {
            int src_top = std::max(rect.ymin(), 0);
            int src_left = std::max(rect.xmin(), 0);
            int src_height = std::min(rect.ymax(), height_) - src_top;
            int src_width = std::min(rect.xmax(), width_) - src_left;
            int dst_top = std::max(-rect.ymin(), 0);
            int dst_left = std::max(-rect.xmin(), 0);
            dst.Reset(rect.GetWidth(), rect.GetHeight(), channels_);
            dst.Fill(0);
            for (int i = 0; i < src_height; ++i) {
                std::memcpy(dst.at(dst_top + i, dst_left), at(src_top + i, src_left),
                            sizeof(D) * src_width * channels_);
            }
        }
    } else {
        INSPIRECV_CHECK(Rect<int>(0, 0, width_, height_).Contains(rect)) << rect;
        auto height = rect.GetHeight();
        auto width = rect.GetWidth();
        dst.Reset(width, height, channels_);
        for (int i = 0; i < height; ++i) {
            std::memcpy(dst.Row(i), this->at(i + rect.ymin(), rect.xmin()),
                        sizeof(D) * width * channels_);
        }
    }
    return dst;
}

template <typename D>
Bitmap<D> Bitmap<D>::Blur(int kernel) const {
    // WARN: the code is to be optimized.
    if (kernel == 1)
        return Clone();
    Bitmap<D> res;
    if (kernel < 5) {
        Bitmap<D> row_blur;
        row_blur.Reset(width_, height_, channels_);
        for (int c = 0; c < Channels(); ++c) {
            for (int i = 0; i < Height(); ++i) {
                auto r = Row(i);
                for (int j = 0; j < Width(); ++j) {
                    int min_k = std::max(0, j - (kernel - 1) / 2);
                    int max_k = std::min(Width() - 1, j + kernel / 2);
                    float s = 0;
                    for (int k = min_k; k <= max_k; ++k) {
                        s += r[k * Channels() + c];
                    }
                    row_blur.at(i, j)[c] = s / (max_k - min_k + 1);
                }
            }
        }

        res.Reset(width_, height_, channels_);
        for (int c = 0; c < Channels(); ++c) {
            for (int i = 0; i < Height(); ++i) {
                for (int j = 0; j < Width(); ++j) {
                    int min_k = std::max(0, i - (kernel - 1) / 2);
                    int max_k = std::min(Height() - 1, i + kernel / 2);
                    float s = 0;
                    for (int k = min_k; k <= max_k; ++k) {
                        s += row_blur.at(k, j)[c];
                    }
                    res.at(i, j)[c] = s / (max_k - min_k + 1);
                }
            }
        }
    } else {
        INSPIRECV_CHECK(channels_ == 1) << "channels: " << channels_;
        Bitmap<float> sum;
        sum.Reset(Width(), Height(), Channels());
        for (int i = 0; i < DataSize(); ++i)
            sum.Data()[i] = Data()[i];
        for (int i = 0; i < DataSize(); ++i)
            if (i % Width() != 0)
                sum.Data()[i] += sum.Data()[i - 1];
        for (int i = Width(); i < DataSize(); ++i)
            sum.Data()[i] += sum.Data()[i - Width()];

        res.Reset(Width(), Height(), Channels());
        for (int y = 0; y < Height(); ++y)
            for (int x = 0; x < Width(); ++x) {
                int x1 = x - kernel / 2 - 1;
                int x2 = x + (kernel - 1) / 2;
                int y1 = y - kernel / 2 - 1;
                int y2 = y + (kernel - 1) / 2;
                float v11 = (x1 < 0 || y1 < 0) ? 0 : sum.at(y1, x1)[0];
                float v12 = x1 < 0 ? 0 : sum.at(std::min(y2, Height() - 1), x1)[0];
                float v21 = y1 < 0 ? 0 : sum.at(y1, std::min(x2, Width() - 1))[0];
                float v22 = sum.at(std::min(y2, Height() - 1), std::min(x2, Width() - 1))[0];
                int count_x = std::min(x2, Width() - 1) - std::max(x1, -1);
                int count_y = std::min(y2, Height() - 1) - std::max(y1, -1);
                res.at(y, x)[0] = (v22 - v12 - v21 + v11) / (count_x * count_y);
            }
    }
    return res;
}

template <typename D>
Bitmap<D> Bitmap<D>::GaussianBlur(int ksize, float sigmaX) const {
    if (ksize <= 1)
        return Clone();
    if ((ksize & 1) == 0)
        ++ksize;  // enforce odd

    if (sigmaX <= 0.f) {
        // OpenCV-compatible heuristic for sigma from kernel size
        float half = 0.5f * (ksize - 1);
        sigmaX = 0.3f * (half - 1.f) + 0.8f;
        if (sigmaX <= 0.f)
            sigmaX = 0.8f;
    }

    const int radius = ksize / 2;

    // Build 1D Gaussian kernel
    std::vector<float> kernel(ksize);
    const float inv2Sigma2 = 1.0f / (2.0f * sigmaX * sigmaX);
    float sumw = 0.f;
    for (int i = -radius; i <= radius; ++i) {
        float w = std::exp(-(i * i) * inv2Sigma2);
        kernel[i + radius] = w;
        sumw += w;
    }
    const float invSum = 1.0f / sumw;
    for (int i = 0; i < ksize; ++i) kernel[i] *= invSum;
    // Force exact sum to 1.0f to preserve constant images after separable passes
    {
        float s = 0.f;
        for (int i = 0; i < ksize; ++i) s += kernel[i];
        kernel[radius] += (1.0f - s);
    }

    // Temporary buffer for horizontal pass (float for precision)
    std::vector<float> tmp(static_cast<size_t>(width_) * height_ * channels_);

    // Precompute clamped index tables to remove inner clamps
    std::vector<int> horizIndices(static_cast<size_t>(width_) * ksize);
    for (int x = 0; x < width_; ++x) {
        for (int k = -radius; k <= radius; ++k) {
            int idx = x + k;
            if (idx < 0) idx = 0; else if (idx >= width_) idx = width_ - 1;
            horizIndices[static_cast<size_t>(x) * ksize + (k + radius)] = idx;
        }
    }
    std::vector<int> vertIndices(static_cast<size_t>(height_) * ksize);
    for (int y = 0; y < height_; ++y) {
        for (int k = -radius; k <= radius; ++k) {
            int idy = y + k;
            if (idy < 0) idy = 0; else if (idy >= height_) idy = height_ - 1;
            vertIndices[static_cast<size_t>(y) * ksize + (k + radius)] = idy;
        }
    }

    // Parallel setup
    unsigned int hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 4;
    unsigned int numWorkers = std::min<unsigned int>(hw, static_cast<unsigned int>(height_));
    auto parallelForRows = [&](int rows, const std::function<void(int,int)> &fn) {
        if (numWorkers <= 1 || rows < 2) {
            fn(0, rows);
            return;
        }
        std::vector<std::thread> workers;
        workers.reserve(numWorkers);
        int chunk = (rows + static_cast<int>(numWorkers) - 1) / static_cast<int>(numWorkers);
        int start = 0;
        for (unsigned int t = 0; t < numWorkers && start < rows; ++t) {
            int end = std::min(start + chunk, rows);
            workers.emplace_back([=,&fn]() { fn(start, end); });
            start = end;
        }
        for (auto &th : workers) th.join();
    };

    // Horizontal pass (row-parallel)
    parallelForRows(height_, [&](int yBegin, int yEnd) {
        for (int y = yBegin; y < yEnd; ++y) {
            const D* rowPtr = Row(y);
            // Fast path: single-channel + small kernels (3 or 5) with SIMD
            if (channels_ == 1 && (ksize == 3 || ksize == 5)) {
                const float w0 = kernel[0];
                const float w1 = kernel[1];
                const float w2 = kernel[2];
                const float w3 = (ksize == 5 ? kernel[3] : 0.f);
                const float w4 = (ksize == 5 ? kernel[4] : 0.f);

                // Prepare a float row view
                std::vector<float> rf;
                const float* rowFloat = nullptr;
                if (std::is_same<D, float>::value) {
                    rowFloat = reinterpret_cast<const float*>(rowPtr);
                } else {
                    rf.resize(static_cast<size_t>(width_));
#if defined(__AVX2__)
                    int x = 0;
                    for (; x + 8 <= width_; x += 8) {
                        __m128i b8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(rowPtr + x));
                        __m128i u16 = _mm_cvtepu8_epi16(b8);
                        __m256i u32 = _mm256_cvtepu16_epi32(u16);
                        __m256 f32 = _mm256_cvtepi32_ps(u32);
                        _mm256_storeu_ps(rf.data() + x, f32);
                    }
                    for (; x < width_; ++x) rf[x] = static_cast<float>(rowPtr[x]);
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
                    int x = 0;
                    for (; x + 8 <= width_; x += 8) {
                        uint8x8_t u8 = vld1_u8(reinterpret_cast<const uint8_t*>(rowPtr) + x);
                        uint16x8_t u16 = vmovl_u8(u8);
                        uint32x4_t lo32 = vmovl_u16(vget_low_u16(u16));
                        uint32x4_t hi32 = vmovl_u16(vget_high_u16(u16));
                        float32x4_t flo = vcvtq_f32_u32(lo32);
                        float32x4_t fhi = vcvtq_f32_u32(hi32);
                        vst1q_f32(rf.data() + x, flo);
                        vst1q_f32(rf.data() + x + 4, fhi);
                    }
                    for (; x < width_; ++x) rf[x] = static_cast<float>(rowPtr[x]);
#else
                    for (int x = 0; x < width_; ++x) rf[x] = static_cast<float>(rowPtr[x]);
#endif
                    rowFloat = rf.data();
                }

                float* tmpRow = &tmp[static_cast<size_t>(y) * width_];

                if (ksize == 3) {
                    // edges with replicate
                    if (width_ >= 1) {
                        float acc0 = w0 * rowFloat[0] + w1 * rowFloat[0] + w2 * (width_ > 1 ? rowFloat[1] : rowFloat[0]);
                        tmpRow[0] = acc0;
                    }
                    if (width_ >= 2) {
                        int end = width_ - 1;
                        int x = 1;
#if defined(__AVX2__)
                        __m256 vw0 = _mm256_set1_ps(w0);
                        __m256 vw1 = _mm256_set1_ps(w1);
                        __m256 vw2 = _mm256_set1_ps(w2);
                        for (; x + 8 <= end; x += 8) {
                            __m256 l = _mm256_loadu_ps(rowFloat + x - 1);
                            __m256 c = _mm256_loadu_ps(rowFloat + x);
                            __m256 r = _mm256_loadu_ps(rowFloat + x + 1);
                            __m256 acc = _mm256_add_ps(_mm256_mul_ps(l, vw0), _mm256_mul_ps(c, vw1));
                            acc = _mm256_add_ps(acc, _mm256_mul_ps(r, vw2));
                            _mm256_storeu_ps(tmpRow + x, acc);
                        }
#elif defined(__SSE2__)
                        __m128 vw0 = _mm_set1_ps(w0);
                        __m128 vw1 = _mm_set1_ps(w1);
                        __m128 vw2 = _mm_set1_ps(w2);
                        for (; x + 4 <= end; x += 4) {
                            __m128 l = _mm_loadu_ps(rowFloat + x - 1);
                            __m128 c = _mm_loadu_ps(rowFloat + x);
                            __m128 r = _mm_loadu_ps(rowFloat + x + 1);
                            __m128 acc = _mm_add_ps(_mm_mul_ps(l, vw0), _mm_mul_ps(c, vw1));
                            acc = _mm_add_ps(acc, _mm_mul_ps(r, vw2));
                            _mm_storeu_ps(tmpRow + x, acc);
                        }
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
                        float32x4_t vw0 = vdupq_n_f32(w0);
                        float32x4_t vw1 = vdupq_n_f32(w1);
                        float32x4_t vw2 = vdupq_n_f32(w2);
                        for (; x + 4 <= end; x += 4) {
                            float32x4_t l = vld1q_f32(rowFloat + x - 1);
                            float32x4_t c = vld1q_f32(rowFloat + x);
                            float32x4_t r = vld1q_f32(rowFloat + x + 1);
                            float32x4_t acc = vmlaq_f32(vmulq_f32(l, vw0), c, vw1);
                            acc = vmlaq_f32(acc, r, vw2);
                            vst1q_f32(tmpRow + x, acc);
                        }
#endif
                        for (; x < end; ++x) {
                            tmpRow[x] = w0 * rowFloat[x - 1] + w1 * rowFloat[x] + w2 * rowFloat[x + 1];
                        }
                        // last element
                        tmpRow[end] = w0 * rowFloat[end - 1] + w1 * rowFloat[end] + w2 * rowFloat[end];
                    }
                } else {  // ksize == 5
                    // edges with replicate: x=0,1 and last two
                    if (width_ >= 1) {
                        float a = rowFloat[0], b = (width_ > 1 ? rowFloat[1] : a), c = (width_ > 2 ? rowFloat[2] : b);
                        tmpRow[0] = w0 * a + w1 * a + w2 * a + w3 * b + w4 * c;
                    }
                    if (width_ >= 2) {
                        float a0 = rowFloat[0], a1 = rowFloat[1], a2 = (width_ > 2 ? rowFloat[2] : a1), a3 = (width_ > 3 ? rowFloat[3] : a2);
                        tmpRow[1] = w0 * a0 + w1 * a0 + w2 * a1 + w3 * a2 + w4 * a3;
                    }
                    if (width_ >= 3) {
                        int x = 2;
                        int end = width_ - 3;
#if defined(__AVX2__)
                        __m256 vw0 = _mm256_set1_ps(w0);
                        __m256 vw1 = _mm256_set1_ps(w1);
                        __m256 vw2 = _mm256_set1_ps(w2);
                        __m256 vw3 = _mm256_set1_ps(w3);
                        __m256 vw4 = _mm256_set1_ps(w4);
                        for (; x + 8 <= end; x += 8) {
                            __m256 a = _mm256_loadu_ps(rowFloat + x - 2);
                            __m256 b = _mm256_loadu_ps(rowFloat + x - 1);
                            __m256 c = _mm256_loadu_ps(rowFloat + x);
                            __m256 d = _mm256_loadu_ps(rowFloat + x + 1);
                            __m256 e = _mm256_loadu_ps(rowFloat + x + 2);
                            __m256 acc = _mm256_mul_ps(a, vw0);
                            acc = _mm256_fmadd_ps(b, vw1, acc);
                            acc = _mm256_fmadd_ps(c, vw2, acc);
                            acc = _mm256_fmadd_ps(d, vw3, acc);
                            acc = _mm256_fmadd_ps(e, vw4, acc);
                            _mm256_storeu_ps(tmpRow + x, acc);
                        }
#elif defined(__SSE2__)
                        __m128 vw0 = _mm_set1_ps(w0);
                        __m128 vw1 = _mm_set1_ps(w1);
                        __m128 vw2 = _mm_set1_ps(w2);
                        __m128 vw3 = _mm_set1_ps(w3);
                        __m128 vw4 = _mm_set1_ps(w4);
                        for (; x + 4 <= end; x += 4) {
                            __m128 a = _mm_loadu_ps(rowFloat + x - 2);
                            __m128 b = _mm_loadu_ps(rowFloat + x - 1);
                            __m128 c = _mm_loadu_ps(rowFloat + x);
                            __m128 d = _mm_loadu_ps(rowFloat + x + 1);
                            __m128 e = _mm_loadu_ps(rowFloat + x + 2);
                            __m128 acc = _mm_mul_ps(a, vw0);
                            acc = _mm_add_ps(acc, _mm_mul_ps(b, vw1));
                            acc = _mm_add_ps(acc, _mm_mul_ps(c, vw2));
                            acc = _mm_add_ps(acc, _mm_mul_ps(d, vw3));
                            acc = _mm_add_ps(acc, _mm_mul_ps(e, vw4));
                            _mm_storeu_ps(tmpRow + x, acc);
                        }
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
                        float32x4_t vw0 = vdupq_n_f32(w0);
                        float32x4_t vw1 = vdupq_n_f32(w1);
                        float32x4_t vw2 = vdupq_n_f32(w2);
                        float32x4_t vw3 = vdupq_n_f32(w3);
                        float32x4_t vw4 = vdupq_n_f32(w4);
                        for (; x + 4 <= end; x += 4) {
                            float32x4_t a = vld1q_f32(rowFloat + x - 2);
                            float32x4_t b = vld1q_f32(rowFloat + x - 1);
                            float32x4_t c = vld1q_f32(rowFloat + x);
                            float32x4_t d = vld1q_f32(rowFloat + x + 1);
                            float32x4_t e = vld1q_f32(rowFloat + x + 2);
                            float32x4_t acc = vmulq_f32(a, vw0);
                            acc = vmlaq_f32(acc, b, vw1);
                            acc = vmlaq_f32(acc, c, vw2);
                            acc = vmlaq_f32(acc, d, vw3);
                            acc = vmlaq_f32(acc, e, vw4);
                            vst1q_f32(tmpRow + x, acc);
                        }
#endif
                        for (; x <= end; ++x) {
                            tmpRow[x] = w0 * rowFloat[x - 2] + w1 * rowFloat[x - 1] +
                                        w2 * rowFloat[x] + w3 * rowFloat[x + 1] +
                                        w4 * rowFloat[x + 2];
                        }
                        // trailing with replicate
                        int e2 = width_ - 1;
                        if (width_ >= 4) {
                            int e1 = width_ - 2;
                            tmpRow[e1] = w0 * rowFloat[e1 - 2] + w1 * rowFloat[e1 - 1] +
                                         w2 * rowFloat[e1] + w3 * rowFloat[e1 + 1] +
                                         w4 * rowFloat[e1 + 1];
                        }
                        // last element
                        tmpRow[e2] = w0 * rowFloat[std::max(0, e2 - 2)] +
                                     w1 * rowFloat[std::max(0, e2 - 1)] +
                                     w2 * rowFloat[e2] +
                                     w3 * rowFloat[e2] +
                                     w4 * rowFloat[e2];
                    }
                }
                continue;  // next row
            }

            // Generic path (any channels / any ksize)
            if (channels_ == 3 && (ksize == 3 || ksize == 5)) {
                // Horizontal pass for 3-channel interleaved using float accumulators
                // Build planar float rows for B,G,R
                std::vector<float> rfB(static_cast<size_t>(width_));
                std::vector<float> rfG(static_cast<size_t>(width_));
                std::vector<float> rfR(static_cast<size_t>(width_));
                if (std::is_same<D, float>::value) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
                    int x = 0;
                    for (; x + 4 <= width_; x += 4) {
                        float32x4x3_t tri = vld3q_f32(reinterpret_cast<const float*>(rowPtr) + x * 3);
                        vst1q_f32(rfB.data() + x, tri.val[0]);
                        vst1q_f32(rfG.data() + x, tri.val[1]);
                        vst1q_f32(rfR.data() + x, tri.val[2]);
                    }
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
                    int tailStart = (width_ & ~3);
#else
                    int tailStart = 0;
#endif
                    for (int xx = tailStart; xx < width_; ++xx) {
                        const float* p = reinterpret_cast<const float*>(rowPtr) + xx * 3;
                        rfB[xx] = p[0]; rfG[xx] = p[1]; rfR[xx] = p[2];
                    }
                } else {
                    // D == uint8_t
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
                    int x = 0;
                    for (; x + 16 <= width_; x += 16) {
                        const uint8_t* src = reinterpret_cast<const uint8_t*>(rowPtr) + x * 3;
                        uint8x16x3_t v = vld3q_u8(src);
                        // low 8
                        uint8x8_t b_lo8 = vget_low_u8(v.val[0]);
                        uint8x8_t g_lo8 = vget_low_u8(v.val[1]);
                        uint8x8_t r_lo8 = vget_low_u8(v.val[2]);
                        uint16x8_t b_lo16 = vmovl_u8(b_lo8);
                        uint16x8_t g_lo16 = vmovl_u8(g_lo8);
                        uint16x8_t r_lo16 = vmovl_u8(r_lo8);
                        uint32x4_t b_lo32_0 = vmovl_u16(vget_low_u16(b_lo16));
                        uint32x4_t b_lo32_1 = vmovl_u16(vget_high_u16(b_lo16));
                        uint32x4_t g_lo32_0 = vmovl_u16(vget_low_u16(g_lo16));
                        uint32x4_t g_lo32_1 = vmovl_u16(vget_high_u16(g_lo16));
                        uint32x4_t r_lo32_0 = vmovl_u16(vget_low_u16(r_lo16));
                        uint32x4_t r_lo32_1 = vmovl_u16(vget_high_u16(r_lo16));
                        float32x4_t b_lo_f0 = vcvtq_f32_u32(b_lo32_0);
                        float32x4_t b_lo_f1 = vcvtq_f32_u32(b_lo32_1);
                        float32x4_t g_lo_f0 = vcvtq_f32_u32(g_lo32_0);
                        float32x4_t g_lo_f1 = vcvtq_f32_u32(g_lo32_1);
                        float32x4_t r_lo_f0 = vcvtq_f32_u32(r_lo32_0);
                        float32x4_t r_lo_f1 = vcvtq_f32_u32(r_lo32_1);
                        vst1q_f32(rfB.data() + x + 0, b_lo_f0);
                        vst1q_f32(rfB.data() + x + 4, b_lo_f1);
                        vst1q_f32(rfG.data() + x + 0, g_lo_f0);
                        vst1q_f32(rfG.data() + x + 4, g_lo_f1);
                        vst1q_f32(rfR.data() + x + 0, r_lo_f0);
                        vst1q_f32(rfR.data() + x + 4, r_lo_f1);
                        // high 8
                        uint8x8_t b_hi8 = vget_high_u8(v.val[0]);
                        uint8x8_t g_hi8 = vget_high_u8(v.val[1]);
                        uint8x8_t r_hi8 = vget_high_u8(v.val[2]);
                        uint16x8_t b_hi16 = vmovl_u8(b_hi8);
                        uint16x8_t g_hi16 = vmovl_u8(g_hi8);
                        uint16x8_t r_hi16 = vmovl_u8(r_hi8);
                        uint32x4_t b_hi32_0 = vmovl_u16(vget_low_u16(b_hi16));
                        uint32x4_t b_hi32_1 = vmovl_u16(vget_high_u16(b_hi16));
                        uint32x4_t g_hi32_0 = vmovl_u16(vget_low_u16(g_hi16));
                        uint32x4_t g_hi32_1 = vmovl_u16(vget_high_u16(g_hi16));
                        uint32x4_t r_hi32_0 = vmovl_u16(vget_low_u16(r_hi16));
                        uint32x4_t r_hi32_1 = vmovl_u16(vget_high_u16(r_hi16));
                        float32x4_t b_hi_f0 = vcvtq_f32_u32(b_hi32_0);
                        float32x4_t b_hi_f1 = vcvtq_f32_u32(b_hi32_1);
                        float32x4_t g_hi_f0 = vcvtq_f32_u32(g_hi32_0);
                        float32x4_t g_hi_f1 = vcvtq_f32_u32(g_hi32_1);
                        float32x4_t r_hi_f0 = vcvtq_f32_u32(r_hi32_0);
                        float32x4_t r_hi_f1 = vcvtq_f32_u32(r_hi32_1);
                        vst1q_f32(rfB.data() + x + 8, b_hi_f0);
                        vst1q_f32(rfB.data() + x + 12, b_hi_f1);
                        vst1q_f32(rfG.data() + x + 8, g_hi_f0);
                        vst1q_f32(rfG.data() + x + 12, g_hi_f1);
                        vst1q_f32(rfR.data() + x + 8, r_hi_f0);
                        vst1q_f32(rfR.data() + x + 12, r_hi_f1);
                    }
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
                    int tailStart = (width_ & ~15);
#else
                    int tailStart = 0;
#endif
                    for (int xx = tailStart; xx < width_; ++xx) {
                        const uint8_t* p = reinterpret_cast<const uint8_t*>(rowPtr) + xx * 3;
                        rfB[xx] = static_cast<float>(p[0]);
                        rfG[xx] = static_cast<float>(p[1]);
                        rfR[xx] = static_cast<float>(p[2]);
                    }
                }

                auto convolve3 = [&](const std::vector<float>& src, float* dst) {
                    if (width_ >= 1) {
                        dst[0] = kernel[0] * src[0] + kernel[1] * src[0] + kernel[2] * (width_ > 1 ? src[1] : src[0]);
                    }
#if defined(__SSE2__)
                    if (width_ >= 3) {
                        int x = 1;
                        int end = width_ - 1;
                        __m128 vw0 = _mm_set1_ps(kernel[0]);
                        __m128 vw1 = _mm_set1_ps(kernel[1]);
                        __m128 vw2 = _mm_set1_ps(kernel[2]);
                        for (; x + 4 <= end; x += 4) {
                            __m128 l = _mm_loadu_ps(src.data() + x - 1);
                            __m128 c = _mm_loadu_ps(src.data() + x);
                            __m128 r = _mm_loadu_ps(src.data() + x + 1);
                            __m128 acc = _mm_mul_ps(l, vw0);
                            acc = _mm_add_ps(acc, _mm_mul_ps(c, vw1));
                            acc = _mm_add_ps(acc, _mm_mul_ps(r, vw2));
                            _mm_storeu_ps(dst + x, acc);
                        }
                        for (; x < end; ++x) {
                            dst[x] = kernel[0] * src[x - 1] + kernel[1] * src[x] + kernel[2] * src[x + 1];
                        }
                        // last
                        dst[end] = kernel[0] * src[end - 1] + kernel[1] * src[end] + kernel[2] * src[end];
                        return;
                    }
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
                    if (width_ >= 3) {
                        int x = 1;
                        int end = width_ - 1;
                        float32x4_t vw0 = vdupq_n_f32(kernel[0]);
                        float32x4_t vw1 = vdupq_n_f32(kernel[1]);
                        float32x4_t vw2 = vdupq_n_f32(kernel[2]);
                        for (; x + 4 <= end; x += 4) {
                            float32x4_t l = vld1q_f32(src.data() + x - 1);
                            float32x4_t c = vld1q_f32(src.data() + x);
                            float32x4_t r = vld1q_f32(src.data() + x + 1);
                            float32x4_t acc = vmulq_f32(l, vw0);
                            acc = vmlaq_f32(acc, c, vw1);
                            acc = vmlaq_f32(acc, r, vw2);
                            vst1q_f32(dst + x, acc);
                        }
                        for (; x < end; ++x) {
                            dst[x] = kernel[0] * src[x - 1] + kernel[1] * src[x] + kernel[2] * src[x + 1];
                        }
                        // last
                        dst[end] = kernel[0] * src[end - 1] + kernel[1] * src[end] + kernel[2] * src[end];
                        return;
                    }
#endif
                    // scalar fallback
                    for (int x = 1; x < std::max(1, width_ - 1); ++x) {
                        dst[x] = kernel[0] * src[x - 1] + kernel[1] * src[x] + kernel[2] * src[x + 1];
                    }
                    if (width_ >= 2) {
                        int end = width_ - 1;
                        dst[end] = kernel[0] * src[end - 1] + kernel[1] * src[end] + kernel[2] * src[end];
                    }
                };

                auto convolve5 = [&](const std::vector<float>& src, float* dst) {
                    if (width_ >= 1) {
                        float a = src[0], b = (width_ > 1 ? src[1] : a), c = (width_ > 2 ? src[2] : b);
                        dst[0] = kernel[0] * a + kernel[1] * a + kernel[2] * a + kernel[3] * b + kernel[4] * c;
                    }
                    if (width_ >= 2) {
                        float a0 = src[0], a1 = src[1], a2 = (width_ > 2 ? src[2] : a1), a3 = (width_ > 3 ? src[3] : a2);
                        dst[1] = kernel[0] * a0 + kernel[1] * a0 + kernel[2] * a1 + kernel[3] * a2 + kernel[4] * a3;
                    }
#if defined(__SSE2__)
                    if (width_ >= 5) {
                        int x = 2;
                        int end = width_ - 3;
                        __m128 vw0 = _mm_set1_ps(kernel[0]);
                        __m128 vw1 = _mm_set1_ps(kernel[1]);
                        __m128 vw2 = _mm_set1_ps(kernel[2]);
                        __m128 vw3 = _mm_set1_ps(kernel[3]);
                        __m128 vw4 = _mm_set1_ps(kernel[4]);
                        for (; x + 4 <= end; x += 4) {
                            __m128 a = _mm_loadu_ps(src.data() + x - 2);
                            __m128 b = _mm_loadu_ps(src.data() + x - 1);
                            __m128 c = _mm_loadu_ps(src.data() + x);
                            __m128 d = _mm_loadu_ps(src.data() + x + 1);
                            __m128 e = _mm_loadu_ps(src.data() + x + 2);
                            __m128 acc = _mm_mul_ps(a, vw0);
                            acc = _mm_add_ps(acc, _mm_mul_ps(b, vw1));
                            acc = _mm_add_ps(acc, _mm_mul_ps(c, vw2));
                            acc = _mm_add_ps(acc, _mm_mul_ps(d, vw3));
                            acc = _mm_add_ps(acc, _mm_mul_ps(e, vw4));
                            _mm_storeu_ps(dst + x, acc);
                        }
                        for (; x <= end; ++x) {
                            dst[x] = kernel[0] * src[x - 2] + kernel[1] * src[x - 1] +
                                     kernel[2] * src[x] + kernel[3] * src[x + 1] +
                                     kernel[4] * src[x + 2];
                        }
                        // tails replicate
                        if (width_ >= 4) {
                            int e1 = width_ - 2;
                            dst[e1] = kernel[0] * src[e1 - 2] + kernel[1] * src[e1 - 1] +
                                      kernel[2] * src[e1] + kernel[3] * src[e1 + 1] +
                                      kernel[4] * src[e1 + 1];
                        }
                        int e2 = width_ - 1;
                        dst[e2] = kernel[0] * src[std::max(0, e2 - 2)] +
                                  kernel[1] * src[std::max(0, e2 - 1)] +
                                  kernel[2] * src[e2] +
                                  kernel[3] * src[e2] +
                                  kernel[4] * src[e2];
                        return;
                    }
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
                    if (width_ >= 5) {
                        int x = 2;
                        int end = width_ - 3;
                        float32x4_t vw0 = vdupq_n_f32(kernel[0]);
                        float32x4_t vw1 = vdupq_n_f32(kernel[1]);
                        float32x4_t vw2 = vdupq_n_f32(kernel[2]);
                        float32x4_t vw3 = vdupq_n_f32(kernel[3]);
                        float32x4_t vw4 = vdupq_n_f32(kernel[4]);
                        for (; x + 4 <= end; x += 4) {
                            float32x4_t a = vld1q_f32(src.data() + x - 2);
                            float32x4_t b = vld1q_f32(src.data() + x - 1);
                            float32x4_t c = vld1q_f32(src.data() + x);
                            float32x4_t d = vld1q_f32(src.data() + x + 1);
                            float32x4_t e = vld1q_f32(src.data() + x + 2);
                            float32x4_t acc = vmulq_f32(a, vw0);
                            acc = vmlaq_f32(acc, b, vw1);
                            acc = vmlaq_f32(acc, c, vw2);
                            acc = vmlaq_f32(acc, d, vw3);
                            acc = vmlaq_f32(acc, e, vw4);
                            vst1q_f32(dst + x, acc);
                        }
                        for (; x <= end; ++x) {
                            dst[x] = kernel[0] * src[x - 2] + kernel[1] * src[x - 1] +
                                     kernel[2] * src[x] + kernel[3] * src[x + 1] +
                                     kernel[4] * src[x + 2];
                        }
                        // tails replicate
                        if (width_ >= 4) {
                            int e1 = width_ - 2;
                            dst[e1] = kernel[0] * src[e1 - 2] + kernel[1] * src[e1 - 1] +
                                      kernel[2] * src[e1] + kernel[3] * src[e1 + 1] +
                                      kernel[4] * src[e1 + 1];
                        }
                        int e2 = width_ - 1;
                        dst[e2] = kernel[0] * src[std::max(0, e2 - 2)] +
                                  kernel[1] * src[std::max(0, e2 - 1)] +
                                  kernel[2] * src[e2] +
                                  kernel[3] * src[e2] +
                                  kernel[4] * src[e2];
                        return;
                    }
#endif
                    // scalar fallback
                    for (int x = 2; x <= std::max(2, width_ - 3); ++x) {
                        if (x >= width_ - 2) break;
                        dst[x] = kernel[0] * src[x - 2] + kernel[1] * src[x - 1] +
                                 kernel[2] * src[x] + kernel[3] * src[x + 1] +
                                 kernel[4] * src[x + 2];
                    }
                    if (width_ >= 4) {
                        int e1 = width_ - 2;
                        dst[e1] = kernel[0] * src[e1 - 2] + kernel[1] * src[e1 - 1] +
                                  kernel[2] * src[e1] + kernel[3] * src[e1 + 1] +
                                  kernel[4] * src[e1 + 1];
                    }
                    int e2 = width_ - 1;
                    dst[e2] = kernel[0] * src[std::max(0, e2 - 2)] +
                              kernel[1] * src[std::max(0, e2 - 1)] +
                              kernel[2] * src[e2] +
                              kernel[3] * src[e2] +
                              kernel[4] * src[e2];
                };

                // Output buffers for this row
                std::vector<float> outB(static_cast<size_t>(width_));
                std::vector<float> outG(static_cast<size_t>(width_));
                std::vector<float> outR(static_cast<size_t>(width_));

                if (ksize == 3) {
                    convolve3(rfB, outB.data());
                    convolve3(rfG, outG.data());
                    convolve3(rfR, outR.data());
                } else {
                    convolve5(rfB, outB.data());
                    convolve5(rfG, outG.data());
                    convolve5(rfR, outR.data());
                }

                // Store interleaved into tmp
                for (int x = 0; x < width_; ++x) {
                    size_t base = (static_cast<size_t>(y) * width_ + x) * 3;
                    tmp[base + 0] = outB[x];
                    tmp[base + 1] = outG[x];
                    tmp[base + 2] = outR[x];
                }
                continue;
            }
            for (int x = 0; x < width_; ++x) {
                const int* xIdx = &horizIndices[static_cast<size_t>(x) * ksize];
                size_t base = (static_cast<size_t>(y) * width_ + x) * channels_;
                for (int c = 0; c < channels_; ++c) {
                    float acc = 0.f;
                    for (int k = 0; k < ksize; ++k) {
                        int xx = xIdx[k];
                        acc += static_cast<float>(rowPtr[xx * channels_ + c]) * kernel[k];
                    }
                    tmp[base + c] = acc;
                }
            }
        }
    });

    // Vertical pass -> output (row-parallel)
    Bitmap<D> dst;
    dst.Reset(width_, height_, channels_);
    D* dst_data = dst.Data();

    // Precompute numeric limits for saturation
    const float lo = static_cast<float>(std::numeric_limits<D>::lowest());
    const float hi = static_cast<float>(std::numeric_limits<D>::max());

    parallelForRows(height_, [&](int yBegin, int yEnd) {
        for (int y = yBegin; y < yEnd; ++y) {
            const int* yIdx = &vertIndices[static_cast<size_t>(y) * ksize];
            if (channels_ == 1) {
                int x = 0;
#if defined(__AVX2__)
                for (; x + 8 <= width_; x += 8) {
                    __m256 acc0 = _mm256_setzero_ps();
                    for (int k = 0; k < ksize; ++k) {
                        int yy = yIdx[k];
                        const float* src = &tmp[static_cast<size_t>(yy) * width_ + x];
                        __m256 s = _mm256_loadu_ps(src);
                        __m256 w = _mm256_set1_ps(kernel[k]);
#  if defined(__FMA__)
                        acc0 = _mm256_fmadd_ps(s, w, acc0);
#  else
                        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(s, w));
#  endif
                    }
                    alignas(32) float buf[8];
                    _mm256_storeu_ps(buf, acc0);
                    size_t baseOut = static_cast<size_t>(y) * width_ + x;
                    for (int i = 0; i < 8; ++i) {
                        float v = buf[i];
                        if (std::is_integral<D>::value) v = std::floor(v + 0.5f);
                        v = v < lo ? lo : (v > hi ? hi : v);
                        dst_data[baseOut + i] = static_cast<D>(v);
                    }
                }
#elif defined(__SSE2__)
                for (; x + 4 <= width_; x += 4) {
                    __m128 acc0 = _mm_setzero_ps();
                    for (int k = 0; k < ksize; ++k) {
                        int yy = yIdx[k];
                        const float* src = &tmp[static_cast<size_t>(yy) * width_ + x];
                        __m128 s = _mm_loadu_ps(src);
                        __m128 w = _mm_set1_ps(kernel[k]);
                        acc0 = _mm_add_ps(acc0, _mm_mul_ps(s, w));
                    }
                    alignas(16) float buf[4];
                    _mm_storeu_ps(buf, acc0);
                    size_t baseOut = static_cast<size_t>(y) * width_ + x;
                    for (int i = 0; i < 4; ++i) {
                        float v = buf[i];
                        if (std::is_integral<D>::value) v = std::floor(v + 0.5f);
                        v = v < lo ? lo : (v > hi ? hi : v);
                        dst_data[baseOut + i] = static_cast<D>(v);
                    }
                }
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
                for (; x + 4 <= width_; x += 4) {
                    float32x4_t acc0 = vdupq_n_f32(0.0f);
                    for (int k = 0; k < ksize; ++k) {
                        int yy = yIdx[k];
                        const float* src = &tmp[static_cast<size_t>(yy) * width_ + x];
                        float32x4_t s = vld1q_f32(src);
                        float32x4_t w = vdupq_n_f32(kernel[k]);
                        acc0 = vmlaq_f32(acc0, s, w);
                    }
                    alignas(16) float buf[4];
                    vst1q_f32(buf, acc0);
                    size_t baseOut = static_cast<size_t>(y) * width_ + x;
                    for (int i = 0; i < 4; ++i) {
                        float v = buf[i];
                        if (std::is_integral<D>::value) v = std::floor(v + 0.5f);
                        v = v < lo ? lo : (v > hi ? hi : v);
                        dst_data[baseOut + i] = static_cast<D>(v);
                    }
                }
#endif
                // scalar tail
                for (; x < width_; ++x) {
                    float acc = 0.f;
                    for (int k = 0; k < ksize; ++k) {
                        int yy = yIdx[k];
                        acc += tmp[static_cast<size_t>(yy) * width_ + x] * kernel[k];
                    }
                    if (std::is_integral<D>::value) acc = std::floor(acc + 0.5f);
                    acc = acc < lo ? lo : (acc > hi ? hi : acc);
                    dst_data[static_cast<size_t>(y) * width_ + x] = static_cast<D>(acc);
                }
            } else if (channels_ == 3) {
                int x = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
                for (; x + 4 <= width_; x += 4) {
                    float32x4_t accB = vdupq_n_f32(0.0f);
                    float32x4_t accG = vdupq_n_f32(0.0f);
                    float32x4_t accR = vdupq_n_f32(0.0f);
                    for (int k = 0; k < ksize; ++k) {
                        int yy = yIdx[k];
                        size_t baseIn = (static_cast<size_t>(yy) * width_ + x) * 3;
                        float32x4x3_t tri = vld3q_f32(&tmp[baseIn]);
                        float32x4_t w = vdupq_n_f32(kernel[k]);
                        accB = vmlaq_f32(accB, tri.val[0], w);
                        accG = vmlaq_f32(accG, tri.val[1], w);
                        accR = vmlaq_f32(accR, tri.val[2], w);
                    }
                    size_t baseOut = (static_cast<size_t>(y) * width_ + x) * 3;
                    if (std::is_same<D, float>::value) {
                        float* out = reinterpret_cast<float*>(dst_data) + baseOut;
                        float32x4x3_t outv{accB, accG, accR};
                        vst3q_f32(out, outv);
                    } else {
                        alignas(16) float bb[4], gg[4], rr[4];
                        vst1q_f32(bb, accB);
                        vst1q_f32(gg, accG);
                        vst1q_f32(rr, accR);
                        for (int i = 0; i < 4; ++i) {
                            float vb = bb[i], vg = gg[i], vr = rr[i];
                            vb = vb < lo ? lo : (vb > hi ? hi : vb);
                            vg = vg < lo ? lo : (vg > hi ? hi : vg);
                            vr = vr < lo ? lo : (vr > hi ? hi : vr);
                            dst_data[baseOut + i * 3 + 0] = static_cast<D>(std::floor(vb + 0.5f));
                            dst_data[baseOut + i * 3 + 1] = static_cast<D>(std::floor(vg + 0.5f));
                            dst_data[baseOut + i * 3 + 2] = static_cast<D>(std::floor(vr + 0.5f));
                        }
                    }
                }
#endif
                for (; x < width_; ++x) {
                    float accB = 0.f, accG = 0.f, accR = 0.f;
                    for (int k = 0; k < ksize; ++k) {
                        int yy = yIdx[k];
                        size_t baseIn = (static_cast<size_t>(yy) * width_ + x) * 3;
                        accB += tmp[baseIn + 0] * kernel[k];
                        accG += tmp[baseIn + 1] * kernel[k];
                        accR += tmp[baseIn + 2] * kernel[k];
                    }
                    size_t baseOut = (static_cast<size_t>(y) * width_ + x) * 3;
                    if (std::is_same<D, float>::value) {
                        dst_data[baseOut + 0] = static_cast<D>(accB);
                        dst_data[baseOut + 1] = static_cast<D>(accG);
                        dst_data[baseOut + 2] = static_cast<D>(accR);
                    } else {
                        float vb = std::floor(accB + 0.5f);
                        float vg = std::floor(accG + 0.5f);
                        float vr = std::floor(accR + 0.5f);
                        vb = vb < lo ? lo : (vb > hi ? hi : vb);
                        vg = vg < lo ? lo : (vg > hi ? hi : vg);
                        vr = vr < lo ? lo : (vr > hi ? hi : vr);
                        dst_data[baseOut + 0] = static_cast<D>(vb);
                        dst_data[baseOut + 1] = static_cast<D>(vg);
                        dst_data[baseOut + 2] = static_cast<D>(vr);
                    }
                }
            } else {
                // Fallback: original scalar path for multi-channel
                for (int x = 0; x < width_; ++x) {
                    size_t baseOut = (static_cast<size_t>(y) * width_ + x) * channels_;
                    for (int c = 0; c < channels_; ++c) {
                        float acc = 0.f;
                        for (int k = 0; k < ksize; ++k) {
                            int yy = yIdx[k];
                            size_t baseIn = (static_cast<size_t>(yy) * width_ + x) * channels_ + c;
                            acc += tmp[baseIn] * kernel[k];
                        }
                        if (std::is_integral<D>::value) acc = std::floor(acc + 0.5f);
                        acc = acc < lo ? lo : (acc > hi ? hi : acc);
                        dst_data[baseOut + c] = static_cast<D>(acc);
                    }
                }
            }
        }
    });

    return dst;
}

template <typename D>
Bitmap<D> Bitmap<D>::MinFilter(int kernel_left, int kernel_right, int kernel_top,
                             int kernel_bottom) const {
    INSPIRECV_CHECK(Channels() == 1) << "channels=" << Channels();
    const int W = Width();
    const int H = Height();
    // Fast path: 3x3 erode (u8 single-channel)
    if (std::is_same<D, uint8_t>::value &&
        kernel_left == 1 && kernel_right == 1 && kernel_top == 1 && kernel_bottom == 1) {
        Bitmap<D> dst;
        dst.Reset(W, H, 1);
        D* out = dst.Data();
        for (int y = 0; y < H; ++y) {
            const D* r0 = Row(std::max(0, y - 1));
            const D* r1 = Row(y);
            const D* r2 = Row(std::min(H - 1, y + 1));
            int x = 0;
            // left edge
            if (W >= 1) {
                D v = std::min<D>(std::min<D>(std::min(r0[0], r0[0]), std::min(r0[1 < W ? 1 : 0], r0[0])),
                                  std::min<D>(std::min<D>(std::min(r1[0], r1[0]), std::min(r1[1 < W ? 1 : 0], r1[0])),
                                              std::min<D>(std::min(r2[0], r2[0]), std::min(r2[1 < W ? 1 : 0], r2[0]))));
                out[y * W + 0] = v;
                x = 1;
            }
#if defined(__AVX2__)
            {
                static bool logged = false;
                if (!logged) {
                    INSPIRECV_LOG(INFO) << "MinFilter: using AVX2 3x3 fast path";
                    logged = true;
                }
            }
            for (; x + 32 <= W - 1; x += 32) {
                __m256i r0_l = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r0 + x - 1));
                __m256i r0_c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r0 + x));
                __m256i r0_r = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r0 + x + 1));
                __m256i h0 = _mm256_min_epu8(_mm256_min_epu8(r0_l, r0_c), r0_r);

                __m256i r1_l = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r1 + x - 1));
                __m256i r1_c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r1 + x));
                __m256i r1_r = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r1 + x + 1));
                __m256i h1 = _mm256_min_epu8(_mm256_min_epu8(r1_l, r1_c), r1_r);

                __m256i r2_l = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r2 + x - 1));
                __m256i r2_c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r2 + x));
                __m256i r2_r = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r2 + x + 1));
                __m256i h2 = _mm256_min_epu8(_mm256_min_epu8(r2_l, r2_c), r2_r);

                __m256i vmin = _mm256_min_epu8(_mm256_min_epu8(h0, h1), h2);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + y * W + x), vmin);
            }
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
            {
                static bool logged = true;
                if (!logged) {
                    INSPIRECV_LOG(INFO) << "MinFilter: using NEON 3x3 fast path";
                    logged = true;
                }
            }
            for (; x + 16 <= W - 1; x += 16) {
                uint8x16_t r0_l = vld1q_u8(reinterpret_cast<const uint8_t*>(r0 + x - 1));
                uint8x16_t r0_c = vld1q_u8(reinterpret_cast<const uint8_t*>(r0 + x));
                uint8x16_t r0_r = vld1q_u8(reinterpret_cast<const uint8_t*>(r0 + x + 1));
                uint8x16_t h0 = vminq_u8(vminq_u8(r0_l, r0_c), r0_r);

                uint8x16_t r1_l = vld1q_u8(reinterpret_cast<const uint8_t*>(r1 + x - 1));
                uint8x16_t r1_c = vld1q_u8(reinterpret_cast<const uint8_t*>(r1 + x));
                uint8x16_t r1_r = vld1q_u8(reinterpret_cast<const uint8_t*>(r1 + x + 1));
                uint8x16_t h1 = vminq_u8(vminq_u8(r1_l, r1_c), r1_r);

                uint8x16_t r2_l = vld1q_u8(reinterpret_cast<const uint8_t*>(r2 + x - 1));
                uint8x16_t r2_c = vld1q_u8(reinterpret_cast<const uint8_t*>(r2 + x));
                uint8x16_t r2_r = vld1q_u8(reinterpret_cast<const uint8_t*>(r2 + x + 1));
                uint8x16_t h2 = vminq_u8(vminq_u8(r2_l, r2_c), r2_r);

                uint8x16_t vmin = vminq_u8(vminq_u8(h0, h1), h2);
                vst1q_u8(reinterpret_cast<uint8_t*>(out + y * W + x), vmin);
            }
#else
            {
                static bool logged = false;
                if (!logged) {
                    INSPIRECV_LOG(INFO) << "MinFilter: SIMD fast path not compiled (no AVX2/NEON)";
                    logged = true;
                }
            }
#endif
            // scalar tail (interior)
            for (; x < std::max(1, W - 1); ++x) {
                D v = 255;
                for (int ky = -1; ky <= 1; ++ky) {
                    const D* ry = Row(std::min(H - 1, std::max(0, y + ky)));
                    for (int kx = -1; kx <= 1; ++kx) {
                        int xx = std::min(W - 1, std::max(0, x + kx));
                        v = std::min<D>(v, ry[xx]);
                    }
                }
                out[y * W + x] = v;
            }
            // right edge
            if (W >= 2) {
                int xe = W - 1;
                D v = 255;
                for (int ky = -1; ky <= 1; ++ky) {
                    const D* ry = Row(std::min(H - 1, std::max(0, y + ky)));
                    for (int kx = -1; kx <= 1; ++kx) {
                        int xx = std::min(W - 1, std::max(0, xe + kx));
                        v = std::min<D>(v, ry[xx]);
                    }
                }
                out[y * W + xe] = v;
            }
        }
        return dst;
    }
    const int L = kernel_left + 1;
    const int R = kernel_right + 1;
    const int T = kernel_top + 1;
    const int B = kernel_bottom + 1;

    Bitmap<D> tmp_image;
    tmp_image.Reset(W, H, 1);

    // Horizontal pass using trailing minima (left) and leading minima (right via reversed trailing)
    {
        std::vector<D> left_trailing(static_cast<size_t>(W));
        std::vector<D> right_leading(static_cast<size_t>(W));

        auto trailing_min = [&](const D* row, D* out, int n, int win) {
            std::deque<int> dq;
            for (int i = 0; i < n; ++i) {
                while (!dq.empty() && row[dq.back()] >= row[i]) dq.pop_back();
                dq.push_back(i);
                int head = i - win + 1;  // head of window
                if (dq.front() < head) dq.pop_front();
                out[i] = row[dq.front()];
            }
        };

        auto leading_min = [&](const D* row, D* out, int n, int win) {
            // compute trailing minima on reversed row, then map back
            std::deque<int> dq;
            for (int i = n - 1; i >= 0; --i) {
                int ri = n - 1 - i;  // reversed index increasing
                D v = row[i];
                while (!dq.empty() && row[n - 1 - dq.back()] >= v) dq.pop_back();
                dq.push_back(ri);
                int head = ri - win + 1;
                if (dq.front() < head) dq.pop_front();
                out[i] = row[n - 1 - dq.front()];
            }
        };

        for (int y = 0; y < H; ++y) {
            const D* row = Row(y);
            D* tmp_row = tmp_image.Row(y);

            if (kernel_left == 0 && kernel_right == 0) {
                std::memcpy(tmp_row, row, static_cast<size_t>(W) * sizeof(D));
                continue;
            }

            trailing_min(row, left_trailing.data(), W, L);
            leading_min(row, right_leading.data(), W, R);

            // Combine per-position minima
            int x = 0;
#if defined(__AVX2__)
            if (std::is_same<D, uint8_t>::value) {
                for (; x + 32 <= W; x += 32) {
                    __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(left_trailing.data() + x));
                    __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(right_leading.data() + x));
                    __m256i m = _mm256_min_epu8(a, b);
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp_row + x), m);
                }
            }
#endif
#if defined(__SSE2__)
            if (std::is_same<D, uint8_t>::value) {
                for (; x + 16 <= W; x += 16) {
                    __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(left_trailing.data() + x));
                    __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(right_leading.data() + x));
                    __m128i m = _mm_min_epu8(a, b);
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(tmp_row + x), m);
                }
            }
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
            if (std::is_same<D, uint8_t>::value) {
                for (; x + 16 <= W; x += 16) {
                    uint8x16_t a = vld1q_u8(reinterpret_cast<const uint8_t*>(left_trailing.data() + x));
                    uint8x16_t b = vld1q_u8(reinterpret_cast<const uint8_t*>(right_leading.data() + x));
                    uint8x16_t m = vminq_u8(a, b);
                    vst1q_u8(reinterpret_cast<uint8_t*>(tmp_row + x), m);
                }
            }
#endif
            for (; x < W; ++x) tmp_row[x] = std::min(left_trailing[x], right_leading[x]);
        }
    }

    // Vertical pass using column-wise trailing/leading minima
    if (kernel_top == 0 && kernel_bottom == 0) {
        return tmp_image;
    }

    Bitmap<D> res_image;
    res_image.Reset(W, H, 1);

    std::vector<D> col(static_cast<size_t>(H));
    std::vector<D> top_trailing(static_cast<size_t>(H));
    std::vector<D> bottom_leading(static_cast<size_t>(H));

    auto trailing_min_col = [&](const D* c, D* out, int n, int win) {
        std::deque<int> dq;
        for (int i = 0; i < n; ++i) {
            while (!dq.empty() && c[dq.back()] >= c[i]) dq.pop_back();
            dq.push_back(i);
            int head = i - win + 1;
            if (dq.front() < head) dq.pop_front();
            out[i] = c[dq.front()];
        }
    };

    auto leading_min_col = [&](const D* c, D* out, int n, int win) {
        std::deque<int> dq;
        for (int i = n - 1; i >= 0; --i) {
            int ri = n - 1 - i;
            D v = c[i];
            while (!dq.empty() && c[n - 1 - dq.back()] >= v) dq.pop_back();
            dq.push_back(ri);
            int head = ri - win + 1;
            if (dq.front() < head) dq.pop_front();
            out[i] = c[n - 1 - dq.front()];
        }
    };

    for (int x = 0; x < W; ++x) {
        for (int y = 0; y < H; ++y) col[y] = tmp_image.Row(y)[x];
        trailing_min_col(col.data(), top_trailing.data(), H, T);
        leading_min_col(col.data(), bottom_leading.data(), H, B);
        for (int y = 0; y < H; ++y) res_image.Row(y)[x] = std::min(top_trailing[y], bottom_leading[y]);
    }

    return res_image;
}

template <typename D>
Bitmap<D> Bitmap<D>::MaxFilter(int kernel_left, int kernel_right, int kernel_top,
                             int kernel_bottom) const {
    INSPIRECV_CHECK(Channels() == 1) << "channels=" << Channels();
    const int W = Width();
    const int H = Height();
    // Fast path: 3x3 dilate (u8 single-channel)
    if (std::is_same<D, uint8_t>::value &&
        kernel_left == 1 && kernel_right == 1 && kernel_top == 1 && kernel_bottom == 1) {
        Bitmap<D> dst;
        dst.Reset(W, H, 1);
        D* out = dst.Data();
        for (int y = 0; y < H; ++y) {
            const D* r0 = Row(std::max(0, y - 1));
            const D* r1 = Row(y);
            const D* r2 = Row(std::min(H - 1, y + 1));
            int x = 0;
            // left edge
            if (W >= 1) {
                D v = 0;
                for (int ky = -1; ky <= 1; ++ky) {
                    const D* ry = Row(std::min(H - 1, std::max(0, y + ky)));
                    for (int kx = -1; kx <= 1; ++kx) {
                        int xx = std::min(W - 1, std::max(0, 0 + kx));
                        v = std::max<D>(v, ry[xx]);
                    }
                }
                out[y * W + 0] = v;
                x = 1;
            }
#if defined(__AVX2__)
            for (; x + 32 <= W - 1; x += 32) {
                __m256i r0_l = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r0 + x - 1));
                __m256i r0_c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r0 + x));
                __m256i r0_r = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r0 + x + 1));
                __m256i h0 = _mm256_max_epu8(_mm256_max_epu8(r0_l, r0_c), r0_r);

                __m256i r1_l = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r1 + x - 1));
                __m256i r1_c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r1 + x));
                __m256i r1_r = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r1 + x + 1));
                __m256i h1 = _mm256_max_epu8(_mm256_max_epu8(r1_l, r1_c), r1_r);

                __m256i r2_l = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r2 + x - 1));
                __m256i r2_c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r2 + x));
                __m256i r2_r = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r2 + x + 1));
                __m256i h2 = _mm256_max_epu8(_mm256_max_epu8(r2_l, r2_c), r2_r);

                __m256i vmaxv = _mm256_max_epu8(_mm256_max_epu8(h0, h1), h2);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + y * W + x), vmaxv);
            }
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
            for (; x + 16 <= W - 1; x += 16) {
                uint8x16_t r0_l = vld1q_u8(reinterpret_cast<const uint8_t*>(r0 + x - 1));
                uint8x16_t r0_c = vld1q_u8(reinterpret_cast<const uint8_t*>(r0 + x));
                uint8x16_t r0_r = vld1q_u8(reinterpret_cast<const uint8_t*>(r0 + x + 1));
                uint8x16_t h0 = vmaxq_u8(vmaxq_u8(r0_l, r0_c), r0_r);

                uint8x16_t r1_l = vld1q_u8(reinterpret_cast<const uint8_t*>(r1 + x - 1));
                uint8x16_t r1_c = vld1q_u8(reinterpret_cast<const uint8_t*>(r1 + x));
                uint8x16_t r1_r = vld1q_u8(reinterpret_cast<const uint8_t*>(r1 + x + 1));
                uint8x16_t h1 = vmaxq_u8(vmaxq_u8(r1_l, r1_c), r1_r);

                uint8x16_t r2_l = vld1q_u8(reinterpret_cast<const uint8_t*>(r2 + x - 1));
                uint8x16_t r2_c = vld1q_u8(reinterpret_cast<const uint8_t*>(r2 + x));
                uint8x16_t r2_r = vld1q_u8(reinterpret_cast<const uint8_t*>(r2 + x + 1));
                uint8x16_t h2 = vmaxq_u8(vmaxq_u8(r2_l, r2_c), r2_r);

                uint8x16_t vmaxv = vmaxq_u8(vmaxq_u8(h0, h1), h2);
                vst1q_u8(reinterpret_cast<uint8_t*>(out + y * W + x), vmaxv);
            }
#endif
            // scalar tail (interior)
            for (; x < std::max(1, W - 1); ++x) {
                D v = 0;
                for (int ky = -1; ky <= 1; ++ky) {
                    const D* ry = Row(std::min(H - 1, std::max(0, y + ky)));
                    for (int kx = -1; kx <= 1; ++kx) {
                        int xx = std::min(W - 1, std::max(0, x + kx));
                        v = std::max<D>(v, ry[xx]);
                    }
                }
                out[y * W + x] = v;
            }
            // right edge
            if (W >= 2) {
                int xe = W - 1;
                D v = 0;
                for (int ky = -1; ky <= 1; ++ky) {
                    const D* ry = Row(std::min(H - 1, std::max(0, y + ky)));
                    for (int kx = -1; kx <= 1; ++kx) {
                        int xx = std::min(W - 1, std::max(0, xe + kx));
                        v = std::max<D>(v, ry[xx]);
                    }
                }
                out[y * W + xe] = v;
            }
        }
        return dst;
    }
    const int L = kernel_left + 1;
    const int R = kernel_right + 1;
    const int T = kernel_top + 1;
    const int B = kernel_bottom + 1;

    Bitmap<D> tmp_image;
    tmp_image.Reset(W, H, 1);

    // Horizontal pass using trailing maxima (left) and leading maxima (right via reversed trailing)
    {
        std::vector<D> left_trailing(static_cast<size_t>(W));
        std::vector<D> right_leading(static_cast<size_t>(W));

        auto trailing_max = [&](const D* row, D* out, int n, int win) {
            std::deque<int> dq;
            for (int i = 0; i < n; ++i) {
                while (!dq.empty() && row[dq.back()] <= row[i]) dq.pop_back();
                dq.push_back(i);
                int head = i - win + 1;
                if (dq.front() < head) dq.pop_front();
                out[i] = row[dq.front()];
            }
        };

        auto leading_max = [&](const D* row, D* out, int n, int win) {
            std::deque<int> dq;
            for (int i = n - 1; i >= 0; --i) {
                int ri = n - 1 - i;
                D v = row[i];
                while (!dq.empty() && row[n - 1 - dq.back()] <= v) dq.pop_back();
                dq.push_back(ri);
                int head = ri - win + 1;
                if (dq.front() < head) dq.pop_front();
                out[i] = row[n - 1 - dq.front()];
            }
        };

        for (int y = 0; y < H; ++y) {
            const D* row = Row(y);
            D* tmp_row = tmp_image.Row(y);

            if (kernel_left == 0 && kernel_right == 0) {
                std::memcpy(tmp_row, row, static_cast<size_t>(W) * sizeof(D));
                continue;
            }

            trailing_max(row, left_trailing.data(), W, L);
            leading_max(row, right_leading.data(), W, R);

            int x = 0;
#if defined(__AVX2__)
            if (std::is_same<D, uint8_t>::value) {
                for (; x + 32 <= W; x += 32) {
                    __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(left_trailing.data() + x));
                    __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(right_leading.data() + x));
                    __m256i m = _mm256_max_epu8(a, b);
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp_row + x), m);
                }
            }
#endif
#if defined(__SSE2__)
            if (std::is_same<D, uint8_t>::value) {
                for (; x + 16 <= W; x += 16) {
                    __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(left_trailing.data() + x));
                    __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(right_leading.data() + x));
                    __m128i m = _mm_max_epu8(a, b);
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(tmp_row + x), m);
                }
            }
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
            if (std::is_same<D, uint8_t>::value) {
                for (; x + 16 <= W; x += 16) {
                    uint8x16_t a = vld1q_u8(reinterpret_cast<const uint8_t*>(left_trailing.data() + x));
                    uint8x16_t b = vld1q_u8(reinterpret_cast<const uint8_t*>(right_leading.data() + x));
                    uint8x16_t m = vmaxq_u8(a, b);
                    vst1q_u8(reinterpret_cast<uint8_t*>(tmp_row + x), m);
                }
            }
#endif
            for (; x < W; ++x) tmp_row[x] = std::max(left_trailing[x], right_leading[x]);
        }
    }

    if (kernel_top == 0 && kernel_bottom == 0) {
        return tmp_image;
    }

    Bitmap<D> res_image;
    res_image.Reset(W, H, 1);

    std::vector<D> col(static_cast<size_t>(H));
    std::vector<D> top_trailing(static_cast<size_t>(H));
    std::vector<D> bottom_leading(static_cast<size_t>(H));

    auto trailing_max_col = [&](const D* c, D* out, int n, int win) {
        std::deque<int> dq;
        for (int i = 0; i < n; ++i) {
            while (!dq.empty() && c[dq.back()] <= c[i]) dq.pop_back();
            dq.push_back(i);
            int head = i - win + 1;
            if (dq.front() < head) dq.pop_front();
            out[i] = c[dq.front()];
        }
    };

    auto leading_max_col = [&](const D* c, D* out, int n, int win) {
        std::deque<int> dq;
        for (int i = n - 1; i >= 0; --i) {
            int ri = n - 1 - i;
            D v = c[i];
            while (!dq.empty() && c[n - 1 - dq.back()] <= v) dq.pop_back();
            dq.push_back(ri);
            int head = ri - win + 1;
            if (dq.front() < head) dq.pop_front();
            out[i] = c[n - 1 - dq.front()];
        }
    };

    for (int x = 0; x < W; ++x) {
        for (int y = 0; y < H; ++y) col[y] = tmp_image.Row(y)[x];
        trailing_max_col(col.data(), top_trailing.data(), H, T);
        leading_max_col(col.data(), bottom_leading.data(), H, B);
        for (int y = 0; y < H; ++y) res_image.Row(y)[x] = std::max(top_trailing[y], bottom_leading[y]);
    }

    return res_image;
}

template <typename D>
Bitmap<D> Bitmap<D>::FlipLeftRight() const {
    Bitmap<D> dst;
    dst.Reset(width_, height_, channels_);
#if (defined(__ARM_NEON) || defined(__ARM_NEON__)) 
    if (std::is_same<D, uint8_t>::value) {
        const int C = channels_;
        for (int i = 0; i < height_; ++i) {
            const uint8_t* src_row = reinterpret_cast<const uint8_t*>(Row(i));
            uint8_t* dst_row = reinterpret_cast<uint8_t*>(dst.Row(i));
            if (C == 1) {
                int x = 0;
                for (; x + 16 <= width_; x += 16) {
                    const uint8_t* s = src_row + (width_ - x - 16);
                    uint8x16_t v = vld1q_u8(s);
                    uint8x16_t t = vrev64q_u8(v);
                    uint8x16_t r = vextq_u8(t, t, 8);
                    vst1q_u8(dst_row + x, r);
                }
                for (; x < width_; ++x) {
                    dst_row[x] = src_row[width_ - 1 - x];
                }
            } else if (C == 3) {
                int x = 0;
                for (; x + 16 <= width_; x += 16) {
                    const uint8_t* s = src_row + (width_ - x - 16) * 3;
                    uint8x16x3_t bgr = vld3q_u8(s);
                    uint8x16_t t0 = vrev64q_u8(bgr.val[0]);
                    uint8x16_t t1 = vrev64q_u8(bgr.val[1]);
                    uint8x16_t t2 = vrev64q_u8(bgr.val[2]);
                    uint8x16_t r0 = vextq_u8(t0, t0, 8);
                    uint8x16_t r1 = vextq_u8(t1, t1, 8);
                    uint8x16_t r2 = vextq_u8(t2, t2, 8);
                    uint8x16x3_t out = {r0, r1, r2};
                    vst3q_u8(dst_row + x * 3, out);
                }
                for (; x < width_; ++x) {
                    const uint8_t* sp = src_row + (width_ - 1 - x) * 3;
                    uint8_t* dp = dst_row + x * 3;
                    dp[0] = sp[0]; dp[1] = sp[1]; dp[2] = sp[2];
                }
            } else {
                for (int j = 0; j < width_; ++j) {
                    memcpy(dst.at(i, j), at(i, width_ - j - 1), sizeof(D) * C);
                }
            }
        }
        return dst;
    }
#endif
#if defined(__SSSE3__)
    if (kX86U8C3KernelsEnabled && std::is_same<D, uint8_t>::value &&
        channels_ == 3) {
        x86::FlipHorizontalU8C3(
          reinterpret_cast<const uint8_t*>(Data()),
          reinterpret_cast<uint8_t*>(dst.Data()), width_, height_);
        return dst;
    }
#endif
#if defined(__AVX2__) || defined(__SSSE3__)
    if (std::is_same<D, uint8_t>::value && channels_ == 1) {
        const int C = 1;
        const __m128i rev16 = _mm_setr_epi8(
            15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
#if defined(__AVX2__)
        const __m256i rev256 = _mm256_insertf128_si256(
            _mm256_castsi128_si256(rev16), rev16, 1);
#endif
        for (int i = 0; i < height_; ++i) {
            const uint8_t* src_row = reinterpret_cast<const uint8_t*>(Row(i));
            uint8_t* dst_row = reinterpret_cast<uint8_t*>(dst.Row(i));
            int x = 0;
#if defined(__AVX2__)
            // 32-byte blocks: swap 128-bit halves then reverse bytes within lanes
            for (; x + 32 <= width_; x += 32) {
                const uint8_t* s = src_row + (width_ - x - 32) * C;
                __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s));
                __m256i swapped = _mm256_permute2x128_si256(v, v, 0x01);
                __m256i r = _mm256_shuffle_epi8(swapped, rev256);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst_row + x * C), r);
            }
#endif
#if defined(__SSSE3__)
            // 16-byte blocks with pshufb reverse
            for (; x + 16 <= width_; x += 16) {
                const uint8_t* s = src_row + (width_ - x - 16) * C;
                __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s));
                __m128i r = _mm_shuffle_epi8(v, rev16);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(dst_row + x * C), r);
            }
#endif
            for (; x < width_; ++x) {
                dst_row[x] = src_row[width_ - 1 - x];
            }
        }
        return dst;
    }
#endif
    for (int i = 0; i < height_; ++i) {
        for (int j = 0; j < width_; ++j) {
            memcpy(dst.at(i, j), at(i, width_ - j - 1), sizeof(D) * channels_);
        }
    }
    return dst;
}

template <typename D>
Bitmap<D> Bitmap<D>::FlipUpDown() const {
    Bitmap<D> dst;
    dst.Reset(width_, height_, channels_);
    for (int i = 0; i < height_; ++i) {
        memcpy(dst.Row(i), Row(height_ - i - 1), sizeof(D) * width_ * channels_);
    }
    return dst;
}

template <typename D>
Bitmap<D> Bitmap<D>::FlipChannels() const {
    Bitmap<D> dst;
    dst.Reset(width_, height_, channels_);
    auto src_iter = Data();
    auto dst_iter = dst.Data();
    for (int i = 0; i < height_ * width_; ++i) {
        for (int c = 0; c < channels_; ++c) {
            *(dst_iter++) = src_iter[channels_ - c - 1];
        }
        src_iter += channels_;
    }
    return dst;
}

template <typename D>
Bitmap<D> Bitmap<D>::Rotate180() const {
    // Fast path: 180° rotation == vertical flip + horizontal flip
    // Both flips already have optimized implementations (NEON/SIMD).
    return this->FlipUpDown().FlipLeftRight();
}

template <typename D>
Bitmap<D> Bitmap<D>::Rotate90() const {
#if (defined(__ARM_NEON) || defined(__ARM_NEON__))
    // NEON fast paths
    if (channels_ == 1) {
        // u8 1ch: 8x8 tile
        if (std::is_same<D, uint8_t>::value) {
            Bitmap<D> dst;
            dst.Reset(height_, width_, channels_);
            const int Hd = dst.height_;
            const int Wd = dst.width_;
            const int Hs = height_;
            auto transpose8x8 = [](uint8x8_t a0, uint8x8_t a1, uint8x8_t a2, uint8x8_t a3,
                                   uint8x8_t a4, uint8x8_t a5, uint8x8_t a6, uint8x8_t a7,
                                   uint8x8_t &o0, uint8x8_t &o1, uint8x8_t &o2, uint8x8_t &o3,
                                   uint8x8_t &o4, uint8x8_t &o5, uint8x8_t &o6, uint8x8_t &o7) {
                uint8x8x2_t b0 = vtrn_u8(a0, a1);
                uint8x8x2_t b1 = vtrn_u8(a2, a3);
                uint8x8x2_t b2 = vtrn_u8(a4, a5);
                uint8x8x2_t b3 = vtrn_u8(a6, a7);
                uint16x4x2_t c0 = vtrn_u16(vreinterpret_u16_u8(b0.val[0]), vreinterpret_u16_u8(b1.val[0]));
                uint16x4x2_t c1 = vtrn_u16(vreinterpret_u16_u8(b0.val[1]), vreinterpret_u16_u8(b1.val[1]));
                uint16x4x2_t c2 = vtrn_u16(vreinterpret_u16_u8(b2.val[0]), vreinterpret_u16_u8(b3.val[0]));
                uint16x4x2_t c3 = vtrn_u16(vreinterpret_u16_u8(b2.val[1]), vreinterpret_u16_u8(b3.val[1]));
                uint32x2x2_t d0 = vtrn_u32(vreinterpret_u32_u16(c0.val[0]), vreinterpret_u32_u16(c2.val[0]));
                uint32x2x2_t d1 = vtrn_u32(vreinterpret_u32_u16(c1.val[0]), vreinterpret_u32_u16(c3.val[0]));
                uint32x2x2_t d2 = vtrn_u32(vreinterpret_u32_u16(c0.val[1]), vreinterpret_u32_u16(c2.val[1]));
                uint32x2x2_t d3 = vtrn_u32(vreinterpret_u32_u16(c1.val[1]), vreinterpret_u32_u16(c3.val[1]));
                o0 = vreinterpret_u8_u32(d0.val[0]);
                o1 = vreinterpret_u8_u32(d1.val[0]);
                o2 = vreinterpret_u8_u32(d2.val[0]);
                o3 = vreinterpret_u8_u32(d3.val[0]);
                o4 = vreinterpret_u8_u32(d0.val[1]);
                o5 = vreinterpret_u8_u32(d1.val[1]);
                o6 = vreinterpret_u8_u32(d2.val[1]);
                o7 = vreinterpret_u8_u32(d3.val[1]);
            };
            int i = 0;
            for (; i + 7 < Hd; i += 8) {
                int j = 0;
                for (; j + 7 < Wd; j += 8) {
                    const int r0 = Hs - 1 - (j + 0);
                    const int r1 = Hs - 1 - (j + 1);
                    const int r2 = Hs - 1 - (j + 2);
                    const int r3 = Hs - 1 - (j + 3);
                    const int r4 = Hs - 1 - (j + 4);
                    const int r5 = Hs - 1 - (j + 5);
                    const int r6 = Hs - 1 - (j + 6);
                    const int r7 = Hs - 1 - (j + 7);
                    const int c0 = i;
                    uint8x8_t a0 = vld1_u8(reinterpret_cast<const uint8_t*>(at(r0, c0)));
                    uint8x8_t a1 = vld1_u8(reinterpret_cast<const uint8_t*>(at(r1, c0)));
                    uint8x8_t a2 = vld1_u8(reinterpret_cast<const uint8_t*>(at(r2, c0)));
                    uint8x8_t a3 = vld1_u8(reinterpret_cast<const uint8_t*>(at(r3, c0)));
                    uint8x8_t a4 = vld1_u8(reinterpret_cast<const uint8_t*>(at(r4, c0)));
                    uint8x8_t a5 = vld1_u8(reinterpret_cast<const uint8_t*>(at(r5, c0)));
                    uint8x8_t a6 = vld1_u8(reinterpret_cast<const uint8_t*>(at(r6, c0)));
                    uint8x8_t a7 = vld1_u8(reinterpret_cast<const uint8_t*>(at(r7, c0)));
                    uint8x8_t o0, o1, o2, o3, o4, o5, o6, o7;
                    transpose8x8(a0, a1, a2, a3, a4, a5, a6, a7, o0, o1, o2, o3, o4, o5, o6, o7);
                    vst1_u8(reinterpret_cast<uint8_t*>(dst.at(i + 0, j)), o0);
                    vst1_u8(reinterpret_cast<uint8_t*>(dst.at(i + 1, j)), o1);
                    vst1_u8(reinterpret_cast<uint8_t*>(dst.at(i + 2, j)), o2);
                    vst1_u8(reinterpret_cast<uint8_t*>(dst.at(i + 3, j)), o3);
                    vst1_u8(reinterpret_cast<uint8_t*>(dst.at(i + 4, j)), o4);
                    vst1_u8(reinterpret_cast<uint8_t*>(dst.at(i + 5, j)), o5);
                    vst1_u8(reinterpret_cast<uint8_t*>(dst.at(i + 6, j)), o6);
                    vst1_u8(reinterpret_cast<uint8_t*>(dst.at(i + 7, j)), o7);
                }
            }
            // tails (scalar)
            for (; i < Hd; ++i) {
                for (int j = 0; j < Wd; ++j) {
                    *reinterpret_cast<uint8_t*>(dst.at(i, j)) =
                        *reinterpret_cast<const uint8_t*>(at(height_ - 1 - j, i));
                }
            }
            return dst;
        }
        // f32 1ch: 4x4 tile
        if (std::is_same<D, float>::value) {
            Bitmap<D> dst;
            dst.Reset(height_, width_, channels_);
            const int Hd = dst.height_;
            const int Wd = dst.width_;
            const int Hs = height_;
            int i = 0;
            for (; i + 3 < Hd; i += 4) {
                int j = 0;
                for (; j + 3 < Wd; j += 4) {
                    const int r0 = Hs - 1 - (j + 0);
                    const int r1 = Hs - 1 - (j + 1);
                    const int r2 = Hs - 1 - (j + 2);
                    const int r3 = Hs - 1 - (j + 3);
                    const int c0 = i;
                    float32x4_t a0 = vld1q_f32(reinterpret_cast<const float*>(at(r0, c0)));
                    float32x4_t a1 = vld1q_f32(reinterpret_cast<const float*>(at(r1, c0)));
                    float32x4_t a2 = vld1q_f32(reinterpret_cast<const float*>(at(r2, c0)));
                    float32x4_t a3 = vld1q_f32(reinterpret_cast<const float*>(at(r3, c0)));
                    float32x4x2_t t0 = vtrnq_f32(a0, a1);
                    float32x4x2_t t1 = vtrnq_f32(a2, a3);
                    float32x4_t s0 = vcombine_f32(vget_low_f32(t0.val[0]), vget_low_f32(t1.val[0]));
                    float32x4_t s1 = vcombine_f32(vget_high_f32(t0.val[0]), vget_high_f32(t1.val[0]));
                    float32x4_t s2 = vcombine_f32(vget_low_f32(t0.val[1]), vget_low_f32(t1.val[1]));
                    float32x4_t s3 = vcombine_f32(vget_high_f32(t0.val[1]), vget_high_f32(t1.val[1]));
                    vst1q_f32(reinterpret_cast<float*>(dst.at(i + 0, j)), s0);
                    vst1q_f32(reinterpret_cast<float*>(dst.at(i + 1, j)), s1);
                    vst1q_f32(reinterpret_cast<float*>(dst.at(i + 2, j)), s2);
                    vst1q_f32(reinterpret_cast<float*>(dst.at(i + 3, j)), s3);
                }
                // tail columns for the processed 4 rows
                for (; j < Wd; ++j) {
                    const int r = Hs - 1 - j;
                    float* d0 = reinterpret_cast<float*>(dst.at(i + 0, j));
                    float* d1 = reinterpret_cast<float*>(dst.at(i + 1, j));
                    float* d2 = reinterpret_cast<float*>(dst.at(i + 2, j));
                    float* d3 = reinterpret_cast<float*>(dst.at(i + 3, j));
                    d0[0] = *reinterpret_cast<const float*>(at(r, i + 0));
                    d1[0] = *reinterpret_cast<const float*>(at(r, i + 1));
                    d2[0] = *reinterpret_cast<const float*>(at(r, i + 2));
                    d3[0] = *reinterpret_cast<const float*>(at(r, i + 3));
                }
            }
            for (; i < Hd; ++i) {
                for (int j = 0; j < Wd; ++j) {
                    *reinterpret_cast<float*>(dst.at(i, j)) =
                        *reinterpret_cast<const float*>(at(height_ - 1 - j, i));
                }
            }
            return dst;
        }
    } else if (channels_ == 3) {
        // u8 3ch: 8x8 tile de/rec-interleave
        if (std::is_same<D, uint8_t>::value) {
            Bitmap<D> dst;
            dst.Reset(height_, width_, channels_);
            const int Hd = dst.height_;
            const int Wd = dst.width_;
            const int Hs = height_;
            auto transpose8x8 = [](uint8x8_t a0, uint8x8_t a1, uint8x8_t a2, uint8x8_t a3,
                                   uint8x8_t a4, uint8x8_t a5, uint8x8_t a6, uint8x8_t a7,
                                   uint8x8_t &o0, uint8x8_t &o1, uint8x8_t &o2, uint8x8_t &o3,
                                   uint8x8_t &o4, uint8x8_t &o5, uint8x8_t &o6, uint8x8_t &o7) {
                uint8x8x2_t b0 = vtrn_u8(a0, a1);
                uint8x8x2_t b1 = vtrn_u8(a2, a3);
                uint8x8x2_t b2 = vtrn_u8(a4, a5);
                uint8x8x2_t b3 = vtrn_u8(a6, a7);
                uint16x4x2_t c0 = vtrn_u16(vreinterpret_u16_u8(b0.val[0]), vreinterpret_u16_u8(b1.val[0]));
                uint16x4x2_t c1 = vtrn_u16(vreinterpret_u16_u8(b0.val[1]), vreinterpret_u16_u8(b1.val[1]));
                uint16x4x2_t c2 = vtrn_u16(vreinterpret_u16_u8(b2.val[0]), vreinterpret_u16_u8(b3.val[0]));
                uint16x4x2_t c3 = vtrn_u16(vreinterpret_u16_u8(b2.val[1]), vreinterpret_u16_u8(b3.val[1]));
                uint32x2x2_t d0 = vtrn_u32(vreinterpret_u32_u16(c0.val[0]), vreinterpret_u32_u16(c2.val[0]));
                uint32x2x2_t d1 = vtrn_u32(vreinterpret_u32_u16(c1.val[0]), vreinterpret_u32_u16(c3.val[0]));
                uint32x2x2_t d2 = vtrn_u32(vreinterpret_u32_u16(c0.val[1]), vreinterpret_u32_u16(c2.val[1]));
                uint32x2x2_t d3 = vtrn_u32(vreinterpret_u32_u16(c1.val[1]), vreinterpret_u32_u16(c3.val[1]));
                o0 = vreinterpret_u8_u32(d0.val[0]);
                o1 = vreinterpret_u8_u32(d1.val[0]);
                o2 = vreinterpret_u8_u32(d2.val[0]);
                o3 = vreinterpret_u8_u32(d3.val[0]);
                o4 = vreinterpret_u8_u32(d0.val[1]);
                o5 = vreinterpret_u8_u32(d1.val[1]);
                o6 = vreinterpret_u8_u32(d2.val[1]);
                o7 = vreinterpret_u8_u32(d3.val[1]);
            };
            int i = 0;
            for (; i + 7 < Hd; i += 8) {
                int j = 0;
                for (; j + 7 < Wd; j += 8) {
                    const int r0 = Hs - 1 - (j + 0);
                    const int r1 = Hs - 1 - (j + 1);
                    const int r2 = Hs - 1 - (j + 2);
                    const int r3 = Hs - 1 - (j + 3);
                    const int r4 = Hs - 1 - (j + 4);
                    const int r5 = Hs - 1 - (j + 5);
                    const int r6 = Hs - 1 - (j + 6);
                    const int r7 = Hs - 1 - (j + 7);
                    const int c0 = i;
                    uint8x8x3_t p0 = vld3_u8(reinterpret_cast<const uint8_t*>(at(r0, c0)));
                    uint8x8x3_t p1 = vld3_u8(reinterpret_cast<const uint8_t*>(at(r1, c0)));
                    uint8x8x3_t p2 = vld3_u8(reinterpret_cast<const uint8_t*>(at(r2, c0)));
                    uint8x8x3_t p3 = vld3_u8(reinterpret_cast<const uint8_t*>(at(r3, c0)));
                    uint8x8x3_t p4 = vld3_u8(reinterpret_cast<const uint8_t*>(at(r4, c0)));
                    uint8x8x3_t p5 = vld3_u8(reinterpret_cast<const uint8_t*>(at(r5, c0)));
                    uint8x8x3_t p6 = vld3_u8(reinterpret_cast<const uint8_t*>(at(r6, c0)));
                    uint8x8x3_t p7 = vld3_u8(reinterpret_cast<const uint8_t*>(at(r7, c0)));
                    uint8x8_t b0 = p0.val[0], g0 = p0.val[1], r0c = p0.val[2];
                    uint8x8_t b1 = p1.val[0], g1 = p1.val[1], r1c = p1.val[2];
                    uint8x8_t b2 = p2.val[0], g2 = p2.val[1], r2c = p2.val[2];
                    uint8x8_t b3 = p3.val[0], g3 = p3.val[1], r3c = p3.val[2];
                    uint8x8_t b4 = p4.val[0], g4 = p4.val[1], r4c = p4.val[2];
                    uint8x8_t b5 = p5.val[0], g5 = p5.val[1], r5c = p5.val[2];
                    uint8x8_t b6 = p6.val[0], g6 = p6.val[1], r6c = p6.val[2];
                    uint8x8_t b7 = p7.val[0], g7 = p7.val[1], r7c = p7.val[2];
                    uint8x8_t tb0, tb1, tb2, tb3, tb4, tb5, tb6, tb7;
                    uint8x8_t tg0, tg1, tg2, tg3, tg4, tg5, tg6, tg7;
                    uint8x8_t tr0, tr1, tr2, tr3, tr4, tr5, tr6, tr7;
                    transpose8x8(b0, b1, b2, b3, b4, b5, b6, b7, tb0, tb1, tb2, tb3, tb4, tb5, tb6, tb7);
                    transpose8x8(g0, g1, g2, g3, g4, g5, g6, g7, tg0, tg1, tg2, tg3, tg4, tg5, tg6, tg7);
                    transpose8x8(r0c, r1c, r2c, r3c, r4c, r5c, r6c, r7c, tr0, tr1, tr2, tr3, tr4, tr5, tr6, tr7);
                    uint8x8x3_t q;
                    q.val[0] = tb0; q.val[1] = tg0; q.val[2] = tr0; vst3_u8(reinterpret_cast<uint8_t*>(dst.at(i + 0, j)), q);
                    q.val[0] = tb1; q.val[1] = tg1; q.val[2] = tr1; vst3_u8(reinterpret_cast<uint8_t*>(dst.at(i + 1, j)), q);
                    q.val[0] = tb2; q.val[1] = tg2; q.val[2] = tr2; vst3_u8(reinterpret_cast<uint8_t*>(dst.at(i + 2, j)), q);
                    q.val[0] = tb3; q.val[1] = tg3; q.val[2] = tr3; vst3_u8(reinterpret_cast<uint8_t*>(dst.at(i + 3, j)), q);
                    q.val[0] = tb4; q.val[1] = tg4; q.val[2] = tr4; vst3_u8(reinterpret_cast<uint8_t*>(dst.at(i + 4, j)), q);
                    q.val[0] = tb5; q.val[1] = tg5; q.val[2] = tr5; vst3_u8(reinterpret_cast<uint8_t*>(dst.at(i + 5, j)), q);
                    q.val[0] = tb6; q.val[1] = tg6; q.val[2] = tr6; vst3_u8(reinterpret_cast<uint8_t*>(dst.at(i + 6, j)), q);
                    q.val[0] = tb7; q.val[1] = tg7; q.val[2] = tr7; vst3_u8(reinterpret_cast<uint8_t*>(dst.at(i + 7, j)), q);
                }
            }
            // tails
            for (int i2 = 0; i2 < Hd; ++i2) {
                for (int j2 = (Wd & ~7); j2 < Wd; ++j2) {
                    memcpy(dst.at(i2, j2), at(height_ - 1 - j2, i2), sizeof(D) * 3);
                }
            }
            for (int i2 = (Hd & ~7); i2 < Hd; ++i2) {
                for (int j2 = 0; j2 < Wd; ++j2) {
                    memcpy(dst.at(i2, j2), at(height_ - 1 - j2, i2), sizeof(D) * 3);
                }
            }
            return dst;
        }
        // f32 3ch: fallback to scalar for correctness (avoid tile artifacts)
        if (std::is_same<D, float>::value) {
            Bitmap<D> dst;
            dst.Reset(height_, width_, channels_);
            for (int i = 0; i < dst.height_; ++i) {
                for (int j = 0; j < dst.width_; ++j) {
                    memcpy(dst.at(i, j), at(height_ - 1 - j, i), sizeof(D) * 3);
                }
            }
            return dst;
        }
    }
#endif
#if defined(__SSE2__)
    // x86 fast path: float single-channel, 4x4 tile transpose + index mapping
    if (std::is_same<D, float>::value && channels_ == 1) {
        Bitmap<D> dst;
        dst.Reset(height_, width_, channels_);
        const int Hs = height_;
        const int Ws = width_;
        const int Hd = dst.height_;  // == Ws
        const int Wd = dst.width_;   // == Hs
        int i = 0;
        for (; i + 3 < Hd; i += 4) {
            int j = 0;
            for (; j + 3 < Wd; j += 4) {
                // src rows (descending by j), src cols (ascending by i..i+3)
                const int r0 = Hs - 1 - (j + 0);
                const int r1 = Hs - 1 - (j + 1);
                const int r2 = Hs - 1 - (j + 2);
                const int r3 = Hs - 1 - (j + 3);
                const int c0 = i + 0;
                const int c1 = i + 1;
                const int c2 = i + 2;
                const int c3 = i + 3;
                __m128 s0 = _mm_loadu_ps(reinterpret_cast<const float*>(at(r0, c0)));
                __m128 s1 = _mm_loadu_ps(reinterpret_cast<const float*>(at(r1, c0)));
                __m128 s2 = _mm_loadu_ps(reinterpret_cast<const float*>(at(r2, c0)));
                __m128 s3 = _mm_loadu_ps(reinterpret_cast<const float*>(at(r3, c0)));
                _MM_TRANSPOSE4_PS(s0, s1, s2, s3);
                _mm_storeu_ps(reinterpret_cast<float*>(dst.at(i + 0, j)), s0);
                _mm_storeu_ps(reinterpret_cast<float*>(dst.at(i + 1, j)), s1);
                _mm_storeu_ps(reinterpret_cast<float*>(dst.at(i + 2, j)), s2);
                _mm_storeu_ps(reinterpret_cast<float*>(dst.at(i + 3, j)), s3);
            }
            // tail columns (Wd % 4)
            for (; j < Wd; ++j) {
                const int r = Hs - 1 - j;
                float* d0 = reinterpret_cast<float*>(dst.at(i + 0, j));
                float* d1 = reinterpret_cast<float*>(dst.at(i + 1, j));
                float* d2 = reinterpret_cast<float*>(dst.at(i + 2, j));
                float* d3 = reinterpret_cast<float*>(dst.at(i + 3, j));
                d0[0] = *reinterpret_cast<const float*>(at(r, i + 0));
                d1[0] = *reinterpret_cast<const float*>(at(r, i + 1));
                d2[0] = *reinterpret_cast<const float*>(at(r, i + 2));
                d3[0] = *reinterpret_cast<const float*>(at(r, i + 3));
            }
        }
        // tail rows (Hd % 4)
        for (; i < Hd; ++i) {
            for (int j = 0; j < Wd; ++j) {
                *reinterpret_cast<float*>(dst.at(i, j)) =
                    *reinterpret_cast<const float*>(at(Hs - 1 - j, i));
            }
        }
        return dst;
    }
#endif
#if defined(__SSSE3__)
    if (kX86U8C3KernelsEnabled && channels_ == 3 &&
        std::is_same<D, uint8_t>::value) {
        Bitmap<D> dst;
        dst.Reset(height_, width_, channels_);
        x86::Rotate90U8C3(
          reinterpret_cast<const uint8_t*>(Data()),
          reinterpret_cast<uint8_t*>(dst.Data()), width_, height_);
        return dst;
    }
#endif
    // Fallback scalar
    Bitmap<D> dst;
    dst.Reset(height_, width_, channels_);
    for (int i = 0; i < dst.height_; ++i) {
        for (int j = 0; j < dst.width_; ++j) {
            memcpy(dst.at(i, j), at(height_ - j - 1, i), sizeof(D) * channels_);
        }
    }
    return dst;
}

template <typename D>
Bitmap<D> Bitmap<D>::Rotate270() const {
#if (defined(__ARM_NEON) || defined(__ARM_NEON__))
    // NEON fast paths
    if (channels_ == 1) {
        // u8 1ch: 8x8 tile
        if (std::is_same<D, uint8_t>::value) {
            Bitmap<D> dst;
            dst.Reset(height_, width_, channels_);
            const int Hd = dst.height_;
            const int Wd = dst.width_;
            const int Ws = width_;
            auto transpose8x8 = [](uint8x8_t a0, uint8x8_t a1, uint8x8_t a2, uint8x8_t a3,
                                   uint8x8_t a4, uint8x8_t a5, uint8x8_t a6, uint8x8_t a7,
                                   uint8x8_t &o0, uint8x8_t &o1, uint8x8_t &o2, uint8x8_t &o3,
                                   uint8x8_t &o4, uint8x8_t &o5, uint8x8_t &o6, uint8x8_t &o7) {
                uint8x8x2_t b0 = vtrn_u8(a0, a1);
                uint8x8x2_t b1 = vtrn_u8(a2, a3);
                uint8x8x2_t b2 = vtrn_u8(a4, a5);
                uint8x8x2_t b3 = vtrn_u8(a6, a7);
                uint16x4x2_t c0 = vtrn_u16(vreinterpret_u16_u8(b0.val[0]), vreinterpret_u16_u8(b1.val[0]));
                uint16x4x2_t c1 = vtrn_u16(vreinterpret_u16_u8(b0.val[1]), vreinterpret_u16_u8(b1.val[1]));
                uint16x4x2_t c2 = vtrn_u16(vreinterpret_u16_u8(b2.val[0]), vreinterpret_u16_u8(b3.val[0]));
                uint16x4x2_t c3 = vtrn_u16(vreinterpret_u16_u8(b2.val[1]), vreinterpret_u16_u8(b3.val[1]));
                uint32x2x2_t d0 = vtrn_u32(vreinterpret_u32_u16(c0.val[0]), vreinterpret_u32_u16(c2.val[0]));
                uint32x2x2_t d1 = vtrn_u32(vreinterpret_u32_u16(c1.val[0]), vreinterpret_u32_u16(c3.val[0]));
                uint32x2x2_t d2 = vtrn_u32(vreinterpret_u32_u16(c0.val[1]), vreinterpret_u32_u16(c2.val[1]));
                uint32x2x2_t d3 = vtrn_u32(vreinterpret_u32_u16(c1.val[1]), vreinterpret_u32_u16(c3.val[1]));
                o0 = vreinterpret_u8_u32(d0.val[0]);
                o1 = vreinterpret_u8_u32(d1.val[0]);
                o2 = vreinterpret_u8_u32(d2.val[0]);
                o3 = vreinterpret_u8_u32(d3.val[0]);
                o4 = vreinterpret_u8_u32(d0.val[1]);
                o5 = vreinterpret_u8_u32(d1.val[1]);
                o6 = vreinterpret_u8_u32(d2.val[1]);
                o7 = vreinterpret_u8_u32(d3.val[1]);
            };
            int i = 0;
            for (; i + 7 < Hd; i += 8) {
                int j = 0;
                for (; j + 7 < Wd; j += 8) {
                    const int r0 = j + 0;
                    const int r1 = j + 1;
                    const int r2 = j + 2;
                    const int r3 = j + 3;
                    const int r4 = j + 4;
                    const int r5 = j + 5;
                    const int r6 = j + 6;
                    const int r7 = j + 7;
                    const int c0 = Ws - 8 - i;
                    uint8x8_t a0 = vld1_u8(reinterpret_cast<const uint8_t*>(at(r0, c0)));
                    uint8x8_t a1 = vld1_u8(reinterpret_cast<const uint8_t*>(at(r1, c0)));
                    uint8x8_t a2 = vld1_u8(reinterpret_cast<const uint8_t*>(at(r2, c0)));
                    uint8x8_t a3 = vld1_u8(reinterpret_cast<const uint8_t*>(at(r3, c0)));
                    uint8x8_t a4 = vld1_u8(reinterpret_cast<const uint8_t*>(at(r4, c0)));
                    uint8x8_t a5 = vld1_u8(reinterpret_cast<const uint8_t*>(at(r5, c0)));
                    uint8x8_t a6 = vld1_u8(reinterpret_cast<const uint8_t*>(at(r6, c0)));
                    uint8x8_t a7 = vld1_u8(reinterpret_cast<const uint8_t*>(at(r7, c0)));
                    uint8x8_t o0, o1, o2, o3, o4, o5, o6, o7;
                    transpose8x8(a0, a1, a2, a3, a4, a5, a6, a7, o0, o1, o2, o3, o4, o5, o6, o7);
                    // Store rows using reversed column index (o7..o0) without lane reversal.
                    vst1_u8(reinterpret_cast<uint8_t*>(dst.at(i + 0, j)), o7);
                    vst1_u8(reinterpret_cast<uint8_t*>(dst.at(i + 1, j)), o6);
                    vst1_u8(reinterpret_cast<uint8_t*>(dst.at(i + 2, j)), o5);
                    vst1_u8(reinterpret_cast<uint8_t*>(dst.at(i + 3, j)), o4);
                    vst1_u8(reinterpret_cast<uint8_t*>(dst.at(i + 4, j)), o3);
                    vst1_u8(reinterpret_cast<uint8_t*>(dst.at(i + 5, j)), o2);
                    vst1_u8(reinterpret_cast<uint8_t*>(dst.at(i + 6, j)), o1);
                    vst1_u8(reinterpret_cast<uint8_t*>(dst.at(i + 7, j)), o0);
                }
            }
            // tails
            const int i_aligned = Hd & ~7;
            const int j_aligned = Wd & ~7;
            // tail columns for processed rows
            for (int i2 = 0; i2 < i_aligned; ++i2) {
                for (int j2 = j_aligned; j2 < Wd; ++j2) {
                    *reinterpret_cast<uint8_t*>(dst.at(i2, j2)) =
                        *reinterpret_cast<const uint8_t*>(at(j2, width_ - 1 - i2));
                }
            }
            // remaining rows (all columns)
            for (int i2 = i_aligned; i2 < Hd; ++i2) {
                for (int j2 = 0; j2 < Wd; ++j2) {
                    *reinterpret_cast<uint8_t*>(dst.at(i2, j2)) =
                        *reinterpret_cast<const uint8_t*>(at(j2, width_ - 1 - i2));
                }
            }
            return dst;
        }
        // f32 1ch: 4x4 tile
        if (std::is_same<D, float>::value) {
            Bitmap<D> dst;
            dst.Reset(height_, width_, channels_);
            const int Hd = dst.height_;
            const int Wd = dst.width_;
            const int Ws = width_;
            int i = 0;
            for (; i + 3 < Hd; i += 4) {
                int j = 0;
                for (; j + 3 < Wd; j += 4) {
                    const int r0 = j + 0;
                    const int r1 = j + 1;
                    const int r2 = j + 2;
                    const int r3 = j + 3;
                    const int c0 = Ws - 4 - i;
                    float32x4_t a0 = vld1q_f32(reinterpret_cast<const float*>(at(r0, c0)));
                    float32x4_t a1 = vld1q_f32(reinterpret_cast<const float*>(at(r1, c0)));
                    float32x4_t a2 = vld1q_f32(reinterpret_cast<const float*>(at(r2, c0)));
                    float32x4_t a3 = vld1q_f32(reinterpret_cast<const float*>(at(r3, c0)));
                    float32x4x2_t t0 = vtrnq_f32(a0, a1);
                    float32x4x2_t t1 = vtrnq_f32(a2, a3);
                    float32x4_t s0 = vcombine_f32(vget_low_f32(t0.val[0]), vget_low_f32(t1.val[0]));
                    float32x4_t s1 = vcombine_f32(vget_high_f32(t0.val[0]), vget_high_f32(t1.val[0]));
                    float32x4_t s2 = vcombine_f32(vget_low_f32(t0.val[1]), vget_low_f32(t1.val[1]));
                    float32x4_t s3 = vcombine_f32(vget_high_f32(t0.val[1]), vget_high_f32(t1.val[1]));
                    // Store rows using reversed column index (s3..s0) without lane reversal.
                    vst1q_f32(reinterpret_cast<float*>(dst.at(i + 0, j)), s3);
                    vst1q_f32(reinterpret_cast<float*>(dst.at(i + 1, j)), s2);
                    vst1q_f32(reinterpret_cast<float*>(dst.at(i + 2, j)), s1);
                    vst1q_f32(reinterpret_cast<float*>(dst.at(i + 3, j)), s0);
                }
            }
            // tails
            for (int i2 = 0; i2 < Hd; ++i2) {
                for (int j2 = (Wd & ~3); j2 < Wd; ++j2) {
                    *reinterpret_cast<float*>(dst.at(i2, j2)) =
                        *reinterpret_cast<const float*>(at(j2, width_ - 1 - i2));
                }
            }
            for (int i2 = (Hd & ~3); i2 < Hd; ++i2) {
                for (int j2 = 0; j2 < Wd; ++j2) {
                    *reinterpret_cast<float*>(dst.at(i2, j2)) =
                        *reinterpret_cast<const float*>(at(j2, width_ - 1 - i2));
                }
            }
            return dst;
        }
    } else if (channels_ == 3) {
        // u8 3ch: 8x8 tile de/rec-interleave
        if (std::is_same<D, uint8_t>::value) {
            Bitmap<D> dst;
            dst.Reset(height_, width_, channels_);
            const int Hd = dst.height_;
            const int Wd = dst.width_;
            const int Ws = width_;
            auto transpose8x8 = [](uint8x8_t a0, uint8x8_t a1, uint8x8_t a2, uint8x8_t a3,
                                   uint8x8_t a4, uint8x8_t a5, uint8x8_t a6, uint8x8_t a7,
                                   uint8x8_t &o0, uint8x8_t &o1, uint8x8_t &o2, uint8x8_t &o3,
                                   uint8x8_t &o4, uint8x8_t &o5, uint8x8_t &o6, uint8x8_t &o7) {
                uint8x8x2_t b0 = vtrn_u8(a0, a1);
                uint8x8x2_t b1 = vtrn_u8(a2, a3);
                uint8x8x2_t b2 = vtrn_u8(a4, a5);
                uint8x8x2_t b3 = vtrn_u8(a6, a7);
                uint16x4x2_t c0 = vtrn_u16(vreinterpret_u16_u8(b0.val[0]), vreinterpret_u16_u8(b1.val[0]));
                uint16x4x2_t c1 = vtrn_u16(vreinterpret_u16_u8(b0.val[1]), vreinterpret_u16_u8(b1.val[1]));
                uint16x4x2_t c2 = vtrn_u16(vreinterpret_u16_u8(b2.val[0]), vreinterpret_u16_u8(b3.val[0]));
                uint16x4x2_t c3 = vtrn_u16(vreinterpret_u16_u8(b2.val[1]), vreinterpret_u16_u8(b3.val[1]));
                uint32x2x2_t d0 = vtrn_u32(vreinterpret_u32_u16(c0.val[0]), vreinterpret_u32_u16(c2.val[0]));
                uint32x2x2_t d1 = vtrn_u32(vreinterpret_u32_u16(c1.val[0]), vreinterpret_u32_u16(c3.val[0]));
                uint32x2x2_t d2 = vtrn_u32(vreinterpret_u32_u16(c0.val[1]), vreinterpret_u32_u16(c2.val[1]));
                uint32x2x2_t d3 = vtrn_u32(vreinterpret_u32_u16(c1.val[1]), vreinterpret_u32_u16(c3.val[1]));
                o0 = vreinterpret_u8_u32(d0.val[0]);
                o1 = vreinterpret_u8_u32(d1.val[0]);
                o2 = vreinterpret_u8_u32(d2.val[0]);
                o3 = vreinterpret_u8_u32(d3.val[0]);
                o4 = vreinterpret_u8_u32(d0.val[1]);
                o5 = vreinterpret_u8_u32(d1.val[1]);
                o6 = vreinterpret_u8_u32(d2.val[1]);
                o7 = vreinterpret_u8_u32(d3.val[1]);
            };
            int i = 0;
            for (; i + 7 < Hd; i += 8) {
                int j = 0;
                for (; j + 7 < Wd; j += 8) {
                    const int r0 = j + 0;
                    const int r1 = j + 1;
                    const int r2 = j + 2;
                    const int r3 = j + 3;
                    const int r4 = j + 4;
                    const int r5 = j + 5;
                    const int r6 = j + 6;
                    const int r7 = j + 7;
                    const int c0 = Ws - 8 - i;
                    uint8x8x3_t p0 = vld3_u8(reinterpret_cast<const uint8_t*>(at(r0, c0)));
                    uint8x8x3_t p1 = vld3_u8(reinterpret_cast<const uint8_t*>(at(r1, c0)));
                    uint8x8x3_t p2 = vld3_u8(reinterpret_cast<const uint8_t*>(at(r2, c0)));
                    uint8x8x3_t p3 = vld3_u8(reinterpret_cast<const uint8_t*>(at(r3, c0)));
                    uint8x8x3_t p4 = vld3_u8(reinterpret_cast<const uint8_t*>(at(r4, c0)));
                    uint8x8x3_t p5 = vld3_u8(reinterpret_cast<const uint8_t*>(at(r5, c0)));
                    uint8x8x3_t p6 = vld3_u8(reinterpret_cast<const uint8_t*>(at(r6, c0)));
                    uint8x8x3_t p7 = vld3_u8(reinterpret_cast<const uint8_t*>(at(r7, c0)));
                    uint8x8_t b0 = p0.val[0], g0 = p0.val[1], r0c = p0.val[2];
                    uint8x8_t b1 = p1.val[0], g1 = p1.val[1], r1c = p1.val[2];
                    uint8x8_t b2 = p2.val[0], g2 = p2.val[1], r2c = p2.val[2];
                    uint8x8_t b3 = p3.val[0], g3 = p3.val[1], r3c = p3.val[2];
                    uint8x8_t b4 = p4.val[0], g4 = p4.val[1], r4c = p4.val[2];
                    uint8x8_t b5 = p5.val[0], g5 = p5.val[1], r5c = p5.val[2];
                    uint8x8_t b6 = p6.val[0], g6 = p6.val[1], r6c = p6.val[2];
                    uint8x8_t b7 = p7.val[0], g7 = p7.val[1], r7c = p7.val[2];
                    uint8x8_t tb0, tb1, tb2, tb3, tb4, tb5, tb6, tb7;
                    uint8x8_t tg0, tg1, tg2, tg3, tg4, tg5, tg6, tg7;
                    uint8x8_t tr0, tr1, tr2, tr3, tr4, tr5, tr6, tr7;
                    transpose8x8(b0, b1, b2, b3, b4, b5, b6, b7, tb0, tb1, tb2, tb3, tb4, tb5, tb6, tb7);
                    transpose8x8(g0, g1, g2, g3, g4, g5, g6, g7, tg0, tg1, tg2, tg3, tg4, tg5, tg6, tg7);
                    transpose8x8(r0c, r1c, r2c, r3c, r4c, r5c, r6c, r7c, tr0, tr1, tr2, tr3, tr4, tr5, tr6, tr7);
                    uint8x8x3_t q;
                    q.val[0] = tb7; q.val[1] = tg7; q.val[2] = tr7; vst3_u8(reinterpret_cast<uint8_t*>(dst.at(i + 0, j)), q);
                    q.val[0] = tb6; q.val[1] = tg6; q.val[2] = tr6; vst3_u8(reinterpret_cast<uint8_t*>(dst.at(i + 1, j)), q);
                    q.val[0] = tb5; q.val[1] = tg5; q.val[2] = tr5; vst3_u8(reinterpret_cast<uint8_t*>(dst.at(i + 2, j)), q);
                    q.val[0] = tb4; q.val[1] = tg4; q.val[2] = tr4; vst3_u8(reinterpret_cast<uint8_t*>(dst.at(i + 3, j)), q);
                    q.val[0] = tb3; q.val[1] = tg3; q.val[2] = tr3; vst3_u8(reinterpret_cast<uint8_t*>(dst.at(i + 4, j)), q);
                    q.val[0] = tb2; q.val[1] = tg2; q.val[2] = tr2; vst3_u8(reinterpret_cast<uint8_t*>(dst.at(i + 5, j)), q);
                    q.val[0] = tb1; q.val[1] = tg1; q.val[2] = tr1; vst3_u8(reinterpret_cast<uint8_t*>(dst.at(i + 6, j)), q);
                    q.val[0] = tb0; q.val[1] = tg0; q.val[2] = tr0; vst3_u8(reinterpret_cast<uint8_t*>(dst.at(i + 7, j)), q);
                }
            }
            // tails
            const int i_aligned = Hd & ~7;
            const int j_aligned = Wd & ~7;
            // tail columns for processed rows
            for (int i2 = 0; i2 < i_aligned; ++i2) {
                for (int j2 = j_aligned; j2 < Wd; ++j2) {
                    memcpy(dst.at(i2, j2), at(j2, width_ - 1 - i2), sizeof(D) * 3);
                }
            }
            // remaining rows (all columns)
            for (int i2 = i_aligned; i2 < Hd; ++i2) {
                for (int j2 = 0; j2 < Wd; ++j2) {
                    memcpy(dst.at(i2, j2), at(j2, width_ - 1 - i2), sizeof(D) * 3);
                }
            }
            return dst;
        }
        // f32 3ch: fallback to scalar for correctness (avoid tile artifacts)
        if (std::is_same<D, float>::value) {
            Bitmap<D> dst;
            dst.Reset(height_, width_, channels_);
            for (int i = 0; i < dst.height_; ++i) {
                for (int j = 0; j < dst.width_; ++j) {
                    memcpy(dst.at(i, j), at(j, width_ - 1 - i), sizeof(D) * 3);
                }
            }
            return dst;
        }
    }
#endif
#if defined(__SSE2__)
    // x86 fast path: float single-channel, 4x4 tile transpose + index mapping (90 deg CCW)
    if (std::is_same<D, float>::value && channels_ == 1) {
        Bitmap<D> dst;
        dst.Reset(height_, width_, channels_);
        const int Hs = height_;
        const int Ws = width_;
        const int Hd = dst.height_;  // == Ws
        const int Wd = dst.width_;   // == Hs
        int i = 0;
        for (; i + 3 < Hd; i += 4) {
            int j = 0;
            for (; j + 3 < Wd; j += 4) {
                // src rows (ascending by j), src cols (descending by i)
                const int r0 = j + 0;
                const int r1 = j + 1;
                const int r2 = j + 2;
                const int r3 = j + 3;
                const int c0 = Ws - 1 - (i + 0);
                const int c1 = Ws - 1 - (i + 1);
                const int c2 = Ws - 1 - (i + 2);
                const int c3 = Ws - 1 - (i + 3);
                // load contiguous as [c3,c2,c1,c0] then transpose then reverse to [c0..c3]
                __m128 s0 = _mm_loadu_ps(reinterpret_cast<const float*>(at(r0, c3)));
                __m128 s1 = _mm_loadu_ps(reinterpret_cast<const float*>(at(r1, c3)));
                __m128 s2 = _mm_loadu_ps(reinterpret_cast<const float*>(at(r2, c3)));
                __m128 s3 = _mm_loadu_ps(reinterpret_cast<const float*>(at(r3, c3)));
                _MM_TRANSPOSE4_PS(s0, s1, s2, s3);
                const int rev = _MM_SHUFFLE(0, 1, 2, 3); // produce [3,2,1,0]
                s0 = _mm_shuffle_ps(s0, s0, rev);
                s1 = _mm_shuffle_ps(s1, s1, rev);
                s2 = _mm_shuffle_ps(s2, s2, rev);
                s3 = _mm_shuffle_ps(s3, s3, rev);
                _mm_storeu_ps(reinterpret_cast<float*>(dst.at(i + 0, j)), s0);
                _mm_storeu_ps(reinterpret_cast<float*>(dst.at(i + 1, j)), s1);
                _mm_storeu_ps(reinterpret_cast<float*>(dst.at(i + 2, j)), s2);
                _mm_storeu_ps(reinterpret_cast<float*>(dst.at(i + 3, j)), s3);
            }
            for (; j < Wd; ++j) {
                float* d0 = reinterpret_cast<float*>(dst.at(i + 0, j));
                float* d1 = reinterpret_cast<float*>(dst.at(i + 1, j));
                float* d2 = reinterpret_cast<float*>(dst.at(i + 2, j));
                float* d3 = reinterpret_cast<float*>(dst.at(i + 3, j));
                d0[0] = *reinterpret_cast<const float*>(at(j, Ws - 1 - (i + 0)));
                d1[0] = *reinterpret_cast<const float*>(at(j, Ws - 1 - (i + 1)));
                d2[0] = *reinterpret_cast<const float*>(at(j, Ws - 1 - (i + 2)));
                d3[0] = *reinterpret_cast<const float*>(at(j, Ws - 1 - (i + 3)));
            }
        }
        for (; i < Hd; ++i) {
            for (int j = 0; j < Wd; ++j) {
                *reinterpret_cast<float*>(dst.at(i, j)) =
                    *reinterpret_cast<const float*>(at(j, Ws - 1 - i));
            }
        }
        return dst;
    }
#endif
    // Fallback scalar
    Bitmap<D> dst;
    dst.Reset(height_, width_, channels_);
    for (int i = 0; i < dst.height_; ++i) {
        for (int j = 0; j < dst.width_; ++j) {
            memcpy(dst.at(i, j), at(j, width_ - i - 1), sizeof(D) * channels_);
        }
    }
    return dst;
}

template <typename D>
Bitmap<D> Bitmap<D>::RgbToGray() const {
    Bitmap<D> dst;
    INSPIRECV_CHECK_EQ(channels_, 3);
    dst.Reset(width_, height_, 1);
#if defined(__AVX2__)
    // AVX2 fast path for interleaved BGR -> Gray (u8)
    if (std::is_same<D, uint8_t>::value) {
        const uint8_t* src = reinterpret_cast<const uint8_t*>(Data());
        uint8_t* out = reinterpret_cast<uint8_t*>(dst.Data());
        const int strideIn = width_ * 3;
        const int strideOut = width_;

        // Shuffle masks for extracting B/G/R from three 16B chunks (48B = 16 pixels)
        const __m128i mB0 = _mm_setr_epi8(0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i mB1 = _mm_setr_epi8(2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i mB2 = _mm_setr_epi8(1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

        const __m128i mG0 = _mm_setr_epi8(1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i mG1 = _mm_setr_epi8(0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i mG2 = _mm_setr_epi8(2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

        const __m128i mR0 = _mm_setr_epi8(2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i mR1 = _mm_setr_epi8(1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i mR2 = _mm_setr_epi8(0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

        const __m128i wB = _mm_set1_epi16(29);
        const __m128i wG = _mm_set1_epi16(150);
        const __m128i wR = _mm_set1_epi16(77);
        const __m128i v128 = _mm_set1_epi16(128);

        for (int y = 0; y < height_; ++y) {
            const uint8_t* row = src + static_cast<size_t>(y) * strideIn;
            uint8_t* dstrow = out + static_cast<size_t>(y) * strideOut;
            int x = 0;
            for (; x + 16 <= width_; x += 16) {
                const uint8_t* p = row + x * 3;
                __m128i c0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 0));
                __m128i c1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 16));
                __m128i c2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 32));

                // Extract B, then place into contiguous 16 bytes
                __m128i b0 = _mm_shuffle_epi8(c0, mB0);
                __m128i b1 = _mm_shuffle_epi8(c1, mB1);
                __m128i b2 = _mm_shuffle_epi8(c2, mB2);
                __m128i b = _mm_or_si128(_mm_or_si128(b0, _mm_slli_si128(b1, 6)), _mm_slli_si128(b2, 11));

                // Extract G
                __m128i g0 = _mm_shuffle_epi8(c0, mG0);
                __m128i g1 = _mm_shuffle_epi8(c1, mG1);
                __m128i g2 = _mm_shuffle_epi8(c2, mG2);
                __m128i g = _mm_or_si128(_mm_or_si128(g0, _mm_slli_si128(g1, 5)), _mm_slli_si128(g2, 11));

                // Extract R
                __m128i r0 = _mm_shuffle_epi8(c0, mR0);
                __m128i r1 = _mm_shuffle_epi8(c1, mR1);
                __m128i r2 = _mm_shuffle_epi8(c2, mR2);
                __m128i r = _mm_or_si128(_mm_or_si128(r0, _mm_slli_si128(r1, 5)), _mm_slli_si128(r2, 10));

                // Low 8 pixels
                __m128i b_lo = _mm_cvtepu8_epi16(b);
                __m128i g_lo = _mm_cvtepu8_epi16(g);
                __m128i r_lo = _mm_cvtepu8_epi16(r);
                __m128i s_lo = _mm_add_epi16(_mm_add_epi16(_mm_mullo_epi16(b_lo, wB),
                                                           _mm_mullo_epi16(g_lo, wG)),
                                             _mm_mullo_epi16(r_lo, wR));
                s_lo = _mm_add_epi16(s_lo, v128);
                s_lo = _mm_srli_epi16(s_lo, 8);

                // High 8 pixels
                __m128i b_hi = _mm_cvtepu8_epi16(_mm_srli_si128(b, 8));
                __m128i g_hi = _mm_cvtepu8_epi16(_mm_srli_si128(g, 8));
                __m128i r_hi = _mm_cvtepu8_epi16(_mm_srli_si128(r, 8));
                __m128i s_hi = _mm_add_epi16(_mm_add_epi16(_mm_mullo_epi16(b_hi, wB),
                                                           _mm_mullo_epi16(g_hi, wG)),
                                             _mm_mullo_epi16(r_hi, wR));
                s_hi = _mm_add_epi16(s_hi, v128);
                s_hi = _mm_srli_epi16(s_hi, 8);

                __m128i y8 = _mm_packus_epi16(s_lo, s_hi);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(dstrow + x), y8);
            }
            // tail
            for (; x < width_; ++x) {
                const uint8_t* p = row + x * 3;
                unsigned int yy = 29u * p[0] + 150u * p[1] + 77u * p[2] + 128u;
                yy >>= 8;
                dstrow[x] = static_cast<uint8_t>(yy);
            }
        }
        return dst;
    }
#endif
#if (defined(__ARM_NEON) || defined(__ARM_NEON__))
    // NEON fast paths for interleaved BGR
    if (std::is_same<D, uint8_t>::value) {
        const uint8_t* src = reinterpret_cast<const uint8_t*>(Data());
        uint8_t* out = reinterpret_cast<uint8_t*>(dst.Data());
        const int strideIn = width_ * 3;
        const int strideOut = width_;
        const uint16x8_t kB = vdupq_n_u16(29);   // ~0.114 * 256
        const uint16x8_t kG = vdupq_n_u16(150);  // ~0.587 * 256
        const uint16x8_t kR = vdupq_n_u16(77);   // ~0.299 * 256
        const uint16x8_t k128 = vdupq_n_u16(128);
        for (int y = 0; y < height_; ++y) {
            const uint8_t* row = src + static_cast<size_t>(y) * strideIn;
            uint8_t* dstrow = out + static_cast<size_t>(y) * strideOut;
            int x = 0;
            for (; x + 16 <= width_; x += 16) {
                uint8x16x3_t bgr = vld3q_u8(row + x * 3);
                uint8x16_t b = bgr.val[0];
                uint8x16_t g = bgr.val[1];
                uint8x16_t r = bgr.val[2];
                // low 8
                uint16x8_t bl = vmovl_u8(vget_low_u8(b));
                uint16x8_t gl = vmovl_u8(vget_low_u8(g));
                uint16x8_t rl = vmovl_u8(vget_low_u8(r));
                uint16x8_t sl = vmulq_u16(bl, kB);
                sl = vmlaq_u16(sl, gl, kG);
                sl = vmlaq_u16(sl, rl, kR);
                sl = vaddq_u16(sl, k128);
                sl = vshrq_n_u16(sl, 8);
                // high 8
                uint16x8_t bh = vmovl_u8(vget_high_u8(b));
                uint16x8_t gh = vmovl_u8(vget_high_u8(g));
                uint16x8_t rh = vmovl_u8(vget_high_u8(r));
                uint16x8_t sh = vmulq_u16(bh, kB);
                sh = vmlaq_u16(sh, gh, kG);
                sh = vmlaq_u16(sh, rh, kR);
                sh = vaddq_u16(sh, k128);
                sh = vshrq_n_u16(sh, 8);
                uint8x16_t y8 = vcombine_u8(vqmovn_u16(sl), vqmovn_u16(sh));
                vst1q_u8(dstrow + x, y8);
            }
            // tail
            for (; x < width_; ++x) {
                const uint8_t* p = row + x * 3;
                unsigned int yy = 29u * p[0] + 150u * p[1] + 77u * p[2] + 128u;
                yy >>= 8;
                dstrow[x] = static_cast<uint8_t>(yy);
            }
        }
        return dst;
    }
    if (std::is_same<D, float>::value) {
        const float32x4_t cB = vdupq_n_f32(0.114f);
        const float32x4_t cG = vdupq_n_f32(0.587f);
        const float32x4_t cR = vdupq_n_f32(0.299f);
        const float* src = reinterpret_cast<const float*>(Data());
        float* out = reinterpret_cast<float*>(dst.Data());
        const int strideIn = width_ * 3;
        const int strideOut = width_;
        for (int y = 0; y < height_; ++y) {
            const float* row = src + static_cast<size_t>(y) * strideIn;
            float* dstrow = out + static_cast<size_t>(y) * strideOut;
            int x = 0;
            for (; x + 4 <= width_; x += 4) {
                float32x4x3_t bgr = vld3q_f32(row + x * 3);
                float32x4_t acc = vmulq_f32(bgr.val[0], cB);
                acc = vmlaq_f32(acc, bgr.val[1], cG);
                acc = vmlaq_f32(acc, bgr.val[2], cR);
                vst1q_f32(dstrow + x, acc);
            }
            for (; x < width_; ++x) {
                const float* p = row + x * 3;
                dstrow[x] = 0.114f * p[0] + 0.587f * p[1] + 0.299f * p[2];
            }
        }
        return dst;
    }
#endif
    for (int i = 0; i < height_; ++i)
        for (int j = 0; j < width_; ++j) {
            auto p = this->at(i, j);
            // Backend adopts BGR channel order; OpenCV uses rounding to nearest for U8
            double y = 0.114 * p[0] + 0.587 * p[1] + 0.299 * p[2];
            if (std::is_integral<D>::value) {
                dst.at(i, j)[0] = static_cast<D>(std::floor(y + 0.5));
            } else {
                dst.at(i, j)[0] = static_cast<D>(y);
            }
        }
    return dst;
}

template <typename D>
Bitmap<D> Bitmap<D>::SwapRB() const {
    INSPIRECV_CHECK_EQ(channels_, 3);
    Bitmap<D> dst;
    dst.Reset(width_, height_, channels_);
#if defined(__SSSE3__)
    if (kX86U8C3KernelsEnabled && std::is_same<D, uint8_t>::value) {
        x86::SwapRbU8C3(
          reinterpret_cast<const uint8_t*>(Data()),
          reinterpret_cast<uint8_t*>(dst.Data()), width_, height_);
        return dst;
    }
#endif
#if defined(__AVX2__)
    // AVX2 fast path for float 3-channel: gather/swap/scatter 8 pixels per iter
    if (std::is_same<D, float>::value) {
        const float* src = reinterpret_cast<const float*>(Data());
        float* out = reinterpret_cast<float*>(dst.Data());
        const int stride = width_ * 3;
        alignas(32) int idxB[8], idxG[8], idxR[8];
        for (int y = 0; y < height_; ++y) {
            const float* row = src + static_cast<size_t>(y) * stride;
            float* dstrow = out + static_cast<size_t>(y) * stride;
            int x = 0;
            for (; x + 8 <= width_; x += 8) {
                for (int i = 0; i < 8; ++i) {
                    int base = (x + i) * 3;
                    idxB[i] = base + 0;
                    idxG[i] = base + 1;
                    idxR[i] = base + 2;
                }
                __m256i vB = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(idxB));
                __m256i vG = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(idxG));
                __m256i vR = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(idxR));
                __m256 B = _mm256_i32gather_ps(row, vB, 4);
                __m256 G = _mm256_i32gather_ps(row, vG, 4);
                __m256 R = _mm256_i32gather_ps(row, vR, 4);
                alignas(32) float b[8], g[8], r[8];
                _mm256_store_ps(b, B);
                _mm256_store_ps(g, G);
                _mm256_store_ps(r, R);
                for (int i = 0; i < 8; ++i) {
                    int base = (x + i) * 3;
                    dstrow[base + 0] = r[i];
                    dstrow[base + 1] = g[i];
                    dstrow[base + 2] = b[i];
                }
            }
            for (; x < width_; ++x) {
                const float* p = row + x * 3;
                float* q = dstrow + x * 3;
                q[0] = p[2]; q[1] = p[1]; q[2] = p[0];
            }
        }
        return dst;
    }
#endif
#if (defined(__ARM_NEON) || defined(__ARM_NEON__))
    if (std::is_same<D, uint8_t>::value) {
        const uint8_t* src = reinterpret_cast<const uint8_t*>(Data());
        uint8_t* out = reinterpret_cast<uint8_t*>(dst.Data());
        const int stride = width_ * 3;
        for (int y = 0; y < height_; ++y) {
            const uint8_t* row = src + static_cast<size_t>(y) * stride;
            uint8_t* dstrow = out + static_cast<size_t>(y) * stride;
            int x = 0;
            for (; x + 16 <= width_; x += 16) {
                uint8x16x3_t bgr = vld3q_u8(row + x * 3);
                uint8x16x3_t rgb;
                rgb.val[0] = bgr.val[2];
                rgb.val[1] = bgr.val[1];
                rgb.val[2] = bgr.val[0];
                vst3q_u8(dstrow + x * 3, rgb);
            }
            for (; x < width_; ++x) {
                const uint8_t* p = row + x * 3;
                uint8_t* q = dstrow + x * 3;
                q[0] = p[2]; q[1] = p[1]; q[2] = p[0];
            }
        }
        return dst;
    }
    if (std::is_same<D, float>::value) {
        const float* src = reinterpret_cast<const float*>(Data());
        float* out = reinterpret_cast<float*>(dst.Data());
        const int stride = width_ * 3;
        for (int y = 0; y < height_; ++y) {
            const float* row = src + static_cast<size_t>(y) * stride;
            float* dstrow = out + static_cast<size_t>(y) * stride;
            int x = 0;
            for (; x + 4 <= width_; x += 4) {
                float32x4x3_t bgr = vld3q_f32(row + x * 3);
                float32x4x3_t rgb;
                rgb.val[0] = bgr.val[2];
                rgb.val[1] = bgr.val[1];
                rgb.val[2] = bgr.val[0];
                vst3q_f32(dstrow + x * 3, rgb);
            }
            for (; x < width_; ++x) {
                const float* p = row + x * 3;
                float* q = dstrow + x * 3;
                q[0] = p[2]; q[1] = p[1]; q[2] = p[0];
            }
        }
        return dst;
    }
#endif
    auto src_iter = Data();
    auto dst_iter = dst.Data();
    for (int i = 0; i < height_ * width_; ++i) {
        dst_iter[0] = src_iter[2];
        dst_iter[1] = src_iter[1];
        dst_iter[2] = src_iter[0];
        src_iter += channels_;
        dst_iter += channels_;
    }
    return dst;
}

template <typename D>
Rect2i Bitmap<D>::GetMaskRect(D threshold) const {
    INSPIRECV_CHECK(!Empty());

    int left = Width() - 1;
    int right = 0;
    int top = Height() - 1;
    int bottom = 0;

    auto iter = Data();
    for (int i = 0; i < Height(); ++i)
        for (int j = 0; j < Width(); ++j) {
            if (*iter > threshold) {
                left = std::min(left, j);
                right = std::max(right, j);
                top = std::min(top, i);
                bottom = std::max(bottom, i);
            }
            ++iter;
        }

    return Rect2i(left, top, right, bottom);
}

template <typename D>
Status Bitmap<D>::FillRect(const Rect2i &rect, const std::vector<D> &color) {
    if (color.size() != channels_) {
        std::stringstream msg;
        msg << "color.size()=" << color.size() << ", channels_=" << channels_;
        return Status(error::INVALID_ARGUMENT, msg.str());
    }
    auto xmin = std::max(0, rect.xmin());
    auto ymin = std::max(0, rect.ymin());
    auto xmax = std::min(width_, rect.xmax());
    auto ymax = std::min(height_, rect.ymax());
    for (int i = ymin; i < ymax; ++i) {
        auto iter = at(i, xmin);
        for (int j = xmin; j < xmax; ++j) {
            memcpy(iter, color.data(), channels_ * sizeof(D));
            iter += channels_;
        }
    }
    return Status::OK();
}

template <typename D>
Status Bitmap<D>::FillCircle(const Point2f &center, float radius, const std::vector<D> &color) {
    if (color.size() != static_cast<size_t>(channels_) || radius < 0.0f) {
        return Status(error::INVALID_ARGUMENT, "Invalid circle color or radius");
    }
    int top = std::ceil(std::max(0.f, center.y - radius));
    int bottom = std::min(height_ - 1.f, center.y + radius);
    for (int y = top; y <= bottom; ++y) {
        double d = std::sqrt(radius * radius - (y - center.y) * (y - center.y));
        int left = std::max(static_cast<int>(std::ceil(center.x - d)), 0);
        int right = std::min(static_cast<int>(center.x + d), width_ - 1);
        auto iter = at(y, left);
        for (int x = left; x <= right; ++x) {
            memcpy(iter, color.data(), channels_ * sizeof(D));
            iter += channels_;
        }
    }
    return Status::OK();
}

template <typename D>
Status Bitmap<D>::DrawPoint(const Point2f &point, float thinkness, const std::vector<D> &color) {
    INSPIRECV_RETURN_IF_ERROR(FillCircle(point, thinkness, color));
    return Status::OK();
}

template <typename D>
Status Bitmap<D>::DrawCircle(const Point2f &center, float radius,
                            const std::vector<D> &color, int thickness) {
    if (color.size() != static_cast<size_t>(channels_) || radius < 0.0f || thickness == 0) {
        return Status(error::INVALID_ARGUMENT, "Invalid circle color, radius, or thickness");
    }
    if (thickness < 0) return FillCircle(center, radius, color);

    const float half_thickness = 0.5f * static_cast<float>(std::max(1, thickness));
    const float inner_radius = std::max(0.0f, radius - half_thickness);
    const float outer_radius = radius + half_thickness;
    const float inner_squared = inner_radius * inner_radius;
    const float outer_squared = outer_radius * outer_radius;
    const int top = std::max(0, static_cast<int>(std::floor(center.y - outer_radius)));
    const int bottom = std::min(height_ - 1,
                                static_cast<int>(std::ceil(center.y + outer_radius)));
    const int left = std::max(0, static_cast<int>(std::floor(center.x - outer_radius)));
    const int right = std::min(width_ - 1,
                               static_cast<int>(std::ceil(center.x + outer_radius)));

    for (int y = top; y <= bottom; ++y) {
        const float dy = static_cast<float>(y) - center.y;
        for (int x = left; x <= right; ++x) {
            const float dx = static_cast<float>(x) - center.x;
            const float distance_squared = dx * dx + dy * dy;
            if (distance_squared < inner_squared || distance_squared > outer_squared) continue;
            std::copy_n(color.data(), channels_, at(y, x));
        }
    }
    return Status::OK();
}

template <typename D>
Status Bitmap<D>::DrawLine(const Point2i &p0, const Point2i &p1, const std::vector<D> &color,
                          int thickness) {
    if (p0 == p1) {
        return Status(error::INVALID_ARGUMENT, "Same points!");
    }
    int b0 = thickness / 2;
    int b1 = thickness - b0;
    if (p0.x == p1.x) {
        Rect2i rect(p0.x - b0, std::min(p0.y, p1.y), p0.x + b1, std::max(p0.y, p1.y));
        INSPIRECV_RETURN_IF_ERROR(FillRect(rect, color));
    } else if (p0.y == p1.y) {
        Rect2i rect(std::min(p0.x, p1.x), p0.y - b0, std::max(p0.x, p1.x), p0.y + b1);
        INSPIRECV_RETURN_IF_ERROR(FillRect(rect, color));
    } else {
        double k = static_cast<double>(p1.y - p0.y) / (p1.x - p0.x);
        double b = p0.y - p0.x * k;
        if (-1 <= k && k <= 1) {
            auto xmin = std::max(std::min(p0.x, p1.x), 0);
            auto xmax = std::min(std::max(p0.x, p1.x), width_ - 1);
            for (int x = xmin; x <= xmax; ++x) {
                int y_c = static_cast<int>(k * x + b + 0.5);
                int ymin = std::max(0, y_c - b0);
                int ymax = std::min(height_ - 1, y_c + b1 - 1);
                for (int y = ymin; y <= ymax; ++y)
                    memcpy(at(y, x), color.data(), channels_ * sizeof(D));
            }
        } else {
            auto ymin = std::max(std::min(p0.y, p1.y), 0);
            auto ymax = std::min(std::max(p0.y, p1.y), height_ - 1);
            for (int y = ymin; y <= ymax; ++y) {
                int x_c = static_cast<int>((y - b) / k + 0.5);
                int xmin = std::max(0, x_c - b0);
                int xmax = std::min(width_ - 1, x_c + b1 - 1);
                for (int x = xmin; x <= xmax; ++x)
                    memcpy(at(y, x), color.data(), channels_ * sizeof(D));
            }
        }
    }
    return Status::OK();
}

template <typename D>
Status Bitmap<D>::DrawRect(const Rect2i &rect, const std::vector<D> &color, int thickness) {
    INSPIRECV_RETURN_IF_ERROR(DrawLine(Point2i(rect.xmin(), rect.ymin()),
                                       Point2i(rect.xmin(), rect.ymax()), color, thickness));
    INSPIRECV_RETURN_IF_ERROR(DrawLine(Point2i(rect.xmin(), rect.ymax()),
                                       Point2i(rect.xmax(), rect.ymax()), color, thickness));
    INSPIRECV_RETURN_IF_ERROR(DrawLine(Point2i(rect.xmax(), rect.ymax()),
                                       Point2i(rect.xmax(), rect.ymin()), color, thickness));
    INSPIRECV_RETURN_IF_ERROR(DrawLine(Point2i(rect.xmax(), rect.ymin()),
                                       Point2i(rect.xmin(), rect.ymin()), color, thickness));
    return Status::OK();
}

template <typename D>
Rect2i Bitmap<D>::GetSafeRect(const Rect2i &rect) const {
    return Rect2i(std::max(0, rect.xmin()), std::max(0, rect.ymin()),
                  std::min(width_ - 1, rect.xmax()), std::min(height_ - 1, rect.ymax()));
}

#ifdef INSPIRECV_BACKEND_OKCV_USE_OPENCV

template <typename dtype>
void Bitmap<dtype>::FromCVMat(const cv::Mat &mat, bool swap_channels) {
    Reset(mat.size().width, mat.size().height, mat.channels());
    auto data_ptr = Data();
    auto depth_type = (mat.type() & CV_MAT_DEPTH_MASK);
    INSPIRECV_CHECK(depth_type == CV_8U || depth_type == CV_32F);
    if (depth_type == CV_8U) {
        for (int i = 0; i < height_; ++i) {
            for (int j = 0; j < width_; ++j) {
                if (channels_ == 3) {
                    cv::Vec3b intensity = mat.at<cv::Vec3b>(i, j);
                    if (swap_channels) {
                        *data_ptr++ = static_cast<dtype>(intensity.val[2]);
                        *data_ptr++ = static_cast<dtype>(intensity.val[1]);
                        *data_ptr++ = static_cast<dtype>(intensity.val[0]);
                    } else {
                        *data_ptr++ = static_cast<dtype>(intensity.val[0]);
                        *data_ptr++ = static_cast<dtype>(intensity.val[1]);
                        *data_ptr++ = static_cast<dtype>(intensity.val[2]);
                    }
                } else {
                    *data_ptr++ = static_cast<dtype>(mat.at<uchar>(i, j));
                }
            }
        }
    } else {
        for (int i = 0; i < height_; ++i) {
            for (int j = 0; j < width_; ++j) {
                if (channels_ == 3) {
                    cv::Vec3f intensity = mat.at<cv::Vec3f>(i, j);
                    if (swap_channels) {
                        *data_ptr++ = static_cast<dtype>(intensity.val[2]);
                        *data_ptr++ = static_cast<dtype>(intensity.val[1]);
                        *data_ptr++ = static_cast<dtype>(intensity.val[0]);
                    } else {
                        *data_ptr++ = static_cast<dtype>(intensity.val[0]);
                        *data_ptr++ = static_cast<dtype>(intensity.val[1]);
                        *data_ptr++ = static_cast<dtype>(intensity.val[2]);
                    }
                } else {
                    *data_ptr++ = static_cast<dtype>(mat.at<float>(i, j));
                }
            }
        }
    }
}

template <>
void Bitmap<float>::ToCVMat(cv::Mat &mat, bool swap_channels) const {
    INSPIRECV_CHECK(channels_ == 1 || channels_ == 3 || channels_ == 4)
      << "channels_ = " << channels_;
    int flag = channels_ == 1 ? CV_32FC1 : CV_32FC3;
    if (channels_ == 4)
        flag = CV_32FC4;
    mat = cv::Mat(height_, width_, flag);
    auto data_ptr = Data();
    float *mat_data_ptr = reinterpret_cast<float *>(mat.data);
    for (int i = 0; i < height_ * width_; ++i) {
        if (channels_ == 3) {
            if (swap_channels) {
                *mat_data_ptr++ = data_ptr[2];
                *mat_data_ptr++ = data_ptr[1];
                *mat_data_ptr++ = data_ptr[0];
            } else {
                *mat_data_ptr++ = data_ptr[0];
                *mat_data_ptr++ = data_ptr[1];
                *mat_data_ptr++ = data_ptr[2];
            }
            data_ptr += 3;
        } else if (channels_ == 1) {
            *mat_data_ptr++ = *data_ptr++;
        } else {
            if (swap_channels) {
                *mat_data_ptr++ = data_ptr[2];
                *mat_data_ptr++ = data_ptr[1];
                *mat_data_ptr++ = data_ptr[0];
            } else {
                *mat_data_ptr++ = data_ptr[0];
                *mat_data_ptr++ = data_ptr[1];
                *mat_data_ptr++ = data_ptr[2];
            }
            *mat_data_ptr++ = data_ptr[3];
            data_ptr += 4;
        }
    }
}
template <>
void Bitmap<uint8_t>::ToCVMat(cv::Mat &mat, bool swap_channels) const {
    INSPIRECV_CHECK(channels_ == 1 || channels_ == 3 || channels_ == 4)
      << "channels_ = " << channels_;
    int flag = channels_ == 1 ? CV_8UC1 : CV_8UC3;
    if (channels_ == 4)
        flag = CV_8UC4;
    mat = cv::Mat(height_, width_, flag);
    auto data_ptr = Data();
    auto mat_data_ptr = mat.data;
    for (int i = 0; i < height_ * width_; ++i) {
        if (channels_ == 3) {
            if (swap_channels) {
                *mat_data_ptr++ = data_ptr[2];
                *mat_data_ptr++ = data_ptr[1];
                *mat_data_ptr++ = data_ptr[0];
            } else {
                *mat_data_ptr++ = data_ptr[0];
                *mat_data_ptr++ = data_ptr[1];
                *mat_data_ptr++ = data_ptr[2];
            }
            data_ptr += 3;
        } else if (channels_ == 1) {
            *mat_data_ptr++ = *data_ptr++;
        } else {
            if (swap_channels) {
                *mat_data_ptr++ = data_ptr[2];
                *mat_data_ptr++ = data_ptr[1];
                *mat_data_ptr++ = data_ptr[0];
            } else {
                *mat_data_ptr++ = data_ptr[0];
                *mat_data_ptr++ = data_ptr[1];
                *mat_data_ptr++ = data_ptr[2];
            }
            *mat_data_ptr++ = data_ptr[3];
            data_ptr += 4;
        }
    }
}

#endif  // INSPIRECV_BACKEND_OKCV_USE_OPENCV

template class Bitmap<uint8_t>;
template class Bitmap<float>;

}  // namespace okcv
