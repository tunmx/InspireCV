#ifndef INSPIRECV_IMPL_OKCV_IMAGE_OKCV_H
#define INSPIRECV_IMPL_OKCV_IMAGE_OKCV_H

#include "okcv/okcv.h"
#include "inspirecv/core/image.h"
#include "inspirecv/core/rect.h"
#include "inspirecv/core/point.h"
#include "inspirecv/core/transform_matrix.h"
#include "logging.h"

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

template<typename Pixel>
class ImageT<Pixel>::Impl {
public:
    // Creation and conversion
    Impl(const okcv::Image<Pixel>& mat) {
        image_ = mat.Clone();
    }

    Impl() : image_() {}

    Impl(int width, int height, int channels, const Pixel* data = nullptr,
         bool copy_data = true) {
        image_.Reset(width, height, channels, data, copy_data);
    }

    void Reset(int width, int height, int channels, const Pixel* data = nullptr,
               bool copy_data = true) {
        image_.Reset(width, height, channels, data, copy_data);
    }

    ImageT<Pixel> Clone() const {
        ImageT<Pixel> result;
        result.impl_->image_ = image_.Clone();
        return result;
    }

    ~Impl() = default;

    // Basic properties
    int Width() const {
        return image_.Width();
    }
    int Height() const {
        return image_.Height();
    }
    int Channels() const {
        return image_.Channels();
    }
    bool Empty() const {
        return image_.Empty();
    }
    const Pixel* Data() const {
        return image_.Data();
    }

    // Get internal image implementation
    void* GetInternalImage() const {
        return static_cast<void*>(const_cast<Pixel*>(image_.Data()));
    }

    // I/O operations
    bool Read(const std::string& filename, int channels) {
        image_.Read(filename, channels);
        return !Empty();
    }

    bool Write(const std::string& filename) const {
        image_.Write(filename);
        return true;
    }

    void Show(const std::string& window_name, int delay) const {
        image_.Show(window_name, delay);
    }

    // Access to core image for interop if needed
    const okcv::Image<Pixel>& Core() const { return image_; }

    // Basic operations
    void Fill(double value) {
        image_.Fill(static_cast<Pixel>(value));
    }

    ImageT<Pixel> Mul(double scale) const {
        ImageT<Pixel> result;
        result.impl_->image_ = image_.Mul(static_cast<float>(scale));
        return result;
    }

    ImageT<Pixel> Add(double value) const {
        ImageT<Pixel> result;
        result.impl_->image_ = image_.MulAdd(1.0f, static_cast<float>(value));
        return result;
    }

    // Geometric transformations
    ImageT<Pixel> Resize(int width, int height, bool use_linear) const {
        ImageT<Pixel> result;
        if (use_linear) {
            result.impl_->image_ = image_.ResizeBilinear(width, height);
        } else {
            result.impl_->image_ = image_.ResizeNearest(width, height);
        }
        return result;
    }

    ImageT<Pixel> Crop(const Rect<int>& rect) const {
        ImageT<Pixel> result;
        int x2 = rect.GetX() + std::max(1, rect.GetWidth());
        int y2 = rect.GetY() + std::max(1, rect.GetHeight());

        okcv::Rect2i okcv_rect(rect.GetX(), rect.GetY(), x2, y2);

        result.impl_->image_ = image_.Crop(okcv_rect);
        return result;
    }

    ImageT<Pixel> WarpAffine(const TransformMatrix& matrix, int width, int height) const {
        ImageT<Pixel> result;
        okcv::TransformMatrix okcv_matrix =
          *static_cast<okcv::TransformMatrix*>(matrix.GetInternalMatrix());
        result.impl_->image_ = image_.AffineBilinear(width, height, okcv_matrix);
        return result;
    }

    ImageT<Pixel> Rotate90() const {
        ImageT<Pixel> result;
        result.impl_->image_ = image_.Rotate90();
        return result;
    }

    ImageT<Pixel> Rotate180() const {
        ImageT<Pixel> result;
        result.impl_->image_ = image_.Rotate180();
        return result;
    }

    ImageT<Pixel> Rotate270() const {
        ImageT<Pixel> result;
        result.impl_->image_ = image_.Rotate270();
        return result;
    }

    ImageT<Pixel> SwapRB() const {
        ImageT<Pixel> result;
        result.impl_->image_ = image_.SwapRB();
        return result;
    }

    ImageT<Pixel> FlipHorizontal() const {
        ImageT<Pixel> result;
        result.impl_->image_ = image_.FlipLeftRight();
        return result;
    }

    ImageT<Pixel> FlipVertical() const {
        ImageT<Pixel> result;
        result.impl_->image_ = image_.FlipUpDown();
        return result;
    }

    ImageT<Pixel> Pad(int top, int bottom, int left, int right, const std::vector<double>& color) const {
        ImageT<Pixel> result;
        result.impl_->image_ = image_.Pad(top, bottom, left, right, static_cast<Pixel>(color[0]));
        return result;
    }

    // Image processing
    ImageT<Pixel> GaussianBlur(int kernel_size, double sigma) const {
        ImageT<Pixel> result;
        result.impl_->image_ = image_.GaussianBlur(kernel_size, static_cast<float>(sigma));
        return result;
    }

    ImageT<Pixel> Erode(int kernel_size, int iterations) const {
        INSPIRECV_CHECK(image_.Channels() == 1) << "Erode only supports single-channel";
        int left = kernel_size / 2;
        int right = kernel_size - left - 1;
        int top = kernel_size / 2;
        int bottom = kernel_size - top - 1;
        ImageT<Pixel> result;
        okcv::Image<Pixel> cur = image_.Clone();
        for (int it = 0; it < std::max(1, iterations); ++it) {
            okcv::Image<Pixel> tmp = cur.MinFilter(left, right, top, bottom);
            cur = std::move(tmp);
        }
        result.impl_->image_ = std::move(cur);
        return result;
    }

    ImageT<Pixel> Dilate(int kernel_size, int iterations) const {
        INSPIRECV_CHECK(image_.Channels() == 1) << "Dilate only supports single-channel";
        int left = kernel_size / 2;
        int right = kernel_size - left - 1;
        int top = kernel_size / 2;
        int bottom = kernel_size - top - 1;
        ImageT<Pixel> result;
        okcv::Image<Pixel> cur = image_.Clone();
        for (int it = 0; it < std::max(1, iterations); ++it) {
            okcv::Image<Pixel> tmp = cur.MaxFilter(left, right, top, bottom);
            cur = std::move(tmp);
        }
        result.impl_->image_ = std::move(cur);
        return result;
    }

    ImageT<Pixel> Threshold(double thresh, double maxval, int type) const {
        // Support binary only: type==0 -> THRESH_BINARY
        INSPIRECV_CHECK(image_.Channels() == 1);
        ImageT<Pixel> result;
        result.impl_->image_.Reset(image_.Width(), image_.Height(), 1);
        const Pixel* src = image_.Data();
        Pixel* dst = result.impl_->image_.Data();
        const int total = image_.DataSize();
        const Pixel t = static_cast<Pixel>(thresh);
        const Pixel mv = static_cast<Pixel>(maxval);
        for (int i = 0; i < total; ++i) dst[i] = src[i] >= t ? mv : static_cast<Pixel>(0);
        return result;
    }

    ImageT<Pixel> AbsDiff(const typename ImageT<Pixel>::Impl& other) const {
        INSPIRECV_CHECK(image_.Width() == other.Width() && image_.Height() == other.Height() &&
                        image_.Channels() == other.Channels());
        ImageT<Pixel> result;
        result.impl_->image_.Reset(image_.Width(), image_.Height(), image_.Channels());
        const Pixel* a = image_.Data();
        const Pixel* b = other.image_.Data();
        Pixel* d = result.impl_->image_.Data();
        const int n = image_.DataSize();
        for (int i = 0; i < n; ++i) {
            auto diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
            double ad = std::abs(diff);
            d[i] = static_cast<Pixel>(ad);
        }
        return result;
    }

    ImageT<Pixel> MeanChannels() const {
        INSPIRECV_CHECK(image_.Channels() >= 1);
        if (image_.Channels() == 1) return Clone();
        ImageT<Pixel> result;
        result.impl_->image_.Reset(image_.Width(), image_.Height(), 1);
        for (int y = 0; y < image_.Height(); ++y) {
            const Pixel* row = image_.Row(y);
            Pixel* out = result.impl_->image_.Row(y);
            for (int x = 0; x < image_.Width(); ++x) {
                double s = 0;
                for (int c = 0; c < image_.Channels(); ++c) s += row[x * image_.Channels() + c];
                out[x] = static_cast<Pixel>(s / image_.Channels());
            }
        }
        return result;
    }

    ImageT<Pixel> Blend(const typename ImageT<Pixel>::Impl& other, const typename ImageT<uint8_t>::Impl& mask) const {
        INSPIRECV_CHECK(image_.Width() == other.Width() && image_.Height() == other.Height() &&
                        image_.Channels() == other.Channels());
        INSPIRECV_CHECK(mask.Channels() == 1 && mask.Width() == image_.Width() &&
                        mask.Height() == image_.Height());

        const int width = image_.Width();
        const int height = image_.Height();
        const int cn = image_.Channels();

        ImageT<Pixel> result;
        result.impl_->image_.Reset(width, height, cn);

        const Pixel* a = image_.Data();
        const Pixel* b = other.Data();
        const uint8_t* m = mask.Data();
        Pixel* d = result.impl_->image_.Data();

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
        image_.DrawLine(start, end, okcv_color, thickness);
    }

    void DrawRect(const Rect<int>& rect, const std::vector<double>& color, int thickness = 1) {
        okcv::Rect2i okcv_rect = *static_cast<okcv::Rect2i*>(rect.GetInternalRect());
        std::vector<Pixel> okcv_color;
        for (const auto& c : color) {
            okcv_color.push_back(static_cast<Pixel>(c));
        }
        image_.DrawRect(okcv_rect, okcv_color, thickness);
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
        image_.DrawPoint(center_point, static_cast<float>(thickness), okcv_color);
    }

    void Fill(const Rect<int>& rect, const std::vector<double>& color) {
        okcv::Rect2i okcv_rect = *static_cast<okcv::Rect2i*>(rect.GetInternalRect());
        std::vector<Pixel> okcv_color;
        for (const auto& c : color) {
            okcv_color.push_back(static_cast<Pixel>(c));
        }
        image_.FillRect(okcv_rect, okcv_color);
    }

    // Image format conversion
    ImageT<Pixel> ToGray() const {
        ImageT<Pixel> result;
        result.impl_->image_ = image_.RgbToGray();
        return result;
    }

    void Print(std::ostream& os) const {
        const int N = 10;  // Threshold for truncated display
        if (image_.Height() > N || image_.Width() > N) {
            // For large matrices, show truncated view
            os << "[";
            for (int i = 0; i < std::min(3, image_.Height()); i++) {
                if (i > 0)
                    os << " ";
                os << "[";
                // Show first 3 elements
                for (int j = 0; j < std::min(3, image_.Width()); j++) {
                    if (j > 0)
                        os << " ";
                    if (image_.Channels() == 1) {
                        os << (int)*(image_.at(i, j));
                    } else {
                        os << "[";
                        for (int c = 0; c < image_.Channels(); c++) {
                            if (c > 0)
                                os << " ";
                            os << (int)*(image_.at(i, j) + c);
                        }
                        os << "]";
                    }
                }
                if (image_.Width() > 3)
                    os << " ... ";
                // Show last 3 elements if there are more columns
                if (image_.Width() > 6) {
                    for (int j = image_.Width() - 3; j < image_.Width(); j++) {
                        if (image_.Channels() == 1) {
                            os << (int)*(image_.at(i, j)) << " ";
                        } else {
                            os << "[";
                            for (int c = 0; c < image_.Channels(); c++) {
                                if (c > 0)
                                    os << " ";
                                os << (int)*(image_.at(i, j) + c);
                            }
                            os << "] ";
                        }
                    }
                }
                os << "]\n";
            }
            if (image_.Height() > 6) {
                os << "...\n";
                // Show last 3 rows
                for (int i = image_.Height() - 3; i < image_.Height(); i++) {
                    os << "[";
                    for (int j = 0; j < std::min(3, image_.Width()); j++) {
                        if (j > 0)
                            os << " ";
                        if (image_.Channels() == 1) {
                            os << (int)*(image_.at(i, j));
                        } else {
                            os << "[";
                            for (int c = 0; c < image_.Channels(); c++) {
                                if (c > 0)
                                    os << " ";
                                os << (int)*(image_.at(i, j) + c);
                            }
                            os << "]";
                        }
                    }
                    if (image_.Width() > 3)
                        os << " ... ";
                    if (image_.Width() > 6) {
                        for (int j = image_.Width() - 3; j < image_.Width(); j++) {
                            if (image_.Channels() == 1) {
                                os << (int)*(image_.at(i, j)) << " ";
                            } else {
                                os << "[";
                                for (int c = 0; c < image_.Channels(); c++) {
                                    if (c > 0)
                                        os << " ";
                                    os << (int)*(image_.at(i, j) + c);
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
            for (int i = 0; i < image_.Height(); i++) {
                if (i > 0)
                    os << " ";
                os << "[";
                for (int j = 0; j < image_.Width(); j++) {
                    if (j > 0)
                        os << " ";
                    if (image_.Channels() == 1) {
                        os << (int)*(image_.at(i, j));
                    } else {
                        os << "[";
                        for (int c = 0; c < image_.Channels(); c++) {
                            if (c > 0)
                                os << " ";
                            os << (int)*(image_.at(i, j) + c);
                        }
                        os << "]";
                    }
                }
                os << "]\n";
            }
            os << "]\n";
        }
        os << "Size(H x W x C): " << image_.Height() << " x " << image_.Width() << " x "
           << image_.Channels() << "\n";
    }

private:
    okcv::Image<Pixel> image_;
};

}  // namespace inspirecv

#endif  // INSPIRECV_IMPL_OKCV_IMAGE_OKCV_H
