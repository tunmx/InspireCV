#ifndef INSPIRECV_IMPL_OKCV_IMAGE_OKCV_H
#define INSPIRECV_IMPL_OKCV_IMAGE_OKCV_H

#include "okcv/okcv.h"
#include "inspirecv/core/image.h"
#include "inspirecv/core/rect.h"
#include "inspirecv/core/point.h"
#include "inspirecv/core/transform_matrix.h"
#include "logging.h"

namespace inspirecv {

class Image::Impl {
public:
    // Creation and conversion
    Impl(const okcv::Image<uint8_t>& mat) {
        image_ = mat.Clone();
    }

    Impl() : image_() {}

    Impl(int width, int height, int channels, const uint8_t* data = nullptr) {
        image_.Reset(width, height, channels, data);
    }

    void Reset(int width, int height, int channels, const uint8_t* data = nullptr) {
        image_.Reset(width, height, channels, data);
    }

    Image Clone() const {
        Image result;
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
    const uint8_t* Data() const {
        return image_.Data();
    }

    // Get internal image implementation
    void* GetInternalImage() const {
        return static_cast<void*>(const_cast<uint8_t*>(image_.Data()));
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

    // Basic operations
    void Fill(double value) {
        image_.Fill(static_cast<uint8_t>(value));
    }

    Image Mul(double scale) const {
        Image result;
        result.impl_->image_ = image_.Mul(static_cast<float>(scale));
        return result;
    }

    Image Add(double value) const {
        Image result;
        result.impl_->image_ = image_.MulAdd(1.0f, static_cast<float>(value));
        return result;
    }

    // Geometric transformations
    Image Resize(int width, int height, bool use_linear) const {
        Image result;
        if (use_linear) {
            result.impl_->image_ = image_.ResizeBilinear(width, height);
        } else {
            result.impl_->image_ = image_.ResizeNearest(width, height);
        }
        return result;
    }

    Image Crop(const Rect<int>& rect) const {
        Image result;
        int x2 = rect.GetX() + std::max(1, rect.GetWidth());
        int y2 = rect.GetY() + std::max(1, rect.GetHeight());

        okcv::Rect2i okcv_rect(rect.GetX(), rect.GetY(), x2, y2);

        result.impl_->image_ = image_.Crop(okcv_rect);
        return result;
    }

    Image WarpAffine(const TransformMatrix& matrix, int width, int height) const {
        Image result;
        okcv::TransformMatrix okcv_matrix =
          *static_cast<okcv::TransformMatrix*>(matrix.GetInternalMatrix());
        result.impl_->image_ = image_.AffineBilinear(width, height, okcv_matrix);
        return result;
    }

    Image Rotate90() const {
        Image result;
        result.impl_->image_ = image_.Rotate90();
        return result;
    }

    Image Rotate180() const {
        Image result;
        result.impl_->image_ = image_.Rotate180();
        return result;
    }

    Image Rotate270() const {
        Image result;
        result.impl_->image_ = image_.Rotate270();
        return result;
    }

    Image SwapRB() const {
        Image result;
        result.impl_->image_ = image_.SwapRB();
        return result;
    }

    Image FlipHorizontal() const {
        Image result;
        result.impl_->image_ = image_.FlipLeftRight();
        return result;
    }

    Image FlipVertical() const {
        Image result;
        result.impl_->image_ = image_.FlipUpDown();
        return result;
    }

    Image Pad(int top, int bottom, int left, int right, const std::vector<double>& color) const {
        Image result;
        result.impl_->image_ = image_.Pad(top, bottom, left, right, static_cast<uint8_t>(color[0]));
        return result;
    }

    // Image processing
    Image GaussianBlur(int kernel_size, double sigma) const {
        Image result;
        result.impl_->image_ = image_.Blur(kernel_size);
        return result;
    }

    Image Threshold(double thresh, double maxval, int type) const {
        // TODO: Implement when OKCV adds threshold
        INSPIRECV_LOG(WARN) << "Threshold is not implemented in OKCV";
        return Clone();
    }

    // Drawing operations
    void DrawLine(const Point<int>& p1, const Point<int>& p2, const std::vector<double>& color,
                  int thickness = 1) {
        okcv::Point2i start = *static_cast<okcv::Point2i*>(p1.GetInternalPoint());
        okcv::Point2i end = *static_cast<okcv::Point2i*>(p2.GetInternalPoint());
        std::vector<uint8_t> okcv_color;
        for (const auto& c : color) {
            okcv_color.push_back(static_cast<uint8_t>(c));
        }
        image_.DrawLine(start, end, okcv_color, thickness);
    }

    void DrawRect(const Rect<int>& rect, const std::vector<double>& color, int thickness = 1) {
        okcv::Rect2i okcv_rect = *static_cast<okcv::Rect2i*>(rect.GetInternalRect());
        std::vector<uint8_t> okcv_color;
        for (const auto& c : color) {
            okcv_color.push_back(static_cast<uint8_t>(c));
        }
        image_.DrawRect(okcv_rect, okcv_color, thickness);
    }

    void DrawCircle(const Point<int>& center, int radius, const std::vector<double>& color,
                    int thickness = 1) {
        okcv::Point2f center_point =
          okcv::Point2f(static_cast<okcv::Point2i*>(center.GetInternalPoint())->x,
                        static_cast<okcv::Point2i*>(center.GetInternalPoint())->y);
        std::vector<uint8_t> okcv_color;
        for (const auto& c : color) {
            okcv_color.push_back(static_cast<uint8_t>(c));
        }
        image_.DrawPoint(center_point, static_cast<float>(thickness), okcv_color);
    }

    void Fill(const Rect<int>& rect, const std::vector<double>& color) {
        okcv::Rect2i okcv_rect = *static_cast<okcv::Rect2i*>(rect.GetInternalRect());
        std::vector<uint8_t> okcv_color;
        for (const auto& c : color) {
            okcv_color.push_back(static_cast<uint8_t>(c));
        }
        image_.FillRect(okcv_rect, okcv_color);
    }

    // Image format conversion
    Image ToGray() const {
        Image result;
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
    okcv::Image<uint8_t> image_;
};

}  // namespace inspirecv

#endif  // INSPIRECV_IMPL_OKCV_IMAGE_OKCV_H
