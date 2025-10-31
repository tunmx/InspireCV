#ifndef INSPIRECV_IMPL_OPENCV_IMAGE_OPENCV_H
#define INSPIRECV_IMPL_OPENCV_IMAGE_OPENCV_H

#include "opencv2/opencv.hpp"
#include "inspirecv/core/image.h"
#include "inspirecv/core/rect.h"
#include "inspirecv/core/point.h"
#include "inspirecv/core/transform_matrix.h"

namespace inspirecv {

template<typename Pixel>
class ImageT<Pixel>::Impl {
public:
    // Creation and conversion
    Impl(const cv::Mat& mat) : image_(std::move(mat)) {}

    // copy_data is ignored
    Impl(int width, int height, int channels, const Pixel* data = nullptr,
         bool copy_data = true) {
        image_.create(height, width,
                      std::is_same<Pixel, float>::value ? CV_32FC(channels) : CV_8UC(channels));
        if (data) {
            std::memcpy(image_.data, data, sizeof(Pixel) * width * height * channels);
        }
    }

    // copy_data is ignored
    void Reset(int width, int height, int channels, const Pixel* data = nullptr,
               bool copy_data = true) {
        image_.create(height, width,
                      std::is_same<Pixel, float>::value ? CV_32FC(channels) : CV_8UC(channels));
        if (data) {
            std::memcpy(image_.data, data, sizeof(Pixel) * width * height * channels);
        }
    }

    ImageT<Pixel> Clone() const {
        ImageT<Pixel> result;
        result.impl_->image_ = image_.clone();
        return result;
    }

    // Basic properties
    ~Impl() = default;

    // Basic properties
    int Width() const {
        return image_.cols;
    }
    int Height() const {
        return image_.rows;
    }
    int Channels() const {
        return image_.channels();
    }
    bool Empty() const {
        return image_.empty();
    }
    const Pixel* Data() const { return reinterpret_cast<const Pixel*>(image_.data); }

    // Get internal image implementation
    void* GetInternalImage() const {
        return image_.data;
    }

    // I/O operations
    bool Read(const std::string& filename, int channels) {
        int flag = channels == 3 ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE;
        image_ = cv::imread(filename, flag);
        if (std::is_same<Pixel, float>::value && !image_.empty()) {
            cv::Mat tmp;
            image_.convertTo(tmp, CV_32F);
            if (channels == 3 && image_.channels() == 1) cv::cvtColor(tmp, tmp, cv::COLOR_GRAY2BGR);
            image_ = tmp;
        }
        return !image_.empty();
    }

    bool Write(const std::string& filename) const {
        if (image_.depth() == CV_8U) return cv::imwrite(filename, image_);
        cv::Mat u8; image_.convertTo(u8, CV_8U);
        return cv::imwrite(filename, u8);
    }

    void Show(const std::string& window_name, int delay) const {
        if (image_.depth() == CV_8U) {
            cv::imshow(window_name, image_);
        } else {
            cv::Mat u8; image_.convertTo(u8, CV_8U);
            cv::imshow(window_name, u8);
        }
        cv::waitKey(delay);
    }

    // Expose read-only Mat view for internal interop where necessary
    const cv::Mat& MatRef() const { return image_; }

    // Basic operations
    void Fill(double value) {
        image_.setTo(cv::Scalar::all(value));
    }

    ImageT<Pixel> Mul(double scale) const {
        ImageT<Pixel> result;
        image_.convertTo(result.impl_->image_, -1, scale);
        return result;
    }

    ImageT<Pixel> Add(double value) const {
        ImageT<Pixel> result;
        cv::Scalar scalar;
        if (image_.channels() == 3) {
            scalar = cv::Scalar(value, value, value);
        } else {
            scalar = cv::Scalar(value);
        }
        cv::add(image_, scalar, result.impl_->image_);
        return result;
    }

    ImageT<Pixel> Resize(int width, int height, bool use_linear = true) const {
        ImageT<Pixel> result;
        cv::resize(image_, result.impl_->image_, cv::Size(width, height), 0, 0,
                   use_linear ? cv::INTER_LINEAR : cv::INTER_NEAREST);
        return result;
    }

    ImageT<Pixel> Crop(const Rect<int>& rect) const {
        ImageT<Pixel> result;
        cv::Rect roi(rect.GetX(), rect.GetY(), rect.GetWidth(), rect.GetHeight());
        result.impl_->image_ = image_(roi).clone();
        return result;
    }

    ImageT<Pixel> WarpAffine(const TransformMatrix& matrix, int width, int height) const {
        ImageT<Pixel> result;
        const cv::Mat& M = *static_cast<const cv::Mat*>(matrix.GetInternalMatrix());
        cv::Mat inv_mat;
        // Because opencv's warpaffine requires the inverse of the transform matrix
        cv::invertAffineTransform(M.rowRange(0, 2),
                                  inv_mat);  // Get first 2 rows for affine transform
        cv::warpAffine(image_, result.impl_->image_, inv_mat, cv::Size(width, height));
        return result;
    }

    ImageT<Pixel> Rotate90() const {
        ImageT<Pixel> result;
        cv::Mat result_mat;
        cv::rotate(image_, result_mat, cv::ROTATE_90_CLOCKWISE);
        result.impl_->image_ = result_mat;
        return result;
    }

    ImageT<Pixel> Rotate180() const {
        ImageT<Pixel> result;
        cv::Mat result_mat;
        cv::rotate(image_, result_mat, cv::ROTATE_180);
        result.impl_->image_ = result_mat;
        return result;
    }

    ImageT<Pixel> Rotate270() const {
        ImageT<Pixel> result;
        cv::Mat result_mat;
        cv::rotate(image_, result_mat, cv::ROTATE_90_COUNTERCLOCKWISE);
        result.impl_->image_ = result_mat;
        return result;
    }

    ImageT<Pixel> SwapRB() const {
        ImageT<Pixel> result;
        cv::cvtColor(image_, result.impl_->image_, cv::COLOR_BGR2RGB);
        return result;
    }

    ImageT<Pixel> FlipHorizontal() const {
        ImageT<Pixel> result;
        cv::Mat result_mat;
        cv::flip(image_, result_mat, 1);
        result.impl_->image_ = result_mat;
        return result;
    }

    ImageT<Pixel> FlipVertical() const {
        ImageT<Pixel> result;
        cv::Mat result_mat;
        cv::flip(image_, result_mat, 0);
        result.impl_->image_ = result_mat;
        return result;
    }

    ImageT<Pixel> Pad(int top, int bottom, int left, int right, const std::vector<double>& color) const {
        ImageT<Pixel> result;
        cv::Scalar cv_color;
        if (color.empty()) {
            cv_color = cv::Scalar(0);
        } else if (image_.channels() == 1) {
            cv_color = cv::Scalar(color[0]);
        } else if (image_.channels() == 3) {
            // Convert RGB to BGR for OpenCV
            cv_color = cv::Scalar(color[2], color.size() > 1 ? color[1] : 0,
                                  color.size() > 2 ? color[0] : 0);
        } else if (image_.channels() == 4) {
            // Convert RGBA to BGRA for OpenCV
            cv_color = cv::Scalar(color[2], color.size() > 1 ? color[1] : 0,
                                  color.size() > 2 ? color[0] : 0, color.size() > 3 ? color[3] : 0);
        }
        cv::copyMakeBorder(image_, result.impl_->image_, top, bottom, left, right,
                           cv::BORDER_CONSTANT, cv_color);
        return result;
    }

    // Image format conversion
    ImageT<Pixel> ToGray() const {
        ImageT<Pixel> result;
        cv::cvtColor(image_, result.impl_->image_, cv::COLOR_BGR2GRAY);
        return result;
    }

    // Drawing operations
    void DrawLine(const Point<int>& p1, const Point<int>& p2, const std::vector<double>& color,
                  int thickness = 1) {
        cv::Scalar cv_color;
        if (!color.empty()) {
            if (image_.channels() >= 3) {
                // Convert RGB to BGR for OpenCV
                cv_color = cv::Scalar(color[2], color.size() > 1 ? color[1] : 0,
                                      color.size() > 2 ? color[0] : 0);
            } else {
                cv_color = cv::Scalar(color[0]);
            }
        }
        cv::Point cv_p1 = *static_cast<cv::Point*>(p1.GetInternalPoint());
        cv::Point cv_p2 = *static_cast<cv::Point*>(p2.GetInternalPoint());
        cv::line(image_, cv_p1, cv_p2, cv_color, thickness);
    }

    void DrawRect(const Rect<int>& rect, const std::vector<double>& color, int thickness = 1) {
        cv::Scalar cv_color;
        if (!color.empty()) {
            if (image_.channels() >= 3) {
                // Convert RGB to BGR for OpenCV
                cv_color = cv::Scalar(color[2], color.size() > 1 ? color[1] : 0,
                                      color.size() > 2 ? color[0] : 0);
            } else {
                cv_color = cv::Scalar(color[0]);
            }
        }
        cv::Rect cv_rect = *static_cast<cv::Rect*>(rect.GetInternalRect());
        cv::rectangle(image_, cv_rect, cv_color, thickness);
    }

    void DrawCircle(const Point<int>& center, int radius, const std::vector<double>& color,
                    int thickness = 1) {
        cv::Scalar cv_color;
        if (!color.empty()) {
            if (image_.channels() >= 3) {
                // Convert RGB to BGR for OpenCV
                cv_color = cv::Scalar(color[2], color.size() > 1 ? color[1] : 0,
                                      color.size() > 2 ? color[0] : 0);
            } else {
                cv_color = cv::Scalar(color[0]);
            }
        }
        cv::Point cv_center = *static_cast<cv::Point*>(center.GetInternalPoint());
        cv::circle(image_, cv_center, radius, cv_color, thickness);
    }

    void Fill(const Rect<int>& rect, const std::vector<double>& color) {
        cv::Scalar cv_color;
        if (!color.empty()) {
            if (image_.channels() >= 3) {
                // Convert RGB to BGR for OpenCV
                cv_color = cv::Scalar(color[2], color.size() > 1 ? color[1] : 0,
                                      color.size() > 2 ? color[0] : 0);
            } else {
                cv_color = cv::Scalar(color[0]);
            }
        }
        cv::Rect cv_rect = *static_cast<cv::Rect*>(rect.GetInternalRect());
        image_(cv_rect).setTo(cv_color);
    }

    ImageT<Pixel> GaussianBlur(int kernel_size, double sigma) const {
        ImageT<Pixel> result;
        cv::GaussianBlur(image_, result.impl_->image_, cv::Size(kernel_size, kernel_size), sigma,
                         sigma);
        return result;
    }

    ImageT<Pixel> Erode(int kernel_size, int iterations) const {
        ImageT<Pixel> result;
        if ((kernel_size & 1) == 0) ++kernel_size;
        cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
        cv::erode(image_, result.impl_->image_, element, cv::Point(-1,-1), std::max(1, iterations));
        return result;
    }

    ImageT<Pixel> Dilate(int kernel_size, int iterations) const {
        ImageT<Pixel> result;
        if ((kernel_size & 1) == 0) ++kernel_size;
        cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
        cv::dilate(image_, result.impl_->image_, element, cv::Point(-1,-1), std::max(1, iterations));
        return result;
    }

    ImageT<Pixel> Threshold(double thresh, double maxval, int type) const {
        ImageT<Pixel> result;
        cv::Mat result_mat;
        cv::threshold(image_, result_mat, thresh, maxval, type);
        result.impl_->image_ = result_mat;
        return result;
    }

    ImageT<Pixel> AbsDiff(const typename ImageT<Pixel>::Impl& other) const {
        ImageT<Pixel> result;
        cv::absdiff(image_, other.image_, result.impl_->image_);
        return result;
    }

    ImageT<Pixel> MeanChannels() const {
        if (image_.channels() == 1) return Clone();
        ImageT<Pixel> result;
        std::vector<cv::Mat> ch;
        cv::split(image_, ch);
        CV_Assert(!ch.empty());
        cv::Mat acc;
        ch[0].convertTo(acc, CV_32F);
        for (size_t i = 1; i < ch.size(); ++i) {
            cv::Mat tmp;
            ch[i].convertTo(tmp, CV_32F);
            acc += tmp;
        }
        acc *= (1.0f / static_cast<float>(ch.size()));
        acc.convertTo(result.impl_->image_, std::is_same<Pixel,float>::value ? CV_32F : CV_8U);
        return result;
    }

    ImageT<Pixel> Blend(const typename ImageT<Pixel>::Impl& other, const typename ImageT<uint8_t>::Impl& mask) const {
        ImageT<Pixel> result;
        // Ensure same size/channels between A and B
        CV_Assert(image_.size() == other.image_.size());
        CV_Assert(image_.channels() == other.image_.channels());
        const int cn = image_.channels();
        // Build single-channel 8U mask (require 1 channel)
        CV_Assert(mask.MatRef().channels() == 1);
        cv::Mat M1 = mask.MatRef();

        if (std::is_same<Pixel,float>::value) {
            // Float blend: out = m*A + (1-m)*B, m in [0,1]
            cv::Mat A32, B32, M32;
            if (image_.depth() == CV_32F) A32 = image_; else image_.convertTo(A32, CV_32F);
            if (other.image_.depth() == CV_32F) B32 = other.image_; else other.image_.convertTo(B32, CV_32F);
            M1.convertTo(M32, CV_32F, 1.0 / 255.0);
            if (A32.channels() == 1 && cn == 3) cv::cvtColor(A32, A32, cv::COLOR_GRAY2BGR);
            if (B32.channels() == 1 && cn == 3) cv::cvtColor(B32, B32, cv::COLOR_GRAY2BGR);

            std::vector<cv::Mat> mv(static_cast<size_t>(cn), M32);
            cv::Mat Mx; if (cn == 1) Mx = M32; else cv::merge(mv, Mx);
            cv::Mat inv; cv::subtract(cv::Scalar::all(1.0), Mx, inv);
            cv::Mat p1, p2, out; cv::multiply(Mx, A32, p1); cv::multiply(inv, B32, p2); cv::add(p1, p2, out);
            out.convertTo(result.impl_->image_, CV_32F);
            return result;
        } else {
            // 8U reference path
            cv::Mat A = image_, B = other.image_;
            if (A.channels() == 1 && cn == 3) cv::cvtColor(A, A, cv::COLOR_GRAY2BGR);
            if (B.channels() == 1 && cn == 3) cv::cvtColor(B, B, cv::COLOR_GRAY2BGR);

            result.impl_->image_.create(A.rows, A.cols, A.type());
            const int width = A.cols; const int height = A.rows;
            for (int y = 0; y < height; ++y) {
                const uchar* pa = A.ptr<uchar>(y);
                const uchar* pb = B.ptr<uchar>(y);
                const uchar* pm = M1.ptr<uchar>(y);
                uchar* po = result.impl_->image_.template ptr<uchar>(y);
                for (int x = 0; x < width; ++x) {
                    int m = pm[x]; int inv = 255 - m;
                    if (cn == 1) {
                        int a0 = pa[x]; int b0 = pb[x]; int out0 = (m * a0 + inv * b0 + 127) / 255;
                        po[x] = static_cast<uchar>(out0);
                    } else {
                        int idx = x * 3;
                        for (int c = 0; c < 3; ++c) {
                            int a0 = pa[idx + c]; int b0 = pb[idx + c]; int out0 = (m * a0 + inv * b0 + 127) / 255;
                            po[idx + c] = static_cast<uchar>(out0);
                        }
                    }
                }
            }
            return result;
        }
    }

    explicit Impl(void* internal_data) {
        image_ = *static_cast<cv::Mat*>(internal_data);
    }

    void Print(std::ostream& os) const {
        // Unified print using cv::format for simplicity across depths
        os << cv::format(image_, cv::Formatter::FMT_PYTHON) << "\n";
        os << "Size(H x W x C): " << image_.rows << " x " << image_.cols << " x "
           << image_.channels() << "\n";
    }

private:
    cv::Mat image_;
};

}  // namespace inspirecv

#endif  // INSPIRECV_IMPL_OPENCV_IMAGE_OPENCV_H
