#include <memory>
#include <cmath>
#include "impl.h"
#include "logging.h"

namespace inspirecv {

template<typename Pixel>
ImageT<Pixel>::~ImageT() = default;

template<typename Pixel>
ImageT<Pixel>::ImageT() {
    impl_ = std::make_unique<Impl>(0, 0, 0);
}

template<typename Pixel>
ImageT<Pixel>::ImageT(ImageT&& other) noexcept = default;

template<typename Pixel>
ImageT<Pixel>& ImageT<Pixel>::operator=(ImageT&& other) noexcept = default;

template<typename Pixel>
ImageT<Pixel>::ImageT(int width, int height, int channels, const Pixel* data, bool copy_data) {
    impl_ = std::make_unique<Impl>(width, height, channels, data, copy_data);
}

template<typename Pixel>
void ImageT<Pixel>::Reset(int width, int height, int channels, const Pixel* data) {
    impl_->Reset(width, height, channels, data);
}

template<typename Pixel>
ImageT<Pixel> ImageT<Pixel>::Clone() const {
    ImageT<Pixel> new_image = impl_->Clone();
    return new_image;
}

template<typename Pixel>
int ImageT<Pixel>::Width() const {
    return impl_->Width();
}

template<typename Pixel>
int ImageT<Pixel>::Height() const {
    return impl_->Height();
}

template<typename Pixel>
int ImageT<Pixel>::Channels() const {
    return impl_->Channels();
}

template<typename Pixel>
bool ImageT<Pixel>::Empty() const {
    return impl_->Empty();
}

template<typename Pixel>
const Pixel* ImageT<Pixel>::Data() const {
    return impl_->Data();
}

template<typename Pixel>
void* ImageT<Pixel>::GetInternalImage() const {
    return impl_->GetInternalImage();
}

template<typename Pixel>
bool ImageT<Pixel>::Read(const std::string& filename, int channels) {
    return impl_->Read(filename, channels);
}

template<typename Pixel>
bool ImageT<Pixel>::Write(const std::string& filename) const {
    return impl_->Write(filename);
}

template<typename Pixel>
void ImageT<Pixel>::Show(const std::string& window_name, int delay) const {
    impl_->Show(window_name, delay);
}

template<typename Pixel>
void ImageT<Pixel>::Fill(double value) {
    impl_->Fill(value);
}

template<typename Pixel>
ImageT<Pixel> ImageT<Pixel>::Mul(double scale) const {
    return impl_->Mul(scale);
}

template<typename Pixel>
ImageT<Pixel> ImageT<Pixel>::Add(double value) const {
    return impl_->Add(value);
}

template<typename Pixel>
ImageT<Pixel> ImageT<Pixel>::Resize(int width, int height, bool use_linear) const {
    return impl_->Resize(width, height, use_linear);
}

template<typename Pixel>
ImageT<Pixel> ImageT<Pixel>::Crop(const Rect<int>& rect) const {
    return impl_->Crop(rect);
}

template<typename Pixel>
ImageT<Pixel> ImageT<Pixel>::WarpAffine(const TransformMatrix& matrix, int width, int height) const {
    return impl_->WarpAffine(matrix, width, height);
}

template<typename Pixel>
ImageT<Pixel> ImageT<Pixel>::Rotate90() const {
    return impl_->Rotate90();
}

template<typename Pixel>
ImageT<Pixel> ImageT<Pixel>::Rotate180() const {
    return impl_->Rotate180();
}

template<typename Pixel>
ImageT<Pixel> ImageT<Pixel>::Rotate270() const {
    return impl_->Rotate270();
}

template<typename Pixel>
ImageT<Pixel> ImageT<Pixel>::SwapRB() const {
    return impl_->SwapRB();
}

template<typename Pixel>
ImageT<Pixel> ImageT<Pixel>::FlipHorizontal() const {
    return impl_->FlipHorizontal();
}

template<typename Pixel>
ImageT<Pixel> ImageT<Pixel>::FlipVertical() const {
    return impl_->FlipVertical();
}

template<typename Pixel>
ImageT<Pixel> ImageT<Pixel>::Pad(int top, int bottom, int left, int right, const std::vector<double>& color) const {
    return impl_->Pad(top, bottom, left, right, color);
}

template<typename Pixel>
void ImageT<Pixel>::DrawLine(const Point<int>& p1, const Point<int>& p2, const std::vector<double>& color,
                     int thickness) {
    impl_->DrawLine(p1, p2, color, thickness);
}

template<typename Pixel>
void ImageT<Pixel>::DrawRect(const Rect<int>& rect, const std::vector<double>& color, int thickness) {
    impl_->DrawRect(rect, color, thickness);
}

template<typename Pixel>
void ImageT<Pixel>::DrawCircle(const Point<int>& center, int radius, const std::vector<double>& color,
                       int thickness) {
    impl_->DrawCircle(center, radius, color, thickness);
}

template<typename Pixel>
void ImageT<Pixel>::Fill(const Rect<int>& rect, const std::vector<double>& color) {
    impl_->Fill(rect, color);
}

template<typename Pixel>
ImageT<Pixel> ImageT<Pixel>::GaussianBlur(int kernel_size, double sigma) const {
    return impl_->GaussianBlur(kernel_size, sigma);
}

template<typename Pixel>
ImageT<Pixel> ImageT<Pixel>::Erode(int kernel_size, int iterations) const {
    return impl_->Erode(kernel_size, iterations);
}

template<typename Pixel>
ImageT<Pixel> ImageT<Pixel>::Dilate(int kernel_size, int iterations) const {
    return impl_->Dilate(kernel_size, iterations);
}

template<typename Pixel>
ImageT<Pixel> ImageT<Pixel>::Threshold(double thresh, double maxval, int type) const {
    return impl_->Threshold(thresh, maxval, type);
}

template<typename Pixel>
ImageT<Pixel> ImageT<Pixel>::ToGray() const {
    return impl_->ToGray();
}

template<typename Pixel>
ImageT<Pixel> ImageT<Pixel>::AbsDiff(const ImageT& other) const {
    return impl_->AbsDiff(*other.impl_);
}

template<typename Pixel>
ImageT<Pixel> ImageT<Pixel>::MeanChannels() const {
    return impl_->MeanChannels();
}

template<typename Pixel>
ImageT<Pixel> ImageT<Pixel>::Blend(const ImageT& other, const ImageT<uint8_t>& mask) const {
    return impl_->Blend(*other.impl_, *mask.impl_);
}

template<typename Pixel>
ImageT<Pixel> ImageT<Pixel>::Create(int width, int height, int channels, const Pixel* data, bool copy_data) {
    return ImageT(width, height, channels, data, copy_data);
}

template<typename Pixel>
ImageT<Pixel> ImageT<Pixel>::Create() {
    return ImageT(0, 0, 0);
}

template<typename Pixel>
ImageT<Pixel> ImageT<Pixel>::Create(const std::string& filename, int channels) {
    ImageT image;
    bool success = image.Read(filename, channels);
    if (!success) {
        INSPIRECV_LOG(WARN) << "Failed to read image from " << filename << " with channels "
                            << channels << " : " << (success ? "success" : "failure");
    }
    return image;
}

template<typename Pixel>
ImageT<Pixel>::ImageT(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

template<typename Pixel>
std::ostream& operator<<(std::ostream& os, const ImageT<Pixel>& image) {
    image.impl_->Print(os);
    return os;
}

// Explicit instantiation for common pixel types
template class ImageT<uint8_t>;
template std::ostream& operator<< <uint8_t>(std::ostream& os, const ImageT<uint8_t>& image);

template class ImageT<float>;
template std::ostream& operator<< <float>(std::ostream& os, const ImageT<float>& image);

}  // namespace inspirecv
