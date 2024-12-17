#include "image_rga.h"

namespace okcv {

template <typename D>
Image<D>::Image(Image &&that) {
    data_ = std::move(that.data_);
    height_ = that.height_;
    width_ = that.width_;
    channels_ = that.channels_;

    that.height_ = 0;
    that.width_ = 0;
}

template <typename D>
Image<D> &Image<D>::operator=(Image &&that) {
    data_ = std::move(that.data_);
    height_ = that.height_;
    width_ = that.width_;
    channels_ = that.channels_;

    that.height_ = 0;
    that.width_ = 0;
    return *this;
}

template <typename D>
void Image<D>::Reset() {
    height_ = 0;
    width_ = 0;
    channels_ = 0;
    data_ = nullptr;
}

}  // namespace okcv