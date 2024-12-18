#include "inspirecv/core/processor.h"
#include "processor_impl.h"

namespace inspirecv {

Processor::Processor() : impl_(std::make_unique<Impl>()) {}

Processor::~Processor() = default;

bool Processor::PreallocateBuffer(int width, int height, int channels) {
    return impl_->PreallocateBuffer(width, height, channels);
}

bool Processor::Resize(const Image& src, Image& dst, int dst_width, int dst_height) {
    return impl_->Resize(src, dst, dst_width, dst_height);
}

bool Processor::Resize(const uint8_t* src_data, int src_width, int src_height, int channels,
                       uint8_t** dst_data, int dst_width, int dst_height) {
    return impl_->Resize(src_data, src_width, src_height, channels, dst_data, dst_width,
                         dst_height);
}

bool Processor::SwapColor(const Image& src, Image& dst) {
    return impl_->SwapColor(src, dst);
}

bool Processor::Padding(const Image& src, Image& dst, int padding_left, int padding_right,
                        int padding_top, int padding_bottom) {
    return impl_->Padding(src, dst, padding_left, padding_right, padding_top, padding_bottom);
}

std::unique_ptr<Processor> Processor::Create() {
    return std::make_unique<Processor>();
}

}  // namespace inspirecv
