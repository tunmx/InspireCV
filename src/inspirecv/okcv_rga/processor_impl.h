#ifndef INSPIRECV_OKCV_RGA_PROCESSOR_IMPL_H
#define INSPIRECV_OKCV_RGA_PROCESSOR_IMPL_H

#include "inspirecv/core/processor.h"
#include "rga_processor.h"

namespace inspirecv {

class Processor::Impl {
public:
    Impl() = default;
    ~Impl() = default;

    bool PreallocateBuffer(int width, int height, int channels) {
        rga_processor_.PreallocateBuffer(width, height, channels);
        return true;
    }

    bool Resize(const uint8_t* src_data, int src_width, int src_height, int channels,
                uint8_t** dst_data, int dst_width, int dst_height) {
        return rga_processor_.Resize(src_data, src_width, src_height, channels, dst_data, dst_width,
                                     dst_height);
    }

    bool Resize(const Image& src, Image& dst, int dst_width, int dst_height) {
        // Convert inspirecv::Image to okcv::Image
        okcv::Image<uint8_t> okcv_src;
        TimeSpend time_spend("RGA Resize create okcv::Image");
        time_spend.Start();
        okcv_src.Reset(src.Width(), src.Height(), src.Channels(), src.Data(), false);
        time_spend.Stop();
        // std::cout << time_spend << std::endl;

        okcv::Image<uint8_t> okcv_dst;

        bool success = rga_processor_.Resize(okcv_src, okcv_dst, dst_width, dst_height);
        if (!success) {
            return false;
        }

        // Convert back to inspirecv::Image
        dst = Image::Create(dst_width, dst_height, okcv_dst.Channels(), okcv_dst.Data());
        return true;
    }

    bool SwapColor(const Image& src, Image& dst) {
        // Convert inspirecv::Image to okcv::Image
        okcv::Image<uint8_t> okcv_src;
        okcv_src.Reset(src.Width(), src.Height(), src.Channels(), src.Data(), false);
        okcv::Image<uint8_t> okcv_dst;
        auto ret = rga_processor_.SwapColor(okcv_src, okcv_dst);
        if (!ret) {
            return false;
        }
        dst =
          Image::Create(okcv_dst.Width(), okcv_dst.Height(), okcv_dst.Channels(), okcv_dst.Data());
        return true;
    }

    bool Padding(const Image& src, Image& dst, int padding_left, int padding_right, int padding_top,
                 int padding_bottom) {
        // Convert inspirecv::Image to okcv::Image
        okcv::Image<uint8_t> okcv_src;
        okcv_src.Reset(src.Width(), src.Height(), src.Channels(), src.Data(), false);
        okcv::Image<uint8_t> okcv_dst;
        auto ret = rga_processor_.Padding(okcv_src, okcv_dst, padding_left, padding_right,
                                          padding_top, padding_bottom);
        if (!ret) {
            return false;
        }
        dst =
          Image::Create(okcv_dst.Width(), okcv_dst.Height(), okcv_dst.Channels(), okcv_dst.Data());
        return true;
    }

private:
    RGAProcessor rga_processor_;
};

}  // namespace inspirecv

#endif  // INSPIRECV_OKCV_RGA_PROCESSOR_IMPL_H
