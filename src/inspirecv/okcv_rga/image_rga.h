#ifndef OKCV_RGA_IMAGE_RGA_H_
#define OKCV_RGA_IMAGE_RGA_H_

#include <okcv/okcv.h>
#include <linux/stddef.h>
#include <iostream>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>
#include "im2d.hpp"
#include "RgaUtils.h"
#include "utils.h"
#include "dma_alloc.h"

namespace okcv {

class OKCV_API ImageRga {
public:
    ImageRga() {
        image_ = std::make_shared<Image<uint8_t>>();
    }

    ~ImageRga();

    ImageRga(ImageRga&& image);
    ImageRga& operator=(ImageRga&& image);

    inline bool Empty() const {
        return image_->Empty();
    }

    inline int Width() const {
        return image_->Width();
    }

    inline int Height() const {
        return image_->Height();
    }

    inline int Channels() const {
        return image_->Channels();
    }

    // 基本操作接口
    void Reset();
    void Reset(int width, int height, int channels, const uint8_t* data = nullptr);
    void Read(const std::string& filename, int channels = 3);
    void Write(const std::string& filename) const;

    // 图像变换
    bool ResizeRga(int dst_width, int dst_height, ImageRga& dst) const;

    // 预分配目标缓冲区
    void PreallocateDestBuffer(int width, int height, int channels);

private:
    void InitRgaBuffer();
    void AllocateRgaBuffer();
    void ReleaseRgaBuffer();
    bool SyncToRga();    // 将Image数据同步到RGA缓冲区
    bool SyncFromRga();  // 将RGA缓冲区数据同步到Image

    // 获取或创建目标缓冲区
    bool EnsureDestBuffer(int width, int height, int channels);
    void ReleaseDestBuffer();

private:
    std::shared_ptr<Image<uint8_t>> image_;
    rga_buffer_t rga_buffer_{};           // RGA缓冲区描述符
    rga_buffer_handle_t rga_handle_{0};   // RGA缓冲区句柄
    int rga_dma_fd_{-1};                  // DMA文件描述符
    uint8_t* rga_virtual_addr_{nullptr};  // RGA缓冲区虚拟地址

    // RGA destination buffer (lazy initialization)
    struct DestBuffer {
        rga_buffer_t buffer{};
        rga_buffer_handle_t handle{0};
        int dma_fd{-1};
        uint8_t* virtual_addr{nullptr};
        int width{0};
        int height{0};
        int channels{0};
    };
    std::unique_ptr<DestBuffer> dest_buffer_;
};

}  // namespace okcv

#endif  // OKCV_RGA_IMAGE_RGA_H_