#include "image_rga.h"
#include <cmath>
#include <string>
#include <utility>
#include <cstring>
#include <sstream>
#include <algorithm>
#include <vector>
#include "check.h"
#include <fstream>
#include "../okcv/io/stb_warpper.h"
#include "inspirecv/time_spend.h"
namespace okcv {

ImageRga::ImageRga(ImageRga&& image) = default;

ImageRga::~ImageRga() {
    ReleaseRgaBuffer();
    ReleaseDestBuffer();
}

void ImageRga::Reset() {
    // 释放 RGA 资源
    ReleaseRgaBuffer();
    // 重置 Image 对象
    image_->Reset();
}

void ImageRga::Reset(int width, int height, int channels, const uint8_t* data) {
    // 检查是否需要重新分配
    bool need_realloc = (width * height * channels != image_->DataSize());

    if (need_realloc) {
        ReleaseRgaBuffer();
    }

    // 重置 Image 对象
    image_->Reset(width, height, channels, data);

    // 如果需要重新分配，或者 RGA 缓冲区尚未分配
    if (need_realloc || rga_dma_fd_ < 0) {
        AllocateRgaBuffer();
    }

    // 如果有输入数据，同步到 RGA 缓冲区
    if (data) {
        SyncToRga();
    }
}

void ImageRga::Read(const std::string& filename, int channels) {
    // 先读取到 Image 对象中
    image_->Read(filename, channels);

    // 释放旧的 RGA 缓冲区（如果有的话）
    ReleaseRgaBuffer();

    // 分配新的 RGA 缓冲区
    AllocateRgaBuffer();

    // 将数据同步到 RGA 缓冲区
    SyncToRga();
}

void ImageRga::Write(const std::string& filename) const {
    // 确保 RGA 缓冲区的数据已同步到 Image 对象
    if (rga_virtual_addr_ && rga_dma_fd_ >= 0) {
        const_cast<ImageRga*>(this)->SyncFromRga();
    }

    // for (size_t i = 0; i < image_->DataSize(); i++) {
    //     std::cout << image_->Data()[i] << " ";
    // }
    // std::cout << std::endl;

    // 写入文件
    image_->Write(filename);
}

void ImageRga::InitRgaBuffer() {
    // 只有当图像有效时才初始化 RGA 缓冲区
    if (image_->Width() > 0 && image_->Height() > 0) {
        AllocateRgaBuffer();
    }
}

void ImageRga::ReleaseRgaBuffer() {
    // 释放 RGA 句柄
    if (rga_handle_) {
        releasebuffer_handle(rga_handle_);
        rga_handle_ = 0;
    }

    // 释放 DMA 缓冲区
    if (rga_dma_fd_ >= 0) {
        int buf_size = image_->DataSize();
        dma_buf_free(buf_size, &rga_dma_fd_, rga_virtual_addr_);
        rga_dma_fd_ = -1;
        rga_virtual_addr_ = nullptr;
    }

    // 清空 RGA buffer 描述符
    memset(&rga_buffer_, 0, sizeof(rga_buffer_));
}

bool ImageRga::SyncToRga() {
    if (!rga_virtual_addr_ || rga_dma_fd_ < 0) {
        return false;
    }

    std::memcpy(rga_virtual_addr_, image_->Data(), image_->DataSize());
    return true;
}

bool ImageRga::SyncFromRga() {
    if (!rga_virtual_addr_ || rga_dma_fd_ < 0) {
        return false;
    }

    std::memcpy(image_->Data(), rga_virtual_addr_, image_->DataSize());
    return true;
}

void ImageRga::AllocateRgaBuffer() {
    int buf_size = image_->DataSize();

    // 分配DMA缓冲区
    int ret =
      dma_buf_alloc(RV1106_CMA_HEAP_PATH, buf_size, &rga_dma_fd_, (void**)&rga_virtual_addr_);
    if (ret < 0) {
        INSPIRECV_LOG(ERROR) << "Failed to allocate RGA buffer";
        return;
    }

    // 导入缓冲区
    rga_handle_ = importbuffer_fd(rga_dma_fd_, buf_size);
    if (rga_handle_ == 0) {
        INSPIRECV_LOG(ERROR) << "Failed to import RGA buffer";
        dma_buf_free(buf_size, &rga_dma_fd_, rga_virtual_addr_);
        rga_dma_fd_ = -1;
        rga_virtual_addr_ = nullptr;
        return;
    }

    // 包装缓冲区
    rga_buffer_ =
      wrapbuffer_handle(rga_handle_, image_->Width(), image_->Height(), RK_FORMAT_RGB_888);
}

void ImageRga::ReleaseDestBuffer() {
    if (dest_buffer_) {
        if (dest_buffer_->handle) {
            releasebuffer_handle(dest_buffer_->handle);
            dest_buffer_->handle = 0;
        }

        if (dest_buffer_->dma_fd >= 0) {
            int buf_size = dest_buffer_->width * dest_buffer_->height * dest_buffer_->channels;
            dma_buf_free(buf_size, &dest_buffer_->dma_fd, dest_buffer_->virtual_addr);
            dest_buffer_->dma_fd = -1;
            dest_buffer_->virtual_addr = nullptr;
        }

        memset(&dest_buffer_->buffer, 0, sizeof(rga_buffer_t));
        dest_buffer_.reset();
    }
}

bool ImageRga::EnsureDestBuffer(int width, int height, int channels) {
    // 检查是否需要重新分配
    if (dest_buffer_ && dest_buffer_->width == width && dest_buffer_->height == height &&
        dest_buffer_->channels == channels) {
        return true;  // 已经有合适的缓冲区
    }

    // 释放旧的缓冲区
    ReleaseDestBuffer();

    // 创建新的缓冲区
    dest_buffer_ = std::make_unique<DestBuffer>();
    dest_buffer_->width = width;
    dest_buffer_->height = height;
    dest_buffer_->channels = channels;

    int buf_size = width * height * channels;

    // 分配DMA缓冲区
    int ret = dma_buf_alloc(RV1106_CMA_HEAP_PATH, buf_size, &dest_buffer_->dma_fd,
                            (void**)&dest_buffer_->virtual_addr);
    if (ret < 0) {
        INSPIRECV_LOG(ERROR) << "Failed to allocate destination RGA buffer";
        ReleaseDestBuffer();
        return false;
    }

    // 导入缓冲区
    dest_buffer_->handle = importbuffer_fd(dest_buffer_->dma_fd, buf_size);
    if (dest_buffer_->handle == 0) {
        INSPIRECV_LOG(ERROR) << "Failed to import destination RGA buffer";
        ReleaseDestBuffer();
        return false;
    }

    // 包装缓冲区
    dest_buffer_->buffer =
      wrapbuffer_handle(dest_buffer_->handle, width, height, RK_FORMAT_RGB_888);
    return true;
}

void ImageRga::PreallocateDestBuffer(int width, int height, int channels) {
    auto ret = EnsureDestBuffer(width, height, channels);
    if (!ret) {
        INSPIRECV_LOG(ERROR) << "Failed to ensure destination buffer";
    }
}

bool ImageRga::ResizeRga(int dst_width, int dst_height, ImageRga& dst) const {
    int ret = 0;

    // 1. 确保源图像已经同步到RGA缓冲区
    if (!const_cast<ImageRga*>(this)->SyncToRga()) {
        INSPIRECV_LOG(ERROR) << "Failed to sync source image to RGA";
        return false;
    }

    // 2. 确保目标缓冲区准备就绪
    if (!const_cast<ImageRga*>(this)->EnsureDestBuffer(dst_width, dst_height, image_->Channels())) {
        INSPIRECV_LOG(ERROR) << "Failed to ensure destination buffer";
        return false;
    }

    // 3. 检查参数有效性
    ret = imcheck(rga_buffer_, dest_buffer_->buffer, {}, {});
    if (IM_STATUS_NOERROR != ret) {
        INSPIRECV_LOG(ERROR) << "RGA parameter check failed: " << imStrError((IM_STATUS)ret);
        return false;
    }

    // 4. 执行RGA缩放
    inspirecv::TimeSpend time_spend("RGA Resize inner");
    time_spend.Start();
    ret = imresize(rga_buffer_, dest_buffer_->buffer);
    time_spend.Stop();
    std::cout << time_spend << std::endl;
    if (ret != IM_STATUS_SUCCESS) {
        INSPIRECV_LOG(ERROR) << "Failed to resize image: " << imStrError((IM_STATUS)ret);
        return false;
    }

    // 5. 更新目标图像
    dst.Reset(dst_width, dst_height, image_->Channels());

    // 6. 将结果从RGA缓冲区复制到目标图像
    std::memcpy(dst.image_->Data(), dest_buffer_->virtual_addr,
                dst_width * dst_height * image_->Channels());

    // dst.Write("input_resize_rga.png");

    // Write raw binary data to file
    std::string bin_filename = "rga_output.bin";
    std::ofstream outfile(bin_filename, std::ios::binary);
    if (!outfile) {
        INSPIRECV_LOG(ERROR) << "Failed to open binary file for writing: " << bin_filename;
        return false;
    }
    outfile.write(reinterpret_cast<const char*>(dest_buffer_->virtual_addr),
                  dst_width * dst_height * image_->Channels());
    outfile.close();

    return true;
}

}  // namespace okcv