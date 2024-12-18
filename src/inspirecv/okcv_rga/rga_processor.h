#ifndef INSPIRECV_OKCV_RGA_RGA_PROCESSOR_H
#define INSPIRECV_OKCV_RGA_RGA_PROCESSOR_H

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
#include <memory>
#include <unordered_map>
#include "im2d.hpp"
#include "RgaUtils.h"
#include "utils.h"
#include "dma_alloc.h"
#include "inspirecv/logging.h"
#include "inspirecv/time_spend.h"

namespace inspirecv {

class RGAProcessor {
private:
    // RGA缓冲区结构
    struct RGABuffer {
        int width{0};
        int height{0};
        int channels{0};
        int dma_fd{-1};
        void* virtual_addr{nullptr};
        size_t buffer_size{0};
        rga_buffer_handle_t handle{0};
        rga_buffer_t buffer{};

        bool Allocate(int w, int h, int c) {
            width = w;
            height = h;
            channels = c;
            buffer_size = width * height * channels;

            // 分配DMA缓冲区
            int ret = dma_buf_alloc(RV1106_CMA_HEAP_PATH, buffer_size, &dma_fd, &virtual_addr);
            if (ret < 0) {
                INSPIRECV_LOG(ERROR) << "Failed to allocate DMA buffer";
                return false;
            }

            // 导入缓冲区
            handle = importbuffer_fd(dma_fd, buffer_size);
            if (handle == 0) {
                INSPIRECV_LOG(ERROR) << "Failed to import buffer";
                Release();
                return false;
            }

            // 包装缓冲区
            buffer = wrapbuffer_handle(handle, width, height, RK_FORMAT_RGB_888);
            return true;
        }

        void Release() {
            if (handle) {
                releasebuffer_handle(handle);
                handle = 0;
            }
            if (dma_fd >= 0) {
                dma_buf_free(buffer_size, &dma_fd, virtual_addr);
                dma_fd = -1;
                virtual_addr = nullptr;
            }
        }

        ~RGABuffer() {
            Release();
        }
    };

    // 缓存键结构
    struct BufferKey {
        int width;
        int height;
        int channels;

        bool operator==(const BufferKey& other) const {
            return width == other.width && height == other.height && channels == other.channels;
        }
    };

    // 缓存键的哈希函数
    struct BufferKeyHash {
        std::size_t operator()(const BufferKey& key) const {
            return std::hash<int>()(key.width) ^ (std::hash<int>()(key.height) << 1) ^
                   (std::hash<int>()(key.channels) << 2);
        }
    };

public:
    RGAProcessor() = default;
    ~RGAProcessor() = default;

    // 禁用拷贝
    RGAProcessor(const RGAProcessor&) = delete;
    RGAProcessor& operator=(const RGAProcessor&) = delete;

    bool Resize(const uint8_t* src_data, int src_width, int src_height, int channels,
                uint8_t** dst_data, int dst_width, int dst_height) {
        // 1. Get or create source buffer
        BufferKey src_key{src_width, src_height, channels};
        auto& src_buffer = GetOrCreateBuffer(src_key);

        // 2. Get or create destination buffer
        BufferKey dst_key{dst_width, dst_height, channels};
        auto& dst_buffer = GetOrCreateBuffer(dst_key);

        // 3. Copy source data to RGA buffer
        memcpy(src_buffer.virtual_addr, src_data, src_buffer.buffer_size);

        dma_sync_cpu_to_device(src_buffer.dma_fd);
        dma_sync_cpu_to_device(dst_buffer.dma_fd);

        // 4. Execute RGA resize
        int ret = imcheck(src_buffer.buffer, dst_buffer.buffer, {}, {});
        if (IM_STATUS_NOERROR != ret) {
            INSPIRECV_LOG(ERROR) << "RGA parameter check failed: " << imStrError((IM_STATUS)ret);
            return false;
        }

        ret = imresize(src_buffer.buffer, dst_buffer.buffer);

        if (ret != IM_STATUS_SUCCESS) {
            INSPIRECV_LOG(ERROR) << "RGA resize failed: " << imStrError((IM_STATUS)ret);
            return false;
        }

        // 5. Return pointer to destination buffer
        *dst_data = static_cast<uint8_t*>(dst_buffer.virtual_addr);

        return true;
    }

    bool Resize(const okcv::Image<uint8_t>& src, okcv::Image<uint8_t>& dst, int dst_width,
                int dst_height) {
        // 1. 获取或创建源缓冲区
        BufferKey src_key{src.Width(), src.Height(), src.Channels()};
        auto& src_buffer = GetOrCreateBuffer(src_key);

        // 2. 获取或创建目标缓冲区
        BufferKey dst_key{dst_width, dst_height, src.Channels()};
        auto& dst_buffer = GetOrCreateBuffer(dst_key);

        // 3. 将源数据复制到RGA缓冲区
        memcpy(src_buffer.virtual_addr, src.Data(), src_buffer.buffer_size);

        dma_sync_cpu_to_device(src_buffer.dma_fd);
        dma_sync_cpu_to_device(dst_buffer.dma_fd);

        // 4. 执行RGA缩放
        int ret = imcheck(src_buffer.buffer, dst_buffer.buffer, {}, {});
        if (IM_STATUS_NOERROR != ret) {
            INSPIRECV_LOG(ERROR) << "RGA parameter check failed: " << imStrError((IM_STATUS)ret);
            return false;
        }

        ret = imresize(src_buffer.buffer, dst_buffer.buffer);

        if (ret != IM_STATUS_SUCCESS) {
            INSPIRECV_LOG(ERROR) << "RGA resize failed: " << imStrError((IM_STATUS)ret);
            return false;
        }

        // 5. 更新目标图像
        dma_sync_device_to_cpu(dst_buffer.dma_fd);
        dst.Reset(dst_width, dst_height, src.Channels(), (uint8_t*)dst_buffer.virtual_addr, false);
        // memcpy(dst.Data(), dst_buffer.virtual_addr, dst_buffer.buffer_size);
        dma_sync_cpu_to_device(dst_buffer.dma_fd);

        return true;
    }

    bool SwapColor(const okcv::Image<uint8_t>& src, okcv::Image<uint8_t>& dst) {
        // 1. Get or create source buffer
        BufferKey src_key{src.Width(), src.Height(), src.Channels()};
        auto& src_buffer = GetOrCreateBuffer(src_key);

        // 2. Get or create destination buffer with same dimensions
        BufferKey dst_key{src.Width(), src.Height(), src.Channels()};
        auto& dst_buffer = GetOrCreateBuffer(dst_key);

        // 3. Copy source data to RGA buffer
        memcpy(src_buffer.virtual_addr, src.Data(), src_buffer.buffer_size);

        dma_sync_cpu_to_device(src_buffer.dma_fd);
        dma_sync_cpu_to_device(dst_buffer.dma_fd);

        // 4. Execute RGA color conversion
        int ret = imcheck(src_buffer.buffer, dst_buffer.buffer, {}, {});
        if (IM_STATUS_NOERROR != ret) {
            INSPIRECV_LOG(ERROR) << "RGA parameter check failed: " << imStrError((IM_STATUS)ret);
            return false;
        }

        ret =
          imcvtcolor(src_buffer.buffer, dst_buffer.buffer, RK_FORMAT_RGB_888, RK_FORMAT_BGR_888);
        if (ret != IM_STATUS_SUCCESS) {
            INSPIRECV_LOG(ERROR) << "RGA color conversion failed: " << imStrError((IM_STATUS)ret);
            return false;
        }

        // 5. Update destination image
        dma_sync_device_to_cpu(dst_buffer.dma_fd);
        dst.Reset(src.Width(), src.Height(), src.Channels(), (uint8_t*)dst_buffer.virtual_addr,
                  false);
        dma_sync_cpu_to_device(dst_buffer.dma_fd);

        return true;
    }

    bool Padding(const okcv::Image<uint8_t>& src, okcv::Image<uint8_t>& dst, int top, int bottom,
                 int left, int right) {
        // 1. Get or create source buffer
        BufferKey src_key{src.Width(), src.Height(), src.Channels()};
        auto& src_buffer = GetOrCreateBuffer(src_key);

        // 2. Get or create destination buffer with padded dimensions
        int dst_width = src.Width() + left + right;
        int dst_height = src.Height() + top + bottom;
        BufferKey dst_key{dst_width, dst_height, src.Channels()};
        auto& dst_buffer = GetOrCreateBuffer(dst_key);

        // 3. Copy source data to RGA buffer
        memcpy(src_buffer.virtual_addr, src.Data(), src_buffer.buffer_size);

        dma_sync_cpu_to_device(src_buffer.dma_fd);
        dma_sync_cpu_to_device(dst_buffer.dma_fd);

        // 4. Execute RGA padding operation
        int ret = imcheck(src_buffer.buffer, dst_buffer.buffer, {}, {});
        if (IM_STATUS_NOERROR != ret) {
            INSPIRECV_LOG(ERROR) << "RGA parameter check failed: " << imStrError((IM_STATUS)ret);
            return false;
        }

        ret = immakeBorder(src_buffer.buffer, dst_buffer.buffer, top, bottom, left, right,
                           IM_BORDER_REFLECT);
        if (ret != IM_STATUS_SUCCESS) {
            INSPIRECV_LOG(ERROR) << "RGA padding failed: " << imStrError((IM_STATUS)ret);
            return false;
        }

        // 5. Update destination image
        dma_sync_device_to_cpu(dst_buffer.dma_fd);
        dst.Reset(dst_width, dst_height, src.Channels(), (uint8_t*)dst_buffer.virtual_addr, false);
        dma_sync_cpu_to_device(dst_buffer.dma_fd);

        return true;
    }

    void PreallocateBuffer(int width, int height, int channels) {
        BufferKey key{width, height, channels};
        GetOrCreateBuffer(key);
    }

    void ClearBuffers() {
        buffer_cache_.clear();
    }

private:
    RGABuffer& GetOrCreateBuffer(const BufferKey& key) {
        auto it = buffer_cache_.find(key);
        if (it == buffer_cache_.end()) {
            auto& buffer = buffer_cache_[key];
            if (!buffer.Allocate(key.width, key.height, key.channels)) {
                INSPIRECV_LOG(ERROR) << "Failed to allocate RGA buffer";
                throw std::runtime_error("RGA buffer allocation failed");
            }
            return buffer;
        }
        return it->second;
    }

private:
    std::unordered_map<BufferKey, RGABuffer, BufferKeyHash> buffer_cache_;
};

}  // namespace inspirecv

#endif  // INSPIRECV_OKCV_RGA_RGA_PROCESSOR_H