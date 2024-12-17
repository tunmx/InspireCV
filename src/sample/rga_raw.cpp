/*
 * Copyright (C) 2022  Rockchip Electronics Co., Ltd.
 * Authors:
 *     YuQiaowei <cerf.yu@rock-chips.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define LOG_NDEBUG 0
#undef LOG_TAG
#define LOG_TAG "rga_resize_demo"

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
#include <linux/stddef.h>

#include "RgaUtils.h"
#include "im2d.hpp"

#include "librga_utils/utils.h"
#include "dma_alloc.h"

#include "inspirecv/inspirecv.h"

#define LOCAL_FILE_PATH "data/"

#define RV1106_CMA_HEAP_PATH "/dev/rk_dma_heap/rk-dma-heap-cma"

int main() {
    int ret = 0;
    int src_width, src_height, src_format;
    int dst_width, dst_height, dst_format;
    char *src_buf, *dst_buf;
    int src_buf_size, dst_buf_size;
    int src_dma_fd, dst_dma_fd;
    rga_buffer_t src_img = {};
    rga_buffer_t dst_img = {};
    im_rect src_rect = {};
    im_rect dst_rect = {};
    rga_buffer_handle_t src_handle, dst_handle;
    int64_t ts;

    memset(&src_img, 0, sizeof(src_img));
    memset(&dst_img, 0, sizeof(dst_img));

    src_width = 640;
    src_height = 960;
    src_format = RK_FORMAT_RGB_888;

    dst_width = src_width / 2;
    dst_height = src_height / 2;
    dst_format = RK_FORMAT_RGB_888;

    src_buf_size = src_width * src_height * get_bpp_from_format(src_format);
    dst_buf_size = dst_width * dst_height * get_bpp_from_format(dst_format);

    std::cout << "src_buf_size: " << src_buf_size << std::endl;
    std::cout << "dst_buf_size: " << dst_buf_size << std::endl;

    inspirecv::TimeSpend a1("Alloc DMA buffer");
    a1.Start();
    /* Allocate dma_buf from CMA, return dma_fd and virtual address */
    ret = dma_buf_alloc(RV1106_CMA_HEAP_PATH, src_buf_size, &src_dma_fd, (void **)&src_buf);
    if (ret < 0) {
        printf("alloc src CMA buffer failed: %d\n", ret);
        return -1;
    }
    std::cout << "Alloc src CMA buffer success" << std::endl;

    ret = dma_buf_alloc(RV1106_CMA_HEAP_PATH, dst_buf_size, &dst_dma_fd, (void **)&dst_buf);
    if (ret < 0) {
        printf("alloc dst CMA buffer failed: %d\n", ret);
        dma_buf_free(src_buf_size, &src_dma_fd, src_buf);
        return -1;
    }
    a1.Stop();

    std::cout << "Alloc dst CMA buffer success" << std::endl;

    std::cout << a1 << std::endl;
    /* fill image data */
    if (0 != read_image_from_file(src_buf, LOCAL_FILE_PATH, src_width, src_height, src_format, 0)) {
        printf("src image read err\n");
        draw_rgba(src_buf, src_width, src_height);
    }

    memset(dst_buf, 0x33, dst_buf_size);
    std::cout << "Read image from file success" << std::endl;

    src_handle = importbuffer_fd(src_dma_fd, src_buf_size);
    dst_handle = importbuffer_fd(dst_dma_fd, dst_buf_size);
    if (src_handle == 0 || dst_handle == 0) {
        printf("importbuffer failed!\n");
        // goto release_buffer;
    }

    std::cout << "Import buffer success" << std::endl;
    src_img = wrapbuffer_handle(src_handle, src_width, src_height, src_format);
    dst_img = wrapbuffer_handle(dst_handle, dst_width, dst_height, dst_format);
    std::cout << "Wrap buffer success" << std::endl;

    /*
     * Scale up the src image to 1920*1080.
        --------------    ---------------------
        |            |    |                   |
        |  src_img   |    |     dst_img       |
        |            | => |                   |
        --------------    |                   |
                          |                   |
                          ---------------------
     */

    ret = imcheck(src_img, dst_img, {}, {});
    if (IM_STATUS_NOERROR != ret) {
        printf("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret));
        return -1;
    }
    std::cout << "Check success" << std::endl;

    inspirecv::TimeSpend s1("RGA");
    s1.Start();
    ret = imresize(src_img, dst_img);
    if (ret == IM_STATUS_SUCCESS) {
        printf("%s running success!\n", LOG_TAG);
    } else {
        printf("%s running failed, %s\n", LOG_TAG, imStrError((IM_STATUS)ret));
        // goto release_buffer;
    }
    std::cout << "imresize success" << std::endl;
    s1.Stop();

    std::cout << s1 << std::endl;

    write_image_to_file(dst_buf, LOCAL_FILE_PATH, dst_width, dst_height, dst_format, 0);
    std::cout << "Write image to file success" << std::endl;
release_buffer:
    if (src_handle)
        releasebuffer_handle(src_handle);
    if (dst_handle)
        releasebuffer_handle(dst_handle);

    if (src_buf)
        free(src_buf);
    if (dst_buf)
        free(dst_buf);

    return ret;
}