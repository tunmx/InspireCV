
#include "inspirecv/inspirecv.h"
#include "im2d.hpp"
#include "RgaUtils.h"
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

#undef LOG_TAG
#define LOG_TAG "rga_resize_demo"

int rga_test(inspirecv::Image &image) {
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

    inspirecv::TimeSpend s1("imresize");
    memset(&src_img, 0, sizeof(src_img));
    memset(&dst_img, 0, sizeof(dst_img));

    src_width = image.Width();
    src_height = image.Height();
    src_format = RK_FORMAT_RGB_888;

    dst_width = src_width / 2;
    dst_height = src_height / 2;
    dst_format = RK_FORMAT_RGB_888;

    src_buf_size = src_width * src_height * get_bpp_from_format(src_format);
    dst_buf_size = dst_width * dst_height * get_bpp_from_format(dst_format);

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

    memset(dst_buf, 0x33, dst_buf_size);

    memcpy(src_buf, image.Data(), src_buf_size);

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
    s1.Start();
    ret = imresize(src_img, dst_img);
    if (ret == IM_STATUS_SUCCESS) {
        printf("%s running success!\n", LOG_TAG);
    } else {
        printf("%s running failed, %s\n", LOG_TAG, imStrError((IM_STATUS)ret));
        // goto release_buffer;
    }
    s1.Stop();
    std::cout << s1 << std::endl;

    inspirecv::Image dst_image = inspirecv::Image::Create();
    dst_image.Reset(dst_width, dst_height, 3, (uint8_t *)dst_buf);
    dst_image.Write("output_rga.png");

    return 0;
}

int main() {
    inspirecv::Image image = inspirecv::Image::Create();
    image.Read("input.png");
    std::cout << image.Width() << " " << image.Height() << " " << image.Channels() << std::endl;

    inspirecv::TimeSpend s1("Resize@general");
    s1.Start();
    auto img = image.Resize(image.Width() / 2, image.Height() / 2);
    s1.Stop();

    std::cout << s1 << std::endl;

    rga_test(img);

    return 0;
}