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

template <typename D>
class OKCV_API ImageRga : public Image<uint8_t> {
public:
    ImageRga() : Image<uint8_t>() {}
    ImageRga(Image &&image);
    ImageRga &operator=(Image &&image);

    void Reset();

    void Reset(int width, int height, int channels, const D *data = nullptr);

protected:
    rga_buffer_t rga_buffer_{};
    rga_buffer_handle_t rga_handle_{};
    int rga_dma_fd_{-1};
    // rga transform cache
    rga_buffer_t rga_dest_cache_buffer_{};
    rga_buffer_handle_t rga_dest_cache_handle_{};
    int rga_dest_cache_dma_fd_{-1};
};

}  // namespace okcv

#endif  // OKCV_RGA_IMAGE_RGA_H_