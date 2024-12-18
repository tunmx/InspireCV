#ifndef INSPIRECV_PROCESSOR_H
#define INSPIRECV_PROCESSOR_H

#include "define.h"
#include "image.h"
#include <memory>

namespace inspirecv {

class INSPIRECV_API Processor {
public:
    Processor();
    ~Processor();

    // Add move constructor and move assignment operator
    Processor(Processor&&) = default;
    Processor& operator=(Processor&&) = default;

    // Delete copy constructor and copy assignment operator
    Processor(const Processor&) = delete;
    Processor& operator=(const Processor&) = delete;

    bool PreallocateBuffer(int width, int height, int channels);
    bool Resize(const Image& src, Image& dst, int dst_width, int dst_height);
    bool Resize(const uint8_t* src_data, int src_width, int src_height, int channels,
                uint8_t** dst_data, int dst_width, int dst_height);
    bool SwapColor(const Image& src, Image& dst);
    bool Padding(const Image& src, Image& dst, int padding_left, int padding_right, int padding_top,
                 int padding_bottom);
    static std::unique_ptr<Processor> Create();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace inspirecv

#endif  // INSPIRECV_PROCESSOR_H