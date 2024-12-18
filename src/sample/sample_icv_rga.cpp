#include <iostream>

#include "inspirecv/core/processor.h"
#include "inspirecv/time_spend.h"

int main() {
    auto processor = inspirecv::Processor::Create();
    auto image = inspirecv::Image::Create("input.png");
    auto dst = inspirecv::Image::Create();

    // resize CPU
    inspirecv::TimeSpend time_spend_cpu("CPU Resize");
    for (int i = 0; i < 10; i++) {
        time_spend_cpu.Start();
        dst = image.Resize(image.Width() / 2, image.Height() / 2);
        time_spend_cpu.Stop();
    }
    std::cout << time_spend_cpu << std::endl;
    dst.Write("icv_cpu_resize.png");

    // resize RGA
    processor->PreallocateBuffer(image.Width() / 2, image.Height() / 2, image.Channels());
    inspirecv::TimeSpend time_spend("RGA Resize");
    for (int i = 0; i < 10; i++) {
        time_spend.Start();
        processor->Resize(image, dst, image.Width() / 2, image.Height() / 2);
        time_spend.Stop();
        std::cout << "================" << std::endl;
    }
    std::cout << time_spend << std::endl;
    dst.Write("icv_rga_resize.png");

    // // Use raw
    // inspirecv::TimeSpend time_spend_raw("RGA Resize Raw");
    // auto data = image.Data();
    // uint8_t* dst_data = nullptr;
    // for (int i = 0; i < 10; i++) {
    //     time_spend_raw.Start();
    //     processor->Resize(data, image.Width(), image.Height(), image.Channels(), &dst_data,
    //                       image.Width() / 2, image.Height() / 2);
    //     std::cout << "================" << std::endl;
    //     time_spend_raw.Stop();
    // }
    // std::cout << time_spend_raw << std::endl;
    // // to image
    // dst.Reset(image.Width() / 2, image.Height() / 2, image.Channels(), dst_data);
    // dst.Write("icv_rga_resize_raw.png");

    // inspirecv::Image dst_icv;
    inspirecv::TimeSpend time_spend_swap_icv("CPU Swap Color");
    for (int i = 0; i < 10; i++) {
        time_spend_swap_icv.Start();
        dst = dst.SwapRB();
        time_spend_swap_icv.Stop();
    }
    std::cout << time_spend_swap_icv << std::endl;
    dst.Write("icv_swap_color.png");

    // swap color
    inspirecv::TimeSpend time_spend_swap("RGA Swap Color");
    for (int i = 0; i < 10; i++) {
        time_spend_swap.Start();
        processor->SwapColor(dst, dst);
        time_spend_swap.Stop();
    }
    std::cout << time_spend_swap << std::endl;
    dst.Write("icv_rga_swap_color.png");

    // padding CPU
    inspirecv::Image dst_cpu;
    inspirecv::TimeSpend time_spend_padding_cpu("CPU Padding");
    for (int i = 0; i < 10; i++) {
        time_spend_padding_cpu.Start();
        dst_cpu = dst.Pad(10, 10, 10, 10, {0, 0, 0});
        time_spend_padding_cpu.Stop();
    }
    std::cout << time_spend_padding_cpu << std::endl;
    dst_cpu.Write("icv_cpu_padding.png");

    // padding RGA
    inspirecv::Image dst_rga;
    inspirecv::TimeSpend time_spend_padding("RGA Padding");
    for (int i = 0; i < 10; i++) {
        time_spend_padding.Start();
        processor->Padding(dst, dst_rga, 10, 10, 10, 10);
        time_spend_padding.Stop();
    }
    std::cout << time_spend_padding << std::endl;
    dst_rga.Write("icv_rga_padding.png");

    return 0;
}
