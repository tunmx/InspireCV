#include "okcv_rga/image_rga.h"
#include "inspirecv/time_spend.h"
#include "okcv_rga/rga_processor.h"

int main(int argc, char **argv) {
    okcv::Image<uint8_t> image;
    image.Read("input.png");
    image.Write("input_ori.png");

    inspirecv::TimeSpend time_spend0("Image Resize");
    for (int i = 0; i < 10; i++) {
        time_spend0.Start();
        okcv::Image<uint8_t> dst0 = image.ResizeBilinear(image.Width() / 2, image.Height() / 2);
        time_spend0.Stop();
    }
    std::cout << time_spend0 << std::endl;

    inspirecv::RGAProcessor rga_processor;
    okcv::Image<uint8_t> dst;
    rga_processor.PreallocateBuffer(image.Width() / 2, image.Height() / 2, image.Channels());
    inspirecv::TimeSpend time_spend("RGA Resize");
    for (int i = 0; i < 10; i++) {
        time_spend.Start();
        rga_processor.Resize(image, dst, image.Width() / 2, image.Height() / 2);
        time_spend.Stop();
    }
    std::cout << time_spend << std::endl;

    dst.Write("input_resize.png");

    return 0;
}