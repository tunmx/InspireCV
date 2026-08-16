// Color & channel operations example: ToGray, SwapRB, MeanChannels.
#include "../include/inspirecv/inspirecv.h"
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage:\n"
                  << "  example_image_color <input_image>\n"
                  << "Output:\n"
                  << "  ./exp_out_gray.jpg\n"
                  << "  ./exp_out_swaprb.jpg\n"
                  << "  ./exp_out_mean.jpg\n";
        return 0;
    }
    const std::string input_path = argv[1];

    inspirecv::Image img = inspirecv::Image::Create();
    if (!img.Read(input_path, /*channels=*/3) || img.Empty()) {
        std::cerr << "Failed to read image: " << input_path << std::endl;
        return 1;
    }

    inspirecv::Image gray   = img.ToGray();
    inspirecv::Image swaprb = img.SwapRB();
    inspirecv::Image mean   = img.MeanChannels();

    if (!gray.Write("exp_out_gray.jpg")) {
        std::cerr << "Failed to write: exp_out_gray.jpg\n";
        return 2;
    }
    if (!swaprb.Write("exp_out_swaprb.jpg")) {
        std::cerr << "Failed to write: exp_out_swaprb.jpg\n";
        return 3;
    }
    if (!mean.Write("exp_out_mean.jpg")) {
        std::cerr << "Failed to write: exp_out_mean.jpg\n";
        return 4;
    }

    std::cout << "Wrote: exp_out_gray.jpg, exp_out_swaprb.jpg, exp_out_mean.jpg\n";
    return 0;
}


