// Minimal user-facing example: read an image, rotate 90°, write out.
#include "../include/inspirecv/inspirecv.h"
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage:\n"
                  << "  example_image_io <input_image>\n"
                  << "Example:\n"
                  << "  example_image_io images/aimg.jpg\n"
                  << "Output:\n"
                  << "  ./exp_out_rot90.jpg\n";
        return 0;
    }
    const std::string input_path = argv[1];
    const std::string output_path = "exp_out_rot90.jpg";

    inspirecv::Image img = inspirecv::Image::Create();
    if (!img.Read(input_path, /*channels=*/3) || img.Empty()) {
        std::cerr << "Failed to read image: " << input_path << std::endl;
        return 1;
    }
    inspirecv::Image rotated = img.Rotate90();
    if (!rotated.Write(output_path)) {
        std::cerr << "Failed to write image: " << output_path << std::endl;
        return 2;
    }
    std::cout << "Wrote: " << output_path << std::endl;
    return 0;
}

