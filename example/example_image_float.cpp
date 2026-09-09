// Float workflow example: read as float, run a few ops, write results.
#include "../include/inspirecv/inspirecv.h"
#include <iostream>
#include <string>

static bool write_or_fail(const inspirecv::ImageT<float>& img, const std::string& path) {
    if (!img.Write(path)) {
        std::cerr << "Failed to write: " << path << "\n";
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage:\n"
                  << "  example_image_float <input_image>\n"
                  << "Output:\n"
                  << "  ./exp_out_f32_blur5.jpg\n"
                  << "  ./exp_out_f32_resize_half.jpg\n"
                  << "  ./exp_out_f32_rot90.jpg\n";
        return 0;
    }
    const std::string input_path = argv[1];

    inspirecv::ImageT<float> fimg = inspirecv::ImageT<float>::Create();
    if (!fimg.Read(input_path, /*channels=*/3) || fimg.Empty()) {
        std::cerr << "Failed to read image (float): " << input_path << std::endl;
        return 1;
    }

    // Gaussian blur (float)
    auto fblur = fimg.GaussianBlur(5, 1.0);
    if (!write_or_fail(fblur, "exp_out_f32_blur5.jpg")) return 2;

    // Resize to half (ensure at least 1x1)
    int half_w = fimg.Width() > 1 ? fimg.Width() / 2 : 1;
    int half_h = fimg.Height() > 1 ? fimg.Height() / 2 : 1;
    auto fhalf = fimg.Resize(half_w, half_h, /*use_linear=*/true);
    if (!write_or_fail(fhalf, "exp_out_f32_resize_half.jpg")) return 3;

    // Rotate 90 degrees
    auto frot90 = fimg.Rotate90();
    if (!write_or_fail(frot90, "exp_out_f32_rot90.jpg")) return 4;

    std::cout << "Wrote: exp_out_f32_blur5.jpg, exp_out_f32_resize_half.jpg, exp_out_f32_rot90.jpg\n";
    return 0;
}


