// Filter + Morphology + Threshold + Alpha blend example.
#include "../include/inspirecv/inspirecv.h"
#include <iostream>
#include <string>

static bool write_or_fail(const inspirecv::Image& img, const std::string& path) {
    if (!img.Write(path)) {
        std::cerr << "Failed to write: " << path << "\n";
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage:\n"
                  << "  example_image_filter_morph_alpha <input_image>\n"
                  << "Output:\n"
                  << "  ./exp_out_gauss_k5_s1p2.jpg\n"
                  << "  ./exp_out_thresh_100.jpg\n"
                  << "  ./exp_out_morph_open_k3.jpg\n"
                  << "  ./exp_out_blend.jpg\n";
        return 0;
    }
    const std::string input_path = argv[1];

    // Read color image (3 channels)
    inspirecv::Image img = inspirecv::Image::Create();
    if (!img.Read(input_path, /*channels=*/3) || img.Empty()) {
        std::cerr << "Failed to read image: " << input_path << std::endl;
        return 1;
    }

    // 1) Gaussian blur -> save
    const int k_gauss = 5;
    const double sigma = 1.2;
    inspirecv::Image blur = img.GaussianBlur(k_gauss, sigma);
    if (!write_or_fail(blur, "exp_out_gauss_k5_s1p2.jpg")) return 2;

    // 2) Threshold on blurred grayscale -> save
    inspirecv::Image blur_gray = blur.ToGray();
    const double thresh = 100.0;
    const double maxval = 255.0;
    inspirecv::Image th = blur_gray.Threshold(thresh, maxval, /*type*/0);
    if (!write_or_fail(th, "exp_out_thresh_100.jpg")) return 3;

    // 3) Morphology (open: erode then dilate) on thresholded image -> save
    const int k_morph = 3;
    inspirecv::Image er = th.Erode(k_morph, /*iterations=*/1);
    inspirecv::Image morph = er.Dilate(k_morph, /*iterations=*/1);
    if (!write_or_fail(morph, "exp_out_morph_open_k3.jpg")) return 4;

    // 4) Before blending, change color via multiplicative scaling
    inspirecv::Image color_changed = img.Mul(1.1);  // brighten slightly

    //    Blend the color-changed image with the original using the morph result as mask -> save
    //    Semantics: out = m*this + (1-m)*other, where m is from 8U mask in [0..255].
    inspirecv::Image blended = color_changed.Blend(img, morph);
    if (!write_or_fail(blended, "exp_out_blend.jpg")) return 5;

    std::cout << "Wrote: exp_out_gauss_k5_s1p2.jpg, exp_out_thresh_100.jpg, "
                 "exp_out_morph_open_k3.jpg, exp_out_blend.jpg\n";
    return 0;
}


