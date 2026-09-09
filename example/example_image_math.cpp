// Pointwise math & comparison example: Add, Mul, AbsDiff.
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
                  << "  example_image_math <input_image> [second_image]\n"
                  << "Output:\n"
                  << "  ./exp_out_add.jpg\n"
                  << "  ./exp_out_mul.jpg\n"
                  << "  ./exp_out_absdiff.jpg\n";
        return 0;
    }
    const std::string input_path = argv[1];
    const std::string second_path = (argc >= 3) ? argv[2] : std::string();

    inspirecv::Image img = inspirecv::Image::Create();
    if (!img.Read(input_path, /*channels=*/3) || img.Empty()) {
        std::cerr << "Failed to read image: " << input_path << std::endl;
        return 1;
    }

    // Add and Mul
    inspirecv::Image add_img = img.Add(10.0);
    inspirecv::Image mul_img = img.Mul(1.2);
    if (!write_or_fail(add_img, "exp_out_add.jpg")) return 2;
    if (!write_or_fail(mul_img, "exp_out_mul.jpg")) return 3;

    // Prepare a second image for AbsDiff
    inspirecv::Image other;
    bool have_other = false;
    if (!second_path.empty()) {
        other = inspirecv::Image::Create();
        if (other.Read(second_path, /*channels=*/img.Channels()) && !other.Empty()) {
            have_other = true;
        }
    }
    if (!have_other) {
        // Fallback: generate a different version from the input
        other = img.Rotate180();
        have_other = true;
    }
    // Ensure same size
    if (other.Width() != img.Width() || other.Height() != img.Height()) {
        other = other.Resize(img.Width(), img.Height(), /*use_linear=*/true);
    }
    // Ensure same channels
    inspirecv::Image lhs = img.Clone();
    inspirecv::Image rhs = other.Clone();
    if (lhs.Channels() != rhs.Channels()) {
        lhs = lhs.ToGray();
        rhs = rhs.ToGray();
    }
    inspirecv::Image diff = lhs.AbsDiff(rhs);
    if (!write_or_fail(diff, "exp_out_absdiff.jpg")) return 4;

    std::cout << "Wrote: exp_out_add.jpg, exp_out_mul.jpg, exp_out_absdiff.jpg\n";
    return 0;
}


