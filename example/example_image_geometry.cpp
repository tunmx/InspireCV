// Geometry example: Resize, Rotate(90/180/270), Flip(H/V), Crop, Pad.
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
                  << "  example_image_geometry <input_image>\n"
                  << "Output:\n"
                  << "  ./exp_out_resize_half.jpg\n"
                  << "  ./exp_out_rot90.jpg\n"
                  << "  ./exp_out_rot180.jpg\n"
                  << "  ./exp_out_rot270.jpg\n"
                  << "  ./exp_out_flip_h.jpg\n"
                  << "  ./exp_out_flip_v.jpg\n"
                  << "  ./exp_out_crop.jpg\n"
                  << "  ./exp_out_pad.jpg\n"
                  << "  ./exp_out_warp_affine.jpg\n";
        return 0;
    }
    const std::string input_path = argv[1];

    inspirecv::Image img = inspirecv::Image::Create();
    if (!img.Read(input_path, /*channels=*/3) || img.Empty()) {
        std::cerr << "Failed to read image: " << input_path << std::endl;
        return 1;
    }

    // Resize: half size (at least 1x1)
    int half_w = img.Width() > 1 ? img.Width() / 2 : 1;
    int half_h = img.Height() > 1 ? img.Height() / 2 : 1;
    inspirecv::Image resized = img.Resize(half_w, half_h, /*use_linear=*/true);
    if (!write_or_fail(resized, "exp_out_resize_half.jpg")) return 2;

    // Rotations
    if (!write_or_fail(img.Rotate90(),  "exp_out_rot90.jpg"))  return 3;
    if (!write_or_fail(img.Rotate180(), "exp_out_rot180.jpg")) return 4;
    if (!write_or_fail(img.Rotate270(), "exp_out_rot270.jpg")) return 5;

    // Flips
    if (!write_or_fail(img.FlipHorizontal(), "exp_out_flip_h.jpg")) return 6;
    if (!write_or_fail(img.FlipVertical(),   "exp_out_flip_v.jpg")) return 7;

    // Crop: centered box of ~1/2 size
    int cw = img.Width()  / 2;
    int ch = img.Height() / 2;
    if (cw < 1) cw = 1;
    if (ch < 1) ch = 1;
    int cx = (img.Width()  - cw) / 2;
    int cy = (img.Height() - ch) / 2;
    inspirecv::Rect<int> roi(cx, cy, cw, ch);
    inspirecv::Image cropped = img.Crop(roi);
    if (!write_or_fail(cropped, "exp_out_crop.jpg")) return 8;

    // Pad: 30px on all sides with gray color
    inspirecv::Image padded = img.Pad(30, 30, 30, 30, {128.0, 128.0, 128.0});
    if (!write_or_fail(padded, "exp_out_pad.jpg")) return 9;

    // Affine: construct directly as coefficients (a11, a12, b1, a21, a22, b2), similar to sample_warp_affine
    // Slightly smaller scale to bring more content into view
    inspirecv::TransformMatrix tm = inspirecv::TransformMatrix::Create(
        1.41236329f,  0.03819365f, -60.09374106f,
       -0.03819365f,  1.61236329f,  -50.86148856f
    );
    inspirecv::Image warped = img.WarpAffine(tm, img.Width(), img.Height());
    if (!write_or_fail(warped, "exp_out_warp_affine.jpg")) return 10;

    std::cout << "Wrote geometry outputs.\n";
    return 0;
}


