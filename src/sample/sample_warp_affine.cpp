#include <iostream>
#include <string>
#include "inspirecv/inspirecv.h"

using namespace inspirecv;

int main() {
    // Inputs copied from sample_pipeline.cpp
    const std::string target_path = "../images/face_sample.png";
    const std::string fake_path = "../images/aimg.jpg";

    // Read images
    Image target_img = Image::Create(target_path, 3);
    if (target_img.Empty()) {
        std::cerr << "Failed to read target image: " << target_path << std::endl;
        return 1;
    }
    Image bgr_fake = Image::Create(fake_path, 3);
    if (bgr_fake.Empty()) {
        std::cerr << "Failed to read bgr_fake: " << fake_path << std::endl;
        return 1;
    }

    // Default affine matrix copied from sample_pipeline.cpp
    TransformMatrix M = TransformMatrix::Create(
        0.18727215f, 0.06365608f, -34.09374106f,
       -0.06365608f, 0.18727215f,  -5.86148856f
    );

    // Prepare white 1-channel image (aligned space) and warp both to target space
    const std::string aimg_path = "../images/aimg.jpg";
    Image aimg = Image::Create(aimg_path, 3);
    if (aimg.Empty()) {
        std::cerr << "Failed to read aimg: " << aimg_path << std::endl;
        return 1;
    }
    const int aligned_w = aimg.Width();
    const int aligned_h = aimg.Height();
    Image img_white = Image::Create(aligned_w, aligned_h, 1);
    img_white.Fill(255);

    const int TW = target_img.Width();
    const int TH = target_img.Height();

    // Loop affine and measure time
    const int iters = 50;
    Image bgr_fake_warp, img_white_warp;
    {
        TimeSpend t1(std::string("WarpAffineColor-") + GetCVBackend());
        for (int i = 0; i < iters; ++i) {
            t1.Start();
            bgr_fake_warp = bgr_fake.WarpAffine(M, TW, TH);
            t1.Stop();
        }
        std::cout << t1 << std::endl;
    }
    {
        TimeSpend t2(std::string("WarpAffineGray-") + GetCVBackend());
        for (int i = 0; i < iters; ++i) {
            t2.Start();
            img_white_warp = img_white.WarpAffine(M, TW, TH);
            t2.Stop();
        }
        std::cout << t2 << std::endl;
    }

    // Save processed images
    std::string out_color = "warp_color_" + std::string(GetCVBackend()) + ".jpg";
    std::string out_white = "warp_white_" + std::string(GetCVBackend()) + ".png";
    bgr_fake_warp.Write(out_color.c_str());
    img_white_warp.Write(out_white.c_str());
    std::cout << "Saved: " << out_color << ", " << out_white << std::endl;
    return 0;
}


