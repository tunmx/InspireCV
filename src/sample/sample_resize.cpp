#include <inspirecv/inspirecv.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <cmath>
#include <chrono>
static uint64_t checksum_u8(const inspirecv::Image& img) {
    const uint8_t* p = img.Data();
    uint64_t s = 0;
    for (int i = 0; i < img.Width() * img.Height() * img.Channels(); ++i) s += p[i];
    return s;
}
static double checksum_f32(const inspirecv::ImageT<float>& img) {
    const float* p = img.Data();
    double s = 0.0;
    for (int i = 0; i < img.Width() * img.Height() * img.Channels(); ++i) s += p[i];
    return s;
}
int main(int argc, char** argv) {
    std::string in_path = (argc > 1) ? argv[1] : std::string("../images/kun.jpg");
    std::string mode = (argc > 2) ? argv[2] : std::string("linear"); // linear | nearest
    int iters = (argc > 3) ? std::max(1, std::stoi(argv[3])) : 50;
    bool use_linear = (mode != "nearest");
    // Read u8 image (3ch)
    auto img_u8 = inspirecv::Image::Create(in_path, 3);
    if (img_u8.Empty()) {
        std::cerr << "Failed to read image: " << in_path << std::endl;
        return 1;
    }
    std::cout << "Input: " << in_path << " (" << img_u8.Width() << "x" << img_u8.Height()
              << "x" << img_u8.Channels() << ")" << std::endl;
    int upw = img_u8.Width() * 2, uph = img_u8.Height() * 2;
    int dnw = std::max(1, img_u8.Width() / 2), dnh = std::max(1, img_u8.Height() / 2);
    // Warmup + save
    auto up_u8 = img_u8.Resize(upw, uph, use_linear);
    auto dn_u8 = img_u8.Resize(dnw, dnh, use_linear);
    // Save u8 outputs
    if (use_linear) {
        up_u8.Write("resize_u8_up2_linear.jpg");
        dn_u8.Write("resize_u8_dn2_linear.jpg");
    } else {
        up_u8.Write("resize_u8_up2_nearest.jpg");
        dn_u8.Write("resize_u8_dn2_nearest.jpg");
    }
    // Loop timings
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) up_u8 = img_u8.Resize(upw, uph, use_linear);
    auto t1 = std::chrono::steady_clock::now();
    double up_ms_u8 = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) dn_u8 = img_u8.Resize(dnw, dnh, use_linear);
    t1 = std::chrono::steady_clock::now();
    double dn_ms_u8 = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    std::cout << "[u8] checksums:" << std::endl;
    std::cout << "  up2_" << mode << "  sum=" << checksum_u8(up_u8) << " avg_ms=" << up_ms_u8 << " (iters=" << iters << ")" << std::endl;
    std::cout << "  dn2_" << mode << "  sum=" << checksum_u8(dn_u8) << " avg_ms=" << dn_ms_u8 << " (iters=" << iters << ")" << std::endl;
    // Float path: convert to gray and test linear upsample on optimized path
    auto gray_u8 = img_u8.ToGray();
    // normalize to float [0..255] (ImageT<float>::Create reads and converts internally if given path,
    // here we build from u8 gray for explicitness)
    std::vector<float> gray_f(gray_u8.Width() * gray_u8.Height());
    const uint8_t* gp = gray_u8.Data();
    for (size_t i = 0; i < gray_f.size(); ++i) gray_f[i] = static_cast<float>(gp[i]);
    auto img_f32 = inspirecv::ImageT<float>::Create(gray_u8.Width(), gray_u8.Height(), 1, gray_f.data());
    auto up_f32 = img_f32.Resize(img_f32.Width() * 2, img_f32.Height() * 2, use_linear);
    auto dn_f32 = img_f32.Resize(std::max(1, img_f32.Width() / 2), std::max(1, img_f32.Height() / 2), use_linear);
    // Save float outputs (内部 Write 会做 round+饱和为 u8 输出)
    if (use_linear) {
        up_f32.Write("resize_f32_up2_linear.png");
        dn_f32.Write("resize_f32_dn2_linear.png");
    } else {
        up_f32.Write("resize_f32_up2_nearest.png");
        dn_f32.Write("resize_f32_dn2_nearest.png");
    }
    // Loop timings
    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) up_f32 = img_f32.Resize(img_f32.Width() * 2, img_f32.Height() * 2, use_linear);
    t1 = std::chrono::steady_clock::now();
    double up_ms_f32 = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) dn_f32 = img_f32.Resize(std::max(1, img_f32.Width() / 2), std::max(1, img_f32.Height() / 2), use_linear);
    t1 = std::chrono::steady_clock::now();
    double dn_ms_f32 = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    std::cout << "[f32 gray] checksums:" << std::endl;
    std::cout << "  up2_" << mode << "  sum=" << checksum_f32(up_f32) << " avg_ms=" << up_ms_f32 << " (iters=" << iters << ")" << std::endl;
    std::cout << "  dn2_" << mode << "  sum=" << checksum_f32(dn_f32) << " avg_ms=" << dn_ms_f32 << " (iters=" << iters << ")" << std::endl;
    std::cout << "Saved: resize_u8_*.jpg, resize_f32_*.png (mode=" << mode << ", iters=" << iters << ")" << std::endl;
    return 0;
}

