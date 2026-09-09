#include <inspirecv/inspirecv.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <algorithm>
static uint64_t checksum_u8(const inspirecv::Image& img) {
    const uint8_t* p = img.Data();
    uint64_t s = 0;
    for (int i = 0; i < img.Width() * img.Height() * img.Channels(); ++i) s += p[i];
    return s;
}
static uint64_t count_nonzero_u8(const inspirecv::Image& img) {
    const uint8_t* p = img.Data();
    uint64_t c = 0;
    for (int i = 0; i < img.Width() * img.Height() * img.Channels(); ++i) c += (p[i] != 0);
    return c;
}
static double checksum_f32(const inspirecv::ImageT<float>& img) {
    const float* p = img.Data();
    double s = 0.0;
    for (int i = 0; i < img.Width() * img.Height() * img.Channels(); ++i) s += p[i];
    return s;
}
static uint64_t count_nonzero_f32(const inspirecv::ImageT<float>& img) {
    const float* p = img.Data();
    uint64_t c = 0;
    for (int i = 0; i < img.Width() * img.Height() * img.Channels(); ++i) c += (p[i] != 0.0f);
    return c;
}
int main(int argc, char** argv) {
    std::string in_path = (argc > 1) ? argv[1] : std::string("../images/kun.jpg");
    double thresh = (argc > 2) ? std::stod(argv[2]) : 128.0;
    double maxval = (argc > 3) ? std::stod(argv[3]) : 255.0;
    // Read as grayscale (1ch)
    auto img_u8 = inspirecv::Image::Create(in_path, 1);
    if (img_u8.Empty()) {
        std::cerr << "Failed to read image: " << in_path << std::endl;
        return 1;
    }
    std::cout << "Input(gray): " << in_path << " (" << img_u8.Width() << "x" << img_u8.Height()
              << "x" << img_u8.Channels() << ")" << std::endl;
    // u8 threshold (OpenCV binary semantics '>'): type=0
    auto th_u8 = img_u8.Threshold(thresh, maxval, 0);
    th_u8.Write("threshold_u8.png");
    std::cout << "[u8] sum=" << checksum_u8(th_u8)
              << ", nz=" << count_nonzero_u8(th_u8) << std::endl;
    // float path: convert u8 gray to float
    std::vector<float> gray_f(img_u8.Width() * img_u8.Height());
    const uint8_t* gp = img_u8.Data();
    for (size_t i = 0; i < gray_f.size(); ++i) gray_f[i] = static_cast<float>(gp[i]);
    auto img_f32 = inspirecv::ImageT<float>::Create(img_u8.Width(), img_u8.Height(), 1, gray_f.data());
    auto th_f32 = img_f32.Threshold(thresh, maxval, 0);
    th_f32.Write("threshold_f32.png");
    std::cout << "[f32] sum=" << checksum_f32(th_f32)
              << ", nz=" << count_nonzero_f32(th_f32) << std::endl;
    std::cout << "Saved: threshold_u8.png, threshold_f32.png" << std::endl;
    return 0;
}

