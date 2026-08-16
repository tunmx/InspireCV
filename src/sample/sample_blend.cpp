#include <inspirecv/inspirecv.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
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
    std::string a_path = (argc > 1) ? argv[1] : std::string("../images/kun.jpg");
    std::string b_path = (argc > 2) ? argv[2] : std::string("../images/kun_copy.jpg");
    std::string mode   = (argc > 3) ? argv[3] : std::string("grad"); // grad | half
    int iters = (argc > 4) ? std::max(1, std::stoi(argv[4])) : 50;
    // Read two color images
    auto A = inspirecv::Image::Create(a_path, 3);
    auto B = inspirecv::Image::Create(b_path, 3);
    if (A.Empty() || B.Empty()) {
        std::cerr << "Failed to read images." << std::endl;
        return 1;
    }
    // Match size
    if (A.Width() != B.Width() || A.Height() != B.Height()) {
        B = B.Resize(A.Width(), A.Height(), true);
    }
    // Make 1ch mask
    const int w = A.Width(), h = A.Height();
    std::vector<uint8_t> m(w * h);
    if (mode == "half") {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                m[y * w + x] = (x < w / 2) ? 0 : 255;
            }
        }
    } else { // grad
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                float t = w > 1 ? static_cast<float>(x) / static_cast<float>(w - 1) : 0.f;
                int v = static_cast<int>(t * 255.f + 0.5f);
                if (v < 0) v = 0; if (v > 255) v = 255;
                m[y * w + x] = static_cast<uint8_t>(v);
            }
        }
    }
    auto M = inspirecv::Image::Create(w, h, 1, m.data());
    // Blend u8
    auto OutU8 = A.Blend(B, M); // warmup
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        OutU8 = A.Blend(B, M);
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms_u8 = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    OutU8.Write("blend_u8.png");
    std::cout << "[u8] sum=" << checksum_u8(OutU8) << std::endl;
    std::cout << "[u8] avg_ms_per_iter=" << ms_u8 << " (iters=" << iters << ")\n";
    // Blend float path for reference
    std::vector<float> af(w * h * 3), bf(w * h * 3);
    for (size_t i = 0; i < af.size(); ++i) {
        af[i] = static_cast<float>(A.Data()[i]);
        bf[i] = static_cast<float>(B.Data()[i]);
    }
    auto AF = inspirecv::ImageT<float>::Create(w, h, 3, af.data());
    auto BF = inspirecv::ImageT<float>::Create(w, h, 3, bf.data());
    auto OutF = AF.Blend(BF, M); // warmup
    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
        OutF = AF.Blend(BF, M);
    }
    t1 = std::chrono::steady_clock::now();
    double ms_f32 = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
    OutF.Write("blend_f32.png");
    std::cout << "[f32] sum=" << checksum_f32(OutF) << std::endl;
    std::cout << "[f32] avg_ms_per_iter=" << ms_f32 << " (iters=" << iters << ")\n";
    std::cout << "Saved: blend_u8.png, blend_f32.png" << std::endl;
    return 0;
}