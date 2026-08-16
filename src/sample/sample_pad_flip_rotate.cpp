#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <chrono>
#include <inspirecv/inspirecv.h>

using namespace inspirecv;

static uint64_t sum_u8(const Image& img) {
    const uint8_t* p = img.Data();
    const int n = img.Width() * img.Height() * img.Channels();
    uint64_t s = 0;
    for (int i = 0; i < n; ++i) s += p[i];
    return s;
}

int main(int argc, char** argv) {
    const std::string img_path = (argc >= 2) ? std::string(argv[1]) : "../images/kun.jpg";
    int iters = (argc >= 3) ? std::max(1, std::atoi(argv[2])) : 50;
    int pad = (argc >= 4) ? std::max(0, std::atoi(argv[3])) : 10;
    int pad_val = (argc >= 5) ? std::max(0, std::min(255, std::atoi(argv[4]))) : 128;

    Image src = Image::Create(img_path.c_str(), 3);
    if (src.Empty()) {
        std::cerr << "Failed to read image: " << img_path << std::endl;
        return 1;
    }
    std::cout << "Input: " << img_path << " (" << src.Height() << "x" << src.Width()
              << "x" << src.Channels() << "), iters=" << iters
              << ", pad=" << pad << ", val=" << pad_val << std::endl;

    // Pad
    auto t0 = std::chrono::steady_clock::now();
    Image padded;
    for (int i = 0; i < iters; ++i) {
        padded = src.Pad(pad, pad, pad, pad, std::vector<double>{static_cast<double>(pad_val)});
    }
    auto t1 = std::chrono::steady_clock::now();
    double avg_pad_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / std::max(1, iters);
    uint64_t sum_pad = sum_u8(padded);

    // FlipHorizontal
    t0 = std::chrono::steady_clock::now();
    Image fh;
    for (int i = 0; i < iters; ++i) {
        fh = src.FlipHorizontal();
    }
    t1 = std::chrono::steady_clock::now();
    double avg_fh_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / std::max(1, iters);
    uint64_t sum_fh = sum_u8(fh);

    // FlipVertical
    t0 = std::chrono::steady_clock::now();
    Image fv;
    for (int i = 0; i < iters; ++i) {
        fv = src.FlipVertical();
    }
    t1 = std::chrono::steady_clock::now();
    double avg_fv_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / std::max(1, iters);
    uint64_t sum_fv = sum_u8(fv);

    // Rotate90
    t0 = std::chrono::steady_clock::now();
    Image r90;
    for (int i = 0; i < iters; ++i) {
        r90 = src.Rotate90();
    }
    t1 = std::chrono::steady_clock::now();
    double avg_r90_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / std::max(1, iters);
    uint64_t sum_r90 = sum_u8(r90);

    // Rotate180
    t0 = std::chrono::steady_clock::now();
    Image r180;
    for (int i = 0; i < iters; ++i) {
        r180 = src.Rotate180();
    }
    t1 = std::chrono::steady_clock::now();
    double avg_r180_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / std::max(1, iters);
    uint64_t sum_r180 = sum_u8(r180);

    // Rotate270
    t0 = std::chrono::steady_clock::now();
    Image r270;
    for (int i = 0; i < iters; ++i) {
        r270 = src.Rotate270();
    }
    t1 = std::chrono::steady_clock::now();
    double avg_r270_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / std::max(1, iters);
    uint64_t sum_r270 = sum_u8(r270);

    std::cout << "[Pad]        sum=" << sum_pad  << " avg_ms=" << avg_pad_ms  << " (iters=" << iters << ")\n";
    std::cout << "[FlipH]      sum=" << sum_fh   << " avg_ms=" << avg_fh_ms   << " (iters=" << iters << ")\n";
    std::cout << "[FlipV]      sum=" << sum_fv   << " avg_ms=" << avg_fv_ms   << " (iters=" << iters << ")\n";
    std::cout << "[Rotate90]   sum=" << sum_r90  << " avg_ms=" << avg_r90_ms  << " (iters=" << iters << ")\n";
    std::cout << "[Rotate180]  sum=" << sum_r180 << " avg_ms=" << avg_r180_ms << " (iters=" << iters << ")\n";
    std::cout << "[Rotate270]  sum=" << sum_r270 << " avg_ms=" << avg_r270_ms << " (iters=" << iters << ")\n";

    // Save outputs
    padded.Write(("pad_" + std::string(GetCVBackend()) + ".jpg").c_str());
    fh.Write(("flip_h_" + std::string(GetCVBackend()) + ".jpg").c_str());
    fv.Write(("flip_v_" + std::string(GetCVBackend()) + ".jpg").c_str());
    r90.Write(("rotate90_" + std::string(GetCVBackend()) + ".jpg").c_str());
    r180.Write(("rotate180_" + std::string(GetCVBackend()) + ".jpg").c_str());
    r270.Write(("rotate270_" + std::string(GetCVBackend()) + ".jpg").c_str());
    std::cout << "Saved: pad_*.jpg, flip_*.jpg, rotate*.jpg" << std::endl;
    return 0;
}


