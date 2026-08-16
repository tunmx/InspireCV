// Image benchmark: U8 resize (nearest vs linear)
#include "../../common/common.h"
#include "image_test_utils.h"
#include <inspirecv/inspirecv.h>
#include <inspirecv/time_spend.h>
#include <vector>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <string>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <fstream>
#include <algorithm>

using inspirecv::Image;
using inspirecv::TimeSpend;
using inspirecv::TransformMatrix;
using inspirecv::ImageT;

namespace {

static bool save_benchmark_images() {
    const char* value = std::getenv("INSPIRECV_IMAGE_BENCHMARK_SAVE_IMAGES");
    return value == nullptr || std::strcmp(value, "0") != 0;
}

static inline void debug_save_image(const Image& img, const std::string& filename) {
    if (save_benchmark_images()) inspirecv_test_write_image(img, filename);
}
static inline void debug_save_image(const ImageT<float>& img, const std::string& filename) {
    if (save_benchmark_images()) inspirecv_test_write_image(img, filename);
}

static std::string make_name(const char* prefix, int inW, int inH, int outW, int outH, const char* extra = nullptr) {
    std::string name = std::string(prefix) + "_in" + std::to_string(inW) + "x" + std::to_string(inH)
                     + "_out" + std::to_string(outW) + "x" + std::to_string(outH);
    if (extra && *extra) {
        name += "_";
        name += extra;
    }
    name += ".png";
    return name;
}

// Bench result POD used by all measurements
struct BenchResult {
    double total_ms;
    double avg_ms;
    double mpix_per_s;
};

// Reporting infrastructure
struct BenchCsvRow {
    std::string dtype;     // "u8" or "f32"
    std::string op;        // tag/op name
    int inW, inH;
    int outW, outH;
    int loops;
    double total_ms;
    double avg_ms;
    double mpix_per_s;
};
static std::vector<BenchCsvRow> g_rows;
static const char* g_current_dtype = "u8";

static inline std::string to_lower_underscore(const char* s) {
    if (!s) return std::string();
    std::string out; out.reserve(std::strlen(s));
    for (const char* p = s; *p; ++p) {
        unsigned char uc = static_cast<unsigned char>(*p);
        char c = static_cast<char>(std::tolower(uc));
        out.push_back(std::isalnum(uc) ? c : '_');
    }
    // collapse multiple underscores and trim
    std::string res;
    bool prev_us = false;
    for (char c : out) {
        if (c == '_') {
            if (!prev_us) res.push_back(c);
            prev_us = true;
        } else {
            res.push_back(c);
            prev_us = false;
        }
    }
    if (!res.empty() && res.front() == '_') res.erase(res.begin());
    if (!res.empty() && res.back() == '_') res.pop_back();
    return res;
}

static void record_row(const char* dtype, const char* tag, int inW, int inH, int outW, int outH,
                       int loops, const BenchResult& r) {
    BenchCsvRow row;
    row.dtype = dtype ? dtype : "";
    row.op = tag ? tag : "";
    row.inW = inW; row.inH = inH; row.outW = outW; row.outH = outH;
    row.loops = loops;
    row.total_ms = r.total_ms;
    row.avg_ms = r.avg_ms;
    row.mpix_per_s = r.mpix_per_s;
    g_rows.push_back(std::move(row));
}

static void bench_report_flush() {
    if (g_rows.empty()) return;
    std::sort(g_rows.begin(), g_rows.end(), [](const BenchCsvRow& a, const BenchCsvRow& b){
        if (a.dtype != b.dtype) return a.dtype < b.dtype;
        if (a.op != b.op) return a.op < b.op;
        if (a.inW != b.inW) return a.inW < b.inW;
        if (a.inH != b.inH) return a.inH < b.inH;
        if (a.outW != b.outW) return a.outW < b.outW;
        if (a.outH != b.outH) return a.outH < b.outH;
        if (a.mpix_per_s != b.mpix_per_s) return a.mpix_per_s > b.mpix_per_s;
        return a.avg_ms < b.avg_ms;
    });
    const auto& bi = inspirecv::GetLibraryInfo();
    // Always write to a single text file like task bench style
    std::ofstream ofs("image_benchmark.txt", std::ios::out | std::ios::trunc);
    if (!ofs) return;
    // Header with build/optimization toggles
    ofs << "Image Benchmark\n";
    ofs << "System: " << (bi.system_name ? bi.system_name : "") << " (" << (bi.system_processor ? bi.system_processor : "") << ")\n";
    ofs << "Compiler: " << (bi.compiler_name ? bi.compiler_name : "") << " " << (bi.compiler_version ? bi.compiler_version : "") << " (C++" << bi.cxx_standard << ")\n";
    ofs << "BuildType: " << (bi.build_type ? bi.build_type : "") << "\n";
    ofs << "LTO: " << (bi.lto_enabled ? "true" : "false") << "\n";
    ofs << "SSE: " << (bi.sse_enabled ? "true" : "false") << "\n";
    ofs << "AVX2: " << (bi.avx2_enabled ? "true" : "false") << "\n";
    ofs << "NEON: " << (bi.neon_enabled ? "true" : "false") << "\n";
    ofs << "TilingOff: " << (bi.task_tiling_enabled ? "false" : "true") << "\n";
    ofs << "TileWidth: " << bi.task_tile_width << "\n";
    if (bi.cxx_flags && bi.cxx_flags[0]) {
        ofs << "CXXFlags: " << bi.cxx_flags << "\n";
        }
    ofs << "\n";
    ofs << "dtype  op  inWxH  outWxH  loops  total_ms  avg_ms  mpix_per_s\n";
    for (const auto& e : g_rows) {
        ofs << e.dtype << "  " << e.op << "  "
            << e.inW << "x" << e.inH << "  "
            << e.outW << "x" << e.outH << "  "
            << e.loops << "  "
            << std::fixed << std::setprecision(3) << e.total_ms << "  "
            << std::fixed << std::setprecision(3) << e.avg_ms << "  "
            << std::fixed << std::setprecision(3) << e.mpix_per_s << "\n";
    }
    ofs.flush();
}

struct ReportAtExit {
    ~ReportAtExit() { bench_report_flush(); }
};
static ReportAtExit g_report_at_exit;

static Image make_input_image_u8(int N, int channels) {
    const char* env = std::getenv("INSPIRECV_IMAGES_DIR");
    std::string dir = (env && *env) ? std::string(env) : std::string("../images");
    std::string path = dir + "/kun.jpg";
    auto img = Image::Create(path, channels);
    if (img.Empty()) {
        return Image::Create(); // empty indicates skip
    }
    if (img.Width() != N || img.Height() != N) {
        img = img.Resize(N, N, /*use_linear=*/true);
    }
    return img;
}

static ImageT<float> make_input_image_f32(int N, int channels) {
    const char* env = std::getenv("INSPIRECV_IMAGES_DIR");
    std::string dir = (env && *env) ? std::string(env) : std::string("../images");
    std::string path = dir + "/kun.jpg";
    auto img = ImageT<float>::Create(path, channels);
    if (img.Empty()) {
        return ImageT<float>::Create(); // empty indicates skip
    }
    if (img.Width() != N || img.Height() != N) {
        img = img.Resize(N, N, /*use_linear=*/true);
    }
    return img;
}

static BenchResult bench_resize_u8(const Image& src, int outW, int outH, bool use_linear, int loops) {
    // Warmup
    for (int i = 0; i < 5; ++i) {
        auto tmp = src.Resize(outW, outH, use_linear);
        (void)tmp;
    }
    TimeSpend t("resize_u8");
    t.Start();
    for (int i = 0; i < loops; ++i) {
        auto out = src.Resize(outW, outH, use_linear);
        (void)out;
    }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    const double total_ms = total_us / 1000.0;
    const double avg_ms = total_ms / std::max(1, loops);
    const double mpix_per_s = (static_cast<double>(outW) * outH * loops) / total_us; // MPix/s
    return { total_ms, avg_ms, mpix_per_s };
}

static BenchResult bench_resize_f32(const ImageT<float>& src, int outW, int outH, bool use_linear, int loops) {
    for (int i = 0; i < 5; ++i) { auto tmp = src.Resize(outW, outH, use_linear); (void)tmp; }
    TimeSpend t("resize_f32"); t.Start();
    for (int i = 0; i < loops; ++i) { auto out = src.Resize(outW, outH, use_linear); (void)out; }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    const double total_ms = total_us / 1000.0;
    const double avg_ms = total_ms / std::max(1, loops);
    const double mpix_per_s = (static_cast<double>(outW) * outH * loops) / total_us;
    return { total_ms, avg_ms, mpix_per_s };
}

static void print_row(const char* tag, int inW, int inH, int outW, int outH,
                      int loops, const BenchResult& r) {
    std::cout << "[" << tag << "] "
              << "in=" << inW << "x" << inH
              << " -> out=" << outW << "x" << outH
              << " loops=" << loops
              << " total_ms=" << std::fixed << std::setprecision(3) << r.total_ms
              << " avg_ms=" << std::fixed << std::setprecision(3) << r.avg_ms
              << " MPix/s=" << std::fixed << std::setprecision(3) << r.mpix_per_s
              << std::endl;
    record_row(g_current_dtype, tag, inW, inH, outW, outH, loops, r);
}

static BenchResult bench_warp_affine_u8(const Image& src, const TransformMatrix& M, int outW, int outH, int loops) {
    // Warmup
    for (int i = 0; i < 5; ++i) {
        auto tmp = src.WarpAffine(M, outW, outH);
        (void)tmp;
    }
    TimeSpend t("warp_affine_u8");
    t.Start();
    for (int i = 0; i < loops; ++i) {
        auto out = src.WarpAffine(M, outW, outH);
        (void)out;
    }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    const double total_ms = total_us / 1000.0;
    const double avg_ms = total_ms / std::max(1, loops);
    const double mpix_per_s = (static_cast<double>(outW) * outH * loops) / total_us; // MPix/s
    return { total_ms, avg_ms, mpix_per_s };
}

static BenchResult bench_warp_affine_f32(const ImageT<float>& src, const TransformMatrix& M, int outW, int outH, int loops) {
    for (int i = 0; i < 5; ++i) { auto tmp = src.WarpAffine(M, outW, outH); (void)tmp; }
    TimeSpend t("warp_affine_f32"); t.Start();
    for (int i = 0; i < loops; ++i) { auto out = src.WarpAffine(M, outW, outH); (void)out; }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    const double total_ms = total_us / 1000.0;
    const double avg_ms = total_ms / std::max(1, loops);
    const double mpix_per_s = (static_cast<double>(outW) * outH * loops) / total_us;
    return { total_ms, avg_ms, mpix_per_s };
}

static BenchResult bench_rotate90_u8(const Image& src, int loops) {
    const int outW = src.Width(), outH = src.Height();
    for (int i = 0; i < 5; ++i) { auto tmp = src.Rotate90(); (void)tmp; }
    TimeSpend t("rotate90_u8"); t.Start();
    for (int i = 0; i < loops; ++i) { auto out = src.Rotate90(); (void)out; }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    return { total_us / 1000.0, (total_us / 1000.0) / std::max(1, loops),
             (static_cast<double>(outW) * outH * loops) / total_us };
}

static BenchResult bench_rotate90_f32(const ImageT<float>& src, int loops) {
    const int outW = src.Width(), outH = src.Height();
    for (int i = 0; i < 5; ++i) { auto tmp = src.Rotate90(); (void)tmp; }
    TimeSpend t("rotate90_f32"); t.Start();
    for (int i = 0; i < loops; ++i) { auto out = src.Rotate90(); (void)out; }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    return { total_us / 1000.0, (total_us / 1000.0) / std::max(1, loops),
             (static_cast<double>(outW) * outH * loops) / total_us };
}

static BenchResult bench_rotate180_u8(const Image& src, int loops) {
    const int outW = src.Width(), outH = src.Height();
    for (int i = 0; i < 5; ++i) { auto tmp = src.Rotate180(); (void)tmp; }
    TimeSpend t("rotate180_u8"); t.Start();
    for (int i = 0; i < loops; ++i) { auto out = src.Rotate180(); (void)out; }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    return { total_us / 1000.0, (total_us / 1000.0) / std::max(1, loops),
             (static_cast<double>(outW) * outH * loops) / total_us };
}

static BenchResult bench_rotate180_f32(const ImageT<float>& src, int loops) {
    const int outW = src.Width(), outH = src.Height();
    for (int i = 0; i < 5; ++i) { auto tmp = src.Rotate180(); (void)tmp; }
    TimeSpend t("rotate180_f32"); t.Start();
    for (int i = 0; i < loops; ++i) { auto out = src.Rotate180(); (void)out; }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    return { total_us / 1000.0, (total_us / 1000.0) / std::max(1, loops),
             (static_cast<double>(outW) * outH * loops) / total_us };
}

static BenchResult bench_rotate270_u8(const Image& src, int loops) {
    const int outW = src.Width(), outH = src.Height();
    for (int i = 0; i < 5; ++i) { auto tmp = src.Rotate270(); (void)tmp; }
    TimeSpend t("rotate270_u8"); t.Start();
    for (int i = 0; i < loops; ++i) { auto out = src.Rotate270(); (void)out; }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    return { total_us / 1000.0, (total_us / 1000.0) / std::max(1, loops),
             (static_cast<double>(outW) * outH * loops) / total_us };
}

static BenchResult bench_rotate270_f32(const ImageT<float>& src, int loops) {
    const int outW = src.Width(), outH = src.Height();
    for (int i = 0; i < 5; ++i) { auto tmp = src.Rotate270(); (void)tmp; }
    TimeSpend t("rotate270_f32"); t.Start();
    for (int i = 0; i < loops; ++i) { auto out = src.Rotate270(); (void)out; }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    return { total_us / 1000.0, (total_us / 1000.0) / std::max(1, loops),
             (static_cast<double>(outW) * outH * loops) / total_us };
}

static BenchResult bench_crop_u8(const Image& src, int outW, int outH, int loops) {
    inspirecv::Rect<int> roi((src.Width() - outW) / 2, (src.Height() - outH) / 2, outW, outH);
    for (int i = 0; i < 5; ++i) { auto tmp = src.Crop(roi); (void)tmp; }
    TimeSpend t("crop_u8"); t.Start();
    for (int i = 0; i < loops; ++i) { auto out = src.Crop(roi); (void)out; }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    return { total_us / 1000.0, (total_us / 1000.0) / std::max(1, loops),
             (static_cast<double>(outW) * outH * loops) / total_us };
}

static BenchResult bench_crop_f32(const ImageT<float>& src, int outW, int outH, int loops) {
    inspirecv::Rect<int> roi((src.Width() - outW) / 2, (src.Height() - outH) / 2, outW, outH);
    for (int i = 0; i < 5; ++i) { auto tmp = src.Crop(roi); (void)tmp; }
    TimeSpend t("crop_f32"); t.Start();
    for (int i = 0; i < loops; ++i) { auto out = src.Crop(roi); (void)out; }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    return { total_us / 1000.0, (total_us / 1000.0) / std::max(1, loops),
             (static_cast<double>(outW) * outH * loops) / total_us };
}

static BenchResult bench_pad_u8(const Image& src, int top, int bottom, int left, int right, int loops) {
    const int outW = src.Width() + left + right;
    const int outH = src.Height() + top + bottom;
    const std::vector<double> black = {0.0, 0.0, 0.0};
    for (int i = 0; i < 5; ++i) { auto tmp = src.Pad(top, bottom, left, right, black); (void)tmp; }
    TimeSpend t("pad_u8"); t.Start();
    for (int i = 0; i < loops; ++i) { auto out = src.Pad(top, bottom, left, right, black); (void)out; }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    return { total_us / 1000.0, (total_us / 1000.0) / std::max(1, loops),
             (static_cast<double>(outW) * outH * loops) / total_us };
}

static BenchResult bench_pad_f32(const ImageT<float>& src, int top, int bottom, int left, int right, int loops) {
    const int outW = src.Width() + left + right;
    const int outH = src.Height() + top + bottom;
    const std::vector<double> black = {0.0, 0.0, 0.0};
    for (int i = 0; i < 5; ++i) { auto tmp = src.Pad(top, bottom, left, right, black); (void)tmp; }
    TimeSpend t("pad_f32"); t.Start();
    for (int i = 0; i < loops; ++i) { auto out = src.Pad(top, bottom, left, right, black); (void)out; }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    return { total_us / 1000.0, (total_us / 1000.0) / std::max(1, loops),
             (static_cast<double>(outW) * outH * loops) / total_us };
}

static BenchResult bench_togray_u8(const Image& src, int loops) {
    const int outW = src.Width(), outH = src.Height();
    for (int i = 0; i < 5; ++i) { auto tmp = src.ToGray(); (void)tmp; }
    TimeSpend t("togray_u8"); t.Start();
    for (int i = 0; i < loops; ++i) { auto out = src.ToGray(); (void)out; }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    return { total_us / 1000.0, (total_us / 1000.0) / std::max(1, loops),
             (static_cast<double>(outW) * outH * loops) / total_us };
}

static BenchResult bench_togray_f32(const ImageT<float>& src, int loops) {
    const int outW = src.Width(), outH = src.Height();
    for (int i = 0; i < 5; ++i) { auto tmp = src.ToGray(); (void)tmp; }
    TimeSpend t("togray_f32"); t.Start();
    for (int i = 0; i < loops; ++i) { auto out = src.ToGray(); (void)out; }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    return { total_us / 1000.0, (total_us / 1000.0) / std::max(1, loops),
             (static_cast<double>(outW) * outH * loops) / total_us };
}

static BenchResult bench_swaprb_u8(const Image& src, int loops) {
    const int outW = src.Width(), outH = src.Height();
    for (int i = 0; i < 5; ++i) { auto tmp = src.SwapRB(); (void)tmp; }
    TimeSpend t("swaprb_u8"); t.Start();
    for (int i = 0; i < loops; ++i) { auto out = src.SwapRB(); (void)out; }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    return { total_us / 1000.0, (total_us / 1000.0) / std::max(1, loops),
             (static_cast<double>(outW) * outH * loops) / total_us };
}

static BenchResult bench_swaprb_f32(const ImageT<float>& src, int loops) {
    const int outW = src.Width(), outH = src.Height();
    for (int i = 0; i < 5; ++i) { auto tmp = src.SwapRB(); (void)tmp; }
    TimeSpend t("swaprb_f32"); t.Start();
    for (int i = 0; i < loops; ++i) { auto out = src.SwapRB(); (void)out; }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    return { total_us / 1000.0, (total_us / 1000.0) / std::max(1, loops),
             (static_cast<double>(outW) * outH * loops) / total_us };
}

static BenchResult bench_mean_channels_u8(const Image& src, int loops) {
    const int outW = src.Width(), outH = src.Height();
    for (int i = 0; i < 5; ++i) { auto tmp = src.MeanChannels(); (void)tmp; }
    TimeSpend t("mean_channels_u8"); t.Start();
    for (int i = 0; i < loops; ++i) { auto out = src.MeanChannels(); (void)out; }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    return { total_us / 1000.0, (total_us / 1000.0) / std::max(1, loops),
             (static_cast<double>(outW) * outH * loops) / total_us };
}

static BenchResult bench_mean_channels_f32(const ImageT<float>& src, int loops) {
    const int outW = src.Width(), outH = src.Height();
    for (int i = 0; i < 5; ++i) { auto tmp = src.MeanChannels(); (void)tmp; }
    TimeSpend t("mean_channels_f32"); t.Start();
    for (int i = 0; i < loops; ++i) { auto out = src.MeanChannels(); (void)out; }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    return { total_us / 1000.0, (total_us / 1000.0) / std::max(1, loops),
             (static_cast<double>(outW) * outH * loops) / total_us };
}

static BenchResult bench_mul_u8(const Image& src, double scale, int loops) {
    const int outW = src.Width(), outH = src.Height();
    for (int i = 0; i < 5; ++i) { auto tmp = src.Mul(scale); (void)tmp; }
    TimeSpend t("mul_u8"); t.Start();
    for (int i = 0; i < loops; ++i) { auto out = src.Mul(scale); (void)out; }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    return { total_us / 1000.0, (total_us / 1000.0) / std::max(1, loops),
             (static_cast<double>(outW) * outH * loops) / total_us };
}

static BenchResult bench_mul_f32(const ImageT<float>& src, double scale, int loops) {
    const int outW = src.Width(), outH = src.Height();
    for (int i = 0; i < 5; ++i) { auto tmp = src.Mul(scale); (void)tmp; }
    TimeSpend t("mul_f32"); t.Start();
    for (int i = 0; i < loops; ++i) { auto out = src.Mul(scale); (void)out; }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    return { total_us / 1000.0, (total_us / 1000.0) / std::max(1, loops),
             (static_cast<double>(outW) * outH * loops) / total_us };
}

static BenchResult bench_add_u8(const Image& src, double value, int loops) {
    const int outW = src.Width(), outH = src.Height();
    for (int i = 0; i < 5; ++i) { auto tmp = src.Add(value); (void)tmp; }
    TimeSpend t("add_u8"); t.Start();
    for (int i = 0; i < loops; ++i) { auto out = src.Add(value); (void)out; }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    return { total_us / 1000.0, (total_us / 1000.0) / std::max(1, loops),
             (static_cast<double>(outW) * outH * loops) / total_us };
}

static BenchResult bench_add_f32(const ImageT<float>& src, double value, int loops) {
    const int outW = src.Width(), outH = src.Height();
    for (int i = 0; i < 5; ++i) { auto tmp = src.Add(value); (void)tmp; }
    TimeSpend t("add_f32"); t.Start();
    for (int i = 0; i < loops; ++i) { auto out = src.Add(value); (void)out; }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    return { total_us / 1000.0, (total_us / 1000.0) / std::max(1, loops),
             (static_cast<double>(outW) * outH * loops) / total_us };
}

static BenchResult bench_threshold_u8(const Image& gray, double thr, double maxv, int type, int loops) {
    const int outW = gray.Width(), outH = gray.Height();
    for (int i = 0; i < 5; ++i) { auto tmp = gray.Threshold(thr, maxv, type); (void)tmp; }
    TimeSpend t("threshold_u8"); t.Start();
    for (int i = 0; i < loops; ++i) { auto out = gray.Threshold(thr, maxv, type); (void)out; }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    return { total_us / 1000.0, (total_us / 1000.0) / std::max(1, loops),
             (static_cast<double>(outW) * outH * loops) / total_us };
}

static BenchResult bench_threshold_f32(const ImageT<float>& gray, double thr, double maxv, int type, int loops) {
    const int outW = gray.Width(), outH = gray.Height();
    for (int i = 0; i < 5; ++i) { auto tmp = gray.Threshold(thr, maxv, type); (void)tmp; }
    TimeSpend t("threshold_f32"); t.Start();
    for (int i = 0; i < loops; ++i) { auto out = gray.Threshold(thr, maxv, type); (void)out; }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    return { total_us / 1000.0, (total_us / 1000.0) / std::max(1, loops),
             (static_cast<double>(outW) * outH * loops) / total_us };
}

static BenchResult bench_gauss_u8(const Image& src, int ksize, double sigma, int loops) {
    const int outW = src.Width(), outH = src.Height();
    for (int i = 0; i < 5; ++i) { auto tmp = src.GaussianBlur(ksize, sigma); (void)tmp; }
    TimeSpend t("gauss_u8"); t.Start();
    for (int i = 0; i < loops; ++i) { auto out = src.GaussianBlur(ksize, sigma); (void)out; }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    return { total_us / 1000.0, (total_us / 1000.0) / std::max(1, loops),
             (static_cast<double>(outW) * outH * loops) / total_us };
}

static BenchResult bench_gauss_f32(const ImageT<float>& src, int ksize, double sigma, int loops) {
    const int outW = src.Width(), outH = src.Height();
    for (int i = 0; i < 5; ++i) { auto tmp = src.GaussianBlur(ksize, sigma); (void)tmp; }
    TimeSpend t("gauss_f32"); t.Start();
    for (int i = 0; i < loops; ++i) { auto out = src.GaussianBlur(ksize, sigma); (void)out; }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    return { total_us / 1000.0, (total_us / 1000.0) / std::max(1, loops),
             (static_cast<double>(outW) * outH * loops) / total_us };
}

static BenchResult bench_erode_u8(const Image& gray, int ksize, int iters, int loops) {
    const int outW = gray.Width(), outH = gray.Height();
    for (int i = 0; i < 5; ++i) { auto tmp = gray.Erode(ksize, iters); (void)tmp; }
    TimeSpend t("erode_u8"); t.Start();
    for (int i = 0; i < loops; ++i) { auto out = gray.Erode(ksize, iters); (void)out; }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    return { total_us / 1000.0, (total_us / 1000.0) / std::max(1, loops),
             (static_cast<double>(outW) * outH * loops) / total_us };
}

static BenchResult bench_erode_f32(const ImageT<float>& gray, int ksize, int iters, int loops) {
    const int outW = gray.Width(), outH = gray.Height();
    for (int i = 0; i < 5; ++i) { auto tmp = gray.Erode(ksize, iters); (void)tmp; }
    TimeSpend t("erode_f32"); t.Start();
    for (int i = 0; i < loops; ++i) { auto out = gray.Erode(ksize, iters); (void)out; }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    return { total_us / 1000.0, (total_us / 1000.0) / std::max(1, loops),
             (static_cast<double>(outW) * outH * loops) / total_us };
}

static BenchResult bench_dilate_u8(const Image& gray, int ksize, int iters, int loops) {
    const int outW = gray.Width(), outH = gray.Height();
    for (int i = 0; i < 5; ++i) { auto tmp = gray.Dilate(ksize, iters); (void)tmp; }
    TimeSpend t("dilate_u8"); t.Start();
    for (int i = 0; i < loops; ++i) { auto out = gray.Dilate(ksize, iters); (void)out; }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    return { total_us / 1000.0, (total_us / 1000.0) / std::max(1, loops),
             (static_cast<double>(outW) * outH * loops) / total_us };
}

static BenchResult bench_dilate_f32(const ImageT<float>& gray, int ksize, int iters, int loops) {
    const int outW = gray.Width(), outH = gray.Height();
    for (int i = 0; i < 5; ++i) { auto tmp = gray.Dilate(ksize, iters); (void)tmp; }
    TimeSpend t("dilate_f32"); t.Start();
    for (int i = 0; i < loops; ++i) { auto out = gray.Dilate(ksize, iters); (void)out; }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    return { total_us / 1000.0, (total_us / 1000.0) / std::max(1, loops),
             (static_cast<double>(outW) * outH * loops) / total_us };
}

static BenchResult bench_flip_h_u8(const Image& src, int loops) {
    const int outW = src.Width(), outH = src.Height();
    for (int i = 0; i < 5; ++i) { auto tmp = src.FlipHorizontal(); (void)tmp; }
    TimeSpend t("flip_h_u8"); t.Start();
    for (int i = 0; i < loops; ++i) { auto out = src.FlipHorizontal(); (void)out; }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    return { total_us / 1000.0, (total_us / 1000.0) / std::max(1, loops),
             (static_cast<double>(outW) * outH * loops) / total_us };
}

static BenchResult bench_flip_h_f32(const ImageT<float>& src, int loops) {
    const int outW = src.Width(), outH = src.Height();
    for (int i = 0; i < 5; ++i) { auto tmp = src.FlipHorizontal(); (void)tmp; }
    TimeSpend t("flip_h_f32"); t.Start();
    for (int i = 0; i < loops; ++i) { auto out = src.FlipHorizontal(); (void)out; }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    return { total_us / 1000.0, (total_us / 1000.0) / std::max(1, loops),
             (static_cast<double>(outW) * outH * loops) / total_us };
}

static BenchResult bench_flip_v_u8(const Image& src, int loops) {
    const int outW = src.Width(), outH = src.Height();
    for (int i = 0; i < 5; ++i) { auto tmp = src.FlipVertical(); (void)tmp; }
    TimeSpend t("flip_v_u8"); t.Start();
    for (int i = 0; i < loops; ++i) { auto out = src.FlipVertical(); (void)out; }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    return { total_us / 1000.0, (total_us / 1000.0) / std::max(1, loops),
             (static_cast<double>(outW) * outH * loops) / total_us };
}

static BenchResult bench_flip_v_f32(const ImageT<float>& src, int loops) {
    const int outW = src.Width(), outH = src.Height();
    for (int i = 0; i < 5; ++i) { auto tmp = src.FlipVertical(); (void)tmp; }
    TimeSpend t("flip_v_f32"); t.Start();
    for (int i = 0; i < loops; ++i) { auto out = src.FlipVertical(); (void)out; }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    return { total_us / 1000.0, (total_us / 1000.0) / std::max(1, loops),
             (static_cast<double>(outW) * outH * loops) / total_us };
}

static BenchResult bench_absdiff_u8(const Image& a, const Image& b, int loops) {
    const int outW = a.Width(), outH = a.Height();
    for (int i = 0; i < 5; ++i) { auto tmp = a.AbsDiff(b); (void)tmp; }
    TimeSpend t("absdiff_u8"); t.Start();
    for (int i = 0; i < loops; ++i) { auto out = a.AbsDiff(b); (void)out; }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    return { total_us / 1000.0, (total_us / 1000.0) / std::max(1, loops),
             (static_cast<double>(outW) * outH * loops) / total_us };
}

static BenchResult bench_absdiff_f32(const ImageT<float>& a, const ImageT<float>& b, int loops) {
    const int outW = a.Width(), outH = a.Height();
    for (int i = 0; i < 5; ++i) { auto tmp = a.AbsDiff(b); (void)tmp; }
    TimeSpend t("absdiff_f32"); t.Start();
    for (int i = 0; i < loops; ++i) { auto out = a.AbsDiff(b); (void)out; }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    return { total_us / 1000.0, (total_us / 1000.0) / std::max(1, loops),
             (static_cast<double>(outW) * outH * loops) / total_us };
}

static inspirecv::Image make_gradient_mask_u8(int w, int h) {
    std::vector<uint8_t> m(static_cast<size_t>(w) * h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float fx = static_cast<float>(x) / std::max(1, w - 1);
            float fy = static_cast<float>(y) / std::max(1, h - 1);
            float v = 255.0f * 0.5f * (fx + fy);
            m[static_cast<size_t>(y) * w + x] = static_cast<uint8_t>(std::max(0.f, std::min(255.f, v)));
        }
    }
    return Image::Create(w, h, 1, m.data(), true);
}

static BenchResult bench_blend_u8(const Image& a, const Image& b, const Image& mask, int loops) {
    const int outW = a.Width(), outH = a.Height();
    for (int i = 0; i < 5; ++i) { auto tmp = a.Blend(b, mask); (void)tmp; }
    TimeSpend t("blend_u8"); t.Start();
    for (int i = 0; i < loops; ++i) { auto out = a.Blend(b, mask); (void)out; }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    return { total_us / 1000.0, (total_us / 1000.0) / std::max(1, loops),
             (static_cast<double>(outW) * outH * loops) / total_us };
}

static BenchResult bench_blend_f32(const ImageT<float>& a, const ImageT<float>& b, const Image& mask, int loops) {
    const int outW = a.Width(), outH = a.Height();
    for (int i = 0; i < 5; ++i) { auto tmp = a.Blend(b, mask); (void)tmp; }
    TimeSpend t("blend_f32"); t.Start();
    for (int i = 0; i < loops; ++i) { auto out = a.Blend(b, mask); (void)out; }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    return { total_us / 1000.0, (total_us / 1000.0) / std::max(1, loops),
             (static_cast<double>(outW) * outH * loops) / total_us };
}

static BenchResult bench_draw_line_u8(Image& canvas, int loops) {
    const int outW = canvas.Width(), outH = canvas.Height();
    inspirecv::Point<int> p1(0, 0), p2(outW - 1, outH - 1);
    const std::vector<double> red = {0.0, 0.0, 255.0};
    for (int i = 0; i < 5; ++i) { canvas.DrawLine(p1, p2, red, 1); }
    TimeSpend t("draw_line_u8"); t.Start();
    for (int i = 0; i < loops; ++i) { canvas.DrawLine(p1, p2, red, 1); }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    return { total_us / 1000.0, (total_us / 1000.0) / std::max(1, loops),
             (static_cast<double>(outW) * outH * loops) / total_us };
}

static BenchResult bench_draw_line_f32(ImageT<float>& canvas, int loops) {
    const int outW = canvas.Width(), outH = canvas.Height();
    inspirecv::Point<int> p1(0, 0), p2(outW - 1, outH - 1);
    const std::vector<double> red = {0.0, 0.0, 255.0};
    for (int i = 0; i < 5; ++i) { canvas.DrawLine(p1, p2, red, 1); }
    TimeSpend t("draw_line_f32"); t.Start();
    for (int i = 0; i < loops; ++i) { canvas.DrawLine(p1, p2, red, 1); }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    return { total_us / 1000.0, (total_us / 1000.0) / std::max(1, loops),
             (static_cast<double>(outW) * outH * loops) / total_us };
}

static BenchResult bench_draw_rect_u8(Image& canvas, int loops) {
    const int outW = canvas.Width(), outH = canvas.Height();
    inspirecv::Rect<int> r(outW / 4, outH / 4, outW / 2, outH / 2);
    const std::vector<double> green = {0.0, 255.0, 0.0};
    for (int i = 0; i < 5; ++i) { canvas.DrawRect(r, green, 1); }
    TimeSpend t("draw_rect_u8"); t.Start();
    for (int i = 0; i < loops; ++i) { canvas.DrawRect(r, green, 1); }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    return { total_us / 1000.0, (total_us / 1000.0) / std::max(1, loops),
             (static_cast<double>(outW) * outH * loops) / total_us };
}

static BenchResult bench_draw_rect_f32(ImageT<float>& canvas, int loops) {
    const int outW = canvas.Width(), outH = canvas.Height();
    inspirecv::Rect<int> r(outW / 4, outH / 4, outW / 2, outH / 2);
    const std::vector<double> green = {0.0, 255.0, 0.0};
    for (int i = 0; i < 5; ++i) { canvas.DrawRect(r, green, 1); }
    TimeSpend t("draw_rect_f32"); t.Start();
    for (int i = 0; i < loops; ++i) { canvas.DrawRect(r, green, 1); }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    return { total_us / 1000.0, (total_us / 1000.0) / std::max(1, loops),
             (static_cast<double>(outW) * outH * loops) / total_us };
}

static BenchResult bench_draw_circle_u8(Image& canvas, int loops) {
    const int outW = canvas.Width(), outH = canvas.Height();
    inspirecv::Point<int> c(outW / 2, outH / 2);
    const std::vector<double> blue = {255.0, 0.0, 0.0};
    for (int i = 0; i < 5; ++i) { canvas.DrawCircle(c, std::max(1, outW / 6), blue, 1); }
    TimeSpend t("draw_circle_u8"); t.Start();
    for (int i = 0; i < loops; ++i) { canvas.DrawCircle(c, std::max(1, outW / 6), blue, 1); }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    return { total_us / 1000.0, (total_us / 1000.0) / std::max(1, loops),
             (static_cast<double>(outW) * outH * loops) / total_us };
}

static BenchResult bench_draw_circle_f32(ImageT<float>& canvas, int loops) {
    const int outW = canvas.Width(), outH = canvas.Height();
    inspirecv::Point<int> c(outW / 2, outH / 2);
    const std::vector<double> blue = {255.0, 0.0, 0.0};
    for (int i = 0; i < 5; ++i) { canvas.DrawCircle(c, std::max(1, outW / 6), blue, 1); }
    TimeSpend t("draw_circle_f32"); t.Start();
    for (int i = 0; i < loops; ++i) { canvas.DrawCircle(c, std::max(1, outW / 6), blue, 1); }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    return { total_us / 1000.0, (total_us / 1000.0) / std::max(1, loops),
             (static_cast<double>(outW) * outH * loops) / total_us };
}

static BenchResult bench_fill_rect_u8(Image& canvas, int loops) {
    const int outW = canvas.Width(), outH = canvas.Height();
    inspirecv::Rect<int> r(outW / 3, outH / 3, outW / 3, outH / 3);
    const std::vector<double> white = {255.0, 255.0, 255.0};
    for (int i = 0; i < 5; ++i) { canvas.Fill(r, white); }
    TimeSpend t("fill_rect_u8"); t.Start();
    for (int i = 0; i < loops; ++i) { canvas.Fill(r, white); }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    return { total_us / 1000.0, (total_us / 1000.0) / std::max(1, loops),
             (static_cast<double>(outW) * outH * loops) / total_us };
}

static BenchResult bench_fill_rect_f32(ImageT<float>& canvas, int loops) {
    const int outW = canvas.Width(), outH = canvas.Height();
    inspirecv::Rect<int> r(outW / 3, outH / 3, outW / 3, outH / 3);
    const std::vector<double> white = {255.0, 255.0, 255.0};
    for (int i = 0; i < 5; ++i) { canvas.Fill(r, white); }
    TimeSpend t("fill_rect_f32"); t.Start();
    for (int i = 0; i < loops; ++i) { canvas.Fill(r, white); }
    t.Stop();
    const double total_us = static_cast<double>(t.Total());
    return { total_us / 1000.0, (total_us / 1000.0) / std::max(1, loops),
             (static_cast<double>(outW) * outH * loops) / total_us };
}

static TransformMatrix make_centered_affine(float angle_deg, float sx, float sy, float tx, float ty,
                                            int width, int height) {
    const float cx = (static_cast<float>(width)  - 1.0f) * 0.5f;
    const float cy = (static_cast<float>(height) - 1.0f) * 0.5f;
    const float rad = angle_deg * static_cast<float>(M_PI) / 180.0f;
    const float a = std::cos(rad);
    const float b = std::sin(rad);
    // dest(x,y) -> src
    const float a11 = a * sx;
    const float a12 = b * sy;
    const float a21 = -b * sx;
    const float a22 = a * sy;
    const float b1 = cx - a11 * cx - a12 * cy + tx;
    const float b2 = cy - a21 * cx - a22 * cy + ty;
    return TransformMatrix::Create(a11, a12, b1, a21, a22, b2);
}

} // namespace

TEST_CASE("image_benchmark_resize_nearest_u8", "[bench][image][resize][u8][nearest]") {
    const int sizes[] = {128, 256, 512, 1024};
    const int channels = 3;
    const int loops = 100;
    g_current_dtype = "u8";

    for (int N : sizes) {
        auto img = make_input_image_u8(N, channels);
        if (img.Empty()) {
            INFO("Skip benchmark: input image not found (set INSPIRECV_IMAGES_DIR or ensure ../images/kun.jpg exists)");
            SUCCEED();
            return;
        }

        // Downsample N -> N/2
        {
            int outW = std::max(1, N / 2);
            int outH = std::max(1, N / 2);
            auto r = bench_resize_u8(img, outW, outH, /*use_linear=*/false, loops);
            print_row("resize_u8_nearest", N, N, outW, outH, loops, r);
            auto out = img.Resize(outW, outH, /*use_linear=*/false);
            debug_save_image(out, make_name("resize_u8_nearest", N, N, outW, outH, "down"));
        }
        // Upsample N -> 2N
        {
            int outW = N * 2;
            int outH = N * 2;
            auto r = bench_resize_u8(img, outW, outH, /*use_linear=*/false, loops);
            print_row("resize_u8_nearest", N, N, outW, outH, loops, r);
            auto out = img.Resize(outW, outH, /*use_linear=*/false);
            debug_save_image(out, make_name("resize_u8_nearest", N, N, outW, outH, "up"));
        }
    }
    SUCCEED();
}

// =========================== F32 BENCHMARKS ===========================

TEST_CASE("image_benchmark_resize_nearest_f32", "[bench][image][resize][f32][nearest]") {
    const int sizes[] = {128, 256, 512, 1024};
    const int channels = 3;
    const int loops = 100;
    g_current_dtype = "f32";

    for (int N : sizes) {
        auto img = make_input_image_f32(N, channels);
        if (img.Empty()) {
            INFO("Skip benchmark: input image not found (set INSPIRECV_IMAGES_DIR or ensure ../images/kun.jpg exists)");
            SUCCEED();
            return;
        }
        // Downsample N -> N/2
        {
            int outW = std::max(1, N / 2), outH = std::max(1, N / 2);
            auto r = bench_resize_f32(img, outW, outH, /*use_linear=*/false, loops);
            print_row("resize_f32_nearest", N, N, outW, outH, loops, r);
            auto out = img.Resize(outW, outH, /*use_linear=*/false);
            debug_save_image(out, make_name("resize_f32_nearest", N, N, outW, outH, "down"));
        }
        // Upsample N -> 2N
        {
            int outW = N * 2, outH = N * 2;
            auto r = bench_resize_f32(img, outW, outH, /*use_linear=*/false, loops);
            print_row("resize_f32_nearest", N, N, outW, outH, loops, r);
            auto out = img.Resize(outW, outH, /*use_linear=*/false);
            debug_save_image(out, make_name("resize_f32_nearest", N, N, outW, outH, "up"));
        }
    }
    SUCCEED();
}

TEST_CASE("image_benchmark_resize_linear_f32", "[bench][image][resize][f32][linear]") {
    const int sizes[] = {128, 256, 512, 1024};
    const int channels = 3;
    const int loops = 100;
    g_current_dtype = "f32";

    for (int N : sizes) {
        auto img = make_input_image_f32(N, channels);
        if (img.Empty()) {
            INFO("Skip benchmark: input image not found (set INSPIRECV_IMAGES_DIR or ensure ../images/kun.jpg exists)");
            SUCCEED();
            return;
        }
        // Downsample N -> N/2
        {
            int outW = std::max(1, N / 2), outH = std::max(1, N / 2);
            auto r = bench_resize_f32(img, outW, outH, /*use_linear=*/true, loops);
            print_row("resize_f32_linear", N, N, outW, outH, loops, r);
            auto out = img.Resize(outW, outH, /*use_linear=*/true);
            debug_save_image(out, make_name("resize_f32_linear", N, N, outW, outH, "down"));
        }
        // Upsample N -> 2N
        {
            int outW = N * 2, outH = N * 2;
            auto r = bench_resize_f32(img, outW, outH, /*use_linear=*/true, loops);
            print_row("resize_f32_linear", N, N, outW, outH, loops, r);
            auto out = img.Resize(outW, outH, /*use_linear=*/true);
            debug_save_image(out, make_name("resize_f32_linear", N, N, outW, outH, "up"));
        }
    }
    SUCCEED();
}

TEST_CASE("image_benchmark_resize_linear_1p5x_f32", "[bench][image][resize][f32][linear][1p5x]") {
    const int sizes[] = {128, 256, 512, 1024};
    const int channels = 3;
    const int loops = 100;
    g_current_dtype = "f32";
    for (int N : sizes) {
        auto img = make_input_image_f32(N, channels);
        if (img.Empty()) { INFO("Skip benchmark: input image not found"); SUCCEED(); return; }
        int outW = N + N / 2, outH = N + N / 2;
        auto r = bench_resize_f32(img, outW, outH, /*use_linear=*/true, loops);
        print_row("resize_f32_linear_1p5x", N, N, outW, outH, loops, r);
        auto out = img.Resize(outW, outH, /*use_linear=*/true);
        debug_save_image(out, make_name("resize_f32_linear_1p5x", N, N, outW, outH));
    }
    SUCCEED();
}

TEST_CASE("image_benchmark_rotate_f32", "[bench][image][rotate][f32]") {
    const int sizes[] = {128, 256, 512, 1024};
    const int channels = 3;
    const int loops = 100;
    g_current_dtype = "f32";
    for (int N : sizes) {
        auto img = make_input_image_f32(N, channels);
        if (img.Empty()) { INFO("Skip benchmark: input image not found"); SUCCEED(); return; }
        { auto r = bench_rotate90_f32(img, loops);  print_row("rotate90_f32",  N, N, N, N, loops, r);  debug_save_image(img.Rotate90(),  make_name("rotate90_f32",  N, N, N, N)); }
        { auto r = bench_rotate180_f32(img, loops); print_row("rotate180_f32", N, N, N, N, loops, r); debug_save_image(img.Rotate180(), make_name("rotate180_f32", N, N, N, N)); }
        { auto r = bench_rotate270_f32(img, loops); print_row("rotate270_f32", N, N, N, N, loops, r); debug_save_image(img.Rotate270(), make_name("rotate270_f32", N, N, N, N)); }
    }
    SUCCEED();
}

TEST_CASE("image_benchmark_crop_pad_f32", "[bench][image][crop][pad][f32]") {
    const int sizes[] = {128, 256, 512, 1024};
    const int channels = 3;
    const int loops = 100;
    g_current_dtype = "f32";
    for (int N : sizes) {
        auto img = make_input_image_f32(N, channels);
        if (img.Empty()) { INFO("Skip benchmark: input image not found"); SUCCEED(); return; }
        {
            int outW = std::max(1, N / 2), outH = std::max(1, N / 2);
            auto r = bench_crop_f32(img, outW, outH, loops);
            print_row("crop_f32_center", N, N, outW, outH, loops, r);
            auto roi = inspirecv::Rect<int>((N - outW) / 2, (N - outH) / 2, outW, outH);
            debug_save_image(img.Crop(roi), make_name("crop_f32_center", N, N, outW, outH));
        }
        {
            int top = 10, bottom = 10, left = 10, right = 10;
            auto r = bench_pad_f32(img, top, bottom, left, right, loops);
            print_row("pad_f32_10_all", N, N, N + left + right, N + top + bottom, loops, r);
            const std::vector<double> black = {0.0, 0.0, 0.0};
            debug_save_image(img.Pad(top, bottom, left, right, black),
                             make_name("pad_f32_10_all", N, N, N + left + right, N + top + bottom));
        }
    }
    SUCCEED();
}

TEST_CASE("image_benchmark_color_f32", "[bench][image][color][f32]") {
    const int sizes[] = {128, 256, 512, 1024};
    const int channels = 3;
    const int loops = 100;
    g_current_dtype = "f32";
    for (int N : sizes) {
        auto img = make_input_image_f32(N, channels);
        if (img.Empty()) { INFO("Skip benchmark: input image not found"); SUCCEED(); return; }
        { auto r = bench_togray_f32(img, loops);        print_row("to_gray_f32", N, N, N, N, loops, r); debug_save_image(img.ToGray(), make_name("to_gray_f32", N, N, N, N)); }
        { auto r = bench_swaprb_f32(img, loops);        print_row("swap_rb_f32", N, N, N, N, loops, r); debug_save_image(img.SwapRB(), make_name("swap_rb_f32", N, N, N, N)); }
        { auto r = bench_mean_channels_f32(img, loops); print_row("mean_channels_f32", N, N, N, N, loops, r); debug_save_image(img.MeanChannels(), make_name("mean_channels_f32", N, N, N, N)); }
    }
    SUCCEED();
}

TEST_CASE("image_benchmark_arith_thresh_f32", "[bench][image][arith][thresh][f32]") {
    const int sizes[] = {128, 256, 512, 1024};
    const int channels = 3;
    const int loops = 100;
    g_current_dtype = "f32";
    for (int N : sizes) {
        auto img = make_input_image_f32(N, channels);
        if (img.Empty()) { INFO("Skip benchmark: input image not found"); SUCCEED(); return; }
        { auto r = bench_mul_f32(img, 1.2, loops);  print_row("mul_f32_1p2", N, N, N, N, loops, r); debug_save_image(img.Mul(1.2), make_name("mul_f32_1p2", N, N, N, N)); }
        { auto r = bench_add_f32(img, 10.0, loops); print_row("add_f32_10",  N, N, N, N, loops, r); debug_save_image(img.Add(10.0), make_name("add_f32_10",  N, N, N, N)); }
        auto gray = img.ToGray();
        { auto r = bench_threshold_f32(gray, 100.0, 255.0, 0, loops); print_row("threshold_f32_100_255", N, N, N, N, loops, r);
          debug_save_image(gray.Threshold(100.0, 255.0, 0), make_name("threshold_f32_100_255", N, N, N, N)); }
    }
    SUCCEED();
}

TEST_CASE("image_benchmark_filter_morph_f32", "[bench][image][filter][morph][f32]") {
    const int sizes[] = {128, 256, 512, 1024};
    const int channels = 3;
    const int loops = 100;
    g_current_dtype = "f32";
    for (int N : sizes) {
        auto img = make_input_image_f32(N, channels);
        if (img.Empty()) { INFO("Skip benchmark: input image not found"); SUCCEED(); return; }
        { auto r = bench_gauss_f32(img, 5, 0.0, loops); print_row("gauss5_f32", N, N, N, N, loops, r); debug_save_image(img.GaussianBlur(5, 0.0), make_name("gauss5_f32", N, N, N, N)); }
        auto gray = img.ToGray();
        { auto r = bench_erode_f32(gray, 3, 1, loops);  print_row("erode3_f32", N, N, N, N, loops, r); debug_save_image(gray.Erode(3, 1), make_name("erode3_f32", N, N, N, N)); }
        { auto r = bench_dilate_f32(gray, 3, 1, loops); print_row("dilate3_f32", N, N, N, N, loops, r); debug_save_image(gray.Dilate(3, 1), make_name("dilate3_f32", N, N, N, N)); }
    }
    SUCCEED();
}

TEST_CASE("image_benchmark_warp_affine_f32", "[bench][image][warp_affine][f32]") {
    const int sizes[] = {128, 256, 512, 1024};
    const int channels = 3;
    const int loops = 100;
    g_current_dtype = "f32";
    for (int N : sizes) {
        auto img = make_input_image_f32(N, channels);
        if (img.Empty()) { INFO("Skip benchmark: input image not found"); SUCCEED(); return; }
        const int outW = N, outH = N;
        { auto M = TransformMatrix::Identity();
          auto r = bench_warp_affine_f32(img, M, outW, outH, loops); print_row("warp_affine_f32_identity", N, N, outW, outH, loops, r);
          debug_save_image(img.WarpAffine(M, outW, outH), make_name("warp_affine_f32_identity", N, N, outW, outH)); }
        { float tx = N * 0.1f, ty = -N * 0.1f;
          auto M = make_centered_affine(0.0f, 1.0f, 1.0f, tx, ty, outW, outH);
          auto r = bench_warp_affine_f32(img, M, outW, outH, loops); print_row("warp_affine_f32_translate", N, N, outW, outH, loops, r);
          debug_save_image(img.WarpAffine(M, outW, outH), make_name("warp_affine_f32_translate", N, N, outW, outH)); }
        { auto M = make_centered_affine(0.0f, 0.5f, 0.5f, 0.0f, 0.0f, outW, outH);
          auto r = bench_warp_affine_f32(img, M, outW, outH, loops); print_row("warp_affine_f32_scale05", N, N, outW, outH, loops, r);
          debug_save_image(img.WarpAffine(M, outW, outH), make_name("warp_affine_f32_scale05", N, N, outW, outH)); }
        { auto M = make_centered_affine(30.0f, 1.0f, 1.0f, 0.0f, 0.0f, outW, outH);
          auto r = bench_warp_affine_f32(img, M, outW, outH, loops); print_row("warp_affine_f32_rot30", N, N, outW, outH, loops, r);
          debug_save_image(img.WarpAffine(M, outW, outH), make_name("warp_affine_f32_rot30", N, N, outW, outH)); }
    }
    SUCCEED();
}

TEST_CASE("image_benchmark_absdiff_blend_f32", "[bench][image][absdiff][blend][f32]") {
    const int sizes[] = {128, 256, 512, 1024};
    const int channels = 3;
    const int loops = 100;
    g_current_dtype = "f32";
    for (int N : sizes) {
        auto img = make_input_image_f32(N, channels);
        if (img.Empty()) { INFO("Skip benchmark: input image not found"); SUCCEED(); return; }
        auto other = img.Rotate180();
        { auto r = bench_absdiff_f32(img, other, loops); print_row("absdiff_f32", N, N, N, N, loops, r);
          debug_save_image(img.AbsDiff(other), make_name("absdiff_f32", N, N, N, N)); }
        { auto swapped = img.SwapRB(); auto mask = make_gradient_mask_u8(N, N);
          auto r = bench_blend_f32(img, swapped, mask, loops); print_row("blend_f32_gradmask", N, N, N, N, loops, r);
          debug_save_image(img.Blend(swapped, mask), make_name("blend_f32_gradmask", N, N, N, N)); }
    }
    SUCCEED();
}

TEST_CASE("image_benchmark_draw_ops_f32", "[bench][image][draw][f32]") {
    const int sizes[] = {128, 256, 512, 1024};
    const int channels = 3;
    const int loops = 100;
    g_current_dtype = "f32";
    for (int N : sizes) {
        auto base = make_input_image_f32(N, channels);
        if (base.Empty()) { INFO("Skip benchmark: input image not found"); SUCCEED(); return; }
        auto canvas = base.Clone();
        { auto r = bench_draw_line_f32(canvas, loops);   print_row("draw_line_f32",   N, N, N, N, loops, r);   debug_save_image(canvas, make_name("draw_line_f32",   N, N, N, N)); }
        { auto r = bench_draw_rect_f32(canvas, loops);   print_row("draw_rect_f32",   N, N, N, N, loops, r);   debug_save_image(canvas, make_name("draw_rect_f32",   N, N, N, N)); }
        { auto r = bench_draw_circle_f32(canvas, loops); print_row("draw_circle_f32", N, N, N, N, loops, r);   debug_save_image(canvas, make_name("draw_circle_f32", N, N, N, N)); }
        { auto r = bench_fill_rect_f32(canvas, loops);   print_row("fill_rect_f32",   N, N, N, N, loops, r);   debug_save_image(canvas, make_name("fill_rect_f32",   N, N, N, N)); }
    }
    SUCCEED();
}

TEST_CASE("image_benchmark_resize_linear_1p5x_u8", "[bench][image][resize][u8][linear][1p5x]") {
    const int sizes[] = {128, 256, 512, 1024};
    const int channels = 3;
    const int loops = 100;
    g_current_dtype = "u8";

    for (int N : sizes) {
        auto img = make_input_image_u8(N, channels);
        if (img.Empty()) {
            INFO("Skip benchmark: input image not found (set INSPIRECV_IMAGES_DIR or ensure ../images/kun.jpg exists)");
            SUCCEED();
            return;
        }
        // Non-integer resize: N -> 1.5N (round to nearest int)
        int outW = N + N / 2;
        int outH = N + N / 2;
        auto r = bench_resize_u8(img, outW, outH, /*use_linear=*/true, loops);
        print_row("resize_u8_linear_1p5x", N, N, outW, outH, loops, r);
        auto out = img.Resize(outW, outH, /*use_linear=*/true);
        debug_save_image(out, make_name("resize_u8_linear_1p5x", N, N, outW, outH));
    }
    SUCCEED();
}

TEST_CASE("image_benchmark_flip_u8", "[bench][image][flip][u8]") {
    const int sizes[] = {128, 256, 512, 1024};
    const int channels = 3;
    const int loops = 100;
    g_current_dtype = "u8";

    for (int N : sizes) {
        auto img = make_input_image_u8(N, channels);
        if (img.Empty()) {
            INFO("Skip benchmark: input image not found (set INSPIRECV_IMAGES_DIR or ensure ../images/kun.jpg exists)");
            SUCCEED();
            return;
        }
        {
            auto r = bench_flip_h_u8(img, loops);
            print_row("flip_h_u8", N, N, N, N, loops, r);
            auto out = img.FlipHorizontal();
            debug_save_image(out, make_name("flip_h_u8", N, N, N, N));
        }
        {
            auto r = bench_flip_v_u8(img, loops);
            print_row("flip_v_u8", N, N, N, N, loops, r);
            auto out = img.FlipVertical();
            debug_save_image(out, make_name("flip_v_u8", N, N, N, N));
        }
    }
    SUCCEED();
}

TEST_CASE("image_benchmark_absdiff_blend_u8", "[bench][image][absdiff][blend][u8]") {
    const int sizes[] = {128, 256, 512, 1024};
    const int channels = 3;
    const int loops = 100;
    g_current_dtype = "u8";

    for (int N : sizes) {
        auto img = make_input_image_u8(N, channels);
        if (img.Empty()) {
            INFO("Skip benchmark: input image not found (set INSPIRECV_IMAGES_DIR or ensure ../images/kun.jpg exists)");
            SUCCEED();
            return;
        }
        auto other = img.Rotate180();
        {
            auto r = bench_absdiff_u8(img, other, loops);
            print_row("absdiff_u8", N, N, N, N, loops, r);
            auto out = img.AbsDiff(other);
            debug_save_image(out, make_name("absdiff_u8", N, N, N, N));
        }
        {
            auto swapped = img.SwapRB();
            auto mask = make_gradient_mask_u8(N, N);
            auto r = bench_blend_u8(img, swapped, mask, loops);
            print_row("blend_u8_gradmask", N, N, N, N, loops, r);
            auto out = img.Blend(swapped, mask);
            debug_save_image(out, make_name("blend_u8_gradmask", N, N, N, N));
        }
    }
    SUCCEED();
}

TEST_CASE("image_benchmark_warp_affine_more_u8", "[bench][image][warp_affine][u8][more]") {
    const int sizes[] = {128, 256, 512, 1024};
    const int channels = 3;
    const int loops = 100;
    g_current_dtype = "u8";

    auto make_centered_shear = [](float kx, float ky, int W, int H) {
        const float cx = (static_cast<float>(W)  - 1.0f) * 0.5f;
        const float cy = (static_cast<float>(H) - 1.0f) * 0.5f;
        const float a11 = 1.0f, a12 = kx;
        const float a21 = ky,   a22 = 1.0f;
        const float b1 = cx - a11 * cx - a12 * cy;
        const float b2 = cy - a21 * cx - a22 * cy;
        return TransformMatrix::Create(a11, a12, b1, a21, a22, b2);
    };

    for (int N : sizes) {
        auto img = make_input_image_u8(N, channels);
        if (img.Empty()) {
            INFO("Skip benchmark: input image not found (set INSPIRECV_IMAGES_DIR or ensure ../images/kun.jpg exists)");
            SUCCEED();
            return;
        }
        const int outW = N, outH = N;
        // Non-uniform scale (0.75, 1.25)
        {
            auto M = make_centered_affine(/*deg=*/0.0f, /*sx=*/0.75f, /*sy=*/1.25f, 0.0f, 0.0f, outW, outH);
            auto r = bench_warp_affine_u8(img, M, outW, outH, loops);
            print_row("warp_affine_u8_scale_nu", N, N, outW, outH, loops, r);
            auto out = img.WarpAffine(M, outW, outH);
            debug_save_image(out, make_name("warp_affine_u8_scale_nu", N, N, outW, outH));
        }
        // Shear kx=0.2
        {
            auto M = make_centered_shear(/*kx=*/0.2f, /*ky=*/0.0f, outW, outH);
            auto r = bench_warp_affine_u8(img, M, outW, outH, loops);
            print_row("warp_affine_u8_shear_kx02", N, N, outW, outH, loops, r);
            auto out = img.WarpAffine(M, outW, outH);
            debug_save_image(out, make_name("warp_affine_u8_shear_kx02", N, N, outW, outH));
        }
        // Shear ky=0.2
        {
            auto M = make_centered_shear(/*kx=*/0.0f, /*ky=*/0.2f, outW, outH);
            auto r = bench_warp_affine_u8(img, M, outW, outH, loops);
            print_row("warp_affine_u8_shear_ky02", N, N, outW, outH, loops, r);
            auto out = img.WarpAffine(M, outW, outH);
            debug_save_image(out, make_name("warp_affine_u8_shear_ky02", N, N, outW, outH));
        }
    }
    SUCCEED();
}

TEST_CASE("image_benchmark_pad_variants_u8", "[bench][image][pad][u8][variants]") {
    const int sizes[] = {128, 256, 512, 1024};
    const int channels = 3;
    g_current_dtype = "u8";

    const std::vector<double> white = {255.0, 255.0, 255.0};
    const std::vector<double> red   = {0.0, 0.0, 255.0};

    for (int N : sizes) {
        // Keep every timed sample long enough for a strict low-single-digit
        // regression gate. A fixed 100 iterations only measures ~0.2 ms at
        // 128x128 on modern ARM64 hardware and is dominated by timer/scheduler
        // noise.
        const int output_extent = N + 20;
        const int loops = std::max(
          100, 500000000 / (output_extent * output_extent));
        auto img = make_input_image_u8(N, channels);
        if (img.Empty()) {
            INFO("Skip benchmark: input image not found (set INSPIRECV_IMAGES_DIR or ensure ../images/kun.jpg exists)");
            SUCCEED();
            return;
        }
        {
            int top = 10, bottom = 10, left = 10, right = 10;
            // white
            for (int i = 0; i < 5; ++i) { auto tmp = img.Pad(top, bottom, left, right, white); (void)tmp; }
            TimeSpend t("pad_u8_white"); t.Start();
            for (int i = 0; i < loops; ++i) { auto out = img.Pad(top, bottom, left, right, white); (void)out; }
            t.Stop();
            const double total_us = static_cast<double>(t.Total());
            BenchResult r{ total_us/1000.0, (total_us/1000.0)/std::max(1,loops),
                           (static_cast<double>(N+left+right)*(N+top+bottom)*loops)/total_us };
            print_row("pad_u8_white_10_all", N, N, N + left + right, N + top + bottom, loops, r);
            auto out = img.Pad(top, bottom, left, right, white);
            debug_save_image(out, make_name("pad_u8_white_10_all", N, N, out.Width(), out.Height()));
        }
        {
            int top = 10, bottom = 10, left = 10, right = 10;
            // red
            for (int i = 0; i < 5; ++i) { auto tmp = img.Pad(top, bottom, left, right, red); (void)tmp; }
            TimeSpend t("pad_u8_red"); t.Start();
            for (int i = 0; i < loops; ++i) { auto out = img.Pad(top, bottom, left, right, red); (void)out; }
            t.Stop();
            const double total_us = static_cast<double>(t.Total());
            BenchResult r{ total_us/1000.0, (total_us/1000.0)/std::max(1,loops),
                           (static_cast<double>(N+left+right)*(N+top+bottom)*loops)/total_us };
            print_row("pad_u8_red_10_all", N, N, N + left + right, N + top + bottom, loops, r);
            auto out = img.Pad(top, bottom, left, right, red);
            debug_save_image(out, make_name("pad_u8_red_10_all", N, N, out.Width(), out.Height()));
        }
    }
    SUCCEED();
}

TEST_CASE("image_benchmark_threshold_variants_u8", "[bench][image][threshold][u8][variants]") {
    const int sizes[] = {128, 256, 512, 1024};
    const int channels = 3;
    const int loops = 100;
    g_current_dtype = "u8";

    for (int N : sizes) {
        auto img = make_input_image_u8(N, channels);
        if (img.Empty()) {
            INFO("Skip benchmark: input image not found (set INSPIRECV_IMAGES_DIR or ensure ../images/kun.jpg exists)");
            SUCCEED();
            return;
        }
        auto gray = img.ToGray();
        {
            auto r = bench_threshold_u8(gray, 200.0, 255.0, 0, loops);
            print_row("threshold_u8_200_255", N, N, N, N, loops, r);
            auto out = gray.Threshold(200.0, 255.0, 0);
            debug_save_image(out, make_name("threshold_u8_200_255", N, N, N, N));
        }
    }
    SUCCEED();
}

TEST_CASE("image_benchmark_draw_ops_u8", "[bench][image][draw][u8]") {
    const int sizes[] = {128, 256, 512, 1024};
    const int channels = 3;
    const int loops = 100;
    g_current_dtype = "u8";

    for (int N : sizes) {
        auto base = make_input_image_u8(N, channels);
        if (base.Empty()) {
            INFO("Skip benchmark: input image not found (set INSPIRECV_IMAGES_DIR or ensure ../images/kun.jpg exists)");
            SUCCEED();
            return;
        }
        auto canvas = base.Clone();
        {
            auto r = bench_draw_line_u8(canvas, loops);
            print_row("draw_line_u8", N, N, N, N, loops, r);
            debug_save_image(canvas, make_name("draw_line_u8", N, N, N, N));
        }
        {
            auto r = bench_draw_rect_u8(canvas, loops);
            print_row("draw_rect_u8", N, N, N, N, loops, r);
            debug_save_image(canvas, make_name("draw_rect_u8", N, N, N, N));
        }
        {
            auto r = bench_draw_circle_u8(canvas, loops);
            print_row("draw_circle_u8", N, N, N, N, loops, r);
            debug_save_image(canvas, make_name("draw_circle_u8", N, N, N, N));
        }
        {
            auto r = bench_fill_rect_u8(canvas, loops);
            print_row("fill_rect_u8", N, N, N, N, loops, r);
            debug_save_image(canvas, make_name("fill_rect_u8", N, N, N, N));
        }
    }
    SUCCEED();
}

TEST_CASE("image_benchmark_rotate_u8", "[bench][image][rotate][u8]") {
    const int sizes[] = {128, 256, 512, 1024};
    const int channels = 3;
    const int loops = 100;
    g_current_dtype = "u8";

    for (int N : sizes) {
        auto img = make_input_image_u8(N, channels);
        if (img.Empty()) {
            INFO("Skip benchmark: input image not found (set INSPIRECV_IMAGES_DIR or ensure ../images/kun.jpg exists)");
            SUCCEED();
            return;
        }
        {
            auto r = bench_rotate90_u8(img, loops);
            print_row("rotate90_u8", N, N, N, N, loops, r);
            auto out = img.Rotate90();
            debug_save_image(out, make_name("rotate90_u8", N, N, N, N));
        }
        {
            auto r = bench_rotate180_u8(img, loops);
            print_row("rotate180_u8", N, N, N, N, loops, r);
            auto out = img.Rotate180();
            debug_save_image(out, make_name("rotate180_u8", N, N, N, N));
        }
        {
            auto r = bench_rotate270_u8(img, loops);
            print_row("rotate270_u8", N, N, N, N, loops, r);
            auto out = img.Rotate270();
            debug_save_image(out, make_name("rotate270_u8", N, N, N, N));
        }
    }
    SUCCEED();
}

TEST_CASE("image_benchmark_crop_pad_u8", "[bench][image][crop][pad][u8]") {
    const int sizes[] = {128, 256, 512, 1024};
    const int channels = 3;
    const int loops = 100;
    g_current_dtype = "u8";

    for (int N : sizes) {
        auto img = make_input_image_u8(N, channels);
        if (img.Empty()) {
            INFO("Skip benchmark: input image not found (set INSPIRECV_IMAGES_DIR or ensure ../images/kun.jpg exists)");
            SUCCEED();
            return;
        }
        // Crop center to N/2
        {
            int outW = std::max(1, N / 2), outH = std::max(1, N / 2);
            auto r = bench_crop_u8(img, outW, outH, loops);
            print_row("crop_u8_center", N, N, outW, outH, loops, r);
            auto roi = inspirecv::Rect<int>((N - outW) / 2, (N - outH) / 2, outW, outH);
            auto out = img.Crop(roi);
            debug_save_image(out, make_name("crop_u8_center", N, N, outW, outH));
        }
        // Pad 10 pixels each side
        {
            int top = 10, bottom = 10, left = 10, right = 10;
            auto r = bench_pad_u8(img, top, bottom, left, right, loops);
            print_row("pad_u8_10_all", N, N, N + left + right, N + top + bottom, loops, r);
            const std::vector<double> black = {0.0, 0.0, 0.0};
            auto out = img.Pad(top, bottom, left, right, black);
            debug_save_image(out, make_name("pad_u8_10_all", N, N, out.Width(), out.Height()));
        }
    }
    SUCCEED();
}

TEST_CASE("image_benchmark_color_u8", "[bench][image][color][u8]") {
    const int sizes[] = {128, 256, 512, 1024};
    const int channels = 3;
    const int loops = 100;
    g_current_dtype = "u8";

    for (int N : sizes) {
        auto img = make_input_image_u8(N, channels);
        if (img.Empty()) {
            INFO("Skip benchmark: input image not found (set INSPIRECV_IMAGES_DIR or ensure ../images/kun.jpg exists)");
            SUCCEED();
            return;
        }
        {
            auto r = bench_togray_u8(img, loops);
            print_row("to_gray_u8", N, N, N, N, loops, r);
            auto out = img.ToGray();
            debug_save_image(out, make_name("to_gray_u8", N, N, out.Width(), out.Height()));
        }
        {
            auto r = bench_swaprb_u8(img, loops);
            print_row("swap_rb_u8", N, N, N, N, loops, r);
            auto out = img.SwapRB();
            debug_save_image(out, make_name("swap_rb_u8", N, N, N, N));
        }
        {
            auto r = bench_mean_channels_u8(img, loops);
            print_row("mean_channels_u8", N, N, N, N, loops, r);
            auto out = img.MeanChannels();
            debug_save_image(out, make_name("mean_channels_u8", N, N, out.Width(), out.Height()));
        }
    }
    SUCCEED();
}

TEST_CASE("image_benchmark_arith_thresh_u8", "[bench][image][arith][thresh][u8]") {
    const int sizes[] = {128, 256, 512, 1024};
    const int channels = 3;
    const int loops = 100;
    g_current_dtype = "u8";

    for (int N : sizes) {
        auto img = make_input_image_u8(N, channels);
        if (img.Empty()) {
            INFO("Skip benchmark: input image not found (set INSPIRECV_IMAGES_DIR or ensure ../images/kun.jpg exists)");
            SUCCEED();
            return;
        }
        {
            auto r = bench_mul_u8(img, 1.2, loops);
            print_row("mul_u8_1p2", N, N, N, N, loops, r);
            auto out = img.Mul(1.2);
            debug_save_image(out, make_name("mul_u8_1p2", N, N, N, N));
        }
        {
            auto r = bench_add_u8(img, 10.0, loops);
            print_row("add_u8_10", N, N, N, N, loops, r);
            auto out = img.Add(10.0);
            debug_save_image(out, make_name("add_u8_10", N, N, N, N));
        }
        {
            auto gray = img.ToGray();
            auto r = bench_threshold_u8(gray, 100.0, 255.0, 0, loops);
            print_row("threshold_u8_100_255", N, N, N, N, loops, r);
            auto out = gray.Threshold(100.0, 255.0, 0);
            debug_save_image(out, make_name("threshold_u8_100_255", N, N, N, N));
        }
    }
    SUCCEED();
}

TEST_CASE("image_benchmark_filter_morph_u8", "[bench][image][filter][morph][u8]") {
    const int sizes[] = {128, 256, 512, 1024};
    const int channels = 3;
    const int loops = 100;
    g_current_dtype = "u8";

    for (int N : sizes) {
        auto img = make_input_image_u8(N, channels);
        if (img.Empty()) {
            INFO("Skip benchmark: input image not found (set INSPIRECV_IMAGES_DIR or ensure ../images/kun.jpg exists)");
            SUCCEED();
            return;
        }
        {
            auto r = bench_gauss_u8(img, 5, 0.0, loops);
            print_row("gauss5_u8", N, N, N, N, loops, r);
            auto out = img.GaussianBlur(5, 0.0);
            debug_save_image(out, make_name("gauss5_u8", N, N, N, N));
        }
        {
            auto gray = img.ToGray();
            auto r = bench_erode_u8(gray, 3, 1, loops);
            print_row("erode3_u8", N, N, N, N, loops, r);
            auto out = gray.Erode(3, 1);
            debug_save_image(out, make_name("erode3_u8", N, N, N, N));
        }
        {
            auto gray = img.ToGray();
            auto r = bench_dilate_u8(gray, 3, 1, loops);
            print_row("dilate3_u8", N, N, N, N, loops, r);
            auto out = gray.Dilate(3, 1);
            debug_save_image(out, make_name("dilate3_u8", N, N, N, N));
        }
    }
    SUCCEED();
}

TEST_CASE("image_benchmark_resize_linear_u8", "[bench][image][resize][u8][linear]") {
    const int sizes[] = {128, 256, 512, 1024};
    const int channels = 3;
    const int loops = 100;
    g_current_dtype = "u8";

    for (int N : sizes) {
        auto img = make_input_image_u8(N, channels);
        if (img.Empty()) {
            INFO("Skip benchmark: input image not found (set INSPIRECV_IMAGES_DIR or ensure ../images/kun.jpg exists)");
            SUCCEED();
            return;
        }

        // Downsample N -> N/2 (linear)
        {
            int outW = std::max(1, N / 2);
            int outH = std::max(1, N / 2);
            auto r = bench_resize_u8(img, outW, outH, /*use_linear=*/true, loops);
            print_row("resize_u8_linear", N, N, outW, outH, loops, r);
            auto out = img.Resize(outW, outH, /*use_linear=*/true);
            debug_save_image(out, make_name("resize_u8_linear", N, N, outW, outH, "down"));
        }
        // Upsample N -> 2N (linear)
        {
            int outW = N * 2;
            int outH = N * 2;
            auto r = bench_resize_u8(img, outW, outH, /*use_linear=*/true, loops);
            print_row("resize_u8_linear", N, N, outW, outH, loops, r);
            auto out = img.Resize(outW, outH, /*use_linear=*/true);
            debug_save_image(out, make_name("resize_u8_linear", N, N, outW, outH, "up"));
        }
    }
    SUCCEED();
}

TEST_CASE("image_benchmark_warp_affine_u8", "[bench][image][warp_affine][u8]") {
    const int sizes[] = {128, 256, 512, 1024};
    const int channels = 3;
    const int loops = 100;
    g_current_dtype = "u8";

    for (int N : sizes) {
        auto img = make_input_image_u8(N, channels);
        if (img.Empty()) {
            INFO("Skip benchmark: input image not found (set INSPIRECV_IMAGES_DIR or ensure ../images/kun.jpg exists)");
            SUCCEED();
            return;
        }

        const int outW = N, outH = N;

        // Identity
        {
            auto M = TransformMatrix::Identity();
            auto r = bench_warp_affine_u8(img, M, outW, outH, loops);
            print_row("warp_affine_u8_identity", N, N, outW, outH, loops, r);
            auto out = img.WarpAffine(M, outW, outH);
            debug_save_image(out, make_name("warp_affine_u8_identity", N, N, outW, outH));
        }
        // Translate by +/- 0.1N (centered formulation degenerates to plain translate)
        {
            float tx = static_cast<float>(N) * 0.1f;
            float ty = static_cast<float>(N) * -0.1f;
            auto M = make_centered_affine(/*deg=*/0.0f, /*sx=*/1.0f, /*sy=*/1.0f, tx, ty, outW, outH);
            auto r = bench_warp_affine_u8(img, M, outW, outH, loops);
            print_row("warp_affine_u8_translate", N, N, outW, outH, loops, r);
            auto out = img.WarpAffine(M, outW, outH);
            debug_save_image(out, make_name("warp_affine_u8_translate", N, N, outW, outH));
        }
        // Uniform scale 0.5
        {
            auto M = make_centered_affine(/*deg=*/0.0f, /*sx=*/0.5f, /*sy=*/0.5f, /*tx=*/0.0f, /*ty=*/0.0f, outW, outH);
            auto r = bench_warp_affine_u8(img, M, outW, outH, loops);
            print_row("warp_affine_u8_scale05", N, N, outW, outH, loops, r);
            auto out = img.WarpAffine(M, outW, outH);
            debug_save_image(out, make_name("warp_affine_u8_scale05", N, N, outW, outH));
        }
        // Rotation 30 degrees
        {
            auto M = make_centered_affine(/*deg=*/30.0f, /*sx=*/1.0f, /*sy=*/1.0f, /*tx=*/0.0f, /*ty=*/0.0f, outW, outH);
            auto r = bench_warp_affine_u8(img, M, outW, outH, loops);
            print_row("warp_affine_u8_rot30", N, N, outW, outH, loops, r);
            auto out = img.WarpAffine(M, outW, outH);
            debug_save_image(out, make_name("warp_affine_u8_rot30", N, N, outW, outH));
        }
        // Compound: rotate 30° + scale 0.8 + translate 0.05N
        {
            float tx = static_cast<float>(N) * 0.05f;
            float ty = static_cast<float>(N) * 0.05f;
            auto M = make_centered_affine(/*deg=*/30.0f, /*sx=*/0.8f, /*sy=*/0.8f, tx, ty, outW, outH);
            auto r = bench_warp_affine_u8(img, M, outW, outH, loops);
            print_row("warp_affine_u8_combo", N, N, outW, outH, loops, r);
            auto out = img.WarpAffine(M, outW, outH);
            debug_save_image(out, make_name("warp_affine_u8_combo", N, N, outW, outH));
        }
    }
    SUCCEED();
}
