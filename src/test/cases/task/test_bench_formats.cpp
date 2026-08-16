#include "../../common/common.h"
#include <inspirecv/task/task.h>
#include <inspirecv/core/image.h>
#include <inspirecv/core/transform_matrix.h>
#include <cstdio>
#include <vector>
#include <string>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <iomanip>

using inspirecv::task::PixelFormat;

static std::string getImagesDir() {
    const char* env = std::getenv("INSPIRECV_IMAGES_DIR");
    if (env && *env) return std::string(env);
    // Local default: prefer ../images for developer workflow
    return std::string("../images");
}
static std::string getGTDir() {
    const char* env = std::getenv("INSPIRECV_GT_DIR");
    if (env && *env) return std::string(env);
    return getImagesDir() + "/task_gt";
}
static std::string getInputImagePath() { return getImagesDir() + "/kun.jpg"; }
static std::string getBenchOutputPath() {
    const char* env = std::getenv("INSPIRECV_BENCH_OUTPUT");
    if (env && *env) return std::string(env);
    return std::string("benchmark.txt");
}

struct Case {
    const char* name;
    PixelFormat srcFmt;
    const uint8_t* src;
    int iw, ih;
    int stride;
};

struct AlignedBuf {
    uint8_t* ptr{nullptr};
    size_t size{0};
    void alloc(size_t n, size_t align = 64) {
        free();
        size = n;
        void* p = nullptr;
        int rc = posix_memalign(&p, align, n);
        if (rc != 0) {
            p = std::malloc(n);
        }
        ptr = static_cast<uint8_t*>(p);
    }
    void free() { if (ptr) { ::free(ptr); ptr = nullptr; size = 0; } }
    ~AlignedBuf() { free(); }
};

static bool readFile(const std::string& path, std::vector<uint8_t>& buf) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;
    ifs.seekg(0, std::ios::end);
    std::streamsize sz = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    buf.resize(static_cast<size_t>(sz));
    return static_cast<bool>(ifs.read(reinterpret_cast<char*>(buf.data()), sz));
}

static double benchCase(const Case& c, int loops) {
    inspirecv::task::PipelineOptions options;
    options.input_format = c.srcFmt;
    options.output_format = PixelFormat::kBgr;
    options.sampling = inspirecv::task::SamplingMode::kNearest;
    options.border = inspirecv::task::BorderMode::kReplicate;
    inspirecv::task::Pipeline pipeline(options);
    pipeline.SetTransform(inspirecv::TransformMatrix::Identity());
    AlignedBuf dst;
    dst.alloc(static_cast<size_t>(c.iw) * c.ih * 3);
    inspirecv::task::RawImageView source;
    source.data = c.src;
    source.width = c.iw;
    source.height = c.ih;
    source.row_stride_bytes = c.stride;
    inspirecv::task::TensorBuffer destination;
    destination.data = dst.ptr;
    destination.width = c.iw;
    destination.height = c.ih;
    destination.channels = 3;
    destination.element_type = inspirecv::task::ElementType::kUInt8;
    destination.order = inspirecv::task::TensorOrder::kHwc;
    pipeline.Run(source, destination);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < loops; ++i) {
        pipeline.Run(source, destination);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / std::max(1, loops);
}

static inspirecv::TransformMatrix invertAffine2x3(float a, float b, float c, float d, float e, float f) {
    float det = a * e - b * d;
    float ia =  e / det;
    float ib = -b / det;
    float id = -d / det;
    float ie =  a / det;
    float ic = (b * f - c * e) / det;
    float iff= (c * d - a * f) / det;
    return inspirecv::TransformMatrix(ia, ib, ic, id, ie, iff);
}

static double benchCaseAffine(const Case& c, int loops, const inspirecv::TransformMatrix& Minv) {
    inspirecv::task::PipelineOptions options;
    options.input_format = c.srcFmt;
    options.output_format = PixelFormat::kBgr;
    options.sampling = inspirecv::task::SamplingMode::kLinear;
    options.border = inspirecv::task::BorderMode::kReplicate;
    inspirecv::task::Pipeline pipeline(options);
    pipeline.SetTransform(Minv);
    AlignedBuf dst;
    dst.alloc(static_cast<size_t>(c.iw) * c.ih * 3);
    inspirecv::task::RawImageView source;
    source.data = c.src;
    source.width = c.iw;
    source.height = c.ih;
    source.row_stride_bytes = c.stride;
    inspirecv::task::TensorBuffer destination;
    destination.data = dst.ptr;
    destination.width = c.iw;
    destination.height = c.ih;
    destination.channels = 3;
    destination.element_type = inspirecv::task::ElementType::kUInt8;
    destination.order = inspirecv::task::TensorOrder::kHwc;
    pipeline.Run(source, destination);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < loops; ++i) {
        pipeline.Run(source, destination);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / std::max(1, loops);
}

TEST_CASE("bench_formats_print", "[bench][no_check]") {
    std::string path = getInputImagePath();
    // Keep the shortest format case above roughly 50 ms so a scheduler tick
    // cannot masquerade as a low-single-digit kernel regression.
    const int loops = 800;
    std::string outPath = getBenchOutputPath();
    std::ofstream ofs(outPath, std::ios::out | std::ios::trunc);
    REQUIRE(ofs.good());

    auto bgrImg = inspirecv::Image::Create(path, 3);
    REQUIRE_FALSE(bgrImg.Empty());
    int W = bgrImg.Width(), H = bgrImg.Height();
    if ((W & 1) || (H & 1)) {
        // ensure even dims for YUV compatibility
        bgrImg = bgrImg.Crop(inspirecv::Rect<int>(0, 0, W & ~1, H & ~1));
        W = bgrImg.Width(); H = bgrImg.Height();
    }

    // Prepare RGB/BGRA/RGBA/GRAY via InspireCV (no OpenCV dependency)
    auto rgbImg = bgrImg.SwapRB(); // BGR -> RGB
    inspirecv::Image bgraImg;
    {
        bgraImg = inspirecv::Image::Create(W, H, 4);
        const uint8_t* src = bgrImg.Data();
        uint8_t* dst = const_cast<uint8_t*>(bgraImg.Data());
        for (int i = 0; i < W * H; ++i) {
            dst[4 * i + 0] = src[3 * i + 0];
            dst[4 * i + 1] = src[3 * i + 1];
            dst[4 * i + 2] = src[3 * i + 2];
            dst[4 * i + 3] = 255;
        }
    }
    inspirecv::Image rgbaImg;
    {
        rgbaImg = inspirecv::Image::Create(W, H, 4);
        const uint8_t* src = rgbImg.Data();
        uint8_t* dst = const_cast<uint8_t*>(rgbaImg.Data());
        for (int i = 0; i < W * H; ++i) {
            dst[4 * i + 0] = src[3 * i + 0];
            dst[4 * i + 1] = src[3 * i + 1];
            dst[4 * i + 2] = src[3 * i + 2];
            dst[4 * i + 3] = 255;
        }
    }
    auto grayImg = bgrImg.ToGray();

    // Load raw YUV inputs from GT directory
    std::vector<uint8_t> nv21, nv12, i420;
    auto gtDir = getGTDir();
    REQUIRE(readFile(gtDir + "/SRC_nv21_w" + std::to_string(W) + "_h" + std::to_string(H) + ".bin", nv21));
    REQUIRE(readFile(gtDir + "/SRC_nv12_w" + std::to_string(W) + "_h" + std::to_string(H) + ".bin", nv12));
    REQUIRE(readFile(gtDir + "/SRC_i420_w" + std::to_string(W) + "_h" + std::to_string(H) + ".bin", i420));

    std::vector<Case> cases = {
        {"BGR->BGR",   PixelFormat::kBgr, bgrImg.Data(),  W, H, W * 3},
        {"RGB->BGR",   PixelFormat::kRgb, rgbImg.Data(),  W, H, W * 3},
        {"BGRA->BGR",  PixelFormat::kBgra,bgraImg.Data(), W, H, W * 4},
        {"RGBA->BGR",  PixelFormat::kRgba,rgbaImg.Data(), W, H, W * 4},
        {"GRAY->BGR",  PixelFormat::kGray,grayImg.Data(), W, H, W * 1},
        {"NV21->BGR",  PixelFormat::kNv21, nv21.data(), W, H, 0},
        {"NV12->BGR",  PixelFormat::kNv12, nv12.data(), W, H, 0},
        {"I420->BGR",  PixelFormat::kI420, i420.data(), W, H, 0},
    };

    std::printf("loops=%d\n", loops);
    ofs << "loops=" << loops << "\n";
    for (const auto& c : cases) {
        if (!c.src) continue;
        double avg = benchCase(c, loops);
        std::printf("%-10s  %8.3f ms\n", c.name, avg);
        ofs << c.name << "  " << std::fixed << std::setprecision(6) << avg << " ms\n";
    }

    std::printf("\n-- With affine (BILINEAR) --\n");
    ofs << "\n-- With affine (BILINEAR) --\n";
    float cx = 0.5f * W, cy = 0.5f * H;
    auto rotMinv = [&](float deg) {
        float rad = deg * float(M_PI) / 180.0f;
        float cs = std::cos(rad), sn = std::sin(rad);
        float a = cs, b = sn, c = (1 - cs) * cx - sn * cy;
        float d = -sn, e = cs, f = sn * cx + (1 - cs) * cy;
        return invertAffine2x3(a, b, c, d, e, f);
    };
    std::vector<std::pair<const char*, inspirecv::TransformMatrix>> mats = {
        {"identity", invertAffine2x3(1,0,0, 0,1,0)},
        {"translate", invertAffine2x3(1,0,20, 0,1,-15)},
        {"scale0.5", invertAffine2x3(0.5f,0,0, 0,0.5f,0)},
        {"scale2.0", invertAffine2x3(2.0f,0,0, 0,2.0f,0)},
        {"rot30", rotMinv(30.0f)},
        {"rot-45", rotMinv(-45.0f)},
    };

    for (const auto& kv : mats) {
        const char* mname = kv.first;
        const auto& Minv = kv.second;
        for (const auto& c : cases) {
            if (!c.src) continue;
            double avg = benchCaseAffine(c, loops, Minv);
            std::printf("%-6s + %-10s  %8.3f ms\n", mname, c.name, avg);
            ofs << mname << " + " << c.name << "  " << std::fixed
                << std::setprecision(6) << avg << " ms\n";
        }
    }
    ofs.flush();
}
