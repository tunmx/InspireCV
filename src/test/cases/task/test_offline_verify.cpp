#include "../../common/common.h"
#include "../image/image_test_utils.h"
#include <inspirecv/task/task.h>
#include <inspirecv/core/transform_matrix.h>
#include <inspirecv/core/image.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>

using inspirecv::task::PixelFormat;

// Resolve paths via env; fallback to relative when not provided
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
static std::string getInputImagePath() {
    return getImagesDir() + "/kun.jpg";
}

static int computeMaxDiff(const uint8_t* a, const uint8_t* b, size_t n) {
    int md = 0;
    for (size_t i = 0; i < n; ++i) {
        int d = std::abs(int(a[i]) - int(b[i]));
        if (d > md) md = d;
    }
    return md;
}

static uint64_t hashBytes(const uint8_t* data, size_t size) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t index = 0; index < size; ++index) {
        hash ^= data[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static double computePSNR(const uint8_t* a, const uint8_t* b, size_t n, int channels) {
    if (n == 0) return 100.0;
    double sse = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = double(a[i]) - double(b[i]);
        sse += d * d;
    }
    if (sse <= 1e-12) return 100.0;
    double mse = sse / double(n);
    return 10.0 * std::log10((255.0 * 255.0) / mse);
}

static bool readFile(const std::string& path, std::vector<uint8_t>& buf) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;
    ifs.seekg(0, std::ios::end);
    std::streamsize sz = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    buf.resize(static_cast<size_t>(sz));
    if (!ifs.read(reinterpret_cast<char*>(buf.data()), sz)) return false;
    return true;
}

static bool parseWHFromName(const std::string& filename, int& w, int& h) {
    // expect ..._w{W}_h{H}.bin
    auto posW = filename.rfind("_w");
    auto posH = filename.rfind("_h");
    auto posDot = filename.rfind('.');
    if (posW == std::string::npos || posH == std::string::npos || posDot == std::string::npos) return false;
    std::string ws = filename.substr(posW + 2, posH - (posW + 2));
    std::string hs = filename.substr(posH + 2, posDot - (posH + 2));
    w = std::stoi(ws);
    h = std::stoi(hs);
    return true;
}

static inspirecv::TransformMatrix invertAffine2x3(float a, float b, float c, float d, float e, float f) {
    // M = [a b c; d e f]
    float det = a * e - b * d;
    float ia =  e / det;
    float ib = -b / det;
    float id = -d / det;
    float ie =  a / det;
    float ic = (b * f - c * e) / det;
    float iff= (c * d - a * f) / det;
    return inspirecv::TransformMatrix(ia, ib, ic, id, ie, iff);
}

TEST_CASE("offline_verify_geometry", "[offline]") {
    DRAW_SPLIT_LINE
    std::string gt = getGTDir();
    auto geoBase = gt + std::string("/GT_geo_");
    // Use provided input image directly (must match GT generation source)
    auto bgr = inspirecv::Image::Create(getInputImagePath(), 3);
    REQUIRE_FALSE(bgr.Empty());
    const int W = bgr.Width(), H = bgr.Height();

    struct Case { const char* name; inspirecv::TransformMatrix Minv; };
    float cx = 0.5f * W, cy = 0.5f * H;
    auto rotMinv = [&](float deg) {
        float rad = deg * float(M_PI) / 180.0f;
        float cs = std::cos(rad), sn = std::sin(rad);
        float a = cs, b = sn, c = (1 - cs) * cx - sn * cy;
        float d = -sn, e = cs, f = sn * cx + (1 - cs) * cy;
        return invertAffine2x3(a, b, c, d, e, f);
    };

    // Match GT filenames:
    // GT_geo_affine_identity.png
    // GT_geo_affine_translate.png
    // GT_geo_affine_scale0_5.png
    // GT_geo_affine_scale2_0.png
    // GT_geo_affine_rot30.png
    // GT_geo_affine_rot_45.png  (note: generated from -45 deg)
    std::vector<Case> cases = {
        {"affine_identity", invertAffine2x3(1,0,0, 0,1,0)},
        {"affine_translate", invertAffine2x3(1,0,20, 0,1,-15)},
        {"affine_scale0_5", invertAffine2x3(0.5f,0,0, 0,0.5f,0)},
        {"affine_scale2_0", invertAffine2x3(2,0,0, 0,2,0)},
        {"affine_rot30", rotMinv(30.0f)},
        {"affine_rot_45", rotMinv(-45.0f)},
    };

    for (auto& cs : cases) {
        auto gtPath = geoBase + std::string(cs.name) + ".png";
        auto gtimg = inspirecv::Image::Create(gtPath, 3);
        REQUIRE_FALSE(gtimg.Empty());
        REQUIRE(gtimg.Width() == W);
        REQUIRE(gtimg.Height() == H);

        inspirecv::task::PipelineOptions options;
        options.sampling = inspirecv::task::SamplingMode::kLinear;
        options.input_format = PixelFormat::kBgr;
        options.output_format = PixelFormat::kBgr;
        options.border = inspirecv::task::BorderMode::kReplicate;
        inspirecv::task::Pipeline pipeline(options);
        pipeline.SetTransform(cs.Minv);
        inspirecv::Image actual;
        REQUIRE(pipeline.Run(bgr, W, H, &actual) ==
                inspirecv::task::Status::kOk);
        inspirecv_test_write_comparison(actual, gtimg, std::string("task_") + cs.name);

        const size_t output_bytes = static_cast<size_t>(W) * H * 3;
        int maxDiff = computeMaxDiff(gtimg.Data(), actual.Data(), output_bytes);
        REQUIRE(maxDiff <= 3);
        double psnr = computePSNR(gtimg.Data(), actual.Data(), output_bytes, 3);
        REQUIRE(psnr > 50.0);
        if (std::string(cs.name) == "affine_rot30") {
            // Freeze each AppleClang P0 execution route's complete BGR output,
            // including endpoint/step rounding used by bilinear sampling.
#if defined(__clang__)
#if defined(INSPIRECV_TASK_USE_NEON)
            constexpr uint64_t kExpectedHash = UINT64_C(0xacb0fab76e327630);
#elif defined(__x86_64__) || defined(_M_X64)
            constexpr uint64_t kExpectedHash = UINT64_C(0xdf0d3381fcaf8ce1);
#else
            constexpr uint64_t kExpectedHash = UINT64_C(0x7366af06bb199483);
#endif
            REQUIRE(hashBytes(actual.Data(), output_bytes) == kExpectedHash);
#endif
        }
    }
}

TEST_CASE("offline_verify_formats", "[offline]") {
    DRAW_SPLIT_LINE
    std::string gt = getGTDir();
    // BGR->RGB
    {
        auto srcBgr = inspirecv::Image::Create(getInputImagePath(), 3);
        auto gtImg = inspirecv::Image::Create(gt + "/GT_bgr2rgb.png", 3);
        REQUIRE_FALSE(srcBgr.Empty());
        REQUIRE_FALSE(gtImg.Empty());
        inspirecv::task::PipelineOptions options;
        options.input_format = PixelFormat::kBgr;
        options.output_format = PixelFormat::kRgb;
        options.sampling = inspirecv::task::SamplingMode::kNearest;
        inspirecv::task::Pipeline pipeline(options);
        pipeline.SetTransform(inspirecv::TransformMatrix::Identity());
        inspirecv::Image actual;
        REQUIRE(pipeline.Run(srcBgr, srcBgr.Width(), srcBgr.Height(),
                             &actual) == inspirecv::task::Status::kOk);
        inspirecv_test_write_comparison(actual, gtImg, "task_bgr_to_rgb");
        REQUIRE(computeMaxDiff(gtImg.Data(), actual.Data(),
                               static_cast<size_t>(actual.Width()) *
                                 actual.Height() * actual.Channels()) == 0);
    }
    // NV21/NV12/I420 -> BGR
    struct YCase { const char* raw; const char* gt; PixelFormat fmt; };
    std::vector<YCase> ycases = {
        {"SRC_nv21", "GT_nv21_to_bgr.png", PixelFormat::kNv21},
        {"SRC_nv12", "GT_nv12_to_bgr.png", PixelFormat::kNv12},
        {"SRC_i420", "GT_i420_to_bgr.png", PixelFormat::kI420},
    };
    for (auto& yc : ycases) {
        // find a bin file (any with w/h)
        std::string patternBase = gt + "/" + yc.raw;
        // We assume a single resolution dump exists; construct name using SRC_BGR.png dims
        auto srcBgr = inspirecv::Image::Create(getInputImagePath(), 3);
        int W = srcBgr.Width(), H = srcBgr.Height();
        std::string rawPath = patternBase + "_w" + std::to_string(W) + "_h" + std::to_string(H) + ".bin";
        std::vector<uint8_t> raw;
        REQUIRE(readFile(rawPath, raw));
        auto gtImg = inspirecv::Image::Create(gt + "/" + yc.gt, 3);
        REQUIRE_FALSE(gtImg.Empty());
        inspirecv::task::PipelineOptions options;
        options.input_format = yc.fmt;
        options.output_format = PixelFormat::kBgr;
        options.sampling = inspirecv::task::SamplingMode::kNearest;
        inspirecv::task::Pipeline pipeline(options);
        pipeline.SetTransform(inspirecv::TransformMatrix::Identity());
        std::vector<uint8_t> out((size_t)W*H*3);
        inspirecv::task::RawImageView source;
        source.data = raw.data();
        source.width = W;
        source.height = H;
        inspirecv::task::TensorBuffer destination;
        destination.data = out.data();
        destination.width = W;
        destination.height = H;
        destination.channels = 3;
        destination.element_type = inspirecv::task::ElementType::kUInt8;
        destination.order = inspirecv::task::TensorOrder::kHwc;
        REQUIRE(pipeline.Run(source, destination) ==
                inspirecv::task::Status::kOk);
        auto actual = inspirecv::Image::Create(W, H, 3, out.data());
        inspirecv_test_write_comparison(actual, gtImg, std::string("task_") + yc.raw);
        int maxDiff = computeMaxDiff(gtImg.Data(), out.data(), out.size());
        double psnr = computePSNR(gtImg.Data(), out.data(), out.size(), 3);
        INFO("YUV conversion " << yc.raw << ": maxDiff=" << maxDiff << ", PSNR=" << psnr << " dB");
        REQUIRE(maxDiff <= 33);
        REQUIRE(psnr > 25.0);
    }
}
