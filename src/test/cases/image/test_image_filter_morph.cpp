#include "../../common/common.h"
#include "image_test_utils.h"
#include <inspirecv/inspirecv.h>
#include <string>
#include <cstdlib>
#include <utility>
#include <iostream>
#include <iomanip>
#include <cmath>

using inspirecv::Image;

static std::string gt_dir() {
    return inspirecv_test_images_dir() + "/image_gt";
}
static std::string gt_path(const std::string& name) {
    return gt_dir() + "/" + name;
}
static std::string compare_dir() {
    const char* env = std::getenv("INSPIRECV_COMPARE_DIR");
    if (env && *env) return std::string(env);
    return std::string("");
}
static Image try_load_from_dir(const std::string& dir, const std::string& fname, int channels) {
    if (dir.empty()) return Image::Create();
    return Image::Create(dir + "/" + fname, channels);
}
static void print_diff_stats(const uint8_t* a, const uint8_t* b, size_t n) {
    size_t cnt = 0;
    double sum = 0.0, maxd = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = std::abs(int(a[i]) - int(b[i]));
        if (d != 0.0) ++cnt;
        sum += d;
        if (d > maxd) maxd = d;
    }
    double mae = sum / double(n);
    INFO("diff stats - count!=" << 0 << ": " << cnt << ", MAE: " << mae << ", max: " << maxd);
}
static void print_full_diff(const Image& A, const Image& B, const char* tag) {
    const uint8_t* a = A.Data();
    const uint8_t* b = B.Data();
    const size_t n = static_cast<size_t>(A.Width()) * A.Height() * A.Channels();
    double sum = 0.0, maxd = 0.0, mse = 0.0;
    size_t c1 = 0, c2 = 0, c3 = 0, c5 = 0;
    for (size_t i = 0; i < n; ++i) {
        double d = std::abs(int(a[i]) - int(b[i]));
        sum += d;
        mse += d * d;
        if (d > maxd) maxd = d;
        if (d >= 1.0) ++c1;
        if (d >= 2.0) ++c2;
        if (d >= 3.0) ++c3;
        if (d >= 5.0) ++c5;
    }
    double mae = sum / double(n);
    double rmse = std::sqrt(mse / double(n));
    double psnr = rmse > 0.0 ? 20.0 * std::log10(255.0 / rmse) : INFINITY;
    std::cout << "[diff:" << tag << "] "
              << "N=" << n
              << " MAE=" << std::fixed << std::setprecision(4) << mae
              << " RMSE=" << rmse
              << " PSNR(dB)=" << psnr
              << " max=" << maxd
              << " |>=1:" << c1 << " >=2:" << c2 << " >=3:" << c3 << " >=5:" << c5
              << std::endl;
}
static void save_concat(const Image& left, const Image& right, const std::string& out_name) {
    int h = std::min(left.Height(), right.Height());
    int wl = left.Width();
    int wr = right.Width();
    int c = left.Channels();
    if (right.Channels() != c || wl <= 0 || wr <= 0 || h <= 0) return;
    int w = wl + wr;
    std::vector<uint8_t> buf(static_cast<size_t>(w) * h * c);
    const uint8_t* L = left.Data();
    const uint8_t* R = right.Data();
    size_t ls = static_cast<size_t>(wl) * c;
    size_t rs = static_cast<size_t>(wr) * c;
    size_t os = static_cast<size_t>(w) * c;
    for (int y = 0; y < h; ++y) {
        uint8_t* row = buf.data() + static_cast<size_t>(y) * os;
        const uint8_t* lrow = L + static_cast<size_t>(y) * ls;
        const uint8_t* rrow = R + static_cast<size_t>(y) * rs;
        std::memcpy(row, lrow, ls);
        std::memcpy(row + ls, rrow, rs);
    }
    auto cat = Image::Create(w, h, c, buf.data());
    inspirecv_test_write_image(cat, out_name);
}

static bool check_gaussian_with_border_policy(const Image& g, const Image& gt,
                                              int eps_interior, int eps_border,
                                              int radius,
                                              long max_interior_outliers,
                                              long max_border_outliers,
                                              const char* tag) {
    const int W = g.Width(), H = g.Height(), C = g.Channels();
    const uint8_t* a = g.Data();
    const uint8_t* b = gt.Data();
    size_t stride = static_cast<size_t>(W) * C;
    long fail_in = 0, fail_bd = 0;
    for (int y = 0; y < H; ++y) {
        bool border_y = (y < radius) || (y >= H - radius);
        for (int x = 0; x < W; ++x) {
            bool border = border_y || (x < radius) || (x >= W - radius);
            size_t base = static_cast<size_t>(y) * stride + static_cast<size_t>(x) * C;
            for (int c = 0; c < C; ++c) {
                int d = std::abs(int(a[base + c]) - int(b[base + c]));
                if (!border) {
                    if (d > eps_interior) { ++fail_in; break; }
                } else {
                    if (d > eps_border) { ++fail_bd; break; }
                }
            }
        }
    }
    if (fail_in > max_interior_outliers || fail_bd > max_border_outliers) {
        std::cout << "[gauss-check:" << tag << "] interior_eps=" << eps_interior
                  << " border_eps=" << eps_border
                  << " radius=" << radius
                  << " interior_fail=" << fail_in
                  << " border_fail=" << fail_bd << std::endl;
        auto diff = g.AbsDiff(gt);
        inspirecv_test_write_image(diff, std::string("dbg_diff_") + tag + ".png");
        inspirecv_test_write_image(diff.Mul(8.0), std::string("dbg_diffx8_") + tag + ".png");
        save_concat(g, gt, std::string("dbg_concat_") + tag + ".png");
        print_full_diff(g, gt, tag);
        return false;
    }
    return true;
}

TEST_CASE("image_filter_and_morphology_u8", "[image][filter][morph]") {
    const int sizes[] = {128, 256, 512, 1024};
    for (int N : sizes) {
        SECTION(std::string("size_") + std::to_string(N)) {
            // Base color and gray from GT
            auto base = Image::Create(gt_path("resize_linear_" + std::to_string(N) + ".png"), 3);
            auto gray = Image::Create(gt_path("gray_" + std::to_string(N) + ".png"), 1);
            if (base.Empty() || gray.Empty()) {
                INFO("Skip: missing GT base/gray for size " + std::to_string(N));
                SUCCEED();
                continue;
            }
            // Dump inputs for debugging
            inspirecv_test_write_image(base, "filter_base_bgr_" + std::to_string(N) + ".png");
            inspirecv_test_write_image(gray, "filter_base_gray_" + std::to_string(N) + ".png");

            // GaussianBlur 5x5 (sigma auto) on BGR
            SECTION("gauss5_bgr") {
                auto g = base.GaussianBlur(5, 0.0);
                inspirecv_test_write_image(g, "filter_gauss5_bgr_" + std::to_string(N) + ".png");
                auto gt = Image::Create(gt_path("gauss5_bgr_" + std::to_string(N) + ".png"), 3);
                REQUIRE_FALSE(gt.Empty());
                inspirecv_test_write_comparison(g, gt, "gauss5_bgr_" + std::to_string(N));
                // Optional: override result by reading from compare dir if provided
                auto loaded = try_load_from_dir(compare_dir(), "filter_gauss5_bgr_" + std::to_string(N) + ".png", 3);
                if (!loaded.Empty()) g = std::move(loaded);
                // Allow a tiny number of interior outliers due to rounding/FMA differences
                bool ok = check_gaussian_with_border_policy(g, gt, /*eps_interior=*/2, /*eps_border=*/3, /*radius=*/2,
                                                            /*max_interior_outliers=*/10, /*max_border_outliers=*/0,
                                                            "gauss5_bgr");
                REQUIRE(ok);
            }

            // GaussianBlur 5x5 on gray
            SECTION("gauss5_gray") {
                auto g = gray.GaussianBlur(5, 0.0);
                inspirecv_test_write_image(g, "filter_gauss5_gray_" + std::to_string(N) + ".png");
                auto gt = Image::Create(gt_path("gauss5_gray_" + std::to_string(N) + ".png"), 1);
                REQUIRE_FALSE(gt.Empty());
                inspirecv_test_write_comparison(g, gt, "gauss5_gray_" + std::to_string(N));
                auto loaded = try_load_from_dir(compare_dir(), "filter_gauss5_gray_" + std::to_string(N) + ".png", 1);
                if (!loaded.Empty()) g = std::move(loaded);
                bool ok = check_gaussian_with_border_policy(g, gt, /*eps_interior=*/2, /*eps_border=*/3, /*radius=*/2,
                                                            /*max_interior_outliers=*/10, /*max_border_outliers=*/0,
                                                            "gauss5_gray");
                REQUIRE(ok);
            }

            // Erode 3x3 single-iteration
            SECTION("erode3_gray") {
                auto e = gray.Erode(3, 1);
                inspirecv_test_write_image(e, "morph_erode3_gray_" + std::to_string(N) + ".png");
                auto gt = Image::Create(gt_path("erode3_gray_" + std::to_string(N) + ".png"), 1);
                if (!gt.Empty()) inspirecv_test_write_image(gt, "gt_morph_erode3_gray_" + std::to_string(N) + ".png");
                {
                    std::string msg;
                    bool ok = inspirecv::CheckArrayEqual(e.Data(), gt.Data(), static_cast<size_t>(N) * N, msg);
                    if (!ok) {
                        auto diff = e.AbsDiff(gt);
                        inspirecv_test_write_image(diff, "dbg_diff_erode3_gray_" + std::to_string(N) + ".png");
                        inspirecv_test_write_image(diff.Mul(8.0), "dbg_diffx8_erode3_gray_" + std::to_string(N) + ".png");
                        if (!msg.empty()) INFO(msg);
                    }
                    REQUIRE(ok);
                }
            }

            // Dilate 3x3 single-iteration
            SECTION("dilate3_gray") {
                auto d = gray.Dilate(3, 1);
                inspirecv_test_write_image(d, "morph_dilate3_gray_" + std::to_string(N) + ".png");
                auto gt = Image::Create(gt_path("dilate3_gray_" + std::to_string(N) + ".png"), 1);
                if (!gt.Empty()) inspirecv_test_write_image(gt, "gt_morph_dilate3_gray_" + std::to_string(N) + ".png");
                {
                    std::string msg;
                    bool ok = inspirecv::CheckArrayEqual(d.Data(), gt.Data(), static_cast<size_t>(N) * N, msg);
                    if (!ok) {
                        auto diff = d.AbsDiff(gt);
                        inspirecv_test_write_image(diff, "dbg_diff_dilate3_gray_" + std::to_string(N) + ".png");
                        inspirecv_test_write_image(diff.Mul(8.0), "dbg_diffx8_dilate3_gray_" + std::to_string(N) + ".png");
                        if (!msg.empty()) INFO(msg);
                    }
                    REQUIRE(ok);
                }
            }
        }
    }
}

