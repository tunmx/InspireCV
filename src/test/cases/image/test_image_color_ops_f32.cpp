#include "../../common/common.h"
#include "image_test_utils.h"
#include <inspirecv/inspirecv.h>
#include <string>
#include <vector>
#include <cstring>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>

using inspirecv::ImageT;

static std::string gt_dir() {
    return inspirecv_test_images_dir() + "/image_gt";
}
static std::string gt_path(const std::string& name) {
    return gt_dir() + "/" + name;
}

static void save_concat_f32(const ImageT<float>& left, const ImageT<float>& right,
                            const std::string& out_name) {
    int h = std::min(left.Height(), right.Height());
    int wl = left.Width();
    int wr = right.Width();
    int c = left.Channels();
    if (right.Channels() != c || wl <= 0 || wr <= 0 || h <= 0) return;
    int w = wl + wr;
    std::vector<float> buf(static_cast<size_t>(w) * h * c);
    const float* L = left.Data();
    const float* R = right.Data();
    size_t ls = static_cast<size_t>(wl) * c;
    size_t rs = static_cast<size_t>(wr) * c;
    size_t os = static_cast<size_t>(w) * c;
    for (int y = 0; y < h; ++y) {
        float* row = buf.data() + static_cast<size_t>(y) * os;
        const float* lrow = L + static_cast<size_t>(y) * ls;
        const float* rrow = R + static_cast<size_t>(y) * rs;
        std::memcpy(row, lrow, ls * sizeof(float));
        std::memcpy(row + ls, rrow, rs * sizeof(float));
    }
    auto cat = ImageT<float>::Create(w, h, c, buf.data());
    inspirecv_test_write_image(cat, out_name);
}

static void show_concat_if_gui(const ImageT<float>& left, const ImageT<float>& right,
                               const std::string& win_name) {
#if TEST_SHOW && defined(INSPIRECV_BACKEND_OKCV_USE_OPENCV_GUI)
    int h = std::min(left.Height(), right.Height());
    int wl = left.Width();
    int wr = right.Width();
    int c = left.Channels();
    if (right.Channels() != c || wl <= 0 || wr <= 0 || h <= 0) return;
    int w = wl + wr;
    std::vector<float> buf(static_cast<size_t>(w) * h * c);
    const float* L = left.Data();
    const float* R = right.Data();
    size_t ls = static_cast<size_t>(wl) * c;
    size_t rs = static_cast<size_t>(wr) * c;
    size_t os = static_cast<size_t>(w) * c;
    for (int y = 0; y < h; ++y) {
        float* row = buf.data() + static_cast<size_t>(y) * os;
        const float* lrow = L + static_cast<size_t>(y) * ls;
        const float* rrow = R + static_cast<size_t>(y) * rs;
        std::memcpy(row, lrow, ls * sizeof(float));
        std::memcpy(row + ls, rrow, rs * sizeof(float));
    }
    auto cat = ImageT<float>::Create(w, h, c, buf.data());
    cat.Show(win_name, 0);
#else
    (void)left; (void)right; (void)win_name;
#endif
}

static void print_full_diff_f32(const float* a, const float* b, size_t n, const char* tag) {
    double sum = 0.0, mse = 0.0, maxd = 0.0;
    size_t c1 = 0, c2 = 0, c3 = 0, c5 = 0;
    for (size_t i = 0; i < n; ++i) {
        double d = std::fabs(a[i] - b[i]);
        sum += d; mse += d * d;
        if (d > maxd) maxd = d;
        if (d >= 1.0) ++c1;
        if (d >= 2.0) ++c2;
        if (d >= 3.0) ++c3;
        if (d >= 5.0) ++c5;
    }
    double mae = sum / double(n);
    double rmse = std::sqrt(mse / double(n));
    double psnr = rmse > 0.0 ? 20.0 * std::log10(255.0 / rmse) : INFINITY;
    std::cout << "[f32-diff:" << tag << "] N=" << n
              << " MAE=" << std::fixed << std::setprecision(6) << mae
              << " RMSE=" << rmse
              << " PSNR(dB)=" << psnr
              << " max=" << maxd
              << " |>=1:" << c1 << " >=2:" << c2 << " >=3:" << c3 << " >=5:" << c5
              << std::endl;
}

static bool check_with_outliers_f32(const float* a, const float* b, size_t n,
                                    float eps, size_t max_outliers, const char* tag) {
    size_t outliers = 0;
    double maxd = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = std::fabs(a[i] - b[i]);
        if (d > eps) ++outliers;
        if (d > maxd) maxd = d;
    }
    if (outliers > max_outliers) {
        std::cout << "[f32-check:" << tag << "] eps=" << eps
                  << " outliers=" << outliers << " (limit " << max_outliers << ")"
                  << " max=" << maxd << std::endl;
        return false;
    }
    return true;
}

TEST_CASE("image_color_channel_pixel_ops_f32", "[image][color][float]") {
    const int sizes[] = {128, 256, 512, 1024};
    for (int N : sizes) {
        SECTION(std::string("size_") + std::to_string(N)) {
            auto base = ImageT<float>::Create(gt_path("resize_linear_" + std::to_string(N) + ".png"), 3);
            REQUIRE_FALSE(base.Empty());

            SECTION("to_gray_f32") {
                auto y = base.ToGray();
                inspirecv_test_write_image(y, "f32_color_gray_" + std::to_string(N) + ".png");
                auto gt = ImageT<float>::Create(gt_path("gray_" + std::to_string(N) + ".png"), 1);
                inspirecv_test_write_image(gt, "f32_gt_color_gray_" + std::to_string(N) + ".png");
                // Compare after rounding our float result to nearest uint8 to align with GT from PNG
                std::vector<float> yr(static_cast<size_t>(N) * N);
                const float* yd = y.Data();
                for (size_t i = 0; i < yr.size(); ++i) {
                    float v = yd[i];
                    if (v < 0.f) v = 0.f; else if (v > 255.f) v = 255.f;
                    yr[i] = std::floor(v + 0.5f);
                }
                // Save diff/concat for inspection
                auto y_rounded_img = ImageT<float>::Create(N, N, 1, yr.data());
                auto diff = y_rounded_img.AbsDiff(gt);
                inspirecv_test_write_image(diff, "f32_dbg_diff_to_gray_" + std::to_string(N) + ".png");
                inspirecv_test_write_image(diff.Mul(8.0), "f32_dbg_diffx8_to_gray_" + std::to_string(N) + ".png");
                save_concat_f32(y_rounded_img, gt, "f32_dbg_concat_to_gray_" + std::to_string(N) + ".png");
                show_concat_if_gui(y_rounded_img, gt, "to_gray_f32_concat_" + std::to_string(N));
                {
                    size_t max_outliers = std::max<size_t>(32, static_cast<size_t>(N) / 4096);
                    // If GT is in [0,1], scale to [0,255] before check
                    float gt_max = 0.f;
                    const float* gtd = gt.Data();
                    for (size_t i = 0; i < static_cast<size_t>(N) * N; ++i) if (gtd[i] > gt_max) gt_max = gtd[i];
                    std::vector<float> gt_scaled(static_cast<size_t>(N) * N);
                    if (gt_max <= 1.5f) {
                        for (size_t i = 0; i < gt_scaled.size(); ++i) gt_scaled[i] = gtd[i] * 255.0f;
                    } else {
                        std::memcpy(gt_scaled.data(), gtd, gt_scaled.size() * sizeof(float));
                    }
                    bool ok2 = check_with_outliers_f32(yr.data(), gt_scaled.data(), static_cast<size_t>(N) * N,
                                                       /*eps=*/1.01f, max_outliers, "to_gray_f32");
                    if (!ok2) {
                        print_full_diff_f32(yr.data(), gt_scaled.data(), static_cast<size_t>(N) * N, "to_gray_f32");
                    }
                    REQUIRE(ok2);
                }
            }

            SECTION("swap_rb_f32") {
                auto rgb = base.SwapRB();
                inspirecv_test_write_image(rgb, "f32_color_swaprb_" + std::to_string(N) + ".png");
                auto gt = ImageT<float>::Create(gt_path("swaprb_" + std::to_string(N) + ".png"), 3);
                inspirecv_test_write_image(gt, "f32_gt_color_swaprb_" + std::to_string(N) + ".png");
                REQUIRE_NEAR_C_ARRAY(rgb.Data(), gt.Data(), static_cast<size_t>(N) * N * 3, 1e-3);
            }

            SECTION("mean_channels_f32") {
                auto mean = base.MeanChannels();
                inspirecv_test_write_image(mean, "f32_color_mean_" + std::to_string(N) + ".png");
                // Reference: true float average (not integer /3)
                std::vector<float> ref(static_cast<size_t>(N) * N);
                const float* p = base.Data();
                for (int y = 0; y < N; ++y) {
                    for (int x = 0; x < N; ++x) {
                        size_t idx = static_cast<size_t>(y) * N * 3 + static_cast<size_t>(x) * 3;
                        ref[static_cast<size_t>(y) * N + x] =
                          (p[idx + 0] + p[idx + 1] + p[idx + 2]) * (1.0f / 3.0f);
                    }
                }
                REQUIRE_NEAR_C_ARRAY(mean.Data(), ref.data(), static_cast<size_t>(N) * N, 1e-4);
            }

            SECTION("absdiff_flip_h_f32") {
                auto other = ImageT<float>::Create(gt_path("flip_h_" + std::to_string(N) + ".png"), 3);
                auto diff = base.AbsDiff(other);
                inspirecv_test_write_image(diff, "f32_color_absdiff_flip_h_" + std::to_string(N) + ".png");
                auto gt = ImageT<float>::Create(gt_path("absdiff_flip_h_" + std::to_string(N) + ".png"), 3);
                inspirecv_test_write_image(gt, "f32_gt_color_absdiff_flip_h_" + std::to_string(N) + ".png");
                REQUIRE_NEAR_C_ARRAY(diff.Data(), gt.Data(), static_cast<size_t>(N) * N * 3, 1e-3);
            }

            SECTION("blend_maskgrad_green_f32") {
                using inspirecv::Rect;
                ImageT<float> other = ImageT<float>::Create(N, N, 3);
                other.Fill(Rect<int>(0, 0, N, N), {0.f, 255.f, 0.f});
                auto mask = inspirecv::Image::Create(gt_path("mask_grad_" + std::to_string(N) + ".png"), 1);
                auto blended = base.Blend(other, mask);
                inspirecv_test_write_image(blended, "f32_color_blend_maskgrad_green_" + std::to_string(N) + ".png");
                auto gt = ImageT<float>::Create(gt_path("blend_maskgrad_green_" + std::to_string(N) + ".png"), 3);
                inspirecv_test_write_image(gt, "f32_gt_color_blend_maskgrad_green_" + std::to_string(N) + ".png");
                REQUIRE_NEAR_C_ARRAY(blended.Data(), gt.Data(), static_cast<size_t>(N) * N * 3, 1.0);
            }

            SECTION("threshold_gray_gt100_f32") {
                auto gray = ImageT<float>::Create(gt_path("gray_" + std::to_string(N) + ".png"), 1);
                auto th = gray.Threshold(100.0, 255.0, 0);
                inspirecv_test_write_image(th, "f32_color_thresh100_gray_" + std::to_string(N) + ".png");
                auto gt = ImageT<float>::Create(gt_path("thresh100_gray_" + std::to_string(N) + ".png"), 1);
                inspirecv_test_write_image(gt, "f32_gt_color_thresh100_gray_" + std::to_string(N) + ".png");
                REQUIRE_NEAR_C_ARRAY(th.Data(), gt.Data(), static_cast<size_t>(N) * N, 1e-3);
            }
        }
    }
}

