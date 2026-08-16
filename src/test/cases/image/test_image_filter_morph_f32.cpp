#include "../../common/common.h"
#include "image_test_utils.h"
#include <inspirecv/inspirecv.h>
#include <string>
#include <cmath>
#include <iostream>

using inspirecv::ImageT;

static std::string gt_dir() {
    return inspirecv_test_images_dir() + "/image_gt";
}
static std::string gt_path(const std::string& name) {
    return gt_dir() + "/" + name;
}

static bool check_gaussian_f32_with_border(const ImageT<float>& g, const ImageT<float>& gt,
                                           float eps_interior, float eps_border,
                                           int radius,
                                           long max_interior_outliers,
                                           long max_border_outliers,
                                           const char* tag) {
    const int W = g.Width(), H = g.Height(), C = g.Channels();
    const float* a = g.Data();
    const float* b = gt.Data();
    size_t stride = static_cast<size_t>(W) * C;
    long fail_in = 0, fail_bd = 0;
    for (int y = 0; y < H; ++y) {
        bool border_y = (y < radius) || (y >= H - radius);
        for (int x = 0; x < W; ++x) {
            bool border = border_y || (x < radius) || (x >= W - radius);
            size_t base = static_cast<size_t>(y) * stride + static_cast<size_t>(x) * C;
            for (int c = 0; c < C; ++c) {
                float d = std::fabs(a[base + c] - b[base + c]);
                if (!border) {
                    if (d > eps_interior) { ++fail_in; break; }
                } else {
                    if (d > eps_border) { ++fail_bd; break; }
                }
            }
        }
    }
    if (fail_in > max_interior_outliers || fail_bd > max_border_outliers) {
        std::cout << "[gauss-f32-check:" << tag << "] interior_eps=" << eps_interior
                  << " border_eps=" << eps_border
                  << " radius=" << radius
                  << " interior_fail=" << fail_in
                  << " border_fail=" << fail_bd << std::endl;
        return false;
    }
    return true;
}

TEST_CASE("image_filter_and_morphology_f32", "[image][filter][morph][float]") {
    const int sizes[] = {128, 256, 512, 1024};
    for (int N : sizes) {
        SECTION(std::string("size_") + std::to_string(N)) {
            auto base = ImageT<float>::Create(gt_path("resize_linear_" + std::to_string(N) + ".png"), 3);
            auto gray = ImageT<float>::Create(gt_path("gray_" + std::to_string(N) + ".png"), 1);
            REQUIRE_FALSE(base.Empty());
            REQUIRE_FALSE(gray.Empty());

            SECTION("gauss5_bgr_f32") {
                auto g = base.GaussianBlur(5, 0.0);
                inspirecv_test_write_image(g, "f32_filter_gauss5_bgr_" + std::to_string(N) + ".png");
                auto gt = ImageT<float>::Create(gt_path("gauss5_bgr_" + std::to_string(N) + ".png"), 3);
                inspirecv_test_write_image(gt, "f32_gt_filter_gauss5_bgr_" + std::to_string(N) + ".png");
                REQUIRE_FALSE(gt.Empty());
                inspirecv_test_write_comparison(g, gt, "gauss5_bgr_f32_" + std::to_string(N));
                bool ok = check_gaussian_f32_with_border(g, gt, /*eps_interior=*/2.0f, /*eps_border=*/3.0f,
                                                         /*radius=*/2, /*in_outliers=*/20, /*bd_outliers=*/0,
                                                         "gauss5_bgr_f32");
                REQUIRE(ok);
            }

            SECTION("gauss5_gray_f32") {
                auto g = gray.GaussianBlur(5, 0.0);
                inspirecv_test_write_image(g, "f32_filter_gauss5_gray_" + std::to_string(N) + ".png");
                auto gt = ImageT<float>::Create(gt_path("gauss5_gray_" + std::to_string(N) + ".png"), 1);
                inspirecv_test_write_image(gt, "f32_gt_filter_gauss5_gray_" + std::to_string(N) + ".png");
                REQUIRE_FALSE(gt.Empty());
                inspirecv_test_write_comparison(g, gt, "gauss5_gray_f32_" + std::to_string(N));
                bool ok = check_gaussian_f32_with_border(g, gt, /*eps_interior=*/2.0f, /*eps_border=*/3.0f,
                                                         /*radius=*/2, /*in_outliers=*/20, /*bd_outliers=*/0,
                                                         "gauss5_gray_f32");
                REQUIRE(ok);
            }

            SECTION("erode3_gray_f32") {
                auto e = gray.Erode(3, 1);
                inspirecv_test_write_image(e, "f32_morph_erode3_gray_" + std::to_string(N) + ".png");
                auto gt = ImageT<float>::Create(gt_path("erode3_gray_" + std::to_string(N) + ".png"), 1);
                inspirecv_test_write_image(gt, "f32_gt_morph_erode3_gray_" + std::to_string(N) + ".png");
                REQUIRE_NEAR_C_ARRAY(e.Data(), gt.Data(), static_cast<size_t>(N) * N, 1e-3);
            }

            SECTION("dilate3_gray_f32") {
                auto d = gray.Dilate(3, 1);
                inspirecv_test_write_image(d, "f32_morph_dilate3_gray_" + std::to_string(N) + ".png");
                auto gt = ImageT<float>::Create(gt_path("dilate3_gray_" + std::to_string(N) + ".png"), 1);
                inspirecv_test_write_image(gt, "f32_gt_morph_dilate3_gray_" + std::to_string(N) + ".png");
                REQUIRE_NEAR_C_ARRAY(d.Data(), gt.Data(), static_cast<size_t>(N) * N, 1e-3);
            }
        }
    }
}
