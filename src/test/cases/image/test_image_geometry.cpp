#include "../../common/common.h"
#include "image_test_utils.h"
#include <inspirecv/inspirecv.h>
#include <algorithm>
#include <vector>
#include <string>
#include <cmath>
#include <sstream>
#include <utility>

using inspirecv::Image;
using inspirecv::Rect;
using inspirecv::TransformMatrix;

static std::string gt_dir() {
    return inspirecv_test_images_dir() + "/image_gt";
}
static std::string gt_path(const std::string& name) {
    return gt_dir() + "/" + name;
}
static void dump_image(const Image& img, const std::string& tag, int N) {
    std::string fname = "geom_out_" + tag + "_" + std::to_string(N) + ".png";
    inspirecv_test_write_image(img, fname);
}

static bool check_resize_quality(const Image& actual, const Image& expected,
                                 double min_psnr, double max_mae,
                                 std::string& stats) {
    const size_t count = static_cast<size_t>(actual.Width()) * actual.Height() * actual.Channels();
    double abs_sum = 0.0;
    double squared_sum = 0.0;
    int max_diff = 0;
    for (size_t i = 0; i < count; ++i) {
        const int diff = std::abs(int(actual.Data()[i]) - int(expected.Data()[i]));
        abs_sum += diff;
        squared_sum += static_cast<double>(diff) * diff;
        max_diff = std::max(max_diff, diff);
    }
    const double mae = abs_sum / count;
    const double mse = squared_sum / count;
    const double psnr = mse == 0.0 ? INFINITY : 20.0 * std::log10(255.0 / std::sqrt(mse));
    std::ostringstream stream;
    stream << "resize quality: MAE=" << mae << ", PSNR=" << psnr << " dB, max=" << max_diff;
    stats = stream.str();
    return psnr >= min_psnr && mae <= max_mae;
}

TEST_CASE("image_geometry_resize_rotate_flip_crop_pad_affine_u8", "[image][geom]") {
    const std::string base_path = inspirecv_test_image_path("kun.jpg");
    auto src = Image::Create(base_path, 3);
    INFO(std::string("Base image: ") + base_path);
    REQUIRE_FALSE(src.Empty());

    const int sizes[] = {128, 256, 512, 1024};
    for (int N : sizes) {
        SECTION(std::string("size_") + std::to_string(N)) {
            // Resize nearest
            auto rn = src.Resize(N, N, /*use_linear=*/false);
            dump_image(rn, "resize_nearest", N);
            auto rn_gt = Image::Create(gt_path("resize_nearest_" + std::to_string(N) + ".png"), 3);
            REQUIRE_FALSE(rn_gt.Empty());
            REQUIRE(rn.Width() == N);
            REQUIRE(rn.Height() == N);
            REQUIRE_EQ_C_ARRAY(rn.Data(), rn_gt.Data(), static_cast<size_t>(N) * N * 3);

            // Resize linear
            auto rl = src.Resize(N, N, /*use_linear=*/true);
            dump_image(rl, "resize_linear", N);
            auto rl_gt = Image::Create(gt_path("resize_linear_" + std::to_string(N) + ".png"), 3);
            REQUIRE_FALSE(rl_gt.Empty());
            REQUIRE(rl.Width() == N);
            REQUIRE(rl.Height() == N);
            // Note: OpenCV vs OKCV bilinear coordinate mapping may differ (half-pixel rule).
            // Use perceptual aggregate bounds instead of a misleading per-pixel epsilon.
            inspirecv_test_write_comparison(rl, rl_gt, "resize_linear_" + std::to_string(N));
            std::string resize_stats;
            const bool resize_ok = check_resize_quality(rl, rl_gt, /*min_psnr=*/30.0,
                                                        /*max_mae=*/4.0, resize_stats);
            INFO(resize_stats);
            REQUIRE(resize_ok);

            // For subsequent geometry ops (pure permutations), use the OpenCV linear GT as baseline
            // to remove interpolation policy bias.
            auto base = rl_gt.Clone();

            // Rotations
            {
                auto r90 = base.Rotate90();
                dump_image(r90, "rot90", N);
                auto r90_gt = Image::Create(gt_path("rot90_" + std::to_string(N) + ".png"), 3);
                REQUIRE_FALSE(r90_gt.Empty());
                REQUIRE_EQ_C_ARRAY(r90.Data(), r90_gt.Data(), static_cast<size_t>(N) * N * 3);

                auto r180 = base.Rotate180();
                dump_image(r180, "rot180", N);
                auto r180_gt = Image::Create(gt_path("rot180_" + std::to_string(N) + ".png"), 3);
                REQUIRE_FALSE(r180_gt.Empty());
                REQUIRE_EQ_C_ARRAY(r180.Data(), r180_gt.Data(), static_cast<size_t>(N) * N * 3);

                auto r270 = base.Rotate270();
                dump_image(r270, "rot270", N);
                auto r270_gt = Image::Create(gt_path("rot270_" + std::to_string(N) + ".png"), 3);
                REQUIRE_FALSE(r270_gt.Empty());
                REQUIRE_EQ_C_ARRAY(r270.Data(), r270_gt.Data(), static_cast<size_t>(N) * N * 3);
            }

            // Flips
            {
                auto fh = base.FlipHorizontal();
                dump_image(fh, "flip_h", N);
                auto fh_gt = Image::Create(gt_path("flip_h_" + std::to_string(N) + ".png"), 3);
                REQUIRE_FALSE(fh_gt.Empty());
                REQUIRE_EQ_C_ARRAY(fh.Data(), fh_gt.Data(), static_cast<size_t>(N) * N * 3);

                auto fv = base.FlipVertical();
                dump_image(fv, "flip_v", N);
                auto fv_gt = Image::Create(gt_path("flip_v_" + std::to_string(N) + ".png"), 3);
                REQUIRE_FALSE(fv_gt.Empty());
                REQUIRE_EQ_C_ARRAY(fv.Data(), fv_gt.Data(), static_cast<size_t>(N) * N * 3);
            }

            // Crop center region
            {
                int rx = N / 4, ry = N / 4, rw = N / 2, rh = N / 2;
                auto cropped = base.Crop(Rect<int>(rx, ry, rw, rh));
                dump_image(cropped, "crop", N);
                auto crop_gt = Image::Create(gt_path("crop_" + std::to_string(N) +
                                                     "_x" + std::to_string(rx) +
                                                     "_y" + std::to_string(ry) +
                                                     "_w" + std::to_string(rw) +
                                                     "_h" + std::to_string(rh) + ".png"), 3);
                REQUIRE_FALSE(crop_gt.Empty());
                REQUIRE(cropped.Width() == rw);
                REQUIRE(cropped.Height() == rh);
                REQUIRE_EQ_C_ARRAY(cropped.Data(), crop_gt.Data(), static_cast<size_t>(rw) * rh * 3);
            }

            // Pad 10 pixels (black)
            {
                auto padded = base.Pad(10, 10, 10, 10, {0, 0, 0});
                dump_image(padded, "pad_t10b10l10r10", N);
                auto pad_gt = Image::Create(gt_path("pad_" + std::to_string(N) +
                                                    "_t10_b10_l10_r10_black.png"), 3);
                REQUIRE_FALSE(pad_gt.Empty());
                REQUIRE(padded.Width() == N + 20);
                REQUIRE(padded.Height() == N + 20);
                REQUIRE_EQ_C_ARRAY(padded.Data(), pad_gt.Data(), static_cast<size_t>(N + 20) * (N + 20) * 3);
            }

            // WarpAffine: 180-degree rotation via affine matrix
            {
                TransformMatrix M = TransformMatrix::Create(-1.f, 0.f, static_cast<float>(N - 1),
                                                            0.f, -1.f, static_cast<float>(N - 1));
                auto wa = base.WarpAffine(M, N, N);
                dump_image(wa, "warp_rot180", N);
                auto wa_gt = Image::Create(gt_path("warp_rot180_" + std::to_string(N) + ".png"), 3);
                REQUIRE_FALSE(wa_gt.Empty());
                REQUIRE_EQ_C_ARRAY(wa.Data(), wa_gt.Data(), static_cast<size_t>(N) * N * 3);
            }
        }
    }
}
