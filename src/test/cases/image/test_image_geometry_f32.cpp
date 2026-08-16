#include "../../common/common.h"
#include "image_test_utils.h"
#include <inspirecv/inspirecv.h>
#include <string>
#include <utility>
#include <vector>
#include <cstring>
#include <iostream>
#include <cmath>

using inspirecv::ImageT;
using inspirecv::TransformMatrix;
using inspirecv::Rect;

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
        std::cout << "[f32-geom-check:" << tag << "] eps=" << eps
                  << " outliers=" << outliers << " (limit " << max_outliers << ")"
                  << " max=" << maxd << std::endl;
        return false;
    }
    return true;
}

static double compute_psnr_u8(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    double mse = 0.0;
    size_t n = a.size();
    for (size_t i = 0; i < n; ++i) {
        double d = double(int(a[i]) - int(b[i]));
        mse += d * d;
    }
    mse /= double(n);
    if (mse <= 1e-12) return INFINITY;
    return 20.0 * std::log10(255.0 / std::sqrt(mse));
}
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
TEST_CASE("image_geometry_f32", "[image][geom][float]") {
    const int sizes[] = {128, 256, 512, 1024};
    for (int N : sizes) {
        SECTION(std::string("size_") + std::to_string(N)) {
            // Base from OpenCV linear GT to remove interpolation policy bias
            auto base = ImageT<float>::Create(gt_path("resize_linear_" + std::to_string(N) + ".png"), 3);
            REQUIRE_FALSE(base.Empty());

            SECTION("rotations_f32") {
                const double psnr_thresh =
                    (N <= 128) ? 27.0 :
                    (N <= 256) ? 30.0 :
                                  36.0;
                auto r90 = base.Rotate90();
                auto gt90 = ImageT<float>::Create(gt_path("rot90_" + std::to_string(N) + ".png"), 3);
                // Debug artifacts (optional): quantize a copy only for visualization
                std::vector<float> r90q(static_cast<size_t>(N) * N * 3);
                {
                    const float* ad = r90.Data();
                    for (size_t i = 0; i < r90q.size(); ++i) {
                        float v = ad[i]; if (v < 0.f) v = 0.f; else if (v > 255.f) v = 255.f;
                        r90q[i] = std::floor(v + 0.5f);
                    }
                }
                auto r90img = ImageT<float>::Create(N, N, 3, r90q.data());
                auto diff90 = r90img.AbsDiff(gt90);
                inspirecv_test_write_image(diff90, "f32_dbg_diff_rot90_" + std::to_string(N) + ".png");
                inspirecv_test_write_image(diff90.Mul(8.0), "f32_dbg_diffx8_rot90_" + std::to_string(N) + ".png");
                save_concat_f32(r90img, gt90, "f32_dbg_concat_rot90_" + std::to_string(N) + ".png");
                show_concat_if_gui(r90img, gt90, "rot90_f32_concat_" + std::to_string(N));
                {
                    // Round both sides to U8; auto-scale GT if it is in [0,1]
                    size_t n = static_cast<size_t>(N) * N * 3;
                    std::vector<uint8_t> a(n), b(n);
                    // detect scales
                    float act_max = 0.f;
                    const float* ad = r90.Data();
                    for (size_t i = 0; i < n; ++i) if (ad[i] > act_max) act_max = ad[i];
                    float act_scale = (act_max <= 1.5f) ? 255.0f : 1.0f;
                    float gt_max = 0.f;
                    const float* gtd = gt90.Data();
                    for (size_t i = 0; i < n; ++i) if (gtd[i] > gt_max) gt_max = gtd[i];
                    float gt_scale = (gt_max <= 1.5f) ? 255.0f : 1.0f;
                    for (size_t i = 0; i < n; ++i) {
                        float v = ad[i] * act_scale; if (v < 0.f) v = 0.f; else if (v > 255.f) v = 255.f;
                        a[i] = static_cast<uint8_t>(std::floor(v + 0.5f));
                        float g = gt90.Data()[i] * gt_scale; if (g < 0.f) g = 0.f; else if (g > 255.f) g = 255.f;
                        b[i] = static_cast<uint8_t>(std::floor(g + 0.5f));
                    }
                    double psnr = compute_psnr_u8(a, b);
                    if (psnr < psnr_thresh) {
                        std::cout << "[f32-geom-u8:rot90] PSNR=" << psnr << " dB" << std::endl;
                    }
                    REQUIRE(psnr >= psnr_thresh);
                }

                auto r180 = base.Rotate180();
                auto gt180 = ImageT<float>::Create(gt_path("rot180_" + std::to_string(N) + ".png"), 3);
                std::vector<float> r180q(static_cast<size_t>(N) * N * 3);
                {
                    const float* ad = r180.Data();
                    for (size_t i = 0; i < r180q.size(); ++i) {
                        float v = ad[i]; if (v < 0.f) v = 0.f; else if (v > 255.f) v = 255.f;
                        r180q[i] = std::floor(v + 0.5f);
                    }
                }
                auto r180img = ImageT<float>::Create(N, N, 3, r180q.data());
                auto diff180 = r180img.AbsDiff(gt180);
                inspirecv_test_write_image(diff180, "f32_dbg_diff_rot180_" + std::to_string(N) + ".png");
                inspirecv_test_write_image(diff180.Mul(8.0), "f32_dbg_diffx8_rot180_" + std::to_string(N) + ".png");
                save_concat_f32(r180img, gt180, "f32_dbg_concat_rot180_" + std::to_string(N) + ".png");
                show_concat_if_gui(r180img, gt180, "rot180_f32_concat_" + std::to_string(N));
                {
                    size_t n = static_cast<size_t>(N) * N * 3;
                    std::vector<uint8_t> a(n), b(n);
                    float act_max = 0.f;
                    const float* ad = r180.Data();
                    for (size_t i = 0; i < n; ++i) if (ad[i] > act_max) act_max = ad[i];
                    float act_scale = (act_max <= 1.5f) ? 255.0f : 1.0f;
                    float gt_max = 0.f;
                    const float* gtd = gt180.Data();
                    for (size_t i = 0; i < n; ++i) if (gtd[i] > gt_max) gt_max = gtd[i];
                    float gt_scale = (gt_max <= 1.5f) ? 255.0f : 1.0f;
                    for (size_t i = 0; i < n; ++i) {
                        float v = ad[i] * act_scale; if (v < 0.f) v = 0.f; else if (v > 255.f) v = 255.f;
                        a[i] = static_cast<uint8_t>(std::floor(v + 0.5f));
                        float g = gt180.Data()[i] * gt_scale; if (g < 0.f) g = 0.f; else if (g > 255.f) g = 255.f;
                        b[i] = static_cast<uint8_t>(std::floor(g + 0.5f));
                    }
                    double psnr = compute_psnr_u8(a, b);
                    if (psnr < psnr_thresh) {
                        std::cout << "[f32-geom-u8:rot180] PSNR=" << psnr << " dB" << std::endl;
                    }
                    REQUIRE(psnr >= psnr_thresh);
                }

                auto r270 = base.Rotate270();
                auto gt270 = ImageT<float>::Create(gt_path("rot270_" + std::to_string(N) + ".png"), 3);
                std::vector<float> r270q(static_cast<size_t>(N) * N * 3);
                {
                    const float* ad = r270.Data();
                    for (size_t i = 0; i < r270q.size(); ++i) {
                        float v = ad[i]; if (v < 0.f) v = 0.f; else if (v > 255.f) v = 255.f;
                        r270q[i] = std::floor(v + 0.5f);
                    }
                }
                auto r270img = ImageT<float>::Create(N, N, 3, r270q.data());
                auto diff270 = r270img.AbsDiff(gt270);
                inspirecv_test_write_image(diff270, "f32_dbg_diff_rot270_" + std::to_string(N) + ".png");
                inspirecv_test_write_image(diff270.Mul(8.0), "f32_dbg_diffx8_rot270_" + std::to_string(N) + ".png");
                save_concat_f32(r270img, gt270, "f32_dbg_concat_rot270_" + std::to_string(N) + ".png");
                show_concat_if_gui(r270img, gt270, "rot270_f32_concat_" + std::to_string(N));
                {
                    size_t n = static_cast<size_t>(N) * N * 3;
                    std::vector<uint8_t> a(n), b(n);
                    float act_max = 0.f;
                    const float* ad = r270.Data();
                    for (size_t i = 0; i < n; ++i) if (ad[i] > act_max) act_max = ad[i];
                    float act_scale = (act_max <= 1.5f) ? 255.0f : 1.0f;
                    float gt_max = 0.f;
                    const float* gtd = gt270.Data();
                    for (size_t i = 0; i < n; ++i) if (gtd[i] > gt_max) gt_max = gtd[i];
                    float gt_scale = (gt_max <= 1.5f) ? 255.0f : 1.0f;
                    for (size_t i = 0; i < n; ++i) {
                        float v = ad[i] * act_scale; if (v < 0.f) v = 0.f; else if (v > 255.f) v = 255.f;
                        a[i] = static_cast<uint8_t>(std::floor(v + 0.5f));
                        float g = gt270.Data()[i] * gt_scale; if (g < 0.f) g = 0.f; else if (g > 255.f) g = 255.f;
                        b[i] = static_cast<uint8_t>(std::floor(g + 0.5f));
                    }
                    double psnr = compute_psnr_u8(a, b);
                    if (psnr < psnr_thresh) {
                        std::cout << "[f32-geom-u8:rot270] PSNR=" << psnr << " dB" << std::endl;
                    }
                    REQUIRE(psnr >= psnr_thresh);
                }
            }

            SECTION("flips_f32") {
                auto fh = base.FlipHorizontal();
                auto gfh = ImageT<float>::Create(gt_path("flip_h_" + std::to_string(N) + ".png"), 3);
                REQUIRE_NEAR_C_ARRAY(fh.Data(), gfh.Data(), static_cast<size_t>(N) * N * 3, 1e-5);

                auto fv = base.FlipVertical();
                auto gfv = ImageT<float>::Create(gt_path("flip_v_" + std::to_string(N) + ".png"), 3);
                REQUIRE_NEAR_C_ARRAY(fv.Data(), gfv.Data(), static_cast<size_t>(N) * N * 3, 1e-5);
            }

            SECTION("crop_pad_f32") {
                int rx = N / 4, ry = N / 4, rw = N / 2, rh = N / 2;
                auto cropped = base.Crop(Rect<int>(rx, ry, rw, rh));
                auto gcrop = ImageT<float>::Create(gt_path("crop_" + std::to_string(N) +
                                                           "_x" + std::to_string(rx) +
                                                           "_y" + std::to_string(ry) +
                                                           "_w" + std::to_string(rw) +
                                                           "_h" + std::to_string(rh) + ".png"), 3);
                REQUIRE(cropped.Width() == rw);
                REQUIRE(cropped.Height() == rh);
                REQUIRE_NEAR_C_ARRAY(cropped.Data(), gcrop.Data(), static_cast<size_t>(rw) * rh * 3, 1e-5);

                auto padded = base.Pad(10, 10, 10, 10, {0.f, 0.f, 0.f});
                auto gpad = ImageT<float>::Create(gt_path("pad_" + std::to_string(N) +
                                                          "_t10_b10_l10_r10_black.png"), 3);
                REQUIRE(padded.Width() == N + 20);
                REQUIRE(padded.Height() == N + 20);
                REQUIRE_NEAR_C_ARRAY(padded.Data(), gpad.Data(), static_cast<size_t>(N + 20) * (N + 20) * 3, 1e-5);
            }

            SECTION("warp_affine_180_f32") {
                TransformMatrix M = TransformMatrix::Create(-1.f, 0.f, static_cast<float>(N - 1),
                                                            0.f, -1.f, static_cast<float>(N - 1));
                auto wa = base.WarpAffine(M, N, N);
                auto gwa = ImageT<float>::Create(gt_path("warp_rot180_" + std::to_string(N) + ".png"), 3);
                REQUIRE_NEAR_C_ARRAY(wa.Data(), gwa.Data(), static_cast<size_t>(N) * N * 3, 1e-5);
            }
        }
    }
}

