#include "../../common/common.h"
#include "image_test_utils.h"
#include <inspirecv/inspirecv.h>
#include <vector>
#include <string>

TEST_CASE("image_scalar_and_fill_f32", "[image][basic][scalar][fill][float]") {
    using inspirecv::ImageT;
    if (!inspirecv_test_has_image_io()) {
        INFO("Skip float scalar/fill: OpenCV IO not enabled");
        SUCCEED();
        return;
    }
    const std::string path = inspirecv_test_image_path("kun.jpg");
    auto img = ImageT<float>::Create(path, 3);
    REQUIRE_FALSE(img.Empty());
    const int W = img.Width(), H = img.Height(), C = img.Channels();
    const size_t N = static_cast<size_t>(W) * H * C;

    SECTION("fill_all_f32") {
        auto f = img.Clone();
        f.Fill(23.0);
        const float* d = f.Data();
        for (size_t i = 0; i < N; ++i) REQUIRE(std::fabs(d[i] - 23.0f) < 1e-6);
    }

    SECTION("fill_roi_f32") {
        int rw = std::max(1, W / 8), rh = std::max(1, H / 8);
        int rx = std::max(0, W / 4), ry = std::max(0, H / 4);
        auto r = inspirecv::Rect<int>(rx, ry, rw, rh);
        auto t = img.Clone();
        t.Fill(r, {10.0, 20.0, 30.0});
        const float* d = t.Data();
        for (int y = ry; y < ry + rh; ++y) {
            for (int x = rx; x < rx + rw; ++x) {
                size_t base = static_cast<size_t>(y) * W * C + static_cast<size_t>(x) * C;
                REQUIRE(std::fabs(d[base + 0] - 10.0f) <= 1e-5);
                if (C > 1) REQUIRE(std::fabs(d[base + 1] - 20.0f) <= 1e-5);
                if (C > 2) REQUIRE(std::fabs(d[base + 2] - 30.0f) <= 1e-5);
            }
        }
    }

    SECTION("add_mul_roundtrip_f32") {
        auto a = img.Add(10.0);
        auto b = a.Add(-10.0);
        REQUIRE_NEAR_C_ARRAY(b.Data(), img.Data(), N, 1e-6);

        auto z = img.Mul(0.0);
        for (size_t i = 0; i < N; ++i) REQUIRE(std::fabs(z.Data()[i] - 0.0f) < 1e-6);
        auto same = img.Mul(1.0);
        REQUIRE_NEAR_C_ARRAY(same.Data(), img.Data(), N, 1e-6);
    }
}


