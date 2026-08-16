#include "../../common/common.h"
#include "image_test_utils.h"
#include <inspirecv/inspirecv.h>
#include <vector>
#include <cstdint>
#include <string>

TEST_CASE("image_scalar_and_fill_u8", "[image][basic][scalar][fill]") {
    using inspirecv::Image;
    using inspirecv::Rect;

    if (!inspirecv_test_has_image_io()) {
        INFO("Skipping image-based test: OpenCV IO not enabled at build time");
        SUCCEED();
        return;
    }

    const std::string img_path = inspirecv_test_image_path("kun.jpg");
    auto img = Image::Create(img_path, 3);
    REQUIRE_FALSE(img.Empty());
    const int W = img.Width(), H = img.Height(), C = img.Channels();
    const size_t N = static_cast<size_t>(W) * H * C;

    SECTION("Fill whole image with scalar value") {
        auto fill_img = img.Clone();
        const uint8_t v = 23;
        fill_img.Fill(v);
        const uint8_t* d = fill_img.Data();
        for (size_t i = 0; i < N; ++i) {
            REQUIRE(d[i] == v);
        }
    }

    SECTION("Fill rectangular ROI with color") {
        // Choose a mid-area ROI and clamp to image bounds
        int rw = std::max(1, W / 8), rh = std::max(1, H / 8);
        int rx = std::max(0, W / 4), ry = std::max(0, H / 4);
        Rect<int> r(rx, ry, rw, rh);
        // Fill color
        std::vector<double> color = {10, 20, 30};

        auto roi_img = img.Clone();
        roi_img.Fill(r, color);

        const uint8_t* d = roi_img.Data();
        // Check ROI pixels equal to color
        for (int y = ry; y < ry + rh; ++y) {
            for (int x = rx; x < rx + rw; ++x) {
                const size_t base = static_cast<size_t>(y) * W * C + static_cast<size_t>(x) * C;
                REQUIRE(d[base + 0] == static_cast<uint8_t>(color[0]));
                if (C > 1) REQUIRE(d[base + 1] == static_cast<uint8_t>(color[1]));
                if (C > 2) REQUIRE(d[base + 2] == static_cast<uint8_t>(color[2]));
            }
        }
    }

    SECTION("Add scalar uses defined uint8 wraparound") {
        auto added = img.Add(10.0);
        auto back = added.Add(-10.0);
        REQUIRE(back.Width() == img.Width());
        REQUIRE(back.Height() == img.Height());
        REQUIRE(back.Channels() == img.Channels());
        std::vector<uint8_t> expected_added(N);
        std::vector<uint8_t> expected_back(N);
        for (size_t i = 0; i < N; ++i) {
            expected_added[i] = static_cast<uint8_t>(int(img.Data()[i]) + 10);
            expected_back[i] = static_cast<uint8_t>(int(expected_added[i]) - 10);
        }
        REQUIRE_EQ_C_ARRAY(added.Data(), expected_added.data(), N);
        REQUIRE_EQ_C_ARRAY(back.Data(), expected_back.data(), N);
    }

    SECTION("Mul invariants") {
        auto zero = img.Mul(0.0);
        const uint8_t* zd = zero.Data();
        for (size_t i = 0; i < N; ++i) REQUIRE(zd[i] == 0);

        auto same = img.Mul(1.0);
        REQUIRE_EQ_C_ARRAY(same.Data(), img.Data(), N);
    }
}
