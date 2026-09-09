#include "../../common/common.h"
#include "image_test_utils.h"
#include <inspirecv/inspirecv.h>
#include <vector>
#include <cstring>
#include <cmath>
#include <sstream>

TEST_CASE("image_construct_and_properties_u8", "[image][basic]") {
    using inspirecv::Image;

    if (!inspirecv_test_has_image_io()) {
        INFO("Skipping image-based test: OpenCV IO not enabled at build time");
        SUCCEED();
        return;
    }

    const std::string img_path = inspirecv_test_image_path("kun.jpg");
    auto img = Image::Create(img_path, 3);
    REQUIRE_FALSE(img.Empty());
    REQUIRE(img.Channels() == 3);
    REQUIRE(img.Width() > 0);
    REQUIRE(img.Height() > 0);
    const size_t numel = static_cast<size_t>(img.Width()) * img.Height() * img.Channels();

    SECTION("clone is deep copy and equal content") {
        auto clone = img.Clone();
        REQUIRE(clone.Width() == img.Width());
        REQUIRE(clone.Height() == img.Height());
        REQUIRE(clone.Channels() == img.Channels());
        REQUIRE(clone.Data() != img.Data());
        REQUIRE_EQ_C_ARRAY(clone.Data(), img.Data(), numel);
    }

    SECTION("reset with external buffer preserves content") {
        auto empty = Image::Create();
        empty.Reset(img.Width(), img.Height(), img.Channels(), img.Data());
        REQUIRE(empty.Width() == img.Width());
        REQUIRE(empty.Height() == img.Height());
        REQUIRE(empty.Channels() == img.Channels());
        REQUIRE_EQ_C_ARRAY(empty.Data(), img.Data(), numel);
    }

    SECTION("factory create from external buffer (copy) matches content") {
        auto from_buf = Image::Create(img.Width(), img.Height(), img.Channels(), img.Data());
        REQUIRE(from_buf.Width() == img.Width());
        REQUIRE(from_buf.Height() == img.Height());
        REQUIRE(from_buf.Channels() == img.Channels());
        REQUIRE_EQ_C_ARRAY(from_buf.Data(), img.Data(), numel);
    }

    SECTION("create grayscale from file") {
        auto gray = Image::Create(img_path, 1);
        REQUIRE_FALSE(gray.Empty());
        REQUIRE(gray.Channels() == 1);
        REQUIRE(gray.Width() == img.Width());
        REQUIRE(gray.Height() == img.Height());
    }

    SECTION("default empty and then reset size") {
        auto e = Image::Create();
        REQUIRE(e.Empty());
        e.Reset(4, 3, 3);
        REQUIRE_FALSE(e.Empty());
        REQUIRE(e.Width() == 4);
        REQUIRE(e.Height() == 3);
        REQUIRE(e.Channels() == 3);
    }

    SECTION("move semantics keep dimensions and content") {
        std::vector<uint8_t> buf = {
            1,2,3, 4,5,6,
            7,8,9, 10,11,12
        }; // 2x2x3
        Image src(2, 2, 3, buf.data(), true);
        auto before = src.Clone();
        Image moved = std::move(src);
        REQUIRE(moved.Width() == 2);
        REQUIRE(moved.Height() == 2);
        REQUIRE(moved.Channels() == 3);
        REQUIRE_EQ_C_ARRAY(moved.Data(), before.Data(), 2 * 2 * 3);
    }
}


