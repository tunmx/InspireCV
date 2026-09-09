#include "../../common/common.h"
#include "image_test_utils.h"
#include <inspirecv/inspirecv.h>
#include <string>

TEST_CASE("image_basics_f32", "[image][basic][float]") {
    using inspirecv::ImageT;
    if (!inspirecv_test_has_image_io()) {
        INFO("Skip float basics: OpenCV IO not enabled");
        SUCCEED();
        return;
    }
    const std::string path = inspirecv_test_image_path("kun.jpg");
    auto img = ImageT<float>::Create(path, 3);
    REQUIRE_FALSE(img.Empty());
    REQUIRE(img.Channels() == 3);
    REQUIRE(img.Width() > 0);
    REQUIRE(img.Height() > 0);

    SECTION("clone_deepcopy_f32") {
        auto c = img.Clone();
        REQUIRE(c.Width() == img.Width());
        REQUIRE(c.Height() == img.Height());
        REQUIRE(c.Channels() == img.Channels());
        REQUIRE_NEAR_C_ARRAY(c.Data(), img.Data(),
                             static_cast<size_t>(img.Width()) * img.Height() * img.Channels(), 1e-6);
    }

    SECTION("reset_from_buffer_f32") {
        auto e = ImageT<float>::Create();
        e.Reset(img.Width(), img.Height(), img.Channels(), img.Data());
        REQUIRE_NEAR_C_ARRAY(e.Data(), img.Data(),
                             static_cast<size_t>(img.Width()) * img.Height() * img.Channels(), 1e-6);
    }
}


