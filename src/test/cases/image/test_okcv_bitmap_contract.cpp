#include "../../common/common.h"

#include <inspirecv/core/image.h>
#include <inspirecv/backends/okcv/bitmap/bitmap.h>

#include <cstdint>
#include <type_traits>

static_assert(std::is_same<inspirecv::Image,
                           inspirecv::ImageT<uint8_t>>::value,
              "the public Image alias must remain unchanged");
static_assert(!std::is_same<inspirecv::Image,
                            okcv::Bitmap<uint8_t>>::value,
              "public Image and backend Bitmap must remain distinct types");
static_assert(!std::is_copy_constructible<
                okcv::Bitmap<uint8_t>>::value,
              "Bitmap storage must not be copied implicitly");
static_assert(std::is_move_constructible<
                okcv::Bitmap<uint8_t>>::value,
              "Bitmap storage must remain movable");

TEST_CASE("okcv_bitmap_is_an_internal_storage_type",
          "[image][okcv][contract][naming]") {
    REQUIRE((std::is_same<inspirecv::Image,
                          inspirecv::ImageT<uint8_t>>::value));
    REQUIRE_FALSE((std::is_same<inspirecv::Image,
                                okcv::Bitmap<uint8_t>>::value));
}
