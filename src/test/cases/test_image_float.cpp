#include "../common/common.h"
#include <inspirecv/inspirecv.h>

TEST_CASE("test_image_float_ops", "[float]") {
    SECTION("creation and clone") {
        float data_f32[] = {1.0f, 2.5f, 3.0f, 4.25f, 5.0f, 6.75f};
        inspirecv::ImageT<float> image(2, 3, 1, data_f32, true);
        REQUIRE(image.Width() == 2);
        REQUIRE(image.Height() == 3);
        REQUIRE(image.Channels() == 1);
        REQUIRE_EQ_C_ARRAY(image.Data(), data_f32, 2 * 3 * 1);
        auto cloned = image.Clone();
        REQUIRE_EQ_C_ARRAY(cloned.Data(), data_f32, 2 * 3 * 1);
    }

    SECTION("resize no-op") {
        float data[] = {1,2,3,4,5,6,7,8,9,
                        10,11,12,13,14,15,16,17,18};
        auto img = inspirecv::ImageT<float>::Create(2, 3, 3, data);
        auto resized = img.Resize(2, 3);
        REQUIRE_EQ_C_ARRAY(resized.Data(), data, 2 * 3 * 3);

        float data_gray[] = {1,2,3,4,5,6};
        auto img_gray = inspirecv::ImageT<float>::Create(2, 3, 1, data_gray);
        auto resized_gray = img_gray.Resize(2, 3);
        REQUIRE_EQ_C_ARRAY(resized_gray.Data(), data_gray, 2 * 3);
    }

    SECTION("warp affine no-op") {
        float data[] = {1,2,3,4,5,6,7,8,9,
                        10,11,12,13,14,15,16,17,18};
        auto img = inspirecv::ImageT<float>::Create(2, 3, 3, data);
        auto M = inspirecv::TransformMatrix::Create(1, 0, 0, 0, 1, 0);
        auto warped = img.WarpAffine(M, 2, 3);
        REQUIRE_EQ_C_ARRAY(warped.Data(), data, 2 * 3 * 3);

        float data_gray[] = {1,2,3,4,5,6};
        auto img_gray = inspirecv::ImageT<float>::Create(2, 3, 1, data_gray);
        auto warped_gray = img_gray.WarpAffine(M, 2, 3);
        REQUIRE_EQ_C_ARRAY(warped_gray.Data(), data_gray, 2 * 3);
    }

    SECTION("add/mul no-op") {
        float data[] = {1,2,3,4,5,6,7,8,9,
                        10,11,12,13,14,15,16,17,18};
        auto img = inspirecv::ImageT<float>::Create(2, 3, 3, data);
        auto add0 = img.Add(0.0);
        REQUIRE_EQ_C_ARRAY(add0.Data(), data, 2 * 3 * 3);
        auto mul1 = img.Mul(1.0);
        REQUIRE_EQ_C_ARRAY(mul1.Data(), data, 2 * 3 * 3);
    }

    SECTION("crop no-op and rotations") {
        float data[] = {1,2,3,4,5,6,7,8,9,
                        10,11,12,13,14,15,16,17,18};
        auto img = inspirecv::ImageT<float>::Create(2, 3, 3, data);
        auto cropped = img.Crop(inspirecv::Rect<int>(0,0,2,3));
        REQUIRE_EQ_C_ARRAY(cropped.Data(), data, 2 * 3 * 3);
        auto rot = img.Rotate90().Rotate270();
        REQUIRE_EQ_C_ARRAY(rot.Data(), data, 2 * 3 * 3);
        auto flip2 = img.FlipHorizontal().FlipHorizontal();
        REQUIRE_EQ_C_ARRAY(flip2.Data(), data, 2 * 3 * 3);
        auto vflip2 = img.FlipVertical().FlipVertical();
        REQUIRE_EQ_C_ARRAY(vflip2.Data(), data, 2 * 3 * 3);
    }

    SECTION("dimension changes and transforms") {
        float data[] = {1,2,3,4,5,6,7,8,9,
                        10,11,12,13,14,15,16,17,18};
        int width = 2, height = 3;
        auto img = inspirecv::ImageT<float>::Create(width, height, 3, data);
        auto resized = img.Resize(1, 2);
        REQUIRE(resized.Width() == 1);
        REQUIRE(resized.Height() == 2);

        // Rotate 180 via transform and API should match
        float expected_rot180[] = {16,17,18,13,14,15,10,11,12,7,8,9,4,5,6,1,2,3};
        auto T = inspirecv::TransformMatrix::Create(-1, 0, width - 1, 0, -1, height - 1);
        auto warped = img.WarpAffine(T, img.Width(), img.Height());
        REQUIRE_EQ_C_ARRAY(warped.Data(), expected_rot180, 3 * 2 * 3);
        auto rot180 = img.Rotate180();
        REQUIRE_EQ_C_ARRAY(rot180.Data(), expected_rot180, 3 * 2 * 3);
    }

    SECTION("crop small region") {
        float data[] = {1,2,3,4,5,6,7,8,9,
                        10,11,12,13,14,15,16,17,18};
        auto img = inspirecv::ImageT<float>::Create(2, 3, 3, data);
        float expected_crop[] = {1,2,3,7,8,9};
        auto crop = img.Crop(inspirecv::Rect<int>(0,0,1,2));
        REQUIRE(crop.Width() == 1);
        REQUIRE(crop.Height() == 2);
        REQUIRE_EQ_C_ARRAY(crop.Data(), expected_crop, 2 * 1 * 3);
    }

    SECTION("pad image") {
        float data[] = {1,2,3,4,5,6,7,8,9,
                        10,11,12,13,14,15,16,17,18};
        auto img = inspirecv::ImageT<float>::Create(2, 3, 3, data);
        // expected from uint8 test, now as float
        float expected_pad[] = {
            0,0,0,0, 0,0,0,0, 0,0,0,0,
            0,0,0,1,2,3,4,5,6,0,0,0,
            0,0,0,7,8,9,10,11,12,0,0,0,
            0,0,0,13,14,15,16,17,18,0,0,0,
            0,0,0,0, 0,0,0,0, 0,0,0,0
        };
        auto padded = img.Pad(1,1,1,1, {0,0,0});
        REQUIRE(padded.Width() == img.Width() + 2);
        REQUIRE(padded.Height() == img.Height() + 2);
        REQUIRE_NEAR_C_ARRAY(padded.Data(), expected_pad, 3 * 4 * 3, 1e-6);
    }

    SECTION("blend float with u8 mask") {
        // 1x3, 1-channel
        float a_data[] = {100.f, 50.f, 200.f};
        float b_data[] = {0.f,  200.f, 100.f};
        uint8_t m_data[] = {255, 128, 0};
        auto A = inspirecv::ImageT<float>::Create(3, 1, 1, a_data);
        auto B = inspirecv::ImageT<float>::Create(3, 1, 1, b_data);
        auto M = inspirecv::Image::Create(3, 1, 1, m_data);
        auto O = A.Blend(B, M);
        float expected[] = {100.f, 0.5f * 50.f + 0.5f * 200.f, 100.f}; // {100, 125, 100}
        REQUIRE_NEAR_C_ARRAY(O.Data(), expected, 3, 1e-5);
    }

    SECTION("threshold for float single-channel") {
        float d[] = {50.f, 100.f, 150.f, 90.f};
        auto I = inspirecv::ImageT<float>::Create(2, 2, 1, d);
        auto T = I.Threshold(100.0, 255.0, 0);
        float exp[] = {0.f, 255.f, 255.f, 0.f};
        REQUIRE_NEAR_C_ARRAY(T.Data(), exp, 4, 1e-6);
    }
}


