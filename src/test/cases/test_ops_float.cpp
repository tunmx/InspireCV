#include "../common/common.h"
#include <inspirecv/inspirecv.h>
#include <vector>
#include <cmath>

TEST_CASE("test_filters_and_morphology_float", "[ops_float]") {
    DRAW_SPLIT_LINE;

    SECTION("filter/GaussianBlur float") {
        SECTION("ksize=1 returns clone (f32, 1x3)") {
            float data[] = {10.f, 20.f, 30.f};
            auto img = inspirecv::ImageT<float>::Create(3, 1, 1, data);
            auto out = img.GaussianBlur(1, 0.0);
            REQUIRE_NEAR_C_ARRAY(out.Data(), data, 3, 1e-6);
        }

        SECTION("1x3, sigma=1.0, ksize=3, replicate border (f32)") {
            // Input: [0, 100, 0]
            // Separable Gaussian k=3, sigma=1.0
            float in[] = {0.f, 100.f, 0.f};
            auto img = inspirecv::ImageT<float>::Create(3, 1, 1, in);
            auto out = img.GaussianBlur(3, 1.0);
            // a = exp(-0.5) / (1 + 2*exp(-0.5)) ~= 0.27406862
            // b = 1 / (1 + 2*exp(-0.5)) ~= 0.45286276
            // Expected horizontally: [a*100, b*100, a*100]
            float a = std::exp(-0.5f); a = a / (1.f + 2.f * a);
            float b = 1.f / (1.f + 2.f * std::exp(-0.5f));
            float expect[] = {100.f * a, 100.f * b, 100.f * a};
            REQUIRE_NEAR_C_ARRAY(out.Data(), expect, 3, 1e-4);
        }

        SECTION("constant image remains constant (f32)") {
            float in[] = {
              50.f, 50.f, 50.f,
              50.f, 50.f, 50.f,
              50.f, 50.f, 50.f,
            };
            auto img = inspirecv::ImageT<float>::Create(3, 3, 1, in);
            auto out = img.GaussianBlur(5, 0.0);
            REQUIRE_NEAR_C_ARRAY(out.Data(), in, 9, 1e-5);
        }

        SECTION("1x5, sigma auto, ksize=5, replicate border (f32)") {
            float in[] = {0.f, 100.f, 0.f, 0.f, 0.f};
            auto img = inspirecv::ImageT<float>::Create(5, 1, 1, in);
            auto out = img.GaussianBlur(5, 0.0);
#if defined(INSPIRECV_BACKEND_OPENCV)
            // OpenCV's automatic-sigma ksize=5 path uses the exact binomial
            // kernel [1, 4, 6, 4, 1] / 16.
            float expect[] = {25.f, 37.5f, 25.f, 6.25f, 0.f};
#else
            // OKCV preserves its normalized inferred-sigma kernel.
            float expect[] = {24.46f, 36.95f, 24.46f, 7.07f, 0.f};
#endif
            REQUIRE_NEAR_C_ARRAY(out.Data(), expect, 5, 0.02);
        }
    }

    SECTION("morphology/Erode_Dilate float") {
        SECTION("erode: 5x5 with 3x3 solid block shrinks to 1x1 (f32, iter=1)") {
            float in[25] = {0.f};
            auto idx = [&](int y, int x) { return y * 5 + x; };
            for (int y = 1; y <= 3; ++y)
                for (int x = 1; x <= 3; ++x)
                    in[idx(y, x)] = 1.f;
            auto img = inspirecv::ImageT<float>::Create(5, 5, 1, in);
            auto eroded = img.Erode(3, 1);

            float expect[25] = {0.f};
            expect[idx(2, 2)] = 1.f;
            REQUIRE(eroded.Width() == 5);
            REQUIRE(eroded.Height() == 5);
            REQUIRE(eroded.Channels() == 1);
            REQUIRE_NEAR_C_ARRAY(eroded.Data(), expect, 25, 1e-6);
        }

        SECTION("dilate: 5x5 with single 1 expands to 3x3 (f32, iter=1)") {
            float in[25] = {0.f};
            auto idx = [&](int y, int x) { return y * 5 + x; };
            in[idx(2, 2)] = 1.f;
            auto img = inspirecv::ImageT<float>::Create(5, 5, 1, in);
            auto dilated = img.Dilate(3, 1);

            float expect[25] = {0.f};
            for (int y = 1; y <= 3; ++y)
                for (int x = 1; x <= 3; ++x)
                    expect[idx(y, x)] = 1.f;
            REQUIRE_NEAR_C_ARRAY(dilated.Data(), expect, 25, 1e-6);
        }

        SECTION("iterations: erode twice equals applying 3x3 twice (f32)") {
            // 7x7 with centered 5x5 = 1.f; erode k=3 once -> 3x3; twice -> 1x1.
            float in[49] = {0.f};
            auto idx7 = [&](int y, int x) { return y * 7 + x; };
            for (int y = 1; y <= 5; ++y)
                for (int x = 1; x <= 5; ++x)
                    in[idx7(y, x)] = 1.f;
            auto img = inspirecv::ImageT<float>::Create(7, 7, 1, in);
            auto e1 = img.Erode(3, 1);
            auto e2 = e1.Erode(3, 1);

            float expect1[49] = {0.f};
            for (int y = 2; y <= 4; ++y)
                for (int x = 2; x <= 4; ++x)
                    expect1[idx7(y, x)] = 1.f;
            REQUIRE_NEAR_C_ARRAY(e1.Data(), expect1, 49, 1e-6);

            float expect2[49] = {0.f};
            expect2[idx7(3, 3)] = 1.f;
            REQUIRE_NEAR_C_ARRAY(e2.Data(), expect2, 49, 1e-6);
        }
    }
}

TEST_CASE("test_color_and_channels_float", "[ops_float]") {
    DRAW_SPLIT_LINE;

    SECTION("color/ToGray float") {
        float bgr[] = {
          0.f,   0.f, 100.f,   0.f, 100.f, 0.f,
          100.f, 0.f,   0.f,   90.f, 60.f, 30.f
        };
        auto img = inspirecv::ImageT<float>::Create(2, 2, 3, bgr);
        auto gray = img.ToGray();
        float expect[] = {
          100.f * 0.299f,
          100.f * 0.587f,
          100.f * 0.114f,
          30.f * 0.299f + 60.f * 0.587f + 90.f * 0.114f
        };
        REQUIRE_NEAR_C_ARRAY(gray.Data(), expect, 4, 1e-5);
    }

    SECTION("color/MeanChannels float") {
        float rgb[] = {
          3.f, 6.f, 9.f,   12.f, 15.f, 18.f,
          21.f, 24.f, 27.f, 30.f, 33.f, 36.f
        };
        auto img = inspirecv::ImageT<float>::Create(2, 2, 3, rgb);
        auto mean = img.MeanChannels();
        float expect[] = {6.f, 15.f, 24.f, 33.f};
        REQUIRE_NEAR_C_ARRAY(mean.Data(), expect, 4, 1e-6);
    }

    SECTION("color/Threshold float") {
        float data[] = {50.f, 100.f, 101.f, 200.f};
        auto img = inspirecv::ImageT<float>::Create(2, 2, 1, data);
        auto th = img.Threshold(100.0, 255.0, 0);
        // OpenCV THRESH_BINARY uses '>'
        float expect[] = {0.f, 0.f, 255.f, 255.f};
        REQUIRE_NEAR_C_ARRAY(th.Data(), expect, 4, 1e-6);
    }

    SECTION("color/Pad 3ch constant fill float") {
        float center[] = {10.f, 20.f, 30.f};
        auto img = inspirecv::ImageT<float>::Create(1, 1, 3, center);
        auto padded = img.Pad(1, 1, 1, 1, {7.f}); // same value across channels
        REQUIRE(padded.Width() == 3);
        REQUIRE(padded.Height() == 3);
        REQUIRE(padded.Channels() == 3);
        std::vector<float> expect(static_cast<size_t>(3 * 3 * 3), 0.f);
        auto set_px = [&](int y, int x, float r, float g, float b) {
            size_t base = static_cast<size_t>(y) * 3 * 3 + static_cast<size_t>(x) * 3;
            expect[base + 0] = r; expect[base + 1] = g; expect[base + 2] = b;
        };
        for (int y = 0; y < 3; ++y)
            for (int x = 0; x < 3; ++x)
                set_px(y, x, 7.f, 7.f, 7.f);
        set_px(1, 1, 10.f, 20.f, 30.f);
        REQUIRE_NEAR_C_ARRAY(padded.Data(), expect.data(), static_cast<int>(expect.size()), 1e-6);
    }
}

TEST_CASE("test_pixel_ops_float", "[ops_float]") {
    DRAW_SPLIT_LINE;

    SECTION("pixel/AbsDiff float") {
        float a[] = {1.f, -2.f, 3.5f, 10.f};
        float b[] = {0.f, 5.f,  1.5f, 8.f};
        auto A = inspirecv::ImageT<float>::Create(2, 2, 1, a);
        auto B = inspirecv::ImageT<float>::Create(2, 2, 1, b);
        auto D = A.AbsDiff(B);
        float exp[] = {1.f, 7.f, 2.f, 2.f};
        REQUIRE_NEAR_C_ARRAY(D.Data(), exp, 4, 1e-6);
    }

    SECTION("pixel/Blend float 1ch and 3ch") {
        // 1ch
        float A1[] = {100.f, 50.f, 200.f};
        float B1[] = {0.f,   200.f, 100.f};
        uint8_t M1[] = {0, 128, 255};
        auto IA1 = inspirecv::ImageT<float>::Create(3, 1, 1, A1);
        auto IB1 = inspirecv::ImageT<float>::Create(3, 1, 1, B1);
        auto IM1 = inspirecv::Image::Create(3, 1, 1, M1);
        auto O1 = IA1.Blend(IB1, IM1);
        std::vector<float> exp1(3);
        auto blendf = [](float a, float b, uint8_t m) {
            float alpha = static_cast<float>(m) * (1.0f / 255.0f);
            return alpha * a + (1.0f - alpha) * b;
        };
        for (int i = 0; i < 3; ++i) exp1[i] = blendf(A1[i], B1[i], M1[i]);
        REQUIRE_NEAR_C_ARRAY(O1.Data(), exp1.data(), 3, 1e-5);

        // 3ch
        float A3[] = {
          10.f, 20.f, 30.f,
          40.f, 50.f, 60.f,
          70.f, 80.f, 90.f
        };
        float B3[] = {
          90.f, 80.f, 70.f,
          60.f, 50.f, 40.f,
          30.f, 20.f, 10.f
        };
        uint8_t M3[] = {0, 128, 255};
        auto IA3 = inspirecv::ImageT<float>::Create(3, 1, 3, A3);
        auto IB3 = inspirecv::ImageT<float>::Create(3, 1, 3, B3);
        auto IM3 = inspirecv::Image::Create(3, 1, 1, M3);
        auto O3 = IA3.Blend(IB3, IM3);
        std::vector<float> exp3(9);
        for (int x = 0; x < 3; ++x) {
            for (int c = 0; c < 3; ++c) {
                exp3[x * 3 + c] = blendf(A3[x * 3 + c], B3[x * 3 + c], M3[x]);
            }
        }
        REQUIRE_NEAR_C_ARRAY(O3.Data(), exp3.data(), 9, 1e-5);
    }
}

TEST_CASE("test_resize_and_warp_float", "[ops_float]") {
    DRAW_SPLIT_LINE;

    SECTION("resize/Nearest upsample 2x2->4x4 (f32, 1ch)") {
        float in[] = {
          1.f, 2.f,
          3.f, 4.f
        };
        auto img = inspirecv::ImageT<float>::Create(2, 2, 1, in);
        auto out = img.Resize(4, 4, /*use_linear*/false);
        float exp[] = {
          1.f, 1.f, 2.f, 2.f,
          1.f, 1.f, 2.f, 2.f,
          3.f, 3.f, 4.f, 4.f,
          3.f, 3.f, 4.f, 4.f
        };
        REQUIRE_NEAR_C_ARRAY(out.Data(), exp, 16, 1e-6);
    }

    SECTION("resize/Linear 2x2->3x3 (f32, corners check)") {
        float in[] = {
          0.f, 10.f,
          20.f, 30.f
        };
        auto img = inspirecv::ImageT<float>::Create(2, 2, 1, in);
        auto out = img.Resize(3, 3, /*use_linear*/true);
        REQUIRE(out.Width() == 3);
        REQUIRE(out.Height() == 3);
        auto at_out = [&](int y, int x) -> float { return out.Data()[y * 3 + x]; };
        REQUIRE(std::fabs(at_out(0,0) - 0.f) < 1e-5);
        REQUIRE(std::fabs(at_out(0,2) - 10.f) < 1e-5);
        REQUIRE(std::fabs(at_out(2,0) - 20.f) < 1e-5);
        REQUIRE(std::fabs(at_out(2,2) - 30.f) < 1e-5);
    }

    SECTION("warp/Scale 4x4->2x2, m = diag(2,2), f32 1ch") {
        float in[16];
        for (int y = 0; y < 4; ++y)
            for (int x = 0; x < 4; ++x)
                in[y * 4 + x] = static_cast<float>(y * 4 + x);
        auto src = inspirecv::ImageT<float>::Create(4, 4, 1, in);
        auto M = inspirecv::TransformMatrix::Create(
          2.f, 0.f, 0.f,
          0.f, 2.f, 0.f);
        auto dst = src.WarpAffine(M, 2, 2);
        float exp[] = {
          0.f, 2.f,
          8.f, 10.f
        };
        REQUIRE_NEAR_C_ARRAY(dst.Data(), exp, 4, 1e-5);
    }

    SECTION("warp/Translate 4x4->2x2, m = I + [tx=1, ty=1], f32 1ch") {
        float in[16];
        for (int y = 0; y < 4; ++y)
            for (int x = 0; x < 4; ++x)
                in[y * 4 + x] = static_cast<float>(y * 4 + x);
        auto src = inspirecv::ImageT<float>::Create(4, 4, 1, in);
        auto M = inspirecv::TransformMatrix::Create(
          1.f, 0.f, 1.f,
          0.f, 1.f, 1.f);
        auto dst = src.WarpAffine(M, 2, 2);
        float exp[] = {
          5.f, 6.f,
          9.f, 10.f
        };
        REQUIRE_NEAR_C_ARRAY(dst.Data(), exp, 4, 1e-5);
    }
}
