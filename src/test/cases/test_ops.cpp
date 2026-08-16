#include "../common/common.h"
#include <inspirecv/inspirecv.h>
#include <vector>

TEST_CASE("test_filters_and_morphology", "[ops]") {
    DRAW_SPLIT_LINE;

    SECTION("filter/GaussianBlur") {
        SECTION("ksize=1 returns clone (u8, 1x3)") {
            uint8_t data[] = {10, 20, 30};
            auto img = inspirecv::Image::Create(3, 1, 1, data);
            auto out = img.GaussianBlur(1, 0.0);
            REQUIRE_EQ_C_ARRAY(out.Data(), data, 3);
        }

        SECTION("1x3, sigma=1.0, ksize=3, replicate border (u8)") {
            // Input: [0, 100, 0]
            // Separable Gaussian k=3, sigma=1.0, normalized weights:
            // a = exp(-0.5) / (1 + 2*exp(-0.5)) ~= 0.27406862
            // b = 1 / (1 + 2*exp(-0.5)) ~= 0.45286276
            // Horizontal pass gives [~27.4069, ~45.2863, ~27.4069]
            // Vertical pass on 1-row with replicate border is identity.
            uint8_t in[] = {0, 100, 0};
            uint8_t expect[] = {27, 45, 27};  // rounded with +0.5 then floor for u8
            auto img = inspirecv::Image::Create(3, 1, 1, in);
            auto out = img.GaussianBlur(3, 1.0);
            REQUIRE_EQ_C_ARRAY(out.Data(), expect, 3);
        }

        SECTION("constant image remains constant (u8)") {
            uint8_t in[] = {
              50, 50, 50,
              50, 50, 50,
              50, 50, 50,
            };
            auto img = inspirecv::Image::Create(3, 3, 1, in);
            auto out = img.GaussianBlur(5, 0.0);  // any odd k, sigma inferred
            REQUIRE_EQ_C_ARRAY(out.Data(), in, 9);
        }

        SECTION("1x5, sigma auto, ksize=5, replicate border (u8)") {
            // The second output pixel uses source indices [0, 0, 1, 2, 3].
            // This catches asymmetric left-border indexing in optimized paths.
            uint8_t in[] = {0, 100, 0, 0, 0};
#if defined(INSPIRECV_BACKEND_OPENCV)
            // OpenCV defines sigma=0 for ksize=5 with its binomial kernel
            // [1, 4, 6, 4, 1] / 16. Preserve that backend contract exactly.
            uint8_t expect[] = {25, 38, 25, 6, 0};
#else
            // OKCV infers sigma and constructs a normalized Gaussian kernel.
            uint8_t expect[] = {24, 37, 24, 7, 0};
#endif
            auto img = inspirecv::Image::Create(5, 1, 1, in);
            auto out = img.GaussianBlur(5, 0.0);
            REQUIRE_EQ_C_ARRAY(out.Data(), expect, 5);
        }
    }

    SECTION("morphology/Erode_Dilate") {
        SECTION("erode: 5x5 with 3x3 solid block shrinks to 1x1 (u8, iter=1)") {
            // Build 5x5 zeros with a centered 3x3 block of 255
            uint8_t in[25] = {0};
            auto idx = [&](int y, int x) { return y * 5 + x; };
            for (int y = 1; y <= 3; ++y)
                for (int x = 1; x <= 3; ++x)
                    in[idx(y, x)] = 255;
            auto img = inspirecv::Image::Create(5, 5, 1, in);
            auto eroded = img.Erode(3, 1);

            // Expect only center pixel remains 255
            uint8_t expect[25] = {0};
            expect[idx(2, 2)] = 255;
            REQUIRE(eroded.Width() == 5);
            REQUIRE(eroded.Height() == 5);
            REQUIRE(eroded.Channels() == 1);
            REQUIRE_EQ_C_ARRAY(eroded.Data(), expect, 25);
        }

        SECTION("dilate: 5x5 with single 255 expands to 3x3 (u8, iter=1)") {
            uint8_t in[25] = {0};
            auto idx = [&](int y, int x) { return y * 5 + x; };
            in[idx(2, 2)] = 255;
            auto img = inspirecv::Image::Create(5, 5, 1, in);
            auto dilated = img.Dilate(3, 1);

            uint8_t expect[25] = {0};
            for (int y = 1; y <= 3; ++y)
                for (int x = 1; x <= 3; ++x)
                    expect[idx(y, x)] = 255;
            REQUIRE_EQ_C_ARRAY(dilated.Data(), expect, 25);
        }

        SECTION("iterations: erode twice equals applying 3x3 twice (u8)") {
            // Start from 7x7 with centered 5x5 = 255; erode k=3 once -> 3x3; twice -> 1x1.
            uint8_t in[49] = {0};
            auto idx7 = [&](int y, int x) { return y * 7 + x; };
            for (int y = 1; y <= 5; ++y)
                for (int x = 1; x <= 5; ++x)
                    in[idx7(y, x)] = 255;
            auto img = inspirecv::Image::Create(7, 7, 1, in);
            auto e1 = img.Erode(3, 1);
            auto e2 = e1.Erode(3, 1);

            uint8_t expect1[49] = {0};
            for (int y = 2; y <= 4; ++y)
                for (int x = 2; x <= 4; ++x)
                    expect1[idx7(y, x)] = 255;
            REQUIRE_EQ_C_ARRAY(e1.Data(), expect1, 49);

            uint8_t expect2[49] = {0};
            expect2[idx7(3, 3)] = 255;
            REQUIRE_EQ_C_ARRAY(e2.Data(), expect2, 49);
        }
    }
}

TEST_CASE("test_color_and_channels", "[ops]") {
    DRAW_SPLIT_LINE;

    SECTION("color/ToGray u8 and float") {
        // 2x2, 3ch, layout: [B,G,R]
        uint8_t bgr_u8[] = {
          0,   0, 100,   0, 100, 0,
          100, 0,   0,   90, 60, 30
        };
        auto img_u8 = inspirecv::Image::Create(2, 2, 3, bgr_u8);
        auto gray_u8 = img_u8.ToGray();
        // Y = 0.299*R + 0.587*G + 0.114*B (OpenCV rounds to nearest)
        auto r8 = [](double v) -> uint8_t { return static_cast<uint8_t>(std::floor(v + 0.5)); };
        uint8_t y00 = r8(100 * 0.299);                 // 30
        uint8_t y01 = r8(100 * 0.587);                 // 59
        uint8_t y10 = r8(100 * 0.114);                 // 11
        uint8_t y11 = r8(30 * 0.299 + 60 * 0.587 + 90 * 0.114); // 54
        uint8_t expect_u8[] = {y00, y01, y10, y11};
        REQUIRE_EQ_C_ARRAY(gray_u8.Data(), expect_u8, 4);

        float bgr_f32[] = {
          0.f,   0.f, 100.f,   0.f, 100.f, 0.f,
          100.f, 0.f,   0.f,   90.f, 60.f, 30.f
        };
        auto img_f32 = inspirecv::ImageT<float>::Create(2, 2, 3, bgr_f32);
        auto gray_f32 = img_f32.ToGray();
        float expect_f32[] = {
          100.f * 0.299f,
          100.f * 0.587f,
          100.f * 0.114f,
          30.f * 0.299f + 60.f * 0.587f + 90.f * 0.114f
        };
        REQUIRE_NEAR_C_ARRAY(gray_f32.Data(), expect_f32, 4, 1e-5);
    }

    SECTION("color/MeanChannels u8 and float") {
        // 2x2, 3ch
        uint8_t rgb_u8[] = {
          3, 6, 9,   12, 15, 18,
          21, 24, 27, 30, 33, 36
        };
        auto img_u8 = inspirecv::Image::Create(2, 2, 3, rgb_u8);
        auto mean_u8 = img_u8.MeanChannels();
        // truncate after average
        uint8_t expect_u8[] = {
          static_cast<uint8_t>((3 + 6 + 9) / 3),      // 6
          static_cast<uint8_t>((12 + 15 + 18) / 3),   // 15
          static_cast<uint8_t>((21 + 24 + 27) / 3),   // 24
          static_cast<uint8_t>((30 + 33 + 36) / 3)    // 33
        };
        REQUIRE_EQ_C_ARRAY(mean_u8.Data(), expect_u8, 4);

        float rgb_f32[] = {
          3.f, 6.f, 9.f,   12.f, 15.f, 18.f,
          21.f, 24.f, 27.f, 30.f, 33.f, 36.f
        };
        auto img_f32 = inspirecv::ImageT<float>::Create(2, 2, 3, rgb_f32);
        auto mean_f32 = img_f32.MeanChannels();
        float expect_f32[] = {6.f, 15.f, 24.f, 33.f};
        REQUIRE_NEAR_C_ARRAY(mean_f32.Data(), expect_f32, 4, 1e-6);
    }

    SECTION("color/Threshold u8") {
        uint8_t data[] = {50, 100, 101, 200};
        auto img = inspirecv::Image::Create(2, 2, 1, data);
        auto th = img.Threshold(100.0, 255.0, 0);
        // OpenCV THRESH_BINARY uses '>'
        uint8_t expect[] = {0, 0, 255, 255};
        REQUIRE_EQ_C_ARRAY(th.Data(), expect, 4);
    }

    SECTION("color/Pad 3ch constant fill") {
        // 1x1, 3ch center pixel, pad 1 on all sides with value=7 for all channels
        uint8_t center[] = {10, 20, 30};
        auto img = inspirecv::Image::Create(1, 1, 3, center);
        auto padded = img.Pad(1, 1, 1, 1, {7}); // use same value across channels
        REQUIRE(padded.Width() == 3);
        REQUIRE(padded.Height() == 3);
        REQUIRE(padded.Channels() == 3);
        // Build expected interleaved 3x3x3
        std::vector<uint8_t> expect(3 * 3 * 3, 0);
        auto set_px = [&](int y, int x, uint8_t r, uint8_t g, uint8_t b) {
            size_t base = static_cast<size_t>(y) * 3 * 3 + static_cast<size_t>(x) * 3;
            expect[base + 0] = r; expect[base + 1] = g; expect[base + 2] = b;
        };
        // Fill all border with (7,7,7)
        for (int y = 0; y < 3; ++y) {
            for (int x = 0; x < 3; ++x) {
                set_px(y, x, 7, 7, 7);
            }
        }
        // Center retains original
        set_px(1, 1, 10, 20, 30);
        REQUIRE_EQ_C_ARRAY(padded.Data(), expect.data(), static_cast<int>(expect.size()));
    }
}

TEST_CASE("test_pixel_ops", "[ops]") {
    DRAW_SPLIT_LINE;

    SECTION("pixel/AbsDiff u8 and float") {
        uint8_t a_u8[] = {1, 50, 200, 7, 9, 255};
        uint8_t b_u8[] = {3, 20, 100, 7, 30, 0};
        auto A = inspirecv::Image::Create(3, 2, 1, a_u8);
        auto B = inspirecv::Image::Create(3, 2, 1, b_u8);
        auto D = A.AbsDiff(B);
        uint8_t exp_u8[] = {2, 30, 100, 0, 21, 255};
        REQUIRE_EQ_C_ARRAY(D.Data(), exp_u8, 6);

        float a_f[] = {1.f, -2.f, 3.5f, 10.f};
        float b_f[] = {0.f, 5.f,  1.5f, 8.f};
        auto AF = inspirecv::ImageT<float>::Create(2, 2, 1, a_f);
        auto BF = inspirecv::ImageT<float>::Create(2, 2, 1, b_f);
        auto DF = AF.AbsDiff(BF);
        float exp_f[] = {1.f, 7.f, 2.f, 2.f};
        REQUIRE_NEAR_C_ARRAY(DF.Data(), exp_f, 4, 1e-6);
    }

    SECTION("pixel/Blend u8 1ch endpoints and mid mask") {
        // width=3, 1ch
        uint8_t A[] = {100,  50, 200};
        uint8_t B[] = {0,   200, 100};
        uint8_t M[] = {0,   128, 255};
        auto IA = inspirecv::Image::Create(3, 1, 1, A);
        auto IB = inspirecv::Image::Create(3, 1, 1, B);
        auto IM = inspirecv::Image::Create(3, 1, 1, M);
        auto O = IA.Blend(IB, IM);
        // expected = round((m/255)*A + (1-m/255)*B)
        auto rr = [](uint8_t a, uint8_t b, uint8_t m) -> uint8_t {
            double alpha = static_cast<double>(m) / 255.0;
            double v = alpha * a + (1.0 - alpha) * b;
            return static_cast<uint8_t>(std::floor(v + 0.5));
        };
        uint8_t exp[] = {rr(100,0,0), rr(50,200,128), rr(200,100,255)};
        REQUIRE_EQ_C_ARRAY(O.Data(), exp, 3);
    }

    SECTION("pixel/Blend float 3ch") {
        // width=3, 3ch; broadcast mask (1ch)
        float A[] = {
          10.f, 20.f, 30.f,
          40.f, 50.f, 60.f,
          70.f, 80.f, 90.f
        };
        float B[] = {
          90.f, 80.f, 70.f,
          60.f, 50.f, 40.f,
          30.f, 20.f, 10.f
        };
        uint8_t M[] = {0, 128, 255};
        auto IA = inspirecv::ImageT<float>::Create(3, 1, 3, A);
        auto IB = inspirecv::ImageT<float>::Create(3, 1, 3, B);
        auto IM = inspirecv::Image::Create(3, 1, 1, M);
        auto O = IA.Blend(IB, IM);
        std::vector<float> exp(9);
        auto blendf = [](float a, float b, uint8_t m) {
            float alpha = static_cast<float>(m) * (1.0f / 255.0f);
            return alpha * a + (1.0f - alpha) * b;
        };
        for (int x = 0; x < 3; ++x) {
            for (int c = 0; c < 3; ++c) {
                exp[x * 3 + c] = blendf(A[x * 3 + c], B[x * 3 + c], M[x]);
            }
        }
        REQUIRE_NEAR_C_ARRAY(O.Data(), exp.data(), 9, 1e-5);
    }
}

TEST_CASE("test_resize_and_warp", "[ops]") {
    DRAW_SPLIT_LINE;

    SECTION("resize/Nearest upsample 2x2->4x4 (1ch)") {
        uint8_t in[] = {
          1, 2,
          3, 4
        };
        auto img = inspirecv::Image::Create(2, 2, 1, in);
        auto out = img.Resize(4, 4, /*use_linear*/false);
        uint8_t exp[] = {
          1, 1, 2, 2,
          1, 1, 2, 2,
          3, 3, 4, 4,
          3, 3, 4, 4
        };
        REQUIRE_EQ_C_ARRAY(out.Data(), exp, 16);
    }

    SECTION("resize/Nearest downsample 3x3->2x2 (1ch)") {
        uint8_t in[] = {
          1, 2, 3,
          4, 5, 6,
          7, 8, 9
        };
        auto img = inspirecv::Image::Create(3, 3, 1, in);
        auto out = img.Resize(2, 2, /*use_linear*/false);
        // Picks rows 0,1 and cols 0,1
        uint8_t exp[] = {1, 2, 4, 5};
        REQUIRE_EQ_C_ARRAY(out.Data(), exp, 4);
    }

    SECTION("resize/Linear 2x2->3x3 (float, corners check)") {
        float in[] = {
          0.f, 10.f,
          20.f, 30.f
        };
        auto img = inspirecv::ImageT<float>::Create(2, 2, 1, in);
        auto out = img.Resize(3, 3, /*use_linear*/true);
        REQUIRE(out.Width() == 3);
        REQUIRE(out.Height() == 3);
        // Corners should match source corners for both backends
        auto at = [&](int y, int x) -> float { return out.Data()[y * 3 + x]; };
        REQUIRE(std::fabs(at(0,0) - 0.f) < 1e-5);
        REQUIRE(std::fabs(at(0,2) - 10.f) < 1e-5);
        REQUIRE(std::fabs(at(2,0) - 20.f) < 1e-5);
        REQUIRE(std::fabs(at(2,2) - 30.f) < 1e-5);
    }

    SECTION("warp/Scale 4x4->2x2, m = diag(2,2)") {
        // Source 4x4 gradient: val = y*4 + x
        uint8_t in[16];
        for (int y = 0; y < 4; ++y)
            for (int x = 0; x < 4; ++x)
                in[y * 4 + x] = static_cast<uint8_t>(y * 4 + x);
        auto src = inspirecv::Image::Create(4, 4, 1, in);
        auto M = inspirecv::TransformMatrix::Create(
          2.f, 0.f, 0.f,
          0.f, 2.f, 0.f);
        auto dst = src.WarpAffine(M, 2, 2);
        uint8_t exp[] = {
          0, 2,
          8, 10
        };
        REQUIRE_EQ_C_ARRAY(dst.Data(), exp, 4);
    }

    SECTION("warp/Translate 4x4->2x2, m = I + [tx=1, ty=1]") {
        uint8_t in[16];
        for (int y = 0; y < 4; ++y)
            for (int x = 0; x < 4; ++x)
                in[y * 4 + x] = static_cast<uint8_t>(y * 4 + x);
        auto src = inspirecv::Image::Create(4, 4, 1, in);
        auto M = inspirecv::TransformMatrix::Create(
          1.f, 0.f, 1.f,
          0.f, 1.f, 1.f);
        auto dst = src.WarpAffine(M, 2, 2);
        uint8_t exp[] = {
          5, 6,
          9, 10
        };
        REQUIRE_EQ_C_ARRAY(dst.Data(), exp, 4);
    }
}
