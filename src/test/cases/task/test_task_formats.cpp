#include "../../common/common.h"
#include <inspirecv/task/legacy.h>
#include <inspirecv/task/task.h>
#include <inspirecv/core/transform_matrix.h>
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>

using inspirecv::task::StreamTask;
using inspirecv::task::StreamFormat;
using inspirecv::task::Filter;
using inspirecv::task::Wrap;

static int channelsOf(StreamFormat fmt) {
    switch (fmt) {
        case StreamFormat::GRAY: return 1;
        case StreamFormat::BGR:
        case StreamFormat::RGB: return 3;
        case StreamFormat::BGRA:
        case StreamFormat::RGBA: return 4;
        default: return 0; // unknown or YUV (not used as dest in these tests)
    }
}

static inspirecv::task::Status convertU8(
  const uint8_t* src, int iw, int ih, int istride, StreamFormat sfmt,
  uint8_t* dst, int ow, int oh, int ostride, StreamFormat dfmt,
  Filter filter = Filter::NEAREST, Wrap wrap = Wrap::CLAMP_TO_EDGE) {
    inspirecv::task::PipelineOptions options;
    options.input_format = static_cast<inspirecv::task::PixelFormat>(sfmt);
    options.output_format = static_cast<inspirecv::task::PixelFormat>(dfmt);
    options.sampling = static_cast<inspirecv::task::SamplingMode>(filter);
    options.border = static_cast<inspirecv::task::BorderMode>(wrap);
    inspirecv::task::Pipeline pipeline(options);
    pipeline.SetTransform(inspirecv::TransformMatrix::Identity());
    inspirecv::task::RawImageView source;
    source.data = src;
    source.width = iw;
    source.height = ih;
    source.row_stride_bytes = istride;
    inspirecv::task::TensorBuffer destination;
    destination.data = dst;
    destination.width = ow;
    destination.height = oh;
    destination.channels = channelsOf(dfmt);
    destination.element_type = inspirecv::task::ElementType::kUInt8;
    destination.order = inspirecv::task::TensorOrder::kHwc;
    destination.row_stride_bytes = ostride;
    return pipeline.Run(source, destination);
}

static void runConvertU8(const uint8_t* src, int iw, int ih, int istride,
                         StreamFormat sfmt,
                         uint8_t* dst, int ow, int oh, int ostride,
                         StreamFormat dfmt) {
    auto st = convertU8(src, iw, ih, istride, sfmt, dst, ow, oh, ostride, dfmt);
    REQUIRE(st == inspirecv::task::Status::kOk);
}

TEST_CASE("task_formats_basic_conversions", "[task][formats]") {
    // 2x2 synthetic pixels for easy verification
    // Layout by pixel: P0,P1,P2,P3
    SECTION("RGB -> BGR channel swap") {
        uint8_t rgb[] = {
          // R, G, B
          10, 20, 30,
          40, 50, 60,
          70, 80, 90,
          100,110,120
        };
        uint8_t out[sizeof(rgb)] = {0};
        runConvertU8(rgb, 2, 2, 2 * 3, StreamFormat::RGB, out, 2, 2, 2 * 3, StreamFormat::BGR);
        uint8_t expected[] = {
          30, 20, 10,
          60, 50, 40,
          90, 80, 70,
          120,110,100
        };
        REQUIRE_EQ_C_ARRAY(out, expected, sizeof(expected));
    }

    SECTION("BGR -> BGRA alpha=255 and BGRA -> BGR roundtrip") {
        uint8_t bgr[] = {
          3, 2, 1,
          6, 5, 4,
          9, 8, 7,
          12,11,10
        };
        uint8_t bgra[4 * 4] = {0};
        runConvertU8(bgr, 2, 2, 2 * 3, StreamFormat::BGR, bgra, 2, 2, 2 * 4, StreamFormat::BGRA);
        // Check alpha and RGB keep
        for (int i = 0; i < 4; ++i) {
            REQUIRE(bgra[4 * i + 0] == bgr[3 * i + 0]);
            REQUIRE(bgra[4 * i + 1] == bgr[3 * i + 1]);
            REQUIRE(bgra[4 * i + 2] == bgr[3 * i + 2]);
            REQUIRE(bgra[4 * i + 3] == 255);
        }
        uint8_t back_bgr[sizeof(bgr)] = {0};
        runConvertU8(bgra, 2, 2, 2 * 4, StreamFormat::BGRA, back_bgr, 2, 2, 2 * 3, StreamFormat::BGR);
        REQUIRE_EQ_C_ARRAY(back_bgr, bgr, sizeof(bgr));
    }

    SECTION("GRAY -> BGR replicate channels") {
        uint8_t gray[] = {
          10, 200,
          30,  40
        };
        uint8_t bgr[2 * 2 * 3] = {0};
        runConvertU8(gray, 2, 2, 2 * 1, StreamFormat::GRAY, bgr, 2, 2, 2 * 3, StreamFormat::BGR);
        for (int i = 0; i < 4; ++i) {
            REQUIRE(bgr[3 * i + 0] == gray[i]);
            REQUIRE(bgr[3 * i + 1] == gray[i]);
            REQUIRE(bgr[3 * i + 2] == gray[i]);
        }
    }
}

TEST_CASE("task_formats_stride_variants", "[task][formats][stride]") {
    SECTION("source stride with padding rows") {
        const int W = 3, H = 2, C = 3;
        uint8_t tight[W * H * C] = {
          1,2,3, 4,5,6, 7,8,9,
          10,11,12, 13,14,15, 16,17,18
        };
        // Make a padded buffer: stride = W*C + pad
        const int pad = 5;
        const int stride = W * C + pad;
        std::vector<uint8_t> padded(H * stride, 0xEE);
        for (int y = 0; y < H; ++y) {
            std::memcpy(&padded[y * stride], &tight[y * (W * C)], W * C);
        }
        // Convert padded BGR->BGR and compare with tight
        uint8_t out_tight[sizeof(tight)] = {0};
        runConvertU8(tight, W, H, W * C, StreamFormat::BGR, out_tight, W, H, W * C, StreamFormat::BGR);
        uint8_t out_padded[sizeof(tight)] = {0};
        runConvertU8(padded.data(), W, H, stride, StreamFormat::BGR, out_padded, W, H, W * C, StreamFormat::BGR);
        REQUIRE_EQ_C_ARRAY(out_padded, out_tight, sizeof(tight));
    }

    SECTION("destination stride with padding rows") {
        const int W = 2, H = 2, C = 3;
        uint8_t bgr[W * H * C] = {
          1,2,3, 4,5,6,
          7,8,9, 10,11,12
        };
        // tight output
        uint8_t tight_out[sizeof(bgr)] = {0};
        runConvertU8(bgr, W, H, W * C, StreamFormat::BGR, tight_out, W, H, W * C, StreamFormat::BGR);
        // padded output
        const int pad = 4;
        const int stride = W * C + pad;
        std::vector<uint8_t> padded(H * stride, 0xEE);
        runConvertU8(bgr, W, H, W * C, StreamFormat::BGR, padded.data(), W, H, stride, StreamFormat::BGR);
        const int rowBytes = W * C;
        for (int y = 0; y < H; ++y) {
            REQUIRE_EQ_C_ARRAY(padded.data() + y * stride, tight_out + y * rowBytes, rowBytes);
            for (int x = rowBytes; x < stride; ++x) {
                REQUIRE(padded[y * stride + x] == 0xEE);
            }
        }
    }

    SECTION("destination stride is honored during format conversion") {
        const int W = 3, H = 2, C = 3;
        uint8_t rgb[W * H * C] = {
          1,2,3, 4,5,6, 7,8,9,
          10,11,12, 13,14,15, 16,17,18
        };
        const int rowBytes = W * C;
        const int stride = rowBytes + 5;
        std::vector<uint8_t> tight(H * rowBytes, 0);
        std::vector<uint8_t> padded(H * stride, 0xEE);
        runConvertU8(rgb, W, H, rowBytes, StreamFormat::RGB,
                     tight.data(), W, H, rowBytes, StreamFormat::BGR);
        runConvertU8(rgb, W, H, rowBytes, StreamFormat::RGB,
                     padded.data(), W, H, stride, StreamFormat::BGR);
        for (int y = 0; y < H; ++y) {
            REQUIRE_EQ_C_ARRAY(padded.data() + y * stride, tight.data() + y * rowBytes, rowBytes);
            for (int x = rowBytes; x < stride; ++x) {
                REQUIRE(padded[y * stride + x] == 0xEE);
            }
        }
    }
}

TEST_CASE("task_formats_report_unsupported_configuration", "[task][formats][errors]") {
    uint8_t src[2 * 2 * 4] = {
        1,2,3,4, 5,6,7,8,
        9,10,11,12, 13,14,15,16
    };
    std::vector<uint8_t> dst(64, 0xEE);

    SECTION("bicubic does not silently fall back to nearest") {
        auto st = convertU8(src, 2, 2, 2 * 3, StreamFormat::BGR,
                            dst.data(), 3, 3, 3 * 3, StreamFormat::BGR,
                            Filter::BICUBIC, Wrap::CLAMP_TO_EDGE);
        REQUIRE(st == inspirecv::task::Status::kUnsupportedSampling);
    }

    SECTION("repeat wrap does not silently behave as clamp") {
        auto st = convertU8(src, 2, 2, 2 * 3, StreamFormat::BGR,
                            dst.data(), 3, 3, 3 * 3, StreamFormat::BGR,
                            Filter::NEAREST, Wrap::REPEAT);
        REQUIRE(st == inspirecv::task::Status::kUnsupportedSampling);
    }

    SECTION("unsupported conversion status is propagated") {
        auto st = convertU8(src, 2, 2, 2 * 4, StreamFormat::BGRA,
                            dst.data(), 2, 2, 2 * 3, StreamFormat::HSV);
        REQUIRE(st == inspirecv::task::Status::kUnsupportedConversion);
    }

    SECTION("undersized destination stride is rejected") {
        auto st = convertU8(src, 2, 2, 2 * 3, StreamFormat::BGR,
                            dst.data(), 2, 2, 5, StreamFormat::BGR);
        REQUIRE(st == inspirecv::task::Status::kInvalidArgument);
    }
}

TEST_CASE("task_draw_uses_exact_color_bytes", "[task][draw]") {
    const int W = 3, H = 3, C = 3;
    std::vector<uint8_t> image(W * H * C, 0);
    const int regions[] = {1, 0, 2};
    const uint8_t color[] = {7, 8, 9};

    StreamTask::Config cfg;
    cfg.sourceFormat = StreamFormat::BGR;
    cfg.destFormat = StreamFormat::BGR;
    auto proc = StreamTask::Create(cfg);
    proc->SetDraw();
    proc->Draw(image.data(), W, H, C, regions, 1, color);
    StreamTask::Destroy(proc);

    const uint8_t expected[] = {
        0,0,0, 0,0,0, 0,0,0,
        7,8,9, 7,8,9, 7,8,9,
        0,0,0, 0,0,0, 0,0,0
    };
    REQUIRE_EQ_C_ARRAY(image.data(), expected, image.size());
}

TEST_CASE("task_formats_yuv_constraints", "[task][formats][yuv]") {
    auto makeNV21 = [](int W, int H, std::vector<uint8_t>& buf, int& yStride, int& uvStride) {
        yStride = W;
        uvStride = ((W + 1) / 2) * 2;
        size_t Y = (size_t)H * yStride;
        size_t UV = (size_t)((H + 1) / 2) * uvStride;
        buf.resize(Y + UV);
        // Fill Y
        for (size_t i = 0; i < Y; ++i) buf[i] = static_cast<uint8_t>((i * 7) & 0xFF);
        // Fill VU interleaved
        uint8_t* uv = buf.data() + Y;
        for (size_t r = 0; r < (size_t)((H + 1) / 2); ++r) {
            for (size_t c = 0; c < (size_t)uvStride; c += 2) {
                size_t idx = r * uvStride + c;
                uv[idx + 0] = static_cast<uint8_t>((10 + r + c) & 0xFF);  // V
                uv[idx + 1] = static_cast<uint8_t>((200 - r - c) & 0xFF); // U
            }
        }
    };
    auto makeNV12 = [](int W, int H, std::vector<uint8_t>& buf, int& yStride, int& uvStride) {
        yStride = W;
        uvStride = ((W + 1) / 2) * 2;
        size_t Y = (size_t)H * yStride;
        size_t UV = (size_t)((H + 1) / 2) * uvStride;
        buf.resize(Y + UV);
        for (size_t i = 0; i < Y; ++i) buf[i] = static_cast<uint8_t>((i * 5) & 0xFF);
        uint8_t* uv = buf.data() + Y;
        for (size_t r = 0; r < (size_t)((H + 1) / 2); ++r) {
            for (size_t c = 0; c < (size_t)uvStride; c += 2) {
                size_t idx = r * uvStride + c;
                uv[idx + 0] = static_cast<uint8_t>((50 + r + c) & 0xFF);   // U
                uv[idx + 1] = static_cast<uint8_t>((150 - r - c) & 0xFF);  // V
            }
        }
    };
    auto makeI420 = [](int W, int H, std::vector<uint8_t>& buf, int& yStride, int& uStride, int& vStride) {
        yStride = W;
        uStride = ((W + 1) / 2);
        vStride = ((W + 1) / 2);
        size_t Y = (size_t)H * yStride;
        size_t U = (size_t)((H + 1) / 2) * uStride;
        size_t V = (size_t)((H + 1) / 2) * vStride;
        buf.resize(Y + U + V);
        for (size_t i = 0; i < Y; ++i) buf[i] = static_cast<uint8_t>((i * 3) & 0xFF);
        uint8_t* Up = buf.data() + Y;
        uint8_t* Vp = Up + U;
        for (size_t i = 0; i < U; ++i) Up[i] = static_cast<uint8_t>((80 + i) & 0xFF);
        for (size_t i = 0; i < V; ++i) Vp[i] = static_cast<uint8_t>((160 + i) & 0xFF);
    };

    SECTION("NV21 even width/height: stride 0 equals explicit stride") {
        // For even W, UV row stride equals W,单一stride对Y与UV均成立
        const int W = 6, H = 4;
        std::vector<uint8_t> src;
        int yStride = 0, uvStride = 0;
        makeNV21(W, H, src, yStride, uvStride);
        // Convert with stride=0
        std::vector<uint8_t> out0((size_t)W * H * 3);
        runConvertU8(src.data(), W, H, 0, StreamFormat::YUV_NV21, out0.data(), W, H, 0, StreamFormat::BGR);
        // Convert with explicit stride (W)
        std::vector<uint8_t> out1((size_t)W * H * 3);
        runConvertU8(src.data(), W, H, yStride, StreamFormat::YUV_NV21, out1.data(), W, H, 0, StreamFormat::BGR);
        REQUIRE_EQ_C_ARRAY(out0.data(), out1.data(), out0.size());
        // Basic sanity: not all zeros
        bool any = false; for (auto v : out0) if (v) { any = true; break; } REQUIRE(any);
    }

    SECTION("NV21 odd width/height: stride 0 works (no explicit stride equality)") {
        const int W = 3, H = 5;
        std::vector<uint8_t> src;
        int yStride = 0, uvStride = 0;
        makeNV21(W, H, src, yStride, uvStride);
        std::vector<uint8_t> out0((size_t)W * H * 3);
        runConvertU8(src.data(), W, H, 0, StreamFormat::YUV_NV21, out0.data(), W, H, 0, StreamFormat::BGR);
        bool any = false; for (auto v : out0) if (v) { any = true; break; } REQUIRE(any);
    }

    SECTION("NV21 odd height accepts the compact floor-height chroma plane") {
        const int W = 4, H = 5;
        const size_t yBytes = static_cast<size_t>(W) * H;
        const size_t uvBytes = static_cast<size_t>(W) * (H / 2);
        std::vector<uint8_t> src(yBytes + uvBytes, 100);
        uint8_t* uv = src.data() + yBytes;
        // Two chroma rows. The unpaired final luma row must reuse row 1,
        // rather than reading a non-existent ceil(H / 2) row.
        std::fill(uv, uv + W, 128);
        std::fill(uv + W, uv + 2 * W, 96);

        std::vector<uint8_t> out(static_cast<size_t>(W) * H * 3);
        runConvertU8(src.data(), W, H, 0, StreamFormat::YUV_NV21,
                     out.data(), W, H, 0, StreamFormat::BGR);
        REQUIRE_EQ_C_ARRAY(out.data() + static_cast<size_t>(3) * W * 3,
                           out.data() + static_cast<size_t>(4) * W * 3,
                           static_cast<size_t>(W) * 3);
    }

    SECTION("NV12 even width/height: stride 0 equals explicit stride") {
        const int W = 8, H = 6;
        std::vector<uint8_t> src;
        int yStride = 0, uvStride = 0;
        makeNV12(W, H, src, yStride, uvStride);
        std::vector<uint8_t> out0((size_t)W * H * 3);
        runConvertU8(src.data(), W, H, 0, StreamFormat::YUV_NV12, out0.data(), W, H, 0, StreamFormat::BGR);
        std::vector<uint8_t> out1((size_t)W * H * 3);
        runConvertU8(src.data(), W, H, yStride, StreamFormat::YUV_NV12, out1.data(), W, H, 0, StreamFormat::BGR);
        REQUIRE_EQ_C_ARRAY(out0.data(), out1.data(), out0.size());
        bool any = false; for (auto v : out0) if (v) { any = true; break; } REQUIRE(any);
    }

    SECTION("NV12 odd width/height: stride 0 works (no explicit stride equality)") {
        const int W = 5, H = 3;
        std::vector<uint8_t> src;
        int yStride = 0, uvStride = 0;
        makeNV12(W, H, src, yStride, uvStride);
        std::vector<uint8_t> out0((size_t)W * H * 3);
        runConvertU8(src.data(), W, H, 0, StreamFormat::YUV_NV12, out0.data(), W, H, 0, StreamFormat::BGR);
        bool any = false; for (auto v : out0) if (v) { any = true; break; } REQUIRE(any);
    }

    SECTION("I420 odd width/height: stride 0 works") {
        const int W = 3, H = 3;
        std::vector<uint8_t> src;
        int yStride = 0, uStride = 0, vStride = 0;
        makeI420(W, H, src, yStride, uStride, vStride);
        std::vector<uint8_t> out0((size_t)W * H * 3);
        runConvertU8(src.data(), W, H, 0, StreamFormat::YUV_I420, out0.data(), W, H, 0, StreamFormat::BGR);
        bool any = false; for (auto v : out0) if (v) { any = true; break; } REQUIRE(any);
    }
}
