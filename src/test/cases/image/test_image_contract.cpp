#include "../../common/common.h"

#include <inspirecv/inspirecv.h>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace {

template <typename Pixel>
void RequirePixels(const inspirecv::ImageT<Pixel>& image,
                   const std::vector<Pixel>& expected) {
    const std::size_t count = static_cast<std::size_t>(image.Width()) *
                              image.Height() * image.Channels();
    REQUIRE(count == expected.size());
    REQUIRE_EQ_C_ARRAY(image.Data(), expected.data(), count);
}

}  // namespace

TEST_CASE("image_external_storage_contract_u8", "[image][contract][storage]") {
    using inspirecv::Image;

    std::vector<uint8_t> external = {1, 2, 3, 4};

    SECTION("factory honors zero-copy and observes caller updates") {
        auto image = Image::Create(2, 2, 1, external.data(), false);
        REQUIRE(image.Data() == external.data());
        external[2] = 91;
        REQUIRE(image.Data()[2] == 91);
    }

    SECTION("moving a zero-copy image preserves its view") {
        auto source = Image::Create(2, 2, 1, external.data(), false);
        Image destination = std::move(source);
        REQUIRE(destination.Width() == 2);
        REQUIRE(destination.Height() == 2);
        REQUIRE(destination.Channels() == 1);
        REQUIRE(destination.Data() == external.data());
        RequirePixels(destination, external);
    }

    SECTION("arithmetic reads zero-copy storage") {
        auto image = Image::Create(2, 2, 1, external.data(), false);
        RequirePixels(image.Mul(2.0), std::vector<uint8_t>{2, 4, 6, 8});
        RequirePixels(image.Add(5.0), std::vector<uint8_t>{6, 7, 8, 9});
    }

    SECTION("same-size reset transitions from external to owned storage") {
        auto image = Image::Create(2, 2, 1, external.data(), false);
        const std::vector<uint8_t> replacement = {11, 12, 13, 14};
        image.Reset(2, 2, 1, replacement.data());
        REQUIRE(image.Data() != external.data());
        RequirePixels(image, replacement);
        external[0] = 99;
        REQUIRE(image.Data()[0] == 11);
    }
}

TEST_CASE("image_multichannel_pad_contract_u8", "[image][contract][pad]") {
    using inspirecv::Image;

    const std::vector<uint8_t> source = {10, 20, 30};
    auto image = Image::Create(1, 1, 3, source.data());

    SECTION("per-channel border color is preserved") {
        auto padded = image.Pad(1, 1, 1, 1, {1.0, 2.0, 3.0});
        REQUIRE(padded.Width() == 3);
        REQUIRE(padded.Height() == 3);
        REQUIRE(padded.Channels() == 3);
        for (int y = 0; y < padded.Height(); ++y) {
            for (int x = 0; x < padded.Width(); ++x) {
                const uint8_t* pixel = padded.Data() + (y * padded.Width() + x) * 3;
                if (x == 1 && y == 1) {
                    REQUIRE(pixel[0] == 10);
                    REQUIRE(pixel[1] == 20);
                    REQUIRE(pixel[2] == 30);
                } else {
                    REQUIRE(pixel[0] == 1);
                    REQUIRE(pixel[1] == 2);
                    REQUIRE(pixel[2] == 3);
                }
            }
        }
    }

    SECTION("single border value broadcasts to every channel") {
        auto padded = image.Pad(1, 0, 0, 0, {7.0});
        REQUIRE(padded.Data()[0] == 7);
        REQUIRE(padded.Data()[1] == 7);
        REQUIRE(padded.Data()[2] == 7);
    }

    SECTION("empty border color uses black") {
        auto padded = image.Pad(1, 0, 0, 0, {});
        REQUIRE(padded.Data()[0] == 0);
        REQUIRE(padded.Data()[1] == 0);
        REQUIRE(padded.Data()[2] == 0);
    }
}

TEST_CASE("image_multichannel_pad_contract_f32",
          "[image][contract][pad][float]") {
    const std::vector<float> source = {
      10.0f, 20.0f, 30.0f, 40.0f,
      50.0f, 60.0f, 70.0f, 80.0f,
    };
    const std::vector<float> border = {0.25f, 1.5f, -2.0f, 9.0f};
    const auto image =
      inspirecv::ImageT<float>::Create(2, 1, 4, source.data());

    SECTION("asymmetric four-channel padding preserves channel values") {
        const auto padded = image.Pad(0, 1, 2, 0, {0.25, 1.5, -2.0, 9.0});
        REQUIRE(padded.Width() == 4);
        REQUIRE(padded.Height() == 2);
        REQUIRE(padded.Channels() == 4);
        for (int pixel = 0; pixel < 2; ++pixel) {
            REQUIRE_EQ_C_ARRAY(padded.Data() + pixel * 4, border.data(), 4);
        }
        REQUIRE_EQ_C_ARRAY(padded.Data() + 8, source.data(), source.size());
        for (int pixel = 0; pixel < 4; ++pixel) {
            REQUIRE_EQ_C_ARRAY(padded.Data() + (4 + pixel) * 4,
                               border.data(), 4);
        }
    }

    SECTION("zero-width padding is an exact copy") {
        const auto copied = image.Pad(0, 0, 0, 0, {0.25, 1.5, -2.0, 9.0});
        REQUIRE_EQ_C_ARRAY(copied.Data(), source.data(), source.size());
    }
}

TEST_CASE("image_circle_contract_u8", "[image][contract][drawing]") {
    using inspirecv::Image;
    using inspirecv::Point2i;

    auto canvas = Image::Create(15, 15, 1);
    canvas.Fill(0.0);
    canvas.DrawCircle(Point2i(7, 7), 4, {255.0}, 1);

    const auto at = [&canvas](int x, int y) {
        return canvas.Data()[y * canvas.Width() + x];
    };
    REQUIRE(at(11, 7) == 255);
    REQUIRE(at(3, 7) == 255);
    REQUIRE(at(7, 11) == 255);
    REQUIRE(at(7, 3) == 255);
    REQUIRE(at(7, 7) == 0);
    REQUIRE(at(12, 7) == 0);
}

TEST_CASE("image_float_blend_mask_boundary_contract",
          "[image][contract][blend][storage]") {
    const int widths[] = {1, 7, 8, 9, 15, 16, 17};
    const int channel_counts[] = {1, 3, 4};
    for (int width : widths) {
        for (int channels : channel_counts) {
            const size_t element_count =
              static_cast<size_t>(width) * channels;
            std::vector<float> foreground(element_count);
            std::vector<float> background(element_count);
            std::vector<uint8_t> mask(static_cast<size_t>(width));
            for (size_t index = 0; index < element_count; ++index) {
                foreground[index] = static_cast<float>(index * 1.25 + 3.0);
                background[index] = static_cast<float>(200.0 - index * 0.75);
            }
            for (int x = 0; x < width; ++x) {
                mask[static_cast<size_t>(x)] =
                  static_cast<uint8_t>((x * 61 + 13) & 255);
            }

            const auto first = inspirecv::ImageT<float>::Create(
              width, 1, channels, foreground.data());
            const auto second = inspirecv::ImageT<float>::Create(
              width, 1, channels, background.data());
            const auto mask_image = inspirecv::Image::Create(
              width, 1, 1, mask.data(), false);
            const auto blended = first.Blend(second, mask_image);

            REQUIRE(blended.Width() == width);
            REQUIRE(blended.Channels() == channels);
            for (int x = 0; x < width; ++x) {
                const float weight = mask[static_cast<size_t>(x)] / 255.0f;
                for (int channel = 0; channel < channels; ++channel) {
                    const size_t index =
                      static_cast<size_t>(x) * channels + channel;
                    const float expected =
                      weight * foreground[index] +
                      (1.0f - weight) * background[index];
                    REQUIRE(blended.Data()[index] ==
                            Approx(expected).margin(2e-5f));
                }
            }
        }
    }
}

TEST_CASE("image_nearest_2x_vector_and_tail_contract",
          "[image][contract][resize][vector]") {
    constexpr int kSourceWidth = 17;
    constexpr int kSourceHeight = 2;
    constexpr int kOutputWidth = 34;
    constexpr int kOutputHeight = 4;

    for (int channels : {1, 3}) {
        std::vector<uint8_t> source(
          static_cast<size_t>(kSourceWidth) * kSourceHeight * channels);
        for (size_t index = 0; index < source.size(); ++index) {
            source[index] = static_cast<uint8_t>((index * 29 + 7) & 255);
        }
        const auto image = inspirecv::Image::Create(
          kSourceWidth, kSourceHeight, channels, source.data());
        const auto resized = image.Resize(kOutputWidth, kOutputHeight, false);
        for (int y = 0; y < kOutputHeight; ++y) {
            for (int x = 0; x < kOutputWidth; ++x) {
                for (int channel = 0; channel < channels; ++channel) {
                    const size_t source_index =
                      (static_cast<size_t>(y / 2) * kSourceWidth + x / 2) *
                        channels +
                      channel;
                    const size_t output_index =
                      (static_cast<size_t>(y) * kOutputWidth + x) * channels +
                      channel;
                    REQUIRE(resized.Data()[output_index] == source[source_index]);
                }
            }
        }
    }

    std::vector<float> source(
      static_cast<size_t>(kSourceWidth) * kSourceHeight);
    for (size_t index = 0; index < source.size(); ++index) {
        source[index] = static_cast<float>(index) * 0.75f - 3.25f;
    }
    const auto image = inspirecv::ImageT<float>::Create(
      kSourceWidth, kSourceHeight, 1, source.data());
    const auto resized = image.Resize(kOutputWidth, kOutputHeight, false);
    for (int y = 0; y < kOutputHeight; ++y) {
        for (int x = 0; x < kOutputWidth; ++x) {
            REQUIRE(resized.Data()[y * kOutputWidth + x] ==
                    source[(y / 2) * kSourceWidth + x / 2]);
        }
    }
}

TEST_CASE("image_u8c3_reorder_vector_boundaries_match_reference",
          "[image][contract][geometry][color][vector]") {
    const int widths[] = {
      1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65,
    };
    const int heights[] = {1, 2, 3, 17, 31, 32, 33, 65};
    const int offsets[] = {0, 1, 7, 15};

    for (int width : widths) {
        for (int height : heights) {
            const std::size_t element_count =
              static_cast<std::size_t>(width) * height * 3;
            for (int offset : offsets) {
                std::vector<uint8_t> storage(
                  element_count + static_cast<std::size_t>(offset) + 16,
                  0xA5);
                uint8_t* source = storage.data() + offset;
                for (std::size_t index = 0; index < element_count; ++index) {
                    source[index] = static_cast<uint8_t>(
                      (index * 73 + width * 17 + height * 29) & 255);
                }
                const std::vector<uint8_t> original(
                  source, source + element_count);
                const auto image = inspirecv::Image::Create(
                  width, height, 3, source, false);

                const auto swapped = image.SwapRB();
                REQUIRE(swapped.Width() == width);
                REQUIRE(swapped.Height() == height);
                REQUIRE(swapped.Channels() == 3);
                for (std::size_t pixel = 0;
                     pixel < element_count / 3; ++pixel) {
                    REQUIRE(swapped.Data()[pixel * 3] ==
                            original[pixel * 3 + 2]);
                    REQUIRE(swapped.Data()[pixel * 3 + 1] ==
                            original[pixel * 3 + 1]);
                    REQUIRE(swapped.Data()[pixel * 3 + 2] ==
                            original[pixel * 3]);
                }

                const auto flipped = image.FlipHorizontal();
                REQUIRE(flipped.Width() == width);
                REQUIRE(flipped.Height() == height);
                REQUIRE(flipped.Channels() == 3);
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        const std::size_t source_index =
                          (static_cast<std::size_t>(y) * width +
                           (width - 1 - x)) * 3;
                        const std::size_t destination_index =
                          (static_cast<std::size_t>(y) * width + x) * 3;
                        REQUIRE_EQ_C_ARRAY(
                          flipped.Data() + destination_index,
                          original.data() + source_index, 3);
                    }
                }

                const auto rotated = image.Rotate90();
                REQUIRE(rotated.Width() == height);
                REQUIRE(rotated.Height() == width);
                REQUIRE(rotated.Channels() == 3);
                for (int y = 0; y < width; ++y) {
                    for (int x = 0; x < height; ++x) {
                        const std::size_t source_index =
                          (static_cast<std::size_t>(height - 1 - x) * width +
                           y) * 3;
                        const std::size_t destination_index =
                          (static_cast<std::size_t>(y) * height + x) * 3;
                        REQUIRE_EQ_C_ARRAY(
                          rotated.Data() + destination_index,
                          original.data() + source_index, 3);
                    }
                }

                REQUIRE_EQ_C_ARRAY(source, original.data(), element_count);
                for (int index = 0; index < offset; ++index) {
                    REQUIRE(storage[static_cast<std::size_t>(index)] == 0xA5);
                }
                for (std::size_t index =
                       static_cast<std::size_t>(offset) + element_count;
                     index < storage.size(); ++index) {
                    REQUIRE(storage[index] == 0xA5);
                }
            }
        }
    }
}
