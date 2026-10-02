#include "../common/common.h"
#include <inspirecv/inspirecv.h>
#include <inspirecv/task/task.h>
#include "inspirecv/core/runtime/cpu_features.h"
#include "inspirecv/task/kernels/cpu/sampling_ops.h"
#include "inspirecv/task/kernels/cpu/color_ops.h"
#include "inspirecv/task/kernels/cpu/yuv_ops.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif

#if defined(_WIN32) || defined(__unix__) || defined(__APPLE__)
namespace {
// A byte canary detects stores but cannot detect loads. Place the exact buffer
// at an inaccessible page boundary so any vector overread/overwrite faults.
class GuardedBuffer {
public:
    explicit GuardedBuffer(size_t bytes) : bytes_(bytes) {
#if defined(_WIN32)
        SYSTEM_INFO info;
        GetSystemInfo(&info);
        const size_t page = info.dwPageSize;
#else
        const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
#endif
        const size_t usable = ((bytes + page - 1) / page) * page;
        allocation_bytes_ = usable + page;
#if defined(_WIN32)
        allocation_ = static_cast<uint8_t*>(VirtualAlloc(nullptr, allocation_bytes_,
            MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
        if (!allocation_) throw std::runtime_error("VirtualAlloc failed");
        DWORD previous = 0;
        if (!VirtualProtect(allocation_ + usable, page, PAGE_NOACCESS, &previous)) {
            VirtualFree(allocation_, 0, MEM_RELEASE);
            throw std::runtime_error("VirtualProtect failed");
        }
#else
        void* mapping = mmap(nullptr, allocation_bytes_, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapping == MAP_FAILED) throw std::runtime_error("mmap failed");
        allocation_ = static_cast<uint8_t*>(mapping);
        if (mprotect(allocation_ + usable, page, PROT_NONE) != 0) {
            munmap(allocation_, allocation_bytes_);
            throw std::runtime_error("mprotect failed");
        }
#endif
        data_ = allocation_ + usable - bytes;
        std::memset(data_, 0x53, bytes);
    }
    ~GuardedBuffer() {
#if defined(_WIN32)
        VirtualFree(allocation_, 0, MEM_RELEASE);
#else
        munmap(allocation_, allocation_bytes_);
#endif
    }
    GuardedBuffer(const GuardedBuffer&) = delete;
    GuardedBuffer& operator=(const GuardedBuffer&) = delete;
    uint8_t* data() { return data_; }
    size_t size() const { return bytes_; }
private:
    uint8_t* allocation_ = nullptr;
    uint8_t* data_ = nullptr;
    size_t allocation_bytes_ = 0, bytes_ = 0;
};
std::vector<uint32_t> GuardMasks() {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    return {0, inspirecv::cpu::kSsse3 | inspirecv::cpu::kSse41,
            inspirecv::cpu::kAllCpuFeatures};
#else
    return {inspirecv::cpu::kAllCpuFeatures};
#endif
}
}  // namespace

TEST_CASE("simd_packed_rows_do_not_access_beyond_exact_buffers",
          "[simd-dispatch][guard-page][task]") {
    using namespace inspirecv::task;
    const PixelFormat formats[] = {PixelFormat::kGray, PixelFormat::kRgb, PixelFormat::kRgba};
    kernels::sampling::Sampler* nearest[] = {kernels::sampling::NearestMono,
        kernels::sampling::NearestTriple, kernels::sampling::NearestQuad};
    kernels::sampling::Sampler* linear[] = {kernels::sampling::BilinearMono,
        kernels::sampling::BilinearTriple, kernels::sampling::BilinearQuad};
    for (uint32_t mask : GuardMasks()) {
        const inspirecv::cpu::ScopedCpuFeatureMask features(mask);
        for (int f = 0; f < 3; ++f) for (int width : {1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 257}) {
            const int channels = f == 0 ? 1 : f + 2;
            const size_t bytes = size_t(width) * channels;
            GuardedBuffer input(bytes), output(bytes), floats(bytes * sizeof(float));
            for (size_t i = 0; i < bytes; ++i) input.data()[i] = uint8_t(i * 71 + 17);
            const std::vector<uint8_t> before(input.data(), input.data() + bytes);
            CAPTURE(mask, channels, width);
            // Clamp every output to the last source pixel. A gather at C1/C3
            // would cross the protected page unless the implementation checks
            // its full load width, not just the logical pixel coordinate.
            for (auto sampler : {nearest[f], linear[f]}) {
                Point line[2] = {{float(width) - .75f, 0.f}, {.25f, 0.f}};
                sampler(input.data(), output.data(), line, 0, width, width, width, 1, int(bytes));
                for (int x = 0; x < width; ++x) for (int c = 0; c < channels; ++c)
                    REQUIRE(output.data()[x * channels + c] == input.data()[(width - 1) * channels + c]);
            }
            PipelineOptions options;
            options.input_format = formats[f];
            options.output_format = formats[f];
            options.backend_preference = BackendPreference::kCpu;
            Pipeline pipeline(options);
            TensorBuffer destination;
            destination.data = floats.data();
            destination.width = width; destination.height = 1; destination.channels = channels;
            destination.element_type = ElementType::kFloat32;
            const RawImageView source{input.data(), 0, width, 1};
            REQUIRE(pipeline.Run(source, destination) == Status::kOk);
            for (size_t i = 0; i < bytes; ++i)
                REQUIRE(reinterpret_cast<float*>(floats.data())[i] == float(input.data()[i]));
            REQUIRE(std::equal(before.begin(), before.end(), input.data()));
        }
    }
}

TEST_CASE("simd_color_and_yuv_tails_do_not_cross_guard_pages",
          "[simd-dispatch][guard-page][task]") {
    using namespace inspirecv::task;
    for (uint32_t mask : GuardMasks()) {
        const inspirecv::cpu::ScopedCpuFeatureMask features(mask);
        for (size_t count : {size_t(1), size_t(7), size_t(8), size_t(9), size_t(15), size_t(16), size_t(17), size_t(31), size_t(32), size_t(33), size_t(257)}) {
            CAPTURE(mask, count);
            GuardedBuffer rgb(3 * count), color(3 * count), packed(2 * count);
            for (auto convert : {kernels::color::RgbToYCrCb, kernels::color::RgbToYuv,
                                 kernels::color::RgbToXyz, kernels::color::RgbToHsv,
                                 kernels::color::RgbToHsvFull})
                convert(rgb.data(), color.data(), count);
            kernels::color::RgbToBgr555(rgb.data(), packed.data(), count);
            kernels::color::RgbToBgr565(rgb.data(), packed.data(), count);
            GuardedBuffer yuv(count + 2 * ((count + 1) / 2)), rgba(4 * count);
            kernels::yuv::ToRgb(yuv.data(), color.data(), count);
            kernels::yuv::ToRgba(yuv.data(), rgba.data(), count);
            SUCCEED("all loads/stores stayed inside mapped buffers");
        }
    }
}

#if !defined(INSPIRECV_BACKEND_OPENCV)
TEST_CASE("image_simd_reads_only_exact_external_storage",
          "[simd-dispatch][guard-page][image]") {
    for (uint32_t mask : GuardMasks()) {
        const inspirecv::cpu::ScopedCpuFeatureMask features(mask);
        for (int channels : {1, 3, 4}) for (int width : {1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65}) {
            GuardedBuffer storage(size_t(width) * channels);
            const auto input = inspirecv::Image::Create(width, 1, channels, storage.data(), false);
            CAPTURE(mask, channels, width);
            REQUIRE(input.Rotate90().Width() == 1);
            REQUIRE(input.Rotate270().Width() == 1);
            REQUIRE(input.Rotate180().Width() == width);
            REQUIRE(input.FlipHorizontal().Width() == width);
            REQUIRE(input.Resize(width + 9, 5, true).Width() == width + 9);
            if (channels == 1) REQUIRE(input.Threshold(80, 255, 0).Width() == width);
            REQUIRE(input.AbsDiff(input).Width() == width);
            if (channels == 3) {
                REQUIRE(input.ToGray().Width() == width);
                REQUIRE(input.MeanChannels().Width() == width);
            }
        }
    }
}

TEST_CASE("image_float_arithmetic_and_geometry_preserve_exact_external_storage",
          "[simd-dispatch][guard-page][image][float]") {
    // The final float touches an inaccessible page. This catches full-vector
    // loads beyond an arithmetic tail and the last C4 pixel in a rotation.
    const uint32_t patterns[] = {
        0x00000000u, 0x80000000u, 0x3f000000u, 0xbf000000u,
        0x3f800000u, 0xbf800000u, 0x7f800000u, 0xff800000u,
        0x7fc12345u, 0xffc54321u, 0x00000001u, 0x80000001u,
        0x00800000u, 0x80800000u, 0x42ff8000u, 0xc37f8000u
    };
    for (uint32_t mask : GuardMasks()) {
        const inspirecv::cpu::ScopedCpuFeatureMask features(mask);
        for (int channels : {1, 3, 4})
        for (int width : {1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 65})
        for (int height : {1, 3, 8, 9}) {
            const size_t count = size_t(width) * height * channels;
            GuardedBuffer storage(count * sizeof(float));
            float* data = reinterpret_cast<float*>(storage.data());
            for (size_t i = 0; i < count; ++i) {
                const uint32_t bits = patterns[(i * 7) % 16];
                std::memcpy(data + i, &bits, sizeof(bits));
            }
            const std::vector<uint8_t> before(storage.data(), storage.data() + storage.size());
            const auto input = inspirecv::ImageT<float>::Create(width, height, channels, data, false);
            CAPTURE(mask, channels, width, height);
            for (float scalar : {-.5f, 0.f, 1.f, 2.f, 7.5f}) {
                const auto added = input.Add(scalar);
                const auto multiplied = input.Mul(scalar);
                CAPTURE(scalar);
                REQUIRE(added.Data() != data);
                REQUIRE(multiplied.Data() != data);
                for (size_t i = 0; i < count; ++i) {
                    // A binary64 product of binary32 operands is exact before
                    // rounding to float; no production arithmetic is reused.
                    const float sum = float(double(data[i]) + double(scalar));
                    const float product = float(double(data[i]) * double(scalar));
                    REQUIRE((std::memcmp(added.Data() + i, &sum, sizeof(float)) == 0 ||
                             (std::isnan(added.Data()[i]) && std::isnan(sum))));
                    REQUIRE((std::memcmp(multiplied.Data() + i, &product, sizeof(float)) == 0 ||
                             (std::isnan(multiplied.Data()[i]) && std::isnan(product))));
                }
            }
            const auto rotated = input.Rotate90();
            REQUIRE(rotated.Width() == height);
            REQUIRE(rotated.Height() == width);
            REQUIRE(rotated.Channels() == channels);
            REQUIRE(rotated.Data() != data);
            for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) {
                const size_t source_pixel = (size_t(y) * width + x) * channels;
                const size_t destination_pixel = (size_t(x) * height + height - 1 - y) * channels;
                REQUIRE(std::memcmp(rotated.Data() + destination_pixel, data + source_pixel,
                                    size_t(channels) * sizeof(float)) == 0);
            }
            for (int output_width : {std::max(1, width / 2), width + 9}) {
                const int output_height = height + 2;
                const auto resized = input.Resize(output_width, output_height, false);
                REQUIRE(resized.Width() == output_width);
                REQUIRE(resized.Height() == output_height);
                REQUIRE(resized.Channels() == channels);
                REQUIRE(resized.Data() != data);
                const float scale_x = float(width) / float(output_width);
                const float scale_y = float(height) / float(output_height);
                for (int y = 0; y < output_height; ++y) for (int x = 0; x < output_width; ++x) {
                    // Nearest uses truncated binary32 coordinates. Volatile
                    // intermediates keep that rounding explicit in the oracle.
                    volatile float position_x = float(x) * scale_x;
                    volatile float position_y = float(y) * scale_y;
                    const int sx = std::min(int(position_x), width - 1);
                    const int sy = std::min(int(position_y), height - 1);
                    const size_t source_pixel = (size_t(sy) * width + sx) * channels;
                    const size_t destination_pixel = (size_t(y) * output_width + x) * channels;
                    REQUIRE(std::memcmp(resized.Data() + destination_pixel, data + source_pixel,
                                        size_t(channels) * sizeof(float)) == 0);
                }
            }
            REQUIRE(std::memcmp(storage.data(), before.data(), before.size()) == 0);
        }
    }
}
#endif
#endif
