// Standalone warm-cache latency probe. Link identical probe code against the
// baseline and candidate archives; compare repeated, alternating invocations.
#include <inspirecv/backends/okcv/bitmap/bitmap.h>
#include <inspirecv/task/core/matrix.h>
#include <inspirecv/task/task.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
volatile float observed = 0;

template <typename Operation>
void Measure(const char* name, const Operation& operation) {
    for (int i = 0; i < 5; ++i) operation();
    auto batch = [&](int loops) {
        const auto start = Clock::now();
        for (int i = 0; i < loops; ++i) operation();
        return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
    };
    int calibration = 1;
    double duration;
    while ((duration = batch(calibration)) < 1000 && calibration < 1048576) calibration *= 2;
    const int loops = std::max(1, std::min(4194304, int(std::ceil(4000 * calibration / duration))));
    std::vector<double> samples;
    for (int i = 0; i < 11; ++i) samples.push_back(batch(loops) / loops);
    auto ordered = samples;
    std::sort(ordered.begin(), ordered.end());
    std::printf("%s,%.6f,%.6f,%d", name, ordered[5], ordered[10], loops);
    for (double value : samples) std::printf(",%.6f", value);
    std::putchar('\n');
}

template <typename Pixel>
void Images(const char* type, okcv::BorderMode border) {
    for (int channels : {1, 3}) {
        for (int size : {17, 128, 512, 1024}) {
            std::vector<Pixel> bytes(size_t(size) * size * channels);
            for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<Pixel>((i * 37 + i / 7) % 251);
            okcv::Bitmap<Pixel> source;
            source.Reset(size, size, channels, bytes.data());
            const float transforms[][6] = {
              {1, 0, 0, 0, 1, 0}, {1, 0, 5, 0, 1, 3},
              {1, 0, 0.5f, 0, 1, 0.25f}, {0.75f, 0, -0.5f, 0, 0.5f, -0.25f}};
            const char* names[] = {"identity", "translate", "fractional", "scale_border"};
            for (int index = 0; index < 4; ++index) {
                const auto* t = transforms[index];
                const okcv::TransformMatrix matrix({t[0], t[1], t[2], t[3], t[4], t[5]});
                char label[128];
                std::snprintf(label, sizeof(label), "%s%s/c%d/%d/%s",
                              border == okcv::BORDER_MODE_CONSTANT ? "constant/" : "",
                              type, channels, size, names[index]);
                Measure(label, [&] {
                    const auto output = source.AffineBilinearOptimized(
                      size, size, matrix, border, Pixel(0));
                    observed = output.Data()[size * channels / 2];
                });
            }
        }
    }
}

void Matrices() {
    using inspirecv::task::Matrix;
    using inspirecv::task::Point;
    for (bool perspective : {false, true}) {
        Matrix first, second, output;
        first.setAll(1.25f, -0.375f, 17.5f, 0.625f, 0.75f, -9.25f,
                     perspective ? 0.00125f : 0, perspective ? -0.005f : 0, 1);
        second.setAll(0.8f, 0.03f, 3.2f, -0.1f, 1.1f, -6.0f, 0, 0, 1);
        Measure(perspective ? "matrix/concat/projective" : "matrix/concat/affine", [&] {
            output.setConcat(first, second); observed = output.getScaleX();
        });
        Measure(perspective ? "matrix/inverse/projective" : "matrix/inverse/affine", [&] {
            if (!first.invert(&output)) std::abort();
            observed = output.getScaleX();
        });
        Point input[5] = {{1, 2}, {-5, 17}, {32, 51}, {63, 64}, {127, 129}}, mapped[5];
        Measure(perspective ? "matrix/map5/projective" : "matrix/map5/affine", [&] {
            first.mapPoints(mapped, input, 5); observed = mapped[4].fX;
        });
    }
}

void Tasks() {
    using namespace inspirecv::task;
    PipelineOptions options;
    options.input_format = PixelFormat::kBgr;
    options.output_format = PixelFormat::kBgr;
    options.sampling = SamplingMode::kNearest;
    options.border = BorderMode::kReplicate;
    options.mean = {{127.5f, 127.5f, 127.5f, 0}};
    options.scale = {{1.f / 128, 1.f / 128, 1.f / 128, 1}};
    Pipeline pipeline(options);
    if (pipeline.SetTransform(inspirecv::TransformMatrix::Identity()) != Status::kOk) std::abort();
    for (int size : {112, 160, 320, 640}) {
        const size_t count = size_t(size) * size * 3;
        std::vector<uint8_t> input_storage(count + 8192);
        std::vector<float> output_storage(count + 2048);
        const auto input_page = (reinterpret_cast<uintptr_t>(input_storage.data()) + 4095) & ~uintptr_t(4095);
        const auto output_page = (reinterpret_cast<uintptr_t>(output_storage.data()) + 4095) & ~uintptr_t(4095);
        auto* source = reinterpret_cast<uint8_t*>(input_page);
        for (size_t i = 0; i < count; ++i) source[i] = static_cast<uint8_t>((29 * i + 17) & 255);
        for (int offset : {0, 64, 256}) {
            RawImageView input;
            input.data = source;
            input.width = input.height = size;
            TensorBuffer tensor;
            tensor.data = reinterpret_cast<float*>(output_page + offset);
            tensor.width = tensor.height = size;
            tensor.channels = 3;
            tensor.element_type = ElementType::kFloat32;
            tensor.order = TensorOrder::kChw;
            char label[128];
            std::snprintf(label, sizeof(label), "task/nchw/%d/page_offset_%d", size, offset);
            Measure(label, [&] {
                if (pipeline.Run(input, tensor) != Status::kOk) std::abort();
                observed = static_cast<float*>(tensor.data)[count - 1];
            });
        }
    }
}

void UnaffectedImages() {
    for (int size : {128, 256, 512, 1024}) {
        const size_t count = size_t(size) * size * 3;
        std::vector<uint8_t> byte_storage(count + 8192);
        std::vector<float> float_storage(count + 2048);
        auto* bytes = reinterpret_cast<uint8_t*>(
          ((reinterpret_cast<uintptr_t>(byte_storage.data()) + 4095) & ~uintptr_t(4095)) + 64);
        auto* floats = reinterpret_cast<float*>(
          ((reinterpret_cast<uintptr_t>(float_storage.data()) + 4095) & ~uintptr_t(4095)) + 64);
        for (size_t i = 0; i < count; ++i) {
            bytes[i] = static_cast<uint8_t>((37 * i + i / 7) % 251);
            floats[i] = bytes[i];
        }
        const auto u8 = inspirecv::Image::Create(size, size, 3, bytes, false);
        const auto f32 = inspirecv::ImageT<float>::Create(size, size, 3, floats, false);
        const auto gray = inspirecv::ImageT<float>::Create(size, size, 1, floats, false);
        const inspirecv::TransformMatrix matrix(0.8660254f, -0.5f, size * 0.3f,
                                                0.5f, 0.8660254f, size * -0.1f);
        char label[128];
        std::snprintf(label, sizeof(label), "unaffected/f32/%d/threshold", size);
        Measure(label, [&] { const auto out = gray.Threshold(100, 255, 0); observed = out.Data()[size]; });
        std::snprintf(label, sizeof(label), "unaffected/u8/%d/mul", size);
        Measure(label, [&] { const auto out = u8.Mul(1.2); observed = out.Data()[size]; });
        std::snprintf(label, sizeof(label), "unaffected/u8/%d/add", size);
        Measure(label, [&] { const auto out = u8.Add(10); observed = out.Data()[size]; });
        std::snprintf(label, sizeof(label), "unaffected/f32/%d/rotate90", size);
        Measure(label, [&] { const auto out = f32.Rotate90(); observed = out.Data()[size]; });
        std::snprintf(label, sizeof(label), "unaffected/f32/%d/warp_rotate", size);
        Measure(label, [&] { const auto out = f32.WarpAffine(matrix, size, size); observed = out.Data()[size]; });
    }
}
}  // namespace

int main(int argc, char** argv) {
    const bool task_only = argc == 2 && std::strcmp(argv[1], "--task-only") == 0;
    const bool unaffected = argc == 2 && std::strcmp(argv[1], "--unaffected") == 0;
    if (argc > 2 || (argc == 2 && !task_only && !unaffected && std::strcmp(argv[1], "--constant-border") != 0)) return 2;
    const auto border = argc == 2 ? okcv::BORDER_MODE_CONSTANT : okcv::BORDER_MODE_REPLICATE;
    std::puts("case,median_us,p95_us,loops,sample_us_0,sample_us_1,sample_us_2,sample_us_3,sample_us_4,sample_us_5,sample_us_6,sample_us_7,sample_us_8,sample_us_9,sample_us_10");
    if (task_only) {
        Tasks();
        return 0;
    }
    if (unaffected) {
        UnaffectedImages();
        return 0;
    }
    Images<uint8_t>("u8", border);
    Images<float>("f32", border);
    Matrices();
}
