// One public Image operation per process; no test registration or image IO.
// Compile this object once, then link it against each library revision.
#include <inspirecv/core/image.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
volatile double observed = 0;

struct Case {
    std::string type, operation;
    int size, output;
    std::string Id() const {
        return type + "/" + operation + "/" + std::to_string(size) + "x" +
               std::to_string(size) + "/" + std::to_string(output) + "x" + std::to_string(output);
    }
};

std::vector<Case> Cases() {
    std::vector<Case> result;
    for (const std::string type : {"u8", "f32"}) {
        for (int n : {128, 256, 512, 1024}) {
            auto add = [&](const std::string& op, int output) { result.push_back({type, op, n, output}); };
            for (const std::string op : {"absdiff", "dilate3", "draw_circle", "draw_line",
                  "draw_rect", "erode3", "fill_rect", "gauss5", "mean_channels", "rotate180",
                  "rotate270", "rotate90", "swap_rb", "to_gray"}) add(op + "_" + type, n);
            add("add_" + type + "_10", n);
            add("mul_" + type + "_1p2", n);
            add("blend_" + type + "_gradmask", n);
            add("crop_" + type + "_center", n / 2);
            add("pad_" + type + "_10_all", n + 20);
            add("threshold_" + type + "_100_255", n);
            for (const std::string mode : {"linear", "nearest"}) {
                add("resize_" + type + "_" + mode, n / 2);
                add("resize_" + type + "_" + mode, n * 2);
            }
            add("resize_" + type + "_linear_1p5x", n * 3 / 2);
            for (const std::string transform : {"identity", "rot30", "scale05", "translate"})
                add("warp_affine_" + type + "_" + transform, n);
            if (type == "u8") {
                for (const std::string op : {"flip_h_u8", "flip_v_u8", "threshold_u8_200_255",
                     "warp_affine_u8_combo", "warp_affine_u8_scale_nu",
                     "warp_affine_u8_shear_kx02", "warp_affine_u8_shear_ky02"}) add(op, n);
                add("pad_u8_red_10_all", n + 20);
                add("pad_u8_white_10_all", n + 20);
            }
        }
    }
    std::sort(result.begin(), result.end(), [](const Case& a, const Case& b) { return a.Id() < b.Id(); });
    return result;
}

template <typename Pixel>
struct Buffer {
    std::vector<Pixel> storage;
    Pixel* data;
    explicit Buffer(size_t count) : storage(count + 8192 / sizeof(Pixel)) {
        // Fix relative input page offset across independent processes/revisions.
        data = reinterpret_cast<Pixel*>(
          ((reinterpret_cast<uintptr_t>(storage.data()) + 4095) & ~uintptr_t(4095)) + 64);
    }
};

template <typename Pixel>
uint64_t Hash(const inspirecv::ImageT<Pixel>& output) {
    if (output.Empty()) throw std::runtime_error("empty output");
    uint64_t hash = UINT64_C(14695981039346656037);
    const auto* bytes = reinterpret_cast<const uint8_t*>(output.Data());
    const size_t count = size_t(output.Width()) * output.Height() * output.Channels() * sizeof(Pixel);
    for (size_t i = 0; i < count; ++i) hash = (hash ^ bytes[i]) * UINT64_C(1099511628211);
    return hash;
}

inspirecv::TransformMatrix Matrix(const std::string& op, int n) {
    const float center = (n - 1) * 0.5f;
    if (op.find("shear_") != std::string::npos) {
        const float kx = op.find("kx02") != std::string::npos ? 0.2f : 0;
        const float ky = op.find("ky02") != std::string::npos ? 0.2f : 0;
        return inspirecv::TransformMatrix(1, kx, -kx * center, ky, 1, -ky * center);
    }
    float angle = 0, sx = 1, sy = 1, tx = 0, ty = 0;
    if (op.find("rot30") != std::string::npos) angle = 30;
    if (op.find("scale05") != std::string::npos) sx = sy = 0.5f;
    if (op.find("scale_nu") != std::string::npos) { sx = 0.75f; sy = 1.25f; }
    if (op.find("translate") != std::string::npos) { tx = n * 0.1f; ty = -n * 0.1f; }
    if (op.find("combo") != std::string::npos) { angle = 30; sx = sy = 0.8f; tx = ty = n * 0.05f; }
    const float rad = angle * 0.017453292519943295f;
    const float a11 = std::cos(rad) * sx, a12 = std::sin(rad) * sy;
    const float a21 = -std::sin(rad) * sx, a22 = std::cos(rad) * sy;
    return inspirecv::TransformMatrix(a11, a12, center - a11 * center - a12 * center + tx,
                                     a21, a22, center - a21 * center - a22 * center + ty);
}

template <typename Pixel>
void Run(const Case& test, double batch_us, int samples) {
    using Image = inspirecv::ImageT<Pixel>;
    using inspirecv::Point;
    using inspirecv::Rect;
    const int n = test.size;
    const auto& op = test.operation;
    const int channels = op.find("threshold_") == 0 || op.find("erode3_") == 0 ||
                         op.find("dilate3_") == 0 ? 1 : 3;
    const size_t count = size_t(n) * n * channels;
    Buffer<Pixel> input(count), other(count);
    Buffer<uint8_t> mask_bytes(size_t(n) * n);
    for (size_t i = 0; i < count; ++i) input.data[i] = Pixel((i * 37 + i / 7) % 251);
    for (int y = 0; y < n; ++y) for (int x = 0; x < n; ++x) {
        const size_t index = size_t(y) * n + x;
        mask_bytes.data[index] = uint8_t(255.f * 0.5f * (float(x) / (n - 1) + float(y) / (n - 1)));
        for (int c = 0; c < channels; ++c) {
            other.data[index * channels + c] = op.find("blend_") == 0
              ? input.data[index * channels + (channels - 1 - c)]
              : input.data[(size_t(n) * n - 1 - index) * channels + c];
        }
    }
    auto source = Image::Create(n, n, channels, input.data, false);
    const auto second = Image::Create(n, n, channels, other.data, false);
    const auto mask = inspirecv::Image::Create(n, n, 1, mask_bytes.data, false);
    const auto matrix = Matrix(op, n);
    const std::vector<double> red = {0, 0, 255}, green = {0, 255, 0}, blue = {255, 0, 0};
    const std::vector<double> white = {255, 255, 255}, black = {0, 0, 0};
    std::function<Image()> allocating;
    std::function<void()> drawing;
    if (op.find("resize_") == 0) allocating = [&] { return source.Resize(test.output, test.output, op.find("nearest") == std::string::npos); };
    else if (op.find("warp_affine_") == 0) allocating = [&] { return source.WarpAffine(matrix, n, n); };
    else if (op.find("crop_") == 0) allocating = [&] { return source.Crop(Rect<int>(n / 4, n / 4, n / 2, n / 2)); };
    else if (op.find("pad_") == 0) allocating = [&] { return source.Pad(10, 10, 10, 10, op.find("red") != std::string::npos ? red : op.find("white") != std::string::npos ? white : black); };
    else if (op.find("threshold_") == 0) allocating = [&] { return source.Threshold(op.find("200") != std::string::npos ? 200 : 100, 255, 0); };
    else if (op.find("absdiff_") == 0) allocating = [&] { return source.AbsDiff(second); };
    else if (op.find("blend_") == 0) allocating = [&] { return source.Blend(second, mask); };
    else if (op.find("add_") == 0) allocating = [&] { return source.Add(10); };
    else if (op.find("mul_") == 0) allocating = [&] { return source.Mul(1.2); };
    else if (op.find("gauss5_") == 0) allocating = [&] { return source.GaussianBlur(5, 0); };
    else if (op.find("erode3_") == 0) allocating = [&] { return source.Erode(3, 1); };
    else if (op.find("dilate3_") == 0) allocating = [&] { return source.Dilate(3, 1); };
    else if (op.find("mean_channels_") == 0) allocating = [&] { return source.MeanChannels(); };
    else if (op.find("to_gray_") == 0) allocating = [&] { return source.ToGray(); };
    else if (op.find("swap_rb_") == 0) allocating = [&] { return source.SwapRB(); };
    else if (op.find("flip_h_") == 0) allocating = [&] { return source.FlipHorizontal(); };
    else if (op.find("flip_v_") == 0) allocating = [&] { return source.FlipVertical(); };
    else if (op.find("rotate90_") == 0) allocating = [&] { return source.Rotate90(); };
    else if (op.find("rotate180_") == 0) allocating = [&] { return source.Rotate180(); };
    else if (op.find("rotate270_") == 0) allocating = [&] { return source.Rotate270(); };
    else if (op.find("draw_line_") == 0) drawing = [&] { source.DrawLine(Point<int>(0, 0), Point<int>(n - 1, n - 1), red, 1); };
    else if (op.find("draw_rect_") == 0) drawing = [&] { source.DrawRect(Rect<int>(n / 4, n / 4, n / 2, n / 2), green, 1); };
    else if (op.find("draw_circle_") == 0) drawing = [&] { source.DrawCircle(Point<int>(n / 2, n / 2), std::max(1, n / 6), blue, 1); };
    else if (op.find("fill_rect_") == 0) drawing = [&] { source.Fill(Rect<int>(n / 3, n / 3, n / 3, n / 3), white); };
    else throw std::runtime_error("unimplemented case: " + test.Id());

    auto operation = [&] {
        if (drawing) drawing();
        else { const auto output = allocating(); observed = output.Data()[0]; }
    };
    for (int i = 0; i < 5; ++i) operation();
    auto batch = [&](int loops) {
        const auto start = Clock::now();
        for (int i = 0; i < loops; ++i) operation();
        return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
    };
    int loops = 1;
    while (batch(loops) < batch_us && loops < 4194304) loops *= 2;
    std::vector<double> values;
    for (int i = 0; i < samples; ++i) values.push_back(batch(loops) / loops);
    auto sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const auto output = drawing ? source.Clone() : allocating();
    if (output.Width() != test.output || output.Height() != test.output)
        throw std::runtime_error("unexpected output dimensions");
    std::printf("{\"case\":\"%s\",\"channels\":%d,\"output_channels\":%d,\"loops\":%d,"
                "\"median_us\":%.9f,\"p95_batch_mean_us\":%.9f,\"hash\":\"%016llx\",\"samples_us\":[",
                test.Id().c_str(), channels, output.Channels(), loops, sorted[samples / 2],
                sorted[size_t(std::ceil(samples * .95)) - 1], static_cast<unsigned long long>(Hash(output)));
    for (int i = 0; i < samples; ++i) std::printf("%s%.9f", i ? "," : "", values[i]);
    std::puts("]}");
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const auto cases = Cases();
        if (argc == 2 && std::string(argv[1]) == "--list") {
            for (const auto& test : cases) std::puts(test.Id().c_str());
            return 0;
        }
        if (argc < 3 || std::string(argv[1]) != "--case" || argc > 5)
            throw std::runtime_error("usage: image_latency_probe --list | --case ID [batch_us=5000] [samples=9]");
        const double batch_us = argc >= 4 ? std::stod(argv[3]) : 5000;
        const int samples = argc >= 5 ? std::stoi(argv[4]) : 9;
        if (!std::isfinite(batch_us) || batch_us < 100 || batch_us > 1000000 ||
            samples < 3 || samples > 101 || samples % 2 == 0) throw std::runtime_error("invalid timing parameters");
        for (const auto& test : cases) if (test.Id() == argv[2]) {
            if (test.type == "u8") Run<uint8_t>(test, batch_us, samples);
            else Run<float>(test, batch_us, samples);
            return 0;
        }
        throw std::runtime_error("unknown case");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 2;
    }
}
