// Link this probe against two builds and compare stdout byte-for-byte. It uses
// runtime-generated inputs so constant folding in the probe cannot mask a
// change to matrix arithmetic or scanline sampling in the library.
#include <inspirecv/inspirecv.h>
#include <inspirecv/task/task.h>
#include <inspirecv/task/core/matrix.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
uint32_t state = 0x517cc1b7u;
uint32_t Next() {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}
float Value() { return (static_cast<int>(Next() % 4097) - 2048) / 997.0f; }
void PrintFloats(const float* values, int count) {
    for (int i = 0; i < count; ++i) {
        uint32_t bits;
        std::memcpy(&bits, values + i, sizeof(bits));
        std::printf(" %08x", bits);
    }
    std::putchar('\n');
}
uint64_t Hash(const uint8_t* bytes, size_t size) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}
}  // namespace

int main(int argc, char** argv) {
    const bool full_pixels = argc == 2 && std::strcmp(argv[1], "--full-pixels") == 0;
    if (argc > 2 || (argc == 2 && !full_pixels)) return 1;
    using inspirecv::task::Matrix;
    using inspirecv::task::Point;
    for (int index = 0; index < 256; ++index) {
        float a[9], b[9], values[9];
        for (int j = 0; j < 9; ++j) { a[j] = Value(); b[j] = Value(); }
        a[0] += 3; a[4] += 3; b[0] += 3; b[4] += 3;
        a[6] *= 0.001f; a[7] *= 0.001f; a[8] = 1;
        b[6] *= 0.001f; b[7] *= 0.001f; b[8] = 1;
        if (index % 2 == 0) { a[6] = a[7] = b[6] = b[7] = 0; }
        Matrix first, second, composed, inverse;
        first.set9(a); second.set9(b); composed.setConcat(first, second);
        composed.get9(values);
        std::printf("compose/%d", index); PrintFloats(values, 9);
        if (!composed.invert(&inverse)) return 2;
        inverse.get9(values);
        std::printf("inverse/%d", index); PrintFloats(values, 9);
        Point input[17], output[17];
        for (auto& point : input) { point.fX = Value() * 128; point.fY = Value() * 128; }
        for (int count : {1, 2, 4, 5, 6, 17}) {
            composed.mapPoints(output, input, count);
            std::printf("map/%d/%d", index, count);
            for (int j = 0; j < count; ++j) {
                const float pair[] = {output[j].fX, output[j].fY};
                uint32_t bits[2]; std::memcpy(bits, pair, sizeof(bits));
                std::printf(" %08x %08x", bits[0], bits[1]);
            }
            std::putchar('\n');
        }
    }
    using namespace inspirecv::task;
    std::vector<uint8_t> source_bytes(67 * 43 * 3);
    for (auto& byte : source_bytes) byte = static_cast<uint8_t>(Next());
    const auto source = inspirecv::Image::Create(67, 43, 3, source_bytes.data());
    for (int index = 0; index < 64; ++index) {
        const float a = Value() * 0.125f + 1, b = Value() * 0.0625f;
        const float c = Value() * 0.0625f, d = Value() * 0.125f + 1;
        const float tx = Value() * 3, ty = Value() * 3;
        for (auto sampling : {SamplingMode::kNearest, SamplingMode::kLinear}) {
            PipelineOptions options;
            options.input_format = PixelFormat::kBgr;
            options.output_format = PixelFormat::kRgb;
            options.sampling = sampling;
            options.border = BorderMode::kReplicate;
            Pipeline pipeline(options);
            if (pipeline.SetTransform(inspirecv::TransformMatrix(a, b, tx, c, d, ty)) != Status::kOk) return 3;
            for (int width : {1, 4, 5, 17, 63, 257}) {
                inspirecv::Image actual;
                if (pipeline.Run(source, width, 19, &actual) != Status::kOk) return 4;
                std::printf("pixels/%d/%d/%d %016llx", index, int(sampling), width,
                            static_cast<unsigned long long>(Hash(actual.Data(), size_t(width) * 19 * 3)));
                if (full_pixels) {
                    std::putchar(' ');
                    for (size_t i = 0; i < size_t(width) * 19 * 3; ++i) {
                        std::printf("%02x", actual.Data()[i]);
                    }
                }
                std::putchar('\n');
            }
        }
    }
}
