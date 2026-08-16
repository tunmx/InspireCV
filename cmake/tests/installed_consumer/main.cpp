#include <inspirecv/inspirecv.h>
#include <inspirecv/task/pipeline.h>

#include <cstdint>

int main() {
    const auto& library = inspirecv::GetLibraryInfo();
    if (library.version_major != 1 || library.version_minor != 0) return 1;

    const uint8_t pixels[] = {
      1, 2, 3, 4, 5, 6,
      7, 8, 9, 10, 11, 12,
    };
    const auto source = inspirecv::Image::Create(2, 2, 3, pixels);

    inspirecv::task::PipelineOptions options;
    options.input_format = inspirecv::task::PixelFormat::kBgr;
    options.output_format = inspirecv::task::PixelFormat::kBgr;
    inspirecv::task::Pipeline pipeline(options);
    if (pipeline.ConfigurationStatus() != inspirecv::task::Status::kOk ||
        pipeline.SetTransform(inspirecv::TransformMatrix::Identity()) !=
          inspirecv::task::Status::kOk) {
        return 2;
    }

    inspirecv::Image destination;
    if (pipeline.Run(source, 2, 2, &destination) !=
        inspirecv::task::Status::kOk) {
        return 3;
    }
    if (destination.Width() != 2 || destination.Height() != 2 ||
        destination.Channels() != 3) {
        return 4;
    }
    for (size_t index = 0; index < sizeof(pixels); ++index) {
        if (destination.Data()[index] != pixels[index]) return 5;
    }
    return 0;
}
