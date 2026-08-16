// Task API example: read NV21 raw file and convert to BGR image.
#include "../include/inspirecv/task/task.h"
#include "../include/inspirecv/inspirecv.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <regex>

static bool read_file(const std::string& path, std::vector<uint8_t>& data) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;
    ifs.seekg(0, std::ios::end);
    std::streampos sz = ifs.tellg();
    if (sz <= 0) return false;
    data.resize(static_cast<size_t>(sz));
    ifs.seekg(0, std::ios::beg);
    ifs.read(reinterpret_cast<char*>(data.data()), sz);
    return ifs.good();
}

static bool try_parse_wh_from_name(const std::string& name, int& w, int& h) {
    std::regex re(".*[_-]w(\\d+)[_-]h(\\d+)\\.", std::regex::icase);
    std::smatch m;
    if (std::regex_match(name, m, re) && m.size() == 3) {
        w = std::stoi(m[1].str());
        h = std::stoi(m[2].str());
        return true;
    }
    return false;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage:\n"
                  << "  example_task_nv21_decode <nv21_file> [width height] [output_image]\n"
                  << "Example:\n"
                  << "  example_task_nv21_decode SRC_nv21_w256_h320.bin 256 320 exp_out_task_nv21_bgr.jpg\n"
                  << "  example_task_nv21_decode SRC_nv21_w256_h320.bin  # width/height parsed from name\n";
        return 0;
    }
    const std::string nv21_path = argv[1];
    int width = 0, height = 0;
    if (argc >= 4) {
        width = std::stoi(argv[2]);
        height = std::stoi(argv[3]);
    } else {
        if (!try_parse_wh_from_name(nv21_path, width, height)) {
            std::cerr << "Width/height not provided and cannot be parsed from filename.\n";
            return 1;
        }
    }
    const std::string out_path = (argc >= 5) ? argv[4] : std::string("exp_out_task_nv21_bgr.jpg");

    std::vector<uint8_t> nv21;
    if (!read_file(nv21_path, nv21)) {
        std::cerr << "Failed to read NV21 file: " << nv21_path << "\n";
        return 2;
    }
    const size_t expect = static_cast<size_t>(width) * static_cast<size_t>(height) * 3 / 2;
    if (nv21.size() < expect) {
        std::cerr << "NV21 file size too small: got " << nv21.size() << ", expect at least " << expect << "\n";
        return 3;
    }

    inspirecv::task::PipelineOptions options;
    options.input_format = inspirecv::task::PixelFormat::kNv21;
    options.output_format = inspirecv::task::PixelFormat::kBgr;
    options.sampling = inspirecv::task::SamplingMode::kNearest;
    options.border = inspirecv::task::BorderMode::kReplicate;
    inspirecv::task::Pipeline pipeline(options);

    inspirecv::task::RawImageView source;
    source.data = nv21.data();
    source.width = width;
    source.height = height;
    source.row_stride_bytes = 0;  // derive NV21 plane strides

    inspirecv::Image bgr;
    const inspirecv::task::Status status =
      pipeline.Run(source, width, height, &bgr);
    if (status != inspirecv::task::Status::kOk) {
        std::cerr << "Task pipeline failed: "
                  << inspirecv::task::StatusMessage(status) << "\n";
        return 4;
    }

    if (!bgr.Write(out_path)) {
        std::cerr << "Failed to write: " << out_path << "\n";
        return 5;
    }
    std::cout << "Wrote: " << out_path << "\n";
    return 0;
}
