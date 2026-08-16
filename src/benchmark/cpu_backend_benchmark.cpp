#include <inspirecv/inspirecv.h>
#include <inspirecv/task/task.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

volatile double g_sink = 0.0;

struct Options {
    int samples = 101;
    int warmups = 10;
    int opencv_threads = 1;
    std::string report = "cpu_backend_benchmark.csv";
    std::string machine = "unspecified";
    std::string suite = "full";
};

struct Distribution {
    double p50_us = 0.0;
    double p95_us = 0.0;
};

struct Accuracy {
    size_t elements = 0;
    size_t mismatches = 0;
    double maximum_absolute_error = 0.0;
    double mean_absolute_error = 0.0;
};

struct BenchmarkResult {
    Distribution inspirecv;
    Distribution opencv;
    Accuracy accuracy;
};

Options ParseOptions(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument(argv[index]);
        const auto consume = [&](const char* name) -> std::string {
            if (index + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++index];
        };
        if (argument == "--samples") {
            options.samples = std::stoi(consume("--samples"));
        } else if (argument == "--warmups") {
            options.warmups = std::stoi(consume("--warmups"));
        } else if (argument == "--opencv-threads") {
            options.opencv_threads = std::stoi(consume("--opencv-threads"));
        } else if (argument == "--report") {
            options.report = consume("--report");
        } else if (argument == "--machine") {
            options.machine = consume("--machine");
        } else if (argument == "--suite") {
            options.suite = consume("--suite");
        } else if (argument == "--help") {
            std::cout
              << "Usage: inspirecv_cpu_benchmark [--samples N] [--warmups N] "
                 "[--opencv-threads N] [--report FILE] [--machine LABEL] "
                 "[--suite full|u8c3|matrix]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + argument);
        }
    }
    if (options.samples < 3 || (options.samples & 1) == 0) {
        throw std::runtime_error("--samples must be an odd integer of at least 3");
    }
    if (options.warmups < 1) {
        throw std::runtime_error("--warmups must be positive");
    }
    if (options.suite != "full" && options.suite != "u8c3" &&
        options.suite != "matrix") {
        throw std::runtime_error("--suite must be full, u8c3, or matrix");
    }
    return options;
}

double Percentile(std::vector<double> samples, double percentile) {
    std::sort(samples.begin(), samples.end());
    const size_t index = static_cast<size_t>(
      std::ceil(percentile * static_cast<double>(samples.size())) - 1.0);
    return samples[std::min(index, samples.size() - 1)];
}

Distribution Summarize(const std::vector<double>& samples) {
    Distribution result;
    result.p50_us = Percentile(samples, 0.50);
    result.p95_us = Percentile(samples, 0.95);
    return result;
}

double TimeOne(const std::function<void()>& operation) {
    const auto begin = std::chrono::steady_clock::now();
    operation();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(end - begin).count();
}

std::pair<Distribution, Distribution> MeasureAlternating(
  const std::function<void()>& inspirecv_operation,
  const std::function<void()>& opencv_operation, const Options& options) {
    for (int iteration = 0; iteration < options.warmups; ++iteration) {
        if ((iteration & 1) == 0) {
            inspirecv_operation();
            opencv_operation();
        } else {
            opencv_operation();
            inspirecv_operation();
        }
    }

    std::vector<double> inspirecv_samples;
    std::vector<double> opencv_samples;
    inspirecv_samples.reserve(options.samples);
    opencv_samples.reserve(options.samples);
    for (int sample = 0; sample < options.samples; ++sample) {
        if ((sample & 1) == 0) {
            inspirecv_samples.push_back(TimeOne(inspirecv_operation));
            opencv_samples.push_back(TimeOne(opencv_operation));
        } else {
            opencv_samples.push_back(TimeOne(opencv_operation));
            inspirecv_samples.push_back(TimeOne(inspirecv_operation));
        }
    }
    return {Summarize(inspirecv_samples), Summarize(opencv_samples)};
}

std::vector<uint8_t> MakePixels(int width, int height, int channels,
                                uint32_t seed = 17) {
    std::vector<uint8_t> pixels(
      static_cast<size_t>(width) * height * channels);
    uint32_t state = seed;
    for (size_t index = 0; index < pixels.size(); ++index) {
        state = state * 1664525u + 1013904223u;
        pixels[index] = static_cast<uint8_t>((state >> 24) ^ (index * 29u));
    }
    return pixels;
}

Accuracy CompareU8(const inspirecv::Image& actual, const cv::Mat& expected) {
    if (actual.Width() != expected.cols || actual.Height() != expected.rows ||
        actual.Channels() != expected.channels() || expected.depth() != CV_8U) {
        throw std::runtime_error("u8 comparison shape mismatch");
    }
    Accuracy result;
    result.elements = static_cast<size_t>(actual.Width()) * actual.Height() *
                      actual.Channels();
    double absolute_sum = 0.0;
    size_t offset = 0;
    for (int row = 0; row < expected.rows; ++row) {
        const uint8_t* expected_row = expected.ptr<uint8_t>(row);
        const size_t row_elements =
          static_cast<size_t>(expected.cols) * expected.channels();
        for (size_t column = 0; column < row_elements; ++column, ++offset) {
            const double error = std::abs(
              static_cast<int>(actual.Data()[offset]) - expected_row[column]);
            if (error != 0.0) ++result.mismatches;
            result.maximum_absolute_error =
              std::max(result.maximum_absolute_error, error);
            absolute_sum += error;
        }
    }
    result.mean_absolute_error = absolute_sum / result.elements;
    return result;
}

Accuracy CompareFloat(const std::vector<float>& actual,
                      const std::vector<float>& expected) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error("float comparison shape mismatch");
    }
    Accuracy result;
    result.elements = actual.size();
    double absolute_sum = 0.0;
    for (size_t index = 0; index < actual.size(); ++index) {
        const double error = std::fabs(
          static_cast<double>(actual[index]) - expected[index]);
        if (std::memcmp(&actual[index], &expected[index], sizeof(float)) != 0) {
            ++result.mismatches;
        }
        result.maximum_absolute_error =
          std::max(result.maximum_absolute_error, error);
        absolute_sum += error;
    }
    result.mean_absolute_error = absolute_sum / result.elements;
    return result;
}

template <typename InspireOperation, typename OpenCvOperation>
BenchmarkResult RunImageCase(InspireOperation inspire_operation,
                             OpenCvOperation opencv_operation,
                             const Options& options) {
    inspirecv::Image inspire_output = inspire_operation();
    cv::Mat opencv_output = opencv_operation();
    BenchmarkResult result;
    result.accuracy = CompareU8(inspire_output, opencv_output);

    const auto measured = MeasureAlternating(
      [&]() {
          inspirecv::Image output = inspire_operation();
          g_sink += output.Data()[output.Width() * output.Channels() / 2];
      },
      [&]() {
          cv::Mat output = opencv_operation();
          g_sink += output.ptr<uint8_t>(output.rows / 2)[
            output.cols * output.channels() / 2];
      },
      options);
    result.inspirecv = measured.first;
    result.opencv = measured.second;
    return result;
}

class CsvReport {
public:
    CsvReport(const Options& options) : output_(options.report) {
        if (!output_) throw std::runtime_error("cannot open report: " + options.report);
        const inspirecv::LibraryInfo& info = inspirecv::GetLibraryInfo();
        const int reported_opencv_threads = cv::getNumThreads();
        output_ << "# benchmark=inspirecv_vs_opencv_cpu\n"
                << "# machine=" << options.machine << "\n"
                << "# inspirecv=" << info.version_string << "\n"
                << "# opencv=" << CV_VERSION << "\n"
                << "# compiler=" << info.compiler_name << ' '
                << info.compiler_version << "\n"
                << "# build_type=" << info.build_type << "\n"
                << "# neon=" << (info.neon_enabled ? 1 : 0) << "\n"
                << "# sse=" << (info.sse_enabled ? 1 : 0) << "\n"
                << "# avx2_objects=" << (info.avx2_enabled ? 1 : 0) << "\n"
                << "# opencv_threads_requested=" << options.opencv_threads << "\n"
                << "# opencv_threads_reported=" << reported_opencv_threads << "\n"
                << "# opencv_thread_control_honored="
                << (reported_opencv_threads == options.opencv_threads ? 1 : 0)
                << "\n"
                << "# opencv_optimized=" << (cv::useOptimized() ? 1 : 0) << "\n"
                << "# suite=" << options.suite << "\n"
                << "# samples=" << options.samples << "\n"
                << "# warmups=" << options.warmups << "\n";
        output_ << "family,operation,type,channels,source_width,source_height,"
                   "destination_width,destination_height,timing_scope,contract,"
                   "tolerance,inspirecv_p50_us,inspirecv_p95_us,opencv_p50_us,"
                   "opencv_p95_us,inspirecv_speedup,winner,elements,mismatches,"
                   "max_abs_error,mean_abs_error,accuracy\n";
    }

    void Add(const char* family, const char* operation, const char* type,
             int channels, int source_width, int source_height,
             int destination_width, int destination_height,
             const char* timing_scope, bool comparable, double tolerance,
             const BenchmarkResult& result) {
        const bool accurate = result.accuracy.maximum_absolute_error <= tolerance;
        const char* accuracy = comparable
                                 ? (accurate ? (result.accuracy.mismatches == 0
                                                  ? "exact"
                                                  : "within_tolerance")
                                             : "failed")
                                 : "different_contract";
        const double speedup =
          result.opencv.p50_us / result.inspirecv.p50_us;
        const char* winner = speedup > 1.02
                               ? "inspirecv"
                               : (speedup < 0.98 ? "opencv" : "tie");
        output_ << family << ',' << operation << ',' << type << ',' << channels
                << ',' << source_width << ',' << source_height << ','
                << destination_width << ',' << destination_height << ','
                << timing_scope << ','
                << (comparable ? "aligned" : "different") << ',' << tolerance
                << ',' << std::fixed << std::setprecision(3)
                << result.inspirecv.p50_us << ',' << result.inspirecv.p95_us
                << ',' << result.opencv.p50_us << ',' << result.opencv.p95_us
                << ',' << std::setprecision(4) << speedup << ',' << winner << ','
                << result.accuracy.elements << ',' << result.accuracy.mismatches
                << ',' << std::setprecision(7)
                << result.accuracy.maximum_absolute_error << ','
                << result.accuracy.mean_absolute_error << ',' << accuracy << '\n';
        std::cout << std::left << std::setw(26)
                  << (std::string(family) + "/" + operation) << " InspireCV "
                  << std::right << std::setw(9) << std::setprecision(3)
                  << std::fixed << result.inspirecv.p50_us << " us  OpenCV "
                  << std::setw(9) << result.opencv.p50_us << " us  speedup "
                  << std::setw(7) << std::setprecision(3) << speedup << "x  "
                  << accuracy << '\n';
        if (comparable && !accurate) failed_ = true;
    }

    bool failed() const { return failed_; }

private:
    std::ofstream output_;
    bool failed_ = false;
};

struct ResizeCase {
    int source_width;
    int source_height;
    int destination_width;
    int destination_height;
    bool nearest_contract_aligned;
};

void AddImageResizeCase(CsvReport* report, const Options& options,
                        const ResizeCase& current) {
    const std::vector<uint8_t> pixels = MakePixels(
      current.source_width, current.source_height, 3,
      static_cast<uint32_t>(current.source_width));
    const inspirecv::Image image = inspirecv::Image::Create(
      current.source_width, current.source_height, 3, pixels.data(), false);
    const cv::Mat mat(current.source_height, current.source_width, CV_8UC3,
                      const_cast<uint8_t*>(pixels.data()));

    BenchmarkResult nearest = RunImageCase(
      [&]() {
          return image.Resize(current.destination_width,
                              current.destination_height, false);
      },
      [&]() {
          cv::Mat output;
          cv::resize(mat, output,
                     cv::Size(current.destination_width,
                              current.destination_height),
                     0.0, 0.0, cv::INTER_NEAREST);
          return output;
      },
      options);
    report->Add("image", "resize_nearest", "u8", 3,
                current.source_width, current.source_height,
                current.destination_width, current.destination_height,
                "public_api_allocation", current.nearest_contract_aligned,
                0.0, nearest);

    BenchmarkResult linear = RunImageCase(
      [&]() {
          return image.Resize(current.destination_width,
                              current.destination_height, true);
      },
      [&]() {
          cv::Mat output;
          cv::resize(mat, output,
                     cv::Size(current.destination_width,
                              current.destination_height),
                     0.0, 0.0, cv::INTER_LINEAR);
          return output;
      },
      options);
    report->Add("image", "resize_linear", "u8", 3,
                current.source_width, current.source_height,
                current.destination_width, current.destination_height,
                "public_api_allocation", false, 0.0, linear);
}

void AddImageResizeCases(CsvReport* report, const Options& options) {
    const ResizeCase cases[] = {
      {128, 128, 256, 256, true}, {512, 512, 256, 256, true},
      // The two libraries intentionally use different nearest-neighbour source
      // coordinate rules for non-integral, aspect-changing scale factors.
      {1920, 1080, 224, 224, false},
      {1920, 1080, 3840, 2160, true}};
    for (const ResizeCase& current : cases) {
        AddImageResizeCase(report, options, current);
    }
}

void AddImageResizeMatrixCases(CsvReport* report, const Options& options) {
    const ResizeCase cases[] = {
      {64, 64, 128, 128, true},       {128, 128, 256, 256, true},
      {224, 224, 112, 112, true},     {512, 512, 256, 256, true},
      {640, 480, 320, 240, true},     {1920, 1080, 960, 540, true},
      // Non-integral, aspect-changing nearest-neighbour coordinates differ.
      {1920, 1080, 224, 224, false},
      {1920, 1080, 3840, 2160, true},
    };
    for (const ResizeCase& current : cases) {
        AddImageResizeCase(report, options, current);
    }
}

void AddImagePointwiseCases(CsvReport* report, const Options& options,
                            int width, int height, bool include_gray = true) {
    const std::vector<uint8_t> pixels = MakePixels(width, height, 3, 73);
    const inspirecv::Image image = inspirecv::Image::Create(
      width, height, 3, pixels.data(), false);
    const cv::Mat mat(height, width, CV_8UC3,
                      const_cast<uint8_t*>(pixels.data()));

    BenchmarkResult rotate = RunImageCase(
      [&]() { return image.Rotate90(); },
      [&]() {
          cv::Mat output;
          cv::rotate(mat, output, cv::ROTATE_90_CLOCKWISE);
          return output;
      },
      options);
    report->Add("image", "rotate90", "u8", 3, width, height, height, width,
                "public_api_allocation", true, 0.0, rotate);

    BenchmarkResult flip = RunImageCase(
      [&]() { return image.FlipHorizontal(); },
      [&]() {
          cv::Mat output;
          cv::flip(mat, output, 1);
          return output;
      },
      options);
    report->Add("image", "flip_horizontal", "u8", 3, width, height, width,
                height, "public_api_allocation", true, 0.0, flip);

    BenchmarkResult swap = RunImageCase(
      [&]() { return image.SwapRB(); },
      [&]() {
          cv::Mat output;
          cv::cvtColor(mat, output, cv::COLOR_BGR2RGB);
          return output;
      },
      options);
    report->Add("image", "swap_rb", "u8", 3, width, height, width, height,
                "public_api_allocation", true, 0.0, swap);

    if (!include_gray) return;

    BenchmarkResult gray = RunImageCase(
      [&]() { return image.ToGray(); },
      [&]() {
          cv::Mat output;
          cv::cvtColor(mat, output, cv::COLOR_BGR2GRAY);
          return output;
      },
      options);
    report->Add("image", "to_gray", "u8", 3, width, height, width, height,
                "public_api_allocation", true, 1.0, gray);
}

void AddU8C3ScaleCases(CsvReport* report, const Options& options) {
    struct SizeCase {
        int width;
        int height;
    };
    const SizeCase cases[] = {
      {16, 16},     {32, 32},      {64, 64},       {112, 112},
      {224, 224},   {512, 512},    {640, 480},     {1280, 720},
      {1920, 1080}, {3840, 2160},
    };
    for (const SizeCase& current : cases) {
        AddImagePointwiseCases(report, options, current.width, current.height,
                               false);
    }
}

void AddImageGrayCases(CsvReport* report, const Options& options,
                       int width, int height) {
    const std::vector<uint8_t> pixels = MakePixels(width, height, 1, 91);
    const inspirecv::Image image = inspirecv::Image::Create(
      width, height, 1, pixels.data(), false);
    const cv::Mat mat(height, width, CV_8UC1,
                      const_cast<uint8_t*>(pixels.data()));

    BenchmarkResult threshold = RunImageCase(
      [&]() { return image.Threshold(100.0, 255.0, 0); },
      [&]() {
          cv::Mat output;
          cv::threshold(mat, output, 100.0, 255.0, cv::THRESH_BINARY);
          return output;
      },
      options);
    report->Add("image", "threshold", "u8", 1, width, height, width, height,
                "public_api_allocation", true, 0.0, threshold);

    BenchmarkResult gaussian = RunImageCase(
      [&]() { return image.GaussianBlur(5, 0.0); },
      [&]() {
          cv::Mat output;
          cv::GaussianBlur(mat, output, cv::Size(5, 5), 0.0, 0.0,
                           cv::BORDER_REPLICATE);
          return output;
      },
      options);
    // sigma=0 is not an aligned contract: OpenCV substitutes its exact 5-tap
    // binomial kernel, while InspireCV evaluates the documented sigma heuristic.
    // Keep the error in the report, but do not turn a kernel-policy difference
    // into a false numerical-regression failure.
    report->Add("image", "gaussian5", "u8", 1, width, height, width, height,
                "public_api_allocation", false, 0.0, gaussian);

    BenchmarkResult erode = RunImageCase(
      [&]() { return image.Erode(3, 1); },
      [&]() {
          cv::Mat output;
          const cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_RECT, cv::Size(3, 3));
          cv::erode(mat, output, kernel);
          return output;
      },
      options);
    report->Add("image", "erode3", "u8", 1, width, height, width, height,
                "public_api_allocation", true, 0.0, erode);
}

void AddImageWarpCase(CsvReport* report, const Options& options,
                      int width, int height) {
    const std::vector<uint8_t> pixels = MakePixels(width, height, 3, 113);
    const inspirecv::Image image = inspirecv::Image::Create(
      width, height, 3, pixels.data(), false);
    const cv::Mat mat(height, width, CV_8UC3,
                      const_cast<uint8_t*>(pixels.data()));
    const float angle = 0.17f;
    const float cosine = std::cos(angle);
    const float sine = std::sin(angle);
    const float center_x = (width - 1) * 0.5f;
    const float center_y = (height - 1) * 0.5f;
    const float translate_x = center_x - cosine * center_x - sine * center_y;
    const float translate_y = center_y + sine * center_x - cosine * center_y;
    const inspirecv::TransformMatrix matrix(
      cosine, sine, translate_x, -sine, cosine, translate_y);
    const cv::Matx23d destination_to_source(
      cosine, sine, translate_x, -sine, cosine, translate_y);

    BenchmarkResult warp = RunImageCase(
      [&]() { return image.WarpAffine(matrix, width, height); },
      [&]() {
          cv::Mat output;
          cv::warpAffine(mat, output, destination_to_source,
                         cv::Size(width, height),
                         cv::INTER_LINEAR | cv::WARP_INVERSE_MAP,
                         cv::BORDER_CONSTANT);
          return output;
      },
      options);
    report->Add("image", "warp_affine_linear", "u8", 3, width, height,
                width, height, "public_api_allocation", false, 0.0, warp);
}

void AddImageAlgorithmMatrixCases(CsvReport* report, const Options& options) {
    AddImageResizeMatrixCases(report, options);

    struct SizeCase {
        int width;
        int height;
    };
    const SizeCase common_sizes[] = {
      {64, 64},       {112, 112},   {224, 224},
      {512, 512},     {640, 480},   {1280, 720},
      {1920, 1080},   {3840, 2160},
    };
    for (const SizeCase& current : common_sizes) {
        AddImagePointwiseCases(report, options, current.width, current.height);
        AddImageGrayCases(report, options, current.width, current.height);
    }

    const SizeCase warp_sizes[] = {
      {112, 112}, {224, 224}, {512, 512}, {1280, 720}, {1920, 1080},
    };
    for (const SizeCase& current : warp_sizes) {
        AddImageWarpCase(report, options, current.width, current.height);
    }
}

BenchmarkResult RunTaskCase(int source_width, int source_height,
                            int destination_width, int destination_height,
                            inspirecv::task::TensorOrder order,
                            bool use_resize, const Options& options) {
    const std::vector<uint8_t> pixels = MakePixels(
      source_width, source_height, 3,
      static_cast<uint32_t>(source_width + destination_width));
    const size_t output_elements =
      static_cast<size_t>(destination_width) * destination_height * 3;
    std::vector<float> inspire_output(output_elements);
    std::vector<float> opencv_output(output_elements);

    inspirecv::task::PipelineOptions pipeline_options;
    pipeline_options.input_format = inspirecv::task::PixelFormat::kBgr;
    pipeline_options.output_format = inspirecv::task::PixelFormat::kRgb;
    pipeline_options.sampling = use_resize
                                  ? inspirecv::task::SamplingMode::kLinear
                                  : inspirecv::task::SamplingMode::kNearest;
    pipeline_options.mean = {{127.5f, 127.5f, 127.5f, 0.0f}};
    pipeline_options.scale = {{1.0f / 128.0f, 1.0f / 128.0f,
                               1.0f / 128.0f, 1.0f}};
    pipeline_options.backend_preference =
      inspirecv::task::BackendPreference::kCpu;
    inspirecv::task::Pipeline pipeline(pipeline_options);
    if (use_resize) {
        const inspirecv::TransformMatrix transform(
          static_cast<float>(source_width) / destination_width, 0.0f, 0.0f,
          0.0f, static_cast<float>(source_height) / destination_height, 0.0f);
        pipeline.SetTransform(transform);
    }
    inspirecv::task::RawImageView source;
    source.data = pixels.data();
    source.width = source_width;
    source.height = source_height;
    source.row_stride_bytes = static_cast<size_t>(source_width) * 3;
    inspirecv::task::TensorBuffer destination;
    destination.data = inspire_output.data();
    destination.width = destination_width;
    destination.height = destination_height;
    destination.channels = 3;
    destination.element_type = inspirecv::task::ElementType::kFloat32;
    destination.order = order;

    const cv::Mat input(source_height, source_width, CV_8UC3,
                        const_cast<uint8_t*>(pixels.data()));
    cv::Mat resized(destination_height, destination_width, CV_8UC3);
    cv::Mat rgb(destination_height, destination_width, CV_8UC3);
    cv::Mat interleaved_float(destination_height, destination_width, CV_32FC3);
    cv::Mat hwc_output(destination_height, destination_width, CV_32FC3,
                       opencv_output.data());
    std::vector<cv::Mat> planes;
    if (order == inspirecv::task::TensorOrder::kChw) {
        const size_t plane_elements =
          static_cast<size_t>(destination_width) * destination_height;
        for (int channel = 0; channel < 3; ++channel) {
            planes.emplace_back(destination_height, destination_width, CV_32FC1,
                                opencv_output.data() + channel * plane_elements);
        }
    }

    const std::function<void()> run_inspirecv = [&]() {
        const inspirecv::task::Status status = pipeline.Run(source, destination);
        if (status != inspirecv::task::Status::kOk) {
            throw std::runtime_error(inspirecv::task::StatusMessage(status));
        }
        g_sink += inspire_output[output_elements / 2];
    };
    const std::function<void()> run_opencv = [&]() {
        const cv::Mat* transformed = &input;
        if (use_resize) {
            cv::resize(input, resized,
                       cv::Size(destination_width, destination_height),
                       0.0, 0.0, cv::INTER_LINEAR);
            transformed = &resized;
        }
        cv::cvtColor(*transformed, rgb, cv::COLOR_BGR2RGB);
        if (order == inspirecv::task::TensorOrder::kChw) {
            rgb.convertTo(interleaved_float, CV_32FC3, 1.0 / 128.0,
                          -127.5 / 128.0);
            cv::split(interleaved_float, planes);
        } else {
            rgb.convertTo(hwc_output, CV_32FC3, 1.0 / 128.0,
                          -127.5 / 128.0);
        }
        g_sink += opencv_output[output_elements / 2];
    };

    run_inspirecv();
    run_opencv();
    BenchmarkResult result;
    result.accuracy = CompareFloat(inspire_output, opencv_output);
    const auto measured = MeasureAlternating(
      run_inspirecv, run_opencv, options);
    result.inspirecv = measured.first;
    result.opencv = measured.second;
    return result;
}

struct TaskCase {
    int source_width;
    int source_height;
    int destination_width;
    int destination_height;
    inspirecv::task::TensorOrder order;
    bool resize;
};

void AddTaskCase(CsvReport* report, const Options& options,
                 const TaskCase& current) {
    const BenchmarkResult result = RunTaskCase(
      current.source_width, current.source_height,
      current.destination_width, current.destination_height,
      current.order, current.resize, options);
    const bool comparable = !current.resize;
    report->Add("task",
                current.order == inspirecv::task::TensorOrder::kChw
                  ? (current.resize ? "resize_bgr_to_rgb_f32_chw"
                                    : "identity_bgr_to_rgb_f32_chw")
                  : "identity_bgr_to_rgb_f32_hwc",
                "u8_to_f32", 3, current.source_width,
                current.source_height, current.destination_width,
                current.destination_height, "preallocated_end_to_end",
                comparable, comparable ? 1.0e-6 : 0.0, result);
}

void AddTaskCases(CsvReport* report, const Options& options) {
    const TaskCase cases[] = {
      {112, 112, 112, 112, inspirecv::task::TensorOrder::kChw, false},
      {224, 224, 224, 224, inspirecv::task::TensorOrder::kChw, false},
      {640, 640, 640, 640, inspirecv::task::TensorOrder::kChw, false},
      {224, 224, 224, 224, inspirecv::task::TensorOrder::kHwc, false},
      {640, 480, 112, 112, inspirecv::task::TensorOrder::kChw, true},
      {1920, 1080, 224, 224, inspirecv::task::TensorOrder::kChw, true},
    };
    for (const TaskCase& current : cases) {
        AddTaskCase(report, options, current);
    }
}

void AddTaskMatrixCases(CsvReport* report, const Options& options) {
    const TaskCase cases[] = {
      {64, 64, 64, 64, inspirecv::task::TensorOrder::kChw, false},
      {64, 64, 64, 64, inspirecv::task::TensorOrder::kHwc, false},
      {112, 112, 112, 112, inspirecv::task::TensorOrder::kChw, false},
      {112, 112, 112, 112, inspirecv::task::TensorOrder::kHwc, false},
      {224, 224, 224, 224, inspirecv::task::TensorOrder::kChw, false},
      {224, 224, 224, 224, inspirecv::task::TensorOrder::kHwc, false},
      {320, 320, 320, 320, inspirecv::task::TensorOrder::kChw, false},
      {320, 320, 320, 320, inspirecv::task::TensorOrder::kHwc, false},
      {640, 640, 640, 640, inspirecv::task::TensorOrder::kChw, false},
      {640, 640, 640, 640, inspirecv::task::TensorOrder::kHwc, false},
      {640, 480, 112, 112, inspirecv::task::TensorOrder::kChw, true},
      {1280, 720, 224, 224, inspirecv::task::TensorOrder::kChw, true},
      {1920, 1080, 224, 224, inspirecv::task::TensorOrder::kChw, true},
      {3840, 2160, 640, 640, inspirecv::task::TensorOrder::kChw, true},
    };
    for (const TaskCase& current : cases) {
        AddTaskCase(report, options, current);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = ParseOptions(argc, argv);
        cv::setUseOptimized(true);
        cv::setNumThreads(options.opencv_threads);
        CsvReport report(options);

        if (options.suite == "u8c3") {
            AddU8C3ScaleCases(&report, options);
        } else if (options.suite == "matrix") {
            AddImageAlgorithmMatrixCases(&report, options);
            AddTaskMatrixCases(&report, options);
        } else {
            AddImageResizeCases(&report, options);
            AddImagePointwiseCases(&report, options, 512, 512);
            AddImagePointwiseCases(&report, options, 1920, 1080);
            AddImageGrayCases(&report, options, 512, 512);
            AddImageGrayCases(&report, options, 1920, 1080);
            AddImageWarpCase(&report, options, 512, 512);
            AddImageWarpCase(&report, options, 1920, 1080);
            AddTaskCases(&report, options);
        }

        std::cout << "report=" << options.report << " sink=" << g_sink << '\n';
        if (report.failed()) {
            std::cerr << "accuracy gate failed\n";
            return 2;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
