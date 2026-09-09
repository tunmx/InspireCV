// Common helpers for image-based tests
#pragma once
#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <inspirecv/inspirecv.h>
#if defined(_WIN32)
#include <direct.h>
#endif

inline bool inspirecv_test_has_image_io() {
    // The OpenCV backend uses cv::imread/imwrite. The standalone OKCV backend
    // always has its bundled STB reader/writer available.
    return true;
}

inline std::string inspirecv_test_images_dir() {
    const char* env = std::getenv("INSPIRECV_IMAGES_DIR");
    if (env && *env) return std::string(env);
    return std::string("../images");
}

inline std::string inspirecv_test_image_path(const std::string& filename) {
    return inspirecv_test_images_dir() + "/" + filename;
}

inline std::string inspirecv_test_out_dir() {
    const char* env = std::getenv("INSPIRECV_TEST_OUT_DIR");
    if (env && *env) return std::string(env);
    return std::string("inspirecv_test_output");
}

inline bool inspirecv_test_dump_images_enabled() {
    const char* env = std::getenv("INSPIRECV_TEST_DUMP_IMAGES");
    if (!env || !*env) return false;
    const std::string value(env);
    return value == "1" || value == "true" || value == "TRUE" ||
           value == "on" || value == "ON" || value == "yes" || value == "YES";
}

inline bool inspirecv_test_ensure_dir(const std::string& dir) {
    if (dir.empty() || dir == "." || dir == "./") return true;
    errno = 0;
#if defined(_WIN32)
    const int result = _mkdir(dir.c_str());
#else
    const int result = mkdir(dir.c_str(), 0777);
#endif
    return result == 0 || errno == EEXIST;
}

template <typename Pixel>
inline bool inspirecv_test_write_image(const inspirecv::ImageT<Pixel>& img,
                                       const std::string& filename) {
    if (!inspirecv_test_dump_images_enabled()) return false;

    const std::string out = inspirecv_test_out_dir();
    if (!inspirecv_test_ensure_dir(out)) {
        std::cerr << "[test-image-dump] cannot create directory: " << out << std::endl;
        return false;
    }

    const std::string path = (out == "." || out == "./") ? filename : out + "/" + filename;
    const bool write_result = img.Write(path);
    struct stat file_info;
    const bool exists = stat(path.c_str(), &file_info) == 0;
    if (!write_result || !exists) {
        std::cerr << "[test-image-dump] cannot write image: " << path << std::endl;
        return false;
    }
    return true;
}

template <typename Pixel>
inline void inspirecv_test_write_comparison(const inspirecv::ImageT<Pixel>& actual,
                                            const inspirecv::ImageT<Pixel>& expected,
                                            const std::string& stem) {
    if (!inspirecv_test_dump_images_enabled()) return;
    inspirecv_test_write_image(actual, stem + "_actual.png");
    inspirecv_test_write_image(expected, stem + "_expected.png");
    if (actual.Width() != expected.Width() || actual.Height() != expected.Height() ||
        actual.Channels() != expected.Channels() || actual.Empty() || expected.Empty()) {
        std::cerr << "[test-image-dump] cannot diff images with different shapes: " << stem << std::endl;
        return;
    }
    auto diff = actual.AbsDiff(expected);
    inspirecv_test_write_image(diff, stem + "_diff.png");
    inspirecv_test_write_image(diff.Mul(8.0), stem + "_diff_x8.png");
}
