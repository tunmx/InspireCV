#ifndef INSPIRECV_OKCV_IO_STB_WRAPPER_H_
#define INSPIRECV_OKCV_IO_STB_WRAPPER_H_

#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <cstring>
#include <cctype>
#include "check.h"

namespace okcv {

class ImageReader {
public:
    // Image data structure
    struct ImageData {
        int width = 0;
        int height = 0;
        int channels = 0;
        std::vector<unsigned char> data;

        bool isValid() const {
            return !data.empty() && width > 0 && height > 0;
        }

        void clear() {
            width = height = channels = 0;
            data.clear();
        }

        // Get total bytes
        size_t getDataSize() const {
            return static_cast<size_t>(width) * height * channels;
        }
    };

    enum class ColorOrder {
        RGB,  // Keep stb_image's default RGB order
        BGR   // Convert to BGR order
    };

    // Write configuration structure
    struct WriteConfig {
        int jpg_quality;         // JPEG quality (1-100)
        int png_compression;     // PNG compression level (0-9)
        ColorOrder color_order;  // Color order

        // Constructor to initialize members
        WriteConfig() : jpg_quality(100), png_compression(9), color_order(ColorOrder::BGR) {}
    };

    // Error codes
    enum class ErrorCode {
        OK = 0,
        FILE_NOT_FOUND,
        INVALID_CHANNEL,
        DECODE_ERROR,
        INVALID_PARAMETER
    };

    // Get last error message
    static std::string GetLastError();

    // Read/Write functions
    static bool Read(const std::string& filename, ImageData& outData, int requestedChannels,
                     ColorOrder order = ColorOrder::BGR);

    static bool Write(const std::string& filename, const unsigned char* data, int width, int height,
                      int channels, const WriteConfig& config = WriteConfig());

    static bool Write(const std::string& filename, const ImageData& data,
                      const WriteConfig& config = WriteConfig());

    // Utility functions
    static bool ReadInfo(const std::string& filename, int& width, int& height, int& channels);
    static bool IsSupportedImage(const std::string& filename);
    static int GetOriginalChannels(const std::string& filename);

private:
    static std::string GetFileExtension(const std::string& filename);
    static std::string lastError;
};

}  // namespace okcv

#endif  // INSPIRECV_OKCV_IO_STB_WRAPPER_H_