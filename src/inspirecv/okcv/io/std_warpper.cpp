#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "stb_warpper.h"

namespace okcv {

std::string ImageReader::lastError;

std::string ImageReader::GetLastError() {
    return lastError;
}

bool ImageReader::Read(const std::string& filename, ImageData& outData, int requestedChannels,
                       ColorOrder order) {
    // Clear output data
    outData.clear();
    lastError.clear();

    // Validate requested channels
    if (requestedChannels != 1 && requestedChannels != 3) {
        lastError =
          "Only 1 or 3 channels supported (requested: " + std::to_string(requestedChannels) + ")";
        INSPIRECV_LOG(ERROR) << lastError;
        return false;
    }

    // Read image data
    int width, height, channels;
    unsigned char* data =
      stbi_load(filename.c_str(), &width, &height, &channels, requestedChannels);

    if (!data) {
        lastError = "Decode failed: ";
        lastError += stbi_failure_reason();
        return false;
    }

    // Set output data
    outData.width = width;
    outData.height = height;
    outData.channels = requestedChannels;

    // Calculate data size and copy data
    size_t dataSize = static_cast<size_t>(width) * height * requestedChannels;
    outData.data.resize(dataSize);

    // Direct copy for grayscale or if no color conversion needed
    if (requestedChannels == 1 || order == ColorOrder::RGB) {
        std::memcpy(outData.data.data(), data, dataSize);
    }
    // Convert to BGR if needed
    else if (order == ColorOrder::BGR && requestedChannels == 3) {
        for (int i = 0; i < width * height; ++i) {
            outData.data[i * 3 + 0] = data[i * 3 + 2];  // B <- R
            outData.data[i * 3 + 1] = data[i * 3 + 1];  // G <- G
            outData.data[i * 3 + 2] = data[i * 3 + 0];  // R <- B
        }
    }

    // Free stb allocated memory
    stbi_image_free(data);
    return true;
}

bool ImageReader::Write(const std::string& filename, const unsigned char* data, int width,
                        int height, int channels, const WriteConfig& config) {
    if (!data || width <= 0 || height <= 0 || (channels != 1 && channels != 3)) {
        lastError = "Invalid input parameters";
        return false;
    }

    std::string ext = GetFileExtension(filename);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    bool success = false;

    // Convert color order if needed
    std::vector<unsigned char> rgb_data;
    const unsigned char* write_data = data;

    // Only convert if 3 channels and input is BGR
    if (channels == 3 && config.color_order == ColorOrder::BGR) {
        rgb_data.resize(static_cast<size_t>(width) * height * channels);
        for (int i = 0; i < width * height; ++i) {
            rgb_data[i * 3 + 0] = data[i * 3 + 2];  // R <- B
            rgb_data[i * 3 + 1] = data[i * 3 + 1];  // G <- G
            rgb_data[i * 3 + 2] = data[i * 3 + 0];  // B <- R
        }
        write_data = rgb_data.data();
    }

    // Choose save format based on extension
    if (ext == "png") {
        success = stbi_write_png(filename.c_str(), width, height, channels, write_data,
                                 width * channels) != 0;
    } else if (ext == "jpg" || ext == "jpeg") {
        success = stbi_write_jpg(filename.c_str(), width, height, channels, write_data,
                                 config.jpg_quality) != 0;
    } else if (ext == "bmp") {
        success = stbi_write_bmp(filename.c_str(), width, height, channels, write_data) != 0;
    } else {
        lastError = "Unsupported image format: " + ext;
        INSPIRECV_LOG(ERROR) << lastError;
        return false;
    }

    if (!success) {
        lastError = "Failed to write image: " + filename;
        INSPIRECV_LOG(ERROR) << lastError;
    }

    return success;
}

bool ImageReader::Write(const std::string& filename, const ImageData& data,
                        const WriteConfig& config) {
    if (!data.isValid()) {
        lastError = "Invalid image data";
        INSPIRECV_LOG(ERROR) << lastError;
        return false;
    }
    return Write(filename, data.data.data(), data.width, data.height, data.channels, config);
}

bool ImageReader::ReadInfo(const std::string& filename, int& width, int& height, int& channels) {
    lastError.clear();

    if (!stbi_info(filename.c_str(), &width, &height, &channels)) {
        lastError = "Failed to read image info: ";
        lastError += stbi_failure_reason();
        return false;
    }

    return true;
}

bool ImageReader::IsSupportedImage(const std::string& filename) {
    int width, height, channels;
    return stbi_info(filename.c_str(), &width, &height, &channels) != 0;
}

int ImageReader::GetOriginalChannels(const std::string& filename) {
    int width, height, channels;
    if (stbi_info(filename.c_str(), &width, &height, &channels)) {
        return channels;
    }
    return -1;
}

std::string ImageReader::GetFileExtension(const std::string& filename) {
    size_t pos = filename.find_last_of('.');
    if (pos != std::string::npos && pos + 1 < filename.length()) {
        return filename.substr(pos + 1);
    }
    return "";
}

}  // namespace okcv