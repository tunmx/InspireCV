#include <algorithm>
#include <map>
#include <inspirecv/task/core/stream_task.h>
#include <inspirecv/task/core/stream_task_utils.h>
#include <inspirecv/task/core/st_defs.h>
#include <memory>
#include <vector>
#include <cstring>

namespace inspirecv {
namespace task {

struct StreamTask::Inside {
    Config config_;
    std::unique_ptr<inspirecv::StreamTaskUtils> proc_;
};

void StreamTask::Destroy(StreamTask* pro) {
    if (nullptr != pro) {
        delete pro;
    }
}

StreamTask::~StreamTask() {
    delete inside_;
}

StreamTask::StreamTask(const Config& config) {
    inside_          = new Inside;
    inside_->config_ = config;
    inside_->proc_.reset(new inspirecv::StreamTaskUtils(config));
}

StreamTask* StreamTask::Create(const Config& config) { return new StreamTask(config); }

void StreamTask::SetMatrix(const task::Matrix& matrix) {
    inside_->proc_->setMatrix(matrix);
    transform_ = matrix;
}

static int _getBpp(task::StreamFormat format) {
    switch (format) {
        case task::RGB:
        case task::BGR:
        case task::YCrCb:
        case task::YUV:
        case task::HSV:
        case task::XYZ:
            return 3;
        case task::RGBA:
        case task::BGRA:
            return 4;
        case task::GRAY:
            return 1;
        case task::BGR555:
        case task::BGR565:
            return 2;
        default:
            break;
    }
    return 0;
}

TaskStatus StreamTask::Convert(const uint8_t* source, int iw, int ih, int stride,
                               void* dest, int ow, int oh, int outputBpp,
                               int /*outputStride*/, halide_type_t type) {
    int ic = _getBpp(inside_->config_.sourceFormat);
    int oc = outputBpp;
    if (outputBpp == 0) {
        oc = _getBpp(inside_->config_.destFormat);
    }
    // Ultra-fast path: identity + same format + uint8 + same size
    if (inside_->config_.sourceFormat == inside_->config_.destFormat &&
        oc == ic && halide_type_bytes(type) == 1 &&
        ow == iw && oh == ih && transform_.isIdentity()) {
        int srcStride = stride == 0 ? iw * ic : stride;
        const int rowBytes = ow * oc;
        uint8_t* dstPtr = reinterpret_cast<uint8_t*>(dest);
        if (srcStride == rowBytes) {
            ::memcpy(dstPtr, source, (size_t)rowBytes * (size_t)oh);
            return SUCCESS;
        } else {
            for (int y = 0; y < oh; ++y) {
                ::memcpy(dstPtr + (size_t)y * rowBytes, source + (size_t)y * srcStride, rowBytes);
            }
            return SUCCESS;
        }
    }
    inside_->proc_->SetPadding(padding_value_);
    inside_->proc_->ResizeFunc(ic, iw, ih, oc, ow, oh, type, stride);
    return inside_->proc_->ExecFunc(source, stride, dest);
}

void StreamTask::SetDraw() {
    if (inside_ && inside_->proc_) {
        inside_->proc_->SetDraw();
    }
}

void StreamTask::Draw(uint8_t* img, int w, int h, int c, const int* regions, int num, const uint8_t* color) {
    std::vector<int32_t> tmpReg(3 * num);
    ::memcpy(tmpReg.data(), (void*)regions, 4 * 3 * num);
    double tmpBuf[4];
    ::memcpy(tmpBuf, color, 4 * sizeof(double));
        inside_->proc_->ResizeFunc(c, w, h, c, w, h, halide_type_of<uint8_t>());
        inside_->proc_->Draw(img, w, h, c, tmpReg.data(), num, reinterpret_cast<uint8_t*>(tmpBuf));
}

} // namespace task
} // namespace inspirecv



