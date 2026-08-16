#include <inspirecv/task/core/stream_task.h>
#include <inspirecv/task/runtime/conversion_session.h>
#include <algorithm>
#include <memory>

namespace inspirecv {
namespace task {
namespace {

internal::PipelineConfig ToPipelineConfig(const StreamTask::Config& config) {
    internal::PipelineConfig pipeline;
    pipeline.filter = config.filterType;
    pipeline.source_format = config.sourceFormat;
    pipeline.destination_format = config.destFormat;
    std::copy_n(config.mean, 4, pipeline.mean);
    std::copy_n(config.normal, 4, pipeline.scale);
    pipeline.wrap = config.wrap;
    pipeline.preserve_identity_float_order = config.preserveIdentityFloatOrder;
    return pipeline;
}

}  // namespace

struct StreamTask::Inside {
    explicit Inside(const Config& task_config)
        : session(new internal::ConversionSession(ToPipelineConfig(task_config))) {}

    std::unique_ptr<internal::ConversionSession> session;
};

void StreamTask::Destroy(StreamTask* task) { delete task; }

StreamTask::~StreamTask() { delete inside_; }

StreamTask::StreamTask(const Config& config) : inside_(new Inside(config)) {}

StreamTask* StreamTask::Create(const Config& config) { return new StreamTask(config); }

void StreamTask::SetMatrix(const task::Matrix& matrix) {
    inside_->session->SetMatrix(matrix);
    transform_ = matrix;
}

TaskStatus StreamTask::Convert(const uint8_t* source, int iw, int ih, int stride, void* dest,
                               int ow, int oh, int outputBpp, int outputStride,
                               halide_type_t type) {
    inside_->session->SetPadding(padding_value_);
    return inside_->session->Convert(source, iw, ih, stride, dest, ow, oh,
                                     outputBpp, outputStride, type);
}

TaskStatus StreamTask::Convert(const uint8_t* source, int iw, int ih, int stride,
                               const TensorView& output) {
    inside_->session->SetPadding(padding_value_);
    return inside_->session->Convert(source, iw, ih, stride, output);
}

void StreamTask::SetDraw() {
    if (inside_ && inside_->session) {
        inside_->session->EnableDrawing();
    }
}

void StreamTask::Draw(uint8_t* img, int w, int h, int c, const int* regions, int num,
                      const uint8_t* color) {
    if (!inside_ || !inside_->session) return;
    inside_->session->DrawRegions(img, w, h, c, regions, num, color);
}

}  // namespace task
}  // namespace inspirecv
