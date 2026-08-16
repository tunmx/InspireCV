#include <inspirecv/task/pipeline.h>

#include <climits>
#include <memory>
#include <utility>

#include <inspirecv/task/core/preprocess_types.h>
#include <inspirecv/task/task_status.h>

#include "inspirecv/task/api/public_conversions.h"
#include "inspirecv/task/runtime/conversion_session.h"

namespace inspirecv {
namespace task {
namespace {

bool ToInternalTensor(const TensorBuffer& source, TensorView* destination) {
    if (!destination) return false;
    destination->data = source.data;
    destination->width = source.width;
    destination->height = source.height;
    destination->channels = source.channels;
    switch (source.element_type) {
        case ElementType::kUInt8:
            destination->type = halide_type_of<uint8_t>();
            break;
        case ElementType::kFloat32:
            destination->type = halide_type_of<float>();
            break;
        default:
            return false;
    }
    switch (source.order) {
        case TensorOrder::kHwc:
            destination->layout = TensorLayout::NHWC;
            break;
        case TensorOrder::kChw:
            destination->layout = TensorLayout::NCHW;
            break;
        case TensorOrder::kChannelPacked4:
            destination->layout = TensorLayout::NC4HW4;
            break;
        default:
            return false;
    }
    destination->rowStride = source.row_stride_bytes;
    destination->channelStride = source.channel_stride_bytes;
    return true;
}

}  // namespace

struct Pipeline::Impl {
    explicit Impl(const PipelineOptions& options)
        : output_format(options.output_format) {
        configuration_status = api_internal::ConfigurePipeline(options, &config);
        if (configuration_status == Status::kOk) {
            session.reset(new internal::ConversionSession(config));
        }
    }

    internal::PipelineConfig config;
    std::unique_ptr<internal::ConversionSession> session;
    PixelFormat output_format;
    Status configuration_status = Status::kInvalidArgument;
};

Pipeline::Pipeline(const PipelineOptions& options)
    : impl_(new Impl(options)) {}

Pipeline::~Pipeline() = default;

Pipeline::Pipeline(Pipeline&& other) noexcept
    : impl_(std::move(other.impl_)) {}

Pipeline& Pipeline::operator=(Pipeline&& other) noexcept {
    if (this != &other) impl_ = std::move(other.impl_);
    return *this;
}

Status Pipeline::SetTransform(const TransformMatrix& transform) {
    if (!impl_) return Status::kInvalidArgument;
    if (impl_->configuration_status != Status::kOk) {
        return impl_->configuration_status;
    }
    if (!impl_->session) return Status::kInvalidArgument;
    const Matrix candidate = api_internal::ToTaskMatrix(transform);
    if (!candidate.invert(nullptr)) {
        return Status::kNonInvertibleTransform;
    }
    impl_->session->SetMatrix(candidate);
    return Status::kOk;
}

void Pipeline::SetBorderValue(uint8_t value) {
    if (impl_ && impl_->session) impl_->session->SetPadding(value);
}

Status Pipeline::ConfigurationStatus() const noexcept {
    return impl_ ? impl_->configuration_status : Status::kInvalidArgument;
}

PixelFormat Pipeline::OutputFormat() const noexcept {
    return impl_ ? impl_->output_format : PixelFormat::kRgba;
}

ExecutionBackend Pipeline::LastExecutionBackend() const noexcept {
    return impl_ && impl_->session
             ? impl_->session->LastExecutionBackend()
             : ExecutionBackend::kCpu;
}

Status Pipeline::Run(const Image& source, int output_width, int output_height,
                     Image* destination) {
    if (!impl_) return Status::kInvalidArgument;
    if (impl_->configuration_status != Status::kOk) {
        return impl_->configuration_status;
    }
    if (!destination || source.Empty() ||
        output_width <= 0 || output_height <= 0) {
        return Status::kInvalidArgument;
    }
    const int source_channels =
      internal::ChannelCount(impl_->config.source_format);
    if (source_channels <= 0 || source.Channels() != source_channels) {
        return Status::kInvalidArgument;
    }

    RawImageView raw_source;
    raw_source.data = source.Data();
    raw_source.width = source.Width();
    raw_source.height = source.Height();
    raw_source.row_stride_bytes =
      static_cast<size_t>(source.Width()) * source.Channels();
    return Run(raw_source, output_width, output_height, destination);
}

Status Pipeline::Run(const RawImageView& source, int output_width,
                     int output_height, Image* destination) {
    if (!impl_) return Status::kInvalidArgument;
    if (impl_->configuration_status != Status::kOk) {
        return impl_->configuration_status;
    }
    if (!impl_->session || !destination ||
        !source.data || source.width <= 0 || source.height <= 0 ||
        output_width <= 0 || output_height <= 0) {
        return Status::kInvalidArgument;
    }
    const int destination_channels =
      internal::ChannelCount(impl_->config.destination_format);
    if (destination_channels <= 0) return Status::kInvalidArgument;

    Image candidate =
      Image::Create(output_width, output_height, destination_channels);
    if (candidate.Empty()) return Status::kInvalidArgument;
    const Status status = RunInto(source, &candidate);
    if (status == Status::kOk) *destination = std::move(candidate);
    return status;
}

Status Pipeline::RunInto(const Image& source, Image* destination) {
    if (!impl_) return Status::kInvalidArgument;
    if (impl_->configuration_status != Status::kOk) {
        return impl_->configuration_status;
    }
    if (!destination || source.Empty() ||
        &source == destination || source.Data() == destination->Data()) {
        return Status::kInvalidArgument;
    }
    const int source_channels =
      internal::ChannelCount(impl_->config.source_format);
    if (source_channels <= 0 || source.Channels() != source_channels) {
        return Status::kInvalidArgument;
    }

    RawImageView raw_source;
    raw_source.data = source.Data();
    raw_source.width = source.Width();
    raw_source.height = source.Height();
    raw_source.row_stride_bytes =
      static_cast<size_t>(source.Width()) * source.Channels();
    return RunInto(raw_source, destination);
}

Status Pipeline::RunInto(const RawImageView& source, Image* destination) {
    if (!impl_) return Status::kInvalidArgument;
    if (impl_->configuration_status != Status::kOk) {
        return impl_->configuration_status;
    }
    if (!impl_->session || !destination ||
        destination->Empty() || !source.data || source.width <= 0 ||
        source.height <= 0 ||
        source.row_stride_bytes > static_cast<size_t>(INT32_MAX) ||
        source.data == destination->Data()) {
        return Status::kInvalidArgument;
    }
    const int destination_channels =
      internal::ChannelCount(impl_->config.destination_format);
    if (destination_channels <= 0 ||
        destination->Channels() != destination_channels ||
        destination->Width() <= 0 || destination->Height() <= 0) {
        return Status::kInvalidArgument;
    }

    return api_internal::ToPublicStatus(impl_->session->Convert(
      source.data, source.width, source.height,
      static_cast<int>(source.row_stride_bytes),
      destination->GetInternalImage(), destination->Width(),
      destination->Height(), destination_channels, 0,
      halide_type_of<uint8_t>()));
}

Status Pipeline::Run(const Image& source,
                     const TensorBuffer& destination) {
    if (!impl_) return Status::kInvalidArgument;
    if (impl_->configuration_status != Status::kOk) {
        return impl_->configuration_status;
    }
    if (source.Empty()) {
        return Status::kInvalidArgument;
    }
    const int source_channels =
      internal::ChannelCount(impl_->config.source_format);
    if (source_channels <= 0 || source.Channels() != source_channels) {
        return Status::kInvalidArgument;
    }
    RawImageView raw_source;
    raw_source.data = source.Data();
    raw_source.width = source.Width();
    raw_source.height = source.Height();
    raw_source.row_stride_bytes =
      static_cast<size_t>(source.Width() * source.Channels());
    return Run(raw_source, destination);
}

Status Pipeline::Run(const RawImageView& source,
                     const TensorBuffer& destination) {
    if (!impl_) return Status::kInvalidArgument;
    if (impl_->configuration_status != Status::kOk) {
        return impl_->configuration_status;
    }
    if (!impl_->session || !source.data ||
        source.width <= 0 || source.height <= 0 ||
        source.row_stride_bytes > static_cast<size_t>(INT32_MAX)) {
        return Status::kInvalidArgument;
    }
    TensorView internal_destination;
    if (!ToInternalTensor(destination, &internal_destination)) {
        return Status::kInvalidArgument;
    }
    return api_internal::ToPublicStatus(impl_->session->Convert(
      source.data, source.width, source.height,
      static_cast<int>(source.row_stride_bytes), internal_destination));
}

const char* StatusMessage(Status status) {
    switch (status) {
        case Status::kOk:
            return "ok";
        case Status::kInvalidArgument:
            return "invalid argument";
        case Status::kUnsupportedSampling:
            return "unsupported sampling mode";
        case Status::kUnsupportedConversion:
            return "unsupported pixel conversion";
        case Status::kUnsupportedElementType:
            return "unsupported element type";
        case Status::kAccelerationDisabled:
            return "acceleration disabled";
        case Status::kAccelerationUnavailable:
            return "acceleration unavailable";
        case Status::kAccelerationFailure:
            return "acceleration failure";
        case Status::kNonInvertibleTransform:
            return "transform is not invertible";
    }
    return "unknown task status";
}

}  // namespace task
}  // namespace inspirecv
