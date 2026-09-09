#include "../../common/common.h"

#include "inspirecv/task/planning/compiled_conversion.h"

namespace {

inspirecv::task::internal::ConversionRequest MakeRequest(
  int source_channels, int destination_channels,
  halide_type_t type = halide_type_of<uint8_t>(),
  inspirecv::task::TensorLayout layout = inspirecv::task::TensorLayout::NHWC) {
    inspirecv::task::internal::ConversionRequest request;
    request.source_channels = source_channels;
    request.source_width = 32;
    request.source_height = 24;
    request.destination_channels = destination_channels;
    request.destination_width = 16;
    request.destination_height = 12;
    request.destination_type = type;
    request.destination_layout = layout;
    if (layout == inspirecv::task::TensorLayout::NC4HW4) {
        request.destination_stride = 16 * 4 * static_cast<int>(sizeof(float));
    }
    return request;
}

}  // namespace

TEST_CASE("task_compiled_conversion_selects_one_stable_execution_route",
          "[task][request-compiler][kernel-contract]") {
    using inspirecv::task::BGR;
    using inspirecv::task::GRAY;
    using inspirecv::task::RGB;
    using inspirecv::task::TensorLayout;
    using inspirecv::task::internal::CompileConversion;
    using inspirecv::task::internal::CompiledConversion;
    using inspirecv::task::internal::ExecutionRoute;
    using inspirecv::task::internal::PipelineConfig;

    inspirecv::task::Matrix identity;
    inspirecv::task::Matrix scaled;
    scaled.setScale(2.0f, 2.0f);

    PipelineConfig config;
    config.source_format = BGR;
    config.destination_format = BGR;
    CompiledConversion compiled;
    REQUIRE(CompileConversion(config, identity, MakeRequest(3, 3), false,
                              &compiled) == inspirecv::SUCCESS);
    REQUIRE(compiled.route == ExecutionRoute::kByteRows);
    REQUIRE(compiled.effective_source_stride == 32 * 3);
    REQUIRE(compiled.effective_destination_stride == 16 * 3);

    config.destination_format = RGB;
    REQUIRE(CompileConversion(config, identity, MakeRequest(3, 3), false,
                              &compiled) == inspirecv::SUCCESS);
    REQUIRE(compiled.route == ExecutionRoute::kPackedRows);

    REQUIRE(CompileConversion(config, scaled, MakeRequest(3, 3), false,
                              &compiled) == inspirecv::SUCCESS);
    REQUIRE(compiled.route == ExecutionRoute::kPackedTriple);

    REQUIRE(CompileConversion(
              config, scaled,
              MakeRequest(3, 4, halide_type_of<float>(), TensorLayout::NC4HW4),
              false, &compiled) == inspirecv::SUCCESS);
    REQUIRE(compiled.route == ExecutionRoute::kBlockedFloat);

    config.source_format = GRAY;
    config.destination_format = BGR;
    REQUIRE(CompileConversion(config, scaled, MakeRequest(1, 3), false,
                              &compiled) == inspirecv::SUCCESS);
    REQUIRE(compiled.route == ExecutionRoute::kGenericTiles);

    REQUIRE(CompileConversion(config, identity, MakeRequest(3, 3), true,
                              &compiled) == inspirecv::SUCCESS);
    REQUIRE(compiled.route == ExecutionRoute::kDrawRegions);
}

TEST_CASE("task_request_compiler_failure_does_not_replace_last_valid_program",
          "[task][request-compiler][reuse]") {
    using inspirecv::task::BGR;
    using inspirecv::task::HSV;
    using inspirecv::task::YUV_NV21;
    using inspirecv::task::internal::CompileConversion;
    using inspirecv::task::internal::CompiledConversion;
    using inspirecv::task::internal::ExecutionRoute;
    using inspirecv::task::internal::PipelineConfig;

    inspirecv::task::Matrix identity;
    PipelineConfig config;
    config.source_format = BGR;
    config.destination_format = BGR;
    CompiledConversion compiled;
    REQUIRE(CompileConversion(config, identity, MakeRequest(3, 3), false,
                              &compiled) == inspirecv::SUCCESS);
    REQUIRE(compiled.route == ExecutionRoute::kByteRows);

    config.source_format = YUV_NV21;
    config.destination_format = HSV;
    REQUIRE(CompileConversion(config, identity, MakeRequest(0, 3), false,
                              &compiled) == inspirecv::UNSUPPORTED_CONVERSION);
    REQUIRE(compiled.route == ExecutionRoute::kByteRows);
    REQUIRE(compiled.source_width == 32);
    REQUIRE(compiled.destination_width == 16);
}
