# Task preprocessing source manifest.
#
# These lists describe responsibilities even before the corresponding files are
# moved into their final directories. All entries are still compiled directly
# into the top-level inspirecv target for InspireFace OBJECT compatibility.
set(INSPIRECV_TASK_API_SOURCES
    ${TASK_DIR}/api/legacy_stream_task.cc
    ${TASK_DIR}/api/pipeline.cc
    ${TASK_DIR}/api/public_conversions.cc
    ${TASK_DIR}/cuda/cuda_control.cc
    ${TASK_DIR}/cuda/cuda_auto_policy.cc
)

if(INSPIRECV_ENABLE_CUDA)
    set(INSPIRECV_TASK_CUDA_SOURCES
        ${TASK_DIR}/cuda/cuda_backend.cu)
    set_source_files_properties(${INSPIRECV_TASK_CUDA_SOURCES}
        PROPERTIES COMPILE_OPTIONS "--fmad=false")
else()
    set(INSPIRECV_TASK_CUDA_SOURCES
        ${TASK_DIR}/cuda/cuda_dispatch_stub.cc)
endif()

set(INSPIRECV_TASK_PLANNING_SOURCES
    ${TASK_DIR}/planning/compiled_conversion.cc
    ${TASK_DIR}/planning/conversion_request.cc
    ${TASK_DIR}/planning/kernel_registry.cc
    ${TASK_DIR}/planning/pixel_program.cc
)

set(INSPIRECV_TASK_RUNTIME_SOURCES
    ${TASK_DIR}/runtime/conversion_session.cc
)

set(INSPIRECV_TASK_EXECUTION_SOURCES
    ${TASK_DIR}/execution/execution_engine.cc
    ${TASK_DIR}/execution/row_schedule.cc
    ${TASK_DIR}/execution/row_routes.cc
    ${TASK_DIR}/execution/tile_pipeline.cc
)

set(INSPIRECV_TASK_PORTABLE_KERNEL_SOURCES
    ${TASK_DIR}/kernels/cpu/channel_ops.cc
    ${TASK_DIR}/kernels/cpu/color_ops.cc
    ${TASK_DIR}/kernels/cpu/sampling_ops.cc
    ${TASK_DIR}/kernels/cpu/tensor_writers.cc
    ${TASK_DIR}/kernels/cpu/yuv_ops.cc
)

set(INSPIRECV_TASK_GEOMETRY_SOURCES
    ${TASK_DIR}/geometry/matrix.cc
    ${TASK_DIR}/geometry/matrix_classification.cc
    ${TASK_DIR}/geometry/matrix_composition.cc
    ${TASK_DIR}/geometry/gnu_projective_math.cc
    ${TASK_DIR}/geometry/matrix_inverse.cc
    ${TASK_DIR}/geometry/matrix_mapping.cc
    ${TASK_DIR}/geometry/matrix_state.cc
    ${TASK_DIR}/geometry/matrix_transform.cc
    ${TASK_DIR}/geometry/projective_solver.cc
)

set(INSPIRECV_TASK_PLATFORM_SOURCES
    ${TASK_DIR}/platform/cpu_features.cc
    ${TASK_DIR}/platform/avx2_memcpy.cc
)

set(INSPIRECV_TASK_SOURCES
    ${TASK_DIR}/api/legacy_stream_task.cc
    ${TASK_DIR}/api/pipeline.cc
    ${TASK_DIR}/api/public_conversions.cc
    ${TASK_DIR}/cuda/cuda_control.cc
    ${TASK_DIR}/cuda/cuda_auto_policy.cc
    ${TASK_DIR}/planning/compiled_conversion.cc
    ${TASK_DIR}/planning/conversion_request.cc
    ${TASK_DIR}/runtime/conversion_session.cc
    ${TASK_DIR}/execution/execution_engine.cc
    ${TASK_DIR}/planning/kernel_registry.cc
    ${TASK_DIR}/planning/pixel_program.cc
    ${TASK_DIR}/execution/row_schedule.cc
    ${TASK_DIR}/execution/row_routes.cc
    ${TASK_DIR}/execution/tile_pipeline.cc
    ${INSPIRECV_TASK_PORTABLE_KERNEL_SOURCES}
    ${INSPIRECV_TASK_GEOMETRY_SOURCES}
    ${INSPIRECV_TASK_PLATFORM_SOURCES}
    ${INSPIRECV_TASK_CUDA_SOURCES}
)

set(INSPIRECV_TASK_FAST_MATH_SOURCES
    ${TASK_DIR}/execution/execution_engine.cc
    ${TASK_DIR}/execution/row_schedule.cc
    ${TASK_DIR}/execution/tile_pipeline.cc
    ${TASK_DIR}/kernels/cpu/color_ops.cc
    ${TASK_DIR}/kernels/cpu/sampling_ops.cc
    ${TASK_DIR}/kernels/cpu/tensor_writers.cc
    ${INSPIRECV_TASK_GEOMETRY_SOURCES}
)

set(INSPIRECV_TASK_AVX2_SOURCES
    ${TASK_DIR}/platform/avx2_memcpy.cc
)

set(INSPIRECV_TASK_NEON_SOURCES
    ${TASK_DIR}/kernels/arm/neon_kernels.cc
)
