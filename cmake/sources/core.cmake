# Core facade and process-wide support sources.
#
# Keep this as a source manifest rather than a nested library: InspireFace
# consumes $<TARGET_OBJECTS:inspirecv> directly, so every translation unit must
# remain owned by the top-level inspirecv target.
set(INSPIRECV_CORE_SOURCES
    ${INSPIRECV_SRC_DIR}/core/api/point.cpp
    ${INSPIRECV_SRC_DIR}/core/api/rect.cpp
    ${INSPIRECV_SRC_DIR}/core/api/size.cpp
    ${INSPIRECV_SRC_DIR}/core/api/transform_matrix.cpp
    ${INSPIRECV_SRC_DIR}/core/api/image.cpp
    ${INSPIRECV_SRC_DIR}/core/runtime/acceleration_state.cpp
    ${INSPIRECV_SRC_DIR}/core/runtime/image_acceleration.cpp
    ${INSPIRECV_SRC_DIR}/core/runtime/costman.cpp
    ${INSPIRECV_SRC_DIR}/core/runtime/logging.cpp
    ${INSPIRECV_LIBRARY_INFO_SOURCE}
)

if(INSPIRECV_ENABLE_CUDA)
    list(APPEND INSPIRECV_CORE_SOURCES
        ${INSPIRECV_SRC_DIR}/core/cuda/image_geometry_backend.cu)
    set_source_files_properties(
        ${INSPIRECV_SRC_DIR}/core/cuda/image_geometry_backend.cu
        PROPERTIES COMPILE_OPTIONS "--fmad=false")
else()
    list(APPEND INSPIRECV_CORE_SOURCES
        ${INSPIRECV_SRC_DIR}/core/cuda/image_geometry_dispatch_stub.cc)
endif()
