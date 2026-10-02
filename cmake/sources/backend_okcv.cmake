# Standalone OKCV backend sources.
#
# Production sources are deliberately explicit. A recursive glob can silently
# change the object payload embedded by InspireFace when an unrelated .cpp file
# appears below this directory.
set(INSPIRECV_OKCV_SOURCES
    ${OKCV_DIR}/base/types.cpp
    ${OKCV_DIR}/bitmap/bitmap.cpp
    ${OKCV_DIR}/geometry/cv_point.cpp
    ${OKCV_DIR}/geometry/cv_transform_matrix.cpp
)

if(ST_TARGET_PROCESSOR MATCHES "^(x86_64|X86_64|AMD64|i686|i386)$")
    set(INSPIRECV_OKCV_X86_SOURCES
        ${OKCV_DIR}/kernels/x86/u8c3_ops.cc)
    set(INSPIRECV_OKCV_AVX2_SOURCES
        ${OKCV_DIR}/kernels/x86/image_ops_avx2.cc
        ${OKCV_DIR}/kernels/x86/image_affine_avx2.cc)
    list(APPEND INSPIRECV_OKCV_SOURCES ${INSPIRECV_OKCV_X86_SOURCES}
                                     ${INSPIRECV_OKCV_AVX2_SOURCES})
    list(APPEND INSPIRECV_PRIVATE_DEFINITIONS
         INSPIRECV_HAVE_IMAGE_AVX2_KERNELS)
    inspirecv_configure_x86_kernels(SSSE3 ${INSPIRECV_OKCV_X86_SOURCES})
    inspirecv_configure_x86_kernels(AVX2 ${INSPIRECV_OKCV_AVX2_SOURCES})
endif()
