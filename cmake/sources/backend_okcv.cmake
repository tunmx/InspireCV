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
    list(APPEND INSPIRECV_OKCV_SOURCES ${INSPIRECV_OKCV_X86_SOURCES})
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        set_property(SOURCE ${INSPIRECV_OKCV_X86_SOURCES} APPEND
                     PROPERTY COMPILE_OPTIONS -mssse3)
    elseif(MSVC AND CMAKE_SIZEOF_VOID_P EQUAL 4)
        set_property(SOURCE ${INSPIRECV_OKCV_X86_SOURCES} APPEND
                     PROPERTY COMPILE_OPTIONS /arch:SSE2)
    endif()
endif()
