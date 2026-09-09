cmake_minimum_required(VERSION 3.15)
project(InspireCV VERSION 1.0.2 LANGUAGES C CXX)

set(INSPIRECV_CXX_STANDARD 14)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(INSPIRECV_IS_TOP_LEVEL ON)
else()
    set(INSPIRECV_IS_TOP_LEVEL OFF)
endif()

# Pick a useful default only for a standalone, single-config build. When
# InspireCV is brought in with add_subdirectory(), the host project remains in
# full control of its build type and toolchain flags.
if(INSPIRECV_IS_TOP_LEVEL AND NOT CMAKE_CONFIGURATION_TYPES AND
   NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
                 Debug Release RelWithDebInfo MinSizeRel)
endif()

# == OpenCV Backend Mode ==
# OpenCV Backend Mode:
# When enabled, InspireCV will use OpenCV as the underlying image processing engine.
# This provides access to OpenCV's extensive functionality and optimizations.
# When disabled, InspireCV will use its own lightweight OKCV backend implementation.
# OKCV provides basic image processing capabilities with minimal dependencies.
option(INSPIRECV_BACKEND_OPENCV "If you want to use OpenCV as the CV backend" OFF)

# == OKCV Backend Mode ==
# OKCV Backend Mode:
# When enabled, InspireCV will use its own lightweight OKCV backend implementation.
# OKCV provides basic image processing capabilities with minimal dependencies.
# When using the OKCV backend, you can optionally enable OpenCV's I/O and GUI APIs for debugging convenience.
# This allows you to use OpenCV's image reading/writing functions and visualization tools like imshow() 
# while still using OKCV as the core processing engine.
# These OpenCV features are typically enabled during development to help with debugging and visualization.
option(INSPIRECV_BACKEND_OKCV_USE_OPENCV "Turn on some OpenCV api support inside OKCV" OFF)
option(INSPIRECV_BACKEND_OKCV_USE_OPENCV_IO "Let okcv internally use OpenCV's image codec and read-write api." OFF)
option(INSPIRECV_BACKEND_OKCV_USE_OPENCV_GUI "Make okcv internally dependent on some of OpenCV's curtain functions, like imshow." OFF)

option(INSPIRECV_BUILD_OBJECT_LIBS "Build object libs" ON)
option(INSPIRECV_BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(INSPIRECV_BUILD_TESTS "Build tests" OFF)
option(INSPIRECV_BUILD_SAMPLE "Build sample" OFF)
option(INSPIRECV_BUILD_EXAMPLES "Build user-facing examples in root example/ directory" OFF)
option(INSPIRECV_BUILD_CPU_BENCHMARKS
       "Build the direct InspireCV versus OpenCV CPU benchmark" OFF)
option(INSPIRECV_INSTALL "Generate install rules and a CMake package" ON)

# == Task preprocessing options ==
option(INSPIRECV_TASK_ENABLE_ARM_NEON "Enable ARM NEON imageprocess kernels" ON)
option(INSPIRECV_TASK_DISABLE_TILING "Disable row tiling (force single-tile per row)" OFF)
option(INSPIRECV_ENABLE_CUDA "Build the optional CUDA acceleration backends" OFF)
option(INSPIRECV_TASK_ENABLE_CUDA
       "Compatibility switch for enabling InspireCV CUDA acceleration" OFF)
if(INSPIRECV_TASK_ENABLE_CUDA)
    set(INSPIRECV_ENABLE_CUDA ON CACHE BOOL
        "Build the optional CUDA acceleration backends" FORCE)
endif()
set(INSPIRECV_CUDA_ARCHITECTURES "86" CACHE STRING
    "CUDA architectures compiled into the optional CUDA backends")

set(INSPIRECV_PRIVATE_DEFINITIONS "")
set(INSPIRECV_PRIVATE_OPTIONS "")

set(EIGEN3_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/Eigen-3.4.0-Headers")

if(INSPIRECV_BACKEND_OKCV_USE_OPENCV_IO OR INSPIRECV_BACKEND_OKCV_USE_OPENCV_GUI)
    set(INSPIRECV_BACKEND_OKCV_USE_OPENCV ON)
endif()

set(NEED_LINK_LIBS "")

# logging.cpp writes to logcat on Android. Keep the platform library on the
# target's public link interface so static and object-library consumers do not
# need to rediscover this implementation dependency.
if(ANDROID)
    list(APPEND NEED_LINK_LIBS log)
endif()

if(INSPIRECV_ENABLE_CUDA)
    if(CMAKE_VERSION VERSION_LESS 3.18)
        message(FATAL_ERROR
            "INSPIRECV_ENABLE_CUDA requires CMake 3.18 or newer")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    list(APPEND INSPIRECV_PRIVATE_DEFINITIONS
         INSPIRECV_HAS_CUDA=1 INSPIRECV_TASK_HAS_CUDA=1)
    list(APPEND NEED_LINK_LIBS CUDA::cudart_static)
endif()

set(OTHER_SUFFIX "")
if(INSPIRECV_BACKEND_OKCV_USE_OPENCV)
    find_package(OpenCV REQUIRED)
    list(APPEND INSPIRECV_PRIVATE_DEFINITIONS
         INSPIRECV_BACKEND_OKCV_USE_OPENCV)
    if (INSPIRECV_BACKEND_OKCV_USE_OPENCV_IO)
        list(APPEND INSPIRECV_PRIVATE_DEFINITIONS
             INSPIRECV_BACKEND_OKCV_USE_OPENCV_IO)
        set(OTHER_SUFFIX "${OTHER_SUFFIX}@OpenCV-IO")
    else()
        set(OTHER_SUFFIX "${OTHER_SUFFIX}@OpenCV")
    endif()
    if (INSPIRECV_BACKEND_OKCV_USE_OPENCV_GUI)
        list(APPEND INSPIRECV_PRIVATE_DEFINITIONS
             INSPIRECV_BACKEND_OKCV_USE_OPENCV_GUI)
        set(OTHER_SUFFIX "${OTHER_SUFFIX}@OpenCV-GUI")
    endif()
    set(NEED_LINK_LIBS ${NEED_LINK_LIBS} ${OpenCV_LIBS})

endif()

set(INSPIRECV_BACKEND "OKCV")
if(INSPIRECV_BACKEND_OPENCV)
    set(INSPIRECV_BACKEND "OpenCV")
    message(STATUS "InspireCV Backend: ${INSPIRECV_BACKEND}")
    list(APPEND INSPIRECV_PRIVATE_DEFINITIONS INSPIRECV_BACKEND_OPENCV)
    find_package(OpenCV REQUIRED)
    set(NEED_LINK_LIBS ${NEED_LINK_LIBS} ${OpenCV_LIBS})
endif()

# Keep the generated runtime record aligned with the CMake project version.
set(INSPIRECV_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(INSPIRECV_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(INSPIRECV_VERSION_PATCH ${PROJECT_VERSION_PATCH})
string(TIMESTAMP BUILD_TIMESTAMP "%Y-%m-%d")

set(VERSION_SUFFIX "")
if(INSPIRECV_BACKEND STREQUAL "OKCV")
    if(INSPIRECV_BACKEND_OKCV_USE_OPENCV_IO OR INSPIRECV_BACKEND_OKCV_USE_OPENCV_GUI OR INSPIRECV_BACKEND_OKCV_USE_OPENCV)
        set(VERSION_SUFFIX "@General(With OpenCV)")
    else()
        set(VERSION_SUFFIX "@General")
    endif()
else()
    set(VERSION_SUFFIX "@OpenCV")
endif()

set(INSPIRECV_VERSION_STRING 
    "InspireCV v${INSPIRECV_VERSION_MAJOR}.${INSPIRECV_VERSION_MINOR}.${INSPIRECV_VERSION_PATCH}${VERSION_SUFFIX} - Build Time: ${BUILD_TIMESTAMP}")

set(INSPIRECV_BACKEND_STRING "")
if(INSPIRECV_BACKEND STREQUAL "OpenCV")
    set(INSPIRECV_BACKEND_STRING "OpenCV")
else()
    set(INSPIRECV_BACKEND_STRING "OKCV")
endif()

# Start of the project
set(INSPIRECV_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(INSPIRECV_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/inspirecv)
set(CPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/)
set(OKCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/inspirecv/backends/okcv)

# Generate the single library identity after Task feature detection so the
# common record includes both CV backend and preprocessing acceleration data.
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(INSPIRECV_LIBRARY_INFO_SOURCE
    "${CMAKE_CURRENT_BINARY_DIR}/generated/library_info.cpp")

set(INSPIRECV_PRIVATE_INCLUDE_DIRS
    ${EIGEN3_INCLUDE_DIRS}
    ${INSPIRECV_SRC_DIR}
    ${INSPIRECV_INCLUDE_DIRS}
    ${INSPIRECV_INCLUDE_DIRS}/inspirecv
    ${CPP_DIR}
)

set(INSPIRECV_CONSUMER_INCLUDE_DIRS
    ${INSPIRECV_PRIVATE_INCLUDE_DIRS}
)

# ================= Task preprocessing =================
# Only algorithms are built (no executables/benches here).
set(TASK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/inspirecv/task)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/sources/task.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_task_dependencies.cmake)
inspirecv_check_task_dependency_boundaries(${TASK_DIR})
inspirecv_check_core_dependency_boundaries(${INSPIRECV_SRC_DIR}/core)
inspirecv_check_backend_dependency_boundaries(${INSPIRECV_SRC_DIR}/backends)

# CMAKE_SYSTEM_PROCESSOR describes the host in a normal macOS build, even when
# CMAKE_OSX_ARCHITECTURES requests a single cross-architecture target (for
# example x86_64 under Rosetta on Apple Silicon). Select kernels from the target
# architecture in that case so an x86 build never receives ARM intrinsics, and
# vice versa.
set(ST_TARGET_PROCESSOR "${CMAKE_SYSTEM_PROCESSOR}")
if(APPLE AND CMAKE_OSX_ARCHITECTURES)
    list(LENGTH CMAKE_OSX_ARCHITECTURES ST_OSX_ARCH_COUNT)
    if(ST_OSX_ARCH_COUNT EQUAL 1)
        list(GET CMAKE_OSX_ARCHITECTURES 0 ST_TARGET_PROCESSOR)
    endif()
endif()
# IPO/LTO
include(CheckIPOSupported)
check_ipo_supported(RESULT st_ipo_supported OUTPUT st_ipo_error)

# Generated config
set(ST_TILE_WIDTH 256)
set(ST_HAVE_LTO 0)
string(TOUPPER "${CMAKE_BUILD_TYPE}" ST_BUILD_TYPE_UPPER)
if(st_ipo_supported AND
   (CMAKE_INTERPROCEDURAL_OPTIMIZATION OR
    CMAKE_INTERPROCEDURAL_OPTIMIZATION_${ST_BUILD_TYPE_UPPER}))
    set(ST_HAVE_LTO 1)
endif()

set(ST_HAVE_SSE 0)
set(ST_HAVE_AVX2 0)
set(ST_HAVE_NEON 0)
set(ST_DISABLE_TILING 0)
if(INSPIRECV_TASK_DISABLE_TILING)
    set(ST_DISABLE_TILING 1)
endif()

# Gather a diagnostic flag snapshot without changing host-project flags.
if(CMAKE_CONFIGURATION_TYPES)
    set(ST_ACTIVE_CXX_FLAGS "multi-config; ${CMAKE_CXX_FLAGS}")
else()
    string(TOUPPER "${CMAKE_BUILD_TYPE}" ST_BUILD_TYPE_UPPER)
    set(ST_ACTIVE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${ST_BUILD_TYPE_UPPER}}")
    string(STRIP "${ST_ACTIVE_CXX_FLAGS}" ST_ACTIVE_CXX_FLAGS)
endif()

# The compatibility reference builds sampling, transform, and matrix arithmetic
# with fast floating-point semantics. Keep these Task units under the same
# contract so bilinear interpolation remains bit-exact for existing callers.
# Scope this to the preprocessing translation units: project-wide
# fast-math would weaken NaN/infinity handling in unrelated geometry/Eigen code.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    # These translation units have a frozen, bit-level compatibility contract
    # that was originally produced with O3 in every build configuration.
    # Keep that contract local to Task instead of forcing O3 on the host and
    # the rest of InspireCV.
    set(ST_TASK_FAST_MATH_OPTIONS "-O3" "-ffast-math")
    # The frozen x86 compatibility contract does not use fused multiply-add.
    # A project-wide -mfma must not change preprocessing arithmetic or its
    # last-bit rounding.
    if(ST_TARGET_PROCESSOR MATCHES "^(x86_64|X86_64|AMD64|i686|i386)$")
        list(APPEND ST_TASK_FAST_MATH_OPTIONS "-ffp-contract=off")
    endif()
    set_property(SOURCE ${INSPIRECV_TASK_FAST_MATH_SOURCES} APPEND
                 PROPERTY COMPILE_OPTIONS ${ST_TASK_FAST_MATH_OPTIONS})
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
       ST_TARGET_PROCESSOR MATCHES "^(x86_64|X86_64|AMD64|i686|i386)$")
        # GCC 13 SLP replaces the two scanline divisions with an approximate
        # reciprocal under fast-math (even 5/5 can become 0x3f7fffff). These
        # coordinates are an observable sampling contract, not an approximation.
        set_property(SOURCE ${TASK_DIR}/execution/row_schedule.cc APPEND
                     PROPERTY COMPILE_OPTIONS -fno-unsafe-math-optimizations)
    endif()
    # LTO historically has its own evaluation contract. Keep that path intact;
    # stabilize the non-LTO ARM Release path affected by newer Clang versions.
    set_property(SOURCE ${INSPIRECV_TASK_FAST_MATH_SOURCES} APPEND
                 PROPERTY COMPILE_DEFINITIONS
                 INSPIRECV_TASK_PRESERVE_LTO_EVALUATION=${ST_HAVE_LTO})
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT ST_HAVE_LTO AND
       ST_TARGET_PROCESSOR MATCHES "^(arm64|aarch64|ARM64)$")
        # Keep within-expression FMA, but honor the inverse's local
        # contract(off) scope instead of fusing across statement boundaries.
        set_property(SOURCE ${TASK_DIR}/geometry/matrix_inverse.cc APPEND
                     PROPERTY COMPILE_OPTIONS -ffp-contract=on)
    endif()
elseif(MSVC)
    set_property(SOURCE ${INSPIRECV_TASK_FAST_MATH_SOURCES} APPEND
                 PROPERTY COMPILE_OPTIONS /O2 /fp:fast)
endif()

set(INSPIRECV_SRC_FILES ${INSPIRECV_SRC_FILES} ${INSPIRECV_TASK_SOURCES})

# CPU feature paths
if (ST_TARGET_PROCESSOR MATCHES "^(x86_64|X86_64|AMD64|i686|i386)$")
    message(STATUS "InspireCV Task: enabling x86 SSE4.1 optimizations")
    list(APPEND INSPIRECV_PRIVATE_DEFINITIONS INSPIRECV_TASK_USE_SSE)
    set(ST_HAVE_SSE 1)
    set(INSPIRECV_TASK_SSE_SOURCES
        ${TASK_DIR}/kernels/cpu/channel_ops.cc
        ${TASK_DIR}/kernels/cpu/sampling_ops.cc
        ${TASK_DIR}/kernels/cpu/tensor_writers.cc
        ${TASK_DIR}/kernels/cpu/yuv_ops.cc)
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        set_property(SOURCE ${INSPIRECV_TASK_SSE_SOURCES} APPEND
                     PROPERTY COMPILE_OPTIONS -msse4.1)
    elseif(MSVC)
        # x64 MSVC has an SSE2 baseline and exposes SSE4.1 intrinsics without
        # a separate /arch switch. Runtime CPUID checks guard their use.
    endif()
    # Keep AVX2 in isolated translation units and dispatch at runtime. This
    # preserves the SSE4.1 baseline when full-project AVX2 is disabled.
    set(ST_AVX2_SOURCES ${INSPIRECV_TASK_AVX2_SOURCES})
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        set_property(SOURCE ${ST_AVX2_SOURCES} APPEND
                     PROPERTY COMPILE_OPTIONS -mavx2)
    elseif (MSVC)
        set_property(SOURCE ${ST_AVX2_SOURCES} APPEND
                     PROPERTY COMPILE_OPTIONS /arch:AVX2)
    endif()
    set(ST_HAVE_AVX2 1)
    # Building every source with AVX2 removes the runtime baseline and can make
    # the library unusable on older x86 CPUs (or under Rosetta). Keep it an
    # explicit opt-in; isolated AVX2 helpers above remain runtime-dispatched.
    option(INSPIRECV_ENABLE_AVX2 "Enable AVX2 for OKCV kernels and targets on x86" OFF)
    if (INSPIRECV_ENABLE_AVX2)
        message(STATUS "INSPIRECV: enabling AVX2 (-mavx2 -mfma) for all C++ sources")
        if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
            list(APPEND INSPIRECV_PRIVATE_OPTIONS -mavx2 -mfma)
        elseif (MSVC)
            list(APPEND INSPIRECV_PRIVATE_OPTIONS /arch:AVX2)
        endif()
    endif()
endif()

if (INSPIRECV_TASK_ENABLE_ARM_NEON)
    if (ST_TARGET_PROCESSOR MATCHES "^(aarch64|arm64)")
        message(STATUS "InspireCV Task: enabling ARM64 NEON kernels")
        list(APPEND INSPIRECV_PRIVATE_DEFINITIONS
             INSPIRECV_TASK_USE_NEON INSPIRECV_TASK_ENABLE_ARM_NEON)
        set(ST_HAVE_NEON 1)
        list(APPEND INSPIRECV_SRC_FILES
             ${INSPIRECV_TASK_NEON_SOURCES})
    elseif (ST_TARGET_PROCESSOR MATCHES "^(arm|armv7|armv7l|armhf|armv8l)")
        message(STATUS "InspireCV Task: enabling ARM32 NEON kernels")
        list(APPEND INSPIRECV_PRIVATE_DEFINITIONS
             INSPIRECV_TASK_USE_NEON INSPIRECV_TASK_ENABLE_ARM_NEON)
        set(ST_HAVE_NEON 1)
        list(APPEND INSPIRECV_SRC_FILES
             ${INSPIRECV_TASK_NEON_SOURCES})
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
            set_property(SOURCE ${INSPIRECV_TASK_NEON_SOURCES} APPEND
                         PROPERTY COMPILE_OPTIONS -mfpu=neon)
        endif()
    endif()
endif()
# Generate the unified InspireCV identity after feature detection.
configure_file(${INSPIRECV_SRC_DIR}/core/version/library_info.cpp.in
               ${INSPIRECV_LIBRARY_INFO_SOURCE} @ONLY)
# ================= End Task preprocessing =================

if(INSPIRECV_BACKEND STREQUAL "OKCV")
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/sources/backend_okcv.cmake)
    set(INSPIRECV_SRC_FILES ${INSPIRECV_SRC_FILES} ${INSPIRECV_OKCV_SOURCES})
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/sources/core.cmake)
set(INSPIRECV_SRC_FILES ${INSPIRECV_SRC_FILES} ${INSPIRECV_CORE_SOURCES})

# message(STATUS "INSPIRECV_SRC_FILES: ${INSPIRECV_SRC_FILES}")

if(INSPIRECV_BUILD_OBJECT_LIBS)
    add_library(inspirecv OBJECT ${INSPIRECV_SRC_FILES})
else()
    if(INSPIRECV_BUILD_SHARED_LIBS)
        add_library(inspirecv SHARED ${INSPIRECV_SRC_FILES})
    else()
        add_library(inspirecv STATIC ${INSPIRECV_SRC_FILES})
    endif()
endif()
set_target_properties(inspirecv PROPERTIES
    CXX_STANDARD ${INSPIRECV_CXX_STANDARD}
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
    POSITION_INDEPENDENT_CODE ON)
if(INSPIRECV_ENABLE_CUDA)
    set_target_properties(inspirecv PROPERTIES
        CUDA_STANDARD ${INSPIRECV_CXX_STANDARD}
        CUDA_STANDARD_REQUIRED YES
        CUDA_EXTENSIONS NO)
endif()
target_compile_features(inspirecv PUBLIC cxx_std_14)
target_compile_definitions(inspirecv PRIVATE ${INSPIRECV_PRIVATE_DEFINITIONS})
target_compile_options(inspirecv PRIVATE ${INSPIRECV_PRIVATE_OPTIONS})
# Preserve the historical public-symbol contract without leaking a directory-
# wide definition into the host project.
if(WIN32)
    if(NOT INSPIRECV_BUILD_OBJECT_LIBS AND INSPIRECV_BUILD_SHARED_LIBS)
        target_compile_definitions(inspirecv
            PRIVATE "INSPIRECV_API=__declspec(dllexport)"
            INTERFACE "INSPIRECV_API=__declspec(dllimport)")
    endif()
else()
    target_compile_definitions(inspirecv PUBLIC
        "INSPIRECV_API=__attribute__((visibility(\"default\")))")
endif()
target_include_directories(inspirecv
    PRIVATE ${INSPIRECV_PRIVATE_INCLUDE_DIRS}
    PUBLIC
        $<BUILD_INTERFACE:${INSPIRECV_INCLUDE_DIRS}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
if(OpenCV_FOUND)
    target_include_directories(inspirecv PRIVATE ${OpenCV_INCLUDE_DIRS})
endif()
target_link_libraries(inspirecv PUBLIC ${NEED_LINK_LIBS} ${EXTENSION_LIBS})
if(INSPIRECV_ENABLE_CUDA)
    set_property(TARGET inspirecv PROPERTY CUDA_ARCHITECTURES
                 "${INSPIRECV_CUDA_ARCHITECTURES}")
endif()
if(WIN32 AND NOT INSPIRECV_BUILD_OBJECT_LIBS AND INSPIRECV_BUILD_SHARED_LIBS)
    set_property(TARGET inspirecv PROPERTY WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

# The build-tree target remains an OBJECT library by default because
# InspireFace embeds it directly. A normal archive is produced alongside it
# for installation and find_package() consumers.
if(INSPIRECV_BUILD_OBJECT_LIBS)
    add_library(inspirecv_package STATIC $<TARGET_OBJECTS:inspirecv>)
    set_target_properties(inspirecv_package PROPERTIES
        OUTPUT_NAME inspirecv
        EXPORT_NAME inspirecv
        POSITION_INDEPENDENT_CODE ON)
    target_compile_features(inspirecv_package PUBLIC cxx_std_14)
    if(NOT WIN32)
        target_compile_definitions(inspirecv_package INTERFACE
            "INSPIRECV_API=__attribute__((visibility(\"default\")))")
    endif()
    target_include_directories(inspirecv_package PUBLIC
        $<BUILD_INTERFACE:${INSPIRECV_INCLUDE_DIRS}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
    target_link_libraries(inspirecv_package PUBLIC
        ${NEED_LINK_LIBS} ${EXTENSION_LIBS})
    add_library(InspireCV::inspirecv ALIAS inspirecv_package)
    set(INSPIRECV_INSTALL_TARGET inspirecv_package)
else()
    set_target_properties(inspirecv PROPERTIES EXPORT_NAME inspirecv)
    if(INSPIRECV_BUILD_SHARED_LIBS)
        set_target_properties(inspirecv PROPERTIES
            VERSION ${PROJECT_VERSION}
            SOVERSION ${PROJECT_VERSION_MAJOR})
    endif()
    add_library(InspireCV::inspirecv ALIAS inspirecv)
    set(INSPIRECV_INSTALL_TARGET inspirecv)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_inspirecv_target.cmake)
inspirecv_check_target_contract(inspirecv "${INSPIRECV_SRC_FILES}")

function(inspirecv_link_consumer target_name)
    target_compile_features(${target_name} PRIVATE cxx_std_14)
    target_include_directories(${target_name} PRIVATE
        ${INSPIRECV_CONSUMER_INCLUDE_DIRS})
    if(INSPIRECV_BUILD_OBJECT_LIBS)
        target_sources(${target_name} PRIVATE $<TARGET_OBJECTS:inspirecv>)
        target_link_libraries(${target_name} PRIVATE
            ${NEED_LINK_LIBS} ${EXTENSION_LIBS})
    else()
        target_link_libraries(${target_name} PRIVATE inspirecv)
    endif()
endfunction()

if(INSPIRECV_BUILD_CPU_BENCHMARKS)
    if(INSPIRECV_BACKEND_OPENCV)
        message(FATAL_ERROR
            "CPU competitor benchmarks require the default OKCV backend")
    endif()
    find_package(OpenCV REQUIRED COMPONENTS core imgproc)
    add_executable(inspirecv_cpu_benchmark
        ${CMAKE_CURRENT_SOURCE_DIR}/src/benchmark/cpu_backend_benchmark.cpp)
    inspirecv_link_consumer(inspirecv_cpu_benchmark)
    target_include_directories(inspirecv_cpu_benchmark PRIVATE
        ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(inspirecv_cpu_benchmark PRIVATE ${OpenCV_LIBS})
endif()

if (INSPIRECV_BUILD_SAMPLE)    
    set(INSPIRECV_SAMPLE_TARGETS
        sample_point
        sample_image
        sample_image_mem
        sample_gauss_blur
        sample_morphology
        sample_pipeline
        sample_resize
        sample_threshold
        sample_blend
        sample_warp_affine
        sample_pad_flip_rotate)
    foreach(sample_target IN LISTS INSPIRECV_SAMPLE_TARGETS)
        add_executable(${sample_target} ${CPP_DIR}/sample/${sample_target}.cpp)
        inspirecv_link_consumer(${sample_target})
    endforeach()

    # Task samples (require OpenCV macros enabled)
    if (OpenCV_FOUND)
        set(INSPIRECV_TASK_SAMPLE_TARGETS
            sample_task_bench
            sample_task_bench_formats
            sample_task_verify)
        foreach(sample_target IN LISTS INSPIRECV_TASK_SAMPLE_TARGETS)
            add_executable(${sample_target} ${CPP_DIR}/sample/${sample_target}.cpp)
            inspirecv_link_consumer(${sample_target})
        endforeach()
    endif()
endif()

include(CTest)
if (INSPIRECV_BUILD_TESTS)
    file(GLOB_RECURSE TEST_CASE_FILES "${CPP_DIR}/test/cases/*.cpp")
    add_executable(inspirecv_tests ${CPP_DIR}/test/test.cpp ${TEST_CASE_FILES})
    target_include_directories(inspirecv_tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/Catch2/single_include")
    target_compile_definitions(inspirecv_tests PRIVATE
        INSPIRECV_TEST_HAVE_SSE=${ST_HAVE_SSE}
        INSPIRECV_TEST_HAVE_AVX2=${ST_HAVE_AVX2}
        INSPIRECV_TEST_HAVE_NEON=${ST_HAVE_NEON}
        INSPIRECV_TEST_BACKEND_OPENCV=$<BOOL:${INSPIRECV_BACKEND_OPENCV}>)
    if(INSPIRECV_ENABLE_CUDA)
        target_compile_definitions(inspirecv_tests PRIVATE
            INSPIRECV_HAS_CUDA=1 INSPIRECV_TASK_HAS_CUDA=1)
    endif()
    if(ST_HAVE_SSE)
        target_compile_definitions(inspirecv_tests PRIVATE
                                   INSPIRECV_TASK_USE_SSE)
    endif()
    if(ST_HAVE_NEON)
        target_compile_definitions(inspirecv_tests PRIVATE
                                   INSPIRECV_TASK_USE_NEON)
    endif()
    if(INSPIRECV_BACKEND_OPENCV)
        target_compile_definitions(inspirecv_tests PRIVATE
                                   INSPIRECV_BACKEND_OPENCV)
    endif()
    if(INSPIRECV_BACKEND_OKCV_USE_OPENCV_GUI)
        target_compile_definitions(inspirecv_tests PRIVATE
                                   INSPIRECV_BACKEND_OKCV_USE_OPENCV_GUI)
    endif()
    inspirecv_link_consumer(inspirecv_tests)
    if(BUILD_TESTING)
        add_test(NAME inspirecv.unit
                 COMMAND inspirecv_tests "~[benchmark]~[bench]")
        set_tests_properties(inspirecv.unit PROPERTIES
            ENVIRONMENT
                "INSPIRECV_IMAGES_DIR=${CMAKE_CURRENT_SOURCE_DIR}/images;INSPIRECV_GT_DIR=${CMAKE_CURRENT_SOURCE_DIR}/images/task_gt"
            LABELS "unit")
        if(INSPIRECV_INSTALL AND INSPIRECV_IS_TOP_LEVEL)
            add_test(NAME inspirecv.installed_consumer
                COMMAND ${CMAKE_COMMAND}
                    -DINSPIRECV_TEST_CMAKE=${CMAKE_COMMAND}
                    -DINSPIRECV_TEST_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/cmake/tests/installed_consumer
                    -DINSPIRECV_TEST_LIBRARY_BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}
                    -DINSPIRECV_TEST_GENERATOR=${CMAKE_GENERATOR}
                    -DINSPIRECV_TEST_CONFIG=$<CONFIG>
                    -DINSPIRECV_TEST_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                    -DINSPIRECV_TEST_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
                    -DINSPIRECV_TEST_OPENCV_DIR=${OpenCV_DIR}
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/run_installed_consumer_test.cmake)
            set_tests_properties(inspirecv.installed_consumer PROPERTIES
                LABELS "integration;install")
        endif()
        if(INSPIRECV_IS_TOP_LEVEL)
            add_test(NAME inspirecv.source_consumer
                COMMAND ${CMAKE_COMMAND}
                    -DINSPIRECV_TEST_CMAKE=${CMAKE_COMMAND}
                    -DINSPIRECV_TEST_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/cmake/tests/source_consumer
                    -DINSPIRECV_TEST_LIBRARY_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
                    -DINSPIRECV_TEST_LIBRARY_BUILD_DIR=${CMAKE_CURRENT_BINARY_DIR}
                    -DINSPIRECV_TEST_GENERATOR=${CMAKE_GENERATOR}
                    -DINSPIRECV_TEST_CONFIG=$<CONFIG>
                    -DINSPIRECV_TEST_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                    -DINSPIRECV_TEST_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/run_source_consumer_test.cmake)
            set_tests_properties(inspirecv.source_consumer PROPERTIES
                LABELS "integration;source-tree")
        endif()
    endif()
endif()

if (INSPIRECV_BUILD_EXAMPLES)
    set(INSPIRECV_EXAMPLE_TARGETS
        example_image_io
        example_image_color
        example_image_math
        example_image_filter_morph_alpha
        example_image_geometry
        example_image_float
        example_task_nv21_decode)
    foreach(example_target IN LISTS INSPIRECV_EXAMPLE_TARGETS)
        add_executable(${example_target}
            ${CMAKE_CURRENT_SOURCE_DIR}/example/${example_target}.cpp)
        inspirecv_link_consumer(${example_target})
    endforeach()
endif()

message(STATUS "================================")
message(STATUS "Version: ${INSPIRECV_VERSION_STRING}")
if(CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Build Type: multi-config (${CMAKE_CONFIGURATION_TYPES})")
else()
    message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
endif()
message(STATUS "InspireCV Backend: ${INSPIRECV_BACKEND}")

if(INSPIRECV_BACKEND STREQUAL "OpenCV" OR INSPIRECV_BACKEND_OKCV_USE_OPENCV)
    message(STATUS "OpenCV Found")
    message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
    message(STATUS "OpenCV Path: ${OpenCV_DIR}")
endif()

if(INSPIRECV_BACKEND STREQUAL "OKCV")
    message(STATUS "OKCV Found")
endif()

if(INSPIRECV_BACKEND STREQUAL "OKCV")
    message(STATUS "Use OpenCV: ${INSPIRECV_BACKEND_OKCV_USE_OPENCV}")
    message(STATUS "Use OpenCV IO: ${INSPIRECV_BACKEND_OKCV_USE_OPENCV_IO}")
    message(STATUS "Use OpenCV GUI: ${INSPIRECV_BACKEND_OKCV_USE_OPENCV_GUI}")
    if(INSPIRECV_BACKEND_OKCV_USE_OPENCV)
        message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
        message(STATUS "OpenCV Path: ${OpenCV_DIR}")
    endif()
    
endif()
message(STATUS "================================")

# Install
if(INSPIRECV_INSTALL)
    install(TARGETS ${INSPIRECV_INSTALL_TARGET}
        EXPORT InspireCVTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/inspirecv
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
            FILES_MATCHING PATTERN "*.h")

    set(INSPIRECV_PACKAGE_NEEDS_OPENCV OFF)
    if(INSPIRECV_BACKEND_OPENCV OR INSPIRECV_BACKEND_OKCV_USE_OPENCV)
        set(INSPIRECV_PACKAGE_NEEDS_OPENCV ON)
    endif()
    set(INSPIRECV_PACKAGE_NEEDS_CUDA ${INSPIRECV_ENABLE_CUDA})
    set(INSPIRECV_PACKAGE_CMAKE_DIR
        "${CMAKE_INSTALL_LIBDIR}/cmake/InspireCV")
    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/InspireCVConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/InspireCVConfig.cmake
        INSTALL_DESTINATION ${INSPIRECV_PACKAGE_CMAKE_DIR})
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/InspireCVConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion)
    install(EXPORT InspireCVTargets
        FILE InspireCVTargets.cmake
        NAMESPACE InspireCV::
        DESTINATION ${INSPIRECV_PACKAGE_CMAKE_DIR})
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/InspireCVConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/InspireCVConfigVersion.cmake
        DESTINATION ${INSPIRECV_PACKAGE_CMAKE_DIR})

    if(INSPIRECV_BUILD_TESTS)
        install(TARGETS inspirecv_tests
                RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()
endif()
