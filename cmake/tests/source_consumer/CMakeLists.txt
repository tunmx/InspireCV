cmake_minimum_required(VERSION 3.15)
project(InspireCVSourceConsumer LANGUAGES CXX)

if(NOT DEFINED INSPIRECV_SOURCE_DIR OR INSPIRECV_SOURCE_DIR STREQUAL "")
    message(FATAL_ERROR "INSPIRECV_SOURCE_DIR is required")
endif()

# These host settings are sentinels: add_subdirectory(InspireCV) must not
# rewrite them.
set(CMAKE_CXX_STANDARD 17)
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO
        "/O1 /Zi /DINSPIRECV_HOST_FLAG_SENTINEL" CACHE STRING "" FORCE)
else()
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO
        "-O1 -g -DINSPIRECV_HOST_FLAG_SENTINEL" CACHE STRING "" FORCE)
endif()
set(expected_host_flags "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
set(expected_install_prefix "${CMAKE_BINARY_DIR}/host-prefix")
set(CMAKE_INSTALL_PREFIX "${expected_install_prefix}" CACHE PATH "" FORCE)

set(INSPIRECV_BUILD_OBJECT_LIBS ON CACHE BOOL "" FORCE)
set(INSPIRECV_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(INSPIRECV_BUILD_SAMPLE OFF CACHE BOOL "" FORCE)
set(INSPIRECV_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(INSPIRECV_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory("${INSPIRECV_SOURCE_DIR}" inspirecv)

if(NOT CMAKE_CONFIGURATION_TYPES AND
   NOT CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    message(FATAL_ERROR "InspireCV changed the host build type")
endif()
if(NOT "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}" STREQUAL "${expected_host_flags}")
    message(FATAL_ERROR "InspireCV changed the host compiler flags")
endif()
if(NOT "${CMAKE_INSTALL_PREFIX}" STREQUAL "${expected_install_prefix}")
    message(FATAL_ERROR "InspireCV changed the host install prefix")
endif()

add_executable(inspirecv_source_consumer
    "${CMAKE_CURRENT_LIST_DIR}/../installed_consumer/main.cpp")
target_link_libraries(inspirecv_source_consumer PRIVATE InspireCV::inspirecv)

enable_testing()
add_test(NAME source_api COMMAND inspirecv_source_consumer)

if(WIN32)
    # InspireFace embeds this exact OBJECT payload into its own DLL. Verify
    # the public API exports without WINDOWS_EXPORT_ALL_SYMBOLS masking gaps.
    target_compile_definitions(inspirecv PRIVATE "INSPIRECV_API=__declspec(dllexport)")
    add_library(inspirecv_embedded SHARED $<TARGET_OBJECTS:inspirecv>
        halide_cv.cpp halide_host.cpp)
    set_target_properties(inspirecv_embedded PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS OFF
        LINKER_LANGUAGE CXX)
    target_include_directories(inspirecv_embedded INTERFACE
        "${INSPIRECV_SOURCE_DIR}/include")
    target_include_directories(inspirecv_embedded PRIVATE
        "${INSPIRECV_SOURCE_DIR}/include")
    target_compile_definitions(inspirecv_embedded INTERFACE
        "INSPIRECV_API=__declspec(dllimport)")
    add_executable(inspirecv_embedded_consumer
        "${CMAKE_CURRENT_LIST_DIR}/../installed_consumer/main.cpp")
    target_compile_definitions(inspirecv_embedded_consumer PRIVATE
        INSPIRECV_TEST_EMBEDDED_HALIDE_HOST)
    target_link_libraries(inspirecv_embedded_consumer PRIVATE inspirecv_embedded)
    if(MSVC)
        # Force real helper calls in this ABI regression even in Release.
        # Production sources retain their normal optimization settings.
        set_source_files_properties(halide_cv.cpp halide_host.cpp PROPERTIES
            COMPILE_OPTIONS "/Od;/Ob0")
    endif()
    add_test(NAME embedded_dll_api COMMAND inspirecv_embedded_consumer)
endif()
