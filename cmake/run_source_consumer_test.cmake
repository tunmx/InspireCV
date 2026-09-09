foreach(required_variable IN ITEMS
        INSPIRECV_TEST_CMAKE
        INSPIRECV_TEST_SOURCE_DIR
        INSPIRECV_TEST_LIBRARY_SOURCE_DIR
        INSPIRECV_TEST_LIBRARY_BUILD_DIR)
    if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
        message(FATAL_ERROR "Missing ${required_variable}")
    endif()
endforeach()

if(NOT DEFINED INSPIRECV_TEST_CONFIG OR
   "${INSPIRECV_TEST_CONFIG}" STREQUAL "")
    set(INSPIRECV_TEST_CONFIG Release)
endif()

set(consumer_build
    "${INSPIRECV_TEST_LIBRARY_BUILD_DIR}/source-consumer/${INSPIRECV_TEST_CONFIG}")
set(configure_command
    "${INSPIRECV_TEST_CMAKE}"
    -S "${INSPIRECV_TEST_SOURCE_DIR}"
    -B "${consumer_build}"
    "-DINSPIRECV_SOURCE_DIR=${INSPIRECV_TEST_LIBRARY_SOURCE_DIR}"
    "-DCMAKE_BUILD_TYPE=RelWithDebInfo")
if(DEFINED INSPIRECV_TEST_GENERATOR AND
   NOT "${INSPIRECV_TEST_GENERATOR}" STREQUAL "")
    list(APPEND configure_command -G "${INSPIRECV_TEST_GENERATOR}")
endif()
if(DEFINED INSPIRECV_TEST_CXX_COMPILER AND
   NOT "${INSPIRECV_TEST_CXX_COMPILER}" STREQUAL "")
    list(APPEND configure_command
         "-DCMAKE_CXX_COMPILER=${INSPIRECV_TEST_CXX_COMPILER}")
endif()
if(DEFINED INSPIRECV_TEST_OSX_ARCHITECTURES AND
   NOT "${INSPIRECV_TEST_OSX_ARCHITECTURES}" STREQUAL "")
    list(APPEND configure_command
         "-DCMAKE_OSX_ARCHITECTURES=${INSPIRECV_TEST_OSX_ARCHITECTURES}")
endif()

execute_process(COMMAND ${configure_command} RESULT_VARIABLE configure_result)
if(NOT configure_result EQUAL 0)
    message(FATAL_ERROR
            "Source consumer configuration failed: ${configure_result}")
endif()

execute_process(
    COMMAND "${INSPIRECV_TEST_CMAKE}" --build "${consumer_build}"
            --config "${INSPIRECV_TEST_CONFIG}"
    RESULT_VARIABLE build_result)
if(NOT build_result EQUAL 0)
    message(FATAL_ERROR "Source consumer build failed: ${build_result}")
endif()

execute_process(
    COMMAND "${CMAKE_CTEST_COMMAND}"
            -C "${INSPIRECV_TEST_CONFIG}" --output-on-failure
    WORKING_DIRECTORY "${consumer_build}"
    RESULT_VARIABLE test_result)
if(NOT test_result EQUAL 0)
    message(FATAL_ERROR "Source consumer test failed: ${test_result}")
endif()
