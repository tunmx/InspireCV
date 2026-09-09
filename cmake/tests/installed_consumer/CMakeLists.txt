cmake_minimum_required(VERSION 3.15)
project(InspireCVInstalledConsumer LANGUAGES CXX)

find_package(InspireCV 1.0 CONFIG REQUIRED)

add_executable(inspirecv_installed_consumer main.cpp)
target_link_libraries(inspirecv_installed_consumer PRIVATE
                      InspireCV::inspirecv)

enable_testing()
add_test(NAME installed_api COMMAND inspirecv_installed_consumer)
