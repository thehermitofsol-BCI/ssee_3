# CMakeLists.txt for Native FastCV + AHardwareBuffer Zero-Copy Preprocessor
cmake_minimum_required(VERSION 3.22.1)
project("fastcv_zero_copy")

add_library(
    fastcv_zero_copy
    SHARED
    fastcv_ahardwarebuffer_preprocessor.cpp
)

find_library(
    log-lib
    log
)

find_library(
    android-lib
    android
)

target_link_libraries(
    fastcv_zero_copy
    ${log-lib}
    ${android-lib}
)
