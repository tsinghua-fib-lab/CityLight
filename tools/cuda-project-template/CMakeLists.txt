cmake_minimum_required(VERSION 3.16)

# project info
project({{PROJECT_NAME}} LANGUAGES CXX CUDA)

# global settings

# activate intellisense
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set runtime output dir
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/build/bin/")

set(CMAKE_CXX_STANDARD 17)
# include cmake
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/" ${CMAKE_MODULE_PATH})
include(BuildType)

add_subdirectory(src)
