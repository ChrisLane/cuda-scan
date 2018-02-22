cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(assignment1 CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_50")

include_directories(/opt/cuda/include)
include_directories(/opt/cuda/samples/common/inc/)

add_executable(main src/main.cu src/dummy.cpp)