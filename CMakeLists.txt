cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(assignment1 CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")

set(CMAKE_CUDA_STANDARD 11)

include_directories(/opt/cuda/include)

add_executable(main src/main.cu src/dummy.cpp)