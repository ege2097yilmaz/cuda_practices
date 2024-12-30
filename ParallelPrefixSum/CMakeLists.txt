cmake_minimum_required(VERSION 3.12)
project(ParallelPrefixSum CUDA)

# Set C++ and CUDA standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Add source files
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)

include_directories(${INCLUDE_DIR})

add_executable(prefix_sum ${SRC_DIR}/main.cu ${SRC_DIR}/prefix_sum.cu)

# Specify target properties
set_target_properties(prefix_sum PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)