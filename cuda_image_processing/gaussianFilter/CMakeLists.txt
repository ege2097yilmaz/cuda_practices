cmake_minimum_required(VERSION 3.10)
project(GaussianFilter LANGUAGES CXX CUDA)

# Enable C++11
set(CMAKE_CXX_STANDARD 11)

# Find CUDA
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(include)
include_directories(${CMAKE_SOURCE_DIR}/include)

set(CUDA_SRCS src/gaussian_filter.cu)

add_executable(gaussianFilter src/main.cpp ${CUDA_SRCS})


target_link_libraries(gaussianFilter ${CUDA_LIBRARIES} ${PYTHON_LIBRARIES})


# Set CUDA architecture
set_target_properties(gaussianFilter PROPERTIES CUDA_ARCHITECTURES 60)
