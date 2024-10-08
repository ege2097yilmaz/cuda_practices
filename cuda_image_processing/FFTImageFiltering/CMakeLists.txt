cmake_minimum_required(VERSION 3.10)
project(FFTImageFiltering LANGUAGES CXX CUDA)

# Enable C++11
set(CMAKE_CXX_STANDARD 11)

# Find CUDA
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -gencode arch=compute_52,code=sm_52")


find_package(PythonLibs REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(include)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS})


# set(CUDA_SRCS src/gaussian_filter.cu)
file(GLOB SOURCES "src/*.cpp" "src/*.cu")

set(IMAGE_PATH "${CMAKE_SOURCE_DIR}/images")

add_compile_definitions(IMAGE_PATH="${IMAGE_PATH}")

# add_executable(gaussianFilter src/main.cpp ${CUDA_SRCS})
cuda_add_executable(fft_image_filtering ${SOURCES})
target_link_libraries(fft_image_filtering ${CUDA_LIBRARIES} ${CUFFT_LIBRARIES} ${OpenCV_LIBS} ${CUDA_CUFFT_LIBRARIES})
