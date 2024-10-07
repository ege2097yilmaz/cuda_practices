cmake_minimum_required(VERSION 3.10)
project(OccupancyGridProcessor LANGUAGES CXX CUDA)

# Enable C++11
set(CMAKE_CXX_STANDARD 11)

# Find CUDA
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(include)

set(CUDA_SRCS src/occupancy_grid_processor.cu)

add_executable(OccupancyGridProcessor src/main.cpp ${CUDA_SRCS})


target_link_libraries(OccupancyGridProcessor ${CUDA_LIBRARIES} ${PYTHON_LIBRARIES})


# Set CUDA architecture
set_target_properties(OccupancyGridProcessor PROPERTIES CUDA_ARCHITECTURES 60)
