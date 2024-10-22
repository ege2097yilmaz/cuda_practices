cmake_minimum_required(VERSION 3.10)
project(VoxelFilterProject LANGUAGES CXX CUDA)

# Set C++ standard to C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ensure CUDA uses the same C++ standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find PCL
find_package(PCL 1.8 REQUIRED)

# Find CUDA
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
    link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
else()
    message(FATAL_ERROR "CUDA not found. Please make sure you have CUDA installed.")
endif()

# Set include directories
include_directories(${PCL_INCLUDE_DIRS} include)

# Link PCL libraries
link_directories(${PCL_LIBRARY_DIRS})

# Add the executable
add_executable(voxel_filter src/main.cpp src/voxel_filter.cu src/voxel_filter_cpu.cpp)

# Link against PCL and CUDA
target_link_libraries(voxel_filter ${PCL_LIBRARIES} ${CUDA_LIBRARIES})

# Add compiler options for CUDA (optional)
set_target_properties(voxel_filter PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
