cmake_minimum_required(VERSION 3.12)
project(NavierStokesSim CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include_directories(${CMAKE_SOURCE_DIR}/include)

# Source files
set(SRC_FILES
    src/main.cu
    src/fluid_sim.cu
)

add_executable(navier_stokes_sim ${SRC_FILES})

set_target_properties(navier_stokes_sim PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)