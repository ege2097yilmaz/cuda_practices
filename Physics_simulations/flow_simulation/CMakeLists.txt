cmake_minimum_required(VERSION 3.10)
project(FluidSimulation LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(FluidSimulation src/main.cu src/solver.cu src/visualization.cpp src/boundary_conditions.cu src/utils.cu)
target_include_directories(FluidSimulation PRIVATE include)
set_target_properties(FluidSimulation PROPERTIES CUDA_ARCHITECTURES "50;60;70;75;80")