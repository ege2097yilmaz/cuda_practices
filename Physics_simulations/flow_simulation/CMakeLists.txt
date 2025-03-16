cmake_minimum_required(VERSION 3.10)
project(CUDA_2D_Pipe_Flow LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include_directories(${CMAKE_SOURCE_DIR}/include)

set(SOURCES
    src/main.cpp
    src/fluid_simulation.cu
    # src/visualization.cpp
)

add_executable(pipe_flow ${SOURCES})

set_target_properties(pipe_flow PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(pipe_flow PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIR}
    ${GLUT_INCLUDE_DIR}
)

target_link_libraries(pipe_flow
    ${CUDA_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${GLUT_LIBRARY}
)