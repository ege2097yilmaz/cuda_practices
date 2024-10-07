# CUDA Practice Repository

This repository contains a collection of simple CUDA (Compute Unified Device Architecture) practice projects. The projects aim to demonstrate the basics of parallel programming with CUDA, including matrix operations, memory management, and image processing.

## Projects

1. **Matrix Addition**
   - Example of adding two matrices using CUDA kernels.
   
2. **ID Assignment**
   - Demonstrates thread and block ID assignment and their usage in parallel tasks.
   
3. **Grid-Block-Thread Examples**
   - Various examples showcasing the concept of grids, blocks, and threads in CUDA programming.

4. **Memory Practices**
   - Examples on how to manage memory in CUDA, including global memory allocation, shared memory usage, and memory transfers between host and device.

5. **Occupancy Grid Process**
   - Practice on handling occupancy grids and optimizing resource usage across threads and blocks.

6. **Image Processing**
   - Basic image processing examples using CUDA, including tasks like filtering and transformations.

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit (https://developer.nvidia.com/cuda-toolkit)
- C++ compiler (e.g., GCC)
  
## Getting Started

To compile and run any of the examples, follow these general steps:

1. Clone this repository:
   ```bash
   cd cuda-practice
   git clone https://github.com/yourusername/cuda-practice.git
   ```


2. You need to compile each projects by handexcept for occupancy_grid project which can be compiled with cmake
    ```bash
    nvcc -o matrix_addition matrix_addition.cu
    ./matrix_addition

    ```

    Replace matrix_addition.cu with the appropriate .cu file for other projects.