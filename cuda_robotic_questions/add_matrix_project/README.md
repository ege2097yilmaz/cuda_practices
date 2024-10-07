# CUDA Element-wise Array Addition

## Overview

This project demonstrates how to calculate the optimal number of blocks and threads in a CUDA kernel for processing a 1D array of size `N`. The CUDA kernel performs element-wise addition of two arrays using an optimal configuration.

### Problem
Given an array size of `N = 1000`, this code calculates the optimal number of blocks and threads required to efficiently execute the addition of two arrays on a GPU using CUDA.

### Key Concepts
- **CUDA Threads and Blocks**: 
  - A CUDA kernel runs on many parallel threads.
  - Threads are grouped into blocks, and blocks are grouped into a grid.
  - To optimize performance, we want to find the right configuration of blocks and threads per block for a given problem size `N`.
  
  Typically, the number of threads per block is a multiple of 32 (the warp size). In this example, we use 256 threads per block, which is a common choice for GPUs.

## Code Breakdown

1. **Kernel Function**:
   The `addMatrix` function is a CUDA kernel that performs element-wise addition of two arrays, `A` and `B`, and stores the result in `C`. Each thread computes one element in the result array.
   
   ```cpp
   __global__ void addMatrix(float *A, float *B, float *C, int n) {
       int index = threadIdx.x + blockIdx.x * blockDim.x;
       if (index < n) {
           C[index] = A[index] + B[index];
       }
   }
   ```

2. **Block and Thread Configuration**:

    The number of threads per block is set to 256. The number of blocks is calculated by dividing the total number of elements by the number of threads per block. To ensure we cover all elements of the array, we compute:

    ```cpp
    int blockPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    ```

3. **Host and Device Memory**:
    The arrays A, B, and C are created and initialized on the host. Memory for these arrays is allocated on the GPU using cudaMalloc, and data is transferred between the host and device using cudaMemcpy.

4. **Launching the Kernel**:
    The kernel is launched using:

    ```cuda
    addMatrix<<<blockPerGrid, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
    ```

# Requirements
* A system with an NVIDIA GPU and CUDA toolkit installed.
* A C++ compiler with CUDA support (e.g., nvcc).

# How to Compile and Run
1. Make sure you have nvcc installed.
2. Save the code to a file, e.g., matrix_add.cu.
3. Compile the code with the following command:
```bash
nvcc -o matrix_add matrix_add.cu
```
4. Run the compiled program:
```bash
./matrix_add
```
5. The output for the first 10 elements will be:
```bash
Results:
0 2 4 6 8 10 12 14 16 18
```
