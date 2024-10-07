#include "occupancy_grid_processor.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <iostream>

__global__ void processGridKernel(int* d_grid, int rows, int cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < rows && j < cols) {

        // printf("Thread (%d, %d) in block (%d, %d) processes element (%d, %d)\n",
        //        threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, i, j);

        d_grid[i * cols + j] = !d_grid[i * cols + j];
    }
}

void processOccupancyGrid(std::vector<std::vector<int>>& grid) {

    int rows = grid.size();
    int cols = grid[0].size();
    int gridSize = rows * cols * sizeof(int);

    // Flatten grid to a 1D array
    std::vector<int> flatGrid(rows * cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            flatGrid[i * cols + j] = grid[i][j];
        }
    }

    // ---------- GPU Processing with Memory Transfer Timing ----------
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu);

    // Allocate memory on the GPU
    int* d_grid;
    cudaMalloc((void**)&d_grid, gridSize);

    // Copy data to the GPU
    cudaMemcpy(d_grid, flatGrid.data(), gridSize, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y);

    // Launch the kernel
    processGridKernel<<<gridDim, blockDim>>>(d_grid, rows, cols);

    // cudaDeviceSynchronize();
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     std::cerr << "CUDA output: " << cudaGetErrorString(err) << std::endl;
    // }

    // Copy the result back to the host
    cudaMemcpy(flatGrid.data(), d_grid, gridSize, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop_gpu);

    cudaEventSynchronize(stop_gpu);

    float gpu_duration = 0;
    cudaEventElapsedTime(&gpu_duration, start_gpu, stop_gpu);
    std::cout << "GPU processing time in processing: " << gpu_duration / 1000.0f << " seconds\n"; // Convert from ms to seconds

    // Cleanup CUDA events
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    // Copy back the flattened data into the 2D grid
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            grid[i][j] = flatGrid[i * cols + j];
        }
    }

    // Free GPU memory
    cudaFree(d_grid);
}
