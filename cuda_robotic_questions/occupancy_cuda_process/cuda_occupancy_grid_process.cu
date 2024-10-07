/**
 * @file cuda_occupancy_grid_process.cu
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2024-10-07
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cuda_runtime.h>
#include <iostream>

#define WIDTH 16  // Width of the occupancy grid
#define HEIGHT 16  // Height of the occupancy grid

// Kernel function to process the occupancy grid
__global__ void processOccupancyGrid(int* grid, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;

        // Example processing: expanding obstacles (marked as 1)
        if (grid[idx] == 1) {
            if (x > 0 && grid[idx - 1] == 0) grid[idx - 1] = 1;  // Left
            if (x < width - 1 && grid[idx + 1] == 0) grid[idx + 1] = 1;  // Right
            if (y > 0 && grid[idx - width] == 0) grid[idx - width] = 1;  // Top
            if (y < height - 1 && grid[idx + width] == 0) grid[idx + width] = 1;  // Bottom
        }
    }
}

// Function to display the occupancy grid as text
void displayGrid(int* grid, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int value = grid[y * width + x];
            if (value == 0)
                std::cout << ". ";  // Free space
            else if (value == 1)
                std::cout << "# ";  // Obstacle
            else
                std::cout << "? ";  // Unknown space
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    int size = WIDTH * HEIGHT * sizeof(int);

    // allocate memory
    int* h_grid = (int*)malloc(size);

    // initialize the occupancy grid (0 = free, 1 = obstacle, -1 = unknown)
    for (int i = 0; i < WIDTH * HEIGHT; i++) {
        h_grid[i] = rand() % 3 - 1; 
    }

    // display the grid 
    std::cout << "Grid Before Processing:" << std::endl;
    displayGrid(h_grid, WIDTH, HEIGHT);

    int* d_grid;
    cudaMalloc(&d_grid, size);

    cudaMemcpy(d_grid, h_grid, size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);  
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, (HEIGHT + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    processOccupancyGrid<<<gridSize, blockSize>>>(d_grid, WIDTH, HEIGHT);

    cudaMemcpy(h_grid, d_grid, size, cudaMemcpyDeviceToHost);

    // display the grid
    std::cout << "Grid After Processing:" << std::endl;
    displayGrid(h_grid, WIDTH, HEIGHT);

    cudaFree(d_grid);

    free(h_grid);
    return 0;
}
