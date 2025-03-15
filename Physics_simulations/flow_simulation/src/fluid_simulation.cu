/*
Fluid Simulation CUDA Kernels
*/

#include "fluid_simulation.h"

float *h_velocityX, *h_velocityY, *h_pressure;
float *d_velocityX, *d_velocityY, *d_pressure;

__global__ void update_velocity(float *velocityX, float *velocityY, float *pressure) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > 0 && x < WIDTH-1 && y > 0 && y < HEIGHT-1) {
        int idx = y * WIDTH + x;
        velocityX[idx] -= TIME_STEP * (pressure[idx + 1] - pressure[idx - 1]);
        velocityY[idx] -= TIME_STEP * (pressure[idx + WIDTH] - pressure[idx - WIDTH]);
    }
}

__global__ void apply_boundaries(float *velocityX, float *velocityY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * WIDTH + x;
    if (x == 0 || x == WIDTH - 1 || y == 0 || y == HEIGHT - 1) {
        velocityX[idx] = 0;
        velocityY[idx] = 0;
    }
}

void init_simulation() {
    int size = WIDTH * HEIGHT * sizeof(float);
    cudaMalloc((void**)&d_velocityX, size);
    cudaMalloc((void**)&d_velocityY, size);
    cudaMalloc((void**)&d_pressure, size);
    h_velocityX = (float*)malloc(size);
    h_velocityY = (float*)malloc(size);
    h_pressure = (float*)malloc(size);
}

void step_simulation() {
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(WIDTH / BLOCK_SIZE, HEIGHT / BLOCK_SIZE);
    
    update_velocity<<<gridDim, blockDim>>>(d_velocityX, d_velocityY, d_pressure);
    apply_boundaries<<<gridDim, blockDim>>>(d_velocityX, d_velocityY);
    cudaMemcpy(h_velocityX, d_velocityX, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_velocityY, d_velocityY, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);
}
