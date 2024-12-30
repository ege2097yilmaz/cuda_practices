#include "fluid_sim.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>


// CUDA kernels for fluid dynamics

__global__ void advect(float* field, float* field0, int gridSize, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= gridSize || j >= gridSize) return;

    // Backtracking particles for advection
    float x = i - field0[i + j * gridSize] * dt;
    float y = j - field0[i + j * gridSize] * dt;

    x = max(0.0f, min((float)gridSize - 1, x));
    y = max(0.0f, min((float)gridSize - 1, y));

    int i0 = (int)x;
    int i1 = i0 + 1;
    int j0 = (int)y;
    int j1 = j0 + 1;

    float s1 = x - i0;
    float s0 = 1 - s1;
    float t1 = y - j0;
    float t0 = 1 - t1;

    field[i + j * gridSize] = s0 * (t0 * field0[i0 + j0 * gridSize] + t1 * field0[i0 + j1 * gridSize]) +
                              s1 * (t0 * field0[i1 + j0 * gridSize] + t1 * field0[i1 + j1 * gridSize]);
}

__global__ void diffuse(float* field, float* field0, float diffusion, int gridSize, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= gridSize || j >= gridSize) return;

    for (int k = 0; k < 20; k++) { // Gauss-Seidel relaxation
        field[i + j * gridSize] =
            (field0[i + j * gridSize] + diffusion * dt *
             (field[i - 1 + j * gridSize] + field[i + 1 + j * gridSize] +
              field[i + (j - 1) * gridSize] + field[i + (j + 1) * gridSize])) /
            (1 + 4 * diffusion * dt);
    }
}

// Host function to run the simulation
void simulate_fluid(const FluidSimParams& params, int steps) {
    int gridSize = params.gridSize;
    int gridSize2 = gridSize * gridSize;
    size_t gridBytes = gridSize2 * sizeof(float);

    float *field, *field0;
    cudaMalloc(&field, gridBytes);
    cudaMalloc(&field0, gridBytes);

    cudaMemset(field, 0, gridBytes);
    cudaMemset(field0, 0, gridBytes);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((gridSize + 15) / 16, (gridSize + 15) / 16);

    for (int step = 0; step < steps; step++) {
        // Advection step
        advect<<<numBlocks, threadsPerBlock>>>(field, field0, gridSize, params.dt);

        // Diffusion step
        diffuse<<<numBlocks, threadsPerBlock>>>(field, field0, params.diffusion, gridSize, params.dt);

        cudaMemcpy(field0, field, gridBytes, cudaMemcpyDeviceToDevice);
    }

    cudaFree(field);
    cudaFree(field0);

    write_grid_to_csv(field, gridSize, "fluid_sim_output.csv");
}

void write_grid_to_csv(const float* field, int gridSize, const std::string& filename) {
    std::vector<float> hostData(gridSize * gridSize);
    cudaMemcpy(hostData.data(), field, gridSize * gridSize * sizeof(float), cudaMemcpyDeviceToHost);

    std::ofstream file(filename);
    for (int j = 0; j < gridSize; j++) {
        for (int i = 0; i < gridSize; i++) {
            file << hostData[i + j * gridSize];
            if (i < gridSize - 1) file << ",";
        }
        file << "\n";
    }
    file.close();
}