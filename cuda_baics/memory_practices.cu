#include <iostream>
#include <cuda_runtime.h>

#define N 4        // Size of the matrix (N x N)
#define TILE_SIZE 2 // Size of the tile used for shared memory

// CUDA Kernel for Matrix Multiplication
__global__ void matrixMul(float *A, float *B, float *C, int n) {
    // Calculate the row and column index for the current thread
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    // Shared memory for storing sub-matrices (tiles) of A and B
    __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE];

    // Value to store the result of the matrix multiplication for this thread
    float value = 0.0f;

    // Loop over the tiles of A and B that contribute to C[row][col]
    for (int tileIndex = 0; tileIndex < n / TILE_SIZE; ++tileIndex) {
        // Load elements into shared memory
        sharedA[threadIdx.y][threadIdx.x] = A[row * n + (tileIndex * TILE_SIZE + threadIdx.x)];
        sharedB[threadIdx.y][threadIdx.x] = B[(tileIndex * TILE_SIZE + threadIdx.y) * n + col];
        __syncthreads();  // Synchronize threads to ensure all data is loaded

        // Perform the matrix multiplication for this tile
        for (int k = 0; k < TILE_SIZE; k++) {
            value += sharedA[threadIdx.y][k] * sharedB[k][threadIdx.x];
        }
        __syncthreads();  // Synchronize before moving to the next tile
    }

    // Store the result in the global memory
    C[row * n + col] = value;
}

// Utility function to initialize matrices
void initializeMatrix(float *matrix, int size, float value) {
    for (int i = 0; i < size; ++i) {
        matrix[i] = value;
    }
}

// Utility function to print matrices
void printMatrix(float *matrix, int width, int height) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << matrix[i * width + j] << " ";
        }
        std::cout << std::endl;
    }
}

void matrixMulCPU(float *A, float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float value = 0.0f;
            for (int k = 0; k < n; k++) {
                value += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = value;
        }
    }
}

bool compareMatrices(float *A, float *B, int size) {
    for (int i = 0; i < size; i++) {
        if (abs(A[i] - B[i]) > 1e-5) {
            return false;
        }
    }
    return true;
}

int main() {
    int size = N * N * sizeof(float);

    // Allocate host memory
    float A[N * N], B[N * N], C_GPU[N * N], C_CPU[N * N];

    // Initialize the matrices
    initializeMatrix(A, N * N, 1.0f); 
    initializeMatrix(B, N * N, 2.0f); 

    // printMatrix(A, N, N);
    // std::cout << "*************\n";
    // printMatrix(B, N, N);

    // Allocate device memory 
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy host matrices A and B to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Define the dimensions for the grid and block
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE); 
    dim3 numBlocks(N / TILE_SIZE, N / TILE_SIZE);

    // Launch the kernel
    matrixMul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy the result matrix C from device to host
    cudaMemcpy(C_GPU, d_C, size, cudaMemcpyDeviceToHost);

    matrixMulCPU(A, B, C_CPU, N);

    // Print the GPU result
    std::cout << "Matrix C from GPU:" << std::endl;
    printMatrix(C_GPU, N, N);

    // Print the CPU result
    std::cout << "Matrix C from CPU:" << std::endl;
    printMatrix(C_CPU, N, N);

    // Compare the results from CPU and GPU
    if (compareMatrices(C_GPU, C_CPU, N * N)) {
        std::cout << "Results match!" << std::endl;
    } else {
        std::cout << "Results do not match!" << std::endl;
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
