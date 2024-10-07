#include <iostream>
#include <cuda_runtime.h>

#define WIDTH 4
#define HEIGHT 4

__global__ void matrixAdd(int *A, int *B, int *C, int width, int height){
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if(col < width && row < height){
        int index = row * width + col;
        C[index] = A[index] + B[index];
    }
}

int main() {
    int width = WIDTH;
    int height = HEIGHT;
    int size = width * height * sizeof(int);

    int A[width * height], B[width * height], C[width * height];

    for(int i = 0; i < width * height; i++){
        A[i] = i;
        B[i] = i * 2;
    }

    int *d_a, *d_b, *d_c;

    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    cudaMemcpy(d_a, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(2, 2);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixAdd<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, width, height);
    
    cudaMemcpy(C, d_c, size, cudaMemcpyDeviceToHost);

    std::cout << "results: \n";

    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            std::cout << C[i * width + j] << " ";
        }
        std::cout << std::endl; 
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}