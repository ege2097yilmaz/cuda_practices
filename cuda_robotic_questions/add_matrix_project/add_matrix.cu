/** 
 * Question:
    How do you calculate the optimal number of blocks and threads in a CUDA kernel for processing a 1D array of size N? 
    Explain the reasoning and write a simple CUDA kernel that performs element-wise addition of two arrays using an optimal configuration.

    N = 1000 assuming
**/

#include <iostream>
#include <cuda_runtime.h>

#define N 1000
#define THREADS_PER_BLOCK 256

__global__ void addMatrix(float *A, float *B, float *C, int n){
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    C[index] = A[index] + B[index];
}

int main(){
    int size = N * sizeof(float);

    float a[N], b[N], c[N];

    // initialize matrixes
    for(int i = 0; i < N; ++i){
        a[i] = i;
        b[i] = i;
    }

    // allocate the device variables
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // todo calculate the blocks and threads
    int blockPerGrid = (N - THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK; 

    // todo implement adding function
    addMatrix<<<blockPerGrid,THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    std::cout << "results \n";
    for(int i = 0; i < 10; ++i){
        std::cout << c[i] << " ";
    }

    std::cout << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}