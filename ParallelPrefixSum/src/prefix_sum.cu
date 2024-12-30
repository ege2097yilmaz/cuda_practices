#include "prefix_sum.h"
#include "cuda_runtime.h"
#include <vector>
#include <iostream>


__global__ void inclusive_scan_kernel(int* d_input, int* d_output, int n) {
    extern __shared__ int temp[]; // Shared memory for partial sums
    int tid = threadIdx.x;
    int offset = 1;

    // Load input into shared memory
    if (2 * tid < n) temp[2 * tid] = d_input[2 * tid];
    if (2 * tid + 1 < n) temp[2 * tid + 1] = d_input[2 * tid + 1];

    __syncthreads();

    // Up-sweep phase
    for (int d = 1; d < n; d *= 2) {
        if (tid < n / (2 * d)) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
        __syncthreads();
    }

    // Clear the last element for exclusive scan
    if (tid == 0) temp[n - 1] = 0;

    // Down-sweep phase
    for (int d = n / 2; d >= 1; d /= 2) {
        offset /= 2;
        if (tid < n / (2 * d)) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
        __syncthreads();
    }

    // Write back to global memory
    if (2 * tid < n) d_output[2 * tid] = temp[2 * tid];
    if (2 * tid + 1 < n) d_output[2 * tid + 1] = temp[2 * tid + 1];
}


std::vector<int> paralel_prefix_sum(const std::vector<int>& input){
    int n = input.size();
    int* d_input;
    int* d_output;

    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));

    cudaMemcpy(d_input,input.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 1024;
    int blocks = (n + threads - 1) / threads;

    inclusive_scan_kernel<<<blocks, threads, 2 * threads * sizeof(int)>>>(d_input, d_output, n);

    std::vector<int> output(n);
    cudaMemcpy(output.data(), d_output, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}