#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "voxel_filter.h"
#include <cuda_runtime.h>
#include <iostream>

// CUDA Kernel for Voxel Filtering
__global__ void voxel_filter_kernel(float* x, float* y, float* z, int n, float leaf_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Apply voxel filtering logic (grid down-sampling)
        x[idx] = round(x[idx] / leaf_size) * leaf_size;
        y[idx] = round(y[idx] / leaf_size) * leaf_size;
        z[idx] = round(z[idx] / leaf_size) * leaf_size;
    }
}

void voxelFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
    int n = cloud->points.size();
    float leaf_size = 0.01f;  // Define the leaf size for filtering

    // Allocate host memory
    float *h_x = new float[n];
    float *h_y = new float[n];
    float *h_z = new float[n];

    // Fill host arrays with point cloud data
    for (int i = 0; i < n; ++i) {
        h_x[i] = cloud->points[i].x;
        h_y[i] = cloud->points[i].y;
        h_z[i] = cloud->points[i].z;
    }

    // Allocate device memory
    float *d_x, *d_y, *d_z;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));
    cudaMalloc(&d_z, n * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the voxel filter kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    voxel_filter_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_z, n, leaf_size);

    // Copy filtered data back to host
    cudaMemcpy(h_x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_z, d_z, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Update the point cloud with the filtered data
    for (int i = 0; i < n; ++i) {
        cloud->points[i].x = h_x[i];
        cloud->points[i].y = h_y[i];
        cloud->points[i].z = h_z[i];
    }

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    // Free host memory
    delete[] h_x;
    delete[] h_y;
    delete[] h_z;

    std::cout << "Voxel filter applied." << std::endl;
}