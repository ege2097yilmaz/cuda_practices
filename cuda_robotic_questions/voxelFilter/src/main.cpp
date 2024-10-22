#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "voxel_filter.h"
#include "voxel_filter_cpu.h" 
#include <chrono>  

int main(int argc, char** argv) {
    // Load the point cloud from a PCD file
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("../data/table_scene.pcd", *cloud) == -1) {
        PCL_ERROR("Couldn't read file sample.pcd \n");
        return (-1);
    }
    std::cout << "Loaded " << cloud->width * cloud->height
              << " data points from sample.pcd" << std::endl;

    float voxel_size = 0.01f;  // Adjust voxel size as needed

    // Measure the time for the CPU voxel filter
    auto start_cpu = std::chrono::high_resolution_clock::now();
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud_cpu = voxel_filter_cpu(cloud, voxel_size);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = end_cpu - start_cpu;
    std::cout << "CPU Voxel Filter took " << cpu_duration.count() << " seconds." << std::endl;

    // Save the CPU-filtered point cloud
    pcl::io::savePCDFileASCII("../data/filtered_sample_cpu.pcd", *filtered_cloud_cpu);
    std::cout << "Filtered point cloud (CPU) saved as filtered_sample_cpu.pcd" << std::endl;

    // Measure the time for the GPU voxel filter
    auto start_gpu = std::chrono::high_resolution_clock::now();
    voxelFilter(cloud);  // GPU voxel filter modifies the cloud in place
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_duration = end_gpu - start_gpu;
    std::cout << "GPU Voxel Filter took " << gpu_duration.count() << " seconds." << std::endl;

    // Save the GPU-filtered point cloud
    pcl::io::savePCDFileASCII("../data/filtered_sample_gpu.pcd", *cloud);
    std::cout << "Filtered point cloud (GPU) saved as filtered_sample_gpu.pcd" << std::endl;

    return 0;
}