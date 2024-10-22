#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "voxel_filter.h"

int main(int argc, char** argv) {
    // Load a point cloud from a PCD file
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());

    if (pcl::io::loadPCDFile<pcl::PointXYZ>("../data/table_scene.pcd", *cloud) == -1) {
        PCL_ERROR("Couldn't read file sample.pcd \n");
        return (-1);
    }
    std::cout << "Loaded " << cloud->width * cloud->height
              << " data points from sample.pcd" << std::endl;

    // Call voxel filter
    voxelFilter(cloud);

    std::string output_filename = "../data/filtered_sample.pcd";
    pcl::io::savePCDFileASCII(output_filename, *cloud);
    std::cout << "Filtered point cloud saved as " << output_filename << std::endl;

    return 0;
}
