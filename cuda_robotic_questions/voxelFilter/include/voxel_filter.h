#ifndef VOXEL_FILTER_H
#define VOXEL_FILTER_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// Declare the CUDA voxel filter function
void voxelFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

#endif