#include "voxel_filter_cpu.h"

pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_filter_cpu(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float voxel_size) {
    // Create a hash map to store the sum of points and the number of points in each voxel
    std::unordered_map<VoxelIndex, std::pair<pcl::PointXYZ, int>> voxel_map;

    // Iterate over all points in the input cloud
    for (const auto& point : cloud->points) {
        // Calculate the voxel index for the current point
        int voxel_x = static_cast<int>(std::floor(point.x / voxel_size));
        int voxel_y = static_cast<int>(std::floor(point.y / voxel_size));
        int voxel_z = static_cast<int>(std::floor(point.z / voxel_size));
        VoxelIndex voxel_idx = {voxel_x, voxel_y, voxel_z};

        // Accumulate the points and count the number of points in each voxel
        if (voxel_map.find(voxel_idx) == voxel_map.end()) {
            // If this voxel is new, initialize it with the current point and a count of 1
            voxel_map[voxel_idx] = {point, 1};
        } else {
            // Otherwise, accumulate the current point into the voxel
            voxel_map[voxel_idx].first.x += point.x;
            voxel_map[voxel_idx].first.y += point.y;
            voxel_map[voxel_idx].first.z += point.z;
            voxel_map[voxel_idx].second += 1;  // Increment the point count
        }
    }

    // Create the filtered point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // Compute the centroids of each voxel and add them to the filtered point cloud
    for (const auto& kv : voxel_map) {
        const VoxelIndex& voxel_idx = kv.first;
        const pcl::PointXYZ& point_sum = kv.second.first;
        int point_count = kv.second.second;

        // Compute the centroid of the points in this voxel
        pcl::PointXYZ centroid;
        centroid.x = point_sum.x / point_count;
        centroid.y = point_sum.y / point_count;
        centroid.z = point_sum.z / point_count;

        // Add the centroid to the filtered point cloud
        filtered_cloud->points.push_back(centroid);
    }

    // Set the width, height, and density of the filtered point cloud
    filtered_cloud->width = filtered_cloud->points.size();
    filtered_cloud->height = 1;  // Since it's an unorganized point cloud
    filtered_cloud->is_dense = true;  // Mark the cloud as dense (no NaN points)

    return filtered_cloud;
}