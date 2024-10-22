#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <unordered_map>
#include <cmath>

struct VoxelIndex {
    int x, y, z;

    bool operator==(const VoxelIndex& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

namespace std {
    template <>
    struct hash<VoxelIndex> {
        std::size_t operator()(const VoxelIndex& vi) const {
            return std::hash<int>()(vi.x) ^ std::hash<int>()(vi.y) ^ std::hash<int>()(vi.z);
        }
    };
}


pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_filter_cpu(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float voxel_size);
