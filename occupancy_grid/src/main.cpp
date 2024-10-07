#include <iostream>
#include <vector>
#include "occupancy_grid_processor.h"
#include <unistd.h>
#include <chrono>

void processGridCPU(std::vector<std::vector<int>>& grid) {
    int rows = grid.size();
    int cols = grid[0].size();

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            grid[i][j] = !grid[i][j];  // Flip occupancy value
        }
    }
}


int main() {
    int rows = 10000, cols = 10000;

    // Create an occupancy grid and fill with random data
    std::vector<std::vector<int>> occupancyGrid(rows, std::vector<int>(cols, 0));

    // Generate some dummy data
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            occupancyGrid[i][j] = (i + j) % 2;  
        }
    }

    auto start_cpu = std::chrono::high_resolution_clock::now();

    processGridCPU(occupancyGrid); // CPU processing

    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = end_cpu - start_cpu;
    std::cout << "CPU processing time: " << cpu_duration.count() << " seconds\n";

    // Reset the grid to original state before GPU processing
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            occupancyGrid[i][j] = (i + j) % 2;  // Reset pattern
        }
    }

    while(true){
        processOccupancyGrid(occupancyGrid);
        sleep(0.06);
    }

    return 0;
}
