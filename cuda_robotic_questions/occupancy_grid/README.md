# Occupancy Grid Processing with CUDA and matplotlib-cpp

This project demonstrates how to process an occupancy grid using both CPU and GPU (with CUDA) and visualize the results using `matplotlib-cpp`. The goal is to compare the processing speeds of the CPU and GPU, and to visualize the processed data.

## Features

- **Occupancy Grid Processing**: An occupancy grid is represented as a 2D array, where each element indicates whether a cell is occupied or free.
- **CUDA Integration**: The grid is processed on the GPU using CUDA for parallel processing, with a simple kernel flipping occupancy values.
- **CPU vs GPU Performance Comparison**: The project includes both CPU and GPU implementations of the grid processing algorithm, allowing for a comparison of their speeds.
- **Visualization with matplotlib-cpp**: The grid is visualized using `matplotlib-cpp`, which provides a C++ interface to Python's matplotlib for generating plots.

## Requirements

- **C++11** or later
- **CUDA Toolkit** (version 9.0 or later)
- **CMake** (version 3.10 or later)
- **Python** (with `matplotlib` installed)
- **matplotlib-cpp** (included as an external library)

## Installation

### 1. Clone the repository

```bash
cd occupancy-grid-cuda
git clone https://github.com/your-repo/occupancy-grid-cuda.git
```

2. Install dependencies
Ensure you have the following installed on your system:

CUDA: CUDA Toolkit
CMake: CMake Installation Guide

Python with matplotlib:
pip install matplotlib

3. Clone matplotlib-cpp
```bash
Clone the matplotlib-cpp library inside the external/ directory:
git clone https://github.com/lava/matplotlib-cpp.git external/matplotlib-cpp
```

4. Build the project
Create a build directory and run CMake:

```bash
mkdir build
cd build
cmake ..
make
```

5. Run the program
Once the project is built, you can run the program:

```bash
./OccupancyGridProcessor
```

# Usage
The program will process a randomly generated occupancy grid of a specified size (default is 1000x1000). It will process the grid on both the CPU and GPU, print the processing times for each, and visualize the processed grid using matplotlib.

## Expected Output;
CPU Processing Time: The time taken for the CPU to process the grid will be displayed in the terminal.
GPU Processing Time: The time taken for the GPU to process the grid (including data transfers to and from the GPU) will also be displayed in the terminal.


CPU processing time: X.XXXXX seconds
GPU processing time (with memory transfer): Y.YYYYY seconds
A window will also pop up with a visualization of the processed grid.
