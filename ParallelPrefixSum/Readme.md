# Parallel Prefix Sum (Scan) with CUDA

This project demonstrates the implementation of a Parallel Prefix Sum (Scan) algorithm using CUDA. The algorithm leverages the GPU's parallel processing capabilities to efficiently compute prefix sums on large input arrays.

## Project Structure
    ParallelPrefixSum/
    ├── CMakeLists.txt        # Build system configuration
    ├── src/
    │   ├── main.cu           # Main entry point for the program
    │   ├── prefix_sum.cu     # Implementation of the prefix sum algorithm
    ├── include/
    │   └── prefix_sum.h      # Header file for prefix sum functions
    └── build/                # Directory for compiled files (created during build)

## Algorithm Overview

The prefix sum (also known as scan) is a fundamental parallel algorithm that computes all partial sums of an input array. For an input array `A` of size `n`, the output array `B` is calculated as:

` B[i] = A[0] + A[1] + ... + A[i], for all i in [0, n-1]. `

This implementation uses the exclusive scan variant and employs the following phases:

1. Up-Sweep Phase (Reduce):

    * Compute partial sums in a tree-like manner.

    * Each thread operates on a segment of the array, storing intermediate results in shared memory.

2. Down-Sweep Phase:

    * Propagate the results back down the tree to compute the final prefix sums.

Shared memory is used for efficient inter-thread communication.

## Requirements

* CUDA-capable GPU

* CMake (version 3.12 or higher)

* C++17 and CUDA 11.0 or later

## Building and Running the Project

1. Clone the repository and navigate to the project directory:

```bash
git clone <repository_url>
cd ParallelPrefixSum
```

2. Create a build directory and configure the project with CMake:

```bash
mkdir build
cd build
cmake ..
```

3. Build the project:

```bash
make
```

4. Run

```bash
./prefix_sum
```

Example Output:

```bash
    [0, 1, 3, 6, 10, 15, 21, 28]
```

## File Descriptions

* CMakeLists.txt: Defines the project structure, compiler options, and dependencies.

* main.cu: Contains the main function that initializes input data and invokes the prefix sum function.

* prefix_sum.cu: Implements the CUDA kernel for prefix sum computation and the host function wrapper.

* prefix_sum.h: Declares the host function for prefix sum computation.