#ifndef FLUID_SIMULATION_H
#define FLUID_SIMULATION_H

#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>

#define WIDTH 128
#define HEIGHT 64
#define BLOCK_SIZE 16
#define TIME_STEP 0.01f

#define PIPE_START (WIDTH / 4)  
#define PIPE_END (3 * WIDTH / 4)

extern float *h_velocityX, *h_velocityY, *h_pressure;
extern float *d_velocityX, *d_velocityY, *d_pressure;

void init_simulation();
void step_simulation();

__global__ void update_velocity(float *velocityX, float *velocityY, float *pressure);
__global__ void update_velocity_kernel(float *velocityX, float *velocityY, float *pressure);
__global__ void pressure_solver_kernel(float *pressure, float *velocityX, float *velocityY);
__global__ void apply_boundary_conditions_kernel(float *velocityX, float *velocityY);


#endif
