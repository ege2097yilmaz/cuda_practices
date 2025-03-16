#ifndef FLUID_SIMULATION_H
#define FLUID_SIMULATION_H

#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <thread>
#include <GL/glut.h>

#define WIDTH 128
#define HEIGHT 64
#define BLOCK_SIZE 16
#define TIME_STEP 0.01f
#define PIPE_START 40
#define PIPE_END 88

extern float *h_velocityX, *h_velocityY, *h_pressure;
extern float *d_velocityX, *d_velocityY, *d_pressure;
extern bool stop_simulation;

void init_simulation();
void step_simulation();
void handle_user_input();
void visualize_simulation();
void display();
void update(int value);

__global__ void update_velocity_kernel(float *velocityX, float *velocityY, float *pressure);
__global__ void pressure_solver_kernel(float *pressure, float *velocityX, float *velocityY);
__global__ void apply_boundary_conditions_kernel(float *velocityX, float *velocityY);

#endif
