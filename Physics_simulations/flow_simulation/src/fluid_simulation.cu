/*
Fluid Simulation CUDA Kernels
*/

#include "fluid_simulation.h"

float *h_velocityX, *h_velocityY, *h_pressure;
float *d_velocityX, *d_velocityY, *d_pressure;
bool stop_simulation = false;

void visualize_simulation() {
    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_POINTS);
    
    float max_velocity = 1.0f; // Set a reference max velocity
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            int idx = y * WIDTH + x;
            float velocity_magnitude = sqrt(h_velocityX[idx] * h_velocityX[idx] + h_velocityY[idx] * h_velocityY[idx]);
            float normalized_velocity = velocity_magnitude / max_velocity;
            
            // Color mapping: Red for high velocity, blue for low velocity
            glColor3f(normalized_velocity, 0.0f, 1.0f - normalized_velocity);
            glVertex2f((float)x / WIDTH * 2 - 1, (float)y / HEIGHT * 2 - 1);
        }
    }

    glEnd();
    glutSwapBuffers();
}

// OpenGL Display Loop
void display() {
    visualize_simulation();
}

void update(int value) {
    if (!stop_simulation) {
        glutPostRedisplay();
        glutTimerFunc(16, update, 0);
    }
}
__global__ void update_velocity(float *velocityX, float *velocityY, float *pressure) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > 0 && x < WIDTH-1 && y > 0 && y < HEIGHT-1) {
        int idx = y * WIDTH + x;
        velocityX[idx] -= TIME_STEP * (pressure[idx + 1] - pressure[idx - 1]);
        velocityY[idx] -= TIME_STEP * (pressure[idx + WIDTH] - pressure[idx - WIDTH]);
    }
}

// CUDA Kernel: Update Velocity Field
__global__ void update_velocity_kernel(float *velocityX, float *velocityY, float *pressure) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > 0 && x < WIDTH - 1 && y > 0 && y < HEIGHT - 1) {
        int idx = y * WIDTH + x;
        // Compute velocity updates using pressure gradient
        velocityX[idx] -= TIME_STEP * (pressure[idx + 1] - pressure[idx - 1]);
        velocityY[idx] -= TIME_STEP * (pressure[idx + WIDTH] - pressure[idx - WIDTH]);
    }
}


// Solve Pressure Poisson Equation
__global__ void pressure_solver_kernel(float *pressure, float *velocityX, float *velocityY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > 0 && x < WIDTH - 1 && y > 0 && y < HEIGHT - 1) {
        int idx = y * WIDTH + x;
        // Approximate pressure using neighboring values
        pressure[idx] = 0.25f * (pressure[idx + 1] + pressure[idx - 1] +
                                 pressure[idx + WIDTH] + pressure[idx - WIDTH] -
                                 (velocityX[idx] - velocityX[idx - 1] +
                                  velocityY[idx] - velocityY[idx - WIDTH]));
    }
}


// Apply Boundary Conditions
__global__ void apply_boundary_conditions_kernel(float *velocityX, float *velocityY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * WIDTH + x;
    
    // No-slip condition on pipeline surfac
    if (y == HEIGHT / 2 && x > PIPE_START && x < PIPE_END) {
        velocityX[idx] = 0.0f;
        velocityY[idx] = 0.0f;
    }
}

void init_simulation() {
    int size = WIDTH * HEIGHT * sizeof(float);
    cudaMalloc((void**)&d_velocityX, size);
    cudaMalloc((void**)&d_velocityY, size);
    cudaMalloc((void**)&d_pressure, size);
    h_velocityX = (float*)malloc(size);
    h_velocityY = (float*)malloc(size);
    h_pressure = (float*)malloc(size);
}

void step_simulation() {
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(WIDTH / BLOCK_SIZE, HEIGHT / BLOCK_SIZE);
    
    update_velocity_kernel<<<gridDim, blockDim>>>(d_velocityX, d_velocityY, d_pressure);
    pressure_solver_kernel<<<gridDim, blockDim>>>(d_pressure, d_velocityX, d_velocityY);
    apply_boundary_conditions_kernel<<<gridDim, blockDim>>>(d_velocityX, d_velocityY);
    
    cudaMemcpy(h_velocityX, d_velocityX, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_velocityY, d_velocityY, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pressure, d_pressure, WIDTH * HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);
}

