#ifndef FLUID_SIM_H
#define FLUID_SIM_H

#include <vector>
#include <fstream>
#include <string>

// Fluid simulation parameters
struct FluidSimParams {
    float viscosity;   
    float diffusion;     
    float dt;           
    int gridSize;        
};

// Host function to initialize and run the simulation
void simulate_fluid(const FluidSimParams& params, int steps);

void write_grid_to_csv(const float* field, int gridSize, const std::string& filename);

#endif // FLUID_SIM_H