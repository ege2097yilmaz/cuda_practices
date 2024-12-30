#include "fluid_sim.h"
#include <iostream>

int main() {
    FluidSimParams params;
    params.viscosity = 0.1f;
    params.diffusion = 0.1f;
    params.dt = 0.01f; // smaller value for accuracy
    params.gridSize = 128;

    float total_time = 55.0f; 
    int simulationSteps = static_cast<int>(total_time / params.dt);

    std::cout << "Starting fluid simulation...\n";
    simulate_fluid(params, simulationSteps);
    std::cout << "Simulation complete!\n";

    return 0;
}