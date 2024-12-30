#include "fluid_sim.h"
#include <iostream>

int main() {
    FluidSimParams params;
    params.viscosity = 0.1f;
    params.diffusion = 0.1f;
    params.dt = 0.1f;
    params.gridSize = 128;

    int simulationSteps = 100;

    std::cout << "Starting fluid simulation...\n";
    simulate_fluid(params, simulationSteps);
    std::cout << "Simulation complete!\n";

    return 0;
}