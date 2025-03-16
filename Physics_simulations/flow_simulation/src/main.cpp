/*
CUDA 2D Pipe Flow Simulation
- Uses Navier-Stokes equation (simplified)
- CUDA parallel processing
- OpenGL for visualization
*/

#include "fluid_simulation.h"
// #include "visualization.h"

int main(int argc, char** argv) {
    init_simulation();
    
    // Initialize OpenGL
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(WIDTH * 5, HEIGHT * 5);
    glutCreateWindow("CUDA 2D Pipeline Flow");
    glutDisplayFunc(display);
    glutTimerFunc(16, update, 0);
    
    // Start simulation and user input handling in separate threads
    std::thread simulation_thread(step_simulation);
    std::thread input_thread(handle_user_input);
    
    glutMainLoop();
    
    simulation_thread.join();
    input_thread.join();
    return 0;
}
