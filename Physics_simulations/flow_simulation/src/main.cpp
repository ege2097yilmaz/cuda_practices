/*
CUDA 2D Pipe Flow Simulation
- Uses Navier-Stokes equation (simplified)
- CUDA parallel processing
- OpenGL for visualization
*/

#include "fluid_simulation.h"
#include "visualization.h"

int main(int argc, char **argv) {
    init_simulation();
    
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(WIDTH * 5, HEIGHT * 5);
    glutCreateWindow("CUDA 2D Pipe Flow");
    glutDisplayFunc(display);
    glutTimerFunc(16, update, 0);
    
    glutMainLoop();
    return 0;
}
