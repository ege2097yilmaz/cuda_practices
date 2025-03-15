/*
Visualization using OpenGL
*/

#include "visualization.h"
#include "fluid_simulation.h"

#include <iostream>
#include <cstring>

void display() {
    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_POINTS);
    
    // ASCII gradient for terminal visualization
    const char* gradient = " .:-=+*%@#";  // From slowest to fastest
    int grad_size = strlen(gradient) - 1;

    // std::cout << "\033[H"; // Move cursor to top of terminal 

    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            int idx = y * WIDTH + x;
            float speed = sqrt(h_velocityX[idx] * h_velocityX[idx] + h_velocityY[idx] * h_velocityY[idx]);

            // OpenGL rendering
            glColor3f(speed, 0, 1 - speed);
            glVertex2f(x / (float)WIDTH, y / (float)HEIGHT);

            // Terminal visualization
            float normalized_speed = fmin(fmax(speed, 0.0f), 1.0f);
            int char_index = static_cast<int>(normalized_speed * grad_size);
            std::cout << gradient[char_index];  // Print corresponding ASCII character
        }
        std::cout << std::endl; // New row for terminal output
    }

    glEnd();
    glutSwapBuffers();
}

void update(int value) {
    std::cout << "update step" << std::endl;
    step_simulation();
    glutPostRedisplay();
    glutTimerFunc(16, update, 0);
}
