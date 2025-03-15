/*
Visualization using OpenGL
*/

#include "visualization.h"
#include "fluid_simulation.h"

void display() {
    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_POINTS);
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            int idx = y * WIDTH + x;
            float speed = sqrt(h_velocityX[idx] * h_velocityX[idx] + h_velocityY[idx] * h_velocityY[idx]);
            glColor3f(speed, 0, 1 - speed);
            glVertex2f(x / (float)WIDTH, y / (float)HEIGHT);
        }
    }
    glEnd();
    glutSwapBuffers();
}

void update(int value) {
    step_simulation();
    glutPostRedisplay();
    glutTimerFunc(16, update, 0);
}
