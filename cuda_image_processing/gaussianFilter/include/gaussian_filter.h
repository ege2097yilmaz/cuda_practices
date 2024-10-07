#include <iostream>

/**
 * @brief 
 * 
 * @param sigma 
 */
void generateGaussianKernel(float sigma);

/**
 * @brief 
 * 
 * @param inputImage 
 * @param outputImage 
 * @param width 
 * @param height 
 * @param channels 
 */
void applyGaussianBlur(const unsigned char* inputImage, 
                       unsigned char* outputImage, 
                       int width, int height, int channels);