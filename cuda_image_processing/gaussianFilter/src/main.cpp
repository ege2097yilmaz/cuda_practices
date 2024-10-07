/**
 * @author Ege YILMAZ
 * @brief basic cuda gaussian filter for image processing
 * @version 0.1
 * @date 2024-10-07
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "gaussian_filter.h"

#define SIGMA 1.0f

// todo stb kütüphanesi ile image çekmeye bak !!!

int main(){
    int width = 512;
    int height = 512;
    int chanels = 3;

    // unsigned char* input_image = new unsigned char[width * height * chanels];

    unsigned char* inputImage = stbi_load("input.jpg", &width, &height, &chanels, 0);
    if (!inputImage) {
        std::cerr << "Failed to load image!" << std::endl;
        return -1;
    }

    unsigned char* output_image = new unsigned char[width * height * chanels];

    generateGaussianKernel(SIGMA);

    applyGaussianBlur(inputImage,output_image, width, height, chanels);

    if (!stbi_write_jpg("output.jpg", width, height, chanels, output_image, 100)) {
        std::cerr << "Failed to save image!" << std::endl;
        return -1;
    }

    // Cleanup
    stbi_image_free(inputImage);
    delete[] output_image;

    std::cout << "Gaussian blur applied successfully, output saved as output.jpg" << std::endl;
    return 0;
}