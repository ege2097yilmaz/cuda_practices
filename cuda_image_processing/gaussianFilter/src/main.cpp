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

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#define SIGMA 1.0f

const std::string image_path = std::string(IMAGE_PATH) + "/input.png";
const std::string output_path = std::string(IMAGE_PATH) + "/output.jpg";

int main(int argc, char** argv) {

    std::cout << "inputpath is: " << image_path << std::endl;
    std::cout << "output path is: " << output_path << std::endl;

    int width = 512;
    int height = 512;
    int chanels = 3;

    // unsigned char* input_image = new unsigned char[width * height * chanels];

    unsigned char* inputImage = stbi_load(image_path.c_str(), &width, &height, &chanels, 0);
    if (!inputImage) {
        std::cerr << "Failed to load image!" << std::endl;
        return -1;
    }

    unsigned char* output_image = new unsigned char[width * height * chanels];

    generateGaussianKernel(SIGMA);

    applyGaussianBlur(inputImage,output_image, width, height, chanels);

    if (!stbi_write_jpg(output_path.c_str(), width, height, chanels, output_image, 100)) {
        std::cerr << "Failed to save image!" << std::endl;
        return -1;
    }

    // Cleanup
    stbi_image_free(inputImage);
    delete[] output_image;

    std::cout << "Gaussian blur applied successfully, output saved as output.jpg" << std::endl;
    return 0;
}