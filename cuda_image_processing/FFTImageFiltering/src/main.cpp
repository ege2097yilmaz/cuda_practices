#include "fft_filter.hpp"

const std::string image_path = std::string(IMAGE_PATH) + "/input.png";

int main(){

    std::cout << "starting FFt filtering in GPU" << std::endl;
    
    std::cout << "inputpath is: " << image_path << std::endl;

    fft_filter::run_fft_filter(image_path);

    return 0;
}