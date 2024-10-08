#include "fft_filter.hpp"


#define IMG_WIDTH 512
#define IMG_HEIGHT 512

const std::string output_path = std::string(IMAGE_PATH) + "/output.png";

namespace fft_filter{

    void run_fft_filter(const std::string image_path){

        cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
        if(img.empty()) {
            std::cerr << "failed loading image" << std::endl;
        }

        cv::resize(img, img, cv::Size(IMG_WIDTH, IMG_HEIGHT));

        img.convertTo(img, CV_32F, 1.0 / 255.0);

        float* h_inputImage =(float*)malloc(IMG_WIDTH * IMG_HEIGHT * sizeof(float));
        memcpy(h_inputImage, img.ptr<float>(), IMG_WIDTH * IMG_HEIGHT * sizeof(float));
        float* h_outputImage = (float*)malloc(IMG_HEIGHT * IMG_WIDTH * sizeof(float));

        float* d_inputImage;
        cufftComplex* d_freqDomain;

        float* d_outputImage;

        // allocations
        cudaMalloc(&d_inputImage, IMG_WIDTH * IMG_HEIGHT * sizeof(float));
        cudaMalloc(&d_freqDomain, IMG_WIDTH * IMG_HEIGHT * sizeof(cufftComplex));
        cudaMalloc(&d_outputImage, IMG_WIDTH * IMG_HEIGHT * sizeof(float));

        // copy to device mem
        cudaMemcpy(d_inputImage, h_inputImage, 
                    IMG_WIDTH * IMG_HEIGHT * sizeof(float), cudaMemcpyHostToDevice);

        // perform fat fourier transform filter
        fftForward(d_inputImage, d_freqDomain, IMG_WIDTH, IMG_HEIGHT);

        applyFilter(d_freqDomain, IMG_WIDTH, IMG_HEIGHT, true, 50.0f); // 50 is arbitrary value for cut off

        fftInverse(d_freqDomain, d_outputImage, IMG_WIDTH, IMG_HEIGHT);

        // copy from device to host
        cudaMemcpy(h_outputImage, d_outputImage, IMG_WIDTH * IMG_HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);

        // normalize the image
        for(int i = 0; i < IMG_WIDTH * IMG_HEIGHT; i++){
            h_outputImage[i] /= IMG_WIDTH * IMG_HEIGHT;
        }

        cv::Mat outputImg(IMG_HEIGHT, IMG_WIDTH, CV_32F, h_outputImage);
        cv::imwrite(output_path, outputImg * 255);

        cudaFree(d_inputImage);
        cudaFree(d_freqDomain);
        cudaFree(d_outputImage);
        free(h_inputImage);
        free(h_outputImage);

        std::cout << "FFT-based image filtering completed and saved as filtered_output.jpg." << std::endl;

    }

} // dnamespace