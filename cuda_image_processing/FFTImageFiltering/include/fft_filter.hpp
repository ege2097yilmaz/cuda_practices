#ifndef FFT_FILTER_HPP
#define FFT_FILTER_HPP

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>


namespace fft_filter{
    /**
     * @brief 
     * 
     * @param image_path 
     */
    void run_fft_filter(const std::string image_path);

    /**
     * @brief 
     * 
     */
    void applyFilter(cufftComplex* d_freq, int width, int height,
                     bool lowPass, float cutoff);

    /**
     * @brief 
     * 
     * @param d_inputImage 
     * @param d_freqDomain 
     * @param width 
     * @param height 
     */
    void fftForward(float* d_inputImage, cufftComplex* d_freqDomain,
                    int width, int height);

    /**
     * @brief 
     * 
     * @param d_freeDomain 
     * @param d_outputImage 
     * @param width 
     * @param height 
     */
    void fftInverse(cufftComplex* d_freeDomain, float* d_outputImage, 
                    int width, int height);
}

#endif // FFT_FILTER_H
