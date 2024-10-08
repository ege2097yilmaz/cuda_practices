#include "fft_filter.hpp"

__global__ void applyFilterKernel(cufftComplex* d_freq, int width, 
                                  int height, bool lowPass, float cutoff){
        
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int idx = y * width + x;

    float dist = sqrtf((x - width / 2) * (x - width / 2) + (y - height / 2) * (y - height / 2));

    if(lowPass){
        if(dist > cutoff){
            d_freq[idx].x = 0;
            d_freq[idx].y = 0;
        }
    }else{
        if(dist < cutoff){
            d_freq[idx].x = 0;
            d_freq[idx].y = 0;
        }
    }
}

void fft_filter::applyFilter(cufftComplex* d_freq, int width, int height,
                     bool lowPass, float cutoff)
{
    //definig lbock and grid size
    dim3 blockSize(16, 16); // 256 block 
    dim3 gridSize((width + blockSize.x - 1 / blockSize.x), (height + blockSize.y - 1) / blockSize.y);

    applyFilterKernel<<<gridSize, blockSize>>>(d_freq, width, height, lowPass, cutoff);

    cudaDeviceSynchronize();
}

void fft_filter::fftForward(float* d_inputImage, cufftComplex* d_freqDomain,
                        int width, int height)
{
    cufftHandle plan;
    cufftPlan2d(&plan, width, height, CUFFT_R2C);

    cufftExecR2C(plan, d_inputImage, d_freqDomain);

    cufftDestroy(plan);
}


void fft_filter::fftInverse(cufftComplex* d_freeDomain, float* d_outputImage, 
                    int width, int height)
{
    cufftHandle plan;
    cufftPlan2d(&plan, width, height, CUFFT_C2R);

    cufftExecC2R(plan, d_freeDomain, d_outputImage);

    cufftDestroy(plan);
}