#include "gaussian_filter.h"
#include "cuda_runtime.h"
#include "iostream"


#define KERNEL_RADIUS 2
#define KERNEL_WIDTH (2 * KERNEL_RADIUS + 1)

__constant__ float d_kernel[KERNEL_WIDTH * KERNEL_WIDTH];

__global__ void gaussianBlurKernel(const unsigned char* inputImage, unsigned char* outputImage, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = (y * width + x) * channels;

    if (x >= width || y >= height) return;

    float sum[3] = {0.0f, 0.0f, 0.0f};
    int kernelIndex = 0;

    for (int ky = -KERNEL_RADIUS; ky <= KERNEL_RADIUS; ++ky) {
        for (int kx = -KERNEL_RADIUS; kx <= KERNEL_RADIUS; ++kx) {
            int xx = min(max(x + kx, 0), width - 1);
            int yy = min(max(y + ky, 0), height - 1);
            int imgIdx = (yy * width + xx) * channels;

            for (int c = 0; c < channels; ++c) {
                sum[c] += inputImage[imgIdx + c] * d_kernel[kernelIndex];
            }
            kernelIndex++;
        }
    }

    for (int c = 0; c < channels; ++c) {
        outputImage[idx + c] = min(max(int(sum[c]), 0), 255);
    }
}

void applyGaussianBlur(const unsigned char* inputImage, unsigned char* outputImage, int width, int height, int channels) {
    unsigned char *d_inputImage, *d_outputImage;

    size_t imgSize = width * height * channels * sizeof(unsigned char);

    cudaMalloc(&d_inputImage, imgSize);
    cudaMalloc(&d_outputImage, imgSize);

    cudaMemcpy(d_inputImage, inputImage, imgSize, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    gaussianBlurKernel<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, width, height, channels);

    cudaMemcpy(outputImage, d_outputImage, imgSize, cudaMemcpyDeviceToHost);

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
}

void generateGaussianKernel(float sigma) {
    float kernel[KERNEL_WIDTH * KERNEL_WIDTH];
    float sum = 0.0f;
    int index = 0;

    for (int y = -KERNEL_RADIUS; y <= KERNEL_RADIUS; ++y) {
        for (int x = -KERNEL_RADIUS; x <= KERNEL_RADIUS; ++x) {
            float value = expf(-(x * x + y * y) / (2 * sigma * sigma));
            kernel[index++] = value;
            sum += value;
        }
    }

    for (int i = 0; i < KERNEL_WIDTH * KERNEL_WIDTH; ++i) {
        kernel[i] /= sum;
    }

    cudaMemcpyToSymbol(d_kernel, kernel, KERNEL_WIDTH * KERNEL_WIDTH * sizeof(float));
}