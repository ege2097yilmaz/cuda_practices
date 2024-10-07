#include <iostream>

__global__ void add(int *a, int *b, int *c){
    int index = threadIdx.x;
    c[index] = a[index] + b[index];
}

int main(){
    const int SIZE = 10;
    int a[SIZE], b[SIZE], c[SIZE];
    int *d_a, *d_b, *d_c;

    cudaMalloc((void**)&d_a, SIZE * sizeof(int));
    cudaMalloc((void**)&d_b, SIZE * sizeof(int));
    cudaMalloc((void**)&d_c, SIZE * sizeof(int));

    // initialize the 
    for(int i = 0; i < SIZE; i++){
        a[i] = i;
        b[i] = i * i;
    }

    cudaMemcpy(d_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, SIZE * sizeof(int), cudaMemcpyHostToDevice);

    add<<<1, SIZE>>>(d_a, d_b, d_c);

    cudaMemcpy(c, d_c, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < SIZE; i++){
        std::cout << a[i] << " + " << b[i] << " = " << c[i] << std::endl; 
    }
     
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}