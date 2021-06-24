//
// Created by paul on 01/07/2020.
//

#include <iostream>

__global__ void kernel(float* buffer) {
    buffer[threadIdx.x] = threadIdx.x;
}

void init_cuda() {
    cudaDeviceSynchronize();
#if defined(DEBUG)
    std::cout << "Initialising CUDA..." << std::endl;
#endif
    float* dummy_buffer;
    cudaMalloc(&dummy_buffer, 100);
    kernel<<<100, 1>>>(dummy_buffer);
    cudaDeviceSynchronize();
    cudaFree(dummy_buffer);
#if defined(DEBUG)
    std::cout << "Done initialising CUDA..." << std::endl;
#endif
}