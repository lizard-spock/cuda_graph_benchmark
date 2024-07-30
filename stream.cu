#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <vector>
#include <chrono>
#include "setup.h"


#define N 10000000  // Adjusted size of vectors for 16 GB memory per GPU
#define NUM_GRAPHS 100
#define NUM_OPERATIONS 200

__global__ void vectorAdd(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void _check_cuda(cudaError_t err, std::string filename, int line) {
  if(err != cudaSuccess) {
    throw std::runtime_error(
      "cuda no success at " + filename + ":" + write_with_ss(line));
  }
}
#define check_cuda(call) _check_cuda(call, __FILE__, __LINE__)

int main() {
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    // Initialize vectors
    std::vector<float> h_A(N, 1.0f);
    std::vector<float> h_B(N, 2.0f);
    cudaMemcpy(d_A, h_A.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start));
    check_cuda(cudaEventCreate(&stop));
    check_cuda(cudaEventRecord(start, stream));
    
    for (int i = 0; i < NUM_GRAPHS; ++i) {
        for (int j = 0; j < NUM_OPERATIONS; ++j) {
            vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_A, d_B, d_C, N);
        }
        cudaStreamSynchronize(stream);
    }
    
    check_cuda(cudaEventRecord(stop, stream));
    check_cuda(cudaEventSynchronize(stop));
    float msec = 0.0f;
    check_cuda(cudaEventElapsedTime(&msec, start, stop));

    std::cout << "Time for 100 stream with cudaEventRecord: " << msec << " milliseconds\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaStreamDestroy(stream);

    return 0;
}
