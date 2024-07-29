#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <vector>
#include <chrono>

#define N 10000000  // Adjusted size of vectors for 16 GB memory per GPU
#define NUM_GRAPHS 100
#define NUM_OPERATIONS 20

__global__ void vectorAdd(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

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

    bool graphCreated = false;
    cudaGraph_t graph;
    cudaGraphExec_t instance;

    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_exec_time;
    for (int i = 0; i < NUM_GRAPHS; ++i) {
        if (!graphCreated) {
            auto initstart = std::chrono::high_resolution_clock::now();
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
            for (int j = 0; j < NUM_OPERATIONS; ++j) {
                vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_A, d_B, d_C, N);
            }
            cudaStreamEndCapture(stream, &graph);
            cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
            graphCreated = true;
            auto initend = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed = initend - initstart;
            std::cout << "Graph Create time: " << elapsed.count() << " milliseconds\n";
        }
        auto execstart = std::chrono::high_resolution_clock::now();
        cudaGraphLaunch(instance, stream);
        cudaStreamSynchronize(stream);
        auto execend = std::chrono::high_resolution_clock::now();
        total_exec_time = total_exec_time + execend - execstart;

    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Time for 100 CUDA Graphs with stream captures: " << elapsed.count() << " milliseconds\n";
    std::cout << "Exec time: " << total_exec_time.count() << " milliseconds\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaStreamDestroy(stream);
    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(instance);

    return 0;
}
