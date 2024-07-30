#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <vector>
#include <chrono>
#include "setup.h"

#define N 100000  // Adjusted size of vectors for 16 GB memory per GPU
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

double flops(uint64_t ni, uint64_t nj, uint64_t nk, int nmm, float msec)
{
  double f = 1.0*(ni*nj*nk*uint64_t(nmm));
  double ret = f / double(msec);
  ret *= 1000.0;
  DOUT("in flops")
  DOUT(ret)
  return ret;
}

// struct event_loop_t {
//   event_loop_t(cudaStream_t s, env_t& e, data_env_t& d)
//     : stream(s), env(e), data(d)
//   {}

//   void run(int n) {
//     while(n != 0) {
//       launch();
//       std::unique_lock lk(m_notify);
//       cv_notify.wait(lk);
//       n -= 1;
//     }
//   }

//   void launch() {
//     vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_A, d_B, d_C, N);

//     check_cuda(cudaStreamAddCallback(
//       stream,
//       [](cudaStream_t stream, cudaError_t status, void* user_data) {
//         event_loop_t* self = reinterpret_cast<event_loop_t*>(user_data);
//         self->callback();
//       },
//       reinterpret_cast<void*>(this),
//       0));
//   };

//   void callback() {
//     {
//       std::unique_lock lk(m_notify);
//       // modify the shared state here (there isn't any)
//     }

//     cv_notify.notify_one();
//   }

//   cudaStream_t stream;
//   env_t& env;
//   data_env_t& data;

//   std::mutex m_notify;
//   std::condition_variable cv_notify;
// };

/* This main function creates only one cuda graph and runs it */
int main_cudagraph01() {
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

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start));
    check_cuda(cudaEventCreate(&stop));
    check_cuda(cudaEventRecord(start, stream));

    auto execstart = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_GRAPHS; ++i) {
        cudaGraphLaunch(instance, stream);
    }
    cudaStreamSynchronize(stream);
    auto execend = std::chrono::high_resolution_clock::now();
    
    check_cuda(cudaEventRecord(stop, stream));
    check_cuda(cudaEventSynchronize(stop));


    float msec = 0.0f;
    check_cuda(cudaEventElapsedTime(&msec, start, stop));

    std::chrono::duration<double, std::milli> elapsed = execend - execstart;
    // std::cout << "Time for 100 CUDA Graphs with stream captures: " << elapsed.count() << " milliseconds\n";
    std::cout << "Time for 100 CUDA Graphs with cudaEventRecord: " << msec << " milliseconds\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaStreamDestroy(stream);
    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(instance);

    return 0;
}

int main_stream01() {
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
    }
    cudaStreamSynchronize(stream);
    
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


int main(int argc, char** argv) {
    main_cudagraph01();
    main_stream01();
    return 0;
}