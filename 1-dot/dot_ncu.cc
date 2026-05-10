#include <random>
#include <cmath>
#include <ctime>
#include <vector>
#include <chrono>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "utils.h"
#include "dot_vx.h"

using KernelFunc = int (*)(const float *, const float *, float *, int, cudaStream_t);

void helper(KernelFunc func,
            const float *A, const float *B, float *C, int length, cudaStream_t stream,
            int warmup, int run, std::vector<float> &c, std::string tag)
{
    // first time, only run for correctness check
    checkCudaErrors(cudaMemsetAsync(C, 0, c.size() * sizeof(float), stream));
    func(A, B, C, length, stream);
    checkCudaErrors(cudaStreamSynchronize(stream));

    // check diff
    std::vector<float> c_from_device = {0};
    checkCudaErrors(cudaMemcpy(c_from_device.data(), C,
                               c.size() * sizeof(float), cudaMemcpyDeviceToHost));
    check_difference(c, c_from_device, tag);

    // warm up
    for (int ii = 0; ii < warmup; ++ii)
    {
        func(A, B, C, length, stream);
    }
    checkCudaErrors(cudaStreamSynchronize(stream));

    // create event
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    float milliseconds = 0;

    // run
    cudaEventRecord(start_gpu, stream);
    for (int ii = 0; ii < run; ++ii)
    {
        func(A, B, C, length, stream);
    }
    cudaEventRecord(stop_gpu, stream);
    cudaEventSynchronize(stop_gpu);
    checkCudaErrors(cudaGetLastError());

    // get cost
    cudaEventElapsedTime(&milliseconds, start_gpu, stop_gpu);
    spdlog::info("{} cost: {}us", tag, static_cast<int64_t>(milliseconds * 1000.0));

    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);
    return;
}

int main(int argc, char **argv)
{
    int length = std::pow(2, 10);
    if (argc > 1)
    {
        length = std::stoi(argv[1]);
    }

    /* DEVICE PART */
    // init stream
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));

    // init device mat in global memory
    float *A, *B, *C;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&A), length * sizeof(float)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&B), length * sizeof(float)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&C), sizeof(float)));
    checkCudaErrors(cudaStreamSynchronize(stream));

    // dot_v0
    dot_v0(A, B, C, length, stream);
    checkCudaErrors(cudaStreamSynchronize(stream));
    checkCudaErrors(cudaGetLastError());

    // free
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaStreamDestroy(stream);

    return 0;
}
