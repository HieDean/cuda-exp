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
    int length = std::pow(2, 24);
    if (argc > 1)
    {
        length = std::stoi(argv[1]);
    }

    // init random engine
    std::default_random_engine random_engine;
    random_engine.seed(time(0));
    std::uniform_real_distribution<float> uniform_dist(-1.0, 1.0);

    /* HOST PART */
    // init host mat
    std::vector<float> a, b;
    for (int ii = 0; ii < length; ++ii)
    {
        a.push_back(uniform_dist(random_engine));
        b.push_back(uniform_dist(random_engine));
    }

    // host dot
    std::vector<float> c = {0};
    std::chrono::steady_clock::time_point start_cpu = std::chrono::steady_clock::now();
    for (int ii = 0; ii < length; ++ii)
    {
        c[0] += a[ii] * b[ii];
    }
    std::chrono::steady_clock::time_point end_cpu = std::chrono::steady_clock::now();
    spdlog::info("host dot cost: {}us",
                 std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count());

    /* DEVICE PART */
    // init stream
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));

    // init device mat in global memory
    float *A, *B, *C;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&A), a.size() * sizeof(float)));
    checkCudaErrors(cudaMemcpyAsync(A, a.data(), a.size() * sizeof(float),
                                    cudaMemcpyHostToDevice, stream));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&B), b.size() * sizeof(float)));
    checkCudaErrors(cudaMemcpyAsync(B, b.data(), b.size() * sizeof(float),
                                    cudaMemcpyHostToDevice, stream));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&C), c.size() * sizeof(float)));
    checkCudaErrors(cudaMemsetAsync(C, 0, sizeof(float), stream));

    // wait for stream
    checkCudaErrors(cudaStreamSynchronize(stream));

    // dot_v0
    helper(dot_v0, A, B, C, length, stream, 5, 10, c, "dot_v0");

    {
        // cublasSdot
        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasSetStream(handle, stream);
        checkCudaErrors(cudaMemset(C, 0, sizeof(float)));

        // first time, only for correctness check
        cublasSdot(handle, length, A, 1, B, 1, C);

        // check diff
        std::vector<float> c_from_device = {0};
        checkCudaErrors(cudaMemcpy(c_from_device.data(), C,
                                   c.size() * sizeof(float), cudaMemcpyDeviceToHost));
        check_difference(c, c_from_device, "cublasSdot");

        // warm up
        for (int ii = 0; ii < 5; ++ii)
        {
            cublasSdot(handle, length, A, 1, B, 1, C);
        }

        // create event
        cudaEvent_t start_gpu, stop_gpu;
        cudaEventCreate(&start_gpu);
        cudaEventCreate(&stop_gpu);
        float milliseconds = 0;

        // cublasSgemm
        cudaEventRecord(start_gpu, stream);
        for (int ii = 0; ii < 10; ++ii)
        {
            cublasSdot(handle, length, A, 1, B, 1, C);
        }
        cudaEventRecord(stop_gpu, stream);
        cudaEventSynchronize(stop_gpu);
        checkCudaErrors(cudaGetLastError());

        // get cost
        cudaEventElapsedTime(&milliseconds, start_gpu, stop_gpu);
        spdlog::info("cublasSdot cost: {}us",
                     static_cast<int64_t>(milliseconds * 1000.0));

        cublasDestroy(handle);
        cudaEventDestroy(start_gpu);
        cudaEventDestroy(stop_gpu);
    }

    // free
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaStreamDestroy(stream);

    return 0;
}
