#include <random>
#include <cmath>
#include <ctime>
#include <vector>
#include <chrono>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "utils.h"
#include "elementwise_vx.h"

using KernelFunc = int (*)(const float *, const float *, float *, int, int, cudaStream_t);

void helper(KernelFunc func,
            const float *A, const float *B, float *C, int m, int n, cudaStream_t stream,
            int warmup, int run, std::vector<float> &c, std::string tag)
{
    // first time, only run for correctness check
    func(A, B, C, m, n, stream);
    checkCudaErrors(cudaStreamSynchronize(stream));

    std::vector<float> c_from_device(m * n);
    checkCudaErrors(cudaMemcpy(c_from_device.data(), C,
                               m * n * sizeof(float), cudaMemcpyDeviceToHost));

    // check diff
    check_difference(c, c_from_device, tag);

    // warm up
    for (int ii = 0; ii < warmup; ++ii)
    {
        func(A, B, C, m, n, stream);
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
        func(A, B, C, m, n, stream);
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
    int m = 4096;
    int n = 4096;
    if (argc == 3)
    {
        m = std::stoi(argv[1]);
        n = std::stoi(argv[2]);
    }

    // init random engine
    std::default_random_engine random_engine;
    random_engine.seed(time(0));
    std::uniform_real_distribution<float> uniform_dist(-1.0, 1.0);

    /* HOST PART */
    // init host mat
    std::vector<float> a(m * n), b(m * n), c(m * n);
    for (int ii = 0; ii < m; ++ii)
    {
        for (int jj = 0; jj < n; ++jj)
        {
            a[ii * n + jj] = uniform_dist(random_engine);
            b[ii * n + jj] = uniform_dist(random_engine);
        }
    }

    // host elementwise
    std::chrono::steady_clock::time_point start_cpu = std::chrono::steady_clock::now();
    for (int ii = 0; ii < m * n; ++ii)
    {
        c[ii] = a[ii] * b[ii];
    }
    std::chrono::steady_clock::time_point end_cpu = std::chrono::steady_clock::now();
    spdlog::info("host elementwise cost: {}us",
                 std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count());

    /* DEVICE PART */
    // init stream
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));

    // init device mat in global memory
    float *A, *B, *C;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&A), m * n * sizeof(float)));
    checkCudaErrors(cudaMemcpyAsync(A, a.data(), a.size() * sizeof(float),
                                    cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&B), m * n * sizeof(float)));
    checkCudaErrors(cudaMemcpyAsync(B, b.data(), b.size() * sizeof(float),
                                    cudaMemcpyHostToDevice, stream));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&C), m * n * sizeof(float)));

    // wait for stream
    checkCudaErrors(cudaStreamSynchronize(stream));

    // elementwise_v0
    helper(elementwise_v0, A, B, C, m, n, stream, 5, 10, c, "elementwise_v0");

    // elementwise_v1
    helper(elementwise_v1, A, B, C, m, n, stream, 5, 10, c, "elementwise_v1");

    // free
    cudaFree(A);
    cudaFree(B);
    cudaStreamDestroy(stream);

    return 0;
}
