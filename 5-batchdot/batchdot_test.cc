#include <random>
#include <cmath>
#include <ctime>
#include <vector>
#include <chrono>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "utils.h"
#include "batchdot_vx.h"

using KernelFunc = int (*)(const float *, const float *, float *, int, int, cudaStream_t);

void helper(KernelFunc func,
            const float *A, const float *B, float *C, int bs, int length, cudaStream_t stream,
            int warmup, int run, std::vector<float> &c, std::string tag)
{
    // first time, only run for correctness check
    checkCudaErrors(cudaMemsetAsync(C, 0, c.size() * sizeof(float), stream));
    func(A, B, C, bs, length, stream);
    checkCudaErrors(cudaStreamSynchronize(stream));

    std::vector<float> c_from_device(c.size(), 0);
    checkCudaErrors(cudaMemcpy(c_from_device.data(), C,
                               c.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // eliminate error accumulation
    for (int i = 0; i < bs; ++i) {
        c[i] /= length;
        c_from_device[i] /= length;
    }

    // check diff
    check_difference(c, c_from_device, tag);

    // warm up
    for (int ii = 0; ii < warmup; ++ii)
    {
        func(A, B, C, bs, length, stream);
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
        func(A, B, C, bs, length, stream);
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
    int bs = 1024;
    int length = std::pow(2, 12);
    if (argc > 2)
    {
        bs = std::stoi(argv[1]);
        length = std::stoi(argv[2]);
    }

    // init random engine
    std::default_random_engine random_engine;
    random_engine.seed(time(0));
    std::uniform_real_distribution<float> uniform_dist(-1.0, 1.0);

    /* HOST PART */
    // init host mat
    std::vector<float> a, b;
    for (int ii = 0; ii < bs * length; ++ii)
    {
        a.push_back(uniform_dist(random_engine));
        b.push_back(uniform_dist(random_engine));
    }

    // host batchdot
    std::vector<float> c(bs, 0);
    std::chrono::steady_clock::time_point start_cpu = std::chrono::steady_clock::now();
    for (int ii = 0; ii < bs; ++ii)
    {
        for (int jj = 0; jj < length; ++jj)
        {
            c[ii] += a[ii * length + jj] * b[ii * length + jj];
        }
    }

    std::chrono::steady_clock::time_point end_cpu = std::chrono::steady_clock::now();
    spdlog::info("host batchdot cost: {}us",
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
    checkCudaErrors(cudaMemsetAsync(C, 0, c.size() * sizeof(float), stream));

    // wait for stream
    checkCudaErrors(cudaStreamSynchronize(stream));

    // batchdot_v0
    helper(batchdot_v0, A, B, C, bs, length, stream, 5, 10, c, "batchdot_v0");

    {
        // cuda 似乎没有针对 [bs, length] \dot [bs, length] => [bs] 这种场景的实现, 所以这里就不对比了;
        // (deepseek 老师告诉我的)
    }

    // free
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaStreamDestroy(stream);

    return 0;
}
