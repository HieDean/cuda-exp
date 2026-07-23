#include <random>
#include <cmath>
#include <ctime>
#include <vector>
#include <chrono>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "utils.h"
#include "softmax_vx.h"

using KernelFunc = int (*)(const float *, float *, int, cudaStream_t);

void helper(KernelFunc func,
            const float *A, float *B, int n, cudaStream_t stream,
            int warmup, int run, std::vector<float> &b, std::string tag)
{
    // first time, only run for correctness check
    func(A, B, n, stream);
    checkCudaErrors(cudaStreamSynchronize(stream));

    std::vector<float> b_from_device(n);
    checkCudaErrors(cudaMemcpy(b_from_device.data(), B,
                               n * sizeof(float), cudaMemcpyDeviceToHost));

    // check diff
    check_difference(b, b_from_device, tag);

    // warm up
    for (int ii = 0; ii < warmup; ++ii)
    {
        func(A, B, n, stream);
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
        func(A, B, n, stream);
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
    int n = 4096;
    if (argc >= 2)
    {
        n = std::stoi(argv[1]);
    }

    // init random engine
    std::default_random_engine random_engine;
    random_engine.seed(time(0));
    std::uniform_real_distribution<float> uniform_dist(-1.0, 1.0);

    /* HOST PART */
    // init host mat
    std::vector<float> a(n), b(n);
    for (int ii = 0; ii < n; ++ii)
    {
        a[ii] = uniform_dist(random_engine);
    }

    // host softmax
    std::chrono::steady_clock::time_point start_cpu = std::chrono::steady_clock::now();
    float sum = 0.0f;
    float max = -INFINITY;
    for (int ii = 0; ii < n; ++ii)
    {
        max = std::max(max, a[ii]);
    }
    printf("max = %f\n", max);
    for (int ii = 0; ii < n; ++ii)
    {
        b[ii] = std::exp(a[ii] - max);
        sum += b[ii];
    }
    printf("sum = %f\n", sum);
    for (int ii = 0; ii < n; ++ii)
    {
        b[ii] = b[ii] / sum;
    }
    std::chrono::steady_clock::time_point end_cpu = std::chrono::steady_clock::now();
    spdlog::info("host transpose cost: {}us",
                 std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count());

    /* DEVICE PART */
    // init stream
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));

    // init device mat in global memory
    float *A, *B;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&A), n * sizeof(float)));
    checkCudaErrors(cudaMemcpyAsync(A, a.data(), a.size() * sizeof(float),
                                    cudaMemcpyHostToDevice, stream));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&B), n * sizeof(float)));

    // wait for stream
    checkCudaErrors(cudaStreamSynchronize(stream));

    // softmax_v0
    helper(softmax_v0, A, B, n, stream, 5, 10, b, "softmax_v0");

    {
        // // cublasSgemm
        // cublasHandle_t handle;
        // cublasCreate(&handle);
        // cublasSetStream(handle, stream);
        // float alpha = 1.0f, beta = 0.0f;

        // // first time, only for correctness check
        // cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        //             m, n,
        //             &alpha, A, n,
        //             &beta, NULL, m,
        //             B, m);

        // // check diff
        // std::vector<float> b_from_device(m * n);
        // checkCudaErrors(cudaMemcpy(b_from_device.data(), B,
        //                            m * n * sizeof(float), cudaMemcpyDeviceToHost));
        // check_difference(b, b_from_device, "cublasSgemm");

        // // warm up
        // for (int ii = 0; ii < 5; ++ii)
        // {
        //     cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        //                 m, n,
        //                 &alpha, A, n,
        //                 &beta, NULL, m,
        //                 B, m);
        // }

        // // create event
        // cudaEvent_t start_gpu, stop_gpu;
        // cudaEventCreate(&start_gpu);
        // cudaEventCreate(&stop_gpu);
        // float milliseconds = 0;

        // // cublasSgemm
        // cudaEventRecord(start_gpu, stream);
        // for (int ii = 0; ii < 10; ++ii)
        // {
        //     cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        //                 m, n,
        //                 &alpha, A, n,
        //                 &beta, NULL, m,
        //                 B, m);
        // }
        // cudaEventRecord(stop_gpu, stream);
        // cudaEventSynchronize(stop_gpu);
        // checkCudaErrors(cudaGetLastError());

        // // get cost
        // cudaEventElapsedTime(&milliseconds, start_gpu, stop_gpu);
        // spdlog::info("cublasSgemm cost: {}us",
        //              static_cast<int64_t>(milliseconds * 1000.0));

        // cublasDestroy(handle);
        // cudaEventDestroy(start_gpu);
        // cudaEventDestroy(stop_gpu);
    }

    // free
    cudaFree(A);
    cudaFree(B);
    cudaStreamDestroy(stream);

    return 0;
}
