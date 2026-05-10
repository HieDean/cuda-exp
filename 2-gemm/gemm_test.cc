/**
 * simple compile:
 *     nvcc -o gemm gemm.cu -lspdlog -lfmt -lcudart -lcublas
 * for ncu and nsys:
 *     nvcc -o gemm gemm.cu -lspdlog -lfmt -lcudart -lcublas -lineinfo
 *     ncu --set full -o report -f ./gemm
 *     nsys profile --stats true ./gemm
 * for specific arch:
 *     nvcc -o gemm gemm.cu -lspdlog -lfmt -lcudart -lcublas -lineinfo -arch=compute_120 -code=sm_120
 */

#include <random>
#include <cmath>
#include <ctime>
#include <vector>
#include <chrono>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "utils.h"
#include "gemm_vx.h"

using KernelFunc = int (*)(const float *, const float *, float *, int, int, int, cudaStream_t);

void helper(KernelFunc func,
            const float *A, const float *B, float *C, int m, int n, int k, cudaStream_t stream,
            int warmup, int run, std::vector<float> &c, std::string tag)
{
    // first time, only run for correctness check
    checkCudaErrors(cudaMemsetAsync(C, 0, c.size() * sizeof(float), stream));
    func(A, B, C, m, n, k, stream);
    checkCudaErrors(cudaStreamSynchronize(stream));

    // check diff
    std::vector<float> c_from_device(c.size());
    checkCudaErrors(cudaMemcpy(c_from_device.data(), C,
                               c.size() * sizeof(float), cudaMemcpyDeviceToHost));
    check_difference(c, c_from_device, tag);

    // warm up
    for (int ii = 0; ii < warmup; ++ii)
    {
        func(A, B, C, m, n, k, stream);
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
        func(A, B, C, m, n, k, stream);
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

int host_gemm(const float *a, const float *b, float *c, int m, int n, int k)
{
    for (int mm = 0; mm < m; ++mm)
    {
        for (int nn = 0; nn < n; ++nn)
        {
            c[mm * n + nn] = 0.0;
            for (int kk = 0; kk < k; ++kk)
            {
                c[mm * n + nn] += a[mm * k + kk] * b[kk * n + nn];
            }
        }
    }
    return 0;
}

int main(int argc, char **argv)
{

    int m = 1024;
    int n = 1024;
    int k = 1024;
    if (argc == 4)
    {
        m = std::stoi(argv[1]);
        n = std::stoi(argv[2]);
        k = std::stoi(argv[3]);
    }

    // init random engine
    std::default_random_engine random_engine;
    random_engine.seed(time(0));
    std::uniform_real_distribution<float> uniform_dist(-1.0, 1.0);

    /* HOST PART */
    // init host mat
    std::vector<float> a, b, c;
    for (int ii = 0; ii < m * k; ++ii)
    {
        a.push_back(uniform_dist(random_engine));
    }
    for (int ii = 0; ii < k * n; ++ii)
    {
        b.push_back(uniform_dist(random_engine));
    }
    for (int ii = 0; ii < m * n; ++ii)
    {
        c.push_back(0.0);
    }

    // host gemm
    std::chrono::steady_clock::time_point start_cpu = std::chrono::steady_clock::now();
    host_gemm(a.data(), b.data(), c.data(), m, n, k);
    std::chrono::steady_clock::time_point end_cpu = std::chrono::steady_clock::now();
    spdlog::info("host gemm cost: {}us",
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

    // gemm_v0
    helper(gemm_v0, A, B, C, m, n, k, stream,
           5, 10, c, "gemm_v0");

    // gemm_v1
    helper(gemm_v1, A, B, C, m, n, k, stream,
           5, 10, c, "gemm_v1");

    // gemm_v2
    helper(gemm_v2, A, B, C, m, n, k, stream,
           5, 10, c, "gemm_v2");

    // gemm_v3
    helper(gemm_v3, A, B, C, m, n, k, stream,
           5, 10, c, "gemm_v3");

    {
        // cublasSgemm
        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasSetStream(handle, stream);
        float alpha = 1.0, beta = 0.0f;
        checkCudaErrors(cudaMemset(C, 0, c.size() * sizeof(float)));

        // first time, only for correctness check
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, n, A, k, &beta, C, n);

        // check diff
        std::vector<float> c_from_device(c.size());
        checkCudaErrors(cudaMemcpy(c_from_device.data(), C, c.size() * sizeof(float), cudaMemcpyDeviceToHost));
        check_difference(c, c_from_device, "cublasSgemm");

        // warm up
        for (int ii = 0; ii < 5; ++ii)
        {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, n, A, k, &beta, C, n);
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
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, n, A, k, &beta, C, n);
        }
        cudaEventRecord(stop_gpu, stream);
        cudaEventSynchronize(stop_gpu);
        checkCudaErrors(cudaGetLastError());

        // get cost
        cudaEventElapsedTime(&milliseconds, start_gpu, stop_gpu);
        spdlog::info("cublasSgemm cost: {}us",
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
