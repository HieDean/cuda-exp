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

    /************* host *************/
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
    // printf("host gemm cost: %ldus\n",
    //        std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count());

    /************* device *************/
    // init stream
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));

    // init device mat in global memory
    float *device_a, *device_b, *device_c;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&device_a), a.size() * sizeof(float)));
    checkCudaErrors(cudaMemcpyAsync(device_a, a.data(), a.size() * sizeof(float),
                                    cudaMemcpyHostToDevice, stream));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&device_b), b.size() * sizeof(float)));
    checkCudaErrors(cudaMemcpyAsync(device_b, b.data(), b.size() * sizeof(float),
                                    cudaMemcpyHostToDevice, stream));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&device_c), c.size() * sizeof(float)));
    checkCudaErrors(cudaMemsetAsync(device_c, 0, c.size() * sizeof(float), stream));

    // wait for stream
    checkCudaErrors(cudaStreamSynchronize(stream));

    // create event
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);
    float milliseconds = 0;

    {
        /*** gemm_v0 ***/

        // first time, only for correctness check
        checkCudaErrors(cudaMemsetAsync(device_c, 0, c.size() * sizeof(float), stream));
        gemm_v0(device_a, device_b, device_c, m, n, k, stream);
        checkCudaErrors(cudaStreamSynchronize(stream));

        // get diff
        std::vector<float> c_from_device(c.size());
        checkCudaErrors(cudaMemcpy(c_from_device.data(), device_c, c.size() * sizeof(float), cudaMemcpyDeviceToHost));
        check_difference(c, c_from_device, "gemm_v0 vs host");

        // warm up
        for (int ii = 0; ii < 5; ++ii)
        {
            gemm_v0(device_a, device_b, device_c, m, n, k, stream);
        }
        checkCudaErrors(cudaStreamSynchronize(stream));

        // device gemm
        cudaEventRecord(start_gpu, stream);
        for (int ii = 0; ii < 10; ++ii)
        {
            gemm_v0(device_a, device_b, device_c, m, n, k, stream);
        }
        cudaEventRecord(stop_gpu, stream);
        cudaEventSynchronize(stop_gpu);
        checkCudaErrors(cudaGetLastError());

        // get cost
        cudaEventElapsedTime(&milliseconds, start_gpu, stop_gpu);
        spdlog::info("gemm_v0 cost: {}us",
                     static_cast<int64_t>(milliseconds * 1000.0));
    }

    {
        /*** gemm_v1 ***/

        // first time, only for correctness check
        checkCudaErrors(cudaMemsetAsync(device_c, 0, c.size() * sizeof(float), stream));
        gemm_v1(device_a, device_b, device_c, m, n, k, stream);
        checkCudaErrors(cudaStreamSynchronize(stream));

        // get diff
        std::vector<float> c_from_device(c.size());
        checkCudaErrors(cudaMemcpy(c_from_device.data(), device_c, c.size() * sizeof(float), cudaMemcpyDeviceToHost));
        check_difference(c, c_from_device, "gemm_v1 vs host");

        // warm up
        for (int ii = 0; ii < 5; ++ii)
        {
            gemm_v1(device_a, device_b, device_c, m, n, k, stream);
        }
        checkCudaErrors(cudaStreamSynchronize(stream));

        // device gemm
        cudaEventRecord(start_gpu, stream);
        for (int ii = 0; ii < 10; ++ii)
        {
            gemm_v1(device_a, device_b, device_c, m, n, k, stream);
        }
        cudaEventRecord(stop_gpu, stream);
        cudaEventSynchronize(stop_gpu);
        checkCudaErrors(cudaGetLastError());

        // get cost
        cudaEventElapsedTime(&milliseconds, start_gpu, stop_gpu);
        spdlog::info("gemm_v1 cost: {}us",
                     static_cast<int64_t>(milliseconds * 1000.0));
    }

    {
        /*** gemm_v2 ***/

        // first time, only for correctness check
        checkCudaErrors(cudaMemsetAsync(device_c, 0, c.size() * sizeof(float), stream));
        gemm_v2(device_a, device_b, device_c, m, n, k, stream);
        checkCudaErrors(cudaStreamSynchronize(stream));

        // get diff
        std::vector<float> c_from_device(c.size());
        checkCudaErrors(cudaMemcpy(c_from_device.data(), device_c, c.size() * sizeof(float), cudaMemcpyDeviceToHost));
        check_difference(c, c_from_device, "gemm_v2 vs host");

        // warm up
        for (int ii = 0; ii < 5; ++ii)
        {
            gemm_v2(device_a, device_b, device_c, m, n, k, stream);
        }
        checkCudaErrors(cudaStreamSynchronize(stream));

        // device gemm
        cudaEventRecord(start_gpu, stream);
        for (int ii = 0; ii < 10; ++ii)
        {
            gemm_v2(device_a, device_b, device_c, m, n, k, stream);
        }
        cudaEventRecord(stop_gpu, stream);
        cudaEventSynchronize(stop_gpu);
        checkCudaErrors(cudaGetLastError());

        // get cost
        cudaEventElapsedTime(&milliseconds, start_gpu, stop_gpu);
        spdlog::info("gemm_v2 cost: {}us",
                     static_cast<int64_t>(milliseconds * 1000.0));
    }

    {
        /*** cublas ***/
        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasSetStream(handle, stream);
        float alpha = 1.0, beta = 0.0f;
        checkCudaErrors(cudaMemset(device_c, 0, c.size() * sizeof(float)));

        // first time, only for correctness check
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, device_b, n, device_a, k, &beta, device_c, n);

        // get diff
        std::vector<float> c_from_device(c.size());
        checkCudaErrors(cudaMemcpy(c_from_device.data(), device_c, c.size() * sizeof(float), cudaMemcpyDeviceToHost));
        check_difference(c, c_from_device, "cublasSgemm vs host");

        // warm up
        for (int ii = 0; ii < 5; ++ii)
        {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, device_b, n, device_a, k, &beta, device_c, n);
        }

        // cublasSgemm
        cudaEventRecord(start_gpu, stream);
        for (int ii = 0; ii < 10; ++ii)
        {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, device_b, n, device_a, k, &beta, device_c, n);
        }
        cudaEventRecord(stop_gpu, stream);
        cudaEventSynchronize(stop_gpu);
        checkCudaErrors(cudaGetLastError());

        // get cost
        cudaEventElapsedTime(&milliseconds, start_gpu, stop_gpu);
        spdlog::info("cublasSgemm cost: {}us",
                     static_cast<int64_t>(milliseconds * 1000.0));

        cublasDestroy(handle);
    }

    // free
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
    cudaStreamDestroy(stream);
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    return 0;
}
