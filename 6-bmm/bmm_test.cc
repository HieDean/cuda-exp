#include <random>
#include <cmath>
#include <ctime>
#include <vector>
#include <chrono>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "utils.h"
#include "bmm_vx.h"

using KernelFunc = int (*)(const float *, const float *, float *, int, int, int, int, cudaStream_t);

void checkCublasStatus(cublasStatus_t status, const char *operation)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        spdlog::critical("cuBLAS error in {}: {} (status {})",
                         operation, cublasGetStatusString(status),
                         static_cast<int>(status));
        std::exit(EXIT_FAILURE);
    }
}

void helper(KernelFunc func,
            const float *A, const float *B, float *C, int bs, int m, int n, int k, cudaStream_t stream,
            int warmup, int run, std::vector<float> &c, std::string tag)
{
    // // first time, only run for correctness check
    // checkCudaErrors(cudaMemsetAsync(C, 0, c.size() * sizeof(float), stream));
    // func(A, B, C, bs, m, n, k, stream);
    // checkCudaErrors(cudaStreamSynchronize(stream));

    // // check diff
    // std::vector<float> c_from_device(c.size());
    // checkCudaErrors(cudaMemcpy(c_from_device.data(), C,
    //                            c.size() * sizeof(float), cudaMemcpyDeviceToHost));
    // for (int ii = 0; ii < c.size(); ++ii) {
    //     c[ii] = c[ii] / k;
    //     c_from_device[ii] = c_from_device[ii] / k;
    // }
    // check_difference(c, c_from_device, tag);
    // for (int ii = 0; ii < c.size(); ++ii) {
    //     c[ii] = c[ii] * k;
    // }

    // warm up
    for (int ii = 0; ii < warmup; ++ii)
    {
        func(A, B, C, bs, m, n, k, stream);
    }
    checkCudaErrors(cudaStreamSynchronize(stream));

    // create event
    std::vector<cudaEvent_t> starts(run), stops(run);
    std::vector<float> kernelTimes(run);
    for (int ii = 0; ii < run; ++ii)
    {
        cudaEventCreate(&starts[ii]);
        cudaEventCreate(&stops[ii]);
    }

    // run
    for (int ii = 0; ii < run; ++ii)
    {
        cudaEventRecord(starts[ii], stream);
        func(A, B, C, bs, m, n, k, stream);
        cudaEventRecord(stops[ii], stream);
    }
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    // get cost
    for (int ii = 0; ii < run; ++ii)
    {
        cudaEventElapsedTime(&kernelTimes[ii], starts[ii], stops[ii]);
    }
    auto statistics = get_statistics(kernelTimes);
    spdlog::info("{} Cost(ms): min: {} med: {} max: {} avg: {} var: {} p95: {}",
                 tag,
                 statistics.min, statistics.median, statistics.max,
                 statistics.average, statistics.variance, statistics.p95);

    for (int ii = 0; ii < run; ++ii)
    {
        cudaEventDestroy(starts[ii]);
        cudaEventDestroy(stops[ii]);
    }
    return;
}

void cublas_helper(cublasHandle_t handle,
                   const float *A, const float *B, float *C,
                   int bs, int m, int n, int k, cudaStream_t stream,
                   int warmup, int run, std::vector<float> &c)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS is column-major. Interpreting the row-major buffers as transposed
    // column-major matrices gives C^T = B^T * A^T without moving any data.
    auto run_cublas = [&](cublasComputeType_t compute_type, const char *label) {
        checkCublasStatus(
            cublasGemmStridedBatchedEx(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, k,
                &alpha,
                B, CUDA_R_32F, n, static_cast<long long>(k) * n,
                A, CUDA_R_32F, k, static_cast<long long>(m) * k,
                &beta,
                C, CUDA_R_32F, n, static_cast<long long>(m) * n,
                bs,
                compute_type,
                CUBLAS_GEMM_DEFAULT),
            label);
    };

    // // Correctness check.
    // run_cublas(CUBLAS_COMPUTE_32F_PEDANTIC, "cuBLAS FP32");
    // checkCudaErrors(cudaStreamSynchronize(stream));

    // std::vector<float> c_from_device(c.size());
    // checkCudaErrors(cudaMemcpy(c_from_device.data(), C,
    //                            c.size() * sizeof(float), cudaMemcpyDeviceToHost));
    // for (size_t ii = 0; ii < c.size(); ++ii)
    // {
    //     c[ii] /= k;
    //     c_from_device[ii] /= k;
    // }
    // check_difference(c, c_from_device, "cuBLAS FP32");
    // for (size_t ii = 0; ii < c.size(); ++ii)
    // {
    //     c[ii] *= k;
    // }

    auto benchmark_cublas = [&](cublasComputeType_t compute_type, const char *label) {
        for (int ii = 0; ii < warmup; ++ii)
        {
            run_cublas(compute_type, label);
        }
        checkCudaErrors(cudaStreamSynchronize(stream));

        std::vector<cudaEvent_t> starts(run), stops(run);
        std::vector<float> kernelTimes(run);
        for (int ii = 0; ii < run; ++ii)
        {
            checkCudaErrors(cudaEventCreate(&starts[ii]));
            checkCudaErrors(cudaEventCreate(&stops[ii]));
        }

        for (int ii = 0; ii < run; ++ii)
        {
            checkCudaErrors(cudaEventRecord(starts[ii], stream));
            run_cublas(compute_type, label);
            checkCudaErrors(cudaEventRecord(stops[ii], stream));
        }
        checkCudaErrors(cudaStreamSynchronize(stream));

        for (int ii = 0; ii < run; ++ii)
        {
            checkCudaErrors(cudaEventElapsedTime(&kernelTimes[ii], starts[ii], stops[ii]));
        }
        auto statistics = get_statistics(kernelTimes);
        spdlog::info("{} Cost(ms): min: {} med: {} max: {} avg: {} var: {} p95: {}",
                     label,
                     statistics.min, statistics.median, statistics.max,
                     statistics.average, statistics.variance, statistics.p95);

        for (int ii = 0; ii < run; ++ii)
        {
            checkCudaErrors(cudaEventDestroy(starts[ii]));
            checkCudaErrors(cudaEventDestroy(stops[ii]));
        }
    };

    benchmark_cublas(CUBLAS_COMPUTE_32F_PEDANTIC, "cuBLAS FP32");
    benchmark_cublas(CUBLAS_COMPUTE_32F_FAST_TF32, "cuBLAS TF32");
}

int host_bmm(const float *a, const float *b, float *c, int bs, int m, int n, int k)
{
    for (int bi = 0; bi < bs; ++bi)
    {
        for (int mm = 0; mm < m; ++mm)
        {
            for (int nn = 0; nn < n; ++nn)
            {
                c[bi * m * n + mm * n + nn] = 0.0;
                for (int kk = 0; kk < k; ++kk)
                {
                    c[bi * m * n + mm * n + nn] += a[bi * m * k + mm * k + kk] *
                                                   b[bi * n * k + kk * n + nn];
                }
            }
        }
    }
    return 0;
}

int main(int argc, char **argv)
{
    int bs = 16;
    int m = 1024;
    int n = 1024;
    int k = 1024;
    if (argc >= 5)
    {
        bs = std::stoi(argv[1]);
        m = std::stoi(argv[2]);
        n = std::stoi(argv[3]);
        k = std::stoi(argv[4]);
    }

    // init random engine
    std::default_random_engine random_engine;
    random_engine.seed(time(0));
    std::uniform_real_distribution<float> uniform_dist(-1.0, 1.0);

    /* HOST PART */
    // init host mat
    std::vector<float> a, b, c;
    for (int ii = 0; ii < bs * m * k; ++ii)
    {
        a.push_back(uniform_dist(random_engine));
    }
    for (int ii = 0; ii < bs * k * n; ++ii)
    {
        b.push_back(uniform_dist(random_engine));
    }
    for (int ii = 0; ii < bs * m * n; ++ii)
    {
        c.push_back(0.0);
    }

    // // host bmm
    // std::chrono::steady_clock::time_point start_cpu = std::chrono::steady_clock::now();
    // host_bmm(a.data(), b.data(), c.data(), bs, m, n, k);
    // std::chrono::steady_clock::time_point end_cpu = std::chrono::steady_clock::now();
    // spdlog::info("host bmm cost: {}us",
    //              std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count());

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

    // bmm_v0
    helper(bmm_v0, A, B, C, bs, m, n, k, stream, 10, 50, c, "bmm_v0");

    // // bmm_v1_32_8
    // helper(bmm_v1_32_8, A, B, C, bs, m, n, k, stream, 10, 50, c, "bmm_v1_32_8");

    // // bmm_v1_32_16
    // helper(bmm_v1_32_16, A, B, C, bs, m, n, k, stream, 10, 50, c, "bmm_v1_32_16");

    // // bmm_v1_32_32
    // helper(bmm_v1_32_32, A, B, C, bs, m, n, k, stream, 10, 50, c, "bmm_v1_32_32");

    // // bmm_v1_32_64
    // helper(bmm_v1_32_64, A, B, C, bs, m, n, k, stream, 10, 50, c, "bmm_v1_32_64");

    // // bmm_v1_64_4
    // helper(bmm_v1_64_4, A, B, C, bs, m, n, k, stream, 10, 50, c, "bmm_v1_64_4");

    // // bmm_v1_64_8
    // helper(bmm_v1_64_8, A, B, C, bs, m, n, k, stream, 10, 50, c, "bmm_v1_64_8");

    // // bmm_v1_64_16
    // helper(bmm_v1_64_16, A, B, C, bs, m, n, k, stream, 10, 50, c, "bmm_v1_64_16");

    // // bmm_v1_64_32
    // helper(bmm_v1_64_32, A, B, C, bs, m, n, k, stream, 10, 50, c, "bmm_v1_64_32");

    // // bmm_v1_64_64
    // helper(bmm_v1_64_64, A, B, C, bs, m, n, k, stream, 10, 50, c, "bmm_v1_64_64");

    // // bmm_v1_128_4
    // helper(bmm_v1_128_4, A, B, C, bs, m, n, k, stream, 10, 50, c, "bmm_v1_128_4");

    // // bmm_v1_128_8
    // helper(bmm_v1_128_8, A, B, C, bs, m, n, k, stream, 10, 50, c, "bmm_v1_128_8");

    // // bmm_v1_128_16
    // helper(bmm_v1_128_16, A, B, C, bs, m, n, k, stream, 10, 50, c, "bmm_v1_128_16");

    // bmm_v1_128_32
    helper(bmm_v1_128_32, A, B, C, bs, m, n, k, stream, 10, 50, c, "bmm_v1_128_32");

    // // bmm_v1_256_4
    // helper(bmm_v1_256_4, A, B, C, bs, m, n, k, stream, 10, 50, c, "bmm_v1_256_4");

    // // bmm_v1_256_8
    // helper(bmm_v1_256_8, A, B, C, bs, m, n, k, stream, 10, 50, c, "bmm_v1_256_8");

    // // bmm_v1_256_16
    // helper(bmm_v1_256_16, A, B, C, bs, m, n, k, stream, 10, 50, c, "bmm_v1_256_16");

    // bmm_v2
    helper(bmm_v2, A, B, C, bs, m, n, k, stream, 10, 50, c, "bmm_v2");

    // bmm_v3_2_16
    helper(bmm_v3_2_16, A, B, C, bs, m, n, k, stream, 10, 50, c, "bmm_v3_2_16");

    // bmm_v3_4_8
    helper(bmm_v3_4_8, A, B, C, bs, m, n, k, stream, 10, 50, c, "bmm_v3_4_8");

    // // bmm_v3_8_4
    // helper(bmm_v3_8_4, A, B, C, bs, m, n, k, stream, 10, 50, c, "bmm_v3_8_4");

    // // bmm_v3_16_2
    // helper(bmm_v3_16_2, A, B, C, bs, m, n, k, stream, 10, 50, c, "bmm_v3_16_2");

    // bmm_v4
    helper(bmm_v4, A, B, C, bs, m, n, k, stream, 10, 50, c, "bmm_v4");

    // bmm_v5
    helper(bmm_v5, A, B, C, bs, m, n, k, stream, 10, 50, c, "bmm_v5");

    // bmm_v51
    helper(bmm_v51, A, B, C, bs, m, n, k, stream, 10, 50, c, "bmm_v51");

    cublasHandle_t cublas_handle;
    checkCublasStatus(cublasCreate(&cublas_handle), "cublasCreate");
    checkCublasStatus(cublasSetStream(cublas_handle, stream), "cublasSetStream");
    cublas_helper(cublas_handle, A, B, C, bs, m, n, k, stream, 10, 50, c);
    checkCublasStatus(cublasDestroy(cublas_handle), "cublasDestroy");

    // free
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaStreamDestroy(stream);

    return 0;
}
