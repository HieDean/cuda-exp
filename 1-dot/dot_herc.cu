/**
 * nvcc -o dot_herc dot_herc.cu -lcublas
 * for ncu: nvcc -o dot_herc dot_herc.cu -lcublas -lineinfo
 * ncu --set full -o report -f ./dot_herc
 * nsys profile --stats true ./dot_herc
 * for specific arch: nvcc -o dot_herc dot_herc.cu -lcublas -lineinfo -arch=compute_120 -code=sm_120
 */

#include <stdio.h>
#include <random>
#include <cmath>
#include <ctime>
#include <vector>
#include <chrono>

#include "cuda_runtime.h"
#include "cublas_v2.h"

#ifndef checkCudaErrors
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrors(cudaError_t err, const char *file, const int line) {
  if (cudaSuccess != err) {
    const char *errorStr = cudaGetErrorString(err);
    fprintf(stderr,
            "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, "
            "line %i.\n",
            err, errorStr, file, line);
    exit(EXIT_FAILURE);
  }
}
#endif

__global__ void dot_1(float *device_result, const float *device_a, const float *device_b, const int length) {

    extern __shared__ float sum[];

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_threads_per_block = blockDim.x;
    const int id = bid * num_threads_per_block + tid;

    const float *a = device_a + blockIdx.x * blockDim.x;
    const float *b = device_b + blockIdx.x * blockDim.x;

    if (id < length) {
        sum[tid] = a[tid] * b[tid];
    } else {
        sum[tid] = 0.0;
    }
    __syncthreads();

    for (int stride = num_threads_per_block / 2; stride > 32 ; stride>>=1) {
        if (tid < stride) {
            sum[tid] = sum[tid] + sum[tid+stride];
        }
        __syncthreads();
    }

    if (tid < 32) {
        sum[tid] = sum[tid] + sum[tid+32];
        __syncwarp();
        sum[tid] = sum[tid] + sum[tid+16];
        __syncwarp();
        sum[tid] = sum[tid] + sum[tid+8];
        __syncwarp();
        sum[tid] = sum[tid] + sum[tid+4];
        __syncwarp();
        sum[tid] = sum[tid] + sum[tid+2];
        __syncwarp();
        sum[tid] = sum[tid] + sum[tid+1];
    }

    if (tid == 0) {
        device_result[bid] = sum[tid];
    }

    return;
}

__global__ void dot_2(float *device_result_final, const float *device_result, int length) {

    extern __shared__ float sum[];

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_threads_per_block = blockDim.x;

    const int num_nums_per_thread = (length + num_threads_per_block - 1) / num_threads_per_block;
    sum[tid] = 0.0;
    for (int ii = 0; ii < num_nums_per_thread; ++ii) {
        sum[tid] += device_result[tid * num_nums_per_thread + ii];
    }
    __syncthreads();

    for (int stride = num_threads_per_block / 2; stride > 32 ; stride>>=1) {
        if (tid < stride) {
            sum[tid] = sum[tid] + sum[tid+stride];
        }
        __syncthreads();
    }

    if (tid < 32) {
        sum[tid] = sum[tid] + sum[tid+32];
        __syncwarp();
        sum[tid] = sum[tid] + sum[tid+16];
        __syncwarp();
        sum[tid] = sum[tid] + sum[tid+8];
        __syncwarp();
        sum[tid] = sum[tid] + sum[tid+4];
        __syncwarp();
        sum[tid] = sum[tid] + sum[tid+2];
        __syncwarp();
        sum[tid] = sum[tid] + sum[tid+1];
    }

    if (tid == 0) {
        device_result_final[bid] = sum[tid];
    }

    return;
}

int main(int argc, char **argv) {

    int length = std::pow(2, 30);
    if (argc > 1) {
        length = std::stoi(argv[1]);
    }

    // init random engine
    std::default_random_engine random_engine;
    random_engine.seed(time(0));
    std::uniform_real_distribution<float> uniform_dist(-1.0, 1.0);

    // init host vec
    std::vector<float> a, b;
    for (int ii = 0; ii < length; ++ii) {
        a.push_back(uniform_dist(random_engine));
        b.push_back(uniform_dist(random_engine));
    }

    // host compute
    float result = 0;
    std::chrono::steady_clock::time_point start_cpu = std::chrono::steady_clock::now();
    for (int ii = 0; ii < length; ++ii) {
        result += a[ii] * b[ii];
    }
    std::chrono::steady_clock::time_point end_cpu = std::chrono::steady_clock::now();
    printf("host result: %.9f, cost: %ld us\n", result,
        std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu).count()
    );



    /************* gpu *************/
    // init stream
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));

    // config
    int numThreadsPerBlock = 1024;
    int numBlocksPerGrid = (length + numThreadsPerBlock - 1) / numThreadsPerBlock;
    dim3 blockDims(numThreadsPerBlock);
    dim3 gridDims(numBlocksPerGrid);

    // init device vec
    float *device_a, *device_b, *device_result, *device_result_final;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&device_a), a.size() * sizeof(float)));
    checkCudaErrors(cudaMemcpyAsync(device_a, a.data(), a.size() * sizeof(float),
                                    cudaMemcpyHostToDevice, stream));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&device_b), b.size() * sizeof(float)));
    checkCudaErrors(cudaMemcpyAsync(device_b, b.data(), b.size() * sizeof(float),
                                    cudaMemcpyHostToDevice, stream));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&device_result), numBlocksPerGrid * sizeof(float)));
    checkCudaErrors(cudaMemset(device_result, 0, numBlocksPerGrid * sizeof(float)));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&device_result_final), sizeof(float)));
    checkCudaErrors(cudaMemset(device_result_final, 0, sizeof(float)));

    // wait for stream
    checkCudaErrors(cudaStreamSynchronize(stream));

    // warmup
    for (int ii = 0; ii < 5; ++ii) {
        checkCudaErrors(cudaMemsetAsync(device_result, 0, numBlocksPerGrid * sizeof(float), stream));
        checkCudaErrors(cudaMemsetAsync(device_result_final, 0, sizeof(float), stream));
        dot_1<<<gridDims, blockDims, numThreadsPerBlock*sizeof(float), stream>>>(device_result, device_a, device_b, length);
        dot_2<<<1, std::min(1024, numBlocksPerGrid), std::min(1024, numBlocksPerGrid)*sizeof(float), stream>>>(device_result_final, device_result, numBlocksPerGrid);
    }

    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu, stream);
    for (int ii = 0; ii < 10; ++ii) {
        checkCudaErrors(cudaMemsetAsync(device_result, 0, numBlocksPerGrid * sizeof(float), stream));
        checkCudaErrors(cudaMemsetAsync(device_result_final, 0, sizeof(float), stream));
        dot_1<<<gridDims, blockDims, numThreadsPerBlock*sizeof(float), stream>>>(device_result, device_a, device_b, length);
        dot_2<<<1, std::min(1024, numBlocksPerGrid), std::min(1024, numBlocksPerGrid)*sizeof(float), stream>>>(device_result_final, device_result, numBlocksPerGrid);
    }
    cudaEventRecord(stop_gpu, stream);

    cudaEventSynchronize(stop_gpu);
    // checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    // get result
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_gpu, stop_gpu);

    checkCudaErrors(cudaMemcpy(&result, device_result_final, sizeof(float), cudaMemcpyDeviceToHost));
    printf("device result: %.9f, cost: %ld us\n", result, static_cast<int64_t>(milliseconds * 1000 / 10));

    // cublas
    // cublasHandle_t handle;
    // cublasCreate(&handle);
    // cublasSetStream(handle, stream);
    // for (int ii = 0; ii < 5; ++ii) {
    //     cublasSdot(handle, length, device_a, 1, device_b, 1, device_result);
    // }
    // cudaEventRecord(start_gpu);
    // for (int ii = 0; ii < 10; ++ii) {
    //     cudaMemsetAsync(device_result, 0, sizeof(float), stream);
    //     cublasSdot(handle, length, device_a, 1, device_b, 1, device_result);
    // }
    // cudaEventRecord(stop_gpu);
    // cudaEventSynchronize(stop_gpu);

    // cudaEventElapsedTime(&milliseconds, start_gpu, stop_gpu);

    // cudaMemcpy(&result, device_result, sizeof(float), cudaMemcpyDeviceToHost);
    // printf("cublas result: %.9f, cost: %ld us\n", result, static_cast<int64_t>(milliseconds * 1000 / 10));


    // free
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_result);
    cudaStreamDestroy(stream);
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    // cublasDestroy(handle);

    return 0;
}
