/**
 * nvcc -o dot dot.cu -lcublas
 * for ncu: nvcc -o dot dot.cu -lcublas -lineinfo
 * ncu --set full -o report -f ./dot
 * nsys profile --stats true ./dot
 * for specific arch: nvcc -o dot dot.cu -lcublas -lineinfo -arch=compute_120 -code=sm_120
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

__global__ void dot(float *device_result, const float *device_a, const float *device_b, const int length) {

    // const int bid = blockIdx.z * gridDim.x * gridDim.y +
    //                 blockIdx.y * gridDim.x +
    //                 blockIdx.x;
    // const int tid = threadIdx.z * blockDim.x * blockDim.y +
    //                 threadIdx.y * blockDim.x +
    //                 threadIdx.x;
    // const int num_threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    // const int id = bid * num_threads_per_block + tid;

    extern __shared__ float sum[];

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_threads_per_block = blockDim.x;
    const int id = bid * num_threads_per_block + tid;

    /**
     * vec load
     * test result shows, the perform will drop for a little
     */
    // const float *a = device_a + blockIdx.x * blockDim.x * 4;
    // const float *b = device_b + blockIdx.x * blockDim.x * 4;

    // /**
    //  * multiply
    //  */
    // if (id * 4 + 3 < length) {
        
    //     float4 reg_a = reinterpret_cast<const float4*>(a)[tid];;
    //     float4 reg_b = reinterpret_cast<const float4*>(b)[tid];;
    //     sum[tid] = reg_a.x * reg_b.x + reg_a.y * reg_b.y +
    //                reg_a.z * reg_b.z + reg_a.w * reg_b.w;
    // } else {
    //     sum[tid] = 0.0;
    //     for (int ii = 0; ii < length - id * 4; ++ii) {
    //         sum[tid] += a[tid * 4 + ii] * b[tid * 4 + ii];
    //     }
    // }
    // __syncthreads();

    const float *a = device_a + blockIdx.x * blockDim.x;
    const float *b = device_b + blockIdx.x * blockDim.x;

    /**
     * multiply
     */
    if (id < length) {
        sum[tid] = a[tid] * b[tid];
    } else {
        sum[tid] = 0.0;
    }
    __syncthreads();

    /**
     * reduce v1
     */
    // for (int stride = num_threads_per_block / 2; stride > 0 ; stride>>=1) {
    //     if (tid < stride) {
    //         sum[tid] = sum[tid] + sum[tid+stride];
    //     }
    //     __syncthreads();
    // }

    /**
     * reduce v2
     * 当stride<=32时, 在version1中, 会由于if(tid<stride)的条件判断, 造成warp divergence;
     * version2直接对stride<=32的情况进行循环展开:
     *  1. 由于if(tid<32), 所以不存在warp divergence;
     *  2. 我们只需要第一个warp的结果, 所以不需要__syncthreads()去同步等待block内其他warp的结果;
     */
    for (int stride = num_threads_per_block / 2; stride > 32 ; stride>>=1) {
        if (tid < stride) {
            sum[tid] = sum[tid] + sum[tid+stride];
        }
        __syncthreads();
    }
    /**
     * warp内部的线程只是并行, 但并不同步, 由于计算之间存在依赖关系, 所以这里需要__syncwarp()同步warp内的所有线程;
     * 如果是Volta架构之前, warp内共享一个程序计数器, 每一步执行都是锁步的, 那可以不用__syncwarp(),
     * 但在Volta及Volta之后, warp内的每个线程拥有自己的程序计数器, 那就需要__syncwarp()去同步;
     * 不过__syncwarp()肯定是要比__syncthreads()快很多, 因为前者是warp内的线程同步, 后者是block内的线程同步;
     */
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
        atomicAdd(device_result, sum[tid]);
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

    // init device vec
    float *device_a, *device_b, *device_result;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&device_a), a.size() * sizeof(float)));
    checkCudaErrors(cudaMemcpyAsync(device_a, a.data(), a.size() * sizeof(float),
                                    cudaMemcpyHostToDevice, stream));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&device_b), b.size() * sizeof(float)));
    checkCudaErrors(cudaMemcpyAsync(device_b, b.data(), b.size() * sizeof(float),
                                    cudaMemcpyHostToDevice, stream));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&device_result), sizeof(float)));
    checkCudaErrors(cudaMemset(device_result, 0, sizeof(float)));

    // wait for stream
    checkCudaErrors(cudaStreamSynchronize(stream));

    // dot
    int numThreadsPerBlock = 1024;
    int numBlocksPerGrid = (length + numThreadsPerBlock - 1) / numThreadsPerBlock;
    dim3 blockDims(numThreadsPerBlock);
    dim3 gridDims(numBlocksPerGrid);

    // warmup
    for (int ii = 0; ii < 5; ++ii) {
        checkCudaErrors(cudaMemsetAsync(device_result, 0, sizeof(float), stream));
        dot<<<gridDims, blockDims, numThreadsPerBlock*sizeof(float), stream>>>(device_result, device_a, device_b, length);
    }

    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu, stream);
    for (int ii = 0; ii < 10; ++ii) {
        checkCudaErrors(cudaMemsetAsync(device_result, 0, sizeof(float), stream));
        dot<<<gridDims, blockDims, numThreadsPerBlock*sizeof(float), stream>>>(device_result, device_a, device_b, length);
    }
    cudaEventRecord(stop_gpu, stream);

    cudaEventSynchronize(stop_gpu);
    // checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    // get result
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_gpu, stop_gpu);

    checkCudaErrors(cudaMemcpy(&result, device_result, sizeof(float), cudaMemcpyDeviceToHost));
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
