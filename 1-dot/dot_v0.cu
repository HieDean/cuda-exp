#include "dot_vx.h"

__global__ void dot_kernel_v0(const float *A, const float *B, float *C, int length)
{
    extern __shared__ float workload[];

    int id = blockIdx.x * blockDim.x + threadIdx.x;

    const float *a0 = A + blockIdx.x * blockDim.x;
    const float *b0 = B + blockIdx.x * blockDim.x;

    workload[threadIdx.x] = id < length ? a0[threadIdx.x] * b0[threadIdx.x] : 0.0;
    __syncthreads();

    // for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    // {
    //     if (threadIdx.x < stride)
    //     {
    //         workload[threadIdx.x] = workload[threadIdx.x] + workload[threadIdx.x + stride];
    //     }
    //     __syncthreads();
    // }

    // if (threadIdx.x == 0)
    // {
    //     atomicAdd(C, workload[threadIdx.x]);
    // }

    return;
}

int dot_v0(const float *A, const float *B, float *C, int length, cudaStream_t stream)
{
    constexpr int numThreadsPerBlock = 256;
    dim3 blockDims(numThreadsPerBlock);

    int numBlocksPerGrid = (length + numThreadsPerBlock - 1) / numThreadsPerBlock;
    dim3 gridDims(numBlocksPerGrid);

    dot_kernel_v0<<<gridDims, blockDims, numThreadsPerBlock * sizeof(float), stream>>>(A, B, C, length);
    return 0;
}