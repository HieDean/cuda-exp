#include "dot_vx.h"

__global__ void dot_kernel_v0(const float *A, const float *B, float *C, int length)
{
    int x0 = blockIdx.x * blockDim.x;
    int xx = threadIdx.x;

    atomicAdd(C, A[x0 + xx] * B[x0 + xx]);

    return;
}

int dot_v0(const float *A, const float *B, float *C, int length, cudaStream_t stream)
{
    constexpr int numThreadsPerBlock = 128;
    dim3 blockDims(numThreadsPerBlock);

    int numBlocksPerGrid = (length + numThreadsPerBlock - 1) / numThreadsPerBlock;
    dim3 gridDims(numBlocksPerGrid);

    dot_kernel_v0<<<gridDims, blockDims, 0, stream>>>(A, B, C, length);
    return 0;
}