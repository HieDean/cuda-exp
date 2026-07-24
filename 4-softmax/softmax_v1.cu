#include <cstdio>
#include "softmax_vx.h"

template <int BLOCKSIZE, int WARPSIZE, int NUMWARPS, int MAXNUMELEMENTSPERTHREAD>
__global__ void softmax_kernel_v1(const float *A, float *B, int bs, int num)
{
    __shared__ float shared_sum[NUMWARPS];
    __shared__ float shared_max[NUMWARPS];

    // 用 BLOCKSIZE 个线程处理 num 个数;
    // 每个线程处理 numElements 个数;
    int numElements = (num + BLOCKSIZE - 1) / BLOCKSIZE;
    float regA[MAXNUMELEMENTSPERTHREAD];
    float sum = 0.0f;
    float max = -INFINITY;
    for (int ii = 0; ii < numElements; ++ii)
    {
        if (ii * BLOCKSIZE + threadIdx.x < num)
        {
            // 使用寄存器数组存储, 避免后续二次读取全局内存
            regA[ii] = A[blockIdx.x * num + ii * BLOCKSIZE + threadIdx.x];
            float _max = fmaxf(max, regA[ii]);
            sum = sum * expf(max - _max) + expf(regA[ii] - _max);
            max = _max;
        }
    }

    // reduce
    for (int mask = WARPSIZE >> 1; mask > 0; mask >>= 1) {
        float _sum = __shfl_down_sync(0xffffffff, sum, mask, WARPSIZE);
        float _max = __shfl_down_sync(0xffffffff, max, mask, WARPSIZE);
        
        float new_max = fmaxf(max, _max);
        sum = sum * expf(max - new_max) + _sum * expf(_max - new_max);
        max = new_max;
    }

    // broadcast
    if (threadIdx.x % WARPSIZE == 0)
    {
        shared_sum[threadIdx.x / WARPSIZE] = sum;
        shared_max[threadIdx.x / WARPSIZE] = max;
    }
    __syncthreads();

    // reduce again
    if (threadIdx.x < NUMWARPS) {
        sum = shared_sum[threadIdx.x];
        max = shared_max[threadIdx.x];
        for (int mask = NUMWARPS >> 1; mask > 0; mask >>= 1) {
            float _sum = __shfl_down_sync(0x000000ff, sum, mask, NUMWARPS);
            float _max = __shfl_down_sync(0x000000ff, max, mask, NUMWARPS);

            float new_max = fmaxf(max, _max);
            sum = sum * expf(max - new_max) + _sum * expf(_max - new_max);
            max = new_max;
        }
    }

    // broadcast
    if (threadIdx.x == 0) {
        shared_sum[0] = sum;
        shared_max[0] = max;
    }
    __syncthreads();

    sum = shared_sum[0];
    max = shared_max[0];

    // div
    for (int ii = 0; ii < numElements; ++ii)
    {
        if (ii * BLOCKSIZE + threadIdx.x < num)
        {
            // 直接读取寄存器数组
            B[blockIdx.x * num + ii * BLOCKSIZE + threadIdx.x] = expf(regA[ii] - max) / sum;
        }
    }

    return;
}

int softmax_v1(const float *A, float *B, int bs, int num, cudaStream_t stream)
{
    constexpr int blockDimX = 256;
    constexpr int warpSize = 32;
    constexpr int numWarps = blockDimX / warpSize;
    constexpr int maxNumElementsPerThread = 32;
    dim3 blockDims(blockDimX);

    int gridDimX = bs;
    dim3 gridDims(gridDimX);

    softmax_kernel_v1<blockDimX, warpSize, numWarps, maxNumElementsPerThread>
        <<<gridDims, blockDims, 0, stream>>>(A, B, bs, num);

    return 0;
}